import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "ava";
import { execa, type ResultPromise } from "execa";
import {
  abortTraining,
  buildLoraConfigYaml,
  buildTrainingArgs,
  needsLoraConfig,
  checkpointToRestore,
  commitPendingSave,
  noteValLoss,
  parseTrainingLogLine,
  restoreBestAdapter,
  shouldStopEarly,
  shouldTreatAsStop,
  stopOnAbort,
  type ValTracker,
} from "./mlx.js";
import type { MLXTrainingOptions } from "./mlx.js";

test("parseTrainingLogLine reads a combined train and val line", (t) => {
  const parsed = parseTrainingLogLine(
    "Iter 10: Train loss 1.234, Val loss 1.456",
  );
  t.is(parsed?.iteration, 10);
  t.is(parsed?.trainLoss, 1.234);
  t.is(parsed?.valLoss, 1.456);
});

test("parseTrainingLogLine reads a train line without val loss", (t) => {
  const parsed = parseTrainingLogLine("Iter 50: Train loss 0.456");
  t.is(parsed?.iteration, 50);
  t.is(parsed?.trainLoss, 0.456);
  t.is(parsed?.valLoss, undefined);
});

test("parseTrainingLogLine reads a train line with iteration speed", (t) => {
  const parsed = parseTrainingLogLine(
    "Iter 100 (15.2 it/s): Train loss 0.342, Val loss 0.298",
  );
  t.is(parsed?.iteration, 100);
  t.is(parsed?.trainLoss, 0.342);
  t.is(parsed?.valLoss, 0.298);
});

test("parseTrainingLogLine reads mlx's separate val line", (t) => {
  const parsed = parseTrainingLogLine(
    "Iter 50: Val loss 0.456, Val took 1.230s",
  );
  t.is(parsed?.iteration, 50);
  t.is(parsed?.valLoss, 0.456);
  t.is(parsed?.trainLoss, undefined);
});

test("parseTrainingLogLine reads mlx-lm 0.32 rich output, ANSI included", (t) => {
  const val = parseTrainingLogLine(
    " \u001b[38;5;244m  50\u001b[0m \u001b[1;35mval\u001b[0m \u001b[1m0.456\u001b[0m \u001b[38;5;244m1.23s\u001b[0m",
  );
  t.is(val?.iteration, 50);
  t.is(val?.valLoss, 0.456);

  const train = parseTrainingLogLine("   10 1.234 ▼ 1,234  12.3k");
  t.is(train?.iteration, 10);
  t.is(train?.trainLoss, 1.234);
  t.is(train?.valLoss, undefined);
});

test("a mlx-lm 0.32 log stops at patience and names the best save", (t) => {
  const log = [
    " \u001b[38;5;244m   1\u001b[0m \u001b[1;35mval\u001b[0m \u001b[1m2.100\u001b[0m \u001b[38;5;244m0.40s\u001b[0m",
    "   10 1.800 \u25bc 1,234  12.3k",
    "   50 val 1.200 0.41s",
    "   50 0.900 \u25bc 1,100  20.0k",
    "  100 val 0.800 0.41s",
    "  100 0.700 \u25bc 1,000  30.0k",
    "  150 val 0.850 0.41s",
    "  200 val 0.900 0.41s",
    "  250 val 0.950 0.41s",
  ];
  let tracker: ValTracker | null = null;
  let stoppedAt: number | null = null;
  for (const line of log) {
    const event = parseTrainingLogLine(line);
    t.truthy(event, line);
    if (!event || event.valLoss === undefined) continue;
    tracker = noteValLoss(tracker, event.iteration, event.valLoss, 50);
    if (shouldStopEarly(tracker, 3)) {
      stoppedAt = event.iteration;
      break;
    }
  }
  t.is(stoppedAt, 250);
  t.is(tracker?.bestSavedIteration, 100);
  t.is(
    checkpointToRestore(tracker, {
      iterations: 400,
      earlyStop: true,
      loadBest: false,
      userAborted: false,
    }),
    100,
  );
});

test("parseTrainingLogLine ignores checkpoint chatter", (t) => {
  t.is(
    parseTrainingLogLine(
      "Iter 50: Saved adapter weights to adapters.safetensors and 0000050_adapters.safetensors.",
    ),
    null,
  );
});

function vals(
  reports: Array<[number, number]>,
  saveEvery: number,
): ValTracker | null {
  let tracker: ValTracker | null = null;
  for (const [iteration, loss] of reports) {
    tracker = noteValLoss(tracker, iteration, loss, saveEvery);
  }
  return tracker;
}

test("noteValLoss stops after patience validation checks without improvement", (t) => {
  const tracker = vals(
    [
      [50, 1.0],
      [100, 0.9],
      [150, 0.95],
      [200, 0.96],
    ],
    50,
  );
  t.is(tracker?.bestIteration, 100);
  t.is(tracker?.bestSavedIteration, 100);
  t.is(tracker?.evalsSinceBest, 2);
  t.false(shouldStopEarly(vals([[50, 1.0], [100, 0.9], [150, 0.95]], 50), 2));
  t.true(shouldStopEarly(tracker, 2));
  t.false(shouldStopEarly(tracker, 0));
});

test("noteValLoss does not treat an equal loss as improvement", (t) => {
  const tracker = vals(
    [
      [50, 1.0],
      [100, 1.0],
    ],
    50,
  );
  t.is(tracker?.bestIteration, 50);
  t.is(tracker?.evalsSinceBest, 1);
  t.is(tracker?.bestSavedIteration, 50);
});

test("noteValLoss only remembers a checkpoint mlx actually saved", (t) => {
  const tracker = vals(
    [
      [50, 0.5],
      [100, 0.8],
      [150, 0.4],
      [200, 0.9],
    ],
    100,
  );
  t.is(tracker?.bestIteration, 150);
  t.is(tracker?.bestSavedIteration, 100);
  t.is(tracker?.bestSavedValLoss, 0.8);
});

test("noteValLoss updates the saved checkpoint when a later save is better", (t) => {
  const tracker = commitPendingSave(
    vals(
      [
        [100, 0.8],
        [200, 0.7],
      ],
      100,
    ),
  );
  t.is(tracker?.bestSavedIteration, 200);
  t.is(tracker?.bestSavedValLoss, 0.7);
});

test("a save-step val is not restorable until that step has finished", (t) => {
  const tracker = vals(
    [
      [10, 1.0],
      [20, 1.2],
    ],
    20,
  );
  t.true(shouldStopEarly(tracker, 1));
  t.is(tracker?.bestSavedIteration, null);
  t.is(tracker?.pendingSaveIteration, 20);
});

test("noteValLoss ignores a non-finite loss until a real one arrives", (t) => {
  t.is(noteValLoss(null, 50, Number.NaN, 50), null);
  const tracker = noteValLoss(vals([[50, 1.0]], 50), 100, Number.NaN, 50);
  t.is(tracker?.bestIteration, 50);
  t.is(tracker?.evalsSinceBest, 1);
  t.is(tracker?.bestSavedIteration, 50);
});

function trackerAt(
  bestIteration: number,
  bestSavedIteration: number | null,
  lastValLoss = 0.4,
): ValTracker {
  return {
    bestValLoss: 0.4,
    bestIteration,
    evalsSinceBest: 0,
    bestSavedIteration,
    bestSavedValLoss: bestSavedIteration == null ? null : 0.8,
    pendingSaveIteration: null,
    pendingSaveValLoss: null,
    lastValIteration: bestIteration,
    lastValLoss,
  };
}

test("checkpointToRestore leaves a finished run alone when the last step was best", (t) => {
  t.is(
    checkpointToRestore(trackerAt(150, 100), {
      iterations: 150,
      earlyStop: false,
      loadBest: true,
      userAborted: false,
    }),
    null,
  );
});

test("checkpointToRestore keeps the final step when its val beats the saved checkpoint", (t) => {
  const tracker = {
    ...trackerAt(70, 50, 0.6),
    lastValIteration: 80,
  };
  t.is(
    checkpointToRestore(tracker, {
      iterations: 80,
      earlyStop: false,
      loadBest: true,
      userAborted: false,
    }),
    null,
  );
});

test("checkpointToRestore returns the best save when a later step is worse", (t) => {
  t.is(
    checkpointToRestore(trackerAt(100, 100), {
      iterations: 200,
      earlyStop: false,
      loadBest: true,
      userAborted: false,
    }),
    100,
  );
});

test("checkpointToRestore restores on early stop even without load-best", (t) => {
  t.is(
    checkpointToRestore(trackerAt(100, 100), {
      iterations: 200,
      earlyStop: true,
      loadBest: false,
      userAborted: false,
    }),
    100,
  );
});

test("checkpointToRestore does nothing on Ctrl+C or when both switches are off", (t) => {
  const tracker = trackerAt(100, 100);
  t.is(
    checkpointToRestore(tracker, {
      iterations: 200,
      earlyStop: true,
      loadBest: true,
      userAborted: true,
    }),
    null,
  );
  t.is(
    checkpointToRestore(tracker, {
      iterations: 200,
      earlyStop: false,
      loadBest: false,
      userAborted: false,
    }),
    null,
  );
  t.is(
    checkpointToRestore(trackerAt(100, null), {
      iterations: 200,
      earlyStop: true,
      loadBest: true,
      userAborted: false,
    }),
    null,
  );
});

test("restoreBestAdapter copies the numbered snapshot over adapters.safetensors", (t) => {
  const dir = mkdtempSync(join(tmpdir(), "nanotune-ckpt-"));
  try {
    writeFileSync(join(dir, "0000100_adapters.safetensors"), "best");
    writeFileSync(join(dir, "adapters.safetensors"), "last");
    t.true(restoreBestAdapter(dir, 100));
    t.is(readFileSync(join(dir, "adapters.safetensors"), "utf8"), "best");
    t.false(restoreBestAdapter(dir, 200));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("TrainingProgress structure is correct", (t) => {
  const progress = {
    iteration: 50,
    totalIterations: 150,
    trainLoss: 0.456,
    valLoss: 0.423,
  };

  t.is(progress.iteration, 50);
  t.is(progress.totalIterations, 150);
  t.is(progress.trainLoss, 0.456);
  t.is(progress.valLoss, 0.423);
});

function trainingOptions(
  overrides: Partial<MLXTrainingOptions> = {},
): MLXTrainingOptions {
  return {
    model: "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    dataPath: "/path/to/data",
    adapterPath: "/path/to/adapters",
    iterations: 150,
    learningRate: 5e-5,
    batchSize: 4,
    numLayers: 16,
    stepsPerEval: 50,
    saveEvery: 50,
    resume: false,
    fineTuneType: "lora",
    loraRank: 8,
    loraAlpha: 20,
    loraDropout: 0,
    maxSeqLength: 2048,
    gradCheckpoint: false,
    valBatches: 25,
    seed: 0,
    earlyStoppingPatience: 0,
    loadBestModelAtEnd: false,
    ...overrides,
  };
}

// Reads the value that follows `flag`, so a test fails loudly if a flag is
// dropped or wired to the wrong field rather than passing on a bare includes().
function argValue(args: string[], flag: string): string | undefined {
  const index = args.indexOf(flag);
  return index === -1 ? undefined : args[index + 1];
}

test("buildTrainingArgs passes the hyperparameter flags through to mlx_lm", (t) => {
  const args = buildTrainingArgs(
    trainingOptions({
      fineTuneType: "dora",
      maxSeqLength: 1024,
      valBatches: 10,
      seed: 42,
    }),
  );

  t.is(argValue(args, "--fine-tune-type"), "dora");
  t.is(argValue(args, "--max-seq-length"), "1024");
  t.is(argValue(args, "--val-batches"), "10");
  t.is(argValue(args, "--seed"), "42");
});

test("buildTrainingArgs adds --grad-checkpoint only when enabled", (t) => {
  t.false(buildTrainingArgs(trainingOptions()).includes("--grad-checkpoint"));
  t.true(
    buildTrainingArgs(trainingOptions({ gradCheckpoint: true })).includes(
      "--grad-checkpoint",
    ),
  );
});

test("buildTrainingArgs passes the lora config path when given one", (t) => {
  const args = buildTrainingArgs(trainingOptions(), "/tmp/lora.yaml");
  t.is(argValue(args, "-c"), "/tmp/lora.yaml");
});

test("buildTrainingArgs omits -c when no lora config was written", (t) => {
  t.false(buildTrainingArgs(trainingOptions()).includes("-c"));
});

test("buildTrainingArgs still wires --resume-adapter-file", (t) => {
  const args = buildTrainingArgs(trainingOptions({ resume: true }));
  t.is(
    argValue(args, "--resume-adapter-file"),
    "/path/to/adapters/adapters.safetensors",
  );
});

test("needsLoraConfig is false for full fine-tuning only", (t) => {
  // `full` trains weights directly, so mlx_lm never reads lora_parameters.
  t.true(needsLoraConfig("lora"));
  t.true(needsLoraConfig("dora"));
  t.false(needsLoraConfig("full"));
});

test("buildLoraConfigYaml produces the expected lora_parameters block", (t) => {
  const yaml = buildLoraConfigYaml(8, 20, 0);
  t.is(yaml, "lora_parameters:\n  rank: 8\n  scale: 20\n  dropout: 0\n");
});

test("buildLoraConfigYaml reflects overridden values", (t) => {
  const yaml = buildLoraConfigYaml(16, 32, 0.05);
  t.is(yaml, "lora_parameters:\n  rank: 16\n  scale: 32\n  dropout: 0.05\n");
});

test("buildLoraConfigYaml avoids exponent notation PyYAML would read as a string", (t) => {
  // PyYAML's 1.1 float resolver rejects `1e-7`, so String() is not enough.
  const yaml = buildLoraConfigYaml(8, 20, 1e-7);
  t.false(yaml.includes("e-"));
  t.regex(yaml, /dropout: 0\.0000001/);
});

// ── graceful stop (Ctrl+C) ────────────────────────────────────────────

function fakeSubprocess() {
  const signals: string[] = [];
  const subprocess = {
    kill(signal: string) {
      signals.push(signal);
      return true;
    },
  } as unknown as ResultPromise;
  return { subprocess, signals };
}

test("abortTraining sends SIGINT so MLX can flush its checkpoint", (t) => {
  const { subprocess, signals } = fakeSubprocess();
  abortTraining(subprocess);
  t.deepEqual(signals, ["SIGINT"]);
});

test("stopOnAbort does nothing until the signal aborts", (t) => {
  const { subprocess, signals } = fakeSubprocess();
  const controller = new AbortController();
  stopOnAbort(subprocess, controller.signal);
  t.deepEqual(signals, []);

  controller.abort();
  t.deepEqual(signals, ["SIGINT"]);
});

test("stopOnAbort stops a signal that is already aborted", (t) => {
  // An already-aborted signal never fires an `abort` event, so a bare
  // addEventListener would leave the trainer running forever.
  const { subprocess, signals } = fakeSubprocess();
  const controller = new AbortController();
  controller.abort();
  stopOnAbort(subprocess, controller.signal);
  t.deepEqual(signals, ["SIGINT"]);
});

test("stopOnAbort signals only once", (t) => {
  const { subprocess, signals } = fakeSubprocess();
  const controller = new AbortController();
  stopOnAbort(subprocess, controller.signal);
  controller.abort();
  controller.abort();
  t.deepEqual(signals, ["SIGINT"]);
});

test("stopOnAbort without a signal never touches the subprocess", (t) => {
  const { subprocess, signals } = fakeSubprocess();
  const detach = stopOnAbort(subprocess, undefined);
  detach();
  t.deepEqual(signals, []);
});

test("stopOnAbort detach stops a later abort from signalling a dead process", (t) => {
  // A caller-owned signal outlives the run: once training is over, aborting it
  // must not SIGINT a closed subprocess whose PID may have been recycled.
  const { subprocess, signals } = fakeSubprocess();
  const controller = new AbortController();
  const detach = stopOnAbort(subprocess, controller.signal);

  detach();
  controller.abort();

  t.deepEqual(signals, []);
});

test("stopOnAbort detach is safe to call twice", (t) => {
  const { subprocess, signals } = fakeSubprocess();
  const controller = new AbortController();
  const detach = stopOnAbort(subprocess, controller.signal);

  detach();
  detach();
  controller.abort();

  t.deepEqual(signals, []);
});

test("stopOnAbort detach after an abort leaves the stop intact", (t) => {
  const { subprocess, signals } = fakeSubprocess();
  const controller = new AbortController();
  const detach = stopOnAbort(subprocess, controller.signal);

  controller.abort();
  detach();

  t.deepEqual(signals, ["SIGINT"]);
});

test("stopOnAbort detach for an already-aborted signal is a no-op", (t) => {
  const { subprocess, signals } = fakeSubprocess();
  const controller = new AbortController();
  controller.abort();

  const detach = stopOnAbort(subprocess, controller.signal);
  detach();

  t.deepEqual(signals, ["SIGINT"]);
});

test("stopOnAbort terminates a real running child process", async (t) => {
  const child = execa("node", ["-e", "setInterval(() => {}, 1000)"]);
  t.truthy(child.pid);

  const controller = new AbortController();
  stopOnAbort(child, controller.signal);
  controller.abort();

  // The child would run forever, so this only settles because it was killed.
  t.truthy(await t.throwsAsync(child));
});

test("shouldTreatAsStop returns true when signal is aborted", (t) => {
  const controller = new AbortController();
  controller.abort();
  t.true(shouldTreatAsStop(controller.signal));
});

test("shouldTreatAsStop returns false when signal is not aborted", (t) => {
  const controller = new AbortController();
  t.false(shouldTreatAsStop(controller.signal));
});

test("shouldTreatAsStop returns false when signal is undefined", (t) => {
  t.false(shouldTreatAsStop(undefined));
});
