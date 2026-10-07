import {existsSync, mkdirSync, readdirSync, readFileSync} from 'node:fs';
import {join} from 'node:path';
import {z} from 'zod';
import {TrainingConfigSchema} from '../types/index.js';
import {
	getProjectDir,
	initializeProjectDirs,
	writeFileAtomic,
} from './config.js';

const RUNS_DIR = 'runs';

export const TrainingRunRecordSchema = z.object({
	id: z.string().uuid(),
	startedAt: z.string().datetime(),
	finishedAt: z.string().datetime(),
	status: z.enum(['completed', 'stopped', 'failed']),
	baseModel: z.string(),
	training: TrainingConfigSchema,
	examples: z.object({
		train: z.number().int().nonnegative(),
		validation: z.number().int().nonnegative(),
	}),
	durationMs: z.number().int().nonnegative(),
	lossHistory: z.array(
		z.object({
			iteration: z.number().int().nonnegative(),
			trainLoss: z.number().finite(),
			valLoss: z.number().finite().optional(),
		}),
	),
	finalTrainLoss: z.number().finite().nullable(),
	finalValLoss: z.number().finite().nullable(),
	resume: z.boolean(),
	adapterPath: z.string(),
	adapterModifiedAt: z.string().datetime().nullable(),
	error: z.string().optional(),
});

export type TrainingRunRecord = z.infer<typeof TrainingRunRecordSchema>;

export function getTrainingRunsDir(): string {
	return join(getProjectDir(), RUNS_DIR);
}

/** Create the local-only run-history directory when a record is written. */
export function ensureTrainingRunsDir(): string {
	const dir = getTrainingRunsDir();
	mkdirSync(dir, {recursive: true});
	return dir;
}

/** Persist a complete record with an atomic rename so readers never see partial JSON. */
export function saveTrainingRun(record: TrainingRunRecord): string {
	const validated = TrainingRunRecordSchema.parse(record);
	// Back-fill `runs/` in projects initialized by older Nanotune versions too.
	initializeProjectDirs();
	const dir = ensureTrainingRunsDir();
	const filename = `${validated.id}.json`;
	const path = join(dir, filename);
	writeFileAtomic(path, `${JSON.stringify(validated, null, 2)}\n`);
	return path;
}

/** Read valid run records newest first; a corrupt/incomplete file is skipped. */
export function listTrainingRuns(): TrainingRunRecord[] {
	const dir = getTrainingRunsDir();
	if (!existsSync(dir)) {
		return [];
	}
	return readdirSync(dir)
		.filter(name => name.endsWith('.json'))
		.map(name => {
			try {
				return TrainingRunRecordSchema.parse(
					JSON.parse(readFileSync(join(dir, name), 'utf8')),
				);
			} catch {
				return null;
			}
		})
		.filter((record): record is TrainingRunRecord => record !== null)
		.sort((a, b) => b.startedAt.localeCompare(a.startedAt));
}

/** Compact terminal summary; `--json` exposes each complete loss history. */
export function formatTrainingRuns(runs: TrainingRunRecord[]): string {
	if (runs.length === 0) {
		return 'No training runs found.';
	}
	return runs
		.map(run => {
			const losses =
				run.finalTrainLoss === null
					? 'no loss points'
					: `train ${run.finalTrainLoss.toFixed(4)}${
							run.finalValLoss === null
								? ''
								: ` / validation ${run.finalValLoss.toFixed(4)}`
						}`;
			return [
				run.startedAt,
				run.status,
				run.baseModel,
				`${run.training.iterations} iterations`,
				`${run.examples.train} train / ${run.examples.validation} validation`,
				`${Math.round(run.durationMs / 1000)}s`,
				losses,
			].join('  ');
		})
		.join('\n');
}
