---
title: "nanotune runs"
description: "List saved training runs and inspect their loss histories"
sidebar_order: 9
---

# nanotune runs

List recent training runs, including the model, settings, dataset counts,
duration, outcome, and final losses. Run records are stored locally in
`.nanotune/runs/` and are excluded from Git.

## Usage

```bash
nanotune runs
nanotune runs --limit 1
```

Use `--json` to get the complete records, including per-iteration train and
validation loss points:

```bash
nanotune runs --json | jq '.[0].lossHistory'
```

## Options

| Flag | Description |
|------|-------------|
| `--limit <count>` | Maximum number of recent runs to show (default: `10`) |
| `--json` | Print complete run records as JSON on stdout |

The command requires a Nanotune project. Outside one, it writes the reason to
stderr and exits with status `1`.
