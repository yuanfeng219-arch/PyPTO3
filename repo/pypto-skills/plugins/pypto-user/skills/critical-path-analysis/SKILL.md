---
name: critical-path-analysis
description: Use when explaining why a chip-swimlane run took as long as it did — reconstructing the dependency-limited critical path and the as-executed path, splitting a makespan into compute versus data-wait, core-wait, and front-gap stall, or deciding whether a workload is dependency-bound, resource-bound, or compute-bound.
---

# Analyze a Run's Critical Path

Resolve the target repository root first. If it contains
`docs/dfx/chip-swimlane-profiling.md`, read that canonical guide before drawing
conclusions; `simpler_setup/tools/README.md` documents the tool surface. Use
this skill for artifact resolution, safe invocation, validation, and
interpretation.

The analysis is pure post-processing. It needs no device, no build, and no
repository checkout — only the wheel that produced the run.

## Bound system-test execution

Never run the full system-test suite locally. Run only system-test cases
directly relevant to the changed or requested scope; use CI for the full
system-test suite. If CI cannot run it, report that limitation instead of
substituting a local full-suite run.

## What the report answers

For each rank, the tool builds a happens-before graph from the dependency
graph oriented by observed timestamps, then reports two paths:

- **Static CPM path** — the longest duration-weighted path: the
  dependency-limited latency floor with unlimited cores.
- **Observed path** — the as-executed backward blame walk from the
  last-finishing task. Each task's compute plus the scheduling stall in front
  of it tiles the makespan exactly.

Stall on the observed path is attributed to one of three kinds:

| Kind | Means |
| ---- | ----- |
| `data-wait` | waiting on an upstream producer |
| `core-wait` | waiting for the assigned core to free up (resource serialization) |
| `front-gap` | launch or dispatch delay before the first task |

## Resolve the input

The tool discovers every directory under the given root that holds
`chip_swimlane_records.json` (or the legacy `l2_perf_records.json`),
`deps.json`, and `name_map*.json` as siblings. Point it at a whole run tree
and it analyzes each rank or device separately, writing one report beside each
records file — not one combined report at the scan root.

If the artifacts do not exist yet, or a capture must be planned, follow the
[capture contract](../../lib/dfx/capture.md). Do not invent capture flags here.

## Run

```bash
python -m simpler_setup.tools.critical_path <run-dir>
```

Useful options:

```bash
python -m simpler_setup.tools.critical_path <run-dir> --top 25 --stdout
```

- `--report` takes a bare filename, never a path; the report is always written
  beside its source records file. A path is rejected with exit status 2.
- `--top` sets the number of rows in the kernel-family table.
- `--tol` is the tick tolerance for deciding that one task finished before
  another started. Raise it only to absorb known clock skew, and say so in the
  report — a larger tolerance admits more edges into the path.
- `--stdout` adds the full report to standard output on top of the per-rank
  one-line summary that always prints.

Exit status 2 means the root does not exist, no rank directory was found, or a
`merged_swimlane*.json` was unreadable. The "no rank directory" message names
the chip-swimlane flag, but any one of the three artifacts being absent
produces it. After a split capture the missing one is the dependency graph, not
the timing — check for `deps.json` before believing the message.

## Validate before interpreting

A written report is not evidence that the analyzed path is real. Check all of
the following, and report any that fail instead of quoting numbers past them.

- **Capture integrity.** Confirm the run log's reconciliation lines per the
  capture contract. Dropped records understate every percentage in the report.
- **Tiling check.** Each rank prints `tiling check: compute+stall = ... vs
  makespan ...`. It must read `exact`. A non-zero difference means the walk did
  not tile the makespan; the per-task attribution is unsound.
- **Kernel names resolved.** Families named `unknown` or `cid<N>` mean the name
  map did not resolve. Family-level conclusions are meaningless until it does.
- **Warm-up.** With multiple rounds, capture happens on the first round only,
  so the makespan includes warm-up cost. Never present it as steady state.
- **Rank coverage.** The summary prints one line per rank. A multi-rank run
  that reports one rank means the other ranks' artifacts are incomplete.
- **Independent timing.** Cross-check the makespan against a host-side or
  device wall-clock measurement of the same run. A makespan far from it means
  the capture covered a different window than the one under discussion.
- **One capture, one sample.** Every percentage comes from a single run. Two
  captures of one unchanged workload can differ by several points of stall
  share, so never compare configurations from one capture each; repeat both.

## Interpret

Read the two headline percentages first, then the stall breakdown.

| Signal | Conclusion | Where to act |
| ------ | ---------- | ------------ |
| Static CPM near the makespan | dependency-bound: the graph itself is the floor, more cores cannot help | restructure the graph — shorten the chain, split long tasks, cut fan-in serialization |
| Static CPM well below the makespan, stall high | execution is losing time the dependencies do not require | read the stall kinds below |
| `core-wait` dominant | resource serialization: the path waits on busy cores | widen the work, rebalance core assignment, or reduce concurrent demand |
| `data-wait` dominant | a producer on the path is late | the named upstream kernel is the target, not the waiting task |
| `front-gap` large | launch or dispatch delay before any task ran | host-side or orchestration launch cost, not kernel cost |
| compute high, stall low | genuinely compute-bound | optimize the top families in the kernel-family table |

Then confirm the target before recommending work:

- The kernel-family table ranks families by compute time on the path. Use it to
  pick the target, and the full per-task listing to confirm the family is
  broadly costly rather than one outlier instance.
- A task on the observed path is not necessarily on the static CPM path.
  Optimizing a task that only the observed path visits removes stall, not the
  dependency floor; the reverse removes the floor. Say which one a proposal
  addresses.
- Percentages are of one rank's makespan. Do not add them across ranks.

When `merged_swimlane*.json` was present, the tool also writes `CPM_static.json`
and `CPM_observed.json`: the full Perfetto trace with off-path task bars
renamed so the selected path stands out. Offer these for visual confirmation;
they carry no numbers the report lacks.

## Reporting

Return:

- the analyzed run directory and how many ranks were discovered;
- makespan, static CPM absolute time and percentage, and the compute versus
  stall split with each stall kind;
- the validation checks above, including any that failed;
- the bounding classification — dependency-bound, resource-bound, or
  compute-bound — with the specific evidence behind it;
- the top kernel families with their compute share; and
- the limits of the measurement: which round it covers, whether the capture was
  paired or split, and the device measurement needed to confirm any proposed
  change.
