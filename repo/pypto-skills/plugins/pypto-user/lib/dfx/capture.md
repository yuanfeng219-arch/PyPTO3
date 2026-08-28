# Capture chip-swimlane DFX artifacts

Shared capture contract for every skill that post-processes a chip-swimlane
run. Analysis skills read this file; they never restate the capture rules.

## The three artifacts

A post-processing tool discovers a *rank directory*: one directory holding all
three files as siblings.

| File | Produced by | Carries |
| ---- | ----------- | ------- |
| `chip_swimlane_records.json` (legacy name `l2_perf_records.json`) | chip-swimlane capture | per-task timing for one run |
| `deps.json` | dep_gen capture | the static task graph for one topology |
| `name_map*.json` | the same run | `callable_id_to_name`, used for kernel labels |

`merged_swimlane*.json` is optional. When present beside the triple, the
critical-path tool also emits Perfetto traces; when absent it warns and still
writes its Markdown report.

Only the timing-based analyses need the full triple co-located. An analysis of
the task graph's structure consumes `deps.json` on its own, so a capture made
for one of those can leave the chip-swimlane level at whatever the run already
used.

Artifacts land under the run's `output_prefix`. Two layouts are common and
both are supported by the tools:

- `outputs/<case>_<timestamp>/` — scene-test style runs.
- `build_output/<case>/dfx_outputs/` — JIT / model-library runs, including
  nested per-rank and per-device subdirectories.

## Turning capture on

The capture gate is the call configuration object, not a CLI flag. Any Python
caller can set it:

```python
config.enable_chip_swimlane = 4   # 0 disables; see the level table below
config.enable_dep_gen = True
```

A runner that exposes these as command-line flags is offering a convenience
wrapper over the same two fields:

```bash
python <case>.py --platform <target> -d <device> --enable-chip-swimlane 4 --enable-dep-gen
```

`--enable-chip-swimlane` takes an optional integer level; a bare flag means 4.

| Level | Adds |
| ----- | ---- |
| 0 | disabled |
| 1 | AICore task timing only |
| 2 | + AICPU dispatch / finish timestamps |
| 3 | + scheduler phases |
| 4 | + orchestrator phases |

Level 1 is enough for a critical-path report — it needs task start/end plus
the graph, and a level-1 capture carries no AICPU task records at all yet still
produces a complete report. Choose a higher level only when the same capture
must also feed a scheduler-overhead analysis.

**An output directory is mandatory.** With dep_gen enabled alongside any other
diagnostic, the runtime throws unless `output_prefix` is set. Scene-test
harnesses set it for you; a hand-written runner must set it explicitly.

## Paired capture or split capture

**Paired** — one run with both flags. Both artifacts land in the same
directory, so a critical-path analysis works with no further steps. Combined
per-round overhead is well under 10 µs on measured workloads. This is the
right default.

**Split** — dep_gen in one run, chip-swimlane in another. Use it when
measuring at µs resolution, when one topology is measured under several
configurations, or when a workload is large enough that the dep_gen replay
would dominate the measured run.

Split capture puts the two artifacts in *different* directories, and the
critical-path tool has no flag to join across directories — it requires the
triple to be co-located. Bridge it by copying the captured graph into the
measured run's directory before analyzing:

```bash
cp <dep-gen-run-dir>/deps.json <swimlane-run-dir>/deps.json
```

This is sound only while the topology is unchanged. Re-capture `deps.json`
after any change to the task graph.

## Keep the capture small

Capture a small number of layers or iterations, not a full-model run. A
full-model swimlane trace reaches megabytes per layer, and the device-side
record buffers overflow. Two layers give one warm-up plus one steady-state
layer, which is enough to read the structure.

With `--rounds N`, records are collected on the **first** round only, so the
steady-state benchmark is left unperturbed. The analyzed makespan is therefore
a first-round makespan and carries warm-up cost.

## Validate the capture before analyzing it

The collectors reconcile their device-side and host-side counters when a run
ends, but the *positive* confirmation is emitted at info level and the usual
default is quieter. An absent line is therefore not a passing capture. Raise the
level on any run you intend to analyze:

```bash
python <case>.py ... --enable-chip-swimlane 1 --enable-dep-gen --log-level info
grep reconcile <run-log>
```

A complete capture prints one line per collector pool plus one for the graph:

```text
ChipSwimlane reconcile: PERF counts match (collected=128, dropped=0, device_total=128)
ChipSwimlane reconcile: SCHED_PHASE counts match (collected=157, dropped=0, device_total=157)
ChipSwimlane reconcile: ORCH_PHASE counts match (collected=128, dropped=0, device_total=128)
dep_gen reconcile: counts match (collected=128, dropped=0, device_total=128, overflow=0)
```

Every pool the capture level enables must appear, and every `dropped` and
`overflow` must be zero. A pool missing from the list is not a pass.

Failures are warnings, so they stay visible even at a default level:

- `records dropped on device side` — records were lost; per-task timing is
  incomplete and every downstream percentage is understated. Raise the
  per-core / per-thread profiling buffer counts, or shrink the capture, and
  re-run.
- `count mismatch` — silent loss. Treat the artifacts as unusable.
- An un-flushed buffer reported at `stop()` is a flush defect, not a tail loss
  to tune around. Report it rather than analyzing around it.

A simulator target exercises the export pipeline but its device clock is not
realistic. Never draw absolute-timing conclusions from a simulated run; use
hardware for any steady-state number.
