---
name: dependency-redundancy
description: Use when auditing a task dependency graph for edges that are already implied by a longer path — deciding whether explicit dependencies in an orchestration can be dropped, reading the transitive-reduction and dataflow-verified modes of the dependency viewer, or explaining why a reduction reports nothing removable.
---

# Audit a Dependency Graph for Redundant Edges

Resolve the target repository root first. If it contains `docs/dfx/dep-gen.md`,
read that canonical guide before drawing conclusions; `simpler_setup/tools/README.md`
documents the tool surface. Use this skill for input resolution, mode selection,
validation, and interpretation.

The analysis is pure post-processing over a captured graph. It needs no device,
no timing artifacts, and no build — only `deps.json` and the wheel that produced
it.

## Bound system-test execution

Never run the full system-test suite locally. Run only system-test cases
directly relevant to the changed or requested scope; use CI for the full
system-test suite. If CI cannot run it, report that limitation instead of
substituting a local full-suite run.

## What counts as redundant

An edge `(u, v)` is redundant when `v` is still reachable from `u` along some
other path — the ordering it expresses is already implied by a longer chain.
Removing such an edge cannot change execution order; it only removes
bookkeeping the scheduler would otherwise carry.

Whether the tool will *offer* to remove it is a separate question, and it turns
on the edge's `source`:

| `source` | Meaning | Reducible |
| -------- | ------- | --------- |
| `explicit` | an ordering the orchestration stated itself | yes |
| `tensormap` | an ordering derived from tensor dataflow | yes |
| `creator` | keeps alive the task that owns a tensor the consumer still references | **protected**, except under the dataflow modes |

A `creator` edge survives structural reduction unconditionally, because
execution order is not what it encodes.

## Read the edge sources and the depth first

Do this before selecting a mode. Both numbers decide whether any mode can say
anything at all, and skipping them is how a `0` gets misread as "the graph is
already minimal".

```bash
python - "$DEPS_JSON" <<'PY'
import collections, json, sys
graph = json.load(open(sys.argv[1]))
edges = graph["edges"]
print("edges by source:", collections.Counter(e.get("source") for e in edges))
successors = collections.defaultdict(list)
for edge in edges:
    successors[str(edge["pred"])].append(str(edge["succ"]))
seen = {}
def depth(node):
    if node not in seen:
        seen[node] = 0
        seen[node] = 1 + max((depth(n) for n in successors.get(node, [])), default=-1)
    return seen[node]
print("tasks:", len(graph["tasks"]), "edges:", len(edges),
      "depth:", max((depth(str(t["task_id"])) for t in graph["tasks"]), default=0))
PY
```

**Depth 1 ends the audit.** A graph with no two-hop path — a bipartite
producer/consumer set, for example — cannot contain a redundant edge at all.
Report that and stop.

**Any `creator` share at all makes `reduced` under-report.** Protection is
per-pair: one creator row anywhere on a `(pred, succ)` pair protects the whole
edge, and Step-A creator retention emits such a row for exactly the pairs a
dataflow dependency would also cover. A mixed-source graph is therefore no
safer than an all-`creator` one — a measured graph of 5120 `creator` plus 1008
`tensormap` edges had every one of its 2032 redundant pairs creator-annotated,
so `reduced` reported `0` while `reduced_dataflow` removed 992.

The consequence is a rule, not a judgement call: **never report from `reduced`
alone.** Its zero is evidence about the mode, not about the graph. Use the
source mix to explain a gap between the two modes, never to decide whether the
second one is worth running.

## Resolve the input

This analysis consumes `deps.json` alone; the timing artifacts a swimlane
analysis needs are irrelevant here. To produce one, follow the dep_gen half of
the [capture contract](../../lib/dfx/capture.md) — the chip-swimlane flag and
its level do not matter for this audit.

With no path given, the tool takes the newest `deps.json` under `./outputs/`.
Name the file explicitly whenever more than one run is present.

## Run

```bash
python -m simpler_setup.tools.deps_viewer <deps.json> --edge-mode reduced
python -m simpler_setup.tools.deps_viewer <deps.json> --edge-mode reduced_dataflow
```

- `reduced` keeps the non-redundant edges; `omitted` keeps their complement.
  Both print the same headline count and the same `pred -> succ` list of what
  reduction would drop, so either answers "which edges are redundant".
- `reduced_dataflow` and `omitted_dataflow` are the same pair with `creator`
  edges made eligible: one is dropped only when every creator annotation on the
  pair is an exactly-known `INOUT` region and every byte provably flows from an
  earlier `Output` and on to a later `INOUT` owned by the same creator. An
  `OUTPUT_EXISTING` edge begins a reuse generation and is always kept, and
  missing, ambiguous, or over-complex stride metadata keeps the edge too.
- Output goes to `deps_viewer_<mode>.txt` unless `-o` names a file, so a
  reduction never overwrites a full-graph render beside it. Add
  `--format html` for a graph view, which keeps every edge in the layout and
  paints unselected ones as background.
- `--func-names <name_map*.json>` replaces numeric labels with kernel names and
  makes the printed edge list readable.

## Validate before interpreting

- **A cycle silently disables the mode.** The tool warns on standard error,
  emits the full graph, and **still exits 0** — with the output stem falling
  back to `deps_viewer`. Read stderr; never take the exit status as proof a
  reduction ran.
- **Check the pair of modes.** Run `reduced` and `reduced_dataflow` both. Equal
  results mean the byte-level proof licensed nothing beyond the structural
  answer, which is a finding worth stating, not a reason to report one number.
- **A zero-edge output file is not an empty graph.** In text mode only the
  selected edges are written, so an `omitted` run with nothing redundant
  correctly writes a file with zero edges.
- **The graph is one capture of one topology.** A `deps.json` describes the
  submits of the run that produced it. Do not carry a conclusion across a shape,
  batch size, or layer count the capture did not cover.

## Interpret

| Result | Conclusion | Action |
| ------ | ---------- | ------ |
| `reduced` removes N > 0 | those `explicit` or `tensormap` edges are already implied by a longer path | drop them from the orchestration; ordering is unchanged and the scheduler carries less fan-in |
| `reduced` removes 0 and any edge carries `creator` | uninformative by construction | run `reduced_dataflow` before saying anything |
| `reduced_dataflow` removes more than `reduced` | those creator edges are provably inside one reuse generation | they are droppable, but say plainly that the proof is byte-level and rests on the capture's tensor metadata |
| both remove 0, depth ≥ 2 | nothing is *removable* | say exactly that; redundant-but-retained edges are invisible to every mode, so it is not evidence the graph is minimal |
| depth 1 | no two-hop path exists | do not run the audit |

An edge that reduction would drop is dead scheduling bookkeeping, not dead
ordering. Never describe removing one as a change in execution order, and never
promise a speedup the run did not measure — the gain is fewer dependency
records, and whether that is visible belongs to a timing measurement.

## Reporting

Return:

- the analyzed `deps.json`, its task and edge counts, its edge-source mix, and
  its depth;
- the count and the listed `pred -> succ` edges each mode selected, for both
  `reduced` and the dataflow variant;
- the validation checks above, including whether a cycle warning appeared;
- which edges are safe to drop and which are retained as lifetime references;
  and
- the limits: one capture, one topology, and the measurement that would be
  needed to show any effect of acting on the result.
