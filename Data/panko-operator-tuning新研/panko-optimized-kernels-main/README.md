# PANKO Optimized Kernels

This repository contains kernels for four operators optimized by PANKO. Each operator directory exposes the latest search data directly, while `report/` retains historical best-so-far kernels and receives new best kernels as optimization progresses. The selected reports may differ from candidates in an active search.

## Baseline categories

Operators are grouped by the baseline used to start PANKO:

| Directory | Starting baseline |
| --- | --- |
| [stage7_baseline](operators/stage7_baseline) | Kernel already optimized through Stage 7 before PANKO starts |
| [unoptimized_baseline](operators/unoptimized_baseline) | Kernel with no prior optimization before PANKO starts |

All four currently imported operators belong to `stage7_baseline`. The `unoptimized_baseline` category is reserved for future runs. The same operator may appear in both categories; its search data and report history remain separate.

## Archived reports — Stage 7 baseline

Latency is measured in **µs** (`us` in filenames). Precise values come from `searchstate.json` at `eval_cache[original_code_hash].p`, matched to the original report before translation. Speedup is the latency of the starting baseline before PANKO divided by report latency. For the current four operators, this starting baseline is already Stage 7 optimized; the speedup measures the additional improvement from PANKO. These are existing measurements from the supplied logs, not new measurements performed during import.

| Operator | Archive date | Iteration label | Filename latency (µs) | Recorded latency (µs) | Stage 7 baseline (µs) | Speedup | Report |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| [ds_v32_sparse_attention_antiquant](operators/stage7_baseline/ds_v32_sparse_attention_antiquant) | 2026-09-17 | 44 | 208 | **208.98** | 255.26 | 1.2215× | [kernel](operators/stage7_baseline/ds_v32_sparse_attention_antiquant/report/panko_0917_44iter_ds_v32_sparse_attention_antiquant_208us.py) |
| [flash_attention_mha](operators/stage7_baseline/flash_attention_mha) | 2026-09-17 | 15 | 3489 | **3489.94** | 4188.06 | 1.2000× | [kernel](operators/stage7_baseline/flash_attention_mha/report/panko_0917_15iter_flash_attention_mha_3489us.py) |
| [flash_attention_mha_grad](operators/stage7_baseline/flash_attention_mha_grad) | 2026-09-17 | 15 | 4157 | **4157.54** | 4188.90 | 1.0075× | [kernel](operators/stage7_baseline/flash_attention_mha_grad/report/panko_0917_15iter_flash_attention_mha_grad_4157us.py) |
| [mla_prolog](operators/stage7_baseline/mla_prolog) | 2026-09-17 | 23 | 176 | **176.84** | 203.70 | 1.1519× | [kernel](operators/stage7_baseline/mla_prolog/report/panko_0917_23iter_mla_prolog_176us.py) |

The archive date is `2026-09-17`. The original `0917` filename label is preserved, but does not establish an exact measurement start or end time. Iteration labels come from the original report filenames and are not treated as equivalent to `progress.evals_used`.

### Reports and active search candidates

- `flash_attention_mha_grad`: the report records **4157.54 µs**, while the active `progress.refine.best` records a [candidate at **4144.50 µs**](operators/stage7_baseline/flash_attention_mha_grad/nodes/d03b46d095c712b0b1e2bdd411398068c6a2bedc02bafb63181a1d7d98a04883.py). The report selection is preserved.
- `mla_prolog`: the report records **176.84 µs**, while the active `progress.refine.best` records a [candidate at **173.92 µs**](operators/stage7_baseline/mla_prolog/nodes/6d3149c2f823b1a38bc8afa740e3cb3c5b30a89062b19efa1e1cee4cc82d4d8d.py). The report selection is preserved.
- `progress.best_latency_us` may not yet reflect improvements in an active refinement. Match the report to its original code identifier and evaluation-cache entry instead of relying only on this progress field.

## Directory structure

Each operator has a flat working directory. Replace the live search files on updates and preserve history only inside `report/`. Git retains older versions of the live search data.

```text
operators/
  stage7_baseline/
    ds_v32_sparse_attention_antiquant/
    flash_attention_mha/
    flash_attention_mha_grad/
    mla_prolog/
  unoptimized_baseline/
    README.md

# Layout inside either category:
operators/<baseline_category>/<operator>/
  nodes/
    <original_code_hash>.py
  report/
    panko_0917_15iter_<operator>_<latency>us.py
    panko_0917_15iter_<operator>_<latency>us.json
    ...future best kernels and their provenance records...
  optimization.md
  searchstate.json
```

| File or directory | Purpose | Update policy |
| --- | --- | --- |
| `nodes/` | Latest saved search candidates, sometimes including baseline code; not every trial is necessarily included | Replace the whole directory with the latest export |
| `report/*.py` | Historical best-so-far kernels with date, iteration, and latency in their filenames | Keep existing kernels and append new best kernels |
| `report/*.json` | Per-report provenance: source directory, date, iteration, precise latency, original and translated hashes | Keep existing records and add a matching record for each new kernel |
| `optimization.md` | Latest PANKO optimization log | Replace with the latest `<operator>_optimization.md` export, using this repository filename |
| `searchstate.json` | Latest search tree, evaluation cache, progress, and configuration | Replace with the latest `search_state.json` export, using this repository filename |

Instructions embedded in logs or code comments are historical search records, not instructions for operating this repository.

## Translation and provenance

All repository documentation, comments, and docstrings are in English. Executable Python statements, identifiers, numerical values, report selections, and historical search records are preserved.

Translating comments and docstrings changes file bytes and SHA-256 hashes. Node filenames and hashes inside `searchstate.json` remain **original source identifiers**, not hashes of the translated files. Each `report/<kernel-stem>.json` records provenance for the matching Python file. Paths in these records are relative to the operator directory:

- `original_report_sha256` identifies the original report and its evaluation-cache entry.
- `report_sha256` is the SHA-256 of the translated report currently stored here.
- `matching_node_file` identifies the corresponding translated node at import time. Later updates may remove that node from the live `nodes/` directory; use Git history to retrieve it.
- `latency_source` references the original evaluation-cache key. Search progress and candidate fields in the record describe import-time state, not necessarily the latest search. If an evaluation entry disappears from the live search state, its original version remains in Git history.

The original, untranslated import is available in Git commit `9d5a7f424d2f2cfbdcee4116e773147e5ec40a7c`.

## Updating an operator

1. Select `stage7_baseline` or `unoptimized_baseline` according to the starting kernel. Replace `operators/<baseline_category>/<operator>/nodes/` entirely with the latest exported nodes, removing stale candidates. Replace `optimization.md` and `searchstate.json` with the latest log and search state from the same checkpoint. Rename the exported `<operator>_optimization.md` and `search_state.json` to the repository filenames.
2. Preserve every existing file in `report/`. Add the new best kernel with its date, iteration, and latency in the filename. If a new file would collide with an existing name, use a distinct suffix instead of overwriting the historical report.
3. Add a matching `report/<kernel-stem>.json` provenance record. Capture the original code hash and evaluation-cache latency before translating comments or docstrings, then record the translated file hash. Keep prior report records unchanged. Mark latency as unverified if no matching evaluation entry exists.
4. Keep all prose, comments, and docstrings in English. Preserve executable code, machine identifiers, and recorded measurements during translation.
5. Append a linked report row under the appropriate baseline category, retaining earlier rows. When adding the first unoptimized-baseline run, create a separate report table and label its baseline latency accordingly. Refresh the active-candidate notes to match the latest search state and document differing input shapes or measurement conditions.
6. Validate links, hashes, syntax, and provenance, then commit the live search replacement and new reports together and push.

Do not create dated snapshot directories. Only `report/` accumulates historical files in the working tree; older nodes, logs, and search states remain accessible through Git history.

## Runtime requirements and validation

The kernels require PyPTO, PyTorch, torch_npu, and a compatible Ascend/CANN environment. Device IDs in the logs are device indices, not hardware model names. Complete environment versions, input data, and test harnesses were not included in the supplied directories.

Some search states contain absolute paths and a `test_command` from the original environment. They are preserved as provenance and require environment-specific adjustments elsewhere. This repository alone does not establish reproducibility of the recorded latency or correctness checks.

Import validation confirmed that all 32 original files were byte-identical to the supplied files. Translation validation checks Python syntax and executable AST equivalence after excluding docstrings, JSON syntax, README links, translated report/node equality, and the mapping to original evaluation records. NPU execution, correctness testing, and latency remeasurement have not been performed. Original copyright notices are preserved.
