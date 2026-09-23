# CFG frontend structure

The canonical URL remains `/llvmcfg-standalone.html`. Since 2026-09-07 the entry references local assets; deploying or copying it requires the `assets/llvmcfg/` directory as well.

## Ownership

| Path | Responsibility |
| --- | --- |
| `llvmcfg-standalone.html` | Document metadata, root container, ordered resource loading |
| `assets/llvmcfg/styles/base.css` | Extracted existing application styles |
| `assets/llvmcfg/styles/custom.css` | Maintained UI styles, including semantic CFG nodes |
| `assets/llvmcfg/scripts/code-enhancements.js` | Source highlighting and DOM enhancements |
| `assets/llvmcfg/scripts/review.js` | Four-stage integration, review state, graph presentation and interactions |
| `assets/llvmcfg/data/deepseek-evidence.js` | Static compiler evidence and compressed CFG data |
| `assets/llvmcfg/vendor/` | Extracted PTO tokens, styles and shell dependencies |
| `assets/llvmcfg/runtime/` | Existing compiled React/React Flow/Dagre application and license notices |
| `docs/evidence/` | Static contracts and extraction tests |

Stylesheets retain their original cascade order. Classic scripts retain their original synchronous execution order: shell dependencies → compiled runtime → code enhancements → evidence → review integration. Evidence uses `window.LLVMCfgEvidence` to preserve immediate initialization without introducing an asynchronous fetch. The existing `LLVMStage2` and bridge interfaces are preserved.

The runtime is a preserved compiled dependency, not newly recovered component source. This migration does not claim to regenerate it from `llvm-flow-frontend/src`; that source tree has separate, preexisting changes and is not a verified equivalent of the current UI. Do not overwrite the canonical entry or assets using its exporter. Further decomposition of review logic can proceed at tested function boundaries.

## Local development

From the repository root:

```sh
rtk npm start
rtk npm run check
```

Requires Node.js 18+ and Python 3; no new npm packages are needed. Open `http://127.0.0.1:8790/llvmcfg-standalone.html`. An existing server on port 8790 can serve the extracted assets without restarting. Static checks resolve the real linked files, parse scripts, and retain source, graph and evidence checks. Browser acceptance is performed by the user unless explicitly requested.

## Backup and migration proof

The pre-migration single-file page is saved under `backups/before-asset-split-20260907-181724/`, with `SHA256SUMS`. Backups stay local and are excluded from Git. Open that saved HTML directly to inspect the prior artifact, or copy it back to the root entry for rollback. External assets may remain unused after rollback.

`rtk npm run verify:migration` verifies that all 12 original CSS/JS blocks, including extracted evidence, reassemble byte-for-byte to their recorded SHA-256 hashes. This is a migration snapshot check, separate from ongoing regression checks: future intentional code changes can invalidate the old hashes.

For distribution, preserve the relative directory structure of the entry and all assets. This page is now a multi-file static application; the historical filename remains for URL compatibility.
