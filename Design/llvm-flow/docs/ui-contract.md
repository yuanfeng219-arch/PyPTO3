# UI contract and acceptance record

Last updated: 2026-09-07. Scope: canonical `llvmcfg-standalone.html` and `assets/llvmcfg/`.

## Authority and baseline status

This document preserves explicit user requirements across sessions, not agent design preferences. Latest explicit user direction takes precedence. Read with `AGENTS.md` before changes.

**No exact user-approved baseline file/commit/hash has been established in this session.** The user repeatedly required the pre-redesign product experience, but the current file has not received overall visual acceptance. Do not label it approved, revert the dirty tree, or select an old commit by guesswork.

When a future restoration needs pixel/behavior comparison, identify the user-confirmed reference first. A saved candidate copy or screenshot is not an approved baseline until the user confirms it. Record that confirmation and reference here.

## Confirmed requirements to preserve

| Area | Required contract |
| --- | --- |
| Artifact | Maintain canonical HTML entry and extracted assets; no source rebuild/export overwrite. |
| Product | Original LLVM Flow-derived control-flow product, not merely an explanatory Pass Diff tool. |
| Framework | Four stages, original pane layout, existing navigation and CFG engine. |
| Sample replacement | Deepseek V4 is the requested default; Stage 1 source analysis and Stage 2 use the same case, no Stage 2 case picker. Data replacement does not authorize redesign. |
| Source code | Original code panel, full source with line numbers, search, control-region location, code/constraint-review tabs. |
| Constraint review | Original numeric length field, quick values, unroll-list factor buttons, read-only intent summary, constraint rows, and bottom action-card layout. Do not replace with parameter/function dropdown cards. |
| Bottom cards | Code and review bottom action cards remain anchored at pane bottom while main content scrolls. |
| Graph | Readable visual relationships and comparable compact nodes; consistent edge labels/tooltips, labels mapped to the correct targets, no clipped card content. Reuse existing renderer; no unrelated redesign. |
| Evidence | Dynamic length is operator-specific. Do not equate candidate values with runtime facts or infer Task/performance results from unroll factors. No unsupported “complete tail coverage.” |
| Acceptance | No browser automation or screenshots by default. Static pass and user visual acceptance are separate. |

## Current implementation, not approval

- Latest correction: both graph-side wrappers and the graph grid have no background fill; BEFORE/AFTER `.react-flow__pane` surfaces use `#F8F8F8` with a shared sparse 32px screen-space dot pattern. Header fill remains `#F8F8F8`.

- Latest user direction after moving to `/Users/yin/PyPTO3-main/Design/llvm-flow`: remove AFTER graph frame/canvas gray fill; CFG headers use `#F8F8F8`. Keep existing spacing and corners. Visual acceptance pending.

- 2026-09-07 user requested 8px outer graph spacing, 16px corners, black/white selected region buttons; removal of full-function/fit buttons and timeline search, filter, notice and totals; purple text-diff tags and two-line pass-purpose descriptions with full hover text. Entries without both graph snapshots are explicitly disabled with gray titles. This authorization supersedes earlier availability requirements for those entries only.

- 2026-09-07 user requested `cfg-frame` background `#F8F8F8`, removal of graph explanatory copy and the Stage 2 center title, and removal of Vertical/Horizontal controls. This supersedes the earlier requirement to retain direction controls. Visual acceptance remains pending.

- 2026-09-07 user authorized backup and structural separation of CSS, JS and data. Canonical URL is preserved; source ownership is documented in `frontend-architecture.md`. This does not establish visual acceptance.

- 2026-09-07 latest user direction supersedes bare 45×45 `%x` nodes: show readable operation summaries, verified unroll correspondence and path information; keep identifiers secondary, edge labels small and concise, and source code in expanded details. Preserve dragging and distinct forward/backward ports and the four-stage framework. Semantic-node implementation is pending visual/interactive acceptance.

- Deepseek source: `passes_dump/00_frontend.py`, 1793 lines / 24 functions; 55 parameters; actual dynamic input value unknown.
- Stage 2: 51 adjacent comparisons, with reviewed control-flow cases 02/03/09. These are existing compilation artifacts, not results of editing a UI draft.
- Constraint controls are restored in the original class/layout vocabulary. Candidate length starts at 16; factor draft is separate from source evidence. The initial draft is not an observed runtime configuration.
- The bottom “运行 Diff” action is currently disabled because candidate compilation/target-loop binding is not connected. This is a disclosed implementation limitation, **not approval to disable other controls or a confirmed final UX decision**.
- Stage 3/4 currently show unconnected-evidence states inside existing panes. This does not establish that developer data lacks those artifacts, nor does it constitute acceptance of the downstream UX.
- Last targeted fix: action SVG now reuses the original icon class. Static regression passed; visual acceptance remains pending.

## Rejected / superseded directions

- A separate explanatory workspace replacing the main product was rejected. `llvmcfg-pass-diff-explainer.html` is a historical copy, not the mainline baseline.
- Stage 2-only case selection was superseded by the user's default-case replacement request.
- Whole-pane Stage 1 replacement and read-only parameter/function dropdown cards replacing t/unroll interaction were explicitly rejected.
- Earlier design plans and implementation logs remain historical evidence. Their “implemented” status does not override these corrections.

## Pending decisions

1. Exact approved visual/reference baseline: not yet recorded.
2. Simpler demo: fixed two-iteration inner unroll of `indexer_topk_group_wave` was recommended, **not selected by the user**. Do not switch the sample automatically.
3. Real target-loop binding, candidate compilation, and final Diff availability: unresolved. Do not invent generated results to complete the interaction.
4. Overall visual/interactive acceptance of restored UI: pending.

## Acceptance gate

Run `rtk node docs/evidence/check-ui-contract.cjs` from the repository root. This checks static UI contracts and evidence, not pixels or actual browser behavior.

Before handoff compare the patch to requested scope and report: changed areas, preserved areas, tests run, unverified behavior. Do not update this document to “accepted” without explicit user confirmation.

## New-session verification

Open a new session in `/Users/yin/gitcode/llvm-flow`. First ask, without editing or browser use:

> Read AGENTS.md and docs/ui-contract.md. List loaded instruction paths; explain whether replacing sample data authorizes UI changes, whether missing compiler integration authorizes disabling controls, whether a confirmed baseline exists, which demo choice remains pending, and which validation is permitted. Do not change files.

Expected: data-only scope by default; no unilateral control changes; no approved snapshot yet; no automatic demo switch; static tests only unless browser validation is explicitly authorized.

Verification performed on 2026-09-07: a fresh `codex exec --ephemeral --sandbox read-only` session, started in the project directory without this conversation history, read both project files and correctly answered all five questions above. It distinguished the currently disabled Diff action from approved UX. This verifies document retrieval and comprehension in that run, not guaranteed future compliance or visual acceptance. The static regression gate also passed.
