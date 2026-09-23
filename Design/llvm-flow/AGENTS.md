# LLVM CFG standalone workflow

- The canonical product is `llvmcfg-standalone.html` plus `assets/llvmcfg/`. Read `docs/frontend-architecture.md` for ownership and loading order.
- The user explicitly authorized CSS/JS/data extraction on 2026-09-07. Edit `assets/llvmcfg/styles/custom.css` for visual changes and `assets/llvmcfg/scripts/review.js` for review behavior; source highlighting lives in `scripts/code-enhancements.js`.
- Keep entry resource order and bridge initialization intact. Preserve existing compiled runtime behavior; do not rebuild/export or copy frontend build output over the canonical application.
- The default exporter writes `llvmcfg-standalone.generated.html`; only an explicit `--force` may target the canonical standalone file.
- Before editing, inspect current changes and preserve work from other sessions. Use targeted patches instead of whole-file rewrites.
- Validate with static checks by default. Do not run browser automation or screenshots unless the user asks.

## Persistent product boundaries

- Before editing, read `docs/ui-contract.md`. It records confirmed requirements, rejected directions, and pending acceptance; do not treat implementation notes as user approval.
- Replacing sample data does NOT authorize changes to layout, component structure, controls, navigation, or workflow. Modify bindings inside existing components by default.
- A request scoped to one stage does NOT authorize redesigning another stage or the whole product. Preserve the four-stage Source → Transformation → Generated → Execution flow and the original CFG visualization engine.
- Missing data or compiler integration does NOT authorize deleting, replacing, or disabling an interaction. Explain the gap and obtain direction before changing interaction availability. Keep source facts, candidate settings, and runtime observations separate.
- “Restore UI” means restore confirmed controls, placement, behavior, and styling against an identified reference, not construct a similar replacement. If the reference cannot be identified, ask rather than inventing it.
- Reuse original components and icon classes. Do not create replacement panes, overlays, or standalone explanatory workspaces unless explicitly requested.
- Start edits with one concise statement of what changes and what stays unchanged. Ask only when an unresolved choice would materially change scope; do not turn routine fixes into approval ceremonies.
- A proposal, agent-generated plan, passing test, or “implemented” note is not user acceptance. Never silently promote an unaccepted UI to the baseline.
- Latest explicit user instructions govern scope. Historical plans do not authorize reverting a later correction. Mark superseded decisions as historical rather than treating every document as simultaneously current.

## Handoff and regression gate

- After relevant standalone changes run `rtk node docs/evidence/check-ui-contract.cjs` from this project; it runs the retained static JS/data/UI-contract and AST tests. Do not weaken checks to make a regression pass. Update expectations only for an explicitly authorized behavior change, documenting why.
- Tests must retain coverage of original source controls, candidate/evidence isolation, stage navigation and CFG preservation, extension initialization before/after subscription, and bounded action icons.
- Static checks do not validate browser layout, clipping, edge routing, or actual interaction. Report “static checks passed; visual/interactive acceptance pending” when that is all that was verified. Never claim “UI restored/visually accepted” solely from syntax or element-tree tests.
- Respect the no-browser-validation default even if broader workflow guidance recommends a smoke test. Browser automation/screenshots require explicit user authorization.
- Maintain `docs/ui-contract.md` when the user confirms or rejects a product decision. Keep the rule set concise; retrospective narrative belongs in supporting docs, not duplicated instruction files.
