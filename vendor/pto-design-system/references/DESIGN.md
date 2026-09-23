---
version: alpha
name: PTO
description: Dark-first developer workstation design system for PTO modules, graph surfaces, operator workbenches, and review previews. Model architecture gallery pages use the openPangu-2.0-Flash shared model-graph profile: modelArchitectureColormap(), true neutral tensor fills, transparent parent containers, consistent node typography, left-side Parameter/Weight inputs, right-side State/Cache auxiliaries, readable fit, and .pto-model-architecture-stage.
colors:
  primary: "#4369EF"
  primary-hover: "#5A92E6"
  accent: "#7C8DB8"
  success: "#04D793"
  warning: "#FFAA3B"
  danger: "#FF4B7B"
  background: "#101010"
  background-elevated: "#141414"
  surface-1: "#161616"
  surface-2: "#1C1C1C"
  surface-3: "#262626"
  surface-4: "#313131"
  foreground: "#E6E6E6"
  foreground-secondary: "#999999"
  foreground-muted: "#666666"
  border-subtle: "#202020"
  border-default: "#292929"
  border-strong: "#3A3A3A"
  primary-foreground: "#F5F9FF"
typography:
  display:
    fontFamily: "Inter, Source Han Sans SC, PingFang SC, Noto Sans SC, sans-serif"
    fontSize: 28px
    fontWeight: 700
    lineHeight: 1.25
    letterSpacing: 0
  title-1:
    fontFamily: "Inter, Source Han Sans SC, PingFang SC, Noto Sans SC, sans-serif"
    fontSize: 20px
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: 0
  title-2:
    fontFamily: "Inter, Source Han Sans SC, PingFang SC, Noto Sans SC, sans-serif"
    fontSize: 16px
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: 0
  body:
    fontFamily: "Inter, Source Han Sans SC, PingFang SC, Noto Sans SC, sans-serif"
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: 0
  body-sm:
    fontFamily: "Inter, Source Han Sans SC, PingFang SC, Noto Sans SC, sans-serif"
    fontSize: 12px
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: 0
  label:
    fontFamily: "Inter, Source Han Sans SC, PingFang SC, Noto Sans SC, sans-serif"
    fontSize: 12px
    fontWeight: 500
    lineHeight: 1.2
    letterSpacing: 0.5px
  micro:
    fontFamily: "Inter, Source Han Sans SC, PingFang SC, Noto Sans SC, sans-serif"
    fontSize: 11px
    fontWeight: 500
    lineHeight: 1.2
    letterSpacing: 0.5px
  mono:
    fontFamily: "JetBrains Mono, Fira Code, Consolas, monospace"
    fontSize: 12px
    fontWeight: 500
    lineHeight: 1.4
    letterSpacing: 0
spacing:
  1: 4px
  2: 8px
  3: 12px
  4: 16px
  5: 20px
  6: 24px
rounded:
  sm: 6px
  md: 8px
  lg: 12px
  xl: 16px
  pill: 999px
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "#000000"
    typography: "{typography.label}"
    rounded: "{rounded.lg}"
    height: 36px
    padding: 12px
  button-primary-hover:
    backgroundColor: "{colors.primary-hover}"
    textColor: "#000000"
    typography: "{typography.label}"
    rounded: "{rounded.lg}"
    height: 36px
    padding: 12px
  primary-foreground-swatch:
    backgroundColor: "{colors.primary-foreground}"
    textColor: "{colors.background}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: 8px
  button-solid:
    backgroundColor: "{colors.foreground}"
    textColor: "{colors.background}"
    typography: "{typography.label}"
    rounded: "{rounded.lg}"
    height: 36px
    padding: 12px
  button-secondary:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.foreground}"
    typography: "{typography.label}"
    rounded: "{rounded.lg}"
    height: 36px
    padding: 12px
  button-ghost:
    backgroundColor: "{colors.surface-1}"
    textColor: "{colors.foreground-secondary}"
    typography: "{typography.label}"
    rounded: "{rounded.lg}"
    height: 36px
    padding: 12px
  input:
    backgroundColor: "{colors.surface-1}"
    textColor: "{colors.foreground}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    height: 34px
    padding: 12px
  panel:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.lg}"
    padding: 16px
  panel-shell:
    backgroundColor: "{colors.background-elevated}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.xl}"
    padding: 16px
  card:
    backgroundColor: "{colors.surface-1}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.lg}"
    padding: 16px
  card-selected:
    backgroundColor: "{colors.surface-4}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.lg}"
    padding: 16px
  tag:
    backgroundColor: "{colors.surface-3}"
    textColor: "{colors.foreground-secondary}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    height: 20px
    padding: 8px
  metadata:
    backgroundColor: "{colors.surface-1}"
    textColor: "{colors.foreground-secondary}"
    typography: "{typography.body-sm}"
    rounded: "{rounded.sm}"
    padding: 8px
  muted-rule:
    backgroundColor: "{colors.foreground-muted}"
    height: 1px
  tag-accent:
    backgroundColor: "{colors.surface-3}"
    textColor: "{colors.accent}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    height: 20px
    padding: 8px
  status-success:
    backgroundColor: "{colors.surface-3}"
    textColor: "{colors.success}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    height: 20px
    padding: 8px
  status-warning:
    backgroundColor: "{colors.surface-3}"
    textColor: "{colors.warning}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    height: 20px
    padding: 8px
  status-danger:
    backgroundColor: "{colors.surface-3}"
    textColor: "{colors.danger}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    height: 20px
    padding: 8px
  divider-subtle:
    backgroundColor: "{colors.border-subtle}"
    textColor: "{colors.foreground}"
    height: 1px
  divider-default:
    backgroundColor: "{colors.border-default}"
    textColor: "{colors.foreground}"
    height: 1px
  divider-strong:
    backgroundColor: "{colors.border-strong}"
    textColor: "{colors.foreground}"
    height: 1px
---

# PTO DESIGN.md

## 1. Visual Theme & Atmosphere

PTO is not a marketing site. It is a developer workstation for understanding what data becomes across source, pass, runtime, and hardware views.

The UI should feel:

- Technical, calm, and precise
- Dense enough for expert work, but never visually noisy
- Dark-first by default for graph and timeline work
- Capable of light mode, but never pastel, playful, or decorative
- Consistent across modules so the user feels they are moving through one toolchain, not several disconnected pages

The visual benchmark is closer to developer-focused dark products such as Cursor, OpenCode AI, Warp, and Resend than to consumer dashboards.

Do not optimize for “pretty card gallery” aesthetics.
Optimize for:

- traceability
- evidence visibility
- fast scanning
- state clarity
- low visual drift across modules

## 2. Product Surfaces

PTO has three major surface types. They must not be styled as if they are the same thing.

### A. Workbench Surface

Used by:

- `op-ide-assistant`
- `op-ide-assistant-v2`
- future operator workbenches

Characteristics:

- three-column or split-pane workflows
- form inputs, code panes, agent chat, action bars
- clear pane affordances without stacked card chrome
- IDE-like pages inherit the shared `ide-frame` skin: 100%-intensity multi-point gradient/aura background, 80% translucent blurred pane wells, 72% pane-header fill, transparent top chrome, and floating playback chrome when playback is present

### B. Visualization Surface

Used by:

- `pass-ir`
- `swimlane`
- `mem_viewer`
- `execution-overlay`

Characteristics:

- canvas / SVG / minimap / overlays
- neutral dark stage
- restrained chrome
- color reserved for meaning, not decoration

### C. Preview / Review Surface

Used by:

- `design-system-preview.html`
- component previews
- design system review pages

Characteristics:

- explanatory, not product-like
- token demonstrations
- side-by-side comparisons

## 3. Pattern Sources

PTO has reusable graph, timeline, and hardware visualization primitives that are more specific than base components. These are pattern sources, not decorative examples.

The shared pattern registry is:

- `patterns/patterns.json`

Current pattern sources:

- `patterns/swimlane-task/pattern.json`
- `patterns/swimlane-task/pattern.html`
- `patterns/swimlane-task/pattern.css`
- `patterns/swimlane-task/pattern.js`
- `patterns/memory-architecture/pattern.json`
- `patterns/memory-architecture/pattern.html`
- `patterns/memory-architecture/pattern.css`
- `patterns/memory-architecture/pattern.js`
- `patterns/memory-reuse-viewer/pattern.json`
- `patterns/memory-reuse-viewer/pattern.html`
- `patterns/memory-reuse-viewer/pattern.css`
- `patterns/memory-reuse-viewer/pattern.js`
- `patterns/hardware-architecture-viewport/pattern.json`
- `patterns/hardware-architecture-viewport/pattern.html`
- `patterns/hardware-architecture-viewport/pattern.css`
- `patterns/hardware-architecture-viewport/pattern.js`
- `patterns/aic-core-object/pattern.json`
- `patterns/aic-core-object/pattern.html`
- `patterns/aic-core-object/pattern.css`
- `patterns/aic-core-object/pattern.js`
- `patterns/aiv-core-object/pattern.json`
- `patterns/aiv-core-object/pattern.html`
- `patterns/aiv-core-object/pattern.css`
- `patterns/aiv-core-object/pattern.js`
- `patterns/pass-ir-graph-node/pattern.json`
- `patterns/pass-ir-graph-node/pattern.html`
- `patterns/pass-ir-graph-node/pattern.css`
- `patterns/pass-ir-graph-node/pattern.js`
- `patterns/model-graphviz/pattern.json`
- `patterns/model-graphviz/pattern.html`
- `patterns/model-graphviz/pattern.css`
- `patterns/model-graphviz/pattern.js`
- `patterns/model-training-graphviz/pattern.json`
- `patterns/model-training-graphviz/pattern.html`
- `patterns/model-training-graphviz/pattern.css`
- `patterns/model-training-graphviz/pattern.js`
- `patterns/model-architecture-3d-deck/pattern.json`
- `patterns/model-architecture-3d-deck/pattern.html`
- `patterns/model-architecture-3d-deck/pattern.css`
- `patterns/model-architecture-3d-deck/pattern.js`
- `patterns/model-parallel-rank-deck/pattern.json`
- `patterns/model-parallel-rank-deck/pattern.html`
- `patterns/model-parallel-rank-deck/pattern.css`
- `patterns/model-parallel-rank-deck/pattern.js`
- `patterns/model-architecture-training-sidecar/pattern.json`
- `patterns/model-architecture-training-sidecar/pattern.html`
- `patterns/model-architecture-training-sidecar/pattern.css`
- `patterns/model-architecture-training-sidecar/pattern.js`
- `patterns/training-metrics-chart/pattern.json`
- `patterns/training-metrics-chart/pattern.html`
- `patterns/training-metrics-chart/pattern.css`
- `patterns/training-metrics-chart/pattern.js`
- `patterns/tensor-volume-canvas/pattern.json`
- `patterns/tensor-volume-canvas/pattern.html`
- `patterns/tensor-volume-canvas/pattern.css`
- `patterns/tensor-volume-canvas/pattern.js`
- `patterns/workbench-shell/pattern.json`
- `patterns/workbench-shell/pattern.html`
- `patterns/workbench-shell/pattern.css`
- `patterns/workbench-shell/pattern.js`
- `patterns/ide-frame/pattern.json`
- `patterns/ide-frame/pattern.html`
- `patterns/ide-frame/pattern.css`
- `patterns/ide-frame/pattern.js`
- `patterns/ide-frame/vscode.css`
- `patterns/floating-playback-control/pattern.json`
- `patterns/floating-playback-control/pattern.html`
- `patterns/floating-playback-control/pattern.css`
- `patterns/floating-playback-control/pattern.js`

Rules:

- preview pages may wrap a pattern for layout, but must not redefine the pattern's internal classes
- pattern preview wrappers must be borderless: section/card/stage/embed/iframe layers may provide spacing only, not stacked borders, shadows, or rounded frame chrome; the pattern itself owns any visual boundary it needs
- agents must read the pattern JSON before reusing a graph or timeline primitive
- every pattern must expose a stable consumer entry: CSS class names for static visual primitives, and a `window.Pto*Pattern` or `window.Pto*` helper for DOM/SVG/canvas generation, drag behavior, zoom, routing, persistence, or synchronized state
- allowed overrides must be explicit CSS variables or geometry fields documented by the pattern
- forbidden overrides include internal radius, segment typography, divider shadows, and state rules unless the pattern source itself is updated

For swimlane task bars, reuse pattern id `swimlane-task-bar`. The canonical source is the canvas renderer, task colormap helper, and hover tooltip helper in `patterns/swimlane-task/pattern.js`, aligned to `pypto-swimlane-perf-tool/js/swimlane.js` and the PMU swimlane `.sl-tooltip` interaction. Call `createTaskColormap` for semantic, stitch, engine, or subgraph colors instead of local hash palettes. Do not rebuild it with DOM/CSS or rewrite segment math, color hashing, colormap selection, color mixing, border alpha, label truncation, or hover tip behavior locally.

For fixed-view three-dimensional tensor volumes, reuse pattern id `tensor-volume-canvas`. The canonical source is `patterns/tensor-volume-canvas/pattern.css` plus `patterns/tensor-volume-canvas/pattern.js`; consume it directly with `window.PtoTensorVolumeCanvas.render(canvas, scene, options)`, not through an iframe. The page owns Load3D, Conv, tiling, code-recovery, playback, and selection rules and maps them to the business-neutral `extent`, `axes`, and `voxels` scene contract. The renderer owns fixed projection, voxel geometry, depth sorting, DPR, ResizeObserver behavior, and semantic-token palette resolution. Use the returned `update`, `resize`, and `destroy` lifecycle, calling `resize()` after theme changes. Do not add camera, pan, zoom, drag interaction, a private palette, or product chrome to the Pattern.

For UB, L1, and L0C tensor lifetime or reuse analysis, reuse pattern id `memory-reuse-viewer`. The canonical source is `patterns/memory-reuse-viewer/pattern.css` plus `patterns/memory-reuse-viewer/pattern.js`; call `window.PtoMemoryReuseViewer.render(container, data, options)` with Memory.txt-derived buffers, tensor rows, source snippets, and CCE snippets. It owns the lifetime Canvas, offset/ruler geometry, peak usage, reuse links, tensor details, and controller lifecycle. Mount it directly in an unscaled panel and keep `resize()` and `destroy()` attached to the host. Do not embed it inside the architecture pan/zoom canvas, copy the renderer, mutate generated rectangles, or add a second bordered frame around the Pattern.

For memory hierarchy diagrams, reuse pattern id `memory-architecture-layout`. The canonical source is the hybrid renderer in `patterns/memory-architecture/pattern.js`, extracted from `mem_viewer` DOM, BPG grid logic, MTE overlay behavior, hardware hover tips, and `ascend-hardware-map-v3` path-focus hover. New hardware pages such as 950B should extend the preset/config surface there and call `attachPathFocusInteractions` or `setPathFocus` / `clearPathFocus` for node and flow-step hover instead of copying `mem_viewer/index.html`, redrawing route geometry, or reimplementing `is-path-focused` / route glow locally. When the diagram is embedded in an iframe or inspector pane, consume the graph-only pattern surface: wrap the rendered stage in the shared memory architecture viewport/sizer/canvas classes, call `createZoomController`, default the viewport to `0.6`, and rely on drag pan plus Command-wheel zoom. Visible zoom buttons are optional product chrome, not part of the required memory-architecture iframe.

For architecture-map host controls and message coordination, reuse pattern id `hardware-architecture-viewport`. Load its `pattern.css` and `pattern.js`, then call `window.PtoHardwareArchitectureViewport.mount(root, options)` for the shared dotted viewport, transparent title toolbar, detail visibility, zoom readout, iframe readiness, size synchronization, and standard hardware message protocol. Product pages provide architecture lists, preset mapping, and callbacks. Do not fork the toolbar grammar, add header metadata chips, replace the dotted stage, or invent alternatives to `hardware-ready`, `hardware-size`, `hardware-details`, `hardware-focus`, and `hardware-arch-change`.

For AIC internal object shells, reuse pattern id `aic-core-object`. The canonical source is `patterns/aic-core-object/pattern.js`, driven by preset data rather than handwritten page DOM. Extend the preset to add or resize intermediate buffers for 950B; do not restyle the object chrome or clone the generated markup in local pages.

For AIV internal object shells, reuse pattern id `aiv-core-object`. The canonical source is `patterns/aiv-core-object/pattern.js`, driven by preset data for DCache, ICache or ND-DNA Cache, UB, Scalar, SIMT, SIMD, Aux Scalar, and Vector. Keep AIC and AIV spacing, buffer grid math, and route treatment visually consistent when widening or rescaling architecture diagrams.

For Pass-IR graph nodes, reuse pattern id `pass-ir-graph-node`. The canonical source is `patterns/pass-ir-graph-node/pattern.js`, extracted from the real Pass-IR `css/style.css` and `js/renderer.js` contract. It covers op, tensor, incast, outcast, selected, compact, and group cards. Compact group cards keep the Group Node shell, title, count, and thumbnail stack while dropping detailed rows. Do not use the old static graph-node examples in `design-system-preview.html` or `pass-ir/node-preview.html` as source of truth.

For Tensor visualization, treat `tensor-volume-canvas` and `matrix-canvas` as sibling patterns in the `tensor-visualization` family. Use `tensor-volume-canvas` for fixed-view 3D voxel structure and `matrix-canvas` for exact 2D row/column relationships. Both patterns default to `neutral` tone when there is no selection, playback, current-processing state, or other explicit semantic emphasis; object identity such as input or output alone must not automatically introduce a semantic tone. `matrix-canvas` owns the API Visualizer Load3D A2 visual language, rounded matrix clipping, continuous square internal grid lines, row/column/written/padding/current states, bounded vertical and horizontal scrolling, modifier-wheel zoom, label LOD, hit testing, tooltip fallback, and span-based aggregate cells. Internal cells must not introduce per-cell gaps, rounded corners, or card-like outlines. It deliberately has no drag-to-pan interaction and must not allow wheel input to move the matrix beyond its scroll bounds like an infinite canvas. For large matrices, preserve the source matrix extent and let the business inject arbitrary `rowSpan` / `columnSpan` cells or a target `thumbnailRows` / `thumbnailColumns` count. Aggregate cells stay in the same continuous grid and add one centered rounded marker; `summary.intensity` controls marker strength and selected cells use the full primary fill. Consumers may generate these cells with `window.PtoMatrixCanvas.createAggregatedCells`. Do not merge the 2D and 3D renderer APIs, flatten thumbnail-grid dimensions into the source shape, or copy canvas geometry into product pages.

For TorchVista / model Graphviz views, reuse pattern id `model-graphviz`. The canonical visual contract is `patterns/model-graphviz/pattern.js`, aligned to `patterns/model-graphviz/graphviz/deepseek_v32_source_graph.html`; `patterns/model-graphviz/pattern.html` iframes the real DeepSeek V3.2 source page for parity preview instead of redrawing a simplified sample. The source/config schema asset bundle also includes DeepSeek V3.2, Qwen-7B, Gemma, openPangu-2.0-Flash, and local Pangu MoE previews/schemas under `patterns/model-graphviz/assets/`. The old source preview is now colocated under the model-graphviz pattern and is not a separate top-level design-system pattern reference. The contract covers DeepSeek V3.2 default depth-2 folding, dark graph stage, white edges and arrowheads, full-pill module/op nodes with `height / 2` radius, rounded-rect tensor nodes for Input/Output/MTP/Constant/Parameter/Tensor, strict semantic module hierarchy, top-right fold controls on expanded clusters, right-inset expand controls on collapsed modules, centered type labels, semantic-first colormap defaults for common op roles with unknown keys expanded through the shared palette, the shared `modelArchitectureColormap()` profile and `.pto-model-architecture-stage` treatment derived from openPangu-2.0-Flash for model architecture gallery previews, stronger light-mode node fills, transparent expanded parent containers, explicit `sem:*` op colors before parent fallback, folded representative edge dedupe, stable expand/collapse viewport anchoring, no graph shadows, and no-border clicked-operator preview cards with dark code surfaces. Node fill colors are the only data-viz exception; all non-node UI colors, typography, spacing, radii, borders, and code surfaces must remain token-derived. Do not render tensor nodes as ellipses or full pills, place expand icons outside node bounds, drift fold icons away from the cluster top-right corner anchor, add graph drop-shadows, add border strokes to operator preview cards, use light code panels, hard-code non-node UI styles outside tokens, draw detail modules as same-level peers of their collapsed semantic parent, draw multiple parallel representative edges between the same folded modules, recenter/refit the whole graph after expand/collapse, or create duplicate standalone parent nodes next to expanded clusters.

For model Graphviz report overlays, keep the overlay in the same `model-graphviz` pattern. Source pages should preserve Graphviz graph `margin=0.22`, `pad=0.38`, and expanded cluster `margin=36` so title pills and left-side priority tags have breathing room. Container report titles use a filled, same-priority pill centered against the container bbox, with the P0/P1/P2 tag inside the pill on the left and SVG geometric vertical centering. P0/P1/P2 report colors are red/orange/yellow; do not use blue for P2. Priority should be expressed through fill color, not border-only styling. If a centered title crosses an edge, lift it into a root overlay layer above edge strokes instead of adding a visible outline. Right-side report inspectors should reuse `panel-shell`; graph-side Mapped Nodes lists should focus the original graph, not just update inspector text.

For training-time model architecture overlays, reuse pattern id `model-training-graphviz`. It is a hybrid pattern extracted from `graph-evidence-workbench`: the base architecture must still render through `model-graphviz`, while training evidence hover panels, edge type tags, phase-driven selected nodes, related-node emphasis, and drag pan / Command-wheel zoom live in `patterns/model-training-graphviz/pattern.js`. Product pages should call `controller.setPhase({ nodeId, relatedNodeIds })` when training steps change; do not refit, recenter, iframe a page shell, dim non-selected operators with low opacity, or reimplement the hover/evidence overlay locally. Light mode selected states must use neutral emphasis rather than saturated blue borders. Source/config/profiling evidence belongs in hover copy or page-side evidence panels; graph-side inputs should be real tensors, parameters, states, modules, or operators. Do not draw tall source-evidence columns inside the architecture graph.

For depth-stacked model architecture views, reuse pattern id `model-architecture-3d-deck`. It is a hybrid pattern extracted from `pangu-moe-trainviz/op-rank-time-openpangu-flash-events.html`: `patterns/model-architecture-3d-deck/pattern.js` preserves the source page's 46 decoder layers, Dense L0-L1 / MoE L2-L45 split, DSA cadence, block-post layers, complete Sparse MLA operator graph, Dense/MoE branches, mHC paths, input objects, LM Head/logits path, and MTP L46-L48 tail. It also owns dark/light switching, iso/front/right transforms, drag rotate/direct pan, wheel zoom and fit, DOMMatrix-based PP volume projection, DP/PP/TP/EP badge anchoring, and right-view operator row labels. Consumers pass model configuration, parallel annotations, and callbacks through `window.PtoModelArchitecture3dDeck.render`; they must not simplify the semantic graph, copy generated DOM, freeze projected wireframes into static paths, hand-position parallel badges, or duplicate view/theme-state logic in product CSS. The original training visualization still uses its integrated renderer because its node selection is coupled to Three.js, inspector, timeline, profiling, and execution state; reconnect it in a dedicated integration pass rather than maintaining a partial bridge.

For Three.js model-to-rank placement, reuse pattern id `model-parallel-rank-deck`. Load `model-graphviz`, `model-architecture-3d-deck`, then `patterns/model-parallel-rank-deck/pattern.js`, supply its pinned local Three.js module, and call `window.PtoModelParallelRankDeck.render`. Its canonical demo partitions openPangu into TP2 × PP4 × CP1 × EP8 × EDP2 = 128 Ranks, generates explicit RankManifest ownership, and compiles all 1472 PP-owned Layer placements from the base pattern's shared `renderLayerScene()` output into one WebGL scene. It preserves every original Node, Cluster, Edge, label, and semantic ID in ModelSceneSpec/SceneIndex; overview LOD may suppress text but may not replace complete Layers with representative payloads, screenshots, textures, or glyphs. The combination pattern owns parallel topology layouts, single-click relationship selection, double-click Rank focus, Inspector, URL state, and communication overlays. Do not introduce a second Layer renderer or wording that promotes the demo topology to an official training configuration.

For training semantics over the canonical 3D deck right view, reuse pattern id `model-architecture-training-sidecar`. Load it after `model-graphviz` and `model-architecture-3d-deck`, then call `window.PtoModelArchitectureTrainingSidecar.render`. The sidecar owns forward/backward flows, parameter-gradient and optimizer lifecycles, loss composition, selectable metric bands, in-scene Layer slices, tooltips, and the fixed Inspector while leaving the base deck immutable. Extend data and `renderLayerDetail`; do not restyle or replace base deck internals.

For training metric series, anomalies, interest windows, and step cursors, reuse pattern id `training-metrics-chart`. Load its `pattern.css` and `pattern.js`, then call `window.PtoTrainingMetricsChart.render`. The SVG renderer owns axes, dual series, anomaly marks, brushing, cursor geometry, and controller lifecycle. Use visualization highlight ramps, `--danger` for anomalies, and `--primary` for brushes; do not reconstruct chart or interaction geometry locally.

For split-pane resize behavior, reuse pattern id `workbench-shell`. It is a resize kernel, not an application shell. The canonical contract is `patterns/workbench-shell/pattern.css` plus `patterns/workbench-shell/pattern.js`. Use `window.PtoWorkbenchShell.initResizablePanes`, `createSplitGutter`, or `initNestedResizablePanes` for horizontal, vertical, and nested splits with min sizes, default ratio sizes, localStorage persistence, callbacks, keyboard resize, and destroy lifecycle. Pane sizes divide the space left after fixed gutters, so drag bars must not push the last pane outside the frame. Do not put page background, pane background, topbar, brand, title, subtitle, badge, pane header typography, canvas controls, or fixed preview height assumptions into `workbench-shell`. `initWorkbenchShell` and `initCanvasControls` remain compatibility APIs only and are deprecated for new pages.

For PTO IDE framework shells, reuse pattern id `ide-frame`. The canonical contract is `patterns/ide-frame/pattern.css` plus `patterns/ide-frame/pattern.js`; load `patterns/ide-frame/vscode.css` for VS Code webviews. `ide-frame` owns a TRAE-like IDE frame: 4:3 host window ratio, the accepted standalone PTO IDE skin, transparent top chrome without divider borders or macOS traffic-light controls, dense 12-13px pane titles and mono metadata, with dense typography limited to chrome while pane body copy, descriptions, help, empty states, and Inspector prose remain on the 14px body role, utility window controls, optional activity rail, pane headers, pane-local preview tabs, domain toolbars, generic preview surfaces, renderer slots, inspector docks, optional floating playback mounting, and split initialization. The accepted standalone skin is defined in `pattern.css`: `--ide-gradient-bg` and `--ide-aura-bg` provide a 100%-intensity multi-point gradient/aura background; `--ide-frame-pane-fill` is `color-mix(... 80%, transparent)`; `--ide-frame-pane-header-fill` is 72%; `--ide-frame-pane-backdrop-filter` is `blur(18px) saturate(1.18)`; and `--ide-frame-pane-shadow` provides the soft pane lift. Product pages must not recreate those values locally or replace them with opaque panels. Standalone pages may hide the left activity rail with `hidden`, `data-activity-rail="hidden"`, or `data-hide-activity-rail="true"` when the product flow does not need search/git/terminal navigation. Standalone explorer starts at 300px by default through `data-pixel-sizes`, then the remaining panes divide the remaining width by ratio. Tabs belong inside the preview/editor pane, not in a separate top-level chrome band. Playback must use `floating-playback-control`; do not recreate a footer playback bar, scrubber, collapse state, or shell chrome in `ide-frame`. It must not ship business content: no file names, kernel names, graph node data, timeline lanes, trace data, inspector values, placeholder tab names, placeholder code rows, or default textual slot content. Consuming pages provide all domain content and renderers. It must call the `workbench-shell` resize helper and must not override `.pto-workbench-shell__*` internals. Use `data-host="standalone"` when PTO owns topbar, navigation, preview-pane tabs, and status strip; use `data-host="vscode-webview"` when VS Code owns explorer, editor tabs, search, git, terminal, settings, command palette, keybindings, global status, notifications, progress, theme colors, and baseline fonts.

For floating playback controls, reuse pattern id `floating-playback-control`. The canonical contract is `patterns/floating-playback-control/pattern.css` plus state helpers in `pattern.js`, extracted from the `mem_viewer` bottom toolbar. Pages own step data, timers, and renderer updates, but use `.pto-floating-playback*`, `window.PtoFloatingPlaybackControl.init`, and `initScrubberHover` for chrome, collapse state, collapsed play/pause icon, and scrubber hover behavior. Do not redefine floating-shell styling, range thumb, tooltip, or collapse synchronization locally.

Current extraction progress:

- `swimlane-task-bar`: shared canvas renderer and hover tooltip helper registered and previewed
- `tensor-volume-canvas`: shared fixed-view Canvas renderer registered and previewed with NCHW/A1 fixtures, padding/window/current/skipped states, responsive DPR resize, and a direct-embedding controller API
- `memory-reuse-viewer`: shared UB/L1/L0C tensor lifetime and reuse renderer registered and previewed with theme bridge, unscaled direct embedding, and resize/destroy lifecycle
- `memory-architecture-layout`: shared full-stage 950B renderer registered and previewed with graph-only iframe, default 60% zoom, drag pan, and Command-wheel zoom
- `hardware-architecture-viewport`: shared architecture host toolbar, dotted viewport, detail/zoom controls, size synchronization, and hardware message protocol registered and previewed
- `aic-core-object`: shared config-driven object renderer registered and previewed
- `aiv-core-object`: shared config-driven object renderer registered and previewed
- `pass-ir-graph-node`: shared hybrid renderer registered and previewed; original Pass-IR business page still uses its local renderer until a separate integration pass is approved
- `tensor-volume-canvas`: shared fixed-view 3D Tensor renderer registered and previewed for voxel structure and Load3D state
- `matrix-canvas`: shared interactive 2D Matrix Tensor renderer registered and previewed with detail cells, readable label LOD, bounded horizontal/vertical scrolling, and business-injected aggregate thumbnail counts
- `model-graphviz`: shared SVG visual renderer registered and previewed; original TorchVista Graphviz page still owns DOT generation, D3 zoom, popup behavior, and graph reloads until a separate integration pass is approved
- `model-training-graphviz`: shared model-graphviz training overlay renderer registered and previewed; training products should pass graph/evidence/phase data instead of embedding a second Graphviz page shell
- `model-architecture-3d-deck`: shared hybrid CSS 3D renderer registered and previewed with openPangu-2.0-Flash, projected PP partitions, DP/PP/TP/EP annotations, and iso/front/right views; the original training product remains on its integrated local renderer pending a dedicated state-bridge refactor
- `model-parallel-rank-deck`: shared 128-rank Three.js composition registered and previewed with complete GPU-compiled Layer records, deterministic PP/TP/EP/EDP manifests, topology transforms, relationship selection, double-click Rank focus, and Inspector-backed semantic details
- `model-architecture-training-sidecar`: shared non-destructive training overlay registered and previewed with forward/backward flows, gradients, optimizer state, loss composition, metric bands, Layer slices, and fixed Inspector details
- `training-metrics-chart`: shared SVG training metrics renderer registered and previewed with dual series, anomaly marks, interest-window brushing, and step cursor synchronization
- `workbench-shell`: shared split / resize kernel registered as a secondary dependency; old visual shell APIs are compatibility-only and it is not shown as a top-level preview card
- `ide-frame`: shared PTO IDE framework shell registered and previewed with standalone and VS Code webview host modes; content-free slots only; standalone host mode includes the shared gradient/aura + translucent blurred pane skin by default
- `floating-playback-control`: shared DOM/CSS playback toolbar registered, previewed, and adopted by Memory Viewer v1/v2

## 4. Color Palette & Roles

### Core UI Palette

Use token source of truth from:

- `tokens/foundation.css`
- `tokens/semantic.css`
- `tokens/components.css`

The design rule is:

- neutral surfaces carry layout
- accent colors carry meaning
- never use large-area saturated fills for normal UI panels

### Semantic Roles

- `primary`: the main interactive accent
- `accent`: auxiliary emphasis
- `success`: healthy / positive result
- `warning`: caution / pending concern
- `danger`: breakage / invalid / destructive

### Data Visualization Colors

Visualization colors are a separate system from general UI colors.

Examples:

- pass-ir node types
- swimlane semantic labels
- mem-viewer storage tier ramps

These colors may be vivid, but only inside charts, graphs, chips, legends, and traces.

Do not reuse visualization colors as generic panel backgrounds.

## 5. Typography Rules

### Type Families

- Display: `Space Grotesk`
- Body UI: `Inter`
- Code / metrics / IDs: `Space Mono` or the shared mono token family

### Typography Intent

- Titles should be compact and controlled, not editorially oversized
- Labels and metadata should be explicit and easy to scan
- Code and numeric identifiers should always use mono
- Body copy, descriptions, inspector prose, help text, and empty-state explanations use 14px with 1.5 line height
- Dense controls, form labels, tables, trees, code, and metadata never render below 12px
- 11px is reserved for short micro labels, badges, and visualization ticks; it must not carry prose
- Text below 11px is forbidden outside an explicitly documented zoomable data-visualization exception with a readable tooltip or detail view
- Do not solve overflow by shrinking content with `zoom`, `transform: scale(...)`, or a smaller body token; change the layout, wrapping, truncation, or scrolling strategy

### Hierarchy

- Page title: strong but compact
- Panel title: one clear level below page title
- Section label: quiet, uppercase or mono where appropriate
- Metadata: muted, never low-contrast to the point of illegibility

## 6. Component Stylings

### Buttons

Buttons should be grouped into a small, stable set:

- primary action
- secondary action
- ghost/icon action
- selected toggle / segmented state

Do not invent page-local button archetypes unless they are first proven in preview pages and absorbed into tokens.

### Inputs

Inputs must feel like tool inputs, not consumer rounded pills by default.

Rules:

- clear boundaries
- readable placeholder text
- focused state must be obvious
- repeated row layouts must align as a grid

### Panels

All modules should use a consistent shell language:

- rounded container
- consistent border alpha when a border is necessary
- at most one visible boundary around a surface; do not stack parent, card, stage, embed, and iframe borders around the same content
- stable elevation hierarchy
- similar header/body/footer rhythm

### Code Editors

Code panes are not generic cards.

They should read as editor surfaces:

- fixed mono typography
- line-number gutter
- horizontal scroll for long lines
- predictable top chrome
- quiet syntax colors

## 7. Layout Principles

### Pane Logic

For three-column workbenches:

- left = fixed utility rail or form rail
- center = primary content, stretches
- right = fixed assistant / detail rail

If a module breaks this pattern, it must be intentional and documented.

### Spacing

Use token spacing consistently.

Avoid:

- random 7px / 13px / 19px spacing
- one-off panel padding overrides unless strictly necessary

### Alignment

Repeated structures must align to visible grids:

- form rows
- button rows
- stat cards
- legends

Misalignment is one of the fastest ways PTO looks inconsistent.

## 8. Depth & Elevation

Dark PTO should rely on:

- subtle border contrast
- low-to-medium elevation
- restrained glass/blur usage
- the only approved large IDE blur/gradient combination is the shared `ide-frame` skin; consume its variables instead of layering extra module-local gradients, glass cards, or opaque panes
- Liquid Glass surfaces only through the shared `data-liquid-glass` adapter, and only for floating auxiliary overlays that sit above content. This follows Apple HIG Materials guidance: material is for controls/navigation without hiding underlying content, and larger material regions need stronger opacity for legibility. Native `backdrop-filter: blur(...) saturate(...)` must be visible first; static highlight layers are additive only. Do not add cursor-following glow.

Do not stack multiple styling tricks at once:

- strong blur
- bright gradients
- high drop shadow
- saturated fills
- nested border frames

Pick one emphasis mechanism, not four.

## 9. Do’s and Don’ts

### Do

- keep neutral surfaces consistent across modules
- reserve saturated color for semantic meaning
- build preview pages before spreading a new visual pattern
- keep forms, panels, and code panes rhythmically aligned
- maintain both light and dark previews for shared system elements

### Don’t

- hardcode colors in module CSS when a token exists
- use inline styles for reusable UI states
- let each module invent its own panel chrome
- use visualization colors as generic UI decoration
- treat tokens as optional suggestions

## 10. Responsive Behavior

Desktop first, but responsive.

Rules:

- fixed side rails may collapse only below clear breakpoints
- center work area should remain dominant
- toolbars should wrap cleanly rather than overflow unpredictably
- chat input, form rows, and code tabs should remain usable on smaller widths

Do not solve narrow layouts by shrinking text until unreadable.

## 11. Design System Governance

### Source of Truth

The source of truth for implementation tokens is:

- `tokens/foundation.css`
- `tokens/semantic.css`
- `tokens/components.css`

Generated artifacts:

- `tokens/tokens.js`
- `tokens/tokens.json`

must be generated from those CSS sources with `python3 tokens/generate_tokens.py`, not edited by hand.

Before sharing the bundle, run `rtk node scripts/audit-theme.mjs` to check light/dark token coverage, hard-coded UI color counts in shared CSS, and baseline contrast ratios. Run `rtk node scripts/audit-typography.mjs` to verify typography token integrity, the 14px body baseline, and unapproved sub-11px text.

### Review Rule

Any new shared visual pattern must:

1. appear in a preview page
2. be named in tokens or component docs
3. be reused in at least one product surface

### Module Rule

A module is not considered “design-system integrated” until:

- it imports the shared token chain
- it avoids unapproved hardcoded UI colors
- its core UI patterns appear in preview or review material

## 12. Prompt Guide For Agents

When asking an AI agent to build PTO UI, use directions like:

- “Use PTO’s dark-first developer workstation style”
- “Keep left/right rails fixed width and center content flexible”
- “Use neutral panel shells and reserve vivid color for semantic signals”
- “Treat code panes as editor surfaces, not cards”
- “Align forms to a visible grid and avoid pill-like consumer inputs unless explicitly requested”

Avoid directions like:

- “make it more modern”
- “make it more premium”
- “make it more stylish”

Those usually increase inconsistency instead of reducing it.
