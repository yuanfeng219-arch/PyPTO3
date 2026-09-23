# PTO Design System Map

Use this as the default classification when building a new module.

## Shared component families

- `Buttons`: `soft`, `solid`
- `Toggle`
- `Toggle Group`
- `Chip Filter`
- `Labels & Badges`
- `Card`
- `Data Viz Exempt`
- `Graph Node Pattern`
- `Swimlane Event Pattern`
- `Tensor Volume Canvas Pattern`
- `Memory Reuse Viewer Pattern`
- `Hardware Architecture Viewport Pattern`
- `Model Training & Parallel Patterns`
- `Training Metrics Chart Pattern`
- `IDE Frame Pattern`

The source of truth for target appearance is `/Users/yin/pto-design-system/design-system-preview.html`.

## Matching guidance

### Buttons

- `soft`: open, import, load, select, browse, local resource entry
- `solid`: primary workflow commit such as run, generate, apply, execute

Do not split equivalent entry actions into multiple visual styles. For example:

- `Open Pass Folder`
- `Open single JSON`
- `加载本地资源`

These should map to the same entry-button style.

### Toggle and toggle group

Use for mode switches, compact/semantic switches, before/after, TB/LR, and similar mutually exclusive controls.

Do not create pill-only one-off styles when the behavior is still a toggle group.

### Labels and badges

Status labels should be based on the neutral badge shape plus semantic color. Do not create unrelated tag shapes unless they are truly data-viz specific.

### Card

Use for inspector panels, action-list panels, popup detail cards, and compact in-place info blocks.

Do not treat every large surface as a card. Large canvases, viz shells, and IDE/workbench page frames belong outside this class.

During retrofit, do not keep a legacy card's private frame just because its colors were converted to tokens. Full borders, left accent rails, inset-left shadows, pseudo-element side bars, and side gradients should be removed unless the target PTO class explicitly owns that decoration. Dense right-side details should usually become `.inspector-section` plus one optional `.inspector-soft-card`, not a stack of bordered cards.

### Data Viz Exempt

Allowed exemptions:

- color maps
- graph node semantic accents
- swimlane event colors
- stitch colors
- dependency line/dot colors
- other visualization-only encodings

Even exempt patterns should be previewed and documented when they materially affect readability.

### Tensor Volume Canvas Pattern

Use `patterns/tensor-volume-canvas` for NCHW, A1, Load3D, Conv, tiling, and code-recovery views that need the shared fixed three-dimensional Tensor projection. Load `pattern.css` and `pattern.js` directly and call `window.PtoTensorVolumeCanvas.render(canvas, scene, options)`; the consuming page owns business rules and maps them to `extent`, `axes`, and `voxels`.

The Pattern owns projection, voxel geometry, depth sorting, DPR, ResizeObserver behavior, and semantic-token palette resolution. Do not reimplement those locally or add iframe embedding, camera interaction, a private palette, or product chrome.

### Memory Reuse Viewer Pattern

Use `patterns/memory-reuse-viewer` for UB, L1, and L0C tensor lifetime, buffer offset, peak usage, reuse links, and source/CCE correlation. Call `PtoMemoryReuseViewer.render` in an unscaled panel and keep its resize/destroy lifecycle. Do not mount it inside architecture pan/zoom, copy the Canvas renderer, mutate generated rectangles, or add another bordered shell.

### Hardware Architecture Viewport Pattern

Use `patterns/hardware-architecture-viewport` for the shared dotted architecture host, transparent toolbar, detail visibility, zoom readout, iframe readiness, size sync, and hardware message protocol. Product pages supply presets and callbacks; they do not fork the toolbar grammar or standard message names.

### Model Training & Parallel Patterns

- Use `model-training-graphviz` for training evidence and phase overlays on `model-graphviz`.
- Use `model-architecture-3d-deck` for the canonical openPangu depth stack and parallel badges.
- Use `model-parallel-rank-deck` for complete model-to-Rank Three.js placement and ownership inspection.
- Use `model-architecture-training-sidecar` for forward/backward flows, gradients, optimizer/loss semantics, metrics, and Layer telemetry without mutating the base deck.

### Training Metrics Chart Pattern

Use `patterns/training-metrics-chart` for shared SVG metric series, anomalies, interest-window brushing, and step cursors. Pass data and state through its controller instead of recreating chart geometry locally.

### IDE Frame Pattern

Use `patterns/ide-frame` for PTO IDE pages, workbench pages, multi-pane analysis tools, code/preview/inspector layouts, and pages that need floating playback. The pattern owns the default PTO IDE skin: 100%-intensity multi-point gradient/aura background, 80% translucent blurred panes, pane-header fill, pane shadow, transparent top chrome, activity rail, status strip, and playback mount.

Do not map an IDE/workbench page to generic `.panel-shell` or `.card-demo` containers. Product pages fill `ide-frame` slots with business content; they should not locally redefine the gradient background, pane opacity, backdrop blur, split gutter visuals, or floating playback chrome.

## Forbidden moves

- Creating a new `.xxx-btn` visual system when shared buttons already fit
- Hard-coding neutral grays, borders, shadows, or radii inside a new module
- Token-swapping old card borders or left highlight rails instead of removing them
- Rebuilding a PTO IDE page from generic panels instead of consuming `patterns/ide-frame`
- Reimplementing fixed Tensor voxel projection instead of consuming `patterns/tensor-volume-canvas`
- Copying Memory Reuse, model deck, Rank placement, training sidecar, or metric-chart renderers into product pages
- Introducing a new type scale disconnected from existing tokens
- Shipping unapproved new component visuals directly in the module
