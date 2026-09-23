# PTO Quick Reference

One-page cheat sheet of the most common tokens and classes. **For the full system, read `DESIGN.md`.** This file is what to glance at while writing markup.

All variables are defined in `tokens/foundation.css` and `tokens/semantic.css`. Class implementations live in `css/style.css`.

---

## Color tokens (semantic, dark theme default)

| Use for | CSS variable | Hex |
|---|---|---|
| Page background | `--background` | `#101010` |
| Elevated background (panels under nav) | `--background-elevated` | `#141414` |
| Card / inspector surface | `--surface-1` | `#161616` |
| Panel surface | `--surface-2` | `#1c1c1c` |
| Tag / chip surface | `--surface-3` | `#262626` |
| Selected card | `--surface-4` | `#313131` |
| Primary text | `--foreground` | white 90% |
| Secondary text | `--foreground-secondary` | white 60% |
| Muted / placeholder text | `--foreground-muted` | white 40% |
| Disabled text | `--foreground-disabled` | white 25% |
| Subtle border | `--border-subtle` | white 6% |
| Default border | `--border-default` | white 10% |
| Strong border | `--border-strong` | white 16% |
| Primary action | `--primary` | `#4369EF` |
| Primary hover | `--primary-hover` | `#5A92E6` |
| Success | `--success` | `#04D793` |
| Warning | `--warning` | `#FFAA3B` |
| Danger | `--danger` | `#FF4B7B` |
| Accent (auxiliary) | `--accent` | `#7C8DB8` |

**Never hard-code these hex values in module CSS — always reference the variable.**

---

## Spacing

| Token | Value | Use for |
|---|---|---|
| `--space-1` | 4px | tight icon gap, badge inner pad |
| `--space-2` | 8px | row gap inside cards, chip pad |
| `--space-3` | 12px | button h-padding, inline gap |
| `--space-4` | 16px | card / panel padding |
| `--space-5` | 20px | section gap |
| `--space-6` | 24px | page-level breathing |

---

## Radius

| Token | Value | Use for |
|---|---|---|
| `--radius-sm` | 6px | metadata, small chips |
| `--radius-md` | 8px | inputs |
| `--radius-lg` | 12px | buttons, cards, panels |
| `--radius-xl` | 16px | panel-shell, modals |
| `--radius-pill` | 999px | tags, badges, status pills |

---

## Typography

Use the body font stack (`--font-sans`) by default. Use `--font-mono` only for code, IDs, numeric readouts. Body copy and explanatory text use 14px; dense UI and form labels use at least 12px. Reserve 11px for short micro labels, badges, and data-viz ticks.

| Role | Size | Weight | Composite token |
|---|---|---|---|
| display | 28px | 700 | `--type-display` |
| title-1 | 20px | 600 | `--type-title-1` |
| title-2 | 16px | 600 | `--type-title-2` |
| title-3 | 14px | 600 | `--type-title-3` |
| body | 14px | 400 | `--type-body` |
| body-sm | 12px | 400 | `--type-body-sm` |
| label | 12px | 500 | `--type-label` |
| micro | 11px | 500 | `--type-micro` |
| mono | 12px | 500 | `--type-mono` |

Use the `--type-*` composite tokens with the `font` property. Use `--font-sans` and `--font-mono` only with `font-family`; use `--foreground`, `--foreground-secondary`, and `--foreground-muted` only for text color. Do not reuse one custom property across these categories.

---

## Common classes

### Buttons

| Class | When to use |
|---|---|
| `.btn` | Secondary / entry action — *open, load, import, browse, select*. Default subtle surface. |
| `.btn .btn-solid` | Primary / commit — *run, apply, generate, execute*. High contrast (white fill, dark text). |
| `.btn .btn-ghost` | Tertiary / icon-only. Transparent until hover. |
| `.btn .btn-icon` | Square icon button. Combine with `.btn-ghost` for toolbar icons. |
| `.btn .btn-sm` / `.btn-lg` / `.btn-compact` | Size modifiers. |
| `.btn.is-selected` | Toggle / filter selected state. |

Do **not** create new `.xxx-btn`, `.cta`, `.action-button` classes. Map everything to the four roles above.

### Tabs and segmented controls

| Class | When to use |
|---|---|
| `.tab-control`, `.tab-control-item` | Page-level navigation such as Workbench / Review / Health. Selected tab uses neutral surface, not primary action fill. |
| `.segmented-control` | Strong mode choice where selected state should be high-contrast. |
| `.segmented-control.segmented-control-muted` | Secondary in-panel filters such as Memory path / Compute path / Control path. |
| `.toolbar-control`, `.toolbar-readout` | Local canvas tools such as Fit / zoom readout. |

Do not use `.btn-solid` or white selected buttons for tabs. Reserve white fill for the primary commit action.

### Panels & cards

- `.workbench-frame`, `.workbench-frame-split`, `.workbench-pane` — structural split-pane classes for `workbench-shell`; visual fills belong to the consuming pattern
- `.workbench-pane-body-fill` — inner gray body fill for canvas or dense inspector content
- `.panel-shell` — outermost panel container with header / body
- `.panel-shell.panel-shell-quiet` — main workbench section surface; neutral fill, no visual border
- `.panel-shell-header`, `.panel-shell-title`, `.panel-shell-meta`, `.panel-shell-close`
- `.panel-shell-body`
- `.card-demo`, `.card-demo-header`, `.card-demo-title`, `.card-demo-description`, `.card-demo-content`, `.card-demo-footer` — content card primitives

When retrofitting an existing demo, remove legacy card frames that do not exist in these shared classes. Do not keep old full borders, left highlight rails, inset-left shadows, or pseudo-element side bars by changing their colors to PTO tokens.

### Pattern boundaries

- `workbench-shell` — resize kernel only. Use `PtoWorkbenchShell.initResizablePanes`, `createSplitGutter`, or `initNestedResizablePanes`; ratio sizes divide the space left after fixed gutters. Do not use it for page chrome, pane fill, pane titles, or canvas controls.
- `ide-frame` — PTO IDE framework shell. Use `.pto-ide-frame*` classes, `data-host="standalone"` or `data-host="vscode-webview"`, and `PtoIdeFrame.init`; product pages fill the slots. The host window defaults to 4:3 and standalone explorer starts at 300px via `data-pixel-sizes`. Standalone host mode owns the PTO IDE skin: 100%-intensity multi-point gradient/aura background, 80% translucent pane fill, 72% pane-header fill, `blur(18px) saturate(1.18)` pane backdrop filtering, pane shadow, transparent borderless top chrome, and window controls. Preview tabs live inside the preview/editor pane, not in a separate top chrome band. Playback uses `floating-playback-control` through `data-ide-floating-playback`; do not recreate a footer playback bar. Do not put business sample data, placeholder tab names, placeholder code rows, or default textual slot content in the pattern and do not override `.pto-workbench-shell__*` internals. Do not locally replace the shared IDE gradient or pane blur; update `patterns/ide-frame` variables first.
- `vscode.css` — maps VS Code `--vscode-*` variables and hides standalone chrome in webviews.
- `tensor-volume-canvas` — load its `pattern.css` and `pattern.js`, then call `window.PtoTensorVolumeCanvas.render(canvas, scene, options)`. The page converts business state to `extent`, `axes`, and `voxels`, then uses the returned `update`, `resize`, and `destroy` lifecycle. Do not copy renderer math, depth sorting, DPR, or resize logic; do not use an iframe, camera controls, a private palette, or product chrome.
- `memory-reuse-viewer` — call `PtoMemoryReuseViewer.render` with buffer and tensor lifetime data inside an unscaled panel. Keep `resize` and `destroy`; do not place it inside architecture pan/zoom, copy its Canvas renderer, mutate generated rectangles, or add a second frame.
- `hardware-architecture-viewport` — call `PtoHardwareArchitectureViewport.mount` for the dotted architecture host, transparent toolbar, detail/zoom controls, iframe readiness, size sync, and standard hardware messages. Product pages supply presets and callbacks, not local toolbar or protocol variants.
- `model-training-graphviz` — load after `model-graphviz`, call `PtoModelTrainingGraphvizPattern.render`, and drive phases through `controller.setPhase`. Do not add a second graph shell or recenter/dim the graph locally.
- `model-architecture-3d-deck` — call `PtoModelArchitecture3dDeck.render` for the canonical openPangu depth stack, transforms, pan/zoom, projected PP stages, and parallel badges. Do not duplicate model semantics or projection math.
- `model-parallel-rank-deck` — load after the 3D deck with pinned Three.js and call `PtoModelParallelRankDeck.render`. Preserve every Layer Node, Cluster, and Edge; do not replace Rank payloads with representative glyphs or screenshots.
- `model-architecture-training-sidecar` — load after the 3D deck and call `PtoModelArchitectureTrainingSidecar.render`. Add training data and Layer details without mutating the base deck.
- `training-metrics-chart` — call `PtoTrainingMetricsChart.render` for metric series, anomaly marks, brushing, and step cursors. Do not rebuild its SVG or interaction geometry locally.

### Navigation

- `.nav-bar`, `.nav-shell`, `.nav-left`, `.nav-right` — top navigation
- `.nav-pill`, `.nav-pill-menu`, `.nav-pill-item` — pill-shaped menu
- `.nav-stage-chip` — stage indicator chip
- `.nav-pass-dot` — small dot indicator

### Inspector / details

- `.detail-header`, `.detail-body`, `.detail-close`, `.detail-connections`, `.detail-conn-chip`
- `.inspector-rail` — scroll body for right-side detail rails; uses quiet padding and no card gap
- `.inspector-section`, `.inspector-section-head`, `.inspector-section-title`, `.inspector-section-kicker` — continuous detail sections separated by `--border-subtle`
- `.inspector-soft-card` — emphasized content inside a section, no full border, neutral soft surface
- `.inspector-soft-card.is-info` / `.is-warning` / `.is-danger` / `.is-success` — status emphasis via tinted fill, not a border

Use inspector sections for dense right rails. Do not wrap every metric row in a bordered card; prefer section dividers, neutral row surfaces, and one soft emphasized block for the current conclusion or next action.

### Toggles & control flow blocks (specialized)

- Control-flow viewer: `.cf-panel`, `.cf-node`, `.cf-toggle-btn`, `.cf-reopen-btn`, etc.
- Color panel: `.color-panel`
- Floating playback controls: use `patterns/floating-playback-control/pattern.css` and `pattern.js`; consume `.pto-floating-playback*` classes or `window.PtoFloatingPlaybackControl.createControl()`, then call `init()` and `initScrubberHover()`. Do not recreate the floating shell, range thumb, collapse sync, or scrubber hover locally.
- Glass surfaces: add `data-liquid-glass` only to floating auxiliary overlays, not page sections, cards, dense tables, or primary content panels. `scripts/liquid-glass.js` owns support flags only; CSS must keep native `backdrop-filter` blur visible before adding static highlight styling. The approved large-area IDE blur/gradient treatment is `ide-frame`; do not recreate it with page-local glass cards or cursor glow.

For full class index, grep `css/style.css` or open `design-system-preview.html`.

---

## State conventions

| State | How to express |
|---|---|
| hover | semantic var or `.btn:hover` mixes `--state-hover` |
| pressed | `--state-press` |
| selected | add `.is-selected` to `.btn`, or surface-4 fill on cards |
| focus | `--focus-ring` 3px ring via `box-shadow: 0 0 0 3px var(--focus-ring)` |
| disabled | `opacity: var(--button-disabled-opacity)` + `pointer-events: none` |

---

## Status pills

Pill shape (`--radius-pill`) on `--surface-3` background; text color = status semantic:

```html
<span class="badge badge--success">Done</span>
<span class="badge badge--warning">Running</span>
<span class="badge badge--danger">Failed</span>
```

If `.badge--*` is missing in `css/style.css`, inline-style is acceptable for retrofit (text uses `var(--success | --warning | --danger)`), but flag it so the system can absorb it later.

---

## Data-viz exemptions

These are allowed to use hard-coded colors **outside** the semantic palette, because they encode data:

- Color maps (heatmaps, density)
- Graph node semantic accents
- Swimlane event colors: use `PtoSwimlaneTaskPattern.createTaskColormap()` for task categories, engine lanes, stitches, and subgraphs
- Stitch / dependency colors

Document the choice. Reusable swimlane color rules belong in the shared pattern colormap; one-off data encodings stay module-local and must not become global UI tokens. Task identity fallback colors use the categorical palette in `patterns/swimlane-task/pattern.js`, not raw `hash % 360` hue mapping.

---

## When to STOP and ask for approval (Workflow A)

If the page needs any of the following that don't exist in the system:

- A button variant beyond the 4 roles
- A new card / panel shape
- A new badge / tag shape
- A new chart legend / control pattern that is reusable

→ produce a preview HTML, list state coverage, and wait for explicit user approval. See `preview-gate.md`.
