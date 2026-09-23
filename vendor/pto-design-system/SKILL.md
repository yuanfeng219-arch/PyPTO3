---
name: pto-design-system
description: "PTO 设计系统优先的页面创建与改造规范。用于创建或改造任何 PTO 页面：所有 PTO 页面必须以 `patterns/ide-frame` 为基础 shell，并从共享 tokens、components、patterns 进行 pattern-first 组合。不要发明新的视觉语言。"
---

# PTO 设计系统 Skill

这个 Skill 用来让 AI 生成新 PTO 页面，或把已有 demo 刷成 PTO 视觉风格。

## 使用方式

1. 把整个 `design-system-share/` 文件夹交给 AI，包括这个文件、`references/DESIGN.md`、`design-system-preview.html`、`tokens/`、`css/`、`references/`、`patterns/`。
2. 先判断任务入口，但所有 PTO 页面最终都进入「工作流 C：统一 Pattern-first 页面流程」：
   - 从产品需求新建页面：先用「工作流 A」做需求拆解，再进入工作流 C。
   - 把已有 demo 改成 PTO 风格：先用「工作流 B」做 shell-first retrofit，再进入工作流 C。
   - 已经明确 pattern / block / page 组合：直接进入工作流 C。
3. 所有完整 PTO 页面都必须以 `patterns/ide-frame` 作为基础 shell。`design-system-preview.html` 只用于查基础组件外观；完整页面、复杂图谱、workbench、timeline、architecture 必须读取对应 `patterns/<pattern-id>/pattern.json`。

核心原则：所有 PTO 页面 = `ide-frame` 基础 shell + pattern-first 内容组合。产品页只负责 domain data、pane content、commands 和必要 glue code；不要新造按钮、toggle、badge、card、panel、间距、色彩或边框语言。

可读性基线：正文、描述、Inspector 说明、帮助与空状态文案使用 `--type-body`，即 14px / 1.5；紧凑控件、表格、树、表单 label、代码和 metadata 不低于 12px；11px 只用于短 badge、eyebrow、坐标轴等 micro label，不得承载正文。低于 11px 默认禁止，只有可缩放 data-viz 内部标注可以作为 pattern contract 中有理由、有 tooltip/detail fallback 的显式例外。不要用 `zoom`、`transform: scale(...)` 或更小正文 token 解决拥挤，应调整 pane、换行、截断或滚动。

页面 chrome 基线：PTO 页面默认使用 `ide-frame` 提供的透明顶栏和 workarea。不要加实色 header 背景、装饰性 header band，或 header 下方额外大间距，除非用户明确要求分离式页面 chrome。第一个内容 shell 应贴近 header，下钻到 shell/pane 内部再使用 token 间距。

IDE frame 基线：所有 PTO 页面必须用 `patterns/ide-frame` 作为外框，并继承 `patterns/ide-frame/pattern.css` 的默认皮肤。已接受的 standalone skin 是 100% 强度多点 gradient/aura 背景、80% 半透明模糊 pane、72% pane header fill、`blur(18px) saturate(1.18)` backdrop filter、透明 top chrome、可选 activity rail、可选 `floating-playback-control`。不要用页面私有 opaque panel、私有 gradient、generic dashboard shell 或复制出来的 workbench CSS 替换它。需要改皮肤时，先改共享 `ide-frame` 变量或 pattern，再由产品页消费。

光标追踪基线：standalone `ide-frame` 可以在 `.pto-ide-frame` 根节点启用 `data-cursor-dots="true"`，并在 pointer movement 更新 `--ide-cursor-x`、`--ide-cursor-y`、`--ide-cursor-alpha`、`--ide-dot-opacity`。共享点阵密度是 `--ide-dot-gap: 20px`；产品页不要再加密 grid。觉得光标场太强时，调 opacity、alpha 或 hot radius，或通过获批的页面变量 override 降噪，不要做私有高密度 grid。

播放条基线：floating playback 不是默认页面 chrome。只有页面确实有 demo time、execution steps、trace playback、animation scrubber 或 timeline state 时才添加播放条。静态 architecture、graph、report、map、inspector、dashboard 页面默认不要添加 playback mount，也不要 import/use `floating-playback-control`，除非用户明确要求。

## 必读文件

按顺序读取：

1. `references/DESIGN.md`：完整系统规范，包括 theme、surface、palette、typography、spacing、component、governance。
2. `references/quick-reference.md`：tokens 和 class 速查。
3. `references/retrofit-container-audit.md`：retrofit 时删除 legacy card/panel decoration 的强制规则。
4. `patterns/patterns.json`：IDE frame、workbench split、graph、timeline、architecture、playback 等 pattern 注册表。
5. `tokens/foundation.css`、`tokens/semantic.css`、`tokens/components.css`：CSS variables 实现。
6. `css/style.css`：具体组件 class 实现。
7. `design-system-preview.html`：只作为基础组件辅助预览，不作为完整页面或复杂 pattern 的权威来源。

layout-heavy 或 visualization-heavy 工作必须先读匹配的 `patterns/<pattern-id>/pattern.json`。`pattern.json` 是 allowed overrides、forbidden overrides、required APIs 的权威 contract。

完成前从设计系统根目录运行 `rtk node scripts/audit-typography.mjs <module-or-css-path>`；修复所有 prose 低于 14px、普通 UI 低于 12px、未批准的低于 11px 字号、typography token 混用和页面根缩放。修改共享 tokens 或 patterns 时，同时运行不带 target 的 `rtk node scripts/audit-typography.mjs`。

## 工作流 A：需求拆解（随后进入统一 Pattern-first 流程）

工作流 A 不直接产出一套独立页面结构。它只把产品需求拆解成 panes、patterns、theme、playback、data 和 interaction，然后进入「工作流 C」用 `ide-frame` + pattern-first 组合实现。

写代码前，先列出页面需要的 UI 部件：

- header / toolbar
- buttons，包含 entry / commit 类按钮
- toggle、toggle group
- chip filter
- labels / badges
- card、inspector、popup
- input、select
- typography roles：body、body-sm、label、micro、mono
- data-viz-only patterns
- IDE frame、workbench split、graph、timeline、architecture、playback patterns

如果需求没有明确提到 time、step、playback，选择 optional playback 前必须问一个短问题：

`是否需要调用播放条（demo 演示时间轴）组件？`

如果没有回答，但可以继续推进，默认 **不加 playback**。生成页面不能默认包含 `.pto-ide-frame__floating-playback-mount`、`data-ide-floating-playback`、scrubber、play/pause 控制或 demo step state。

如果需求没有明确默认主题，选择页面 theme 前必须问一个短问题：

`默认使用 light mode 还是 dark mode？`

如果没有回答，但可以继续推进：retrofit 旧页时保留原页面 theme；新 standalone PTO/workbench 页面默认 dark mode，并在最终说明里写明这是推断。

然后用 `references/pto-design-system-map.md`、`patterns/patterns.json` 和匹配的 `patterns/<id>/pattern.json` 做映射。`design-system-preview.html` 只用于确认基础组件外观。

进入工作流 C 写页面时使用：

- `css/style.css` 里的 HTML class。
- `tokens/*.css` 里的 CSS variables，例如 `var(--surface-2)`、`var(--foreground-secondary)`、`var(--space-3)`。
- typography 使用 `font: var(--type-body)`、`var(--type-body-sm)`、`var(--type-label)`、`var(--type-micro)`、`var(--type-mono)`；不要把 font shorthand、font family 和 text color 混用在同一个 custom property。
- flex/grid 做局部 layout；模块私有 layout class 可以存在。
- 必须以 `patterns/ide-frame` 作为页面 shell，再消费匹配的 `patterns/`，例如 graph node、swimlane、memory tier、floating playback、workbench split kernel。

## 工作流 C：统一 Pattern-first 页面流程

这是所有 PTO 页面的唯一构建流程。无论页面看起来是 report、dashboard、graph、timeline、architecture、map、IDE、workbench，还是 blocks-style preview，都先用 `patterns/ide-frame` 作为基础 shell，再选择 pane / view pattern；不要从空白 HTML/CSS 或 generic dashboard 开始造页面。

### 1. 先选页面 shell

所有 PTO 页面都使用 `patterns/ide-frame`。不要用 shadcn dashboard、simple page、standalone canvas、普通 blocks layout 或 generic app shell 假装 PTO 页面。

standalone PTO 页面加载 `patterns/ide-frame/pattern.css` 后，应直接继承它的 gradient、pane fill、pane blur、pane shadow、background aura。当前默认背景强度是 100%；preview 页面里的 debug-panel 实验控件不能成为产品页自带私有 palette 的理由。

如果 PTO 页面布局不明确，写代码前先问：

- 页面需要几个 pane / split region。
- 每个 pane 放什么，例如 explorer、code、graph、timeline、report、inspector、console、preview。
- host 是 `standalone` 还是 `vscode-webview`。
- pane 是否需要拖拽和持久化。
- 是否需要 playback、step、scrubber；用问题 `是否需要调用播放条（demo 演示时间轴）组件？`，未回答默认不加。
- 默认主题是 light 还是 dark；用问题 `默认使用 light mode 还是 dark mode？`，retrofit 未回答保留原主题，新 standalone PTO/workbench 未回答默认 dark。

`patterns/workbench-shell` 只作为 draggable split kernel。它不拥有页面 chrome、pane fills、typography、title 或 canvas controls。PTO 页面加载 `ide-frame`，由它委托 `workbench-shell` 做拖拽。

### 1.1. IDE Frame 接入检查清单

每个 standalone PTO 页面，在写业务内容样式前先检查：

- Root：`.pto-ide-frame[data-ide-frame][data-host="standalone"]`；只有实现 pointer tracking 时才加 `data-cursor-dots="true"`。
- Chrome：`.pto-ide-frame__topbar` 透明；页面标题用简洁英文；不要默认塞 tag/chip，除非用户要求上下文 badge。
- Body：`.pto-ide-frame__body` 里包含 `.pto-ide-frame__activity-rail` 和 `.pto-ide-frame__workarea`。
- Activity rail：四个共享按钮，顺序和语义固定为 Explorer、Search、Source control、Terminal。不要换成页面私有 chart/file icon，不要遗漏 rail；只有 `vscode-webview` 或用户明确要求时才隐藏。
- Split：使用 `.pto-ide-frame__split[data-ide-split="standalone-main"]`，并设置页面专属 `data-storage-key`；每个 pane 必须是 direct child，并带 `data-ide-pane`。
- Explorer toggle：如果有 Explorer rail button，页面必须同时具备 `[data-ide-toggle="explorer"]`、`[data-ide-pane="explorer"]`、`[data-ide-split="standalone-main"]`，让 `PtoIdeFrame.init` 能同时折叠 pane 和 gutter。
- Inspector：右上角 inspector control 必须是真 toggle，能重复打开/关闭。不要留下过期 `aria-expanded`、`aria-pressed`、hidden pane 或孤立 gutter。
- Terminal：页面需要底部 terminal-like panel 时，在右上角加 `data-ide-toggle="terminal"`，`aria-controls` 指向 terminal panel。terminal icon 使用标准 rectangle + prompt glyph；不要换成 chart 或 settings icon。
- Bottom dock：如果页面已有底部 visualization split，把 visualization 和 terminal 放进同一个 bottom dock，使用 `data-ide-bottom-panel="visualization"` 和 `data-ide-bottom-panel="terminal"`。两者互斥；打开 Terminal 替换 visualization，关闭 Terminal 时恢复原来的 visualization。
- Status strip：只用于真实页面状态，保持低密度。不要再造第二条 bottom chrome band。
- Initialization：先加载 `patterns/workbench-shell/pattern.js`，再加载 `patterns/ide-frame/pattern.js`，然后调用 `window.PtoIdeFrame.init(frame)` 或 `initAll()`。
- Cursor tracking：有 `data-cursor-dots="true"` 时，pointer movement 必须更新 `--ide-cursor-x/y`、`--ide-cursor-alpha`、`--ide-dot-opacity`，pointer leave 必须把 alpha 和 dot opacity 重置为 `0`。
- Theme：dark/light 必须一致。不要默认状态里出现 light business canvas + dark IDE pane 的混搭，除非页面明确提供 theme switch，并且两个状态都检查过。
- Typography：dense 只限 pane header、tab、status strip 等 chrome；pane 正文和 Inspector prose 使用 14px `--type-body`。最终在 100% browser zoom 下检查 computed font size。

architecture、graph、map pane 在 IDE frame 内时还要检查：

- 默认状态不要选中业务对象、不要高亮路径、不要 dim 其他区域、不要显示残留 route/tag focus，除非用户明确要求 guided starting state。
- 默认视图必须能 pan/drag 和 zoom。不要设置过紧的 zoom clamp；如果确实需要 clamp，说明 domain 原因。
- 主图默认在 pane 内居中，并设置用户要求的默认 zoom。不要复用其他页面的 persisted localStorage；用页面专属 storage key，review 时必要就 reset stored view。
- pane 标题和控制项放在 `.pto-ide-frame__pane-header`；不要塞进 pattern body，避免 body padding 造成 header 漂移。

### 2. 匹配 pane / view 到 pattern

| 用户需求 | 先读哪个 pattern |
|---|---|
| PTO IDE / workbench shell | `patterns/ide-frame/pattern.json` |
| 可拖拽 horizontal / vertical / nested panes | `patterns/workbench-shell/pattern.json` |
| execution trace、swimlane、timeline task bars | `patterns/swimlane-task/pattern.json` |
| NCHW / A1 / Load3D / Conv / tiling / code-recovery 三维 Tensor 体素 | `patterns/tensor-volume-canvas/pattern.json` |
| Pass-IR op、tensor、incast、outcast、group node cards | `patterns/pass-ir-graph-node/pattern.json` |
| 二维 Matrix Tensor、逐元素格、大矩阵聚合格 | `patterns/matrix-canvas/pattern.json` |
| TorchVista、model Graphviz、Qwen-7B / openPangu-2.0-Flash 折叠模型架构图、report overlays | `patterns/model-graphviz/pattern.json` |
| 模型训练 evidence、phase、edge tags、related-node overlay | `patterns/model-training-graphviz/pattern.json` |
| openPangu 深度堆叠模型架构、iso/front/right、平行标注 | `patterns/model-architecture-3d-deck/pattern.json` |
| Three.js model-to-rank 放置、TP/PP/EP/EDP ownership | `patterns/model-parallel-rank-deck/pattern.json` |
| 模型训练流、梯度、loss、metric、Layer telemetry sidecar | `patterns/model-architecture-training-sidecar/pattern.json` |
| 训练指标曲线、异常、brush、step cursor | `patterns/training-metrics-chart/pattern.json` |
| 完整 memory architecture diagrams | `patterns/memory-architecture/pattern.json` |
| 硬件架构 viewport toolbar、zoom readout、iframe readiness、消息协议 | `patterns/hardware-architecture-viewport/pattern.json` |
| UB / L1 / L0C tensor lifetime、buffer offset、peak usage、reuse links | `patterns/memory-reuse-viewer/pattern.json` |
| AIC / AIV 内部对象壳 | `patterns/aic-core-object/pattern.json` 或 `patterns/aiv-core-object/pattern.json` |
| Floating playback、step、pause、scrubber、collapsed playback chrome | 先问是否需要 playback；需要时再读 `patterns/floating-playback-control/pattern.json` |

如果匹配的 pattern 是 rendered、canvas、SVG 或 hybrid，不要用静态 HTML/CSS 重写。加载它的 `pattern.css` 和 `pattern.js`，调用 `pattern.json` 里记录的 `window.Pto*Pattern` 或 `window.Pto*` API。

消费或修改 pattern preview 前，先做 capability contract check：

1. 读取 `design-system-preview.html` 对应卡片，列出它承诺的能力，例如 hover tips、playback、collapse、resize、zoom、overlay、scrubber。
2. 确认这些能力在 `patterns/<pattern-id>/pattern.json` 中声明，尤其是 `requiredApis`、`allowedOverrides`、`forbiddenOverrides`。
3. 确认 `patterns/<pattern-id>/pattern.js` 确实导出对应 `window.Pto*Pattern` / `window.Pto*` API，并实现该行为。
4. 确认 `patterns/<pattern-id>/pattern.html` 在 preview 中调用 API，能被视觉验证。

四点不一致时，先修共享 pattern contract 或 preview，再给产品页消费。不要让 preview card 承诺未导出、未执行的行为。

#### Tensor Volume Canvas 复用规则

For fixed-view three-dimensional tensor volumes, reuse pattern id `tensor-volume-canvas`. Load `patterns/tensor-volume-canvas/pattern.css` and `patterns/tensor-volume-canvas/pattern.js`, then call `window.PtoTensorVolumeCanvas.render(canvas, scene, options)`. The consuming page owns Load3D, Conv, tiling, code-recovery, playback, and selection rules and maps them only to the business-neutral `extent`, `axes`, and `voxels` scene contract. Use the returned controller's `update`, `resize`, and `destroy` lifecycle methods, and call `resize()` after theme changes so Canvas paint resolves the new semantic token values. Do not copy the fixed projection, voxel geometry, depth sorting, DPR, or ResizeObserver logic; do not add an iframe, camera, pan, zoom, drag interaction, private palette, or product chrome to the Pattern.

#### Memory Reuse Viewer 复用规则

For UB, L1, and L0C tensor lifetime or reuse analysis, reuse pattern id `memory-reuse-viewer`. Load its `pattern.css` and `pattern.js`, then call `window.PtoMemoryReuseViewer.render(container, data, options)` with Memory.txt-derived data or the preview fixture from `createDemoData`. Keep the returned `resize` and `destroy` lifecycle attached to the host. Mount the viewer directly inside an unscaled panel; do not place it inside the pan/zoom architecture canvas, copy its Canvas lifetime renderer, mutate generated tensor rectangles, or wrap it in an additional bordered preview frame.

#### Memory Architecture 发布规则

改动或消费 `memory-architecture-layout` 时，完成前检查：

1. route geometry 放在 `patterns/memory-architecture/pattern.js` 的 preset data 和 overlay helpers 中。不要在产品页用局部 CSS 或 DOM traversal 修 route。
2. Direct CV detour routes 必须锚定具体硬件节点，例如 `data-aiv-node="exec:SIMD"`，不要锚到整列 AIV exec column 这种大容器。
3. Hover、path focus、playback、step focus 不得改变 route `stroke-width`。用 opacity、color、glow 表示状态，保持线宽稳定。
4. 共享 pattern 文件变化后，同步更新 `patterns/memory-architecture/pattern.html` 和 `design-system-preview.html` cache key。
5. 如果 `/Users/yin/pto` 页面通过 `vendor/pto-design-system` 消费 pattern，同一次 release 要更新 vendor checkout pointer，并 bump 产品页 resource query string。只 push `pto-design-system` 不会更新 `compute-graph-viewer` GitHub Pages。
6. 告诉用户 Pages 已更新前，验证 published HTML 引用了新 resource query，并确认 published pattern JS 包含预期 route selector。

### 3. 选择 iframe 或 direct embedding

当保留完整源页面比本地组合更重要时，用 iframe。适合 parity preview，以及原页面拥有重运行时行为的场景，例如 Graphviz DOT generation、D3 zoom、popup、graph reload、report-overlay focus state。

当新 PTO 页面拥有数据并需要本地组合 pattern 时，用 direct embedding。加载 pattern dependencies，并调用 `pattern.json` 的 required API。

model Graphviz report overlay 遵循现有做法：parity preview iframe 真实 Graphviz 源页面；只有轻量 rendered example 或新页面自己拥有 graph data 时，才直接调用 `PtoModelGraphvizPattern.render`。

新建 model architecture 页面时，遵循 `qwen7b_modelviz.html` 和 `patterns/model-graphviz/pattern.json` 的约定：cluster box 由可见 children 的 bounding box 推导；折叠 module 时 parent 应 reflow 并 shrink；decoder chain 单列自上而下布局，只有 Gate/Up 并列；cluster fold controls 放右上角；cluster corner 使用 `--radius-xl`；expanded cluster 填充 10% white；node label+type stack 垂直居中；pan/zoom 使用 transform-based `translate` + `scale`；初始状态不得默认选中任何算子，selection 只在用户点击/键盘选择后显示单条 white stroke，去掉浏览器 focus ring；repeat-count badge 放 cluster 左上标题下方。

model Graphviz 节点取色使用 shared renderer 的 semantic-first colormap：常见 `sem:*` key（`embedding`、`norm`、`attention`、`position`/`rope`、`qknorm`、`linear`/`head`、`mlp`/`act`、`gate`、`moe`、`comm`）先映射到固定语义色，未知 key 再由 shared palette 自动扩展；自动生成 hue 必须避开正红色和正绿色区间。Light mode 保持较高饱和度和 `0.90` node fill opacity；op/module 节点优先使用自身 `colorKey`，只有缺失 `colorKey` 时才 fallback 到 parent cluster 色。不要在页面局部重新洗淡 palette、恢复 parent 覆盖 `sem:*`，或硬编码页面私有节点色。Tensor/parameter 节点默认保持中性灰，cluster frame 默认透明填充；training/report 的 P0/P1/P2 这类业务优先级不进入 pattern badge。

### 4. 组合时不要加额外 chrome

Pattern wrapper 可以提供 spacing 和 sizing，但不能重定义内部 pattern classes，也不能在同一 surface 外面再套 stacked borders、shadow、rounded frame 或额外 card shell。

PTO frame blocks 规则：

- block 应展示 PTO typical page，不要复制 shadcn demo。
- `ide-frame` 拥有 activity rail、pane headers、preview/editor slots、inspector docks、split initialization。
- `ide-frame` 拥有 PTO frame skin：multi-point gradient background、aura layer、80% translucent blurred panes、pane header fill、pane shadows、transparent top chrome、bottom status strip、可选 floating playback mounting。
- 多 pane PTO/workbench 页面里，每个 pane 的 title/meta/control row 必须处于同一层 `.pto-ide-frame__pane-header`。不要把 pane title 放进 embedded pattern toolbar 或 `.pto-ide-frame__pane-body`，否则 body padding 和 pattern toolbar 默认值会造成 header 高度与 baseline 漂移。
- embedded pattern 例如 `hardware-architecture-viewport` 需要 controls mount root 时，让 pane 自身成为 pattern root，或适配 mount root，但 title row 仍保留在 `ide-frame` pane header。
- explorer collapse 必须同时移除 pane 和 gutter，剩余 panes 填满空间。
- 整个 preview page 应可正常滚动；embedded iframe 不能吞掉正常页面滚动。
- icon 使用 Lucide-style SVG；stroke color 用不透明色，用整 SVG opacity 控制状态，避免 path 交叉处透明叠加。
- playback controls 是 opt-in。不要为了让 frame 看起来完整而加；只有页面数据有 time、step、trace 语义时才加。

## 工作流 B：把已有 demo 改成 PTO 风格（shell-first retrofit）

已有 HTML/CSS demo 需要迁移到 PTO 时：

1. 自上而下阅读 demo，列出所有视觉元素：buttons、inputs、panels、badges、toggles、headings、surfaces、hard-coded colors。
2. 先判断 demo 是否已经使用 `.pto-ide-frame`。没有时，第一项迁移就是接入 `patterns/ide-frame`，把旧页面 shell、topbar、side rail、bottom panel、inspector、canvas host 映射进 frame；不要只改色而保留 legacy shell。
3. 对每个元素，用 `references/pto-design-system-map.md`、`patterns/patterns.json`、匹配的 `pattern.json` 和 `references/quick-reference.md` 映射到 PTO。
4. 必须执行 `references/retrofit-container-audit.md` 的 container decoration audit。每个 card、panel、list item、inspector block、sidebar section 都要检查。
5. 先给用户迁移表再动手：

   | Demo 元素 | PTO 等价物 | 使用的 class / token | 需要删除的 legacy decoration |
   |---|---|---|---|
   | `<button class="cta">Run</button>` | solid button | `btn btn-solid` | 私有 shadow / gradient |
   | `background: #1a1a1a` | surface-2 | `var(--surface-2)` | 私有背景色 |
   | `padding: 16px` | space-4 | `var(--space-4)` | 无 |

   第四列用于列出旧 full borders、left rails、accent bars、inset highlights、shadows、pseudo-elements、gradients。这些应删除，不是 token 化。

6. 用户看过表后，替换 HTML class 和 inline styles；把 hard-coded colors、shadows、radii、spacing 改成 tokens。
7. 删除 legacy container decoration。不要把私有 card border 或 left accent bar 换成 PTO token 值后继续保留，除非目标 PTO component 明确有该 decoration。
8. 用工作流 C 重新检查页面 shell、pane、pattern embedding、theme、playback、terminal/bottom dock、cursor tracking 和默认 graph state。
9. 执行 `references/retrofit-container-audit.md` 的 post-migration residue check，并在最终回复里说明结果。
10. 跳过 preview gate；这里不是创建新视觉，只是消费现有系统。
11. 最终回复列出：
   - 使用了哪些 PTO classes / tokens。
   - `ide-frame` shell 是否已接入，旧 shell 删除了哪些部分。
   - 哪些元素没有找到等价物，需要用户决策；不要静默新造风格。
   - container decoration residue check 结果。

## 硬性规则

- 所有完整 PTO 页面必须以 `patterns/ide-frame` 为基础 shell；不允许 simple page、generic dashboard、standalone canvas、standalone report 或 shadcn layout 直接作为最终 PTO 页面。
- 所有 PTO 页面都必须 pattern-first；产品页只负责数据、pane content、domain commands 和 glue code，不要从空白 HTML/CSS 发明页面结构。
- 不要创建私有 button / toggle / badge / card 系统。
- 有现成 token 时，不要 hard-code colors、radii、shadows、font sizes、borders、spacing。
- 正文、描述、Inspector prose、帮助和空状态文案必须使用 14px `--type-body`；不能用 `body-sm`、label、micro 或 metadata 样式代替。
- 表单 label、按钮、表格、树、代码与 metadata 不得低于 12px；11px 只用于短 micro label；低于 11px 必须是 pattern contract 明确批准的 data-viz 例外。
- 不要用 `zoom`、`transform: scale(...)`、页面根字号缩放或更小字号解决布局拥挤。
- `--type-*` 只用于 `font`，`--font-*` 只用于 font family，`--foreground*` 只用于 text color；不要创建或覆盖同时承担多种含义的 typography token。
- 不要保留 legacy card / panel decoration 后只把颜色换成 token。
- generic cards / panels / inspector blocks 不要保留 `border-left`、`border-inline-start`、inset left `box-shadow`、pseudo-element rails、side gradients，除非它们编码数据或是获批的 selected state。
- 不要把 border 或 outline 作为主要视觉语言。避免 border-heavy cards、outlined panels、outline-only buttons、nested stroked containers、decorative outlines。优先用共享 fills、elevation、spacing、typography、opacity、state tokens。只在必要 pane boundaries、form controls、accessibility focus rings、table/grid separators、data-viz encodings，或 pattern contract 明确要求时保留细 border/outline。
- 未经用户审批，不要新增 module-local visual tokens。
- 不要带着未审批的新视觉发业务模块。
- 不要把 `design-system-preview.html` 当成 graph、timeline、architecture、IDE、workbench 行为的权威来源。
- 当 `pattern.js` 拥有 geometry、rendering、truncation、drag、zoom、synchronized state 时，不要把 pattern 截图复制成静态 DOM/CSS。
- 不要在产品页或高阶 pattern 覆盖 `.pto-workbench-shell__*` internals。
- 不要本地复刻 floating playback chrome；只有用户确认需要 playback 时才使用 `floating-playback-control`。
- 不要在本地重写 `ide-frame` pane backgrounds、backdrop blur、gradient background、pane shadows；先改 `patterns/ide-frame` 变量，让所有 PTO 页面继承同一 skin。
- 不要改 `--ide-dot-gap` 或在产品页添加私有 cursor-grid background；保留共享 grid density，通过获批的 opacity/radius 变量调可见性。
- 不要同时显示 Terminal 和 bottom visualization panel；它们是同一个 bottom dock 的替换视图。
- 不要让 PTO 页面默认选中对象、默认高亮路径、默认 dim 其他 architecture 区域，或显示残留 tag，除非这是用户明确要求的 initial story。

允许模块私有 layout 和结构 class；不允许新视觉语言。

## 常见失败模式

- 复制 shadcn Blocks 内容，而不是用 PTO patterns 搭 PTO typical page。
- 把 Pattern-first 当成只适用于复杂图、IDE 或某类页面，静态 report / dashboard 另起一套结构。
- 用普通 div、shadcn、dashboard shell 或 standalone report 搭 PTO 页面，跳过 `ide-frame`。
- PTO 页面没问布局就猜 pane 数量和内容。
- 静态页面默认添加 floating playback。
- 没问 dark/light mode 就猜，或默认状态混合 light business content 与 dark IDE chrome。
- 改 activity rail icon set、漏掉左侧 rail，或用页面私有 icon 替代 Explorer / Search / Source control / Terminal。
- 添加用户没要求的右上角工具 icon 或 tag chips。
- 做成 border/outline-heavy 页面，所有 card、panel、button、nested container 都描边，而不是用 PTO fills 和 spacing。
- 为了塞下更多内容把正文降到 12px 或更小，把 11px label 用于整段说明，或缩放整个 pane。
- 文件树视觉上关闭了，但 gutter 或窄残留 pane 还在。
- inspector、explorer、bottom panel 按钮只能打开一次，第二次不能关闭。
- 启用 cursor dots 但没有 pointer tracking 和 pointer-leave reset。
- 通过改密 `--ide-dot-gap` 让 cursor grid 看起来更密，而不是调 opacity。
- 只让 iframe 或 preview stage 内部滚动，整个页面不能正常滚动。
- 把 Graphviz、swimlane、memory architecture、Pass-IR node 当成 CSS-only component。
- legacy card borders、left rails、pseudo-elements、inset highlights、gradients 只是换成 token 色后继续保留。

## 缺失样式审批门

工作流 A / C 中，如果现有系统无法满足需要：

1. 写最终模块样式前停止。
2. 创建 `<module>/component-preview.html`，展示：
   - 最接近的现有 system pattern。
   - proposed new pattern。
   - normal / hover / active / selected 状态覆盖。
   - token 使用。
   - 当前系统为什么不够。
3. 等用户明确批准。
4. 批准后，先把新 pattern 吸收到共享系统，再由模块消费。

完整规则见 `references/preview-gate.md`。

## 最终回复必须说明

- 是否使用 `patterns/ide-frame` 作为基础 shell；如果是 retrofit，旧 shell 哪些部分被删除或替换。
- 复用了哪些现有系统部件。
- 匹配了哪些 pattern id。
- 每个匹配的 pattern 是 iframe 还是 direct embedding，以及原因。
- PTO/workbench 页面中，pane 数量和内容是用户确认的，还是合理推断的。
- 默认 theme 是用户确认的，还是推断的。
- 哪些需求超出现有系统。
- 是否创建了 preview page。
- 用户是否批准新视觉。
- 获批视觉是否已吸收到共享系统。
- 工作流 B：完整 migration table。
- 工作流 B：container decoration residue check 结果。
- Typography audit 结果：正文 computed font size、最小普通 UI 字号，以及所有低于 11px 的 data-viz 例外。

## 参考文件

- `references/quick-reference.md`：token 和 class 速查。
- `references/pto-design-system-map.md`：元素分类规则。
- `references/preview-gate.md`：审批流程。
- `patterns/patterns.json`：共享 pattern registry。
- `patterns/<pattern-id>/pattern.json`：每个 pattern 的权威复用 contract。
- `references/DESIGN.md`：设计系统权威规范。
