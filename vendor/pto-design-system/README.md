# PTO 设计系统｜协作使用说明

这是一份打包好的 **PTO 设计系统**，可以丢给任何 AI 编程工具（Claude Code、Cursor、Windsurf、ChatGPT 文件上传…），让它**自动**按 PTO 的颜色、字体、间距、组件规范来生成或改造网页。

> 简单说：你给 AI 一个产品需求或现有 demo，AI 会按 PTO 样式产出网页，**不需要你再手动调样式**。

## 在线预览

- Design System Preview: <https://yinyucheng0601.github.io/pto-design-system/design-system-preview.html>

## 这个文件夹里有什么

```
design-system-share/
├── README.md                    ← 当前文件，给你看的使用说明
├── SKILL.md                     ← 给 AI 看的规则（让 AI 先读这个）
├── DESIGN.md                    ← 设计系统全景说明（颜色 / 字体 / 间距 / 组件 / 治理）
├── design-system-preview.html   ← 基础组件辅助预览；完整页面优先看 patterns
├── references/
│   ├── quick-reference.md       ← 一页速查：所有 token 和 class 名
│   ├── pto-design-system-map.md ← 元素分类规则（什么时候用什么按钮）
│   ├── retrofit-container-audit.md ← 改造 demo 时检查旧卡片/面板边框残留
│   └── preview-gate.md          ← 遇到系统没有的样式时的审批流程
├── tokens/                      ← CSS 变量（颜色 / 间距 / 圆角 / 字体）
│   ├── foundation.css
│   ├── semantic.css
│   ├── components.css
│   ├── generate_tokens.py       ← 从 CSS token 源生成 tokens.json / tokens.js
│   ├── tokens.json              ← 生成产物，不手改
│   └── tokens.js                ← 生成产物，不手改
├── css/style.css                ← 真正的 class 实现
├── scripts/audit-theme.mjs      ← light/dark token、硬编码颜色、对比度检查
├── scripts/audit-typography.mjs ← 14px 正文基线、token 完整性、data-viz 小字号例外检查
├── assets/                      ← pattern 运行时需要的 SVG 资源（例如 fx.svg）
├── swimlane/styles.css          ← swimlane 模块样式（预览页要用）
└── patterns/                    ← 页面级 / 可视化级复用 pattern，优先级高于基础组件拼装
    ├── patterns.json            ← pattern 注册表
    ├── ide-frame/               ← PTO 典型 IDE 框架与默认皮肤：渐变背景 / 磨砂 pane / topbar / activity rail / 目录树 / 代码区 / View slot
    ├── workbench-shell/         ← 可拖拽分屏 resize kernel，只负责 split / gutter / localStorage
    ├── floating-playback-control/ ← 悬浮播放、step、scrubber 控制条
    ├── model-graphviz/          ← TorchVista / DeepSeek V3.2 Graphviz / Qwen-7B / openPangu-2.0-Flash model architecture / report overlay
    ├── model-training-graphviz/ ← 模型训练 evidence、phase、edge tag overlay
    ├── model-architecture-3d-deck/ ← openPangu 深度堆叠模型架构与平行标注
    ├── model-parallel-rank-deck/ ← Three.js 模型到 Rank 的并行放置
    ├── model-architecture-training-sidecar/ ← 模型架构训练语义与 telemetry sidecar
    ├── training-metrics-chart/  ← 训练指标、异常、brush 和 step cursor SVG 图表
    ├── memory-architecture/     ← 910B / 950 内存架构图与 route overlay
    ├── memory-reuse-viewer/     ← UB / L1 / L0C tensor lifetime 与复用检查器
    ├── hardware-architecture-viewport/ ← 硬件架构 viewport、toolbar、zoom 和消息协议
    ├── aic-core-object/         ← AIC core object 可配置结构图
    ├── aiv-core-object/         ← AIV core object 可配置结构图
    ├── pass-ir-graph-node/      ← Pass-IR graph node 卡片和 compact/group 状态
    ├── tensor-volume-canvas/    ← API Visualizer 风格的固定视角三维 Tensor Canvas renderer
    ├── matrix-canvas/           ← 二维矩阵 Tensor Canvas，支持逐元素格与大矩阵聚合格
    └── swimlane-task/           ← 执行 trace / timeline 泳道任务条 canvas renderer
```

## Patterns 是重点

`patterns/` 不是 demo 素材库，而是 PTO 页面搭建的高层积木。遇到 IDE 工作台、多分屏分析、Graphviz、泳道、内存架构图、Pass-IR 节点、播放控制时，先命中 pattern，再填业务内容；不要只用基础按钮和卡片临摹截图。

| Pattern | 什么时候用 | 入口文件 |
|---|---|---|
| `ide-frame` | PTO IDE / 工作台 / 多视图分析页面的整体框架；默认套用 100% 强度多点渐变背景、80% 透明磨砂 pane、透明 top chrome | `patterns/ide-frame/pattern.html` |
| `workbench-shell` | 任意需要拖拽调整大小的分屏 | `patterns/workbench-shell/pattern.js` |
| `floating-playback-control` | step、播放、暂停、scrubber、回放控制 | `patterns/floating-playback-control/pattern.js` |
| `model-graphviz` | TorchVista / 模型 Graphviz / Qwen-7B / openPangu-2.0-Flash 架构页 / 报告 overlay | `patterns/model-graphviz/pattern.html` |
| `model-training-graphviz` | 在共享模型 Graphviz 上叠加训练 evidence、phase、edge tag 与相关节点强调 | `patterns/model-training-graphviz/pattern.js` |
| `model-architecture-3d-deck` | openPangu 深度堆叠架构、iso/front/right 视图和平行标注 | `patterns/model-architecture-3d-deck/pattern.js` |
| `model-parallel-rank-deck` | 将完整模型 Layer 编译到 Three.js Rank volumes 并检查并行归属 | `patterns/model-parallel-rank-deck/pattern.js` |
| `model-architecture-training-sidecar` | 在 3D deck 右视图上增加训练流、梯度、loss、metric 与 Layer 下钻 | `patterns/model-architecture-training-sidecar/pattern.js` |
| `training-metrics-chart` | 训练指标曲线、异常点、interest-window brush 和 step cursor | `patterns/training-metrics-chart/pattern.js` |
| `memory-architecture` | 910B / 950 内存层级、MTE route、AIC/AIV 组合图、graph-only iframe、60% 默认缩放、拖拽和 Command-wheel 缩放 | `patterns/memory-architecture/pattern.js` |
| `memory-reuse-viewer` | UB / L1 / L0C tensor lifetime、buffer offset、peak usage 与复用关系检查 | `patterns/memory-reuse-viewer/pattern.js` |
| `hardware-architecture-viewport` | 架构图 toolbar、detail visibility、zoom readout、iframe readiness 和标准消息协议 | `patterns/hardware-architecture-viewport/pattern.js` |
| `aic-core-object` / `aiv-core-object` | 单独渲染 AIC / AIV 内部结构 | `patterns/aic-core-object/pattern.js` / `patterns/aiv-core-object/pattern.js` |
| `pass-ir-graph-node` | Pass-IR 节点卡、group node、compact node | `patterns/pass-ir-graph-node/pattern.js` |
| `tensor-volume-canvas` | NCHW / A1 / Load3D / Conv / Tiling / Code Recovery 的固定视角三维 Tensor 体素渲染 | `patterns/tensor-volume-canvas/pattern.js` |
| `matrix-canvas` | 二维矩阵 Tensor、Load3D A2 状态，以及业务可注入数量的缩略格概览 | `patterns/matrix-canvas/pattern.js` |
| `swimlane-task` | 执行 trace / timeline 的泳道任务条 | `patterns/swimlane-task/pattern.js` |

## 在三种环境里加载

有两种用法：**装成 Skill**（Claude 系工具，自动触发、按需读文件）和 **当文件包**（其他工具，手动丢给它）。

### 1. CLI — Claude Code

技能目录名**必须等于 `SKILL.md` 里的 `name` 字段**（本系统为 `pto-design-system`）。

```bash
# 个人级（所有项目可用）
git clone https://github.com/yinyucheng0601/pto-design-system \
  ~/.claude/skills/pto-design-system

# 或项目级（只在某个项目可用）
git clone https://github.com/yinyucheng0601/pto-design-system \
  <你的项目>/.claude/skills/pto-design-system
```

重启 Claude Code，`/` 列表里能看到 `pto-design-system`。命中 description 描述的场景（建 PTO 新页面 / 改造 demo）会**自动触发**，也可手动 `/pto-design-system` 调用。

### 2. 网页版 + 桌面 App — Claude.ai / Claude Desktop

两者同一账号，**上传一次两端都生效**，桌面 App 不用单独装。

1. 把 `design-system-share/` 打包成 **.zip**，确保 zip 内顶层就是包含 `SKILL.md` 的文件夹（GitHub「Download ZIP」会多一层 `-main` 外壳，要先解压再重新打包）。
2. 进入 Settings → Capabilities → **Skills**（需 Pro / Max / Team / Enterprise）。
3. 上传 zip，启用。之后新对话里会按 description 自动触发。

> 不想装 Skill 也可以用 **Projects**：把文件加进项目知识库即可，但不会自动触发，且文件夹层级会被拍平，体验弱于 Skill。

### 3. Codex / Cursor / Windsurf / ChatGPT 等

这些工具**没有 Skill 自动加载机制**，只能当文件包用：

- **Codex**：把 `design-system-share/` 放进项目，在项目根的 `AGENTS.md` 里加一句——「生成或改造页面前先读 `design-system-share/SKILL.md`，严格复用其中的 token 和 class」。
- **Cursor / Windsurf / ChatGPT 等**：直接把整个文件夹丢进对话，并在指令里要求「先读 `SKILL.md`」。

具体手动用法见下面「怎么用」。

## 怎么用

> 下面的「丢文件夹给 AI」适用于**当文件包**的场景；如果已按上面装成 Skill，文件已在本地，AI 会自己按需读，你只要直接说需求即可。

### 第 0 步：先看系统边界

`design-system-preview.html` 只能辅助查看基础按钮、标签、卡片、输入框等原子组件。完整页面、IDE 框架、可拖拽分屏、Graphviz、泳道、模型架构图等，优先看 `patterns/`，不要让 AI 只照着 `design-system-preview.html` 拼页面。

IDE / 工作台页面默认套用 `patterns/ide-frame` 的皮肤：100% 强度多点渐变和 aura 背景、80% 填充的半透明磨砂 pane、72% pane header 填充、`blur(18px) saturate(1.18)`、透明 topbar、悬浮播放条。不要让业务页复制一份本地渐变、实底面板或私有 workbench CSS；需要改皮肤时先改 `patterns/ide-frame`。

### 场景 A：从产品需求生成新页面

1. 把整个 `design-system-share/` 文件夹一起丢给 AI 工具
2. 给 AI 这样的指令：

   > 先读 `SKILL.md`。我要一个 PTO 样式的新页面，用途是：[在这里写你的产品需求，比如"算子调试面板，左侧文件树、中间代码编辑器、右侧 inspector"]。按 Workflow A 来做。

3. AI 会：
   - 列出页面需要哪些 UI 元素（按钮、面板、表格、卡片…）
   - 把每个元素对应到 PTO 已有的 class 和 token
   - 产出符合 PTO 样式的 HTML / CSS

### 场景 B：把现有 demo 改成 PTO 样式

1. 把 `design-system-share/` 文件夹 **+ 你的 demo 文件** 一起丢给 AI
2. 给 AI 这样的指令：

   > 先读 `SKILL.md`。把这个 demo 改造成 PTO 样式，按 Workflow B 来做。

3. AI 会先给你一张 **迁移对照表**，告诉你它打算把哪个元素换成哪个 PTO class、哪些颜色换成哪个 token、哪些旧容器装饰要删除：

   | demo 里的元素 | 对应 PTO 组件 | 用的 class / token | 要删除的旧装饰 |
   |---|---|---|---|
   | `<button class="cta">运行</button>` | solid 主按钮 | `btn btn-solid` | 无 |
   | `background: #1a1a1a` | surface-2 | `var(--surface-2)` | 无 |
   | `.card { border-left: 3px solid #5b8cff }` | inspector section / soft card | `inspector-section` / `inspector-soft-card` | 左侧高亮描边 |

4. 你看一遍没问题，让它继续，AI 就会真的改 HTML / CSS

### 场景 C：用 Patterns 搭 PTO 典型页面

当需求不是一个普通表单或卡片，而是“IDE 工作台 / 多分屏分析工具 / 图可视化 / 时间线 / 架构图 / 带播放控制的调试页面”时，要让 AI 先走 `patterns/`。

你可以这样说：

> 先读 `SKILL.md`，按 Pattern-first workflow 做。我要一个 PTO IDE 页面，用 `ide-frame` 搭页面框架。请先问我页面要几分屏、每个分屏是什么视图，再判断各视图命中哪个 pattern。

AI 应该先问清楚：

- 页面是独立网页，还是 VS Code Webview
- 页面要几分屏，例如左侧目录树 / 中间图或代码 / 右侧 inspector / 底部 timeline 或 console
- 每个分屏分别是什么视图，例如文件树、代码、Graphviz 模型图、Pass-IR 节点图、swimlane 时间线、memory architecture、报告 inspector
- 分屏是否需要拖拽调整大小；如果需要，用 `workbench-shell`，但它只负责拖拽，不负责页面视觉
- 是否需要播放、step、scrubber；如果需要，用 `floating-playback-control`

常见命中关系：

| 需求 | 应命中 |
|---|---|
| PTO IDE / 工作台页面 | `patterns/ide-frame`，默认套用共享 IDE 皮肤；可用 `hidden` 或 `data-activity-rail="hidden"` 隐藏左侧 activity rail |
| 可拖拽分屏 | `patterns/workbench-shell` |
| 执行 trace / 时间线任务条 | `patterns/swimlane-task` |
| Pass-IR graph 节点卡 | `patterns/pass-ir-graph-node` |
| 固定视角三维 Tensor / Load3D 体素状态 | `patterns/tensor-volume-canvas`；页面将业务状态转换为 voxel 数据并直接调用共享 renderer |
| 二维矩阵 Tensor / 大矩阵缩略概览 | `patterns/matrix-canvas`；页面保持源 shape，通过任意 span + summary 或目标缩略格行列数表达聚合区域 |
| TorchVista / 模型 Graphviz / Qwen-7B / openPangu-2.0-Flash 架构页 / 报告 overlay | `patterns/model-graphviz` |
| 模型训练 evidence / phase / edge tag overlay | `patterns/model-training-graphviz` |
| 深度堆叠模型架构与平行标注 | `patterns/model-architecture-3d-deck` |
| 模型到 Rank 的 Three.js 并行放置 | `patterns/model-parallel-rank-deck` |
| 模型架构训练流、梯度、loss 与 Layer telemetry | `patterns/model-architecture-training-sidecar` |
| 训练指标曲线、异常与 step cursor | `patterns/training-metrics-chart` |
| AIC / AIV / 950B 内存架构图 | `patterns/memory-architecture`、`patterns/aic-core-object`、`patterns/aiv-core-object`；iframe/inspector 嵌入默认 graph-only，使用 `createZoomController` 的 60% zoom、拖拽和 Command-wheel 缩放 |
| 硬件架构 viewport toolbar、zoom 和消息同步 | `patterns/hardware-architecture-viewport` |
| UB / L1 / L0C tensor lifetime 与复用检查 | `patterns/memory-reuse-viewer` |
| 底部播放和 step 控制 | `patterns/floating-playback-control` |

如果一个 pattern 已经有完整运行环境，比如 Graphviz overlay 原页面仍然负责 DOT 生成、D3 zoom、popup、展开折叠，那么 pattern 预览可以用 iframe 保持一致；如果是在新页面里消费同一套视觉和渲染逻辑，则直接加载对应 `pattern.css` / `pattern.js` 并调用 `window.Pto*Pattern` 或 `window.Pto*` API。

## 验收 AI 输出时检查这几点

打开生成的页面，优先对照 `SKILL.md`、`DESIGN.md`、命中的 `patterns/<id>/pattern.json` 和对应 pattern 预览；`design-system-preview.html` 只用于辅助检查基础组件。确认：

- 所有颜色都用语义 token，例如 `var(--background)` / `var(--surface-*)` / `var(--foreground-*)` / `var(--primary)`，**没有硬编码的 `#xxxxxx`**
- 间距用 `var(--space-1)` ~ `var(--space-6)`，**不要写死 `padding: 13px`**
- 按钮用 `btn` / `btn btn-solid` / `btn btn-ghost`，**没有 `.my-button`、`.custom-cta` 这种自创 class**
- IDE / graph / swimlane / memory architecture 这类页面要命中对应 pattern，不要只用基础组件临摹截图；IDE 页不要本地重写渐变背景、pane 透明度、pane blur 或播放条 chrome
- 如果用了 iframe，要说明是为了保留完整运行环境；如果直接嵌入，要说明调用了哪个 pattern API
- 旧 demo 的卡片/面板边框已消除：尤其不要留下旧的 `border-left`、伪元素竖条、左侧 inset shadow、侧向渐变高亮
- AI 在最后会列出**"复用了哪些 PTO 组件"**和**"哪些地方系统没覆盖到"**
- 改造 demo 时，AI 在最后会列出 **Container decoration residue**，说明旧容器装饰是否已删除或为什么保留
- 如果 AI 偷偷加了新颜色 / 新按钮样式但**没有标注**，直接打回让它改

## 遇到 PTO 系统没覆盖的情况怎么办

如果你的需求里有一个组件 PTO 现在没有（比如要一个特殊的进度条），AI 会：

1. **停下来**，不会瞎造
2. 做一个 preview 给你看：现有最接近的是什么、它想新加的是什么、各种状态长什么样
3. 等你说"可以用这个"再继续

详见 `references/preview-gate.md`。

## 一些常见问题

**Q：我必须用 Claude 吗？**
不用。任何能读文件夹的 AI 都行（Cursor / Windsurf / Cline / ChatGPT 文件上传 / Gemini 等）。SKILL.md 是给 AI 看的纯文本规则。

**Q：AI 没按 SKILL.md 来怎么办？**
在指令里**强调一次**：「请先读 `SKILL.md`，并按里面的 Workflow A/B/C 输出」。大多数情况这一句就够了。

**Q：我自己改了 PTO 主仓库的 token，这个文件夹会自动同步吗？**
不会，这是一份**复制**。主仓库改动后需要把 `DESIGN.md` / `design-system-preview.html` / `tokens/*` / `css/style.css` / `swimlane/styles.css` / `patterns/` 重新复制进来。

**Q：可以把 AI 生成的新组件回流到 PTO 系统里吗？**
可以而且应该。流程是：AI 先做 preview → 你审核 → 审核通过后**先把样式塞进 `tokens/` 和 `css/style.css`**，**再**让业务模块去消费。绝对不要先在业务模块里用、回头再补到系统里。

## 反馈 / 改进

发现 AI 经常踩同一个坑，告诉我，我会把规则写进 `SKILL.md`。
