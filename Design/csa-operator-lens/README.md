# CSA Operator Lens · 整层 → Kernel 工作区

入口：`index.html`（也可从仓库根目录 `launch.html` 进入）。支持 `#L1` / `#L2` 直达，L2 可写 `#L2:pipeline`、`#L2:chip`；旧链接 `#L3` / `#L4` 仍可用，分别落到 L2 qk_pv 的计算图与硅片落点。

以 `Data/DeepseekV4/deepseek_v4_flash_dspark/decode_csa.py` 为例，把两个已有 Demo 的能力接成一条阅读路径。原来的 L2 Kernel、L3 Tile、L4 硅片已合并成一个 **Kernel 工作区**：上方是计算逻辑图，下方切换 Assignment / Pipeline / 硅片落点三个视图。

| 层 / 视图 | 中间画布 | 左侧源码 | 右侧 | 来源 |
|---|---|---|---|---|
| L1 整层 | 语义计算图：5 个阶段框、Tensor / 计算 / Cache 状态节点，Sparse Attention 与 Merge · Publish 可展开；图元与 PyPTO-Compute-Explorer-UI 的 Graph 页签相同 | `decode_csa.py` 151–499，作用域 / 派发 / 通信标签 | 节点的是什么 / 为什么 / 数据 / 怎么执行、源码段、可下钻的 Kernel | CSA 语义模型按 Explorer-UI schema v2 编写并通过其校验，行号按当前源码重核 |
| L2 上方 · 计算图 | 所选 Kernel 的 Tile 计算图（目前是 qk_pv 的循环嵌套与计算链）；可切到“模块 Kernel 链”（显式 TaskId 才画箭头） | — | — | 复用 compile-board-demo 的 workbench |
| L2 下方 · Assignment / Pipeline | 所选 SPMD Kernel 的 **Assignment**（Generation → Ordering → Assignment，可播放）与 **Pipeline**（Lane × 轮次，Rank / Request / Query / Valid KV Rows 过滤） | Kernel 所在函数 | Kernel、Lane、Work Item 的规则公式 | 规则取自源码；qk_pv 的有效 / 空块为 Synthetic 输入 |
| L2 下方 · 硅片落点 | AIC / AIV 存储落点（只有 qk_pv） | qk_pv 208–313 代码行 + 硬件标签 | qk_pv 洞察（问题分析） | pass 36 `AllocateMemoryAddr` 真实分配；UP_DOWN / stage 改动为推算 |

下钻统一用**双击**，右侧详情栏只做说明、不放下钻按钮：
- L1：派发多个 Kernel 的模块（Sparse Attention、Indexer、HC Pre、O Proj 等 10 个）是可展开的容器，双击展开 / 收起，子节点就是它的 Kernel；双击其中的 Kernel（如 qk_pv）进入 L2。只派发一个 Kernel 的模块（RoPE 表、KV Writeback、Merge · Publish、HC Post）双击直接进入 L2。双击数据 / 状态节点会展开（或进入）产生它的模块；`cache_ready_dep` 只承载依赖，没有 L2。
- L2：在模块 Kernel 链里双击 qk_pv，上方换成它的计算图；双击计算图节点，或在 Assignment / Pipeline 里双击一个 Work Item，下方切到硅片落点。单击只选中、看详情。

Kernel 链来源：
- 手工建模 4 条：Sparse Attention（kv_touch → qk_plan → qk_pv → rope_cs）、KV Writeback（64 block）、Merge · Pack · Publish（48 worker，按目标 Rank 着色）、RoPE 表（CORE_GROUP 单任务）。其中 qk_pv、KV Writeback、Merge · Publish 有 Assignment / Pipeline。
- 其余 10 个模块由 `js/kernel-scan.js` 从子模块源码扫描：在模块调用的函数里找 `pl.spmd` / `pl.at` 派发，读出 name_hint、派发数表达式、deps 与 `as` 捕获的 TaskId，按显式 TaskId 连线；共 55 个 Kernel，名字都能在 pass 36 编译产物里找到。这些 Kernel 只给派发规则卡片，不画 Lane（派发数是运行期表达式）。

L2 上方的模块 Kernel 链与 L1 用同一套图元和渲染（`graph.html?mode=chain`，见 `js/chain-graph-model.js`）：Kernel 是计算节点；虚线是源码写出的 TaskId 依赖；实线是按张量读写推出的数据依赖（Kernel A 写、后面的 Kernel B 读同名张量），边上标张量名；链外传入、交给外部的 TaskId 画成数据节点。下方是 Assignment / Pipeline / 硅片落点或派发规则卡片，中间可拖动调整高度。

L2 上方：所选 Kernel 有 Tile 计算图（目前是 qk_pv）时默认显示它。整页只有这一张算子逻辑图（`#hpLogic`，由 `lens-tile.js` 渲染），三个下方视图共用：
- Assignment / Pipeline：单击一个 Work Item，图上只亮它会走的路径，并在节点上标出行数等实例数据；
- 硅片落点：源码栏换成 qk_pv 带硬件标签的摘录，右侧换成问题分析；悬停源码行、图上节点或硅片区域，三处一起亮起。切到硅片落点时上方固定为计算图，默认把更多高度让给下方。

右侧详情的算子卡片（`js/op-cards.js` 数据、`js/op-card-view.js` 渲染、`styles/op-card.css`）：
- 功能摘要：一句话 + 输入 → 算子 → 输出的张量胶囊（颜色 = 数据归属）；
- 数学表达式：取自各模块的 golden_* torch 参考实现，排版成公式，附符号说明；
- 输入输出示例：默认配置（TP = 2，T = 256）下的形状图，宽高按维度的 log₂ 比例，分段表示行的归属；
- 支持的场景及限制：场景矩阵（平台、TP、阶段、压缩比等，取自命令行 choices 与常量）+ 限制清单（固定常量 / 整除约束 / 运行期行为，附源码行号）。

## 联动

- 悬停图上节点 ↔ 源码行互相点亮；点源码行时，右栏的“这一行在各层”给出它在 L1、L2、计算图、硅片分别是什么。
- 顶栏路径与 L1 / L2 切换随时可回到上一层；计算图与硅片落点目前只有 qk_pv，选其他 Kernel 时硅片落点页签置灰。
- Work Item 上下文贯穿三个视图：在 qk_pv 的 Assignment / Pipeline 里点一个 Work Item，详情栏预览它在循环体里会走哪条路（滑窗块：窗口行数与成段数；压缩块：逐行搬的有效行与占位行；空块：只写初值），上方计算图同时只亮这条路径；切到硅片落点（或双击这个格子）后，硅片图上方出现 Work Item 条，源码摘录只亮对应行，点 × 回到通用循环体。这些数字由源码规则加该项的 position 推出，position 本身是 Synthetic 输入。

## 文件

| 文件 | 说明 |
|---|---|
| `js/source-snapshot.js` | `decode_csa.py`、`decode_sparse_attn_csa.py` 及 9 个子模块文件的逐字快照与 SHA-256（2026-11-10 生成）；用 `tools/make-snapshot.js` 重新生成，原文件更新后需核对写死的行号 |
| `js/kernel-scan.js` | 从子模块源码扫描各模块的 Kernel 派发，补齐 L2 链与源码标签 |
| `js/lens-data.js` | 模块说明、L2 Kernel 与源码标签、qk_pv Work Item 规则（按 Rank 生成 Synthetic 输入） |
| `js/csa-graph-model.js` | L1 的 CSA 语义图（Explorer-UI schema v2） |
| `graph.html` · `js/graph-host.js` | 图宿主：在 iframe 里调用 Explorer-UI 的 `PtoVisualization2Graph.mount`，L1 整层图与 L2 Kernel 链共用；与外层只用 postMessage 互通；PTO 设计系统样式不影响外层 |
| `js/op-cards.js` · `js/op-card-view.js` · `styles/op-card.css` | 右侧详情的算子卡片：数据与可视化 |
| `js/chain-graph-model.js` | 把模块的 Kernel 链写成 Explorer-UI schema v2 语义图 |
| `js/lens.js` | 层级切换、L2 Assignment / Pipeline、源码联动、详情栏 |
| `js/lens-tile.js` | 复制自 `compile-board-demo/path-csa.js`，渲染 L2 上方的 qk_pv 计算图与下方硅片落点；限定源码查询范围、只在硅片落点视图响应事件，并暴露 `window.CsaTile` |
| `vendor/` | 复制自 `hardware-native-ide-ui`、`hardware-native-ide`、`compile-board-demo` 的样式与图元脚本，原文件未改动 |
| `vendor/explorer-ui/` | 复制自 `PyPTO-Compute-Explorer-UI` 的 Graph 渲染链（语义图校验 / 布局 / 场景 / AI Infra 图元皮肤）与样式；`ai-infra-presentation.js` 只摘取 `ai-infra-ui.js` 的图元部分。PTO 设计系统仍引用仓库共享的 `vendor/pto-design-system/` |
| `styles/lens.css` | 本 Demo 新增样式 |

## 口径

- 位置表达依赖与归属，不表达耗时、实际重叠或带宽；未接入真实编译器或 Runtime。
- L2 的 Lane 列是领取顺序，不是时间轴。
- 所有资源可离线打开（可直接用 file:// 打开）；用静态服务器时以 `PyPTO3/` 为根。L1 图在 iframe 里，与外层只用 postMessage 通信，所以 file:// 下同样能双击下钻、联动源码。
