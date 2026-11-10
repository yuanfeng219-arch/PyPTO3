# CSA Operator Lens · 四层下钻

入口：`index.html`（也可从仓库根目录 `launch.html` 进入）。支持 `#L1`–`#L4` 直达某一层，L2 可写 `#L2:pipeline`。

以 `Data/DeepseekV4/deepseek_v4_flash_dspark/decode_csa.py` 为例，把两个已有 Demo 的能力接成一条阅读路径：

| 层 | 中间画布 | 左侧源码 | 右侧 | 来源 |
|---|---|---|---|---|
| L1 整层 | 语义计算图：5 个阶段框、Tensor / 计算 / Cache 状态节点，Sparse Attention 与 Merge · Publish 可展开；图元与 PyPTO-Compute-Explorer-UI 的 Graph 页签相同 | `decode_csa.py` 151–499，作用域 / 派发 / 通信标签 | 节点的是什么 / 为什么 / 数据 / 怎么执行、源码段、可下钻的 Kernel | CSA 语义模型按 Explorer-UI schema v2 编写并通过其校验，行号按当前源码重核 |
| L2 Kernel | 模块内的 Kernel 派发链（显式 TaskId 才画箭头）+ 所选 SPMD Kernel 的 **Assignment**（Generation → Ordering → Assignment，可播放）与 **Pipeline**（Lane × 轮次，Rank / Request / Query / Valid KV Rows 过滤） | Kernel 所在函数 | Kernel、Lane、Work Item 的规则公式 | 规则取自源码；qk_pv 的有效 / 空块为 Synthetic 输入 |
| L3 Tile | qk_pv 循环嵌套与计算链 | qk_pv 208–313 代码行 + 硬件标签 | qk_pv 洞察（问题分析） | 复用 compile-board-demo 的 workbench |
| L4 硅片 | 算子逻辑 + AIC / AIV 存储落点 | 同上 | 同上 | pass 36 `AllocateMemoryAddr` 真实分配；UP_DOWN / stage 改动为推算 |

下钻统一用**双击节点**，右侧详情栏只做说明、不放下钻按钮：
- L1：双击 14 个计算 / 通信模块进入它的 L2；展开后的子节点（如 Sparse Attention 里的 qk_plan）直达对应 Kernel；双击数据 / 状态节点进入产生它的模块。`cache_ready_dep` 只承载依赖，没有 L2。
- L2：双击 qk_pv 的 Kernel 节点进入 L3；在 Assignment / Pipeline 里双击一个 Work Item，带着它进入 L3。单击只选中、看详情。

Kernel 链来源：
- 手工建模 4 条：Sparse Attention（kv_touch → qk_plan → qk_pv → rope_cs）、KV Writeback（64 block）、Merge · Pack · Publish（48 worker，按目标 Rank 着色）、RoPE 表（CORE_GROUP 单任务）。其中 qk_pv、KV Writeback、Merge · Publish 有 Assignment / Pipeline。
- 其余 10 个模块由 `js/kernel-scan.js` 从子模块源码扫描：在模块调用的函数里找 `pl.spmd` / `pl.at` 派发，读出 name_hint、派发数表达式、deps 与 `as` 捕获的 TaskId，按显式 TaskId 连线；共 55 个 Kernel，名字都能在 pass 36 编译产物里找到。这些 Kernel 只给派发规则卡片，不画 Lane（派发数是运行期表达式）。

## 联动

- 悬停图上节点 ↔ 源码行互相点亮；点源码行时，右栏的“这一行在各层”给出它在 L1–L4 分别是什么，并可直接跳到对应层。
- 顶栏路径与 L1–L4 切换随时可回到上一层；L3 / L4 固定为 qk_pv。
- L2 → L3 带上下文：在 qk_pv 的 Assignment / Pipeline 里点一个 Work Item，详情栏预览它在循环体里会走哪条路（滑窗块：窗口行数与成段数；压缩块：逐行搬的有效行与占位行；空块：只写初值），双击这个格子进入 L3。L3 顶部出现来源条，逻辑图只亮这条路径并标出行数，源码只亮对应行；L3 ↔ L4 之间保留，从 L1 / L2 直接切到 L3 或点 × 回到通用循环体。这些数字由源码规则加该项的 position 推出，position 本身是 Synthetic 输入。

## 文件

| 文件 | 说明 |
|---|---|
| `js/source-snapshot.js` | `decode_csa.py`、`decode_sparse_attn_csa.py` 及 9 个子模块文件的逐字快照与 SHA-256（2026-11-10 生成）；用 `tools/make-snapshot.js` 重新生成，原文件更新后需核对写死的行号 |
| `js/kernel-scan.js` | 从子模块源码扫描各模块的 Kernel 派发，补齐 L2 链与源码标签 |
| `js/lens-data.js` | 模块说明、L2 Kernel 与源码标签、qk_pv Work Item 规则（按 Rank 生成 Synthetic 输入） |
| `js/csa-graph-model.js` | L1 的 CSA 语义图（Explorer-UI schema v2） |
| `l1-graph.html` · `js/l1-graph-host.js` | L1 图宿主：在 iframe 里调用 Explorer-UI 的 `PtoVisualization2Graph.mount`，经 `LensGraph` / `LensL1Bridge` 与外层互通；PTO 设计系统样式不影响外层 |
| `js/lens.js` | 层级切换、L2 Assignment / Pipeline、源码联动、详情栏 |
| `js/lens-tile.js` | 复制自 `compile-board-demo/path-csa.js`，限定源码查询范围、只在 L3 / L4 响应事件，并暴露 `window.CsaTile` |
| `vendor/` | 复制自 `hardware-native-ide-ui`、`hardware-native-ide`、`compile-board-demo` 的样式与图元脚本，原文件未改动 |
| `vendor/explorer-ui/` | 复制自 `PyPTO-Compute-Explorer-UI` 的 Graph 渲染链（语义图校验 / 布局 / 场景 / AI Infra 图元皮肤）与样式；`ai-infra-presentation.js` 只摘取 `ai-infra-ui.js` 的图元部分。PTO 设计系统仍引用仓库共享的 `vendor/pto-design-system/` |
| `styles/lens.css` | 本 Demo 新增样式 |

## 口径

- 位置表达依赖与归属，不表达耗时、实际重叠或带宽；未接入真实编译器或 Runtime。
- L2 的 Lane 列是领取顺序，不是时间轴。
- 所有资源可离线打开（可直接用 file:// 打开）；用静态服务器时以 `PyPTO3/` 为根。L1 图在 iframe 里，与外层只用 postMessage 通信，所以 file:// 下同样能双击下钻、联动源码。
