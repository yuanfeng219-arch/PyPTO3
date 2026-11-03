# DeepSeek V4 CSA · 第一阶段

入口：index.html → 资源管理器 decode_csa.py → **Compute Explorer**。

本场景用于阅读源码中的数据归属与依赖。默认 TP=2。T 是运行时本地 token 数；数字 Shape 按本地默认容量 T=256、整组 2T=512 说明，不是每次运行的固定输入规模。

| 视角 | 整体 | 局部 |
|---|---|---|
| Assignment | 两个 Rank 的 Local / Gathered 数据、三种 Cache、Attention Group 重分布 | rank-major 行段、Cache 生产者与消费者、Group 到目标 Rank 的归属 |
| Pipeline | 按 Rank 分组的主要计算与通信依赖 | AllGather 信号阶段、Cache TaskID 就绪、Publish / All-to-All / ReduceScatter |
| Graph | 五个语义 Stage 与 Module、数据和状态依赖 | 展开 Module 查看源码可支持的内部计算关系 |

选择节点后，Inspector 提供 What / Why / Data / How / Source。Assignment 与 Pipeline 保留语义对象和 Rank；某个 Tensor 不直接出现在目标视角时，选择映射到对应模块。源码按钮返回 decode_csa.py 的真实行段；子算子证据可在详情中展开。两个文件的源码位置与两个场景的可视化状态分别保存。

## 关键阅读口径

- Q 与 Indexer query 读取 local token 行；KV 和两个 Compressor 读取 gathered stream。local 数据独立不意味着 Q 已与 AllGather 重叠：Q 调用位于后续 scope。
- AllGather 得到的是每个 Rank 上的全组副本，顺序为 [Rank 0 rows | Rank 1 rows]。
- Original、Compressed、Indexer KV Cache 独立。Main 与 Indexer Compressor 也各自维护请求状态。
- cache_ready_dep 合并 Original 和 Main 压缩 Cache 的写入 TaskID；Indexer 另外等待 idx_cache_write_tid。图中不添加虚构的全局 barrier。
- Attention 输出先按 Group 归属融合 Put / Notify，再由 o_group_a2a 等待、读回和回收信号；它不是再发送一次相同 payload。
- O Projection 的各 Rank 部分和发回 token owner，owner 做 FP32 累加。ReduceScatter 不是单纯切分 token。
- l3_decode_csa 对数据取 [rank]，传 tp_rank=rank 与 device=rank。host rank 循环不代表两个设备串行执行。

## 源码与工程边界

证据来自本地相对路径 ../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_csa.py，以及同目录的 decode_cp_token_allgather.py、decode_o_proj.py、config.py。

js/decode-csa-source.js 保存上述源码的本地快照，主文件包含原始相对路径与 SHA-256。源码展示和详情不需要运行时 Fetch、CDN 或外部站点；原始源码更新后需同步快照及行段。运行静态服务器时以 PyPTO3/ 为根，保留现有 ../../vendor/pto-design-system/ 相对依赖。

复用既有 IDE Frame、深色主题、按钮和 Tab；图形通过共享 model-graphviz.renderController 直接嵌入，使用其节点、选择、拖拽和缩放。CSA 的数据和状态独立于 Paged Attention 引擎。

布局、连线和行段块为静态简化示意，不推断运行耗时、实际重叠、带宽、物理地址或 Tile 指令。Pipeline 全览密度较高，可用“阅读”、缩放及局部视图查看细节。未接入真实编译器或 Runtime，CSA Graph 复用共享语义 Graph 渲染器；未做 Morph、播放模拟或完整 Tile 级细节。

## QK Work Assignment · 第二阶段局部场景

入口：Decode CSA → Compute Explorer → Assignment → 工具栏 **QK Work Assignment**（再次点击返回 Assignment）。它是 Assignment 的一个局部模式，不新增主 Tab；退出后原有 Gather / Cache / Output 视图不变。

用于解释 `decode_sparse_attn_csa.py` 的 `qk_plan → qk_pv`：

| 区域 | 内容 |
|---|---|
| 显式任务依赖 | `qk_plan_tid → qk_pv`、`cache_ready_dep → qk_pv`；`cache_ready_dep` 可展开为 `ori_cache_write_tid` / `cmp_cache_write_tid`。只画本段源码显式声明的依赖 |
| A · Original Work Matrix | 5 Tokens × 5 Sparse Blocks = 25 个 `(token, sparseBlock)` Work Item，格内为扁平 ID |
| B · Reordered Work Queue | `qk_order`：valid 段在前，invalid 追加到尾部（不删除），两段之间有边界标记 |
| C · SPMD Block Assignment | 每行一个逻辑 SPMD Block，按 Iteration 顺序列出领取的 Work Items |
| 右侧 Inspector | Work Item / Block / Dispatch 三种粒度；含队列位置、Block、Iteration、规则公式、执行路径、真实源码行段 |
| 术语 | 固定解释 Dispatch / SPMD Block / Work Item / Iteration 的区别 |

工程口径：

- `qk_pv` 是**一次** `pl.spmd(NUM_QK_CORES)` 派发，展开为 24 个逻辑 Block；不是 24 次 Submit。
- Work Item 是 `(token, sparseBlock)`，不是独立 Runtime Task；`workItemId = token * SPARSE_BLOCKS + sb`，重排后 ID 不变。
- 领取规则 `qk_flat = qk_core + qk_it * N`，因此 `block = queuePosition % N`、`iteration = ⌊queuePosition / N⌋`。
- 底部教学示例（4 Blocks，非真实 CSA 配置）与顶部源码配置（24 Blocks，示例输入）共用同一组 25 个教学 Work Item，只切换 Block 数并重算分配。
- SB0 恒为 valid（源码直接把 `valid_block_mask[*,0]` 写 1），不可切换；SB1–SB4 可切换，切换只改本地教学数据，不代表修改真实 `idx_topk` 或重跑算子。
- 源码证据：`decode_sparse_attn_csa.py` L134–196 / 208–224 / 224–314 以本地快照行段展示；`decode_csa.py` L323、353、361、384–393 通过编辑器导航跳转（会先退出 Explorer 再滚动定位）。
- 不出现 Runtime Duration、物理 Core 时间线、valid 数量换算的耗时；`pl.pipeline(..., stage=2)` 只作为源码声明的结构说明。

实现：`js/csa-qk-work.js`（纯函数 `buildWorkItems` / `buildQueue` / `assignToBlocks` / `getWorkItemDetails` / `getBlockDetails` + 自带选择状态的渲染器）、`styles/csa-qk-work.css`（`csa-qk-` 前缀）。`csa-explorer.js` 只增加 `qkMode` 状态、工具栏入口、`render()` 分支与源码跳转；QK 视图有独立 host，切换主 Tab 不丢选择。


## Project Explorer 与多文件源码导航

文件树从 2 个文件扩展为两个可折叠分组：`Paged Attention`（1 个文件）与 `DeepSeek V4 / Decode CSA`（`decode_csa.py` + 12 个真实模块，共 13 个）。默认打开的仍是 `paged_attention_pypto.py`。

| 层 | 实现 |
|---|---|
| 文件身份 | `js/project-files.js`：`fileId`（`csa/<name>`、`paged-attention/<name>`）、`name`、`sourcePath`、`scene`、`content` |
| 源码来源 | 主文件 `PTO_CSA_SOURCE.text`；11 个模块 `PTO_CSA_SOURCE.dependencies`；`utils.py` 单独快照 `js/csa-utils-source.js`（与仓库文件逐字节一致） |
| 文件树 | `renderTree()` 写 `#projectTree`，沿用既有 `.kf-tree-row` / `kf-caret` / `kf-file-icon` 样式，`depth-2` 缩进、长名省略、分组计数徽标 |
| Source Tab | 只列已打开文件，同一文件重复打开不新增 Tab；切回旧 Tab 恢复该文件上次滚动位置与区域选中 |
| 文件级导航 | `PtoComputeIDE.openSource({file,start,end})`，`file` 接受 fileId 或文件名；`showSource(start,end)` 仍定位“当前文件”，`showSourceFile(name,start,end)` 保留兼容 |
| 跨文件符号 | 扫描真实 `from x import y` / `import x` 与目标文件顶格 `def` / `class` / `NAME =`；解析不到的名字保持普通文本，不猜测跳转 |
| Source Region | 两个主文件沿用人工编写的区域表；12 个子文件按真实 `def` / `class` 自动分区，区域行号由源码行生成 |

工程口径：

- `decode_csa.py` 的 `from decode_sparse_attn_csa import sparse_attn_csa` 与所有调用点可点击，落到 `decode_sparse_attn_csa.py` L105 `def sparse_attn_csa(`。
- 子文件仍属 `csa` 场景，从子文件点 `Compute Explorer` 进入的仍是 Decode CSA，不会切到 Paged Attention。
- `csa-explorer.js` 的源码跳转统一走 `openSource({file,...})`：QK Work Assignment 与 Graph / Assignment Inspector 都按各自真实文件名定位，不再假设当前打开的是主文件。
- 全部为本地快照，无 GitHub API / Raw / CDN 依赖，GitHub Pages 直接可用。
- `golden.py` 在仓库中不存在，未注册、未伪造；外部库（`pypto` / `torch` / `sys` 等）不进入文件树。
