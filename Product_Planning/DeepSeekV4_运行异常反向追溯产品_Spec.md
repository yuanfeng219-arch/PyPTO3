# DeepSeek V4 运行异常反向追溯产品 Spec

> 文档版本：v0.13  
> 更新时间：2026-09-11  
> 文档状态：v0.13 删除证据模式与待取证控件，统一 Inspector 最小宽度并补充泳道概览；真实致因 Pass 未定位，设备实验尚未接通  
> 适用范围：PyPTO / PTO Runtime 泳道异常、任务依赖、编译决策与性能调优  
> 当前验证数据：`Data/DeepseekV4/`  
> 对应原型：`Design/deepseek-gap-investigator/`（v0.13；实现范围与剩余缺口见 11.7，不代表完成设备运行）  
> 旧版备份：`Design/deepseek-gap-investigator-v0.1-backup-20260910/`

v0.2 将产品入口从独立诊断报告调整为完整 Trace 工作台，将六步流程收进调查抽屉，并补充标准泳道基座与 PyPTO 工程实验的衔接契约。v0.3 通过消融实验去掉重复的 Workspace / Explorer 栏、Activity Rail 和无上下文工具入口，将 Rank、泳道范围、数据源和调查点收敛到泳道顶部。

v0.4 将泳道表头压缩为一行，把人工标注、自动候选与手动 Case 合并到同一份调查点列表。六步流程改为可拖动高度的底部非模态悬浮抽屉，Task Inspector 独立在右侧，允许调查与事件详情同时显示。

v0.5 曾将原报告完整嵌入抽屉，包括独立相关任务泳道与正文内 IR 文本对照。该版本保留了调查内容，但仍是“Trace 上叠加报告”，尚未形成主视图驱动的调查工作流。

v0.6 根据 2026-09-10 的交互与代码审查，替代 v0.5 的整页复用要求：保留问题、证据、诊断能力和 PTO 组件，删除抽屉内独立泳道；统一 Task 定位；Pass IR 与泳道 Diff 使用主工作区；工程直接导入、自动关联结果为目标路径，手工导出／回填仅作离线降级。下文区分当前事实、目标规格与待验证能力。

v0.7 根据 2026-09-11 的界面审查，删除 Step 04 的说明卡、源码卡、决策表和“打开 Pass IR”中间入口。进入 Step 04 后，正文直接替换为相邻 Pass 前后图；快照选择位于对照图头部，节点继续下钻原始 IR 行。主 Trace 保持原视口，不因查看编译变化而切走。

v0.8 继续消除只读信息的非必要下钻：Pass 对照默认选择存在结构变化的 49→50 快照并 Fit 全图，节点详情改为 hover／键盘聚焦提示；源码与参数定义改为原位折叠代码块。诊断文案采用“观测结果／数据缺口／建议”的短句结构，避免多层否定和无法由数据支持的归因。

v0.9 曾给默认 49→50 对照补充“无法解释等待区间”的说明；这一做法仍将无关变化放进调查流程，已由 v0.10 替代。该版本保留的改进是源码顺序边与中文状态文案。

v0.10 取消写死的第 50 个 Pass、按 wait 名称取图及全部快照选择器。Step 04 先要求变化与当前异常有可追溯的关联；只有“有结构变化”不能成为展示理由。当前两个等待调查缺少最晚到达生产者和 Task→编译子图映射，因此不展示 Pass 图，改为呈现本次等待 Task、Trace 已记录的直接前驱、具体缺失记录，以及返回上游排查的操作。已有用户状态中保存的 Pass 编号不再影响此步骤。

v0.11 根据用户明确允许 mock 的要求，为 01 / Rank 0 增加“真实证据／模拟补全”切换。真实数据仍不足以判断致因 Pass，这不表示编译过程与异常无关。模拟模式补齐等待记录、任务到图的映射、内存分配决策、单变量方案与模拟泳道结果，用于验证完整交互；模拟内容不写入真实 Case 证据、工程、实验或测量结果。

v0.12 删除 Step 04 的“查看模拟 Pass Diff”二次入口，进入该步骤即显示前后图。模拟图不再使用“打包／暂存／发布”三个抽象业务块：基线采用真实 Pass 23 中 `csa_merge_pack_publish_spmd → publish_tid → o_group_a2a_wait` 的任务依赖，变化侧复用真实 Pass 24 的 `gm_pipe_buffer` 创建与 SPMD 入参写法。新增 buffer 与当前 producer 的对应关系、320 µs 代价和优化结果仍是显式模拟，不冒充归档事实。

v0.13 删除调查头部的“真实证据／模拟补全”切换与“待取证”状态标签；模拟仅保留显式 Demo URL，不占用常规调查界面。右侧 Inspector 每次打开使用 360px 最小宽度，允许打开后临时拉宽；未选择事件时显示当前筛选泳道的汇总，选中真实或模拟事件后显示对象详情。

---

## 1. 文档目标

本文定义一个从运行异常反向追溯到调优入口的诊断产品，解决开发者在泳道图中看到空洞、碎片、长尾或核间不均衡后，无法把观测结果转化为下一轮调优动作的问题。

产品必须连续回答以下七个问题：

1. 空洞期间是真的没有就绪任务，还是存在就绪任务但没有及时下发？
2. 任务未运行是因为上游依赖未满足，还是 AIC／AIV、内存、队列或其他资源不可用？
3. 如果上游任务过慢，它慢在计算、数据搬运、同步通信还是调度开销？
4. 当前子图为什么被这样切分或融合？
5. 哪个 Pass 做出了或物化了这项决定？
6. 开发者应该修改 TileShape、Loop、融合范围、调度策略、Runtime 参数还是源码计算结构？
7. 修改后预计以及实际减少的是搬运、任务数、同步等待还是核间不均衡？

本文不是编译器全量可视化需求，也不是新增一张独立图的设计说明。它定义的是一条有证据等级、有未知项、有验证闭环的诊断工作流。

---

## 2. 产品定义

### 2.1 一句话定位

> 从泳道异常出发，将 Runtime 状态、Execute／Block／Tile 图、Pass 决策和可修改入口连接成一条可验证的调优证据链。

### 2.2 核心价值

现有工具相对擅长：

```text
泳道异常 → 相关 Task → Execute / Block Graph
```

本产品补齐：

```text
直接等待原因 → 上游瓶颈 → 编译决策 → 控制入口 → 实验验证
```

完整闭环为：

```text
确认异常
  → 定位任务
  → 排查原因
  → 追溯编译变化
  → 制定调优方案
  → 运行实验与对比
```

### 2.3 产品边界

产品做：

- 从泳道中的空洞、细碎任务、长尾和核间不均衡建立调查对象；
- 聚合与异常时间窗有关的 Runtime、图、Pass、源码和硬件证据；
- 对每个诊断问题给出“确认、候选或未知”的明确状态；
- 推荐可修改入口，并说明依据、预期直接变化、风险和验证指标；
- 对比修改前后的图结构、任务、搬运、等待、负载均衡和端到端耗时。

产品不做：

- 不根据泳道形状直接声称某个编译或 Runtime 根因；
- 不在缺少 `ready`、`blocked_reason` 或 Pass 决策日志时伪造确定答案；
- 不替代完整 Execute Graph、Block Graph、源码编辑器或 Pass IR 查看器；
- 不自动修改源码、参数或编译配置，除非用户明确确认；
- 不使用单一综合健康分代替可核验的诊断证据。

---

## 3. 设计原则

### 3.1 Trace-first，工作流附着于异常

完整泳道是产品一级入口。开发者先浏览一次 Run 的全量 Trace，再从手动框选或统一调查点进入调查。六步工作流以底部悬浮抽屉承载，始终绑定当前 `run_id`、Rank、时间范围、选中泳道与锚点 Task。事件详情独立显示在右侧，不替换调查内容。

主工作区承载 Trace 与实验泳道对比；调查抽屉负责组织“当前能得出什么结论、依据是什么、下一步做什么”。Step 04 在确认调查关联后直接显示 Pass 前后图；未关联时显示追查断点和下一步。生命周期与小规模证据也留在抽屉；不另建独立首页，也不在抽屉中复制第二套 Trace。

### 3.2 每一步使用统一回答契约

每个诊断步骤必须同时呈现：

```text
当前问题
├─ 当前结论
├─ 支持结论的直接证据
├─ 尚不能确定的部分
└─ 下一步所需数据或操作
```

系统不能只给一个结论，也不能只把证据平铺给用户自行理解。

### 3.3 事实、推导、假设和未知严格分离

界面不得把以下内容混写：

- 原始观测；
- 编译产物中确认的事实；
- 基于多个事实得到的推导；
- 尚待验证的候选解释；
- 因数据缺失而无法作答的问题。

### 3.4 推荐必须能回到控制入口

“这里 Copy 很多”“这里存在等待”不是完整结论。调优建议必须指向具体源码、TileShape、Loop、融合 scope、Pass 参数、Runtime 参数或调度策略，并说明它通过哪条因果路径影响哪个指标。

### 3.5 所有优化建议最终回到实验 Diff

建议不是结果。产品必须记录修改假设，并通过重新编译和实测对比判断它是否成立。

---

## 4. 核心对象：异常调查 Case

用户在泳道上选择一个异常时间窗后，系统创建一个 Investigation Case。

### 4.1 Case 最小字段

| 字段 | 含义 |
|---|---|
| `case_id` | 调查对象稳定 ID |
| `run_id` | 对应运行与编译产物 |
| `rank_ids` | 涉及的 Rank |
| `time_range` | 异常起止时间 |
| `symptom_type` | 空洞、碎片、长尾、核间不均衡或依赖等待 |
| `selected_lanes` | AIC、AIV、CPU、通信或其他泳道 |
| `viewport_state` | 创建 Case 时的缩放、滚动、过滤与可见时间窗快照 |
| `anchor_task_ids` | 与异常相邻或重叠的任务 |
| `root_hash` | 跳转 Execute Graph 的稳定关联键 |
| `call_op_magic` | 调用或执行实体关联键 |
| `leaf_hash` | 跳转 Block Graph / 叶子实现的关联键 |
| `diagnosis_state` | 六步工作流的完成状态 |
| `evidence_refs` | Trace、IR、Pass、源码与指标引用 |
| `hypotheses` | 待验证假设及状态 |
| `experiments` | 修改方案、基线和验证结果 |

### 4.2 Case 状态

| 状态 | 含义 |
|---|---|
| 新建 | 仅选定异常时间窗 |
| 定位中 | 正在关联 Task、Execute、Block 和 Rank |
| 诊断中 | 正在判断 Runtime 直接原因与上游瓶颈 |
| 待取证 | 关键数据缺失，已生成采集要求 |
| 待实验 | 已形成可修改入口与验证假设 |
| 已验证 | 候选修改已完成基线对比 |
| 未解决 | 实验未改善或证据仍不足 |

---

## 5. 六步诊断工作流

## 5.1 第一步：确认异常

### 用户问题

- 这是空洞、任务过碎、长尾、核间不均衡还是依赖等待？
- 异常发生在哪些 Rank、核类型和时间范围？
- 它是偶发局部现象还是重复出现的稳定模式？

### 输入

- 泳道任务的开始时间、结束时间、核类型和 Rank；
- 设备利用率；
- 任务数量和持续时间分布；
- 可选的历史 Run 或其他 Rank 基线。

### 页面产出

- 异常时间窗；
- 异常类型与严重度；
- 受影响泳道、Rank 和任务；
- 与对照 Rank／历史 Run 的差异；
- 调查方向，而不是根因结论。

### 交互要求

- 框选泳道时间窗创建 Case；
- 在主 Trace 支持 Rank 0、Rank 1 与对齐比较，明确 Run 内相对时间、事件锚点对齐或有证据支持的绝对时间；不得默认视为时钟已同步；
- 点击异常卡片回到泳道中的准确位置；
- 可切换每核视图与核类型汇总视图。

抽屉保留范围、指标和调查方向，不再绘制“运行时间线 · 相关任务切片”。原切片使用真实任务摘录，但多核汇总跨度不等于单个 Worker 原始事件时长；汇总数据应标注口径，并联动主 Trace 中的全部对应实例。

### 完成标准

用户能够回答“发生了什么、发生在哪里、影响多大”，但此步不得声称“为什么发生”。

---

## 5.2 第二步：定位任务

### 用户问题

- 空洞前最后完成的任务是什么？
- 空洞后第一个启动的任务是什么？
- 哪些 Task、Execute 调用、Block Graph 和源码范围与异常相关？

### 输入

- Task ID、Task Name、Rank、Lane；
- `rootHash`、`callOpMagic`、`leafHash`；
- Execute / Block / Tile 图节点 ID；
- 编译产物和源码映射。

### 页面产出

以异常时间窗为锚点给出四类集合：

1. 空洞前的候选前驱；
2. 空洞后启动的下游任务；
3. 与时间窗重叠但运行在其他核或 Rank 的任务；
4. 图上与下游任务存在依赖或资源关系的候选任务。

### 交互要求

- 泳道任务、因果节点、Execute、Block 和源码双向联动；
- 默认只展示与调查 Case 有关的局部子图；
- 可跳转完整 Execute Graph 和 Block Graph；
- 跳转后保留 Case 上下文和选中态。

所有入口共用同一定位契约，不能只更新 Inspector：

```text
选择逻辑 Task / Worker 实例
  → 解析 Run、Rank、pid、tid 与事件身份
  → 显示目标 Rank 和泳道，解除阻挡定位的显示过滤
  → 调整主 Trace 时间视口和纵向滚动
  → 高亮对应事件并打开独立 Inspector
```

- 点击逻辑 Task 高亮全部 Worker 实例，并提供实例选择；点击实例精确定位单个事件，不能任取一个 Worker 代替整个 Task。
- 目标事件必须位于抽屉上方可见区域；必要时提供“展开 Trace”动作，不允许定位成功但事件被遮住。
- 跨 Rank 定位可切换主视图 Rank 或进入对照视图，但不得更改 Case 绑定的 Rank、调查范围及步骤。
- 浏览状态与调查状态分离；定位 Task 后，通过调查抽屉内与 Case 同上下文的“回到调查范围”恢复异常区间，不在主泳道底栏暴露隐式的“定位前视口”快照。
- 无精确事件或稳定映射时显示缺口，不按相似名称静默跳转。

### 完成标准

用户能够确定“调查的是哪一组运行任务和哪一段编译／源码结构”。

---

## 5.3 第三步：排查原因

此步是产品的核心，不应被简化为一张“最小因果切片”。系统首先判断任务生命周期，再决定需要展示哪一种下钻视图。

### 5.3.1 一级诊断树

```text
空洞期间是否存在候选下游 Task？
│
├─ 尚未创建
│  └─ 追溯任务生成、控制流或图切分
│
├─ 已创建，但依赖未满足
│  └─ 展开未满足依赖与最慢前驱
│
├─ 已 ready，但未进入调度队列
│  └─ 检查 Runtime 状态推进与队列治理
│
├─ 已 enqueued，但未 dispatched
│  └─ 检查调度策略、核与资源可用性
│
├─ 已 dispatched，但未 started
│  └─ 检查设备队列、核流水和 Runtime 资源
│
└─ 生命周期数据缺失
   └─ 标记 UNKNOWN，并生成补采集要求
```

### 5.3.2 必须回答的状态问题

| 问题 | 可确认所需的最小证据 |
|---|---|
| 是否存在候选任务 | `created_at` 或 Runtime task create 事件 |
| 是否已经 ready | `ready_at`、未满足依赖计数与依赖解除事件 |
| 是否进入调度队列 | `enqueued_at` 与 queue id |
| 是否已下发 | `dispatched_at`、目标核与 dispatch 事件 |
| 是否已开始 | `started_at` |
| 为什么没有推进 | `blocked_reason` 与资源快照 |

缺少上述证据时，页面必须显示 UNKNOWN，不能根据空白色块推断任务状态。

### 5.3.3 资源与依赖排查

若任务已经 ready 但没有及时运行，系统继续区分：

| 候选原因 | 所需证据 |
|---|---|
| AIC 不可用 | 目标 AIC 占用、队列、并发上限 |
| AIV 不可用 | 目标 AIV 占用、队列、伙伴核约束 |
| 内存不足 | UB／L1／L0／GM 容量、分配失败或等待事件 |
| DMA／搬运资源忙 | DMA 队列、端口占用、Copy 任务状态 |
| 同步资源等待 | Barrier、Flag、Signal、Collective 与 peer 到达状态 |
| Runtime 队列限制 | Ready Queue、Task Ring、Dependency Pool 水位 |
| 调度策略限制 | 优先级、亲和性、OoO、Scope 或并发限制 |

页面不要求一次证明所有资源都可用，但必须列出已排除、仍可能和无法检查的项目。

### 5.3.4 上游瓶颈分解

若下游任务因依赖未满足而未 ready，系统沿关键依赖链分解前驱时间：

```text
上游总耗时
├─ Compute
├─ Data Movement
├─ Synchronization / Communication
├─ Runtime Queue / Dispatch
└─ Device Start / Pipeline Stall
```

每一段必须可点击回到对应 Task、指令阶段、泳道区间或原始事件。若事件粒度不支持拆分，则显示“任务总时长已知，内部构成未知”。

### 5.3.5 “最小因果切片”的准确定义

最小因果切片是第三步中的一个下钻视图，不是六步工作流本身。

定义：

> 围绕选中的异常时间窗，只保留解释目标下游任务为何在该时刻启动所必需的前驱任务、依赖／同步事件、状态迁移和资源阻塞事件。

默认范围：

- 目标下游任务；
- 直接未满足依赖；
- 每条依赖上的最后一个关键生产者；
- 与等待时长重叠的同步或资源事件；
- 下游从 ready 到 started 的生命周期事件。

视图标题应描述当前判断，例如：

- “下游未 Ready：等待 `attention_signal`”；
- “已 Ready 未下发：AIV 队列受并发上限阻塞”；
- “状态未知：缺少 Ready 与 Dispatch 事件”。

不再使用宽泛的“从前驱到下游启动”作为默认标题。

### 完成标准

第三步必须给出以下三种状态之一：

1. 已确认直接等待原因；
2. 有证据支持的候选原因及未排除项；
3. 数据不足，并明确缺失字段与补采集方式。

---

## 5.4 第四步：追溯编译变化

### 用户问题

- 当前任务／子图为何被切分、融合、插入 Copy 或同步？
- 哪个 Pass 首次产生了这一结构变化？
- 哪个 Pass 只是物化结果，而不是作出决定？

### 追溯链

```text
Runtime Task
  → Execute Call
  → Block / Subgraph
  → Pass 前后结构 Diff
  → 首次变化 Pass
  → 决策原因与约束
  → 源码或配置来源
```

### Pass 事件最小字段

| 字段 | 含义 |
|---|---|
| `pass_name` | Pass 名称 |
| `before_graph_ref` | 变换前图或 IR |
| `after_graph_ref` | 变换后图或 IR |
| `affected_node_ids` | 被切分、融合、插入或删除的节点 |
| `decision_type` | partition、fusion、copy、sync、memory、schedule 等 |
| `reason_code` | 容量、依赖、布局、合法性、硬件映射等结构化原因 |
| `reason_detail` | 阈值、估算量、约束冲突与候选方案 |
| `source_refs` | 对应源码、Tile、Loop 或 Scope |

### 页面产出

- 首次发生结构变化的 Pass；
- Pass 前后局部图 Diff；
- 新增、删除、切分、融合、搬运和同步节点清单；
- 决策原因与阈值；
- 该决策对任务数、搬运量、同步和并行度的静态影响。

### 展示 Pass 对照前的关联要求

必须能说明“为什么检查这个 Pass”，并提供以下关联：

1. 当前异常的运行记录，以及正在排查的具体任务或资源问题；
2. 该任务与编译节点／子图的可追溯对应关系；
3. 相邻快照中影响这些节点的变化，以及它可能影响计算、搬运、同步或并行度的具体方式。

这不要求先证明最终根因，但需要可检验的关联。函数同名、位于同一归档、Pass 名称看似相关或任意位置出现新增节点，都不能替代以上依据。选择器仅容纳满足关联要求的变化；不允许遍历全部 Pass 后挑一个“有变化”的结果填充页面。

当前 01 / 02 的 Trace 可以定位等待 Task 与直接前驱，却缺少最后到达数据的 Rank／Worker／时间及 Task→编译节点映射。Step 04 因此显示“尚未定位到相关的编译变化”，列出对应等待事件和前驱，支持在主 Trace 定位，并提供“返回 03 · 排查上游耗时”。规则空闲调查尚未确认阻塞任务时，返回 Step 02 定位任务。此状态不代表不存在编译问题。

### 已关联但缺失决策日志时的降级

Pass Dump 只能证明“某结构在两个阶段之间发生了变化”，不一定能证明“为什么变化”。没有 `reason_code` 或决策记录时，界面应显示：

> 结构在所选快照之间发生变化；形成原因未知。以下解释仅为候选假设。

只有完整追踪相关结构、核验相邻快照和跨阶段对应关系后，才能进一步表述“首次出现在 Pass X”；仅比较 Frontend 与某个后续阶段不能作此结论。

### 显式模拟补全（v0.13）

模拟仅通过 `?demo=compiler` 显式进入 01 / Rank 0 的 Step 04；常规调查头部不显示模拟切换。Step 04 直接显示带口径标识的 Pass Diff，不设置“查看模拟 Pass Diff”二次按钮。02 与规则空闲调查不套用此模拟场景。

浏览器地址与标题必须同步：存在 `demo=compiler` 时标题标为“模拟 Pass Diff”；使用常规入口或切换到不支持该场景的调查时删除此参数，标题标为“真实证据”。刷新不得出现网址标记模拟而正文仍为真实缺口页的矛盾状态。

固定场景：模拟生产者 `SIM-PRODUCER-01` 的片上峰值为 96 KiB，超过示例预算 64 KiB。图的真实基线取自 Pass 23 的任务链；变化侧借用 Pass 24 在其他 SPMD 任务中实际出现的 `gm_pipe_buffer_0 = pl.tensor.create(..., manual_dep=True)` 与追加函数入参形式，模拟其被应用到 `csa_merge_pack_publish_spmd`。模拟代价仍设为 320 µs。真实归档中的 Pass 24 没有修改该 producer，因此 producer 映射、容量预算、耗时和致因结论均为演示定义，不声称是目标设备事实。

| 步骤 | 模拟内容与交互 |
| --- | --- |
| 01／02 | 继续显示真实 Trace 现象和真实任务，主泳道不改写 |
| 03 | 以模拟等待起点为 0：搬运 200–520 µs，发布 520–647 µs，信号到达 663 µs；消费任务 664 µs 入队、666 µs 下发、668 µs 开始。明确显示模拟记录，不能与真实 Rank 1 时钟混比 |
| 04 | 直接显示 Pass 23→24 对照，不再经过按钮；节点采用真实 IR 名称、任务 ID 和依赖边。变化侧新增 `gm_pipe_buffer`，经修改后的 `pl.spmd_submit`、`publish_tid` 连到 `o_group_a2a_wait`；hover 显示归档行、真实语法与模拟口径；默认 Fit 全图 |
| 05 | 演示参数 `DEMO_PUBLISH_TILE_ROWS` 从 64 改为 32，峰值降到 48 KiB；可折叠示例配置明确标为非工程源码。采用方案只更新本页模拟会话 |
| 06 | 点击生成模拟结果：移除上述 320 µs 搬运，等待 663→343 µs，局部场景 710→390 µs；点击对比在主工作区打开两份全模拟 Trace。正确性和真实端到端收益仍显示未执行／未测量 |

模拟等待 663 µs 借用真实 01 的时长作为场景尺寸，其分解和候选结果均为构造数据。其余开销固定是演示假设，不构成真实收益预测。图节点同时显示真实参考文件与行号、以及“真实语法 + 模拟映射”口径；Inspector 显示“模拟事件”，不得把参考行号描述成当前异常的已证实来源。

模拟状态仅存在专用内存会话，不进入 localStorage 的 Case／真实实验、不修改 `INVESTIGATION_DATA`，不将模拟模型加入真实 Run 注册表。离开显式 Demo URL 后移除模拟对比页签，恢复真实诊断缺口。真实工程的导入、执行服务与采集能力不因模拟流程完成而改变。

### Pass IR 直接对照与来源契约

满足上述关联要求后，Step 04 正文直接呈现相关子图前后对照，不增加“打开 Pass IR”中间按钮，也不把主 Trace 切换为另一页。有关联的快照选择合并进图头部；图节点通过 hover 或键盘聚焦显示算子、状态、符号和 IR 片段，不再点击跳转。优先复用 `/Users/yin/pto/pass-ir/` 的图解析、布局和渲染栈，不重新绘制一套近似计算图。

| 内容 | 明确名称与用途 |
| --- | --- |
| 模型计算源码 | 开发者编写的 scope、Loop 和计算结构；按钮命名“查看模型源码” |
| 编译 IR 快照 | `passes_dump/*.py` 的阶段产物；按钮命名“查看 IR 文本” |
| Pass 实现源码 | 编译器变换的实现；仅有真实文件映射时提供“查看 Pass 实现”，不能用模型源码代替 |

- 展示关联子图时默认 Fit 全图，突出与调查目标有关的新增、删除、边界、搬运与同步变化。节点详情使用 hover／聚焦提示；没有关联变化时保留调查断点，不改选无关 Pass。
- 默认比较相邻快照；跨多个 Pass 的比较显式标注范围，不归因于其中单个 Pass。
- 旧版 39→40 对照及 00 到 09／25／48 的跨阶段选项已退出诊断默认流程，不能直接作为单 Pass 决策证据。
- 当前 Pass IR 工具接收计算图 JSON，DeepSeek 归档是 Python 形式 IR；必须新增格式适配或接入编译器结构化导出，不能把 `.py` 直接当成现有 JSON。
- 适配输出保留快照、函数、scope、节点与边的来源位置；数据边、控制边、调用关系分别表示。无数据输入的副作用语句可以连接到源码中的前一项操作，但必须标为“源码顺序”，不得冒充数据依赖、运行时调度顺序或性能因果。不支持的语法标为未知，不能虚构语义关系。
- 49→50 `InsertCommFence` 中的 `pl.system.cacheinvalid()` 位于 wait 循环之后；当前没有它与 01 等待异常的关联证据，必须从诊断内容中移除，不得通过附加免责声明保留默认展示。
- 不执行用户导入的 Python 文件来“读取”工程或绘图；静态解析失败时保留原始 IR 文本与错误位置。图是结构证据，不替代缺失的决策理由。

### 完成标准

用户能够区分：

- 首次引入变化的 Pass；
- 后续物化或保持该结构的 Pass；
- 编译器明确记录的理由；
- 工具根据结构提出但尚未验证的解释。

---

## 5.5 第五步：制定调优方案

用户界面不再使用含糊的“修改入口”作为步骤名称。此步回答“改什么、为什么改、风险是什么”；内部数据契约仍可使用控制入口这一术语。

### 用户问题

- 应该从哪里开始修改？
- 修改会通过什么因果路径影响当前异常？
- 它可能带来哪些副作用？

### 调优入口分类

| 类型 | 示例 |
|---|---|
| 源码计算结构 | 生产者／消费者关系、通信组织、计算阶段拆分 |
| TileShape | 行列 Tile、Chunk、分块数量 |
| Loop | range、parallel、pipeline、展开与顺序 |
| 融合范围 | Scope、Inline、Group、子图边界 |
| 调度策略 | OoO、优先级、亲和性、并发度 |
| Runtime 参数 | Worker 数、队列容量、Signal / Barrier 策略 |
| Pass 参数 | Partition 阈值、Fusion 条件、内存规划策略 |

### 调优建议卡字段

每个建议必须包含：

| 字段 | 内容 |
|---|---|
| 修改对象与位置 | 文件、行、符号、参数或 Pass 配置 |
| 当前值 | 当前源码或编译配置值 |
| 建议实验值 | 一个或多个候选值，不直接替用户修改 |
| 依据 | 来自哪个 Runtime、图或 Pass 证据 |
| 因果路径 | 为什么该入口可能影响当前异常 |
| 预期直接变化 | 搬运、任务、同步、资源或均衡中的哪一项 |
| 预期最终指标 | 空洞、P95/P99、吞吐或端到端耗时 |
| 风险 | 容量、精度、并行度、其他 shape 或硬件的回退 |
| 置信度 | 已证实规律、强假设或探索性实验 |

### 推荐映射示例

| 观察 | 候选入口 | 预期直接变化 |
|---|---|---|
| UB spill 与 Copy 增多 | TileShape、融合范围 | 降低工作集或减少 spill Copy |
| 子图边界过多 | 融合 scope、Partition 参数 | 减少 Task 与同步边界 |
| publish 等待受慢 Worker 主导 | Worker 数、工作划分、Loop | 缩短同步等待并改善核间均衡 |
| 已 Ready 但 queue delay 长 | OoO、优先级、并发限制 | 降低排队与下发延迟 |
| 真正关键前驱 Compute 过慢 | 源码计算结构、Tile、Pipeline | 缩短 Compute 时间 |

### 完成标准

任何“建议修改 X”的结论都必须能回溯到具体证据，并明确其预期改变的中间指标。

`ATTENTION_PUBLISH_WORKERS`、`COMM_ROW_TILE` 等当前预设候选不等于已证实的修复方案。先检查工程当前值、候选值合法性、信号计数、整除／尾块与资源约束；无法验证时标为探索性实验，不承诺定量收益，不自动应用修改。

---

## 5.6 第六步：运行实验与对比

### 用户问题

- 修改是否改变了预期的图结构或 Runtime 行为？
- 局部指标改善是否转化为端到端收益？
- 是否出现新的资源、正确性或其他 shape 回退？

### 实验定义

每次验证至少包含：

- 基线 Run；
- 候选 Run；
- 修改文件与配置 Diff；
- 模型、输入、Shape、硬件、版本和采集配置；
- 调优假设；
- 预期改变的直接指标；
- 正确性门禁和性能判定阈值。

实验定义不是孤立下载文件，还必须绑定到一个可执行的工程上下文：

- `workspace_ref`：PyPTO 开发工程位置；只导入构建归档时另标为“归档浏览”，不视作已绑定源码重编译工程；
- `source_revision`：不可变源码 revision 或内容摘要；分支名称仅作辅助信息；
- `working_tree_state`：是否存在未提交修改及其 Diff 引用；
- `edit_target`：源码文件、符号、参数和修改前后值；
- `build_entry`：编译或生成构建产物的入口；
- `correctness_entry`：正确性验证入口；
- `performance_entry`：性能运行和 Trace 采集入口；
- `artifact_contract`：候选 Run、Trace、Pass Dump、指标与日志的输出位置。

目标路径是直接导入工程、预检查、用户确认修改、执行实验与自动回收结果。系统管理 `experiment_id`，开发者不需要手工复制 ID 或反复导出／导入结果。手工导出实验定义与导入结果包仅作为离线降级，两种形态使用同一 Schema；复制命令不等于已执行。

### 直接导入与执行契约

1. 导入源码工程与基线归档，识别入口、源码、配置、Pass Dump 和各 Rank Trace；本例允许默认预载已核实的 `Data/DeepseekV4/` 文件。
2. 展示识别结果及来源，自动填充可从文件确认的信息；缺失项才要求补充。目录选择不是只列出 JSON 让用户逐个挑选 Trace。
3. 检查 PyPTO／工具链／辅助模块、源码版本、设备或执行服务、输入和采集配置；源码存在、依赖齐备、可编译、可设备运行分别记录，不能混成“导入成功”。
4. 用户审查单变量 Diff 与运行范围后，在独立候选目录应用修改；不得覆盖原始数据、基线产物或用户未提交修改。
5. 从源码重新编译，运行真实正确性验证，并在相同条件下对基线和候选重复测量及采集。历史基线无法复现时，重新采集新的基线，不能将不一致条件直接混比。
6. 自动收集 manifest、Trace、Pass Dump、指标、校验与性能日志，关联候选 Run；失败或取消保留日志和阶段状态，不生成成功结论。
7. 主工作区打开真实 Before／After 泳道，抽屉显示假设验证结果及证据链。

工程导入只做读取和索引；执行需要受控本地服务或远端 runner。导入文件里的命令是待验证数据，不因导入而自动执行。没有执行服务时仍可浏览工程和既有产物，但明确显示缺失能力，不以假进度或模拟候选冒充端到端完成。

### 主 Trace Diff 契约

- 基线与候选必须为独立 Run；同一 Run 的 Rank 0／Rank 1 是跨 Rank 对照，不是优化前后对比。
- 两侧展示完整 Trace，支持联动缩放、平移、Rank／lane 对应与异常范围定位；可切换 Run 相对起点或明确事件锚点对齐，显示对齐方式。
- 按 10.5 的匹配等级建立逻辑计算范围对应，Worker 数发生变化时展示一对多／多对一与未匹配实例，不强行按序号一一配对。
- 显示任务新增／删除、等待跨度、任务数、长尾、核间分布与端到端指标；Copy 字节或生命周期分段未采集时保留未知。
- 事件对照可进入独立 Inspector；每项差异可追溯到两侧原始事件及产物来源。
- 只有指标 JSON 时页面命名为“指标对照”；两份真实 Trace 未加载前，不显示“泳道 Diff 已完成”。

### 必须对比的指标

| 维度 | 指标示例 |
|---|---|
| 图结构 | 子图数、融合边界、Copy／Sync 节点变化 |
| 任务 | Task 总数、细碎任务数、平均与尾部时长 |
| 搬运 | Copy 次数、字节数、Data Movement 时间 |
| 同步 | Wait／Barrier／Collective 时间 |
| 调度 | ready→enqueue、enqueue→dispatch、dispatch→start |
| 资源 | AIC/AIV 利用率、UB/L1 水位、队列水位 |
| 均衡 | Rank 间、核间耗时方差与长尾 |
| 结果 | 端到端耗时、吞吐、P95/P99、正确性 |

### 结果状态

| 状态 | 判定 |
|---|---|
| 假设成立 | 直接指标和最终指标均按预期改善 |
| 局部成立 | 直接指标改善，但端到端收益不明显 |
| 假设不成立 | 目标中间指标未改善 |
| 引入回退 | 改善目标指标，但正确性或其他关键指标回退 |
| 不可比较 | Run 条件不一致或数据不足 |

### 完成标准

页面必须把“预计减少什么”和“实际减少什么”并列展示，不能只给优化后单次结果。

---

## 6. 页面信息架构

### 6.1 一级入口与用户路径

```text
打开 Run / 默认加载 DeepSeek V4 Trace
  → 浏览、搜索、缩放或筛选完整泳道
  → 在异常区间点击“调查此区间”，或框选后点击“调查此选区”
  → 创建／恢复 Case，打开非模态调查抽屉
  → 在泳道、图、Pass、源码之间保留 Case 上下文
  → 创建并运行工程实验
  → 基线与候选 Trace 对比，回写调查结论
```

系统发现的异常只能称为“候选异常”，并显示检测规则、统计范围和置信度。当前 DeepSeek V4 的两个 Case 是基于真实数据整理的预设调查点，默认标为“已标注”，不得表达为算法已经自动发现全部空泡。

### 6.2 IDE Frame 页面骨架

新基座使用 PTO Design System 的 `ide-frame` 作为唯一页面框架，并通过 `PtoIdeFrame.init` 接管标准窗体、面板插槽和 resize 行为。

```text
┌────────────────────── PTO IDE Frame ──────────────────────┐
│ Trace Investigator · 数据源       Search       事件 / 主题 │
├───────────────────────────────────────────┬───────────────┤
│ Rank · 泳道 · 异常调查          显示 / 缩放 │ Task Inspector│
│ Trace / 实验对比 · 按上下文切换            │ 事件详情      │
│ 上部可继续浏览、缩放、点击事件             │ 独立选择      │
│ ┄┄┄┄┄┄ 可拖动高度的悬浮抽屉上沿 ┄┄┄┄┄┄┄┄ │ 不切换 Case   │
│ 调查点列表 │ 六步调查 / 问题 / 证据 / 实验 │               │
│ 来源、范围 │ 结论 / 依据 / 未知项 / 下一步 │               │
│           │ Step 04 直接 Pass 图；Diff 联动│               │
├───────────────────────────────────────────┴───────────────┤
│ Run / Rank / lanes / 单位 / 解析状态                       │
└───────────────────────────────────────────────────────────┘
```

插槽分工：

| IDE Frame 区域 | 本产品内容 |
|---|---|
| Global search | Task、lane、label、ID 搜索 |
| Topbar 数据源 | 当前数据源与低频文件／工程导入；实验对比从实验上下文打开，不恢复 Workspace 栏 |
| Preview header / toolbar | 合并成一行：Rank、泳道范围、异常调查；右侧显示设置与缩放；框选保留可发现的辅助入口 |
| 调查点入口 | 唯一列表位于抽屉内，主 Trace 区间标记与选区动作使用同一数据集合；无第二套异常弹窗 |
| Preview | 默认完整 Trace；实验产生真实候选后可切换泳道 Diff；Step 04 不切走 Trace |
| Inspector | 默认以 360px 最小宽度打开；无选中对象时显示当前 Rank 与可见泳道汇总，选中后显示真实或模拟事件详情；与当前 Case 和步骤独立 |
| 底部悬浮区域 | 在 Preview 内绝对定位；PTO pane 承载六步流程，workbench-shell 拖拽内核调整高度 |
| 抽屉内证据下钻 | 生命周期、少量依赖关系、Step 04 Pass 前后图、源码片段与日志；泳道 Diff 在主工作区展示，无第二个底部 Dock |
| Status strip | 数据源、Rank、时间单位、过滤状态、解析告警 |

独立页面使用 `data-host="standalone"`；未来进入 VS Code Webview 时使用 `data-host="vscode-webview"`，并让宿主 IDE 提供文件树、编辑器、终端和全局状态。Webview 内不模拟一套 VS Code 外壳，也不在只读面板内直接编辑源码。

### 6.3 调查抽屉

调查抽屉是覆盖在泳道下部、可手动调整高度的非模态区域。“悬浮”指叠加在画布之上，不是鼠标悬停即展开；点击调查点或框选后打开。无遮罩、不锁定全页焦点，打开和拖动高度不改变 Canvas 尺寸、时间视口与滚动位置。右侧 Inspector 不被抽屉覆盖；点击事件只更新 Task 详情。关闭抽屉后保存 Case、步骤和当前证据状态。

拖拽调用 `PtoWorkbenchShell.initResizablePanes`：透明上部空间与下部 PTO pane 构成绝对定位的 vertical split，透明区域允许指针穿透，只有 gutter 和抽屉接收操作。上沿支持指针及方向键调整，比例保存至 `gap-v4-drawer-height`。窄屏仍使用底部非模态抽屉，列表与正文切换；Task 详情保留独立 Sheet，关闭后回到原调查。

- 顶部：Case 名称、症状类型、Rank 与固定调查时间范围；不重复放置证据模式和待取证状态控件；
- 导航：紧凑的 01—06 步骤，不重复展示步骤说明；
- 正文：当前问题、结论等级、直接证据、未知项和一个主要下一步；
- 按需入口：七个诊断问题、完整证据、补采集、原始数据；
- 底部：上一步、下一步及当前唯一主要动作。

#### 调查内容复用与主工作区契约（v0.6，替代 v0.5）

复用原报告的问题、证据和诊断能力，不再要求完整报告布局原样嵌入。既有 `investigation/report.js`、`report.css` 与只读备份可作为内容和组件来源；备份保持不变，不增加第二套页面 shell。删除独立时间线不等于删除诊断证据，而是将其呈现和交互迁到唯一主 Trace。

| 步骤／区域 | 抽屉保留 | 主工作区联动 |
| --- | --- | --- |
| 调查列表与标题 | 统一 registry、Case 上下文、状态、七问与证据入口；PTO `cp-btn` 六步导航 | 同一 Case 标记，不增加重复异常列表 |
| 01 确认异常 | 范围、指标、口径与调查方向 | 定位异常区间、受影响 lane；跨 Rank 对照也使用主 Trace |
| 02 定位任务 | 相关 Task 表、实例选择、映射缺口与关系证据 | 跳转真实事件，逻辑 Task 高亮全部实例；不止打开 Inspector |
| 03 排查原因 | 生命周期、依赖与资源、上游耗时、最小因果切片四类诊断 | 选中的证据高亮主 Trace 或关联图节点 |
| 04 追溯编译变化 | 已关联：直接显示相关 Pass 前后图并 Fit；未关联：显示当前任务、追查断点及返回上游排查的操作 | 进入步骤保持主 Trace 视口；明确点击任务后才定位 |
| 05 制定调优方案 | 假设、修改对象与位置、当前／候选值、指标、风险及确认动作 | 联动模型源码或相关图范围 |
| 06 运行实验与对比 | 工程检查、修改审查、执行状态、验证结论 | 真实 Before／After Trace 与对应结构变化；只有指标时明确降级 |
| 阅读与材质 | 保留 PTO 组件、单列证据阅读、正文滚动及固定操作区；使用 `data-surface="solid"` | 不复制 Trace 的 Rank、缩放、框选状态，不让大图挤占抽屉正文 |

调查列表在抽屉左侧常驻，不设置左上角展开／收起按钮；窄屏缩窄列表但仍与调查正文同时存在。跨 Rank 定位遵循 5.2，主视图可以切换，但 Case 归属不变。抽屉的打开、收起和手动 resize 本身不得改变 Trace 浏览状态；显式“定位事件”“回到调查范围”等上下文动作才调整视口。

验收以证据覆盖和真实联动为准，不再要求抽屉中出现双 Rank 12 条任务汇总图。保留四类原因诊断与未知项，验证主 Trace 定位可见、Pass IR 来源可追溯、实验对比是真实产物。

调查范围和浏览范围必须分离。开发者为查看前驱而平移或缩放泳道时，Case 的 `time_range` 不随之变化；只有执行“更新调查范围”才修改 Case。一次集中展示一个 Case，多个候选异常通过抽屉内统一列表切换，各自保存进度。

统一调查点契约：人工标注显示“具名等待”，规则候选显示“Worker AIC 共同空闲 ≥ 80 µs”，手动 Case 显示“手动框选”。规则固定基于当前 Run / Rank 的完整 Worker AIC 集合，包含 Run 两端，不随显示过滤器改变；没有 AIC 泳道时不生成共同空闲候选。列表与画布标记共用 key、编号与时间范围；打开已有候选复用 Case，不重复追加。时间重叠但统计语义不同的调查保留来源说明，不据此合并为已确认的同一个异常。桌面保留列表，可从抽屉左上“调查点”收起或重新展开；窄屏选择后收起。标记必须可识别、可聚焦、可点击，悬停／聚焦显示调查动作；避免为每个候选同时铺满整列背景，但不能以难以命中的细线和编号作为唯一入口。

### 6.4 工作流状态

每一步显示以下状态：

- 未开始；
- 有候选；
- 已确认；
- 数据缺失；
- 待验证；
- 已验证。

用户可回到任一步，但系统不得允许后续建议掩盖前序证据缺失。例如第三步为 UNKNOWN 时，第五步只能显示探索性实验，不能显示确定性修复。

### 6.5 证据呈现

抽屉正文只显示支撑当前回答的最小证据集，其余内容从“证据”按需展开。Task 详情与调查使用独立状态和容器：查看事件、依赖高亮或关闭 Inspector 均不清空 Case 和六步进度；切换调查步骤也不清空已选 Task。避免使用“返回调查”让用户在两者之间反复切换。

### 6.6 大视图下钻规则

- Task 生命周期和少量依赖关系可在抽屉内展示；
- Step 04 在满足 5.4 的关联要求时直接展示 Pass 前后图；数据未关联时显示追查断点。大型任务图和泳道 Diff 可在主工作区切换。源码片段与日志可在抽屉下钻，长源码优先进入源码视图／真实工程位置；完整 Execute／Block 映射缺失时说明缺少哪些对应关系；
- 返回 Trace 时恢复滚动、缩放、过滤、选区与当前 Case；显式事件定位则按 5.2 更新浏览状态，不修改调查状态；
- “最小因果切片”是第三步的“查看相关依赖链”，不能脱离等待状态证据宣称完整因果链。

---

## 7. 结论与证据等级

### 7.1 证据来源标签

| 标签 | 含义 |
|---|---|
| RUNTIME | Runtime 事件或状态事实 |
| HARDWARE | PMU、资源与设备实测 |
| COMPILED | IR、Pass Dump、Codegen 中确认的事实 |
| SOURCE | 源码和配置中的直接事实 |
| DERIVED | 由明确规则从多个事实推导出的结论 |
| HYPOTHESIS | 尚待实验或补采集验证的候选解释 |
| UNKNOWN | 当前数据无法回答 |

### 7.2 结论状态

| 状态 | 使用条件 |
|---|---|
| CONFIRMED | 有直接证据，且关键替代解释已排除 |
| SUPPORTED | 多项证据支持，但仍存在未排除项 |
| CANDIDATE | 合理候选，尚缺关键证据 |
| UNKNOWN | 无法从当前数据判断 |
| DISPROVED | 已被证据或实验否定 |

`DERIVED` 是证据类型，不等于结论已经确认。任何推导都必须展示使用的输入事实和规则。

### 7.3 文案约束

允许：

> 观察到 `o_group_a2a_wait` 与空洞重叠 663.00 µs；它支持“同步等待主导该时间窗”的解释。

不允许：

> 空洞期间没有 Ready Task。

除非存在该时间窗内完整的 Task 生命周期和 Ready Queue 证据。

---

## 8. 数据契约

## 8.0 原始 Trace 解析与泳道身份

基座直接接受 Chrome Trace / Perfetto 格式的 `traceEvents`，并保留 `M`、`X`、`C`、Flow 等原始事件供下钻。UI 的统计对象必须显式选择事件层级，不能把不同进程中的同名线程自动合并。

泳道稳定身份为：

```text
run_id + rank_id + pid + tid
```

`thread_name` 和 `process_name` 仅用于显示和分类。`AIC_0` 等名称会同时出现在 Worker View 和 Scheduler View 中，不能作为唯一键。默认执行利用率只统计 Worker View 的执行事件；Scheduler、AICPU 和 Orchestrator 作为独立泳道组按需显示。若同一执行单元存在重叠事件，利用率计算应基于时间区间并集，最终限制在 0—100%，同时以解析告警提示重叠来源。

时间语义必须区分：

- “空泡”是指定泳道集合在指定区间内没有可计入的执行事件；
- “wait 任务”是一个具名 Runtime 事件及其跨度；
- 两者重叠是调查线索，不自动构成因果证明；
- 跨 Rank 比较默认使用各 Rank 的 Run 内相对时间，只有存在同步时钟依据时才做绝对时间对齐。

## 8.1 Runtime Task 生命周期

目标数据序列：

```text
created → dependencies_ready → enqueued → dispatched → started → finished
```

建议最小 Schema：

```json
{
  "run_id": "...",
  "rank_id": 0,
  "task_id": "r2t56",
  "task_name": "o_group_a2a_wait",
  "target_type": "AIV",
  "target_id": 12,
  "created_at": 0,
  "dependencies_ready_at": 0,
  "enqueued_at": 0,
  "dispatched_at": 0,
  "started_at": 0,
  "finished_at": 0,
  "blocked_reason": null,
  "unresolved_dependencies": [],
  "queue_id": "...",
  "root_hash": "...",
  "call_op_magic": "...",
  "leaf_hash": "..."
}
```

### blocked_reason 建议枚举

- `DEPENDENCY`；
- `AIC_UNAVAILABLE`；
- `AIV_UNAVAILABLE`；
- `MEMORY_CAPACITY`；
- `DMA_BUSY`；
- `QUEUE_LIMIT`；
- `SYNC_SIGNAL`；
- `COLLECTIVE_PEER`；
- `SCHEDULER_POLICY`；
- `DEVICE_QUEUE`；
- `UNKNOWN`。

枚举不能替代详情。事件还需携带被等待对象、资源 ID、当前值、期望值和解除时间。

## 8.2 资源快照

异常时间窗内至少需要：

- AIC／AIV running 与 queued task；
- 核亲和性和可调度集合；
- Ready Queue、Task Ring、Dependency Pool 水位；
- UB／L1／L0 的分配、占用和失败原因；
- DMA／搬运队列；
- Signal、Barrier、Collective 与 peer 到达状态；
- Runtime 并发上限和调度策略。

## 8.3 图与编译映射

每个 Runtime Task 需要稳定关联：

```text
Task ID
↔ Execute Call
↔ Block / Subgraph
↔ Tile / Op
↔ Pass 节点 ID
↔ Source Range
```

`rootHash`、`callOpMagic`、`leafHash` 可作为现有桥梁，但仍需明确每个键的稳定范围、唯一性和跨 Pass 保留策略。

## 8.4 Pass 决策记录

仅保存 Pass Dump 不足以解释“为什么”。编译器应额外输出：

- 决策类型；
- 被评估的候选方案；
- 采用与拒绝原因；
- 容量、成本或合法性阈值；
- 受影响节点与源码；
- 可覆盖的控制参数。

## 8.5 调优控制入口目录

需要维护机器可读的控制入口映射：

```text
诊断特征
→ 可能的控制入口
→ 影响机制
→ 适用条件
→ 风险
→ 验证指标
```

它可以由编译器元数据、源码索引和专家知识共同构成，但来源必须可见。

---

## 9. DeepSeek V4 当前数据验证切片

本节只记录当前仓库数据能够支持的产品展示，不将其扩大为完整根因结论。

### 9.0 默认预加载与已验证兼容性

新基座默认加载 Rank 0 的完整原始泳道，并允许切换 Rank 1 或进入 Rank 对照；用户仍可通过本地文件／目录替换默认数据。

| 角色 | 默认文件 |
|---|---|
| Primary | `_jit_l3_decode_csa_20260903_010617/dfx_outputs/rank0/d0/merged_swimlane_20260903_010746.json` |
| Rank 对照 | `_jit_l3_decode_csa_20260903_010617/dfx_outputs/rank1/d0/merged_swimlane_20260903_010747.json` |

浏览器实测确认现有 `/Users/yin/pto/swimlane/` 的 `?file=` 加载链路可以解析 Rank 0 完整文件，并读到 9,480 个 `X` 事件、79 个按现有规则聚合的 lane 和约 4.9 ms 总跨度。原文件约 9 MB，适合作为真实数据兼容与渲染性能基线。

该结果只证明格式能够加载，不证明现有统计语义正确。当前解析器按 `threadName` 聚合，导致 Worker 与 Scheduler 的同名 `AIC_n` / `AIV_n` 进入同一 lane，页面出现部分利用率超过 100% 的结果。新基座必须先完成 8.0 的身份与统计修正，才能把利用率或空泡标注用于诊断。

两份文件在 Demo 中应通过相对 URL 配置引用，不复制为另一份手工裁剪的默认 Trace。`build-data.cjs` 继续负责生成两个调查 Case 的索引、源码与 Pass 锚点；它不替代完整泳道输入。

### 9.1 Case A：`o_group_a2a_wait`

当前可观察内容：

- Rank 0 的 `o_group_a2a_wait` 为 663.00 µs；
- Rank 1 同阶段为 1.34 µs；
- wait 结束后存在 `o_group_a2a_gather`；
- 源码 `decode_o_proj.py` 中存在 `o_group_a2a_wait`、`publish_count`、`ATTENTION_PUBLISH_WORKERS = 48` 和后续 gather；
- Pass Dump 中可见 wait Task 及其显式依赖；
- Codegen 产物中存在对应 AIV Kernel。

当前可以表达：

> Rank 0 的长时间同步等待与所选空洞重叠，是该时间窗的主要可观察耗时；Rank 间差异明显。

当前不能确认：

- 空洞期间是否存在其他已经 Ready 的 AIC／AIV Task；
- wait 前的慢方具体是哪个 Worker、Rank 或 payload；
- 调度器是否曾有可下发但未下发的任务；
- `ATTENTION_PUBLISH_WORKERS` 是否为最优修改入口；
- 哪个 Pass 决定了当前通信／子图切分形态。

因此当前结论等级应为 SUPPORTED，而不是 CONFIRMED 根因。

### 9.2 Case B：`cp_token_allgather_payload_wait`

当前可观察内容：

- Rank 0 的 `cp_token_allgather_payload_wait` 为 1080.52 µs；
- Rank 1 同阶段为 21.02 µs；
- 源码 `decode_cp_token_allgather.py` 中存在 `COMM_ROW_TILE = 8`、Worker 循环和 payload wait；
- Runtime 命名映射、泳道数据、Pass Dump 和 Codegen 产物中均存在对应任务。

当前可以表达：

> Rank 0 的 payload wait 显著长于 Rank 1，支持“跨 Rank payload 到达存在长尾”的候选解释。

当前不能确认：

- 具体慢 peer、chunk 或链路；
- 慢在发送端计算、数据搬运、同步还是调度；
- `COMM_ROW_TILE` 或 Worker 数变化是否一定降低等待；
- 修改后是否会增加任务数、搬运或其他 Rank 压力。

需要补充 peer／chunk 到达事件和 Runtime stall reason 后，才能进入确定性诊断。

### 9.3 v0.6 审查：当前实现、工程文件与执行边界

核实日期：2026-09-10。以下是本地文件与实现的检查结果，不代表已执行新实验。

| 审查项 | 当前事实 | 对规格的影响 |
| --- | --- | --- |
| 异常入口 | 主 Trace 标记使用 2px 细线和编号，点击区窄；工具栏名为“调查点” | 按 10.1 增加可发现的区间与选区调查动作 |
| 独立任务切片 | `report.js` 使用 `data.js` 的真实任务摘录，多核条为汇总跨度；拥有独立 Rank、缩放与选择状态 | 数据并非无关，但重复主图且口径易混淆，应移除独立图，改为主 Trace 联动 |
| Step 02 | `selectEvidenceTask()` 解析原始事件后仅调用 `selectTask()`，未调用主 Trace 的 `focus()`；跨 Rank 只加载 Inspector | 统一事件定位，不将“打开详情”当作“定位成功” |
| Step 04 | “查看源码”实际指向模型源码；前后对照为 IR 文本，并非 Pass 实现源码 | 明确三类来源，复用 Pass IR 图能力并补数据适配 |
| Step 06 | 保存工程路径与命令文本，未执行；导入结果 JSON 后展示指标表；`artifacts.trace` 未被加载为候选泳道 | 当前“Run Diff”不能作为真实泳道 Diff 或端到端运行验收 |

实现来源：[app.js](../Design/deepseek-gap-investigator/app.js)、[report.js](../Design/deepseek-gap-investigator/investigation/report.js)、[experiments.js](../Design/deepseek-gap-investigator/investigation/experiments.js)、[Trace 渲染器](../Design/deepseek-gap-investigator/render/trace.js)。

#### 已存在的源码与产物

- [源码目录](../Data/DeepseekV4/deepseek_v4_flash_dspark/)：存在模型与子模块源码，不应要求用户重新手填所有已知文件。
- [decode_csa.py](../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_csa.py)：第 1957 行起有实际 CLI 入口，包含 `--compile-only`、`--dump-passes`、`--enable-chip-swimlane`、设备／TP 配置；同时存在输入构造、golden 计算和结果检查代码。
- [decode_o_proj.py](../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_o_proj.py)：当前 `oWait` 引用第 280 行，`oConst` 引用第 66 行；这些是模型源码位置，不是编译器 Pass 实现。
- [decode_cp_token_allgather.py](../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_cp_token_allgather.py)：当前 wait 引用第 123 行，参数引用第 61 行。
- [编译归档](../Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/)：包含 00—51 共 52 个 IR 快照、生成代码／构建产物，以及同一次 Run 的 Rank 0／Rank 1 Trace。
- [debug/run.py](../Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/debug/run.py)：属于构建产物重放脚本；第 111 行标量输入待补，第 127 行比较函数为 TODO。它的不完整不能被用来推断模型源码目录也不存在有效编译或校验入口。
- [PTO Pass IR 工作台](/Users/yin/pto/pass-ir/index.html)与[解析器](/Users/yin/pto/js/parser.js)：现有工具使用图 JSON；本例 Python IR 快照尚未适配。共享脚本顺序和消费者约束须遵守该模块 `AGENTS.md`。

#### 尚未验证或缺失的条件

- 当前 `Data/DeepseekV4/` 目录未找到入口依赖的 `golden` 辅助模块；不等于其他工程或环境中不存在，需要定位并绑定正确版本。
- 模型源码与历史 Trace 的精确 revision、输入、工具链和采集条件对应关系尚未核实；文件名称相关不能替代基线溯源。
- 当前检查主机为 Darwin arm64，原型未连接目标编译／设备执行服务；未验证可用远端环境，不能宣称已经能够运行设备实验。
- 在当前 DeepSeek 数据归档中未找到优化后的候选 Run。Rank 0／Rank 1 不能充当 Before／After；不得合成“优化后”事件作为真实验证结果。

#### 可推进范围

1. 当前文件足以实现工程／归档导入、入口识别、基线浏览与缺口检查。
2. 主 Trace 定位和候选 Trace 导入／Diff 能力可实现；真实优化对比仍需独立候选产物和可比条件。
3. “修改 → 编译 → 正确性 → 采集 → 自动回收 → 泳道 Diff”的完整验收需要补齐依赖、runner、设备及基线条件；未满足时保留明确阻塞状态，不把手工结果表当作已完成的闭环。

---

## 10. 关键交互规格

### 10.1 创建调查

1. 页面先加载完整 Trace，调查抽屉保持关闭；
2. 用户手动框选异常时间窗，或点击系统／预设的候选异常标记；
3. 系统识别相邻 Task、Lane、Rank 和可用关联键，并在创建前展示选区摘要；
4. 用户创建或打开 Case，底部悬浮抽屉进入第一步（已有 Case 恢复原步骤）；
5. 系统预计算六步的可回答程度，标出已有证据与需补采集项；
6. 主泳道持续高亮 Case 时间范围、选中 lane 和锚点 Task。

入口可发现性要求：

- 一行工具栏明确使用“异常调查”，分别说明已标注数量和规则候选数量，不使用无来源的“异常总数”。
- 主 Trace 的区间标记悬停／键盘聚焦时展示范围、现象、来源和“调查此区间”；点击区间标记可直接打开调查。
- 框选完成后就地显示“调查此选区”和范围摘要，不要求用户先发现一个特殊模式；实现必须明确框选与拖动平移的手势区分，并保留按钮／键盘替代操作。
- 抽屉收起后保留当前 Case 的底部恢复入口；不自动弹出遮挡主图的全屏异常列表。
- 工具栏、区间标记、选区动作和恢复入口访问同一 registry，重复打开不新增同一 Case。

### 10.2 在六步间导航

- 点击步骤切换主视图；
- 完成当前步骤后自动推荐下一步，但不强制线性操作；
- 每一步保留选中 Task、Rank、时间窗和证据上下文；
- 浏览泳道或打开下钻视图后保持 Case 高亮；
- UNKNOWN 状态可直接生成补采集配置或任务清单。

### 10.3 证据联动

- 点击结论，高亮其全部支持证据；
- 点击任务证据执行 5.2 的统一定位；Step 04 仅呈现满足 5.4 关联要求的 Pass 对照，大型任务图在主工作区打开，源码片段和实验日志可在抽屉中下钻；
- 点击候选解释，显示支持项、反证和未排除项；
- 人工可确认、否定或备注候选解释，但不能覆盖原始证据。

### 10.4 创建实验

- 用户从第五步选择一个控制入口；
- 直接导入或使用已绑定工程，自动识别源码与产物、生成单变量优先的实验定义；检查 revision、依赖、构建、正确性、性能和设备条件，仅对缺失字段要求补充；
- 用户审查具体文件／符号的修改 Diff 与运行范围；
- 接入执行服务时由同一入口触发真实执行和产物回收；未接入时保留工程浏览与离线协议，明确“未执行”，未经验证的命令只标为候选说明；
- 执行产物带 `experiment_id` 和 `case_id` 回传，绑定新的候选 Run；
- 新 Run 完成后，主 Preview 切换到 Trace 对比，第六步展示结构、Runtime、端到端和正确性 Diff；
- 实验结果回写对应假设状态和 Case 历史。

### 10.5 跨 Run 对应关系

重新编译后 Task ID、图 Hash 和子图边界可能变化，目标 wait 任务也可能消失。因此候选 Run 不得只按旧 Task ID 匹配。系统按以下优先级寻找对应计算范围，并显示匹配等级：

1. 稳定编译映射键和明确的跨 Run 节点 lineage；
2. 源码范围、具名 scope、调用上下文与 Rank；
3. 任务语义、邻接关系和时间位置的组合候选；
4. 无可靠对应时标为 UNKNOWN，并要求人工选择，不能静默比较错误对象。

---

## 11. 标准泳道基座副本改造

### 11.1 来源与复用方式

以 `/Users/yin/pto/swimlane/` 为行为参考，将可用能力整理为当前产品可维护的基座副本。不能通过 iframe 嵌入旧页面，也不能整体复制其历史 UI 和私有视觉类。新副本应把“数据解析、泳道模型、渲染内核、视口交互”与“页面框架、调查业务、工程闭环”分层。

候选实现审查：

| 候选 | 可复用价值 | 不作为主基座的原因 |
|---|---|---|
| `/Users/yin/pto/swimlane/` | `?file=`、本地文件／目录、搜索、缩放、测量、Task、Flow、对比和 Pass IR 桥接最完整 | 选为行为与代码来源，但需移除历史页面层 |
| `/Users/yin/pto/pypto-swimlane-perf-tool/` | 本地 JSON、拖拽与性能分析逻辑 | 页面结构偏分析报告，和六步调查抽屉重复 |
| `/Users/yin/pto/pto-swimlane-profiler/` | 工作台布局、生命周期和图联动可参考 | 数据与页面耦合较强，不是通用 Trace 加载底座 |
| `/Users/yin/pto/swimlane-bench/` | 合成大数据与预处理性能测试 | 作为测试工具保留，不承担产品页面 |

基座副本归入当前 `Design/deepseek-gap-investigator/` 产品模块，运行时不依赖 `/Users/yin/pto/` 的绝对路径。复用代码保留来源说明；PTO 视觉与模式只从当前仓库 `vendor/pto-design-system/` 加载。原 `/Users/yin/pto/swimlane/` 保持不变，避免为了本次产品裁剪影响其他 Demo。

建议目录职责：

```text
trace-workbench/
├─ adapters/       原始 Trace、Rank、Program 与 Case 数据适配
├─ model/          lane/task/flow 索引、统计和跨 Run 匹配
├─ render/         虚拟化泳道、坐标系、命中测试、依赖 overlay
├─ investigation/  六步状态、证据、抽屉与实验协议
├─ components/     仅组合 PTO Design System 组件
└─ index.html       IDE Frame 插槽与启动配置
```

### 11.2 保留能力

| 能力 | 保留方式 | 原因 |
|---|---|---|
| Chrome Trace / core-task JSON 解析 | 提取为 adapter，并补 Rank / pid / tid 语义 | 真实数据入口 |
| `?file=` 和本地文件／目录加载 | 复用加载协议，默认配置 DeepSeek V4 | 支持开箱演示与用户数据 |
| 搜索、过滤、Fit、缩放、滚动 | 放入 Preview pane 的领域工具栏 | 泳道基本浏览能力 |
| 区间选择和测量 | 统一为“框选调查范围”模式 | Case 的主要创建入口 |
| Task 命中与详情 | 复用索引和命中逻辑，详情进入 Inspector | 连接异常与 Task |
| Flow／依赖 overlay | 作为按需能力保留 | 支持最小依赖下钻 |
| Primary / Candidate 对比 | 重命名为基线 / 候选 Run，在 Step 06 解锁 | 验证闭环 |
| Program / Pass IR 桥接 | 保留数据协议，视图进入抽屉内证据区 | 连接编译证据 |
| 大数据虚拟化思路 | 保留可见区渲染与命中测试 | 真实 9 MB Trace 的性能需要 |

### 11.3 删除或延后能力

| 旧能力 | 处理 | 原因 |
|---|---|---|
| Before / After 内置样例切换 | 删除 | 与当前 Run / Rank / 实验语义重复 |
| 默认自动展开 Journey panel | 删除 | 六步调查抽屉已承担引导 |
| 自动生成的“全局认知／找瓶颈／深入任务”卡 | 删除 | 重复工作流，且部分启发式结论缺证据 |
| Task popup、Detail panel、Journey panel 三套详情入口 | 合并 | 统一进入 Inspector 的 Task 或 Case 状态 |
| `sw-analysis-row` 和隐藏 insight ghost nodes | 删除 | 无可见产品价值，仅维持历史数据绑定 |
| 独立资源大面板 | 收敛为 Trace header 中的“打开 Trace” | 加载资源是低频上下文操作，不占主分析区域 |
| Workspace / Explorer 左栏 | 删除 | Rank、泳道范围、Case 和数据加载都与泳道顶部重复，却持续占用水平空间 |
| Activity Rail | 删除 | Explorer、搜索、工程和终端入口已有更贴近上下文的位置 |
| 全局证据 / 终端按钮 | 默认隐藏或删除 | 尚无证据或未连接执行服务时是死入口；抽屉内证据由具体下钻动作打开 |
| 原型备份 / Spec 链接 | 从运行时状态栏删除 | 属于设计评审导航，不属于诊断任务 |
| 默认 Bubbles 开关和按气泡排序入口 | 归入统一异常调查或高级显示 | 避免把启发式气泡当作根因 |
| Marker 的 `window.prompt` 编辑 | 替换为 Case 创建／重命名流程 | 接入稳定调查对象与可访问表单 |
| 任意时刻显示 Compare / Diff | 延后到 Step 06 或显式加载参考 Run | 减少默认模式和无上下文对比 |
| 私有 `sw-*-btn`、`sw-*-panel` 视觉组件 | 删除 | 统一调用 PTO 组件和模式 |

### 11.4 PTO Design System 调用契约

本节列出该产品范围内的完整组件契约。实现顺序固定为：PTO 已有组件或 Pattern → 语义化原生元素 → shadcn/ui 开源组件的结构与样式适配。不得跳过已有 PTO 能力直接创建私有组件。

#### 11.4.1 基础资源加载顺序

```text
tokens/foundation.css
→ tokens/semantic.css
→ tokens/components.css
→ css/style.css
→ patterns/workbench-shell/*
→ patterns/ide-frame/*
→ patterns/swimlane-task/*
→ 按实际打开视图加载 pass-ir-graph-node 等可选 Pattern
→ 本产品只含布局与 shadcn 适配的 components/ui.css
```

业务页面不得复制设计系统文件、覆盖 `.pto-workbench-shell__*` 内部规则，或用本地颜色、圆角、阴影重建 IDE Frame。独立页面使用 `data-host="standalone"`，并让 IDE Frame 直接占满浏览器 viewport；优先把 `.pto-ide-frame[data-ide-frame]` 直接设在 `body`，不在外层保留带 padding 的展示页，不增加固定宽高比、圆角、边框或整框阴影。`data-preview-frame="true"` 仅供设计系统文档预览，产品页禁用。VS Code Webview 使用 `data-host="vscode-webview"`。

#### 11.4.2 PTO 原生组件与 Pattern 矩阵

| 页面对象／场景 | PTO 调用契约 | 状态与限制 |
|---|---|---|
| 工作台框架 | `patterns/ide-frame`、`PtoIdeFrame.init(root, options)` | Standalone 根节点与浏览器 viewport 同层，优先直接使用 `body`；本产品只使用 Preview、Inspector、contextual bottom dock 和 status slots，不强制填满 Explorer / Activity Rail；不套展示卡片，不自建第二套页面 chrome |
| 面板分割与调整宽度 | IDE Frame 间接调用 `patterns/workbench-shell` | 保留键盘和拖拽 resize；业务代码不直接修改 gutter 内部样式 |
| 顶部领域工具栏 | `.toolbar`、`.toolbar-l`、`.toolbar-r` | 只放 Run 上下文、搜索和高频操作；低频操作进入菜单 |
| Preview 页签 | `.tab-control`、`.tab-control-item` | 用于上下文中的 Trace / 实验对比；Step 04 的 Pass 图不建立页签，选中态不使用 `.btn-solid` |
| Rank、原因与颜色模式 | `.segmented-control.segmented-control-muted` + `.btn.is-selected` | 用于同层互斥选择；必须有 `aria-pressed` 或 tab 语义 |
| 次要、主要、图标操作 | `.btn`、`.btn-solid`、`.btn-ghost`、`.btn-icon`、`.btn-sm` / `.btn-compact` | 每个区域最多一个 `.btn-solid`；图标按钮必须有可读名称 |
| 缩放与读数 | `.toolbar-control`、`.toolbar-readout` | Fit、放大、缩小和比例读数作为一组 |
| 内容面板 | `.panel-shell*` 或 `.panel-shell.panel-shell-quiet` | 仅用于独立区域；不把每个字段包装成卡片 |
| 解释与建议卡 | `.card-demo*` | 只承载需要明显成组的假设、建议或实验摘要 |
| Inspector / 调查正文 | `.inspector-rail`、`.inspector-section*`、`.inspector-soft-card` | Task 详情独立在右侧，Case 正文在底部悬浮 pane；两者可同时显示 |
| 空状态 | `.empty-state`、`.empty-card`、`.empty-title`、`.empty-sub`、`.empty-actions` | 区分未加载 Trace、无搜索结果、未创建 Case 和无实验结果 |
| 轻量指标与状态 | `.stat-chip`；状态文字使用 `--success` / `--warning` / `--danger` | 证据来源与结论等级是两组语义，不混成一个标签 |
| Swimlane Task | `PtoSwimlaneTaskPattern.drawTaskBar` | 传入真实 task 数据和 selected / related / emphasized 状态；不复制 DOM 条形实现 |
| Swimlane 配色 | `createTaskColormap`、`colorFromColormap` | 只为语义、执行单元、Stitch、子图使用数据色；界面 chrome 仍用语义 token |
| Task Tooltip | `PtoSwimlaneTaskPattern.initHoverTooltip` | 用于 Canvas Task hover / focus；不再实现第二套 Task tooltip |
| Task 分段 | `buildTaskSegmentSpec` | 只有真实 `inputRawMagic` / `outputRawMagic` 时显示 IN / Compute / OUT |
| Pass / 局部图节点 | `PtoPassIrGraphNodePattern.buildNodeCardElement` 或 `renderCardSet` | 第 2—4 步的节点卡必须由 Pattern 生成 |
| 可选播放控制 | `PtoFloatingPlaybackControl.createControl()` | 只有引入 Trace 时间播放时使用；M0 不默认展示 |

#### 11.4.3 shadcn/ui 缺口适配矩阵

实现核验（2026-09-10）：当前 vendor 文档列出了 `.inspector-section` / `.inspector-soft-card`，但 CSS 未提供实际实现。v0.2 在 `components/ui.css` 中按本节缺口规则，使用 shadcn Item / Separator / Alert 结构补齐，并记录参考 URL；不将仅存在于文档中的类名声称为已调用的原生组件。Inspector 外框仍由 PTO IDE Frame 提供。

下表中的组件在当前 PTO Design System 中没有完整、可直接调用的结构或交互实现。允许参考 [shadcn/ui Components](https://ui.shadcn.com/docs/components) 的开源 DOM 分层、状态属性和视觉节奏进行本地适配；当前原型是原生 HTML / CSS / JavaScript，不声明安装或实际调用 React、Radix、Base UI、Vaul 或 Tailwind 包。现有 `components/ui.css` 中已经完成的 Table / Dialog / Field 适配可以保留，按本矩阵补齐其余缺口。

| 产品对象／场景 | shadcn/ui 参考组件 | 适配要求 |
|---|---|---|
| 窄屏 Task 详情 | [Sheet](https://ui.shadcn.com/docs/components/base/sheet) | 仅窄屏 Task 详情使用右侧 Sheet；调查在所有视口使用底部非模态区域，Header / Content / Footer 复用现有结构 |
| 创建实验、查看全部证据、导入错误 | [Dialog](https://ui.shadcn.com/docs/components/base/dialog) | 使用原生 `<dialog>` 或等价可访问行为；支持 Escape、初始焦点、焦点约束与关闭后焦点恢复 |
| 放弃未保存实验等确认 | [Alert Dialog](https://ui.shadcn.com/docs/components/base/alert-dialog) | 只用于需要明确确认的高影响操作；普通导航不弹确认框 |
| Trace 数据源／更多菜单 | [Dropdown Menu](https://ui.shadcn.com/docs/components/base/dropdown-menu) | 仅从 Preview header 打开；Trigger、Content、Group、Label、Separator、Item 层级完整；支持键盘移动和 Escape |
| Task / lane / label 全局搜索 | [Command](https://ui.shadcn.com/docs/components/base/command) + [Combobox](https://ui.shadcn.com/docs/components/base/combobox) | 支持输入、分组结果、空结果、上下键和 Enter 定位；结果必须区分 Rank、lane 与对象类型 |
| 参数、候选值和运行条件 | [Field](https://ui.shadcn.com/docs/components/base/field)、Input、Textarea、[Select](https://ui.shadcn.com/docs/components/base/select) | Label、Description、Control、Error 信息形成一组；原生控件保持可提交和键盘可用 |
| Lane 多选与显示偏好 | [Checkbox](https://ui.shadcn.com/docs/components/base/checkbox) / Switch | 互斥模式继续用 PTO segmented control；仅真正布尔或多选设置使用 Checkbox / Switch |
| 证据、实验指标和 Run 条件 | [Table](https://ui.shadcn.com/docs/components/base/table) | 保留 Caption、Header、Body、Row、Cell；窄屏允许横向滚动，不把表格拆成重复卡片 |
| 调查抽屉、证据与代码长内容 | [Scroll Area](https://ui.shadcn.com/docs/components/base/scroll-area) | 使用原生滚动作为能力基础，复用其 viewport / scrollbar 视觉；不得劫持滚轮或破坏横向 Trace 滚动 |
| 证据详情、低频运行配置 | [Collapsible](https://ui.shadcn.com/docs/components/base/collapsible) / Accordion | 当前结论和 UNKNOWN 永远可见；仅折叠补充字段、原始参数和低频配置 |
| UNKNOWN、采集缺口、执行失败 | [Alert](https://ui.shadcn.com/docs/components/base/alert) | 使用标题、说明和可选动作；颜色映射 PTO warning / danger / info token，不靠图标单独表达状态 |
| 非 Task 图标说明 | [Tooltip](https://ui.shadcn.com/docs/components/base/tooltip) | 只补充图标按钮或截断文本；Task hover 继续使用 PTO Swimlane Tooltip |
| 异常标记 | PTO Swimlane 标记 + 抽屉列表 | 点击编号直接打开对应 Case；不再显示中间摘要弹窗。编号和列表来自同一集合 |
| Trace 解析、索引和实验执行进度 | [Progress](https://ui.shadcn.com/docs/components/base/progress) + Spinner | 已知总量用 Progress；未知时长用 Spinner；同时提供文字状态和失败出口 |
| 首屏 Trace / Inspector 加载占位 | [Skeleton](https://ui.shadcn.com/docs/components/base/skeleton) | 骨架形状对应真实布局；不伪造可交互 Task，不长时间替代明确错误状态 |
| 保存 Case、导出完成、导入成功／失败 | [Toast](https://ui.shadcn.com/docs/components/base/toast) | 短暂、非阻塞反馈；关键失败仍在相关区域显示，不只依赖 Toast |
| 列表和分区边界 | [Separator](https://ui.shadcn.com/docs/components/base/separator) | 使用 `--border-subtle`；优先由 Inspector section 自带分隔，不重复画线 |
| 快捷键提示 | [Kbd](https://ui.shadcn.com/docs/components/base/kbd) | 仅显示实际可用快捷键，例如 `/` 搜索、`Esc` 关闭浮层 |

#### 11.4.4 shadcn 样式到 PTO token 的映射

| shadcn 视觉角色 | PTO token |
|---|---|
| background | `--background` |
| card / panel | `--surface-1`、`--surface-2`、`--card-bg`、`--panel-bg` |
| popover / dialog | `--background-elevated`、`--panel-shell-bg` |
| foreground / muted foreground | `--foreground`、`--foreground-secondary`、`--foreground-muted` |
| border / separator | `--border-default`、`--border-subtle` |
| input / focus ring | `--input-bg`、`--input-border`、`--focus-ring` |
| accent / selected | `--state-hover`、`--state-press`、`--state-selected` |
| destructive / warning / success | `--danger`、`--warning`、`--success` 及对应 tone background |
| radius | `--radius-sm` 至 `--radius-xl`；状态 chip 使用 `--radius-pill` |
| spacing | `--space-1` 至 `--space-6` |
| shadow / stacking | `--shadow-sm` 至 `--shadow-lg`、`--z-dropdown` / `--z-modal` / `--z-toast` |

不得复制 shadcn 默认 Tailwind 色值、圆角、阴影、字体和间距。适配层只复用组件结构、状态层次和交互节奏，最终视觉必须完全响应 PTO 深色、浅色和 IDE host token。

#### 11.4.5 状态与可访问性契约

所有交互组件至少覆盖 default、hover、focus-visible、active、selected / expanded、disabled、loading、empty 和 error。浮层使用 `aria-expanded`、`aria-controls`、`aria-haspopup` 与正确的 dialog / menu / listbox 角色；Tab、Command、Menu、Select 和 Accordion 支持对应键盘操作。图标不能代替文字状态，颜色不能成为唯一信息载体。

页面只为坐标、视口、选区、异常 overlay 和密度可视化编写必要的业务样式。任何未列入矩阵的新通用组件，必须先检查 PTO Design System；确认缺失后才能按同一规则适配 shadcn/ui，并在组件文件头记录参考 URL、使用范围和 token 映射。

### 11.5 默认界面消融结果

本轮以“移除一个区域后，核心诊断链是否仍完整”为消融判据。结果是 Workspace / Explorer 左栏、Activity Rail、空证据入口和空终端都不影响“泳道 → 调查 → 证据 → 控制入口 → 实验”链路，因此从默认界面移除。

默认界面保留全局搜索与数据源；泳道表头只有一行：Rank、泳道范围、异常调查与必要的框选辅助，以及显示设置和缩放。颜色与单 lane 多选收进“显示”。删除单独的默认 Trace 标签行、调查点第三行和“查找异常”模态弹窗。调查点在底部可调高度抽屉内切换；Task 在右侧独立查看。v0.6 删除抽屉内独立泳道，将 Pass IR / 大图 / 泳道 Diff 改为主工作区上下文视图；v0.7 又将 Step 04 的 Pass IR 收回当前步骤正文并直接展示，仅大型任务图与泳道 Diff 继续使用主工作区，不增加默认空页签。

### 11.6 实施边界

v0.6 目标范围为：异常入口可发现性、唯一主 Trace 与 Task 定位、Pass IR 数据适配、调优方案命名、直接工程导入，以及有真实候选数据时的主泳道 Diff。真实设备结果需受控执行服务及完整工程环境；缺少候选 Run 时不能冒充执行成功。v0.12 另提供经用户授权、明确标记、基于真实 Pass 结构且隔离的模拟演示，见 5.4。9.3 为改造前的能力核实，当前交付见 11.7。

### 11.7 v0.6 实施记录（2026-09-10）

| 范围 | 当前实现 | 尚未完成 |
| --- | --- | --- |
| 异常入口 | 可聚焦区间按钮、悬停／聚焦调查动作、框选就地确认；抽屉左侧常驻同一 registry | 更大规模候选的重叠聚合 |
| 主 Trace 定位 | Rank／过滤／时间／纵向滚动／高亮／独立 Inspector；逻辑 Task 与实例选择；返回浏览位置 | 完整 Execute／Block 稳定映射仍缺失 |
| Task 身份 | 同一 Run／Rank／进程内按 taskId、shortId、FuncId 与执行单元分组；同一 r2t53 含 24 AIC + 48 AIV，不能把所选 AIC 函数的 Worker 数写成 72 | 此键仅用于本次运行的执行函数范围；跨构建逻辑身份需要独立映射 |
| 抽屉消融 | 删除独立时间线及其状态，保留四类原因诊断、七问与证据 | 不新增无证据的等待根因 |
| Pass IR | 保留 52 快照的 104 个局部静态 AST 索引和图渲染组件；v0.10 移除无关联的默认对照，01／02 显示真实等待任务、直接前驱及上游取证路径 | 尚无运行异常→编译变化的可追溯关联；不展示诊断 Pass 图。循环携带依赖、内存副作用、稳定结构 Diff、首次变化归属与决策原因未实现 |
| 模拟补全 | v0.13 通过显式 Demo URL 为 01 / Rank 0 提供模拟原因→真实结构约束的 Pass Diff→单变量方案→全模拟 Trace 对比；常规调查不显示模式切换 | 只验证产品交互；模拟不证明真实致因 Pass，不完成工程执行或正确性验证 |
| 工程导入 | 默认只读加载 49 个 Python 文件并验证摘要；识别编译／golden／DFX CLI；另支持源码目录选择；核验候选符号和当前值 | 依赖环境、历史基线构建 revision、输入和设备未验证；导入状态暂存内存，刷新需重读 |
| 实验 | 自动填充已识别绑定，展示单变量修改与预检查阻塞；保留离线 schema v2 | 无受控 runner，未应用修改、重编译、正确性实跑、重复测量或自动收集产物 |
| 对比 | 导入带 Case／实验关联字段的独立候选 Trace 后，主工作区双泳道联动；Run 起点／事件锚点对齐，未匹配泳道明确标注；指标单独命名 | 当前归档没有真实优化候选；尚无稳定逻辑计算范围的一对多映射、确定性任务增删／Copy Diff；测试候选仅存在隔离浏览器，不进入产品默认数据 |

实现和数据刷新说明见 [原型 README](../Design/deepseek-gap-investigator/README.md)。本轮不修改原始 Trace、模型源码或 v0.1 备份。12.5 的完整端到端设备验收仍未通过，不以只读导入或前端对比测试替代。

---

## 12. 功能验收标准

### 12.1 七问可见性验收

打开任意 Investigation Case，用户无需猜测即可找到以下问题的入口与状态：

| 编号 | 问题 | 验收要求 |
|---|---|---|
| Q1 | 是否存在 Ready Task 未及时下发 | 第三步显示答案或 UNKNOWN，并列出所需生命周期字段 |
| Q2 | 依赖还是资源不可用 | 第三步显示已排除、仍可能和无法检查的原因 |
| Q3 | 上游慢在哪一类时间 | 第三步显示关键路径分类或粒度不足说明 |
| Q4 | 为什么切分或融合 | 第四步显示决策原因、候选解释或 UNKNOWN |
| Q5 | 哪个 Pass 做出决定 | 第四步区分首次变化 Pass 与后续物化 Pass |
| Q6 | 应修改什么 | 第五步给出带依据、因果路径和风险的控制入口 |
| Q7 | 修改后改善什么 | 第六步并列预期值、实际值和实验结论 |

### 12.2 证据诚实性验收

- 所有结论均显示来源和等级；
- UNKNOWN 不得被隐藏在折叠区；
- HYPOTHESIS 不得使用确定性语言；
- 缺少 `ready_at` 时不得断言任务未 Ready；
- 缺少资源快照时不得断言资源可用；
- 缺少 Pass reason 时不得断言某个 Pass 的决策动机；
- 所有建议均能跳回至少一个证据和一个可修改入口。

### 12.3 闭环验收

- 初始打开的是完整 Trace，调查抽屉默认关闭；
- 能从两个预设异常标记或手动框选创建调查；
- 能从异常时间窗进入相关 Task；
- 能从 Task 进入 Runtime 原因分析；
- 能从 Runtime 任务回到图和 Pass；
- 能从 Pass 或源码进入候选控制入口；
- 能从控制入口创建验证实验；
- 能将实验结果回写为假设成立、局部成立、不成立或回退。

### 12.4 泳道基座验收

- Rank 0 默认完整 Trace 能在浏览器中加载，Rank 1 可切换或作为对照；
- lane 使用 `run_id + rank_id + pid + tid` 唯一标识，Worker 与 Scheduler 不合并；
- Worker 利用率按时间区间并集计算且不超过 100%；
- 搜索、缩放、Fit、Lane 过滤、Task 选择和范围框选可用；
- 浏览视口变化不修改已创建 Case 的调查范围；
- Task Inspector 与底部 Case 工作流独立，允许同时存在；抽屉内不存在第二套泳道；
- Step 04 有调查关联才在正文展示 Pass 图；01／02 当前不得出现 49→50 对照、全量 Pass 选择器或无关变化免责声明。旧版保存的 Pass 编号不得恢复这些内容。返回上游排查和主 Trace 任务定位需实测通过；大型任务图与泳道 Diff 在主 Preview 打开，源码片段和日志可在抽屉内下钻；
- 默认画面不存在 Journey、隐藏占位卡、重复详情面板和 Before / After 样例开关；
- PTO 已有的按钮、页签、面板、Inspector、Task bar 和 Task Tooltip 均直接调用设计系统；缺失组件只按 11.4.3 复用 shadcn/ui 样式结构，并按 11.4.4 映射 PTO tokens；
- 9 MB 级默认 Trace 的首次可交互、缩放和平移没有明显主线程长时间冻结。

v0.6 联动专项验收：区间标记可通过指针及键盘发现并打开；框选就地创建 Case；Step 02 的逻辑 Task 与 Worker 实例分别定位正确对象，跨 Rank 定位和返回均不改变 Case；定位目标不被抽屉覆盖；Step 04 图中节点／边有真实 IR 来源，相邻与跨阶段 Diff 标记准确，模型源码、IR 文本和 Pass 实现不混淆。

v0.7 诊断模式修订：异常标记不得占用额外泳道或时间轴行。诊断模式按钮以黑底白字表示开启；开启后主 Trace 的全部 Task 去色，异常区间以贯穿当前可见泳道的淡黄色覆盖层显示，标签位于区间中央。黄色空白区域可直接打开对应调查，覆盖区内真实 Task 仍优先响应 Task Inspector。关闭诊断模式后隐藏区间并恢复原任务配色。底部调查抽屉整体使用 `#F8F8F8`，且保持可调高度和非模态交互。

v0.8 调查导航消融：调查抽屉左侧调查列表常驻，标题栏不再设置“调查点”展开按钮。删除主泳道底栏中的“返回定位前视口”和“继续调查”两个脱离上下文的操作；重新进入调查统一点击黄色异常区间，返回 Case 范围统一使用调查正文内的“回到调查范围”。底栏只保留当前时间范围与泳道浏览提示。

v0.9 异常范围二维化：诊断覆盖层必须同时表达时间区间和受影响 Core，不得默认贯穿全部可见泳道。具名调查绑定到真实 wait Task 所在的具体 AIV 泳道，以实线黄色矩形显示；规则候选来自完整 Worker AIC 集合的共同无 X 事件窗口，只覆盖参与规则计算的 24 条 AIC 泳道，以虚线黄色矩形显示。前者表示已建立 Task／源码／Pass 关联的调查，不代表根因已经确认；后者只表示满足几何空闲规则，仍缺 Task 归属、生命周期、资源状态和 blocked reason。手动选区的分类视觉暂不扩展。

### 12.5 工程实验验收

- Step 05 的候选入口能定位到真实源码文件、符号和当前 revision；
- Step 06 在运行前明确展示修改 Diff、构建入口、正确性入口、性能入口与产物位置；
- 直接导入能识别真实工程入口和已有产物，显示依赖／版本／设备缺口，不要求重复手工填写已识别字段；导入不执行代码；
- 无执行服务时，导出协议显式标注未绑定与未验证字段，并提供稳定 `experiment_id`；有执行服务时无需人工搬运 ID 和结果 JSON；
- 候选产物能通过 `experiment_id + case_id` 回到原调查；
- 跨 Run 对应关系显示匹配等级，无法可靠匹配时不生成确定性 Diff；
- 性能结论至少同时检查正确性、重复测量、目标直接指标和端到端指标。

- 完整端到端验收必须实际完成：工程预检查 → 用户确认 Diff → 独立候选编译 → 正确性与重复采集 → 自动关联产物 → 两份真实 Trace 的联动对比。仅保存定义、导入指标或跨 Rank 比较均不算通过。
- 执行失败、取消、缺依赖、候选 Trace 缺失、映射不可靠均有明确状态；原始工程和基线产物不被覆盖。

---

## 13. MVP 分期

### M0：Trace-first 基座与调查入口

- IDE Frame 页面骨架；
- Rank / pid / tid 正确分组的完整 Trace；
- 默认加载 DeepSeek V4 Rank 0，Rank 1 可切换；
- 搜索、缩放、过滤、Task 详情和框选；
- 两个真实 Case 作为“已标注调查点”叠加；
- 六步调查抽屉与七问按需入口；
- 事实、候选、未知分级；
- 不增加未经采集的数据。

### M1：直接原因诊断

- Task 生命周期数据接入；
- Ready / Enqueue / Dispatch / Start 分段；
- 未满足依赖与关键前驱；
- Rank 对齐；
- 资源快照与 blocked reason；
- 补采集配置生成。

### M2：编译决策追溯

- Runtime Task 到 Execute／Block／Tile 的稳定映射；
- Pass 前后局部图 Diff；
- 首次结构变化定位；
- Pass reason 记录；
- Copy、Sync、Partition、Fusion 形成过程。

### M3：调优与验证闭环

- 控制入口目录；
- 调优建议卡；
- 与 PyPTO 工程、源码 revision 和执行入口绑定的单变量实验定义；
- 直接工程导入与依赖／执行环境检查；接入 runner 完成重新编译、验证和采集，手工导出仅作离线降级；
- 通过 `experiment_id + case_id` 自动关联候选 Run；
- 图、任务、搬运、等待、资源和端到端 Diff；
- 历史 Case 与已验证规律沉淀。

---

## 14. 待确认问题

1. Runtime 当前是否能输出 Task create、ready、enqueue、dispatch 和 blocked reason？
2. Ready Queue、Task Ring、Dependency Pool 与核资源是否存在可低开销采集接口？
3. `rootHash`、`callOpMagic`、`leafHash` 是否能跨 Pass 和不同构建稳定关联？
4. Pass Pipeline 是否已有结构化变换事件，还是只能通过相邻 Dump 做 Diff？
5. 编译器能否输出 Partition／Fusion／Copy／Sync 决策的候选、阈值和拒绝原因？
6. TileShape、Loop、Scope、Pass 和 Runtime 参数是否存在统一元数据或需单独建设控制入口目录？
7. 已找到的 `decode_csa.py` 入口所需 `golden` 辅助模块、PyPTO 版本、输入与编译环境能否完整绑定，是否存在机器可读 manifest？
8. DeepSeek V4 的两个 Case 是否具有可重复运行环境，可用于 M1/M2 的真实数据补采集？
9. `/Users/yin/pto/swimlane/` 的渲染内核是否会沉淀为共享 pattern / module API，还是由本产品维护经过裁剪的副本？
10. Rank 间绝对时间是否有可靠同步依据；没有时应采用哪一种相对锚点对齐？
11. 工程侧如何将 `experiment_id`、源码 revision 和候选产物写入 Run manifest？
12. 自动异常检测的第一版范围是仅标记空泡，还是同时包含碎片、长尾和核间不均衡？

---

## 15. 产品判断

本产品能否成立，不取决于绘制多少张图，而取决于系统能否诚实且连续地完成以下转换：

```text
现象
→ 运行状态
→ 直接原因
→ 形成该结构的编译决策
→ 可修改控制入口
→ 可验证实验
```

前端可以组织证据、缩小范围和生成候选解释，但不能替代缺失的 Runtime 等待原因和编译器决策原因。若底层暂时无法提供这些数据，产品仍应把“当前无法回答什么、为什么无法回答、需要补什么数据”作为正式结果，而不是用图结构猜测填补空白。
