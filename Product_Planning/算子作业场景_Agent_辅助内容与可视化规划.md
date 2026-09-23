# 算子作业场景下的 Agent 辅助内容与可视化规划

> 文档版本：v1.0  
> 更新时间：2026-08-13  
> 适用范围：PyPTO / PTO 算子开发、编译、调试、正确性验证、性能调优与模型集成  
> 产品形态：嵌入源码编辑器、编译产物和运行结果中的上下文型 Agent

---

## 1. 文档目标

算子开发者面对的不是一个孤立的“写代码”问题，而是一条跨越源码、编译器和昇腾硬件的连续作业链：

`理解算子 → 编写 DSL → 编译与变换 → 数值验证 → 故障定位 → 性能调优 → 模型集成`

每个阶段都会产生不同的问题：

- 编码时，用户需要知道代码表达了什么，而不只是语法是否正确；
- 编译时，用户需要知道 Pass 改变了什么，以及改变是否合法；
- 调试时，用户需要找到首个异常点，而不是阅读更多无关联日志；
- 调优时，用户需要理解瓶颈和优化因果，而不是盲目尝试参数；
- 集成时，用户需要确认单算子收益是否转化为模型或服务收益。

因此，Agent 的核心价值应定义为：

> **理解当前作业上下文，把源码、编译产物、运行证据和硬件知识组织成可视、可解释、可操作的辅助内容，帮助用户更快做出正确判断。**

本文不把 Agent 设计为固定流程向导，而是从典型任务场景出发，规划用户在作业现场需要看到的内容、可视化形式、Agent 结论和可执行动作。

---

## 2. 从 PyPTO 文档提炼出的知识底座

### 2.1 编程模型

PyPTO 源码中需要被 Agent 理解和呈现的核心语义包括：

- `Tensor`：位于 DDR / GM 的片外全局数据；
- `Tile`：位于 UB、L1、L0A、L0B、L0C 等片上存储中的计算分块；
- `Scalar`：控制、索引和标量计算数据；
- shape、dtype、layout、stride 和 memory space；
- `In`、`Out`、`InOut` 参数方向；
- `load`、`move`、`matmul`、Vector 运算、归约、cast、assemble 和 store；
- `pl.range`、`pl.parallel`、`pl.pipeline`、`yield_` 等控制和调度结构；
- Orchestration、Group、InCore、AIC、AIV、Inline 等层级和作用域；
- 自动依赖、显式 `deps`、TaskId、Fence 和跨核数据交接。

### 2.2 编译模型

PyPTO 编译过程不是黑盒，而是有序 Pass 对 IR 的连续变换。Agent 可利用的关键对象包括：

- Inline、SSA 转换、控制流变换、Scope Outline；
- Tensor 到 Tile 的转换、Tile 二维化、Matmul 自动分块；
- layout 解析、混合 Kernel 拆分、内存空间推断；
- MemRef 初始化、内存复用、地址分配；
- 自动依赖推导、运行时 Scope 物化和 Codegen；
- Pass 的 required、produced、invalidated 属性；
- Type、SSA、作用域、依赖、地址、资源等 Verifier 诊断。

### 2.3 正确性与调试能力

现有文档已经提供了可被产品化的证据来源：

- 原始 Program 的 Torch Codegen 数值验证；
- 默认 Pass Pipeline 后的 Torch Codegen 验证；
- 对每一个 Pass Dump 逐一执行数值校验；
- Golden、Tensor Dump、误差指标和首次分歧 Pass；
- Runtime 的依赖图、L2 Swimlane、PMU、参数 Dump 和 Scope Stats；
- 对已有 Build Output 的重放与再次取证。

### 2.4 昇腾硬件模型

硬件可视化需要建立在以下对象上：

- Cluster：1 个 Cube Core 与 2 个伙伴 Vector Core；
- Cube：矩阵乘与累加；Vector：逐元素、归约、归一化等计算；
- GM / DDR、L1、UB、L0A、L0B、L0C 等存储层级；
- DMA / TMOV 等数据搬运路径；
- Cube 与 Vector 之间的 SET / WAIT Flag；
- TPUSH / TPOP 多槽环形缓冲；
- A2/A3 使用 GM 作为跨核缓冲，A5 可使用消费者片上 SRAM；
- Buffer 地址、容量、复用和生命周期约束。

这些知识不应原样堆到界面中，而应按用户当前任务进行选择性呈现。

---

## 3. Agent 的统一工作方式

所有场景遵循同一条内容生成逻辑：

`用户任务 → 当前上下文 → 需要做出的判断 → 可视化内容 → Agent 结论 → 可执行动作 → 验证证据`

### 3.1 Agent 需要读取的上下文

| 上下文 | 典型内容 |
|---|---|
| 当前编辑上下文 | 文件、函数、选区、光标位置、未保存修改、诊断 |
| 工程上下文 | 调用点、相邻算子、配置、测试、模型位置 |
| 编译上下文 | Backend、Pass Pipeline、IR Dump、Verifier、Codegen 产物 |
| 运行上下文 | 输入、Golden、Tensor、Task、依赖、Trace、PMU、日志 |
| 硬件上下文 | A2/A3/A5、核心类型、存储容量、指令和布局能力 |
| 基线上下文 | 正确性基线、性能基线、历史实验和已知问题 |

### 3.2 内容可信度

Agent 展示的每条结论都应标记来源，避免把静态推断包装成实测结果。

| 标记 | 含义 | 可用于什么判断 |
|---|---|---|
| 源码事实 | 直接来自当前源码 | 契约、循环、cast、Chunk、调用关系 |
| 跨文件解析 | 来自配置、调用点、测试和相邻实现 | 模型上下游、变体关系、测试覆盖 |
| 编译事实 | 来自 IR、Pass、Verifier、PTOAS | 实际变换、依赖、layout、地址 |
| 静态估算 | 由 shape、dtype 和硬件规则推导 | 逻辑字节数、工作集、资源预警 |
| 运行实测 | 来自设备、Trace、PMU、Tensor 和 benchmark | 数值、时序、吞吐、真实瓶颈 |

### 3.3 通用交互原则

- 源码行、计算图节点、IR 节点、任务和 Tensor 应支持双向联动；
- 默认先展示结论、影响和下一步，再允许下钻原始证据；
- 可视化必须回答问题，不能只把原始数据换一种画法；
- Agent 建议需要说明依据、预期收益、风险和验证方式；
- 修改代码、执行高开销采集、应用调优配置时由用户确认；
- 同一 Run 使用统一标识关联源码、编译、运行和性能证据。

---

## 4. 场景总览

| 作业场景 | 用户主要问题 | Agent 首要产出 | 核心可视化 |
|---|---|---|---|
| 算子 Coding | 代码应该怎样写，当前代码表达了什么？ | 契约、计算结构、数据路径和编码风险 | 计算图、精度流、Tile 图、硬件路径 |
| 编译与 Lowering | 为什么编不过，Pass 改变了什么？ | 失败归因、结构 Diff 和合法性证据 | Pass 时间线、IR Diff、Verifier 面板 |
| 算子调试 | 错值、Hang 或异常最早从哪里出现？ | 首个异常点和跨层证据链 | Tensor Diff、任务泳道、依赖图 |
| 正确性与精度 | 结果是否可信，误差从哪里引入？ | Oracle 对比和误差归因 | 误差热图、精度路径、逐 Pass 二分 |
| 性能调优 | 为什么慢，改什么最有价值？ | 瓶颈排序和候选优化实验 | 耗时瀑布、硬件利用、Baseline Diff |
| 模型与服务集成 | 单算子放回模型后是否仍然有效？ | 覆盖、Fallback、端到端影响 | 模型图、请求下钻、SLO 分解 |

---

## 5. 算子 Coding 场景

### 5.1 用户正在完成的任务

- 阅读一个陌生算子；
- 新增一个算子或补齐已有实现；
- 修改 shape、dtype、layout、Chunk、Tile、Scope 或依赖；
- 把数学表达映射为 PyPTO DSL；
- 判断实现是否适配目标昇腾硬件；
- 为当前实现准备编译与数值测试。

### 5.2 Coding Agent 应回答的问题

1. 这个算子在模型中做什么？
2. 输入、输出和中间值的 shape、dtype、layout 是什么？
3. 代码分成哪些计算阶段，阶段之间怎样传递数据？
4. 哪些计算运行在 Cube，哪些运行在 Vector？
5. 数据如何从 GM 进入片上并写回？
6. 并行域、Chunk、Tile 和 Pipeline 是怎样组织的？
7. 当前写法有哪些越界、对齐、精度、容量或依赖风险？
8. 修改当前代码会影响哪些调用点、变体和测试？

### 5.3 Coding 时最有价值的可视化

#### A. 算子计算图

展示内容：

- 输入、权重、中间 Tensor 和输出；
- RMSNorm、Matmul、RoPE、Softmax、逐元素、归约等计算节点；
- 残差、状态数据和 KV Cache 等旁路；
- 节点的 shape、dtype 和作用域；
- 未实现、占位或无生产者的数据链断点。

用户价值：快速理解计算意图，发现数据链是否闭合。点击节点定位源码，点击源码高亮计算图路径。

#### B. Tensor 契约与 Shape/Layout 视图

展示内容：

- `In / Out / InOut` 方向；
- Tensor 的 runtime shape、dtype、layout 和 stride；
- 每个 op 前后的 shape 变化；
- broadcast、reshape、transpose、slice 和 assemble；
- 动态维度、有效形状和 Padding 区域。

用户价值：在编码阶段发现 shape 不匹配、坐标系混淆、错误广播和 Padding 消费问题。

#### C. 数据与精度流

展示内容：

`输入 dtype → load/cast → 计算 dtype → 累加 dtype → 中间输出 dtype → store dtype`

- BF16 / FP16 / FP32 的转换位置；
- Matmul、归约、Softmax 和残差累加的精度；
- 重复 cast、过早降精度和潜在溢出位置；
- 每个阶段的逻辑数据量。

用户价值：帮助用户理解误差风险，也为后续数值验证选择关键 Tensor。

#### D. Chunk、Tile 与循环映射

展示内容：

- 完整 Tensor 如何切分为 Chunk 和 Tile；
- 循环变量对应哪个维度；
- 每个 Tile 的真实区域、Padding 和尾块；
- Chunk 数、循环次数和 pipeline stage；
- Tile 在矩阵 M/N/K 维度上的映射。

用户价值：直接看到 `8192 = 16 × 512` 这类调度关系，及时发现不可整除、越界和对齐风险。

#### E. 昇腾硬件执行路径

展示内容：

- `GM / DDR → L1 / UB → L0A/L0B → Cube → L0C`；
- `GM / DDR → UB → Vector → UB → GM / DDR`；
- Cube 与 Vector 之间的 TPUSH/TPOP、Flag 和 Ring Buffer；
- 当前源码阶段对应的硬件节点和数据搬运；
- A2/A3 与 A5 的跨核缓冲差异。

用户价值：把 DSL 语义转换为直观的硬件心智模型。图中必须区分“源码静态映射”和“后端实测路径”。

#### F. Scope、并行与依赖图

展示内容：

- Orchestration、Group、InCore、AIC、AIV 的嵌套；
- `pl.parallel`、`pl.range`、`pl.pipeline` 的并行域；
- 自动依赖、显式 TaskId 和跨 Scope 数据依赖；
- 哪些 Tensor 是共享状态或原位更新。

用户价值：避免把并行循环误当成串行循环，或遗漏跨任务依赖和 Out/InOut 写回语义。

#### G. 能力与风险 Lens

展示内容：

- 当前 API 在目标硬件和后端的支持状态；
- dtype、layout、动态 shape 和指令限制；
- UB/L1/L0 容量静态预警；
- 尾块、索引宽度、对齐、广播、作用域和依赖风险；
- 相关样例、测试和已知问题。

用户价值：风险在 Coding 时出现，而不是推迟到编译失败或设备错值。

#### H. 修改影响 Diff

当用户编辑代码时，Agent 对比修改前后：

- 算子契约变化；
- 计算图节点和边变化；
- dtype / cast 变化；
- Tile 数、工作集和静态资源变化；
- 受影响的调用点、同类实现和测试；
- 新增或消除的风险。

### 5.4 Coding Agent 的右侧面板建议

| 页签 | 默认内容 | 主要交互 |
|---|---|---|
| 概览 | 数学语义、模型上下游、契约、计算图、关键结论 | 图与源码联动 |
| 数据与精度 | dtype 路径、Tensor 规模、cast、误差风险 | 选择 Tensor、查看来源 |
| 分块与硬件 | Chunk/Tile、循环、并行、硬件数据路径 | 缩放、平移、阶段联动 |
| 编排与依赖 | Scope、Inline、调用关系、Task 依赖 | 定位调用点和源码段 |
| 验证准备 | 已有测试、缺失证据、边界用例 | 生成测试草案 |

### 5.5 Agent 可执行的辅助动作

- 解释当前选区；
- 查找相似实现和对应文档；
- 生成符合仓库风格的代码片段；
- 生成 shape/dtype/边界测试；
- 添加关键 Tensor 的选择性 Dump 标记；
- 运行最小编译检查；
- 对修改前后生成契约和资源 Diff；
- 在应用修改前给出影响摘要。

---

## 6. 编译与 Lowering 场景

### 6.1 用户正在完成的任务

- 处理 parser、type checker、Pass、Verifier 或 Codegen 错误；
- 检查某个 op 是否在 Pass 中丢失或被错误改写；
- 判断自动 layout、memory、dependency 处理是否符合预期；
- 分析编译耗时或某个 Pass 的异常增长。

### 6.2 Agent 应提供的内容

#### Pass 时间线

- 按真实执行顺序展示 Pass；
- 显示每个 Pass 的输入/输出 IR 规模、耗时和诊断；
- 标记 required、produced、invalidated 属性；
- 聚合“无结构变化”的 Pass，突出关键变化点。

#### 结构化 IR Diff

- 新增、删除、替换和移动的 op；
- shape、dtype、layout、memory space 和 attrs 的变化；
- Scope Outline、混合 Kernel 拆分和调用方向变化；
- 依赖边、MemRef、地址和复用关系变化；
- Diff 节点回链源码 span。

#### Verifier 与约束面板

- 错误对应的源码、IR 节点和规则；
- 违反的 Type、SSA、作用域、依赖、地址或容量不变量；
- 错误影响和最小修复方式；
- 修复后需要重新验证的属性。

#### 编译耗时火焰图或瀑布图

- parse、passes、codegen、kernel_codegen、orchestration_codegen；
- 单 Pass 耗时和相对历史基线的变化；
- IR 规模增长与耗时增长的关联。

### 6.3 Agent 输出示例

> 首个失败发生在 `AllocateMemoryAddr` 之后。当前 Tile 与预留 Ring Buffer 在 UB 地址区间重叠，违反 `AllocatedMemoryAddr`。建议缩小 Tile 工作集或调整 Buffer 预留区；该修改会影响 A5 路径，不影响 A2/A3 的 GM Buffer 路径。

---

## 7. 算子调试场景

### 7.1 症状入口

- 输出全零、局部错值、NaN、Inf；
- Hang、超时、设备错误或长期等待；
- 不同 Backend、环境或版本结果不一致；
- 模型输出异常，但无法定位具体 Kernel。

用户可以从症状进入，Agent 不要求用户先判断问题属于前端、Pass、Codegen 还是 Runtime。

### 7.2 调试 Agent 的核心产出

#### 首个异常点

优先回答以下之一：

- 首个数值分歧 Tensor；
- 首个产生 NaN / Inf 的 op；
- 首个丢失或语义变化的 Pass；
- 首个未满足的依赖；
- 最早进入长期等待的 Task；
- 首个非法地址、复用或生命周期冲突。

#### 跨层证据链

建立可下钻关联：

`源码行 → DSL op → Pass/IR → Kernel → Task/Event/Fence → Tensor → 症状`

#### 任务依赖图

- Task 生产者和消费者；
- 自动依赖与显式依赖；
- 未连接、环路、过度串行和等待边；
- Kernel 真实名称和输入输出 Tensor。

#### Runtime Swimlane

- 每个 Ring / Core / Task 的开始、结束和等待时间；
- Cube、Vector、搬运和同步的重叠；
- Slot wait、Fence wait、backpressure 和空闲区间；
- 与源码 Scope、调用点和依赖边联动。

#### Tensor Diff 工作台

- Golden 与 Device Tensor 的逐元素对比；
- 最大绝对/相对误差、NaN/Inf、异常比例；
- 行列热图、直方图和 Top-K 异常位置；
- Tensor 的生产者、消费者、dtype 和生命周期；
- 只采集关键 Tensor，提示全量 Dump 的性能和容量风险。

#### 内存与生命周期视图

- Tensor / Tile 的分配、最后使用和释放；
- 地址区间、重用关系和重叠冲突；
- Ring Buffer Slot 的占用和释放；
- 读前未写、写后过早复用和越界访问。

### 7.3 调试 Agent 的辅助动作

- 根据症状选择低干扰取证组合；
- 自动插入或建议 `dump_tag` / `dumps`；
- 启用依赖抓取、Swimlane、PMU 或 Scope Stats；
- 重放已有 Build Output；
- 生成最小复现和脱敏证据包；
- 给出下一步验证动作，并估算采集开销。

---

## 8. 正确性与精度场景

### 8.1 用户正在完成的任务

- 为新算子建立 Golden；
- 判断 BF16/FP16/FP32 混合精度是否可接受；
- 验证融合、分块或调优前后数值一致性；
- 定位从哪个 Pass 或中间 Tensor 开始产生误差。

### 8.2 推荐可视化

| 可视化 | 展示内容 | 回答的问题 |
|---|---|---|
| Oracle 矩阵 | Torch、原始 Program、Pass 后 IR、Device 的通过状态 | 哪一层开始不一致？ |
| 精度路径图 | 每个 op 的输入、计算、累加、输出 dtype | 误差可能在哪里引入？ |
| Tensor 误差热图 | 空间分布、异常行列、Padding 区域 | 错误是局部还是系统性？ |
| 误差直方图 | 绝对误差、相对误差、ULP 分布 | 容差是否合理？ |
| 逐 Pass 二分图 | 每个 Pass Dump 的数值状态 | 首个错误 Pass 是谁？ |
| 边界用例矩阵 | 零值、极值、尾块、最大位置、动态 shape | 哪些边界仍未覆盖？ |

### 8.3 Agent 判断原则

- 编译通过只证明结构可生成，不代表数值正确；
- 相邻算子或通用测试属于间接证据，不能替代当前实现的 Golden；
- 不通过放宽断言掩盖问题；
- 容差建议必须结合算子数学特征、dtype 路径和误差分布；
- 性能候选只有通过正确性门禁后才能进入排名。

---

## 9. 性能调优场景

### 9.1 用户正在完成的任务

- 判断算子是计算、搬运、同步、调度还是资源瓶颈；
- 调整 Chunk、Tile、Pipeline Stage、并行度和内存策略；
- 比较融合前后或不同平台上的收益；
- 定位性能回退并建立可复现基线。

### 9.2 性能 Agent 应先给出的结论

1. 当前最大瓶颈是什么；
2. 结论来自哪些指标和时间区间；
3. 哪段源码、哪个 Scope 或哪类 Task 对瓶颈贡献最大；
4. 优先尝试哪些修改；
5. 每个修改的预期收益、约束和验证方式。

### 9.3 推荐可视化

#### A. 耗时瀑布与关键路径

- 总延迟拆解为计算、数据搬运、同步、调度和空闲；
- 标记关键路径和非关键并行任务；
- 从总耗时下钻到 Scope、Kernel 和源码行。

#### B. Cube / Vector / DMA 泳道

- Cube 与 Vector 是否有效重叠；
- Matmul、Vector 后处理和搬运之间的流水空洞；
- TPUSH/TPOP、Flag 和 Slot Wait；
- A2/A3 GM Round Trip 与 A5 片上路径差异。

#### C. 数据搬运与带宽视图

- GM 读写量、片上搬运、重复加载和写回；
- Tensor 的逻辑字节数与实测流量差异；
- layout conversion、冗余 TMOV 和中间 Buffer；
- 带宽利用率及主要流量贡献者。

#### D. 计算利用与指令结构

- Cube / Vector 活跃时间；
- 计算量、有效工作比例和 Padding 浪费；
- 指令类别、Pipeline Stall 和 PMU 事件；
- 计算与带宽瓶颈的证据判断。

#### E. Tile 与参数敏感性视图

- Tile M/N/K、Chunk、Stage、并行度和 Slot 数；
- 参数变化对延迟、带宽、片上占用和正确性的影响；
- 单参数曲线、二维热图或 Pareto 前沿；
- 合法区、资源越界区和数值失败区。

#### F. Baseline 因果 Diff

对比当前候选与可信基线：

- 哪些源码和调度参数发生变化；
- IR、Kernel 数、Task 数和依赖变化；
- 搬运量、等待、利用率和延迟变化；
- 正确性、资源和性能门禁；
- 收益来自哪里，代价是什么。

### 9.4 性能 Agent 的辅助动作

- 推荐最小必要的 Profiling 配置；
- 根据证据生成有限的参数搜索空间；
- 批量运行候选并执行正确性门禁；
- 按目标指标排序，并保留 Pareto 候选；
- 生成调优前后因果报告；
- 将优选结果保存为带适用边界的 Tuning Recipe。

---

## 10. 模型与服务集成场景

虽然重点是算子作业，Agent 仍需帮助用户确认局部优化是否在系统层成立。

### 10.1 推荐内容

- 模型计算图中的算子实现、Fallback 和覆盖状态；
- 当前 Kernel 对应的模型层、Token 和 Decode Step；
- 单算子延迟在 TTFT、TPOT 和吞吐中的贡献；
- KV Cache 生命周期、容量和命中情况；
- 不同 batch、序列长度和并发下的性能边界；
- 算子版本、编译产物、模型和服务配置关系。

### 10.2 推荐可视化

- 模型图覆盖图：PyPTO 实现、库实现、Fallback、缺失；
- 请求延迟瀑布：排队、Prefill、Decode、采样和返回；
- Token / Layer / Kernel 三层时间线；
- SLO 下钻：从 TTFT/TPOT 定位到具体 Scope、Task 和硬件瓶颈；
- 容量曲线：序列长度、并发、KV Cache 与可用内存的关系。

---

## 11. Agent 产品形态建议

### 11.1 不按 Agent 名称割裂界面

用户不必先选择“Coding Agent”还是“Debug Agent”。产品根据当前对象和证据自动切换内容：

- 正在编辑源码：以 Coding 内容为主；
- 编译失败：浮现 Pass、IR 和 Verifier 内容；
- 打开运行结果：浮现 Tensor、Task 和 Trace；
- 打开性能实验：浮现瓶颈、Baseline 和候选对比。

Agent 名称可以作为能力归属，但不应成为用户完成任务的前置选择。

### 11.2 建议的工作台布局

| 区域 | 内容 |
|---|---|
| 左侧资源管理器 | 源码、配置、测试、Pass Dump、运行和实验资产 |
| 中间主工作区 | 源码、IR Diff、Tensor Diff、Timeline 或性能对比 |
| 右侧 Agent 面板 | 针对当前对象的结构化结论、可视化和动作 |
| 底部证据区 | 编译日志、运行日志、诊断、原始指标和可复现命令 |

### 11.3 右侧面板的动态信息结构

右侧面板可使用统一的五层结构：

1. **对象摘要**：当前函数、Kernel、Pass、Tensor 或 Run；
2. **关键结论**：最值得用户注意的 1～3 条判断；
3. **任务可视化**：与当前场景最相关的图；
4. **风险与证据**：可信度、异常、缺失证据和影响；
5. **下一步动作**：解释、定位、生成测试、采集、对比或修改。

---

## 12. 优先级建议

### P0：Coding 与可信编译

- 算子契约与计算图；
- shape、dtype、layout 和精度流；
- Chunk、Tile、Scope 和依赖；
- 源码与可视化双向联动；
- Pass 时间线、结构化 IR Diff 和 Verifier；
- 测试覆盖与缺失证据。

### P1：正确性与调试

- 多 Oracle 状态矩阵；
- Tensor Diff、热图和首个分歧；
- 逐 Pass 数值验证；
- Task 依赖图和 Runtime Swimlane；
- 选择性 Dump 与证据包。

### P2：性能调优

- Scope / Kernel 耗时拆解；
- Cube、Vector、DMA 和等待分析；
- PMU、带宽和数据流量；
- Baseline 因果 Diff；
- 受约束的参数实验与 Tuning Recipe。

### P3：模型与服务闭环

- 模型图覆盖和 Fallback；
- Token / Layer / Kernel 关联；
- TTFT、TPOT、吞吐和容量下钻；
- 算子收益到服务收益的关联验证。

---

## 13. 衡量 Agent 是否真正帮助用户

不以“回答了多少问题”衡量，而以作业结果衡量：

| 阶段 | 建议指标 |
|---|---|
| Coding | 首次编译通过时间、编码期发现的约束问题比例、测试生成采用率 |
| 编译 | 错误定位时间、首个失败 Pass 命中率、无效重编译次数 |
| 调试 | 首个异常点定位时间、日志阅读量、最小复现生成时间 |
| 正确性 | 首个分歧 Tensor/Pass 命中率、边界覆盖率、误判率 |
| 调优 | 达标实验次数、性能收益可归因率、正确性回退拦截率 |
| 集成 | Fallback 发现时间、单算子收益转化率、SLO 问题下钻时间 |

---

## 14. 结论

算子 Agent 最有价值的能力不是“替用户写完代码”，而是把原本分散在源码、文档、IR、日志、Tensor、Trace 和硬件知识中的信息，组织成与当前任务直接相关的判断界面。

在 Coding 阶段，它应让用户看见计算图、数据与精度、Chunk/Tile、Scope/依赖和昇腾硬件路径；在调试阶段，它应给出首个异常点、Tensor Diff、依赖图和任务泳道；在性能阶段，它应把耗时、数据搬运、核心利用、参数变化和基线差异串成可验证的优化因果。

最终产品应形成一条连续的证据链：

`我写了什么 → 编译器怎样理解 → 硬件怎样执行 → 结果是否正确 → 为什么快或慢 → 修改后发生了什么`

当这条链能够被用户直观看见、逐层下钻并执行验证动作时，Agent 才真正成为算子开发工作台的一部分。

---

## 15. 资料来源

本文主要依据以下本地资料整理：

- [PyPTO 语言指南](../repo/pto/docs/zh-cn/user/01-language_guide.md)
- [PyPTO 操作参考](../repo/pto/docs/zh-cn/user/02-operation_reference.md)
- [Torch Codegen 调试指南](../repo/pto/docs/zh-cn/user/03-torch_codegen_debug.md)
- [Pass Manager](../repo/pto/docs/zh-cn/dev/passes/00-pass_manager.md)
- [Pass Diagnostics](../repo/pto/docs/zh-cn/dev/passes/92-diagnostics.md)
- [Verifier](../repo/pto/docs/zh-cn/dev/passes/99-verifier.md)
- [编译 Profiling](../repo/pto/docs/zh-cn/dev/01-compile-profiling.md)
- [运行时 DFX](../repo/pto/docs/zh-cn/dev/03-runtime-dfx.md)
- [PTO ISA 集群架构](../repo/pto/docs/zh-cn/reference/pto-isa/00-cluster_architecture.md)
- [PTO ISA 缓冲区管理](../repo/pto/docs/zh-cn/reference/pto-isa/02-buffer_management.md)
- [算子开发 Agent 能力规划与 Coding Agent 原型](./算子开发_Agent_能力规划与_Coding_Agent_原型.md)
- [PyPTO 3.0 Toolkit 产品功能规划](./PyPTO3.0_Toolkit_产品功能规划.md)
