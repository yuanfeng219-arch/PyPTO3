# hw-native-sys 剩余 6 个仓库 Issues UX 洞察报告

数据来源：

- `github_issues/simpler`
- `github_issues/PTOAS`
- `github_issues/pto-isa`
- `github_issues/pypto-lib`
- `github_issues/pypto_top_level_documents`
- `github_issues/pypto-serving`

说明：本报告沿用之前对 `pypto` 仓库的分析方式，把 issue 作为开发者反馈样本，从开发者完整使用流程出发，梳理用户痛点、用户期望与设计机会点。每个仓库都已过滤 Pull Request，只保留真实 issue。

## 总览

| 仓库 | Total | Open | Closed | 主要特征 |
|---|---:|---:|---:|---|
| `simpler` | 212 | 16 | 196 | 运行时、调度、DFX、性能、AICPU/A5/A2A3 |
| `PTOAS` | 202 | 25 | 177 | 汇编/代码生成、PTODSL、同步、指令语义、性能 |
| `pto-isa` | 66 | 4 | 62 | ISA 语义、指令文档、硬件限制、仿真一致性 |
| `pypto-lib` | 93 | 20 | 73 | 模型算子库、Qwen/DeepSeek、性能对齐、端到端正确性 |
| `pypto_top_level_documents` | 0 | 0 | 0 | 当前无 issue，可作为顶层文档承载入口 |
| `pypto-serving` | 27 | 21 | 6 | LLM serving、KV cache、平台管理、并行策略、性能指标 |

## 跨仓库开发者旅程

从这些仓库的 issue 看，开发者的真实使用链路大致是：

1. 阅读顶层文档，搭建环境，确认版本和硬件。
2. 用 `pypto` / `pypto-lib` 编写或迁移模型 kernel。
3. 通过 `PTOAS` / `pto-isa` 进入低层指令、layout、同步和汇编语义。
4. 通过 `simpler` runtime 调度任务、运行在 AICPU/NPU 设备上。
5. 在 `pypto-lib` 中做真实模型正确性和性能验证。
6. 在 `pypto-serving` 中把模型能力包装成 serving 能力，处理 KV cache、并行、吞吐、TTFT/TPOT 等在线服务问题。

最大的体验风险不是单点 API 不好用，而是跨层调试断裂：当输出错误、性能回退或设备挂死时，开发者需要在 DSL、IR、PTOAS、ISA、runtime、模型库和 serving 层之间来回定位。

---

# 1. simpler

## 仓库角色

`simpler` 更像 PyPTO 体系里的运行时与调度底座。issue 主要集中在 AICPU 调度、任务 dispatch、DFX 诊断、backpressure、A5/A2A3 环境、性能和 runtime fatal code。

代表问题：

- #545 Runtime performance optimization tracking
- #995 DFX capability overview & roadmap
- #997 DFX global backpressure mode
- #1105 Scheduler requeue forever
- #1341 AICPU scheduler starvation
- #1350 Host log prints bare error codes
- #1351 Version guard for A5 SDMA overlay

## 开发者全流程痛点

### 环境与版本阶段

痛点：
- A5、A2A3、SDMA、CANN driver、PTO-ISA overlay 等环境前置条件复杂。
- 版本不匹配时，错误可能在运行结束、driver exit、task dispatch 或设备交互中暴露，而不是一开始就失败。
- host log 有时只输出裸错误码，开发者必须自己查含义。

期望：
- 运行前能明确知道当前 runtime、PTO-ISA、CANN、driver、硬件能力是否匹配。
- 错误码能直接解释含义、可能原因和下一步命令。

机会点：
- 增加 `simpler doctor` 或纳入统一 `pypto doctor`，检查 runtime/ISA/CANN/A5-SDMA overlay 能力。
- 运行时错误输出从“code only”升级为“code + name + likely cause + next action”。
- 对 A5/A2A3 环境差异提供能力矩阵和版本 guard。

### 运行与调度阶段

痛点：
- runtime scheduler 可能出现 requeue forever、slot starvation、full dump backpressure deadlock 等问题。
- 用户很难判断是 kernel 本身慢、调度不公平、任务依赖错误，还是 DFX 记录造成 backpressure。
- dispatch/finish timing 缺少轻量、稳定的观测面。

期望：
- 能看到任务从 submit 到 dispatch、run、finish 的完整 timeline。
- 当任务卡住时，能知道它在等什么：dep、slot、queue、dump buffer、device event 还是 driver。

机会点：
- 提供 `simpler trace report`：展示 task queue、dep wait、dispatch lane、finish timing、backpressure 状态。
- DFX 支持 global backpressure policy，并在 trace 中标记因为记录拥塞导致的阻塞。
- 增加 scheduler fairness / starvation 检测，让饥饿线程、占用 slot 的任务和 idle core 状态可见。

### 调试与复现阶段

痛点：
- 运行时问题常表现为 deadlock、507018、SIGSEGV、scheduler stall，用户需要跨 runtime、driver、kernel 取证。
- DFX 和 full tensor dump 既是诊断工具，也可能引入 backpressure 风险。

期望：
- 诊断工具本身要有可预测的开销和失败模式。
- 复现包应包含 runtime config、task graph、trace、fatal code、driver/CANN 信息。

机会点：
- 一键生成 runtime repro bundle。
- 把 fatal code、runtime state、task graph snapshot 和设备信息统一打包。
- 为 DFX 提供“低干扰模式”和“完整取证模式”两档。

---

# 2. PTOAS

## 仓库角色

`PTOAS` 是 PTODSL、汇编、代码生成和低层 kernel 语义的关键层。issue 主要集中在同步指令、layout 推断、PTODSL 风格、生成代码性能、指令支持和运行时正确性。

代表问题：

- #499 Mixed kernel Flash attention poor performance
- #643 `--enable-insert-sync` much slower than manual-sync kernels
- #734 PTODSL frontend style improvement suggestions
- #800 PTODSL user guide problems
- #804 Simplify Vector Instruction Semantics
- #859 Redesign PTODSL pto.simd/pto.cube/pto.simt subkernel
- #933 False-positive ast_rewrite runtime loop IV exposure

## 开发者全流程痛点

### 学习与编写阶段

痛点：
- PTODSL 用户指南存在问题，说明文档与真实可用语义可能有距离。
- `index` / `integer` 混用、native if/for rewrite、subkernel 风格等概念会影响开发者写法。
- PTODSL 同时面向高层表达和低层指令，抽象边界不稳定时会增加学习负担。

期望：
- 用户能清楚知道何时写 PTODSL、何时依赖 PyPTO 上层、何时需要落到 PTOAS/ISA。
- 文档中的示例应覆盖真实模型 kernel 的常见模式，而不只是玩具示例。

机会点：
- 重写 PTODSL user guide：按任务组织，而不是按 API 罗列。
- 增加“从 Python/DSL 到 PTOAS 到 ISA”的最小路径示例。
- 为 `pto.simd`、`pto.cube`、`pto.simt` 提供统一心智模型和对照表。

### 编译与代码生成阶段

痛点：
- 自动插 sync 可能比手写 sync 慢 10% 或更多，甚至造成性能回退。
- 生成代码变量名、SSA name hint、CHECK 对齐规范都会影响调试可读性。
- False-positive rewrite 或限制过严的校验会让用户不确定该改源码还是改编译器。

期望：
- 自动同步逻辑既正确又能解释：为什么插这个 sync，不插会发生什么，成本是多少。
- 生成代码要保留足够语义线索，便于从源代码定位到汇编和运行时行为。

机会点：
- 增加 sync explain report：列出每条自动 sync 的依赖原因、管线、作用域和预估开销。
- 支持保留 SSA name hint 到 C++/汇编输出，提升可调试性。
- 对 rewrite false-positive 提供 source span 和建议修复方式。

### 性能调优阶段

痛点：
- Mixed kernel、Flash Attention、dynamic GEMM、vector kernel 等场景对同步、layout、tile size 极其敏感。
- 用户很难判断性能差距来自 PTOAS 代码生成、ISA 限制、runtime 调度还是模型层调用方式。

期望：
- 有统一报告能解释生成 kernel 的瓶颈：sync、barrier、pipe utilization、layout conversion、冗余 TMOV。

机会点：
- `ptoas perf explain`：输出 sync/barrier、pipe、tile layout、冗余 move、post-update 优化机会。
- 将 lit regression 和真实模型 kernel 的性能案例沉淀成 benchmark cookbook。

---

# 3. pto-isa

## 仓库角色

`pto-isa` 是最底层 ISA 语义、指令文档和仿真一致性的来源。issue 数量较少，但风险很高，因为它定义了上层工具链的正确性边界。

代表问题：

- #90 Reclassify micro instructions and remove from tile docs
- #106 Add TColGather / TColScatter
- #170 Vector UB usable size appears to be ~184KB, silent corruption
- #197 soft SYNCALL busy-spins up to 1e6 iterations
- #173 TTRANS miscomputes / hangs
- #178 MGatherRowImpl missing wait flag

## 开发者全流程痛点

### 理解语义阶段

痛点：
- 指令到底是 tile instruction、micro instruction 还是内部辅助操作，需要清晰分类。
- 文档与实际硬件行为不一致时，上层开发者会踩到不可见边界。

期望：
- ISA 文档不仅说明指令格式，还要说明能力边界、硬件限制、已知风险和平台差异。

机会点：
- 建立 ISA instruction taxonomy：tile/micro/control/memory/sync 等分类清晰。
- 每条指令增加 platform support、constraints、undefined behavior、known errata。

### 正确性与硬件限制阶段

痛点：
- UB 可用大小、validRow 限制、DMA race、SYNCALL busy spin 等问题可能导致 silent corruption 或长延迟。
- 上层工具链不一定知道这些硬件事实，导致 PyPTO/PTOAS 生成危险代码。

期望：
- ISA 层的硬件限制能上浮为 PTOAS/PyPTO 的 verifier 和 diagnostics。

机会点：
- 输出 machine-readable ISA constraints，让 PTOAS/PyPTO 自动消费。
- 将 #170 这类 silent corruption 风险转成编译期容量检查。
- 对 sync/wait 类指令提供 latency 和 busy-spin 行为说明。

### 仿真与真实设备一致性阶段

痛点：
- cpu_sim、a5sim、真实 A2/A3/A5 之间行为不一致，会让测试 oracle 失去可信度。

期望：
- 明确哪些测试验证 ISA 语义，哪些验证仿真实现，哪些必须上板验证。

机会点：
- 建立 ISA conformance test matrix。
- 每个指令族提供 sim vs device 差异记录和回归用例。

---

# 4. pypto-lib

## 仓库角色

`pypto-lib` 是真实模型算子和端到端模型能力的承载层。issue 主要围绕 Qwen3、DeepSeek V4、mixed C/V kernel、decode/prefill、KV cache、性能 profiling 和模型正确性。

代表问题：

- #115 Mixed C/V kernel end-to-end support
- #294 Chip-level orchestration multi-Out only first Out written back
- #396 Qwen3-14B non-L3 decode generates repetitive text
- #481 no-op self-copy hack to force WAR edge
- #622 Quantifying PyPTO vs CCE paged-attention gap
- #700 DeepSeek V4 Sinkhorn layout underutilizes vector sublanes
- #781 Device-resident DeepSeek V4 cache pools

## 开发者全流程痛点

### 模型迁移阶段

痛点：
- 从模型逻辑迁移到 PyPTO kernel 时，开发者要同时处理 shape、layout、split、C/V 混合、orchestration 和 device memory。
- 模型正确性失败可能表现为 repetitive text、1-ULP drift、丢写、只写回第一个 Out 等，非常难定位。

期望：
- 模型开发者能从“模型语义”一路追踪到 kernel、task、memory、output。
- 对常见模型模式有可复用模板，例如 Qwen decode attention、DeepSeek V4 expert、KV cache path。

机会点：
- 建立 model kernel templates：prefill、decode、paged attention、RMSNorm+RoPE、MoE expert、lm_head。
- 为多 Out、Opaque sub-function、split drift、WAR edge 等风险增加模型层验证器。

### 正确性验证阶段

痛点：
- 端到端输出错误有时来自 PyPTO 编译器依赖、PTOAS 代码生成、runtime 调度或模型库 glue code。
- 如果 golden 也是同一条 on-device code path，可能掩盖错误。

期望：
- 正确性验证应有多 oracle：CPU/reference、CCE/library baseline、PyPTO device、serving output。

机会点：
- 建立 model correctness harness，支持多 oracle 对比。
- 对 token-level 输出、intermediate tensor、kernel-level output 做分层对齐。
- 为 repetitive text、prefix-cache 丢写、WAR edge 缺失等问题提供专门检测规则。

### 性能分析阶段

痛点：
- PyPTO 与 CCE paged-attention 的差距需要被量化和解释，而不是只看到 end-to-end 慢。
- sublane 利用率、GM round trip、KV cache、TPOT、kernel fusion 都跨越多个仓库。

期望：
- 能看到模型层性能指标和底层瓶颈的映射关系。

机会点：
- 输出 model profiling report：TTFT、TPOT、kernel latency、memory traffic、sublane utilization、CCE baseline gap。
- 将 profiling report issue 模板标准化，让性能讨论可对比、可追踪。

---

# 5. pypto_top_level_documents

## 仓库角色

该仓库当前没有真实 issue，但从组织结构看，它天然适合作为 PyPTO 体系的顶层入口文档仓库。

## 开发者全流程痛点

痛点：
- 当前多个仓库分别承载 DSL、runtime、PTOAS、ISA、lib、serving，开发者需要自己拼出全局地图。
- 当问题跨层时，用户不清楚应该先读哪个文档、在哪个仓库提 issue、用哪些诊断命令。

期望：
- 有一个清晰的 top-level developer portal，解释各仓库职责和完整开发链路。

机会点：
- 把该仓库设计成“PyPTO Developer Portal”。
- 内容建议：
  - 架构总览：PyPTO / PTOAS / PTO-ISA / simpler / pypto-lib / pypto-serving 的关系。
  - 新手路径：从安装到跑通第一个 kernel。
  - 故障定位路径：wrong output、hang、performance regression、serving failure 分别怎么查。
  - issue 路由指南：不同类型问题应该提到哪个仓库。

---

# 6. pypto-serving

## 仓库角色

`pypto-serving` 是从 kernel/model 能力走向在线推理服务的产品化层。issue 以开放 feature 为主，集中在 L3 serving runtime、KV cache、platform management、parallel strategy、prefix cache、Qwen/DeepSeek serving、性能指标和 lib-serving 解耦。

代表问题：

- #7 pypto-serving vs vLLM feature support comparison
- #16 NPU-Memory-Aware KV Cache Auto-Sizing
- #18 L3 Serving Runtime Design
- #27 KV Cache NPU-to-SSD Offload Plan
- #28 Fine-grained TTFT / TPOT metrics
- #32 Platform Management Design
- #36 Parallel Strategies Support
- #65 Lib-Serving Decoupling Architecture Design
- #91 Long prompts cause task-ring heap deadlock

## 开发者全流程痛点

### 产品能力理解阶段

痛点：
- 用户会自然拿 pypto-serving 和 vLLM 对比，但当前能力边界、已支持模型、未支持特性需要更清晰。
- Serving 被拆成 platform、model support、runtime、cache、parallel 等子模块，架构理解成本高。

期望：
- 能快速判断：这个 serving 框架现在支持什么模型、什么硬件、什么精度、什么并行策略、什么缓存能力。

机会点：
- 增加 feature support matrix，对标 vLLM：模型、batching、prefix cache、KV offload、parallel、quant、metrics、API。
- 提供 serving architecture map，解释 platform management、model module、L3 worker、cache backend、lib-serving 的职责。

### 部署与资源管理阶段

痛点：
- KV cache sizing、NPU memory、SSD offload、per-token malloc/free、cache pool 生命周期都是部署稳定性的关键。
- 长 prompt、prefix cache cold-hit-miss、task-ring heap deadlock 等问题说明资源管理需要更强保护。

期望：
- Serving 在启动前能做容量规划，在运行中能监控 cache/memory 状态，并在危险前给出告警或降级策略。

机会点：
- 增加 NPU-memory-aware capacity planner：根据模型、batch、seq length、KV dtype、parallel strategy 估算可服务范围。
- 提供 KV cache dashboard/report：命中率、block 使用、碎片、offload、eviction、prefix reuse。
- 长 prompt 场景增加 chunked prefill 和 task-ring heap guard。

### 性能与服务质量阶段

痛点：
- 单一 e2e tok/s 指标混合了 prefill、decode、调度、网络/框架开销，无法指导优化。
- 用户需要 TTFT、TPOT、output throughput、kernel latency、cache hit 等细粒度指标。

期望：
- 性能指标应能对应到服务体验和底层瓶颈。

机会点：
- 标准化 serving metrics：TTFT、TPOT、prefill throughput、decode throughput、output throughput、queue wait、cache hit、NPU memory。
- 增加 benchmark skill / script，覆盖性能和精度评估。
- 支持与 vLLM/CANN baseline 的同配置对比。

---

# 跨仓库系统级机会

## 1. 统一 `doctor`：把环境、版本、硬件能力前置

覆盖仓库：

- `simpler`
- `PTOAS`
- `pto-isa`
- `pypto-lib`
- `pypto-serving`

设计机会：
- 一个顶层 `pypto doctor` 检查 Python、CANN、NPU、PTOAS、PTO-ISA、simpler、runtime pins、A5/A2A3 能力。
- 输出按严重程度分级：blocking、warning、info。
- 提供“如何修复”的下一步命令。

## 2. 统一 `explain`：从源码一路解释到 runtime

覆盖仓库：

- `pypto`
- `PTOAS`
- `pto-isa`
- `simpler`
- `pypto-lib`

设计机会：
- `pypto explain` 支持输入源码、IR dump、PTOAS output、runtime trace。
- 输出 source span、IR pass change、PTOAS sync、ISA constraint、runtime task/event 的关联链。
- 重点解决 wrong output、hang、性能回退这三类高频痛点。

## 3. 统一错误码和诊断词典

覆盖仓库：

- `simpler`
- `PTOAS`
- `pto-isa`
- `pypto-serving`

设计机会：
- 建立 error dictionary：507018、scheduler stall、driver SIGSEGV、runtime fatal code、ISA errata。
- 每个错误包含：含义、可能层级、常见原因、建议收集的证据、下一步命令。

## 4. 统一性能报告

覆盖仓库：

- `PTOAS`
- `simpler`
- `pypto-lib`
- `pypto-serving`

设计机会：
- kernel 层：sync/barrier、pipe、layout、冗余 move、tile utilization。
- runtime 层：dispatch wait、queue、slot、backpressure、task timeline。
- model 层：kernel latency、CCE/vLLM baseline gap、sublane utilization、GM round trip。
- serving 层：TTFT、TPOT、throughput、cache hit、NPU memory。

## 5. 统一跨仓库 issue 路由与模板

覆盖仓库：

- 全部仓库

设计机会：
- 在 `pypto_top_level_documents` 中提供 issue routing guide。
- 模板按症状分类：
  - 编译失败
  - wrong output
  - device hang / 507018
  - performance regression
  - serving instability
  - documentation gap
- 每类模板自动提示用户提供对应证据：commit pins、命令、trace、IR dump、hardware、CANN、expected/actual。

## 优先级建议

1. 先做 `doctor` 和错误码解释。
   - 这是最短路径改善环境和运行问题的 UX。

2. 再做 source-to-runtime traceability。
   - 这是解决跨仓库调试断裂的核心能力。

3. 并行整理 top-level developer portal。
   - 让用户知道每个仓库的职责和排查路径。

4. 针对 PTOAS/pto-isa 建立 machine-readable constraints。
   - 让底层硬件限制上浮为上层编译期诊断。

5. 最后沉淀 performance report 与 serving metrics。
   - 支撑真实模型和在线服务的可持续优化。

