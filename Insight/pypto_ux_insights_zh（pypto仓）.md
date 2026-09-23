# PyPTO Issues UX 洞察报告

数据来源：`pypto_issues.json`、`pypto_issues.md`、`pypto_issues.csv`

分析范围：从 `hw-native-sys/pypto` 下载并过滤 Pull Request 后的 561 个真实 issue。

- Open：45
- Closed：516
- 主要标签：bug 229、enhancement 160、code health 39、rfc 23、documentation 14
- 主要标题类型：Bug 183、Feature 142、Pass Bug 77、Code Health 32、RFC 27
- 已关闭 issue 的中位关闭时间：约 1.7 天

## 核心摘要

PyPTO 的 issue 集合不像一个普通新手产品的反馈池，更像是一个面向编译器、运行时、算子和模型内核开发者的高强度开发者体验 backlog。项目响应速度很快，大量 issue 能在较短时间内关闭；但仍然开放的问题高度集中在高级开发流程中：如何用 DSL 表达硬件相关意图、如何信任编译器变换、如何诊断静默错误、如何理解任务依赖，以及如何在分布式和真实模型内核中进行性能调优。

最重要的 UX 机会点不是做一个更漂亮的界面，而是让开发者拥有一个更安全、更可解释的开发闭环：写 PyPTO 代码、理解代码语义、获得可操作的编译诊断、运行时可追踪，并且能有信心地调优性能。

## 开发者使用全流程

### 1. 环境搭建、运行与验证

代表 issue：#63，"How to run the tests such as tests/ut/ir/core/"。

用户痛点：
- README 或测试命令可能与真实环境、包结构发生漂移。
- `ModuleNotFoundError` 这类导入和环境问题会在用户真正接触产品价值之前就形成阻塞。
- 测试体系横跨 unit test、system test、runtime、硬件、Docker 和 CI，用户很难判断应该运行哪条命令来验证自己的改动。

用户期望：
- 新贡献者能在几分钟内跑通一个最小 smoke test。
- 项目能明确告诉用户 Python path、submodule、runtime pin、硬件环境是否正确。
- 测试命令按目的分层：快速本地检查、compiler-only 检查、device 检查、distributed 检查、完整 CI 等价检查。

设计机会点：
- 增加 `pypto doctor` 或 `scripts/doctor.ps1`，自动检查环境、imports、submodules、CANN/runtime pins 和设备可用性。
- 在文档中提供“第一个成功测试”的路径，包括准确命令和预期输出。
- 增加测试命令地图：例如“我改了 DSL parser 应该跑什么”“我改了 codegen 应该跑什么”“我改了 distributed runtime 应该跑什么”“我改了模型 kernel 应该跑什么”。

### 2. 用 DSL 表达开发意图

代表 issue：#1647、#1968、#1368、#2059、#1189。

用户痛点：
- DSL 有时过早暴露底层实现细节。例如 window buffer 分配要求用户手动计算 byte size，但后续又要再次声明 shape 和 dtype。
- 等价的编程模型之间切换成本过高。例如在 `pl.parallel + pl.at` 和 `pl.spmd` 之间切换时，可能需要重新缩进整个代码块。
- 手动任务依赖模型虽然正确、可预测，但在复杂 pipeline 中非常冗长且容易出错。
- 某些运行时能力已经存在，但前端 DSL 尚未暴露，导致用户无法表达真实模型需求。
- 分布式编程涉及 HOST/CHIP/CORE_GROUP、window、signal、rank、predicate、collective 等概念，需要稳定清晰的心智模型。

用户期望：
- PyPTO 应该允许用户用自己正在思考的抽象层级来表达意图：tensor shape、dtype、任务依赖、dispatch 条件、通信模式等。
- 高级 escape hatch 可以存在，但常见路径应该尽量难以误用。
- 当语义不变时，切换执行策略应该是局部小改，而不是重写整个代码块。

设计机会点：
- 将 byte-oriented API 逐步迁移到 shape/dtype-oriented overload，同时保留底层 byte 形式作为显式 escape hatch。
- 提供“语义化 DSL recipes”：SPMD block、带依赖的 manual scope、跨 rank collective、条件 expert dispatch、window allocation 等。
- 为依赖推导提供 opt-in 自动化机制，并提供 explain mode 输出推导出的依赖边。
- 建立 DSL 使用决策指南，解释何时使用 `pl.parallel`、`pl.at`、`pl.spmd`、`manual_scope` 和 orchestration-level collectives。

### 3. 编译阶段的可信度与诊断

代表 issue：#1525、#1305、#2005、#2006、#2047、#2058。

用户痛点：
- 多个高风险 issue 都属于静默正确性问题：输出全零、cache line 被破坏、写入丢失、依赖丢失、scalar aliasing 语义不清。
- 某些 verifier hint 噪声较多、重复，或者难以映射回源码位置。
- 用户可能写出“编译成功但设备运行错误或结果错误”的代码。
- IR 层问题常常需要专家跨多个 pass dump 做取证式排查。

用户期望：
- 不安全或语义模糊的代码模式应尽可能在编译期报错。
- Warning 应该有源码映射、去重，并且可操作。
- 当编译器做了代码变换，用户应该能理解为什么创建了某个依赖、alias、buffer 或 event。

设计机会点：
- 建立诊断分级体系：correctness error、likely correctness hazard、performance hint、informational note。
- 为 verifier 输出和 pass diagnostics 增加 source span 与“为什么发生”的解释。
- 为已知 foot-gun 增加编译期保护：混合 tensor/scalar stores 到重叠 GM cache line、丢失 WAR dependencies、非法 split semantics、detached `pl.Out` reassignment、scalar aliasing ambiguity。
- 提供一键 repro bundle，包含源码、生成的 IR snapshots、pass order、runtime pins 和设备元数据。

### 4. 运行、调试与复现

代表 issue：#1789、#1869、#1840。

用户痛点：
- 硬件或运行时错误可能表现为不透明的 device error 或 deadlock，例如 AICPU 507018。
- flaky failure 和 daily CI failure 往往累积了大量讨论，但很难转化成用户可理解的根因。
- runtime trace、codegen event 和源码操作之间缺少稳定可导航的连接。

用户期望：
- 设备错误不应只给出底层错误码，而应指向可能的 compiler/runtime/source 原因。
- Flaky test 报告应保留足够的上下文，方便复现。
- 开发者应该能把 source operation、IR op、event ID、fence、runtime task 和 trace span 关联起来。

设计机会点：
- 建立 runtime error explainer，覆盖常见 device/AICPU 错误码、可能的 PyPTO 原因和下一步排查命令。
- 为 source region、IR op、runtime task/event 增加 trace correlation ID。
- 标准化 CI failure issue template，固定记录环境、commit pins、trace artifacts、失败命令和疑似子系统。

### 5. 性能优化

代表 issue：#1475、#2040、#1958、#1980。

用户痛点：
- 编译器优化即使功能正确，也可能破坏 pipeline overlap 或硬件利用率。
- 用户需要理解为什么原生 PyPTO kernel 会慢于 CANN FAI 等库实现。
- memory planning、buffer reuse、valid shape preservation 和 layout decision 很难从源码直接推理。

用户期望：
- 性能回退应该能被解释为 pipeline overlap、memory traffic、fence、buffer reuse、layout 等具体因素。
- 编译器应暴露足够信息，让用户判断某个优化到底是帮助了还是伤害了性能。
- 用户需要可操作的调优抓手，而不是只能深入 IR 取证。

设计机会点：
- 增加 performance report mode，总结 buffer reuse、liveness、pipeline overlap、GM round trip 和 emitted fences。
- 让 performance hint 支持源码映射，并按预期影响排序。
- 提供 before/after IR 和 timeline diff 工具，用于分析优化回退。
- 提供模型级 benchmark 模板，用一致指标比较 PyPTO kernel 和 baseline library kernel。

### 6. 构建分布式与 MoE 工作负载

代表 issue：#1189、#1906、#2027、#2029、#2059。

用户痛点：
- 分布式工作需要同时组合 frontend DSL、compiler lowering、runtime scheduling、HCCL/window 概念和模型级约束。
- MoE 工作负载需要在不阻塞 orchestration 的情况下实现 conditional dispatch 和 skip-empty-expert。
- 通信上下文和 provenance 相关问题很难靠用户手动发现。

用户期望：
- DistributedTensor 和 communication context 应携带足够 provenance，避免非法 merge 和 alias。
- 用户应该能在 DSL 中直接表达 collectives 和 conditional dispatch。
- 系统应在保留 static graph 优势的同时，支持 runtime-dependent scheduling choice。

设计机会点：
- 提供分布式编程指南，覆盖 TP、EP、allreduce、allgather、window buffer、signal、dispatch predicate 等标准示例。
- 用图示解释 HOST/CHIP/CORE_GROUP 执行模型和 task dependency flow。
- 为 DistributedTensor context provenance 和跨 rank 通信不变量增加 validator。

## 跨流程 UX 主题

### 1. 静默失败是最高风险的 UX 问题

多个 issue 都描述了“编译成功，但输出错误、全零、数据损坏、写入丢失、读到旧值或运行死锁”的情况。对于面向专用硬件的编译 DSL，最糟糕的用户体验不是编译失败，而是信任被破坏：用户不再知道问题出在源码、编译器、运行时、硬件还是测试 oracle。

优先机会：
- 将已知静默失败模式转成早期诊断。
- 当无法早期诊断时，增加 runtime assertions 或 trace markers。
- 为失败建立 source-to-runtime provenance。

### 2. 用户需要渐进披露，而不是完全隐藏系统细节

提交这些 issue 的用户大多是高级开发者。他们不需要系统完全隐藏 IR、memory 或 scheduling 细节；他们需要这些细节在合适的层级出现，并且能从源码映射到原因。

优先机会：
- 提供三层解释：入门命令、编译器解释、硬件/运行时细节。
- 允许用户从源码行逐层 drill down 到 IR pass 和 runtime event。

### 3. API 人机工程在频繁改写代码的地方最关键

围绕 `pl.spmd`、`pl.parallel`、window allocation、manual deps 和 split semantics 的问题说明：一个小的 API 或语法设计，会在 kernel 开发中放大成很高的改写成本。

优先机会：
- 优化 edit-locality：改变 dispatch 或 memory strategy 时，不应重写主体逻辑。
- 减少 shape、dtype、dependency 等信息的重复声明。

## 优先级设计机会点

### 1. 面向静默正确性风险的诊断升级

- 影响：非常高
- 证据：#1525、#2005、#2006、#2058、#1789
- 预期结果：减少 wrong-output debugging session，提升用户对编译器的信任。

### 2. Developer doctor 与测试工作流地图

- 影响：高
- 证据：#63、daily CI issues、复杂测试体系
- 预期结果：缩短 onboarding 时间，减少环境问题造成的阻塞。

### 3. Source-to-IR-to-runtime 可追踪性

- 影响：高
- 证据：pass bugs、event deadlocks、runtime failures、noisy perf hints
- 预期结果：更快定位 compiler/hardware 边界问题的根因。

### 4. 更安全的高层 DSL 设计

- 影响：高
- 证据：#1968、#1647、#1368、#2059
- 预期结果：用户能以更少底层 foot-gun 表达真实开发意图。

### 5. 分布式编程指南与 validators

- 影响：中高
- 证据：#1189、#2027、#2029、#1906、#2059
- 预期结果：建立 TP/EP/MoE 工作流的清晰心智模型。

### 6. 性能可解释性工具

- 影响：中高
- 证据：#1475、#2040、#1958、#1980
- 预期结果：开发者无需做大量 IR 考古，也能理解并调优性能回退。

## 建议产品化产物

- `pypto doctor`：检查环境、imports、pins、submodules、设备与 smoke test。
- `pypto explain <file_or_dump>`：总结诊断、源码位置、IR pass 变化、推导依赖和 runtime mapping。
- `pypto trace report`：生成 source-to-task/event/fence timeline。
- Known hazards verifier pack：覆盖 overlapping stores、lost dependencies、invalid split semantics、alias ambiguity、detached outputs 等风险模式。
- 开发者旅程文档：
  - Quickstart 与 first test
  - DSL 心智模型
  - Dependency 与 scheduling 模型
  - Distributed programming model
  - Wrong output debugging
  - Performance tuning workflow

