# PyPTO3 产品上下文

<!-- impeccable:product-schema 1 -->

## Platform

web

## Scope

用户于 2026-11-06 确认，本次初始化聚焦四个已有 Demo，而非仓库所有产品方向：

- `Design/agent-decision-lineage/`
- `Design/operator-tuning-console-v5/`
- `Design/PyPTO-Compute-Explorer-UI/`
- `Design/Toolkit Studio/`

本文件记录上述范围的共享背景；各 Demo 的源码、证据与局部说明保留各自权威性，不假定它们已整合为一个正式产品。

## Users

待确认：首要用户角色与使用优先级。

仓库材料指向的用户假设是 PTO / PyPTO 算子开发、性能调优和编译问题分析人员，以及审查 Agent 生成结果的工程人员；这是从现有功能推断的受众，尚未经用户确认。

## Product Purpose

现有实现围绕计算理解、可信 Kernel 工作流、性能诊断与 Agent 决策审查展开。后续设计应保持这四项任务的语义区别。

- Agent Decision Explorer：审查历史 Softmax Agent Run，从需求与设计追踪源码、正确性、性能实验与停止证据。
- Tuning Console V5：从端到端性能下钻至 L2 调度、L1/L0 流水、编译器和 ISA / 布局，并记录优化实验。
- Compute Explorer：从源码上下文进入 Assignment、Pipeline 和 Graph，理解计算与执行结构。
- Toolkit Studio：围绕计算意图、源码、编译证据、正确性分歧与可信基线组织工作，并展示 Agent 协作和人工接管入口。

待确认：四个 Demo 的产品优先级、量化成功标准，以及是否需要统一成同一产品旅程。

## Operating Context

这是产品设计、研究与浏览器交互原型仓库。四个 Demo 均已有 HTML、CSS 与 JavaScript 实现；不能把现有界面当成空白项目。根目录 `launch.html` 是 Demo 统一入口，静态页面可通过仓库根目录 HTTP 服务访问。运行入口和依赖以各 Demo 当前文件为准；说明中的旧目录链接需要核对。

## Capabilities and Constraints

- Agent Decision Explorer 展示历史材料，不连接 Agent 执行器；不得暗示在线重跑、硬件控制或本机实测。
- 区分实测、推导、设计决策、报告中的尝试、未尝试和缺失证据；展示关系不自动代表时间或因果。
- Tuning Console 的诊断应保留证据作用域、rank 上下文和证据不足信息。模拟案例须明确标识，不能包装成真实归因。
- Tuning Console 当前实验模型每次验证一个假设，先记录正确性，再记录性能，之后决定保留或回退。
- Toolkit Studio 中展示的执行与基线状态不自动证明已接通真实服务；真实能力边界需按具体实现与数据核实。
- 复用现有术语、组件和设计系统；界面改动遵循仓库及目标目录的 `AGENTS.md`。
- 用户未要求本次改造任何页面；初始化不构成重设计授权。

## Evidence on Hand

- `Design/agent-decision-lineage/README.md`：工作流、事实分类、数值口径、限制；同目录 `evidence/` 保存原始材料快照。
- `Design/operator-tuning-console-v5/README.md`：跨层调优模型、实验约束、数据来源和模拟案例边界。部分入口描述沿用了旧版本名称，实际目标为 V5 目录。
- `Design/PyPTO-Compute-Explorer-UI/index.html`：源码入口、Inspector 与计算探索视图；同目录 `js/` 保存计算模型和交互实现。
- `Design/Toolkit Studio/index.html`：可信 Kernel 工作流、Agent 团队、编译与正确性界面。
- `vendor/pto-design-system/`：现有 PTO 设计系统资源。

## Product Principles

以下为从现有材料提炼的工作原则，尚待用户确认其产品优先级：

- 从具体计算、源码或运行对象进入分析，保留上下文。
- 让结论可追溯到证据，并公开证据缺口。
- 将正确性作为性能优化决策的前置条件。
- 支持从概览逐层下钻，避免不同尺度的信息混淆。

## Open Decisions

- 首要用户、核心使用场景与成功标准。
- 四个 Demo 的整合关系及优先级。
- 真实执行服务、目标设备与交付部署要求。
- 产品专属的无障碍要求。
