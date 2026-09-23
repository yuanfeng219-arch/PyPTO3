# Operator Tuning Console V2

V2 是 operator-tuning-console 的独立入口，不替换原 Demo。它复用原始上板数据、五层视图、Pass 轨迹、Inspector 和实验台账，但将默认信息架构改为以调优任务（Investigation）为中心：

Run / 基线 → Investigation → 假设 → 单变量实验 → 回归更新

## 设计原则

- 左栏管理 Investigation，而不是独立的层级告警队列；每个任务绑定目标、状态、证据、假设和实验。
- 假设明确显示证据、待区分的问题与置信度等级；“强支持”“约束耦合”“待验证”不会混为因果结论。
- 每个任务只给出一个当前实验。执行时跳回原工作台的对应发现，并继续使用其原有实验台账，避免两套记录分叉。
- 原有的端到端、L2 调度、单核、编译器、ISA 页签被保留，作为打开证据后的原始观察视图。

## 数据与边界

页面直接读取 ../operator-tuning-console/data.js。数据构建器会把每个 run 的 Investigation、假设、实验计划与 finding 引用一并写入产物；V2 只负责呈现和跳转。V2 不生成预测收益；推断只基于当前 run 的观测与各 finding 的 guardrail，必须经同基线、正确性和端到端回归确认。
