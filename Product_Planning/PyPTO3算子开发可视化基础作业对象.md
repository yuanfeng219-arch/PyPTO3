# PyPTO3 算子开发可视化基础作业对象

## 1. 文档目的

本文用于系统梳理 PyPTO3/PyPTO 算子开发工具中可以抽象、可视化并进一步封装为智能体工具的基础作业对象。

这里的“可视化基础作业对象”不是简单的页面组件，而是同时具备以下能力的领域对象：

- 能表达一个明确的算子开发语义；
- 能被可视化、定位和解释；
- 能被智能体查询、修改、验证或执行；
- 能记录状态、约束、证据和历史；
- 能与其他对象形成可追踪的关系。

PyPTO 的算子体系已经提供了较清晰的对象基础，包括 TensorOp、TileOp、SyncOp、CrossCoreOp 和 PrefetchOp，以及 `Tensor → tile.load → TileOp → tile.store → Tensor` 的典型数据流。

参考：

- [PyPTO 算子系统](https://www.pypto.ai/pypto/zh/dev/ir/05-operators/)
- [PyPTO 算子开发编排示例](https://gitcode.com/cann/cannbot-skills/blob/master/plugins-official/pypto-op-orchestrator/quickstart.md)

## 2. 总体作业链路

```text
算子需求
  ↓
算子规格
  ↓
计算图 / IR
  ↓
Kernel 设计
  ↓
Tile、内存、流水、同步
  ↓
编译产物
  ↓
测试与精度验证
  ↓
性能分析
  ↓
问题定位与调优
```

每个环节都可以抽象为一个或多个可独立操作的作业对象。

## 3. 基础作业对象分类

| 对象层 | 基础对象 | 主要表达内容 | 推荐可视化方式 |
|---|---|---|---|
| 项目上下文 | `Project` | 当前算子项目、目录、版本、依赖 | 工程树、状态面板 |
| 项目上下文 | `Target` | 芯片型号、核心类型、内存层级、指令能力 | 设备卡片、硬件拓扑 |
| 项目上下文 | `Environment` | PyPTO、CANN、Python、编译器版本 | 环境检查面板 |
| 需求规格 | `OperatorSpec` | 输入输出、属性、shape、dtype、算子语义 | 算子规格卡 |
| 约束 | `Constraint` | shape、dtype、layout、对齐、容量、硬件限制 | 约束清单、规则标签 |
| 数据语义 | `Tensor` | Tensor 名称、shape、dtype、设备位置 | 张量卡片、张量关系图 |
| 数据语义 | `Dimension` | 静态维、动态维、符号维、广播关系 | Shape 编辑器 |
| 数据语义 | `Layout` / `View` | layout、stride、reshape、transpose、valid region | Layout 视图、矩阵视图 |
| 计算结构 | `GraphNode` | 计算图节点、输入输出、依赖关系 | 计算图 |
| 计算结构 | `Region` / `Subgraph` | 融合区域、循环区域、归约区域 | 子图框选、区域高亮 |
| Kernel 设计 | `Tile` | Tile shape、valid shape、dtype、memory space | Tile 网格、分块矩阵 |
| Kernel 设计 | `ComputeOp` | add、mul、matmul、reduce、cast 等计算 | 算子节点、指令卡 |
| Kernel 设计 | `DataMovement` | load、store、move、prefetch、copy | 数据流箭头、内存流转图 |
| Kernel 设计 | `Accumulator` | 累加器、初始化条件、归约阶段 | 累加链路图 |
| Kernel 设计 | `MemoryObject` | GM、L2、UB、L0A/L0B/L0C 等内存对象 | 内存层级图 |
| Kernel 设计 | `PipelineStage` | load、compute、store 阶段及重叠关系 | 流水线泳道图 |
| Kernel 设计 | `SyncObject` | barrier、fence、syncall、跨核同步 | 同步依赖图 |
| Kernel 设计 | `CoreMapping` | AIC/AIV、SPMD、核间分工、任务切分 | 核拓扑图 |
| 分布式 | `Communication` | send、recv、all-reduce、remote load 等 | 通信拓扑图 |
| 构建产物 | `CodeArtifact` | Python kernel、IR、生成代码、编译日志 | 代码视图、IR 树 |
| 验证 | `TestCase` | 输入 shape、dtype、随机种子、边界数据 | 测试矩阵 |
| 验证 | `Golden` | 参考实现、参考输出、误差阈值 | 结果对比视图 |
| 验证 | `CheckResult` | 编译检查、shape 检查、精度检查 | 质量门禁 |
| 问题 | `Issue` / `Failure` | 编译错误、精度错误、性能问题 | 问题列表、错误定位 |
| 性能 | `Run` | 一次编译、执行、测试或 profiling 任务 | Run 时间线 |
| 性能 | `Timeline` | kernel、load、compute、sync 的时间关系 | 性能泳道图 |
| 性能 | `Metric` | latency、throughput、bandwidth、occupancy 等 | 指标卡、趋势图 |
| 性能 | `Bottleneck` | 访存瓶颈、计算瓶颈、同步瓶颈、空转 | 瓶颈热力图 |
| 调优 | `Experiment` | Tile shape、pipeline、layout 等参数组合 | 实验对比表 |
| 调优 | `Recommendation` | 智能体给出的修改建议及依据 | 建议卡片 |
| 知识复用 | `Pattern` / `Template` | 常见算子结构、优化模板、经验规则 | 模式库、模板卡 |
| 追踪 | `Provenance` | 对象来源、版本、运行证据、修改历史 | 血缘图、变更时间线 |

## 4. MVP 优先对象

如果第一阶段需要快速形成可用闭环，建议优先实现以下 12 个对象：

1. `OperatorSpec`：算子需求和接口定义；
2. `Tensor`：输入、输出和中间张量；
3. `ShapeConstraint`：shape、动态维、广播、对齐约束；
4. `GraphNode`：计算图节点；
5. `Tile`：Tile shape、valid shape、内存空间；
6. `ComputeOp`：计算操作；
7. `DataMovement`：load、store、move、prefetch；
8. `MemoryObject`：各级内存和 buffer；
9. `PipelineStage`：流水线阶段；
10. `TestCase`：测试输入和覆盖范围；
11. `Run`：编译、执行、验证、profiling 任务；
12. `Issue`：错误、瓶颈和待处理问题。

这组对象可以覆盖：

```text
需求理解 → 算法设计 → Kernel 实现 → 编译 → 验证 → 性能调优
```

## 5. 对象统一模型

一个可复用的作业对象建议至少包含以下字段：

```json
{
  "id": "tile_001",
  "type": "Tile",
  "name": "lhs_tile",
  "status": "validated",
  "source_ref": "kernel.py:42",
  "parent_id": "matmul_region_01",
  "children": [],
  "inputs": ["tensor_a"],
  "outputs": ["matmul_01"],
  "properties": {
    "shape": [32, 128],
    "valid_shape": [32, 100],
    "dtype": "FP16",
    "memory_space": "UB"
  },
  "constraints": [
    "N must be aligned to 16",
    "UB capacity is sufficient"
  ],
  "evidence": [
    "compile_pass",
    "profile_run_003"
  ],
  "actions": [
    "inspect",
    "change_shape",
    "check_capacity",
    "profile"
  ],
  "provenance": {
    "created_by": "designer_agent",
    "created_at": "2026-09-09T00:00:00Z",
    "version": 3
  }
}
```

### 5.1 统一字段建议

| 字段 | 作用 |
|---|---|
| `id` | 全局唯一标识，供智能体引用 |
| `type` | 对象类型，例如 `Tile`、`Tensor`、`Issue` |
| `status` | 对象当前状态 |
| `source_ref` | 对应代码、IR、日志或文档位置 |
| `parent_id` / `children` | 对象层级关系 |
| `inputs` / `outputs` | 数据或作业依赖 |
| `properties` | 对象自身的领域属性 |
| `constraints` | 对象必须满足的规则 |
| `evidence` | 验证、编译、运行、profiling 证据 |
| `actions` | 智能体可执行的动作 |
| `provenance` | 创建者、来源、版本和变更历史 |

## 6. 对象状态机

建议所有主要作业对象使用统一状态模型：

```text
draft
  ↓
specified
  ↓
designed
  ↓
implemented
  ↓
compiled
  ↓
validated
  ↓
profiled
  ↓
optimized
```

异常状态：

```text
failed
blocked
```

状态切换应由证据驱动。例如：

- `compiled`：编译成功且生成目标产物；
- `validated`：测试通过且精度满足阈值；
- `profiled`：已经采集到有效性能数据；
- `optimized`：相对基线达到明确的优化目标。

## 7. 核心对象关系

```text
OperatorSpec
  ├── defines → Tensor
  ├── contains → GraphNode
  └── constrained_by → Constraint

GraphNode
  ├── consumes → Tensor
  ├── produces → Tensor
  ├── lowered_to → ComputeOp
  └── grouped_into → Region

ComputeOp
  ├── operates_on → Tile
  ├── uses → MemoryObject
  ├── connected_by → DataMovement
  ├── synchronized_by → SyncObject
  └── scheduled_as → PipelineStage

PipelineStage
  └── measured_by → Timeline / Metric

Run
  ├── validates → TestCase
  ├── produces → CheckResult
  ├── generates → Issue
  └── compared_with → Experiment
```

## 8. 六类主要可视化视图

### 8.1 计算图视图

展示：

- Tensor 依赖；
- 算子节点；
- 融合区域；
- 数据流；
- 编译前后 IR 变化。

主要服务于规划、架构设计和调试智能体。

### 8.2 Shape / Tensor 视图

展示：

- shape；
- 动态维；
- dtype；
- broadcast；
- transpose；
- valid region；
- layout。

主要服务于算法分析、类型检查和精度验证。

### 8.3 内存与 Tile 视图

展示：

- GM、UB、L0 等内存层级；
- Tile 分块；
- buffer 占用；
- 数据搬运路径；
- 对齐和容量约束。

主要服务于 Kernel 设计和性能优化。

### 8.4 Pipeline / Timeline 视图

展示：

- load、compute、store；
- 各阶段重叠；
- barrier；
- 核间同步；
- stall 和空闲区间。

主要服务于性能分析和瓶颈定位。

### 8.5 验证与问题视图

展示：

- 测试用例覆盖；
- Golden 对比；
- 最大误差；
- 失败输入；
- 错误定位；
- 修复历史。

主要服务于验证和 Debug 智能体。

### 8.6 版本与证据视图

展示：

- 代码变更；
- IR 变更；
- 参数实验；
- profile 对比；
- 建议来源；
- 结论是否已经验证。

主要服务于智能体协作和结果追溯。

## 9. 面向智能体的动作接口

### 9.1 查询类动作

```text
get_object(object_id)
list_children(object_id)
list_dependencies(object_id)
find_by_type(type, filters)
explain_object(object_id)
```

### 9.2 分析类动作

```text
check_constraints(object_id)
infer_shape(object_id)
analyze_precision(object_id)
analyze_memory(object_id)
find_bottleneck(run_id)
trace_failure(issue_id)
```

### 9.3 修改类动作

```text
create_object(type, properties)
update_object(object_id, patch)
change_tile_shape(tile_id, shape)
change_layout(object_id, layout)
fuse_region(region_id, node_ids)
insert_sync(location, sync_type)
modify_pipeline(pipeline_id, patch)
generate_patch(issue_id, strategy)
```

### 9.4 执行类动作

```text
compile(project_id, config)
run_test(test_case_id)
run_profile(run_config)
compare_runs(run_a, run_b)
validate(object_id, policy)
accept_gate(gate_id)
```

## 10. 对象与智能体角色映射

| 智能体角色 | 主要操作对象 |
|---|---|
| Planner | `OperatorSpec`、`Constraint`、质量门禁 |
| Mathematician | `Tensor`、`Dimension`、`DType`、`Layout` |
| Architect | `GraphNode`、`Region`、`Dependency` |
| Designer | `Tile`、`MemoryObject`、`PipelineStage`、`SyncObject` |
| Coder | `CodeArtifact`、`ComputeOp`、`DataMovement` |
| Verifier | `TestCase`、`Golden`、`CheckResult` |
| Debugger | `Issue`、`Failure`、`Trace`、`Provenance` |
| Optimizer | `Run`、`Timeline`、`Metric`、`Experiment`、`Recommendation` |

## 11. 设计原则

### 11.1 不要按 API 一一建模

不建议把 `tile.add`、`tile.mul`、`tile.matmul` 设计成完全独立的顶层对象。

更合适的方式是：

```text
ComputeOp
  ├── ElementwiseOp
  ├── ReductionOp
  ├── MatmulOp
  ├── CastOp
  └── QuantizeOp
```

具体 API 作为对象实例或属性，而不是顶层对象类型。

### 11.2 区分事实、诊断、建议和结论

```text
事实：UB 使用率为 92%
诊断：当前 Tile 可能导致访存压力较高
建议：尝试将 Tile 从 [64, 128] 调整为 [32, 128]
结论：调整后 latency 降低 8.4%
```

这四类信息应分别保存，避免智能体把未经验证的推断当作最终结论。

### 11.3 所有结论都应绑定证据

一个性能建议至少应能关联：

```text
建议
  → 修改对象
  → 编译结果
  → 测试结果
  → Profile
  → 对比结论
```

### 11.4 一个对象支持多个视图

例如同一个 `Tile` 对象可以同时出现在：

- 计算图视图；
- 内存视图；
- Pipeline 视图；
- 性能问题视图；
- 代码定位视图。

对象模型与前端视图应解耦。

## 12. 推荐的第一版对象域

```text
SpecDomain
  ├── OperatorSpec
  ├── Tensor
  └── ShapeConstraint

KernelDomain
  ├── GraphNode
  ├── Tile
  ├── ComputeOp
  ├── DataMovement
  └── MemoryObject

VerifyDomain
  ├── TestCase
  ├── Golden
  ├── CheckResult
  └── Issue

ProfileDomain
  ├── Run
  ├── Timeline
  ├── Metric
  ├── Bottleneck
  └── Experiment

KnowledgeDomain
  ├── Pattern
  ├── Recommendation
  └── Provenance
```

这五个对象域可以覆盖大多数 PyPTO 算子开发场景，并为后续扩展分布式通信、量化、自动调优和知识库提供基础。

## 13. 建议的落地顺序

### 第一阶段：对象底座

- 统一对象 ID 和类型系统；
- 建立对象状态机；
- 建立对象关系图；
- 支持代码、IR、日志和运行结果的来源定位；
- 实现 `get_object`、`list_dependencies`、`explain_object`。

### 第二阶段：算子开发闭环

- 接入 `OperatorSpec`、`Tensor`、`GraphNode`；
- 接入 `Tile`、`ComputeOp`、`DataMovement`；
- 支持编译、测试、精度验证；
- 形成 `Issue` 和证据链。

### 第三阶段：性能闭环

- 接入 `Run`、`Timeline`、`Metric`；
- 构建内存、Tile、流水线视图；
- 支持实验对比；
- 形成自动调优建议。

### 第四阶段：知识复用

- 沉淀算子模式和优化模板；
- 建立问题到解决方案的映射；
- 支持智能体基于历史案例检索和复用；
- 将经过验证的优化方案沉淀为 `Pattern`。

## 14. 最终建议

建议将平台核心定义为：

> 一套面向 PyPTO 算子开发的、可观察、可解释、可修改、可验证、可追踪的作业对象图谱。

其中：

- 对象模型负责统一语义；
- 可视化负责提供不同观察视角；
- 智能体动作负责完成分析和修改；
- 状态机负责控制开发流程；
- 证据链负责保证结论可信；
- Pattern 和 Template 负责沉淀可复用知识。

