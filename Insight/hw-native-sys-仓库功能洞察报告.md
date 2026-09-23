# hw-native-sys 组织仓库功能洞察报告

**分析对象**：https://github.com/orgs/hw-native-sys/repositories
**报告日期**：2026-07-14
**数据来源**：各仓库 README、目录结构、`.gitmodules` 子模块声明、GitHub 组织页元数据

---

## 一、核心结论

组织页显示 8 个仓库，其中 `.github` 是组织元仓库，实质代码/文档仓库为 **7 个**。

这 7 个仓库**不是 7 个独立项目，而是一条完整的垂直技术栈**——华为昇腾（Ascend）NPU 的 **PTO（Parallel Tile Operation）编程与推理软件栈**，从底层虚拟指令集一路向上贯通到 LLM 在线推理服务。

三条硬证据说明它们是一个整体：

1. `pypto/.gitmodules` 将 `hw-native-sys/simpler` 挂载为子模块，路径名直接叫 `runtime`；
2. `pypto-serving/.gitmodules` 将 `hw-native-sys/pypto-lib` 挂载为子模块；
3. `pypto-lib` 的 README 里有一张显式依赖表，同时指向 pypto、simpler、ptoas、pto-isa 四个仓库，并说明"版本 pin 在 CI 配置里"。

所有代码仓库统一采用 **CANN Open Software License 2.0**——该协议限定"仅可用于开发运行在华为 AI 处理器（昇腾 / 麒麟 / 鲲鹏等）上的软件"，这是一条强烈的战略信号：**这是华为在 CUDA 生态之外自建的 Triton / TileLang 类对标技术栈**，以开源社区形式运作，但许可证锁死在自家硬件上。

---

## 二、分层架构总览

```
        ┌──────────────────────────┐
        │      pypto-serving       │  应用层：OpenAI 兼容 API、KV Cache、调度
        └────────────┬─────────────┘
                     │ submodule
        ┌────────────▼─────────────┐
        │        pypto-lib         │  算子/模型层：Qwen3、DeepSeek kernel
        └────────────┬─────────────┘
                     │ 依赖
        ┌────────────▼─────────────┐
        │          pypto           │  编译框架层：Tensor→Tile→Block→Exec 多级 IR
        └──────┬──────────────┬────┘
               │              │
        ┌──────▼─────┐  ┌─────▼──────┐
        │   PTOAS    │  │  pto-isa   │  编译后端 / 虚拟指令集定义
        └──────┬─────┘  └─────┬──────┘
               │              │
        ┌──────▼──────────────▼────┐
        │         simpler          │  运行时层：Host/AICPU/AICore 三程序任务图
        └────────────┬─────────────┘
                     │
        ┌────────────▼─────────────┐
        │   昇腾 NPU (A2/A3/A5)     │
        └──────────────────────────┘

        pypto_top_level_documents ──── 横切：跨仓库顶层设计文档
```

---

## 三、逐仓库解析

### 1. `pto-isa` — 地基：PTO Tile 虚拟指令集

| 项目 | 内容 |
|---|---|
| 语言 | C++ |
| 定位 | **整个技术栈的语义地基** |
| 热度 | 47 stars / 26 forks |

**做什么**：定义并实现 PTO（Parallel Tile Operation）——一套由昇腾 CANN 定义的**面向 Tile 的虚拟 ISA**，目前包含 90+ 条标准 tile 指令，覆盖计算、数据搬运、量化、卷积、归约，以及一套独立的**通信扩展指令集**（点对点、信号同步、集合通信），用于构建计算-通信深度融合的 kernel。

**设计哲学**：它的目标明确写着"不是隐藏底层能力，而是抬高抽象层级同时保留调优空间"——通过 tile 级抽象抹平 A2/A3/A5 各代昇腾芯片的实现差异，降低算子跨代迁移成本，同时保留 tile size / tile shape / 指令排布这些调优维度。

**关键洞察**：
- 它是**唯一一个已经外溢出本组织的仓库**——README 明确提到 PTO 指令已被集成进 [TileLang Ascend](https://github.com/tile-ai/tilelang-ascend/) 以及 gitcode 上的 pypto 镜像。这说明 pto-isa 被定位为**公共接口标准**，而非 pypto 的私有实现细节。
- 提供 **CPU 模拟器**，使得没有昇腾硬件也能做功能验证——这是开源生态获客的关键设计。
- README 的 News 时间线（2025-12 开源 → 2026-01 加归约/MX → 2026-02 加卷积/量化/通信 → 2026-03 加 A5 支持与 CostModel）显示这是**当前迭代最激进的仓库**，指令集仍在快速扩张。

---

### 2. `PTOAS` — 编译后端：PTO 汇编器与优化器

| 项目 | 内容 |
|---|---|
| 语言 | C++ |
| 定位 | LLVM/MLIR 编译后端 |
| 热度 | 26 stars / 53 forks（**fork 数是 star 数的 2 倍**，见下方洞察）|

**做什么**：基于 **LLVM/MLIR（LLVM21 的 VPTO 分支）** 构建的 Out-of-Tree 编译器工具链，专门处理 **PTO Bytecode**。四项职责：

1. 解析 `.pto` 文件，校验 PTO Dialect 的语义正确性；
2. 执行针对**达芬奇架构（Da Vinci）** 的优化 Pass——算子融合、自动同步插入等；
3. 将 PTO IR 下降（Lowering）到 `EmitC` / `Linalg` Dialect，最终生成调用 `pto-isa` C++ 库的代码；
4. 提供 Python 绑定，让 PyPTO / PTODSL / CuTile 等前端能在 Python 侧直接构建和编译 PTO Bytecode。

产出两个命令行工具：`ptoas` 和 `ptobc`。

**关键洞察**：
- **PTOAS 是 pto-isa 的"消费者"**——Lowering 的终点就是生成调用 pto-isa 库的 C++ 代码。二者是"指令集定义"与"指令集编译器"的经典关系。
- README 提到它同时服务 **PyPTO、PTODSL、CuTile** 三个前端。**PTODSL 和 CuTile 在本组织中并不存在**——这暗示华为内部还有未开源（或在别处开源）的 PTO 前端语言，PTOAS 被有意设计成**多前端共享的编译中枢**。CuTile 这个命名尤其值得注意，它对标的是 NVIDIA CUTLASS/CuTe 的 tile 抽象。
- 依赖一个 fork 的 LLVM（`vpto-dev/llvm-project:feature-vpto-llvm21`）和一个私有子模块 `PTO-Gym`（`git@github.com:PTO-ISA/PTO-Gym.git`，SSH 地址，外部不可达）——说明还有**未公开的组织 `PTO-ISA` 和一个疑似性能调优/强化学习环境的 PTO-Gym 项目**。这是本次分析中最值得追踪的线索。
- **异常信号**：53 forks vs 26 stars，且有 34 个 open PR。fork 远超 star 通常意味着**贡献者数量 >> 围观者数量**——即这是一个内部团队在公开仓库上协同开发的仓库，而非社区自发追捧的项目。整个组织都呈现这一特征（见第五节）。

---

### 3. `simpler` — 运行时：任务图执行框架

| 项目 | 内容 |
|---|---|
| 语言 | C++ |
| 定位 | PTO Runtime（**注意：仓库名叫 simpler，README 标题却是 "PTO Runtime"**）|
| 热度 | 19 stars / 51 forks |

**做什么**：在昇腾设备上构建并执行**任务依赖图（DAG）** 的模块化运行时。核心是**三程序模型**——Host（`.so`）、AICPU（`.so`）、AICore（`.o`）三个独立编译的程序通过明确定义的 API 协同工作。

**两种运行时变体**：

| 变体 | 图构建位置 | 用途 |
|---|---|---|
| `host_build_graph` | 主机 CPU | 开发、调试 |
| `tensormap_and_ringbuffer` | AICPU（设备侧） | 生产负载 |

生产变体把图构建下沉到设备侧 AICPU，用 TensorMap + RingBuffer 机制，是典型的**降低 Host-Device 交互开销**的设计。

**四个平台**：`a2a3`（真机）、`a2a3sim`（纯线程模拟，只需 gcc）、`a5`、`a5sim`。

**架构分层**：文档中有 L0–L6 的层级模型和 Orchestrator（DAG 提交）/ Scheduler（DAG 派发）/ Worker（执行）的组件划分。

**关键洞察**：
- **命名与定位严重脱节**。仓库名 `simpler` 毫无信息量，README 标题却是 "PTO Runtime"，而 pypto 把它挂成子模块时重命名为 `runtime/`。这是典型的**内部代号泄漏到公开命名空间**——很可能是内部孵化项目直接开源，未做命名治理。对外部开发者而言这是显著的可发现性障碍。
- 提供**无硬件模拟器**（`a2a3sim` / `a5sim`，只需 gcc/g++），与 pto-isa 的 CPU-SIM 形成呼应——整个栈都在刻意降低"没有昇腾卡就无法参与"的门槛。
- 它是 pypto 的子模块，但**同时被 pypto-lib 和 pypto-serving 直接引用**（serving 的 `python/runtime/` 就是 "Simpler worker wrapper for NPU dispatch"）——说明 simpler 是**跨层被穿透调用的横切组件**，不是严格的分层依赖。

---

### 4. `pypto` — 中枢：Tile 编程框架与多级 IR 编译器

| 项目 | 内容 |
|---|---|
| 语言 | Python 64% / C++ 35% |
| 定位 | **整个组织的旗舰项目与流量入口** |
| 热度 | **72 stars / 68 forks（组织内最高）**，848 commits，51 open issues |

**做什么**：面向 AI 加速器的高性能编程框架，核心是 **Tile-based 编程模型**。通过多级 IR 系统，把用 Python API 写的模型逐级编译为硬件指令：

```
Tensor Graph → Tile Graph → Block Graph → Execution Graph → PTO 虚拟指令 → 可执行代码
```

每一级转换都带一组优化 Pass；最终以 **MPMD（Multiple Program Multiple Data）** 方式调度到设备端处理器核心。

**分层抽象设计（这是它最核心的产品思想）**：

| 开发者角色 | 使用抽象层 | 目的 |
|---|---|---|
| 算法开发者 | **Tensor 层** | 快速实现验证算法，不关心硬件 |
| 性能优化专家 | **Tile / Block 层** | 深度性能调优 |
| 系统开发者 | Tensor/Tile/Block + PTO 虚拟指令层 | 集成三方框架、开发工具链 |

**关键洞察**：
- **pypto 是整条链路的"腰"**：向上被 pypto-lib 和 pypto-serving 使用，向下把 simpler 吞为子模块、把 PTOAS 当编译后端、把 pto-isa 当指令目标。**7 个仓库中有 5 个与它直接相连。**
- 仓库根目录同时存在 `.claude/`、`.gemini/`、`AGENTS.md` 三个 AI 编码助手配置——**这个项目是重度 AI-assisted 开发的**。pypto-lib 里甚至有 `.claude/skills/setup_env/SKILL.md` 这样的自定义 skill。这在华为的开源项目中是个相当现代的工程信号。
- 描述写的是 "A community-driven pypto implementation"（社区驱动的 pypto 实现）——措辞暗示**存在一个"官方"pypto**（大概率是 CANN 内部版本 / gitcode 上的 `cann/pypto`），而 GitHub 这个是社区镜像/共建版。这一点在 pto-isa 的 README 里被证实：它引用的 PyPTO 链接指向 **gitcode.com/cann/pypto** 而非本组织。

---

### 5. `pypto-lib` — 算子与模型库

| 项目 | 内容 |
|---|---|
| 语言 | Python |
| 定位 | pypto 的"标准库"+ 最佳实践样板 |
| 热度 | 8 stars / 30 forks |

**做什么**：基于 pypto 框架编写的 **Tensor 级算子（kernel）与端到端模型实现**，目标硬件是 910B/C 和 950。

结构分三块：
- `examples/`：教学梯度（beginner: hello_world、matmul → intermediate: softmax、rms_norm、rope → advanced: 多阶段融合 + 指令组合 kernel，如 gemm_eltwise、multi_proj、topk）
- `models/`：**真实大模型 kernel**——Qwen3-14B（prefill + decode）、Qwen3-32B（decode）、DeepSeek V3.2-EXP、**DeepSeek V4**
- `golden/`：测试基建——编译、上设备跑、与 PyTorch 结果对比验证

**关键洞察**：
- **这是整个组织的"能力证明"仓库**。一个编译框架好不好，看它能不能跑通最难的模型。`models/deepseek/v4/` 的存在说明该栈正在追踪最前沿的开源模型。
- 文档体系异常完整：`compile-runtime-workflow.md`、`debugging.md`、`performance-tuning.md`（L2 kernel 间 + L1/L0 kernel 内调优、Perfetto swimlane、PMU 计数器）、`precision-tuning.md`（`pl.cast` 舍入模式与 torch 对齐、量化方案、误差分布阈值扫描）。**这四篇文档实际上是整个 PTO 栈的"用户手册"**——比 pypto 自己的 README 更有实操价值。
- 它的 README 是**唯一一份把四个依赖仓库的角色讲清楚的文档**。想快速理解这个组织，pypto-lib 的 README 是最佳入口，而非 pypto 的。

---

### 6. `pypto-serving` — 应用层：LLM 推理服务

| 项目 | 内容 |
|---|---|
| 语言 | Python + C++ |
| 定位 | 栈顶应用，vLLM/SGLang 的昇腾对标 |
| 热度 | **1 star / 4 forks（组织内最低，最年轻）** |

**做什么**：一个本地 LLM 推理服务栈，用 PyPTO kernel 在昇腾 NPU 上跑 **Qwen3-14B** 生成。包含：

- **Python 服务层**：engine、scheduler、KV cache、模型加载、异步服务、批处理
- **CLI + OpenAI 兼容 HTTP API**：`/v1/completions`、`/v1/chat/completions`，支持流式
- **C++ platform 层**：一个独立的分布式系统管理层，围绕 `serving::system::Engine` 构建，负责实例生命周期、RPC 通道、模块服务——**明确声明"不进入 per-token 执行热路径"**，只做编排与监管
- Profiling：支持导出 Chrome trace

**外部依赖**：`platform/extern/` 挂载了两个华为 CSL（中央软件院）的子模块——[HiCR](https://github.com/huawei-csl/HiCR) 和 [TaskR](https://github.com/huawei-csl/TaskR)。

**关键洞察**：
- **这是组织的战略终局，也是当前最薄弱的一环**。1 star、只跑 Qwen3-14B 一个模型、README 里的功能大多是 "quick checks" 级别——**明显处于早期原型阶段**。
- 但它的架构野心不小：Python 模型服务路径 与 C++ 平台管理层**刻意解耦**（模型侧管 batching/KV cache/采样，平台侧管分布式引导、部署元数据、实例监管）。这个切分是冲着**多机多卡生产部署**去的，不是玩具。
- 引入 HiCR / TaskR（华为中央软件院的通用分布式运行时组件）说明 **serving 层正在与华为更广泛的系统软件基建整合**——PTO 栈不是孤岛。
- 顶层文档仓库里有 `pypto_serving_reference_sglang_vllm.md`——**他们在明确对标 vLLM 和 SGLang**。

---

### 7. `pypto_top_level_documents` — 横切：跨仓库顶层设计文档

| 项目 | 内容 |
|---|---|
| 语言 | Markdown（+ 1 个 Python 脚本）|
| 定位 | **跨仓库架构决策的"设计中枢"** |
| 热度 | 4 stars / 4 forks，33 commits |

**做什么**：不是代码仓库，而是**跨越多个仓库的顶层设计文档集散地**。文件清单极具信息量：

| 文档 | 揭示的信息 |
|---|---|
| `HL_ptoisa_newfeature20260306_TPUSH_TPOP.md`、`tpush_tpop_isa_design_v3.md` | pto-isa 正在设计 TPUSH/TPOP 新指令（已迭代到 v3） |
| `HL_new_feature_Expand_Mixed_Kernel_and_call_spmd.md` | pypto 正在扩展混合 kernel 与 SPMD 调用 |
| `simpler_distributed_runtime_design.md`、`runtime_async.md`、`multi_level_runtime_ring_and_pypto_free_api.md` | simpler 的**分布式化**与异步化是当前主攻方向 |
| `pypto_serving_design goal.md`、`pypto_serving_implementation_plan.md`、`pypto_serving_reference_sglang_vllm.md`、`UBL128_serving.md` | serving 层的设计目标、实施计划、竞品参考 |
| `tensor_layout.md`、`tensor_valid_shape.md`、`sharded_tensor.md` | Tensor 抽象的核心设计（含分片，指向分布式） |
| `machine_hierarchy_and_function_hierarchy.md` | 机器层级与函数层级的映射模型——这是整个 tile 抽象的理论根基 |
| `linqu_data_system.md`、`linqu_runtime_design.md` | **"linqu"（临朐？）是一个未在任何仓库中出现的代号** |
| `Gemini_conversation.md` | 一份 Gemini 对话记录被直接提交进仓库——再次印证 AI 辅助设计流程 |

**关键洞察**：
- **想预测这个技术栈未来 6 个月做什么，读这个仓库比读任何代码仓库都有效。** 它是设计先行于实现的证据。
- `linqu_*` 两份文档指向一个**尚未开源的组件代号**——与 PTOAS 中的 `PTO-Gym`、README 中提到的 `PTODSL`/`CuTile` 一起，构成"冰山水下部分"的四条线索。
- 唯一一个不带 LICENSE、无 description 的仓库——**运营上的孤儿**，但内容价值最高。这种"最有价值的仓库最没人打理"的现象，是内部团队直接把工作目录开源的典型特征。

---

## 四、仓库关系图谱

### 4.1 硬依赖（代码级，Git submodule）

```
pypto              ──submodule──▶  simpler  (挂载为 runtime/)
pypto-serving      ──submodule──▶  pypto-lib
pypto-serving      ──submodule──▶  huawei-csl/HiCR, huawei-csl/TaskR  (组织外)
PTOAS              ──submodule──▶  PTO-ISA/PTO-Gym  (私有，SSH，不可访问)
```

### 4.2 软依赖（构建/运行时，版本 pin 在 CI）

```
pypto-lib   ──▶ pypto (编程框架) + simpler (运行时) + PTOAS (编译) + pto-isa (指令)
pypto       ──▶ PTOAS (codegen 后端) + pto-isa (指令目标)
PTOAS       ──▶ pto-isa (Lowering 终点：生成调用 pto-isa C++ 库的代码)
simpler     ──▶ pto-isa (README: "PTO ISA headers 首次运行时自动 clone")
```

### 4.3 三种关系类型

| 类型 | 实例 | 说明 |
|---|---|---|
| **纵向依赖（栈式）** | serving → lib → pypto → PTOAS → pto-isa | 严格的上下游编译链 |
| **横切穿透** | simpler 被 pypto / pypto-lib / pypto-serving 三层同时直接调用 | 运行时不遵守分层，被各层直接引用 |
| **元数据/治理** | pypto_top_level_documents 覆盖所有仓库；`.github` 提供组织模板 | 无代码依赖，但定义了所有仓库的演进方向 |

### 4.4 依赖密度排名

| 仓库 | 被依赖次数 | 角色 |
|---|---|---|
| `pto-isa` | 4（pypto, PTOAS, simpler, pypto-lib） | **地基**——改动它，全栈都要动 |
| `simpler` | 4（pypto, pypto-lib, pypto-serving, + 独立可用） | **横切运行时** |
| `pypto` | 2（pypto-lib, pypto-serving 间接） | **中枢** |
| `PTOAS` | 2（pypto, pypto-lib） | 编译后端 |
| `pypto-lib` | 1（pypto-serving） | 算子库 |
| `pypto-serving` | 0 | 栈顶 |

---

## 五、活跃度与健康度观察

| 仓库 | Stars | Forks | Fork/Star | Open Issues | Open PRs | 最近推送 |
|---|---|---|---|---|---|---|
| pypto | 72 | 68 | 0.94 | 41–51 | 11–19 | 2026-05-20 |
| pto-isa | 47 | 26 | 0.55 | 7 | 5 | 2026-05-20 |
| PTOAS | 26 | 53 | **2.04** | 10 | 34 | 2026-05-20 |
| simpler | 19 | 51 | **2.68** | 10 | 21 | 2026-05-20 |
| pypto-lib | 8 | 30 | **3.75** | 8 | 10 | 2026-05-20 |
| pypto-serving | 1 | 4 | 4.00 | 2 | 2 | 2026-05-19 |
| pypto_top_level_documents | 4 | 4 | 1.00 | 0 | 0 | 2026-04-28 |

**三点解读**：

1. **Fork/Star 比值普遍 > 1，越靠近底层越极端**（pypto-lib 高达 3.75）。正常开源项目该比值通常在 0.1–0.3。**Fork 远超 Star 意味着"来干活的人远多于来围观的人"**——这些 fork 绝大多数是内部/受邀开发者为提 PR 而创建的。这不是一个社区驱动的项目，而是**一个用开源工作流运作的内部工程组织**。pypto 是唯一比值接近 1 的仓库，说明它确实吸引了一部分真实的外部关注。

2. **五个核心仓库在 2026-05-20 同一天推送**——高度同步的发布节奏，几乎可以确定存在跨仓库的联合版本/联调机制（pypto-lib 的 README 也确认"pinned versions live in CI"）。

3. **pypto 的 issue 数（41–51）显著高于其他仓库之和**——它是用户真正接触的界面，问题都集中在这里。这也侧面说明框架的易用性仍是主要痛点。

---

## 六、战略解读与洞察

### 洞察 1：这是华为对 CUDA 编程生态的完整回应

把这 7 个仓库映射到 NVIDIA 生态，对应关系一目了然：

| hw-native-sys | NVIDIA 生态对应物 |
|---|---|
| `pto-isa` | PTX / CuTe（tile 抽象层） |
| `PTOAS` | NVCC 后端 / MLIR-based 编译器 |
| `pypto` | **Triton / TileLang**（核心对标物） |
| `simpler` | CUDA Runtime + Graph |
| `pypto-lib` | CUTLASS + FlashAttention 等 kernel 库 |
| `pypto-serving` | **vLLM / SGLang**（README 文档已明确对标） |

**这不是补齐某个零件，而是复刻整条链路。** 且许可证（CANN OSL 2.0，仅限华为 AI 处理器）明确表明：开源是为了建生态，不是为了通用。

### 洞察 2：抽象分层是它最核心的产品差异化

pypto 的"三类开发者三个抽象层"（算法用 Tensor / 调优专家用 Tile / 系统开发者用 Block+指令）是一个相当清醒的产品判断——**Triton 的主要痛点正是"简单的很简单，难的做不到"**，因为它只暴露一层抽象。PTO 栈选择把 Tile 和 Block 层也开放出来，用复杂度换调优天花板。这个赌注是否成立，要看 pypto-lib 里 DeepSeek V4 那类 kernel 最终能否打到硬件峰值。

### 洞察 3：全栈"无硬件可用"是一个被反复强化的设计原则

- `pto-isa` → CPU Simulator
- `simpler` → `a2a3sim` / `a5sim`（纯线程模拟，只要 gcc）
- `pypto-lib` → 每个 example 都接受 `-p a2a3sim`
- `pypto` → hello_world 可直接跑

**没有昇腾卡也能完整跑通从写 kernel 到验证的全流程。** 这是整个栈里最"懂开源"的一个决策——它直接消除了生态冷启动的最大障碍。

### 洞察 4：水面下至少还有 4 个未开源组件

从依赖和文档中泄漏出的、在本组织中不存在的名字：

| 代号 | 线索来源 | 推测 |
|---|---|---|
| **PTODSL** | PTOAS README（列为 PTOAS 的前端之一） | 另一种 PTO 前端语言 |
| **CuTile** | PTOAS README（列为 PTOAS 的前端之一） | 对标 CUTLASS/CuTe 的 tile 前端 |
| **PTO-Gym** | PTOAS `.gitmodules`（`git@github.com:PTO-ISA/PTO-Gym`） | 疑似性能调优/搜索环境；同时暴露出**另一个组织 `PTO-ISA`** |
| **linqu** | 顶层文档 `linqu_data_system.md` / `linqu_runtime_design.md` | 数据系统 + 运行时，代号未在任何代码中出现 |

追踪这四条线索，是理解该技术栈全貌的下一步。

### 洞察 5：GitHub 是镜像，不是主战场

- pypto 的描述是 "A **community-driven** pypto implementation"
- pto-isa 引用 PyPTO 时指向 **gitcode.com/cann/pypto**，而非 GitHub
- 许可证是 CANN OSL——CANN 是华为昇腾的官方软件栈品牌

**结论：真正的开发主干在华为内部 / gitcode 的 CANN 项目下，GitHub 这个组织是面向国际开发者的镜像与共建入口。** 这解释了为什么命名混乱（`simpler` vs "PTO Runtime"）、文档仓库无人打理、Fork/Star 比值异常——它是一个"外派"的工程前哨，而非项目本体。

---

## 七、给不同读者的行动建议

**如果你想快速理解这个栈**：
读 `pypto-lib/README.md`（唯一讲清全局依赖的文档）→ `pypto/README.md`（框架哲学）→ `pto-isa/README.md`（地基语义）。跳过 `simpler` 的名字带来的困惑。

**如果你想评估技术水平**：
看 `pypto-lib/models/deepseek/v4/` 和 `pypto-lib/docs/performance-tuning.md`。能把 DeepSeek V4 用自研 tile 框架写出来并做 PMU 级调优，说明栈是通的。

**如果你想预测未来路线**：
只读 `pypto_top_level_documents`。TPUSH/TPOP 指令、simpler 分布式化、serving 对标 vLLM——三条主线都写在那里。

**如果你想参与贡献**：
从 `pto-isa`（社区化程度最高、文档最全、有 CPU 模拟器）或 `pypto`（issue 最多、最需要人）切入。`pypto-serving` 虽然最缺人，但架构未定型，早期参与风险高。

---

*报告基于 2026-07-14 的公开仓库快照生成。各仓库最近一次推送为 2026-05-20，数据可能已有变动。*
