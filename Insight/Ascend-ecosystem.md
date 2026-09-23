# 昇腾 AI 软件生态的结构性错位：从全栈闭环到开放硬件后端

## 摘要

昇腾 AI 软件生态面临的核心问题，并非软件能力不足，也不能简单归结为“MindSpore 生态不如 PyTorch”。更准确地说，**昇腾早期软件体系的能力供给边界，与大模型时代开发者的软件资产、技术创新中心和生态治理方式之间存在结构性错位。**

这一错位集中表现为：Ascend 的核心商业价值最终落在算力硬件，但早期软件战略在 Framework、训练套件和推理引擎层建立了较完整的 Huawei-native 技术体系；与此同时，互联网大模型公司的主要软件资产却长期沉淀在 PyTorch、Megatron、Hugging Face、vLLM、SGLang、Triton 等开放生态中。用户若为了采用一种新硬件而同步迁移模型 Framework、训练框架、推理引擎乃至开发范式，其迁移成本往往会侵蚀硬件替代带来的收益。

因此，昇腾真正需要解决的并不是“如何让 Mind 系列取代现有开源生态”，而是：

> **如何让 Ascend 在尽可能不改变用户既有 AI 软件资产的前提下，成为 PyTorch、Megatron、vLLM、Triton 等生态中的一等硬件后端，同时将昇腾独有的图编译、通信、算子、内存与硬件优化能力以可插拔方式向上提供。**

从 2025—2026 年的软件路线看，这一调整已经明显发生。华为明确提出“分层解耦、全面开源开放”，强调为了匹配开发者习惯，与 PyTorch、Triton、vLLM、verl 等社区深度协作；同时，华为公开表示其 AI 商业化重点聚焦硬件。CANN 正逐渐承担开放硬件平台的核心软件底座角色，而 MindSpeed、MindIE 等产品则越来越倾向于成为兼容外部生态的加速组件。

---

# 1. “生态错位”究竟指什么

本文所说的“生态错位”，并不等于产品能力弱，也不意味着 Mind 系列没有使用价值。它描述的是：

> **软件产品所占据的技术层级、控制边界和开发范式，与目标用户真正希望保留的软件资产之间没有完全对齐。**

对于 AI Accelerator，一套软件体系通常可以分成四类价值层：

| 层级                              | 用户真正关心的问题                      | 昇腾对应能力                                           |
| ------------------------------- | ------------------------------ | ------------------------------------------------ |
| L4 模型与业务生态                      | 我的模型、RL、Serving、Agent 系统能否继续使用 | PyTorch / HF / Megatron / vLLM / SGLang / verl 等 |
| L3 Framework / Enablement       | 如何训练、推理、调度和部署                  | MindSpore / MindSpeed / MindIE                   |
| L2 Hardware Adaptation          | 如何让既有 Framework 使用 NPU         | torch_npu / TorchAir / vLLM-Ascend 等             |
| L1 Hardware Software Foundation | 如何编译、通信、执行、开发 Kernel           | CANN / GE / HCCL / Ascend C / Runtime            |
| L0 Hardware                     | 最终提供算力、内存和互联                   | Ascend NPU / SuperPoD                            |

从硬件平台战略看，越靠近 L1/L2，Ascend 越具有不可替代性：GE、HCCL、Runtime、Ascend C、Kernel 和 NPU 架构必须理解昇腾硬件。

但越向 L3/L4 上移，问题开始发生变化。用户在那里已经拥有大量与具体硬件无关的软件资产。

这正是生态错位产生的根源：

> **Ascend 最不可替代的价值位于软件栈下层，而早期 Mind 体系却试图在软件栈上层建立完整闭环。**

---

# 2. 第一重错位：硬件价值捕获与软件入口之间的错位

这一点是理解整个问题最重要的起点。

华为在 2025 年公开明确表示，AI 的商业化战略聚焦于**硬件变现**；与此同时，宣布 CANN 和 Mind 系列全面开源开放。

这意味着 Ascend 商业模型的核心可以抽象为：

```text
更多模型 / Framework / 应用
          ↓
更多工作负载进入 Ascend
          ↓
更高 Ascend 算力使用量
          ↓
Hardware Value Capture
```

如果这个逻辑成立，那么软件最重要的战略任务其实不是形成新的软件平台税，而是：

> **降低 workload 进入 Ascend 的摩擦。**

由此可以看出早期 full-stack 思路中的内在张力。

如果采用 Ascend 同时意味着：

```text
换 Hardware
   +
换 Framework
   +
换 Training Stack
   +
换 Serving Stack
   +
重新建立 Debug / Profiling / Operator workflow
```

那么软件实际上成为了硬件采用的额外门槛。

这形成一种典型的**Complement Friction（互补品摩擦）**：

硬件越希望扩大市场，围绕硬件的上层软件就越应该具有兼容性和可组合性，而不是要求用户同步迁移。

因此，从产业战略角度看：

> **对于以硬件价值捕获为目标的 Accelerator 厂商，最优的软件战略通常不是最大化自有上层软件的占有率，而是最小化任何上层软件进入硬件的成本。**

这也是“分层解耦”对于 Ascend 的意义远大于普通的软件架构重构的原因。

---

# 3. 第二重错位：Framework 中心与用户资产中心的错位——MindSpore 是最典型案例

MindSpore 的原始产品逻辑具有很强的完整性。

官方将 MindSpore 定义为覆盖 Cloud、Edge、Device 的“全场景深度学习框架”，强调易开发、高性能、统一部署，并天然亲和 Ascend。其技术目标并不只是提供一个 Python API，而是建立从编程模型、自动并行、图编译、MindIR 到端边云部署的一套完整 Framework。

从技术设计角度，这种完整性本身具有价值。

问题出现在用户环境发生变化之后。

互联网大模型开发的现实入口通常不是：

```text
选择 Accelerator
      ↓
选择它推荐的 AI Framework
      ↓
开发模型
```

而更接近：

```text
已有 PyTorch Model
       +
已有 HuggingFace ecosystem
       +
已有 Megatron / FSDP / DeepSpeed
       +
已有 RL framework
       +
已有 Serving stack
       ↓
选择可以运行这些 workload 的 Accelerator
```

因此，决策顺序实际上发生了倒置。

### 3.1 MindSpore 面对的问题不是 API 是否足够好

如果已有模型是在 PyTorch 中开发，那么迁移到另外一个 Framework 并不只是“替换几个 API”。

MindSpore 自身长期提供 PyTorch → MindSpore 的 API 映射、MindConverter 和迁移指南；官方文档甚至需要专门解释二者 API、Padding、Tensor 行为等差异。这个事实本身说明 Framework migration 是一个独立工程问题。

一个大型模型系统实际携带的是：

```text
Model Source
   │
   ├─ Custom Operators
   ├─ Distributed Strategy
   ├─ Checkpoint
   ├─ Precision Policy
   ├─ RL / Finetuning
   ├─ Profiling
   ├─ Debugging
   ├─ Evaluation
   ├─ Serving
   └─ Internal Infrastructure
```

因此，对于已有成熟 PyTorch 基础设施的团队：

**Framework Replacement Cost >> Hardware Adaptation Cost**

这就是 MindSpore 与互联网大模型生态最主要的结构性错位。

---

## 3.2 MindSpore 当前实际上也在主动修正这一问题

值得注意的是，这并不是一个静态状态。

2025 年华为技术资料已经明确把 MindSpore 的方向描述为：

> “南向亲和昇腾，北向生态兼容”。

官方同时表示 MindSpore 正通过对齐主流大模型 API，支持 Megatron、Hugging Face、vLLM、SGLang、OpenCompass 等组件。

这说明 MindSpore 的战略方向已经开始从：

```text
建立独立 Framework Ecosystem
```

逐渐增加：

```text
兼容 Existing AI Ecosystem
```

但这里仍然存在一个长期结构性问题：

**兼容 PyTorch 生态，与成为 PyTorch 生态本身并不是同一件事。**

因此，对于互联网大模型市场，MindSpore 更可能形成以下优势场景：

* 对 Framework 选择权要求较低的新建 AI 系统；
* 华为自身及深度 Ascend-native 场景；
* AI for Science；
* 自动并行和超大规模全栈优化；
* 端、边、云一致部署；
* 需要统一 Framework 控制整个生命周期的行业方案。

它未必需要承担“成为互联网 AI 唯一主 Framework”的战略任务。

---

# 4. 第三重错位：垂直完整产品与水平可组合生态之间的错位

大模型时代的软件生态越来越表现为“Composable Stack”。

典型训练体系可能是：

```text
Transformers
     ↓
PyTorch
     ↓
Megatron Core
     ↓
FSDP / Distributed
     ↓
Accelerator Backend
```

推理体系则可能是：

```text
Model
  ↓
vLLM / SGLang
  ↓
Scheduling / KV Cache / PD
  ↓
Hardware Plugin
  ↓
Accelerator
```

其中任何一层都可能独立替换。

而传统全栈软件产品更倾向于：

```text
Model
 ↓
Vendor Framework
 ↓
Vendor Distributed Stack
 ↓
Vendor Runtime
 ↓
Vendor Hardware
```

前者是**Horizontal Ecosystem**，后者是**Vertical Stack**。

Mind 系列最初的产品完整性，在这种变化下反而容易导致边界重叠。

---

# 5. MindSpeed：Mind 系列中错位最小的产品

MindSpeed 是一个非常重要的反例。

它实际上代表了 Ascend 更符合当前生态规律的软件形态。

官方并没有要求用户放弃 Megatron，而是明确把 MindSpeed Core 定义为面向 Ascend 的大模型加速库，通过 `megatron_adapter` 与 Megatron-LM 集成。官方 Quick Start 甚至只要求在 Megatron 代码中增加：

`import mindspeed.megatron_adaptor`

即可开始运行，并进一步开启 Ascend 特有的通信、内存、融合算子和并行优化。

它对应的架构不是：

```text
Megatron
   ×
MindSpeed
```

而是：

```text
          Megatron
              │
      MindSpeed Adapter
              │
 ┌────────────┼────────────┐
 │            │            │
Parallel   Communication  Kernel
Optimization Optimization Optimization
 │            │            │
 └────────────┼────────────┘
              ↓
          torch_npu
              ↓
             CANN
              ↓
           Ascend
```

这意味着 MindSpeed 的核心价值并非重新定义训练 Framework，而是：

> **把 Ascend 的 hardware-specific optimization 注入已有训练生态。**

这与硬件公司的价值边界高度一致。

---

## 5.1 真实互联网项目也正在形成这种集成方式

Alibaba ROLL 对 Ascend 的支持非常具有代表性。

其 Ascend RFC 的目标并不是建设一个 ROLL-Mind 全栈，而是建立统一的 device abstraction：

* 推理采用 vLLM + vLLM-Ascend；
* 训练接入 MindSpeed；
* 同时保持原 CUDA 架构兼容。

2026 年后续 MindSpeed/Megatron 集成 RFC 更明确规定：

* MindSpeed 为 **optional**；
* 不改变 CUDA/CPU 行为；
* 不把 MindSpeed 变成 hard dependency；
* 不改变 Megatron 架构。

这实际上非常准确地揭示了互联网软件生态希望如何“消费”Ascend：

> **Hardware-specific capability 应该进入既有 Framework，而不是要求既有 Framework 进入 Hardware Vendor 的世界。**

从生态战略角度看，MindSpeed 很可能比 MindSpore 更接近 Ascend 面向互联网大模型训练的理想产品形态。

---

# 6. 第四重错位：MindIE 与推理生态控制面的重叠

推理领域的错位更加明显，因为过去几年推理 Framework 的创新速度非常快。

vLLM、SGLang 等系统已经不只是简单的“模型执行器”，而开始承担：

* Continuous batching；
* KV Cache 管理；
* Prefix Cache；
* Speculative decoding；
* PD disaggregation；
* Expert Parallel；
* Scheduler；
* Distributed serving；
* API serving。

于是，如果硬件厂商同时提供一个完整推理 Engine，就会发生功能边界重叠：

```text
                vLLM
                  │
       Scheduler / KV / PD / EP
                  │
                  ?
                  │
               MindIE
                  │
       Scheduler / KV / PD / EP
                  │
                 CANN
```

用户很自然会问：

> 到底谁负责 Scheduling？谁负责 KV Cache？谁负责 PD？谁决定 Parallelism？

这并不完全是功能问题，而是**Control Plane Ownership（控制面归属）问题**。

---

# 7. MindIE 当前最重要的变化：从 Engine 变成 Components

MindIE 当前的产品演进已经明显开始解决这一问题。

官方现在将能力拆解为：

* **MindIE Motor**：服务化、请求路由、集群调度、PD 等；
* **MindIE LLM**：Ascend-native 文本生成；
* **MindIE Turbo**：面向各种推理引擎的通用硬件加速套件；
* 同时支持 vLLM、SGLang 等第三方引擎直接运行。

尤其重要的是，MindIE Turbo 被明确描述为可以叠加在第三方推理框架之上，而且属于**非必选**能力。

因此，一个更符合现实的关系应该是：

```text
                  Application
                       │
                Serving / Control
             ┌─────────┴─────────┐
             │                   │
       MindIE Motor        other platform
             │
     ┌───────┴────────┐
     │                │
MindIE LLM       vLLM / SGLang
                      │
                 vLLM-Ascend
                      │
                MindIE Turbo
                  (optional)
                      │
                    CANN
                      │
                   Ascend
```

这与最初“MindIE 是 Ascend 推理入口”的产品理解已经有明显不同。

它正在向：

> **Serving capability + Optional hardware optimization library**

转型。

这一变化非常重要，因为后者比前者更符合开放推理生态。

---

# 8. vLLM-Ascend：目前最典型的正确生态接口

vLLM-Ascend 是观察 Ascend 软件生态变化最有价值的案例之一。

它已经进入 `vllm-project` 官方组织，并明确采用 Hardware Plugin 架构。

官方描述直接强调：

> Ascend support 通过 hardware-pluggable interface 与 vLLM 解耦。

用户因此可以保持：

```text
Application
    ↓
  vLLM
    ↓
Hardware Plugin
 ┌──────┴──────┐
CUDA        Ascend
             │
        vLLM-Ascend
```

而不需要形成：

```text
Application
    ↓
Vendor-specific inference framework
    ↓
Ascend
```

vLLM-Ascend 当前还保持与上游 vLLM 版本对应，并持续通过 Ascend CI 跟踪 upstream。

这里存在一个非常重要的生态原则：

> **最理想的硬件兼容方式，并不是把上游生态复制一份，而是让硬件能力进入 upstream 定义的 extension point。**

这能够减少 fork、降低 API divergence，同时把模型和 Framework 的创新速度交还给原社区。

---

# 9. 第五重错位：AI 创新速度与 Vendor Release Train 之间的错位

大模型软件现在存在一个非常明显的结构性特征：

**上游变化速度远高于传统硬件软件版本周期。**

例如：

```text
Model architecture
     ↓
Megatron / HF
     ↓
PyTorch
     ↓
vLLM / SGLang
     ↓
New Operator / Parallel Strategy
```

可能在几周至数月内持续变化。

但异构硬件软件通常还存在：

```text
CANN version
     ↕
torch_npu version
     ↕
PyTorch version
     ↕
MindSpeed version
     ↕
Megatron version
```

MindSpeed 官方本身就维护了明确的版本配套矩阵。例如某版本同时绑定特定 MCore、PyTorch、TorchNPU 与 CANN 版本。

vLLM-Ascend 同样要求 vLLM 与插件保持对应版本。

这种版本关系并不是 Ascend 独有的问题，而是异构后端必然面对的工程成本。

但如果 Hardware Vendor 同时掌握更多上层 Framework，这种成本就会被进一步放大：

```text
上游模型变化
       ↓
Framework变化
       ↓
Vendor Framework适配
       ↓
Training/Serving适配
       ↓
Compiler适配
       ↓
Operator适配
       ↓
Hardware
```

因此，在创新高速变化的领域：

> **Vendor-specific layer 越厚，追赶 upstream innovation 的同步成本通常越高。**

这也是为什么硬件生态最终往往趋向“thin compatibility layer + powerful hardware backend”。

---

# 10. 第六重错位：真正具有差异化的能力在底层，上层却承担了过多品牌与产品复杂度

从 Ascend 软件能力本身看，真正不可被通用开源项目替代的核心能力主要集中在：

```text
            Ascend-specific
                  │
      ┌───────────┼───────────┐
      │           │           │
   Compiler     Kernel      Communication
     GE       Ascend C        HCCL
      │           │           │
      └───────────┼───────────┘
                  │
               Runtime
                  │
               Hardware
```

CANN 官方把自己的位置定义得非常清楚：

> 向上支持 MindSpore、PyTorch、TensorFlow 等多种 AI Framework，向下服务 Ascend AI Processor。

其内部包含：

* Framework Adaptor；
* Graph Compiler；
* Operator Library；
* Runtime；
* Graph Executor；
* HCCL；
* AscendCL 等。

这其实意味着：

> **CANN 才是 Ascend 软件生态真正不可替代的“腰部”。**

Framework 可以换。

Serving Engine 可以换。

RL Framework 可以换。

Megatron 可以升级。

模型每个月都会变化。

但是无论这些东西怎么变化：

```text
最终都必须变成
        ↓
Ascend 可执行计算
        ↓
通信 / Kernel / Memory / Runtime
```

所以 Ascend 的长期生态优势更可能来自：

> **“让所有上层生态都能高效访问 CANN 和硬件能力”**

而不是：

> “让所有生态统一迁移到某个 Huawei-defined upper stack”。

---

# 11. TorchNPU / TorchAir 的战略意义也应该放在这个框架下理解

TorchNPU 本身并不是简单的兼容插件。

官方已经把 Ascend Extension for PyTorch 作为完整 PyTorch 开发入口，并在其中直接提供：

* PyTorch 原生 API；
* MindSpeed；
* TorchAir；
* 第三方库；
* 自定义算子；
* 性能优化与迁移工具。

TorchAir 则继承 PyTorch Dynamo，将 FX Graph 转换成 GE Graph，并负责 GE 图在 Ascend 上的编译执行。

因此，它实际解决的是：

```text
PyTorch semantic world
          │
       FX Graph
          │
       TorchAir
          │
       GE Graph
          │
         CANN
          │
        Ascend
```

它的战略价值不是再建立一个 Framework，而是建立：

> **Semantic Bridge。**

这类 Bridge Layer 对 Ascend 生态的重要性很可能高于继续扩张新的上层产品。

---

# 12. 从整个体系看，Mind 系列的“错位程度”并不相同

| 产品               | 原始价值定位                        | 与主流生态的重叠程度 | 当前错位程度 | 当前演化方向                          |
| ---------------- | ----------------------------- | ---------: | -----: | ------------------------------- |
| **MindSpore**    | 完整 AI Framework               |         很高 |  **高** | 北向兼容、保留 Ascend-native / 全场景优势   |
| **MindSpeed**    | 分布式训练加速                       |         较低 |  **低** | Megatron-compatible accelerator |
| **MindIE LLM**   | 完整推理执行体系                      |         较高 | **中高** | 与第三方引擎共存                        |
| **MindIE Motor** | Serving / Cluster control     |          中 |  **中** | 开放服务化和调度平台                      |
| **MindIE Turbo** | Hardware acceleration library |         很低 |  **低** | 可插拔优化组件                         |
| **CANN**         | Hardware software foundation  |         极低 | **最低** | 开放硬件平台核心                        |
| **torch_npu**    | PyTorch hardware adapter      |         极低 | **最低** | 主流 Framework Bridge             |
| **TorchAir**     | PyTorch → GE compiler bridge  |         极低 | **最低** | Compiler Bridge                 |
| **vLLM-Ascend**  | vLLM hardware plugin          |         极低 | **最低** | Upstream-compatible backend     |

这张表揭示了一个非常明显的规律：

> **产品越要求用户接受一种新的上层开发范式，生态错位越严重；产品越接近 Hardware Adapter / Accelerator / Compiler Backend，越符合互联网 AI 的软件消费方式。**

---

# 13. 因此，Mind 系列真正需要解决的不是“生态规模”，而是“生态位置”

很多讨论会把问题表述成：

> MindSpore 开发者够不够多？

或者：

> MindIE 模型支持够不够多？

这些问题当然重要，但还不是根本问题。

真正的问题是：

> **这个组件位于正确的软件抽象层吗？**

例如，即使投入大量资源让 MindIE 支持所有模型，如果用户已经决定使用 vLLM，那么“MindIE 能不能支持这个模型”并不是用户最关心的问题。

用户真正关心的是：

> vLLM 能不能在 Ascend 上运行？

同样，如果用户已经决定使用 Megatron：

问题就不是：

> MindSpeed 能不能重新提供一套训练 Framework？

而是：

> MindSpeed 能不能让 Megatron 在 Ascend 上获得最优性能？

因此：

### Ecosystem Coverage ≠ Ecosystem Alignment

覆盖更多模型、增加更多功能，只解决 Coverage。

真正决定采用的是 Alignment：

```text
用户已有工作流
       │
       │ 尽量不改变
       ↓
Compatibility Layer
       ↓
Ascend Optimization
       ↓
CANN
       ↓
Hardware
```

---

# 14. 昇腾当前正在经历的，本质上是一次“软件价值下沉”

如果把 Ascend 软件战略粗略分成两个阶段，可以得到一个非常清晰的变化。

## 阶段一：Vertical Full Stack

```text
Application
    ↓
MindSpore
    ↓
MindSpeed / MindIE
    ↓
CANN
    ↓
Ascend
```

优势是：

* 华为拥有完整优化控制权；
* 可以实现软硬件联合优化；
* 易形成端到端解决方案；
* 适合高度集成的行业场景。

但代价是用户需要进入新的 Software Stack。

---

## 阶段二：Open Hardware Platform

```text
             User ecosystem
                    │
    ┌───────────────┼───────────────┐
    │               │               │
 PyTorch         Megatron         vLLM
    │               │               │
torch_npu       MindSpeed      vLLM-Ascend
    │               │               │
TorchAir       Ascend Opt.     MindIE Turbo
    └───────────────┼───────────────┘
                    │
                   CANN
                    │
                  Ascend
```

Mind 系列不再必须占据主干路径。

它可以成为：

> **按需插入的 Ascend Optimization Kit。**

这就是“分层解耦”的真正产业意义。

华为官方在 2025 年已经明确提出通过插件化机制提供细粒度、原子化能力，并通过稳定接口兼容硬件演进、保护客户核心资产。

从生态战略看，这实际上是在从：

**Stack Ownership**

转向：

**Capability Provisioning**

---

# 15. 一个更合理的 Ascend 软件生态目标模型

长期看，更合理的架构可能是：

```text
┌─────────────────────────────────────────────┐
│       Open AI Innovation Ecosystem          │
│                                             │
│ PyTorch │ HF │ Megatron │ vLLM │ SGLang   │
│ verl │ Triton │ TileLang │ other projects │
└─────────────────────┬───────────────────────┘
                      │
              Stable Extension Point
                      │
┌─────────────────────▼───────────────────────┐
│            Ascend Adaptation Layer          │
│                                             │
│ torch_npu │ vLLM-Ascend │ MindSpeed        │
│ TorchAir │ optional Mind components        │
└─────────────────────┬───────────────────────┘
                      │
┌─────────────────────▼───────────────────────┐
│       Ascend Differentiation Layer          │
│                                             │
│ GE │ HCCL │ Ascend C │ Kernel │ Runtime    │
│ Memory │ Compiler │ Communication          │
└─────────────────────┬───────────────────────┘
                      │
┌─────────────────────▼───────────────────────┐
│              Ascend Hardware                │
│          NPU / SuperPoD / UnifiedBus        │
└─────────────────────────────────────────────┘
```

这个模型里有一个重要原则：

### 上层追求兼容性，下层追求差异化。

越靠近模型：

> 越应该兼容行业标准。

越靠近硬件：

> 越应该体现 Ascend-specific advantage。

这比“每一层都建立自己的生态”更加可持续。

---

# 16. 对 Mind 系列未来定位的判断

## MindSpore

更合理的长期角色不是承担所有 Ascend workload 的默认入口，而是在能够发挥 full-stack 优势的场景建立明确差异化，例如：

* 超大规模自动并行；
* AI for Science；
* Ascend-native 训练；
* 端边云统一部署；
* 华为内部或行业全栈解决方案。

其竞争逻辑应从：

**Framework Replacement**

转向：

**Differentiated Framework Choice**。

---

## MindSpeed

MindSpeed 很可能是 Mind 系列中最符合互联网生态结构的产品。

它应该继续强化：

> Megatron / PyTorch ecosystem × Ascend optimization。

理想状态甚至是用户只知道：

```text
Megatron works well on Ascend
```

而不需要首先理解 MindSpeed 的所有内部产品边界。

---

## MindIE

MindIE 更合理的发展方向是拆分为：

**Serving Infrastructure + Hardware Optimization Components**

而非继续强化“另一个 vLLM”的认知。

MindIE Turbo 是其中尤其值得关注的一步：

> Ascend-specific optimization 可以存在，但不需要拥有整个 inference framework。

---

# 17. 最深层的问题：Ascend 竞争的对象其实不是 CUDA API，而是 CUDA 的“生态位置”

如果只把 CUDA 理解成一个编程语言或 Kernel API，就很容易把国产 AI 软件生态建设理解成：

```text
CUDA       → Ascend C
cuBLAS     → Ascend library
NCCL       → HCCL
TensorRT   → MindIE
```

然后继续逐层建立对应产品。

但 NVIDIA 真正强大的地方并不只是拥有很多软件。

其更重要的生态结果是：

> **大量第三方 Framework 已经默认把 NVIDIA GPU 当成一个自然执行后端。**

因此 Ascend 最终需要实现的状态也不应该是：

> Ascend 拥有一套与 NVIDIA 数量相同的软件产品。

而应该是：

```text
PyTorch
Megatron
vLLM
SGLang
Triton
verl
各种未来 Framework
        │
        ↓
“Ascend 是一个自然可选的 backend”
```

一旦这一点成立，Mind 系列是否占据用户首屏反而不再重要。

---

# 18. 对昇腾生态战略的最终判断

因此，昇腾 Mind 系列的核心生态问题可以归纳成一句话：

> **历史上的主要错位，是将过多 Ascend 差异化价值封装在用户需要主动进入的上层软件体系中，而大模型时代真正需要的是把这些差异化能力下沉到主流生态可以直接调用的硬件后端。**

这一问题正在被逐步修正。

2025 年之后出现的几条路线实际上指向同一目标：

```text
CANN 分层解耦
        +
全面开源
        +
torch_npu
        +
TorchAir
        +
MindSpeed → Megatron
        +
vLLM-Ascend
        +
MindIE Turbo
        +
Triton / TileLang compatibility
```

它们看起来属于不同产品，背后的战略逻辑却高度一致：

> **降低 Ascend-specific abstraction 对用户已有软件资产的侵入性。**

因此，衡量未来 Ascend 软件生态是否成功，一个比“MindSpore 市占率”更加重要的指标应该是：

### Ascend Native Software Share

并不是最关键的。

真正关键的是：

### Ascend Reachability from Mainstream Ecosystems

即：

> 一个使用 PyTorch、Megatron、vLLM、SGLang、Triton 或未来新 Framework 的开发者，从现有代码迁移到 Ascend，需要改变多少代码、多少系统架构、多少开发习惯，以及在性能优化阶段需要理解多少 Ascend-specific complexity。

**这个距离越短，Ascend 的生态壁垒就越低；而当这条路径同时能够充分释放 Ascend 的硬件差异化能力时，软件生态就开始从“替代生态”转变为真正意义上的“硬件平台生态”。**

这很可能才是 Mind 系列以及整个 Ascend Software Stack 下一阶段最核心的战略命题。

---

# 参考资料索引

**[R1] Huawei，Ascend: Open for All to Build a Vibrant Ecosystem，2025-09。**
华为正式提出 Ascend 架构升级、分层解耦、全面开源开放，并明确与 Triton、PyTorch、vLLM、verl 等社区深度合作。

**[R2] Huawei，Groundbreaking SuperPoD Interconnect: Leading a New Paradigm for AI Infrastructure，2025。**
徐直军公开说明华为 AI 商业化聚焦硬件，并公布 CANN 与 Mind 系列全面开源开放计划。

**[R3] Huawei 2025 Annual Report。**
确认 Mind 系列、CANN 与 AI 训推工具链全面开源开放，并记录华为参与 PyTorch 等主流开源生态。

**[R4] Huawei，《构筑开放基础软件栈，共建昇腾AI算力新生态》。**
系统说明 CANN、MindSpore、MindSpeed、MindIE 的产品结构，并提出插件化、分层解耦、保护客户核心资产等架构方向。

**[R5] Ascend CANN 官方文档，《CANN是什么》。**
定义 CANN 在 Framework 与 Ascend 硬件之间的承上启下位置，以及 Compiler、Runtime、HCCL、Framework Adaptor 等内部能力。

**[R6] MindSpore 官方文档。**
MindSpore 官方定位为全场景深度学习 Framework，支持 Cloud、Edge、Device，并长期提供 PyTorch → MindSpore 的迁移工具与 API 映射。

**[R7] Ascend/MindSpeed 官方仓库及官方 Quick Start。**
说明 MindSpeed Core 与 Megatron-LM 的关系、`megatron_adapter`、版本配套以及 Ascend 专有优化能力。

**[R8] MindIE 官方产品资料。**
说明 MindIE Motor、MindIE LLM、MindIE Turbo 以及 vLLM/SGLang 第三方引擎之间的当前关系，其中 MindIE Turbo 可作为非必选加速能力。

**[R9] vLLM Project，vLLM Ascend Plugin。**
vLLM-Ascend 已进入 vllm-project，采用 Hardware Pluggable 架构，是 vLLM 社区支持 Ascend Backend 的推荐方式。

**[R10] Ascend/TorchAir 官方仓库。**
TorchAir 基于 PyTorch Dynamo，将 FX Graph 转换为 GE 计算图，并提供 Ascend 上的编译与执行能力。

**[R11] Ascend Extension for PyTorch 官方资料。**
说明 torch_npu、MindSpeed、TorchAir、PyTorch API 和第三方库已经被组织为完整 PyTorch-on-Ascend 开发入口。

**[R12] Alibaba ROLL Ascend Support RFC。**
具有代表性的互联网开源项目实践：通过设备抽象保持 CUDA 体系，同时以 vLLM-Ascend 接入推理、MindSpeed 接入训练。

**[R13] Huawei 2024 Annual Report。**
确认 Ascend 已在互联网、电信、金融等行业获得采用，并说明 CANN、MindIE 等软件生态的发展情况。
