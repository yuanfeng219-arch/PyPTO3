# DeepSeek-V3.2 从 GPU 到 Ascend NPU 的推理迁移

## 摘要

大模型迁移通常被描述为“让模型从一种芯片运行到另一种芯片”，但对于 DeepSeek-V3.2 这类已经围绕特定硬件完成深度优化的大规模 MoE 模型，这种描述过于简单。模型权重和模型结构只是推理系统的一部分。真正决定推理效率的，还包括 Attention 与 MoE 的算子实现、Kernel、低精度计算、显存管理、跨卡通信、并行策略、Prefill/Decode 调度以及集群拓扑。

因此，从 NVIDIA GPU 迁移到 Ascend NPU，需要解决两个层次的问题。第一层是**模型语义迁移**：保证 DeepSeek-V3.2 的 Attention、MoE、Router、MLP 等计算在 NPU 上得到正确执行。第二层是**系统效率重建**：将原来围绕 GPU 建立的高性能执行路径，重新映射到 Ascend 的计算单元、内存体系、通信系统和推理软件栈上，并重新验证精度、吞吐、时延和规模化能力。

DeepSeek-V3.2 是一个具有代表性的案例。其官方配置包含 61 层 Transformer，其中 3 层为 Dense Layer，MoE 部分包含 256 个 Routed Expert、1 个 Shared Expert，每个 token 激活 8 个 Routed Expert；与此同时，V3.2 在此前 MLA 和 DeepSeekMoE 架构基础上进一步引入 DeepSeek Sparse Attention（DSA），以降低长上下文 Attention 的计算开销。 这些架构特征直接决定了推理系统需要处理大规模矩阵计算、动态 Expert 路由、跨设备数据交换、KV Cache 管理以及长序列 Attention 等问题。

DeepSeek 已经公开 FlashMLA、DeepGEMM、DeepEP 等基础设施组件，其作用分别覆盖 Attention Kernel、Dense/MoE GEMM 和 Expert Parallel 通信。这说明，一个高性能 DeepSeek 推理系统从来都不只是一份模型权重，而是一套围绕模型特点建立起来的软硬件协同系统。

当前 Ascend 生态已经能够在 vLLM Ascend 上部署 DeepSeek-V3.2。官方文档提供了 A2、A3 平台上的单节点、多节点、Expert Parallel、Prefill-Decode 分离、精度评估和性能评估路径，并提供 DeepSeek-V3.2 W8A8 模型的部署方案。 因此，讨论的重点已经从“Ascend 能否运行 DeepSeek”进一步转向：**DeepSeek 已经在 GPU 上形成的高效执行能力，如何在 Ascend 上得到重新实现和持续优化。**

---

# 背景：模型已经开源，为什么 DeepSeek 仍然有信心？

理解 GPU → NPU 迁移之前，先看一个更直接的问题：DeepSeek 究竟开源了什么？如果第三方已经可以下载模型、购买一台满足部署条件的华为服务器并对外售卖 token，为什么 DeepSeek 仍然有信心开源？

## 从 Hugging Face 下载到的是什么

DeepSeek-V3.2 的官方 Hugging Face 仓库不是一个只有模型名字的展示页，而是一套可下载的模型发布物。仓库页面当前显示总体积约 690 GB，主要包含以下文件和目录：

| 类别 | Hugging Face 中的文件 | 提供的能力 |
| --- | --- | --- |
| 模型权重 | `model-00001-of-000163.safetensors` 至 `model-00163-of-000163.safetensors`，以及 `model.safetensors.index.json` | 保存模型参数及参数到分片文件的索引 |
| 模型结构 | `config.json` | 描述 61 层 Transformer、256 个 Routed Expert、Top-8、DSA/MLA、FP8 等架构和精度配置 |
| 生成配置 | `generation_config.json` | 给出 temperature、top-p、起止 token 等默认生成参数 |
| Tokenizer | `tokenizer.json`、`tokenizer_config.json` | 将文本与模型使用的 token ID 相互转换 |
| 参考推理代码 | `inference/model.py`、`generate.py`、`kernel.py`、`convert.py`、`config_671B_v3.2.json`、`requirements.txt` | 展示模型结构、权重转换、Kernel 调用和生成流程的参考实现 |
| 编码与工具调用 | `encoding/encoding_dsv32.py` 及测试输入输出 | 描述对话、推理与搜索 Agent 场景的消息编码和输出解析 |
| 文档、研究材料与许可 | `README.md`、`LICENSE`、`assets/paper.pdf`、Benchmark 图片和 Olympiad Cases | 解释模型能力、使用方式、技术报告和许可范围 |

这些发布物采用 MIT License。第三方由此获得了下载、修改、部署和商业使用模型的许可基础；但实际对外提供生成式 AI 服务时，仍需自行满足所在地法规、安全、内容治理和运维要求。

需要特别区分：Hugging Face 仓库里的 `inference/` 是模型与参考推理路径的一部分，不是一套已经替第三方调优完成的 Ascend 生产系统。其代码展示了 DeepSeek-V3.2 怎样加载权重和生成 token；在华为服务器上部署仍需要匹配具体硬件的 CANN、TorchNPU、vLLM Ascend 或 MindIE、量化权重格式、算子与通信实现。

## 一台华为服务器能让第三方开始卖 token，但不能复制 DeepSeek 的成本曲线

在硬件容量、模型版本、量化格式和软件版本都满足要求的前提下，第三方确实可以形成这样一条商业链路：

```text
从 Hugging Face 下载权重、配置与 Tokenizer
↓
准备满足容量要求的华为服务器
↓
安装 CANN / TorchNPU / vLLM Ascend 或 MindIE
↓
加载兼容的量化权重并启动推理服务
↓
暴露 OpenAI-compatible API
↓
向客户提供并售卖 token
```

这说明开源真正降低了模型能力的获得门槛。第三方不需要重新训练一个 671B 级 MoE 模型，也不需要猜测模型架构，就可以进入推理服务市场。

但从 Hugging Face 下载的内容，和 DeepSeek 线上系统的完整竞争力并不是同一件事。

| 第三方直接拿到的开源资产 | 没有随下载自动获得的生产能力 |
| --- | --- |
| 模型权重与模型结构 | DeepSeek 的生产集群配置与容量规划 |
| DSA/MLA、MoE、Router 等计算语义 | 针对目标 NPU 调到高利用率的 Kernel |
| Tokenizer、生成配置与编码工具 | 通信拓扑、Expert Placement 与计算通信重叠 |
| 基础推理和权重转换参考代码 | Prefill/Decode 资源配比与持续负载均衡 |
| 使用和修改模型的许可基础 | 真实流量规模、SLO、监控系统与调参经验 |
| 启动一个推理 API 的可能性 | 相同质量、时延和稳定性下的单位 token 成本 |

## DeepSeek 围绕 V2、V3 和 V3.2 的模型结构与真实生产负载，形成了一组以降低推理成本、提高吞吐和降低时延为目标的官方自研优化技术

这些优化技术并不都是 DeepSeek 的闭源或独占能力。FlashMLA、DeepGEMM、DeepEP、EPLB 等项目已经公开，其中有些随模型发布，有些需要从独立仓库获取，有些只公开了系统设计或部分算法。即使源码已经开源，也不意味着面向 CUDA、Tensor Core、NVLink 和 RDMA 的性能可以直接带到昇腾硬件上。

| 官方优化技术栈 | 作用及与 DeepSeek 模型的结合 | 公开与获得边界 | 换成昇腾硬件后，需要重新做什么 |
| --- | --- | --- | --- |
| **MLA / DSA** `Model / Attention` | MLA 通过压缩 KV 表示降低缓存压力；V3.2 的 DSA 先由 Indexer 选择重要 token，再执行稀疏 Attention，进一步降低长上下文计算量。 | 数学结构、模型参数和技术报告公开；模型权重包含这种能力，但不包含某一硬件上的最优执行路径。 | 第三方获得了“算什么”，没有自动获得“怎样最快地算”；Ascend 需要保持 Indexer、稀疏选择和 Attention 语义，并重建 KV Cache 与 Kernel 路径。 |
| **MTP** `Model / Decoding` | Multi-Token Prediction 为一次预测多个后续 token 提供模型基础，可被推理框架用于推测式解码。 | 模型结构和相关参数公开；能否转化为吞吐或时延收益取决于 Serving 实现，而不是加载权重后自动生效。 | Ascend Serving 需要实现草稿 token 生成、验证、调度和回退路径，并在真实接受率与负载下评估收益。 |
| **FlashMLA** `Attention Kernel` | 为 DeepSeek-V3/V3.2 系列的 Dense/Sparse Attention 提供 Prefill、Decode 和 KV Cache 相关高性能 Kernel。 | MIT 开源，但位于独立仓库；当前实现明确面向 NVIDIA SM90/SM100、CUDA 与特定 KV 格式。 | 源码公开不等于 NPU 可直接执行；Ascend 需要按 AI Core、片上内存和数据搬运方式重做 Attention、稀疏访问与 KV Cache Kernel。 |
| **DeepGEMM / Grouped GEMM** `Compute Kernel` | 针对 Dense GEMM、MoE 中 token 数不等的 Expert GEMM，以及 V3.2 Indexer 等 DeepSeek 典型 Shape 组织高性能矩阵计算。 | MIT 开源且独立发布；实现依赖 NVIDIA Tensor Core、CUDA 和具体 Layout，未随 Hugging Face 权重交付。 | Ascend 需要重新实现动态 Expert 分组、Padding/Mask、量化 Scale、Layout 与调度，不能把普通 MatMul 覆盖率等同于 MoE 性能。 |
| **Mega MoE** `Kernel + Communication` | DeepGEMM 当前版本把 EP Dispatch、两层 Expert Linear、SwiGLU 与 Combine 融合或重叠，展示更大的计算—通信融合边界。 | 代码已在 DeepGEMM 中公开；这是该仓库的较新能力，不能仅据此断言它就是 V3.2 线上服务的既有路径。 | Ascend 不仅要逐算子适配，还要重新判断 HCCL、AI Core 和内存系统下值得采用的融合边界；GPU 的 NVLink 融合实现不能直接继承。 |
| **DeepEP** `MoE Communication` | 为 MoE Expert Parallel 的 Dispatch/Combine 提供 All-to-All；分别覆盖偏吞吐的 Prefill 路径和偏低时延的 Decode 路径，并支持计算通信重叠。 | 独立开源；官方实现使用 NVLink、RDMA、CUDA/NVSHMEM 等 GPU 通信机制，不属于模型权重。 | HCCL 支持集合通信并不自动等价于 DeepEP 的负载特化路径；Ascend 仍需分别优化 Prefill/Decode、缓冲区、并发流和通信计算重叠。 |
| **计算—通信重叠** `Runtime` | DeepSeek 的 Prefill 与 Decode 都使用微批次把 MoE All-to-All 尽量隐藏在计算之后，但两阶段占用计算资源和等待通信的方式不同。 | 官方公开了 Profile 数据和部分实现机制，但线上调度参数、拓扑状态与全部运行时控制没有作为单一生产软件包交付。 | 第三方必须在自己的拓扑和负载下重新测量可隐藏比例；Ascend 需要结合 HCCL、Stream/Event、AI Core 占用和内存压力重新排程。 |
| **EPLB 与 Expert Placement** `Distributed Scheduling` | 根据 Expert Load 复制热点 Expert，并结合 Group-limited Routing 把 Expert 放置到合适节点；官方还提供不同规模下的分层与全局均衡策略。 | Placement 算法已开源；官方明确说明 Expert Load 的准确预测不在该仓库范围内。 | 第三方仍需从真实请求中获得负载统计并决定更新周期；Ascend 侧必须结合 Rank、节点拓扑、内存容量和 HCCL 代价重新求解 Placement。 |
| **在线推理多层负载均衡** `Serving` | 官方生产系统分别平衡 Prefill 的 Attention/输入 token、Decode 的 KV Cache/请求数，以及 EP 的 Expert 接收负载。 | 系统设计、部分部署规模和统计数据公开；完整 Request Router、监控、容量系统与生产配置没有作为可直接复制的软件包开源。 | 启动 API 只代表服务可用；第三方还需建立请求路由、KV Cache 感知调度、PD 资源配比、故障恢复和持续容量治理。 |
| **分阶段混合精度** `Precision` | 官方 V3/R1 生产推理中，矩阵乘和 Dispatch 使用 FP8，核心 MLA 与 Combine 使用 BF16，在吞吐、通信量和数值稳定性之间分层取舍。 | 策略公开，但模型 dtype 或权重格式不会自动生成完整的运行时 Tensor 精度图、Scale 处理和精度例外清单。 | Ascend 迁移不能只寻找一个“FP8 对应类型”，而要逐 Tensor、算子和通信阶段建立精度映射，并用 GPU Reference 验证误差。 |
| **3FS** `Storage / Cluster` | 为训练数据、Checkpoint 和推理 KV Cache Lookup 提供基于 SSD 与 RDMA 的共享存储，说明推理效率资产已经延伸到集群存储层。 | 3FS 已独立开源；它不是单机启动模型的必要文件，也不等于 DeepSeek 全部生产存储配置。 | 小规模部署未必需要复刻 3FS；大规模 Ascend Serving 则要重新评估远端 KV Cache 的命中收益、网络开销、一致性和故障边界。 |
| **TileKernels / TileLang 路线** `Kernel DSL` | 用 Tile 级 DSL 表达 MoE Routing、量化和其他 LLM Kernel；官方说明其中部分 Kernel 已用于内部训练和推理场景。 | TileKernels 已开源，但当前要求仍面向 NVIDIA SM90/SM100 与 CUDA；DSL 表达比手写 CUDA 更高层，不代表已有 Ascend 后端。 | 可迁移的是算法、Tile 分解和数据流知识；Ascend 仍需编译后端、硬件映射、算子验证和性能调优，不能把 DSL 直接等同于迁移完成。 |

这张表揭示了比“开源 / 未开源”更重要的三层边界：**随模型获得、从独立项目获得、在生产系统中重新形成。**第三方可以研究甚至修改已经开源的优化组件，但要在不同硬件上复现成本，仍需完成从 GPU 依赖到 NPU 实现、从公开算法到真实负载参数、从单项 Benchmark 到端到端 Serving 的三次转换。

DeepSeek 的信心并不是“别人无法部署”，而是**别人能够复制模型能力，不等于能够立即复制 DeepSeek 的成本曲线**。开源会增加推理服务竞争，但竞争单位已经从一份权重扩展为整个系统：

```text
Model
→ Kernel
→ Communication
→ Serving
→ Cluster
→ 持续的流量反馈与工程迭代
```

模型层面的 MoE 和 MLA 让每个 token 理论上少算、少占 KV Cache；DeepGEMM 与 FlashMLA 把这种理论优势转化为 GPU 上的 Kernel 效率；DeepEP 避免 MoE 的节省被跨卡通信吃掉；Prefill/Decode 分离、Expert Parallel 和多层负载均衡又决定整组设备能否持续产出满足 SLA 的 token。第三方即使获得相关开源组件，也仍需把硬件、网络、软件版本、请求规模和调度策略共同调到合适状态。

因此，低成本的准确定义不是“服务器采购价格更低”，而是**每单位设备时间能够生产更多满足质量、时延和稳定性要求的 token**。这也自然引出本文的迁移问题：既然 Hugging Face 提供的是模型语义、权重和参考实现，那么从 GPU 迁移到 Ascend NPU，真正需要逐步重建的就是模型下面的执行系统。后文将沿 GPU Reference、框架与算子适配、Kernel、低精度、通信、Serving 和生产验收展开这条迁移路径。

### 本节可靠来源

- **模型发布与架构**：[DeepSeek-V3.2 官方 Hugging Face 仓库](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)、[`config.json`](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json)、[`inference/`](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/main/inference)、[`encoding/`](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/main/encoding)、[DeepSeek-V3 官方仓库](https://github.com/deepseek-ai/DeepSeek-V3)与官方技术报告。具体文件和仓库体积以访问时页面为准。
- **Kernel 与通信**：[FlashMLA](https://github.com/deepseek-ai/FlashMLA)、[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)、[DeepEP](https://github.com/deepseek-ai/DeepEP)。这些仓库用于确认公开功能、许可证和 GPU/CUDA 依赖，不把仓库中的单项性能数字外推到 Ascend。
- **调度与生产推理**：[EPLB](https://github.com/deepseek-ai/EPLB)、[V3/R1 计算—通信重叠 Profile](https://github.com/deepseek-ai/profile-data)、[DeepSeek-V3/R1 在线推理系统概览](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)。
- **集群与 Kernel 开发**：[3FS](https://github.com/deepseek-ai/3FS)、[TileKernels](https://github.com/deepseek-ai/TileKernels)与 [DeepSeek Open Infra Index](https://github.com/deepseek-ai/open-infra-index)。
- **证据边界**：表中“公开状态、GPU 依赖和官方生产设计”来自上述一手资料；“Ascend 需要重建什么”是依据这些依赖关系与本文迁移框架作出的工程推断，不代表 DeepSeek 或华为官方承诺。

---

# 1. DeepSeek 的效率竞争力与硬件迁移

## 1.1 模型开源之后，系统工程仍然决定推理效率

DeepSeek 的模型权重、模型结构和部分推理代码已经公开，这使其他平台能够准确理解模型需要完成哪些计算。但知道“模型要计算什么”，并不等于已经解决“这些计算怎样最高效地运行”。

以 DeepSeek-V3 系列的生产推理为例，DeepSeek 官方公开的在线系统使用跨节点 Expert Parallel，通过更大的全局 batch 提高 Expert GEMM 的计算效率，并将不同 Expert 分布到多张 GPU 上。同时，系统还需要解决 Prefill 负载均衡、Decode 负载均衡、Expert 负载均衡以及计算与通信重叠等问题。生产系统中的矩阵乘法和 Dispatch 使用 FP8，而核心 MLA 与 Combine 使用 BF16。

这些优化并不存在于模型权重本身，而存在于模型之下的执行系统中。

因此，一套 DeepSeek 推理能力可以划分为两类资产。

第一类是具有较强可迁移性的**模型语义资产**，包括模型层数、Hidden Size、MLA/DSA 的数学关系、MoE Router、Expert 数量、Top-K 选择规则、MLP 计算以及生成逻辑。这些内容定义模型“应该算什么”。

第二类是与硬件和软件栈高度相关的**执行效率资产**，包括 Attention Kernel、Grouped GEMM、FP8/W8A8 执行路径、Expert Dispatch/Combine、通信与计算重叠、KV Cache 布局、图编译、并行策略以及 Serving 调度。这些内容决定模型“怎样算得快”。

从 GPU 迁移到 NPU，第一类能力原则上需要保持，第二类能力则需要重新映射和验证。

> **梁文锋 2026 投资者交流材料 · 开源不等于获得同样的部署效率**
>
> 梁文锋明确表示，即使模型全部开源，其他人真正“用起来”仍然存在门槛；即使成功部署，要把成本降到 DeepSeek 的水平也很困难。DeepSeek 担心第三方部署时细节没有做好，最终效果下降或成本偏高。
>
> **注释**：模型资产可以开源，但生产推理效率不会随权重自动复制。从模型到低成本 Serving 之间仍存在大量工程优化。这一判断可以作为本节“系统工程仍然决定推理效率”的直接访谈依据，也是“为什么研究 GPU→Ascend 迁移”的现实出发点。

> **梁文锋 2026 投资者交流材料 · 成本是模型竞争的核心维度**
>
> 在讨论大模型长期竞争时，梁文锋把最终差异归纳为成本、时间和用户体验，并明确认为“成本可能是排在第一位的区别”。这里的成本指在提供相同质量服务的条件下，能够以什么资源成本完成服务。
>
> **注释**：成本不是部署完成之后才考虑的运营指标，而是模型和系统竞争力的一部分。这也意味着迁移不能以“模型能跑”为终点，必须进一步验证吞吐、时延和资源效率。

> **梁文锋 2026 投资者交流材料 · DeepSeek 主动追求计算效率**
>
> 梁文锋指出 DeepSeek 比很多商业公司更加在意模型计算效率，并将低成本与团队目标联系起来。他进一步解释，在算力有限的情况下，计算效率越高，就越能承担更大的模型。
>
> **注释**：效率优化同时服务推理成本和模型 Scaling。更高的效率意味着同样资源可以支持更大的模型、更大量的实验或更低的推理成本。这一因果关系避免把 DeepSeek 的低成本理解成单纯的价格策略。

> **梁文锋 2026 投资者交流材料 · DeepSeek 的算力约束是现实背景**
>
> 梁文锋认为中美 AI 发展的主要差距来自算力资源，算力不足不仅直接限制模型规模，也限制实验次数，从而进一步影响人才培养和技术探索。他强调 DeepSeek 会在自己能够承担的模型规模内，通过更高效率做更多研究。
>
> **注释**：DeepSeek 高效率路线具有明确的资源约束背景。当硬件资源不能无限增加时，软件与算法效率的边际价值会被放大。这既解释 DeepSeek 为什么持续投入 FP8、MoE、通信、Kernel 等底层优化，也解释国产算力为何在战略议题中占据位置。

---

## 1.2 高层编程抽象正在改变硬件迁移的边界

2026 年的梁文锋投资者交流材料提出了一个值得关注的判断：随着 TileLang 等更高层的 Kernel 编程方式成熟，开发者可以减少对特定 GPU 编程生态的直接依赖，将更多计算逻辑表达在更高的抽象层，再针对不同硬件建立相应后端。交流材料同时提到 DeepSeek 正参与华为平台适配，并将 TileLang 视为其中的重要技术方向。

> **梁文锋 2026 投资者交流材料 · CUDA 生态的壁垒正在发生变化**
>
> 梁文锋判断 CUDA 的生态护城河正在削弱，原因包括 AI 可以辅助代码开发，以及 TileLang 这类更高层编程方式的出现。他认为这使重新建立一套算子生态所需的成本明显下降。
>
> **注释**：GPU 的竞争优势不仅来自芯片，也来自多年积累的软件生态；而高级 DSL + AI Code Generation 有可能降低重建这层生态的成本。这是讨论“为什么跨 GPU/NPU 迁移开始变得可行”的关键战略依据。

> **梁文锋 2026 投资者交流材料 · DeepSeek 明确参与华为适配**
>
> 梁文锋表示 DeepSeek 当前主要与华为合作，华为进行自身适配，DeepSeek 也会深入参与生态。他将双方当前的重要工作概括为高级语言编译器以及“把 TileLang 做好”。
>
> **注释**：DeepSeek→Ascend 并非纯理论迁移问题，访谈明确给出了现实中的合作与适配方向。这也是本白皮书将“DeepSeek→Ascend”作为具体 case，而非泛化 GPU→NPU 研究的重要依据。

这一判断可以从已经公开的 DeepSeek-V3.2 工程中得到部分技术印证。DeepSeek-V3.2-Exp 官方仓库同时提供两类 Kernel 路径：面向高性能 CUDA 实现的 DeepGEMM 和 FlashMLA，以及更强调可读性和研究开发效率的 TileLang 实现。

> **梁文锋 2026 投资者交流材料 · TileLang 是 DeepSeek 跨硬件判断中的关键技术**
>
> 梁文锋把 TileLang 描述为高级语言，并认为相对于直接编写 CUDA，它需要编写的代码量更少、开发速度更快。在被问到是否会牺牲 inference 效率时，他的回答是“是提高效率，是大幅提高效率”。他同时表示目前 TileLang 仍由人工编写，团队也在尝试让 AI 编写 TileLang。
>
> **注释**：TileLang 的价值不只是代码可移植性，还包括 Kernel 开发效率。DeepSeek 希望把算子优化表达从高度依赖特定 GPU 编程体系的方式提升到更高抽象层。这一观点可连接本白皮书 Compiler / DSL 层的讨论，并进一步对接 Ascend 的 TileLang Backend、PTO/PyPTO。

> **梁文锋 2026 投资者交流材料 · 新编程语言的目标不是用便利性交换大量性能**
>
> 在第 41 页的问答中，对方直接提出使用 TileLang 是否会降低 inference 效率。梁文锋否认存在显著负面影响，并表示即使硬件底层执行效率存在约 1%–2% 的损失，他也认为可以接受。
>
> **注释**：DeepSeek 对 DSL 的目标是保持接近底层实现的性能，同时显著降低开发成本。这进一步引出一个白皮书问题：Tile DSL 跨硬件迁移究竟保留了多少优化知识，Backend 又需要重新解决哪些问题？

这意味着 Kernel 开发正在逐渐形成三层结构：

1. 上层保留 Attention、GEMM、Routing 等算法和数据流语义；
2. 中间层通过 Tile 级编程描述数据分块、计算和流水；
3. 底层再针对 GPU 或 NPU 的具体计算单元、内存层级和指令完成映射。

Ascend 侧也出现了相似方向。PTO（Parallel Tile Operation）由 CANN 定义为面向 Tile Programming 的虚拟 ISA，通过统一的 Tile 级计算和数据流抽象降低不同 Ascend 代际之间的迁移成本，同时保留 Tile Size、Tile Shape 和指令顺序等性能调优空间。目前 PTO 已经接入 PyPTO 与 TileLang Ascend，并支持 A2、A3 和 A5。

因此，高层 DSL 的意义并不是让硬件差异消失，而是改变差异暴露给开发者的位置。模型算法有机会保持相对稳定，而与具体硬件相关的计算映射、数据搬运、流水组织和指令选择被集中到 Backend 和 Kernel 优化层。

这也是理解 DeepSeek 跨 GPU/NPU 迁移的关键基础。

> **梁文锋 2026 投资者交流材料 · V3 跨硬件判断**
>
> 他认为 V3 虽然训练时使用 NVIDIA GPU，但 DeepSeek 已经将越来越多执行能力建立在自己的高级编程体系上，因此对 NVIDIA 软件生态的依赖已经明显降低；按照他的设想，如果把这一套执行体系在华为硬件上重新实现，就能够完成平台迁移。
>
> **注释**：这是访谈对“什么东西真正需要迁移”给出的最直接答案：不是重新设计 V3，而是把模型下面的一套高效执行体系重新映射到目标硬件。这可以作为整份白皮书的核心研究假设。需要注意，其中“几乎不依赖 NVIDIA 生态”“重新做一遍就完成”等强结论必须由 DeepSeek Infra、TileLang、Ascend 官方材料进一步拆解验证。

> **梁文锋 2026 投资者交流材料 · 国产卡生态问题被定义为软件栈问题之一**
>
> 梁文锋认为国产 AI 芯片的生态问题正在快速改善，并认为未来的重要限制可能更多来自产能。他特别把高级语言和编译器视为解决生态问题的重要抓手。
>
> **注释**：“国产卡不好用”不能简单归因于芯片算力，还涉及编程模型、编译器、算子库和工具生态。这支撑本白皮书把分析对象从芯片规格自然扩展到 Framework → Compiler → Kernel → Runtime → Communication 的完整软件栈。

> **梁文锋 2026 投资者交流材料 · 国产算力的现实约束仍然存在**
>
> 梁文锋同时明确承认 Ascend 与最先进 NVIDIA GPU 之间仍存在硬件代际和效率差距，并认为当前国产算力的一个现实限制是产能。
>
> **注释**：生态可迁移与硬件性能等价是两个不同命题。软件生态障碍降低，不意味着芯片自身的算力、能效和供给差距自动消失。这一判断用于平衡本节对 TileLang 等高层抽象的乐观叙述，避免得出“有 TileLang，所以 GPU/NPU 已经没有差别”的错误结论。

---

# 2. DeepSeek-V3.2 推理迁移的技术对象

如果只观察模型文件，迁移似乎只是将 PyTorch 模型加载到新的 Device；如果观察完整推理系统，迁移实际上跨越从 Model 到 Cluster 的多个层次。

| 层级                 | DeepSeek-V3.2 中的具体对象                         | GPU → Ascend 的主要变化                           |
| ------------------ | -------------------------------------------- | -------------------------------------------- |
| Model Architecture | DSA/MLA、MoE、Router、Expert、MLP                | 数学语义保持                                       |
| Framework          | PyTorch、Transformers、vLLM/SGLang             | Device Backend 和部分模型实现需要适配                   |
| Graph / Operator   | MatMul、Attention、TopK、RMSNorm、Grouped GEMM 等 | 算子实现、融合模式和数据格式需要重新映射                         |
| Compiler / DSL     | CUDA、TileLang 等                              | 转换为 Ascend 对应 Backend、Ascend C、PTO/PyPTO 等路径 |
| Kernel             | FlashMLA、DeepGEMM、Dispatch/Combine Kernel    | GPU Kernel 无法直接作为 NPU Kernel 执行，需要重新生成或实现    |
| Runtime / Memory   | Stream、Buffer、KV Cache、Graph Capture         | 根据 NPU 内存与 Runtime 重新组织                      |
| Communication      | EP、TP、All-to-All、P2P                         | 从 GPU 通信栈映射到 HCCL 等 Ascend 通信能力              |
| Serving            | Prefill、Decode、Batching、Scheduling、MTP       | Serving 语义可继承，具体策略重新调优                       |
| Cluster            | GPU 节点和网络拓扑                                  | 映射至 Ascend 节点、超节点和相应网络拓扑                     |
| Validation         | Accuracy、TTFT、TPOT、Throughput、Scale          | 重新建立 Ascend Benchmark 与生产验收基线                |

这十个层级之间存在明确的依赖关系。模型架构决定计算模式，计算模式决定热点算子，热点算子决定 Kernel 和数据访问行为；MoE 和模型规模进一步决定跨卡通信，并行和通信又决定 Serving 与 Cluster 的组织方式。最终，所有这些选择共同体现为推理时延、吞吐和成本。

> **梁文锋 2026 投资者交流材料 · 硬件替换必须从系统级看，而不是比较单卡**
>
> 在讨论华为 950 时，梁文锋使用的是“超节点”与 NVIDIA GB200/GB300 系统进行比较，并从任务能力、价格、卡数和代际差距等方面讨论替代关系，而没有只比较单卡峰值算力。
>
> **注释**：大模型平台竞争最终需要比较系统能力，而非单卡参数。模型可能依靠更多 NPU、不同通信拓扑和系统协同获得相似任务能力。这一判断支撑本白皮书 Cluster / SuperNode 层级为何是迁移链条的一部分。需要说明，访谈中“几张卡对几张卡”“完全平替”等具体数字属于强预测性陈述，不作为本白皮书事实依据。

因此，“DeepSeek-V3.2 已经支持 Ascend”只能说明迁移链条中的基础能力已经成立，并不能单独说明所有 GPU 优化已经在 NPU 上获得完全等价的实现。

---

# 3. DeepSeek-V3.2 的迁移过程

## 3.1 第一阶段：建立 GPU Reference

正式迁移首先需要固定 GPU 参考系统，而不是直接开始修改 NPU 代码。

Reference 的作用是确定迁移前的“正确结果”和“性能目标”。至少需要固定模型版本、权重版本、精度类型、推理框架、输入长度、输出长度、并发度、Batch、并行策略和 Serving 配置，并记录输出结果、Logits/Logprob、TTFT、TPOT、Throughput、显存占用和通信开销。

这样做的原因很直接：模型迁移过程中同时存在大量变量。如果模型版本、精度、Serving 配置和测试 workload 同时变化，最终很难判断性能变化究竟来自硬件差异、模型实现差异还是测试条件差异。

DeepSeek-V3.2 尤其需要固定长上下文 workload。V3.2 相比 V3.1 的关键架构变化是 DSA，设计目标就是降低长序列 Attention 的计算开销。 如果只使用很短的输入测试，可能无法真实暴露 DSA、KV Cache 和通信路径的性能特征。

---

## 3.2 第二阶段：模型运行与框架适配

模型首先需要在 Ascend 上建立完整执行路径。

对于 PyTorch 生态，TorchNPU 是 PyTorch 与 Ascend NPU 之间的适配层。其职责包括将 PyTorch 的设备、算子、动态图、分布式和 Profiling 能力接入 Ascend。官方文档明确将其定位为 PyTorch 使用 Ascend AI Processor 的框架适配插件。

对 DeepSeek-V3.2 来说，这一阶段解决的问题主要包括：

* 模型代码是否能够被当前框架版本识别；
* PyTorch API 是否具有对应的 NPU 支持；
* 模型所需 dtype 是否能够被目标硬件和软件版本处理；
* 自定义算子是否存在可用的 Ascend 实现；
* 模型权重是否能够按照目标并行方式正确切分和加载。

当前 vLLM Ascend 已经提供 DeepSeek-V3.2 的官方部署路径。以量化版本为例，官方文档给出单个 Atlas 800 A3 16 卡节点，或者两个 Atlas 800 A2 8 卡节点的部署方案，并已经包含 Expert Parallel 配置。

因此，在当前生态中，“让 DeepSeek-V3.2 在 Ascend 上启动并生成结果”已经有相对明确的产品化路径。迁移难点进一步向算子性能、分布式效率和生产 Serving 移动。

---

## 3.3 第三阶段：算子覆盖与执行路径映射

模型代码最终需要被拆解为设备能够执行的算子和 Kernel。对于 DeepSeek-V3.2，几个最重要的热点区域包括 Attention、MoE Router、Grouped GEMM、RMSNorm、低精度 MatMul，以及 Expert Dispatch/Combine。

这里需要区分两个问题。

**算子支持问题**意味着某段模型语义没有可用的 NPU 执行路径，例如特定 Shape、dtype 或自定义算子无法运行。这类问题首先影响 Functional Compatibility。

**算子性能问题**意味着已有实现能够执行，但数据搬运、Tiling、融合、流水或硬件利用率并未达到目标。这类问题影响 Performance Compatibility。

这一区分对产品体验非常重要。客户遇到“某个 Operator 有问题”时，需要首先知道这是“没有实现”，还是“实现存在但不够快”，因为两类问题对应完全不同的处理路径。

---

## 3.4 第四阶段：Kernel 优化

当所有计算已经能够运行后，迁移开始进入最核心的性能重建阶段。

### DSA / MLA

DeepSeek-V3.2 延续 MLA，并新增 DSA。DSA 通过 Indexer 从长上下文中筛选一部分更相关的 KV 参与 Attention，官方配置中包含独立的 Indexer Head 和 `index_topk=2048`。

从模型角度看，DSA 规定的是“哪些 KV 应当参与当前 Attention”；从硬件角度看，还需要决定这些数据如何读取、如何分块、如何进入计算单元以及输出如何写回。

DeepSeek 在 GPU 上将相关高性能 Sparse Attention Kernel 放入 FlashMLA，同时在 DeepGEMM 中提供 Indexer Logit Kernel。 迁移到 Ascend 后，这些 CUDA/Hopper 相关实现不能简单复制，Ascend 需要建立等价的 NPU 执行路径，并针对自身的存储层级和 AI Core 重新优化。

因此，这部分迁移保留的是 DSA/MLA 的算法结构，而不是 FlashMLA 本身。

### MoE 与 Grouped GEMM

DeepSeek-V3.2 官方配置包含 256 个 Routed Expert，每个 token 激活其中 8 个，同时还有一个 Shared Expert。 这意味着不同 token 会被送往不同 Expert，不同 Expert 实际接收到的 token 数量也可能不同。

单个 Expert 内部本质上仍然包含 Linear、SiLU 和矩阵乘法，但是如果为每个 Expert 单独发起一次小规模 GEMM，会产生大量 Kernel Launch 和低利用率问题。因此，高性能 MoE 推理通常会把多个 Expert 的计算组合为 Grouped GEMM。

DeepSeek 的 DeepGEMM 就包含针对 Dense 和 MoE 的 FP8 GEMM 实现。 迁到 Ascend 后，Grouped GEMM 的数学需求仍然存在，但具体 Kernel 需要通过 Ascend 高性能算子库、Ascend C、TileLang Ascend、PTO/PyPTO 等执行路径重新承接。

Ascend C 是 CANN 面向自定义算子开发提供的编程语言，算子经过编译和 Runtime 调度后运行在 Ascend AI Processor 上；其性能优化涉及数据搬运、Memory、Pipeline、Instruction 和 Tiling 等多个维度。

因此，Kernel 迁移的准确含义不是“把 CUDA 代码翻译成 Ascend C”，而是根据相同的计算需求，在另一种硬件结构上重新寻找高效的数据分块、计算和流水组织方法。

---

## 3.5 第五阶段：低精度计算与数值验证

DeepSeek 的低成本推理与低精度计算密切相关，但“支持低精度”不能只理解为硬件提供某一种 dtype。

在实际执行中，低精度策略涉及权重、Activation、通信 Tensor 和 KV Cache 分别采用什么格式，Scale 如何计算，量化和反量化发生在哪里，以及对应 Kernel 是否针对这种格式优化。

DeepSeek-V3/R1 的公开生产系统中，矩阵乘法和 Dispatch 使用 FP8，而核心 MLA 与 Combine 使用 BF16。 DeepSeek-V3.2 官方推理配置同样包含 FP8 dtype 和相应 scale format。

而当前 vLLM Ascend 官方 DeepSeek-V3.2 部署路径主要提供 W8A8 量化版本，并通过 `--quantization ascend` 进入 Ascend 的量化执行路径。

这说明迁移过程中不能建立简单的“GPU FP8 = NPU FP8”对应关系。真正需要验证的是：在目标 Ascend SKU 和软件版本上，采用何种数据格式能够同时满足显存、吞吐和精度要求。

因此，低精度迁移必须与 Correctness 验证同时进行。除了最终生成结果，还应根据测试目标观察 Logits、Logprob、关键 Layer Output 和中间 Tensor 的差异，从而区分模型精度变化和实现错误。

---

## 3.6 第六阶段：Expert Parallel 与通信优化

DeepSeek-V3.2 的 MoE 结构使通信成为迁移中的核心问题之一。

当 256 个 Routed Expert 分布在不同设备上时，一个 token 经过 Router 选择 Top-8 Expert 后，可能需要被发送到多个远端 Rank。Expert 计算完成后，结果还要重新汇总。这形成 MoE 中典型的 Dispatch → Expert Compute → Combine 数据流。

在 NVIDIA 系统上，DeepSeek 使用 DeepEP 为这种数据交换提供专门的 Expert Parallel 通信能力，包含面向 Prefill 的高吞吐通信 Kernel、面向 Decode 的低时延 Kernel，以及通信与计算重叠。

Ascend 侧，HCCL 提供 AlltoAll 等集合通信能力。HcclAlltoAll 可以使通信域中的 Rank 相互发送和接收数据；当前 HCCL 文档同时列出了 AlltoAll、AlltoAllV 等能力在节点内和超节点内的支持情况。

但拥有 All-to-All primitive 并不等于已经获得高效 MoE 通信。真正影响性能的仍然包括：

* Expert 如何映射到 Rank；
* 不同 Expert 接收到多少 token；
* Dispatch 的数据量和数据类型；
* Rank 之间的实际链路；
* 通信是否可以被 Expert GEMM 覆盖；
* 是否存在热点 Expert 和 Straggler Rank。

因此，MoE 性能优化的对象不是单独一个 HCCL 算子，而是 Router、Expert Placement、Token Traffic、Grouped GEMM 和通信拓扑共同构成的执行过程。

---

## 3.7 第七阶段：Serving 与集群调优

模型和 Kernel 均达到较好性能之后，仍然不能直接推导生产服务已经达到目标。

大模型推理主要包含 Prefill 和 Decode 两种负载。Prefill 一次处理大量输入 token，通常具有更强的计算密集特征；Decode 则逐 token 生成结果，更容易受到内存访问、KV Cache 和通信时延影响。

DeepSeek 的公开生产系统已经采用 Prefill 与 Decode 分离的设计。Ascend 上的 vLLM Ascend 也已经为 DeepSeek-V3.2 提供 Prefill-Decode Disaggregation。官方文档明确指出，Prefill 节点侧重高吞吐的 Prompt Processing，而 Decode 节点侧重低时延的 Token Generation，并允许两个阶段使用独立的并行配置。

当前官方 DeepSeek-V3.2 示例甚至已经给出一组具体配置：在 A3 的多节点 PD 分离环境中，Prefill 采用 DP2 + TP16，Decode 采用 DP8 + TP4。官方性能示例使用 64K 输入、3K 输出时报告 533 tokens/s 和 32 ms TPOT。该数据说明 Ascend 已经能够把 DeepSeek-V3.2 的迁移推进到生产 Serving 级别，但这个结果只对应特定硬件、版本、模型量化方式和 workload，不能作为不同系统之间的通用性能结论。

这一阶段需要重新调优的参数通常包括 Batch、Concurrency、TP/DP/EP 配置、Prefill/Decode 资源比例、KV Cache 容量以及请求调度策略。

---

# 4. Ascend 在迁移过程中的能力分工

从客户任务出发，Ascend 软件栈可以被理解为一组承担不同迁移职责的能力，而不是一组孤立产品名称。

| 客户任务               | 对应 Ascend 能力                                | 在迁移中的作用                               |
| ------------------ | ------------------------------------------- | ------------------------------------- |
| 让 PyTorch 模型调用 NPU | TorchNPU                                    | Framework 与 Device 适配                 |
| 执行和优化模型图           | CANN 图编译与 Runtime                           | Graph Lowering、算子执行、内存与调度             |
| 使用已有 NPU 算子        | CANN 算子能力                                   | 承接常见计算                                |
| 开发高性能自定义 Kernel    | Ascend C                                    | 针对硬件实现和调优算子                           |
| 进行 Tile 级编程        | PTO / PyPTO                                 | 用更高层抽象表达 Tile Compute 和 Dataflow      |
| 使用跨硬件 DSL          | TileLang Ascend                             | 为 TileLang 建立 Ascend Backend          |
| 实现多 NPU 通信         | HCCL                                        | 提供 AllReduce、AlltoAll、P2P 等通信基础能力     |
| 使用主流推理框架           | vLLM Ascend                                 | 将 vLLM Serving 能力接入 Ascend            |
| 生产级大模型 Serving     | MindIE 等                                    | 提供推理调度、缓存和部署能力                        |
| 定位性能问题             | TorchNPU Profiler、msProf、MindStudio Insight | 观察 Framework、CANN、NPU 和 Communication |

这里最值得关注的是不同能力之间的上下关系。

例如，客户发现 DeepSeek-V3.2 的某个 MoE Layer 很慢，问题可能首先暴露在 vLLM 的模型执行过程中；继续下钻，可能发现时间集中在某个 Grouped GEMM 或 All-to-All；继续下钻后，又可能发现具体瓶颈来自 AI Core 利用率、数据搬运或者某段通信。不同层级的工具分别能够看到这条链的一部分。

TorchNPU Profiler 已经能够采集上层应用、CANN、NPU Operator 和 AI Core 性能数据；在更高采集等级下，还可以输出 HCCL 的 `communication.json` 和 `communication_matrix.json`。

技术能力已经能够提供大量底层事实，产品体验的难点则在于如何把这些事实重新关联回模型语义和客户任务。

---

# 5. 推理迁移中的客户体验问题

DeepSeek-V3.2 的迁移过程可以进一步转换为一条客户旅程。不同阶段的核心问题并不相同。

| 阶段         | 客户需要回答的问题                                | 当前信息通常分散在哪里                       |
| ---------- | ---------------------------------------- | --------------------------------- |
| 环境准备       | 当前模型、框架、CANN、TorchNPU 和硬件是否兼容？           | 安装文档、版本矩阵                         |
| 模型启动       | 模型为什么无法运行？                               | Error Log、框架日志                    |
| 算子适配       | 是 API、Operator、Shape、dtype 还是 Kernel 问题？ | Framework / CANN / Operator 文档    |
| 精度验证       | Ascend 结果从哪里开始偏离 GPU Reference？          | Tensor Dump、Logit、Benchmark       |
| Kernel 调优  | Attention、GEMM、MoE 中具体哪里慢？               | Profiler、算子分析工具                   |
| 分布式调优      | 加卡后为什么吞吐没有按预期增加？                         | HCCL Trace、Topology、Rank 数据       |
| MoE 调优     | 是 Expert 不均衡、通信慢还是 GEMM 慢？               | Router、Expert、Communication 数据    |
| Serving 调优 | 单次 Benchmark 很快，为什么线上 TTFT/TPOT 仍然较差？    | Scheduler、Queue、KV Cache、Profiler |
| 生产验收       | 当前已经“能跑”，距离生产可用还差什么？                     | 多套测试和人工判断                         |

这些问题存在一个共同特征：客户看到的技术信息通常按照软件组件组织，而客户真正关心的问题是按照迁移任务组织。

例如，一个 DeepSeek-V3.2 请求出现 Decode 延迟升高时，客户真正想知道的是：

**这一次延迟升高是由 Attention、MoE Expert Compute、Expert Communication、KV Cache 还是请求排队造成的？**

但现有系统可能分别提供 NPU Operator Timeline、HCCL Trace、Memory Data 和 Serving Log，需要专家自行完成关联。

因此，迁移体验中的核心产品机会并不是增加更多原始指标，而是建立跨层关联能力。

---

## 5.1 Migration Readiness：迁移前识别风险

在真正占用大量 NPU 资源之前，系统可以根据模型、目标硬件、框架和精度配置建立迁移检查。

对于 DeepSeek-V3.2，检查结果至少应覆盖：

* 模型是否已经具有官方适配路径；
* 目标 Ascend SKU 是否支持；
* 当前权重精度是否能够直接使用；
* DSA、MTP、EP、PD 分离等模型和 Serving 特性是否支持；
* 是否包含需要重新开发的 Custom Operator；
* 单节点能否容纳模型；
* 是否必须采用多节点部署。

这种能力解决的不是性能问题，而是客户在迁移开始前对工作量缺乏预期的问题。

---

## 5.2 Model-to-Device Execution Map：解释模型怎样落到 NPU

对于非底层研发人员，单独看到几十万个 Kernel Event 很难形成有效认知。

更有价值的视图应当保留模型语义，例如用户选择 DeepSeek-V3.2 的某一个 MoE Layer 后，可以继续看到：

**MoE Layer → Router → Dispatch → Expert Grouped GEMM → Combine**

随后再继续展开：

**Framework Operator → CANN Operator → NPU Kernel → AI Core / Communication**

这样，用户看到一个耗时异常的 Kernel 时，能够知道它究竟对应模型里的哪段计算。

这一能力对于 GPU/NPU 对照同样适用。GPU 和 NPU 的 Kernel 名称没有必要一一对应，但二者可以通过共同的模型语义进行对齐。例如，同一个 DeepSeek DSA 计算，在 GPU 侧可能进入 FlashMLA 相关实现，在 Ascend 侧则进入另一套 Attention Operator 和 NPU Kernel。对照的中心应该是“DSA 这段模型计算”，而不是两个不同平台的 Kernel 名字。

---

## 5.3 Correctness Diff：定位首次数值分歧

跨硬件迁移中的精度问题通常不应该从最终生成文本反向猜测。

更加有效的方法是同时运行 GPU Reference 和 Ascend Case，并沿 Layer 顺序比较关键 Tensor。当误差在前若干层保持稳定，而从某一个 Layer 开始突然放大时，就可以继续下钻到 Attention、RMSNorm、MoE 或特定 Operator。

用户最终需要得到的结论不应该只是“两个平台结果不同”，而应该接近：

**首次显著差异出现在 Layer N 的某个 Operator；该 Operator 在 GPU 与 NPU 上使用了不同的 dtype 或执行实现。**

这种表达将一个模型级异常转换成可执行的工程问题。

---

## 5.4 MoE Observatory：把 Expert Parallel 从通信 Trace 还原成模型行为

DeepSeek-V3.2 是一个很适合建立 MoE 专用可视化的案例。

在一次推理中，可以同时观察：

* Token 被 Router 分配给哪些 Expert；
* Expert 分布在哪些 Rank；
* 每个 Expert 接收到多少 Token；
* 每个 Rank 的 Expert Compute 时间；
* Dispatch 和 Combine 的通信量；
* All-to-All 的发生时间；
* 通信是否覆盖在计算之下；
* 是否存在热点 Expert 或 Straggler Rank。

这样，当某个 Rank 明显变慢时，用户可以进一步判断：它是因为承担了更多 Expert Token，还是因为通信链路较慢，或者 Grouped GEMM 本身没有达到目标效率。

这比单独显示一个 HCCL All-to-All Event 更接近客户真正需要解决的问题。

---

## 5.5 Performance Attribution：从“慢”定位到可行动原因

迁移完成初期最常见的问题并不是程序报错，而是“模型能运行，但明显比预期慢”。

性能分析应该首先把时间拆成几个可以行动的类别：

* Compute：算力没有被充分利用；
* Memory：数据搬运或 KV Cache 成为瓶颈；
* Communication：TP/EP 等分布式通信占比过高；
* Scheduling：Kernel 或请求之间存在 Bubble；
* Load Balance：不同 Expert、Rank 或请求之间负载不均；
* Host Overhead：Host 与 Device 协同效率低。

继续下钻时，再根据具体数据判断是 Tiling、Memory Access、Pipeline、通信拓扑还是 Parallel Strategy 导致问题。

这种由“现象 → 类型 → 具体原因”的结构，比要求所有客户直接阅读 Profiler Timeline 更适合作为迁移产品入口。

---

# 6. 迁移完成的判定

DeepSeek-V3.2 能够在 Ascend 上输出正确文本，只能说明完成了最基础的 Functional Bring-up。对于生产推理，更合理的完成标准至少包含 Functional、Correctness、Performance、Scale 和 Production 五个维度。

## 6.1 Functional

模型能够稳定完成完整 Forward 和生成流程，核心模型特性均具有有效执行路径，包括 DSA/MLA、MoE、Expert Parallel、目标精度和 Serving 所依赖的必要能力。

当前 vLLM Ascend 已经将 DeepSeek-V3.2 纳入模型支持矩阵，并提供单节点、多节点以及 PD 分离部署指导，因此 Functional 层面已经存在官方验证路径。

## 6.2 Correctness

在固定模型、Tokenizer、精度、解码参数和测试数据之后，对 Ascend 与 Reference 的模型输出进行验证。

Correctness 不宜被简化为一个全行业统一的数值误差阈值，因为 BF16、FP8、W8A8 等路径会产生不同的误差特征。验收应同时覆盖模型任务精度与必要的数值分析。

vLLM Ascend 的 DeepSeek-V3.2 官方文档已经将 Accuracy Evaluation 独立作为部署验证阶段，并提供 AISBench 和 LM Evaluation Harness 两种评估路径。

## 6.3 Performance

生产推理至少需要关注：

* TTFT（Time to First Token）：用户等待第一个输出 token 的时间；
* TPOT（Time per Output Token）：生成阶段每个 token 的平均时间；
* Throughput：单位时间内系统能够完成的 token 数或请求量；
* 并发条件下的 Tail Latency：高负载情况下的尾部时延。

这些指标必须在明确 workload 的前提下比较。输入长度、输出长度、Batch、Concurrency 和 Cache Hit Ratio 的变化都可能显著影响结果。

因此，不能用一个独立的 tokens/s 数字判断 GPU 与 NPU 的优劣，而应比较同一模型、同一业务负载和同一 SLO 下两套系统的表现。

## 6.4 Scale

大规模 MoE 模型的性能高度依赖多设备扩展。

Scale 验证需要回答：增加设备之后，吞吐提升了多少，同时新增了多少通信成本和负载不均衡。

对于 DeepSeek-V3.2，尤其需要观察 Expert Parallel，因为设备规模扩大意味着 Expert 分布范围扩大，同时可能增加跨 Rank Token Traffic。如果 Grouped GEMM 获得的收益低于增加的通信开销，增加设备并不会自然获得理想的 Scaling Efficiency。

## 6.5 Production

生产阶段还需要验证长稳运行、并发流量、KV Cache、请求调度、异常恢复和资源波动。

一个 Kernel Benchmark 达到很高 TFLOPS，并不能直接说明在线服务具有良好的 TTFT；一个离线 Throughput 很高的部署，也可能在实际请求长度分布下出现严重 Tail Latency。

因此，生产验收必须回到真实请求 workload，而不能停留在单算子或离线模型 Benchmark。

## 6.6 Economics

成本最终由业务定义，可以进一步观察 tokens/NPU-hour、单位请求资源成本或 cost per million tokens。

这一层并不存在适用于所有客户的统一硬件标准。它的意义在于确认迁移带来的计算资源、工程投入和运营成本是否满足业务目标。

---

# 结语：从模型兼容走向系统效率迁移

DeepSeek-V3.2 从 GPU 向 Ascend NPU 的迁移，展示了现代大模型迁移与传统软件移植之间的重要差异。

模型架构提供了一套相对稳定的数学语义：DSA/MLA 决定 Attention 如何计算，Router 决定 token 如何选择 Expert，MoE 决定哪些参数被激活。这部分不会因为底层芯片发生变化而重新定义。

但模型真正落到硬件之后，Attention 如何分块、Grouped GEMM 如何执行、Expert 如何跨 Rank 分布、Token 如何通信、KV Cache 如何管理、Prefill 与 Decode 如何配置，都需要重新进入目标硬件的软件与执行体系。

因此，GPU 上已经形成的优化能力并不是全部失效，也不是全部可以直接继承。可以继承的是算法、计算结构、并行思想以及已经被证明有效的系统优化方向；需要重新建立的是这些思想在 Ascend 计算、内存、通信和 Serving 系统中的具体实现。

TileLang、PTO/PyPTO 等更高层抽象正在进一步改变这个边界。它们尝试让开发者在比特定芯片指令更高的层级描述计算和数据流，使同一套算法更容易进入不同硬件 Backend。但性能仍然需要面对真实硬件，因此 Kernel、Tiling、Memory、Pipeline 和 Communication Optimization 不会消失，只会被重新组织到更清晰的抽象层中。

对于产品经理和 UX，理解这条链条的意义并不在于掌握每一种 Kernel 编程方法，而在于重新定义迁移产品需要解决的问题。

一个完整的迁移产品不应只回答“这个模型是否支持 Ascend”，还应持续回答五个问题：

1. 当前模型在目标硬件上的适配程度如何；
2. 迁移目前处于模型运行、精度验证、性能优化还是生产调优阶段；
3. 当前问题发生在模型、算子、Kernel、通信还是 Serving；
4. GPU 已经存在的高效执行能力，在 Ascend 上由什么能力承接；
5. 当前系统距离目标性能和生产可用状态还存在多大差距。

当这些问题能够被统一表达、关联和解释时，模型迁移才从依赖少数专家经验的工程过程，转变为一个可以被观察、诊断和管理的产品流程。

---

## 主要资料与证据范围

本文关键技术结论优先采用以下资料：

**DeepSeek 一手资料**

* DeepSeek-V3.2 / DeepSeek-V3.2-Exp 官方模型与推理配置。
* DeepSeek Open Infra：FlashMLA、DeepEP、DeepGEMM 及 V3/R1 在线推理系统。
* 梁文锋 2026 年投资者交流材料，用于分析 DeepSeek 对 TileLang、国产算力生态与跨硬件适配的战略判断。该材料为录音转写整理稿，其中具体数字和预测性判断不作为本文硬件性能事实依据。

**Ascend 与相关官方/主项目资料**

* Ascend Extension for PyTorch / TorchNPU。
* Ascend C 与算子性能调优文档。
* HCCL 集合通信文档。
* vLLM Ascend DeepSeek-V3.2 部署、精度和性能验证资料。
* PTO Tile Library / PyPTO / TileLang Ascend 相关公开资料。
