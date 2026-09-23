# 《昇腾 AI 软件生态的结构性错位》白皮书生成计划

> 状态：已确认并完成网页生成  
> 源文档：`/Users/yin/PyPTO3-main/Insight/Ascend-ecosystem.md`  
> 图表设计参考：`/Users/yin/PyPTO3-main/Insight/dpsk模型迁移-whitepaper/index.html`，重点参考 FIG-03 与 FIG-08  
> 计划输出：`/Users/yin/PyPTO3-main/Insight/Ascend-ecosystem-whitepaper/`

## 1. 内容理解

### 1.1 主题与读者

- 主题：昇腾 AI 软件生态从垂直全栈向开放硬件后端转型时，产品边界、用户资产与生态治理方式之间的结构性错位。
- 目标读者：AI 基础设施负责人、昇腾生态与产品战略人员、大模型训练/推理架构师。
- 阅读后的下一步任务：判断各昇腾软件组件应该占据哪个抽象层，并用“主流生态可达性”而非单一自有软件份额评估生态建设。
- 一句话论点：Ascend 的不可替代价值集中在硬件适配、编译、通信、算子与运行时；未来竞争力取决于这些能力能否以低侵入、可插拔方式进入 PyTorch、Megatron、vLLM 等主流生态。

### 1.2 保真模式

采用 **Reader-first / 读者优先模式**。

- 保留全部实质论点、限定条件、案例与结论。
- 合并重复论述和 30 组 ASCII 示意。
- 允许重新组织章节，使全文形成连续的技术主线。
- 不改变原文对 MindSpore、MindSpeed、MindIE、CANN 等产品的判断边界。
- 不删除全栈模式的优势、MindSpore 的适用场景、版本绑定的工程必然性等重要限定。

### 1.3 来源与证据边界

源文档没有现成图片、SVG 或外部资产。13 条参考资料只提供索引名称，未提供 URL。

已初步核对的权威来源包括：

- Huawei, *Ascend: Open for All to Build a Vibrant Ecosystem*：确认分层解耦、开放协作以及与 PyTorch、Triton、vLLM、verl 等生态合作。
- Huawei 2025 Annual Report：确认 Mind 系列、CANN 与相关工具链的开源开放方向。
- Ascend/MindSpeed：确认 Megatron-LM 集成、一行 `import mindspeed.megatron_adaptor` 和版本配套关系。
- vLLM Ascend 官方文档：确认 hardware-pluggable interface、推荐插件路径和版本对应关系。
- Ascend/TorchAir 官方文档：确认 PyTorch Dynamo 捕获 FX Graph、转换为 Ascend IR/GE 并在 NPU 上编译执行。

仍需在生成阶段核实：

- Alibaba ROLL Ascend RFC 的具体可选依赖约束。
- MindSpore“南向亲和昇腾，北向生态兼容”的原始出处。
- MindIE Motor、MindIE LLM、MindIE Turbo 的最新产品边界与可选关系。
- 华为“AI 商业化重点聚焦硬件”的原始表述。

如无法取得权威原文，将保持为“源文档观点”或“分析推论”，不包装成官方事实。

### 1.4 来源账本分类

| 分类 | 内容 | 报告中的处理 |
|---|---|---|
| 官方已核实 | 分层解耦与开放合作；MindSpeed/Megatron 接口；vLLM 硬件插件；TorchAir FX→GE 路径 | 给出来源链接和直接事实表述 |
| 源文档事实，待核实 | MindIE 最新拆分；ROLL RFC 约束；MindSpore 最新兼容方向；部分 2025—2026 路线描述 | 核实后升级为官方事实，否则标为源文档陈述 |
| 源文档分析 | “结构性错位”“Complement Friction”“CANN 是生态腰部”“软件价值下沉” | 明确作为分析框架，不伪装成官方结论 |
| 战略推论 | 各产品未来定位、Open Hardware Platform 目标模型、Ascend Reachability | 明确标记为综合判断或建议 |

## 2. 报告结构与内容覆盖

| ID 与标题 | 来源映射 | 读者问题 | 保留内容 | 强调与阅读产出 | 编辑动作 |
|---|---|---|---|---|---|
| `cover` 从全栈闭环到开放硬件后端 | 摘要 | 这篇报告解决什么问题？ | 问题、读者、核心判断、2025—2026 转向 | 三项核心结论与阅读承诺 | 将长摘要压缩为三项独立结论 |
| `misalignment` 错位来源于产品抽象层与用户资产边界不一致 | 第 1 节 | 用户资产与 Ascend 差异化分别位于哪一层？ | L0–L4 分层、用户关切、能力边界 | L0–L4 总览及错位边界 | 层级表转为主视觉，正文缩短 |
| `value-capture` 硬件价值捕获为何要求软件入口变薄 | 第 2 节 | 上层闭环为什么会限制硬件价值？ | Hardware Value Capture、Complement Friction、迁移门槛 | 采用收益与迁移摩擦的因果关系 | 合并两条 ASCII 因果链 |
| `asset-first` 用户先选择软件资产，再选择硬件 | 第 3—4 节 | 为什么 Framework replacement 成本高？ | PyTorch 资产、Framework migration、水平组合、MindSpore 优势边界 | 软件资产系统与可替换边界 | 合并重复决策顺序，保留适用场景 |
| `training-interface` 训练侧：MindSpeed 的正确接口形态 | 第 5 节及 5.1 | 硬件优化如何进入 Megatron？ | adapter、可选依赖、ROLL 案例、CUDA/CPU 不变等限定 | 一条可插拔训练路径 | 保留代码示例，合并训练拓扑 |
| `inference-control` 推理侧：控制面归属决定组件边界 | 第 6—8 节 | Scheduler、KV Cache、PD、Parallelism 应由谁负责？ | MindIE 重叠、组件化、vLLM-Ascend extension point | “冲突—拆分—正确接口”三步逻辑 | 三节合并为一个连续章节 |
| `backend-waist` CANN 与语义桥构成不可替代的腰部 | 第 10—11 节 | 上层生态最终如何进入 Ascend？ | GE、HCCL、Runtime、Ascend C、torch_npu、TorchAir | 共同底座与两条语义桥路径 | 合并底层能力图和 FX→GE 路径 |
| `release-train` 上游创新速度与版本列车 | 第 9 节 | 为什么 vendor layer 越厚，同步成本越高？ | 多层版本绑定、上游变化传播、插件版本对应 | 依赖传播路径与工程边界 | 不引入没有来源的周期或成本数字 |
| `positioning` 生态位置比生态规模更关键 | 第 12—13 节 | 各产品位于正确的抽象层吗？ | 产品矩阵、Coverage 与 Alignment 区别 | 产品定位查阅矩阵 | 原表由视觉矩阵完整承载 |
| `transition` 软件价值下沉：从占有栈到提供能力 | 第 14 节 | 软件战略究竟发生了什么变化？ | Vertical Full Stack 与 Open Hardware Platform | 同尺度阶段比较 | 两阶段 ASCII 图合并 |
| `target-model` 上层兼容、下层差异化 | 第 15—17 节 | 长期架构如何容纳开放生态与 Mind 系列？ | 目标模型、三类产品角色、CUDA 生态位置 | 开放生态目标架构 | 合并目标模型和产品建议 |
| `reachability` 用主流生态可达性衡量成功 | 第 18 节 | 如何评审一次生态接入是否真正低摩擦？ | 代码、架构、习惯、专有调优复杂度 | 四段迁移距离与评审问题 | 不制造总分或伪量化指标 |
| `references` 证据边界与参考资料 | 参考资料索引 | 哪些内容来自事实，哪些是分析？ | 13 条参考资料、术语与证据分类 | 可追溯来源与词汇表 | 补充可确认链接，保留未确认索引 |

不计划删除任何实质论点。主要变化是压缩重复表述、合并 ASCII 图和调整章节顺序。

## 3. 图表设计验收基准

### 3.1 参考页面

本报告的图表设计以以下页面作为验收参考：

`file:///Users/yin/PyPTO3-main/Insight/dpsk%E6%A8%A1%E5%9E%8B%E8%BF%81%E7%A7%BB-whitepaper/index.html`

重点参考其中：

- **FIG-03**：多层解释结构、定义→例子→边界→双路径→差异矩阵→证据边界。
- **FIG-08**：上方连续流程、下方统一字段 Passport、显式“需验证”状态和证据终点。

参考页的 `shared.css` 与 `knowledge-map-theme.css` 已进行源码核对。生成时不会复制整套主题，而是在 whitepaper-generator 的 canonical `shared.css` 基线上移植兼容的图表组件与构图规则。

### 3.2 从参考 FIG-03 继承的设计方法

1. 顶部先给明确的“读图”说明。
2. 先定义对象，再给可观察的小例子或主路径。
3. 中间设置清晰的“可以保留 / 必须改变”边界。
4. 双侧比较使用相同的对象顺序、尺度和字段。
5. 底部用对齐矩阵直接标出决定性差异。
6. Caption 分开书写关键结论与证据边界。
7. 一个复杂图可以逐层深入，但每层必须回答同一个父问题。

重点应用于：`FIG-03`、`FIG-04`、`FIG-09`、`FIG-11`。

### 3.3 从参考 FIG-08 继承的设计方法

1. 上方流程负责建立主路径。
2. 下方 Passport 使用相同字段核对多个对象。
3. 未确认内容直接标为“需验证”。
4. Evidence 是流程终点，而不是隐藏在脚注中。
5. 宽图保留合理最小宽度并允许横向滚动，不通过缩小字体强行塞入视口。

重点应用于：`FIG-05`、`FIG-08`、`FIG-10`、`FIG-12`、`FIG-13`。

## 4. 报告级视觉系统

- 基准坐标：`用户生态 → 稳定扩展点 → Ascend 适配层 → 差异化底座 → 硬件`，同时保留 L4→L0 垂直层级。
- 固定对象：PyTorch、Megatron、vLLM/SGLang、MindSpore、MindSpeed、MindIE、torch_npu、TorchAir、vLLM-Ascend、CANN、Ascend。
- 颜色：开放生态使用蓝色；适配/桥接层使用青色；Ascend 差异化能力使用橙色或紫色；硬件使用深色。
- 颜色只表达技术域，不表达证据可信度。
- 实线：直接执行或依赖路径。
- 虚线：可选组件或非强制接入。
- 点线：战略推论或间接关系。
- 交叉标记：控制面冲突或不应叠加的责任边界。
- 证据标识：`官方资料`、`源文档判断`、`推论`、`需验证` 使用文字标签。
- 所有非定量架构图标注“定性结构，非按比例”。
- 英文技术名保持原样，下方配中文功能说明；API、产品名和标识符不翻译。
- 正文和主标签使用 14px 基线，辅助信息不低于 11px。
- 图框采用 PTO 白色、无边框、20px 圆角基线；内部模块使用 7–14px 圆角和克制的浅色实心分区。
- 主要流程使用 HTML/CSS Grid；复杂拓扑或交叉连线才使用 inline SVG。
- 图族连续性：错位诊断 `FIG-01—04` → 接口机制 `FIG-05—09` → 产品与战略综合 `FIG-10—13`。

## 5. 完整插图清单

### FIG-01｜Ascend 的差异化价值与软件控制边界发生错层

- 章节：`misalignment`
- 来源：摘要、第 1 节、L0–L4 表。
- 读者障碍：难以同时看到“用户资产在哪里”和“Ascend 不可替代能力在哪里”。
- 视觉问题：两个价值中心分别落在哪些层，控制边界在哪里越界？
- 类型/角色：分层架构图；Orientation。
- 设计：L4→L0 五层堆栈；左侧标用户希望保留的资产，右侧标 Ascend 差异化能力，突出 L3/L4 与 L1/L2 的边界错位；底部增加一句含义条。
- 关键结论：差异化集中在下层，历史控制边界延伸到了上层。
- 证据模式：源文档定性结构与分析判断。
- 图前导语：先把用户资产与硬件差异化放在同一张分层图中，错位才会显现。
- 双语：Framework、Enablement、Hardware Adaptation、Foundation 配中文功能说明。
- 重复控制：原层级表转为图例，不再逐项复述。

### FIG-02｜硬件采用收益如何被软件迁移摩擦抵消

- 章节：`value-capture`
- 来源：第 2 节两条因果链。
- 读者障碍：难以理解上层闭环为什么会反过来限制硬件价值捕获。
- 视觉问题：采用收益和迁移摩擦在哪个决策点相遇？
- 类型/角色：双轨因果流；Mechanism。
- 设计：上轨显示 workload→Ascend 使用量→Hardware Value Capture；下轨显示 Framework、Training、Serving、Debug/Profiling 迁移摩擦；在采用决策处汇合。
- 关键结论：软件战略首先需要缩短 workload→hardware 路径。
- 证据模式：源文档战略分析。
- 图前导语：硬件价值并不直接由软件占有率产生，而取决于工作负载能否低摩擦进入硬件。
- 双语：Workload、Complement Friction、Hardware Value Capture 配中文说明。
- 重复控制：图承担因果结构，正文只解释战略含义。

### FIG-03｜一个 PyTorch 模型携带的是资产系统，而非 API 列表

- 章节：`asset-first`
- 来源：第 3 节模型资产树及 Framework 迁移论述。
- 读者障碍：无法直观看到 Framework replacement 的实际影响面。
- 视觉问题：哪些语义和资产需要保留，哪些边界才应该由硬件适配承担？
- 类型/角色：资产定义、实例路径与迁移边界；Mechanism。
- 设计参考：采用参考 FIG-03 的多层结构。
  1. 顶部四类资产定义：模型语义、训练状态、工程工具、生产系统。
  2. 中部给一个“已有 PyTorch workload”小路径。
  3. 设置“应保留 / 应适配”边界。
  4. 下方比较 Framework replacement 与 Hardware adaptation 的影响范围。
  5. 底部矩阵对齐代码、Checkpoint、分布式、Profiling、Serving 等字段。
- 关键结论：迁移影响贯穿整个工程体系，远不只是 API 替换。
- 证据模式：源文档定性结构。
- 图前导语：把模型周围的隐性资产展开后，Framework migration 的实际范围才清楚。
- 双语：Checkpoint、Precision Policy、Profiling、Serving 保留英文并配中文说明。
- 重复控制：资产清单由图承担，正文不再保留长列表。

### FIG-04｜垂直全栈与水平可组合生态的替换边界不同

- 章节：`asset-first`
- 来源：第 4 节三组栈图。
- 读者障碍：两种生态模式看起来都有完整链路，差异不够显性。
- 视觉问题：每一层能否被独立替换，控制权由谁拥有？
- 类型/角色：同尺度双路径与差异矩阵；Comparison。
- 设计参考：参考 FIG-03 的 GPU/Ascend 双 lane。
  1. 左侧 Horizontal Ecosystem。
  2. 右侧 Vertical Stack。
  3. 使用相同层级与对象顺序。
  4. 中间标出替换边界。
  5. 底部比较 Framework、Training、Serving、Backend 的所有权和替换成本。
- 关键结论：产品完整性可能转化为组合摩擦。
- 证据模式：源文档结构与分析。
- 图前导语：两种架构都能形成完整链路，关键区别是每一层能否被独立替换。
- 双语：Horizontal Ecosystem、Vertical Stack、Control Plane 配中文说明。
- 重复控制：三张原始 ASCII 栈图合并为一张。

### FIG-05｜MindSpeed 将 Ascend 优化注入 Megatron，而不接管 Framework

- 章节：`training-interface`
- 来源：第 5 节及 ROLL 案例。
- 读者障碍：MindSpeed 与 Megatron、torch_npu、CANN 的关系容易被误解为另一套训练框架。
- 视觉问题：Ascend-specific optimization 从哪里注入、哪些依赖保持可选？
- 类型/角色：主流程 + Integration Passport；Mechanism/Worked example。
- 设计参考：参考 FIG-08。
  1. 上方：Megatron→MindSpeed adapter→并行/通信/Kernel 优化→torch_npu→CANN。
  2. 下方 Passport：Framework Ownership、Optionality、CUDA/CPU Impact、Version Contract、Evidence。
  3. 侧边案例：ROLL 的训练与推理双路径。
- 关键结论：能力进入已有 Framework，且不把 adapter 变成新的控制主干。
- 证据模式：官方 MindSpeed 资料；ROLL 约束待核实。
- 图前导语：训练侧的关键是在保留既有训练主干的前提下建立稳定优化注入点。
- 双语：Adapter、Parallel Optimization、Communication Optimization、Kernel Optimization 配中文说明。
- 重复控制：一行 import 示例单独保留，图只负责体系关系与契约。

### FIG-06A｜完整推理引擎并置时产生控制面冲突

- 章节：`inference-control`
- 来源：第 6 节。
- 读者障碍：容易把功能重叠误解为普通兼容问题。
- 视觉问题：Scheduler、KV Cache、PD 和 Parallelism 的所有权在哪里冲突？
- 类型/角色：责任边界冲突图；Diagnosis。
- 设计：vLLM 与完整 MindIE Engine 两个控制面重叠，使用相同字段对齐，并直接框出重复职责。
- 关键结论：问题首先是控制权边界，而不只是模型和功能支持。
- 证据模式：源文档分析。
- 图前导语：两个推理引擎都能执行模型并不代表能够自然叠加，控制面重叠才是核心矛盾。
- 双语：Scheduler、KV Cache、PD Disaggregation、Parallelism 配中文说明。
- 重复控制：正文不再逐项复述全部控制面能力。

### FIG-06B｜MindIE 组件化后可与第三方引擎形成组合关系

- 章节：`inference-control`
- 来源：第 7 节。
- 读者障碍：不容易看出组件化如何解决 FIG-06A 的冲突。
- 视觉问题：服务控制、模型执行和硬件优化如何拆分？
- 类型/角色：组件拓扑；Mechanism。
- 设计：Application→MindIE Motor/其他平台→MindIE LLM 或 vLLM/SGLang→可选 MindIE Turbo→CANN；与 FIG-06A 保持相同对象和尺度。
- 关键结论：Serving、Engine 与 Hardware Optimization 可以拆分并独立组合。
- 证据模式：官方资料与源判断分层标记。
- 图前导语：将服务控制、模型执行与硬件优化拆开后，每个组件才有清晰所有权。
- 双语：Motor、LLM、Turbo 旁增加中文职责说明。
- 重复控制：只显示相对 FIG-06A 发生变化的责任边界。

### FIG-07｜vLLM-Ascend 把硬件能力放进上游定义的扩展点

- 章节：`inference-control`
- 来源：第 8 节。
- 读者障碍：插件模式与复制一套 vendor engine 的差别不够直观。
- 视觉问题：谁拥有上游控制面，硬件后端在哪里分叉？
- 类型/角色：插件分叉图；Correct interface。
- 设计：Application→vLLM→Hardware Plugin→CUDA/Ascend，高亮 vLLM-Ascend；旁边仅用缩略反例表示 vendor-specific engine 路径。
- 关键结论：硬件插件保留上游控制面，并降低 fork 与 API divergence。
- 证据模式：vLLM Ascend 官方文档。
- 图前导语：低侵入兼容依赖上游已经定义的 extension point，并应避免复制上游主干。
- 双语：Hardware Plugin、Upstream、Backend 配中文说明。
- 重复控制：反例只保留轮廓，不重复 FIG-06A。

### FIG-08｜Vendor layer 越厚，版本同步路径越长

- 章节：`release-train`
- 来源：第 9 节三条版本链。
- 读者障碍：版本问题容易被看成某一个组件落后，而不是多边界传播问题。
- 视觉问题：一次上游变化必须穿过多少受控层和版本契约？
- 类型/角色：主依赖流程 + Compatibility Passport；Lifecycle/Evidence。
- 设计参考：参考 FIG-08。
  1. 上方：Model→HF/Megatron→PyTorch→vLLM/SGLang→Plugin→CANN。
  2. 下方 Passport：MindSpeed/Megatron、CANN/torch_npu/PyTorch、vLLM/vLLM-Ascend。
  3. 每组统一字段：Upstream Version、Vendor Dependency、Compatibility Rule、Failure Mode、Evidence。
  4. 未核实版本直接标“需验证”。
- 关键结论：同步成本由变化需要穿过的边界决定，不由产品数量单独决定。
- 证据模式：官方兼容矩阵与源文档结构；不展示虚构时间。
- 图前导语：版本问题取决于上游变化需要穿过的受控边界数量与契约复杂度。
- 双语：Release Train、Compatibility Matrix、Version Contract 配中文说明。
- 重复控制：主图呈现结构，具体版本只进入 Passport 或证据注记。

### FIG-09｜CANN 是多种上层生态共享的硬件能力腰部

- 章节：`backend-waist`
- 来源：第 10—11 节。
- 读者障碍：CANN、torch_npu、TorchAir 和上层 Framework 的职责容易混在一起。
- 视觉问题：哪些语义由上层保留，哪些能力必须在 Ascend 底座重建？
- 类型/角色：多层定义、语义边界、双路径与责任矩阵；Mechanism/Cutaway。
- 设计参考：本报告中最接近参考 FIG-03 的主图。
  1. 顶部：主流生态分别定义模型语义、训练控制和推理控制。
  2. 中部：`可以保持 / 必须转译或优化` 的语义边界。
  3. 双路径：PyTorch→FX→TorchAir→GE；vLLM→vLLM-Ascend→CANN。
  4. CANN cutaway：GE、HCCL、Operator/Kernel、Runtime、Memory。
  5. 底部矩阵：Framework、Adapter、Compiler、Runtime、Hardware 的责任和不可替代性。
- 关键结论：上层生态可以变化，但最终都需要共享的硬件语义转换与执行腰部。
- 证据模式：CANN/TorchAir 官方资料与源文档战略分析。
- 图前导语：判断产品边界时，应先找到所有上层工作负载最终必须经过的共同腰部。
- 双语：FX Graph、GE/Ascend IR、Compiler Bridge、Runtime 配中文说明。
- 重复控制：底层组件列表由图承担，正文只解释战略意义。

### FIG-10｜产品越靠近适配层，生态错位通常越低

- 章节：`positioning`
- 来源：第 12—13 节产品表。
- 读者障碍：九个产品的定位、重叠程度和演化方向难以横向比较。
- 视觉问题：产品位于哪个抽象层，要求用户改变多少既有范式？
- 类型/角色：产品位置矩阵 + Product Passport；Comparison。
- 设计参考：参考 FIG-08 的统一字段结构。
  1. 上方：产品按上层 Framework→Adapter→Foundation 排列。
  2. 下方九个产品使用统一字段：Current Role、Overlap、Misalignment Judgment、Target Direction、Evidence Boundary。
  3. 错位等级明确标为“源文档判断”，不伪装成客观评分。
- 关键结论：越接近 Adapter、Accelerator、Compiler Backend，越符合当前互联网 AI 的消费方式。
- 证据模式：源文档产品表与分析。
- 图前导语：把产品放回抽象层后，“生态规模”问题会转化为更具体的“生态位置”问题。
- 双语：Current Role、Overlap、Target Direction 配中文说明。
- 重复控制：原 Markdown 表由此图完整替代，不再复制第二张表。

### FIG-11｜昇腾软件战略从 Stack Ownership 转向 Capability Provisioning

- 章节：`transition`
- 来源：第 14 节。
- 读者障碍：“软件价值下沉”容易被误解成减少软件投入。
- 视觉问题：主干、可选点和控制边界分别发生了什么变化？
- 类型/角色：同尺度阶段对比与变化矩阵；Evolution。
- 设计参考：参考 FIG-03 的双 lane 与差异表。
  1. 左侧 Vertical Full Stack。
  2. 右侧 Open Hardware Platform。
  3. 使用相同层级、对象和路径顺序。
  4. 直接标出 Mind 组件从强制主干变为可选注入点。
  5. 底部比较控制权、用户入口、差异化位置、适用场景和代价。
- 关键结论：转型通过重构软件的交付位置，使强制主干能力转化为可组合能力。
- 证据模式：源文档综合判断与华为开放声明。
- 图前导语：这次转型重新分配软件能力在用户主路径中的必选性与控制权。
- 双语：Stack Ownership、Capability Provisioning、Optional Optimization Kit 配中文说明。
- 重复控制：阶段一、阶段二两张 ASCII 图合为一张。

### FIG-12｜上层兼容、下层差异化的目标生态模型

- 章节：`target-model`
- 来源：第 15—17 节。
- 读者障碍：目标模型与 MindSpore、MindSpeed、MindIE 的角色建议分散在多节文字中。
- 视觉问题：长期架构如何同时容纳开放生态与 Ascend-native 差异化？
- 类型/角色：目标架构 + Role Passport；Synthesis。
- 设计参考：分层架构结合参考 FIG-08 的 Passport。
  1. 上方：开放创新生态→稳定扩展点→Ascend 适配层→差异化层→硬件。
  2. 下方三个 Role Passport：MindSpore、MindSpeed、MindIE。
  3. 统一字段：Primary Role、Best-fit Scenario、Should Preserve、Should Avoid、Evidence/Inference。
- 关键结论：竞争目标是让 Ascend 成为自然后端，而不是复制同数量的软件产品。
- 证据模式：战略推论，明确标注。
- 图前导语：最终目标是在保留 Ascend-specific 能力的同时，将其放置在用户主动感知与迁移成本最低的位置。
- 双语：Open Innovation Ecosystem、Stable Extension Point、Differentiation Layer 配中文说明。
- 重复控制：三个产品的角色只在 Passport 中总结，不再各画独立产品图。

### FIG-13｜Ascend Reachability：从现有工作流到硬件的四段距离

- 章节：`reachability`
- 来源：第 18 节。
- 读者障碍：“生态可达性”是新提出的战略判断，缺少可执行的评审方式。
- 视觉问题：一个现有 workload 接入 Ascend 时，需要在哪些边界发生改变，证据是什么？
- 类型/角色：主流程 + Reachability Passport；Decision synthesis。
- 设计参考：参考 FIG-08。
  1. 上方：现有工作流→代码改动→系统架构→开发习惯→Ascend-specific 调优→可达性结论。
  2. 下方四张 Passport：Code Surface、Architecture Surface、Workflow Surface、Optimization Surface。
  3. 统一字段：Preserved Asset、Required Change、Ascend Interface、Validation Evidence、Unknown Boundary。
  4. 不生成总分、仪表盘或伪精确指数。
- 关键结论：成功标准是迁移距离更短，同时仍能释放 Ascend 的硬件差异化能力。
- 证据模式：源文档最终判断与显式推论。
- 图前导语：生态可达性不需要先发明一个指数，可以先用四个可审查的迁移边界进行判断。
- 双语：Code Surface、Architecture Surface、Workflow Surface、Optimization Surface 配中文说明。
- 重复控制：结论段只保留一句总判断，其余维度由图承担。

## 6. 图表与正文重复审计

- L0–L4 表、CANN 腰部图和目标模型都涉及分层：分别负责“诊断错位”“解释共同底座”“提出目标架构”，不重复组件说明。
- FIG-04 与 FIG-11 共享层级：FIG-04 比较两种生态架构属性，FIG-11 表达 Ascend 自身的阶段性变化。
- FIG-06A/B 与 FIG-07 共享推理链路：前者负责控制面冲突和组件化，后者只解释 upstream extension point。
- FIG-10 与 FIG-12 都包含产品角色：FIG-10 负责九产品横向定位，FIG-12 只保留 MindSpore、MindSpeed、MindIE 的目标角色。
- 30 组 ASCII 图全部合并进 FIG-02—13，不将同一链路同时复制成图、表和长段落。
- 复杂图沿用参考 FIG-03/08 的内部层级，但不会用更多模块重复同一个结论。

## 7. 资产处理

- 现有图片/SVG：无。
- 现有表格：两张。
  - L0–L4 表重绘为 FIG-01。
  - 产品定位表重绘为 FIG-10。
- 现有 ASCII 图：全部重绘或合并，不逐图原样保留。
- 引用块：外部来源内容保留为方角引用；作者自己的总结改成正文或结论框。
- 不使用生成式图片描绘技术架构。
- 图表以 HTML/CSS 和 inline SVG 为主，不额外依赖外部图片资产。
- 不引入市场份额、性能、迁移周期或成本数字。

## 8. 页面与交互

- 静态、可离线打开的 HTML。
- 使用固定左侧 Reading Path；中宽视口收缩为编号轨道；窄屏隐藏。
- 提供页面滚动进度和返回顶部。
- 不使用密集固定顶部导航与左侧阅读路径竞争。
- 不使用 JavaScript 改变正文或图表的 DOM 顺序。
- 所有正文和图表在禁用 JavaScript 后仍可阅读。
- 宽矩阵和拓扑图允许局部横向滚动，并保留标题、图号、读图起点与 Caption。
- 不使用桌面 scroll snap。

## 9. 计划交付

```text
/Users/yin/PyPTO3-main/Insight/Ascend-ecosystem-whitepaper/
├── index.html
└── shared.css
```

若生成阶段需要额外本地媒体，才创建 `assets/`；当前计划不需要外部图片。

## 10. 最终验证

### 10.1 结构与来源

- 每个源标题、实质论点、案例、限定和结论均映射到报告正文或获批图表。
- 所有导航锚点、相对路径和来源链接有效。
- 官方事实、源文档判断、推论和待核实内容在视觉上可区分。
- 13 条参考资料均保留；可验证者补充直接链接。

### 10.2 编辑质量

- 独立扫描所有章节和图表标题，确保脱离上下文仍然清晰。
- 每张实质图前都有紧邻的 `.figure-intro`。
- 检查正文、表格、图表是否重复相同字段和结论。
- 检查所有读者可见英文技术标签是否有中文对应说明。
- 清理“编辑注释”“原文档”“恢复”“删掉的图”等维护叙述。

### 10.3 图表与布局

- 运行 `node scripts/audit_whitepaper.mjs <report-path>` 并处理所有错误与相关警告。
- 检查图 3/图 8 风格组件的层级、对齐、边界标记和 Passport 字段。
- 检查复杂图在目标宽度下的标签碰撞、文字溢出、连线穿字和焦点错误。
- 确保正文和导航为 14px 基线，任何读者可见文字不低于 11px。
- 桌面标准视口检查 Cover、密集正文、宽图、产品矩阵和最终综合图。
- 窄屏检查 Reading Path 隐藏、图表横向滚动和 Caption 完整性。

## 11. 审批状态

白皮书网页已按本计划生成：

- `Ascend-ecosystem-whitepaper/index.html`
- `Ascend-ecosystem-whitepaper/shared.css`

## 12. 内容完整性复核

根据生成后与原始 Markdown 的逐章复核，成稿已补充以下被过度压缩的内容：

- MindSpore“南向亲和昇腾、北向生态兼容”的演进方向、生态治理边界与差异化适用场景。
- Alibaba ROLL Ascend RFC 提出的训练、推理和 CUDA 共存三条路径，以及 optional、非 hard dependency、不改变既有行为与 Megatron 架构等约束；成稿明确区分 RFC 设计与当前发布版状态。
- vLLM / SGLang 与完整 vendor engine 在批处理、缓存、推测解码、PD、专家并行、调度和 Serving 等职责上的控制面重叠。
- vLLM-Ascend 的上游治理位置、版本对应与持续集成责任。
- Ascend Extension for PyTorch 覆盖原生 API、MindSpeed、TorchAir、第三方库、自定义算子、性能优化与迁移工具的完整入口。
- Vertical Full Stack 与 Open Hardware Platform 的长期双路径关系，以及默认入口、控制权和兼容责任的重新分配。
- 以 ROLL RFC 为例的 Reachability 四表面评审，并显式保留原始材料没有提供的性能、生产状态与发布版实现证据。
