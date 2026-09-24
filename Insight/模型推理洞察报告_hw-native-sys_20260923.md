# hw-native-sys 模型推理洞察报告

> 整理日期：2026-09-23
> 资料范围：`hw-native-sys` 组织下 11 个公开仓库的全部 Issue —— 本地归档 1,471 条（抓取于 2026-08-17）+ 实时增量 565 条（2026-08-17 ~ 2026-09-23，其中新建 437 条）
> 标注约定：**【实录】** 数据和结论直接来自 Issue 正文或评论，可回溯到具体条目；**【推断】** 由多条来源交叉得出，来源本身没有直接写出
> 相关文档：性能调优的方法学与单点案例见 [PyPTO 性能调优典型用户案例](PyPTO性能调优典型用户案例_基于GitHub_20260923.md)；仓库职能划分见 [hw-native-sys 仓库功能洞察报告](hw-native-sys-仓库功能洞察报告.md)。本文不重复这两份内容，聚焦「模型推理」这条主线上的实践案例与使用者问题。

---

## 一、结论摘要

1. **推理已经是这个组织的主线工作，而不是某个仓库的子课题。**【实录】按标题筛选，1,471 条归档中 245 条与推理直接相关；8-17 之后新建的 437 条中有 80 条命中，另有 53 条标题未命中但正文高度相关。`pypto-serving` 全部 39 条（归档期）均为推理主题。

2. **瓶颈已经从「算子快不快」转移到「执行模型对不对」。**【实录】这是本报告最重要的发现，证据来自 `pypto-lib#1054`：在匹配形状下，pypto-lib 完成同一个 decode attention 子层只用了参考实现 **2.7–3.0 倍更少的核心微秒**，却花掉 **1.5–1.7 倍的 wall clock**。根因是每条依赖边的派发成本 —— 参考实现 **32.6 ns/边**，pypto-lib **13.5 µs/边**，后者与该芯片上 host-launched eager 算子处于同一量级（p50 20.4 µs）。设备空闲 0.2–0.3% 对 18–25%，cube 核占用 70–71% 对 19–22%。

3. **围绕上一条，三条路线级动作在同一个月内同时启动。**【实录】`pypto#2399` 把 orchestration 编译成 host_build_graph 图执行（录一次、逐层回放，已于 2026-09-20 关闭）；`simpler#2429` 把 Simpler 做成 vLLM 的程序执行后端；`pypto#2857` 打通 ONNX → ATC → OM → ACL 离线推理，脱离 PyPTO runtime 部署。三者指向同一个判断：**在线 JIT + eager 逐算子派发不足以支撑生产推理**。

4. **模型谱系在 8-17 之后跳了一代，并首次走出 DeepSeek / Qwen。**【实录】新增 DeepSeek-V4.1-Flash（`pypto-lib#1205`，40 层混合 attention + Hierarchical Sparse Indexer + Engram）和 GLM-5.3-Flash（`pypto-lib#1267`，TP16/EP16，45 层，288 routed experts）。硬件重心从 a2a3 迁往 A5 / Ascend950PR / Ascend950DT，量化路径从 W8A8 INT8 扩展到 MXFP8/MXFP4。

5. **使用者问题的重心换了。** 归档期以功能打通、挂死和精度为主；8-17 之后前三位变成 **输出非确定性**、**Host 侧开销**、**可观测性自身不稳**。【实录】固定 seed 仍然发散（`pypto-lib#1122`）、两次相同请求返回不同文本（`pypto-serving#183`）、decode 每步 13–20 ms 花在 Host 构表（`pypto-serving#215`）。

6. **serving 正在按 vLLM 的生产特性逐条对齐，对齐过程本身暴露了设计债。**【实录】PD 分离（`pypto-serving#229`）、Mooncake 外部 prefix cache（`#182`）、滑窗 KV 语义（`#218`）。其中 `#218` 指出当前滑窗实现用「紧凑环形表 + 取模覆写」，丢失了逻辑位置，并强制 `sliding_window % token_capacity == 0`，导致合法配置直接失败。

---

## 二、资料来源与方法

| 来源 | 范围 | 时间 | 用途 |
| --- | --- | --- | --- |
| `github_issues/*/*_issues.md` / `.json`（本地归档） | 11 个仓库共 1,471 条 Issue | 2026-08-17 | 全量关键词粗筛，得到 245 条推理相关候选 |
| GitHub REST API（`tools/gh/gh.exe`） | `state=all&since=2026-08-17`，剔除 PR | 2026-09-23 | 增量 565 条（新建 437 条），按同一词表二次筛选 |
| 关键条目的 Issue 评论 | `pypto-lib#622`、`#1054`，`pypto#2040` | 2026-09-23 | 核对结论是否在评论中被修正 |

**筛选词表**：`infer / vllm / serving / decode / prefill / kv cache / prefix cache / throughput / latency / TTFT / TPOT / qwen / deepseek / dsv4 / attention / MoE / MLA / MTP / specul / paged / sampling / 推理 / 部署 / 吞吐 / 时延`。标题命中后，对未命中标题的条目再做正文命中计数（阈值 ≥ 6 次）补捞。

**局限**：

- 归档期的 Issue 评论只有 `pypto-lib/detailed/` 一份，其他仓库仅对本文引用的关键条目实时补读了评论。**这是本报告最主要的风险来源**，原因见第六节。
- 所有性能数据都绑定特定 commit、模型形状与硬件，只能说明方法和量级，不能当作性能承诺。
- 附件图片和 trace 未下载，结论以正文文字为准。
- 增量抓取的原始数据未入库，存放在会话临时目录；如需复现，方法见附录 B。

---

## 三、推理议题全景

### 3.1 分布

| 仓库 | 归档期推理相关 | 归档期总数 | 8-17 后新建推理相关 | 在推理链路中的角色 |
| --- | ---: | ---: | ---: | --- |
| pypto-lib | 89 | 105 | 34 | 模型与算子实现，主战场 |
| simpler | 50 | 315 | 16 | 运行时、派发、泳道 profiling |
| pypto | 45 | 632 | 9 | 编译器 codegen 与 pass |
| pypto-serving | 39 | 39 | 14 | 服务化，100% 推理主题 |
| PTOAS | 17 | 301 | 3 | 汇编与后端 |
| pto-isa | 5 | 76 | 3 | 指令集、CPU 模拟器 |

### 3.2 推理问题的五个层级

同一个现象可能出自任一层，这是排障成本高的直接原因。【推断，依据多条 Issue 的归因路径】

| 层级 | 典型问题 | 代表条目 |
| --- | --- | --- |
| 服务链路 | 调度、构表、bind/H2D、prefix cache 语义 | `pypto-serving#215`、`#218`、`#182` |
| 模型 / 算子 | 精度、KV 契约、并行策略 | `pypto-lib#1133`、`#1043`、`#511` |
| 运行时派发 | 依赖边成本、跨 rank 抖动、图执行 | `pypto-lib#1054`、`pypto-serving#179`、`pypto#2399` |
| 编译器 / ISA | codegen、pass、同步插入 | `pypto#2558`、`pto-isa#282` |
| 模拟器 / 可观测性 | 模拟器与真机不一致、泳道数据缺失 | `pto-isa#289`、simpler chip_swimlane 系列 |

值得注意的是 `pypto-lib#858` 的自述：作者明确写「放在这里而不是 pto-isa，是因为我还无法归因到具体层级，不想让人去错的仓库追查」。**分层归因困难本身已经成为一种被使用者意识到的成本。**【实录】

---

## 四、实践案例

### 4.1 端到端打通（bring-up tracking）

这类 Issue 有统一的写法：明确阶段范围、明确不做什么、给出拓扑与判定标准。是仓库里质量最稳定的一类文档。

| 条目 | 模型 / 拓扑 | 关键约束记录 |
| --- | --- | --- |
| `pypto-lib#135` | Qwen3-14B 单卡 | 选 14B 因为 BF16 下约 28 GB 权重可单卡放下，且 `q_size == hidden_size` 免去 Q/O 维解耦重构 |
| `pypto-lib#156` | DeepSeek-V3.2，16 卡 DP2+TP8+EP16 | 无原生 FP8，主用 BF16 / W8A8 INT8；节点内 HCCS，节点间 RoCE v2 |
| `pypto-lib#1205` | DeepSeek-V4.1-Flash，A5 八卡 TP4/DP2/EP8 | 40 层混合 attention（0–1 层滑窗）、Hierarchical Sparse Indexer；一阶段排除视觉、Engram、DSpark、DSA-CP |
| `pypto-lib#1267` | GLM-5.3-Flash，a2a3 TP16/EP16 | 官方 checkpoint 为 FP8 blockwise，而 `Ascend910_9392` 的 `Intrinsic_mmad` 只有 `s32s8s8`/`u32u8u8`/`u8s8` 及 fp16/fp32 形式，**FP8 无法执行**，只能走 W8A8 转换 |

**共性做法**：第一里程碑一律定为「文本 prompt 经 prefill 和连续 decode，greedy sampling 稳定输出非空文本」，而不是某个算子通过。【实录】

### 4.2 外部基线复现：`pypto-lib#1054`（本报告重点）

这是整批资料中最接近「真实使用者上手实录」的一条，标签为 `enhancement` + `help wanted`。它复现的是 CANN 公开参考实现 `cann-recipes-infer` 的 DeepSeek-V4-Flash，在 8× Ascend 950PR 上。

**结果**【实录】

- 全网络 8 卡端到端跑通并生成正确文本，8K 与 128K 上下文均可。
- decode **19.6 ms/step @ 8K, batch 8**；**24.9 ms/step @ 8K, batch 64（321 tok/chip/s）**，Hybrid MXFP8-MXFP4 checkpoint + `npugraph_ex`。

**上手成本**【实录】

- 作者明确写 **"none of this is in the READMEs"**：950PR 没有发布容器镜像（`ops/pypto_python/README.md` 至今写「待后续发布」），8p 环境无 docker，整栈只能源码构建。
- 数条上游缺口需要绕行，作者认为其中第一条是 `cann-recipes-infer` HEAD 的真实 bug。
- **MTP 无法从公开产物复现**，因此华为已发布的 DSv4-Flash benchmark 行（全部为 MTP1 / MTP3）都无法按原样复现，本文数据为 `next_n = 0`。

**对标结论**（2026-09-03 刷新，pypto-lib main `9368299`，recipes 侧 `3c92667`）【实录】

| decode attention | recipes µs/层 | pypto-lib µs | pypto 相对水平 | 立项时 |
| --- | ---: | ---: | ---: | ---: |
| SWA | 180.6 | 309.3 | 58.4% | 52.0% |
| CSA | 274.0 | 421.3 | 65.0% | 57.6% |
| HCA | 196.8 | 339.7 | 57.9% | 51.5% |

**反直觉的部分**：核心时间的对比方向相反。

| | recipes core-µs | pypto-lib core-µs | 比值 |
| --- | ---: | ---: | ---: |
| CSA | 17,366 | 5,760 | 3.01× |
| HCA | 11,438 | 4,263 | 2.68× |
| SWA | 9,426 | 3,176 | 2.97× |

作者的原话是：**「我们用少 2.7–3.0 倍的核心微秒完成同一个子层，却仍然花掉 1.5–1.7 倍的 wall clock。」** 拆开看：

- 设备空闲 **0.2–0.3%（recipes）对 18–25%（pypto）**；36 个 cube 核的 AIC 占用 **70–71% 对 19–22%**。
- `decode_indexer_compressor` 链在 CSA 程序内只花约 95 core-µs，而对方单个融合 `indexer_compressor` 要 2,063 —— **便宜 22 倍，仍然输掉 wall clock**。
- 机制是**每条依赖边的派发成本**。recipes 用 `aclmdlRICaptureBegin/End`（`exe_mode: npugraph_ex`）把 43 层整网捕获成设备常驻实例，每步一次 host 调用：3,509 个 device task 只花 29–31 µs，即 **8.3 ns/task**，捕获模型内**零 AI_CPU task**；一条依赖只是有序队列里的程序序，实测 **32.6 ns/边**，且不随生产者宽度变化。pypto-lib 则是 **13.5 µs/边**。作者定性为：「这不是调度器慢，这是每条边上的 eager 逐算子派发。」

**方法学上值得学习的地方**【实录】：这条 Issue 自带三重自我纠错 —— 在同一天内撤回了自己当天发布的 provisional caveat，并说明为何原归因看似成立：一次 pin 刷新会同时移动 pypto、simpler、pto-isa，没有任何单一 commit 被隔离过；且精度门是边际而非二元的（`x_out` 在 `ratio=0.8072%` 对 `allowed<=0.8000%` 处失败），所以在无 seed 的 fixture 上，通过与否是一个**比率**，n=3 无法区分「构建坏了」和「门本来就边际」。它还记录了 recipes 侧在 09-01 重采时复现到 0.2% 以内，因此是稳定参考。

### 4.3 自研性能验证报告

| 条目 | 对象 | 方法上的特点 |
| --- | --- | --- |
| `pypto-serving#95` | DeepSeek V4 Flash W8A8，8 卡 TP=8 | 把耗时拆成 prefill / decode / bind+H2D / Device wall / D2H / LM head / sampling 七段；明确声明**不用泳道图作为耗时依据**，改用 Runtime STRACE；明确写「本报告是未含权重与 Cache 驻留优化的基线版本」 |
| `pypto-serving#43` | Qwen3-14B 全流程 | 性能 × 精度 × 稳定性三维矩阵（seq 256→2048，并发 1→32，3 天常稳）；精度真值要求用 AISBench 在 vLLM-Ascend / HF 上、以**相同 prompt 模板与采样参数**测得；多 batch 逐 token 一致性列为硬性要求 |
| `pypto-lib#1233` | DeepSeek-V4-Flash MXFP8/MXFP4 on A5 | 明确写「合成 kernel 或 token loop 跑通，不等于 checkpoint 保真、生成文本有意义或全网吞吐成立」 |
| `pypto-lib#1340` | DeepSeek-V4.1-Flash on Ascend950DT | 立项理由写得很直白：「功能绿不够，其他团队在这个栈上性能经验更少，所以现在就需要一本可审计的性能台账，不能等到后面的优化阶段」 |

### 4.4 serving 能力建设

| 条目 | 能力 | 实现要点与暴露的难点 |
| --- | --- | --- |
| `pypto-serving#229` | PD 分离（chunk-wise Prefill/Decode disaggregation） | Router / P / D 三独立进程，destination-first 预留，Host 管理的 chunk-wise P-push；Mooncake Transfer Engine / AscendDirect 为首个 Device 内存后端，续传数据从 P 的 HBM 直达 D 的 HBM，不经 Router 或 Host。已完成真实 1P1D 验证；prefix cache 的两节点 D2D 验收门仍未过 |
| `pypto-serving#182` | Mooncake 外部 prefix cache（HBM / DRAM / SSD 分层） | DeepSeek V4 的 KV 由 `ori`、`cmp_c128`、`cmp_c4`、`idx`、`hca_state`、`csa_state`、`csa_inner_state` 七类异构 group 构成，block size、压缩率、层数、滑窗语义各不相同，**通用单一 KV block offload 方案不能直接用** |
| `pypto-serving#167` | 分组 prefix cache 零拷贝说明 | `cmp_c128` 每 128 源 token 存 1 行，`cmp_c4` 和 `idx` 存 32 行；`--block-size 128` 只是主 prefix cache 的逻辑匹配粒度，不代表所有 cache 的物理块都有 128 行 |
| `pypto-serving#218` | 滑窗 KV 语义对齐 vLLM | 当前用紧凑环形 block table + 取模覆写，一是强制 `sliding_window % token_capacity == 0`（block=4 / window=6 这类合法配置直接失败），二是丢失完整逻辑位置；vLLM 按逻辑位置索引、释放滑出块并写 `null_block` |
| `pypto-serving#211` | DSpark 投机解码接入 | M1 target-only（verify 行填 noise token，anchor 行接受 → 1 token/step），M2 三层 drafter + Markov head 链，K=7。已于 2026-09-11 关闭 |
| `pypto-serving#7` | 与 vLLM 特性对比 | 已实现：continuous batching、chunked prefill、paged KV、prefix caching、抢占、SSE 流式、OpenAI 兼容 API、多进程 worker、greedy/temperature/top-k/top-p |

### 4.5 调优方法论沉淀

`pypto-lib#828`（**已于 2026-09-07 关闭**）索引了两份一手资料：Qwen3-14B decode 优化语法图鉴（按算法 / 核内 / 调度三层整理 15 类优化语法）与 DeepSeek-V4 泳道图调优记录（20 组实验，每组给出改动前后代码、原因、性能、精度、泳道图和失败尝试）。作者主动标注了适用边界，并提示「后续章节会修正早期判断（如 §17 更正 §16），以靠后的结论为准」。详细拆解见姊妹文档 [PyPTO 性能调优典型用户案例](PyPTO性能调优典型用户案例_基于GitHub_20260923.md)。

---

## 五、使用者问题图谱

### 5.1 输出非确定性（8-17 后的头号问题）

| 条目 | 现象 |
| --- | --- |
| `pypto-lib#1122` | DeepSeek V4 Flash W8A8 在 prompt、采样参数、seed 全部固定时仍间歇产生不同补全；部分发散运行在第 30 个输出 token 从故宫描述跳进无关的购物商品 JSON。`--num-speculative-tokens` 为 1 和 0 都能复现，因此**不是 MTP 专属缺陷**，指向公共 decode / sampling 路径 |
| `pypto-serving#183` | 两次完全相同的请求返回不同文本（`prompt_tokens` 同为 302），且**方向在不同 run 之间翻转**。作者特意说明立 Issue 的理由：该用例占 8 卡约 15 分钟，而且症状会误导人去怀疑自己 PR 里的权重加载或调度改动 |
| `pypto-lib#1043` | EP8 下跨 rank logits 出现有限漂移，在 near-tie 位置翻转 argmax |
| `pypto-lib#368` | `decode_indexer` 的 `topk_idxs` 在 score 仍在容差内时不确定（排序 tie-break） |
| `pypto-lib#951` | DeepSeek-V4-Flash 长 decode 漂移，归因到 RoPE profile 不匹配 + split-K 归约不确定 |

**这一类的共同特征**：现象出现在服务层，根因散落在算子、归约顺序、跨 rank 通信三处，且都需要多次重复运行才能观察到。【推断】

### 5.2 Host 侧开销成为 decode 瓶颈

| 条目 | 数据 |
| --- | --- |
| `pypto-serving#215` | `prepare_early` 每 decode step 均值 **20.1 ms**（中位 17.5）；加单块 ring 行的非缓存快路径后降到 **13.4 ms**，仍是主要 host 成本。主要不是 `copy_shared`（0.104 ms/rank），而是 `rank_build_tables`（1.754 → 1.085 ms/rank × 8 rank） |
| `pypto-serving#198` | 常驻解码元数据导致 `prepare_decode` 耗时显著增长 |
| `pypto#2629` | DP8 TaskArgs 打包占 13.4 ms decode `worker_submit` |
| `pypto#2532` | DP8 Tensor wire 转换每 rank 多约 9 ms |
| `pypto-serving#179` | decode step 延迟跨 rank 不稳：device 执行对齐，慢的是 host 侧调度与 run 生命周期；**慢的 rank 和慢的阶段每次都在变**（`pre_bind` / `runner_run` 之前 / `runner_run` 与 `validate` 之间 / `validate` 内部）。作者提醒「只盯一个 span 会把瓶颈搬家而不是消除」 |

这一组与第四节 `pypto-lib#1054` 的依赖边派发成本是同一个问题的两端：**一端是每条边 13.5 µs 的设备侧派发，另一端是每步 13–20 ms 的主机侧准备**。【推断】

### 5.3 挂死、死锁与超时

`507018`（AICPU / 堆环）是出现最密集的错误码。代表条目：长 prompt 缺 token 级 chunked prefill 触发 task-ring 死锁（`pypto-serving#91`）、prefix cache 冷命中序列崩 Qwen3-14B prefill（`#41`）、MoE EP 多 rank 落在同卡两 die 上 barrier 挂死（`pypto-lib#502`）、Qwen3-14B fused decode 堆环死锁（`pto-isa#147`）、DSpark `l3_prefill_fwd` 在 EP16 首层 MoE `dispatch_meta` 间歇 `TENSOR_WAIT_TIMEOUT`（`pypto-lib#1213`）。

### 5.4 平台与规模泛化

- 仿真与真机不一致：`pto-isa#289`（DSpark HCA 在 a2a3sim / a5sim 数值错误，真机 A2/A3 通过）、`pto-isa#288`（模拟器任务永久 stall）、`simpler#900`（a2a3sim 通过但上板超时或部分零输出）。
- 跨代硬件：`PTOAS#485`、`pypto-lib#110`（A3 通过 A5 挂死）。
- 规模边界：`simpler#1022`（≥ 8192 序列失败）、`pypto-serving#155`（batch-32 在 64-token prefill 失败）。

### 5.5 容量

`pypto-lib#1106`：DSv4-Pro 的 `decode_fwd` / `prefill_fwd` 每 rank 需要 **180 GiB 常驻专家权重**，在 128 GiB 的 Ascend 950 卡上 OOM（`rtMalloc` 请求 64,474,841,088 字节）。Pro 预设为 `hidden_size` 7168、`moe_intermediate_size` 3072、61 层、每 rank 48 个 routed expert。作者主动把归因写在本仓库自己的 layer/expert stacking 上，并明确说明「运行时报错正确且及时，pypto、simpler、ptoas 这里没有什么需要修的」。

### 5.6 可观测性自身不稳

8-17 之后 simpler 出现约 15 条 `chip_swimlane` 重构期缺陷：间歇不落盘（`#2220`、`#2390`）、L3 跨 rank 合并无输入（`#2232`）、fanout 边对不上（`#2191`）、a5sim 段错误（`#2206`）、orch-phase 池按首次 run 的 level 定尺寸导致后续 level-4 run 无记录（`#2375`）。归档期还有 `simpler#860`（tensor dump 跟不上 paged_attention 64bat/8192ctx，host 收集器排空反而成为 kernel hang 根因）。

**影响**：上述所有性能归因工作都依赖这套工具链。`pypto-lib#1054` 的 §8.2 就因为 `_DfxOpts` 自 `f1bb0860` 起从 pypto 移除、而 lib 的 `golden/runner.py` 仍在 import 它，导致**当前 main 对 main 时所有 DFX 开关全部失效**，无法在新工具链上重采。【实录】

---

## 六、一个被反复修正的结论：「2.3 倍差距」事件

这一节单独列出，因为它同时是本报告最重要的方法学教训，也是对本报告自身可信度的说明。

**事件经过**【实录】

1. `pypto-lib#622` 正文（2026-06-26）给出：同形状下 CCE `spmd_paged_attention_highperf` 约 548 µs，pypto `fa_fused` 约 1263 µs，**差距 2.3×**。
2. 次日评论指出**两侧量的根本不是同一个范围**：CCE 侧是独立的 paged attention kernel（QK → softmax → SV → reduce），而 1263 µs 是**整个 Qwen3-14B decode 层**的 on-core busy —— 该层下降为约 35 个 kernel，其中只有 `fa_fused` 和 `online_softmax` 是注意力计算，RMSNorm、QKV 投影、RoPE、out-proj 和整个 MLP 都被算进去了。只取注意力部分约 350 µs，**低于**对方的 548 µs。
3. 2026-06-29 的重测把两侧放在同一 runtime、各自最优 tiling 下对比：pypto 注意力约 **326 µs**（0.82 TB/s，HBM 峰值的 55%），CCE 最优约 **343 µs**（0.78 TB/s，52%），**比值约 0.95×，基本持平**。早先「快 29%」的说法则是因为 CCE 被错误 tiling（runtime 启发式把 `b16/s4096` 路由到慢的长序列 KV-split 路径，约 462 µs）。
4. `pypto#2040`（「fa_fused 比 CANN FAI 慢约 2.3×，归因 A2/A3 的 C2V/V2C GM 往返」）在后续评论中同样被修正：L2 对比显示 PyPTO 注意力核心**略短于**融合的 CCE 注意力核心，剩余的整层差异在 RoPE 融合与调度边界上；原先「GM 往返」的宽泛诊断被作者自己推翻。该 Issue 于 2026-08-27 关闭。

**教训**【推断】

- **只读 Issue 正文会得出与作者最终结论相反的判断。** 本报告在成稿前对引用的关键条目逐条补读了评论，但归档数据中其他仓库的评论未覆盖，因此第五节引用的部分条目仍存在同类风险。
- 这个组织的工程文化里，**结论被自己推翻是常态而非异常**：`pypto-lib#828` 的调优日志提示「以靠后的结论为准」，`pypto-lib#1054` 当天撤回当天的 caveat，`pypto#2040` 作者推翻自己的归因。这既说明数据质量高，也说明**任何基于快照的二手分析都必须标注时间戳**。
- 三次修正的根因是同一个：**统计范围不一致**（层对 kernel、默认 tiling 对最优 tiling、单次读数对多次重复）。这正是 `pypto-serving#28` 想解决的问题的另一种形态 —— 该 Issue 指出 `throughput (e2e)` 把一次性 prefill warmup 混进稳态 decode，使 3.19 tok/s 的报数比真实稳态约 22 tok/s 低了约 7 倍。**口径问题在这个栈上反复出现，且每次都足以反转结论。**

---

## 七、两条路线级动向

### 7.1 执行模型：从 eager 派发走向图捕获

`pypto#2399`（2026-08-18 提出，2026-09-20 关闭）：把重复的 orchestration 区域（典型为一个 transformer decoder 层）编译成 `host_build_graph` 图执行，codegen 产出一个具名 `void(const CoreTaskArgs&)` 函数加每个调用点一次 `rt_submit_graph(...)`；运行时首次调用录制子 DAG，之后回放缓存的 Definition。**N 个相同层从 N × 每层节点数降为一次录制加 N-1 次回放，外层 ring 任务槽从 N × nodes 降为 N。** IR 载体为新的 `FunctionType::Graph`。

这条 RFC 与 `pypto-lib#1054` 的量化结论严丝合缝：对方的 32.6 ns/边正是图捕获带来的，而 pypto 的 13.5 µs/边正是 eager 派发的代价。【推断】

配套还有 `pypto#2653`（统一持久化 JIT 编译缓存，目标之一是让 pypto-serving 删掉自己重复的缓存实现，并让 **vLLM worker 共享兼容产物**）。

### 7.2 生态位：同时向上接 vLLM、向下出 OM

- `simpler#2429`（2026-09-23 新建）：**把 Simpler 确立为 vLLM 的程序执行后端**。从 Pipeline A 的 run-to-run 入队能力出发，终点是固定 Qwen 负载跑通 vLLM eager adapter，再做 capture/replay。设备在算子级仍串行，但 host 可以在前一个 run 执行期间准备并入队后续 run。
- `pypto#2857`（2026-09-21 新建）：**离线图导出**。动机写得很直白 —— PyPTO 目前是在线 JIT 框架，推理必须依赖 PyPTO runtime 在场，而生产与边缘的标准部署路径是 ONNX → ATC → OM → ACL（`aclmdlExecute`），「用 PyPTO DSL 开发的网络被锁死在在线执行形态」。验收标准包含导出的 ONNX 能用 onnxruntime 做 sanity check、能干净转成 `.om`、并且离线模型性能不显著劣于在线 JIT。

这两条合起来是一个清晰的战略姿态：**短期作为 vLLM 的后端进入现有推理生态，长期提供脱离自身 runtime 的部署形态。**【推断】

---

## 八、8-17 前后的变化

| 维度 | 归档期（至 2026-08-17） | 增量期（08-17 ~ 09-23） |
| --- | --- | --- |
| 主力模型 | Qwen3-14B / 32B、DeepSeek-V3.2 / V4 | DeepSeek-V4-Flash / V4-Pro / **V4.1-Flash**、**GLM-5.3-Flash** |
| 主力硬件 | a2a3（910B/910C） | **A5 / Ascend950PR / Ascend950DT** |
| 量化 | W8A8 INT8 为主 | 扩展到 **MXFP8 / MXFP4** 混合 |
| 问题重心 | 功能打通、挂死、精度 | **非确定性、Host 开销、可观测性** |
| serving 议题 | 基础能力（continuous batching、paged KV、prefix cache） | **PD 分离、外部分层 cache、投机解码、与 vLLM 语义对齐** |
| 执行模型 | 在线 JIT，逐算子派发 | **图捕获回放、vLLM 后端、离线 OM 导出** 三线并进 |

**Qwen3 线的状态变化值得单独指出**：`pypto-serving#7`、`#28`、`#43`、`#95` 四份 Qwen3/serving 验证与对比报告自 2026-08-17 起**全部无更新**（仍为 open）；增量期 Qwen3 相关新建条目几乎只剩 `pypto-serving#201`（decode padding 在动态 batch 下竞争非活跃行）。而 `pypto-lib#828`、`#622` 与 `pypto#2040` 三条 Qwen3 性能条目分别在 09-07、09-07、08-27 关闭。**Qwen3 已从主攻目标退为回归基线。**【推断】

---

## 九、对产品与体验的启示【推断，供规划参考】

1. **「口径」应该是工具的一部分，而不是纪律。** 三次结论反转（`#622`、`#2040`、`#28`）全部源于统计范围不一致。可考虑在 profiling 输出中强制携带范围元数据（这是层还是 kernel、是否含 warmup、tiling 是默认还是最优、重复次数），让不可比的两组数据无法被并排放进同一张表。

2. **非确定性需要一等公民的诊断能力。** 目前使用者只能靠「跑 16 次看方向是否翻转」（`pypto-serving#183`）来判断。归约顺序、跨 rank 通信、tie-break 三类不确定性源需要可开关的确定性模式和逐 token 的分歧定位。

3. **归因分层应该被工具承担，而不是由提 Issue 的人承担。** `pypto-lib#858` 的作者明确表示无法判断该报到哪个仓库。跨仓库的一次性诊断入口（给定一次失败运行，指出最可能的层级）会直接降低这类成本。

4. **上手文档与真实环境的差距值得单独治理。** `pypto-lib#1054` 的「none of this is in the READMEs」是最直接的证据：没有 950PR 镜像、必须源码构建、公开 benchmark 因 MTP 产物缺失而不可复现。可复现的 enablement recipe 本身就是高价值内容。

5. **可观测性工具链需要与被测对象解耦的版本契约。** DFX 开关因 `_DfxOpts` 被移除而在 main 对 main 时全线失效，直接阻断了性能重采。

---

## 附录 A：推理相关条目索引

**端到端打通**：`pypto-lib#135`、`#136`、`#156`、`#1205`、`#1211`、`#1217`、`#1238`、`#1267`

**性能基线与对标**：`pypto-lib#1054`、`#1233`、`#1340`、`#465`、`#607`、`#622`、`#314`、`#665`；`pypto-serving#95`、`#43`、`#28`、`#7`；`pypto#2040`

**serving 能力**：`pypto-serving#5`、`#7`、`#11`、`#38`、`#167`、`#175`、`#182`、`#211`、`#218`、`#229`、`#240`

**非确定性**：`pypto-lib#1122`、`#1043`、`#368`、`#951`；`pypto-serving#183`

**Host 开销 / 跨 rank 抖动**：`pypto-serving#179`、`#198`、`#215`；`pypto#2532`、`#2629`

**挂死 / 超时**：`pypto-serving#41`、`#91`、`#155`；`pypto-lib#502`、`#1213`、`#1231`；`pto-isa#147`、`#197`；`simpler#1844`、`#2136`

**容量与内存**：`pypto-lib#1106`、`#544`、`#962`

**KV / prefix cache 契约**：`pypto-lib#511`、`#512`、`#383`、`#717`、`#735`；`pypto#1756`

**可观测性**：`simpler#860`、`#2181`、`#2191`、`#2206`、`#2220`、`#2232`、`#2320`、`#2364`、`#2375`、`#2390`；`pypto-lib#1037`

**执行模型与生态**：`pypto#2399`、`#2653`、`#2857`；`simpler#2429`

**调优方法论**：`pypto-lib#828`、`#590`

> 完整链接形如 `https://github.com/hw-native-sys/<repo>/issues/<number>`，其中 `pypto` 对应本地归档目录 `github_issues/pto/`。

## 附录 B：增量抓取复现方法

```bash
./tools/gh/gh.exe api --paginate \
  "repos/hw-native-sys/<repo>/issues?state=all&since=2026-08-17T00:00:00Z&per_page=100"
```

- 仓库列表来自 `orgs/hw-native-sys/repos`，共 11 个。
- `since` 过滤的是 `updated_at`，因此结果含「旧条目有新活动」；需再用 `created_at >= since` 区分新建与更新。
- 返回结果包含 PR，须按 `pull_request == null` 剔除。
- 本机无 `python` 与 `jq`，本次用 `node` 解析；`gh` 位于 `tools/gh/gh.exe`。
