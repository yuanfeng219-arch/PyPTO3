# Logical Pipeline 语义校验

校验对象：当前 Compute Explorer 使用的源码快照及 `Data/DeepseekV4/deepseek_v4_flash_dspark/` 中的 CSA Python 文件。Logical Tick 与 Logical Step 都是逻辑位置，不是共同的时间单位。

## 源码版本与边界

| 对象 | 本次核对的来源 | 结论 |
| --- | --- | --- |
| Paged Attention | `js/paged-attention-pypto-source.js` 中的 `paged_attention_pypto.py` 快照，SHA-256 `4087a49c…b238e843` | 工作区内未找到同名原始 `.py`；结论只适用于页面当前快照。 |
| Decode CSA 主流程 | `js/decode-csa-source.js` 与 `Data/DeepseekV4/deepseek_v4_flash_dspark/decode_csa.py` | 文本完全一致，SHA-256 `5376b4db…dfca69a0f`。 |
| Decode CSA qk_pv | `Data/DeepseekV4/deepseek_v4_flash_dspark/decode_sparse_attn_csa.py` | 页面原先没有此文件的源码快照；本次按原文件生成 `js/decode-sparse-attn-csa-source.js` 并注册，SHA-256 `7e4afa5e…aa8800b`。 |

## Paged Attention

| 结论 | 证据等级 | 依据 |
| --- | --- | --- |
| Task 内 Stack 数由有效序列长度与 `STACK_TOKENS=512` 决定；最终 `ctx` 由累积状态 `o / l_sum` 转 BF16 后写入 `out2d`。单个 Stack 的局部结果不是最终 Task Output。 | Verified / Source-confirmed | 快照 L266、L864–872。 |
| 主循环为 `pl.range(stack_count + PRE_LAUNCH)`；`PRE_LAUNCH=2`，produce 在 `tick < stack_count`，consume 在 `tick >= PRE_LAUNCH`，`consume_stack=tick-PRE_LAUNCH`。Task 48 的六个 Stack 因而映射到八个 Logical Ticks。 | Verified / Model-derived 位置 | 快照 L31、L758–833；前端规则 `produceTick=stack.index`、`consumeTick=stack.index+preLaunch`。 |
| `TRANSFER_SLOTS=PRE_LAUNCH+1=3`；Stack 使用 `stack.index % TRANSFER_SLOTS`。同 Slot 后续 Stack 的虚线仅表示 Buffer 资源复用。 | Verified / Model-derived Slot | 快照 L31–32、L281、L365、L763、L828；`visualization2-anatomy.js` 与 `visualization2-pipeline-intra.js`。 |
| QK→Softmax→PV→Accumulate 的传递分别通过 `score_transfer`、`probability_transfer`、`pv_transfer`；READY SET/WAIT 是独立同步关系。 | Verified / Source-confirmed | 快照中三组 `*_READY_EVENT` 的 SET/WAIT、transfer 的 store/load；详见当前 Inspector 的 Source Evidence。 |
| 同一 Tick 的节点间距、阶段行距与 Slot 虚线形状。 | Presentation Choice | 前端 SVG 布局常量；不表示实际并发、占用时间或数据值复用。 |
| READY WAIT 是否造成 Stall、指令重叠、硬件时长。 | Unsupported / Unverified | 当前快照及前端模型没有对应 Trace 或编译调度证据。 |

## Decode CSA qk_pv

| 结论 | 证据等级 | 依据 |
| --- | --- | --- |
| `QK_ITEMS=t_dim*SPARSE_BLOCKS`；`qk_order` 先排有效、再排空块；SPMD Lane 按 `qk_core + qk_it*NUM_QK_CORES` 取 Work Item。Work Item 是循环中的一次计算对象，并非单独 Runtime Task。 | Verified / Source-confirmed | `decode_sparse_attn_csa.py` L81、L183–195、L213–223。 |
| SB0 从 Original KV 的 Window 按页连续 Gather；压缩部分按索引逐行 Gather；无效填充值由 `sparse_bias` 屏蔽。单个 BF16 `[128,512]` L1 KV Tile 在两个 Head Batch 中复用。 | Verified / Source-confirmed | 同文件 L225–296。 |
| `QK_M_TILE=32`，`H=64`，`pl.pipeline(H // QK_M_TILE, stage=2)` 形成两批 Heads 0–31、32–63。QK 使用 Kᵀ，PV 使用 BF16 `exp` 与同一 KV Tile；Scores 和 Partial O 为 FP32。 | Verified / Source-confirmed | 同文件 L64–65、L278–296。 |
| Softmax 的 `m [32,1]`、`l [32,1]` 写入 Partial；`exp [32,128]` 转 BF16 后供 PV；PV 的 `o [32,512]` 写入 Partial。每批拆成两个 `H_TILE=16` 分段，一个有效 Work Item 共四段。 | Verified / Source-confirmed | 同文件 L289–304。m/l 不直接输入 PV。 |
| Invalid 分支写 `mi=-3.0e38`、`li=0`、`oi=0`，跳过 QK/Softmax/PV。 | Verified / Source-confirmed | 同文件 L305–313。 |
| `csa_merge_pack_publish` 是主流程中依赖 `qk_tid`、`attn_rope_tid` 的下游 Task，消费多个 Sparse Blocks 的 Partial。 | Verified / Source-confirmed | `decode_csa.py` L395–477。它不属于单个 Work Item 内部。 |
| 每个 Work Item 的 Lane Iteration 可由 `qk_order` 位置和 Lane 数推导；Batch 0 Step 2–5、Batch 1 Step 6–9 只是保守可读排列。 | Model-derived / Presentation Choice | Lane 映射来自源码循环；具体 SVG Step 由前端固定排布。`stage=2` 不证明准确调度时刻或重叠。 |
| Head Batch 的真实重叠、Operation Duration、硬件等待。 | Unsupported / Unverified | 缺少该源码版本对应的编译 Schedule 与 Runtime Trace。 |

## 修复及数据契约

P7.5 核对：Paged Attention 的 PV 主路径位于快照 L364–503，Tail 分支位于 L504–718，PV_READY SET 位于 L719–724。原单段 L512–724 漏掉主路径；现在 QK、Softmax、PV、Accumulate 的 Operation Evidence 均支持多段源码范围，Inspector 可逐段跳转。`PRE_LAUNCH=2` 与 `TRANSFER_SLOTS=3` 仍由 L31–32 及各 produce/consume 索引规则支持；源码快照未变，SHA 保持原值。

两种算子的 Descriptor `inputs`、`outputs` 统一为数据对象数组。Paged Attention 只列已证实的 transfer buffer；CSA 列有源码支持的 tensor、dtype 与 shape。上游 Operation 身份保留在 `dependencies.from`，并附 `to`；CSA 的 L1 KV Tile 对 QK/PV 同时提供数据输入和共享资源复用关系。缺少证据的 shape/dtype 省略，不推定 Runtime 时长。

- **Incorrect（已修复）**：CSA Inspector 的 `decode_sparse_attn_csa.py` Source Evidence 指向未注册文件；已补同目录原始源码快照，使现有 IDE 源码跳转可解析。
- **Unsupported（已澄清）**：将 CSA 的离散 Step、Paged 的 Tick 距离或 READY WAIT 解释为真实调度、时长、Stall；Inspector 现标注其证据等级。
- **Presentation Choice（保留）**：两种 Timeline 的固定列宽、行距及 CSA 两批分开的 Step；不更改现有布局。

Paged 的 `task.operations[]` 现带 `computeUnitId`、`iterationId`、`operationId`、`operationType`、`logicalPosition`、`inputs`、`outputs`、`dependencies`、`resourceRefs`、`evidence`。CSA Detail 通过 `getOperationDescriptor()` 提供同一最小字段。ID 包含 Task/Work Item 与 Stack/Head Batch，不包含虚构 Runtime ID。依赖类型区分 `data`、`event`/同步和 `resource-reuse`；Merge 属于下游 Task 边界。每个证据附等级，源码证据附文件、范围和版本；推导证据附规则与参数。字段中的输入输出仅列已有模型可确认的关系，不宣称已完整覆盖所有 Tensor。

未来对照编译或 Trace 时，需要先解决源码版本、算子逻辑对象与编译节点的稳定映射，再核对调度位置、资源生命周期及运行事件。当前模型不具备这些映射，也不支持性能结论。
