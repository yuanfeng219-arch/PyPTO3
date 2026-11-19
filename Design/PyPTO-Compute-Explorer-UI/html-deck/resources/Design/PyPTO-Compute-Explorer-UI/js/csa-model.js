/* CSA-only static model. Source facts and high-level dependency abstractions. */
window.PtoCsaModel = Object.freeze({
  "objects": {
    "rank": {
      "id": "rank",
      "label": "Rank",
      "kind": "rank",
      "topic": "gather",
      "what": "一个 decode_csa_test 实例绑定一个物理 device。",
      "why": "Rank 同时拥有本地 query token、全流 KV 副本与一部分 O Group 权重。",
      "data": "TP=2；默认每 Rank 32 requests × 8 tokens = 256 tokens。",
      "how": "l3_decode_csa 对输入取 [rank]，传 tp_rank=rank、device=rank。",
      "source": [
        [
          "decode_csa.py",
          624,
          751
        ],
        [
          "config.py",
          242,
          245
        ]
      ],
      "sem": "sem:attention"
    },
    "pre": {
      "id": "pre",
      "label": "HC Pre / RMSNorm",
      "kind": "op",
      "topic": "gather",
      "what": "HC Pre 混合多路 hidden，再做 Attention RMSNorm。",
      "why": "为两条 Projection 数据路径提供一致的归一化输入。",
      "data": "x_hc [T,4,4096] FP32 → x_normed_t [T,4096] BF16。",
      "how": "两个 pl.scope：HC Pre 与 RMSNorm；RoPE 准备另有 rope_tid。",
      "source": [
        [
          "decode_csa.py",
          216,
          274
        ]
      ],
      "sem": "sem:norm"
    },
    "local": {
      "id": "local",
      "label": "x_normed_t",
      "kind": "tensor",
      "topic": "gather",
      "what": "当前 Rank 的归一化 token 行。",
      "why": "Q Projection 与 Indexer query 只服务本地 token，不需要把 Q 扩成全组行。",
      "data": "[T,4096] BF16；T 为运行时本地 token 数，默认容量 256。",
      "how": "进入 AllGather，同时作为 Q Projection 和 Indexer 的 local 输入。",
      "source": [
        [
          "decode_csa.py",
          273,
          318
        ],
        [
          "decode_csa.py",
          374,
          381
        ]
      ],
      "sem": "sem:attention"
    },
    "gather": {
      "id": "gather",
      "label": "Token AllGather",
      "kind": "comm",
      "topic": "gather",
      "what": "每个 Rank 收到按 source rank 排序的整组 token 行。",
      "why": "KV 和 Compressor 必须处理全组 token，维持每个 Rank 的 Cache 状态。",
      "data": "[T,D] × 2 → 每 Rank 一份 [2T,D]；默认 [512,4096]。",
      "how": "Push / Put → payload wait → readback → completion wait → signal retire。",
      "source": [
        [
          "decode_csa.py",
          276,
          288
        ],
        [
          "decode_cp_token_allgather.py",
          79,
          182
        ]
      ],
      "sem": "sem:comm"
    },
    "full": {
      "id": "full",
      "label": "x_normed_full",
      "kind": "tensor",
      "topic": "gather",
      "what": "AllGather 后的 rank-major 完整 token stream。",
      "why": "两路 Compressor 与 KV Projection 消费同一组行；不是仅收到对端 shard。",
      "data": "[KV_T_DYN,4096] BF16，TP=2 等长输入时 KV_T=2T。",
      "how": "行 [0,T) 来自 Rank 0；[T,2T) 来自 Rank 1。",
      "source": [
        [
          "decode_csa.py",
          276,
          288
        ],
        [
          "decode_cp_token_allgather.py",
          91,
          103
        ]
      ],
      "sem": "sem:attention"
    },
    "q": {
      "id": "q",
      "label": "Q Projection",
      "kind": "op",
      "topic": "gather",
      "what": "本地 x_normed_t 生成 Q 与供 Indexer 使用的量化 qr。",
      "why": "保持 query token 的 Rank-local 归属。",
      "data": "q [T,64,512] BF16；qr [T,1024] INT8；qr_scale [T,1] FP32。",
      "how": "读取 local RoPE；调用位于 AllGather 后的 scope，不能据 local 输入推断它与 AllGather 已重叠。",
      "source": [
        [
          "decode_csa.py",
          290,
          318
        ]
      ],
      "sem": "sem:linear"
    },
    "kv": {
      "id": "kv",
      "label": "KV Projection",
      "kind": "op",
      "topic": "cache",
      "what": "用 gathered token stream 生成 kv_full。",
      "why": "各 Rank 为全组请求更新原始 KV Cache。",
      "data": "kv_full [2T,512] BF16；使用 full-stream RoPE。",
      "how": "kv_proj_rope(x_normed_full, …, kv_full, late_dep)。",
      "source": [
        [
          "decode_csa.py",
          298,
          320
        ]
      ],
      "sem": "sem:linear"
    },
    "write": {
      "id": "write",
      "label": "Original KV Writeback",
      "kind": "op",
      "topic": "cache",
      "what": "把 kv_full 写入 Original KV Cache。",
      "why": "slot mapping 将当前 token 对齐到请求的分页 Cache。",
      "data": "ori_slot_mapping [2T]；只写非负 slot。",
      "how": "csa_cache_writeback SPMD；每个 Block 负责 8 个 token。产生 ori_cache_write_tid。",
      "source": [
        [
          "decode_csa.py",
          322,
          334
        ]
      ],
      "sem": "sem:attention"
    },
    "original": {
      "id": "original",
      "label": "Original KV Cache",
      "kind": "state",
      "topic": "cache",
      "what": "保留未压缩的近期 KV 数据。",
      "why": "Sparse Attention 的滑动窗口分支使用它。",
      "data": "[N_ori,128,1,512] BF16；N_ori 为动态分页块数。",
      "how": "由 KV Writeback 更新；window_swa_indices 指定 Attention 读取项。",
      "source": [
        [
          "decode_csa.py",
          188,
          198
        ],
        [
          "decode_csa.py",
          322,
          334
        ],
        [
          "decode_csa.py",
          383,
          392
        ]
      ],
      "sem": "sem:attention"
    },
    "main": {
      "id": "main",
      "label": "Main Compressor",
      "kind": "op",
      "topic": "cache",
      "what": "对 gathered stream 做 ratio-4 压缩并更新压缩 Cache。",
      "why": "为较长上下文提供压缩 KV；与 Original Writeback 是两个依赖分支。",
      "data": "compress_state FP32；cmp_kv BF16；状态按 request 组织。",
      "how": "compressor_ratio4 返回 cmp_cache_write_tid；与 ori_cache_write_tid 在 cache_ready_dep 汇合。",
      "source": [
        [
          "decode_csa.py",
          336,
          361
        ]
      ],
      "sem": "sem:attention"
    },
    "compressed": {
      "id": "compressed",
      "label": "Compressed KV Cache",
      "kind": "state",
      "topic": "cache",
      "what": "Main Compressor 维护的主 Attention 压缩 KV。",
      "why": "Indexer 返回的 Top-k 用于 Sparse Attention 的压缩检索分支。",
      "data": "[N_cmp,128,1,512] BF16；cmp_block_table [B,CMP_MAX_BLOCKS]。",
      "how": "压缩写入完成与 Original 写入完成后，Sparse Attention 才能读取。",
      "source": [
        [
          "decode_csa.py",
          189,
          191
        ],
        [
          "decode_csa.py",
          352,
          361
        ],
        [
          "decode_csa.py",
          383,
          392
        ]
      ],
      "sem": "sem:attention"
    },
    "idxcomp": {
      "id": "idxcomp",
      "label": "Indexer Compressor",
      "kind": "op",
      "topic": "cache",
      "what": "为索引检索生成独立的量化 Cache。",
      "why": "索引特征与主 Attention KV 不同，不能把两个 Cache 画成同一数据。",
      "data": "idx_kv_cache INT8；idx_kv_scale FP32；inner_compress_state FP32。",
      "how": "消费 x_normed_full；产生 idx_cache_write_tid，显式传入 Indexer。",
      "source": [
        [
          "decode_csa.py",
          363,
          381
        ]
      ],
      "sem": "sem:attention"
    },
    "indexcache": {
      "id": "indexcache",
      "label": "Indexer KV Cache",
      "kind": "state",
      "topic": "cache",
      "what": "用于 Indexer 打分的压缩量化索引。",
      "why": "它帮助选择压缩 token，本身不是 Sparse Attention 使用的主 KV 值。",
      "data": "[N_idx,128,1,128] INT8 + [N_idx,128,1,1] FP32 scale。",
      "how": "Indexer Compressor 写；Indexer 读；需 idx_cache_write_tid。",
      "source": [
        [
          "decode_csa.py",
          192,
          194
        ],
        [
          "decode_csa.py",
          365,
          381
        ]
      ],
      "sem": "sem:attention"
    },
    "indexer": {
      "id": "indexer",
      "label": "Indexer",
      "kind": "op",
      "topic": "cache",
      "what": "本地 query 与索引 Cache 打分，输出 Top-k 索引。",
      "why": "在主 Attention 前选出压缩上下文。",
      "data": "idx_topk [T,512] INT32；idx_topk_scores [T,512] FP32。",
      "how": "读取 x_normed_t、qr、qr_scale；等待 idx_cache_write_tid。",
      "source": [
        [
          "decode_csa.py",
          220,
          221
        ],
        [
          "decode_csa.py",
          374,
          392
        ]
      ],
      "sem": "sem:attention"
    },
    "ready": {
      "id": "ready",
      "label": "Cache Ready",
      "kind": "sync",
      "topic": "cache",
      "what": "Original 与 Main 压缩 Cache 的写入汇合点。",
      "why": "不能只等其中一路就让 Attention 读取 KV。",
      "data": "task_dummy(deps=[ori_cache_write_tid, cmp_cache_write_tid])。",
      "how": "Indexer 另有自己的 idx_cache_write_tid；三种依赖不要合成一个虚构的全局 barrier。",
      "source": [
        [
          "decode_csa.py",
          352,
          381
        ]
      ],
      "sem": "sem:gate"
    },
    "attention": {
      "id": "attention",
      "label": "Sparse Attention",
      "kind": "op",
      "topic": "output",
      "what": "本地 Q 对 Original 窗口和 Top-k Compressed KV 执行稀疏 Attention。",
      "why": "query 归属仍是本地 token；输出随后按 O Group 重新分配。",
      "data": "输入 q、kv_cache、cmp_kv、idx_topk；输出分块 m/l/o。",
      "how": "等待 cache_ready_dep；产出 qk_tid、attn_rope_tid。",
      "source": [
        [
          "decode_csa.py",
          383,
          393
        ]
      ],
      "sem": "sem:attention"
    },
    "group": {
      "id": "group",
      "label": "Attention Groups",
      "kind": "tensor",
      "topic": "output",
      "what": "将 64 个 Attention Head 分成 8 个 O Group。",
      "why": "O Projection 权重按 Group 分配，因此要从 token-local 转为 group-local。",
      "data": "每 Group 8 Heads × 512 = 4096；每 Rank 负责 4 Groups。",
      "how": "destination_rank = global_group // LOCAL_O_GROUPS。",
      "source": [
        [
          "decode_csa.py",
          118,
          120
        ],
        [
          "decode_csa.py",
          395,
          410
        ]
      ],
      "sem": "sem:attention"
    },
    "publish": {
      "id": "publish",
      "label": "Merge / Pack / Publish",
      "kind": "comm",
      "topic": "output",
      "what": "合并 Sparse 分块结果，完成归一化与输出 RoPE，然后按 Group 发布。",
      "why": "把本地 token 的所有 Head 结果发送给各 O Group 的目标 Rank。",
      "data": "attention_grouped [8×T_PAD,4096] BF16；目标为 attention_window。",
      "how": "csa_merge_pack_publish 的 deps=[qk_tid,attn_rope_tid]；Put 与 Notify 在生产者中融合。",
      "source": [
        [
          "decode_csa.py",
          395,
          479
        ]
      ],
      "sem": "sem:comm"
    },
    "window": {
      "id": "window",
      "label": "attention_window",
      "kind": "tensor",
      "topic": "output",
      "what": "每个目标 Rank 的 Group-major 接收窗口。",
      "why": "目标 Rank 要收齐自己 4 个 Groups 中来自两个 source Rank 的 token。",
      "data": "[LOCAL_O_GROUPS×GROUP_T_PAD,4096]；默认 [2048,4096] BF16。",
      "how": "target_row = local_group×GROUP_T_PAD + source_rank×local_t + token_offset。",
      "source": [
        [
          "decode_csa.py",
          446,
          477
        ],
        [
          "decode_o_proj.py",
          56,
          76
        ]
      ],
      "sem": "sem:attention"
    },
    "a2a": {
      "id": "a2a",
      "label": "All-to-All Completion",
      "kind": "comm",
      "topic": "output",
      "what": "完成已在 Merge / Publish 发起的 Group 重分布。",
      "why": "这不是另一次重复发送：o_group_a2a 等待、读回并回收已发布窗口。",
      "data": "attention_local_groups [4,2T_PAD,4096] BF16；默认 [4,512,4096]。",
      "how": "publish_dep → peer signal wait → local gather → completion / signal retire。",
      "source": [
        [
          "decode_csa.py",
          481,
          489
        ],
        [
          "decode_o_proj.py",
          269,
          335
        ]
      ],
      "sem": "sem:comm"
    },
    "oproj": {
      "id": "oproj",
      "label": "O Projection",
      "kind": "op",
      "topic": "output",
      "what": "当前 Rank 用自己的 4 个 O Groups 权重计算所有 token 的部分输出。",
      "why": "分组权重贡献需要跨 Rank 相加，才能恢复各 token 的完整 hidden。",
      "data": "wo_a [4,1024,4096]；wo_b [4096,4096] INT8；partial [2T,4096]。",
      "how": "o_proj_reduce_scatter 内对 owner token slices 组织计算；内部子算子本轮合并展示。",
      "source": [
        [
          "decode_csa.py",
          490,
          498
        ],
        [
          "decode_o_proj.py",
          457,
          503
        ]
      ],
      "sem": "sem:linear"
    },
    "rs": {
      "id": "rs",
      "label": "ReduceScatter",
      "kind": "comm",
      "topic": "output",
      "what": "把每个 token owner 的部分输出送回 owner，做 FP32 跨 Rank 累加。",
      "why": "既有 reduce 也有 scatter，不能简化成仅将 token 切开。",
      "data": "attn_out [T,4096] BF16；o_window 中按来源 Rank 排列 partial。",
      "how": "Put / Notify → peer wait → owner 对来源 Rank 累加 → completion / retire。",
      "source": [
        [
          "decode_o_proj.py",
          606,
          697
        ],
        [
          "decode_csa.py",
          490,
          498
        ]
      ],
      "sem": "sem:comm"
    },
    "post": {
      "id": "post",
      "label": "HC Post",
      "kind": "op",
      "topic": "output",
      "what": "将本地 Attention 输出与 HC residual 混合，形成 x_out。",
      "why": "恢复当前 Rank 的本地 token 输出边界。",
      "data": "x_out [T,4,4096] FP32。",
      "how": "HC Post 在输出 Projection / ReduceScatter 的 scope 之后。",
      "source": [
        [
          "decode_csa.py",
          500,
          503
        ]
      ],
      "sem": "sem:norm"
    },
    "push": {
      "id": "push",
      "label": "Put to both Ranks",
      "kind": "comm",
      "topic": "gather",
      "what": "每个 source Rank 向所有目标窗口写自己的行。",
      "why": "保证消费者读取已发布的数据或状态。",
      "data": "dst row = source_rank × local_t",
      "how": "16 个 Push workers；每 worker 发送完成后通知对端。",
      "source": [
        [
          "decode_cp_token_allgather.py",
          95,
          118
        ]
      ],
      "sem": "sem:comm"
    },
    "payload": {
      "id": "payload",
      "label": "Payload Wait",
      "kind": "sync",
      "topic": "gather",
      "what": "等待 peer 的 payload 发布。",
      "why": "保证消费者读取已发布的数据或状态。",
      "data": "expected = PUSH_WORKERS = 16",
      "how": "deps=[_push_tid]；每个 peer signal 达到阈值后放行。",
      "source": [
        [
          "decode_cp_token_allgather.py",
          120,
          131
        ]
      ],
      "sem": "sem:gate"
    },
    "readback": {
      "id": "readback",
      "label": "Readback / Retire",
      "kind": "sync",
      "topic": "gather",
      "what": "读回整组行并完成两阶段 signal epoch。",
      "why": "保证消费者读取已发布的数据或状态。",
      "data": "group_rows=TP_SIZE × local_rows",
      "how": "readback deps=[_push_tid,_payload_wait_tid]；完成通知后回收计数。",
      "source": [
        [
          "decode_cp_token_allgather.py",
          133,
          182
        ]
      ],
      "sem": "sem:gate"
    },
    "rope": {
      "id": "rope",
      "label": "RoPE Dependency",
      "kind": "sync",
      "topic": "cache",
      "what": "连接预处理与后续 Projection / Compressor 的任务依赖。",
      "why": "保证消费者读取已发布的数据或状态。",
      "data": "late_dep = task_dummy(deps=[rope_tid])",
      "how": "仅表示 Task dependency，未表达运行耗时。",
      "source": [
        [
          "decode_csa.py",
          234,
          274
        ],
        [
          "decode_csa.py",
          298,
          310
        ]
      ],
      "sem": "sem:gate"
    },
    "rows0": {
      "id": "rows0",
      "label": "Rank 0 rows · [0,T)",
      "kind": "tensor",
      "topic": "gather",
      "what": "完整 Tensor 中来自 Rank 0 的行段。",
      "why": "两路 Compressor 与 KV Projection 消费同一组行；不是仅收到对端 shard。",
      "data": "[T,4096] BF16；在 rank-major 结果中占 [0,T)。",
      "how": "这是逻辑行段；未表达物理地址或存储连续性。",
      "source": [
        [
          "decode_csa.py",
          276,
          288
        ],
        [
          "decode_cp_token_allgather.py",
          91,
          103
        ]
      ],
      "sem": "sem:attention"
    },
    "rows1": {
      "id": "rows1",
      "label": "Rank 1 rows · [T,2T)",
      "kind": "tensor",
      "topic": "gather",
      "what": "完整 Tensor 中来自 Rank 1 的行段。",
      "why": "两路 Compressor 与 KV Projection 消费同一组行；不是仅收到对端 shard。",
      "data": "[T,4096] BF16；在 rank-major 结果中占 [T,2T)。",
      "how": "两个目标 Rank 都得到这两段；不是各取一半。",
      "source": [
        [
          "decode_csa.py",
          276,
          288
        ],
        [
          "decode_cp_token_allgather.py",
          91,
          103
        ]
      ],
      "sem": "sem:attention"
    },
    "group0": {
      "id": "group0",
      "label": "Groups 0–3 → Rank 0",
      "kind": "tensor",
      "topic": "output",
      "what": "当前 source Rank 生成的前四个 O Groups。",
      "why": "O Projection 权重按 Group 分配，因此要从 token-local 转为 group-local。",
      "data": "4 Groups × 本地 T token 行 × O_GROUP_IN=4096。",
      "how": "destination_rank=global_group//4；Rank 0 还会接收另一个 source Rank 的这些 Group。",
      "source": [
        [
          "decode_csa.py",
          118,
          120
        ],
        [
          "decode_csa.py",
          395,
          410
        ]
      ],
      "sem": "sem:attention"
    },
    "group1": {
      "id": "group1",
      "label": "Groups 4–7 → Rank 1",
      "kind": "tensor",
      "topic": "output",
      "what": "当前 source Rank 生成的后四个 O Groups。",
      "why": "O Projection 权重按 Group 分配，因此要从 token-local 转为 group-local。",
      "data": "4 Groups × 本地 T token 行 × O_GROUP_IN=4096。",
      "how": "destination_rank=global_group//4；Rank 1 还会接收另一个 source Rank 的这些 Group。",
      "source": [
        [
          "decode_csa.py",
          118,
          120
        ],
        [
          "decode_csa.py",
          395,
          410
        ]
      ],
      "sem": "sem:attention"
    },
    "mainstate": {
      "id": "mainstate",
      "label": "compress_state · FP32",
      "kind": "tensor",
      "topic": "cache",
      "what": "Main Compressor 持续维护的每请求压缩状态。",
      "why": "为较长上下文提供压缩 KV；与 Original Writeback 是两个依赖分支。",
      "data": "[N_state,MAIN_STATE_BLOCK_SIZE,2048] FP32；2048=2×COFF×HEAD_DIM。",
      "how": "通过 compress_state_block_table 和 state_slot_mapping 寻址；状态并不是主 KV Cache。",
      "source": [
        [
          "decode_csa.py",
          336,
          361
        ]
      ],
      "sem": "sem:attention"
    },
    "innerstate": {
      "id": "innerstate",
      "label": "inner_compress_state · FP32",
      "kind": "tensor",
      "topic": "cache",
      "what": "Indexer Compressor 独立维护的每请求状态。",
      "why": "索引特征与主 Attention KV 不同，不能把两个 Cache 画成同一数据。",
      "data": "[N_inner,INNER_STATE_BLOCK_SIZE,512] FP32；512=2×COFF×IDX_HEAD_DIM。",
      "how": "它与 Main Compressor 的状态分开，不能误画成共享同一压缩缓冲。",
      "source": [
        [
          "decode_csa.py",
          363,
          381
        ]
      ],
      "sem": "sem:attention"
    }
  },
  "topics": {
    "gather": "Local → Gathered",
    "cache": "Cache / Compressor / Indexer",
    "output": "Attention Output Redistribution"
  },
  "kindLabels": {
    "op": "计算",
    "tensor": "数据",
    "state": "持久状态",
    "comm": "通信",
    "sync": "同步 / 控制",
    "rank": "Rank"
  },
  "sourceBase": "../../Data/DeepseekV4/deepseek_v4_flash_dspark/",
  "facts": {
    "tp": 2,
    "localCapacity": 256,
    "groupCapacity": 512,
    "hidden": 4096,
    "heads": 64,
    "headDim": 512,
    "groups": 8,
    "localGroups": 4
  },
  "sourceNote": "本地源码静态关系；位置表达依赖层级，不表达耗时。T 为每 Rank 的运行时 token 数；数字 Shape 使用默认容量 T=256。",
  "relations": {
    "pipeline": [
      [
        "pre",
        "gather"
      ],
      [
        "pre",
        "q"
      ],
      [
        "gather",
        "q",
        "sync"
      ],
      [
        "gather",
        "kv"
      ],
      [
        "gather",
        "main"
      ],
      [
        "gather",
        "idxcomp"
      ],
      [
        "kv",
        "write"
      ],
      [
        "write",
        "ready",
        "sync"
      ],
      [
        "main",
        "ready",
        "sync"
      ],
      [
        "idxcomp",
        "indexer",
        "sync"
      ],
      [
        "q",
        "indexer"
      ],
      [
        "q",
        "attention"
      ],
      [
        "ready",
        "attention",
        "sync"
      ],
      [
        "indexer",
        "attention"
      ],
      [
        "attention",
        "publish",
        "sync"
      ],
      [
        "publish",
        "a2a",
        "sync"
      ],
      [
        "a2a",
        "oproj"
      ],
      [
        "oproj",
        "rs",
        "comm"
      ],
      [
        "rs",
        "post"
      ]
    ],
    "cache": [
      [
        "local",
        "q"
      ],
      [
        "full",
        "kv"
      ],
      [
        "full",
        "main"
      ],
      [
        "full",
        "idxcomp"
      ],
      [
        "kv",
        "write"
      ],
      [
        "write",
        "original"
      ],
      [
        "main",
        "compressed"
      ],
      [
        "idxcomp",
        "indexcache"
      ],
      [
        "write",
        "ready",
        "sync"
      ],
      [
        "main",
        "ready",
        "sync"
      ],
      [
        "idxcomp",
        "indexer",
        "sync"
      ],
      [
        "indexcache",
        "indexer"
      ],
      [
        "q",
        "indexer"
      ],
      [
        "q",
        "attention"
      ],
      [
        "original",
        "attention"
      ],
      [
        "compressed",
        "attention"
      ],
      [
        "ready",
        "attention",
        "sync"
      ],
      [
        "indexer",
        "attention"
      ],
      [
        "rope",
        "kv",
        "sync"
      ],
      [
        "rope",
        "main",
        "sync"
      ],
      [
        "rope",
        "idxcomp",
        "sync"
      ],
      [
        "rope",
        "indexer",
        "sync"
      ]
    ]
  }
});
