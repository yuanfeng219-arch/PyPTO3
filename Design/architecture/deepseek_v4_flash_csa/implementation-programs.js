// Source-curated program graph. Ranges refer to the SHA-256 snapshots in source-manifest.js.
// These are developer program expressions, not compiler passes or kernel boundaries.
window.PtoCsaProgramDetails = {
  main_compressor: {
    file: '../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_compressor_ratio4.py',
    children: [
      ['cmp_projection', 'KV / gate projection', 110, 136, ['csa_main_compressor_ratio4']],
      ['cmp_pool', 'Conditional softmax pooling', 145, 215, ['csa_main_compressor_ratio4']],
      ['cmp_commit', 'Persistent ring state update', 221, 233, ['csa_main_compressor_state']],
      ['cmp_norm', 'Compressed KV RMSNorm', 235, 256, ['csa_main_compressor_ratio4']],
      ['cmp_rope', 'Compressed KV RoPE', 258, 277, ['csa_main_compressor_ratio4']],
      ['cmp_write', 'Compressed value / cache write', 279, 287, ['csa_main_compressed_kv', 'csa_logical_kv_cache']]
    ],
    edges: [['cmp_projection', 'cmp_pool'], ['cmp_projection', 'cmp_commit'], ['cmp_pool', 'cmp_commit', 'control'], ['cmp_pool', 'cmp_norm'], ['cmp_norm', 'cmp_rope'], ['cmp_rope', 'cmp_write']],
    input: 'cmp_projection', output: 'cmp_write',
    inputPorts: { compressor_state: 'cmp_pool', slot_mapping: ['cmp_commit', 'cmp_write'] },
    outputPorts: { compressor_state: 'cmp_commit' },
    conditions: ['Pooling 仅在 (token_pos + 1) % 4 == 0；cache 写入要求 slot >= 0；state 每步提交。']
  },
  indexer_compressor: {
    file: '../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_indexer_compressor.py',
    children: [
      ['idx_projection', 'Indexer KV / gate projection', 115, 142, ['csa_indexer_compressor_ratio4']],
      ['idx_pool', 'Conditional softmax pooling', 151, 225, ['csa_indexer_compressor_ratio4']],
      ['idx_commit', 'Indexer ring state update', 231, 244, ['csa_indexer_compressor_state']],
      ['idx_norm_rope', 'Indexer KV RMSNorm / RoPE', 246, 294, ['csa_indexer_compressor_ratio4']],
      ['idx_rotate', 'Indexer KV Hadamard rotation', 296, 304, ['csa_indexer_kv_representation']],
      ['idx_write', 'INT8 KV / scale cache write', 306, 335, ['csa_indexer_kv_representation', 'csa_indexer_logical_cache']]
    ],
    edges: [['idx_projection', 'idx_pool'], ['idx_projection', 'idx_commit'], ['idx_pool', 'idx_commit', 'control'], ['idx_pool', 'idx_norm_rope'], ['idx_norm_rope', 'idx_rotate'], ['idx_rotate', 'idx_write']],
    input: 'idx_projection', output: 'idx_write',
    inputPorts: { indexer_state: 'idx_pool' }, outputPorts: { indexer_state: 'idx_commit' },
    conditions: ['独立于主 Compressor；ratio-4 边界 pooling；idx_slot_mapping >= 0 才写 cache。']
  },
  lightning_indexer: {
    file: '../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_indexer.py',
    children: [
      ['idx_q_projection', 'Indexer Q projection / dequant', 651, 681, ['csa_indexer_q_rope']],
      ['idx_q_rope', 'Indexer Q RoPE', 682, 721, ['csa_indexer_q_rope']],
      ['idx_q_rotate', 'Hadamard / INT8 query', 722, 759, ['csa_indexer_q_rope']],
      ['idx_weight_projection', 'Score weight projection', 761, 797, ['csa_score_weighting']],
      ['idx_score', 'Query–Index KV score / ReLU', 535, 571, ['csa_query_index_score']],
      ['idx_weighting', 'Score weighting / head sum', 572, 584, ['csa_score_weighting']]
    ],
    edges: [['idx_q_projection', 'idx_q_rope'], ['idx_q_rope', 'idx_q_rotate'], ['idx_q_rotate', 'idx_score'], ['idx_score', 'idx_weighting'], ['idx_weight_projection', 'idx_weighting']],
    input: 'idx_q_projection', output: 'idx_weighting',
    inputPorts: { tp_allgather: 'idx_weight_projection', attention_norm: 'idx_weight_projection', index_cache: 'idx_score', block_table: 'idx_score' }, outputPorts: {},
    extraEdges: [['attention_norm', 'idx_weight_projection']],
    conditions: ['Query 与 weights 使用本地 token rows；score 读取分页 INT8 cache 与 scale。']
  },
  exact_topk: {
    file: '../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_indexer.py',
    children: [
      ['topk_leaf', 'Candidate mask / leaf Top-K', 176, 224, ['csa_topk_select']],
      ['topk_group', 'Group Top-K merge', 228, 312, ['csa_topk_select']],
      ['topk_query', 'Query merge / selected positions', 316, 426, ['csa_topk_select', 'csa_topk_compressed_positions']]
    ],
    edges: [['topk_leaf', 'topk_group'], ['topk_group', 'topk_query']],
    input: 'topk_leaf', output: 'topk_query', outputPorts: {}
  },
  sparse_attention: {
    file: '../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_sparse_attn_csa.py',
    children: [
      ['sparse_plan', 'Candidate union / validity plan', 143, 194, ['csa_candidate_index_union']],
      ['window_gather', 'Recent-window positions / KV read', 226, 265, ['csa_recent_positions_kv', 'csa_logical_kv_cache']],
      ['compressed_gather', 'Selected compressed KV read', 266, 281, ['csa_topk_compressed_positions', 'csa_main_compressed_kv']],
      ['sparse_qk', 'Sparse QK / scale / mask', 280, 292, ['csa_sparse_attention_op']],
      ['sparse_softmax', 'Block softmax statistics', 291, 297, ['csa_sparse_attention_op']],
      ['sparse_pv', 'PV / partial output', 298, 307, ['csa_sparse_attention_op']]
    ],
    ownedNodes: ['window_gather', 'compressed_gather'],
    edges: [['sparse_plan', 'window_gather'], ['sparse_plan', 'compressed_gather'], ['window_gather', 'sparse_qk'], ['compressed_gather', 'sparse_qk'], ['sparse_plan', 'sparse_qk'], ['sparse_qk', 'sparse_softmax'], ['sparse_softmax', 'sparse_pv']],
    input: 'sparse_plan', output: 'sparse_pv',
    inputPorts: { q_projection: 'sparse_qk', window_gather: 'sparse_qk', compressed_gather: 'sparse_qk' }, outputPorts: {},
    // The plan drives both physical accesses. Gather is inside sparse_attn_csa,
    // represented once by the existing canonical implementation nodes.
    extraEdges: [['exact_topk', 'sparse_plan']],
    conditions: ['窗口与压缩候选共用 validity plan；无效位置使用 NEG_INF mask。']
  },
  head_merge: {
    file: '../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_csa.py',
    children: [
      ['merge_statistics', 'Merge block mi / li / oi', 411, 427, ['csa_sparse_attention_op']],
      ['merge_sink', 'Sink normalization', 429, 433, ['csa_sparse_attention_op']],
      ['merge_inverse_rope', 'Output inverse RoPE', 435, 441, ['csa_output_inverse_rope']],
      ['merge_cast', 'BF16 output / concat', 442, 444, ['csa_output_inverse_rope']]
    ],
    edges: [['merge_statistics', 'merge_sink'], ['merge_sink', 'merge_inverse_rope'], ['merge_inverse_rope', 'merge_cast']],
    input: 'merge_statistics', output: 'merge_cast', outputPorts: {}
  },
  grouped_output_projection: {
    file: '../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_o_proj.py',
    children: [
      ['output_group_view', 'Group reshape / owner view', 474, 503, ['csa_group_reshape']],
      ['output_low_rank', 'Grouped low-rank projection', 505, 522, ['csa_grouped_low_rank_projection']],
      ['output_quant', 'Per-group INT8 / scale', 524, 556, ['csa_grouped_low_rank_projection']],
      ['output_projection', 'Output projection matmul', 558, 574, ['csa_output_projection']],
      ['output_dequant', 'Dequant / group partial sum', 576, 609, ['csa_output_projection']]
    ],
    edges: [['output_group_view', 'output_low_rank'], ['output_low_rank', 'output_quant'], ['output_quant', 'output_projection'], ['output_projection', 'output_dequant']],
    input: 'output_group_view', output: 'output_dequant', outputPorts: {},
    conditions: ['此图为当前 TP/CP 实现；量化与官方源码的数值等价尚未验证。']
  }
};
