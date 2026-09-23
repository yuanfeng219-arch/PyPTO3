(function registerCsaFourLevelMapping(global) {
  'use strict';

  const ref = (file, start, end = start, role = 'definition') => ({ file, start, end, role });
  const node = (id, level, label, type, options = {}) => ({
    id,
    level,
    semanticLevel: options.semanticLevel || level,
    label,
    type,
    nodeType: options.nodeType || type,
    parentId: options.parentId || null,
    parentScope: options.parentScope || null,
    mapsTo: options.mapsTo || [],
    provenance: options.provenance || [],
    sourceRefs: options.sourceRefs || [],
    inputPorts: options.inputPorts || ['in'],
    outputPorts: options.outputPorts || ['out'],
    summaryEdges: options.summaryEdges || [],
    expandedEdges: options.expandedEdges || [],
    defaultExpanded: options.defaultExpanded === true,
    applicableScope: options.applicableScope || 'csa_model_semantic_scope',
    phase: options.phase || 'decode',
    applicableLayers: options.applicableLayers || '2,4,…,42',
    artifactStatus: options.artifactStatus || 'verified',
    children: options.children || [],
    mappingType: options.mappingType || null,
    reason: options.reason || '',
    graph: options.graph || null,
    entityKind: options.entityKind || (level === 'L1' ? 'Scope' : type === 'State' ? 'LogicalState' : 'Op'),
    origin: level === 'L1' || level === 'L2' ? 'official-source' : level === 'L3' ? 'developer-pypto' : 'compiler-generated',
    canonicalId: options.canonicalId || id,
    lifecycle: options.lifecycle || null,
    conditions: options.conditions || []
  });

  const decode = '../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_csa.py';
  const qkv = '../../../Data/DeepseekV4/deepseek_v4_flash_dspark/qkv_proj_rope.py';
  const compressor = '../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_compressor_ratio4.py';
  const indexer = '../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_indexer.py';
  const indexerCompressor = '../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_indexer_compressor.py';
  const sparse = '../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_sparse_attn_csa.py';
  const hcPre = '../../../Data/DeepseekV4/deepseek_v4_flash_dspark/hc_pre.py';
  const hcPost = '../../../Data/DeepseekV4/deepseek_v4_flash_dspark/hc_post.py';
  const rmsnorm = '../../../Data/DeepseekV4/deepseek_v4_flash_dspark/rmsnorm.py';
  const oproj = '../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_o_proj.py';
  const allgather = '../../../Data/DeepseekV4/deepseek_v4_flash_dspark/decode_cp_token_allgather.py';
  const officialModel = '../../../Data/DeepSeek-V4-Flash-Official/model.py';

  const l3 = [
    ['hc_input', 'HC State Input', 'State', 'STATE', [ref(decode, 151, 217, 'call')], [], '四路 HC 残差流是跨子层延续的运行时状态。'],
    ['hc_pre', 'HC Pre', 'Compute', 'SAME', [ref(decode, 227, 228, 'call'), ref(hcPre, 1, 220)], [], '直接实现整网图中的 HC Pre 4→1。'],
    ['attention_norm', 'Attention RMSNorm', 'Compute', 'SAME', [ref(decode, 273, 275, 'call'), ref(rmsnorm, 1, 180)], [], '直接实现 mHC-Attention 前的 RMSNorm。'],
    ['tp_allgather', 'CP Token All-Gather', 'Communication', 'DEPLOY', [ref(decode, 277, 289, 'call'), ref(allgather, 1, 280)], [], '为 KV 分支补齐 TP/CP 组内 token 行。'],
    ['q_projection', 'Query Projection + RoPE', 'Compute', 'SAME', [ref(decode, 304, 313, 'call'), ref(qkv, 243, 553)], ['l4_qkv_tensor_graph'], '直接实现 CSA Query Projection 与 RoPE。'],
    ['kv_projection', 'Shared KV Projection', 'Compute', 'EXPANDED', [ref(decode, 315, 319, 'call'), ref(qkv, 555, 760)], ['l4_qkv_tensor_graph'], '把共享 KV 路径展开为可编译投影。'],
    ['slot_mapping', 'Slot Mapping', 'State', 'STATE', [ref(decode, 192, 198, 'call'), ref(decode, 338, 345, 'call')], [], '缓存与压缩状态的逻辑位置被外显为运行时映射。'],
    ['block_table', 'Block Table', 'State', 'STATE', [ref(decode, 174, 198, 'call'), ref(decode, 346, 351, 'call')], [], '分页 KV、索引与压缩状态的块表被外显为运行时资源。'],
    ['raw_kv_cache', 'Raw KV Cache', 'State', 'STATE', [ref(decode, 186, 198, 'call'), ref(decode, 321, 333, 'call')], [], '原始 KV 被显式保存为分页运行时状态。'],
    ['main_compressor', 'Main Compressor · ratio=4', 'Compute', 'SAME', [ref(decode, 335, 360, 'call'), ref(compressor, 1, 330)], [], '直接实现 ratio-4 KV 压缩语义。'],
    ['compressor_state', 'Compressor State', 'State', 'STATE', [ref(decode, 174, 175, 'call'), ref(compressor, 120, 250)], [], '在线压缩跨 token 递推状态被外显。'],
    ['compressed_kv_cache', 'Compressed KV Cache', 'State', 'STATE', [ref(decode, 187, 188, 'call'), ref(compressor, 200, 270)], [], '压缩历史被显式保存为分页 KV Cache。'],
    ['indexer_compressor', 'Indexer Compressor', 'Compute', 'EXPANDED', [ref(decode, 363, 374, 'call'), ref(indexerCompressor, 73, 310)], [], 'Lightning Indexer 的索引构建被拆为独立压缩步骤。'],
    ['indexer_state', 'Indexer State', 'State', 'STATE', [ref(decode, 184, 185, 'call'), ref(decode, 367, 373, 'call')], [], '索引压缩器拥有独立的持久递推状态。'],
    ['index_cache', 'Index KV Cache', 'State', 'STATE', [ref(decode, 189, 191, 'call'), ref(indexer, 625, 810)], [], 'INT8 检索向量与 scale 被外显为运行时缓存。'],
    ['lightning_indexer', 'Lightning Indexer', 'Compute', 'SAME', [ref(decode, 375, 382, 'call'), ref(indexer, 625, 810)], [], '直接实现 Lightning Indexer 检索打分。'],
    ['exact_topk', 'Exact Top-K Select', 'Compute', 'EXPANDED', [ref(decode, 375, 382, 'call'), ref(indexer, 175, 624)], [], '相关位置选择被具体展开为 exact TopK-512。'],
    ['window_gather', 'Recent Window Gather', 'Compute', 'EXPANDED', [ref(decode, 384, 393, 'call'), ref(sparse, 180, 260)], [], '最近窗口被展开为分页 KV gather。'],
    ['compressed_gather', 'Selected KV Gather', 'Compute', 'EXPANDED', [ref(decode, 384, 393, 'call'), ref(sparse, 240, 320)], [], 'Top-K 压缩位置被展开为 selected KV gather。'],
    ['sparse_attention', 'Sparse Shared-KV Attention', 'Compute', 'SAME', [ref(decode, 384, 393, 'call'), ref(sparse, 90, 390)], [], '直接实现 Shared-KV 稀疏注意力。'],
    ['head_merge', 'Sparse Block Merge', 'Compute', 'EXPANDED', [ref(decode, 395, 445, 'call')], [], '显式展开块级 softmax 统计合并、sink 归一化与逆 RoPE。'],
    ['head_group_redistribution', 'Head-group Redistribution', 'Compute', 'EXPANDED', [ref(decode, 446, 453, 'call')], [], '注意力头被重排为输出投影拥有的 group 布局。'],
    ['attention_alltoall', 'Attention All-to-All', 'Communication', 'DEPLOY', [ref(decode, 455, 484, 'call')], [], '按输出 head-group owner 跨 rank 重排。'],
    ['grouped_output_projection', 'Grouped Output Projection', 'Compute', 'SAME', [ref(decode, 486, 491, 'call'), ref(oproj, 456, 683)], [], '直接实现官方 Grouped Output Projection。'],
    ['output_reduce_scatter', 'Output Reduce-Scatter', 'Communication', 'DEPLOY', [ref(decode, 486, 495, 'call'), ref(oproj, 456, 683)], [], '汇总分片输出并散回 token owner。'],
    ['attention_hc_post', 'HC Post', 'Compute', 'SAME', [ref(decode, 497, 499, 'call'), ref(hcPost, 1, 220)], [], 'decode_csa.py 明确包含 HC Post 1→4。'],
    ['attention_hc_output', 'HC State Output', 'State', 'STATE', [ref(decode, 497, 499, 'call')], [], '更新后的四路 HC 残差状态。']
  ];
  const semanticImplementationIds = new Set([
    'q_projection', 'kv_projection', 'main_compressor', 'indexer_compressor',
    'lightning_indexer', 'exact_topk', 'window_gather', 'compressed_gather',
    'sparse_attention', 'head_merge', 'head_group_redistribution', 'grouped_output_projection'
  ]);
  const l3Nodes = l3.map(([id, label, type, mappingType, sourceRefs, mapsTo, reason]) => node(id, 'L3', label, type, {
    parentScope: semanticImplementationIds.has(id) ? 'csa_model_semantic_scope' : 'decode_csa_implementation_scope',
    mapsTo,
    sourceRefs,
    mappingType,
    reason,
    provenance: sourceRefs.map(item => `${item.file}:L${item.start}–${item.end}`)
  }));

  const semanticProvenance = lines => [`DeepSeek official inference/model.py:L${lines}`];
  const semanticNode = (id, label, parentId, lines, options = {}) => node(id, 'L2', label, 'Compute', {
    parentId,
    parentScope: parentId,
    mapsTo: options.mapsTo || [],
    sourceRefs: options.sourceRefs || [ref(officialModel, ...lines, 'call')],
    provenance: semanticProvenance(`${lines[0]}–${lines[1]}`),
    artifactStatus: 'verified',
    ...options
  });

  const compoundSpecs = [
    {
      id: 'csa_query_projection', label: 'Query Projection', children: [
        'csa_q_lora_down', 'csa_q_normalization', 'csa_main_q_rope', 'csa_indexer_q_rope'
      ], sourceRefs: [ref(officialModel, 496, 499, 'call'), ref(officialModel, 457, 459)]
    },
    {
      id: 'csa_recent_window_kv', label: 'Recent-window KV · 128', children: [
        'csa_shared_kv_projection', 'csa_kv_normalization', 'csa_kv_rope', 'csa_recent_positions_kv', 'csa_logical_kv_cache'
      ], sourceRefs: [ref(officialModel, 502, 507, 'call'), ref(officialModel, 460, 461)]
    },
    {
      id: 'csa_compressed_kv_path', label: 'Compressed KV Path · ratio 4', children: [
        'csa_main_compressor_ratio4', 'csa_main_compressed_kv', 'csa_main_compressor_state'
      ], sourceRefs: [ref(officialModel, 530, 533, 'call'), ref(officialModel, 466, 474)]
    },
    {
      id: 'csa_lightning_indexer', label: 'Lightning Indexer', children: [
        'csa_indexer_compressor_ratio4', 'csa_indexer_kv_representation', 'csa_query_index_score',
        'csa_score_weighting', 'csa_topk_select', 'csa_topk_compressed_positions', 'csa_indexer_logical_cache', 'csa_indexer_compressor_state'
      ], sourceRefs: [ref(officialModel, 402, 433, 'call'), ref(officialModel, 380, 400)]
    },
    {
      id: 'csa_sparse_shared_kv_attention', label: 'Sparse Shared-KV Attention', children: [
        'csa_candidate_index_union', 'csa_sparse_attention_op', 'csa_output_inverse_rope'
      ], sourceRefs: [ref(officialModel, 507, 534, 'call'), ref(officialModel, 473, 474)]
    },
    {
      id: 'csa_grouped_output_projection', label: 'Grouped Output Projection', children: [
        'csa_group_reshape', 'csa_grouped_low_rank_projection', 'csa_output_projection'
      ], sourceRefs: [ref(officialModel, 537, 543, 'call'), ref(officialModel, 462, 463)]
    }
  ];

  const l2 = [
    semanticNode('csa_q_lora_down', 'Q LoRA/down projection', 'csa_query_projection', [496, 496], { mapsTo: ['q_projection'] }),
    semanticNode('csa_q_normalization', 'Q normalization', 'csa_query_projection', [496, 496], { mapsTo: ['q_projection'] }),
    semanticNode('csa_main_q_rope', 'Main Q projection/scaling/RoPE', 'csa_query_projection', [497, 499], { mapsTo: ['q_projection'] }),
    semanticNode('csa_indexer_q_rope', 'Indexer Q + RoPE/rotation', 'csa_query_projection', [411, 416], { mapsTo: ['lightning_indexer'] }),

    semanticNode('csa_shared_kv_projection', 'Shared KV Projection', 'csa_recent_window_kv', [502, 502], { mapsTo: ['kv_projection'] }),
    semanticNode('csa_kv_normalization', 'KV normalization', 'csa_recent_window_kv', [503, 503], { mapsTo: ['kv_projection'] }),
    semanticNode('csa_kv_rope', 'KV RoPE / quantization', 'csa_recent_window_kv', [504, 506], { mapsTo: ['kv_projection'] }),
    semanticNode('csa_recent_positions_kv', 'Recent-window positions', 'csa_recent_window_kv', [507, 507], { entityKind: 'Value', mapsTo: ['window_gather'], reason: '窗口索引值来自 get_window_topk_idxs；不是 KV 内容，不重复创建第二份窗口索引。' }),
    semanticNode('csa_logical_kv_cache', 'Attention.kv_cache', 'csa_recent_window_kv', [530, 533], { entityKind: 'LogicalState', mapsTo: ['raw_kv_cache', 'compressed_kv_cache'], lifecycle: '跨 decode step；同一逻辑缓存含近期环形区与压缩区。', sourceRefs: [ref(officialModel, 530, 533, 'call'), ref(officialModel, 473, 474)] }),

    semanticNode('csa_main_compressor_ratio4', 'Main Compressor · ratio 4', 'csa_compressed_kv_path', [532, 532], { mapsTo: ['main_compressor'], sourceRefs: [ref(officialModel, 532, 532, 'call'), ref(officialModel, 316, 377)], conditions: ['decode: (start_pos + 1) % 4 == 0 才生成新压缩值；每步更新状态'] }),
    semanticNode('csa_main_compressed_kv', 'Main compressed KV value', 'csa_compressed_kv_path', [362, 377], { entityKind: 'Value', mapsTo: ['compressed_kv_cache'], conditions: ['should_compress'] }),
    semanticNode('csa_main_compressor_state', 'Main kv_state / score_state', 'csa_compressed_kv_path', [345, 361], { entityKind: 'LogicalState', mapsTo: ['compressor_state'], lifecycle: '跨 token 保存重叠窗口投影与门控分数。', sourceRefs: [ref(officialModel, 345, 361, 'call'), ref(officialModel, 301, 304)] }),

    semanticNode('csa_indexer_compressor_ratio4', 'Indexer Compressor · ratio 4', 'csa_lightning_indexer', [417, 417], { mapsTo: ['indexer_compressor'], sourceRefs: [ref(officialModel, 417, 417, 'call'), ref(officialModel, 316, 377)], conditions: ['独立 Compressor 实例；should_compress 才写入新检索向量'] }),
    semanticNode('csa_indexer_kv_representation', 'Indexer KV value', 'csa_lightning_indexer', [369, 377], { entityKind: 'Value', mapsTo: ['index_cache'], conditions: ['rotate=True；should_compress'] }),
    semanticNode('csa_query_index_score', 'Query–Index KV score', 'csa_lightning_indexer', [420, 420], { mapsTo: ['lightning_indexer'] }),
    semanticNode('csa_score_weighting', 'Score weighting', 'csa_lightning_indexer', [418, 423], { mapsTo: ['lightning_indexer'], reason: 'weights_proj、ReLU、加权与 head sum；world_size > 1 时官方已有 all_reduce。纯语义画布隐藏通信，不删除源码事实。' }),
    semanticNode('csa_topk_select', 'Top-K select', 'csa_lightning_indexer', [427, 433], { mapsTo: ['exact_topk'] }),
    semanticNode('csa_topk_compressed_positions', 'Top-K compressed positions', 'csa_lightning_indexer', [427, 433], { entityKind: 'Value', mapsTo: ['exact_topk', 'compressed_gather'] }),
    semanticNode('csa_indexer_logical_cache', 'Indexer.kv_cache', 'csa_lightning_indexer', [408, 420], { entityKind: 'LogicalState', mapsTo: ['index_cache'], lifecycle: '跨 decode step 保存检索向量；与主 Attention cache 独立。', sourceRefs: [ref(officialModel, 408, 420, 'call'), ref(officialModel, 398, 399)] }),
    semanticNode('csa_indexer_compressor_state', 'Indexer kv_state / score_state', 'csa_lightning_indexer', [345, 361], { entityKind: 'LogicalState', mapsTo: ['indexer_state'], lifecycle: 'Indexer.compressor 独立实例的跨 token 状态。', sourceRefs: [ref(officialModel, 398, 398, 'call'), ref(officialModel, 301, 304)] }),

    semanticNode('csa_candidate_index_union', 'Candidate/index union', 'csa_sparse_shared_kv_attention', [514, 515], { mapsTo: ['window_gather', 'compressed_gather', 'sparse_attention'], reason: '组合窗口与压缩位置索引；压缩 KV 内容不是索引组合的输入。' }),
    semanticNode('csa_sparse_attention_op', 'Sparse Shared-KV Attention', 'csa_sparse_shared_kv_attention', [533, 533], { mapsTo: ['sparse_attention', 'head_merge'], reason: '当前证据止于 sparse_attn 调用；未导入官方 kernel 定义，不凭名称补写内部算子。' }),
    semanticNode('csa_output_inverse_rope', 'Output inverse RoPE', 'csa_sparse_shared_kv_attention', [534, 534], { mapsTo: ['head_merge'] }),

    semanticNode('csa_group_reshape', 'Group reshape', 'csa_grouped_output_projection', [537, 538], { mapsTo: ['grouped_output_projection', 'head_group_redistribution'] }),
    semanticNode('csa_grouped_low_rank_projection', 'Grouped low-rank projection', 'csa_grouped_output_projection', [541, 541], { mapsTo: ['grouped_output_projection'] }),
    semanticNode('csa_output_projection', 'Output projection', 'csa_grouped_output_projection', [542, 542], { mapsTo: ['grouped_output_projection', 'output_reduce_scatter'], reason: 'RowParallelLinear 定义已有 all_reduce；与 PyPTO Reduce-Scatter 的并行布局对应仍需验证。' })
  ];

  const canonicalSummaryEdges = [
    ['csa_scope_input', 'csa_query_projection', 'normalized hidden → query'],
    ['csa_scope_input', 'csa_recent_window_kv', 'normalized hidden → recent KV'],
    ['csa_scope_input', 'csa_compressed_kv_path', 'normalized hidden → compressed KV'],
    ['csa_scope_input', 'csa_lightning_indexer', 'normalized hidden → indexer'],
    ['csa_query_projection', 'csa_lightning_indexer', 'normalized Q features'],
    ['csa_query_projection', 'csa_sparse_shared_kv_attention', 'main attention Q'],
    ['csa_recent_window_kv', 'csa_sparse_shared_kv_attention', 'recent-window KV'],
    ['csa_compressed_kv_path', 'csa_sparse_shared_kv_attention', 'compressed KV'],
    ['csa_lightning_indexer', 'csa_sparse_shared_kv_attention', 'selected compressed positions'],
    ['csa_sparse_shared_kv_attention', 'csa_grouped_output_projection', 'attention output'],
    ['csa_grouped_output_projection', 'csa_scope_output', 'CSA output']
  ].map(([source, target, label], index) => ({ id: `csa_summary_${index}`, source, target, label }));

  const canonicalExpandedEdges = [
    ['csa_scope_input', 'csa_q_lora_down'],
    ['csa_q_lora_down', 'csa_q_normalization'],
    ['csa_q_normalization', 'csa_main_q_rope'],
    ['csa_q_normalization', 'csa_indexer_q_rope'],
    ['csa_scope_input', 'csa_shared_kv_projection'],
    ['csa_shared_kv_projection', 'csa_kv_normalization'],
    ['csa_kv_normalization', 'csa_kv_rope'],
    ['csa_scope_input', 'csa_recent_positions_kv'],
    ['csa_kv_rope', 'csa_logical_kv_cache', 'state-write'],
    ['csa_logical_kv_cache', 'csa_sparse_attention_op', 'state-read'],
    ['csa_scope_input', 'csa_main_compressor_ratio4'],
    ['csa_main_compressor_ratio4', 'csa_main_compressed_kv'],
    ['csa_main_compressor_state', 'csa_main_compressor_ratio4', 'state-read'],
    ['csa_main_compressor_ratio4', 'csa_main_compressor_state', 'state-write'],
    ['csa_main_compressed_kv', 'csa_sparse_attention_op'],
    ['csa_scope_input', 'csa_indexer_compressor_ratio4'],
    ['csa_indexer_compressor_ratio4', 'csa_indexer_kv_representation'],
    ['csa_indexer_q_rope', 'csa_query_index_score'],
    ['csa_indexer_kv_representation', 'csa_indexer_logical_cache', 'state-write'],
    ['csa_indexer_logical_cache', 'csa_query_index_score', 'state-read'],
    ['csa_indexer_compressor_state', 'csa_indexer_compressor_ratio4', 'state-read'],
    ['csa_indexer_compressor_ratio4', 'csa_indexer_compressor_state', 'state-write'],
    ['csa_scope_input', 'csa_score_weighting'],
    ['csa_query_index_score', 'csa_score_weighting'],
    ['csa_score_weighting', 'csa_topk_select'],
    ['csa_topk_select', 'csa_topk_compressed_positions'],
    ['csa_recent_positions_kv', 'csa_candidate_index_union'],
    ['csa_topk_compressed_positions', 'csa_candidate_index_union'],
    ['csa_main_q_rope', 'csa_sparse_attention_op'],
    ['csa_candidate_index_union', 'csa_sparse_attention_op'],
    ['csa_sparse_attention_op', 'csa_output_inverse_rope'],
    ['csa_output_inverse_rope', 'csa_group_reshape'],
    ['csa_group_reshape', 'csa_grouped_low_rank_projection'],
    ['csa_grouped_low_rank_projection', 'csa_output_projection'],
    ['csa_output_projection', 'csa_scope_output']
  ].map(([source, target, semanticEdgeType = 'activation'], index) => ({
    id: `csa_expanded_${index}`, source, target, semanticEdgeType,
    tensor: { name: semanticEdgeType === 'activation' ? 'value' : semanticEdgeType },
    provenance: [officialModel], recurrence: semanticEdgeType.startsWith('state-')
  }));

  // Shared by the left model overview and the right L1 scope. Coordinates are
  // local to the CSA scope so both renderers project the same visual graph.
  const l1Projection = Object.freeze({
    label: 'CSA · Compressed Sparse Attention',
    width: 750,
    height: 560,
    inputAnchor: Object.freeze({ x: 375, y: -58 }),
    outputAnchor: Object.freeze({ x: 375, y: 618 }),
    nodes: Object.freeze({
      csa_query_projection: Object.freeze({ x: 375, y: 100, width: 320, height: 56, colorKey: 'sem:linear' }),
      csa_recent_window_kv: Object.freeze({ x: 170, y: 215, width: 300, height: 56, colorKey: 'sem:attention' }),
      csa_compressed_kv_path: Object.freeze({ x: 560, y: 215, width: 340, height: 56, colorKey: 'sem:attention' }),
      csa_lightning_indexer: Object.freeze({ x: 375, y: 310, width: 270, height: 56, colorKey: 'sem:gate' }),
      csa_sparse_shared_kv_attention: Object.freeze({ x: 375, y: 410, width: 350, height: 56, colorKey: 'sem:attention' }),
      csa_grouped_output_projection: Object.freeze({ x: 375, y: 500, width: 330, height: 56, colorKey: 'sem:linear' })
    })
  });

  const l1Compounds = compoundSpecs.map(spec => node(spec.id, 'L1', spec.label, 'Compound', {
    parentId: 'l1_csa',
    parentScope: 'csa_model_semantic_scope',
    children: spec.children,
    mapsTo: spec.children,
    sourceRefs: spec.sourceRefs,
    provenance: spec.sourceRefs.map(item => `DeepSeek official inference/model.py:L${item.start}–${item.end}`),
    inputPorts: ['in'],
    outputPorts: ['out'],
    summaryEdges: canonicalSummaryEdges.filter(edge => edge.source === spec.id || edge.target === spec.id),
    expandedEdges: canonicalExpandedEdges.filter(edge => spec.children.includes(edge.source) || spec.children.includes(edge.target)),
    defaultExpanded: true
  }));

  // Compatibility anchor only, not a fabricated four-stage compiler graph.
  const l4 = [
    node('l4_qkv_tensor_graph', 'L4', 'qkv_proj_rope 编译血缘', 'Compilation', {
      artifactStatus: 'unavailable',
      parentScope: 'unmatched_compile_snapshot',
      provenance: ['历史编译目录存在；尚无当前源码/配置的匹配证据'],
      graph: { detail: '暂无已验证的 Kernel 映射，请先编译并导入 Pass Dump' }
    })
  ];

  const l1 = node('l1_csa', 'L1', 'Compressed Sparse Attention', 'ModelModule', {
    parentScope: 'DeepSeek V4 Official model architecture',
    mapsTo: l1Compounds.map(item => item.id),
    children: l1Compounds.map(item => item.id),
    provenance: ['DeepSeek official inference/model.py:L442–548'],
    sourceRefs: [ref(officialModel, 484, 543, 'call'), ref(officialModel, 436, 482)],
    summaryEdges: canonicalSummaryEdges,
    expandedEdges: canonicalExpandedEdges,
    defaultExpanded: false
  });

  const contextNodes = [
    node('official_hc_pre', 'L2', 'HC Pre · 模型上下文', 'Compute', { applicableScope: 'official_block', sourceRefs: [ref(officialModel, 690, 690, 'call')] }),
    node('csa_scope_input', 'L2', 'RMSNorm hidden state', 'Tensor', { entityKind: 'Value', applicableScope: 'official_block', sourceRefs: [ref(officialModel, 691, 692, 'call')] }),
    node('official_hc_post', 'L2', 'HC Post · 模型上下文', 'Compute', { applicableScope: 'official_block', sourceRefs: [ref(officialModel, 693, 693, 'call')] }),
    node('csa_scope_output', 'L2', 'CSA semantic output', 'Tensor', { entityKind: 'Value', applicableScope: 'official_block', sourceRefs: [ref(officialModel, 542, 543, 'call')] })
  ];
  // Relations, rather than color tags on isolated nodes, own the transform.
  // These are curated source correspondences, not numerical-equivalence proofs.
  const relation = (id, sourceIds, targetIds, transformTypes, changeKind, explanation) => ({
    id, sourceIds, targetIds, transformTypes, changeKind, introducedBy: 'developer-pypto',
    scope: 'decode_csa_implementation_scope',
    conditions: ['CSA ratio=4', 'decode', '所选 PyPTO TP/CP 配置；并行与精度不默认等同官方运行配置'],
    artifactStatus: 'inferred', evidenceRefs: [],
    explanation: {
      originalMeaning: explanation[0], change: explanation[1], rationale: explanation[2],
      inputOutputChange: explanation[3],
      limitations: explanation[4] || '对应关系为源码对照；未验证数值等价，也不证明 kernel 融合。'
    }
  });
  const lineage = [
    relation('query-group', ['csa_q_lora_down', 'csa_q_normalization', 'csa_main_q_rope'], ['q_projection'], ['PRESERVED', 'GROUPED'], 'grouped', [
      '官方 Q down、归一化、主 Q 投影/scaling/RoPE。', '同一 q_proj_rope 函数承载多个步骤，另有 INT8 qr 与 scale。',
      '开发者显式组织量化投影与位置编码实现。', 'x → q，以及供 Indexer 使用的 qr/qr_scale；Indexer Q 不在此函数计算。'
    ]),
    relation('kv-group', ['csa_shared_kv_projection', 'csa_kv_normalization', 'csa_kv_rope'], ['kv_projection'], ['PRESERVED', 'GROUPED'], 'grouped', [
      '官方共享 KV 投影、归一化、RoPE 与非 RoPE 量化。', 'kv_proj_rope 共同承载投影/归一化/RoPE。',
      '以显式 PyPTO 运算组织 KV 路径。', '组内 token x_normed_full → kv_full；精度策略需分别核对。'
    ]),
    relation('main-compress', ['csa_main_compressor_ratio4'], ['main_compressor'], ['PRESERVED'], 'preserved', [
      '官方 Compressor 对 hidden state 做门控压缩。', 'compressor_ratio4 显式读写状态与压缩 cache。',
      '支持 decode 增量更新。', '组内 hidden rows、position、state → 条件性压缩值与 cache 更新。',
      '压缩值只在压缩边界产生；当前配对尚未做数值等价验证。'
    ]),
    relation('idx-compress', ['csa_indexer_compressor_ratio4'], ['indexer_compressor'], ['PRESERVED', 'EXPLICIT'], 'split', [
      'Indexer 内部拥有独立 Compressor。', '开发者拆为独立 indexer_compressor 调用，先更新检索 cache。',
      'cache 更新运行在 gathered stream，query 检索保持本地 rows。', 'x_normed_full → INT8 index cache 与 scale。'
    ]),
    relation('indexer-group', ['csa_indexer_q_rope', 'csa_query_index_score', 'csa_score_weighting'], ['lightning_indexer'], ['GROUPED'], 'grouped', [
      '官方 Indexer.forward 中 Q 投影/旋转、q·KV、ReLU、加权与 head sum。', 'indexer 函数承载 query、weights、score 等程序。',
      '开发者采用 INT8 query/cache 与分阶段计算。', '本地 qr/qr_scale、hidden rows、paged index cache → scores。',
      '官方有 FP4 simulation 与条件 all_reduce；不能把精度或部署差异宣称等价。'
    ]),
    relation('topk', ['csa_topk_select', 'csa_topk_compressed_positions'], ['exact_topk'], ['EXPLICIT'], 'split', [
      '官方 topk 返回压缩位置索引。', '实现为候选打分及 Top-K forest 分层排序/合并。',
      '显式处理长候选序列和无效位置。', 'scores → topk indices；选择操作与索引输出不是两个算子。'
    ]),
    relation('gather-window', ['csa_recent_positions_kv', 'csa_logical_kv_cache'], ['window_gather'], ['EXPLICIT', 'MATERIALIZED'], 'split', [
      '官方窗口索引与逻辑 KV cache。', '运行时传入 window_swa_indices，按物理 slot 访问近期 KV。',
      '把逻辑位置落实为分页/slot 访问。', '窗口索引与 raw cache → attention 的 KV 读取。',
      '不宣称官方有独立 Gather 算子；窗口索引生成不在 decode_csa 内。'
    ]),
    relation('gather-compressed', ['csa_topk_compressed_positions', 'csa_main_compressed_kv'], ['compressed_gather'], ['EXPLICIT', 'MATERIALIZED'], 'split', [
      '官方压缩位置与压缩 KV 内容。', '通过 cmp_block_table 将选中位置解析到压缩 KV 存储。',
      '稀疏计算只访问候选 KV。', 'topk indices、block table、compressed cache → KV 读取。'
    ]),
    relation('attention-split', ['csa_candidate_index_union', 'csa_sparse_attention_op'], ['window_gather', 'compressed_gather', 'sparse_attention', 'head_merge'], ['SPLIT', 'EXPLICIT'], 'split', [
      '官方组合索引后调用 sparse_attn。', 'PyPTO 显式组织 slot plan、QK/PV、块统计及 softmax 合并。',
      '分块实现稀疏注意力并汇总 partial results。', 'q、候选 KV → mi/li/oi → 合并 attention output。',
      '官方 sparse_attn 内部定义未导入；L2 止于该调用；实现拆分不代表新数学语义。'
    ]),
    relation('inverse-rope', ['csa_output_inverse_rope'], ['head_merge'], ['GROUPED'], 'grouped', [
      '官方 attention 输出执行 inverse RoPE。', '逆 RoPE 在 block merge 的实现范围内进行。',
      '组织合并输出与位置还原步骤。', 'merged attention → inverse-RoPE attention。'
    ]),
    relation('output-group', ['csa_group_reshape', 'csa_grouped_low_rank_projection', 'csa_output_projection'], ['grouped_output_projection'], ['PRESERVED', 'GROUPED'], 'grouped', [
      '官方 group reshape、grouped einsum 与 wo_b。', 'o_proj_reduce_scatter 函数同时承载投影及通信。',
      '输出投影按 group ownership 执行。', 'local head groups → partial output；函数边界不是单 kernel。'
    ]),
    relation('kv-storage', ['csa_logical_kv_cache', 'csa_main_compressed_kv'], ['raw_kv_cache', 'compressed_kv_cache'], ['MATERIALIZED'], 'split', [
      'Attention.kv_cache 已保存窗口区与压缩区。', '拆为显式 kv_cache/cmp_kv 分页参数。',
      '由调用方管理物理缓存并复用历史内容。', '逻辑缓存的区域 → 两个物理 cache 参数；不是新增缓存语义。'
    ]),
    relation('main-state', ['csa_main_compressor_state'], ['compressor_state'], ['MATERIALIZED'], 'preserved', [
      '官方 kv_state / score_state 为跨 token 压缩状态。', '以 compress_state 参数与 state slot/table 外显。',
      '将模块内部状态交给运行时管理。', '隐式 buffer 读写 → 显式状态参数读写。'
    ]),
    relation('idx-state', ['csa_indexer_compressor_state'], ['indexer_state'], ['MATERIALIZED'], 'preserved', [
      'Indexer Compressor 有自己的 kv_state / score_state。', 'inner_compress_state 是独立显式状态。',
      '主压缩与检索压缩不可串联或共享同一状态实例。', '独立逻辑状态 → 独立物理状态参数。'
    ]),
    relation('idx-storage', ['csa_indexer_logical_cache', 'csa_indexer_kv_representation'], ['index_cache'], ['MATERIALIZED'], 'preserved', [
      '官方 Indexer.kv_cache 保存检索向量。', 'INT8 idx_kv_cache 与 FP32 idx_kv_scale 参数。',
      '采用量化检索存储表示。', '逻辑检索历史 → cache/scale；精度策略是开发者实现选择。'
    ]),
    relation('addressing', ['csa_logical_kv_cache', 'csa_main_compressor_state', 'csa_indexer_logical_cache', 'csa_indexer_compressor_state'], ['slot_mapping', 'block_table'], ['MATERIALIZED', 'EXPLICIT'], 'added', [
      '官方使用逻辑位置、切片和内部 buffer。', '调用方传入 ori/cmp/idx/state slots 与 block tables。',
      '把 token/请求逻辑位置映射到分页物理存储。', '元数据输入 → cache 写行、压缩状态定位、检索和 KV 读取。',
      '这些是 RuntimeMetadata，不是生成它们的算子；生成者不在当前 scope 中，生命周期尚未验证。'
    ]),
    relation('token-deployment', ['csa_scope_input'], ['tp_allgather'], ['DEPLOYED'], 'added', [
      '归一化 hidden state 供 Query、KV、Compressor 使用。', '当前 CP/TP 程序收集 local rows 为组内 token stream。',
      'KV 和两个 Compressor 需要组内 rows，query 路径仍用本地 rows。', 'x_normed_t → x_normed_full。',
      '相对纯语义 scope 为实现增量，不代表所有官方部署均无通信。'
    ]),
    relation('head-deployment', ['csa_group_reshape'], ['head_group_redistribution', 'attention_alltoall'], ['DEPLOYED', 'EXPLICIT'], 'added', [
      '官方按 group reshape attention output。', '按 destination rank 打包、put、notify，并等待后收集。',
      '把 attention heads 搬到执行对应输出投影的 owner rank。', '本 rank attention groups → owner rank local groups。',
      'ownership 来自当前 PyPTO 程序；不能从图上位置推测硬件分配。'
    ]),
    relation('output-deployment', ['csa_output_projection'], ['output_reduce_scatter'], ['DEPLOYED'], 'inferred', [
      '官方 wo_b / RowParallelLinear 在多 rank 时已有 all_reduce。', 'PyPTO 发布输出片段、同步并 reduce-scatter 回 token owner。',
      '按当前 token/output ownership 汇总投影贡献。', 'partial output → token-owner output。',
      '尚未验证两种并行布局的严格对应；不宣称 Reduce-Scatter 直接等价替换官方 all_reduce。'
    ]),
    relation('implementation-scope', ['official_hc_pre', 'csa_scope_input', 'official_hc_post', 'csa_scope_output'], ['hc_input', 'hc_pre', 'attention_norm', 'attention_hc_post', 'attention_hc_output'], ['SCOPE_INCLUDED'], 'scope-included', [
      'Block.forward 中的 HC Pre、RMSNorm 与 HC Post 已经存在。', 'decode_csa 将这些上下文纳入同一实现入口。',
      '入口接受并返回 HC residual stream。', 'CSA 的 hidden→output 边界扩大为 HC input→HC output。',
      '范围扩大不是新增模型数学语义；HC residual 是跨子层的值，不因名为 State 就认定为跨 token cache。'
    ])
  ];
  const implementationPrograms = {
    q_projection: {
      children: [
        ['pypto_q_down', 'Q down projection', 268, 298, ['csa_q_lora_down']],
        ['pypto_q_norm_quant', 'Q RMSNorm / INT8 quant', 300, 368, ['csa_q_normalization']],
        ['pypto_q_matmul', 'Main Q matmul', 370, 420, ['csa_main_q_rope']],
        ['pypto_q_rope', 'Q dequant / norm / RoPE', 422, 553, ['csa_main_q_rope']]
      ],
      edges: [['pypto_q_down', 'pypto_q_norm_quant'], ['pypto_q_norm_quant', 'pypto_q_matmul'], ['pypto_q_matmul', 'pypto_q_rope']],
      input: 'pypto_q_down', output: 'pypto_q_rope', outputPorts: { lightning_indexer: 'pypto_q_norm_quant' }
    },
    kv_projection: {
      children: [
        ['pypto_kv_matmul', 'KV projection', 575, 613, ['csa_shared_kv_projection']],
        ['pypto_kv_norm_rope', 'KV RMSNorm / RoPE', 615, 760, ['csa_kv_normalization', 'csa_kv_rope']]
      ],
      edges: [['pypto_kv_matmul', 'pypto_kv_norm_rope']],
      input: 'pypto_kv_matmul', output: 'pypto_kv_norm_rope', outputPorts: {}
    }
  };
  Object.assign(implementationPrograms, global.PtoCsaProgramDetails || {});
  // Preservation describes the source operation, not bitwise numeric equivalence.
  const preservedExpressions = new Set(['pypto_q_down', 'pypto_kv_matmul', 'merge_inverse_rope']);
  const programNodes = Object.entries(implementationPrograms).flatMap(([parentId, program]) => {
    l3Nodes.find(item => item.id === parentId).children = program.children.map(child => child[0]);
    l3Nodes.find(item => item.id === parentId).entityKind = 'Scope';
    return program.children.flatMap(([id, label, start, end, sourceIds]) => {
      const preserved = preservedExpressions.has(id);
      lineage.push(relation(id + '-source', sourceIds, [id], [preserved ? 'PRESERVED' : 'EXPLICIT'], preserved ? 'preserved' : 'split', [
        '官方语义步骤，见源端表达式。', label + ' 在 PyPTO 函数内部显式实现。',
        '暴露函数内步骤，支持与源语义持续对照。', '输入输出依赖以当前 PyPTO 表达式为准。',
        '这是开发者程序步骤，不是编译生成的 kernel；量化数值对应仍未验证。'
      ]));
      const existing = l3Nodes.find(item => item.id === id);
      if (existing) {
        existing.parentId = parentId;
        existing.parentScope = parentId;
        existing.label = label;
        existing.sourceRefs = [existing.sourceRefs[0], ref(program.file, start, end, 'expression')];
        return [];
      }
      return node(id, 'L3', label, 'Compute', {
        parentId, parentScope: parentId, sourceRefs: [ref(program.file || qkv, start, end, 'expression')],
        conditions: program.conditions || [],
        artifactStatus: 'inferred',
        mappingType: 'EXPANDED', reason: 'PyPTO 函数内部步骤；不是 kernel 分配证据。'
      });
    });
  });
  const nodes = [l1, ...l1Compounds, ...l2, ...contextNodes, ...l3Nodes, ...programNodes, ...l4];
  const byId = new Map(nodes.map(item => [item.id, item]));
  ['hc_input', 'attention_hc_output'].forEach(id => { byId.get(id).entityKind = 'Value'; });
  ['slot_mapping', 'block_table'].forEach(id => {
    byId.get(id).entityKind = 'RuntimeMetadata';
    byId.get(id).lifecycle = '调用方提供；持久性尚未验证';
  });
  lineage.forEach(item => {
    item.evidenceRefs = [...item.sourceIds, ...item.targetIds].flatMap(id => byId.get(id)?.sourceRefs || []);
  });
  const relationsFor = id => {
    const ids = new Set();
    const visit = key => { if (ids.has(key)) return; ids.add(key); (byId.get(key)?.children || []).forEach(visit); };
    visit(id);
    return lineage.filter(item => [...item.sourceIds, ...item.targetIds].some(key => ids.has(key)));
  };
  [...l3Nodes, ...programNodes].forEach(item => {
    const relations = relationsFor(item.id);
    item.transformTypes = [...new Set(relations.flatMap(entry => entry.transformTypes))];
    item.mappingType = item.transformTypes.includes('DEPLOYED') ? 'DEPLOY'
      : item.transformTypes.includes('MATERIALIZED') ? 'STATE'
      : item.transformTypes.some(type => ['GROUPED', 'SPLIT', 'EXPLICIT'].includes(type)) ? 'EXPANDED'
      : item.transformTypes.includes('PRESERVED') ? 'SAME' : null;
    item.reason = relations.map(entry => entry.explanation.change).join(' ');
  });
  // L2-relative display classification is independent of source transformation tags.
  const diffResources = new Set(['slot_mapping', 'block_table', 'raw_kv_cache', 'compressed_kv_cache',
    'compressor_state', 'indexer_state', 'index_cache', 'window_gather', 'compressed_gather',
    'cmp_commit', 'cmp_write', 'idx_commit', 'idx_write']);
  const diffDeployment = new Set(['tp_allgather', 'head_group_redistribution', 'attention_alltoall', 'output_reduce_scatter']);
  const diffPreserved = new Set(['hc_input', 'hc_pre', 'attention_norm', 'attention_hc_post', 'attention_hc_output', 'pypto_q_down', 'pypto_kv_matmul', 'merge_inverse_rope', 'output_low_rank', 'output_projection']);
  function diffFor(id) {
    const item = byId.get(id);
    const sourceIds = [...new Set(relationsFor(id).flatMap(entry => entry.sourceIds))]
      .filter(key => byId.get(key)?.level === 'L2');
    let category = 'retained';
    if (diffDeployment.has(id)) category = 'deploy';
    else if (diffResources.has(id)) category = 'state';
    else if (item?.level === 'L3' && item.entityKind !== 'Scope' && sourceIds.length && !diffPreserved.has(id)) category = 'expanded';
    const reasons = {
      retained: '保留 L2 已有计算，或作为实现范围上下文／折叠摘要；不计为新增节点。',
      expanded: '从 L2 计算语义进一步拆出、组合或具体化的实现步骤；不表示新增模型算法。',
      state: 'L2 的逻辑状态、值或位置在 L3 显式表达为缓存、读写、递推更新或寻址资源；不是新增缓存语义。',
      deploy: '相对 L2，部署方案新增的通信或数据所有权重分布。'
    };
    return { category, sourceIds, reason: reasons[category] };
  }
  // Historical dumps are not bound to the currently inspected source snapshot.
  l4.forEach(item => { item.artifactStatus = 'unavailable'; item.graph = { detail: '历史产物尚未与当前源码/配置快照匹配。' }; });
  nodes.forEach(item => item.sourceRefs.forEach(reference => {
    const manifest = global.PtoCsaSourceManifest?.[reference.file];
    reference.hash = manifest?.hash || null;
    reference.version = manifest ? 'local-sha256' : 'unbound';
    const declaration = manifest?.symbols?.find(symbol => symbol.line >= reference.start && symbol.line <= reference.start + 1)
      || manifest?.symbols?.filter(symbol => symbol.line <= reference.start).at(-1);
    reference.symbol = reference.symbol || declaration?.name || 'module expression';
    reference.artifactStatus = manifest && reference.end <= manifest.lines ? 'verified' : 'unavailable';
  }));
  const childToCompound = new Map(l1Compounds.flatMap(item => item.children.map(childId => [childId, item.id])));
  const canonicalGraph = Object.freeze({
    id: 'csa_canonical_hierarchical_graph',
    rootId: l1.id,
    scopeId: 'csa_model_semantic_scope',
    inputId: 'csa_scope_input',
    outputId: 'csa_scope_output',
    compoundIds: Object.freeze(l1Compounds.map(item => item.id)),
    leafIds: Object.freeze(l2.map(item => item.id)),
    l1Projection,
    summaryEdges: Object.freeze(canonicalSummaryEdges),
    expandedEdges: Object.freeze(canonicalExpandedEdges)
  });
  global.PtoCsaMappingRegistry = Object.freeze({
    schemaVersion: 'pto.mapping_registry.v3',
    taskId: 'PTO-MAPPING-L1-L2-REVERSIBLE-ZOOM-003',
    artifactRoot: '../../../Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617',
    artifactStatus: 'unavailable',
    compilation: Object.freeze({ compileRunId: '_jit_l3_decode_csa_20260903_010617', sourceMatch: 'unavailable', configMatch: 'unavailable', artifactStatus: 'unavailable' }),
    context: Object.freeze({ scope: 'CSA', layer: 2, phase: 'decode', ratio: 4, window: 128, extraction: 'source-curated', config: '../../../Data/DeepSeek-V4-Flash-Official/inference-config.json' }),
    lineage: Object.freeze(lineage),
    implementationPrograms: Object.freeze(implementationPrograms),
    defaultExpandedPrograms: Object.freeze(Object.keys(implementationPrograms)),
    implementationEntries: Object.freeze({
      l1_csa: Object.freeze({
        sourceScopeId: 'l1_csa', targetScopeId: 'decode_csa_implementation_scope',
        sourceNodeId: 'hc_input', relationIds: Object.freeze(lineage.map(item => item.id)),
        sourceContext: '官方整网 / Decoder Layer 2 / Attention / CSA · decode · ratio 4',
        targetContext: 'decode_csa.py · 开发者 PyPTO 实现；包含 HC Pre、RMSNorm、HC Post',
        artifactStatus: 'inferred'
      })
    }),
    relationsFor,
    diffFor,
    mappedIds(id, targetLevel) {
      const related = relationsFor(id);
      return [...new Set(related.flatMap(item => targetLevel === 'L3' ? item.targetIds : item.sourceIds))];
    },
    annotationsFor(id) {
      const item = byId.get(id);
      if (!item) return [];
      const own = [
        `# [Explorer 注释] ${item.label} · ${item.entityKind} · ${item.origin}`,
        ...(item.lifecycle ? [`# 生命周期：${item.lifecycle}`] : []),
        ...(item.reason && item.level !== 'L3' ? [`# 语义边界：${item.reason}`] : []),
        ...item.conditions.map(condition => `# 条件：${condition}`)
      ];
      if (!item.sourceRefs.some(reference => reference.role === 'definition')) own.push('# 无独立定义证据：此处定位实际表达式、声明或调用，不伪造函数。');
      const relations = relationsFor(id);
      if (!relations.length) own.push('# 映射：当前没有已知实现对应；未补造节点。');
      return own.concat(relations.flatMap(entry => [
        `# [Explorer · ${entry.transformTypes.join(' / ')} · ${entry.artifactStatus}]`,
        `# 来源：${entry.sourceIds.map(key => byId.get(key)?.label || key).join('、')}`,
        `# 实现：${entry.targetIds.map(key => byId.get(key)?.label || key).join('、')}`,
        `# 原始语义：${entry.explanation.originalMeaning}`,
        `# 变化：${entry.explanation.change}`,
        `# 原因：${entry.explanation.rationale}`,
        `# 输入输出：${entry.explanation.inputOutputChange}`,
        `# 引入者：${entry.introducedBy}；${entry.conditions.join('；')}`,
        `# 限制：${entry.explanation.limitations}`
      ]));
    },
    artifactStatuses: Object.freeze(['verified', 'inferred', 'unavailable']),
    levels: Object.freeze({ L1: '模型架构', L2: '计算语义', L3: 'PyPTO 执行', L4: 'Kernel/硬件' }),
    canonicalGraph,
    nodes: Object.freeze(nodes),
    getNode(id) { return byId.get(id) || null; },
    nodesAt(level) { return nodes.filter(item => item.level === level); },
    compoundFor(id) {
      const item = byId.get(id);
      if (item?.nodeType === 'Compound') return item;
      return byId.get(childToCompound.get(id)) || null;
    },
    resolve(id) {
      const current = byId.get(id);
      return current?.mapsTo.map(target => byId.get(target)).filter(Boolean) || [];
    }
  });
})(window);
