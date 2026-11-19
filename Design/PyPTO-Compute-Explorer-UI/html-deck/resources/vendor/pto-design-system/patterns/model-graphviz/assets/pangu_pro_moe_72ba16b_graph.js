/* Pangu Pro MoE 72BA16B · communication profile.
   This graph follows the paper-style single decoder layer view and keeps the
   visual focus on communication operators. Parallel configuration facts stay
   in the page metadata instead of being rendered as graph nodes. */
(() => {
  const flow = (source, target, extra = {}) => ({ source, target, ...extra });
  const comm = (source, target, tag, extra = {}) => flow(source, target, { tag, edgeType: 'communication', ...extra });
  const param = (source, target, tag = 'W', extra = {}) => flow(source, target, { tag, edgeType: 'parameter', dashed: true, ...extra });
  const node = (id, label, typeLabel, kind, x, y, width, height, colorKey, extra = {}) => ({
    id, label, typeLabel, kind, x, y, width, height, colorKey, ...extra,
  });
  const glyph = (id, x, y, parent) => node(id, '+', 'Add', 'op', x, y, 38, 38, 'sem:add', {
    glyph: true,
    hideTypeLabel: true,
    parent,
  });

  window.PANGU_GRAPH = {
    width: 1500,
    height: 1320,
    clusters: [
      { id: 'decoder_profile', label: 'Decoder Layer Communication Profile', x: 36, y: 34, width: 720, height: 1248, colorKey: 'module:decoder', repeat: 48 },
      { id: 'attention_path', label: 'Attention Path', x: 92, y: 118, width: 568, height: 374, colorKey: 'sem:attention', parent: 'decoder_profile' },
      { id: 'moe_path', label: 'MoE FFN Path', x: 62, y: 526, width: 650, height: 704, colorKey: 'module:moe', parent: 'decoder_profile' },
      { id: 'mulattention_detail', label: 'MulAttention Children', x: 826, y: 48, width: 626, height: 426, colorKey: 'sem:attention', parent: 'attention_path' },
      { id: 'swiftgmm_detail', label: 'SwiftGMM Children', x: 850, y: 536, width: 558, height: 588, colorKey: 'sem:moe', parent: 'moe_path' },
    ],
    nodes: [
      node('hidden_in', 'Hidden States', 'Tensor', 'tensor', 378, 88, 210, 48, 'io:input', { parent: 'attention_path' }),

      node('attn_norm', 'RMSNorm', 'Op', 'op', 378, 168, 232, 54, 'sem:norm', { parent: 'attention_path' }),
      node('attn_all_gather', 'AllGather', 'Comm', 'op', 378, 248, 232, 52, 'sem:comm', { parent: 'attention_path' }),
      node('attention_gqa', 'Attention(GQA)', 'Op', 'op', 378, 330, 258, 62, 'sem:attention', { parent: 'attention_path' }),
      node('attn_reduce_scatter', 'Reduce-Scatter', 'Comm', 'op', 378, 420, 242, 52, 'sem:comm', { parent: 'attention_path' }),
      glyph('attn_residual_add', 378, 494, 'attention_path'),

      node('moe_norm', 'RMSNorm', 'Op', 'op', 378, 594, 232, 54, 'sem:norm', { parent: 'moe_path' }),
      node('moe_all_gather', 'AllGather', 'Comm', 'op', 378, 674, 232, 52, 'sem:comm', { parent: 'moe_path' }),
      node('router_gate', 'Gating', 'Op', 'op', 552, 770, 206, 54, 'sem:gate', { parent: 'moe_path' }),
      node('shared_expert', 'Shared Expert FFN', 'Op', 'op', 256, 806, 258, 68, 'sem:mlp', { parent: 'moe_path' }),
      node('a2a_dispatch', 'All-to-All Dispatch', 'Comm', 'op', 552, 850, 260, 52, 'sem:comm', { parent: 'moe_path' }),
      node('routed_expert', 'Routed Expert FFN', 'Op', 'op', 552, 932, 268, 70, 'sem:moe', { parent: 'moe_path' }),
      node('a2a_combine', 'All-to-All Combine', 'Comm', 'op', 552, 1014, 260, 52, 'sem:comm', { parent: 'moe_path' }),
      glyph('moe_branch_add', 378, 1092, 'moe_path'),
      node('moe_reduce_scatter', 'Reduce-Scatter', 'Comm', 'op', 378, 1164, 242, 52, 'sem:comm', { parent: 'moe_path' }),
      glyph('block_residual_add', 378, 1238, 'moe_path'),
      node('hidden_out', 'Layer Output', 'Tensor', 'tensor', 378, 1310, 200, 48, 'io:output', { parent: 'decoder_profile' }),

      node('attn_norm_gamma', 'Norm Gamma', 'Parameter', 'tensor', 116, 168, 168, 46, 'io:parameter', { parent: 'attention_path' }),
      node('qkv_weight', 'QKV Weight', 'Parameter', 'tensor', 622, 330, 170, 46, 'io:parameter', { parent: 'attention_path' }),
      node('oproj_weight', 'O-Proj Weight', 'Parameter', 'tensor', 622, 420, 174, 46, 'io:parameter', { parent: 'attention_path' }),
      node('moe_norm_gamma', 'Norm Gamma', 'Parameter', 'tensor', 116, 594, 168, 46, 'io:parameter', { parent: 'moe_path' }),
      node('shared_weight', 'Shared FFN Weight', 'Parameter', 'tensor', 18, 806, 188, 46, 'io:parameter', { parent: 'moe_path' }),
      node('router_weight', 'Router Weight', 'Parameter', 'tensor', 820, 770, 170, 46, 'io:parameter', { parent: 'moe_path' }),
      node('expert_weight', 'Expert Weight Bank', 'Parameter', 'tensor', 840, 932, 196, 46, 'io:parameter', { parent: 'moe_path' }),

      node('mul_inner_loop', 'Inner Loop', 'Loop', 'op', 1140, 112, 184, 42, 'sem:loop', { parent: 'mulattention_detail' }),
      node('mul_kv_tiles', 'KV Tiles', 'Tensor', 'tensor', 1140, 172, 224, 46, 'io:state', { parent: 'mulattention_detail' }),
      node('mul_left_blocks', 'Outer Blocks', 'Tensor', 'tensor', 942, 286, 144, 144, 'io:state', { parent: 'mulattention_detail' }),
      node('mul_core', 'QK + Softmax + V', 'Op', 'op', 1140, 286, 238, 58, 'sem:attention', { parent: 'mulattention_detail' }),
      node('mul_right_blocks', 'Output Blocks', 'Tensor', 'tensor', 1340, 286, 150, 144, 'io:state', { parent: 'mulattention_detail' }),
      node('mul_output_tiles', 'Output Tiles', 'Tensor', 'tensor', 1140, 400, 224, 46, 'io:output', { parent: 'mulattention_detail' }),
      node('mul_outer_loop_l', 'Outer Loop', 'Loop', 'op', 934, 420, 150, 42, 'sem:loop', { parent: 'mulattention_detail' }),
      node('mul_outer_loop_r', 'Outer Loop', 'Loop', 'op', 1342, 420, 150, 42, 'sem:loop', { parent: 'mulattention_detail' }),

      node('swift_input_groups', 'Token Groups', 'Tensor', 'tensor', 972, 728, 156, 154, 'io:input', { parent: 'swiftgmm_detail' }),
      node('swift_grouped_matmul', 'Grouped MatMul', 'Op', 'op', 972, 884, 214, 58, 'sem:moe', { parent: 'swiftgmm_detail' }),
      node('swift_weight_n', 'Expert N Weight', 'Parameter', 'tensor', 1214, 650, 194, 62, 'io:parameter', { parent: 'swiftgmm_detail' }),
      node('swift_out_n', 'Expert N Out', 'Tensor', 'tensor', 1214, 754, 194, 48, 'io:output', { parent: 'swiftgmm_detail' }),
      node('swift_weight_next', 'Expert N+1 Weight', 'Parameter', 'tensor', 1214, 898, 194, 62, 'io:parameter', { parent: 'swiftgmm_detail' }),
      node('swift_out_next', 'Expert N+1 Out', 'Tensor', 'tensor', 1214, 1004, 194, 48, 'io:output', { parent: 'swiftgmm_detail' }),
    ],
    edges: [
      flow('hidden_in', 'attn_norm', { sourceAnchor: 'bottom', targetAnchor: 'top' }),
      flow('hidden_in', 'attn_residual_add', {
        sourceAnchor: 'bottom',
        targetAnchor: 'center',
        waypoints: [{ x: 238, y: 158 }, { x: 238, y: 454 }, { x: 336, y: 494 }],
        route: 'smooth',
      }),
      param('attn_norm_gamma', 'attn_norm', 'gamma', { targetAnchor: 'left', curve: 'horizontal' }),
      comm('attn_norm', 'attn_all_gather', 'AllGather'),
      flow('attn_all_gather', 'attention_gqa'),
      param('qkv_weight', 'attention_gqa', 'Wqkv', { sourceAnchor: 'left', targetAnchor: 'right' }),
      param('oproj_weight', 'attention_gqa', 'Wo', { sourceAnchor: 'left', targetAnchor: 'right', curve: 'horizontal' }),
      comm('attention_gqa', 'attn_reduce_scatter', 'ReduceScatter'),
      flow('attn_reduce_scatter', 'attn_residual_add', { targetAnchor: 'center' }),
      flow('attn_residual_add', 'moe_norm', { sourceAnchor: 'bottom', targetAnchor: 'top' }),
      flow('attn_residual_add', 'block_residual_add', {
        sourceAnchor: 'bottom',
        targetAnchor: 'center',
        waypoints: [{ x: 128, y: 650 }, { x: 128, y: 1190 }, { x: 332, y: 1238 }],
        route: 'smooth',
      }),

      param('moe_norm_gamma', 'moe_norm', 'gamma', { targetAnchor: 'left', curve: 'horizontal' }),
      comm('moe_norm', 'moe_all_gather', 'AllGather'),
      flow('moe_all_gather', 'shared_expert', {
        sourceAnchor: 'bottom',
        targetAnchor: 'top',
      }),
      param('shared_weight', 'shared_expert', 'W', { sourceAnchor: 'right', targetAnchor: 'left', curve: 'horizontal' }),
      flow('moe_all_gather', 'router_gate', {
        sourceAnchor: 'bottom',
        targetAnchor: 'top',
      }),
      param('router_weight', 'router_gate', 'W', { sourceAnchor: 'left', targetAnchor: 'right', curve: 'horizontal' }),
      comm('router_gate', 'a2a_dispatch', 'A2A Dispatch'),
      comm('a2a_dispatch', 'routed_expert', 'tokens'),
      param('expert_weight', 'routed_expert', 'W', { sourceAnchor: 'left', targetAnchor: 'right', curve: 'horizontal' }),
      comm('routed_expert', 'a2a_combine', 'A2A Combine'),
      flow('shared_expert', 'moe_branch_add', { sourceAnchor: 'bottom', targetAnchor: 'center' }),
      flow('a2a_combine', 'moe_branch_add', { sourceAnchor: 'bottom', targetAnchor: 'center' }),
      comm('moe_branch_add', 'moe_reduce_scatter', 'ReduceScatter'),
      flow('moe_reduce_scatter', 'block_residual_add', { targetAnchor: 'center' }),
      flow('block_residual_add', 'hidden_out'),

      flow('mul_inner_loop', 'mul_kv_tiles', { tag: 'inner' }),
      flow('mul_kv_tiles', 'mul_core', { tag: 'tile' }),
      flow('mul_left_blocks', 'mul_core', { tag: 'outer', sourceAnchor: 'right', targetAnchor: 'left', curve: 'horizontal' }),
      flow('mul_core', 'mul_right_blocks', { tag: 'outer', sourceAnchor: 'right', targetAnchor: 'left', curve: 'horizontal' }),
      flow('mul_core', 'mul_output_tiles', { tag: 'inner' }),

      flow('swift_input_groups', 'swift_grouped_matmul', { tag: 'group' }),
      param('swift_weight_n', 'swift_grouped_matmul', 'Wn', { sourceAnchor: 'left', targetAnchor: 'right', curve: 'horizontal' }),
      param('swift_weight_next', 'swift_grouped_matmul', 'Wn+1', { sourceAnchor: 'left', targetAnchor: 'right', curve: 'horizontal' }),
      flow('swift_grouped_matmul', 'swift_out_n', { tag: 'out', sourceAnchor: 'right', targetAnchor: 'left', curve: 'horizontal' }),
      flow('swift_grouped_matmul', 'swift_out_next', { tag: 'out', sourceAnchor: 'right', targetAnchor: 'left', curve: 'horizontal' }),
    ],
    trainingEvidence: {
      decoder_profile: {
        dimension: '视图口径',
        metric: 'single decoder layer',
        what: '这是 Pangu Pro MoE 72BA16B 的单层通信剖面图，用来说明 Attention、Shared Expert、Routed Expert 三条路径的通信边界。',
        evidence: ['总模型为 71.99B 参数 / 16.50B 激活参数', '图上只展示通信算子和计算分支', '并行配置保留在页面元信息中，不作为图节点渲染'],
      },
      attn_all_gather: {
        dimension: '通信算子',
        metric: 'AllGather',
        what: 'RMSNorm 后把 TP rank 上的 hidden 分片聚合成 Attention 计算需要的输入视图。',
        evidence: ['论文图把 AllGather 放在 RMSNorm 之后', '这是 AllReduce 拆分为 AllGather + Reduce-Scatter 的一部分'],
        relatedNodeIds: ['attn_norm', 'attention_gqa', 'attn_reduce_scatter'],
      },
      attn_reduce_scatter: {
        dimension: '通信算子',
        metric: 'Reduce-Scatter',
        what: 'Attention 输出先规约再重新切回各 TP rank 的 hidden 分片，用于后续残差和 MoE 输入。',
        relatedNodeIds: ['attention_gqa', 'attn_residual_add'],
      },
      moe_all_gather: {
        dimension: '通信算子',
        metric: 'AllGather',
        what: 'MoE FFN 前聚合 hidden 输入，让 shared expert 和 router/routed expert 分支都能拿到所需 token 表示。',
        relatedNodeIds: ['shared_expert', 'router_gate'],
      },
      a2a_dispatch: {
        dimension: '通信算子',
        metric: 'All-to-All Dispatch',
        what: '根据 Gating 的 expert 选择，把 token 从原始 rank 布局交换到拥有目标专家的 EP rank。',
        evidence: ['这是 EP 路由的核心通信算子', '没有 Dispatch，远端 expert 拿不到被路由到自己的 token'],
        relatedNodeIds: ['router_gate', 'routed_expert'],
      },
      a2a_combine: {
        dimension: '通信算子',
        metric: 'All-to-All Combine',
        what: '把各 expert rank 上计算完成的 token 输出交换回原 token owner，再与 shared expert 输出合并。',
        relatedNodeIds: ['routed_expert', 'moe_branch_add'],
      },
      moe_reduce_scatter: {
        dimension: '通信算子',
        metric: 'Reduce-Scatter',
        what: 'MoE 双分支输出合并后规约并切回 TP 分片，送入块级残差 Add。',
        relatedNodeIds: ['moe_branch_add', 'block_residual_add'],
      },
      attention_gqa: {
        dimension: '算子',
        metric: 'GQA',
        what: 'Grouped Query Attention 主计算。展开父节点后，MulAttention Children 展示其 tile 内循环和外循环的计算结构。',
        relatedNodeIds: ['mul_core', 'mul_inner_loop', 'mul_outer_loop_l', 'mul_outer_loop_r'],
      },
      routed_expert: {
        dimension: '算子',
        metric: 'SwiftGMM',
        what: 'Routed Expert FFN 使用 grouped matmul，把不同 expert 的 token group 与对应 expert 权重批量计算。',
        relatedNodeIds: ['swift_grouped_matmul', 'swift_weight_n', 'swift_weight_next'],
      },
      swift_grouped_matmul: {
        dimension: '算子细节',
        metric: 'Grouped MatMul',
        what: 'SwiftGMM 把按 expert 分组后的 token 输入与 Expert N、Expert N+1 等不同权重块配对执行矩阵乘。',
        evidence: ['AGMM 可理解为 AllGather + MatMul', 'GMMRS 可理解为 GroupedMatMul + ReduceScatter'],
      },
    },
  };
})();
