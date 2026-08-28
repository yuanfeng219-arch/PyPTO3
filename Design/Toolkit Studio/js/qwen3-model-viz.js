(function registerQwen3ModelViz() {
  'use strict';

  const MODEL_ID = 'qwen3';
  window.PtoModelArchitectureState = window.PtoModelArchitectureState || { active: MODEL_ID };

  const baseGraph = {
    width: 1180,
    height: 2050,
    clusters: [
      { id: 'decoder-stack', label: '_decode_layer × 40', x: 210, y: 300, width: 830, height: 1550, repeat: 40 },
      { id: 'scope-1-cluster', label: 'Scope 1 · Input RMS + QKV', x: 245, y: 440, width: 690, height: 330, parent: 'decoder-stack' },
      { id: 'scope-2-cluster', label: 'Scope 2 · Paged Flash Attention', x: 245, y: 790, width: 690, height: 430, parent: 'decoder-stack' },
      { id: 'scope-3-cluster', label: 'Scope 3 · Output + MLP', x: 245, y: 1240, width: 690, height: 440, parent: 'decoder-stack' },
    ],
    nodes: [
      { id: 'hidden-input', label: 'hidden_states', typeLabel: 'Input · [16,5120] · BF16', kind: 'tensor', x: 590, y: 38, width: 270, height: 50, colorKey: 'io:input', phase: 'boundary-in' },
      { id: 'copy-hidden', label: 'copy_hidden', typeLabel: 'CORE_GROUP · BF16 → FP32', kind: 'op', x: 590, y: 112, width: 280, height: 56, colorKey: 'sem:linear', phase: 'boundary-in' },
      { id: 'fp32-carry-in', label: 'cur', typeLabel: 'Inter-layer carry · [16,5120] · FP32', kind: 'tensor', x: 590, y: 190, width: 318, height: 52, colorKey: 'io:activation', phase: 'boundary-in' },
      { id: 'x-gamma0', label: 'x_gamma0', typeLabel: 'Layer 0 only · FP32 × γ → BF16', kind: 'op', x: 590, y: 264, width: 298, height: 56, colorKey: 'sem:norm', phase: 'boundary-in' },

      { id: 'layer-input', label: 'cur + normed_in', typeLabel: 'FP32 residual / BF16 x×γ', kind: 'tensor', x: 590, y: 354, width: 318, height: 52, parent: 'decoder-stack', phase: 'scope-1' },
      { id: 'rms-recip', label: 'rms_recip', typeLabel: 'Input RMS reciprocal · FP32', kind: 'op', x: 410, y: 500, width: 248, height: 56, colorKey: 'sem:norm', parent: 'scope-1-cluster', phase: 'scope-1' },
      { id: 'qkv-weight', label: 'Stacked Q/K/V Weights', typeLabel: 'Parameter · per layer slice', kind: 'state', state_type: 'parameter', x: 110, y: 580, width: 190, height: 48, colorKey: 'io:parameter', phase: 'scope-1' },
      { id: 'qkv-proj', label: 'q_proj · k_proj · v_proj', typeLabel: 'Split-K + atomic add · FP32', kind: 'op', x: 680, y: 580, width: 326, height: 58, colorKey: 'sem:linear', parent: 'scope-1-cluster', phase: 'scope-1' },
      { id: 'qk-norm', label: 'qk_norm', typeLabel: '8 tasks · gamma + reciprocal', kind: 'op', x: 590, y: 680, width: 270, height: 56, colorKey: 'sem:qknorm', parent: 'scope-1-cluster', phase: 'scope-1' },

      { id: 'fa-work-build', label: 'fa_work_build', typeLabel: 'Dense real-block work table · AIV', kind: 'op', x: 405, y: 850, width: 286, height: 56, colorKey: 'sem:comm', parent: 'scope-2-cluster', phase: 'scope-2' },
      { id: 'rope-qkv', label: 'rope_qkv', typeLabel: 'RoPE + Q pad + paged K/V write', kind: 'op', x: 730, y: 850, width: 310, height: 56, colorKey: 'sem:rope', parent: 'scope-2-cluster', phase: 'scope-2' },
      { id: 'paged-kv-cache', label: 'k_cache · v_cache', typeLabel: 'Paged state · layer_cache_base', kind: 'state', x: 1070, y: 970, width: 196, height: 52, colorKey: 'io:state', phase: 'scope-2' },
      { id: 'fa-fused', label: 'fa_fused', typeLabel: 'QK → local softmax → SV · SPMD', kind: 'op', x: 590, y: 980, width: 316, height: 60, colorKey: 'sem:attention', parent: 'scope-2-cluster', phase: 'scope-2' },
      { id: 'online-softmax', label: 'online_softmax', typeLabel: 'Cross-block reduction · attn_out BF16', kind: 'op', x: 590, y: 1110, width: 316, height: 58, colorKey: 'sem:attention', parent: 'scope-2-cluster', phase: 'scope-2' },

      { id: 'output-weight', label: 'Wo / Gate / Up / Down', typeLabel: 'Stacked parameters · per layer', kind: 'state', state_type: 'parameter', x: 110, y: 1320, width: 190, height: 48, colorKey: 'io:parameter', phase: 'scope-3' },
      { id: 'out-proj', label: 'out_proj', typeLabel: '10 × 5 split-N/K · FP32 atomic', kind: 'op', x: 590, y: 1300, width: 298, height: 58, colorKey: 'sem:linear', parent: 'scope-3-cluster', phase: 'scope-3' },
      { id: 'residual-cast', label: 'residual_rms_cast', typeLabel: 'Residual FP32 + post γ → BF16', kind: 'op', x: 430, y: 1390, width: 294, height: 58, colorKey: 'sem:norm', parent: 'scope-3-cluster', phase: 'scope-3' },
      { id: 'post-rms-reduce', label: 'post_rms_reduce', typeLabel: 'Deferred post-RMS reciprocal', kind: 'op', x: 750, y: 1390, width: 278, height: 58, colorKey: 'sem:norm', parent: 'scope-3-cluster', phase: 'scope-3' },
      { id: 'gate-up-proj', label: 'gate_proj · up_proj', typeLabel: 'Split-K interleaved · FP32', kind: 'op', x: 590, y: 1480, width: 292, height: 58, colorKey: 'sem:gate', parent: 'scope-3-cluster', phase: 'scope-3' },
      { id: 'silu', label: 'silu', typeLabel: 'Deferred RMS scale + SwiGLU · BF16', kind: 'op', x: 590, y: 1565, width: 284, height: 58, colorKey: 'sem:mlp', parent: 'scope-3-cluster', phase: 'scope-3' },
      { id: 'down-proj', label: 'down_proj', typeLabel: '17 × 5 split-K/N · FP32 atomic', kind: 'op', x: 590, y: 1645, width: 302, height: 58, colorKey: 'sem:linear', parent: 'scope-3-cluster', phase: 'scope-3' },
      { id: 'dcr-xgamma', label: 'dcr_xgamma', typeLabel: '5-way SPMD · outside manual_scope', kind: 'op', x: 590, y: 1722, width: 314, height: 60, colorKey: 'sem:comm', phase: 'scope-3' },
      { id: 'fp32-carry-out', label: 'out / next_hidden', typeLabel: 'FP32 · next layer carry', kind: 'tensor', x: 405, y: 1810, width: 258, height: 52, colorKey: 'io:activation', phase: 'scope-3' },
      { id: 'next-normed', label: 'normed_out', typeLabel: 'BF16 · next layer x×γ', kind: 'tensor', x: 765, y: 1810, width: 242, height: 52, colorKey: 'io:activation', phase: 'scope-3' },

      { id: 'cast-lmhead', label: 'cast_lmhead_in', typeLabel: 'Final FP32 → BF16 · once', kind: 'op', x: 590, y: 1885, width: 282, height: 56, colorKey: 'sem:linear', phase: 'boundary-out' },
      { id: 'lm-weight', label: 'Final Norm + LM Head', typeLabel: 'Parameter', kind: 'state', state_type: 'parameter', x: 105, y: 1960, width: 220, height: 48, colorKey: 'io:parameter', phase: 'boundary-out' },
      { id: 'rms-lm-head', label: 'rms_lm_head', typeLabel: 'Final RMSNorm + vocabulary projection', kind: 'op', x: 590, y: 1960, width: 320, height: 58, colorKey: 'sem:head', phase: 'boundary-out' },
      { id: 'logits', label: 'out · logits', typeLabel: 'Output · [16,VOCAB]', kind: 'tensor', x: 1010, y: 1960, width: 222, height: 52, colorKey: 'io:output', phase: 'boundary-out' },
    ],
    edges: [
      { source: 'hidden-input', target: 'copy-hidden', tag: 'BF16' }, { source: 'copy-hidden', target: 'fp32-carry-in', tag: 'FP32 once' }, { source: 'fp32-carry-in', target: 'x-gamma0' }, { source: 'x-gamma0', target: 'layer-input', tag: 'layer 0 seed' },
      { source: 'layer-input', target: 'rms-recip', tag: 'cur FP32' }, { source: 'layer-input', target: 'qkv-proj', tag: 'normed_in BF16' }, { source: 'qkv-weight', target: 'qkv-proj', dashed: true, tag: 'Wq/Wk/Wv' }, { source: 'qkv-proj', target: 'qk-norm' }, { source: 'rms-recip', target: 'qk-norm', dashed: true, tag: 'inv_rms' },
      { source: 'qk-norm', target: 'rope-qkv' }, { source: 'layer-input', target: 'fa-work-build', dashed: true, waypoints: [{ x: 270, y: 354 }, { x: 270, y: 850 }], tag: 'seq_lens' }, { source: 'fa-work-build', target: 'fa-fused', tag: 'dense blocks' }, { source: 'rope-qkv', target: 'fa-fused', tag: 'Q padded' }, { source: 'rope-qkv', target: 'paged-kv-cache', dashed: true, tag: 'paged write' }, { source: 'paged-kv-cache', target: 'fa-fused', dashed: true, tag: 'paged read' }, { source: 'fa-fused', target: 'online-softmax', tag: 'block partials' },
      { source: 'online-softmax', target: 'out-proj', tag: 'attn_out BF16' }, { source: 'output-weight', target: 'out-proj', dashed: true, tag: 'Wo' }, { source: 'out-proj', target: 'residual-cast' }, { source: 'out-proj', target: 'post-rms-reduce' }, { source: 'layer-input', target: 'residual-cast', dashed: true, waypoints: [{ x: 960, y: 354 }, { x: 960, y: 1390 }], tag: 'residual FP32' }, { source: 'layer-input', target: 'post-rms-reduce', dashed: true, waypoints: [{ x: 930, y: 354 }, { x: 930, y: 1390 }] },
      { source: 'residual-cast', target: 'gate-up-proj', tag: 'mlp_norm_in BF16' }, { source: 'post-rms-reduce', target: 'silu', dashed: true, tag: 'post inv_rms' }, { source: 'output-weight', target: 'gate-up-proj', dashed: true, tag: 'Wgate/Wup' }, { source: 'gate-up-proj', target: 'silu' }, { source: 'silu', target: 'down-proj' }, { source: 'output-weight', target: 'down-proj', dashed: true, tag: 'Wdown' }, { source: 'down-proj', target: 'dcr-xgamma' }, { source: 'residual-cast', target: 'dcr-xgamma', dashed: true, waypoints: [{ x: 875, y: 1390 }, { x: 875, y: 1722 }], tag: 'post_norm_partial' },
      { source: 'dcr-xgamma', target: 'fp32-carry-out', tag: 'out FP32' }, { source: 'dcr-xgamma', target: 'next-normed', tag: 'x×γ BF16' }, { source: 'fp32-carry-out', target: 'layer-input', dashed: true, waypoints: [{ x: 205, y: 1810 }, { x: 205, y: 354 }], tag: 'next layer ×40' }, { source: 'next-normed', target: 'layer-input', dashed: true, waypoints: [{ x: 970, y: 1810 }, { x: 970, y: 354 }] },
      { source: 'fp32-carry-out', target: 'cast-lmhead', tag: 'after layer 39' }, { source: 'cast-lmhead', target: 'rms-lm-head', tag: 'BF16 once' }, { source: 'lm-weight', target: 'rms-lm-head', dashed: true }, { source: 'rms-lm-head', target: 'logits', tag: 'logits' },
    ],
  };

  const drillSpecs = {
    'copy-hidden': { description: '入口 BF16→FP32 的分块复制。', steps: [['slice hidden_states', 'Tensor slice · BF16', 'io:input'], ['cast FP32', 'Precision boundary', 'sem:linear'], ['assemble cur', 'Tensor write · FP32', 'io:activation']] },
    'x-gamma0': { description: '仅为 layer 0 生成首份预缩放输入。', steps: [['slice cur + gamma₀', 'FP32 tiles', 'io:input'], ['col_expand_mul', 'x × gamma', 'sem:norm'], ['cast BF16', 'Precision boundary', 'sem:linear'], ['assemble normed', 'Layer 0 seed', 'io:activation']] },
    'rms-recip': { description: '输入 RMS reciprocal 的 FP32 归约链。', steps: [['load FP32 chunks', 'pipeline stage=4', 'io:input'], ['mul x × x', 'Elementwise', 'sem:act'], ['row_sum', 'Reduction', 'sem:norm'], ['mean + ε', 'FP32 scalar', 'sem:norm'], ['sqrt → recip', 'inv_rms_states', 'sem:norm']] },
    'qkv-proj': { description: 'Q/K/V 三路 Split-K 投影的代表性内部算子链。', steps: [['seed zero', 'Q/K/V accumulators', 'io:constant'], ['slice normed + weight', 'BF16 tiles', 'io:input'], ['matmul', 'First K tile · FP32 acc', 'sem:linear'], ['pipeline matmul_acc', 'Remaining K tiles', 'sem:linear'], ['assemble atomic add', 'q/k/v_proj · FP32', 'io:activation']] },
    'qk-norm': { description: '每个 KV Head 合并执行 gamma 与 reciprocal。', steps: [['slice q_proj / k_proj', 'Per-head FP32', 'io:input'], ['row_expand_mul', 'Apply inv_rms', 'sem:norm'], ['col_expand_mul gamma', 'Q/K gamma', 'sem:norm'], ['square → row_sum', 'Head RMS reduction', 'sem:norm'], ['sqrt → recip', 'Deferred qk_inv', 'sem:norm']] },
    'fa-work-build': { description: '把 ragged 请求压紧成无空洞的真实块工作表。', steps: [['read seq_lens', 'Per request', 'io:input'], ['ceil_div SEQ_TILE', 'Real block count', 'sem:linear'], ['encode b × MCB + p', 'INT32 work item', 'sem:comm'], ['tensor.write table', 'fa_work_table', 'io:activation'], ['write fa_total', 'Device block count', 'io:output']] },
    'rope-qkv': { description: '完成 Q/K RoPE、Q padding 以及当前 Token 的 Paged K/V 写入。', steps: [['read slot + cos/sin', 'Position metadata', 'io:input'], ['apply deferred qk_inv', 'Q/K head scale', 'sem:norm'], ['RoPE lo / hi', 'Rotate Q and K', 'sem:rope'], ['assemble K/V cache', 'Paged write · BF16', 'io:state'], ['pad Q heads', '5 real → 16 tile rows', 'io:activation']] },
    'fa-fused': { description: '单个真实 KV Block 内的 QK、局部 Softmax 与 SV 融合链。', steps: [['decode work item', 'b / block / physical page', 'sem:comm'], ['load paged K + Q', 'BF16 tiles', 'io:input'], ['QK matmul + scale', 'Cube · FP32 scores', 'sem:attention'], ['max → exp → row_sum', 'Local softmax · AIV', 'sem:norm'], ['load V → SV matmul', 'Cube · FP32 partial', 'sem:attention'], ['write oi / mi / li', 'Block partials', 'io:output']] },
    'online-softmax': { description: '跨 KV Block 合并局部 Softmax 的稳定递推。', steps: [['load oi / mi / li', 'Block partials', 'io:input'], ['maximum mi', 'Stable merge pivot', 'sem:norm'], ['exp alpha / beta', 'Rescale factors', 'sem:act'], ['update li + oi', 'Online recurrence', 'sem:norm'], ['oi ÷ li', 'Normalized context', 'sem:attention'], ['cast + assemble', 'attn_out · BF16', 'io:output']] },
    'out-proj': { description: 'Attention Output 的 Split-N/Split-K 投影。', steps: [['out_seed', 'Zero FP32 accumulator', 'io:constant'], ['slice attn_out + Wo', 'BF16 tiles', 'io:input'], ['matmul', 'First K tile', 'sem:linear'], ['pipeline matmul_acc', 'Remaining K tiles', 'sem:linear'], ['atomic assemble', 'attn_proj_fp32', 'io:output']] },
    'residual-cast': { description: '生成 Attention 残差及 MLP 的 gamma-scaled BF16 输入。', steps: [['slice attn_proj + hidden', 'FP32 chunks', 'io:input'], ['add residual', 'h1 · FP32', 'sem:act'], ['assemble post_norm_partial', 'Raw residual', 'io:activation'], ['multiply post_gamma', 'h1 × gamma', 'sem:norm'], ['cast + assemble BF16', 'mlp_norm_in', 'io:output']] },
    'post-rms-reduce': { description: '与 residual cast 并行计算 Post RMS reciprocal。', steps: [['add residual chunks', 'FP32 h1', 'sem:act'], ['square', 'h1 × h1', 'sem:act'], ['row_sum pipeline', 'FP32 reduction', 'sem:norm'], ['mean + ε', 'Variance', 'sem:norm'], ['sqrt → recip', 'post_inv_rms', 'sem:norm']] },
    'gate-up-proj': { description: 'Gate 与 Up 投影交错调度，共用 MLP 输入。', steps: [['seed gate / up', 'Zero accumulators', 'io:constant'], ['slice mlp_norm_in', 'BF16 K slice', 'io:input'], ['gate matmul_acc', 'FP32 atomic', 'sem:gate'], ['up matmul_acc', 'FP32 atomic', 'sem:linear'], ['assemble accumulators', 'gate_acc / up_acc', 'io:output']] },
    silu: { description: '延迟应用 Post RMS reciprocal 后执行 SwiGLU。', steps: [['load gate + up', 'FP32 chunks', 'io:input'], ['apply post_inv_rms', 'Row scale', 'sem:norm'], ['neg → exp → recip', 'Sigmoid', 'sem:act'], ['gate × sigmoid × up', 'SwiGLU', 'sem:mlp'], ['cast + assemble BF16', 'mlp_tile', 'io:output']] },
    'down-proj': { description: '把 SwiGLU 中间维投影回 Hidden，并累加为 FP32。', steps: [['down_seed', 'Zero FP32 accumulator', 'io:constant'], ['slice mlp_tile + Wdown', 'BF16 tiles', 'io:input'], ['matmul', 'First K tile', 'sem:linear'], ['pipeline matmul_acc', 'Remaining K tiles', 'sem:linear'], ['atomic assemble', 'down_acc_all · FP32', 'io:output']] },
    'dcr-xgamma': { description: '层尾单次读取同时生成两份跨层输出。', steps: [['get block + slice', '5-way disjoint slabs', 'sem:comm'], ['add down + residual', 'out_chunk · FP32', 'sem:act'], ['assemble out', 'Next FP32 carry', 'io:activation'], ['multiply next_gamma', 'out × gamma_next', 'sem:norm'], ['cast + assemble normed', 'Next BF16 QKV input', 'io:output']] },
    'cast-lmhead': { description: '40 层循环后的唯一 FP32→BF16 转换。', steps: [['slice final cur', 'FP32 chunks', 'io:input'], ['cast BF16', 'Precision boundary', 'sem:linear'], ['assemble cur_bf16', 'LM Head input', 'io:output']] },
    'rms-lm-head': { description: '最终归一化和词表投影。', steps: [['final RMS reduction', 'FP32 reciprocal', 'sem:norm'], ['multiply final gamma', 'Normalized hidden', 'sem:norm'], ['LM Head matmul', 'Hidden → Vocabulary', 'sem:head'], ['write logits', '[16,VOCAB]', 'io:output']] },
  };

  baseGraph.nodes.forEach((node) => {
    if (drillSpecs[node.id]) node.collapsed = true;
  });

  function buildDrillDefinition(nodeId) {
    const spec = drillSpecs[nodeId];
    const target = baseGraph.nodes.find((node) => node.id === nodeId);
    if (!spec || !target) return null;
    const stepGap = 70;
    const height = 74 + spec.steps.length * stepGap;
    const top = target.y - target.height / 2;
    const nodes = spec.steps.map((step, index) => ({
      id: `${nodeId}__${index}`,
      label: step[0],
      typeLabel: step[1],
      kind: step[2].startsWith('io:') ? 'tensor' : 'op',
      x: target.x,
      y: top + 58 + index * stepGap,
      width: index === 0 || index === spec.steps.length - 1 ? 258 : 286,
      height: 52,
      colorKey: step[2],
      parent: `drill-${nodeId}`,
      phase: target.phase,
      drillOwner: nodeId,
    }));
    return {
      target,
      spec,
      width: 570,
      height,
      delta: height - target.height + 34,
      cluster: { id: `drill-${nodeId}`, label: `${target.label} · L3 operators`, x: target.x - 285, y: top, width: 570, height, parent: target.parent, drillOwner: nodeId },
      nodes,
      entry: nodes[0].id,
      exit: nodes[nodes.length - 1].id,
      edges: nodes.slice(0, -1).map((node, index) => ({ source: node.id, target: nodes[index + 1].id })),
    };
  }

  const routeHints = {
    'layer-input->fa-work-build': { side: 'left', lane: 222 },
    'layer-input->residual-cast': { side: 'right', lane: 960 },
    'layer-input->post-rms-reduce': { side: 'right', lane: 925 },
    'residual-cast->dcr-xgamma': { side: 'right', lane: 885 },
    'fp32-carry-out->layer-input': { side: 'left', lane: 232 },
    'next-normed->layer-input': { side: 'right', lane: 1015 },
  };

  function routeGraph(graph) {
    const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
    const outgoing = new Map();
    const incoming = new Map();
    graph.edges.forEach((edge) => {
      if (!outgoing.has(edge.source)) outgoing.set(edge.source, []);
      if (!incoming.has(edge.target)) incoming.set(edge.target, []);
      outgoing.get(edge.source).push(edge);
      incoming.get(edge.target).push(edge);
    });
    outgoing.forEach((edges) => edges.sort((a, b) => (nodeById.get(a.target)?.x || 0) - (nodeById.get(b.target)?.x || 0)));
    incoming.forEach((edges) => edges.sort((a, b) => (nodeById.get(a.source)?.x || 0) - (nodeById.get(b.source)?.x || 0)));

    function portOffset(edge, collection, node, axis) {
      const list = collection.get(axis === 'source' ? edge.source : edge.target) || [];
      if (list.length < 2) return 0;
      const index = list.indexOf(edge);
      const raw = (index - (list.length - 1) / 2) * 24;
      return Math.max(-(node.width / 2 - 24), Math.min(node.width / 2 - 24, raw));
    }

    const edges = graph.edges.map((edge) => {
      const source = nodeById.get(edge.source);
      const target = nodeById.get(edge.target);
      if (!source || !target) return { ...edge };
      const key = edge.routeKey || `${edge.source}->${edge.target}`;
      const hint = routeHints[key];
      const routed = { ...edge, waypoints: undefined, curve: undefined, route: 'rounded', cornerRadius: 14 };
      const dx = target.x - source.x;
      const dy = target.y - source.y;

      if (hint) {
        routed.sourceAnchor = { side: hint.side, dy: 0 };
        routed.targetAnchor = { side: hint.side, dy: 0 };
        routed.waypoints = [{ x: hint.lane, y: source.y }, { x: hint.lane, y: target.y }];
        routed.routeClass = 'side-lane';
        return routed;
      }

      if (dy < -40 || Math.abs(dy) > 360) {
        const useLeft = (source.x + target.x) / 2 < graph.width / 2;
        const lane = useLeft ? 170 : graph.width - 145;
        const side = useLeft ? 'left' : 'right';
        routed.sourceAnchor = side;
        routed.targetAnchor = side;
        routed.waypoints = [{ x: lane, y: source.y }, { x: lane, y: target.y }];
        routed.routeClass = 'side-lane';
        return routed;
      }

      if (Math.abs(dy) <= 54 && Math.abs(dx) > 40) {
        const sourceSide = dx > 0 ? 'right' : 'left';
        const targetSide = dx > 0 ? 'left' : 'right';
        routed.sourceAnchor = sourceSide;
        routed.targetAnchor = targetSide;
        if (Math.abs(dy) > 4) {
          const midX = (source.x + target.x) / 2;
          routed.waypoints = [{ x: midX, y: source.y }, { x: midX, y: target.y }];
        } else {
          routed.curve = 'straight';
          routed.route = undefined;
        }
        routed.routeClass = 'horizontal-lane';
        return routed;
      }

      const sourceDx = portOffset(edge, outgoing, source, 'source');
      const targetDx = portOffset(edge, incoming, target, 'target');
      routed.sourceAnchor = { side: dy >= 0 ? 'bottom' : 'top', dx: sourceDx };
      routed.targetAnchor = { side: dy >= 0 ? 'top' : 'bottom', dx: targetDx };
      const startY = source.y + (dy >= 0 ? source.height / 2 : -source.height / 2);
      const endY = target.y + (dy >= 0 ? -target.height / 2 : target.height / 2);
      const midY = (startY + endY) / 2;
      routed.waypoints = [
        { x: source.x + sourceDx, y: midY },
        { x: target.x + targetDx, y: midY },
      ];
      routed.routeClass = Math.abs(dx) < 32 ? 'spine-lane' : 'branch-lane';
      return routed;
    });
    return { ...graph, edges };
  }

  function buildExpandedGraph(nodeId) {
    const drill = buildDrillDefinition(nodeId);
    if (!drill) return routeGraph({ ...baseGraph, nodes: baseGraph.nodes.map((node) => ({ ...node })), edges: baseGraph.edges.map((edge) => ({ ...edge })), clusters: baseGraph.clusters.map((cluster) => ({ ...cluster })) });
    const threshold = drill.target.y;
    const shouldShiftNode = (node) => node.y > threshold || (
      node.y === threshold
      && Math.abs(node.x - drill.target.x) < (drill.width + node.width) / 2
    );
    const nodes = baseGraph.nodes
      .filter((node) => node.id !== nodeId)
      .map((node) => ({ ...node, y: shouldShiftNode(node) ? node.y + drill.delta : node.y }));
    const shiftedDrillNodes = drill.nodes.map((node) => ({ ...node }));
    const clusters = baseGraph.clusters.map((cluster) => {
      const containsTarget = cluster.y <= threshold && cluster.y + cluster.height >= threshold;
      return {
        ...cluster,
        y: cluster.y > threshold ? cluster.y + drill.delta : cluster.y,
        height: containsTarget ? cluster.height + drill.delta : cluster.height,
      };
    });
    clusters.push(drill.cluster);
    const edges = baseGraph.edges.map((edge) => ({
      ...edge,
      routeKey: `${edge.source}->${edge.target}`,
      source: edge.source === nodeId ? drill.exit : edge.source,
      target: edge.target === nodeId ? drill.entry : edge.target,
    }));
    return routeGraph({
      ...baseGraph,
      height: baseGraph.height + drill.delta,
      nodes: [...nodes, ...shiftedDrillNodes],
      clusters,
      edges: [...edges, ...drill.edges],
      activeDrill: nodeId,
    });
  }

  const phaseCopy = {
    all: ['Qwen3 14B · Fused Decode', 'decode_fwd 全链路 · 40 × Decoder Layer · FP32 跨层 carry', '完整链路', '按 decode_layer.py 的实际任务依赖展示输入边界、Scope 1/2/3 与输出边界。'],
    'boundary-in': ['输入边界', 'copy_hidden · x_gamma0 · 仅在 40 层循环前执行', '输入边界', '外部 BF16 hidden_states 只在入口转换为 FP32；layer 0 单独生成首份 BF16 x×γ。'],
    'scope-1': ['Scope 1 · RMS + QKV', 'rms_recip 与 Split-K Q/K/V 投影并行，随后执行 fused qk_norm', 'SCOPE 1', '输入 RMS reciprocal 延后应用，使归约可与 QKV 投影重叠；Q/K Norm 合并 gamma 与 reciprocal。'],
    'scope-2': ['Scope 2 · Paged Flash Attention', 'fa_work_build → rope_qkv → fa_fused → online_softmax', 'SCOPE 2', '仅真实 KV block 进入稠密工作表；fa_fused 逐块计算 QK/Softmax/SV，再跨块归并。'],
    'scope-3': ['Scope 3 · Output + MLP', 'out_proj → residual/RMS → gate/up → SiLU → down → dcr_xgamma', 'SCOPE 3', '层尾 5-way SPMD 同时写出 FP32 residual carry 与下一层 BF16 x×γ，避免层间往返转换。'],
    'boundary-out': ['输出边界', 'cast_lmhead_in → rms_lm_head → logits', '输出边界', '第 40 层 FP32 hidden 只在进入既有 BF16 RMS LM Head 前转换一次。'],
  };

  const phaseBounds = {
    'boundary-in': { x: 250, y: 0, width: 680, height: 320 },
    'scope-1': { x: 0, y: 320, width: 1040, height: 470 },
    'scope-2': { x: 190, y: 760, width: 970, height: 490 },
    'scope-3': { x: 0, y: 1200, width: 1040, height: 650 },
    'boundary-out': { x: 0, y: 1800, width: 1160, height: 250 },
  };

  const details = {
    'copy-hidden': ['copy_hidden', 'Fused decode 的首次精度边界：把外部 BF16 hidden_states 一次性转换为 FP32 cur。', ['阶段', 'decode_fwd pre-loop'], ['代码', 'decode_layer.py:1155'], ['输出', 'cur · [16,5120] · FP32']],
    'qkv-proj': ['q_proj · k_proj · v_proj', '三路投影使用 Split-K、内部 N/K tiling 与 FP32 atomic accumulation；读取上一层预先生成的 BF16 normed_in。', ['阶段', 'Scope 1'], ['并行', 'Q: 10×5 · K/V: 2×5'], ['代码', 'decode_layer.py:429–505']],
    'fa-fused': ['fa_fused', '基于稠密真实块工作表的 block-level SPMD：每个工作项完成 QK、局部 softmax 与 SV。', ['阶段', 'Scope 2'], ['调度', 'NUM_CORES grid-stride'], ['缓存', 'Paged K/V · block_table']],
    'dcr-xgamma': ['dcr_xgamma', '在 manual_scope 外以 5-way SPMD 同时产出层输出和下一层 x×gamma，恢复跨层自动依赖。', ['阶段', 'Scope 3 tail'], ['输出 1', 'out · FP32 carry'], ['输出 2', 'normed_out · BF16']],
    'rms-lm-head': ['rms_lm_head', '消费最后一次 BF16 转换后的 hidden，执行最终 RMSNorm 与词表投影。', ['阶段', 'decode_fwd tail'], ['输入', 'cur_bf16 · [16,5120]'], ['输出', 'logits · [16,VOCAB]']],
    'rms-recip': ['rms_recip', '只计算输入 RMS 的倒数标量；x×gamma 已由上一层 dcr_xgamma 提前生成，因此可与 QKV 投影重叠。', ['阶段', 'Scope 1'], ['精度', 'FP32 reduction'], ['策略', 'deferred scaling']],
    'fa-work-build': ['fa_work_build', '根据 ragged seq_lens 仅压紧真实序列块，构建无空洞工作表。', ['阶段', 'Scope 2 prep'], ['设备', 'AIV task'], ['输出', 'fa_work_table + fa_total']],
    'online-softmax': ['online_softmax', '合并每个 KV block 的局部 m/l/o 中间量，直接写出 BF16 attn_out。', ['阶段', 'Scope 2'], ['工作项', 'BATCH × NUM_KV_HEADS = 128'], ['输出', 'attn_out · BF16']],
  };

  let controller = null;
  let initialized = false;
  let activePhase = 'all';
  let activeDrill = null;
  let currentGraph = buildExpandedGraph(null);

  function setChrome() {
    const activeItem = document.querySelector('[data-model-id="' + MODEL_ID + '"]');
    document.querySelectorAll('[data-model-id]').forEach((item) => {
      const active = item.dataset.modelId === MODEL_ID;
      item.classList.toggle('is-active', active);
      item.setAttribute('aria-current', String(active));
      item.setAttribute('aria-selected', String(active));
      const status = item.querySelector('em');
      if (status) status.textContent = active ? '已加载' : '可视化';
    });
    const selectorTitle = document.querySelector('[data-model-selector-title]');
    const selectorSubtitle = document.querySelector('[data-model-selector-subtitle]');
    if (selectorTitle) selectorTitle.textContent = activeItem?.querySelector('b')?.textContent || 'Qwen3 14B';
    if (selectorSubtitle) selectorSubtitle.textContent = activeItem?.querySelector('small')?.textContent || 'Fused Decode · 源码校准';
    document.querySelectorAll('[data-model-selector-icon]').forEach((icon) => { icon.hidden = icon.dataset.modelSelectorIcon !== MODEL_ID; });
    const factsBody = document.getElementById('modelFactsBody');
    if (factsBody) factsBody.innerHTML = '<div><dt>Decoder</dt><dd>40 层</dd></div><div><dt>Decode batch</dt><dd>16</dd></div><div><dt>Hidden</dt><dd>5,120</dd></div><div><dt>Attention</dt><dd>40 Q / 8 KV</dd></div><div><dt>Head dim</dt><dd>128</dd></div><div><dt>FFN</dt><dd>17,408</dd></div>';
    const status = document.getElementById('modelCanvasStatus');
    if (status) status.textContent = 'Qwen3 14B 架构已预加载';
    const command = document.querySelector('.kf-command');
    if (command) command.textContent = 'MODEL · Qwen3 14B 架构可视化';
  }

  function renderInspector(title, badge, description, rows) {
    document.getElementById('modelInspectorTitle').textContent = title;
    document.getElementById('modelInspectorBody').innerHTML = `<div class="kf-model-inspector__hero"><span>${badge}</span><b>${title}</b><p>${description}</p></div>${rows?.length ? `<dl class="kf-model-node-detail">${rows.map((row) => `<div><dt>${row[0]}</dt><dd>${row[1]}</dd></div>`).join('')}</dl>` : ''}`;
  }

  function nodeDetail(nodeId) {
    const node = currentGraph.nodes.find((item) => item.id === nodeId);
    if (node?.drillOwner) {
      const owner = baseGraph.nodes.find((item) => item.id === node.drillOwner);
      renderInspector(node.label, 'L3 OPERATOR', node.typeLabel, [['父任务', owner?.label || node.drillOwner], ['阶段', owner?.phase || 'shared']]);
      return;
    }
    const data = details[nodeId] || [node?.label || nodeId, node?.typeLabel || 'PyPTO execution task', ['阶段', node?.phase || 'shared']];
    renderInspector(data[0], node?.phase?.toUpperCase() || 'CODE NODE', data[1], data.slice(2));
  }

  function applyNodeFocus(nodeId) {
    const stage = document.getElementById('qwen3ModelGraph');
    if (!stage || !nodeId) return;
    stage.classList.add('has-node-focus');
    stage.dataset.focusedNode = nodeId;
    nodeDetail(nodeId);
  }

  function clearNodeFocus() {
    const stage = document.getElementById('qwen3ModelGraph');
    stage?.classList.remove('has-node-focus');
    if (stage) delete stage.dataset.focusedNode;
    controller?.clearSelection();
  }

  function focusPhaseViewport(phase) {
    if (!controller) return;
    if (phase === 'all' || !phaseBounds[phase]) {
      controller.fit();
      document.getElementById('modelZoomReadout').textContent = '适应';
      return;
    }
    const bounds = phaseBounds[phase];
    const rect = document.getElementById('qwen3ModelGraph').getBoundingClientRect();
    const padding = 34;
    const zoom = Math.max(.18, Math.min(1.05, (rect.width - padding * 2) / bounds.width, (rect.height - padding * 2) / bounds.height));
    controller.setTransform({
      zoom,
      tx: (rect.width - bounds.width * zoom) / 2 - bounds.x * zoom,
      ty: (rect.height - bounds.height * zoom) / 2 - bounds.y * zoom,
    });
    document.getElementById('modelZoomReadout').textContent = `${Math.round(zoom * 100)}%`;
  }

  function focusDrillViewport(nodeId) {
    const cluster = currentGraph.clusters.find((item) => item.id === `drill-${nodeId}`);
    if (!controller || !cluster) return;
    const rect = document.getElementById('qwen3ModelGraph').getBoundingClientRect();
    const padding = 38;
    const zoom = Math.max(.18, Math.min(1.05, (rect.width - padding * 2) / cluster.width, (rect.height - padding * 2) / cluster.height));
    controller.setTransform({
      zoom,
      tx: (rect.width - cluster.width * zoom) / 2 - cluster.x * zoom,
      ty: (rect.height - cluster.height * zoom) / 2 - cluster.y * zoom,
    });
    document.getElementById('modelZoomReadout').textContent = `${Math.round(zoom * 100)}%`;
  }

  function applyPhase(phase) {
    activePhase = phase;
    const copy = phaseCopy[phase] || phaseCopy.all;
    document.querySelector('.kf-model-toolbar h1').textContent = copy[0];
    document.getElementById('modelPhaseSummary').textContent = copy[1];
    document.querySelectorAll('[data-model-phase]').forEach((button) => button.classList.toggle('is-active', button.dataset.modelPhase === phase));
    document.querySelectorAll('#qwen3ModelGraph .pto-model-graphviz-node').forEach((element) => {
      const node = currentGraph.nodes.find((item) => item.id === element.dataset.nodeId);
      element.classList.toggle('is-phase-muted', phase !== 'all' && node?.phase !== phase);
      element.classList.toggle('is-phase-active', phase !== 'all' && node?.phase === phase);
    });
    document.querySelectorAll('#qwen3ModelGraph .pto-model-graphviz-edge, #qwen3ModelGraph .pto-model-graphviz-edge-tag').forEach((element) => {
      const source = currentGraph.nodes.find((node) => node.id === element.dataset.source);
      const target = currentGraph.nodes.find((node) => node.id === element.dataset.target);
      element.classList.toggle('is-phase-muted', phase !== 'all' && source?.phase !== phase && target?.phase !== phase);
      element.classList.toggle('is-phase-active', phase !== 'all' && (source?.phase === phase || target?.phase === phase));
    });
    clearNodeFocus();
    if (activeDrill) focusDrillViewport(activeDrill);
    else focusPhaseViewport(phase);
    renderInspector(copy[0], copy[2], copy[3], [['源码', 'decode_layer.py'], ['模型', 'Qwen3-14B · 40 layers']]);
  }

  function renderGraph() {
    const stage = document.getElementById('qwen3ModelGraph');
    if (!stage || !window.PtoModelGraphvizPattern) return;
    const savedTransform = controller?.getTransform?.();
    stage.classList.remove('has-node-focus');
    delete stage.dataset.focusedNode;
    controller?.destroy();
    currentGraph = buildExpandedGraph(activeDrill);
    controller = window.PtoModelGraphvizPattern.renderController(stage, currentGraph, {
      ariaLabel: 'Qwen3 14B fused decode execution graph',
      colormap: window.PtoModelGraphvizPattern.modelArchitectureColormap(currentGraph),
      fitMode: 'full', viewportPadding: 36, autoFit: false,
      interaction: { panZoom: true, selectableClusters: false }, overlays: { edgeTags: true },
      onSelect: ({ nodeId }) => applyNodeFocus(nodeId),
    });
    requestAnimationFrame(() => {
      applyPhase(activePhase);
      if (savedTransform) {
        controller.setTransform({ ...savedTransform });
        document.getElementById('modelZoomReadout').textContent = `${Math.round(savedTransform.zoom * 100)}%`;
        return;
      }
      if (!activeDrill) return;
      const spec = drillSpecs[activeDrill];
      const owner = baseGraph.nodes.find((node) => node.id === activeDrill);
      renderInspector(owner?.label || activeDrill, 'L3 EXPANDED', spec?.description || '算子级下钻', [['层级', 'Task → PyPTO operators'], ['操作', '点击容器右上角 − 收起']]);
      focusDrillViewport(activeDrill);
    });
  }

  function handleDrillToggle(event) {
    if (window.PtoModelArchitectureState?.active !== MODEL_ID) return;
    const toggle = event.target.closest('.pto-model-graphviz-toggle, .pto-model-graphviz-toggle-icon');
    if (!toggle) return;
    const nodeGroup = toggle.closest('.pto-model-graphviz-node');
    const clusterGroup = toggle.closest('.pto-model-graphviz-cluster');
    const nodeId = nodeGroup?.dataset.nodeId;
    const clusterId = clusterGroup?.dataset.clusterId;
    if (nodeId && drillSpecs[nodeId]) {
      event.preventDefault();
      event.stopImmediatePropagation();
      activeDrill = nodeId;
      activePhase = baseGraph.nodes.find((node) => node.id === nodeId)?.phase || activePhase;
      renderGraph();
      return;
    }
    if (clusterId?.startsWith('drill-')) {
      event.preventDefault();
      event.stopImmediatePropagation();
      activeDrill = null;
      renderGraph();
    }
  }

  function handleCanvasSelectionClear(event) {
    if (window.PtoModelArchitectureState?.active !== MODEL_ID) return;
    if (event.target.closest('.pto-model-graphviz-node, .pto-model-graphviz-edge, .pto-model-graphviz-edge-tag')) return;
    clearNodeFocus();
  }

  function selectPhase(phase) {
    if (window.PtoModelArchitectureState?.active !== MODEL_ID) return;
    activePhase = phase;
    if (activeDrill) {
      activeDrill = null;
      renderGraph();
      return;
    }
    applyPhase(phase);
  }

  function updateZoom(delta) {
    if (!controller) return;
    const current = controller.getTransform();
    const next = Math.max(.18, Math.min(2.6, current.zoom * delta));
    controller.setTransform({ zoom: next });
    document.getElementById('modelZoomReadout').textContent = `${Math.round(next * 100)}%`;
  }

  function init() {
    if (initialized) return;
    const stage = document.getElementById('qwen3ModelGraph');
    if (!stage || !window.PtoModelGraphvizPattern) return;
    stage.addEventListener('click', handleDrillToggle, true);
    stage.addEventListener('pointerdown', handleCanvasSelectionClear, true);
    document.querySelectorAll('[data-model-phase]').forEach((button) => button.addEventListener('click', () => selectPhase(button.dataset.modelPhase)));
    document.querySelector('[data-model-fit]')?.addEventListener('click', () => {
      if (window.PtoModelArchitectureState?.active !== MODEL_ID) return;
      controller?.fit();
      document.getElementById('modelZoomReadout').textContent = '适应';
    });
    document.querySelector('[data-model-zoom="in"]')?.addEventListener('click', () => {
      if (window.PtoModelArchitectureState?.active !== MODEL_ID) return;
      updateZoom(1.18);
    });
    document.querySelector('[data-model-zoom="out"]')?.addEventListener('click', () => {
      if (window.PtoModelArchitectureState?.active !== MODEL_ID) return;
      updateZoom(1 / 1.18);
    });
    initialized = true;
    renderGraph();
  }

  function show() {
    window.PtoModelArchitectureState = { active: MODEL_ID };
    init();
    setChrome();
    renderGraph();
    requestAnimationFrame(() => {
      applyPhase(activePhase);
      if (activeDrill) focusDrillViewport(activeDrill);
    });
  }

  /**
   * 供推理性能分析抽屉回跳：切到该节点所属阶段（视口随之平移）再选中它。
   * applyPhase 内部会 clearNodeFocus，所以选中必须排在它之后。
   */
  function focusNode(nodeId) {
    const target = baseGraph.nodes.find((node) => node.id === nodeId);
    if (!target) return false;
    init();
    const apply = () => {
      if (target.phase && target.phase !== activePhase) applyPhase(target.phase);
      controller?.selectNode(nodeId, { source: 'profiler' });
    };
    if (activeDrill) {
      // 有下钻展开时先收起，renderGraph 会重建 controller 并在 rAF 里跑 applyPhase
      activeDrill = null;
      renderGraph();
      requestAnimationFrame(() => requestAnimationFrame(apply));
    } else {
      apply();
    }
    return true;
  }

  init();
  window.PtoQwen3ModelViz = { show, fit: () => controller?.fit(), setPhase: selectPhase, focusNode, graph: baseGraph };
})();
