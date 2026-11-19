(function attachPtoDv4ArchitecturePattern(global) {
  'use strict';

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const SCENE_WIDTH = 1840;
  const SCENE_HEIGHT = 3040;
  const STREAM_X = 920;
  const FRONT_MAIN_NODE_WIDTH = 390;
  const BUNDLE_OFFSETS = [-7.5, -2.5, 2.5, 7.5];
  const SIDE_LAYER_TOP = 130;
  const SIDE_LAYER_HEIGHT = 820;
  const SIDE_LAYER_BOTTOM = SIDE_LAYER_TOP + SIDE_LAYER_HEIGHT;

  function queryRoot(input) {
    return typeof input === 'string' ? document.querySelector(input) : input;
  }

  function escapeHtml(value) {
    return String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function graphNode(id, label, x, y, options = {}) {
    const kind = options.kind || 'op';
    return {
      id,
      label,
      kind,
      typeLabel: options.typeLabel ?? (
        kind === 'module' ? 'Module' : kind === 'state' ? 'State' : 'Op'
      ),
      colorKey: options.colorKey || 'sem:linear',
      x,
      y,
      width: options.width || 260,
      height: options.height || 56,
      hideTypeLabel: options.hideTypeLabel === true,
      parent: options.parent || '',
      selectable: false,
      collapsed: options.collapsed === true,
      origin: 'source',
      dataState: 'source_only',
      role: options.role || '',
      variant: options.variant || '',
      attrs: options.attrs || {}
    };
  }

  function graphCluster(id, label, x, y, width, height, options = {}) {
    return {
      id,
      label,
      kind: 'module',
      typeLabel: 'Module',
      colorKey: options.colorKey || 'module:decoder',
      x,
      y,
      width,
      height,
      parent: options.parent || '',
      nodes: [],
      children: [],
      structuralRoot: false,
      variant: options.variant || '',
      selectable: false,
      collapsible: options.collapsible
    };
  }

  function graphEdge(id, source, target, tensorName, options = {}) {
    return {
      id,
      source,
      target,
      semanticEdgeType: options.semanticEdgeType || 'activation',
      sourceAnchor: options.sourceAnchor || 'bottom',
      targetAnchor: options.targetAnchor || 'top',
      curve: options.curve || 'vertical',
      waypoints: options.waypoints,
      cornerRadius: options.cornerRadius || 18,
      color: options.color,
      route: options.route,
      tension: options.tension,
      fanCurveY: options.fanCurveY,
      tensor: {
        name: tensorName || 'hidden_states',
        shape: options.shape || '[B,T,d]',
        dtype: options.dtype || 'bf16'
      },
      provenance: [{ source: 'DeepSeek V4 source/config architecture' }],
      variant: options.variant || '',
      bundlePath: options.bundlePath || null,
      bundleDiagonal: options.bundleDiagonal === true
    };
  }

  function buildFrontGraph(selectedLayer, collapsedIds = new Set()) {
    const data = global.DeepSeekV4ArchitectureData;
    const config = data.config;
    const descriptor = data.describeLayer(selectedLayer);
    const layer = descriptor.layer;
    const rootId = `dv4/layer/${layer}`;
    const attnId = `${rootId}/mhc_attn`;
    const hybridId = `${attnId}/hybrid_attention`;
    const csaId = `${hybridId}/csa`;
    const hcaId = `${hybridId}/hca`;
    const ffnId = `${rootId}/mhc_ffn`;
    const moeId = `${ffnId}/deepseek_moe`;
    const routedId = `${moeId}/routed_experts`;

    const ids = {
      input: `${rootId}/input_streams`,
      attnMix: `${attnId}/residual_mix_b`,
      attnPre: `${attnId}/hc_pre`,
      norm1: `${attnId}/rmsnorm`,
      attentionType: `${hybridId}/attention_type`,
      attnPost: `${attnId}/hc_post`,
      attnMerge: `${attnId}/merge`,
      middle: `${rootId}/attention_output_streams`,
      ffnMix: `${ffnId}/residual_mix_b`,
      ffnPre: `${ffnId}/hc_pre`,
      norm2: `${ffnId}/rmsnorm`,
      tokenId: `${moeId}/token_id`,
      router: `${moeId}/router`,
      selection: `${routedId}/selection`,
      combine: `${moeId}/combine`,
      ffnPost: `${ffnId}/hc_post`,
      ffnMerge: `${ffnId}/merge`,
      output: `${rootId}/output_streams`
    };
    const canonicalCsa = global.PtoCsaMappingRegistry?.canonicalGraph || null;
    const csaProjection = canonicalCsa?.l1Projection || null;
    const csaOrigin = { x: 315, y: 570 };
    const csaNodeId = canonicalId => canonicalId;
    const csaSemanticNodes = (canonicalCsa?.compoundIds || []).map(canonicalId => {
      const presentation = csaProjection.nodes[canonicalId];
      const semanticNode = global.PtoCsaMappingRegistry?.getNode(canonicalId);
      const projected = graphNode(
        csaNodeId(canonicalId),
        semanticNode.label,
        csaOrigin.x + presentation.x,
        csaOrigin.y + presentation.y,
        {
          parent: csaId,
          kind: 'module',
          hideTypeLabel: true,
          colorKey: presentation.colorKey,
          width: presentation.width,
          height: presentation.height,
          variant: 'csa'
        }
      );
      projected.canonicalId = canonicalId;
      projected.sourceRefs = semanticNode?.sourceRefs || [];
      return projected;
    });
    const csaUnavailableId = `${csaId}/registry_unavailable`;
    if (!canonicalCsa) {
      csaSemanticNodes.push(graphNode(csaUnavailableId, 'CSA canonical registry unavailable', 697, 835, {
        parent: csaId, kind: 'module', typeLabel: 'Unavailable', colorKey: 'module:mhc', width: 340, variant: 'csa'
      }));
    }
    const csaSummaryEdges = canonicalCsa?.summaryEdges || [];
    const resolveCsaEndpoint = id => {
      if (id === 'csa_scope_input') return ids.attentionType;
      if (id === 'csa_scope_output') return ids.attnPost;
      return csaNodeId(id);
    };
    const csaProjectedEdges = csaSummaryEdges.map((edge, index) => graphEdge(
      `edge/csa-canonical-${index}`,
      resolveCsaEndpoint(edge.source),
      resolveCsaEndpoint(edge.target),
      edge.label,
      { variant: 'csa' }
    ));
    if (!canonicalCsa) {
      csaProjectedEdges.push(
        graphEdge('edge/csa-registry-input', ids.attentionType, csaUnavailableId, 'CSA input', { variant: 'csa' }),
        graphEdge('edge/csa-registry-output', csaUnavailableId, ids.attnPost, 'CSA output', { variant: 'csa' })
      );
    }

    const clusters = [
      graphCluster(rootId, `DeepSeek V4 Decoder Layer · L${layer}`, 60, 70, 1720, 2310, {
        colorKey: 'module:decoder'
      }),
      graphCluster(attnId, 'mHC-Attention · 4 × d → d → 4 × d', 105, 240, 1630, 1090, {
        parent: rootId,
        colorKey: 'module:mhc'
      }),
      graphCluster(hybridId, 'Hybrid Attention · CSA / HCA', 285, 535, 1440, 625, {
        parent: attnId,
        colorKey: 'module:mhc'
      }),
      graphCluster(
        csaId,
        csaProjection?.label || 'CSA · Compressed Sparse Attention',
        csaOrigin.x,
        csaOrigin.y,
        csaProjection?.width || 585,
        csaProjection?.height || 455,
        {
          parent: hybridId,
          colorKey: 'opv:attention',
          variant: 'csa'
        }
      ),
      graphCluster(hcaId, 'HCA · Heavily Compressed Attention', 1085, 570, 620, 550, {
        parent: hybridId,
        colorKey: 'opv:rope',
        variant: 'hca'
      }),
      graphCluster(ffnId, 'mHC-FFN · 4 × d → d → 4 × d', 105, 1435, 1630, 835, {
        parent: rootId,
        colorKey: 'module:ffn'
      }),
      graphCluster(moeId, 'DeepSeekMoE · all decoder layers', 375, 1640, 1280, 455, {
        parent: ffnId,
        colorKey: 'opv:moe'
      }),
      graphCluster(routedId, `Routed Experts · ${config.routedExperts} total`, 500, 1765, 790, 240, {
        parent: moeId,
        colorKey: 'opv:moe'
      })
    ];

    const nodes = [
      graphNode(ids.input, `X${layer} · 4 × d residual streams`, STREAM_X, 160, {
        kind: 'state', typeLabel: 'mHC State', colorKey: 'io:state', width: 390, height: 64, role: 'stream-state'
      }),
      graphNode(ids.attnMix, 'Mix B · 4→4', 230, 710, {
        typeLabel: 'mHC Mapping', colorKey: 'sem:linear', width: 170, height: 42, role: 'residual-mix'
      }),
      graphNode(ids.attnPre, 'HC Pre · 4→1', STREAM_X, 325, {
        typeLabel: 'mHC Projection', colorKey: 'sem:attention', width: FRONT_MAIN_NODE_WIDTH, height: 64, role: 'mhc-projection'
      }),
      graphNode(ids.norm1, 'RMSNorm 1', STREAM_X, 420, {
        colorKey: 'sem:norm', width: FRONT_MAIN_NODE_WIDTH
      }),
      graphNode(ids.attentionType, descriptor.attentionLabel, STREAM_X, 510, {
        typeLabel: `L${layer} architecture schedule`, colorKey: descriptor.attentionType === 'csa' ? 'sem:attention' : 'sem:rope', width: FRONT_MAIN_NODE_WIDTH, height: 64
      }),

      ...csaSemanticNodes,

      graphNode(`${hcaId}/query_projection`, 'Query Projection', 1395, 655, {
        parent: hcaId, colorKey: 'sem:linear', width: 270, variant: 'hca'
      }),
      graphNode(`${hcaId}/kv_compressor`, 'KV Compressor ×128', 1395, 735, {
        parent: hcaId, colorKey: 'sem:rope', width: 270, variant: 'hca'
      }),
      graphNode(`${hcaId}/compressed_kv`, 'Compressed KV', 1265, 835, {
        parent: hcaId, kind: 'state', typeLabel: 'KV State', colorKey: 'io:state', width: 190, variant: 'hca'
      }),
      graphNode(`${hcaId}/swa_kv`, `SWA KV · ${config.slidingWindow}`, 1525, 835, {
        parent: hcaId, kind: 'state', typeLabel: 'Local KV State', colorKey: 'io:state', width: 190, variant: 'hca'
      }),
      graphNode(`${hcaId}/dense_mqa`, 'Dense Shared-KV MQA', 1395, 930, {
        parent: hcaId, colorKey: 'sem:attention', width: 290, variant: 'hca'
      }),
      graphNode(`${hcaId}/grouped_o_projection`, 'Grouped Output Projection', 1395, 1015, {
        parent: hcaId, colorKey: 'sem:linear', width: 300, variant: 'hca'
      }),

      graphNode(ids.attnPost, 'HC Post · 1→4', STREAM_X, 1160, {
        typeLabel: 'mHC Projection', colorKey: 'sem:attention', width: FRONT_MAIN_NODE_WIDTH, height: 64, role: 'mhc-projection'
      }),
      graphNode(ids.attnMerge, '', STREAM_X, 1260, {
        typeLabel: '', colorKey: 'sem:linear', width: 82, height: 72, role: 'mhc-merge'
      }),
      graphNode(ids.middle, `X${layer}′ · 4 × d residual streams`, STREAM_X, 1370, {
        kind: 'state', typeLabel: 'mHC State', colorKey: 'io:state', width: 390, height: 64, role: 'stream-state'
      }),

      graphNode(ids.ffnMix, 'Mix B · 4→4', 230, 1810, {
        typeLabel: 'mHC Mapping', colorKey: 'sem:linear', width: 170, height: 42, role: 'residual-mix'
      }),
      graphNode(ids.ffnPre, 'HC Pre · 4→1', STREAM_X, 1510, {
        typeLabel: 'mHC Projection', colorKey: 'sem:mlp', width: FRONT_MAIN_NODE_WIDTH, height: 64, role: 'mhc-projection'
      }),
      graphNode(ids.norm2, 'RMSNorm 2', STREAM_X, 1590, {
        colorKey: 'sem:norm', width: FRONT_MAIN_NODE_WIDTH
      }),
      graphNode(ids.tokenId, 'Token ID', 500, 1710, {
        kind: 'state', typeLabel: 'Hash Input', colorKey: 'io:input', width: 180, role: 'token-id-input'
      }),
      graphNode(ids.router, descriptor.routerLabel, STREAM_X, 1710, {
        typeLabel: descriptor.routerType === 'hash_moe' ? 'Hash-MoE' : 'MoE Router',
        colorKey: 'sem:gate', width: FRONT_MAIN_NODE_WIDTH, height: 64
      }),
      graphNode(ids.selection, descriptor.routeSelectionLabel, 895, 1845, {
        parent: routedId, typeLabel: 'Routing Selection', colorKey: 'sem:gate', width: 270
      }),
      graphNode(`${routedId}/expert_001`, 'Expert 001', 600, 1935, {
        parent: routedId, colorKey: 'sem:moe', width: 180
      }),
      graphNode(`${routedId}/expert_002`, 'Expert 002', 800, 1935, {
        parent: routedId, colorKey: 'sem:moe', width: 180
      }),
      graphNode(`${routedId}/expert_ellipsis`, '…', 995, 1935, {
        parent: routedId, colorKey: 'sem:moe', width: 90
      }),
      graphNode(`${routedId}/expert_384`, 'Expert 384', 1160, 1935, {
        parent: routedId, colorKey: 'sem:moe', width: 190
      }),
      graphNode(`${moeId}/shared_expert`, 'Shared Expert', 1430, 1935, {
        parent: moeId, typeLabel: 'Always active', colorKey: 'sem:gate', width: 220
      }),
      graphNode(ids.combine, 'Combine', STREAM_X, 2040, {
        parent: moeId, colorKey: 'sem:gate', width: FRONT_MAIN_NODE_WIDTH
      }),
      graphNode(ids.ffnPost, 'HC Post · 1→4', STREAM_X, 2120, {
        typeLabel: 'mHC Projection', colorKey: 'sem:mlp', width: FRONT_MAIN_NODE_WIDTH, height: 64, role: 'mhc-projection'
      }),
      graphNode(ids.ffnMerge, '', STREAM_X, 2210, {
        typeLabel: '', colorKey: 'sem:linear', width: 82, height: 72, role: 'mhc-merge'
      }),
      graphNode(ids.output, `X${layer + 1} · 4 × d residual streams`, STREAM_X, 2320, {
        kind: 'state', typeLabel: 'mHC State', colorKey: 'io:state', width: 390, height: 64, role: 'stream-state'
      })
    ];

    const nodeParents = new Map([
      [ids.input, rootId],
      [ids.attnMix, attnId],
      [ids.attnPre, attnId],
      [ids.norm1, attnId],
      [ids.attentionType, attnId],
      [ids.attnPost, attnId],
      [ids.attnMerge, attnId],
      [ids.middle, rootId],
      [ids.ffnMix, ffnId],
      [ids.ffnPre, ffnId],
      [ids.norm2, ffnId],
      [ids.tokenId, moeId],
      [ids.router, moeId],
      [ids.combine, moeId],
      [ids.ffnPost, ffnId],
      [ids.ffnMerge, ffnId],
      [ids.output, rootId]
    ]);
    nodes.forEach(node => {
      if (nodeParents.has(node.id)) node.parent = nodeParents.get(node.id);
    });

    const edges = [
      graphEdge('edge/input-attn-pre', ids.input, ids.attnPre, 'hc_streams', {
        shape: '[B,T,4,d]',
        bundlePath: [{ x: STREAM_X, y: 192 }, { x: STREAM_X, y: 293 }]
      }),
      graphEdge('edge/input-attn-mix', ids.input, ids.attnMix, 'hc_streams', {
        semanticEdgeType: 'residual', sourceAnchor: 'left', targetAnchor: 'right', shape: '[B,T,4,d]',
        bundleDiagonal: true,
        bundlePath: [
          { x: 725, y: 160 }, { x: 300, y: 160 },
          { x: 300, y: 710 }, { x: 315, y: 710 }
        ]
      }),
      graphEdge('edge/attn-pre-norm', ids.attnPre, ids.norm1, 'collapsed_hidden', { shape: '[B,T,d]' }),
      graphEdge('edge/norm-attention-type', ids.norm1, ids.attentionType, 'normalized_hidden', { shape: '[B,T,d]' }),

      ...csaProjectedEdges,

      graphEdge('edge/type-hca-query', ids.attentionType, `${hcaId}/query_projection`, 'attention_input', { variant: 'hca' }),
      graphEdge('edge/hca-query-compressor', `${hcaId}/query_projection`, `${hcaId}/kv_compressor`, 'query_and_kv_input', { variant: 'hca' }),
      graphEdge('edge/hca-compressor-kv', `${hcaId}/kv_compressor`, `${hcaId}/compressed_kv`, 'compressed_kv', { variant: 'hca' }),
      graphEdge('edge/hca-compressor-swa', `${hcaId}/kv_compressor`, `${hcaId}/swa_kv`, 'local_kv', { variant: 'hca' }),
      graphEdge('edge/hca-kv-mqa', `${hcaId}/compressed_kv`, `${hcaId}/dense_mqa`, 'dense_compressed_kv', { variant: 'hca' }),
      graphEdge('edge/hca-swa-mqa', `${hcaId}/swa_kv`, `${hcaId}/dense_mqa`, 'swa_kv', { variant: 'hca' }),
      graphEdge('edge/hca-mqa-output', `${hcaId}/dense_mqa`, `${hcaId}/grouped_o_projection`, 'mqa_output', { variant: 'hca' }),

      graphEdge('edge/hca-output-post', `${hcaId}/grouped_o_projection`, ids.attnPost, 'attention_output', { variant: 'hca' }),
      graphEdge('edge/attn-post-merge', ids.attnPost, ids.attnMerge, 'hc_contribution', {
        shape: '[B,T,4,d]', bundlePath: [{ x: STREAM_X, y: 1192 }, { x: STREAM_X, y: 1224 }]
      }),
      graphEdge('edge/attn-mix-merge', ids.attnMix, ids.attnMerge, 'mixed_residual_streams', {
        semanticEdgeType: 'residual', sourceAnchor: 'right', targetAnchor: 'left', shape: '[B,T,4,d]',
        bundleDiagonal: true,
        bundlePath: [
          { x: 315, y: 710 }, { x: 300, y: 710 },
          { x: 300, y: 1260 }, { x: 884, y: 1260 }
        ]
      }),
      graphEdge('edge/attn-merge-middle', ids.attnMerge, ids.middle, 'attention_mhc_output', {
        shape: '[B,T,4,d]', bundlePath: [{ x: STREAM_X, y: 1296 }, { x: STREAM_X, y: 1338 }]
      }),

      graphEdge('edge/middle-ffn-pre', ids.middle, ids.ffnPre, 'hc_streams', {
        shape: '[B,T,4,d]', bundlePath: [{ x: STREAM_X, y: 1402 }, { x: STREAM_X, y: 1478 }]
      }),
      graphEdge('edge/middle-ffn-mix', ids.middle, ids.ffnMix, 'hc_streams', {
        semanticEdgeType: 'residual', sourceAnchor: 'left', targetAnchor: 'right', shape: '[B,T,4,d]',
        bundleDiagonal: true,
        bundlePath: [
          { x: 725, y: 1370 }, { x: 300, y: 1370 },
          { x: 300, y: 1810 }, { x: 315, y: 1810 }
        ]
      }),
      graphEdge('edge/ffn-pre-norm', ids.ffnPre, ids.norm2, 'collapsed_hidden', { shape: '[B,T,d]' }),
      graphEdge('edge/norm-router', ids.norm2, ids.router, 'normalized_hidden', { shape: '[B,T,d]' }),
      graphEdge('edge/token-router', ids.tokenId, ids.router, 'input_ids', {
        sourceAnchor: 'right', targetAnchor: 'left', curve: 'horizontal', shape: '[B,T]'
      }),
      graphEdge('edge/router-selection', ids.router, ids.selection, 'expert_indices'),
      graphEdge('edge/selection-expert-1', ids.selection, `${routedId}/expert_001`, 'routed_tokens'),
      graphEdge('edge/selection-expert-2', ids.selection, `${routedId}/expert_002`, 'routed_tokens'),
      graphEdge('edge/selection-expert-ellipsis', ids.selection, `${routedId}/expert_ellipsis`, 'routed_tokens'),
      graphEdge('edge/selection-expert-384', ids.selection, `${routedId}/expert_384`, 'routed_tokens'),
      graphEdge('edge/norm-shared', ids.norm2, `${moeId}/shared_expert`, 'all_tokens', {
        sourceAnchor: 'bottom', targetAnchor: 'top', curve: 'vertical', fanCurveY: 1652
      }),
      graphEdge('edge/expert-1-combine', `${routedId}/expert_001`, ids.combine, 'expert_output'),
      graphEdge('edge/expert-2-combine', `${routedId}/expert_002`, ids.combine, 'expert_output'),
      graphEdge('edge/expert-ellipsis-combine', `${routedId}/expert_ellipsis`, ids.combine, 'expert_output'),
      graphEdge('edge/expert-384-combine', `${routedId}/expert_384`, ids.combine, 'expert_output'),
      graphEdge('edge/shared-combine', `${moeId}/shared_expert`, ids.combine, 'shared_expert_output', {
        sourceAnchor: 'bottom', targetAnchor: 'top', curve: 'vertical'
      }),
      graphEdge('edge/combine-post', ids.combine, ids.ffnPost, 'moe_output'),
      graphEdge('edge/ffn-post-merge', ids.ffnPost, ids.ffnMerge, 'hc_contribution', {
        shape: '[B,T,4,d]', bundlePath: [{ x: STREAM_X, y: 2152 }, { x: STREAM_X, y: 2174 }]
      }),
      graphEdge('edge/ffn-mix-merge', ids.ffnMix, ids.ffnMerge, 'mixed_residual_streams', {
        semanticEdgeType: 'residual', sourceAnchor: 'right', targetAnchor: 'left', shape: '[B,T,4,d]',
        bundleDiagonal: true,
        bundlePath: [
          { x: 315, y: 1810 }, { x: 300, y: 1810 },
          { x: 300, y: 2210 }, { x: 884, y: 2210 }
        ]
      }),
      graphEdge('edge/ffn-merge-output', ids.ffnMerge, ids.output, 'layer_output_streams', {
        shape: '[B,T,4,d]', bundlePath: [{ x: STREAM_X, y: 2246 }, { x: STREAM_X, y: 2288 }]
      })
    ];

    const decoderOffsetY = 250;
    clusters.forEach(cluster => {
      cluster.y += decoderOffsetY;
    });
    nodes.forEach(node => {
      node.y += decoderOffsetY;
    });
    edges.forEach(edge => {
      if (Array.isArray(edge.bundlePath)) {
        edge.bundlePath = edge.bundlePath.map(point => ({ ...point, y: point.y + decoderOffsetY }));
      }
      if (Number.isFinite(edge.fanCurveY)) edge.fanCurveY += decoderOffsetY;
    });

    const modelIds = {
      inputTokens: 'dv4/model/input_tokens',
      embedding: 'dv4/model/token_embedding',
      previousLayers: 'dv4/model/previous_decoder_layers',
      remainingLayers: 'dv4/model/remaining_decoder_layers',
      finalNorm: 'dv4/model/final_rmsnorm',
      lmHead: 'dv4/model/lm_head',
      outputTokens: 'dv4/model/output_tokens'
    };
    nodes.push(
      graphNode(modelIds.inputTokens, 'Input Tokens', STREAM_X, 55, {
        kind: 'state', typeLabel: 'Token IDs', colorKey: 'io:input', width: FRONT_MAIN_NODE_WIDTH, role: 'model-io'
      }),
      graphNode(modelIds.embedding, 'Token Embedding', STREAM_X, 140, {
        colorKey: 'sem:embedding', width: FRONT_MAIN_NODE_WIDTH, role: 'model-io'
      })
    );
    if (layer > 0) {
      nodes.push(graphNode(modelIds.previousLayers, 'Previous Decoder Layers', STREAM_X, 225, {
        kind: 'module', typeLabel: `Layers 0–${layer - 1}`, colorKey: 'module:decoder', width: FRONT_MAIN_NODE_WIDTH,
        role: 'layer-context'
      }));
    }
    const rootBottom = clusters.find(cluster => cluster.id === rootId).y
      + clusters.find(cluster => cluster.id === rootId).height;
    const remainingLayersY = rootBottom + 90;
    if (layer < config.numHiddenLayers - 1) {
      nodes.push(graphNode(modelIds.remainingLayers, 'Remaining Decoder Layers', STREAM_X, remainingLayersY, {
        kind: 'module', typeLabel: `Layers ${layer + 1}–${config.numHiddenLayers - 1}`,
        colorKey: 'module:decoder', width: FRONT_MAIN_NODE_WIDTH, role: 'layer-context'
      }));
    }
    const finalNormY = remainingLayersY + (layer < config.numHiddenLayers - 1 ? 90 : 0);
    nodes.push(
      graphNode(modelIds.finalNorm, 'Final RMSNorm', STREAM_X, finalNormY, {
        colorKey: 'sem:norm', width: FRONT_MAIN_NODE_WIDTH, role: 'model-io'
      }),
      graphNode(modelIds.lmHead, 'LM Head', STREAM_X, finalNormY + 85, {
        colorKey: 'sem:linear', width: FRONT_MAIN_NODE_WIDTH, role: 'model-io'
      }),
      graphNode(modelIds.outputTokens, 'Output Tokens', STREAM_X, finalNormY + 170, {
        kind: 'state', typeLabel: 'Token logits / IDs', colorKey: 'io:output', width: FRONT_MAIN_NODE_WIDTH, role: 'model-io'
      })
    );
    edges.push(graphEdge('edge/model-input-embedding', modelIds.inputTokens, modelIds.embedding, 'token_ids', {
      shape: '[B,T]', dtype: 'int64'
    }));
    if (layer > 0) {
      edges.push(
        graphEdge('edge/model-input-layer-context', modelIds.embedding, modelIds.previousLayers, 'token_embeddings'),
        graphEdge('edge/model-context-layer-input', modelIds.previousLayers, ids.input, 'layer_input_streams', {
          shape: '[B,T,4,d]'
        })
      );
    } else {
      edges.push(graphEdge('edge/model-embedding-layer-input', modelIds.embedding, ids.input, 'layer_input_streams', {
        shape: '[B,T,4,d]'
      }));
    }
    if (layer < config.numHiddenLayers - 1) {
      edges.push(
        graphEdge('edge/model-layer-output-context', ids.output, modelIds.remainingLayers, 'layer_output_streams', {
          shape: '[B,T,4,d]'
        }),
        graphEdge('edge/model-context-final-norm', modelIds.remainingLayers, modelIds.finalNorm, 'decoder_output')
      );
    } else {
      edges.push(graphEdge('edge/model-layer-output-final-norm', ids.output, modelIds.finalNorm, 'decoder_output'));
    }
    edges.push(
      graphEdge('edge/model-final-norm-head', modelIds.finalNorm, modelIds.lmHead, 'normalized_hidden'),
      graphEdge('edge/model-head-output', modelIds.lmHead, modelIds.outputTokens, 'logits')
    );

    const graphHeight = finalNormY + 225;

    const graph = {
      schemaVersion: 'model_architecture_graph.v1',
      id: `deepseek-v4-pro-layer-${layer}`,
      title: `DeepSeek V4 Pro Decoder Layer ${layer}`,
      width: SCENE_WIDTH,
      height: graphHeight,
      nodes,
      edges,
      clusters,
      metadata: {
        modelId: config.modelId,
        selectedLayer: layer,
        attentionType: descriptor.attentionType,
        routerType: descriptor.routerType,
        hcMult: config.hcMult,
        mergeNodeIds: [ids.attnMerge, ids.ffnMerge]
      }
    };
    return projectCollapsedFrontGraph(graph, collapsedIds);
  }

  function projectCollapsedFrontGraph(graph, collapsedIds) {
    const requested = collapsedIds instanceof Set
      ? collapsedIds
      : new Set(collapsedIds || []);
    if (!requested.size) return graph;

    const clustersById = new Map((graph.clusters || []).map(cluster => [cluster.id, cluster]));
    const nodesById = new Map((graph.nodes || []).map(node => [node.id, node]));
    const collapsed = new Set([...requested].filter(id => clustersById.has(id)));
    if (!collapsed.size) return graph;

    function ancestorIds(clusterId) {
      const ancestors = [];
      let parentId = clustersById.get(clusterId)?.parent || '';
      while (parentId) {
        ancestors.push(parentId);
        parentId = clustersById.get(parentId)?.parent || '';
      }
      return ancestors;
    }

    const topLevelCollapsed = new Set([...collapsed].filter(id => (
      !ancestorIds(id).some(ancestorId => collapsed.has(ancestorId))
    )));

    function collapsedOwnerForParent(parentId) {
      let currentId = parentId || '';
      while (currentId) {
        if (topLevelCollapsed.has(currentId)) return currentId;
        currentId = clustersById.get(currentId)?.parent || '';
      }
      return '';
    }

    const hiddenOwnerById = new Map();
    (graph.clusters || []).forEach(cluster => {
      const owner = topLevelCollapsed.has(cluster.id)
        ? cluster.id
        : collapsedOwnerForParent(cluster.parent);
      if (owner) hiddenOwnerById.set(cluster.id, owner);
    });
    (graph.nodes || []).forEach(node => {
      const owner = collapsedOwnerForParent(node.parent);
      if (owner) hiddenOwnerById.set(node.id, owner);
    });

    const placeholderNodes = [...topLevelCollapsed].map(id => {
      const cluster = clustersById.get(id);
      const isMainStreamModule = id === graph.clusters?.[0]?.id
        || /\/(mhc_attn|hybrid_attention|mhc_ffn|deepseek_moe|routed_experts)$/.test(id);
      const width = isMainStreamModule
        ? FRONT_MAIN_NODE_WIDTH
        : Math.min(500, Math.max(300, cluster.width - 80));
      const x = isMainStreamModule ? STREAM_X : cluster.x + cluster.width / 2;
      return graphNode(id, cluster.label, x, cluster.y + cluster.height / 2, {
        kind: 'module',
        typeLabel: 'Module · folded',
        colorKey: cluster.colorKey,
        width,
        height: 64,
        parent: cluster.parent,
        role: 'folded-module',
        variant: cluster.variant,
        collapsed: true
      });
    });

    const nodes = (graph.nodes || []).filter(node => !hiddenOwnerById.has(node.id)).concat(placeholderNodes);
    const clusters = (graph.clusters || []).filter(cluster => !hiddenOwnerById.has(cluster.id));
    const edgeKeys = new Set();
    const edges = [];
    (graph.edges || []).forEach(edge => {
      const source = hiddenOwnerById.get(edge.source) || edge.source;
      const target = hiddenOwnerById.get(edge.target) || edge.target;
      if (source === target) return;
      if (!nodesById.has(edge.source) && !hiddenOwnerById.has(edge.source)) return;
      if (!nodesById.has(edge.target) && !hiddenOwnerById.has(edge.target)) return;
      const key = `${source}::${target}`;
      if (edgeKeys.has(key)) return;
      edgeKeys.add(key);
      const rewritten = source !== edge.source || target !== edge.target;
      edges.push(rewritten ? {
        ...edge,
        id: `${edge.id}::folded`,
        source,
        target,
        sourceAnchor: 'bottom',
        targetAnchor: 'top',
        curve: 'vertical',
        waypoints: undefined,
        route: undefined,
        fanCurveY: undefined,
        bundlePath: null,
        bundleDiagonal: false
      } : edge);
    });

    const projected = {
      ...graph,
      nodes,
      clusters,
      edges,
      metadata: {
        ...graph.metadata,
        collapsedClusterIds: [...topLevelCollapsed],
        mergeNodeIds: (graph.metadata?.mergeNodeIds || []).filter(id => nodes.some(node => node.id === id))
      }
    };
    return compactCollapsedFrontGraph(projected, graph, topLevelCollapsed);
  }

  function compactCollapsedFrontGraph(projected, expandedGraph, topLevelCollapsed) {
    if (!topLevelCollapsed?.size) return projected;
    const allClusters = expandedGraph.clusters || [];
    const clusterById = new Map(allClusters.map(cluster => [cluster.id, cluster]));
    const intervalKeys = new Set();
    const intervals = [];

    [...topLevelCollapsed].forEach(id => {
      const cluster = clusterById.get(id);
      if (!cluster) return;
      const overlappingSiblings = allClusters.filter(candidate => {
        if (candidate.parent !== cluster.parent) return false;
        const overlap = Math.min(cluster.y + cluster.height, candidate.y + candidate.height)
          - Math.max(cluster.y, candidate.y);
        return overlap > Math.min(cluster.height, candidate.height) * .5;
      });
      if (!overlappingSiblings.every(candidate => topLevelCollapsed.has(candidate.id))) return;
      const start = Math.min(...overlappingSiblings.map(candidate => candidate.y));
      const end = Math.max(...overlappingSiblings.map(candidate => candidate.y + candidate.height));
      const key = `${start}:${end}`;
      if (intervalKeys.has(key)) return;
      intervalKeys.add(key);
      intervals.push({ start, end, targetSpan: 128 });
    });
    intervals.sort((left, right) => left.start - right.start);
    if (!intervals.length) return projected;

    function mapY(y) {
      let removed = 0;
      for (const interval of intervals) {
        const span = interval.end - interval.start;
        if (y < interval.start) break;
        if (y <= interval.end) {
          return interval.start - removed
            + (y - interval.start) * (interval.targetSpan / span);
        }
        removed += span - interval.targetSpan;
      }
      return y - removed;
    }

    projected.nodes.forEach(node => {
      node.y = mapY(node.y);
    });
    projected.clusters.forEach(cluster => {
      const top = mapY(cluster.y);
      const bottom = mapY(cluster.y + cluster.height);
      cluster.y = top;
      cluster.height = Math.max(96, bottom - top);
    });
    projected.edges.forEach(edge => {
      if (Array.isArray(edge.bundlePath)) {
        edge.bundlePath = edge.bundlePath.map(point => ({ ...point, y: mapY(point.y) }));
      }
      if (Number.isFinite(edge.fanCurveY)) edge.fanCurveY = mapY(edge.fanCurveY);
    });
    projected.height = Math.max(420, mapY(expandedGraph.height));
    return projected;
  }

  function buildSideGraph(selectedLayer) {
    const data = global.DeepSeekV4ArchitectureData;
    const config = data.config;
    const selected = data.clampLayer(selectedLayer);
    const compactWidth = 42;
    const selectedWidth = 420;
    const layerGap = 12;
    const sidePadding = 100;
    const nodes = [];
    const edges = [];
    const clusters = [];
    const layerRecords = [];
    const mergeNodes = [];
    const residualBundlePaths = [];
    const streamArrowSegments = [];
    let cursorX = sidePadding;

    for (let layer = 0; layer < config.numHiddenLayers; layer += 1) {
      const descriptor = data.describeLayer(layer);
      const isSelected = layer === selected;
      const width = isSelected ? selectedWidth : compactWidth;
      const clusterId = `dv4/side/layer/${layer}`;
      const clusterX = cursorX;
      const centerX = clusterX + width / 2;
      const streamX = centerX;
      const cluster = graphCluster(
        clusterId,
        isSelected ? `Layer ${layer}` : `L${layer}`,
        clusterX,
        SIDE_LAYER_TOP,
        width,
        SIDE_LAYER_HEIGHT,
        { colorKey: 'module:decoder', collapsible: false }
      );
      cluster.layerIndex = layer;
      cluster.selected = isSelected;
      clusters.push(cluster);
      layerRecords.push({ layer, clusterId, streamX, clusterX, width, selected: isSelected });

      if (isSelected) {
        const travelsDown = layer % 2 === 0;
        const layerY = y => travelsDown ? y : SIDE_LAYER_TOP + SIDE_LAYER_BOTTOM - y;
        const mainNodeWidth = 280;
        const mixX = clusterX + 42;
        const attnMergeId = `${clusterId}/attn_merge`;
        const ffnMergeId = `${clusterId}/ffn_merge`;
        const attnPreId = `${clusterId}/attn_pre`;
        const norm1Id = `${clusterId}/rmsnorm_1`;
        const attentionTypeId = `${clusterId}/attention_type`;
        const attentionId = `${clusterId}/attention`;
        const attnPostId = `${clusterId}/attn_post`;
        const middleId = `${clusterId}/middle_streams`;
        const ffnPreId = `${clusterId}/ffn_pre`;
        const norm2Id = `${clusterId}/rmsnorm_2`;
        const moeId = `${clusterId}/moe`;
        const ffnPostId = `${clusterId}/ffn_post`;
        const attentionTypeLabel = descriptor.attentionType === 'csa'
          ? 'CSA · Compressed Sparse Attention'
          : 'HCA · Heavily Compressed Attention';
        const moeLabel = descriptor.routerType === 'hash_moe'
          ? 'DeepSeekMoE · Hash Routing'
          : 'DeepSeekMoE · Learned Routing';
        nodes.push(
          graphNode(attnPreId, 'HC Pre · 4→1', streamX, layerY(225), {
            parent: clusterId, colorKey: 'sem:linear', width: mainNodeWidth, height: 42, role: 'mhc-projection'
          }),
          graphNode(norm1Id, 'RMSNorm 1', streamX, layerY(275), {
            parent: clusterId, colorKey: 'sem:norm', width: mainNodeWidth, height: 40
          }),
          graphNode(attentionTypeId, attentionTypeLabel, streamX, layerY(330), {
            parent: clusterId,
            colorKey: descriptor.attentionType === 'csa' ? 'sem:attention' : 'sem:rope',
            width: mainNodeWidth,
            height: 44
          }),
          graphNode(attentionId, 'Hybrid Attention · CSA / HCA', streamX, layerY(390), {
            parent: clusterId, kind: 'module', typeLabel: 'Main module',
            colorKey: 'module:mhc', width: mainNodeWidth, height: 46, role: 'side-main-module'
          }),
          graphNode(attnPostId, 'HC Post · 1→4', streamX, layerY(450), {
            parent: clusterId, colorKey: 'sem:linear', width: mainNodeWidth, height: 42, role: 'mhc-projection'
          }),
          graphNode(attnMergeId, '', streamX, layerY(505), {
            parent: clusterId, colorKey: 'sem:linear', width: 38, height: 38, role: 'side-merge-anchor'
          }),
          graphNode(middleId, `X${layer}′ · 4 × d streams`, streamX, layerY(565), {
            parent: clusterId, kind: 'state', typeLabel: 'mHC State', colorKey: 'io:state',
            width: mainNodeWidth, height: 44, role: 'stream-state'
          }),
          graphNode(ffnPreId, 'HC Pre · 4→1', streamX, layerY(625), {
            parent: clusterId, colorKey: 'sem:linear', width: mainNodeWidth, height: 42, role: 'mhc-projection'
          }),
          graphNode(norm2Id, 'RMSNorm 2', streamX, layerY(680), {
            parent: clusterId, colorKey: 'sem:norm', width: mainNodeWidth, height: 40
          }),
          graphNode(moeId, moeLabel, streamX, layerY(740), {
            parent: clusterId, kind: 'module', typeLabel: 'Main module',
            colorKey: 'opv:moe', width: mainNodeWidth, height: 46, role: 'side-main-module'
          }),
          graphNode(ffnPostId, 'HC Post · 1→4', streamX, layerY(800), {
            parent: clusterId, colorKey: 'sem:linear', width: mainNodeWidth, height: 42, role: 'mhc-projection'
          }),
          graphNode(ffnMergeId, '', streamX, layerY(855), {
            parent: clusterId, colorKey: 'sem:linear', width: 38, height: 38, role: 'side-merge-anchor'
          })
        );
        const flowNodeIds = [
          attnPreId, norm1Id, attentionTypeId, attentionId, attnPostId, attnMergeId,
          middleId, ffnPreId, norm2Id, moeId, ffnPostId, ffnMergeId
        ];
        flowNodeIds.slice(0, -1).forEach((sourceId, index) => {
          const source = nodes.find(node => node.id === sourceId);
          const target = nodes.find(node => node.id === flowNodeIds[index + 1]);
          const direction = target.y > source.y ? 1 : -1;
          streamArrowSegments.push({
            x: streamX,
            startY: source.y + direction * source.height / 2,
            endY: target.y - direction * target.height / 2,
            direction
          });
        });
        mergeNodes.push(
          { id: attnMergeId, compact: false },
          { id: ffnMergeId, compact: false }
        );
        residualBundlePaths.push(
          [
            { x: streamX, y: layerY(225) }, { x: mixX, y: layerY(225) },
            { x: mixX, y: layerY(505) }, { x: streamX - 20, y: layerY(505) }
          ],
          [
            { x: streamX, y: layerY(565) }, { x: mixX, y: layerY(565) },
            { x: mixX, y: layerY(855) }, { x: streamX - 20, y: layerY(855) }
          ]
        );
      } else {
        const sliceY = y => layer % 2 === 0 ? y : SIDE_LAYER_TOP + SIDE_LAYER_BOTTOM - y;
        const sliceSpecs = [
          ['attn_pre_slice', 255, 'sem:linear'],
          ['attention_slice', 365, descriptor.attentionType === 'csa' ? 'sem:attention' : 'sem:rope'],
          ['ffn_pre_slice', 650, 'sem:linear'],
          ['moe_slice', 750, 'sem:gate']
        ];
        sliceSpecs.forEach(([name, y, colorKey]) => {
          nodes.push(graphNode(`${clusterId}/${name}`, '', streamX, sliceY(y), {
            parent: clusterId,
            colorKey,
            width: name.includes('attention') || name.includes('moe') ? 8 : 6,
            height: 62,
            role: 'side-operator-slice'
          }));
        });
        [505, 855].forEach((y, index) => {
          const id = `${clusterId}/${index ? 'ffn_merge' : 'attn_merge'}`;
          nodes.push(graphNode(id, '', streamX, sliceY(y), {
            parent: clusterId, colorKey: 'sem:linear', width: 20, height: 20, role: 'side-merge-anchor'
          }));
          mergeNodes.push({ id, compact: true });
        });
      }
      cursorX += width + layerGap;
    }

    const firstLayer = layerRecords[0];
    const lastLayer = layerRecords[layerRecords.length - 1];
    const sceneWidth = cursorX - layerGap + sidePadding;
    const embeddingId = 'dv4/side/stages/token_embedding';
    const expandId = 'dv4/side/stages/hc_expand';
    const headId = 'dv4/side/stages/hc_head';
    const finalNormId = 'dv4/side/stages/final_norm';
    const lmHeadId = 'dv4/side/stages/lm_head';
    const headY = SIDE_LAYER_BOTTOM + 85;
    const finalNormY = headY + 65;
    const lmHeadY = finalNormY + 65;
    nodes.push(
      graphNode(embeddingId, 'Token Embedding', firstLayer.streamX, 32, {
        colorKey: 'sem:embedding', width: 190, height: 42
      }),
      graphNode(expandId, 'HC Expand · d→4×d', firstLayer.streamX, 104, {
        colorKey: 'sem:linear', width: 210, height: 44, role: 'mhc-projection'
      }),
      graphNode(headId, 'HC Head · 4×d→d', lastLayer.streamX, headY, {
        colorKey: 'sem:linear', width: 210, height: 44, role: 'mhc-projection'
      }),
      graphNode(finalNormId, 'Final RMSNorm', lastLayer.streamX, finalNormY, {
        colorKey: 'sem:norm', width: 190, height: 42
      }),
      graphNode(lmHeadId, 'LM Head', lastLayer.streamX, lmHeadY, {
        colorKey: 'sem:head', width: 190, height: 42
      })
    );
    edges.push(
      graphEdge('edge/side-embedding-expand', embeddingId, expandId, 'embedding_states'),
      graphEdge('edge/side-head-norm', headId, finalNormId, 'collapsed_hidden'),
      graphEdge('edge/side-norm-lm-head', finalNormId, lmHeadId, 'normalized_hidden')
    );

    const mainStreamPoints = [];
    const appendPoint = (x, y) => {
      const previous = mainStreamPoints.at(-1);
      if (previous?.x === x && previous?.y === y) return;
      mainStreamPoints.push({ x, y });
    };
    appendPoint(firstLayer.streamX, 126);
    layerRecords.forEach((record, index) => {
      const travelsDown = index % 2 === 0;
      const entryY = travelsDown ? SIDE_LAYER_TOP : SIDE_LAYER_BOTTOM;
      const exitY = travelsDown ? SIDE_LAYER_BOTTOM : SIDE_LAYER_TOP;
      appendPoint(record.streamX, entryY);
      appendPoint(record.streamX, exitY);
      const next = layerRecords[index + 1];
      if (!next) return;
      const foldY = travelsDown ? SIDE_LAYER_BOTTOM + 55 : SIDE_LAYER_TOP - 45;
      appendPoint(record.streamX, foldY);
      appendPoint(next.streamX, foldY);
    });
    appendPoint(lastLayer.streamX, headY - 22);

    return {
      schemaVersion: 'model_architecture_graph.v1',
      id: `deepseek-v4-pro-side-layer-${selected}`,
      title: `DeepSeek V4 Pro side view · Layer ${selected}`,
      width: sceneWidth,
      height: lmHeadY + 45,
      nodes,
      edges,
      clusters,
      metadata: {
        modelId: config.modelId,
        view: 'side',
        selectedLayer: selected,
        layerRecords,
        mainStreamPoints,
        mergeNodes,
        residualBundlePaths,
        streamArrowSegments
      }
    };
  }

  function svgElement(name, attributes = {}) {
    const element = document.createElementNS(SVG_NS, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    return element;
  }

  function decorateRoleClasses(host, graph) {
    (graph.nodes || []).forEach(node => {
      const rendered = host.querySelector(`[data-node-id="${CSS.escape(node.id)}"]`);
      if (!rendered) return;
      if (node.role) rendered.classList.add(`is-${node.role}`);
    });
  }

  function bundlePathData(points, offset) {
    return points.map((point, index) => {
      const previous = points[index - 1];
      const next = points[index + 1];
      const adjacent = [previous, next].filter(Boolean);
      const touchesVertical = adjacent.some(entry => entry.x === point.x);
      const touchesHorizontal = adjacent.some(entry => entry.y === point.y);
      // Parallel orthogonal lanes need a horizontal offset on vertical runs
      // and a vertical offset on horizontal runs. At a bend both offsets are
      // applied, preserving four distinct 90-degree corners.
      const x = point.x + (touchesVertical ? offset : 0);
      const y = point.y + (touchesHorizontal ? offset : 0);
      return `${index ? 'L' : 'M'} ${x} ${y}`;
    }).join(' ');
  }

  function roundedPathData(points, radius = 12) {
    const cleanPoints = points.filter(point => Number.isFinite(point.x) && Number.isFinite(point.y));
    if (cleanPoints.length < 2) return '';
    const parts = [`M ${cleanPoints[0].x} ${cleanPoints[0].y}`];
    const pointToward = (from, to, distance) => {
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const length = Math.hypot(dx, dy);
      if (!length) return { ...from };
      const ratio = Math.min(1, distance / length);
      return { x: from.x + dx * ratio, y: from.y + dy * ratio };
    };
    for (let index = 1; index < cleanPoints.length - 1; index += 1) {
      const previous = cleanPoints[index - 1];
      const point = cleanPoints[index];
      const next = cleanPoints[index + 1];
      const bendRadius = Math.min(
        radius,
        Math.hypot(point.x - previous.x, point.y - previous.y) / 2,
        Math.hypot(next.x - point.x, next.y - point.y) / 2
      );
      const before = pointToward(point, previous, bendRadius);
      const after = pointToward(point, next, bendRadius);
      parts.push(`L ${before.x} ${before.y}`);
      parts.push(`Q ${point.x} ${point.y} ${after.x} ${after.y}`);
    }
    const end = cleanPoints[cleanPoints.length - 1];
    parts.push(`L ${end.x} ${end.y}`);
    return parts.join(' ');
  }

  function decorateMhcBundles(host, graph) {
    const svg = host.querySelector('.pto-model-graphviz-svg');
    if (!svg) return;
    const group = svgElement('g', { class: 'pto-dv4-mhc-bundles', 'aria-hidden': 'true' });
    (graph.edges || []).filter(edge => edge.bundlePath?.length > 1).forEach(edge => {
      const rendered = [...host.querySelectorAll('.pto-model-graphviz-edge')].find(item => (
        item.dataset.source === edge.source && item.dataset.target === edge.target
      ));
      if (rendered) rendered.style.display = 'none';
      BUNDLE_OFFSETS.forEach(offset => {
        group.appendChild(svgElement('path', {
          class: 'pto-dv4-mhc-bundle-line',
          d: bundlePathData(edge.bundlePath, offset)
        }));
      });
    });
    const firstNode = svg.querySelector('.pto-model-graphviz-node');
    svg.insertBefore(group, firstNode || null);
  }

  function decorateFanCurves(host, graph, edgeStyle) {
    const nodesById = new Map((graph.nodes || []).map(node => [node.id, node]));
    const renderedEdges = [...host.querySelectorAll('.pto-model-graphviz-edge')];
    (graph.edges || []).filter(edge => Number.isFinite(edge.fanCurveY)).forEach(edge => {
      const source = nodesById.get(edge.source);
      const target = nodesById.get(edge.target);
      const rendered = renderedEdges.find(item => (
        item.dataset.source === edge.source && item.dataset.target === edge.target
      ));
      if (!source || !target || !rendered) return;
      const start = { x: source.x, y: source.y + source.height / 2 };
      const end = { x: target.x, y: target.y - target.height / 2 };
      const path = edgeStyle === 'orthogonal'
        ? roundedPathData([
          start,
          { x: start.x, y: edge.fanCurveY },
          { x: end.x, y: edge.fanCurveY },
          end
        ], 12)
        : `M ${start.x} ${start.y} C ${start.x} ${edge.fanCurveY}, ${end.x} ${edge.fanCurveY}, ${end.x} ${end.y}`;
      rendered.setAttribute('d', path);
    });
  }

  function decorateSideBundles(host, graph) {
    const svg = host.querySelector('.pto-model-graphviz-svg');
    if (!svg) return;
    const streamGroup = svgElement('g', { class: 'pto-dv4-side-main-stream', 'aria-hidden': 'true' });
    BUNDLE_OFFSETS.forEach(offset => {
      streamGroup.appendChild(svgElement('path', {
        class: 'pto-dv4-side-main-stream__line',
        d: bundlePathData(graph.metadata.mainStreamPoints, offset)
      }));
    });
    const residualGroup = svgElement('g', { class: 'pto-dv4-side-residual-bundles', 'aria-hidden': 'true' });
    (graph.metadata.residualBundlePaths || []).forEach(points => {
      BUNDLE_OFFSETS.forEach(offset => {
        residualGroup.appendChild(svgElement('path', {
          class: 'pto-dv4-side-residual-bundle__line',
          d: bundlePathData(points, offset)
        }));
      });
    });
    const arrowGroup = svgElement('g', { class: 'pto-dv4-side-stream-arrows', 'aria-hidden': 'true' });
    (graph.metadata.streamArrowSegments || []).forEach(segment => {
      const span = Math.abs(segment.endY - segment.startY);
      const arrowLength = Math.min(6, Math.max(4, span - 2));
      BUNDLE_OFFSETS.forEach(offset => {
        const x = segment.x + offset;
        const tipY = segment.endY - segment.direction;
        const baseY = tipY - segment.direction * arrowLength;
        arrowGroup.appendChild(svgElement('path', {
          class: 'pto-dv4-side-stream-arrow',
          d: `M ${x} ${tipY} L ${x - 2} ${baseY} L ${x + 2} ${baseY} Z`
        }));
      });
    });
    const firstNode = svg.querySelector('.pto-model-graphviz-node');
    svg.insertBefore(streamGroup, firstNode || null);
    svg.insertBefore(residualGroup, firstNode || null);
    svg.insertBefore(arrowGroup, firstNode || null);
  }

  function decorateSideMerges(host, graph) {
    const svg = host.querySelector('.pto-model-graphviz-svg');
    const nodesById = new Map((graph.nodes || []).map(node => [node.id, node]));
    if (!svg) return;
    (graph.metadata.mergeNodes || []).forEach(entry => {
      const node = nodesById.get(entry.id);
      if (!node) return;
      const compact = entry.compact === true;
      const radius = compact ? 8 : 15;
      const arm = compact ? 4 : 8;
      const group = svgElement('g', {
        class: `pto-dv4-side-merge${compact ? ' is-compact' : ' is-expanded'}`,
        'aria-hidden': 'true'
      });
      group.appendChild(svgElement('circle', {
        class: 'pto-dv4-side-merge__face', cx: node.x, cy: node.y, r: radius
      }));
      group.appendChild(svgElement('path', {
        class: 'pto-dv4-side-merge__plus',
        d: `M ${node.x - arm} ${node.y} H ${node.x + arm} M ${node.x} ${node.y - arm} V ${node.y + arm}`
      }));
      if (!compact) {
        group.appendChild(svgElement('rect', {
          class: 'pto-dv4-side-merge__badge',
          x: node.x + 9,
          y: node.y + 7,
          width: 28,
          height: 16,
          rx: 8
        }));
        const badge = svgElement('text', {
          class: 'pto-dv4-side-merge__badge-text',
          x: node.x + 23,
          y: node.y + 15
        });
        badge.textContent = '×4';
        group.appendChild(badge);
      }
      svg.appendChild(group);
    });
  }

  function decorateSideLayers(host, graph, onSelect) {
    const records = new Map((graph.metadata.layerRecords || []).map(record => [record.layer, record]));
    host.querySelectorAll('[data-cluster-id], [data-node-id]').forEach(item => {
      const id = item.dataset.clusterId || item.dataset.nodeId || '';
      const match = id.match(/^dv4\/side\/layer\/(\d+)/);
      if (!match) return;
      const layer = Number(match[1]);
      const record = records.get(layer);
      item.dataset.sideLayerIndex = String(layer);
      item.classList.add('pto-dv4-side-layer-item');
      if (record?.selected) item.classList.add('is-side-selected-layer');
      if (item.dataset.clusterId) {
        item.classList.add(record?.selected ? 'is-side-selected-cluster' : 'is-side-compact-cluster');
      }
    });
    host.addEventListener('click', event => {
      const item = event.target.closest('[data-side-layer-index]');
      if (!item) return;
      onSelect(Number(item.dataset.sideLayerIndex));
    });
  }

  function decorateMhcMerge(host, graph, nodeId) {
    const svg = host.querySelector('.pto-model-graphviz-svg');
    const node = graph.nodes.find(entry => entry.id === nodeId);
    if (!svg || !node) return;
    const group = svgElement('g', {
      class: 'pto-dv4-mhc-merge',
      role: 'img',
      'aria-label': 'mHC Merge: four residual streams'
    });
    const title = svgElement('title');
    title.textContent = 'mHC Merge ×4 · B(X) + HC Post(F(HC Pre(X))) · [B,T,4,d]';
    group.appendChild(title);
    group.appendChild(svgElement('circle', {
      class: 'pto-dv4-mhc-merge__face', cx: node.x, cy: node.y, r: 24
    }));
    const plus = svgElement('path', {
      class: 'pto-dv4-mhc-merge__plus',
      d: `M ${node.x - 12} ${node.y} H ${node.x + 12} M ${node.x} ${node.y - 12} V ${node.y + 12}`
    });
    group.appendChild(plus);
    group.appendChild(svgElement('rect', {
      class: 'pto-dv4-mhc-merge__badge',
      x: node.x + 15,
      y: node.y + 13,
      width: 44,
      height: 22,
      rx: 11
    }));
    const badge = svgElement('text', {
      class: 'pto-dv4-mhc-merge__badge-text', x: node.x + 37, y: node.y + 24
    });
    badge.textContent = '×4';
    group.appendChild(badge);
    const label = svgElement('text', {
      class: 'pto-dv4-mhc-merge__label', x: node.x + 70, y: node.y
    });
    label.textContent = 'mHC Merge';
    group.appendChild(label);
    svg.appendChild(group);
  }

  function decorateHashInput(host, graph) {
    const hashMode = graph.metadata.routerType === 'hash_moe';
    const tokenNode = (graph.nodes || []).find(node => node.role === 'token-id-input');
    if (!tokenNode) return;
    const renderedNode = host.querySelector(`[data-node-id="${CSS.escape(tokenNode.id)}"]`);
    if (renderedNode && !hashMode) renderedNode.classList.add('is-router-input-inactive');
    const tokenEdges = [...host.querySelectorAll('.pto-model-graphviz-edge')].filter(edge => edge.dataset.source === tokenNode.id);
    tokenEdges.forEach(edge => edge.classList.toggle('is-router-input-inactive', !hashMode));
  }

  function shell(config, options = {}) {
    const layerOptions = Array.from({ length: config.numHiddenLayers }, (_, layer) => (
      `<option value="${layer}">Layer ${layer}</option>`
    )).join('');
    const edgeStyleControl = options.showEdgeStyleControl === false ? '' : `
          <button class="pto-dv4-architecture__button is-edge-style is-active" type="button" data-dv4-edge-style aria-label="切换为贝塞尔曲线连线" aria-pressed="true">连线 · 圆角直角</button>`;
    return `<div class="pto-dv4-architecture__workspace">
      <div class="pto-dv4-architecture__viewport" tabindex="0" aria-label="可缩放拖动的 DeepSeek V4 正视图">
        <div class="pto-dv4-architecture__canvas"></div>
        <div class="pto-dv4-architecture__layer-picker" data-dv4-overlay>
          <div class="pto-dv4-architecture__view-switch" role="group" aria-label="架构视图">
            <button class="pto-dv4-architecture__button" type="button" data-dv4-view="front">正视</button>
            <button class="pto-dv4-architecture__button" type="button" data-dv4-view="side">侧视</button>
          </div>
          <div class="pto-dv4-architecture__layer-controls">
            <button class="pto-dv4-architecture__button is-icon" type="button" data-dv4-layer-step="-1" aria-label="上一层">‹</button>
            <select class="pto-dv4-architecture__layer-select" id="dv4LayerSelect" data-dv4-layer aria-label="Decoder Layer">${layerOptions}</select>
            <button class="pto-dv4-architecture__button is-icon" type="button" data-dv4-layer-step="1" aria-label="下一层">›</button>
          </div>
        </div>
        <div class="pto-dv4-architecture__viewport-actions" data-dv4-overlay>
          <button class="pto-dv4-architecture__button is-icon" type="button" data-dv4-theme aria-label="切换主题">◐</button>
          ${edgeStyleControl}
          <button class="pto-dv4-architecture__button" type="button" data-dv4-fit>适配</button>
          <span class="pto-dv4-architecture__readout" data-dv4-readout>100%</span>
        </div>
        <aside class="pto-dv4-architecture__legend" data-dv4-overlay aria-label="mHC 图例">
          <div class="pto-dv4-architecture__legend-item"><i class="pto-dv4-architecture__legend-mark is-streams"></i><strong>4 × Residual Streams</strong></div>
          <div class="pto-dv4-architecture__legend-item"><i class="pto-dv4-architecture__legend-mark is-merge"></i><strong>mHC Merge ×4</strong></div>
        </aside>
      </div>
    </div>`;
  }

  function render(rootInput, options = {}) {
    const root = queryRoot(rootInput);
    const data = global.DeepSeekV4ArchitectureData;
    if (!root || !data) return null;
    root.classList.add('pto-dv4-architecture', 'pto-model-graphviz-pattern-page');
    root.dataset.sharedPattern = 'model-architecture-training-sidecar-dv4-working-copy';
    root.innerHTML = shell(data.config, options);

    const viewport = root.querySelector('.pto-dv4-architecture__viewport');
    const canvas = root.querySelector('.pto-dv4-architecture__canvas');
    const readout = root.querySelector('[data-dv4-readout]');
    const layerSelect = root.querySelector('[data-dv4-layer]');
    const state = {
      selectedLayer: data.clampLayer(options.initialLayer ?? 3),
      view: options.initialView === 'side' ? 'side' : 'front',
      theme: options.initialTheme === 'light' ? 'light' : 'dark',
      edgeStyle: options.initialEdgeStyle === 'bezier' ? 'bezier' : 'orthogonal',
      zoom: 1,
      panX: 0,
      panY: 0,
      sceneWidth: SCENE_WIDTH,
      sceneHeight: SCENE_HEIGHT,
      drag: null,
      graph: null,
      collapsedFrontIds: new Set(),
      destroyed: false
    };

    function applyTransform() {
      canvas.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
      readout.textContent = `${Math.round(state.zoom * 100)}%`;
    }

    function fit() {
      const rect = viewport.getBoundingClientRect();
      if (!rect.width || !rect.height) return api;
      const reserve = 28;
      state.zoom = Math.min(1.1, Math.max(.16, Math.min(
        (rect.width - reserve * 2) / state.sceneWidth,
        (rect.height - reserve * 2) / state.sceneHeight
      )));
      state.panX = (rect.width - state.sceneWidth * state.zoom) / 2;
      state.panY = (rect.height - state.sceneHeight * state.zoom) / 2;
      applyTransform();
      return api;
    }

    function syncLayerControls() {
      layerSelect.value = String(state.selectedLayer);
      root.querySelectorAll('[data-dv4-view]').forEach(button => {
        const active = button.dataset.dv4View === state.view;
        button.classList.toggle('is-active', active);
        button.setAttribute('aria-pressed', String(active));
      });
      const edgeStyleButton = root.querySelector('[data-dv4-edge-style]');
      const orthogonal = state.edgeStyle === 'orthogonal';
      if (edgeStyleButton) {
        edgeStyleButton.textContent = orthogonal ? '连线 · 圆角直角' : '连线 · 曲线';
        edgeStyleButton.setAttribute('aria-label', orthogonal ? '切换为贝塞尔曲线连线' : '切换为圆角直角连线');
        edgeStyleButton.setAttribute('aria-pressed', String(orthogonal));
        edgeStyleButton.classList.toggle('is-active', orthogonal);
      }
    }

    function renderScene(shouldFit = true) {
      const sideView = state.view === 'side';
      const graph = sideView
        ? buildSideGraph(state.selectedLayer)
        : buildFrontGraph(state.selectedLayer, state.collapsedFrontIds);
      state.graph = graph;
      state.sceneWidth = graph.width;
      state.sceneHeight = graph.height;
      canvas.style.width = `${graph.width}px`;
      canvas.style.height = `${graph.height}px`;
      canvas.innerHTML = `<div class="pto-dv4-architecture__graph is-${state.view}"></div>`;
      const host = canvas.querySelector('.pto-dv4-architecture__graph');
      if (!global.PtoModelGraphvizPattern) {
        host.innerHTML = '<div class="pto-dv4-architecture__empty">model-graphviz renderer unavailable</div>';
        return api;
      }
      global.PtoModelGraphvizPattern.render(host, graph, {
        width: graph.width,
        height: graph.height,
        ariaLabel: sideView
          ? `DeepSeek V4 Pro 61-layer side view · Layer ${state.selectedLayer} expanded`
          : `DeepSeek V4 Pro Decoder Layer ${state.selectedLayer} front view`,
        metricOverlays: false,
        reportOverlays: false,
        performanceHeatmap: { enabled: false },
        colormap: global.PtoModelGraphvizPattern.modelArchitectureColormap(graph, { theme: state.theme }),
        interaction: { panZoom: false, selectable: false },
        selectable: false,
        selectableClusters: false,
        autoFit: false,
        edgeRouting: state.edgeStyle,
        edgeMarkerSize: 6,
        edgeCornerRadius: 12,
        initialTransform: { tx: 0, ty: 0, zoom: 1 },
        onToggle({ nodeId, collapsed }) {
          if (sideView) return;
          toggleFrontCluster(nodeId, collapsed);
        }
      });
      decorateRoleClasses(host, graph);
      if (sideView) {
        decorateSideBundles(host, graph);
        decorateSideMerges(host, graph);
        decorateSideLayers(host, graph, selectLayer);
      } else {
        decorateHashInput(host, graph);
        decorateFanCurves(host, graph, state.edgeStyle);
        decorateMhcBundles(host, graph);
        graph.metadata.mergeNodeIds.forEach(nodeId => decorateMhcMerge(host, graph, nodeId));
      }
      syncLayerControls();
      if (shouldFit) requestAnimationFrame(fit);
      else applyTransform();
      return api;
    }

    function selectLayer(value) {
      const next = data.clampLayer(value);
      if (next === state.selectedLayer) return api;
      state.selectedLayer = next;
      renderScene(false);
      return api;
    }

    function toggleFrontCluster(clusterId, collapsed) {
      if (!clusterId) return api;
      if (collapsed === true) state.collapsedFrontIds.delete(clusterId);
      else state.collapsedFrontIds.add(clusterId);
      renderScene(true);
      return api;
    }

    function setView(view) {
      const next = view === 'side' ? 'side' : 'front';
      if (next === state.view) return api;
      state.view = next;
      renderScene(true);
      return api;
    }

    function setTheme(theme) {
      const next = theme === 'light' ? 'light' : 'dark';
      if (next === state.theme) return api;
      state.theme = next;
      document.documentElement.dataset.theme = next;
      renderScene(false);
      return api;
    }

    function setEdgeStyle(style) {
      const next = style === 'orthogonal' ? 'orthogonal' : 'bezier';
      if (next === state.edgeStyle) return api;
      state.edgeStyle = next;
      renderScene(false);
      return api;
    }

    function onWheel(event) {
      event.preventDefault();
      const rect = viewport.getBoundingClientRect();
      const screenX = event.clientX - rect.left;
      const screenY = event.clientY - rect.top;
      const worldX = (screenX - state.panX) / state.zoom;
      const worldY = (screenY - state.panY) / state.zoom;
      const factor = Math.exp(-event.deltaY * .0012);
      state.zoom = Math.max(.14, Math.min(2.4, state.zoom * factor));
      state.panX = screenX - worldX * state.zoom;
      state.panY = screenY - worldY * state.zoom;
      applyTransform();
    }

    viewport.addEventListener('wheel', onWheel, { passive: false });
    viewport.addEventListener('pointerdown', event => {
      if (
        event.button !== 0
        || event.target.closest('[data-dv4-overlay]')
        || event.target.closest('.pto-model-graphviz-toggle')
        || event.target.closest('[data-side-layer-index]')
      ) return;
      state.drag = { x: event.clientX, y: event.clientY, panX: state.panX, panY: state.panY };
      viewport.setPointerCapture(event.pointerId);
      viewport.classList.add('is-dragging');
    });
    viewport.addEventListener('pointermove', event => {
      if (!state.drag) return;
      state.panX = state.drag.panX + event.clientX - state.drag.x;
      state.panY = state.drag.panY + event.clientY - state.drag.y;
      applyTransform();
    });
    const endDrag = event => {
      if (!state.drag) return;
      state.drag = null;
      viewport.classList.remove('is-dragging');
      if (viewport.hasPointerCapture(event.pointerId)) viewport.releasePointerCapture(event.pointerId);
    };
    viewport.addEventListener('pointerup', endDrag);
    viewport.addEventListener('pointercancel', endDrag);

    layerSelect.addEventListener('change', event => selectLayer(event.target.value));
    root.querySelectorAll('[data-dv4-layer-step]').forEach(button => {
      button.addEventListener('click', () => selectLayer(
        state.selectedLayer + Number(button.dataset.dv4LayerStep)
      ));
    });
    root.querySelectorAll('[data-dv4-view]').forEach(button => {
      button.addEventListener('click', () => setView(button.dataset.dv4View));
    });
    root.querySelector('[data-dv4-edge-style]')?.addEventListener('click', () => {
      setEdgeStyle(state.edgeStyle === 'bezier' ? 'orthogonal' : 'bezier');
    });
    root.querySelector('[data-dv4-fit]').addEventListener('click', fit);
    root.querySelector('[data-dv4-theme]').addEventListener('click', () => (
      setTheme(state.theme === 'light' ? 'dark' : 'light')
    ));

    const resizeObserver = new ResizeObserver(() => fit());
    resizeObserver.observe(viewport);
    document.documentElement.dataset.theme = state.theme;

    const api = {
      root,
      fit,
      selectLayer,
      setView,
      setTheme,
      setEdgeStyle,
      getState() {
        return {
          selectedLayer: state.selectedLayer,
          view: state.view,
          theme: state.theme,
          edgeStyle: state.edgeStyle,
          zoom: state.zoom,
          collapsedFrontIds: [...state.collapsedFrontIds]
        };
      },
      destroy() {
        if (state.destroyed) return;
        state.destroyed = true;
        resizeObserver.disconnect();
        viewport.removeEventListener('wheel', onWheel);
        root.innerHTML = '';
      }
    };

    renderScene(true);
    return api;
  }

  global.PtoDv4ArchitecturePattern = Object.freeze({
    render,
    buildFrontGraph,
    buildSideGraph,
    // Reuse the same folding and visual decorations when an adapter supplies
    // additional source-backed descendants; existing previews are unchanged.
    projectCollapsedFrontGraph,
    decorateFrontGraph(host, graph, edgeStyle = 'orthogonal') {
      decorateRoleClasses(host, graph);
      decorateHashInput(host, graph);
      decorateFanCurves(host, graph, edgeStyle);
      decorateMhcBundles(host, graph);
      (graph.metadata?.mergeNodeIds || []).forEach(id => decorateMhcMerge(host, graph, id));
    }
  });
})(window);
