(function attachPtoCsaImplementationPattern(global) {
  'use strict';

  function queryRoot(input) {
    return typeof input === 'string' ? document.querySelector(input) : input;
  }

  const registry = global.PtoCsaMappingRegistry;

  function mappingFor(id) {
    const item = registry?.getNode(id);
    return [item?.mappingType || null, item?.reason || '暂无映射证据'];
  }

  function implementationColor(id, fallback) {
    const scopes = { q_projection: 'csa_query_projection', kv_projection: 'csa_recent_window_kv',
      main_compressor: 'csa_compressed_kv_path', indexer_compressor: 'csa_lightning_indexer',
      lightning_indexer: 'csa_lightning_indexer', exact_topk: 'csa_lightning_indexer',
      sparse_attention: 'csa_sparse_shared_kv_attention', grouped_output_projection: 'csa_grouped_output_projection' };
    if (scopes[id]) return SEMANTIC_COLOR[scopes[id]];
    const sources = registry.relationsFor(id).flatMap(r => r.sourceIds).map(key => registry.getNode(key))
      .filter(item => item?.level === 'L2');
    const colors = [...new Set(sources.map(leafColor))];
    return colors.length === 1 ? colors[0] : fallback;
  }

  function shell() {
    return `<div class="pto-dv4-architecture__workspace">
      <div class="pto-dv4-architecture__viewport" tabindex="0" aria-label="可缩放拖动的 CSA 分布式算子实现图">
        <div class="pto-dv4-architecture__canvas"></div>
        <div class="pto-dv4-architecture__viewport-actions" data-dv4-overlay>
          <button class="pto-dv4-architecture__button" type="button" data-dv4-fit>适配</button>
        </div>
      </div>
    </div>`;
  }

  const SEMANTIC_COLOR = Object.freeze({
    csa_query_projection: 'sem:linear',
    csa_recent_window_kv: 'sem:attention',
    csa_compressed_kv_path: 'sem:attention',
    csa_lightning_indexer: 'sem:gate',
    csa_sparse_shared_kv_attention: 'sem:attention',
    csa_grouped_output_projection: 'sem:linear'
  });

  function leafColor(item) {
    if (item.entityKind !== 'Op') return 'io:state';
    if (/normalization/i.test(item.label)) return 'sem:norm';
    if (/RoPE/i.test(item.label)) return 'sem:rope';
    if (/Top-K|score|positions/i.test(item.label)) return 'sem:gate';
    return SEMANTIC_COLOR[item.parentId] || 'sem:attention';
  }

  function buildCollapsedSemanticGraph() {
    const canonical = registry.canonicalGraph;
    const projection = canonical.l1Projection;
    const frame = { x: 24, y: 100 };
    const nodeAt = point => ({ x: frame.x + point.x, y: frame.y + point.y });
    const nodes = canonical.compoundIds.map(compoundId => {
      const item = registry.getNode(compoundId);
      const presentation = projection.nodes[compoundId];
      return {
        id: item.id,
        label: item.label,
        kind: 'module',
        hideTypeLabel: true,
        colorKey: presentation.colorKey,
        width: presentation.width,
        height: presentation.height,
        ...nodeAt(presentation),
        parent: canonical.scopeId,
        selectable: true,
        collapsed: true,
        sourceRefs: item.sourceRefs
      };
    });
    nodes.push(
      {
        id: canonical.inputId,
        label: 'RMSNorm hidden state',
        kind: 'tensor',
        typeLabel: 'Upstream context',
        colorKey: 'io:input',
        width: 230,
        height: 44,
        ...nodeAt(projection.inputAnchor),
        selectable: false,
        hideTypeLabel: true
      },
      {
        id: canonical.outputId,
        label: 'CSA semantic output',
        kind: 'tensor',
        typeLabel: 'Downstream context',
        colorKey: 'io:output',
        width: 230,
        height: 44,
        ...nodeAt(projection.outputAnchor),
        selectable: false,
        hideTypeLabel: true
      }
    );
    return {
      width: projection.width + frame.x * 2,
      height: frame.y + Math.max(projection.height, projection.outputAnchor.y + 22) + 24,
      title: 'CSA canonical L1 semantics',
      metadata: {
        graphRole: 'canonical_hierarchical_semantics',
        distributedRuntime: false,
        expandedCompoundIds: [],
        projection: 'canonical_l1'
      },
      nodes,
      clusters: [{
        id: canonical.scopeId,
        label: projection.label,
        x: frame.x,
        y: frame.y,
        width: projection.width,
        height: projection.height,
        parent: null,
        nodes: [...canonical.compoundIds],
        children: [],
        colorKey: 'sem:attention',
        selectable: false,
        collapsible: false
      }],
      edges: canonical.summaryEdges.map(edge => ({
        ...edge,
        sourceAnchor: 'bottom',
        targetAnchor: 'top',
        curve: 'vertical',
        tensor: edge.label ? { name: edge.label } : undefined
      }))
    };
  }

  function buildSemanticGraph(expandedCompoundIds = new Set()) {
    const canonical = registry.canonicalGraph;
    const projection = canonical.l1Projection;
    const expanded = new Set([...expandedCompoundIds].filter(id => canonical.compoundIds.includes(id)));
    if (expanded.size === 0) return buildCollapsedSemanticGraph();
    const frame = { x: 24, y: 100 };
    const baseHeight = 56;
    const childHeight = 44;
    const childGap = 16;
    const clusterHeader = 58;
    const clusterBottom = 18;
    const expandedHeight = compoundId => {
      if (!expanded.has(compoundId)) return baseHeight;
      const childCount = registry.getNode(compoundId).children.length;
      return clusterHeader + childCount * childHeight + Math.max(0, childCount - 1) * childGap + clusterBottom;
    };
    const expansionDelta = compoundId => expanded.has(compoundId)
      ? Math.max(0, expandedHeight(compoundId) - baseHeight + 20)
      : 0;
    const queryDelta = expansionDelta('csa_query_projection');
    const branchDelta = Math.max(
      expansionDelta('csa_recent_window_kv'),
      expansionDelta('csa_compressed_kv_path')
    );
    const indexerDelta = expansionDelta('csa_lightning_indexer');
    const attentionDelta = expansionDelta('csa_sparse_shared_kv_attention');
    const outputDelta = expansionDelta('csa_grouped_output_projection');
    const yOffsets = {
      csa_query_projection: 0,
      csa_recent_window_kv: queryDelta,
      csa_compressed_kv_path: queryDelta,
      csa_lightning_indexer: queryDelta + branchDelta,
      csa_sparse_shared_kv_attention: queryDelta + branchDelta + indexerDelta,
      csa_grouped_output_projection: queryDelta + branchDelta + indexerDelta + attentionDelta
    };
    const nodeAt = (point, yOffset = 0) => ({ x: frame.x + point.x, y: frame.y + point.y + yOffset });
    const nodes = [];
    const clusters = [];

    canonical.compoundIds.forEach(compoundId => {
      const item = registry.getNode(compoundId);
      const presentation = projection.nodes[compoundId];
      const yOffset = yOffsets[compoundId];
      if (!expanded.has(compoundId)) {
        nodes.push({
          id: item.id, label: item.label, kind: 'module', hideTypeLabel: true,
          colorKey: presentation.colorKey, width: presentation.width, height: presentation.height,
          ...nodeAt(presentation, yOffset), parent: canonical.scopeId, selectable: true, collapsed: true,
          sourceRefs: item.sourceRefs
        });
        return;
      }
      const clusterX = frame.x + presentation.x - presentation.width / 2;
      const clusterY = frame.y + presentation.y + yOffset - baseHeight / 2;
      const clusterHeight = expandedHeight(compoundId);
      clusters.push({
        id: item.id, label: item.label, x: clusterX, y: clusterY,
        width: presentation.width, height: clusterHeight,
        parent: canonical.scopeId, nodes: item.children, children: [], colorKey: presentation.colorKey,
        selectable: true, collapsible: true, sourceRefs: item.sourceRefs
      });
      item.children.forEach((childId, index) => {
        const child = registry.getNode(childId);
        nodes.push({
          id: child.id, label: child.label, kind: child.entityKind === 'Op' ? 'op' : 'tensor', hideTypeLabel: true,
          entityKind: child.entityKind,
          colorKey: leafColor(child), width: presentation.width - 20, height: childHeight,
          x: frame.x + presentation.x,
          y: clusterY + clusterHeader + childHeight / 2 + index * (childHeight + childGap),
          parent: item.id, selectable: true,
          sourceRefs: child.sourceRefs
        });
      });
    });

    const totalDelta = queryDelta + branchDelta + indexerDelta + attentionDelta + outputDelta;
    nodes.push(
      {
        id: canonical.inputId, label: 'RMSNorm hidden state', kind: 'tensor', typeLabel: 'Upstream context',
        colorKey: 'io:input', width: 230, height: 44, ...nodeAt(projection.inputAnchor),
        selectable: false, hideTypeLabel: true
      },
      {
        id: canonical.outputId, label: 'CSA semantic output', kind: 'tensor', typeLabel: 'Downstream context',
        colorKey: 'io:output', width: 230, height: 44, ...nodeAt(projection.outputAnchor, totalDelta),
        selectable: false, hideTypeLabel: true
      }
    );

    const parentFor = id => registry.compoundFor(id)?.id || null;
    const visibleEndpoint = id => {
      const parentId = parentFor(id);
      return parentId && !expanded.has(parentId) ? parentId : id;
    };
    const edgeGroups = new Map();
    canonical.expandedEdges.forEach(edge => {
      const source = visibleEndpoint(edge.source);
      const target = visibleEndpoint(edge.target);
      if (source === target) return;
      const key = `${source}::${target}`;
      const group = edgeGroups.get(key) || { source, target, count: 0, semanticEdgeType: edge.semanticEdgeType };
      group.count += 1;
      edgeGroups.set(key, group);
    });
    const edges = [...edgeGroups.values()].map((edge, index) => ({
      id: `csa_projected_edge_${index}`,
      source: edge.source,
      target: edge.target,
      sourceAnchor: 'bottom',
      targetAnchor: 'top',
      curve: 'vertical',
      semanticEdgeType: edge.semanticEdgeType,
      tensor: edge.count > 1 ? { name: `${edge.count} semantic paths` } : undefined
    }));
    clusters.unshift({
      id: canonical.scopeId, label: projection.label,
      x: frame.x, y: frame.y, width: projection.width, height: projection.height + totalDelta, parent: null,
      nodes: canonical.compoundIds.filter(id => !expanded.has(id)),
      children: canonical.compoundIds.filter(id => expanded.has(id)), colorKey: 'sem:attention',
      selectable: false, collapsible: false
    });
    return {
      width: projection.width + frame.x * 2,
      height: frame.y + Math.max(projection.height + totalDelta, projection.outputAnchor.y + totalDelta + 22) + 24,
      title: 'CSA canonical hierarchical semantics',
      metadata: {
        graphRole: 'canonical_hierarchical_semantics',
        distributedRuntime: false,
        expandedCompoundIds: [...expanded]
      },
      nodes,
      clusters,
      edges
    };
  }

  // One architecture carrier: graft the canonical CSA projection into the
  // existing whole-model hierarchy, then apply its original folding policy.
  // Only geometry is derived here; all CSA dependencies come from the registry.
  function buildWholeModelGraph(expandedIds = new Set(), collapsedIds = new Set()) {
    const pattern = global.PtoDv4ArchitecturePattern;
    const graph = pattern.buildFrontGraph(2);
    const semantic = buildSemanticGraph(expandedIds);
    const localScope = semantic.clusters[0];
    const originalScope = graph.clusters.find(item => item.id.endsWith('/csa'));
    const originalId = originalScope.id;
    const normId = 'dv4/layer/2/mhc_attn/rmsnorm';
    const outputId = 'dv4/layer/2/mhc_attn/hc_post';
    const scheduleId = 'dv4/layer/2/mhc_attn/hybrid_attention/attention_type';
    const scopeId = 'l1_csa';
    const dx = originalScope.x - localScope.x;
    const dy = originalScope.y - localScope.y;
    const delta = localScope.height - originalScope.height;
    const bottom = originalScope.y + originalScope.height;
    const shiftY = y => y >= bottom ? y + delta : y;
    graph.nodes = graph.nodes.filter(item => item.parent !== originalId && item.id !== scheduleId);
    graph.nodes.forEach(item => { item.y = shiftY(item.y); });
    graph.clusters = graph.clusters.filter(item => item.id !== originalId);
    graph.clusters.forEach(item => {
      const end = shiftY(item.y + item.height);
      item.y = shiftY(item.y);
      item.height = end - item.y;
    });
    graph.edges = graph.edges.filter(edge => !edge.id.startsWith('edge/csa-') && edge.target !== scheduleId);
    graph.edges.forEach(edge => {
      if (edge.source === scheduleId) edge.source = normId;
      if (edge.bundlePath) edge.bundlePath = edge.bundlePath.map(point => ({ ...point, y: shiftY(point.y) }));
      if (Number.isFinite(edge.fanCurveY)) edge.fanCurveY = shiftY(edge.fanCurveY);
    });
    const resolve = id => id === registry.canonicalGraph.inputId ? normId
      : id === registry.canonicalGraph.outputId ? outputId : id;
    semantic.nodes.filter(item => ![registry.canonicalGraph.inputId, registry.canonicalGraph.outputId].includes(item.id)).forEach(item => {
      graph.nodes.push({ ...item, x: item.x + dx, y: item.y + dy,
        parent: item.parent === localScope.id ? scopeId : item.parent });
    });
    semantic.clusters.forEach(item => graph.clusters.push({ ...item,
      id: item.id === localScope.id ? scopeId : item.id,
      x: item.x + dx, y: item.y + dy,
      parent: item.id === localScope.id ? originalScope.parent : scopeId,
      selectable: true, collapsible: true,
      sourceRefs: item.id === localScope.id ? registry.getNode(scopeId).sourceRefs : item.sourceRefs
    }));
    semantic.edges.forEach(edge => graph.edges.push({ ...edge, id: `official/${edge.id}`,
      source: resolve(edge.source), target: resolve(edge.target) }));
    graph.height += delta;
    // Derive containment lists from identity, not from the visual sibling order.
    graph.clusters.forEach(item => {
      item.nodes = graph.nodes.filter(node => node.parent === item.id).map(node => node.id);
      item.children = graph.clusters.filter(child => child.parent === item.id).map(child => child.id);
    });
    graph.metadata = { ...graph.metadata, ...semantic.metadata, sourceCarrier: 'whole_model',
      sourceScopeId: scopeId, nonCsaArtifactStatus: 'unavailable' };
    const projected = pattern.projectCollapsedFrontGraph(graph, collapsedIds);
    const foldedCsa = projected.nodes.find(item => item.id === scopeId);
    if (foldedCsa) {
      foldedCsa.selectable = true;
      foldedCsa.sourceRefs = registry.getNode(scopeId).sourceRefs;
    }
    return projected;
  }

  function normalizeGraph(payload, level, expandedCompoundIds) {
    if (level === 'l1' || level === 'l2') return buildSemanticGraph(expandedCompoundIds);
    const graph = structuredClone(payload);
    const semanticNodes = new Map();
    const visit = item => {
      if (!item || typeof item !== 'object') return;
      if (item.id) semanticNodes.set(item.id, item);
      (item.children || []).forEach(visit);
    };
    (graph.roots || []).forEach(visit);
    const linkedBoundaryNodes = new Set(['hc_input', 'attention_hc_output']);
    const visibleNodeIds = new Set(
      (graph.nodes || [])
        .filter(node => node.parent === 'csa_operator_group' || linkedBoundaryNodes.has(node.id))
        .map(node => node.id)
    );
    graph.schemaVersion = graph.schemaVersion || 'model_architecture_graph.v1';
    graph.title = graph.title || graph.model?.name || 'CSA Distributed Operator Implementation';
    graph.metadata = {
      ...(graph.metadata || {}),
      graphRole: 'distributed_operator_implementation',
      sourceModelArchitecture: false,
      implementationSource: 'decode_csa.py'
    };
    graph.nodes = (graph.nodes || []).filter(node => visibleNodeIds.has(node.id)).map(node => {
      const semanticNode = semanticNodes.get(node.id) || {};
      const registryNode = registry?.getNode(node.id);
      const mapping = mappingFor(node.id);
      return {
        ...node,
        label: registryNode?.label || node.label,
        entityKind: registryNode?.entityKind || 'Op',
        kind: ['Value', 'LogicalState', 'RuntimeMetadata'].includes(registryNode?.entityKind) ? 'tensor' : node.kind,
        colorKey: ['Value', 'LogicalState', 'RuntimeMetadata'].includes(registryNode?.entityKind) ? 'io:state' : implementationColor(node.id, node.colorKey),
        sourceRefs: registryNode?.sourceRefs || semanticNode.sourceRefs || node.sourceRefs || [],
        attrs: {
          ...(semanticNode.attrs || node.attrs || {}),
          mappingCode: mapping[0],
          mappingReason: mapping[1]
        },
        selectable: true,
        origin: node.origin || 'implementation',
        dataState: node.dataState || 'implementation_only',
        mappingCode: mapping[0],
        mappingReason: mapping[1]
      };
    });
    const extras = [
      { id: 'slot_mapping', label: 'Slot Mapping', kind: 'tensor', typeLabel: 'Runtime Mapping', colorKey: 'io:state', width: 210, height: 48, x: 215, y: 850 },
      { id: 'block_table', label: 'Block Table', kind: 'tensor', typeLabel: 'Runtime Mapping', colorKey: 'io:state', width: 210, height: 48, x: 215, y: 960 },
      { id: 'head_group_redistribution', label: 'Head-group Redistribution', kind: 'op', typeLabel: 'Layout Transform', colorKey: 'sem:attention', width: 270, height: 56, x: 650, y: 1330 }
    ].map(item => {
      const registryNode = registry.getNode(item.id);
      const mapping = mappingFor(item.id);
      return {
        ...item,
        parent: 'decode_csa_implementation_scope',
        selectable: true,
        sourceRefs: registryNode.sourceRefs,
        mappingCode: mapping[0],
        mappingReason: mapping[1]
      };
    });
    graph.nodes.push(...extras);
    const shifted = new Set(['attention_alltoall', 'grouped_output_projection', 'output_reduce_scatter', 'attention_hc_post', 'attention_hc_output']);
    graph.nodes.forEach(item => {
      if (shifted.has(item.id)) item.y += 120;
      const registryNode = registry?.getNode(item.id);
        item.parent = registryNode?.parentScope || 'decode_csa_implementation_scope';
        item.entityKind = registryNode?.entityKind || 'Op';
    });
    const semanticIds = graph.nodes
      .filter(item => registry?.getNode(item.id)?.parentScope === 'csa_model_semantic_scope')
      .map(item => item.id);
    const outerIds = graph.nodes.filter(item => !semanticIds.includes(item.id)).map(item => item.id);
    graph.clusters = [
      {
        id: 'decode_csa_implementation_scope', label: 'decode_csa.py implementation scope · HC Pre → RMSNorm → CSA → HC Post',
        x: 44, y: 44, width: 1398, height: 2030, parent: null,
        nodes: outerIds, children: ['csa_model_semantic_scope'], colorKey: 'module:decoder', selectable: false, collapsible: false
      },
      {
        id: 'csa_model_semantic_scope', label: 'CSA model semantic scope · Q/KV Projection → Grouped Output Projection',
        x: 82, y: 410, width: 1322, height: 1220, parent: 'decode_csa_implementation_scope',
        nodes: semanticIds, children: [], colorKey: 'sem:attention', selectable: false, collapsible: false
      }
    ];
    graph.edges = (graph.edges || []).filter(edge => (
      visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target)
      && !(edge.source === 'head_merge' && edge.target === 'attention_alltoall')
    )).map(edge => ({
      ...edge,
      sourceAnchor: 'bottom',
      targetAnchor: 'top',
      curve: 'vertical'
    }));
    graph.edges.push(
      { id: 'e_slot_raw', source: 'slot_mapping', target: 'raw_kv_cache' },
      { id: 'e_slot_cmp', source: 'slot_mapping', target: 'main_compressor' },
      { id: 'e_table_cache', source: 'block_table', target: 'compressed_gather' },
      { id: 'e_table_index', source: 'block_table', target: 'lightning_indexer' },
      { id: 'e_merge_redistribute', source: 'head_merge', target: 'head_group_redistribution' },
      { id: 'e_redistribute_a2a', source: 'head_group_redistribution', target: 'attention_alltoall' }
    );
    graph.edges.forEach(edge => {
      edge.sourceAnchor = 'bottom';
      edge.targetAnchor = 'top';
      edge.curve = 'vertical';
    });
    graph.height = 2120;
    return graph;
  }

  function expandImplementationPrograms(graph, expandedPrograms = new Set()) {
    const programs = registry.implementationPrograms;
    const originals = new Map(graph.nodes.map(item => [item.id, { ...item }]));
    const owned = new Set(Object.values(programs).flatMap(program => program.ownedNodes || []));
    graph.nodes = graph.nodes.filter(item => !owned.has(item.id));
    graph.clusters.forEach(cluster => { cluster.nodes = cluster.nodes.filter(id => !owned.has(id)); });
    // Reserve one band per original row, using the tallest sibling expansion.
    // Expanding a parallel branch must not serialize its sibling programs.
    const bands = new Map();
    Object.entries(programs).forEach(([id, program]) => {
      const parent = originals.get(id);
      if (!parent || !expandedPrograms.has(id)) return;
      bands.set(parent.y, Math.max(bands.get(parent.y) || 0, 60 + program.children.length * 64 - parent.height));
    });
    const shift = y => [...bands].reduce((sum, [row, delta]) => sum + (row < y ? delta : 0), 0);
    graph.nodes.forEach(item => { item.y += shift(item.y); });
    graph.clusters.forEach(cluster => {
      const bottom = cluster.y + cluster.height;
      cluster.height += shift(bottom) - shift(cluster.y);
      cluster.y += shift(cluster.y);
    });
    graph.height += shift(Infinity);
    const additional = Object.values(programs).flatMap(program => program.extraEdges || []);
    additional.forEach(([source, target], index) => graph.edges.push({ id: `program-dependency-${index}`, source, target }));
    const childOwners = new Map(Object.entries(programs).flatMap(([id, program]) => program.children.map(([child]) => [child, id])));
    // Resolve both ends together: an input port is chosen using the canonical
    // producer identity, even if the producer itself has been expanded.
    const endpoints = (id, peer, direction) => {
      const owner = childOwners.get(id);
      if (owner) return [expandedPrograms.has(owner) ? id : owner];
      const program = programs[id];
      if (!program || !expandedPrograms.has(id)) return [id];
      return [].concat(program[direction + 'Ports']?.[peer] || program[direction]);
    };
    graph.edges = graph.edges.flatMap(edge => endpoints(edge.source, edge.target, 'output').flatMap(source =>
      endpoints(edge.target, edge.source, 'input').map(target => ({ ...edge, id: `${edge.id}:${source}:${target}`, source, target }))));
    Object.entries(programs).forEach(([id, program]) => {
      const parent = graph.nodes.find(item => item.id === id);
      if (!parent) return;
      parent.width = id === 'q_projection' ? 310 : 270;
      parent.hideTypeLabel = true;
      parent.collapsed = true;
      if (!expandedPrograms.has(id)) return;
      const top = parent.y - parent.height / 2;
      const height = 60 + program.children.length * 64;
      graph.clusters.forEach(cluster => {
        cluster.nodes = cluster.nodes.filter(key => key !== id);
        if (cluster.id === parent.parent) cluster.children.push(id);
      });
      graph.clusters.push({ id, label: parent.label, x: parent.x - parent.width / 2, y: top,
        width: parent.width, height, nodes: program.children.map(child => child[0]), children: [],
        parent: parent.parent, selectable: true, collapsible: true, colorKey: parent.colorKey });
      graph.nodes = graph.nodes.filter(item => item.id !== id);
      program.children.forEach(([childId], index) => {
        const child = registry.getNode(childId);
        graph.nodes.push({ id: childId, label: child.label, x: parent.x, y: top + 82 + index * 64,
          width: parent.width - 20, height: 44, kind: 'op', hideTypeLabel: true, colorKey: implementationColor(childId, parent.colorKey),
          parent: id, selectable: true, sourceRefs: child.sourceRefs, mappingCode: child.mappingType, mappingReason: child.reason });
      });
      program.edges.forEach(([source, target, semanticEdgeType = 'activation'], index) => {
        const from = graph.nodes.find(item => item.id === source);
        const to = graph.nodes.find(item => item.id === target);
        const bypass = Math.abs(to.y - from.y) > 65;
        const bottom = from.y + from.height / 2;
        const top = to.y - to.height / 2;
        const lane = parent.x - parent.width / 2 + 5;
        graph.edges.push({ id: id + '-internal-' + index, source, target,
          semanticEdgeType, dashed: semanticEdgeType === 'control',
          tensor: { name: semanticEdgeType === 'control' ? 'state commit dependency' : 'value' },
          ...(bypass ? { waypoints: [{ x: from.x, y: bottom + 8 },
            { x: lane, y: bottom + 8 }, { x: lane, y: top - 8 }, { x: to.x, y: top - 8 }] } : {}),
          sourceAnchor: 'bottom', targetAnchor: 'top', curve: 'vertical' });
      });
    });
    const pairs = new Set();
    graph.edges = graph.edges.filter(edge => {
      const key = `${edge.source}>${edge.target}`;
      if (edge.source === edge.target || pairs.has(key)) return false;
      pairs.add(key); return true;
    }).map(edge => ({ ...edge, sourceAnchor: 'bottom', targetAnchor: 'top', curve: 'vertical' }));
    return graph;
  }

  const overviewScopes = ['dv4/layer/2/mhc_attn', 'dv4/layer/2/mhc_ffn'];
  function buildOverviewGraph(expandedIds = new Set(), collapsedIds = new Set()) {
    const folded = new Set(collapsedIds);
    folded.delete('dv4/layer/2');
    overviewScopes.forEach(id => folded.add(id));
    return buildWholeModelGraph(expandedIds, folded);
  }

  async function render(rootInput, options = {}) {
    const root = queryRoot(rootInput);
    if (!root) return null;
    const requestedLevel = ['l1', 'l2', 'l3'].includes(options.level) ? options.level : 'l3';
    const semanticMode = requestedLevel === 'l1' || requestedLevel === 'l2';
    const wholeModel = semanticMode && options.modelGraph === true;
    root.classList.add('pto-dv4-architecture', 'pto-model-graphviz-pattern-page');
    root.dataset.sharedPattern = 'model-architecture-training-sidecar-dv4-working-copy';
    root.innerHTML = shell(requestedLevel);

    const response = await fetch('./json-architecture.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`Architecture graph unavailable: ${response.status}`);
    const sourceGraph = await response.json();
    const level = requestedLevel;
    root.dataset.mappingLevel = level;
    const viewport = root.querySelector('.pto-dv4-architecture__viewport');
    const canvas = root.querySelector('.pto-dv4-architecture__canvas');
    const semanticStatus = root.querySelector('[data-csa-semantic-status]');
    const autoExpandButton = root.querySelector('[data-csa-auto-expand]');
    const state = {
      theme: options.initialTheme === 'light' ? 'light' : 'dark',
      selectedNodeId: null,
      expandedCompoundIds: new Set(level === 'l2' ? registry.canonicalGraph.compoundIds : []),
      autoExpandOnFocus: false,
      zoom: 1,
      panX: 0,
      panY: 0,
      mappingDiff: false,
      diffCategories: ['expanded', 'state', 'deploy'],
      drag: null,
      lineageIds: [],
      increments: false,
      rootFolded: false,
      collapsedModelIds: new Set(),
      expandedPrograms: new Set(registry.defaultExpandedPrograms)
    };
    let graph = normalizeGraph(sourceGraph, level, state.expandedCompoundIds);

    function applyTransform() {
      canvas.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
    }

    function fit({ whole = false } = {}) {
      const rect = viewport.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const reserve = 28;
      const focusId = state.selectedNodeId || (state.lineageIds.length === 1 ? state.lineageIds[0] : null);
      const focus = !whole && (graph.clusters.find(item => item.id === focusId) || graph.nodes.find(item => item.id === focusId));
      const isCluster = focus && graph.clusters.includes(focus);
      const bounds = focus ? { x: focus.x - (isCluster ? 0 : focus.width / 2), y: focus.y - (isCluster ? 0 : focus.height / 2), width: focus.width, height: focus.height }
        : { x: 0, y: 0, width: graph.width, height: graph.height };
      const widthFit = (rect.width - reserve * 2) / bounds.width;
      const heightFit = (rect.height - reserve * 2) / bounds.height;
      state.zoom = Math.min(1.1, widthFit, heightFit);
      state.panX = (rect.width - bounds.width * state.zoom) / 2 - bounds.x * state.zoom;
      state.panY = (rect.height - bounds.height * state.zoom) / 2 - bounds.y * state.zoom;
      applyTransform();
    }

    function semanticState() {
      const total = registry.canonicalGraph.compoundIds.length;
      const expandedCount = state.expandedCompoundIds.size;
      return {
        expandedCount,
        total,
        state: expandedCount === 0 ? 'l1' : expandedCount === total ? 'l2' : 'partial',
        label: expandedCount === 0 ? 'L1 · 模型架构' : expandedCount === total ? 'L2 · 完全展开' : 'L2 · 局部展开'
      };
    }

    function syncSemanticState({ notifyParent = true } = {}) {
      if (!semanticMode) return;
      const status = semanticState();
      if (semanticStatus) semanticStatus.textContent = status.label;
      if (autoExpandButton) {
        autoExpandButton.setAttribute('aria-pressed', String(state.autoExpandOnFocus));
        autoExpandButton.classList.toggle('is-active', state.autoExpandOnFocus);
      }
      if (notifyParent) {
        window.parent.postMessage({
          type: 'csa-semantic-state',
          rootFolded: state.rootFolded,
          ...status,
          autoExpandOnFocus: state.autoExpandOnFocus
        }, '*');
      }
    }

    function renderScene({ notifyParent = true } = {}) {
      graph = wholeModel ? (state.rootFolded ? buildOverviewGraph : buildWholeModelGraph)(state.expandedCompoundIds, state.collapsedModelIds)
        : normalizeGraph(sourceGraph, level, state.expandedCompoundIds);
      if (!semanticMode) graph = expandImplementationPrograms(graph, state.expandedPrograms);
      canvas.style.width = `${graph.width}px`;
      canvas.style.height = `${graph.height}px`;
      canvas.innerHTML = '<div class="pto-dv4-architecture__graph is-front"></div>';
      const host = canvas.querySelector('.pto-dv4-architecture__graph');
      const svg = global.PtoModelGraphvizPattern.render(host, graph, {
        width: graph.width,
        height: graph.height,
        ariaLabel: semanticMode ? 'CSA canonical hierarchical semantic graph' : 'DeepSeek V4 Flash CSA distributed operator implementation',
        metricOverlays: false,
        reportOverlays: false,
        performanceHeatmap: { enabled: false },
        colormap: global.PtoModelGraphvizPattern.modelArchitectureColormap(graph, { theme: state.theme }),
        interaction: { panZoom: false, selectable: true, selectableClusters: true },
        selectable: true,
        selectableClusters: true,
        activeNodeId: state.selectedNodeId,
        onSelect(selection) {
          if (!['graph', 'cluster', 'keyboard'].includes(selection.source)) return;
          state.lineageIds = [];
          state.selectedNodeId = selection.nodeId;
          root.classList.remove('has-model-boundary-focus');
          root.classList.toggle('has-csa-selection', Boolean(selection.nodeId));
          syncSelectionColors();
          const node = graph.nodes.find(item => item.id === selection.nodeId)
            || graph.clusters.find(item => item.id === selection.nodeId);
          window.parent.postMessage({
            type: 'csa-operator-select',
            nodeId: selection.nodeId,
            label: node?.label || selection.nodeId,
            sourceRefs: node?.sourceRefs || []
          }, '*');
        },
        onToggle({ nodeId, collapsed }) {
          if (wholeModel && state.rootFolded) {
            state.rootFolded = false;
            overviewScopes.forEach(id => state.collapsedModelIds.add(id));
          }
          if (wholeModel && !registry.canonicalGraph.compoundIds.includes(nodeId)) {
            const before = graph.nodes.find(item => item.id === nodeId) || graph.clusters.find(item => item.id === nodeId);
            const anchor = before && { x: before.kind === 'module' && graph.nodes.includes(before) ? before.x : before.x + before.width / 2,
              y: graph.nodes.includes(before) ? before.y - before.height / 2 : before.y };
            if (collapsed) state.collapsedModelIds.delete(nodeId);
            else state.collapsedModelIds.add(nodeId);
            renderScene();
            const after = graph.nodes.find(item => item.id === nodeId) || graph.clusters.find(item => item.id === nodeId);
            if (anchor && after) {
              const x = graph.nodes.includes(after) ? after.x : after.x + after.width / 2;
              const y = graph.nodes.includes(after) ? after.y - after.height / 2 : after.y;
              state.panX += (anchor.x - x) * state.zoom;
              state.panY += (anchor.y - y) * state.zoom;
              applyTransform();
            }
            return;
          }
          if (!semanticMode && registry.implementationPrograms[nodeId]) {
            if (collapsed) state.expandedPrograms.add(nodeId);
            else state.expandedPrograms.delete(nodeId);
            renderScene();
            window.parent.postMessage({ type: 'csa-operator-select', nodeId }, '*');
            return;
          }
          if (!semanticMode || !registry.canonicalGraph.compoundIds.includes(nodeId)) return;
          if (collapsed) state.expandedCompoundIds.add(nodeId);
          else state.expandedCompoundIds.delete(nodeId);
          state.selectedNodeId = nodeId;
          renderScene();
          const compound = registry.getNode(nodeId);
          window.parent.postMessage({
            type: 'csa-operator-select',
            nodeId,
            label: compound?.label || nodeId,
            sourceRefs: compound?.sourceRefs || []
          }, '*');
        },
        autoFit: false,
        edgeRouting: 'orthogonal',
        edgeMarkerSize: 6,
        edgeCornerRadius: 12,
        initialTransform: { tx: 0, ty: 0, zoom: 1 }
      });
      state.graphController = svg?.ptoModelGraphvizController || null;
      if (wholeModel) global.PtoDv4ArchitecturePattern.decorateFrontGraph(host, graph);
      // The shared title defaults to pointer-events:none for pan-only diagrams.
      // Here a selectable scope is an explicit navigation anchor (no visual change).
      host.querySelectorAll('[data-cluster-id][role="button"] > .pto-model-graphviz-cluster-label').forEach(label => {
        label.style.pointerEvents = 'all';
        label.style.cursor = 'pointer';
      });
      // Make the whole visible CSA container a selection target. Child nodes and
      // fold buttons keep their own handlers; clicking blank scope space selects CSA.
      host.querySelectorAll('[data-cluster-id="l1_csa"] > rect:first-child').forEach(rect => {
        rect.style.pointerEvents = 'all';
        rect.style.cursor = 'pointer';
      });
      root.classList.toggle('is-mapping-diff', level === 'l3' && state.mappingDiff);
      graph.nodes.forEach(node => {
        const element = root.querySelector(`[data-node-id="${CSS.escape(node.id)}"]`);
        if (!element) return;
        const item = registry.getNode(node.id);
        element.dataset.entityKind = item?.entityKind || node.entityKind || 'Scope';
        if (node.mappingCode) {
          element.dataset.mappingCode = node.mappingCode.toLowerCase();
          element.setAttribute('aria-description', `${item?.transformTypes?.join(' / ') || node.mappingCode}：${node.mappingReason}`);
        }
        const rect = element.querySelector('rect');
        if (item?.entityKind === 'LogicalState' && rect) {
          rect.setAttribute('stroke-dasharray', '6 5');
          rect.style.stroke = 'var(--foreground-muted)';
        }
      });
      applyLineageFocus();
      syncSelectionColors();
      applyIncrementFocus();
      syncDiffColors();
      applyTransform();
      syncSemanticState({ notifyParent });
    }

    function setTheme(theme) {
      state.theme = theme === 'light' ? 'light' : 'dark';
      document.documentElement.dataset.theme = state.theme;
      renderScene({ notifyParent: false });
    }

    function syncDiffColors() {
      if (level !== 'l3') return;
      const counts = { expanded: 0, state: 0, deploy: 0 };
      root.querySelectorAll('.pto-model-graphviz-node[data-node-id]').forEach(element => {
        const diff = registry.diffFor(element.dataset.nodeId);
        element.dataset.diffCategory = diff.category;
        element.toggleAttribute('data-diff-highlight', state.diffCategories.includes(diff.category));
        if (diff.category in counts) counts[diff.category]++;
      });
      window.parent.postMessage({ type: 'csa-diff-counts', counts }, '*');
    }

    function setMappingDiff(enabled, categories = state.diffCategories) {
      state.mappingDiff = level === 'l3' && Boolean(enabled);
      state.diffCategories = Array.isArray(categories) ? categories.filter(value => ['expanded', 'state', 'deploy'].includes(value)) : state.diffCategories;
      root.classList.toggle('is-mapping-diff', state.mappingDiff);
      syncDiffColors();
    }

    function clearSelection({ notifyParent = true } = {}) {
      const hadFocus = Boolean(state.selectedNodeId || state.lineageIds.length || root.classList.contains('has-model-boundary-focus'));
      if (!hadFocus) return;
      state.selectedNodeId = null;
      state.lineageIds = [];
      root.querySelectorAll('[data-lineage-hit]').forEach(element => delete element.dataset.lineageHit);
      state.graphController?.clearSelection();
      root.classList.remove('has-csa-selection', 'has-model-boundary-focus');
      if (notifyParent) {
        window.parent.postMessage({ type: 'csa-operator-clear' }, '*');
      }
    }

    function focusModelBoundary() {
      clearSelection({ notifyParent: false });
      root.classList.add('has-model-boundary-focus');
    }

    function setAllCompounds(expanded) {
      if (!semanticMode) return;
      state.rootFolded = false;
      if (wholeModel) {
        ['dv4/layer/2', 'dv4/layer/2/mhc_attn', 'dv4/layer/2/mhc_attn/hybrid_attention', 'l1_csa'].forEach(id => state.collapsedModelIds.delete(id));
      }
      state.expandedCompoundIds = new Set(expanded ? registry.canonicalGraph.compoundIds : []);
      const selectedCompound = registry.compoundFor(state.selectedNodeId);
      if (!expanded && selectedCompound) state.selectedNodeId = selectedCompound.id;
      renderScene();
    }

    function focusCompound(compoundId, { autoExpand = state.autoExpandOnFocus } = {}) {
      if (!semanticMode || !registry.canonicalGraph.compoundIds.includes(compoundId)) return;
      state.autoExpandOnFocus = Boolean(autoExpand);
      if (autoExpand) state.expandedCompoundIds.add(compoundId);
      state.selectedNodeId = compoundId;
      renderScene();
      state.graphController?.selectNode(compoundId, { source: 'model' });
    }

    function syncSelectionColors() {
      const keep = new Set(state.lineageIds.length ? state.lineageIds : [state.selectedNodeId].filter(Boolean));
      if (state.lineageIds.length) root.querySelectorAll('.is-model-selected').forEach(el => keep.add(el.dataset.nodeId || el.dataset.clusterId));
      const visit = id => {
        const scope = graph.clusters.find(item => item.id === id);
        [...(scope?.nodes || []), ...(scope?.children || [])].forEach(child => {
          if (!keep.has(child)) { keep.add(child); visit(child); }
        });
      };
      [...keep].forEach(visit);
      root.querySelectorAll('[data-node-id]').forEach(el => {
        el.toggleAttribute('data-selection-keep', keep.has(el.dataset.nodeId));
      });
    }

    function applyLineageFocus() {
      if (!state.lineageIds.length) return;
      state.graphController?.clearSelection();
      const visible = new Map();
      state.lineageIds.forEach(id => {
        const compound = registry.compoundFor(id);
        const programParent = registry.getNode(id)?.parentId;
        let key = semanticMode && compound && !state.expandedCompoundIds.has(compound.id) ? compound.id
          : !semanticMode && registry.implementationPrograms[programParent] && !state.expandedPrograms.has(programParent) ? programParent : id;
        if (wholeModel && ![...graph.nodes, ...graph.clusters].some(item => item.id === key)) {
          key = ['dv4/layer/2', 'dv4/layer/2/mhc_attn', 'dv4/layer/2/mhc_attn/hybrid_attention', 'l1_csa']
            .find(ancestor => graph.nodes.some(node => node.id === ancestor)) || key;
        }
        visible.set(key, (visible.get(key) || 0) + 1);
      });
      root.querySelectorAll('[data-node-id], [data-cluster-id]').forEach(element => {
        const id = element.dataset.nodeId || element.dataset.clusterId;
        element.classList.toggle('is-model-selected', visible.has(id));
        if (visible.has(id)) {
          element.dataset.lineageHit = String(visible.get(id));
          element.setAttribute('aria-description', `命中 ${visible.get(id)} 个源实体；展开可查看，选择不改变另一栏层级。`);
        } else delete element.dataset.lineageHit;
      });
      root.classList.toggle('has-csa-selection', visible.size > 0);
      syncSelectionColors();
    }

    function focusLineage(nodeIds, { autoExpand = false } = {}) {
      state.lineageIds = [...new Set(nodeIds.filter(id => registry.getNode(id)))];
      state.selectedNodeId = null;
      if (semanticMode && autoExpand && !state.rootFolded) state.lineageIds.forEach(id => {
        const parent = registry.compoundFor(id);
        if (parent) state.expandedCompoundIds.add(parent.id);
        if (wholeModel) ['dv4/layer/2', 'dv4/layer/2/mhc_attn', 'dv4/layer/2/mhc_attn/hybrid_attention', 'l1_csa'].forEach(key => state.collapsedModelIds.delete(key));
      });
      renderScene();
    }

    function applyIncrementFocus() {
      graph.nodes.forEach(node => {
        const item = registry.getNode(node.id);
        const incremental = item?.transformTypes?.some(type => ['MATERIALIZED', 'DEPLOYED', 'SCOPE_INCLUDED'].includes(type));
        const element = root.querySelector(`[data-node-id="${CSS.escape(node.id)}"]`);
        if (element) element.style.opacity = state.increments && !incremental ? '.32' : '';
      });
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
        || event.target.closest('.pto-model-graphviz-node, .pto-model-graphviz-cluster')
      ) return;
      state.drag = { x: event.clientX, y: event.clientY, panX: state.panX, panY: state.panY, moved: false };
      viewport.setPointerCapture(event.pointerId);
      viewport.classList.add('is-dragging');
    });
    viewport.addEventListener('pointermove', event => {
      if (!state.drag) return;
      const deltaX = event.clientX - state.drag.x;
      const deltaY = event.clientY - state.drag.y;
      if (!state.drag.moved && Math.hypot(deltaX, deltaY) < 4) return;
      state.drag.moved = true;
      state.panX = state.drag.panX + deltaX;
      state.panY = state.drag.panY + deltaY;
      applyTransform();
    });
    const endDrag = event => {
      if (!state.drag) return;
      const shouldClearSelection = event.type === 'pointerup' && !state.drag.moved;
      state.drag = null;
      viewport.classList.remove('is-dragging');
      if (viewport.hasPointerCapture(event.pointerId)) viewport.releasePointerCapture(event.pointerId);
      if (shouldClearSelection) clearSelection();
    };
    viewport.addEventListener('pointerup', endDrag);
    viewport.addEventListener('pointercancel', endDrag);
    document.addEventListener('keydown', event => {
      if (event.key !== 'Escape') return;
      clearSelection();
    });

    root.querySelector('[data-dv4-fit]').title = '适配所选节点或范围；未选择时适配整图。Shift + 点击适配整图。';
    root.querySelector('[data-dv4-fit]').addEventListener('click', event => fit({ whole: event.shiftKey }));
    root.querySelector('[data-csa-collapse-all]')?.addEventListener('click', () => setAllCompounds(false));
    root.querySelector('[data-csa-expand-all]')?.addEventListener('click', () => setAllCompounds(true));
    autoExpandButton?.addEventListener('click', () => {
      state.autoExpandOnFocus = !state.autoExpandOnFocus;
      syncSemanticState();
    });

    const resizeObserver = new ResizeObserver(fit);
    resizeObserver.observe(viewport);
    document.documentElement.dataset.theme = state.theme;
    renderScene();
    requestAnimationFrame(fit);

    return {
      root,
      get graph() { return graph; },
      fit,
      setTheme,
      setMappingDiff,
      clearSelection,
      focusModelBoundary,
      setAllCompounds,
      focusCompound,
      focusLineage,
      setIncrements(enabled) { state.increments = Boolean(enabled); applyIncrementFocus(); },
      setAllPrograms(expanded) {
        if (semanticMode) return;
        state.expandedPrograms = new Set(expanded ? registry.defaultExpandedPrograms : []);
        renderScene(); fit({ whole: true });
      },
      setRootFolded(folded) {
        if (state.rootFolded === Boolean(folded)) { syncSemanticState(); return; }
        state.rootFolded = Boolean(folded);
        renderScene(); fit({ whole: true });
      },
      selectNode(nodeId) {
        const selectable = graph.nodes.some(node => node.id === nodeId && node.selectable !== false)
          || graph.clusters.some(cluster => cluster.id === nodeId && cluster.selectable !== false);
        if (!selectable) return;
        state.selectedNodeId = nodeId;
        root.classList.remove('has-model-boundary-focus');
        state.graphController?.selectNode(nodeId, { source: 'source' });
      },
      getState: () => ({
        ...state,
        expandedCompoundIds: [...state.expandedCompoundIds],
        collapsedModelIds: [...state.collapsedModelIds],
        drag: undefined,
        graphController: undefined
      }),
      destroy() {
        resizeObserver.disconnect();
        viewport.removeEventListener('wheel', onWheel);
        root.innerHTML = '';
      }
    };
  }

  global.PtoCsaImplementationPattern = Object.freeze({ render, buildGraph: normalizeGraph, buildWholeModelGraph, buildOverviewGraph, expandImplementationPrograms });
})(window);
