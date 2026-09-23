import fs from 'node:fs';

const canonical = JSON.parse(fs.readFileSync(new URL('./model_architecture.json', import.meta.url), 'utf8'));
const projected = JSON.parse(fs.readFileSync(new URL('./model_architecture_graph.json', import.meta.url), 'utf8'));
const sourceById = new Map(canonical.sources.map(source => [source.id, source]));
const sourceNodeById = new Map(canonical.nodes.map(node => [node.id, node]));
const projectedNodeById = new Map(projected.nodes.map(node => [node.id, node]));
const groupById = new Map(canonical.visual_layout.groups.map(group => [group.id, group]));

function sourceRefs(item) {
  return (item.provenance || []).map(reference => ({
    source_id: reference.source,
    path: sourceById.get(reference.source)?.path || reference.source,
    line_start: reference.line,
    line_end: reference.line,
    symbol: item.op_type || item.module_type || item.label,
    evidence: sourceById.get(reference.source)?.kind || 'source',
    fact: reference.fact
  }));
}

function nodeItem(id) {
  const source = sourceNodeById.get(id);
  const layout = projectedNodeById.get(id);
  if (!source || !layout) throw new Error(`Cannot project node ${id}`);
  return {
    id,
    label: source.label,
    kind: layout.kind,
    typeLabel: layout.typeLabel,
    colorKey: layout.colorKey,
    origin: 'implementation',
    dataState: 'implementation_only',
    attrs: source.attrs || {},
    sourceRefs: sourceRefs(source),
    children: []
  };
}

function groupItem(id, children) {
  const group = groupById.get(id);
  return {
    id,
    label: group.label,
    kind: 'module',
    typeLabel: 'Module',
    colorKey: group.colorKey,
    synthetic: true,
    children
  };
}

const csaGroup = groupById.get('csa_operator_group');
const moeGroup = groupById.get('moe_context_group');
const decoderRoot = groupItem('decoder_layer_group', [
  nodeItem('hc_input'),
  groupItem('csa_operator_group', csaGroup.nodes.map(nodeItem)),
  nodeItem('attention_hc_output'),
  groupItem('moe_context_group', moeGroup.nodes.map(nodeItem)),
  nodeItem('layer_output')
]);

const communicationNodes = new Set(['tp_allgather', 'attention_alltoall', 'output_reduce_scatter']);
const stateNodes = new Set(canonical.nodes.filter(node => node.kind === 'state').map(node => node.id));
const positionById = new Map(projected.nodes.map(node => [node.id, node]));

function semanticType(edge) {
  if (edge.dashed) return 'residual';
  if (stateNodes.has(edge.source) || stateNodes.has(edge.target)) return 'state';
  if (communicationNodes.has(edge.source) || communicationNodes.has(edge.target)) return 'communication';
  return 'activation';
}

function portPolicy(edge) {
  const source = positionById.get(edge.source);
  const target = positionById.get(edge.target);
  if (source && target && Math.abs(source.y - target.y) < 1 && source.x !== target.x) {
    const leftToRight = source.x < target.x;
    return {
      sourceAnchor: leftToRight ? 'right' : 'left',
      targetAnchor: leftToRight ? 'left' : 'right',
      curve: 'horizontal'
    };
  }
  return { sourceAnchor: 'bottom', targetAnchor: 'top', curve: 'vertical' };
}

const output = {
  schema_version: 'model_architecture_graph.v1',
  metadata: {
    modelId: 'DeepSeek-V4-Flash-CSA-PyPTO',
    sourceScope: 'distributed_operator_implementation',
    profilingOverlay: false,
    graphRole: 'stage_2_operator_implementation',
    parentGraph: 'DeepSeek V4 source-model architecture / CSA module'
  },
  roots: [{
    id: 'section/distributed_operator_implementation',
    label: 'Stage 2 · CSA Distributed Operator Implementation',
    kind: 'section',
    typeLabel: 'Implementation graph',
    synthetic: true,
    children: [decoderRoot]
  }],
  edges: canonical.edges.map(edge => ({
    ...edge,
    semanticEdgeType: semanticType(edge),
    ...portPolicy(edge)
  })),
  width: projected.width,
  height: projected.height,
  model: projected.model,
  extraction_scope: projected.extraction_scope,
  nodes: projected.nodes,
  clusters: projected.clusters
};

fs.writeFileSync(new URL('./json-architecture.json', import.meta.url), `${JSON.stringify(output, null, 2)}\n`);
