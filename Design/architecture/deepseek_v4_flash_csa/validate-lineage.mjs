import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const directory = path.dirname(fileURLToPath(import.meta.url));
const context = vm.createContext({ window: {}, structuredClone });
const manifestFile = path.join(directory, 'source-manifest.js');
if (fs.existsSync(manifestFile)) vm.runInContext(fs.readFileSync(manifestFile, 'utf8'), context);
vm.runInContext(fs.readFileSync(path.join(directory, 'implementation-programs.js'), 'utf8'), context);
vm.runInContext(fs.readFileSync(path.join(directory, 'mapping-registry.js'), 'utf8'), context);
const registry = context.window.PtoCsaMappingRegistry;

if (process.argv.includes('--manifest')) {
  const manifest = {};
  const files = new Set(registry.nodes.flatMap(node => node.sourceRefs.map(ref => ref.file)));
  files.add(registry.context.config);
  for (const file of [...files].sort()) {
    assert(!file.startsWith('http'), 'Evidence must use a locked local snapshot');
    const content = fs.readFileSync(path.resolve(directory, file), 'utf8');
    let owner = '';
    const symbols = [];
    content.split('\n').forEach((line, index) => {
      const klass = line.match(/^class (\w+)/);
      if (klass) { owner = klass[1]; symbols.push({ line: index + 1, name: owner }); }
      const func = line.match(/^(\s*)def (\w+)/);
      if (func) {
        if (!func[1]) owner = '';
        symbols.push({ line: index + 1, name: owner ? owner + '.' + func[2] : func[2] });
      }
    });
    manifest[file] = { hash: crypto.createHash('sha256').update(content).digest('hex'), lines: content.split('\n').length, symbols };
  }
  process.stdout.write('window.PtoCsaSourceManifest = Object.freeze(' + JSON.stringify(manifest, null, 2) + ');\n');
  process.exit(0);
}

const ids = new Set(registry.nodes.map(node => node.id));
assert.equal(ids.size, registry.nodes.length, 'unique canonical identities');
for (const node of registry.nodes) {
  for (const field of ['id', 'label', 'entityKind', 'origin', 'sourceRefs', 'children', 'artifactStatus']) assert(field in node, node.id + ': ' + field);
  node.children.forEach(child => assert(ids.has(child), 'unresolved child ' + child));
  node.sourceRefs.forEach(ref => {
    const data = fs.readFileSync(path.resolve(directory, ref.file), 'utf8');
    assert(ref.start >= 1 && ref.end >= ref.start && ref.end <= data.split('\n').length, node.id + ': source range');
    assert.equal(ref.hash, crypto.createHash('sha256').update(data).digest('hex'), node.id + ': stale snapshot; audit before regenerating manifest');
    assert(ref.symbol, node.id + ': source symbol');
  });
}
for (const relation of registry.lineage) {
  assert(relation.sourceIds.length && relation.targetIds.length);
  [...relation.sourceIds, ...relation.targetIds].forEach(id => assert(ids.has(id), 'unresolved relation ' + id));
  assert(relation.evidenceRefs.length);
  assert(relation.explanation.rationale && relation.explanation.limitations);
  assert(!relation.transformTypes.includes('FUSED'), 'function grouping is not proven fusion');
}
assert.equal(registry.getNode('csa_indexer_q_rope').mapsTo.join(','), 'lightning_indexer');
assert(!registry.getNode('csa_recent_window_positions'), 'no duplicate window indices');
assert.equal(registry.getNode('slot_mapping').entityKind, 'RuntimeMetadata');
assert.equal(registry.getNode('hc_input').entityKind, 'Value');
assert.equal(registry.getNode('q_projection').sourceRefs.find(ref => ref.role === 'definition').symbol, 'q_proj_rope');
assert.equal(registry.getNode('kv_projection').sourceRefs.find(ref => ref.role === 'definition').symbol, 'kv_proj_rope');
assert(registry.mappedIds('csa_sparse_attention_op', 'L3').includes('head_merge'));
assert(registry.mappedIds('q_projection', 'L2').length >= 3);
assert(registry.mappedIds('l1_csa', 'L3').length >= 20);
assert.equal(registry.compilation.artifactStatus, 'unavailable');
assert(registry.nodesAt('L4').every(node => node.artifactStatus === 'unavailable'));
const entry = registry.implementationEntries.l1_csa;
assert.equal(entry.sourceScopeId, 'l1_csa');
assert.equal(registry.getNode(entry.sourceNodeId).origin, 'developer-pypto');
assert.equal(entry.targetScopeId, 'decode_csa_implementation_scope');
entry.relationIds.forEach(id => assert(registry.lineage.some(relation => relation.id === id)));

vm.runInContext(fs.readFileSync(path.join(directory, 'json-architecture.js'), 'utf8'), context);
const build = context.window.PtoCsaImplementationPattern.buildGraph;
const payload = JSON.parse(fs.readFileSync(path.join(directory, 'json-architecture.json'), 'utf8'));
const canonical = registry.canonicalGraph;
const summary = new Set(canonical.summaryEdges.map(edge => edge.source + '>' + edge.target));
const folded = build(payload, 'l1', new Set());
for (let mask = 0; mask < 64; mask++) {
  const expanded = new Set(canonical.compoundIds.filter((id, index) => mask & (1 << index)));
  const graph = build(payload, 'l2', expanded);
  const visible = new Set([...graph.nodes, ...graph.clusters].map(node => node.id));
  const pairs = new Set();
  const summaries = new Set();
  for (const edge of graph.edges) {
    assert(visible.has(edge.source) && visible.has(edge.target), 'no dangling projected edge');
    const key = edge.source + '>' + edge.target;
    assert(!pairs.has(key), 'no duplicate projected edge ' + key); pairs.add(key);
    const source = registry.compoundFor(edge.source)?.id || edge.source;
    const target = registry.compoundFor(edge.target)?.id || edge.target;
    if (source !== target) summaries.add(source + '>' + target);
  }
  assert.deepEqual([...summaries].sort(), [...summary].sort(), 'collapse(L2) = L1 at mask ' + mask);
  for (const node of graph.nodes) {
    const parent = graph.clusters.find(cluster => cluster.id === node.parent);
    if (!parent) continue;
    assert(node.x - node.width / 2 >= parent.x && node.x + node.width / 2 <= parent.x + parent.width, 'horizontal containment ' + node.id);
    assert(node.y - node.height / 2 >= parent.y && node.y + node.height / 2 <= parent.y + parent.height, 'vertical containment ' + node.id);
  }
}
assert.deepEqual(JSON.stringify(build(payload, 'l1', new Set())), JSON.stringify(folded));
for (const expanded of [[], ['q_projection'], ['kv_projection'], ['q_projection', 'kv_projection']]) {
  const graph = context.window.PtoCsaImplementationPattern.expandImplementationPrograms(build(payload, 'l3', new Set()), new Set(expanded));
  const visible = new Set(graph.nodes.map(node => node.id));
  graph.edges.forEach(edge => assert(visible.has(edge.source) && visible.has(edge.target), 'program port resolves'));
  if (expanded.includes('q_projection')) assert(graph.edges.some(edge => edge.source === 'pypto_q_norm_quant' && edge.target === 'lightning_indexer'), 'Indexer consumes qr, not main Q RoPE');
  for (const id of expanded) {
    assert(!visible.has(id), 'expanded function is a scope, not duplicate op');
    const cluster = graph.clusters.find(item => item.id === id);
    for (const childId of cluster.nodes) {
      const child = graph.nodes.find(node => node.id === childId);
      assert(child.y - child.height / 2 >= cluster.y && child.y + child.height / 2 <= cluster.y + cluster.height, 'program child containment');
    }
  }
}
const shared = path.resolve(directory, '../../../vendor/pto-design-system/patterns/model-architecture-training-sidecar');
for (const name of ['pattern.dv4-architecture-data.js', 'pattern.dv4.js']) vm.runInContext(fs.readFileSync(path.join(shared, name), 'utf8'), context);
const whole = context.window.PtoCsaImplementationPattern.buildWholeModelGraph;
const ancestors = ['l1_csa', 'dv4/layer/2/mhc_attn/hybrid_attention', 'dv4/layer/2/mhc_attn', 'dv4/layer/2'];
const base = whole();
const baseIds = new Set([...base.nodes, ...base.clusters].map(node => node.id));
for (let mask = 0; mask < 64; mask++) {
  const expanded = new Set(canonical.compoundIds.filter((id, index) => mask & (1 << index)));
  for (const ancestor of [null, ...ancestors]) {
    const graph = whole(expanded, new Set(ancestor ? [ancestor] : []));
    const visible = new Set([...graph.nodes, ...graph.clusters].map(item => item.id));
    assert.equal(visible.size, graph.nodes.length + graph.clusters.length, 'whole graph unique IDs');
    const pairs = new Set();
    graph.edges.forEach(edge => {
      assert(visible.has(edge.source) && visible.has(edge.target), `whole graph dangling edge ${edge.source} -> ${edge.target}`);
      const key = edge.source + '>' + edge.target;
      assert(!pairs.has(key), 'whole graph duplicate ' + key); pairs.add(key);
    });
    if (ancestor) {
      assert(graph.nodes.some(node => node.id === ancestor && node.collapsed), 'ancestor placeholder');
      if (ancestor === 'l1_csa') assert(graph.nodes.find(node => node.id === ancestor).selectable, 'folded CSA entry remains selectable');
      assert(!visible.has('csa_q_lora_down'), 'ancestor hides descendants');
    } else {
      baseIds.forEach(id => assert(visible.has(id), 'original architecture identity retained ' + id));
      const csa = graph.clusters.find(node => node.id === 'l1_csa');
      assert.equal(csa.parent, 'dv4/layer/2/mhc_attn/hybrid_attention');
      for (const node of graph.nodes.filter(node => registry.compoundFor(node.id))) {
        const parent = graph.clusters.find(cluster => cluster.id === node.parent);
        assert(parent, 'CSA parent retained');
        assert(node.x - node.width / 2 >= parent.x && node.x + node.width / 2 <= parent.x + parent.width, 'whole horizontal containment');
        assert(node.y - node.height / 2 >= parent.y && node.y + node.height / 2 <= parent.y + parent.height, 'whole vertical containment');
      }
    }
  }
}
assert.equal(JSON.stringify(whole()), JSON.stringify(base), 'whole graph reversible geometry');
const detailIds = Object.keys(registry.implementationPrograms);
const defaultGraph = context.window.PtoCsaImplementationPattern.expandImplementationPrograms(build(payload, 'l3', new Set()), new Set(detailIds));
const defaultIds = new Set(defaultGraph.nodes.map(item => item.id));
for (const id of canonical.leafIds) {
  assert(registry.mappedIds(id, 'L3').some(target => defaultIds.has(target)), 'L2 baseline must have a visible L3 implementation: ' + id);
}
for (let mask = 0; mask < 2 ** detailIds.length; mask++) {
  const expanded = new Set(detailIds.filter((id, index) => mask & (1 << index)));
  const graph = context.window.PtoCsaImplementationPattern.expandImplementationPrograms(build(payload, 'l3', new Set()), expanded);
  const visible = new Set(graph.nodes.map(item => item.id));
  assert.equal(visible.size, graph.nodes.length, 'shared implementation is never duplicated');
  assert.equal(new Set(graph.edges.map(edge => edge.id)).size, graph.edges.length, 'unique edge IDs after multi-port expansion');
  const pairs = new Set();
  graph.edges.forEach(edge => {
    assert(visible.has(edge.source) && visible.has(edge.target), 'expanded port resolves: ' + edge.source + '>' + edge.target);
    const pair = edge.source + '>' + edge.target;
    assert(!pairs.has(pair) && edge.source !== edge.target, 'no duplicate/self implementation edges');
    pairs.add(pair);
  });
  for (const id of expanded) {
    const parent = graph.clusters.find(item => item.id === id);
    assert(parent && !visible.has(id), 'expanded program is a container');
    for (const childId of parent.nodes) {
      const child = graph.nodes.find(item => item.id === childId);
      assert(child && child.parent === id, 'program canonical ownership');
      assert(child.y - child.height / 2 >= parent.y && child.y + child.height / 2 <= parent.y + parent.height, 'detail containment ' + childId);
    }
  }
}
const overview = context.window.PtoCsaImplementationPattern.buildOverviewGraph(new Set(canonical.compoundIds));
for (const id of ['dv4/layer/2/mhc_attn', 'dv4/layer/2/mhc_ffn']) assert(overview.nodes.some(item => item.id === id && item.collapsed), 'L0 folded sublayer');
for (const suffix of ['/input_tokens', '/token_embedding', '/previous_decoder_layers', '/remaining_decoder_layers', '/final_rmsnorm', '/lm_head', '/output_tokens']) {
  assert(overview.nodes.some(item => item.id.endsWith(suffix)), 'L0 preserves model trunk ' + suffix);
}
assert(!overview.nodes.some(item => item.id === 'official_model_root'), 'L0 is not a synthetic single root');
console.log('PASS: hashes, 64 semantic states, 320 whole-model states, ' + 2 ** detailIds.length + ' program states, all 26 L2 anchors visible in default L3, L0 trunk and L4 guard.');

for (const id of ['pypto_q_down', 'pypto_kv_matmul', 'merge_inverse_rope']) {
  assert.equal(registry.getNode(id).mappingType, 'SAME', `${id} retains its source operation`);
}
assert.equal(registry.getNode('pypto_q_norm_quant').mappingType, 'EXPANDED');

// L2-relative Diff is separate from transformation tags and independent of folding.
for (const [id, category] of Object.entries({
  pypto_q_down: 'retained', merge_inverse_rope: 'retained', output_projection: 'retained',
  pypto_q_norm_quant: 'expanded', topk_leaf: 'expanded', sparse_qk: 'expanded',
  slot_mapping: 'state', cmp_commit: 'state', window_gather: 'state',
  tp_allgather: 'deploy', head_group_redistribution: 'deploy', lightning_indexer: 'retained'
})) assert.equal(registry.diffFor(id).category, category, `L2 Diff classification: ${id}`);
for (const item of registry.nodesAt('L3')) {
  const diff = registry.diffFor(item.id);
  for (const id of diff.sourceIds) assert.equal(registry.getNode(id)?.level, 'L2');
  if (diff.category === 'expanded') assert(diff.sourceIds.length, `${item.id} must have L2 provenance`);
}
console.log('PASS: L2-relative Diff categories and provenance.');
