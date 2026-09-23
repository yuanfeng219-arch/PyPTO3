import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const assetDir = path.join(root, 'patterns/model-graphviz/assets');
const reference = 'openpangu_2_0_flash_pass_ir_capsule_modelviz.html';
const modelvizPages = fs.readdirSync(assetDir)
  .filter((name) => name.endsWith('_modelviz.html'))
  .sort();
const migratedPages = modelvizPages.filter((name) => name !== reference);
const failures = [];

function assert(condition, message) {
  if (!condition) failures.push(message);
}

function inlineScripts(html) {
  return [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)]
    .filter((match) => !/\bsrc\s*=/.test(match[1]) && !/application\/json/.test(match[1]))
    .map((match) => match[2]);
}

function localResources(file, html) {
  return [...html.matchAll(/<(?:script|link)\b[^>]*(?:src|href)=["']([^"']+)["']/gi)]
    .map((match) => match[1].split(/[?#]/)[0])
    .filter((value) => value && !/^(?:[a-z]+:|\/\/|#)/i.test(value))
    .map((value) => path.resolve(path.dirname(file), value));
}

for (const name of migratedPages) {
  const file = path.join(assetDir, name);
  const html = fs.readFileSync(file, 'utf8');
  assert(html.includes('pass-ir-graph-node/pattern.css'), `${name}: missing Pass-IR capsule CSS`);
  assert(html.includes('pass-ir-graph-node/pattern.js'), `${name}: missing Pass-IR capsule renderer`);
  assert(html.includes('capsule.css'), `${name}: missing shared model capsule CSS`);
  assert(html.includes('capsule.js'), `${name}: missing shared model capsule adapter`);
  localResources(file, html).forEach((resource) => {
    assert(fs.existsSync(resource), `${name}: missing local resource ${path.relative(root, resource)}`);
  });
  inlineScripts(html).forEach((source, index) => {
    try {
      new vm.Script(source, { filename: `${name}#inline-${index + 1}` });
    } catch (error) {
      failures.push(`${name}: inline script ${index + 1} does not parse: ${error.message}`);
    }
  });
}

const architectureFiles = fs.readdirSync(assetDir)
  .filter((name) => name.endsWith('_model_architecture.json'))
  .sort();
for (const name of architectureFiles) {
  const schema = JSON.parse(fs.readFileSync(path.join(assetDir, name), 'utf8'));
  const nodeById = new Map((schema.nodes || []).map((node) => [String(node.id), node]));
  const layoutNodes = schema.visual_layout?.nodes || {};
  (schema.nodes || []).forEach((node) => {
    const looksLikeWeightState = /weight/i.test(`${node.id || ''} ${node.label || ''}`) &&
      String(node.kind || '').toLowerCase() === 'state';
    assert(
      !looksLikeWeightState || String(node.state_type || '').toLowerCase() === 'parameter',
      `${name}: Weight node ${node.id || node.label} must use kind=state with state_type=parameter`
    );
    const isAuxiliaryState = String(node.kind || '').toLowerCase() === 'state';
    if (!isAuxiliaryState) return;
    const outgoing = (schema.edges || []).filter((edge) => String(edge.source) === String(node.id));
    if (looksLikeWeightState) {
      assert(outgoing.length > 0, `${name}: Weight node ${node.id} must feed a consumer`);
    }
    outgoing.forEach((edge) => {
      const target = nodeById.get(String(edge.target));
      const sourcePoint = layoutNodes[node.id];
      const targetPoint = layoutNodes[edge.target];
      assert(Boolean(target), `${name}: Weight edge ${edge.id || node.id} has a missing target`);
      if (sourcePoint && targetPoint) {
        assert(
          Number(sourcePoint.y) === Number(targetPoint.y),
          `${name}: State/Weight node ${node.id} must share a horizontal rank with ${edge.target}`
        );
        const processingRow = (schema.nodes || []).filter((candidate) => {
          const kind = String(candidate.kind || '').toLowerCase();
          const point = layoutNodes[candidate.id];
          return (kind === 'module' || kind === 'op') &&
            point && Number(point.y) === Number(targetPoint.y);
        });
        if (processingRow.length > 1) {
          const processingXs = processingRow.map((candidate) => Number(layoutNodes[candidate.id].x));
          const sourceX = Number(sourcePoint.x);
          assert(
            sourceX < Math.min(...processingXs) || sourceX > Math.max(...processingXs),
            `${name}: State/Weight node ${node.id} must sit outside its complete processing row`
          );
        }
      }
    });
  });
}

const capsuleJs = fs.readFileSync(path.join(root, 'patterns/model-graphviz/capsule.js'), 'utf8');
const capsuleCss = fs.readFileSync(path.join(root, 'patterns/model-graphviz/capsule.css'), 'utf8');
assert(capsuleJs.includes('PtoModelGraphvizCapsule'), 'capsule.js: missing public API');
assert(capsuleJs.includes("new URL('../../', capsuleScriptUrl).href"), 'capsule.js: must derive the shared fx.svg asset root');
assert(
  capsuleJs.includes('PtoPassIrGraphNodePattern') && capsuleJs.includes('helper.buildNodeCardElement'),
  'capsule.js: must render through Pass-IR node pattern'
);
assert(
  capsuleJs.includes('normalizeLegacyEdges') && capsuleJs.includes('ptoCapsuleBezier'),
  'capsule.js: must normalize legacy DeepSeek edges to cubic Bézier paths'
);
assert(
  capsuleJs.includes('C${spline[1].x},${spline[1].y} ${spline[2].x},${spline[2].y}'),
  'capsule.js: must preserve renderer-provided edge tangents'
);
assert(
  capsuleJs.includes('OPENPANGU_CAPSULE_FRAME') && capsuleJs.includes('legacyWidth: 365'),
  'capsule.js: must share the openPangu capsule width baseline'
);
assert(capsuleCss.includes('.opv-pass-ir-capsule-host .op-pill'), 'capsule.css: missing capsule pill contract');
assert(capsuleCss.includes('font-size: 16px !important'), 'capsule.css: operator names must use one fixed size');

const deepseekSource = fs.readFileSync(path.join(assetDir, 'deepseek_v32_modelviz.html'), 'utf8');
const deepseekSourceGraph = fs.readFileSync(path.join(root, 'patterns/model-graphviz/graphviz/deepseek_v32_source_graph.html'), 'utf8');
assert(
  deepseekSource.includes('x: sourceNode.position.x + (sourceOnLeft ? sourceNode.width / 2 : -sourceNode.width / 2)') &&
    deepseekSource.includes('x: targetNode.position.x + (sourceOnLeft ? -targetNode.width / 2 : targetNode.width / 2)'),
  'deepseek_v32_modelviz.html: side edges must connect auxiliary and consumer side boundaries'
);
assert(
  deepseekSource.includes('is_side_edge: Boolean(isSideEdge)') && deepseekSource.includes('const sideGap = 84'),
  'deepseek_v32_modelviz.html: missing the side-lane edge contract'
);
assert(
  deepseekSource.includes('const visibleSchemaRanks =') &&
    deepseekSource.includes('index * 124') &&
    deepseekSource.includes('const targetBelow ='),
  'deepseek_v32_modelviz.html: drill-down ranks and post-layout edge routing must stay schema-driven'
);
assert(
  deepseekSource.includes('const routedExpertNode = result.nodes.expert_mlp') &&
    deepseekSource.includes('const sharedExpertNode = result.nodes.shared_expert') &&
    !deepseekSource.includes('nodeAtSchemaPoint'),
  'deepseek_v32_modelviz.html: MoE sub-spine must use semantic node IDs instead of brittle coordinates'
);
assert(
  deepseekSource.includes('Math.abs(sourceNode.position.y - targetNode.position.y) < 1'),
  'deepseek_v32_modelviz.html: edge ports must be derived from final rendered geometry'
);
assert(
  deepseekSource.includes('const parallelPeers =') &&
    deepseekSource.includes('const outwardSide =') &&
    deepseekSource.includes('const hasMixedAuxiliaries =') &&
    deepseekSource.includes('const processingRow = [targetNode, ...parallelPeers]') &&
    deepseekSource.includes('const anchorNode = hasMixedAuxiliaries'),
  'deepseek_v32_modelviz.html: all same-rank branch auxiliaries must stay on the outer flanks'
);
assert(
  deepseekSource.includes('schemaNodeType(node.kind, node.state_type)') &&
    deepseekSource.includes('normalizedStateType === "parameter"') &&
    !deepseekSource.includes('kind: "parameter"'),
  'deepseek_v32_modelviz.html: parameter states must map to the Parameter visual type'
);
[
  'const routedExpertNode = result.nodes.expert_mlp',
  'const anchorNode = hasMixedAuxiliaries',
  'Math.abs(sourceNode.position.y - targetNode.position.y) < 1',
  'schemaNodeType(node.kind, node.state_type)'
].forEach((contractMarker) => {
  assert(
    deepseekSourceGraph.includes(contractMarker),
    `deepseek_v32_source_graph.html: missing shared layout contract marker ${contractMarker}`
  );
});

if (failures.length) {
  failures.forEach((message) => console.error(`FAIL ${message}`));
  process.exitCode = 1;
} else {
  console.log(`OK ${migratedPages.length} modelviz pages load the shared Pass-IR capsule contract`);
  console.log('OK inline scripts parse and local capsule resources resolve');
}
