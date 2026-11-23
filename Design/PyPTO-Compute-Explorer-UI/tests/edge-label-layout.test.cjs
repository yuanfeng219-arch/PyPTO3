const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const context = { window: {} };
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../js/semantic-computation-layout.js'), 'utf8'), context);
const { routeEdges, synchronizationBridges } = context.window.PTO_COMPUTATION_LAYOUT;
const node = (id, x, y) => ({ id, x, y, w: 100, h: 48 });
const edge = (s, t, type) => ({ s, t, type });

test('state outputs and feedback labels sit on opposite sides of each branch', () => {
  const source = node('softmax', 0, 0);
  const states = [node('l', 240, 132), node('m', 560, 132)];
  const nodes = [source, ...states];
  const edges = states.flatMap(({ id }) => [edge(source.id, id, 'state'), edge(id, source.id, 'feedback')]);
  const original = routeEdges(nodes, edges, 660, { flowLayout: true });
  const labeled = routeEdges(nodes, edges, 660, { flowLayout: true, branchEdgeLabels: true });

  states.forEach((state, index) => {
    const forward = labeled[index * 2];
    const feedback = labeled[index * 2 + 1];
    assert.equal(forward.labelX, feedback.labelX);
    assert.ok(feedback.labelX > state.x + state.w / 2);
    assert.ok(forward.labelY < feedback.points[1].y);
    assert.ok(feedback.labelY > feedback.points[1].y && feedback.labelY < state.y);
  });
  assert.notEqual(labeled[1].labelX, labeled[3].labelX);
  assert.deepEqual(labeled.map(route => route.path), original.map(route => route.path));
});

test('converging state inputs are labeled before the shared target port', () => {
  const nodes = [node('rescale', 0, 132), node('pending', 240, 0), node('old', 540, 0)];
  const edges = [edge('pending', 'rescale', 'state'), edge('old', 'rescale', 'state')];
  const labels = routeEdges(nodes, edges, 640, { flowLayout: true, branchEdgeLabels: true });
  assert.ok(labels.every(route => route.labelAnchor === 'middle'));
  assert.ok(labels[1].labelX - labels[0].labelX >= 100);
});

test('self-feedback is labeled below its own state node', () => {
  const rescale = node('rescale', 260, 132);
  const nodes = [rescale, node('m', 460, 132)];
  const [route] = routeEdges(nodes, [edge('rescale', 'rescale', 'feedback')], 560,
    { flowLayout: true, branchEdgeLabels: true });
  assert.equal(route.labelAnchor, 'start');
  assert.ok(route.labelX > rescale.x && route.labelX < rescale.x + rescale.w);
  assert.ok(route.labelY > rescale.y + rescale.h);
});

test('synchronization markers avoid cards, edge labels, and other synchronization markers', () => {
  const nodes = [node('producer', 280, 100), node('consumer', 280, 420), node('side-card', 360, 155)];
  const edgeLabel = { x: 480, y: 155, w: 150, h: 22 };
  const graph = {
    nodes, flowLayout: true, width: 900, height: 550,
    edges: [{ label: 'next_state' }],
    edgeRoutes: [{ labelX: edgeLabel.x, labelY: edgeLabel.y + 17, labelAnchor: 'start' }],
    synchronizations: [0, 1].map(eventId => ({
      eventId, producer: { node: 'producer' }, consumer: { node: 'consumer' }
    }))
  };
  const [first, second] = synchronizationBridges(graph);
  const overlap = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
  for (const marker of [first, second]) {
    assert.ok(nodes.every(box => !overlap(marker, box)));
    assert.ok(!overlap(marker, edgeLabel));
    assert.ok(marker.x + marker.w <= graph.width);
  }
  assert.ok(!overlap(first, second));
  assert.equal(first.path, second.path);
});
