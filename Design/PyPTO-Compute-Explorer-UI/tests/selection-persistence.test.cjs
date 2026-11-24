const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

function makeHost() {
  const listeners = new Map();
  let html = '';
  const host = {
    dataset: {},
    hidden: false,
    get innerHTML() { return html; },
    set innerHTML(val) { html = String(val); },
    addEventListener(type, fn) {
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(fn);
    },
    dispatch(type, event) {
      for (const fn of listeners.get(type) || []) fn(event);
    },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    append() {},
    contains() { return true; },
  };
  return host;
}

function makeTarget(attrs = {}) {
  const dataset = {};
  const attrMap = new Map();
  for (const [k, v] of Object.entries(attrs)) {
    attrMap.set(k, String(v));
    if (k.startsWith('data-')) {
      const camel = k.slice(5).replace(/-([a-z])/g, (_, ch) => ch.toUpperCase());
      dataset[camel] = String(v);
    }
  }
  const el = {
    dataset,
    value: attrs.value ?? '',
    hasAttribute: name => attrMap.has(name),
    getAttribute: name => attrMap.get(name) ?? null,
    matches: sel => {
      const m = sel.match(/^\[([a-z0-9-]+)\]$/i);
      return m ? attrMap.has(m[1]) : false;
    },
    closest: sel => {
      const parts = sel.split(',').map(s => s.trim());
      for (const p of parts) {
        if (p === 'button' && attrs._tag === 'button') return el;
        const m = p.match(/^\[([a-z0-9-]+)\]$/i);
        if (m && attrMap.has(m[1])) return el;
      }
      return null;
    },
  };
  return el;
}

function createWindowContext(files) {
  const win = {
    matchMedia: () => ({ matches: true, addEventListener() {} }),
    requestAnimationFrame: fn => fn(),
    document: {
      body: { classList: { contains: () => false } },
      createElement: () => makeHost(),
      querySelector: () => null,
    },
    innerWidth: 1440,
    innerHeight: 900,
    CSS: { escape: s => String(s) },
  };
  win.window = win;
  const ctx = {
    window: win,
    document: win.document,
    requestAnimationFrame: win.requestAnimationFrame,
    innerWidth: 1440,
    innerHeight: 900,
    CSS: win.CSS,
    console,
    Math,
    Number,
    String,
    Object,
    Array,
    Set,
    Map,
    JSON,
    structuredClone,
  };
  for (const rel of files) {
    vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../js', rel), 'utf8'), ctx);
  }
  return win;
}

test('Q Projection Runtime Trace <-> Lane Map preserves kernel, branch, insight, step, and lane/group selection', () => {
  const win = createWindowContext([
    'csa-stage-runtime-data.js',
    'csa-q-projection-data.js',
    'csa-q-projection-model.js',
    'csa-compressor-pipeline.js',
  ]);
  const host = makeHost();
  const kernelChanges = [];
  const viewChanges = [];
  const ctrl = win.PtoCsaCompressorPipeline.create(host, {
    model: win.PtoCsaQProjectionModel,
    rank: 0,
    kernel: 'qr_proj_matmul',
    onKernelChange: (k, b) => kernelChanges.push([k, b]),
    onViewChange: patch => viewChanges.push(patch),
  });
  ctrl.update({});

  // 1. Runtime Trace filter -> Lane Map -> Runtime Trace
  host.dispatch('change', { target: makeTarget({ 'data-cmp-rt-filter': '', value: 'qproj_matmul' }) });
  assert.equal(ctrl.getState().filter, 'qproj_matmul');
  assert.equal(ctrl.getState().kernel, 'qproj_matmul');

  host.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-cmp-overview': 'lane' }) });
  assert.equal(ctrl.getState().overviewMode, 'lane');
  assert.equal(ctrl.getState().kernel, 'qproj_matmul');

  // 2. Lane Map kernel change + lane selection -> Runtime Trace preserves kernel and maps lane -> group
  host.dispatch('change', { target: makeTarget({ 'data-cmp-kernel': '', value: 'q_rope_prepare' }) });
  assert.equal(ctrl.getState().kernel, 'q_rope_prepare');
  host.dispatch('click', { target: makeTarget({ 'data-cmp-lane': '10' }) }); // AIV lane 10 => Group 5
  assert.equal(ctrl.getState().selectedLane, 10);

  host.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-cmp-overview': 'runtime' }) });
  assert.equal(ctrl.getState().overviewMode, 'runtime');
  assert.equal(ctrl.getState().filter, 'q_rope_prepare');
  assert.equal(ctrl.getState().selectedGroup, 5);

  // 3. Runtime Trace compact insight card click -> Lane Map keeps insight's kernel
  host.dispatch('change', { target: makeTarget({ 'data-cmp-rt-filter': '', value: 'all' }) });
  host.dispatch('click', {
    target: makeTarget({
      'data-cmp-rt-insight': 'skew:qproj_dequant_rms_nope_rope',
      'data-cmp-rt-insight-kernel': 'qproj_dequant_rms_nope_rope',
    }),
  });
  assert.equal(ctrl.getState().kernel, 'qproj_dequant_rms_nope_rope');
  host.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-cmp-overview': 'lane' }) });
  assert.equal(ctrl.getState().kernel, 'qproj_dequant_rms_nope_rope');

  // 4. Runtime Trace branch filter -> Lane Map keeps branch and valid kernel in that branch
  host.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-cmp-overview': 'runtime' }) });
  host.dispatch('change', { target: makeTarget({ 'data-cmp-rt-filter': '', value: 'qr' }) });
  host.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-cmp-overview': 'lane' }) });
  assert.equal(ctrl.getState().branch, 'qr');
  assert.equal(win.PtoCsaQProjectionModel.kernels.get(ctrl.getState().kernel).branch, 'qr');

  // 5. External Assignment/Graph kernel update with syncFilter overwrites stale Runtime Trace filter
  host.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-cmp-overview': 'runtime' }) });
  ctrl.update({ kernel: 'qr_rms_norm_quant', syncFilter: true });
  assert.equal(ctrl.getState().kernel, 'qr_rms_norm_quant');
  assert.equal(ctrl.getState().filter, 'qr_rms_norm_quant');
  host.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-cmp-overview': 'lane' }) });
  assert.equal(ctrl.getState().kernel, 'qr_rms_norm_quant');

  assert.ok(kernelChanges.length > 0);
  assert.ok(viewChanges.length > 0);
});

test('Stage Views (KV Proj) preserves selected kernel across Assignment <-> Pipeline <-> Runtime Trace / Kernel Overview', () => {
  const win = createWindowContext([
    'csa-stage-runtime-data.js',
    'csa-stage-traces-data.js',
    'csa-stage-views.js',
  ]);
  const host = makeHost();
  const sv = win.PtoCsaStageViews.mount(host, {});

  sv.update('assignment', 'input/kv', 0);
  const kvStage = win.PTO_CSA_RUNTIME_DATA.stages.find(s => s.id === 'input/kv');
  const picked = kvStage.kernels[1].name;

  // Select kernel in Assignment
  host.dispatch('click', { target: makeTarget({ 'data-sv-kernel': picked }) });
  assert.ok(host.innerHTML.includes(`data-sv-kernel="${picked}"`));
  assert.ok(host.innerHTML.includes(`class="is-selected" data-sv-kernel="${picked}"`));

  // Switch to Pipeline (Runtime Trace): selected kernel stays highlighted
  sv.update('pipeline', 'input/kv', 0);
  assert.ok(host.innerHTML.includes(`csa-rt-step-chip is-active" data-sv-rt-step="${picked}"`));

  // Switch to Kernel Overview: selected kernel stays selected
  host.dispatch('click', { target: makeTarget({ 'data-sv-pmode': 'kernels' }) });
  assert.ok(host.innerHTML.includes(`csa-sv-lane is-selected" data-sv-kernel="${picked}"`));

  // Switch back to Assignment: selected kernel still selected
  sv.update('assignment', 'input/kv', 0);
  assert.ok(host.innerHTML.includes(`class="is-selected" data-sv-kernel="${picked}"`));

  // Switch to a different stage: selection resets cleanly
  sv.update('assignment', 'redistribute/publish', 0);
  assert.ok(!host.innerHTML.includes(`data-sv-kernel="${picked}"`));
});

test('qk_pv Runtime Trace <-> Lane Map preserves Core Group / Lane focus bidirectionally', () => {
  const win = createWindowContext([
    'csa-qk-work.js',
    'csa-qk-runtime-trace-data.js',
    'csa-qk-runtime-trace.js',
    'csa-qk-pipeline.js',
  ]);
  const host = makeHost();
  let rtHost = null;
  host.querySelector = sel => {
    if (sel === '.csa-qp-runtime-host') {
      rtHost = makeHost();
      return rtHost;
    }
    return null;
  };
  let lastFocus = null;
  const ctrl = win.PtoCsaQkPipeline.create(host, {
    moduleId: 'sparse/attention/qk-pv',
    onFocus: f => { lastFocus = f; },
    onSource: () => {},
  });
  const context = { mode: 'source', workload: 'mixed', position: 8192, rank: 0 };
  const scenario = win.PtoCsaQkWork.deriveScenario(context);
  ctrl.update(null, context, scenario, 'instances');
  assert.ok(rtHost, 'Runtime trace host mounted');

  // Select Group 5 in Runtime Trace -> emits laneFocus(5)
  rtHost.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-rt-group': '5' }) });
  assert.equal(lastFocus?.kind, 'lane');
  assert.equal(lastFocus?.laneId, 5);

  // Switch to Lane Map -> renders Lane 5 selected
  host.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-qp-overview': 'lane' }) });
  assert.ok(host.innerHTML.includes('Lane 5 ·'));

  // Select Lane 12 in Lane Map -> switch back to Runtime Trace -> Group 12 selected
  host.dispatch('click', { target: makeTarget({ 'data-qp-lane': '12' }) });
  assert.equal(lastFocus?.laneId, 12);
  host.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-qp-overview': 'runtime' }) });
  assert.ok(rtHost.innerHTML.includes('csa-rt-row is-selected') && rtHost.innerHTML.includes('Group 12'));

  // Closing Group 12 inspector clears lane focus
  rtHost.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-rt-close': '' }) });
  assert.notEqual(lastFocus?.kind, 'lane');
});
