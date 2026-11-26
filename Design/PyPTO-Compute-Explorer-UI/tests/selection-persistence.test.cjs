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

test('qk_plan -> qk_pv -> Merge 3-act runtime trace renders chain diagnostics and fires cross-view drill-downs', () => {
  const win = createWindowContext([
    'csa-qk-work.js',
    'csa-qk-runtime-trace-data.js',
    'csa-qk-runtime-trace.js',
  ]);
  const rtHost = makeHost();
  const events = { act: [], assign: [], graph: [], source: [] };
  const ctrl = win.PtoCsaQkRuntimeTrace.mount(rtHost, {
    onSelectAct: actId => events.act.push(actId),
    onOpenAssignment: payload => events.assign.push(payload),
    onOpenGraph: (target, actId) => events.graph.push([target, actId]),
    onSource: (file, a, z) => events.source.push([file, a, z]),
  });

  // Rank 1 clean reference
  ctrl.update({ rank: 1, scenario: win.PtoCsaQkWork.deriveScenario({ mode: 'source', workload: 'mixed', position: 8192, rank: 1 }) });
  assert.ok(rtHost.innerHTML.includes('csa-rt-story-strip'), '3-act diagnostic strip rendered');
  assert.ok(rtHost.innerHTML.includes('1282 µs'), 'Rank 1 exact chain span shown');
  assert.ok(rtHost.innerHTML.includes('csa-rt-bar is-plan'), 'qk_plan bar rendered on physical AIV core');
  assert.ok(!rtHost.innerHTML.includes('csa-rt-plan-idle-zone'), 'No noisy red hatched idle zone on other 23 groups');
  assert.ok(rtHost.innerHTML.includes('单核串行 114 µs'), 'qk_plan axis flag states single-core serial cause');
  assert.ok(rtHost.innerHTML.includes('csa-rt-bar is-aiv0-tail'), 'AIV 0 zero-fill tail segment rendered');
  assert.ok(!/(UX Demo|mock|placeholder|占位)/i.test(rtHost.innerHTML), 'No forbidden demo/placeholder copy');

  // Act 1 -> opens Act 1 inspector (Graph structural root cause) and fires Graph/Source drill-down
  rtHost.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-rt-act': 'act1-qk-plan-barrier' }) });
  assert.equal(events.act.at(-1), 'act1-qk-plan-barrier');
  assert.ok(rtHost.innerHTML.includes('1 / 72 核'), 'Act 1 inspector shows 1/72 core utilization');
  assert.ok(rtHost.innerHTML.includes('is-plan-focus'), 'Act 1 highlights qk_plan owner group');
  assert.ok(rtHost.innerHTML.includes('<div><b>Graph</b>'), 'Act 1 includes Graph structural root cause');
  rtHost.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-rt-open-graph': 'sparse/attention/plan', 'data-rt-act-ctx': 'act1-qk-plan-barrier' }) });
  assert.deepEqual(events.graph.at(-1), ['sparse/attention/plan', 'act1-qk-plan-barrier']);

  // Act 2 -> Pipeline -> Assignment -> Source primary loop (Graph demoted to optional non-topology cost context)
  rtHost.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-rt-act': 'act2-qk-pv-parity' }) });
  assert.equal(events.act.at(-1), 'act2-qk-pv-parity');
  assert.ok(rtHost.innerHTML.includes('12 慢') && rtHost.innerHTML.includes('bit-exact'));
  assert.ok(rtHost.innerHTML.includes('Pipeline → Assignment → Source'), 'Act 2 shows 2-view primary loop');
  assert.ok(!rtHost.innerHTML.includes('<div><b>Graph</b>'), 'Act 2 does not force a Graph root-cause row');
  assert.ok(rtHost.innerHTML.includes('csa-rt-opt-context') && rtHost.innerHTML.includes('非计算图拓扑问题'), 'Act 2 demotes Graph to optional cost-branch context');
  rtHost.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-rt-open-assignment': 'act2-qk-pv-parity', 'data-rt-assign-lane': '0' }) });
  assert.deepEqual({ ...events.assign.at(-1) }, { actId: 'act2-qk-pv-parity', lane: 0, rank: 1 });

  // Act 3 -> opens Act 3 inspector (Graph missing-edge root cause) and fires source jump
  rtHost.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-rt-act': 'act3-invalid-zero-flow' }) });
  assert.equal(events.act.at(-1), 'act3-invalid-zero-flow');
  assert.ok(rtHost.innerHTML.includes('87.1 MiB') || rtHost.innerHTML.includes('MiB'));
  assert.ok(rtHost.innerHTML.includes('<div><b>Graph</b>'), 'Act 3 includes Graph missing-edge root cause');
  rtHost.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-rt-source': 'decode_csa.py:420:431' }) });
  assert.deepEqual([...events.source.at(-1)], ['decode_csa.py', 420, 431]);

  // Verify graphRole classification in storyChain metadata (only Act 1 & Act 3 are Graph structural root causes)
  const storyActs = [...win.PTO_CSA_QK_RUNTIME_TRACE.storyChain.acts];
  assert.deepEqual(
    [...storyActs.filter(a => a.graphRole !== 'cost-context').map(a => a.id)],
    ['act1-qk-plan-barrier', 'act3-invalid-zero-flow']
  );
  assert.equal(storyActs.find(a => a.id === 'act2-qk-pv-parity').graphCause, null);

  // Optimization comparison & 3-fix diff inspector
  rtHost.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-rt-opt-open': '' }) });
  assert.ok(rtHost.innerHTML.includes('csa-rt-opt-inspector'), 'Opt comparison panel opened');
  assert.ok(rtHost.innerHTML.includes('csa-rt-diff-del') && rtHost.innerHTML.includes('csa-rt-diff-add'), 'Unified before/after diffs rendered');
  assert.ok(rtHost.innerHTML.includes('1281.7 µs') && rtHost.innerHTML.includes('~782.1 µs'), 'Rank 1 before/after totals shown');
  assert.ok(rtHost.innerHTML.includes('−13.3% 整层 CSA'), 'Rank 1 layer saving pct shown cleanly');
  assert.ok(!rtHost.innerHTML.includes('NaN'), 'No NaN in rendered opt panel');
  assert.ok(rtHost.innerHTML.includes('−174.2 MiB'), 'HBM traffic savings shown when Fix 3 enabled');
  const toggle3 = makeTarget({ 'data-rt-opt-toggle': 'act3-invalid-zero-flow' });
  toggle3.checked = false;
  rtHost.dispatch('change', { target: toggle3 });
  assert.ok(rtHost.innerHTML.includes('未启用修复 3'), 'Unchecking Fix 3 updates live HBM & latency estimate');
});

test('Assignment spatial canvas stays pure allocation view with SB0/CMP coloring, balanced order morph, and runtime input sync', () => {
  const win = createWindowContext([
    'csa-qk-runtime-trace-data.js',
    'csa-qk-work.js',
  ]);
  const host = makeHost();
  const events = { pipe: [], graph: [], source: [], sel: [] };
  const ctrl = win.PtoCsaQkWork.create(host, {
    rank: 1,
    onOpenPipeline: p => events.pipe.push(p),
    onOpenGraph: (t, a) => events.graph.push([t, a]),
    onSource: (f, a, z) => events.source.push([f, a, z]),
    onSelection: s => events.sel.push(s),
  });

  // Always renders spatial canvas (never clones Pipeline horizontal bar chart) and defaults to Runtime Trace input
  assert.equal(ctrl.getState().workload, 'trace');
  assert.equal(ctrl.getState().colorBy, 'kv-type');
  assert.equal(ctrl.getState().orderPolicy, 'source');
  assert.ok(host.innerHTML.includes('csa-qk-canvas-viewport'), 'Spatial SVG canvas rendered');
  assert.ok(!host.innerHTML.includes('csa-qk-cost-board'), 'No Pipeline-like horizontal bar chart in Assignment');
  assert.ok(host.innerHTML.includes('25 valid'), 'Lane 0 card shows exact Rank 1 valid count');
  assert.ok(host.innerHTML.includes('核组 00') && host.innerHTML.includes('核组 03'), 'Physical group mapping tags shown');
  assert.equal(ctrl.getScenario().validCount, 583);
  assert.ok(!/(UX Demo|mock|placeholder|占位)/i.test(host.innerHTML), 'No forbidden copy');

  // Switch coloring to static validity -> uniform valid/invalid view
  host.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-qk-color': 'validity' }) });
  assert.equal(ctrl.getState().colorBy, 'validity');
  assert.ok(!host.innerHTML.includes('is-color-type'));

  // Switch back to kv-type and toggle balanced order policy -> all 24 lanes balance to 10-11 SB0 + 13-14 CMP
  host.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-qk-color': 'kv-type' }) });
  host.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-qk-order': 'balanced' }) });
  assert.equal(ctrl.getState().orderPolicy, 'balanced');
  assert.equal(ctrl.getScenario().blocks[0].sb0Count, 11);
  assert.equal(ctrl.getScenario().blocks[0].cmpCount, 14);

  // Select Lane 1 -> maps to physical Group 03 and fires Pipeline jump with physicalGroup 3
  host.dispatch('click', { target: makeTarget({ 'data-qk-lane': '1' }) });
  assert.equal(events.sel.at(-1)?.type, 'block');
  assert.equal(events.sel.at(-1)?.block, 1);
  host.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-qk-open-pipeline': '3' }) });
  assert.deepEqual({ ...events.pipe.at(-1) }, { actId: null, lane: 3, rank: 1 });

  // External setStoryAct(act3) switches to ordering phase and syncs runtime trace input
  host.dispatch('change', { target: makeTarget({ 'data-qk-workload': '', value: 'mixed' }) });
  assert.equal(ctrl.getScenario().validCount, 992);
  ctrl.setStoryAct('act3-invalid-zero-flow');
  assert.equal(ctrl.getState().workload, 'trace');
  assert.equal(ctrl.getState().phase, 'ordering');
  assert.equal(ctrl.getScenario().validCount, 583);
});

test('qk_plan Graph / Assignment / Pipeline reuse shared semantic graph, kernel stage, and pipeline components cleanly', () => {
  const win = createWindowContext([
    'semantic-computation-graph.js',
    'csa-model.js',
    'csa-stage-runtime-data.js',
    'csa-stage-traces-data.js',
    'csa-qk-plan-data.js',
    'csa-qk-plan-model.js',
    'csa-semantic-graph.js',
    'csa-qk-plan-semantic.js',
    'csa-compressor-pipeline-instance.js',
    'csa-compressor-pipeline.js',
  ]);

  // 1. Semantic Graph (Structure + Instance) validates cleanly and exposes both branches & kernels
  const gStruct = win.PtoCsaQkPlanSemantic.getModel({ focused: true, instances: false });
  const gInst = win.PtoCsaQkPlanSemantic.getModel({ focused: true, instances: true });
  assert.equal(gStruct.id, 'csa-qk-plan');
  const stageNode = gStruct.nodes[0];
  assert.equal(stageNode.id, 'sparse');
  const branchIds = [...stageNode.composite.subgraph.nodes.map(n => n.id)];
  assert.deepEqual(branchIds, ['plan', 'rope']);
  const planKernelStruct = stageNode.composite.subgraph.nodes[0].composite.subgraph.nodes.find(n => n.id === 'csa_slots_build_valid_qk_plan');
  const ropeKernelStruct = stageNode.composite.subgraph.nodes[1].composite.subgraph.nodes.find(n => n.id === 'rope_cs');
  assert.equal(planKernelStruct.stageContext.stage, 'sparse/attention/plan');
  assert.equal(ropeKernelStruct.stageContext.stage, 'sparse/attention/plan');
  const planOpsInst = gInst.nodes[0].composite.subgraph.nodes[0].composite.subgraph.nodes
    .find(n => n.id === 'csa_slots_build_valid_qk_plan').composite.subgraph.nodes;
  assert.ok(planOpsInst.some(n => n.id === 'blk-valid:3'), 'Instance scale expands SB1–SB4 block validity reductions');
  const ropeOpsInst = gInst.nodes[0].composite.subgraph.nodes[1].composite.subgraph.nodes
    .find(n => n.id === 'rope_cs').composite.subgraph.nodes;
  assert.ok(ropeOpsInst.some(n => n.id === 'gather-row:7'), 'Instance scale expands 8 row gathers for rope_cs');

  // 2. Pipeline Runtime Trace <-> Lane Map <-> Micro Block Pipeline (Structure + Instance)
  const host = makeHost();
  let instHost = null;
  let lastScene = null;
  host.querySelector = sel => {
    if (sel === '.csa-qp-instance-host') {
      instHost = makeHost();
      lastScene = { style: {}, innerHTML: '' };
      instHost.querySelector = s => {
        if (s === '[data-cmp-pi-scene]') return lastScene;
        if (s === '[data-cmp-pi-summary]') return { textContent: '' };
        if (s === '[data-cmp-pi-note]') return { textContent: '' };
        if (s === '[data-cmp-pi-inspector]') return { hidden: true, innerHTML: '' };
        if (s === '.viz2-pi-viewport') return { clientWidth: 1000, clientHeight: 400, scrollLeft: 0, scrollTop: 0, addEventListener() {} };
        if (s === '[data-cmp-pi-scale]') return { textContent: '' };
        return null;
      };
      return instHost;
    }
    return null;
  };
  const ctrl = win.PtoCsaCompressorPipeline.create(host, {
    model: win.PtoCsaQkPlanModel,
    rank: 0,
    kernel: 'csa_slots_build_valid_qk_plan',
  });
  ctrl.update({});
  assert.ok(host.innerHTML.includes('csa_slots_build_valid_qk_plan'), 'Runtime Trace renders qk_plan kernel');
  assert.ok(host.innerHTML.includes('并行度不足'), 'Runtime Trace surfaces 1/48 AIV underutilization insight');
  assert.ok(!host.innerHTML.includes('NaN'), 'No NaN in qk_plan Runtime Trace');
  assert.ok(!/(UX Demo|mock|placeholder|占位)/i.test(host.innerHTML), 'No forbidden copy in qk_plan Runtime Trace');

  // Switch to Lane Map -> switch kernel to rope_cs -> switch back to Runtime Trace preserves rope_cs
  host.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-cmp-overview': 'lane' }) });
  assert.equal(ctrl.getState().overviewMode, 'lane');
  assert.ok(host.innerHTML.includes('1 Core Group'), 'Lane Map renders 1 Core Group label');
  host.dispatch('change', { target: makeTarget({ 'data-cmp-kernel': '', value: 'rope_cs' }) });
  assert.equal(ctrl.getState().kernel, 'rope_cs');
  host.dispatch('click', { target: makeTarget({ _tag: 'button', 'data-cmp-overview': 'runtime' }) });
  assert.equal(ctrl.getState().filter, 'rope_cs');

  // Drill into Block 0 -> Micro Structure & Instance Pipeline for both kernels
  ctrl.update({ kernel: 'csa_slots_build_valid_qk_plan', block: 0, level: 'block', intraScale: 'structure' });
  assert.ok(host.innerHTML.includes('qk_order / sparse_bias → qk_pv'), 'Structure Pipeline shows qk_pv downstream boundary');
  assert.ok(host.innerHTML.includes('Token 行数'), 'Uses model.caseLabel instead of Q 行数');
  ctrl.update({ kernel: 'csa_slots_build_valid_qk_plan', block: 0, level: 'block', intraScale: 'instances' });
  assert.ok(instHost && instHost.innerHTML.includes('viz2-pi-viewport'), 'Instance Pipeline mounts for csa_slots_build_valid_qk_plan');
  assert.ok(lastScene.innerHTML.includes('Pass 1 · Valid-First') && lastScene.innerHTML.includes('Slot Range Mask'), 'qk_plan instance blocks rendered');
  assert.ok(!lastScene.innerHTML.includes('NaN'), 'No NaN in qk_plan instance scene');
  ctrl.update({ kernel: 'rope_cs', block: 0, level: 'block', intraScale: 'structure' });
  assert.ok(host.innerHTML.includes('Inverse RoPE Params → Merge'), 'Structure Pipeline shows Merge downstream boundary for rope_cs');
  ctrl.update({ kernel: 'rope_cs', block: 0, level: 'block', intraScale: 'instances' });
  assert.ok(instHost && instHost.innerHTML.includes('viz2-pi-viewport'), 'Instance Pipeline mounts for rope_cs');
  assert.ok(lastScene.innerHTML.includes('Build Swap Index') && lastScene.innerHTML.includes('8×Row Gather Cos'), 'rope_cs instance blocks rendered');
  assert.ok(!lastScene.innerHTML.includes('NaN'), 'No NaN in rope_cs instance scene');
});
