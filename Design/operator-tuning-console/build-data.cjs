/**
 * Build the Tuning Console dataset from one real on-device run dump.
 *
 * Source run: Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617
 *   dfx_outputs/rank{0,1}/d0/  merged swimlane trace, chip_swimlane_records, deps, name_map, host STRACE log
 *   report/perf_hints.log      compiler perf hints (PH001, PH-MR-001)
 *   passes_dump/               52 IR dumps
 *   distributed_meta.json      bound parameter shapes / dtypes
 *
 * Every number in data.js is read from those files.
 * Run:  node build-data.cjs
 */
'use strict';
const fs = require('fs');
const path = require('path');

const OUT = path.join(__dirname, 'data.js');
const DATA = path.resolve(__dirname, '../../Data');

/* Two real dumps, captured by different tools at different times. They do not
 * carry the same artifacts, and the console is expected to say so rather than
 * fake the missing layers:
 *   decode_csa          L3, two ranks, host STRACE spans, PH001 + PH-MR-001,
 *                       no PTOAS / kernel sources
 *   decode_fwd_layers   L2, one device, NO host log (so no e2e layer at all),
 *                       PH001 only, but ships ptoas/ + kernels/ + orchestration/
 */
const CASES = [
  {
    id: 'decode_csa',
    label: 'decode_csa',
    sub: 'DeepSeek V4 · L3 · 2 rank',
    root: path.join(DATA, 'DeepseekV4/_jit_l3_decode_csa_20260903_010617'),
    runDir: '_jit_l3_decode_csa_20260903_010617',
    capturedAt: '2026-09-03 01:06:17',
    level: 3,
    model: 'deepseek_v4_flash_dspark / decode_csa',
    sourceRoot: '/data/w00949750/wzh_pypto_github/pypto/pypto-lib/models/deepseek_v4_flash_dspark',
    binaryContext: 'next_levels/decode_csa_test/cache/binary_context.json',
    ranks: [
      { key: 'rank0', dir: 'dfx_outputs/rank0/d0', trace: 'merged_swimlane_20260903_010746.json', host: 'host.2263908.log' },
      { key: 'rank1', dir: 'dfx_outputs/rank1/d0', trace: 'merged_swimlane_20260903_010747.json', host: 'host.2263922.log' },
    ],
  },
  {
    id: 'decode_fwd_layers',
    label: 'decode_fwd_layers',
    sub: 'Qwen3 14B · L2 · 1 device',
    root: path.join(DATA, '_jit_decode_fwd_layers_20260625_184941'),
    runDir: '_jit_decode_fwd_layers_20260625_184941',
    capturedAt: '2026-06-25 18:49:41',
    level: 2,
    model: 'qwen3 / 14b / decode_layer',
    sourceRoot: '/data/w00949750/wzh_pypto_github/pypto/pypto-lib/models/qwen3/14b',
    binaryContext: null,
    ranks: [
      { key: 'device0', dir: 'dfx_outputs', trace: 'merged_swimlane_20260625_185006.json', host: null },
    ],
  },
];

let CUR = CASES[0];
const at = (p) => path.join(CUR.root, p);
const has = (p) => fs.existsSync(at(p));
const rd = (p) => fs.readFileSync(at(p), 'utf8');
const rj = (p) => JSON.parse(rd(p));
const tryRd = (p) => (p && has(p) ? rd(p) : null);
const tryRj = (p) => (p && has(p) ? rj(p) : null);
const r2 = (n) => Math.round(n * 100) / 100;
const r3 = (n) => Math.round(n * 1000) / 1000;

function buildCase(CASE) {
CUR = CASE;
const RANK_KEYS = CASE.ranks.map((r) => r.key);
const PRIMARY = CASE.ranks[0];

/* ---------------------------------------------------------------- case */
const distMeta = tryRj('distributed_meta.json');
const nameMapFile = fs.readdirSync(at(PRIMARY.dir)).find((f) => /^name_map.*\.json$/.test(f));
const nameMap = rj(PRIMARY.dir + '/' + nameMapFile);
const dispatch = tryRj(PRIMARY.dir + '/dispatch_program.json');
const csr = tryRj(PRIMARY.dir + '/chip_swimlane_records.json');
const binCtx = tryRj(CASE.binaryContext);

/* incore scope names as the compiler outlined them — the kernel inventory */
const incoreFile = fs.readdirSync(at('passes_dump')).find((f) => /OutlineIncoreScopes/.test(f));
const incoreNames = incoreFile ? Array.from(new Set(
  (fs.readFileSync(at('passes_dump/' + incoreFile), 'utf8')
    .match(/^\s*def ([a-z_0-9]+)\(/gm) || []).map((s) => s.trim().replace(/^def /, '').replace(/\($/, ''))
)).sort() : [];

/* PTOAS / kernel / orchestration sources: present in the L2 dump, absent in L3 */
const ptoasFiles = has('ptoas') ? fs.readdirSync(at('ptoas')) : [];
const ptoasUnits = Array.from(new Set(ptoasFiles.map((f) => f.replace(/\.(pto|cpp)$/, ''))))
  .filter((n) => ptoasFiles.indexOf(n + '.pto') >= 0)
  .sort()
  .map((n) => {
    const pto = rd('ptoas/' + n + '.pto');
    const cppPath = 'ptoas/' + n + '.cpp';
    const cpp = has(cppPath) ? rd(cppPath) : null;
    return {
      name: n,
      ptoLines: pto.split('\n').length,
      ptoBytes: pto.length,
      cppLines: cpp ? cpp.split('\n').length : null,
      cppBytes: cpp ? cpp.length : null,
    };
  });
const kernelDirs = has('kernels') ? fs.readdirSync(at('kernels')) : [];
const kernelCounts = {};
kernelDirs.forEach((d) => {
  const full = at('kernels/' + d);
  if (fs.statSync(full).isDirectory()) kernelCounts[d] = fs.readdirSync(full).length;
});
const orchFiles = has('orchestration') ? fs.readdirSync(at('orchestration')) : [];
const kernelConfig = tryRd('kernel_config.py');
const runtimeFromConfig = kernelConfig
  ? (kernelConfig.match(/"runtime":\s*"([^"]+)"/) || [])[1] || null : null;
const aicpuThreads = kernelConfig
  ? +((kernelConfig.match(/"aicpu_thread_num":\s*(\d+)/) || [])[1] || 0) || null : null;
/* Parsed traces are shared between the metadata pass and buildRank — a single
 * dump is 6–10 MB of JSON, so parse each one exactly once. */
const TRACES = {};
const traceOf = (r) => (TRACES[r.key] = TRACES[r.key] || rj(r.dir + '/' + r.trace).traceEvents);

/* chip_swimlane_records carries the core inventory. Without it (the L2 dump),
 * read the same facts off the Worker View's thread names. */
const coreTypes = csr ? csr.metadata.core_types : null;
const laneNamesFromTrace = (() => {
  const names = [];
  traceOf(PRIMARY).filter((e) => e.cat === '__metadata' && e.name === 'thread_name' && e.pid === 4)
    .forEach((e) => { if (names.indexOf(e.args.name) < 0) names.push(e.args.name); });
  return names;
})();
const aicCount = coreTypes ? coreTypes.filter((t) => t === 'aic').length
  : laneNamesFromTrace.filter((n) => n.indexOf('AIC') === 0).length;
const aivCount = coreTypes ? coreTypes.filter((t) => t === 'aiv').length
  : laneNamesFromTrace.filter((n) => n.indexOf('AIV') === 0).length;

const perfHintText = tryRd('report/perf_hints.log') || '';
const backend = (perfHintText.match(/for backend (\w+)/) || [])[1]
  || (binCtx && binCtx.platform) || 'a2a3';

const caseInfo = {
  id: CASE.id,
  label: CASE.label,
  sub: CASE.sub,
  level: CASE.level,
  program: dispatch ? dispatch.program : CASE.label,
  model: CASE.model,
  backend: backend,
  ranks: RANK_KEYS,
  device: CASE.level === 3 ? 'd0' : 'device0',
  clockHz: csr ? csr.metadata.clock_freq_hz : null,
  numCores: csr ? csr.metadata.num_cores : laneNamesFromTrace.length,
  aicCount: aicCount,
  aivCount: aivCount,
  threadsPerCore: csr ? Math.max.apply(null, csr.metadata.core_to_thread) + 1 : null,
  callables: Object.keys(nameMap.callable_id_to_name).length,
  swimlaneLevel: csr ? csr.chip_swimlane_level : null,
  metaSchema: distMeta ? distMeta.schema : null,
  params: distMeta
    ? distMeta.params.map((p) => ({ name: p.name, dir: p.direction, shape: p.shape, dtype: p.dtype }))
    : [],
  runDir: CASE.runDir,
  capturedAt: CASE.capturedAt,
  toolchain: {
    platform: binCtx ? binCtx.platform : backend,
    ptoIsaRevision: binCtx ? binCtx.pto_isa_revision : null,
    runtimeName: binCtx ? binCtx.runtime_name : runtimeFromConfig,
    runtimeRevision: binCtx ? binCtx.runtime_revision : null,
    schema: binCtx ? binCtx.schema : null,
    aicpuThreads: aicpuThreads,
  },
  incoreScopes: incoreNames,
  sourceRoot: CASE.sourceRoot,
  /* Neither dump carries a kernel -> source-file map: perf hints are anchored
   * on source locations, the IR keeps only outlined incore scope names. */
  hasKernelSourceMap: false,
  /* what this dump can and cannot answer — the UI reads these directly */
  artifacts: {
    hostSpans: CASE.ranks.every((r) => !!r.host),
    distributedMeta: !!distMeta,
    chipSwimlaneRecords: !!csr,
    binaryContext: !!binCtx,
    ptoas: ptoasUnits.length,
    kernelDirs: kernelCounts,
    orchestration: orchFiles,
    kernelConfig: !!kernelConfig,
  },
  ptoasUnits: ptoasUnits,
};

/* ------------------------------------------------------------ end-to-end */
function hostSpans(file) {
  const out = {};
  for (const line of rd(file).split(/\r?\n/)) {
    if (!line.trim()) continue;
    const name = (line.match(/name=(\S+)/) || [])[1];
    const dur = +(line.match(/dur=(\d+)/) || [])[1];
    const inv = +(line.match(/inv=(\d+)/) || [])[1];
    const clk = /clk=dev/.test(line) ? 'device' : 'host';
    const ts = +(line.match(/ ts=(\d+)/) || [])[1];
    if (!name || !Number.isFinite(dur)) continue;
    (out[inv] = out[inv] || {})[name] = { us: r2(dur / 1000), clk: clk, tsNs: Number.isFinite(ts) ? ts : null };
  }
  return out;
}
const e2e = CASE.ranks.every((r) => r.host)
  ? CASE.ranks.reduce((acc, r) => { acc[r.key] = hostSpans(r.dir + '/' + r.host); return acc; }, {})
  : null;

const parseList = (s) => {
  const m = (s || '').match(/\[([^\]]*)\]/);
  return m && m[1].trim() ? m[1].split(',').map((x) => x.trim()) : [];
};
const sum = (a) => a.reduce((x, y) => x + y, 0);

/* ------------------------------------------------------- swimlane trace
 * One rank at a time: worker lanes, task aggregation, AICPU scheduler lanes,
 * the ready-queue counter, dependency flows and the measured critical path.  */
function buildRank(RK) {
const rank = RK.key;
const traceFile = RK.trace;
const trace = traceOf(RK);
const threadName = {};
const processName = {};
trace.filter((e) => e.cat === '__metadata').forEach((e) => {
  if (e.name === 'thread_name') threadName[e.pid + ':' + e.tid] = e.args.name;
  if (e.name === 'process_name') processName[e.pid] = e.args.name;
});

/* pid 4 = "Worker View": one event per block execution on a core.
 * pid 3 = "Scheduler View": the AICPU-side view of the same block
 *         (dispatch-time-us -> finish-time-us, aicpu-duration-us). */
const blockEvents = trace.filter((e) => e.cat === 'event' && e.ph === 'X'
  && e.pid === 4 && e.name !== 'setup'
  && /Task:/.test(e.args['event-hint'] || '')
  && e.args['duration-us'] !== undefined
  && !!threadName['4:' + e.tid]);
/* separate setup spans (L2 dump only), keyed by taskId + core */
const setupByKey = {};
trace.filter((e) => e.cat === 'event' && e.ph === 'X' && e.pid === 4 && e.name === 'setup')
  .forEach((e) => {
    const core = (e.args['event-hint'] || '').match(/CoreId:(\d+)/);
    const key = String(e.args.taskId) + '@' + (core ? core[1] : e.tid);
    setupByKey[key] = (setupByKey[key] || 0) + (e.args.local_setup_us || e.dur || 0);
  });
const hasOwnSetupEvents = Object.keys(setupByKey).length > 0;
const schedViewEvents = trace.filter((e) => e.cat === 'event' && e.ph === 'X'
  && e.pid === 3 && e.args['aicpu-duration-us'] !== undefined);
const SPAN = r2(Math.max.apply(null, blockEvents.map((e) => e.ts + e.dur)));

const laneNames = Array.from(new Set(blockEvents.map((e) => threadName['4:' + e.tid]).filter(Boolean)))
  .sort((a, b) => {
    const ka = a.indexOf('AIC') === 0 ? 0 : 1;
    const kb = b.indexOf('AIC') === 0 ? 0 : 1;
    return ka - kb || (+a.split('_')[1] - +b.split('_')[1]);
  });
const laneIdx = {};
laneNames.forEach((n, i) => { laneIdx[n] = i; });

const rankDeps = tryRj(RK.dir + '/deps.json');
const depById = {};
if (rankDeps && rankDeps.tasks) rankDeps.tasks.forEach((t) => { depById[t.task_id] = t; });
const taskMap = new Map();
for (const e of blockEvents) {
  const hint = e.args['event-hint'];
  const id = String(e.args.taskId);
  if (!taskMap.has(id)) {
    taskMap.set(id, {
      id: id,
      tag: (hint.match(/Task:(\S+?),/) || [])[1],
      funcId: +(hint.match(/FuncId:(-?\d+)/) || [0, -1])[1],
      rawName: e.name,
      blocks: [],
      fanoutRaw: e.args['fanout-hint'],
      faninRaw: e.args['fanin-hint'],
    });
  }
  const t = taskMap.get(id);
  const core = +(hint.match(/CoreId:(\d+)/) || [0, -1])[1];
  const setup = hasOwnSetupEvents
    ? (setupByKey[id + '@' + core] || 0)
    : (e.args['local_setup_us'] || 0);
  const kdur = hasOwnSetupEvents
    ? e.dur                                  /* L2: the block IS the kernel */
    : (e.args['kernel-duration-us'] || 0);   /* L3: carried on the event    */
  t.blocks.push({
    lane: laneIdx[threadName['4:' + e.tid]],
    core: core,
    ts: hasOwnSetupEvents ? e.ts - setup : e.ts,
    dur: hasOwnSetupEvents ? e.dur + setup : e.dur,
    kdur: kdur,
    setup: setup,
  });
}

/* AICPU-side (dispatch -> finish) view of each block, keyed by taskId */
const schedViewByTask = {};
for (const e of schedViewEvents) {
  const id = String(e.args.taskId);
  (schedViewByTask[id] = schedViewByTask[id] || []).push({
    dispatch: e.args['dispatch-time-us'],
    finish: e.args['finish-time-us'],
    aicpu: e.args['aicpu-duration-us'],
  });
}

const tasks = [];
taskMap.forEach((t) => {
  const b = t.blocks;
  const durs = b.map((x) => x.dur).sort((a, c) => a - c);
  const start = Math.min.apply(null, b.map((x) => x.ts));
  const end = Math.max.apply(null, b.map((x) => x.ts + x.dur));
  const lanes = Array.from(new Set(b.map((x) => laneNames[x.lane])));
  const kind = lanes.every((l) => l.indexOf('AIC') === 0) ? 'aic'
    : lanes.every((l) => l.indexOf('AIV') === 0) ? 'aiv' : 'mix';
  const dd = depById[t.id];
  const setupSum = sum(b.map((x) => x.setup));
  const sv = schedViewByTask[t.id] || [];
  const svAicpu = sv.map((x) => x.aicpu).sort((a, c) => a - c);
  const workerMean = sum(b.map((x) => x.dur)) / b.length;
  tasks.push({
    id: t.id, tag: t.tag, funcId: t.funcId,
    callable: nameMap.callable_id_to_name[String(t.funcId)] || t.rawName,
    rawName: t.rawName,
    ring: +(t.tag.match(/r(\d+)/) || [0, 0])[1],
    kind: kind,
    blockCount: b.length,
    coreCount: new Set(b.map((x) => x.core)).size,
    laneCount: lanes.length,
    blockNum: dd ? dd.block_num : null,
    scope: dd ? dd.scope : null,
    earlyDispatch: dd ? !!dd.early_dispatch : null,
    start: r2(start), end: r2(end), span: r2(end - start),
    durMin: r2(durs[0]), durMax: r2(durs[durs.length - 1]),
    durMed: r2(durs[Math.floor(durs.length / 2)]),
    durMean: r2(sum(durs) / durs.length),
    durP90: r2(durs[Math.min(durs.length - 1, Math.floor(durs.length * 0.9))]),
    busySum: r2(sum(durs)),
    kdurSum: r2(sum(b.map((x) => x.kdur))),
    setupSum: r2(setupSum),
    setupMean: r3(setupSum / b.length),
    setupShare: r3(setupSum / sum(durs)),
    imbalance: r2(durs[durs.length - 1] / Math.max(durs[Math.floor(durs.length / 2)], 1e-9)),
    svBlocks: sv.length,
    svAicpuMean: svAicpu.length ? r2(sum(svAicpu) / svAicpu.length) : null,
    svAicpuMax: svAicpu.length ? r2(svAicpu[svAicpu.length - 1]) : null,
    svOverhead: svAicpu.length ? r2(sum(svAicpu) / svAicpu.length - workerMean) : null,
    pred: parseList(t.faninRaw), succ: parseList(t.fanoutRaw),
    args: dd ? dd.args.map((a) => ({ idx: a.idx, type: a.type, dtype: a.dtype, shape: a.shape })) : [],
  });
});
tasks.sort((a, b) => a.start - b.start);
const taskIndex = {};
tasks.forEach((t, i) => { taskIndex[t.tag] = i; });

const laneBlocks = laneNames.map(() => []);
taskMap.forEach((t) => {
  const ti = taskIndex[t.tag];
  t.blocks.forEach((b) => laneBlocks[b.lane].push([r2(b.ts), r2(b.dur), ti]));
});
laneBlocks.forEach((a) => a.sort((x, y) => x[0] - y[0]));

const lanes = laneNames.map((name, i) => {
  const a = laneBlocks[i];
  let busy = 0, idle = 0, nGap = 0, maxGap = 0, cursor = null;
  for (const row of a) {
    busy += row[1];
    if (cursor !== null && row[0] - cursor > 0.5) {
      idle += row[0] - cursor; nGap++; maxGap = Math.max(maxGap, row[0] - cursor);
    }
    cursor = Math.max(cursor === null ? 0 : cursor, row[0] + row[1]);
  }
  return {
    name: name, kind: name.indexOf('AIC') === 0 ? 'aic' : 'aiv', coreId: +name.split('_')[1],
    blocks: a.length, busy: r2(busy), util: r2((busy / SPAN) * 100),
    idle: r2(idle), nGap: nGap, maxGap: r2(maxGap),
    first: a.length ? r2(a[0][0]) : null,
    last: a.length ? r2(a[a.length - 1][0] + a[a.length - 1][1]) : null,
  };
});

/* ------------------------------------------------------ AICPU scheduler */
const schedEv = trace.filter((e) => e.cat === 'scheduler' && e.ph === 'X');
const schedLanes = Array.from(new Set(schedEv.map((e) => threadName['2:' + e.tid] || 'Sched?'))).sort();
const schedPhases = {};
for (const e of schedEv) {
  const p = e.args.phase;
  const s = (schedPhases[p] = schedPhases[p] || { n: 0, us: 0, tasks: 0 });
  s.n++; s.us += e.dur; s.tasks += e.args.tasks_processed || 0;
}
Object.keys(schedPhases).forEach((k) => {
  const s = schedPhases[k];
  s.us = r2(s.us);
  s.usPerTask = s.tasks ? r3(s.us / s.tasks) : null;
});
const schedWindow = {
  lo: r2(Math.min.apply(null, schedEv.map((e) => e.ts))),
  hi: r2(Math.max.apply(null, schedEv.map((e) => e.ts + e.dur))),
};
const schedBusy = r2(sum(schedEv.map((e) => e.dur)));
const schedBlocks = schedLanes.map((ln) => schedEv
  .filter((e) => (threadName['2:' + e.tid] || 'Sched?') === ln)
  .map((e) => [r2(e.ts), r2(e.dur), e.args.phase, e.args.tasks_processed || 0])
  .sort((a, b) => a[0] - b[0]));
const schedLaneStats = schedLanes.map((ln, i) => {
  const busy = sum(schedBlocks[i].map((b) => b[1]));
  return {
    name: ln, events: schedBlocks[i].length, busy: r2(busy),
    util: r2((busy / (schedWindow.hi - schedWindow.lo)) * 100),
  };
});

const orchEv = trace.filter((e) => e.cat === 'orchestrator').map((e) => [r2(e.ts), r2(e.dur)]);
const orch = {
  count: orchEv.length,
  busy: r2(sum(orchEv.map((e) => e[1]))),
  lo: r2(Math.min.apply(null, orchEv.map((e) => e[0]))),
  hi: r2(Math.max.apply(null, orchEv.map((e) => e[0] + e[1]))),
  blocks: orchEv,
};

/* ready-but-undispatched queue counter */
const qEv = trace.filter((e) => e.cat === 'queue'
  && (e.name === 'shared_ready_queue' || !/^local_ready_buf/.test(e.name || '')))
  .sort((a, b) => a.ts - b.ts);
const readyQueue = qEv.map((e) => [r2(e.ts), e.args.AIC || 0, e.args.AIV || 0, e.args.MIX || 0]);
const KEYS = ['AIC', 'AIV', 'MIX'];
const rqStat = { window: 0, avg: {}, peak: {}, busyTime: {}, busyShare: {} };
KEYS.forEach((k) => { rqStat.avg[k] = 0; rqStat.peak[k] = 0; rqStat.busyTime[k] = 0; });
for (let i = 0; i < readyQueue.length - 1; i++) {
  const dt = readyQueue[i + 1][0] - readyQueue[i][0];
  rqStat.window += dt;
  KEYS.forEach((k, j) => {
    const v = readyQueue[i][j + 1];
    rqStat.avg[k] += v * dt;
    if (v > 0) rqStat.busyTime[k] += dt;
    rqStat.peak[k] = Math.max(rqStat.peak[k], v);
  });
}
KEYS.forEach((k) => {
  rqStat.avg[k] = r3(rqStat.avg[k] / rqStat.window);
  rqStat.busyShare[k] = r2((rqStat.busyTime[k] / rqStat.window) * 100);
  rqStat.busyTime[k] = r2(rqStat.busyTime[k]);
});
rqStat.window = r2(rqStat.window);

const flowCount = {};
trace.filter((e) => e.cat === 'flow').forEach((e) => { flowCount[e.name] = (flowCount[e.name] || 0) + 1; });
const hbPairs = trace.filter((e) => e.cat === 'flow' && e.name === 'hb_violation' && e.ph === 's').map((e) => {
  const fin = trace.find((f) => f.cat === 'flow' && f.name === 'hb_violation' && f.ph === 'f' && f.id === e.id);
  return {
    from: threadName['4:' + e.tid] || String(e.tid),
    to: fin ? (threadName['4:' + fin.tid] || String(fin.tid)) : null,
    ts: r2(e.ts), tsEnd: fin ? r2(fin.ts) : null,
    inputs: e.input_task_count, outputs: e.output_task_count,
  };
});

/* ------------------------------------------------------- critical path */
const byTag = {};
tasks.forEach((t) => { byTag[t.tag] = t; });
const best = {};
for (const t of tasks) {
  let bp = null, bl = 0;
  for (const p of t.pred) {
    const P = byTag[p];
    if (!P) continue;
    const c = (best[p] ? best[p].len : 0) + P.span;
    if (c > bl) { bl = c; bp = p; }
  }
  best[t.tag] = { len: bl, prev: bp };
}
let endTag = null, mx = -1;
for (const t of tasks) {
  const v = best[t.tag].len + t.span;
  if (v > mx) { mx = v; endTag = t.tag; }
}
const critTags = [];
for (let c = endTag; c; c = best[c].prev) critTags.unshift(c);
let cursor = null, posGap = 0, overlap = 0;
const critNodes = critTags.map((tag) => {
  const t = byTag[tag];
  const gap = cursor === null ? 0 : r2(t.start - cursor);
  if (gap > 0) posGap += gap; else overlap += -gap;
  cursor = t.end;
  return { tag: tag, gap: gap };
});
const critical = {
  tags: critTags,
  chainSpan: r2(mx),
  walltime: SPAN,
  nodes: critNodes,
  gapOnPath: r2(posGap),
  overlapOnPath: r2(overlap),
  spanSum: r2(sum(critTags.map((tg) => byTag[tg].span))),
};

/* -------------------------------------------------------------- slack
 * Standard forward/backward pass over the fanin/fanout DAG using the
 * MEASURED span of each task. ES/EF from predecessors, LF/LS from
 * successors, slack = LS - ES. This is structural slack: it says how much
 * the dependency graph would tolerate, and deliberately ignores resource
 * contention — two zero-slack tasks may still be fighting for the same core.
 * Tasks on the measured critical path have slack 0 by construction. */
const topo = tasks.slice().sort((a, b) => a.start - b.start);
const ES = {}, EF = {}, LS = {}, LF = {};
topo.forEach((t) => {
  let es = 0;
  t.pred.forEach((ptag) => { if (EF[ptag] != null && EF[ptag] > es) es = EF[ptag]; });
  ES[t.tag] = es;
  EF[t.tag] = es + t.span;
});
const makespan = Math.max.apply(null, topo.map((t) => EF[t.tag]));
topo.slice().reverse().forEach((t) => {
  let lf = null;
  t.succ.forEach((stag) => { if (LS[stag] != null && (lf === null || LS[stag] < lf)) lf = LS[stag]; });
  LF[t.tag] = lf === null ? makespan : lf;
  LS[t.tag] = LF[t.tag] - t.span;
});
tasks.forEach((t) => {
  t.es = r2(ES[t.tag]);
  t.ef = r2(EF[t.tag]);
  t.ls = r2(LS[t.tag]);
  t.slack = r2(Math.max(0, LS[t.tag] - ES[t.tag]));
  t.onCrit = critTags.indexOf(t.tag) >= 0;
});

/* ------------------------------------------------------------- scopes
 * "Where did the time go" is asked per scope, not per block: every task
 * carrying the same callable is one outlined scope. Σ core-time alone
 * ranks the fat ones; pairing it with the scope's minimum slack separates
 * "fat" from "fat AND on the critical path". */
const scopeMap = {};
tasks.forEach((t) => {
  const k = t.callable;
  const sc = (scopeMap[k] = scopeMap[k] || {
    name: k, tasks: [], tags: [], kinds: {},
    coreTime: 0, kernelTime: 0, setupTime: 0, aicpuTime: 0,
    blocks: 0, cores: new Set(), critNodes: 0,
    first: Infinity, last: -Infinity,
  });
  sc.tasks.push(t.tag);
  sc.tags.push(t.tag);
  sc.kinds[t.kind] = (sc.kinds[t.kind] || 0) + 1;
  sc.coreTime += t.busySum;
  sc.kernelTime += t.kdurSum || (t.busySum - t.setupSum);
  sc.setupTime += t.setupSum;
  if (t.svAicpuMean != null) sc.aicpuTime += t.svAicpuMean * t.svBlocks;
  sc.blocks += t.blockCount;
  sc.cores.add(t.coreCount);
  if (t.onCrit) sc.critNodes++;
  sc.first = Math.min(sc.first, t.start);
  sc.last = Math.max(sc.last, t.end);
});
const totalCoreTime = sum(tasks.map((t) => t.busySum));
const scopes = Object.keys(scopeMap).map((k) => {
  const sc = scopeMap[k];
  const ts = sc.tasks.map((tg) => byTag[tg]);
  const slacks = ts.map((t) => t.slack);
  const kind = Object.keys(sc.kinds).sort((a, b) => sc.kinds[b] - sc.kinds[a])[0];
  /* wall time this scope actually occupies, merging overlapping tasks */
  const iv = ts.map((t) => [t.start, t.end]).sort((a, b) => a[0] - b[0]);
  let wall = 0, cur = null;
  iv.forEach((r) => {
    if (!cur) { cur = r.slice(); return; }
    if (r[0] <= cur[1]) { cur[1] = Math.max(cur[1], r[1]); return; }
    wall += cur[1] - cur[0]; cur = r.slice();
  });
  if (cur) wall += cur[1] - cur[0];
  return {
    name: k,
    kind: kind,
    taskCount: ts.length,
    tags: sc.tags,
    blocks: sc.blocks,
    coreTime: r2(sc.coreTime),
    coreShare: r2((sc.coreTime / totalCoreTime) * 100),
    kernelTime: r2(sc.kernelTime),
    setupTime: r2(sc.setupTime),
    setupShare: r3(sc.setupTime / Math.max(sc.coreTime, 1e-9)),
    aicpuTime: sc.aicpuTime ? r2(sc.aicpuTime) : null,
    wall: r2(wall),
    first: r2(sc.first), last: r2(sc.last),
    minSlack: r2(Math.min.apply(null, slacks)),
    medSlack: r2(slacks.slice().sort((a, b) => a - b)[Math.floor(slacks.length / 2)]),
    critNodes: sc.critNodes,
    onCrit: sc.critNodes > 0,
    /* the fat scope that is also pinned to the critical path is the one worth
     * touching first; a fat scope with slack is a parallelism question. */
    critCoreTime: r2(sum(ts.filter((t) => t.onCrit).map((t) => t.busySum))),
  };
}).sort((a, b) => b.coreTime - a.coreTime);

/* -------------------------------------------------- occupancy windows
 * Bucket the run into fixed windows and measure how many cores were busy
 * in each. This is what "which stretch was the machine idle" needs and it
 * cannot be read off a per-task list. */
const WIN_N = 240;
const winW = SPAN / WIN_N;
const occWindows = (function () {
  const aicN = lanes.filter((l) => l.kind === 'aic').length || 1;
  const aivN = lanes.filter((l) => l.kind === 'aiv').length || 1;
  const aic = new Float64Array(WIN_N);
  const aiv = new Float64Array(WIN_N);
  lanes.forEach((l, li) => {
    const acc = l.kind === 'aic' ? aic : aiv;
    (laneBlocks[li] || []).forEach((b) => {
      /* clamp: SPAN is rounded, so a block can end a hair past it */
      const bs = Math.max(0, b[0]);
      const be = Math.min(SPAN, b[0] + b[1]);
      if (be <= bs) return;
      const w0 = Math.max(0, Math.min(WIN_N - 1, Math.floor(bs / winW)));
      const w1 = Math.max(0, Math.min(WIN_N - 1, Math.floor((be - 1e-9) / winW)));
      for (let w = w0; w <= w1; w++) {
        const ov = Math.min(be, (w + 1) * winW) - Math.max(bs, w * winW);
        if (ov > 0) acc[w] += ov;
      }
    });
  });
  const out = [];
  for (let i = 0; i < WIN_N; i++) {
    out.push([
      r2(i * winW),
      r2((aic[i] / (winW * aicN)) * 100),
      r2((aiv[i] / (winW * aivN)) * 100),
    ]);
  }
  return out;
})();

/* contiguous stretches where both engines sat below the threshold */
const IDLE_PCT = 15;
const idleRuns = (function () {
  const runs = [];
  let start = null;
  for (let i = 0; i < occWindows.length; i++) {
    const low = occWindows[i][1] < IDLE_PCT && occWindows[i][2] < IDLE_PCT;
    if (low && start === null) start = i;
    if ((!low || i === occWindows.length - 1) && start !== null) {
      const endI = low ? i : i - 1;
      const t0 = start * winW;
      const t1 = (endI + 1) * winW;
      const slice = occWindows.slice(start, endI + 1);
      runs.push({
        t0: r2(t0), t1: r2(t1), us: r2(t1 - t0),
        share: r2(((t1 - t0) / SPAN) * 100),
        aic: r2(sum(slice.map((w) => w[1])) / slice.length),
        aiv: r2(sum(slice.map((w) => w[2])) / slice.length),
        /* blocks genuinely executing inside the window, by core-time */
        running: (function () {
          const acc = {};
          let busy = 0;
          lanes.forEach((l, li) => {
            (laneBlocks[li] || []).forEach((b) => {
              const bs = b[0], be = b[0] + b[1];
              if (bs >= t1 || be <= t0) return;
              const ov = Math.min(be, t1) - Math.max(bs, t0);
              busy += ov;
              const t = tasks[b[2]];
              if (!t) return;
              const a = (acc[t.callable] = acc[t.callable] || { callable: t.callable, us: 0, blocks: 0, onCrit: t.onCrit });
              a.us += ov; a.blocks++;
            });
          });
          return {
            busyUs: r2(busy),
            /* share of the window's total core capacity that was busy */
            capacityPct: r2((busy / ((t1 - t0) * lanes.length)) * 100),
            top: Object.keys(acc).map((k) => acc[k])
              .sort((a, b) => b.us - a.us).slice(0, 3)
              .map((a) => ({ callable: a.callable, us: r2(a.us), blocks: a.blocks, onCrit: a.onCrit })),
          };
        })(),
        /* tasks whose envelope merely spans the window — usually the single
         * block that everything else is waiting on */
        spanning: tasks.filter((t) => t.start < t1 && t.end > t0)
          .sort((a, b) => b.span - a.span).slice(0, 3)
          .map((t) => ({ tag: t.tag, callable: t.callable, span: t.span, onCrit: t.onCrit, blocks: t.blockCount })),
      });
      start = null;
    }
  }
  return runs.sort((a, b) => b.us - a.us);
})();

const aicLanes = lanes.filter((l) => l.kind === 'aic');
const aivLanes = lanes.filter((l) => l.kind === 'aiv');

/* ------------------------------------------------ Worker / Scheduler 口径
 * The same block appears twice in this trace. Stating both totals next to
 * each other is the only way to stop them being silently added together. */
const workerTotal = r2(sum(tasks.map((t) => t.busySum)));
const schedTotal = r2(sum(tasks.filter((t) => t.svAicpuMean != null)
  .map((t) => t.svAicpuMean * t.svBlocks)));
const accounting = {
  workerBlocks: sum(tasks.map((t) => t.blockCount)),
  schedBlocks: sum(tasks.map((t) => t.svBlocks)),
  workerCoreTime: workerTotal,
  workerKernelTime: r2(sum(tasks.map((t) => t.kdurSum || (t.busySum - t.setupSum)))),
  workerSetupTime: r2(sum(tasks.map((t) => t.setupSum))),
  schedCoreTime: schedTotal || null,
  handoff: schedTotal ? r2(schedTotal - workerTotal) : null,
  naiveSum: schedTotal ? r2(schedTotal + workerTotal) : null,
};

return {
  rank: rank,
  traceFile: traceFile,
  swimlane: { spanUs: SPAN, laneNames: laneNames, lanes: lanes, blocks: laneBlocks },
  tasks: tasks,
  scopes: scopes,
  occWindows: occWindows,
  occWindowUs: r2(winW),
  idleRuns: idleRuns,
  idlePct: IDLE_PCT,
  accounting: accounting,
  taskIndex: taskIndex,
  critical: critical,
  scheduler: {
    processNames: processName, lanes: schedLanes, laneStats: schedLaneStats,
    phases: schedPhases, window: schedWindow, busy: schedBusy, blocks: schedBlocks,
    perLaneUtil: r2(sum(schedLaneStats.map((s) => s.util)) / schedLaneStats.length),
  },
  orchestrator: orch,
  readyQueue: readyQueue,
  readyStat: rqStat,
  flows: flowCount,
  hbViolations: hbPairs,
  occupancy: {
    aicUtil: r2(sum(aicLanes.map((l) => l.util)) / aicLanes.length),
    aivUtil: r2(sum(aivLanes.map((l) => l.util)) / aivLanes.length),
    aicWorst: aicLanes.slice().sort((a, b) => a.util - b.util)[0],
    aivWorst: aivLanes.slice().sort((a, b) => a.util - b.util)[0],
    aicBest: aicLanes.slice().sort((a, b) => b.util - a.util)[0],
    aivBest: aivLanes.slice().sort((a, b) => b.util - a.util)[0],
    maxGapLane: lanes.slice().sort((a, b) => b.maxGap - a.maxGap)[0],
  },
};
}

const RANKS = {};
CASE.ranks.forEach((r) => { RANKS[r.key] = buildRank(r); });

/* Findings and inspector defaults are anchored on the first rank's trace. */
const R = RANKS[PRIMARY.key];
if (!csr) {
  const ln = R.swimlane.laneNames;
  caseInfo.numCores = ln.length;
  caseInfo.aicCount = ln.filter((n) => n.indexOf('AIC') === 0).length;
  caseInfo.aivCount = ln.filter((n) => n.indexOf('AIV') === 0).length;
}
const SPAN = R.swimlane.spanUs;
const tasks = R.tasks;
const lanes = R.swimlane.lanes;
const critTags = R.critical.tags;
const critical = R.critical;
const schedPhases = R.scheduler.phases;
const schedLanes = R.scheduler.lanes;
const schedLaneStats = R.scheduler.laneStats;
const schedWindow = R.scheduler.window;
const schedBusy = R.scheduler.busy;
const rqStat = R.readyStat;
const hbPairs = R.hbViolations;

/* ------------------------------------------------------- launch skew
 * Both host logs carry ts= on the same host CLOCK_MONOTONIC (sequential pids
 * on one machine), so the two per-rank device traces — each of which starts
 * at its own t=0 — can be placed on one absolute axis. The offset is then
 * used to PREDICT how long rank0 should sit in each collective wait, and the
 * prediction is checked against the measured span. Nothing here is asserted;
 * the residual is carried through to the UI so the reader can judge the fit. */
const TRACED_INV = 2;
const skewOf = (name) => {
  if (!e2e || RANK_KEYS.length < 2) return null;
  const a = e2e[RANK_KEYS[0]][TRACED_INV] && e2e[RANK_KEYS[0]][TRACED_INV][name];
  const b = e2e[RANK_KEYS[1]][TRACED_INV] && e2e[RANK_KEYS[1]][TRACED_INV][name];
  if (!a || !b || a.tsNs == null || b.tsNs == null) return null;
  return r2((b.tsNs - a.tsNs) / 1000);
};
const launchSkew = (function () {
  /* needs host spans on two ranks of the same host clock */
  if (!e2e || RANK_KEYS.length < 2) return null;
  const runnerUs = skewOf('chip.run.runner_run');
  const chipUs = skewOf('chip.run');
  if (runnerUs == null) return null;
  const waitsOf = (rank) => {
    const m = {};
    RANKS[rank].tasks.filter((t) => /_wait$/.test(t.callable || ''))
      .forEach((t) => { m[t.callable] = t; });
    return m;
  };
  const w0 = waitsOf('rank0');
  const w1 = waitsOf('rank1');
  /* rank0 reaches the collective at its own t=start0; rank1 reaches it at
   * skew + start1 on the same axis. The gap is what rank0 has to wait out. */
  const checks = Object.keys(w0).filter((k) => w1[k]).map((k) => {
    /* rank1 reaches this collective at skew + its own offset, on rank0's axis */
    const arrival1 = r2(runnerUs + w1[k].start);
    const bound = r2(arrival1 - w0[k].start);
    const measured = w0[k].span;
    return {
      callable: k,
      tag0: w0[k].tag, tag1: w1[k].tag,
      start0: w0[k].start, start1: w1[k].start,
      arrival1: arrival1,
      measured: measured,
      bound: bound,
      under: measured <= bound,
      fitPct: bound > 0 ? r2((measured / bound) * 100) : null,
    };
  }).sort((a, b) => b.measured - a.measured);
  const busyOf = (rank, kind) => {
    const R2 = RANKS[rank];
    let busy = 0, n = 0;
    R2.swimlane.lanes.forEach((l, i) => {
      if (l.kind !== kind) return;
      n++;
      busy += (R2.swimlane.blocks[i] || []).reduce((acc, b) => acc + b[1], 0);
    });
    return { busy: r2(busy), lanes: n };
  };
  return {
    inv: TRACED_INV,
    runnerUs: runnerUs,
    chipUs: chipUs,
    ts: {
      rank0: e2e.rank0[TRACED_INV]['chip.run.runner_run'].tsNs,
      rank1: e2e.rank1[TRACED_INV]['chip.run.runner_run'].tsNs,
    },
    checks: checks,
    allUnderBound: checks.every((c) => c.under),
    boundSum: r2(sum(checks.map((c) => c.bound))),
    measuredSum: r2(sum(checks.map((c) => c.measured))),
    waitSum: { rank0: r2(sum(Object.keys(w0).map((k) => w0[k].span))), rank1: r2(sum(Object.keys(w1).map((k) => w1[k].span))) },
    /* the load-imbalance null hypothesis: if rank0 were simply doing more
     * work, its busy time would be larger. Measured straight off the blocks. */
    work: {
      rank0: { aic: busyOf('rank0', 'aic'), aiv: busyOf('rank0', 'aiv') },
      rank1: { aic: busyOf('rank1', 'aic'), aiv: busyOf('rank1', 'aiv') },
    },
    spanDelta: r2(RANKS.rank0.swimlane.spanUs - RANKS.rank1.swimlane.spanUs),
  };
})();
const workDeltaPct = launchSkew
  ? r2(Math.abs(launchSkew.work.rank0.aic.busy - launchSkew.work.rank1.aic.busy)
      / launchSkew.work.rank1.aic.busy * 100)
  : null;

/* ------------------------------------------------------- compiler hints */
const hintLines = rd('report/perf_hints.log').split(/\r?\n/).filter((l) => l.trim());
const hints = hintLines.map((line) => {
  const at = line.match(/ at (\/\S+):(\d+):(\d+)\s*$/);
  const code = (line.match(/\[perf_hint ([A-Z0-9-]+)\]/) || [])[1];
  const occ = +(line.match(/\((\d+) occurrences at this source location\)/) || [0, 1])[1];
  const h = {
    code: code,
    file: at ? at[1].split('/').pop() : null,
    module: at ? at[1].replace(/^.*pypto-lib\//, '') : null,
    line: at ? +at[2] : null,
    col: at ? +at[3] : null,
    occurrences: occ,
  };
  if (code === 'PH001') {
    const m = line.match(/TileInnermostDimGranularity: (\S+) has innermost dim = (\d+)B \(tile (\w+)\[([^\]]+)\], target_memory=(\w+)\)/);
    if (m) { h.op = m[1]; h.innermostB = +m[2]; h.dtype = m[3]; h.tileShape = m[4]; h.mem = m[5]; }
    const mv = line.match(/moves (\d+)B as (\d+) x (\d+)B rows/);
    if (mv) { h.movesB = +mv[1]; h.rows = +mv[2]; h.rowB = +mv[3]; }
    const rec = line.match(/recommended >= (\d+)B for backend (\w+)/);
    if (rec) { h.recB = +rec[1]; h.backend = rec[2]; }
    const cl = line.match(/L2 cache line = (\d+)B/);
    if (cl) h.cacheLineB = +cl[1];
    h.kind = 'tile-granularity';
  } else if (code === 'PH-MR-001') {
    const m = line.match(/requested depth (\d+) for pipeline group (\d+) in (\w+), but only (\d+) of (\d+) buffers fit \((\d+) B per stage, (\d+) B free\)/);
    if (m) {
      h.reqDepth = +m[1]; h.group = +m[2]; h.unit = m[3];
      h.fit = +m[4]; h.of = +m[5]; h.perStageB = +m[6]; h.freeB = +m[7];
    }
    const own = line.match(/would fit depth (\d+) on its own/);
    if (own) h.ownDepth = +own[1];
    h.kind = 'pipeline-depth';
  }
  return h;
});
const ph001 = hints.filter((h) => h.code === 'PH001');
const phmr = hints.filter((h) => h.code === 'PH-MR-001');

const tileSites = {};
for (const h of ph001) {
  const k = h.file + ':' + h.line;
  const s = (tileSites[k] = tileSites[k] || {
    file: h.file, module: h.module, line: h.line, n: 0, occ: 0,
    minB: Infinity, mems: {}, ops: {}, dtypes: {}, recB: h.recB, cacheLineB: h.cacheLineB, shapes: {},
  });
  s.n++; s.occ += h.occurrences;
  if (h.innermostB) s.minB = Math.min(s.minB, h.innermostB);
  s.mems[h.mem] = (s.mems[h.mem] || 0) + 1;
  s.ops[h.op] = (s.ops[h.op] || 0) + 1;
  s.dtypes[h.dtype] = (s.dtypes[h.dtype] || 0) + 1;
  if (h.dtype) s.shapes[h.dtype + '[' + h.tileShape + ']'] = 1;
}
const tileSiteList = Object.keys(tileSites).map((k) => {
  const s = tileSites[k];
  return {
    key: k, file: s.file, module: s.module, line: s.line, n: s.n, occ: s.occ,
    minB: s.minB === Infinity ? null : s.minB, recB: s.recB, cacheLineB: s.cacheLineB,
    mems: s.mems, ops: s.ops, dtypes: s.dtypes, shapes: Object.keys(s.shapes).slice(0, 6),
  };
}).sort((a, b) => (a.minB || 1e9) - (b.minB || 1e9) || b.occ - a.occ);

const tileByFile = {};
ph001.forEach((h) => {
  const f = (tileByFile[h.file] = tileByFile[h.file] || { file: h.file, module: h.module, n: 0, occ: 0, minB: Infinity, sites: {} });
  f.n++; f.occ += h.occurrences;
  if (h.innermostB) f.minB = Math.min(f.minB, h.innermostB);
  f.sites[h.line] = 1;
});
const tileFileList = Object.keys(tileByFile).map((k) => {
  const f = tileByFile[k];
  return { file: f.file, module: f.module, n: f.n, occ: f.occ, minB: f.minB === Infinity ? null : f.minB, siteCount: Object.keys(f.sites).length };
}).sort((a, b) => b.occ - a.occ);

const depthSites = {};
for (const h of phmr) {
  const k = h.file + ':' + h.line;
  const s = (depthSites[k] = depthSites[k] || { file: h.file, module: h.module, line: h.line, groups: [], units: {} });
  s.groups.push({ group: h.group, unit: h.unit, reqDepth: h.reqDepth, fit: h.fit, perStageB: h.perStageB, freeB: h.freeB, ownDepth: h.ownDepth });
  s.units[h.unit] = (s.units[h.unit] || 0) + 1;
}
const depthSiteList = Object.keys(depthSites).map((k) => {
  const s = depthSites[k];
  return {
    key: k, file: s.file, module: s.module, line: s.line,
    units: Object.keys(s.units), groupCount: s.groups.length,
    maxReqDepth: Math.max.apply(null, s.groups.map((g) => g.reqDepth)),
    fittedDepth: Math.max.apply(null, s.groups.map((g) => g.fit)),
    perStageB: Math.max.apply(null, s.groups.map((g) => g.perStageB)),
    freeB: Math.max.apply(null, s.groups.map((g) => g.freeB)),
    groups: s.groups,
  };
}).sort((a, b) => b.groupCount - a.groupCount);

/* on-chip free space per staging unit, as this run's MemoryReuse pass reported it */
const budgets = {};
for (const h of phmr) {
  const b = (budgets[h.unit] = budgets[h.unit] || { unit: h.unit, freeB: 0, sites: 0, minStageB: Infinity, maxStageB: 0 });
  b.freeB = Math.max(b.freeB, h.freeB);
  b.minStageB = Math.min(b.minStageB, h.perStageB);
  b.maxStageB = Math.max(b.maxStageB, h.perStageB);
  b.sites++;
}

/* ---------------------------------------------------------- IR / passes */
const passDir = at('passes_dump');
const passFiles = fs.readdirSync(passDir).filter((f) => f.endsWith('.py')).sort();
let prevLines = null;
const passes = passFiles.map((f) => {
  const text = fs.readFileSync(path.join(passDir, f), 'utf8');
  const lines = text.split('\n').length;
  const row = {
    idx: +f.slice(0, 2),
    name: f.replace(/^\d+_(after_)?/, '').replace(/\.py$/, ''),
    file: f,
    lines: lines,
    delta: prevLines === null ? 0 : lines - prevLines,
    counts: {
      pipeline: (text.match(/pl\.pipeline\(/g) || []).length,
      matmul: (text.match(/pl\.tile\.matmul/g) || []).length,
      spmd: (text.match(/pl\.spmd/g) || []).length,
      range: (text.match(/pl\.range/g) || []).length,
      left: (text.match(/pl\.Mem\.Left/g) || []).length,
      right: (text.match(/pl\.Mem\.Right/g) || []).length,
      acc: (text.match(/pl\.Mem\.Acc/g) || []).length,
      vec: (text.match(/pl\.Mem\.Vec/g) || []).length,
    },
  };
  prevLines = lines;
  return row;
});

const passFileFor = (name) => passFiles.find((f) => f.indexOf('_after_' + name + '.py') >= 0) || null;
const passTextFor = (name) => {
  const f = passFileFor(name);
  return f ? fs.readFileSync(path.join(passDir, f), 'utf8') : null;
};
const autoTile = passTextFor('AutoTileMatmulL0') || '';
const pipelineSites = Array.from(autoTile.matchAll(/pl\.pipeline\(([^)]*)\)/g)).map((m) => {
  const args = m[1];
  const stage = +(args.match(/stage=(\d+)/) || [0, 0])[1];
  const trip = args.split(',')[0].trim();
  const carriers = (args.match(/init_values=\(([^)]*)/) || ['', ''])[1].split(',').filter((s) => s.trim()).length;
  const carrier = ((args.match(/init_values=\(([^,)]*)/) || ['', ''])[1] || '').trim();
  return { stage: stage, trip: trip, carriers: carriers, carrier: carrier };
}).filter((s) => s.stage > 0).sort((a, b) => b.stage - a.stage || String(a.trip).localeCompare(String(b.trip)));

const l0Tiles = (function () {
  const agg = {};
  for (const m of autoTile.matchAll(/pl\.Tile\[\[(\d+), (\d+)\], pl\.(\w+), pl\.Mem\.(Left|Right|Acc)/g)) {
    const rows = +m[1], cols = +m[2], dtype = m[3], mem = m[4];
    const bpe = /INT8|FP8/.test(dtype) ? 1 : /BF16|FP16/.test(dtype) ? 2 : 4;
    const key = mem + '|' + dtype + '|' + rows + 'x' + cols;
    const a = (agg[key] = agg[key] || {
      mem: mem, dtype: dtype, rows: rows, cols: cols,
      bytesPerElem: bpe, bytes: rows * cols * bpe, innermostB: cols * bpe, n: 0,
    });
    a.n++;
  }
  return Object.keys(agg).map((k) => agg[k]).sort((a, b) => b.bytes - a.bytes || b.n - a.n);
})();

const frontend = passFiles.indexOf('00_frontend.py') >= 0
  ? fs.readFileSync(path.join(passDir, '00_frontend.py'), 'utf8') : '';
const finalIr = fs.readFileSync(path.join(passDir, passFiles[passFiles.length - 1]), 'utf8');
const dsl = {
  parallel: (frontend.match(/pl\.parallel/g) || []).length,
  range: (frontend.match(/pl\.range/g) || []).length,
  spmd: (frontend.match(/pl\.spmd/g) || []).length,
  pipeline: (frontend.match(/pl\.pipeline/g) || []).length,
  prefetch: (frontend.match(/pl\.prefetch/g) || []).length,
  functions: (frontend.match(/@pl\.function/g) || []).length,
  matmulFinal: (finalIr.match(/pl\.tile\.matmul/g) || []).length,
  rangeFinal: (finalIr.match(/pl\.range/g) || []).length,
  spmdFinal: (finalIr.match(/pl\.spmd/g) || []).length,
};

/* ----------------------------------------------------- IR excerpt pairs */
function excerpt(file, needle, before, after) {
  const text = fs.readFileSync(path.join(passDir, file), 'utf8').split('\n');
  const i = text.findIndex((l) => l.indexOf(needle) >= 0);
  if (i < 0) return null;
  return text.slice(Math.max(0, i - before), i + after).map((l) => l.replace(/\s+$/, ''));
}
const irPairs = (function () {
  const afterFile = passFileFor('AutoTileMatmulL0');
  if (!afterFile) return [];
  const beforeFile = passFiles[passFiles.indexOf(afterFile) - 1];
  if (!beforeFile) return [];
  const afterLines = autoTile.split('\n');
  const i = afterLines.findIndex((l) => /_l0_init_storage/.test(l) && /pl\.tile\.create/.test(l));
  if (i < 0) return [];
  const base = (afterLines[i].match(/^\s*(\w+)_l0_init_storage/) || [])[1];
  const beforeLines = fs.readFileSync(path.join(passDir, beforeFile), 'utf8').split('\n');
  const j = beforeLines.findIndex((l) => l.indexOf(base) >= 0 && /pl\.tile\.matmul/.test(l));
  const clean = (a) => a.map((l) => l.replace(/\s+$/, '')).filter((l) => l.length);
  return [{
    pass: 'AutoTileMatmulL0',
    passIdx: +afterFile.slice(0, 2),
    subject: base,
    beforeFile: beforeFile,
    afterFile: afterFile,
    before: j < 0 ? [] : clean(beforeLines.slice(j, j + 2)),
    after: clean(afterLines.slice(i, i + 7)),
  }];
})();

/* ------------------------------------------------ Pass change evidence
 *
 * The console used to retain only aggregate IR line counts.  Keep a compact
 * evidence index for every adjacent snapshot instead: exact added/removed
 * line totals, the affected function scopes, and a handful of representative
 * hunks.  This is deliberately built from the very same passes_dump files as
 * the rest of this case, rather than borrowing a second demo's prepared UI
 * data (which could belong to a different run).
 *
 * Unique-line anchors give us stable, useful hunks for these generated Python
 * dumps without shipping all 40–50 full snapshots into the tuning console.
 */
function scopeAt(lines, line) {
  for (let i = Math.min(line, lines.length - 1); i >= 0; i--) {
    const m = lines[i].match(/^\s*def\s+([A-Za-z_]\w*)\s*\(/);
    if (m) return m[1];
  }
  return '<program>';
}

function uniqueAnchors(before, after) {
  const a = new Map(); const b = new Map();
  before.forEach((line, i) => {
    const k = line.trimEnd();
    a.set(k, a.has(k) ? -1 : i);
  });
  after.forEach((line, i) => {
    const k = line.trimEnd();
    b.set(k, b.has(k) ? -1 : i);
  });
  const pairs = [];
  a.forEach((i, k) => {
    const j = b.get(k);
    if (i >= 0 && j >= 0) pairs.push([i, j]);
  });
  pairs.sort((x, y) => x[0] - y[0]);
  const tails = []; const prev = Array(pairs.length).fill(-1);
  pairs.forEach((pair, i) => {
    let lo = 0; let hi = tails.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (pairs[tails[mid]][1] < pair[1]) lo = mid + 1;
      else hi = mid;
    }
    if (lo) prev[i] = tails[lo - 1];
    tails[lo] = i;
  });
  const out = [];
  for (let i = tails.length ? tails[tails.length - 1] : -1; i >= 0; i = prev[i]) out.push(pairs[i]);
  return out.reverse();
}

function passEvidence(before, after) {
  const anchors = [[-1, -1]].concat(uniqueAnchors(before, after), [[before.length, after.length]]);
  let add = 0; let del = 0; let groups = 0;
  const scopeCounts = {}; const hunks = [];
  for (let n = 1; n < anchors.length; n++) {
    const [pa, pb] = anchors[n - 1];
    const [na, nb] = anchors[n];
    const a0 = pa + 1; const b0 = pb + 1;
    const aPart = before.slice(a0, na); const bPart = after.slice(b0, nb);
    if (!aPart.length && !bPart.length) continue;
    groups++;
    del += aPart.length; add += bPart.length;
    const scopes = Array.from(new Set([scopeAt(before, a0), scopeAt(after, b0)]));
    scopes.forEach((scope) => { scopeCounts[scope] = (scopeCounts[scope] || 0) + aPart.length + bPart.length; });
    hunks.push({
      beforeLine: a0 + 1,
      afterLine: b0 + 1,
      before: aPart.slice(0, 5).map((line) => line.trimEnd()),
      after: bPart.slice(0, 5).map((line) => line.trimEnd()),
      beforeMore: Math.max(0, aPart.length - 5),
      afterMore: Math.max(0, bPart.length - 5),
      scopes: scopes,
      weight: aPart.length + bPart.length,
    });
  }
  return {
    add: add, del: del, groups: groups,
    scopes: Object.entries(scopeCounts).sort((x, y) => y[1] - x[1]).slice(0, 8)
      .map(([name, lines]) => ({ name: name, lines: lines })),
    hunks: hunks.sort((x, y) => y.weight - x.weight).slice(0, 6),
  };
}

const passEvidenceIndex = (function () {
  let previous = null;
  return passFiles.map((file, i) => {
    const lines = fs.readFileSync(path.join(passDir, file), 'utf8').split('\n');
    const detail = previous == null ? { add: 0, del: 0, groups: 0, scopes: [], hunks: [] }
      : passEvidence(previous, lines);
    previous = lines;
    return { idx: passes[i].idx, from: i ? passes[i - 1].name : null, ...detail };
  });
})();

/* ------------------------------------------------------------- findings */
const pick = (name) => tasks.filter((t) => t.callable === name)[0];

const waitTasks = tasks.filter((t) => /_wait$/.test(t.callable || '')).sort((a, b) => b.span - a.span);
const waitSpan = r2(sum(waitTasks.map((t) => t.span)));
const mixTasks = tasks.filter((t) => t.kind === 'mix').sort((a, b) => b.span - a.span);
const qkpv = mixTasks[0];
/* the Vec-side counterpart, by convention the same name with an _aiv suffix */
const qkpvPeer = qkpv ? pick(qkpv.callable.replace(/_aic$/, '_aiv')) : null;
const worstImb = tasks.filter((t) => t.blockCount >= 32).sort((a, b) => b.imbalance - a.imbalance)[0];
const worstHandoff = tasks.filter((t) => t.svOverhead != null && t.blockCount >= 8)
  .sort((a, b) => b.svOverhead - a.svOverhead)[0];
const worstSetupShare = tasks.filter((t) => t.blockCount >= 8).sort((a, b) => b.setupShare - a.setupShare)[0];
const aicUtil = R.occupancy.aicUtil;
const aivUtil = R.occupancy.aivUtil;
const schedPerLaneUtil = R.scheduler.perLaneUtil;
const rank0Dev = e2e ? e2e[RANK_KEYS[0]][2]['chip.run.runner_run.device_wall'].us : null;
const rank1Dev = e2e && RANK_KEYS[1] ? e2e[RANK_KEYS[1]][2]['chip.run.runner_run.device_wall'].us : null;
const depthDegraded = depthSiteList.filter((s) => s.fittedDepth < s.maxReqDepth);
const setupHeavy = tasks.filter((t) => t.setupShare > 0.05).sort((a, b) => b.setupSum - a.setupSum);

const CAN = {
  F1: waitTasks.length > 0,
  F2: !!worstHandoff,
  F3: !!launchSkew,
  F4: depthDegraded.length > 0,
  F5: ph001.length > 0 && tileSiteList.length > 0,
  F6: !!(schedPhases.complete && schedPhases.dispatch),
  F7: rqStat.window > 0,
  F8: !!worstImb,
  F9: !!qkpv,
  F10: !!frontend,
};

const findings = [
  CAN.F1 && {
    id: 'F1', level: 'l2', severity: 'high', axis: 'comm',
    title: '通信等待独占关键路径 ' + r2((waitSpan / SPAN) * 100) + '%',
    metric: waitSpan + ' us / ' + SPAN + ' us',
    claim: waitTasks.length + ' 个 *_wait 任务合计 ' + waitSpan + ' us，全部单块单核，其中 '
      + waitTasks[0].callable + ' 单独 ' + waitTasks[0].span + ' us；关键路径 ' + critTags.length + ' 个节点里通信与 publish 段占主导。',
    evidence: [
      { artifact: 'merged_swimlane (rank0/d0)', locator: waitTasks.map((t) => t.tag).join(' / '), value: waitTasks.map((t) => t.callable + '=' + t.span + 'us').join(', ') },
      { artifact: 'critical path (fanin/fanout hints)', locator: 'chain ' + critTags.length + ' nodes', value: 'span sum ' + critical.spanSum + ' us' },
    ],
    focus: { view: 'l2', task: waitTasks[0].tag, critOnly: true },
    lever: '按通信算子优先级推进：先确认算法选择（allgather vs 分块 readback），再做通算重叠，最后才调块大小与乒乓。',
    guardrail: '必须同时报告带宽利用率与 overlap 效率；只压缩 wait 时长而不看带宽，会把等待搬到别处。',
    verify: '重测 device_wall 与该 wait 任务 span，并确认 *_wait 仍在关键路径上。',
  },
  CAN.F2 && {
    id: 'F2', level: 'l1', severity: 'high', axis: 'launch',
    title: worstHandoff.callable + ' hand-off 比核上计算还贵（+' + worstHandoff.svOverhead + ' us/块）',
    metric: 'AICPU ' + worstHandoff.svAicpuMean + ' us vs 核上 ' + worstHandoff.durMean + ' us',
    claim: '同一个块有两个视角：Worker View 记核上 ' + worstHandoff.durMean + ' us，Scheduler View 记 dispatch→finish '
      + worstHandoff.svAicpuMean + ' us，差 ' + worstHandoff.svOverhead + ' us 是领取与依赖等待。核上那 '
      + worstHandoff.durMean + ' us 里还有 ' + worstHandoff.setupMean + ' us（' + r2(worstHandoff.setupShare * 100)
      + '%）是 local_setup 而非 kernel；' + worstHandoff.blockCount + ' 块累计 setup ' + worstHandoff.setupSum + ' us。',
    evidence: [
      { artifact: 'Scheduler View (pid 3)', locator: worstHandoff.tag + ' (' + worstHandoff.callable + ')', value: 'aicpu-duration mean ' + worstHandoff.svAicpuMean + ' us, max ' + worstHandoff.svAicpuMax + ' us' },
      { artifact: 'Worker View (pid 4)', locator: 'duration − kernel_duration', value: 'setup mean ' + worstHandoff.setupMean + ' us, kernel mean ' + r2(worstHandoff.kdurSum / worstHandoff.blockCount) + ' us' },
      { artifact: 'same trace', locator: 'setup 占比 > 5% 的任务', value: setupHeavy.length + ' 个，最高 ' + worstSetupShare.callable + ' ' + r2(worstSetupShare.setupShare * 100) + '%' },
    ],
    focus: { view: 'l1', task: worstHandoff.tag },
    lever: '把 publish 段与它的生产者合进同一 mixed kernel，消掉一次 AICPU hand-off；setup 里可复用的准备提到核外或跨块复用。',
    guardrail: '合核会拉长单核占用；合并后要复查该核是否变成新的独占瓶颈。',
    verify: '重测该任务的 aicpu-duration、setup mean 与所在核 util；hand-off 差值应收窄且核 util 不恶化。',
  },
  CAN.F3 && {
    id: 'F3', level: 'e2e', severity: 'high', axis: 'balance',
    title: 'rank1 晚发 ' + launchSkew.runnerUs + ' us，rank0 在集合点替它等',
    metric: 'span +' + launchSkew.spanDelta + ' us，AIC 占用 −'
      + r2(RANKS.rank1.occupancy.aicUtil - RANKS.rank0.occupancy.aicUtil) + ' pt',
    claim: 'rank0 的 trace 跨度比 rank1 长 ' + launchSkew.spanDelta + ' us，AIC 占用却低 '
      + r2(RANKS.rank1.occupancy.aicUtil - RANKS.rank0.occupancy.aicUtil)
      + ' pt——更慢却更闲，多出来的是空转。负载不均的零假设不成立：两卡任务数、块数相同，'
      + 'AIC busy 相差仅 ' + workDeltaPct + '%。两份 host log 的 ts 同属一个 host 单调钟，'
      + '对齐后 rank1 的 runner_run 比 rank0 晚 ' + launchSkew.runnerUs + ' us；把这个偏移加到 rank1 '
      + '各 *_wait 的到达时刻上，就得到 rank0 在该点最多能等多久的上界——'
      + launchSkew.checks.length + ' 个 wait ' + (launchSkew.allUnderBound ? '全部落在上界内' : '有超出上界的项')
      + '，合计 ' + launchSkew.measuredSum + ' / ' + launchSkew.boundSum + ' us，'
      + '两个主导项贴到上界的 ' + launchSkew.checks[0].fitPct + '% 与 ' + launchSkew.checks[1].fitPct + '%。',
    evidence: [
      {
        artifact: 'host.2263908.log / host.2263922.log',
        locator: 'inv=' + launchSkew.inv + ' chip.run.runner_run ts（host CLOCK_MONOTONIC）',
        value: 'rank0 ' + launchSkew.ts.rank0 + ' ns，rank1 ' + launchSkew.ts.rank1
          + ' ns → rank1 晚 ' + launchSkew.runnerUs + ' us（chip.run 口径 ' + launchSkew.chipUs + ' us）',
      },
      {
        artifact: 'merged_swimlane blocks（两 rank）',
        locator: 'AIC / AIV busy 由块时长直接加总，非占用率反推',
        value: 'AIC ' + launchSkew.work.rank0.aic.busy + ' vs ' + launchSkew.work.rank1.aic.busy
          + ' us（差 ' + workDeltaPct + '%），AIV ' + launchSkew.work.rank0.aiv.busy + ' vs '
          + launchSkew.work.rank1.aiv.busy + ' us；两卡同为 ' + RANKS.rank0.tasks.length + ' 任务',
      },
    ].concat(launchSkew.checks.map((c) => ({
      artifact: 'merged_swimlane（两 rank）',
      locator: c.callable + '：rank0 到达 ' + c.start0 + ' us，rank1 到达 ' + c.start1
        + ' us（对齐后 ' + c.arrival1 + ' us）',
      value: '上界 ' + c.bound + ' us，实测 ' + c.measured + ' us'
        + (c.under ? '（占上界 ' + c.fitPct + '%）' : '（超出上界）'),
    }))).concat([
      {
        artifact: 'both host logs',
        locator: 'inv=1 graph_build',
        value: 'rank0 ' + e2e.rank0[1]['chip.run.runner_run.device_wall.graph_build'].us + ' us vs rank1 '
          + e2e.rank1[1]['chip.run.runner_run.device_wall.graph_build'].us + ' us — 首次 JIT 建图，不能当稳定态',
      },
      {
        artifact: '未拆解',
        locator: 'host runner_run vs device_wall（rank0 inv=2）',
        value: e2e.rank0[2]['chip.run.runner_run'].us + ' us（host 钟）对 '
          + e2e.rank0[2]['chip.run.runner_run.device_wall'].us
          + ' us（设备钟）；错峰产生在这段主机时间里，dump 内无更细的 span 可归因',
      },
    ]),
    focus: { view: 'e2e' },
    lever: '对齐两卡的下发时刻（同步 launch、收紧 host 侧提交路径），而不是去调 rank0 的 kernel——它的计算量与 rank1 相同。',
    guardrail: '上界只说明「等待能被错峰解释」，不证明错峰是唯一成因；时钟对齐是从同一主机的 mono ts 推的，dump 里没有显式跨 rank 同步记录；本次仅 2 次调用，偏移量本身没有分布。',
    verify: '固定 case 重跑 ≥10 次，每次记录两卡 runner_run 的 ts 差与 rank0 的 *_wait 合计；偏移收窄，等待应同比收窄。',
  },
  CAN.F4 && {
    id: 'F4', level: 'compiler', severity: 'high', axis: 'pipeline',
    title: depthDegraded.length + ' 处软流水深度被降到 ' + depthDegraded[0].fittedDepth,
    metric: phmr.length + ' 条 PH-MR-001 / ' + depthDegraded.length + ' 个源码点',
    claim: 'MemoryReuse 报告：请求 depth ' + Array.from(new Set(depthSiteList.map((s) => s.maxReqDepth))).sort().join('/')
      + '，实际只有 1 个 buffer 放得下，相距 1 个 stage 的操作共享存储并串行化。'
      + 'Left/Right 每 stage ' + budgets.Right.minStageB + '–' + budgets.Right.maxStageB + ' B，可用 ' + budgets.Right.freeB + ' B；Vec 每 stage ' + budgets.Vec.minStageB + ' B，可用 ' + budgets.Vec.freeB + ' B。',
    evidence: depthSiteList.slice(0, 4).map((s) => ({
      artifact: 'report/perf_hints.log', locator: s.module + ':' + s.line,
      value: 'depth ' + s.maxReqDepth + '→' + s.fittedDepth + '，' + s.groupCount + ' 组 @' + s.units.join('/') + '，' + s.perStageB + ' B/stage，' + s.freeB + ' B free',
    })),
    focus: { view: 'compiler', pass: 'MemoryReuse' },
    lever: '先减少同驻 tile（更小或更少 co-live 操作数），再谈调 stage；Left/Right 是编译器的 L0A/L0B staging 结果，不要当独立 Tile 预算去调。',
    guardrail: '盲目加大 stage 只会让 MemoryReuse 再降一次深度，并多出一条同样的提示。',
    verify: '改后重跑编译，确认该源码点不再出现 PH-MR-001，并复测该 kernel 的块时长。',
  },
  CAN.F5 && {
    id: 'F5', level: 'compiler', severity: 'medium', axis: 'granularity',
    title: sum(ph001.map((h) => h.occurrences)) + ' 次搬运末维 < ' + ph001[0].cacheLineB + 'B cache line',
    metric: '最小 ' + tileSiteList[0].minB + 'B，覆盖 ' + tileFileList.length + ' 个算子文件',
    claim: 'TileInnermostDimGranularity 在 ' + tileSiteList.length + ' 个源码点上报 tile.load/tile.store 末维不足一个 L2 cache line，最极端处只有 '
      + tileSiteList[0].minB + 'B。末维碎片化会让每次搬运只拿到 cache line 的一小部分。',
    evidence: tileFileList.slice(0, 5).map((f) => ({
      artifact: 'report/perf_hints.log', locator: f.module,
      value: f.occ + ' 次 / ' + f.siteCount + ' 个源码点，最小末维 ' + f.minB + 'B',
    })),
    focus: { view: 'compiler', tab: 'granularity' },
    lever: '按 dtype 把末维凑到 512B：BF16 → 256 元素倍数，FP32 → 128，INT8 → 512。',
    guardrail: '加大末维会同时抬高 L0/UB 占用，可能触发 F4 的深度回退；两项要一起看。',
    verify: '重编译后核对 PH001 条数与最小末维，并复测对应 kernel 的 MTE 时间。',
  },
  CAN.F6 && {
    id: 'F6', level: 'l2', severity: 'medium', axis: 'sched',
    title: 'AICPU 调度器平均占用 ' + schedPerLaneUtil + '%',
    metric: schedBusy + ' us busy / ' + schedLanes.length + ' 个调度线程',
    claim: schedLanes.length + ' 个调度线程在 ' + r2(schedWindow.hi - schedWindow.lo) + ' us 窗口内合计 busy ' + schedBusy
      + ' us。complete 阶段最贵：' + schedPhases.complete.us + ' us 处理 ' + schedPhases.complete.tasks
      + ' 次完成，约 ' + schedPhases.complete.usPerTask + ' us/次；dispatch ' + schedPhases.dispatch.usPerTask + ' us/次。',
    evidence: Object.keys(schedPhases).sort((a, b) => schedPhases[b].us - schedPhases[a].us).map((k) => ({
      artifact: 'merged_swimlane scheduler lane', locator: 'phase=' + k,
      value: schedPhases[k].us + ' us / ' + schedPhases[k].n + ' 段 / ' + schedPhases[k].tasks + ' 任务'
        + (schedPhases[k].usPerTask ? ' = ' + schedPhases[k].usPerTask + ' us/任务' : ''),
    })),
    focus: { view: 'l2', overlay: 'sched' },
    lever: '合并相邻核、把外层迭代折进核内，或用 pl.spmd 一次 fan-out 多块，降低完成回收次数。',
    guardrail: 'A3/910C 上「约 50us 级内核」只是实测启发式，不是跨芯片硬规则；合核到多长要按本 case 实测。',
    verify: '重测 complete 段总时长与任务数；单任务代价不变而次数下降才算生效。',
  },
  CAN.F7 && {
    id: 'F7', level: 'l2', severity: 'medium', axis: 'sched',
    title: 'AIC ready-but-undispatched 占窗口 ' + rqStat.busyShare.AIC + '%',
    metric: 'avg ' + rqStat.avg.AIC + ' / peak ' + rqStat.peak.AIC,
    claim: 'shared_ready_queue 在 ' + rqStat.busyTime.AIC + ' us（窗口的 ' + rqStat.busyShare.AIC
      + '%）里有 AIC 任务已 ready 但未派发，峰值 ' + rqStat.peak.AIC + ' 个；AIV 侧 ' + rqStat.busyShare.AIV + '%，峰值 ' + rqStat.peak.AIV + '。'
      + '同期 AIC 平均占用只有 ' + aicUtil + '%。',
    evidence: [
      { artifact: 'merged_swimlane queue counter', locator: 'shared_ready_queue', value: 'AIC avg ' + rqStat.avg.AIC + ', peak ' + rqStat.peak.AIC + ', >0 占 ' + rqStat.busyShare.AIC + '%' },
      { artifact: 'worker lanes', locator: 'AIC 平均 vs AIV 平均', value: aicUtil + '% vs ' + aivUtil + '%' },
      { artifact: 'flow events', locator: 'hb_violation', value: hbPairs.length + ' 对 happens-before 违例标记' },
    ],
    focus: { view: 'l2', overlay: 'ready' },
    lever: '只针对已观测到的关键路径提前 dispatch 或调整依赖，不要全局提前。',
    guardrail: '提前 dispatch、改依赖、延后非关键任务都可能以吞吐换时延；两个指标都要报。',
    verify: '重测 ready>0 时间占比、AIC 平均占用与 device_wall；三者要同向改善。',
  },
  CAN.F8 && {
    id: 'F8', level: 'l1', severity: 'medium', axis: 'balance',
    title: worstImb.callable + ' 块时长离散 ' + worstImb.imbalance + 'x',
    metric: 'max ' + worstImb.durMax + ' us / med ' + worstImb.durMed + ' us',
    claim: worstImb.blockCount + ' 个块摊到 ' + worstImb.coreCount + ' 核（约 '
      + r2(worstImb.blockCount / worstImb.coreCount) + ' 波），最慢块 ' + worstImb.durMax
      + ' us 是中位块的 ' + worstImb.imbalance + ' 倍，尾块决定该任务 span ' + worstImb.span + ' us。',
    evidence: [
      { artifact: 'merged_swimlane blocks', locator: worstImb.tag + ' (' + worstImb.callable + ')', value: 'min ' + worstImb.durMin + ' / med ' + worstImb.durMed + ' / p90 ' + worstImb.durP90 + ' / max ' + worstImb.durMax + ' us' },
      { artifact: 'deps.json', locator: 'task ' + worstImb.id, value: 'block_num=' + worstImb.blockNum + ', scope=' + worstImb.scope },
    ],
    focus: { view: 'l1', task: worstImb.tag },
    lever: '独立循环用 pl.parallel 而非 pl.range；尾块偏长说明每块工作量不齐，需要重新切分而不是加核。',
    guardrail: '先确认慢块是工作量差异还是 MTE/UB 争用；PMU 打开会改变调度，不能与 PMU-off 基线直接比较。',
    verify: '重测该任务 durMax/durMed 与 span；离散度下降且 span 缩短才算生效。',
  },
  CAN.F9 && {
    id: 'F9', level: 'l1', severity: 'medium', axis: 'fusion',
    title: qkpv.callable + ' 混合核跨 ' + qkpv.coreCount + ' 核，单块 ' + qkpv.durMean + ' us',
    metric: 'span ' + qkpv.span + ' us，核上 ' + qkpv.durMean + ' us/块',
    claim: '本 case 只有 ' + tasks.filter((t) => t.kind === 'mix').length + ' 个 mixed kernel 横跨全部 '
      + qkpv.coreCount + ' 个核（AIC+AIV 同核），这是其中最贵的一个：span '
      + qkpv.span + ' us 而单块 ' + qkpv.durMean + ' us，说明 span 几乎等于单块时长，Cube 与 Vec 段是串在块内的。'
      + (qkpvPeer ? '同段还有 ' + qkpvPeer.tag + '（' + qkpvPeer.callable + '）作为 Vec 侧对应体。' : ''),
    evidence: [
      { artifact: 'merged_swimlane', locator: qkpv.tag + ' (' + qkpv.callable + ')', value: qkpv.blockCount + ' 块 / ' + qkpv.coreCount + ' 核，min ' + qkpv.durMin + ' / med ' + qkpv.durMed + ' / max ' + qkpv.durMax + ' us' },
      { artifact: 'deps.json', locator: 'task ' + qkpv.id, value: 'block_num=' + qkpv.blockNum + '，前驱 ' + qkpv.pred.length + ' 个，后继 ' + qkpv.succ.length + ' 个' },
      { artifact: 'critical path', locator: '是否在关键路径上', value: critTags.indexOf(qkpv.tag) >= 0 ? '在，第 ' + (critTags.indexOf(qkpv.tag) + 1) + ' 个节点' : '不在' },
    ],
    focus: { view: 'l1', task: qkpv.tag },
    lever: '按 Flash Attention 的 Cube→Vec→Cube→Vec 解耦思路，用 GM FIFO 让两侧真正并行，而不是块内串行。',
    guardrail: 'S1_TILE、预加载深度、FIFO 槽数与 UB 预算必须一起推导；只单独扫 Tile 会在 F4 的深度回退上撞墙。',
    verify: '重测该任务 span 与单块时长的比值；解耦生效后 span 应显著小于 块数 × 单块时长 / 核数。',
  },
  CAN.F10 && {
    id: 'F10', level: 'l2', severity: 'low', axis: 'reuse',
    title: '本程序 0 处 pl.prefetch，L2 复用杠杆未启用',
    metric: 'prefetch ' + dsl.prefetch + ' / pipeline ' + dsl.pipeline + ' / spmd ' + dsl.spmd + ' / parallel ' + dsl.parallel,
    claim: '前端 IR 里 pl.prefetch 出现 ' + dsl.prefetch + ' 次。权重类输入确实存在（'
      + caseInfo.params.filter((p) => /^w/.test(p.name)).length + ' 个 w* 参数），但没有任何静态预取，也没有 N-group swizzle 证据。',
    evidence: [
      { artifact: 'passes_dump/00_frontend.py', locator: 'pl.prefetch', value: dsl.prefetch + ' 处' },
      { artifact: 'distributed_meta.json', locator: 'w* 参数', value: caseInfo.params.filter((p) => /^w/.test(p.name)).map((p) => p.name.replace(/__ssa_v0$/, '') + ' ' + p.dtype + JSON.stringify(p.shape)).slice(0, 4).join(', ') },
    ],
    focus: { view: 'l2', overlay: 'none' },
    lever: '只对静态、确定会冷、且所有在途 warm 数据能放进 L2 的权重用 pl.prefetch。',
    guardrail: 'prefetch 占 SDMA；错误预取会拖慢通信或挤掉真正需要的数据。本 case 通信已是瓶颈（F1），风险更高。',
    verify: '加预取后同时看 device_wall、通信段 span 与 SDMA 占用，三者不能互相恶化。',
  },
].filter(Boolean);

/* --------------------------------------------------- finding -> subjects
 * A finding is only useful if the reader can see it on the stage. Each one
 * names the concrete objects the centre view should mark (tasks, source
 * sites, scheduler phases), so the stage can number them instead of leaving
 * the reader to guess which parts the inspector is talking about.          */
const taskByTag = {};
tasks.forEach((t) => { taskByTag[t.tag] = t; });

const SUBJECTS = {
  F1: {
    view: 'l2',
    tasks: waitTasks.map((t) => t.tag),
  },
  F2: {
    view: 'l1',
    tasks: worstHandoff ? [worstHandoff.tag] : [],
  },
  F3: {
    view: 'e2e',
    ranks: RANK_KEYS.length > 1 ? RANK_KEYS.slice(0, 2) : [],
  },
  F4: {
    view: 'compiler', tab: 'depth',
    sites: depthSiteList.map((s) => s.key),
  },
  F5: {
    view: 'compiler', tab: 'granularity',
    sites: tileSiteList.slice(0, 12).map((s) => s.key),
    files: tileFileList.map((f) => f.file),
  },
  F6: {
    view: 'l2', overlay: 'sched',
    schedPhases: ['complete', 'dispatch'],
  },
  F7: {
    view: 'l2', overlay: 'ready',
    lanes: lanes.filter((l) => l.kind === 'aic').sort((a, b) => a.util - b.util).slice(0, 6).map((l) => l.name),
  },
  F8: {
    view: 'l1',
    tasks: worstImb ? [worstImb.tag] : [],
  },
  F9: {
    view: 'l1',
    tasks: qkpv ? [qkpv.tag] : [],
  },
  F10: {
    view: 'l2',
    absent: true,
  },
};
findings.forEach((f) => {
  const s = SUBJECTS[f.id] || {};
  f.subjects = {
    view: s.view || (f.focus && f.focus.view) || 'l2',
    tab: s.tab || null,
    overlay: s.overlay || null,
    tasks: s.tasks || [],
    lanes: s.lanes || [],
    sites: s.sites || [],
    files: s.files || [],
    ranks: s.ranks || [],
    schedPhases: s.schedPhases || [],
    absent: !!s.absent,
  };
  /* chips shown in the centre evidence bar, each one jumpable */
  f.chips = []
    .concat(f.subjects.tasks.map((tag) => {
      const t = taskByTag[tag];
      return { kind: 'task', id: tag, label: t ? t.callable : tag, value: t ? t.span + ' us' : '' };
    }))
    .concat(f.subjects.sites.map((key) => {
      const d = depthSiteList.find((x) => x.key === key);
      const g = tileSiteList.find((x) => x.key === key);
      return {
        kind: 'site', id: key, label: key,
        value: d ? 'depth ' + d.maxReqDepth + '→' + d.fittedDepth : (g ? g.minB + 'B' : ''),
      };
    }))
    .concat(f.subjects.lanes.map((name) => {
      const l = lanes.find((x) => x.name === name);
      return { kind: 'lane', id: name, label: name, value: l ? r2(l.util) + '%' : '' };
    }))
    .concat(f.subjects.ranks.map((r) => ({
      kind: 'rank', id: r, label: r,
      value: e2e[r][2]['chip.run.runner_run.device_wall'].us + ' us',
    })))
    .concat(f.subjects.schedPhases.map((p) => ({
      kind: 'phase', id: p, label: 'phase ' + p,
      value: schedPhases[p] ? schedPhases[p].us + ' us' : '',
    })));
});

/* Cross-layer links are emitted only where this run has direct evidence.
 * They keep Pass inspection inside the tuning loop, not beside it. */
const findingIds = new Set(findings.map((f) => f.id));
passEvidenceIndex.forEach((detail) => {
  const pass = passes.find((p) => p.idx === detail.idx);
  detail.links = [];
  if (!pass) return;
  if (pass.name === 'MemoryReuse' && findingIds.has('F4')) {
    detail.links.push({ findingId: 'F4', label: '关联 F4 · 软流水深度回退' });
  }
  if (pass.name === 'AutoTileMatmulL0' && l0Tiles.length) {
    detail.links.push({ view: 'isa', label: '查看 ISA / 布局中的 L0 tile' });
  }
});

/* Investigations are the product-level objects built from this run's raw
 * findings. Views remain evidence lenses; the task, its hypothesis and its
 * experiment are what a developer actually carries through a tuning loop. */
const investigationIds = new Set(findings.map((f) => f.id));
const includeFinding = (id) => investigationIds.has(id) ? id : null;
const compactIds = (ids) => ids.filter(Boolean);
const investigations = [];
const addInvestigation = (id, title, status, target, findingRefs, hypotheses, experiments) => {
  investigations.push({
    id, title, status, target,
    baseline: '当前 run · ' + caseInfo.program,
    owner: '待分派',
    findings: compactIds(findingRefs), hypotheses, experiments,
  });
};

if (investigationIds.has('F3') && investigationIds.has('F1')) {
  addInvestigation('INV-024', '解释 rank 间尾部延迟', '需要实验',
    '缩短关键路径上的通信等待，并验证是否改善端到端尾部。',
    [includeFinding('F3'), includeFinding('F1'), includeFinding('F7'), includeFinding('F6')],
    [
      { id: 'H-01', title: 'rank 启动错位放大集合通信等待', level: '强支持',
        claim: 'F3 的 launch skew 与 F1 的关键路径 wait 有同一条跨层时序锚点；它解释等待来源，但不排除其他成因。',
        evidence: ['F3', 'F1'], need: '在不改通信算法的前提下，缩小启动错位后 wait span 是否同步下降。' },
      { id: 'H-02', title: '调度队列可能是额外贡献因素', level: '待区分',
        claim: 'F7/F6 同时出现，只能作为竞争解释，不能升级为 H-01 的因果前提。',
        evidence: compactIds([includeFinding('F7'), includeFinding('F6')]),
        need: '比较局部 dispatch 调整前后 ready>0 占比、complete 次数与 device wall。' },
    ],
    [{ id: 'EXP-024-01', status: '待执行', name: '只调整关键 rank 的启动 / dispatch 时序',
      change: '不改通信算法、Tile、融合边界', measures: 'rank start skew · wait span · device wall',
      guardrail: '结果校验、吞吐、host bind 时间' }]);
} else if (investigationIds.has('F7') && investigationIds.has('F6')) {
  addInvestigation('INV-031', '解释 AIC 任务已就绪但未派发', '需要实验',
    '确认 ready queue 堵塞是否对 device wall 有可观测贡献。',
    [includeFinding('F7'), includeFinding('F6'), includeFinding('F9')],
    [
      { id: 'H-01', title: '关键路径上的依赖 / dispatch 时机造成队列积压', level: '待验证',
        claim: 'ready queue 指标说明 AIC 任务可运行但未派发；不能单独证明调度器是根因。', evidence: ['F7'],
        need: '仅改变关键路径任务的 dispatch 时机，重测 queue 与 device wall。' },
      { id: 'H-02', title: 'AICPU complete 开销是竞争解释', level: '待区分',
        claim: 'complete 阶段的工作量可能延后派发，需要独立测量而不是与 queue 告警合并。', evidence: ['F6'],
        need: '记录 complete 次数和单位任务开销是否与等待窗口同向变化。' },
    ],
    [{ id: 'EXP-031-01', status: '待执行', name: '关键路径局部提前 dispatch',
      change: '不改变任务数量与 fusion 边界', measures: 'ready>0 · AIC util · device wall',
      guardrail: '吞吐与正确性同基线回归' }]);
} else {
  const firstFinding = findings[0];
  addInvestigation('INV-001', '建立首个可证伪的性能假设', '待分诊',
    '用一个可回滚改动验证最高影响发现是否能改善端到端结果。', [firstFinding && firstFinding.id],
    [{ id: 'H-01', title: '最高优先级发现值得进一步验证', level: '待建模',
      claim: '当前只有同层观测，尚未形成跨层解释。', evidence: [firstFinding && firstFinding.id].filter(Boolean),
      need: '先绑定端到端指标和最小改动，再开始实验。' }],
    [{ id: 'EXP-001-01', status: '待规划', name: '定义最小单变量实验', change: '待选择',
      measures: '局部指标 · device wall · 正确性', guardrail: '保持同一基线与采样条件' }]);
}

if (investigationIds.has('F2') || investigationIds.has('F9')) {
  addInvestigation('INV-025', '评估混合核的任务边界', '待分诊',
    '判断 hand-off 或块内串行是否值得以融合 / 解耦方式处理。',
    [includeFinding('F2'), includeFinding('F9'), includeFinding('F8')],
    [{ id: 'H-01', title: '任务边界引入可避免的 hand-off 或串行段', level: '待建模',
      claim: '先区分任务领取、依赖等待和核内串行，再选择 fusion 或 FIFO 方案。',
      evidence: compactIds([includeFinding('F2'), includeFinding('F9')]),
      need: '定位最小 scope 并做一个只改变边界的对照实验。' }],
    [{ id: 'EXP-025-01', status: '待规划', name: '选择一个 kernel scope 建立对照', change: '待确认',
      measures: '任务 span · hand-off · util', guardrail: '避免形成新的独占核' }]);
}

if (investigationIds.has('F4') || investigationIds.has('F5')) {
  addInvestigation('INV-026', '处理 Tile 与流水资源约束', '需要补证',
    '确认搬运粒度改动是否会触发流水深度回退。', [includeFinding('F4'), includeFinding('F5')],
    [{ id: 'H-01', title: '粒度与流水深度受同一 L0 / UB 预算约束', level: '约束耦合',
      claim: 'F5 的修改可能触发 F4；它是 guardrail 关系，尚不是性能因果结论。',
      evidence: compactIds([includeFinding('F4'), includeFinding('F5')]),
      need: '在同一个 source scope 对比 Tile 预算、PH-MR-001 和 MTE 时间。' }],
    [{ id: 'EXP-026-01', status: '待规划', name: '同 scope 的 Tile 预算对照',
      change: '只调整末维或 pipeline depth 之一', measures: 'PH 提示 · MTE · L0/UB 预算',
      guardrail: '不接受深度回退换来的局部收益' }]);
}

/* ---------------------------------------------------------------- write */
const payload = {
  generatedBy: 'Design/operator-tuning-console/build-data.cjs',
  source: caseInfo.runDir,
  case: caseInfo,
  e2e: e2e,
  ranks: RANKS,
  defaultRank: PRIMARY.key,
  hints: hints,
  tileSites: tileSiteList,
  tileFiles: tileFileList,
  depthSites: depthSiteList,
  budgets: budgets,
  passes: passes,
  passEvidence: passEvidenceIndex,
  pipelineSites: pipelineSites,
  l0Tiles: l0Tiles,
  dsl: dsl,
  irPairs: irPairs,
  findings: findings,
  investigations: investigations,
  launchSkew: launchSkew,
  derived: {
    waitTasks: waitTasks.map((t) => t.tag), waitSpan: waitSpan,
    setupHeavy: setupHeavy.map((t) => t.tag),
    worstImb: worstImb.tag, worstHandoff: worstHandoff.tag, worstSetupShare: worstSetupShare.tag,
  },
};

console.log(CASE.id + '  ' + CASE.sub);
RANK_KEYS.forEach((k) => {
  const x = RANKS[k];
  console.log('   ', k, 'span', x.swimlane.spanUs, 'us | tasks', x.tasks.length,
    '| blocks', sum(x.swimlane.blocks.map((a) => a.length)),
    '| crit', x.critical.tags.length, '| AIC', x.occupancy.aicUtil + '%', 'AIV', x.occupancy.aivUtil + '%');
});
console.log('    hints', hints.length, '| passes', passes.length, '| findings', findings.length,
  '| e2e', e2e ? Object.keys(e2e[RANK_KEYS[0]] || {}).length + ' inv' : 'absent');
return payload;
}

const BUILT = {};
CASES.forEach((c) => { BUILT[c.id] = buildCase(c); });

fs.writeFileSync(OUT, 'window.TUNING_RUNS = ' + JSON.stringify(BUILT) + ';\n'
  + 'window.TUNING_CASES = ' + JSON.stringify(CASES.map((c) => ({ id: c.id, label: c.label, sub: c.sub }))) + ';\n'
  + 'window.TUNING_RUN = window.TUNING_RUNS[' + JSON.stringify(CASES[0].id) + '];\n');
console.log('wrote', OUT, (fs.statSync(OUT).size / 1024).toFixed(1) + ' KB');
