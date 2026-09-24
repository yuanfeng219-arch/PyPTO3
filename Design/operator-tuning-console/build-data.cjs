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
    /* the model source this run was compiled from; absolute, outside the dump */
    sourceDir: path.join(DATA, 'DeepseekV4/deepseek_v4_flash_dspark'),
    entryModule: 'decode_csa.py',
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
    /* the Qwen3 tree this was compiled from is not in the repo */
    sourceDir: null,
    entryModule: null,
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
  /* No dump carries a kernel -> source map. For decode_csa one is rebuilt
   * from the model source by name_hint (see sourceMap); without the source
   * tree — decode_fwd_layers — there is none. */
  hasKernelSourceMap: false,
  kernelSourceRebuilt: !!(CASE.sourceDir && fs.existsSync(CASE.sourceDir)),
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
    /* NOT redundant with the task's funcId: a mixed scope's two halves share
     * one taskId but carry different FuncIds on their blocks. */
    fid: +(hint.match(/FuncId:(-?\d+)/) || [0, -1])[1],
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

  /* ------------------------------------------------ scope -> kernel(s)
   * ExpandMixedKernel splits a mixed InCore function into an AIC and an AIV
   * kernel wrapped in a Group. The scheduler launches the Group, so both
   * halves share ONE taskId -- but their blocks carry different FuncIds.
   * Keying only on taskId therefore files the Vec half's core-time under the
   * Cube half's name. Regroup by FuncId so the pair is visible. */
  const kAcc = {};
  b.forEach((x) => {
    const k = (kAcc[x.fid] = kAcc[x.fid] || {
      funcId: x.fid, name: nameMap.callable_id_to_name[String(x.fid)] || t.rawName,
      engine: null, blocks: 0, cores: new Set(),
      coreTime: 0, kernelTime: 0, setupTime: 0, durs: [],
    });
    const en = laneNames[x.lane].indexOf('AIC') === 0 ? 'aic' : 'aiv';
    k.engine = k.engine === null || k.engine === en ? en : 'mix';
    k.blocks += 1; k.cores.add(x.core);
    k.coreTime += x.dur; k.kernelTime += x.kdur; k.setupTime += x.setup;
    k.durs.push(x.dur);
  });
  const kernels = Object.keys(kAcc).map((f) => {
    const k = kAcc[f];
    const d = k.durs.slice().sort((a, c) => a - c);
    return {
      funcId: k.funcId, name: k.name, engine: k.engine,
      blocks: k.blocks, cores: k.cores.size,
      coreTime: r2(k.coreTime), kernelTime: r2(k.kernelTime), setupTime: r2(k.setupTime),
      durMed: r2(d[Math.floor(d.length / 2)]), durMax: r2(d[d.length - 1]),
      durMean: r2(k.coreTime / k.blocks),
    };
  }).sort((a, c) => c.coreTime - a.coreTime);

  /* The scope is the source-level pl.spmd region; the kernels are what the
   * device actually launched. They are 1:1 except for mixed scopes, where the
   * two halves share the name_hint the scope was written with. */
  const scopeName = (function () {
    const ns = kernels.map((k) => k.name);
    if (ns.length === 1) return ns[0];
    const bases = Array.from(new Set(ns.map((n) => n.replace(/_(aic|aiv)$/, ''))));
    return bases.length === 1 ? bases[0] : ns.slice().sort().join('+');
  })();

  const engineOf = (en) => {
    const ks = kernels.filter((k) => k.engine === en);
    if (!ks.length) return null;
    return {
      blocks: sum(ks.map((k) => k.blocks)),
      cores: sum(ks.map((k) => k.cores)),
      coreTime: r2(sum(ks.map((k) => k.coreTime))),
      setupTime: r2(sum(ks.map((k) => k.setupTime))),
      durMax: Math.max.apply(null, ks.map((k) => k.durMax)),
      durMean: r2(sum(ks.map((k) => k.coreTime)) / sum(ks.map((k) => k.blocks))),
      kernels: ks.map((k) => k.name),
    };
  };
  const engines = { aic: engineOf('aic'), aiv: engineOf('aiv') };

  tasks.push({
    id: t.id, tag: t.tag, funcId: t.funcId,
    callable: scopeName,
    kernels: kernels,
    kernelCount: kernels.length,
    engines: engines,
    /* Cube and Vec halves of one mixed kernel run concurrently on paired
     * cores. If the two longest blocks are ~equal the halves are serialised
     * inside the block; a large gap means one side waits on the other. */
    pairRatio: (engines.aic && engines.aiv)
      ? r3(Math.max(engines.aic.durMax, engines.aiv.durMax)
           / Math.max(Math.min(engines.aic.durMax, engines.aiv.durMax), 1e-9))
      : null,
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

/* ----------------------------------------------------------- by tag */
const byTag = {};
tasks.forEach((t) => { byTag[t.tag] = t; });
/* ------------------------------------------ critical path, attributed
 * Ports simpler_setup.tools.critical_path (repo/simpler). The structural
 * slack below says how much the DEPENDENCY GRAPH would tolerate; it cannot
 * say whether a task waited on a producer or on a busy core. This does.
 *
 *   static CPM   longest duration-weighted path in the happens-before DAG
 *                = the dependency-limited floor with unlimited cores
 *   observed     backward blame walk from the last-finishing task through
 *                whichever predecessor -- data dependency or same-core
 *                resource -- most tightly gated each task's start
 *
 * The observed walk's compute + stall must tile the makespan EXACTLY. That
 * check is the only thing making the per-task attribution sound, so it is
 * computed and shipped rather than asserted in prose. */

/* The reference tool's tolerance is 2 clock ticks. Without a recorded clock
 * there is nothing to convert, so fall back to two quanta of the timestamp
 * resolution (r2 -> 0.01 us) and record which one was used. */
const TOL = caseInfo.clockHz ? r3(2 / (caseInfo.clockHz / 1e6)) : 0.02;
const TOL_SOURCE = caseInfo.clockHz ? 'clock' : 'quantum';

const cpath = (function () {
  const T0 = 0;
  const st = {}, en = {}, dr = {};
  tasks.forEach((t) => { st[t.tag] = t.start; en[t.tag] = t.end; dr[t.tag] = t.span; });

  /* --- happens-before edges: a dependency edge is kept only when the
   * producer actually finished before the consumer started, AND started
   * strictly earlier. The strict-start test makes retention antisymmetric,
   * so the kept graph is acyclic even under tick-level ties. */
  const hbPred = {};
  let kept = 0, dropped = 0;
  tasks.forEach((t) => {
    const keep = [];
    t.pred.forEach((ptag) => {
      if (ptag === t.tag || st[ptag] === undefined) return;
      if (en[ptag] <= st[t.tag] + TOL && st[ptag] < st[t.tag]) { keep.push(ptag); kept++; }
      else dropped++;
    });
    hbPred[t.tag] = keep;
  });

  /* --- same-core resource predecessor: when was the core this task first
   * lands on freed? Running max over earlier slices on that lane, so a
   * pipelined / overlapping slice is handled, not just the previous one. */
  const corePrev = {};
  laneBlocks.forEach((slices) => {
    let bestEnd = -Infinity, bestTag = null;
    slices.forEach((b) => {
      const t = tasks[b[2]];
      if (!t) return;
      const bs = b[0], be = b[0] + b[1];
      if (bestTag !== null && bestTag !== t.tag && bestEnd <= bs + TOL && bs === st[t.tag]) {
        const cur = corePrev[t.tag];
        if (!cur || bestEnd > cur[1]) corePrev[t.tag] = [bestTag, bestEnd];
      }
      if (be > bestEnd) { bestEnd = be; bestTag = t.tag; }
    });
  });

  /* --- topological order over the kept graph */
  const indeg = {}, succ = {};
  tasks.forEach((t) => { indeg[t.tag] = 0; succ[t.tag] = []; });
  tasks.forEach((t) => {
    hbPred[t.tag].forEach((ptag) => { succ[ptag].push(t.tag); indeg[t.tag]++; });
  });
  const q = tasks.filter((t) => indeg[t.tag] === 0).map((t) => t.tag);
  const order = [];
  while (q.length) {
    const u = q.shift();
    order.push(u);
    succ[u].forEach((v) => { if (--indeg[v] === 0) q.push(v); });
  }
  const acyclic = order.length === tasks.length;

  /* --- static CPM: longest duration-weighted path */
  let cpmPath = [], cpmLen = 0;
  if (acyclic) {
    const finish = {}, choice = {};
    order.forEach((v) => {
      let bp = null, bf = 0;
      hbPred[v].forEach((ptag) => { if (finish[ptag] > bf) { bf = finish[ptag]; bp = ptag; } });
      finish[v] = bf + dr[v];
      choice[v] = bp;
    });
    let sink = null;
    Object.keys(finish).forEach((k) => { if (sink === null || finish[k] > finish[sink]) sink = k; });
    for (let v = sink; v; v = choice[v]) cpmPath.push(v);
    cpmPath.reverse();
    cpmLen = r2(finish[sink]);
  }

  /* --- observed: backward blame walk from the last-finishing task */
  let sink = null;
  tasks.forEach((t) => { if (sink === null || en[t.tag] > en[sink]) sink = t.tag; });
  const walk = [];
  const seen = {};
  let v = sink;
  while (v && !seen[v]) {
    seen[v] = 1;
    const cand = [];
    hbPred[v].forEach((ptag) => {
      if (en[ptag] <= st[v] + TOL && st[ptag] < st[v]) cand.push([en[ptag], ptag, 'data-wait']);
    });
    const cp = corePrev[v];
    if (cp && cp[1] <= st[v] + TOL && st[cp[0]] < st[v]) cand.push([cp[1], cp[0], 'core-wait']);
    if (!cand.length) { walk.push([v, 'front-gap']); break; }
    cand.sort((a, b) => a[0] - b[0]);
    const pickd = cand[cand.length - 1];
    walk.push([v, pickd[2]]);
    v = pickd[1];
  }
  walk.reverse();

  /* --- frontier sweep: tile [t0, end(sink)] into compute + stall */
  const segments = [];
  const byKind = { 'data-wait': 0, 'core-wait': 0, 'front-gap': 0 };
  let computeTotal = 0, stallTotal = 0, frontier = T0;
  walk.forEach((w) => {
    const tag = w[0], kind = w[1];
    const a = st[tag], b = en[tag];
    const gap = Math.max(0, a - frontier);
    const eff = Math.max(0, b - Math.max(a, frontier));
    const t = byTag[tag];
    segments.push({
      tag: tag, callable: t.callable, kind: kind,
      start: r2(a), end: r2(b), dur: r2(dr[tag]),
      compute: r2(eff), stall: r2(gap),
      blocks: t.blockCount, cores: t.coreCount,
      onCpm: cpmPath.indexOf(tag) >= 0,
      /* A *_wait task's span occupies the path but is not work. The model
       * calls every path span "compute"; keeping them apart is the
       * difference between "compute-bound" and "waiting on a peer". */
      isWait: /_wait$/.test(t.callable || ''),
    });
    byKind[kind] += gap;
    computeTotal += eff;
    stallTotal += gap;
    frontier = Math.max(frontier, b);
  });

  const makespan = r2(Math.max.apply(null, tasks.map((t) => t.end)) - T0);
  const tiled = r2(computeTotal + stallTotal);
  /* the invariant: the walk must account for every microsecond. r2 rounding
   * of the inputs is the only slack allowed, so compare at that resolution. */
  const tilingDelta = r2(tiled - makespan);
  const tilingExact = Math.abs(tilingDelta) <= 0.01;

  /* kernel families on the observed path, ranked by compute contributed */
  const fam = {};
  segments.forEach((sg) => {
    const f = (fam[sg.callable] = fam[sg.callable] || { callable: sg.callable, compute: 0, stall: 0, n: 0 });
    f.compute += sg.compute; f.stall += sg.stall; f.n++;
  });
  const families = Object.keys(fam).map((k) => ({
    callable: k, compute: r2(fam[k].compute), stall: r2(fam[k].stall), nodes: fam[k].n,
    share: r2((fam[k].compute / Math.max(makespan, 1e-9)) * 100),
  })).sort((a, b) => b.compute - a.compute);

  /* The reference model counts every path span as "compute". On a run whose
   * path is dominated by collective waits that reads as compute-bound, which
   * is the opposite of the truth -- so split it before classifying. */
  const waitSpan = r2(sum(segments.filter((sg) => sg.isWait).map((sg) => sg.compute)));
  const workSpan = r2(computeTotal - waitSpan);
  const waitShare = r2((waitSpan / Math.max(makespan, 1e-9)) * 100);
  const workShare = r2((workSpan / Math.max(makespan, 1e-9)) * 100);

  const cpmShare = r2((cpmLen / Math.max(makespan, 1e-9)) * 100);
  const stallShare = r2((stallTotal / Math.max(makespan, 1e-9)) * 100);
  const worstKind = Object.keys(byKind).sort((a, b) => byKind[b] - byKind[a])[0];
  let bound, boundWhy;
  if (cpmShare >= 85) {
    bound = 'dependency';
    boundWhy = '静态 CPM 占 makespan ' + cpmShare + '%，图本身就是地板，加核改善不了。';
  } else if (waitShare >= 25) {
    bound = 'comm';
    boundWhy = '路径上 ' + segments.filter((sg) => sg.isWait).length + ' 个 *_wait 任务占 '
      + waitShare + '%，真正算的只有 ' + workShare + '%。'
      + '模型把路径任务的 span 一律记作 compute，这里必须拆开看。';
  } else if (stallShare >= 40) {
    bound = worstKind === 'core-wait' ? 'resource' : 'stall';
    boundWhy = 'stall 占 ' + stallShare + '%，其中 ' + worstKind + ' 最大（'
      + r2((byKind[worstKind] / Math.max(stallTotal, 1e-9)) * 100) + '% 的 stall）。';
  } else {
    bound = 'compute';
    boundWhy = '真正计算占 ' + workShare + '%，等待 ' + waitShare + '%，stall ' + stallShare + '%。';
  }

  return {
    /* the filtered graph itself: slack has to run on the same edges the
     * critical path does, or the two disagree about who is critical */
    hbPred: hbPred,
    hbSucc: (function () {
      const out = {};
      tasks.forEach((t) => { out[t.tag] = []; });
      tasks.forEach((t) => { hbPred[t.tag].forEach((ptag) => { out[ptag].push(t.tag); }); });
      return out;
    })(),
    tol: TOL, tolSource: TOL_SOURCE,
    makespan: makespan,
    edgesKept: kept, edgesDropped: dropped,
    acyclic: acyclic,
    cpm: {
      tags: cpmPath, len: cpmLen, nodes: cpmPath.length, share: cpmShare,
      /* nodes on the dependency floor that the observed path never visits:
       * touching one lowers the floor, touching an observed-only node only
       * removes stall. The two are not interchangeable. */
      onlyOnCpm: cpmPath.filter(function (tg) { return !segments.some(function (sg) { return sg.tag === tg; }); }),
      shared: cpmPath.filter(function (tg) { return segments.some(function (sg) { return sg.tag === tg; }); }).length,
    },
    segments: segments,
    computeTotal: r2(computeTotal),
    waitSpan: waitSpan,
    workSpan: workSpan,
    waitShare: waitShare,
    workShare: workShare,
    waitNodes: segments.filter((sg) => sg.isWait).length,
    stallTotal: r2(stallTotal),
    computeShare: r2((computeTotal / Math.max(makespan, 1e-9)) * 100),
    stallShare: stallShare,
    stallByKind: {
      'data-wait': r2(byKind['data-wait']),
      'core-wait': r2(byKind['core-wait']),
      'front-gap': r2(byKind['front-gap']),
    },
    families: families.slice(0, 10),
    tiling: { sum: tiled, makespan: makespan, delta: tilingDelta, exact: tilingExact },
    bound: bound, boundWhy: boundWhy,
    /* rows the reference skill would flag; 1 us is its threshold */
    slowNodes: segments.filter((sg) => sg.stall > 1).length,
  };
})();

/* ------------------------------------------------------- critical path
 * ONE definition, derived from the happens-before graph cpath already
 * filtered by observed timestamps.
 *
 * This used to be an independent longest-path walk over the RAW deps graph.
 * That treated every dependency edge as a happens-before edge, so when a
 * consumer actually started before its producer finished -- early dispatch,
 * or a lifetime edge that encodes ownership rather than ordering -- it added
 * both spans instead of one. On decode_csa/rank0 it reported a 33-node chain
 * of 4985.02 us against a 4879.82 us makespan: 102.2%, which a dependency
 * floor cannot be. Five edges on that chain overlapped, 539.74 us of them.
 *
 * The filtered CPM's 14 nodes are a strict subset of that 33, so the old
 * walk was not mis-ordered, it was over-including. */
const critTags = cpath.cpm.tags;
let cursor = null, posGap = 0, overlap = 0;
const critNodes = critTags.map((tag) => {
  const t = byTag[tag];
  const gap = cursor === null ? 0 : r2(t.start - cursor);
  if (gap > 0) posGap += gap; else overlap += -gap;
  cursor = t.end;
  return { tag: tag, gap: gap };
});
/* A dependency-limited floor cannot exceed the wall time it is a floor for.
 * Nothing checked this before, which is why the old walk shipped wrong. */
const floorValid = cpath.cpm.len <= SPAN + 0.01;
const critical = {
  tags: critTags,
  chainSpan: cpath.cpm.len,
  walltime: SPAN,
  nodes: critNodes,
  gapOnPath: r2(posGap),
  overlapOnPath: r2(overlap),
  spanSum: r2(sum(critTags.map((tg) => byTag[tg].span))),
  share: cpath.cpm.share,
  floorValid: floorValid,
  /* residual overlap must stay inside the edge-retention tolerance */
  overlapWithinTol: overlap <= TOL * critTags.length + 0.01,
};
if (!floorValid) {
  console.error('  !! ' + rank + ' critical path ' + cpath.cpm.len
    + ' us exceeds makespan ' + SPAN + ' us');
}


/* -------------------------------------------------------------- slack
 * Standard forward/backward pass using the MEASURED span of each task.
 * ES/EF from predecessors, LF/LS from successors, slack = LS - ES.
 *
 * It runs on the SAME timestamp-filtered graph as the critical path. Using
 * the raw fanin/fanout edges instead over-constrains the pass: an edge whose
 * producer did not actually gate the consumer still pushes ES forward, so
 * slack comes out too small and far too many tasks read as zero-slack. On
 * decode_csa/rank0 that was 33 zero-slack tasks against a 14-node critical
 * path -- the scope ranking's red "on the critical path" border was on 19
 * scopes that were not.
 *
 * This is still STRUCTURAL slack: it says how much the graph would tolerate
 * and deliberately ignores resource contention. Two zero-slack tasks may
 * still be fighting for the same core -- that part is core-wait in cpath. */
const topo = tasks.slice().sort((a, b) => a.start - b.start);
const hbP = cpath.hbPred, hbS = cpath.hbSucc;
const ES = {}, EF = {}, LS = {}, LF = {};
topo.forEach((t) => {
  let es = 0;
  hbP[t.tag].forEach((ptag) => { if (EF[ptag] != null && EF[ptag] > es) es = EF[ptag]; });
  ES[t.tag] = es;
  EF[t.tag] = es + t.span;
});
const makespan = Math.max.apply(null, topo.map((t) => EF[t.tag]));
topo.slice().reverse().forEach((t) => {
  let lf = null;
  hbS[t.tag].forEach((stag) => { if (LS[stag] != null && (lf === null || LS[stag] < lf)) lf = LS[stag]; });
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
    /* which device kernels this source scope compiled into */
    kernelAcc: {},
    /* core-time on each engine, so a mixed scope does not read as one number */
    eng: {},
    /* spmd fan-out: how wide the launch was and how evenly it filled */
    fanCores: 0, fanBlocks: 0, waves: 0, imbalance: 0, coreSet: {},
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
  (t.kernels || []).forEach((kn) => {
    const a = (sc.kernelAcc[kn.name] = sc.kernelAcc[kn.name]
      || { name: kn.name, engine: kn.engine, funcId: kn.funcId,
           blocks: 0, cores: 0, coreTime: 0, setupTime: 0, durMax: 0 });
    a.blocks += kn.blocks;
    a.cores = Math.max(a.cores, kn.cores);
    a.coreTime += kn.coreTime;
    a.setupTime += kn.setupTime;
    a.durMax = Math.max(a.durMax, kn.durMax);
  });
  ['aic', 'aiv'].forEach((en) => {
    const e = t.engines && t.engines[en];
    if (!e) return;
    const a = (sc.eng[en] = sc.eng[en] || { blocks: 0, coreTime: 0, durMax: 0, cores: 0 });
    a.blocks += e.blocks; a.coreTime += e.coreTime;
    a.durMax = Math.max(a.durMax, e.durMax);
    a.cores = Math.max(a.cores, e.cores);
  });
  sc.fanCores = Math.max(sc.fanCores, t.coreCount);
  sc.fanBlocks = Math.max(sc.fanBlocks, t.blockCount);
  sc.waves = Math.max(sc.waves, t.blockCount / Math.max(t.coreCount, 1));
  sc.imbalance = Math.max(sc.imbalance, t.imbalance || 0);
  sc.coreSet[t.coreCount] = (sc.coreSet[t.coreCount] || 0) + 1;
  if (t.onCrit) sc.critNodes++;
  sc.first = Math.min(sc.first, t.start);
  sc.last = Math.max(sc.last, t.end);
});
/* every block duration, grouped by the scope it belongs to */
const scopeBlockDur = {};
laneBlocks.forEach((slices) => {
  slices.forEach((b) => {
    const t = tasks[b[2]];
    if (!t) return;
    (scopeBlockDur[t.callable] = scopeBlockDur[t.callable] || []).push(b[1]);
  });
});
const qAt = (sorted, q) => (sorted.length
  ? sorted[Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * q)))]
  : 0);

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
  const kernelList = Object.keys(sc.kernelAcc).map((n) => {
    const a = sc.kernelAcc[n];
    return {
      name: a.name, engine: a.engine, funcId: a.funcId,
      blocks: a.blocks, cores: a.cores,
      coreTime: r2(a.coreTime), setupTime: r2(a.setupTime), durMax: r2(a.durMax),
      share: r2((a.coreTime / Math.max(sc.coreTime, 1e-9)) * 100),
    };
  }).sort((a, c) => c.coreTime - a.coreTime);
  const engOut = {};
  ['aic', 'aiv'].forEach((en) => {
    const a = sc.eng[en];
    if (!a) return;
    engOut[en] = {
      blocks: a.blocks, cores: a.cores,
      coreTime: r2(a.coreTime), durMax: r2(a.durMax),
      share: r2((a.coreTime / Math.max(sc.coreTime, 1e-9)) * 100),
    };
  });
  /* --- cost decomposition over this scope's own blocks --- */
  sc.spmdLaunches = ts.length;
  sc.spmdWaves = sc.fanBlocks / Math.max(sc.fanCores, 1);
  sc.spmdCores = sc.fanCores;
  const bd = (scopeBlockDur[k] || []).slice().sort((a, b) => a - b);
  const bMed = qAt(bd, 0.5);
  const bP90 = qAt(bd, 0.9);
  const bMax = bd.length ? bd[bd.length - 1] : 0;
  const bMean = bd.length ? sc.coreTime / bd.length : 0;
  /* Sigma = N x mean is the exact identity. Comparing against N x median
   * says which side the spread falls on, and it is SIGNED: a scope whose
   * median sits ABOVE its mean (qk_pv: med 782 vs mean 648) is left-skewed
   * -- a few fast blocks, not a heavy tail. Clamping that to zero would
   * hide the more interesting case. */
  const uniform = bMed * bd.length;
  const skew = r2(sc.coreTime - uniform);

  return {
    name: k,
    kind: kind,
    /* Sigma = repeat x width x mean, an exact factorisation of the block
     * count. Raw N conflates two different things: fa_fused's 72 blocks are
     * 72 cores in ONE wave (wide), up_proj's 85 blocks are 85 separate
     * launches on ONE core (repeated). Calling both "次数多" is wrong. */
    cost: {
      blocks: bd.length,
      repeat: r2(sc.spmdLaunches * sc.spmdWaves),
      width: sc.spmdCores,
      mean: r2(bMean),
      med: r2(bMed),
      p90: r2(bP90),
      max: r2(bMax),
      min: r2(bd.length ? bd[0] : 0),
      /* N x median, and how far Sigma sits from it (signed) */
      uniform: r2(uniform),
      skew: skew,
      skewShare: r2((skew / Math.max(sc.coreTime, 1e-9)) * 100),
      spread: bMed > 0 ? r3(bP90 / bMed) : null,
    },
    /* the device kernels this scope compiled into: 1, or 2 when mixed */
    kernels: kernelList,
    kernelCount: kernelList.length,
    engines: engOut,
    /* Cube vs Vec longest block. ~1 on a mixed scope means the two halves
     * are serialised inside the block rather than overlapped. */
    pairRatio: (engOut.aic && engOut.aiv)
      ? r3(Math.max(engOut.aic.durMax, engOut.aiv.durMax)
           / Math.max(Math.min(engOut.aic.durMax, engOut.aiv.durMax), 1e-9))
      : null,
    /* spmd launch shape */
    spmd: {
      cores: sc.fanCores,
      blocks: sc.fanBlocks,
      waves: r2(sc.waves),
      launches: ts.length,
      imbalance: r2(sc.imbalance),
      /* the same scope relaunched at a different width is a real signal */
      widths: Object.keys(sc.coreSet).map((n) => +n).sort((a, c) => a - c),
    },
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
  cpath: cpath,
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

/* --------------------------------------------------- kernel -> source
 * The dump has no kernel->source map, but the names are not invented: every
 * outlined scope is named after a pl.spmd(..., name_hint="X") in the model
 * source. Walking the entry module's transitive imports and indexing those
 * hints reconstructs the mapping the dump omits.
 *
 * Two things this deliberately does NOT do:
 *   - it does not claim a unique site when the same hint appears more than
 *     once; those are reported as candidates
 *   - it does not attribute measured time to a source line. The name says
 *     where the scope was written, not which call site produced which block. */
const SUFFIX = /_(aic|aiv)$|_\d+$/;
const sourceIndex = (function () {
  if (!CASE.sourceDir || !fs.existsSync(CASE.sourceDir)) return null;
  const root = CASE.sourceDir;
  const exists = (f) => fs.existsSync(path.join(root, f));
  const seen = new Set();
  const queue = [CASE.entryModule];
  while (queue.length) {
    const f = queue.shift();
    if (!f || seen.has(f) || !exists(f)) continue;
    seen.add(f);
    const text = fs.readFileSync(path.join(root, f), 'utf8');
    for (const m of text.matchAll(/^\s*(?:from|import)\s+([a-z_0-9]+)/gm)) {
      const cand = m[1] + '.py';
      if (exists(cand) && !seen.has(cand)) queue.push(cand);
    }
  }
  const hints = {};
  Array.from(seen).sort().forEach((f) => {
    fs.readFileSync(path.join(root, f), 'utf8').split('\n').forEach((line, i) => {
      const m = line.match(/name_hint="([^"]+)"/);
      if (!m) return;
      (hints[m[1]] = hints[m[1]] || []).push({ file: f, line: i + 1 });
    });
  });
  return { modules: Array.from(seen).sort(), hints: hints };
})();

/* resolve one callable to its source site(s) */
function sourceFor(callable) {
  if (!sourceIndex) return null;
  const direct = sourceIndex.hints[callable];
  const base = callable.replace(SUFFIX, '');
  const viaSuffix = direct ? null : sourceIndex.hints[base];
  const sites = direct || viaSuffix;
  if (!sites || !sites.length) return null;
  return {
    hint: direct ? callable : base,
    exact: !!direct,
    file: sites[0].file,
    line: sites[0].line,
    candidates: sites.length,
    sites: sites.length > 1 ? sites : null,
  };
}

const sourceMap = (function () {
  if (!sourceIndex) return null;
  const out = {};
  let uniq = 0, ambiguous = 0, missing = 0;
  Object.keys(RANKS).forEach((rk) => {
    RANKS[rk].tasks.forEach((t) => {
      if (out[t.callable] !== undefined) return;
      const src = sourceFor(t.callable);
      out[t.callable] = src;
      if (!src) missing++;
      else if (src.candidates > 1) ambiguous++;
      else uniq++;
    });
  });
  return {
    root: path.basename(CASE.sourceDir),
    entry: CASE.entryModule,
    modules: sourceIndex.modules,
    hintCount: Object.keys(sourceIndex.hints).length,
    map: out,
    covered: uniq + ambiguous,
    total: uniq + ambiguous + missing,
    unique: uniq,
    ambiguous: ambiguous,
    missing: missing,
  };
})();

/* hang it on the tasks and scopes so every panel can read it */
if (sourceMap) {
  Object.keys(RANKS).forEach((rk) => {
    RANKS[rk].tasks.forEach((t) => { t.src = sourceMap.map[t.callable] || null; });
    RANKS[rk].scopes.forEach((sc) => { sc.src = sourceMap.map[sc.name] || null; });
  });
}

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

/* ------------------------------------------------------------- findings
 * A finding here is a CHAIN, not a single reading. The unit is "discovered
 * at one layer, followed down into the next, and either landed on a compiler
 * site or stopped on purpose for lack of evidence". Two rules keep the queue
 * honest:
 *   1. a chain must carry a makespan attribution computed from this run --
 *      Sigma core-time, utilisation and hint counts are not bottlenecks;
 *   2. a layer this dump cannot reach is a declared stop with a reason, not
 *      an invented conclusion.
 * Readings that fail rule 1 are still reported, as hygiene items that say so.
 */
const nameMapCount = Object.keys(nameMap.callable_id_to_name || {}).length;

const waitTasks = tasks.filter((t) => /_wait$/.test(t.callable || '')).sort((a, b) => b.span - a.span);
const waitSpan = r2(sum(waitTasks.map((t) => t.span)));
const mixTasks = tasks.filter((t) => t.kind === 'mix').sort((a, b) => b.span - a.span);
const worstImb = tasks.filter((t) => t.blockCount >= 32).sort((a, b) => b.imbalance - a.imbalance)[0];
const worstHandoff = tasks.filter((t) => t.svOverhead != null && t.blockCount >= 8)
  .sort((a, b) => b.svOverhead - a.svOverhead)[0];
const worstSetupShare = tasks.filter((t) => t.blockCount >= 8).sort((a, b) => b.setupShare - a.setupShare)[0];
const aicUtil = R.occupancy.aicUtil;
const aivUtil = R.occupancy.aivUtil;
const schedPerLaneUtil = R.scheduler.perLaneUtil;
const depthDegraded = depthSiteList.filter((s) => s.fittedDepth < s.maxReqDepth);
const setupHeavy = tasks.filter((t) => t.setupShare > 0.05).sort((a, b) => b.setupSum - a.setupSum);

/* ---------------------------------------------------------- wave floor
 * A multi-wave task cannot finish faster than (waves x median block). What
 * it actually took, minus that floor, is time the task existed without
 * computing -- gap, not work. This is the only place in this file where a
 * per-task number becomes a wall-clock claim, so the formula stays visible
 * in the finding text itself.                                             */
const wavesOf = (t) => (t.coreCount ? t.blockCount / t.coreCount : 1);
const waveRows = tasks.map((t) => {
  const waves = wavesOf(t);
  const floor = r2(waves * t.durMed);
  return { t: t, waves: r2(waves), floor: floor, gap: r2(t.span - floor) };
}).filter((x) => x.waves >= 2 && x.t.blockCount >= 32 && x.gap > 0)
  .sort((a, b) => b.gap - a.gap);
const waveGapSum = r2(sum(waveRows.map((x) => x.gap)));
const waveSpanSum = r2(sum(waveRows.map((x) => x.t.span)));
const waveBlocks = sum(waveRows.map((x) => x.t.blockCount));
const schedPerBlockUs = (schedPhases.dispatch && schedPhases.complete)
  ? r2(schedPhases.dispatch.usPerTask + schedPhases.complete.usPerTask) : null;
const schedBlockCost = (schedPerBlockUs && schedLanes.length)
  ? r2((waveBlocks * schedPerBlockUs) / schedLanes.length) : null;

/* Gap that the communication waits do NOT hide. A gap-heavy task overlapping
 * a *_wait is partly free; one running entirely outside every wait sits on
 * the wall clock by itself, and only that part is worth quoting as cost. */
const waitIv = waitTasks.map((t) => [t.start, t.end]);
const insideWait = (t) => waitIv.some((w) => t.start < w[1] && t.end > w[0]);
const exposedRows = waveRows.filter((x) => !insideWait(x.t));
const exposedGap = r2(sum(exposedRows.map((x) => x.gap)));
const unionUs = (ivs) => {
  const s = ivs.slice().sort((a, b) => a[0] - b[0]);
  let tot = 0, lo = null, hi = null;
  s.forEach((iv) => {
    if (lo === null) { lo = iv[0]; hi = iv[1]; return; }
    if (iv[0] > hi) { tot += hi - lo; lo = iv[0]; hi = iv[1]; } else if (iv[1] > hi) hi = iv[1];
  });
  if (lo !== null) tot += hi - lo;
  return r2(tot);
};
const exposedWindow = unionUs(exposedRows.map((x) => [x.t.start, x.t.end]));

/* ------------------------------------------------- early-dispatch pairs
 * The trace marks happens-before violations itself. One whose end lands on a
 * task's first block start is that task being dispatched while its producer
 * is still running -- which is what turns "duration - kernel_duration" into
 * on-core waiting rather than setup work. Only pairs where both ends resolve
 * to a real task are kept; nothing is inferred from the marker alone.      */
const hbLinks = (hbPairs || []).map((h) => {
  const cons = tasks.filter((t) => Math.abs(t.start - h.tsEnd) < 1.5)
    .sort((a, b) => b.span - a.span)[0];
  const prod = tasks.filter((t) => t.start <= h.ts + 1.5 && t.end > h.tsEnd)
    .sort((a, b) => b.span - a.span)[0];
  return cons && prod && cons !== prod ? { h: h, cons: cons, prod: prod } : null;
}).filter(Boolean).sort((a, b) => b.cons.setupSum - a.cons.setupSum);
const hbLink = hbLinks[0] || null;
/* kernel floor: what the blocks would take if they only ran kernel code */
const kernelFloor = (t) => r2(wavesOf(t) * (t.blockCount ? t.kdurSum / t.blockCount : 0));
const stallHost = hbLink ? hbLink.cons : worstHandoff;
const stallFloor = stallHost ? kernelFloor(stallHost) : null;
const stallGap = stallHost ? r2(stallHost.span - stallFloor) : null;

/* ------------------------------------------------------ one-wave giants
 * A task whose blocks all start together has no wave quantisation to tune:
 * its span IS one block. If such a task sits on the dependency critical
 * path, the only lever left is per-block work, and this dump carries no
 * in-block PMU -- so the chain has to stop at L1 and say so.              */
const oneWave = tasks.filter((t) => t.blockCount >= 8 && t.coreCount >= 8 && wavesOf(t) <= 1.5)
  .sort((a, b) => b.span - a.span);
const longBlock = oneWave.filter((t) => critTags.indexOf(t.tag) >= 0)[0] || oneWave[0] || null;
const lbAic = longBlock && longBlock.engines ? longBlock.engines.aic : null;
const lbAiv = longBlock && longBlock.engines ? longBlock.engines.aiv : null;
const lbSerial = lbAic && lbAiv ? r2(lbAic.durMax + lbAiv.durMax) : null;

/* -------------------------------------------------- compiler site joins
 * A compiler hint only earns a place in a chain if the tasks compiled from
 * that source point are the ones the chain is already about. Everything
 * else is a hint, not a root cause, and is counted separately.            */
const srcOf = (t) => (sourceMap && sourceMap.map ? sourceMap.map[t.callable] : null);
const nearSite = (t, file, line, win) => {
  const m = srcOf(t);
  return !!(m && m.file === file && Math.abs(m.line - line) <= (win == null ? 40 : win));
};
const hotTags = new Set(waveRows.slice(0, 8).map((x) => x.t.tag)
  .concat(stallHost ? [stallHost.tag] : [])
  .concat(longBlock ? [longBlock.tag] : []));
const depthHits = depthSiteList.map((s) => {
  const hit = tasks.filter((t) => nearSite(t, s.file, s.line)).sort((a, b) => b.span - a.span);
  return {
    site: s, tasks: hit, topSpan: hit.length ? hit[0].span : 0,
    onChain: hit.filter((t) => hotTags.has(t.tag)),
  };
}).sort((a, b) => b.topSpan - a.topSpan);
const depthHot = depthHits.filter((d) => d.onChain.length);
const depthCold = depthHits.filter((d) => !d.onChain.length);
const depthFor = (tag) => depthHot.filter((d) => d.onChain.some((t) => t.tag === tag))[0] || null;
const gapSite = depthHot.filter((d) => waveRows.slice(0, 8)
  .some((x) => d.onChain.some((t) => t.tag === x.t.tag)))[0] || null;
const stallSite = stallHost ? depthFor(stallHost.tag) : null;
/* name the L0 tile whose size equals one pipeline stage, so the budget
 * argument is arithmetic the reader can redo rather than an assertion */
const tileForStage = (bytes, unit) => (l0Tiles || [])
  .filter((x) => x.bytes === bytes && x.mem === unit).sort((a, b) => b.n - a.n)[0] || null;

/* --------------------------------------------------- tile-granularity mix
 * PH001 counts collapse three different situations into one number. Split
 * them, because only one of the three is tunable.                         */
const tileScalar = tileSiteList.filter((s) => (s.shapes || []).length
  && s.shapes.every((sh) => /\[1\]$/.test(sh)));
const tileScalarSet = new Set(tileScalar.map((s) => s.key));
const tileHalf = tileSiteList.filter((s) => !tileScalarSet.has(s.key) && s.minB * 2 >= s.cacheLineB);
const tileTunable = tileSiteList.filter((s) => !tileScalarSet.has(s.key) && s.minB * 2 < s.cacheLineB);
const occOf = (a) => sum(a.map((s) => s.occ));
const tileOnChain = tileTunable.filter((s) => tasks
  .some((t) => hotTags.has(t.tag) && nearSite(t, s.file, s.line)));

/* worstImb is only a "tail block" story when the task actually runs several
 * waves; at one wave the tail IS the span and the honest reading flips. */
const imbWaves = worstImb ? r2(wavesOf(worstImb)) : null;
const imbFloor = worstImb ? r2(wavesOf(worstImb) * worstImb.durMed) : null;
const imbGap = worstImb ? r2(worstImb.span - wavesOf(worstImb) * worstImb.durMed) : null;
const imbMultiWave = !!(worstImb && wavesOf(worstImb) >= 2);

const CAN = {
  C1: !!(launchSkew && waitTasks.length && launchSkew.checks && launchSkew.checks.length >= 2),
  C2: waveRows.length >= 2 && waveGapSum > 0,
  /* C3 needs the stall to be material, not merely present: at least 3% of
   * makespan AND at least a tenth of the block's own core time. A 12 us
   * hand-off on a 993 us run is noise, and promoting it to a chain would put
   * a rounding error next to a 39% finding. */
  C3: !!(stallHost && stallGap >= SPAN * 0.03 && stallHost.setupShare >= 0.1),
  C4: !!longBlock,
  H1: tileSiteList.length > 0,
  H2: !!worstImb,
  H3: rqStat.window > 0,
  H4: !!frontend,
};

/* Hygiene items point at chains by id. Which chains exist depends on what
 * the dump carries, so those ids are derived, never spelled into the prose. */
const depthChainIds = ['C2', 'C3'].filter((k) => CAN[k]);
const starveChainIds = ['C1', 'C2'].filter((k) => CAN[k]);

/* one shared vocabulary for a chain step, so every chain reads the same way */
const step = (level, role, headline, detail, evidence, subjects) => ({
  level: level, role: role, headline: headline, detail: detail,
  evidence: (evidence || []).filter(Boolean), subjects: subjects || null,
});

const findings = [
  CAN.C1 && {
    id: 'C1', kind: 'chain', level: 'l2', severity: 'high', axis: 'comm',
    title: '集合点等待 ' + launchSkew.measuredSum + ' us，根因在 rank 启动错峰',
    metric: '错峰 ' + launchSkew.runnerUs + ' us · Σ(*_wait) ' + waitSpan + ' us',
    cost: {
      us: launchSkew.measuredSum, share: r2((launchSkew.measuredSum / SPAN) * 100),
      basis: 'Σ(*_wait span)，全部贴合错峰上界',
    },
    claim: waitTasks.length + ' 个 *_wait 合计 ' + waitSpan + ' us（makespan 的 '
      + r2((waitSpan / SPAN) * 100) + '%），其中最大的两个 '
      + launchSkew.checks.slice(0, 2).map((c) => c.callable + ' ' + c.measured + ' us').join(' 与 ')
      + ' 合计 ' + r2(launchSkew.checks.slice(0, 2).reduce((n, c) => n + c.measured, 0))
      + ' us，占全部等待的 '
      + r2((launchSkew.checks.slice(0, 2).reduce((n, c) => n + c.measured, 0) / waitSpan) * 100)
      + '%。把 rank 启动偏移加到对侧到达时刻上得到的上界，这两项分别贴到 '
      + launchSkew.checks[0].fitPct + '% 与 ' + launchSkew.checks[1].fitPct
      + '% —— 等待量级由错峰解释，不是通信算法本身。',
    chain: [
      step('l2', 'observe',
        '等待吃掉 ' + r2((waitSpan / SPAN) * 100) + '% 墙钟',
        waitTasks.length + ' 个 *_wait 任务合计 ' + waitSpan + ' us，全部单块单核；'
          + waitTasks.filter((t) => critTags.indexOf(t.tag) >= 0).length + ' 个落在依赖关键路径（'
          + critTags.length + ' 节点）上。',
        [{ artifact: 'merged_swimlane (' + PRIMARY.key + '/d0)', locator: waitTasks.map((t) => t.tag).join(' / '),
          value: waitTasks.map((t) => t.callable + '=' + t.span + 'us').join(', ') }],
        { view: 'l2', tasks: waitTasks.map((t) => t.tag) }),
      step('e2e', 'root',
        'rank 启动偏移 ' + launchSkew.runnerUs + ' us',
        '两份 host log 的 ts 同属一个 host 单调钟，对齐后晚发 ' + launchSkew.runnerUs
          + ' us（chip.run 口径 ' + launchSkew.chipUs + ' us）。两卡任务数与块数相同，AIC busy 只差 '
          + workDeltaPct + '% —— 更慢却更闲，说明多出来的是空转而不是算力差。'
          + launchSkew.checks.length + ' 个 wait '
          + (launchSkew.allUnderBound ? '全部落在上界内' : '有超出上界的项')
          + '，合计 ' + launchSkew.measuredSum + ' / ' + launchSkew.boundSum + ' us。',
        [{ artifact: 'host log（两 rank）', locator: 'inv=' + launchSkew.inv + ' chip.run.runner_run ts',
          value: launchSkew.ts.rank0 + ' ns vs ' + launchSkew.ts.rank1 + ' ns → 晚 ' + launchSkew.runnerUs + ' us' }]
          .concat(launchSkew.checks.map((c) => ({
            artifact: 'merged_swimlane（两 rank）',
            locator: c.callable + '：到达 ' + c.start0 + ' us vs ' + c.start1 + ' us（对齐后 ' + c.arrival1 + ' us）',
            value: '上界 ' + c.bound + ' us，实测 ' + c.measured + ' us'
              + (c.under ? '（占上界 ' + c.fitPct + '%）' : '（超出上界）'),
          }))),
        { view: 'e2e', ranks: RANK_KEYS.slice(0, 2) }),
      step('l1', 'stop',
        '不下探 L1 / L0',
        '这 ' + waitTasks.length + ' 个 wait 全是单块单核（blockCount=1），核上没有可优化对象：'
          + '压不压得下去取决于对侧什么时候到，不取决于这块核上的代码。'
          + '再往下走只会把一个调度问题包装成一个 kernel 问题。',
        [], null),
    ],
    terminus: { level: 'l1', reason: '单块单核，核上无可调对象；链止于 L2 / E2E' },
    evidence: [],
    focus: { view: 'l2', task: waitTasks[0].tag, critOnly: true },
    lever: '对齐两卡的下发时刻（同步 launch、收紧 host 侧提交路径），而不是去调通信算子或本卡 kernel。',
    guardrail: '上界只说明「等待能被错峰解释」，不证明错峰是唯一成因；时钟对齐是从同一主机的 mono ts 推的，'
      + 'dump 里没有显式跨 rank 同步记录；本次仅 2 次调用，偏移量本身没有分布。',
    verify: '固定 case 重跑 ≥10 次，每次记录两卡 runner_run 的 ts 差与 *_wait 合计；偏移收窄，等待应同比收窄。',
  },
  CAN.C2 && {
    id: 'C2', kind: 'chain', level: 'l2', severity: 'high', axis: 'granularity',
    title: '多波小块任务有 ' + waveGapSum + ' us 是块间间隙，不是核上工作',
    metric: waveRows.length + ' 个任务 / ' + waveBlocks + ' 块 · 间隙 ' + waveGapSum + ' us',
    cost: {
      us: waveGapSum, share: r2((waveGapSum / SPAN) * 100),
      basis: 'Σ(span − 波数 × 块中位时长)，仅取 ≥2 波且 ≥32 块的任务',
    },
    claim: waveRows.length + ' 个多波任务 span 合计 ' + waveSpanSum + ' us，其中 ' + waveGapSum
      + ' us（makespan 的 ' + r2((waveGapSum / SPAN) * 100) + '%）超出「波数 × 块中位时长」的工作量下界。'
      + (exposedRows.length
        ? '其中 ' + exposedRows.length + ' 个任务完全落在所有 *_wait 之外（跨 ' + exposedWindow
          + ' us 墙钟），它们的间隙合计 ' + exposedGap
          + ' us —— 这部分无处可藏，通信等待掩盖不了它。注意这些任务彼此重叠，'
          + '所以间隙合计会大于窗口长度，两个数不能相除。'
        : '')
      + '同期 AIC 平均占用只有 ' + aicUtil + '%。',
    chain: [
      step('l2', 'observe',
        '间隙 ' + waveGapSum + ' us，占 makespan ' + r2((waveGapSum / SPAN) * 100) + '%',
        '按「span − 波数 × 块中位时长」逐任务算，头部是 '
          + waveRows.slice(0, 3).map((x) => x.t.callable + ' ' + x.gap + ' us').join('、')
          + '。这不是 Σ core-time，是墙钟上的空档。',
        waveRows.slice(0, 6).map((x) => ({
          artifact: 'merged_swimlane blocks', locator: x.t.tag + ' (' + x.t.callable + ')',
          value: 'span ' + x.t.span + ' us − ' + x.waves + ' 波 × 中位 ' + x.t.durMed
            + ' us = 间隙 ' + x.gap + ' us',
        })),
        { view: 'l2', tasks: waveRows.slice(0, 6).map((x) => x.t.tag) }),
      step('l1', 'descend',
        '块太小：中位 ' + waveRows[0].t.durMed + ' us，' + waveRows[0].t.blockCount + ' 块摊 '
          + waveRows[0].t.coreCount + ' 核',
        '这些任务每块只有 ' + Math.min.apply(null, waveRows.map((x) => x.t.durMed)) + '–'
          + Math.max.apply(null, waveRows.map((x) => x.t.durMed)) + ' us，却要走完整的 dispatch→complete。'
          + (schedPerBlockUs
            ? 'AICPU 侧每块 dispatch ' + schedPhases.dispatch.usPerTask + ' us + complete '
              + schedPhases.complete.usPerTask + ' us = ' + schedPerBlockUs + ' us；' + waveBlocks
              + ' 块摊到 ' + schedLanes.length + ' 条调度线程 ≈ ' + schedBlockCost + ' us，'
              + '可以解释间隙的 ' + r2((schedBlockCost / waveGapSum) * 100) + '%。'
              + '注意调度线程本身并没有饱和（每线程占用 ' + schedPerLaneUtil
              + '%）—— 贵的是次数，不是线程忙不忙。'
            : ''),
        [schedPhases.dispatch ? { artifact: 'merged_swimlane scheduler lane',
          locator: 'phase=dispatch / complete',
          value: schedPhases.dispatch.us + ' us / ' + schedPhases.dispatch.tasks + ' 次 · '
            + schedPhases.complete.us + ' us / ' + schedPhases.complete.tasks + ' 次' } : null,
          { artifact: 'worker lanes', locator: 'AIC 平均 vs AIV 平均', value: aicUtil + '% vs ' + aivUtil + '%' }],
        { view: 'l2', overlay: 'sched', tasks: waveRows.slice(0, 3).map((x) => x.t.tag),
          schedPhases: schedPhases.complete ? ['complete', 'dispatch'] : [] }),
      gapSite && step('l1', 'descend',
        '块压不下来：' + gapSite.site.units.join('/') + ' 只有 ' + gapSite.site.freeB
          + ' B，一级 stage 就要 ' + gapSite.site.perStageB + ' B',
        '要减少块数就得让单块做更多事，而单块做更多事需要软流水把 load 与 mmad 叠起来。'
          + gapSite.site.units.join('/') + ' 可用 ' + gapSite.site.freeB + ' B，每级 stage '
          + gapSite.site.perStageB + ' B'
          + (tileForStage(gapSite.site.perStageB, gapSite.site.units[0])
            ? '（相当于一块 ' + tileForStage(gapSite.site.perStageB, gapSite.site.units[0]).dtype + ' '
              + tileForStage(gapSite.site.perStageB, gapSite.site.units[0]).rows + '×'
              + tileForStage(gapSite.site.perStageB, gapSite.site.units[0]).cols + ' tile）'
            : '')
          + '：单组双缓冲就要 ' + gapSite.site.perStageB * 2 + ' B，而这里有 ' + gapSite.site.groupCount
          + ' 组同驻。深度 1 是算术上被逼出来的，不是调参不到位。',
        [{ artifact: 'report/perf_hints.log', locator: gapSite.site.module + ':' + gapSite.site.line,
          value: 'depth ' + gapSite.site.maxReqDepth + '→' + gapSite.site.fittedDepth + '，'
            + gapSite.site.groupCount + ' 组 @' + gapSite.site.units.join('/') + '，'
            + gapSite.site.perStageB + ' B/stage，' + gapSite.site.freeB + ' B free' },
          { artifact: 'passes_dump AutoTileMatmulL0', locator: 'Mem.' + gapSite.site.units[0] + ' tile',
            value: (l0Tiles || []).filter((x) => x.mem === gapSite.site.units[0]).slice(0, 3)
              .map((x) => x.dtype + ' ' + x.rows + '×' + x.cols + '=' + x.bytes + 'B').join('、') || 'n/a' }],
        { view: 'isa' }),
      gapSite
        ? step('compiler', 'root',
          'MemoryReuse 在 ' + gapSite.site.file + ':' + gapSite.site.line + ' 把深度降到 '
            + gapSite.site.fittedDepth,
          '这个源码点编出来的任务正是间隙最大的那批（'
            + gapSite.onChain.slice(0, 3).map((t) => t.callable + ' ' + t.span + ' us').join('、')
            + '）。' + depthHot.length + ' / ' + depthHits.length
            + ' 个 PH-MR-001 源码点能接到本链的任务上，其余 ' + depthCold.length
            + ' 个落在更冷的任务上，不应与这条同级展示。',
          depthHot.slice(0, 4).map((d) => ({
            artifact: 'report/perf_hints.log', locator: d.site.module + ':' + d.site.line,
            value: 'depth ' + d.site.maxReqDepth + '→' + d.site.fittedDepth + ' · 命中 '
              + d.onChain.map((t) => t.callable).join('/') + '（最大 span ' + d.topSpan + ' us）',
          })),
          { view: 'compiler', tab: 'depth', sites: depthHot.map((d) => d.site.key) })
        : step('compiler', 'stop',
          '本 dump 没有可接的 PH-MR-001',
          depthSiteList.length
            ? depthSiteList.length + ' 个流水深度回退点没有一个能接到本链的任务上，不能当根因用。'
            : 'report/perf_hints.log 里没有 PH-MR-001，编译器层在这条链上整层缺证据；'
              + '只能先在 L1 侧验证「减少块数」是否收敛间隙。',
          [], null),
    ].filter(Boolean),
    terminus: gapSite
      ? { level: 'compiler', reason: '落到 MemoryReuse 的具体源码点，可编译验证' }
      : { level: 'l1', reason: '缺可接的 PH-MR-001，编译器层无证据；链止于 L1' },
    evidence: [],
    focus: { view: 'l2', task: waveRows[0].t.tag, pass: 'MemoryReuse' },
    lever: '减少块数而不是减少块时长：'
      + (gapSite
        ? '把 ' + gapSite.site.units[0] + ' 侧的 tile 减半（' + gapSite.site.perStageB + ' B → '
          + gapSite.site.perStageB / 2 + ' B）后，单组 depth ' + gapSite.site.maxReqDepth + ' 只要 '
          + (gapSite.site.perStageB / 2) * gapSite.site.maxReqDepth + ' B，' + gapSite.site.freeB
          + ' B 能同时容纳 '
          + Math.floor(gapSite.site.freeB / ((gapSite.site.perStageB / 2) * gapSite.site.maxReqDepth))
          + ' / ' + gapSite.site.groupCount + ' 组 —— 要吃下全部 ' + gapSite.site.groupCount
          + ' 组还得同时把同驻组数降下来，单减 tile 不够。深度上来之后再把块数往下收。'
        : '把外层迭代折进核内，或用 pl.spmd 一次 fan-out 多块，降低 dispatch / complete 次数。'),
    guardrail: '「波数 × 块中位时长」是下界而不是可达目标：块变大后中位时长会上升，'
      + '间隙收敛的同时 span 可能不动。必须同时报 span、块数与块中位时长三项，只报间隙会自欺。'
      + (gapSite ? '另外调大 stage 会再触发一次 MemoryReuse 回退，方向是减小同驻 tile，不是加大 stage。' : ''),
    verify: '重编译后核对该源码点的 PH-MR-001 是否消失、块数是否下降，再重测这批任务的 span 与间隙；'
      + '间隙下降而 span 不降，说明瓶颈已经换了位置。',
  },
  CAN.C3 && {
    id: 'C3', kind: 'chain', level: 'l2', severity: 'high', axis: 'launch',
    title: stallHost.callable + ' 每块 ' + r2(stallHost.setupMean) + ' us 在核上空等生产者',
    metric: stallGap + ' us 墙钟 · setup 占核上 ' + r2(stallHost.setupShare * 100) + '%',
    cost: {
      us: stallGap, share: r2((stallGap / SPAN) * 100),
      basis: 'span − 波数 × 块内 kernel 时长（duration 与 kernel_duration 之差不计入工作量）',
    },
    claim: (hbLink
      ? 'trace 自带的 hb_violation 标出 ' + hbLink.h.from + '→' + hbLink.h.to + ' 区间 '
        + hbLink.h.ts + '–' + hbLink.h.tsEnd + ' us，正好是 ' + hbLink.prod.callable + '（'
        + hbLink.prod.start + '–' + hbLink.prod.end + ' us）还在跑、而 ' + stallHost.callable
        + ' 的块已经在 ' + stallHost.start + ' us 起来了。'
      : '本 case 没有 hb_violation 标记，只有 Scheduler / Worker 两视角的差值。')
      + '所以核上那 ' + stallHost.durMean + ' us 里的 ' + r2(stallHost.setupMean) + ' us（'
      + r2(stallHost.setupShare * 100) + '%）是等数据，不是可复用的准备工作：' + stallHost.blockCount
      + ' 块累计 ' + stallHost.setupSum + ' us 核时间，折到墙钟上是 ' + stallGap + ' us。'
      + '把它当「setup 提到核外复用」来优化，是对错误前提开的药。',
    chain: [
      step('l2', 'observe',
        (hbLink ? 'hb_violation 指到生产者未完成' : '两视角差 ' + stallHost.svOverhead + ' us'),
        (hbPairs || []).length + ' 对 happens-before 违例标记'
          + (hbLink
            ? '，其中 ' + hbLinks.length + ' 对能同时解析出生产者与消费者任务；最贵的一对是 '
              + hbLink.prod.callable + ' → ' + stallHost.callable + '。'
            : '。')
          + '消费者在观测路径上贡献 ' + stallHost.span + ' us（makespan 的 '
          + r2((stallHost.span / SPAN) * 100) + '%）。',
        (hbLink
          ? [{ artifact: 'merged_swimlane flow events',
            locator: 'hb_violation ' + hbLink.h.from + '→' + hbLink.h.to,
            value: hbLink.h.ts + '–' + hbLink.h.tsEnd + ' us（inputs ' + hbLink.h.inputs
              + ' / outputs ' + hbLink.h.outputs + '）' },
            { artifact: 'merged_swimlane blocks', locator: hbLink.prod.tag + ' → ' + stallHost.tag,
              value: '生产者 ' + hbLink.prod.start + '–' + hbLink.prod.end + ' us，消费者 '
                + stallHost.start + '–' + stallHost.end + ' us —— 消费者早起 '
                + r2(hbLink.prod.end - stallHost.start) + ' us' }]
          : [{ artifact: 'Scheduler View (pid 3) vs Worker View (pid 4)', locator: stallHost.tag,
            value: 'aicpu-duration ' + stallHost.svAicpuMean + ' us vs 核上 ' + stallHost.durMean + ' us' }]),
        { view: 'l2', tasks: [stallHost.tag].concat(hbLink ? [hbLink.prod.tag] : []) }),
      step('l1', 'descend',
        '核上 ' + r2(stallHost.setupShare * 100) + '% 不是 kernel',
        'Worker View 记核上 ' + stallHost.durMean + ' us，其中 kernel 只有 '
          + r2(stallHost.kdurSum / stallHost.blockCount) + ' us，差 ' + r2(stallHost.setupMean)
          + ' us 落在 duration − kernel_duration 里；Scheduler View 记 dispatch→finish '
          + stallHost.svAicpuMean + ' us，再差 ' + stallHost.svOverhead + ' us。'
          + '三个口径的差额都指向同一段等待，而不是三段独立开销。',
        [{ artifact: 'Worker View (pid 4)', locator: stallHost.tag + ' duration − kernel_duration',
          value: 'setup mean ' + r2(stallHost.setupMean) + ' us / ' + stallHost.blockCount + ' 块 = '
            + stallHost.setupSum + ' us 核时间' },
          { artifact: 'Scheduler View (pid 3)', locator: stallHost.tag,
            value: 'aicpu-duration mean ' + stallHost.svAicpuMean + ' us, max ' + stallHost.svAicpuMax + ' us' },
          { artifact: 'same trace', locator: 'setup 占比 > 5% 的任务',
            value: setupHeavy.length + ' 个，最高 ' + worstSetupShare.callable + ' '
              + r2(worstSetupShare.setupShare * 100) + '%' }],
        { view: 'l1', tasks: [stallHost.tag] }),
      stallSite && step('l1', 'descend',
        stallSite.site.units.join('/') + ' 深度 ' + stallSite.site.fittedDepth + '，没有双缓冲掩盖这段等待',
        '即使等待无法消除，双缓冲也能让后一 stage 的搬运和前一 stage 的等待重叠。这里 '
          + stallSite.site.units.join('/') + ' 可用 ' + stallSite.site.freeB + ' B，每级 stage 只要 '
          + stallSite.site.perStageB + ' B，' + stallSite.site.groupCount + ' 组，'
          + (stallSite.site.perStageB * stallSite.site.maxReqDepth <= stallSite.site.freeB
            ? '按这条 hint 自己的数字，depth ' + stallSite.site.maxReqDepth + ' 只要 '
              + stallSite.site.perStageB * stallSite.site.maxReqDepth + ' B，远小于可用的 '
              + stallSite.site.freeB + ' B，却仍被降到 ' + stallSite.site.fittedDepth
              + ' —— 这是本次 dump 里最值得单独立假设的编译器异常。'
            : '两级就要 ' + stallSite.site.perStageB * stallSite.site.maxReqDepth + ' B，放不下。'),
        [{ artifact: 'passes_dump AutoTileMatmulL0 / 预算',
          locator: 'Mem.' + stallSite.site.units[0] + ' 可用 vs 每级 stage',
          value: stallSite.site.freeB + ' B free vs ' + stallSite.site.perStageB + ' B/stage × depth '
            + stallSite.site.maxReqDepth + ' = ' + stallSite.site.perStageB * stallSite.site.maxReqDepth + ' B' }],
        { view: 'isa' }),
      stallSite
        ? step('compiler', 'root',
          'MemoryReuse @ ' + stallSite.site.file + ':' + stallSite.site.line,
          '两件事要分开验证：一是这条边为什么会早发（依赖是否声明不足），二是 '
            + stallSite.site.units.join('/') + ' 明显放得下却仍被降级。'
            + '前者是调度问题，后者是 pass 问题，不能合成一个实验。',
          [{ artifact: 'report/perf_hints.log', locator: stallSite.site.module + ':' + stallSite.site.line,
            value: 'unit ' + stallSite.site.units.join('/') + ' · ' + stallSite.site.groupCount
              + ' 组 · depth ' + stallSite.site.maxReqDepth + '→' + stallSite.site.fittedDepth + ' · '
              + stallSite.site.perStageB + ' B/stage · ' + stallSite.site.freeB + ' B free' }],
          { view: 'compiler', tab: 'depth', sites: [stallSite.site.key] })
        : step('compiler', 'stop',
          '编译器层缺证据',
          '这个源码点没有对应的 PH-MR-001，无法把早发或空等接到具体 pass 上；'
            + '先在 L2 侧验证依赖声明与 dispatch 时机。',
          [], null),
    ].filter(Boolean),
    terminus: stallSite
      ? { level: 'compiler', reason: '落到 MemoryReuse 源码点，但「放得下却降级」本身仍需复现' }
      : { level: 'l1', reason: '无对应编译提示；链止于 L1' },
    evidence: R.cpath.edgesDropped
      ? [{ artifact: '依赖关键路径（happens-before 图）',
        locator: 'edgesKept ' + R.cpath.edgesKept + ' / edgesDropped ' + R.cpath.edgesDropped,
        value: '这条早发边被时间戳过滤丢掉，' + stallHost.tag + ' 之后整条尾巴 slack 都是同一个 '
          + stallHost.slack + ' us —— 该 slack 不能当「可以不管」的依据' }]
      : [],
    focus: { view: 'l1', task: stallHost.tag },
    lever: '先把依赖补实或推迟 dispatch，让消费者不要在生产者完成前上核；确认等待无法消除后，'
      + '再考虑把消费段与生产者合进同一 mixed kernel，用 GM FIFO 交接。',
    guardrail: '合核会拉长单核占用；合并后要复查该核是否变成新的独占瓶颈。'
      + '推迟 dispatch 可能把等待搬到队列里而不是消掉，必须同时报 span 与 ready>0 占比。',
    verify: '重测该任务的 duration − kernel_duration、aicpu-duration 与 hb_violation 条数；'
      + 'setup 占比下降而 span 不降，说明等待只是换了地方。',
  },
  CAN.C4 && {
    id: 'C4', kind: 'chain', level: 'l2', severity: 'medium', axis: 'pipeline',
    title: longBlock.callable + ' 一波跑完，整段 ' + longBlock.span + ' us 就是一个块的时长',
    metric: longBlock.blockCount + ' 块 / ' + longBlock.coreCount + ' 核 · ' + r2(wavesOf(longBlock))
      + ' 波 · 最长块 ' + longBlock.durMax + ' us',
    cost: {
      us: longBlock.span, share: r2((longBlock.span / SPAN) * 100),
      basis: 'span 本身即块时长（波数 ≤ 1.5，无波量化可调）',
    },
    claim: longBlock.callable + ' 的 ' + longBlock.blockCount + ' 块摊在 ' + longBlock.coreCount
      + ' 核上一波跑完，最长块 ' + longBlock.durMax + ' us、整段 span ' + longBlock.span
      + ' us —— 两者几乎相等，说明没有靠加核或改波次能拿到的收益。'
      + (lbAic && lbAiv
        ? 'ExpandMixedKernel 把这个 scope 拆成 ' + longBlock.kernels.map((k) => k.name).join(' + ')
          + '，共用一次 Group launch。这里要特别否掉一个常见误读：两半不是块内串行 —— Cube 侧 '
          + lbAic.blocks + ' 块（最长 ' + lbAic.durMax + ' us）与 Vec 侧 ' + lbAiv.blocks + ' 块（最长 '
          + lbAiv.durMax + ' us）跑在不同核上，真串行应该接近 ' + lbSerial + ' us，而实测 span 只有 '
          + longBlock.span + ' us ≈ max(' + lbAic.durMax + ', ' + lbAiv.durMax + ')，它们本来就是并行的。'
        : '')
      + '要压的是块内工作量，而本 dump 没有块内 PMU，链必须停在 L1。',
    chain: [
      step('l2', 'observe',
        (critTags.indexOf(longBlock.tag) >= 0 ? '依赖关键路径上最大的计算段' : '观测路径上的大计算段')
          + ' ' + longBlock.span + ' us',
        (critTags.indexOf(longBlock.tag) >= 0
          ? '依赖关键路径 ' + critTags.length + ' 节点 / ' + critical.chainSpan + ' us（makespan 的 '
            + critical.share + '%），本任务是其中第 ' + (critTags.indexOf(longBlock.tag) + 1) + ' 节点。'
          : '不在依赖关键路径上，但占 makespan ' + r2((longBlock.span / SPAN) * 100) + '%。'),
        [{ artifact: 'merged_swimlane', locator: longBlock.tag + ' (' + longBlock.callable + ')',
          value: longBlock.blockCount + ' 块 / ' + longBlock.coreCount + ' 核，min ' + longBlock.durMin
            + ' / med ' + longBlock.durMed + ' / max ' + longBlock.durMax + ' us' }],
        { view: 'l2', tasks: [longBlock.tag] }),
      step('l1', 'descend',
        r2(wavesOf(longBlock)) + ' 波，span ≈ 单块时长，没有波量化可调',
        '块数 ' + longBlock.blockCount + ' 对核数 ' + longBlock.coreCount + ' 是 '
          + r2(wavesOf(longBlock)) + ' 波，块中位 ' + longBlock.durMed + ' us、最长 ' + longBlock.durMax
          + ' us，而 span ' + longBlock.span + ' us。setup 只占 ' + r2(longBlock.setupShare * 100)
          + '%，hand-off 只有 ' + longBlock.svOverhead + ' us —— 时间确实花在 kernel 里。'
          + (lbAic && lbAiv
            ? 'Cube ' + lbAic.blocks + ' 块 / ' + lbAic.coreTime + ' us，Vec ' + lbAiv.blocks + ' 块 / '
              + lbAiv.coreTime + ' us，两侧并行；解耦 Cube / Vec 不会缩短这一段。'
            : ''),
        [(lbAic && lbAiv) ? {
          artifact: 'merged_swimlane（按 FuncId 拆）',
          locator: longBlock.kernels.map((k) => 'FuncId ' + k.funcId + ' = ' + k.name).join(' / '),
          value: 'AIC ' + lbAic.blocks + ' 块 ' + lbAic.coreTime + ' us（最长 ' + lbAic.durMax + '）· AIV '
            + lbAiv.blocks + ' 块 ' + lbAiv.coreTime + ' us（最长 ' + lbAiv.durMax + '）· 串行下界 '
            + lbSerial + ' us vs 实测 span ' + longBlock.span + ' us',
        } : null,
        { artifact: 'deps.json', locator: 'task ' + longBlock.id,
          value: 'block_num=' + longBlock.blockNum + '，前驱 ' + longBlock.pred.length + ' 个，后继 '
            + longBlock.succ.length + ' 个' }],
        { view: 'l1', tasks: [longBlock.tag] }),
      step('l1', 'stop',
        '不下探 L0 / 编译器：缺块内证据',
        '要判断这 ' + longBlock.durMed + ' us 是 MTE、Cube 还是 Vec 撑起来的，'
          + '需要块内 PMU 或 pipe 级计数，本 dump 只有块级时长。'
          + (depthFor(longBlock.tag) ? '' : '该 scope 也没有对应的 PH-MR-001。')
          + '在补采之前，任何指向具体 pass 的结论都是猜的。',
        [], null),
    ],
    terminus: { level: 'l1', reason: '缺块内 PMU / pipe 计数，无法归因到 L0 或具体 pass；需重采' },
    evidence: [{ artifact: 'name_map.json', locator: 'callable_id_to_name',
      value: nameMapCount + ' 个 kernel 名 vs ' + R.scopes.length + ' 个 scope —— 差的正是被拆开的 mixed scope' }],
    focus: { view: 'l1', task: longBlock.tag },
    lever: '先补采块内 PMU；在此之前唯一可做的是缩小单块工作量（更小的 S1_TILE / 更少的 co-live 操作数），'
      + '并且要作为一次可回滚的对照实验做。',
    guardrail: '不要把「一个任务 span 很大」当成「它被串行化了」。本链已排除 Cube / Vec 串行、'
      + '排除 hand-off、排除波量化；剩下的解释只能靠新数据，不能靠推断。',
    verify: '重采带块内 PMU 的 trace（注意 PMU 打开会改变调度，不能与 PMU-off 基线直接比较），'
      + '确认块时长的构成后再选 pass。',
  },
].filter(Boolean);

/* ------------------------------------------------------------- hygiene
 * Real readings that do NOT carry a makespan attribution. They stay visible
 * -- a reader who saw the number elsewhere should find it here together with
 * the reason it is not in the queue -- but they never compete with a chain.
 */
const hygiene = [
  CAN.H1 && {
    id: 'H1', kind: 'hygiene', level: 'compiler', severity: 'low', axis: 'granularity',
    title: '搬运末维 < cache line ' + occOf(tileSiteList) + ' 次，其中只有 '
      + occOf(tileTunable) + ' 次可调',
    metric: '可调 ' + occOf(tileTunable) + ' / 标量 ' + occOf(tileScalar) + ' / 已过半线 ' + occOf(tileHalf),
    cost: null,
    unattributed: 'PH001 是静态提示，本 run 没有 MTE 级计数，无法把任何一次搬运折成墙钟。',
    claim: 'TileInnermostDimGranularity 在 ' + tileSiteList.length + ' 个源码点共报 '
      + occOf(tileSiteList) + ' 次。拆开看是三件不同的事：' + occOf(tileHalf)
      + ' 次末维已经 ≥ 半条 cache line（影响有限）；' + occOf(tileScalar) + ' 次是 '
      + Array.from(new Set(tileScalar.map((s) => s.shapes.join('')))).slice(0, 3).join(' / ')
      + ' 这类单元素访问，padding 也补不成一条 line（调不了）；真正值得调的是剩下 '
      + occOf(tileTunable) + ' 次 / ' + tileTunable.length + ' 个源码点，其中 ' + occOf(tileOnChain)
      + ' 次落在瓶颈链上任务的源码邻域（±40 行）。把 ' + occOf(tileSiteList)
      + ' 次当一个数字报，会把不可调项和热点项混成同一个优先级。',
    evidence: [
      { artifact: 'report/perf_hints.log', locator: '按最小末维分桶',
        value: Object.keys(tileSiteList.reduce((m, s) => { m[s.minB] = 1; return m; }, {}))
          .map(Number).sort((a, b) => a - b)
          .map((b) => b + 'B: ' + occOf(tileSiteList.filter((s) => s.minB === b)) + ' 次').join('，') },
      { artifact: 'report/perf_hints.log', locator: '单元素访问（shape [1]）',
        value: tileScalar.length + ' 个源码点 / ' + occOf(tileScalar) + ' 次，最小 '
          + (tileScalar.length ? Math.min.apply(null, tileScalar.map((s) => s.minB)) : '—') + 'B' },
    ].concat(tileOnChain.slice(0, 3).map((s) => ({
      artifact: 'report/perf_hints.log', locator: s.key,
      value: '最小末维 ' + s.minB + 'B · ' + Object.keys(s.ops).join('/') + ' · 落在链上任务邻域',
    }))),
    focus: { view: 'compiler', tab: 'granularity' },
    subjectsHint: { view: 'compiler', tab: 'granularity',
      sites: tileOnChain.concat(tileTunable).slice(0, 12).map((s) => s.key),
      files: tileFileList.map((f) => f.file) },
    lever: '只动落在链上的那几处：按 dtype 把末维凑到 '
      + (tileSiteList[0] ? tileSiteList[0].recB : 512) + 'B（BF16 → 256 元素倍数，FP32 → 128，INT8 → 512）。',
    guardrail: '加大末维会抬高 L0 / UB 占用'
      + (depthChainIds.length ? '，可能触发 ' + depthChainIds.join(' / ') + ' 里的深度回退，两项要一起看' : '')
      + '；'
      + '单元素访问不要碰，改不动还会掩盖真正的碎片。',
    verify: '重编译后核对可调桶的条数与最小末维，并复测对应 kernel 的块时长。',
  },
  CAN.H2 && {
    id: 'H2', kind: 'hygiene', level: 'l1', severity: 'low', axis: 'balance',
    title: worstImb.callable + ' 块时长离散 ' + worstImb.imbalance + 'x'
      + (imbMultiWave ? '，但决定 span 的不是尾块' : '，span 就等于最长块'),
    metric: 'max ' + worstImb.durMax + ' / med ' + worstImb.durMed + ' us · span ' + worstImb.span
      + ' us（' + r2((worstImb.span / SPAN) * 100) + '%）',
    cost: null,
    unattributed: imbMultiWave
      ? '离散度只解释 ' + r2(worstImb.durMax - worstImb.durMed) + ' us；该任务 span '
        + worstImb.span + ' us 里的大头是 ' + imbGap + ' us 的块间间隙'
        + (CAN.C2 ? '，已归入 C2' : '') + '。'
      : '整个任务只占 makespan ' + r2((worstImb.span / SPAN) * 100) + '%，'
        + '离散度最多值 ' + r2(worstImb.durMax - worstImb.durMed) + ' us；这个量级折不出墙钟收益。',
    claim: worstImb.blockCount + ' 块摊到 ' + worstImb.coreCount + ' 核（约 ' + imbWaves
      + ' 波），最慢块 ' + worstImb.durMax + ' us 是中位块的 ' + worstImb.imbalance
      + ' 倍 —— 离散度是真的。'
      + (imbMultiWave
        ? '但「尾块决定 span」算不过来：' + imbWaves + ' 波 × 中位 ' + worstImb.durMed + ' us = '
          + imbFloor + ' us 的工作量下界，span 却是 ' + worstImb.span + ' us，差的 ' + imbGap
          + ' us 是间隙不是尾块。'
        : '这是一波跑完的任务，尾块确实决定 span —— 但 span 一共才 ' + worstImb.span
          + ' us，把最慢块压到中位也只省 ' + r2(worstImb.durMax - worstImb.durMed) + ' us。')
      + (critTags.indexOf(worstImb.tag) >= 0 ? '' : '它也不在依赖关键路径上（slack ' + worstImb.slack + ' us）。')
      + (CAN.C2 ? '先看 C2，再谈切分。' : '这条读数本身也没有 makespan 归因。'),
    evidence: [
      { artifact: 'merged_swimlane blocks', locator: worstImb.tag + ' (' + worstImb.callable + ')',
        value: 'min ' + worstImb.durMin + ' / med ' + worstImb.durMed + ' / p90 ' + worstImb.durP90
          + ' / max ' + worstImb.durMax + ' us' },
      { artifact: '同一 trace', locator: '工作量下界 vs 实测 span',
        value: imbWaves + ' 波 × ' + worstImb.durMed + ' us = ' + imbFloor + ' us vs '
          + worstImb.span + ' us' },
      { artifact: 'deps.json', locator: 'task ' + worstImb.id,
        value: 'block_num=' + worstImb.blockNum + ', scope=' + worstImb.scope + ', slack '
          + worstImb.slack + ' us' },
    ],
    focus: { view: 'l1', task: worstImb.tag },
    subjectsHint: { view: 'l1', tasks: [worstImb.tag] },
    lever: '独立循环用 pl.parallel 而非 pl.range；'
      + (CAN.C2 ? '但排在 C2 之后做，否则改了切分也看不出 span 变化。' : '先确认它对 span 有影响再动。'),
    guardrail: '先确认慢块是工作量差异还是 MTE / UB 争用；PMU 打开会改变调度，不能与 PMU-off 基线直接比较。',
    verify: '重测该任务 durMax/durMed、间隙与 span 三项；只有 span 缩短才算生效。',
  },
  CAN.H3 && {
    id: 'H3', kind: 'hygiene', level: 'l2', severity: 'low', axis: 'sched',
    title: 'ready-but-undispatched 占窗口 ' + rqStat.busyShare.AIC + '%，但队列深度只有 ' + rqStat.avg.AIC,
    metric: 'avg ' + rqStat.avg.AIC + ' / peak ' + rqStat.peak.AIC + ' · AIC 占用 ' + aicUtil + '%',
    cost: null,
    unattributed: rqStat.avg.AIC < 1.5
      ? '平均队列深度 ' + rqStat.avg.AIC + ' 意味着那 ' + rqStat.busyShare.AIC
        + '% 的时间里基本只排着一个任务；这解释不了 AIC 占用只有 ' + aicUtil + '%。'
      : '队列平均 ' + rqStat.avg.AIC + '、峰值 ' + rqStat.peak.AIC
        + ' 确实不浅，但本 run 没有 dispatch 延迟计数，排队时长无法折成墙钟。',
    claim: 'shared_ready_queue 在 ' + rqStat.busyTime.AIC + ' us（窗口的 ' + rqStat.busyShare.AIC
      + '%）里有 AIC 任务已 ready 未派发，峰值 ' + rqStat.peak.AIC + ' 个，'
      + (rqStat.avg.AIC < 1.5 ? '但平均只有 ' : '平均 ') + rqStat.avg.AIC + ' 个。'
      + (rqStat.avg.AIC < 1.5
        ? '占用率低的主因是依赖饥饿'
          + (starveChainIds.length ? '（见 ' + starveChainIds.join(' / ') + '）' : '')
          + '，不是队列积压。作为独立瓶颈立不住，留在这里是为了让「我见过这个数」有个落点。'
        : '队列确实是深的，但本 run 没有 dispatch 延迟的直接计数，无法把排队时长折成墙钟；'
          + '要立成一条链得先补采。'),
    evidence: [
      { artifact: 'merged_swimlane queue counter', locator: 'shared_ready_queue',
        value: 'AIC avg ' + rqStat.avg.AIC + ', peak ' + rqStat.peak.AIC + ', >0 占 '
          + rqStat.busyShare.AIC + '%；AIV avg ' + rqStat.avg.AIV + ', peak ' + rqStat.peak.AIV },
      { artifact: 'worker lanes', locator: 'AIC 平均 vs AIV 平均', value: aicUtil + '% vs ' + aivUtil + '%' },
      { artifact: 'scheduler lanes', locator: '每线程占用',
        value: schedLanes.length + ' 条线程 · ' + schedPerLaneUtil + '% —— 都没饱和' },
    ],
    focus: { view: 'l2', overlay: 'ready' },
    subjectsHint: { view: 'l2', overlay: 'ready',
      lanes: lanes.filter((l) => l.kind === 'aic').sort((a, b) => a.util - b.util).slice(0, 6).map((l) => l.name) },
    lever: '不单独动它。'
      + (starveChainIds.length
        ? '如果 ' + starveChainIds.join(' / ') + ' 收敛后队列仍然 >0，再针对关键路径提前 dispatch。'
        : '先补采 dispatch 延迟计数，再决定是否针对关键路径提前 dispatch。'),
    guardrail: '提前 dispatch、改依赖、延后非关键任务都可能以吞吐换时延；两个指标都要报。',
    verify: (starveChainIds.length ? '在 ' + starveChainIds.join(' / ') + ' 的实验里' : '在下一轮实验里')
      + '顺带记录 ready>0 占比与平均深度，看它是否随之下降。',
  },
  CAN.H4 && {
    id: 'H4', kind: 'hygiene', level: 'l2', severity: 'low', axis: 'reuse',
    title: '本程序 0 处 pl.prefetch，属于未启用而非已损失',
    metric: 'prefetch ' + dsl.prefetch + ' / pipeline ' + dsl.pipeline + ' / spmd ' + dsl.spmd
      + ' / parallel ' + dsl.parallel,
    cost: null,
    unattributed: '这是一个缺失项。dump 里没有 L2 命中率或 SDMA 计数，无法说明它现在的损失是多少。',
    claim: '前端 IR 里 pl.prefetch 出现 ' + dsl.prefetch + ' 次。权重类输入确实存在（'
      + caseInfo.params.filter((p) => /^w/.test(p.name)).length
      + ' 个 w* 参数），但没有静态预取，也没有 N-group swizzle 证据。'
      + '把「没用某个能力」写成瓶颈会污染优先级 —— 它要先成为一个假设，再成为一条发现。',
    evidence: [
      { artifact: 'passes_dump/00_frontend.py', locator: 'pl.prefetch', value: dsl.prefetch + ' 处' },
      { artifact: 'distributed_meta.json', locator: 'w* 参数',
        value: caseInfo.params.filter((p) => /^w/.test(p.name))
          .map((p) => p.name.replace(/__ssa_v0$/, '') + ' ' + p.dtype + JSON.stringify(p.shape))
          .slice(0, 4).join(', ') },
    ],
    focus: { view: 'l2', overlay: 'none' },
    subjectsHint: { view: 'l2', absent: true },
    lever: '只对静态、确定会冷、且所有在途 warm 数据能放进 L2 的权重用 pl.prefetch。',
    guardrail: 'prefetch 占 SDMA；错误预取会拖慢通信或挤掉真正需要的数据。'
      + (CAN.C1 ? '本 case 通信已是瓶颈（C1），风险更高。' : ''),
    verify: '加预取后同时看 device_wall、通信段 span 与 SDMA 占用，三者不能互相恶化。',
  },
].filter(Boolean);

/* Chains and hygiene share one list so every existing lens (search, task
 * inspector, ledger) keeps working; `kind` is what tells them apart. */
const allFindings = findings.concat(hygiene);

/* --------------------------------------------------- finding -> subjects
 * A finding is only useful if the reader can see it on the stage. Each one
 * names the concrete objects the centre view should mark (tasks, source
 * sites, scheduler phases), so the stage can number them instead of leaving
 * the reader to guess which parts the inspector is talking about. For a
 * chain, the top-level subjects are its observe step's -- that is where the
 * reader lands -- and every step also carries its own set for the ladder. */
const taskByTag = {};
tasks.forEach((t) => { taskByTag[t.tag] = t; });

const mkSubjects = (s, fallbackView) => ({
  view: (s && s.view) || fallbackView || 'l2',
  tab: (s && s.tab) || null,
  overlay: (s && s.overlay) || null,
  tasks: (s && s.tasks) || [],
  lanes: (s && s.lanes) || [],
  sites: (s && s.sites) || [],
  files: (s && s.files) || [],
  ranks: (s && s.ranks) || [],
  schedPhases: (s && s.schedPhases) || [],
  absent: !!(s && s.absent),
});
const mkChips = (sub) => []
  .concat(sub.tasks.map((tag) => {
    const t = taskByTag[tag];
    return { kind: 'task', id: tag, label: t ? t.callable : tag, value: t ? t.span + ' us' : '' };
  }))
  .concat(sub.sites.map((key) => {
    const d = depthSiteList.find((x) => x.key === key);
    const g = tileSiteList.find((x) => x.key === key);
    return {
      kind: 'site', id: key, label: key,
      value: d ? 'depth ' + d.maxReqDepth + '→' + d.fittedDepth : (g ? g.minB + 'B' : ''),
    };
  }))
  .concat(sub.lanes.map((name) => {
    const l = lanes.find((x) => x.name === name);
    return { kind: 'lane', id: name, label: name, value: l ? r2(l.util) + '%' : '' };
  }))
  .concat(sub.ranks.map((k) => ({
    kind: 'rank', id: k, label: k,
    value: e2e && e2e[k] ? e2e[k][2]['chip.run.runner_run.device_wall'].us + ' us' : '',
  })))
  .concat(sub.schedPhases.map((p) => ({
    kind: 'phase', id: p, label: 'phase ' + p,
    value: schedPhases[p] ? schedPhases[p].us + ' us' : '',
  })));

allFindings.forEach((f) => {
  (f.chain || []).forEach((st) => {
    st.subjects = mkSubjects(st.subjects, st.level === 'l0' ? 'isa' : st.level);
    st.chips = mkChips(st.subjects);
  });
  const observe = (f.chain || []).filter((st) => st.role === 'observe')[0] || (f.chain || [])[0];
  f.subjects = mkSubjects(f.subjectsHint || (observe && observe.subjects) || null,
    (f.focus && f.focus.view) || f.level);
  delete f.subjectsHint;
  /* chips shown in the centre evidence bar, each one jumpable */
  f.chips = mkChips(f.subjects);
  /* every level the chain actually visits, for the level filter */
  f.levels = Array.from(new Set([f.level].concat((f.chain || []).map((st) => st.level))));
  /* the pass a chain landed on, if any -- read off the root step's own tab so
   * the Pass view can link back without a second hand-written table */
  const root = (f.chain || []).filter((st) => st.role === 'root' && st.level === 'compiler')[0];
  f.rootPass = root && root.subjects.tab === 'depth' ? 'MemoryReuse' : null;
  /* flattened evidence keeps the existing evidence list working: the chain's
   * own rows first, then every step's, in ladder order */
  f.evidence = (f.evidence || []).concat(
    (f.chain || []).reduce((a, st) => a.concat(st.evidence || []), []));
});

/* Cross-layer links are emitted only where this run has direct evidence.
 * They keep Pass inspection inside the tuning loop, not beside it. */
const findingIds = new Set(allFindings.map((f) => f.id));
/* a pass links back to the chains that actually landed on it, so the reader
 * arrives at MemoryReuse already knowing which chain sent them */
const chainsRooted = (passName) => allFindings.filter((f) => f.rootPass === passName);
passEvidenceIndex.forEach((detail) => {
  const pass = passes.find((p) => p.idx === detail.idx);
  detail.links = [];
  if (!pass) return;
  chainsRooted(pass.name).forEach((f) => {
    detail.links.push({ findingId: f.id, label: '关联 ' + f.id + ' · ' + f.title });
  });
  if (pass.name === 'AutoTileMatmulL0' && l0Tiles.length) {
    detail.links.push({ view: 'isa', label: '查看 ISA / 布局中的 L0 tile' });
  }
});
/* Investigations are the product-level objects built from this run's chains.
 * Views remain evidence lenses; the task, its hypothesis and its experiment
 * are what a developer actually carries through a tuning loop. One chain is
 * already "discovered -> followed down -> landed or stopped", so an
 * investigation is built per chain rather than by pairing loose readings, and
 * the hygiene items only ever appear as competing explanations to rule out. */
const investigationIds = new Set(allFindings.map((f) => f.id));
const includeFinding = (id) => (investigationIds.has(id) ? id : null);
const compactIds = (ids) => ids.filter(Boolean);
const byId = {};
allFindings.forEach((f) => { byId[f.id] = f; });
const investigations = [];
const addInvestigation = (id, title, status, target, findingRefs, hypotheses, experiments) => {
  investigations.push({
    id, title, status, target,
    baseline: '当前 run · ' + caseInfo.program,
    owner: '待分派',
    findings: compactIds(findingRefs), hypotheses, experiments,
  });
};

/* A chain whose terminus is the compiler can be falsified by a recompile; one
 * that stopped earlier can only be falsified by new data. That difference is
 * the investigation's status, not a note in its body. */
const statusOf = (f) => (f.terminus.level === 'compiler' ? '需要实验' : '需要补证');

if (investigationIds.has('C1')) {
  const f = byId.C1;
  addInvestigation('INV-024', '把集合点等待归因到 rank 启动错峰', statusOf(f),
    '在不改通信算法的前提下对齐两卡下发时刻，验证等待是否同比收窄。',
    [includeFinding('C1'), includeFinding('H3')],
    [
      { id: 'H-01', title: 'rank 启动错峰解释了等待的量级', level: '强支持',
        claim: f.chain.filter((s) => s.role === 'root')[0].detail,
        evidence: ['C1'],
        need: '缩小启动偏移后，' + waitTasks.length + ' 个 *_wait 的合计是否同比下降。' },
      { id: 'H-02', title: '调度队列是竞争解释，不是前提', level: '待区分',
        claim: investigationIds.has('H3')
          ? byId.H3.unattributed
          : '本 case 没有 ready queue 计数，无法作为竞争解释评估。',
        evidence: compactIds([includeFinding('H3')]),
        need: '同一实验里记录 ready>0 占比与平均深度，看它是否随等待一起变化。' },
    ],
    [{ id: 'EXP-024-01', status: '待执行', name: '只对齐两卡的启动 / 下发时刻',
      change: '不改通信算法、Tile、融合边界',
      measures: 'runner_run ts 差 · Σ(*_wait) · device_wall',
      guardrail: '结果校验、吞吐、host bind 时间' }]);
}

if (investigationIds.has('C2')) {
  const f = byId.C2;
  const rooted = f.terminus.level === 'compiler';
  addInvestigation('INV-025', '把块间间隙归因到调度粒度与流水深度', statusOf(f),
    '让单块做更多事、块数下降，验证 ' + f.cost.us + ' us 的间隙是否随之收敛。',
    [includeFinding('C2'), includeFinding('H2'), includeFinding('H1')],
    [
      { id: 'H-01', title: '间隙来自每块的 dispatch / complete 次数，而不是核上算得慢', level: '强支持',
        claim: f.chain.filter((s) => s.role === 'descend')[0].detail,
        evidence: ['C2'],
        need: '块数下降后，间隙与 span 是否同向下降（只有间隙降说明瓶颈换位）。' },
      rooted
        ? { id: 'H-02', title: '块压不下来是因为 L0 预算把流水深度逼到 1', level: '可编译验证',
          claim: f.chain.filter((s) => s.role === 'root')[0].detail,
          evidence: ['C2'],
          need: '减小同驻 tile 后该源码点的 PH-MR-001 是否消失、块数是否下降。' }
        : { id: 'H-02', title: '编译器层在本 dump 中无证据', level: '缺证据',
          claim: f.terminus.reason, evidence: ['C2'],
          need: '先在 L1 侧验证「减少块数」是否收敛间隙，再决定是否需要编译侧证据。' },
      { id: 'H-03', title: '块时长离散是竞争解释', level: '待区分',
        claim: investigationIds.has('H2') ? byId.H2.unattributed : '本 case 无显著离散任务。',
        evidence: compactIds([includeFinding('H2')]),
        need: '同一实验里同时记录 durMax/durMed 与间隙，区分尾块与空档。' },
    ],
    [{ id: 'EXP-025-01', status: rooted ? '待执行' : '待规划',
      name: rooted ? '减小同驻 tile，恢复流水深度' : '把外层迭代折进核内，减少块数',
      change: '只动一个源码点的 tile 或迭代结构',
      measures: 'PH-MR-001 条数 · 块数 · 块中位时长 · 任务 span · 间隙',
      guardrail: '不接受「间隙下降但 span 不动」；不通过调大 stage 换取局部收益' }]);
}

if (investigationIds.has('C3')) {
  const f = byId.C3;
  addInvestigation('INV-026', '区分早发空等与真正的 hand-off 开销', statusOf(f),
    '先证明 duration − kernel_duration 是等数据而不是准备工作，再决定合核还是改依赖。',
    [includeFinding('C3'), includeFinding('C4')],
    [
      { id: 'H-01', title: '消费者在生产者完成前上核，核上时间花在等数据', level: hbLink ? '强支持' : '待验证',
        claim: f.chain.filter((s) => s.role === 'observe')[0].detail,
        evidence: ['C3'],
        need: '推迟该任务的 dispatch 或补实依赖后，setup 占比是否下降、span 是否不变或下降。' },
      { id: 'H-02', title: 'Vec / L0 侧缺双缓冲，等待无法被搬运掩盖', level: stallSite ? '可编译验证' : '缺证据',
        claim: stallSite
          ? f.chain.filter((s) => s.role === 'root')[0].detail
          : f.terminus.reason,
        evidence: ['C3'],
        need: stallSite
          ? '复现「预算明显放得下却仍降级」，这是独立于调度的 pass 问题。'
          : '先在 L2 侧验证依赖声明与 dispatch 时机。' },
    ],
    [{ id: 'EXP-026-01', status: '待执行', name: '只推迟该消费者的 dispatch',
      change: '不改融合边界、不改 Tile',
      measures: 'duration − kernel_duration · aicpu-duration · hb_violation 条数 · span',
      guardrail: 'setup 占比下降但 span 不降，说明等待只是换了地方' }]);
}

if (investigationIds.has('C4')) {
  const f = byId.C4;
  addInvestigation('INV-027', '补采块内证据后再决定 ' + byId.C4.chain[0].subjects.tasks[0] + ' 的方向', '需要补证',
    '这一段占 makespan ' + f.cost.share + '%，但本 dump 无法把它归因到 L0 或具体 pass。',
    [includeFinding('C4')],
    [
      { id: 'H-01', title: '已排除的解释', level: '已排除',
        claim: f.guardrail, evidence: ['C4'],
        need: '不需要再验证：波量化、hand-off、Cube / Vec 串行都已用本 run 的数据排除。' },
      { id: 'H-02', title: '块时长由 MTE / Cube / Vec 中的哪一段撑起来', level: '缺证据',
        claim: f.chain.filter((s) => s.role === 'stop')[0].detail, evidence: ['C4'],
        need: '带块内 PMU 重采一次；PMU 会改变调度，需要 PMU-on 的自有基线。' },
    ],
    [{ id: 'EXP-027-01', status: '待规划', name: '带块内 PMU 重采同一 case',
      change: '不改代码，只改采集',
      measures: 'pipe 级计数 · 块时长构成 · PMU-on 基线 span',
      guardrail: '不与 PMU-off 基线直接比较' }]);
}

if (!investigations.length) {
  const first = allFindings[0];
  addInvestigation('INV-001', '建立首个可证伪的性能假设', '待分诊',
    '用一个可回滚改动验证最高影响发现是否能改善端到端结果。', [first && first.id],
    [{ id: 'H-01', title: '最高优先级发现值得进一步验证', level: '待建模',
      claim: '当前只有同层观测，尚未形成跨层解释。',
      evidence: [first && first.id].filter(Boolean),
      need: '先绑定端到端指标和最小改动，再开始实验。' }],
    [{ id: 'EXP-001-01', status: '待规划', name: '定义最小单变量实验', change: '待选择',
      measures: '局部指标 · device wall · 正确性', guardrail: '保持同一基线与采样条件' }]);
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
  /* one list, two kinds: `chain` rows carry a makespan attribution and a
   * layer ladder, `hygiene` rows carry the reason they do not */
  findings: allFindings,
  chainCount: findings.length,
  hygieneCount: hygiene.length,
  investigations: investigations,
  launchSkew: launchSkew,
  sourceMap: sourceMap,
  /* ------------------------------------------------ evidence ladder
   * Each layer names its input, the one question it answers, and -- the
   * part that matters -- what it CANNOT settle on its own, with a pointer
   * to the layer that can. A layer whose artifact this dump does not carry
   * is a first-class state, not a blank row. */
  evidence: [
    {
      id: 'serving', name: '服务层',
      input: 'serving-strace-swimlane.json、端到端 benchmark',
      answers: '哪个 WorkerProcess 负载或尾耗时异常',
      cannot: 'AICore 内某个 kernel 为什么慢',
      cannotGoto: 'l1',
      state: 'absent',
      note: '本 dump 是单进程 JIT run，没有 serving 侧采集；多 WorkerProcess 的负载与尾耗时无从谈起。',
    },
    {
      id: 'hostdev', name: 'Host / Device',
      input: 'BenchmarkStats、独立 benchmark',
      answers: '延迟是 Host、Device，还是两者共同贡献',
      cannot: 'Device 内的依赖与 pipe 根因',
      cannotGoto: 'l2',
      state: e2e ? 'partial' : 'absent',
      have: e2e ? 'STRACE host span：bind / runner_run / device_wall / sched，两种时钟已对齐' : null,
      note: e2e
        ? 'BenchmarkStats 缺席：没有 rounds / warmup 统计量，本 case 只有 '
          + Object.keys(e2e[RANK_KEYS[0]]).length + ' 次调用，给不出 mean / median。'
        : '本 dump 没有 host STRACE log，Host 与 Device 的拆分整层缺席。',
    },
    {
      id: 'funcs', name: '函数汇总',
      input: 'name_map*.json + Swimlane',
      answers: '慢来自单次慢、次数多，还是波动',
      cannot: '是否影响 wall-clock',
      cannotGoto: 'l2',
      state: 'ok',
      have: R.scopes.length + ' 个 scope / ' + nameMapCount + ' 个 kernel 名，每块时长齐全',
      note: 'Σ core-time 大不等于拖慢墙钟 —— 一个 scope 可能摊在很多核上并行跑完。'
        + '要不要动它，看它在不在路径上，那是下一层的事。',
    },
  ],
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
console.log('    hints', hints.length, '| passes', passes.length,
  '| chains', findings.map((f) => f.id + '→' + f.terminus.level).join(' '),
  '| hygiene', hygiene.length,
  '| e2e', e2e ? Object.keys(e2e[RANK_KEYS[0]] || {}).length + ' inv' : 'absent');
return payload;
}

const BUILT = {};
CASES.forEach((c) => { BUILT[c.id] = buildCase(c); });

fs.writeFileSync(OUT, 'window.TUNING_RUNS = ' + JSON.stringify(BUILT) + ';\n'
  + 'window.TUNING_CASES = ' + JSON.stringify(CASES.map((c) => ({ id: c.id, label: c.label, sub: c.sub }))) + ';\n'
  + 'window.TUNING_RUN = window.TUNING_RUNS[' + JSON.stringify(CASES[0].id) + '];\n');
console.log('wrote', OUT, (fs.statSync(OUT).size / 1024).toFixed(1) + ' KB');
