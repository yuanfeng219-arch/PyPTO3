/* Task history — engineering records for model / operator work.

   Model (mirrors how artifacts land on disk):
     任务 Task   stable target, e.g. Data/_jit_<operator>_*   — identity
       └ 运行 Run   one timestamped execution, e.g. _20260625_184941
           └ 产物   passes_dump / kernels / orchestration / ptoas /
                    dfx_outputs / report / debug
   Artifacts belong to a RUN, never to a task. A task's verdict is its latest
   run's verdict. Artifact buttons carry data-step / data-open-runs so
   demo-v2.js keeps owning stage switching. */
(function () {
  'use strict';

  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.prototype.slice.call((r || document).querySelectorAll(s));
  const esc = (s) => String(s).replace(/[&<>"']/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  const VERDICT = {
    blocked: ['编译阻塞', 'bad'], passed: ['编译通过', 'ok'], warn: ['有告警', 'warn'],
    trusted: ['可信基线', 'ok'], running: ['执行中', 'run'], purged: ['产物已清理', 'idle']
  };
  /* Run state is deliberately separate from diagnostic quality. A completed
     Run can still carry a correctness failure or a performance warning. */
  const DOMAIN_ORDER = ['compilation', 'correctness', 'execution', 'performance', 'resources'];
  /* 一级导航与 Overview 的 domain 卡共用这份标签。Performance 不作为一个页签存在
     （结论与证据都已并入 Execution），所以它在这里是一个域名称「性能」，而不是
     一个英文页签名 —— 免得用户回头去找一个不存在的页签。 */
  const DOMAIN_LABEL = { compilation: '编译', correctness: '正确性', execution: '执行', performance: '性能', resources: '资源' };
  const DOMAIN_VERDICT = {
    pass: ['PASS', 'ok'], warning: ['WARNING', 'warn'], fail: ['FAIL', 'bad'],
    not_evaluated: ['NOT EVALUATED', 'idle'], unknown: ['UNKNOWN', 'idle']
  };
  const STATE_LABEL = { running: '运行中', completed: '已完成', failed: '失败', incomplete: '不完整', cancelled: '已取消' };
  const EVIDENCE_STATUS = { available: ['✓', '可用'], partial: ['◐', '部分'], not_collected: ['—', '未采集'], unavailable: ['—', '不可用'] };
  const EVIDENCE_LABEL = {
    golden_compare: 'Golden Compare', ir_validation: 'IR Validation', tensor_dump: 'Intermediate Tensor',
    dependency_graph: 'Dependency Graph', runtime_timeline: 'Runtime Timeline', pmu: 'PMU', scope_stats: 'Runtime Resources',
    pass_dump: 'Pass Dump', source_location: 'Source Location', args_dump: 'Args Dump', core_trace: 'Core Trace',
    critical_path: 'Critical Path', memory_map: 'Memory Map'
  };

  function makeDomains(values) {
    const unknown = { verdict: 'unknown', summary: '无可用结论' };
    return DOMAIN_ORDER.reduce((all, key) => { all[key] = Object.assign({}, unknown, values[key] || {}); return all; }, {});
  }
  function makeEvidence(values) {
    return Object.keys(EVIDENCE_LABEL).reduce((all, key) => {
      all[key] = Object.assign({ status: 'not_collected', label: EVIDENCE_LABEL[key] }, values[key] || {});
      return all;
    }, {});
  }
  function runModel(state, domains, evidence, findings, extra) {
    return Object.assign({ state, domains: makeDomains(domains), evidence: makeEvidence(evidence), findings: findings || [], baseline: false }, extra || {});
  }
  function getRunModel(run) { return run && run.model || runModel('incomplete', {}, {}, []); }
  function getDomainVerdict(run, domain) { return getRunModel(run).domains[domain] || { verdict: 'unknown', summary: '无可用结论' }; }
  function getEvidenceCoverage(run) { return getRunModel(run).evidence; }
  function getFindings(run) { return getRunModel(run).findings || []; }
  function runDisplayId(run) { return run && (run.displayId || run.id) || '—'; }
  function isCompileFailureStory(run) { return !!getRunModel(run).compileStory; }
  function isCorrectnessFailureStory(run) { return !!getRunModel(run).correctnessStory; }
  function isPerformanceWarningStory(run) { return !!getRunModel(run).performanceStory; }
  function isValidatedOptimizationStory(run) { return !!getRunModel(run).optimizationStory; }
  /* 这一次 Run 的 Correctness 页签自绘诊断视图：正确性失败的 Run 用它，
     磁盘上那次真实 Run 也用它（沿用 compilationDataMatches 的「数据归属」判定，
     不额外写死 run id），其余历史 Run 保持原样。 */
  function usesCorrectnessDiagnosis(run) {
    return isCorrectnessFailureStory(run) || compilationDataMatches(run);
  }
  function getRunDisplayStatus(run) {
    const m = getRunModel(run), d = m.domains;
    if (m.baseline) return ['可信基线', 'ok'];
    if (m.state === 'running') return ['运行中', 'run'];
    const failed = DOMAIN_ORDER.find(key => d[key].verdict === 'fail');
    if (failed) return [{ compilation: '编译失败', correctness: '正确性失败', execution: '执行失败', performance: '性能失败', resources: '资源失败' }[failed], 'bad'];
    const warned = ['performance', 'resources', 'execution', 'compilation', 'correctness'].find(key => d[key].verdict === 'warning');
    if (warned) return [warned === 'performance' ? '性能告警' : warned === 'resources' ? '资源告警' : DOMAIN_LABEL[warned] + ' 告警', 'warn'];
    if (m.state === 'failed') return ['运行失败', 'bad'];
    if (m.state === 'incomplete') return ['证据不完整', 'idle'];
    if (d.compilation.verdict === 'pass' && d.correctness.verdict === 'pass' && d.execution.verdict === 'pass') return ['已验证', 'ok'];
    return [STATE_LABEL[m.state] || '未知状态', 'idle'];
  }
  const DIRLABEL = {
    passes_dump: 'IR 快照', kernels: 'Kernel 源码', orchestration: '编排代码',
    ptoas: '汇编', dfx_outputs: '运行时 Trace', report: '编译报告', debug: '调试'
  };

  /* ---------- facts from the real run ----------
     Everything here is read off ir-kernels-data.js, which is generated from
     the run directory (passes_dump + report/). No shape is invented. */
  function liveRun() {
    const K = window.PTO_IR_KERNELS;
    if (!K) return null;
    const pct = (a, b) => (b ? (a / b) * 100 : 0);

    // worst-loaded memory space per kernel
    const worst = (k) => {
      let w = -1, sp = null, u = 0;
      for (const s of K.spaces) {
        const p = pct(k.mem[s] || 0, K.limits[s]);
        if (p > w) { w = p; sp = s; u = k.mem[s] || 0; }
      }
      return { p: w, sp, u, lim: K.limits[sp] };
    };

    const kmem = K.kernels.map(k => {
      const w = worst(k);
      return { n: k.name, t: k.type, sp: w.sp, u: w.u, lim: w.lim,
               p: Math.round(w.p), diag: k.diags.length };
    }).filter(x => x.u > 0).sort((a, b) => b.p - a.p || b.u - a.u);

    // per-space rollup: how many kernels live there and how hot it gets
    const spaces = K.spaces.map(s => {
      const on = K.kernels.filter(k => (k.mem[s] || 0) > 0);
      const peak = on.reduce((a, k) => Math.max(a, pct(k.mem[s], K.limits[s])), 0);
      return { sp: s, n: on.length, lim: K.limits[s], peak: Math.round(peak) };
    }).filter(x => x.n).sort((a, b) => b.peak - a.peak);

    // functions that never allocate on chip (orchestration / spmd / group
    // wrappers) — shown as a muted group so the grid accounts for all 45
    const noMem = K.kernels.filter(k => !worst(k).u).map(k => ({ n: k.name, t: k.type }));

    const types = {};
    K.kernels.forEach(k => { types[k.type] = (types[k.type] || 0) + 1; });

    const nb = K.kernels.reduce((a, k) => a + k.reuse.before.n, 0);
    const na = K.kernels.reduce((a, k) => a + k.reuse.after.n, 0);
    const bb = K.kernels.reduce((a, k) => a + k.reuse.before.b, 0);
    const ba = K.kernels.reduce((a, k) => a + k.reuse.after.b, 0);
    const noGain = K.kernels.filter(k => k.reuse.before.b > 0 && k.reuse.after.b >= k.reuse.before.b).length;

    const demotedK = K.kernels.filter(k => k.intent.demoted > 0)
      .map(k => ({ n: k.name, d: k.intent.demoted, decl: k.intent.declared.length }));
    const l0K = K.kernels.filter(k => k.intent.l0.length)
      .map(k => ({ n: k.name, tiles: k.intent.l0.length }));

    const m = (K.source || '').match(/^_jit_(.+)_(\d{8})_(\d{6})$/);
    return {
      op: m ? m[1] : (K.source || 'unknown'),
      stamp: m ? m[2] + '_' + m[3] : '',
      time: m ? m[2].replace(/(\d{4})(\d\d)(\d\d)/, '$1-$2-$3') + ' ' +
                m[3].replace(/(\d\d)(\d\d)(\d\d)/, '$1:$2:$3') : '',
      dir: K.source,
      passes: K.passNames.length, kernels: K.kernels.length, types,
      kmem, spaces, noMem,
      atLimit: kmem.filter(x => x.u === x.lim).length,
      over: kmem.filter(x => x.u > x.lim).length,
      peak: kmem.length ? kmem[0].p : 0, peakK: kmem.length ? kmem[0].n : '',
      peakSp: kmem.length ? kmem[0].sp : '',
      errors: 0, warns: 0,
      hints: (K.perfHints || []).slice(), minInner: K.perfMinInnermost || 512,
      // only the "half" cause is a change the operator author can make; "row"
      // is blocked by the paged KV layout and "scalar" is not a defect at all
      actionable: (K.perfHints || []).filter(h => h.cause === "half").length,
      byCause: (K.perfHints || []).reduce((a, h) => { a[h.cause] = (a[h.cause] || 0) + 1; return a; },
                                          { half: 0, row: 0, scalar: 0 }),
      hintKernels: K.kernels.filter(k => k.diags.length).length,
      reuse: { nb, na, bb, ba, noGain, pct: bb ? Math.round((1 - ba / bb) * 100) : 0 },
      declared: K.kernels.reduce((a, k) => a + k.intent.declared.length, 0),
      demotedK, l0K,
      target: K.target,
      inventory: K.inventory || []
    };
  }

  const ART = (over) => Object.assign({
    overview: { k: 'overview', label: '任务契约', step: 0 },
    source: { k: 'source', label: '算子源码', explorer: true },
    compile: { k: 'compile', label: '编译 IR 全流程', step: 2, primary: true },
    correct: { k: 'correct', label: '正确性比对', step: 3 },
    baseline: { k: 'baseline', label: '环境与基线', step: 4 }
  }, over);

  function buildTasks() {
    const r = liveRun();
    const tasks = [];

    /* --- the one task with a real run directory on disk --- */
    tasks.push({
      id: 'task_decode', kind: '算子', model: 'qwen3-14b',
      op: r ? r.op : 'decode_fwd_layers', branch: 'kernel/decode-layer',
      title: 'Decode Layer 融合算子',
      runs: [
        {
          id: 'run_105', displayId: '#105', time: '12:47', target: 'Ascend 910B',
          model: runModel('failed', {
            compilation: { verdict: 'fail', summary: 'LegalizeIndexing 校验失败' },
            correctness: { verdict: 'not_evaluated', summary: '编译未完成' },
            execution: { verdict: 'not_evaluated', summary: '未进入运行时执行' },
            performance: { verdict: 'not_evaluated', summary: '未进入运行时执行' },
            resources: { verdict: 'unknown', summary: '无资源证据' }
          }, {
            ir_validation: { status: 'available' }, pass_dump: { status: 'available' }, source_location: { status: 'available' },
            golden_compare: { status: 'not_collected' }, dependency_graph: { status: 'not_collected' }, runtime_timeline: { status: 'not_collected' },
            pmu: { status: 'not_collected' }, scope_stats: { status: 'not_collected' }
          }, [{
            id: 'F105', severity: 'critical', domain: 'compilation',
            title: '动态索引在 lowering 后不满足 GM store 地址约束',
            summary: 'IR verification 在 LegalizeIndexing 后首次失败，设备代码未生成。',
            location: 'LegalizeIndexing · decode_layer.py:728',
            affectedObjects: [{ kind: 'pass', id: 'LegalizeIndexing' }, { kind: 'ir_op', id: 'tensor.write[index]' }, { kind: 'source', id: 'decode_layer.py:728' }],
            evidence: ['上一个 Pass 校验：PASS', 'LegalizeIndexing：FAIL', 'offending op: tensor.write[index]', 'index 类型 / 地址约束不一致', '源码区间可用'],
            action: { label: '查看编译', route: 'compilation' }
          }], {
            compileStory: true,
            lineage: 'initial',
            source: { file: 'decode_layer.py', line: 728 },
            fix: 'dynamic index → affine fallback'
          }),
          artifacts: [Object.assign({}, ART().source, { meta: 'decode_layer.py:728 · 源码区间', tone: 'ok' }), Object.assign({}, ART().compile, { meta: 'LegalizeIndexing · 校验失败', tone: 'bad', primary: true })]
        },
        {
          id: r ? r.stamp : '20260625_184941', live: !!r,
          verdict: r && r.errors ? 'blocked' : 'passed',
          model: runModel('completed', {
            compilation: { verdict: 'pass', summary: r ? r.passes + ' Pass · ' + r.kernels + ' Kernel' : '—' },
            correctness: { verdict: 'pass', summary: '16 / 16 Golden Compare match' },
            execution: { verdict: 'pass', summary: '428 Task · Dependency Graph 完整' },
            performance: { verdict: 'warning', summary: '关键链等待占比 61%' },
            resources: { verdict: 'pass', summary: 'L0B peak 100% · 无溢出' }
          }, {
            golden_compare: { status: 'available' }, ir_validation: { status: 'available' },
            tensor_dump: { status: 'partial' }, dependency_graph: { status: 'available' },
            runtime_timeline: { status: 'available' }, pmu: { status: 'not_collected' }, scope_stats: { status: 'not_collected' }
          }, [{ id: 'F001', severity: 'warning', domain: 'performance', title: 'RoPE 的 lo/hi 半维被拆成两次搬运', summary: '实际搬运宽度低于目标，影响内存效率。', affectedObjects: [{ kind: 'kernel', id: 'rope_qkv' }], evidence: ['实际宽度 128 / 256 B', '目标宽度 512 B', '16 次触发'], action: { label: '在「执行」中查看性能证据', route: 'performance' } }]),
          time: r ? r.time : '2026-06-25 18:49:41',
          target: r ? r.target : 'Ascend 910B',   // no real compile duration on disk
          dir: r ? 'Data/' + r.dir : '',
          inventory: r ? r.inventory : [],
          artifacts: [
            Object.assign({}, ART().overview, { meta: 'hidden [16,5120] FP32 · 40Q / 8KV', tone: 'ok' }),
            Object.assign({}, ART().source, { meta: 'decode_layer.py · 在资源管理器中打开', tone: 'ok' }),
            Object.assign({}, ART().compile, {
              meta: r ? r.passes + ' pass · ' + r.kernels + ' kernel · 峰值 ' + r.peak + '%' : '',
              tone: r && r.peak >= 90 ? 'warn' : 'ok' }),
            Object.assign({}, ART().correct, { meta: '待运行 · 未产出比对结果', tone: 'idle' }),
            Object.assign({}, ART().baseline, { meta: 'env:8da1bf09 · 4 / 4 门禁', tone: 'ok' })
          ],
          // live runs render measured sections instead of summary chips
          signals: []
        },
        { id: '20260624_101233', verdict: 'purged', time: '2026-06-24 10:12:33',
          duration: '1m03s', target: 'Ascend 910B', purged: true,
          model: runModel('completed', {
            compilation: { verdict: 'pass', summary: '42 Pass · 44 Kernel' }, correctness: { verdict: 'fail', summary: '首个分歧：Tensor T37' },
            execution: { verdict: 'pass', summary: '421 Task · Timeline 完整' }, performance: { verdict: 'not_evaluated', summary: '正确性失败后未评估' }, resources: { verdict: 'pass', summary: 'L0B peak 81%' }
          }, { golden_compare: { status: 'available' }, ir_validation: { status: 'available' }, tensor_dump: { status: 'available' }, dependency_graph: { status: 'available' }, runtime_timeline: { status: 'available' } },
          [{ id: 'F002', severity: 'critical', domain: 'correctness', title: '输出从 Tensor T37 开始分歧', summary: 'Golden Compare 首次出现输出不匹配。', affectedObjects: [{ kind: 'tensor', id: '37' }], evidence: ['Golden Compare FAIL', '首个分歧 Tensor T37', '上游 Tensor T36 match'], action: { label: '查看正确性证据', route: 'correctness' } }]),
          note: '产物目录已清理，仅保留结论：AutoTileMatmulL0 前的版本，L0B 尚未打满。' },
        { id: '20260620_093015', verdict: 'blocked', time: '2026-06-20 09:30:15',
          duration: '3m58s', target: 'Ascend 910B', purged: true,
          model: runModel('failed', {
            compilation: { verdict: 'fail', summary: 'AllocateMemoryAddr 未完成' }, correctness: { verdict: 'not_evaluated', summary: '编译失败，未执行' },
            execution: { verdict: 'not_evaluated', summary: '编译失败，未执行' }, performance: { verdict: 'not_evaluated', summary: '未执行' }, resources: { verdict: 'unknown', summary: '无可用资源结论' }
          }, { ir_validation: { status: 'partial' } },
          [{ id: 'F003', severity: 'critical', domain: 'compilation', title: '片上分配无法满足 L0B 约束', summary: '编译在内存分配阶段中止。', affectedObjects: [{ kind: 'buffer', id: 'q_proj' }], evidence: ['AllocateMemoryAddr FAIL', 'L0B 请求超出上限'], action: { label: '查看编译证据', route: 'compilation' } }]),
          note: '产物目录已清理，仅保留结论：首次跑通全部 42 个 pass。' }
      ]
    });

    /* --- archive tasks (no run directory in Data/) --- */
    tasks.push({
      id: 'task_rmsrope', kind: '算子', model: 'qwen3-32b', op: 'rmsnorm_rope',
      branch: 'kernel/rmsnorm-rope', title: 'RMSNorm + RoPE 融合内核',
      runs: [{
        id: '20260618_100712', verdict: 'trusted', time: '2026-06-18 10:07:12',
        duration: '1m52s', target: 'Ascend 910B', archived: true,
        model: runModel('completed', {
          compilation: { verdict: 'pass', summary: '41 Pass · 38 Kernel' }, correctness: { verdict: 'pass', summary: '16 / 16 Golden Compare match' },
          execution: { verdict: 'pass', summary: '390 Task · Dependency Graph 完整' }, performance: { verdict: 'pass', summary: '关键链预算内' }, resources: { verdict: 'pass', summary: 'UB peak 61%' }
        }, { golden_compare: { status: 'available' }, ir_validation: { status: 'available' }, tensor_dump: { status: 'available' }, dependency_graph: { status: 'available' }, runtime_timeline: { status: 'available' }, pmu: { status: 'partial' }, scope_stats: { status: 'available' } }, [], { baseline: true }),
        artifacts: [
          Object.assign({}, ART().overview, { meta: 'L14 · RMSNorm + RoPE', tone: 'ok' }),
          Object.assign({}, ART().source, { meta: 'rmsnorm_rope.py · 在资源管理器中打开', tone: 'ok' }),
          Object.assign({}, ART().compile, { meta: '编译通过 · 7 约束', tone: 'ok', primary: false }),
          Object.assign({}, ART().correct, { meta: '3 / 3 oracle · 16 / 16 match', tone: 'ok' }),
          Object.assign({}, ART().baseline, { meta: '已签发可信基线', tone: 'ok' })
        ],
        signals: [['ok', '端到端 +8% vs 基线'], ['ok', 'UB 61% · 预算内'], ['ok', '最大绝对误差 0.0004883']]
      }]
    });

    tasks.push({
      id: 'task_dsv4', kind: '模型', model: 'deepseek-v4-flash', op: 'decode_fwd',
      branch: 'model/v4-flash-npu', title: 'DeepSeek V4 Flash · 整网融合替换',
      runs: [{
        id: '20260611_163004', verdict: 'warn', time: '2026-06-11 16:30:04',
        duration: '12m04s', target: 'Ascend 910B', archived: true,
        model: runModel('completed', {
          compilation: { verdict: 'pass', summary: '61 Kernel · IR Validation pass' }, correctness: { verdict: 'pass', summary: '逐层 checkpoint match' },
          execution: { verdict: 'pass', summary: '整网替换完成' }, performance: { verdict: 'warning', summary: 'MoE 路由分支未覆盖' }, resources: { verdict: 'unknown', summary: '未采集 Scope Stats' }
        }, { ir_validation: { status: 'available' }, tensor_dump: { status: 'partial' }, dependency_graph: { status: 'available' }, runtime_timeline: { status: 'partial' } },
        [{ id: 'F004', severity: 'warning', domain: 'performance', title: 'MoE 路由分支未覆盖', summary: '当前性能结论不覆盖所有路由组合。', affectedObjects: [], evidence: ['覆盖集不完整'], action: { label: '在「执行」中查看性能证据', route: 'performance' } }]),
        artifacts: [
          Object.assign({}, ART().overview, { meta: 'MoE · CSA / HCA', tone: 'ok' }),
          Object.assign({}, ART().source, { label: '模型源码', meta: 'torch_npu 融合替换后', tone: 'ok' }),
          Object.assign({}, ART().compile, { meta: '整网 61 kernel', tone: 'warn', primary: false }),
          Object.assign({}, ART().correct, { meta: '逐层 checkpoint 通过', tone: 'ok' }),
          Object.assign({}, ART().baseline, { meta: '待复核', tone: 'warn' })
        ],
        signals: [['warn', 'MoE 路由分支未覆盖'], ['ok', '融合替换命中 18 处']]
      }]
    });

    tasks.push({
      id: 'task_paged', kind: '算子', model: 'qwen3-14b', op: 'fa_fused',
      branch: 'exp/paged-block', title: 'Paged Attention · block 调度实验',
      runs: [{
        id: '20260529_091210', verdict: 'purged', time: '2026-05-29 09:12:10',
        duration: '3m41s', target: 'Ascend 910B', purged: true,
        model: runModel('cancelled', {
          compilation: { verdict: 'pass', summary: '实验编译完成' }, correctness: { verdict: 'unknown', summary: '未保留比对结论' }, execution: { verdict: 'unknown', summary: '未保留运行证据' }, performance: { verdict: 'not_evaluated', summary: '方案已替换' }, resources: { verdict: 'unknown', summary: '无可用结论' }
        }, {}, []),
        note: '实验分支，产物已清理。结论：已被 affine fallback 方案取代。'
      }]
    });

    return tasks;
  }

  const TASKS = buildTasks();
  const latest = (t) => t.runs[0];
  const FILTERS = [['all', '全部'], ['op', '算子'], ['model', '模型'], ['live', '产物在库']];

  const st = {
    task: TASKS[0].id,
    run: TASKS[0].runs[0].id,
    artifact: 'compile',
    artifactGroup: 'input',
    compareRuns: [],
    filter: 'all',
    tab: 'overview',
    selection: null,
    runtimeContext: null,
    compilationEntry: null,
    executionEntry: null,
    runSplitSession: null,
    // Run hides the inspector; this is the explorer width Run is pinned to,
    // in pixels, carried across Activity switches.
    runSplitExplorerPx: null
  };
  let els = null;
  let objectTooltip = null, objectTooltipTarget = null, objectTooltipTimer = null, objectTooltipShowTimer = null;
  let fixPreview = null;
  /* 悬浮不立刻出卡，留一点反应时间；已打开时切换目标直接跟随 */
  const TOOLTIP_SHOW_DELAY = 260;
  const TOOLTIP_GAP = 12;

  function matches(t) {
    if (st.filter === 'all') return true;
    if (st.filter === 'op') return t.kind === '算子';
    if (st.filter === 'model') return t.kind === '模型';
    return t.runs.some(r => r.live);
  }

  /* ---------- mount ---------- */
  function mount() {
    const host = $('[data-side-view="workflow"]');
    if (!host) return false;
    const root = document.createElement('div');
    root.className = 'kf-th';
    root.id = 'taskHistory';
    root.innerHTML =
      '<div class="kf-th-filters" id="thFilters">' +
        FILTERS.map(f => '<button type="button" data-th-filter="' + f[0] + '"' +
          ' aria-pressed="' + (f[0] === st.filter) + '">' + f[1] + '</button>').join('') +
      '</div>' +
      '<div class="kf-th-compare-box" id="thCompareBox" hidden></div>' +
      '<div class="kf-th-list" id="thList"></div>';
    host.insertBefore(root, host.firstChild);
    // Run detail goes at the top of stage 0; the recipe / contract form below
    // it stays as the "start a new task" affordance.
    let detail = null;
    const stage0 = $('.kf-stage[data-stage="0"]');
    if (stage0) {
      detail = $('#runDetail', stage0);
      if (!detail) {
        detail = document.createElement('section');
        detail.className = 'kf-rd';
        detail.id = 'runDetail';
        stage0.insertBefore(detail, stage0.firstChild);
      }
    }

    objectTooltip = document.createElement('div');
    objectTooltip.className = 'kf-object-tooltip';
    objectTooltip.id = 'runObjectTooltip';
    objectTooltip.setAttribute('role', 'tooltip');
    objectTooltip.setAttribute('aria-hidden', 'true');
    document.body.appendChild(objectTooltip);

    /* 修复预览（轻量 dialog）：挂在 body 上，不进 run detail 的 DOM —— 页签
       切换重写 #runTabPanel 时不会被冲掉，也不用参与借用 / 归还流程。 */
    fixPreview = document.createElement('div');
    fixPreview.className = 'kf-rw-fix-modal';
    fixPreview.id = 'runFixPreview';
    fixPreview.hidden = true;
    document.body.appendChild(fixPreview);

    els = { root, detail, list: $('#thList', root), filters: $('#thFilters', root), compareBox: $('#thCompareBox', root) };
    return true;
  }

  /* ---------- content tabs -------------------------------------------------
     What the run concluded, what it is made of, and how the compiler got
     there. They sit inside the run detail, under the KPI strip, so the run's
     identity, verdict and headline numbers stay on screen no matter which tab
     is open — and so the tabs never appear outside 任务与运行.

     The IR tab shows stage 2, which cannot move: demo-v2's renderStage() maps
     state.step onto .kf-stage by DOM index. So the section stays put and this
     borrows its CHILDREN into the panel, returning them the moment stage 0
     stops being the active stage. Nothing is duplicated, and every listener
     inside that stage survives the move. */
  const PANELS = [
    { k: 'overview', label: '概览' },
    // compilation 的 from 只服务「编译失败」Run：那条路径仍借用 stage 2 的
    // Kernel Guard + 失败故事。pass 的 Run 走 renderCompilationTab()，不再搬 DOM。
    { k: 'compilation', label: '编译', from: '.kf-stage[data-stage="2"]' },
    { k: 'correctness', label: '正确性', from: '.kf-stage[data-stage="3"]' },
    { k: 'execution', label: '执行' },
    { k: 'resources', label: '资源' }
  ];
  /* Performance 不再是页签。延迟结论（端到端 / 关键链 / 等待 / 核占用）和它的
     证据（时间轴 / 依赖 / 泳道）来自同一份 dfx_outputs，拆成两个页签的结果是
     结论与证据隔着一次点击，而且同一条关键链在两边各算了一遍。旧的
     performance 路由 —— Overview 的 domain 卡、Finding 的 route、执行地图的
     行动作 —— 统一改道到 execution。 */
  const TAB_ALIAS = { performance: 'execution' };
  const toTabKey = (k) => TAB_ALIAS[k] || k;
  let borrowed = null;

  function releaseBorrowed() {
    if (!borrowed) return;
    const home = $(borrowed.from);
    if (home) borrowed.nodes.forEach(n => home.appendChild(n));
    borrowed = null;
  }

  function borrowInto(from, panel) {
    const src = $(from);
    if (!src) return;
    const nodes = $$(':scope > *', src);
    nodes.forEach(n => panel.appendChild(n));
    borrowed = { from, nodes };
  }

  /* Called whenever the active stage changes: stage 2 must have its content
     back before demo-v2 shows it for any reason other than the IR tab. */
  function syncPanel() {
    const panel = $('#runTabPanel');
    /* Compilation 页签交给 PTO_COMPILATION 之后，面板里是自绘的 .kc 视图，
       而不是从 stage 2 搬来的 DOM。watchRunInspectorLayout 的 MutationObserver
       会绕过 renderDetailBody 直接再跑一次 syncPanel，不在这里挡住就会把新视图
       冲掉换成 Kernel Guard，并把借给它当页签的 #kgTrace 甩成游离节点。 */
    if (window.PTO_COMPILATION?.owns?.(panel)) {
      window.PTO_GUARD?.activate?.();
      return;
    }
    /* Correctness 页签自绘诊断视图时，面板里是我们的 .kf-dg-* 结构，
       同样不能被借用 stage 3 的 DOM 冲掉。 */
    if (panel && panel.querySelector('[data-dg-canvas]')) { releaseBorrowed(); return; }
    const stage0 = $('.kf-stage[data-stage="0"]');
    const p = PANELS.find(x => x.k === st.tab);
    if (!panel || !p || !p.from || !stage0 || !stage0.classList.contains('is-active')) {
      releaseBorrowed();
      return;
    }
    window.PTO_GUARD?.activate?.();
    if (borrowed && borrowed.from === p.from && panel.contains(borrowed.nodes[0])) return;
    releaseBorrowed();
    window.PTO_COMPILATION?.release?.();   // 归还 #kgTrace，否则它会跟着面板一起被冲掉
    panel.innerHTML = '';        // after the release, only our own leftovers remain
    borrowInto(p.from, panel);
  }

  /* Compilation 页签：只有当这次 Run 就是 IR 数据产出的那一次时，才交给新的
     PTO_COMPILATION 渲染；其余情况（历史 Run、编译失败 Run）保持原有行为。
     这样不会把 20260625_184941 的编译数据贴到别的 Run 上。 */
  function compilationDataMatches(r) {
    const K = window.PTO_IR_KERNELS;
    if (!K || !K.source || !r || !r.id) return false;
    return String(K.source).indexOf(String(r.id)) >= 0;
  }

  function renderCompilationTab(panel, r) {
    if (r.id === 'run_106' && st.compilationEntry?.from === 'investigation') {
      const compiler = window.PTO_CORRECTNESS_DIAGNOSTICS.profiles[r.id].compiler || {};
      const nv = compiler.numericalValidation || {};
      panel.innerHTML = '<section class="kf-rd-sec" aria-label="当前 Run 编译校验">' +
        '<div class="kf-rd-h">编译校验证据<small>当前 Run · 已检查输出</small></div>' +
        dl([['结构校验', compiler.structuralVerification?.status === 'pass' ? '通过' : '未确认'],
          ['数值校验', nv.status === 'pass' ? '已检查输出匹配' : '未确认'],
          ['通过 / 已检查 Pass', (nv.passed ?? '—') + ' / ' + (nv.total ?? '—')]]) +
        '<p class="kf-rd-note">' + esc(nv.summary || '未采集数值校验摘要') + '</p>' +
        '<p class="kf-rd-note">当前仅保留校验摘要，未关联逐 Pass 比对明细；不能据此排除所有输入与设备执行路径。</p></section>';
      return;
    }
    const view = window.PTO_COMPILATION;
    if (view && view.ready) {
      // #109 reuses the loaded workspace; only its numerical evidence differs.
      const compilerFixture = r.id === 'run_109';
      const profile = compilerFixture ? window.PTO_CORRECTNESS_DIAGNOSTICS?.get?.(r.id) : null;
      const entry = compilerFixture ? Object.assign({}, st.compilationEntry || { from: 'compilation' }, {
        runId: r.id,
        finding: '数值精度 · 输出不一致',
        failureMode: 'Semantic',
        intent: '正在定位编译语义分歧',
        numericalValidation: profile?.compiler?.numericalValidation || view.numericalFixtures.compiler_semantic_error
      }) : st.compilationEntry;
      window.PTO_RUN_CONTEXT = {
        runId: r.id,
        findings: getFindings(r),
        compilationEntry: entry,
        numericalValidation: entry?.numericalValidation || null
      };
      if (entry?.numericalValidation) {
        view.setNumericalContext?.(entry.numericalValidation, entry);
      } else view.clearNumericalContext?.();
      if ((compilerFixture || compilationDataMatches(r) || entry?.numericalValidation) && view.render(panel)) return;
    }
    syncPanel();
  }

  function isSemanticCompilationHandoff(r) {
    return r?.id === 'run_109' && st.tab === 'compilation';
  }

  function semanticCompilationHandoff(t, r) {
    const first = window.PTO_COMPILATION?.numericalFixtures?.compiler_semantic_error?.firstDivergentPass || 'ExpandMixedKernel';
    return '<section class="kf-compile-handoff-wrap" aria-label="编译调查交接">' +
      '<header class="kf-compile-identity"><b>' + esc(t.title) + '</b><span>Run ' + esc(runDisplayId(r)) + '</span></header>' +
      '<section class="kf-compile-handoff">' +
        '<p class="kf-compile-handoff__from">来自总览 · Compiler / ' + esc(first) + '</p>' +
        '<div class="kf-compile-handoff__grid">' +
          '<div><span class="kf-compile-handoff__label">当前发现</span><b>' + esc(first) + '</b><p>首个数值分歧 Pass。Host IR 在此处首次偏离 Golden；设备尚未执行。</p></div>' +
          '<i class="kf-compile-handoff__arrow" aria-hidden="true">→</i>' +
          '<div><span class="kf-compile-handoff__label">下一步调查</span><b>' + esc(first) + ' 前后 IR</b><p>确认该 Pass 引入了哪项语义变化。</p><button type="button" class="btn btn-solid btn-compact" data-th-compilation-diff="' + esc(first) + '">查看 IR Diff →</button></div>' +
        '</div>' +
      '</section>' +
    '</section>';
  }

  /* Execution keeps the existing composition fan and trace timeline together.
     Their selection is handed to the shared object Inspector instead of a
     second, embedded Inspector. */
  function renderExecution(panel) {
    if (!window.PTO_FAN) {
      panel.innerHTML = '<p class="kf-rd-note is-dim">运行切片需要 passes_dump 与 dfx_outputs。</p>';
      return;
    }
    const fan = document.createElement('section');
    fan.className = 'kf-rd-sec kf-fan-host';
    fan.id = 'runFan';
    panel.appendChild(fan);
    window.PTO_FAN.mount(fan, {
      onSelect(obj) {
        selectObject(Object.assign({ sourceTab: 'execution' }, obj));
      }
    });
    const timeline = document.createElement('div');
    timeline.id = 'runTimeline';
    panel.appendChild(timeline);
    window.PTO_TIMELINE?.mount?.(timeline, {
      inlineInspector: false,
      selectedTaskId: st.selection && st.selection.kind === 'task' ? Number(st.selection.id) : null,
      onSelect(obj) { selectObject(Object.assign({ sourceTab: 'execution' }, obj)); },
      onClear() { clearSelection(); }
    });
  }

  /* 当前真实 Run 的 Execution 页签。Performance 在此 Run 中是 Execution 的
     一个证据维度，而非独立页签，因此 Compilation 的上下文直接落到这里。 */
  function renderExecutionTab(panel, run) {
    /* This exact on-disk run has a dedicated L2 scheduling view. Keep it in the
       task history's Execution tab instead of duplicating the console's data,
       renderers, and scheduling interaction here. */
    if (String(run && run.id) === '20260625_184941') {
      panel.innerHTML = '<section class="kf-tuning-embed" aria-label="L2 调度视图">' +
        '<header><div><span>Execution · L2 调度</span><small>任务放置、依赖与关键路径</small></div></header>' +
        '<iframe class="kf-tuning-embed__frame" title="L2 调度视图" loading="eager" ' +
          'src="../operator-tuning-console/index.html?embed=l2&amp;case=decode_fwd_layers"></iframe>' +
      '</section>';
      return;
    }
    const D = window.PTO_RUN_TRACE, ctx = st.runtimeContext;
    const indices = D && ctx ? runtimeIndices(D, ctx.kernelName) : [];
    const focus = indices.length ? runtimeFocus(D, ctx, indices) : null;
    if (ctx && focus != null) {
      const task = D.tasks[focus];
      const backend = (D.kernels[task.k] || {}).be || '—';
      const chain = (((D.perf || {}).chain || {}).steps || []).find(s => s.i === focus);
      const chainText = chain ? '等待 ' + rtUs(chain.wait) + ' · 执行 ' + rtUs(chain.run) : '不在当前关键链记录中';
      const summary = indices.length === 1
        ? '1 个逻辑运行时任务 · ' + rtUs(task.s) + ' → ' + rtUs(task.e) + ' · ' + task.c + ' 个 ' + backend + ' 核'
        : indices.length + ' 个逻辑运行时任务 · 当前选择 ' + task.id + ' · ' + rtUs(task.e - task.s);
      panel.insertAdjacentHTML('beforeend', contextBanner('编译', runtimeFindingLabel(ctx.findingId), '定位运行时证据', 'Task #' + task.id + ' · ' + ctx.kernelName) +
        '<section class="kf-cv-summary"><div><span>运行时证据</span><b>' + esc(ctx.kernelName) + '</b><small>' +
        esc(summary) +
        '</small></div><div><span>关键链</span><b>' + esc(chainText) +
        '</b><small>这是运行时证据，不单独判定瓶颈。</small></div></section>' + localDependencyHTML(D, focus));
    }
    const fan = document.createElement('section');
    fan.className = 'kf-rd-sec kf-fan-host';
    fan.id = 'runFan';
    panel.appendChild(fan);
    window.PTO_FAN?.mount?.(fan, {
      onSelect(obj) { selectObject(Object.assign({ sourceTab: 'execution' }, obj)); }
    });
    const timeline = document.createElement('div');
    timeline.id = 'runTimeline';
    panel.appendChild(timeline);
    window.PTO_TIMELINE?.mount?.(timeline, {
      inlineInspector: false,
      selectedTaskId: focus != null ? focus : (st.selection && st.selection.kind === 'task' ? Number(st.selection.id) : null),
      onSelect(obj) {
        if (ctx) st.runtimeContext = Object.assign({}, ctx, { taskIndex: Number(obj.id) });
        selectObject(Object.assign({ sourceTab: 'execution' }, obj));
      },
      onClear() { clearSelection(); }
    });
    if (focus != null) {
      const task = D.tasks[focus], pad = Math.max(16, (task.e - task.s) * 1.5);
      window.PTO_TIMELINE?.focusRange?.({ from: task.s - pad, to: task.e + pad });
      window.PTO_TIMELINE?.setHighlight?.({ tasks: [focus] }, { track: 'both' });
    } else {
      window.PTO_TIMELINE?.focusRange?.(null);
      window.PTO_TIMELINE?.clearHighlight?.();
    }
  }

  function tabStrip() {
    return '<nav class="kf-rt" id="runTabs" role="tablist" aria-label="运行内容">' +
      PANELS.map(p =>
        '<button type="button" role="tab" data-th-tab="' + p.k + '"' +
          ' aria-selected="' + (p.k === st.tab) + '">' +
          '<b>' + p.label + '</b></button>').join('') +
    '</nav>';
  }

  /* ---------- render ---------- */
  function runRow(t, r) {
    const v = getRunDisplayStatus(r);
    const on = r.id === st.run && t.id === st.task;
    return '<button type="button" class="kf-th-run' + (on ? ' is-sel' : '') +
      (r.purged ? ' is-purged' : '') + '" data-th-run="' + r.id + '" data-th-of="' + t.id + '">' +
      '<code>' + esc(runDisplayId(r)) + '</code>' +
      '<span class="kf-th-rv is-' + v[1] + '">' + v[0] + '</span>' +
      '<small><i aria-hidden="true">◷</i>' + esc(r.time) + '</small>' +
    '</button>';
  }

  /* ---------- run detail (main area, top of stage 0) ----------
     A run detail page has to answer, without another click: did it pass, where
     is it tight, what did the compiler complain about, and what came out.
     Artifacts are the last section, not the whole page. */
  const GROUPS = [
    ['input',   '输入', '这次运行消费了什么'],
    ['compile', '编译产物', '编译器走过的 IR 与它自己的结论'],
    ['codegen', '生成代码', '最终落盘的可执行物'],
    ['runtime', '运行时证据', '上设备之后采集到的'],
    ['repro',   '复现', '把这次运行原样跑回来']
  ];
  // artifacts that have a viewer wired up — both are tabs of this page now, so
  // opening one stays inside the run detail instead of navigating away
  const OPEN = {
    step2: { tab: 'compilation', k: 'compile' },
    explorer: { explorer: true, k: 'source' }     // source has no tab; the workspace owns it
  };
  // dfx_outputs/ is what the execution timeline on this page is built from, so
  // those three artifacts scroll to it rather than claiming to have no viewer.
  const TL_ARTS = { swimlane: 1, deps: 1, namemap: 1 };

  const kb = (b) => b >= 1048576 ? (b / 1048576).toFixed(1) + ' MB'
                  : b >= 1024 ? Math.round(b / 1024) + ' KB' : b + ' B';
  const tone = (p) => p >= 100 ? 'bad' : p >= 80 ? 'warn' : p >= 50 ? 'mid' : 'ok';

  /* ---------- header ------------------------------------------------------
     Keep the run identity together: the title is followed immediately by the
     immutable run id and timestamp. The live overview puts the two actionable
     metrics in a separate right-hand column. */
  /* Run lifecycle 是最弱的信息，不抢注意力：
     运行中 = 标题旁一个轻量 spinner；已取消 = 弱化文字「— 已取消」；
     完成 / 失败 / 不完整 一律不显示任何 lifecycle 文案 —— 结论交给各 Domain。 */
  function runLifecycle(m) {
    if (m.state === 'running') return '<span class="kf-rd-spinner" role="status" aria-label="运行中"></span>';
    if (m.state === 'cancelled') return '<span class="kf-rd-cancelled">— 已取消</span>';
    return '';
  }
  function headline(t, r, L) {
    const m = getRunModel(r);
    const title = t.title;
    return '<section class="kf-rd-summary">' +
      '<div class="kf-rd-identity">' +
        /* 运行快照图标 · 翻转摄像头。Pixso 导出的 28 网格几何原样保留，
           去掉掩膜 / 滤镜与 opacity=0 的图层，描边改为跟随容器的 currentColor。 */
        '<span class="kf-rd-run-icon" aria-hidden="true">' +
          '<svg viewBox="0 0 28 28" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' +
            '<path d="M7.68301 4.60802C7.97 3.60702 8.88501 2.91602 9.927 2.91602L17.712 2.91602C18.645 2.91602 19.489 3.47202 19.856 4.33002L20.849 6.64602C20.941 6.86002 21.152 6.99902 21.385 6.99902L23.917 6.99902C25.528 6.99902 26.834 8.30502 26.834 9.91602L26.834 21.583C26.834 23.516 25.267 25.083 23.334 25.083L4.66699 25.083C2.73401 25.083 1.16699 23.516 1.16699 21.583L1.16699 9.91602C1.16699 8.30502 2.47299 6.99902 4.08401 6.99902L6.56 6.99902C6.82101 6.99902 7.04999 6.82702 7.121 6.57601L7.68301 4.60802Z"/>' +
            '<circle cx="11" cy="6" r="1" fill="currentColor" stroke="none"/>' +
            '<path d="M1.83291 0.0484024C0.711538 0.984976 0 2.38162 0 3.94168C0 6.76162 2.32482 9.04762 5.19262 9.04762C5.61257 9.04762 6.02081 8.99862 6.41186 8.9061M1.89143 1.40364L1.89143 0" transform="matrix(-1,0,0,-1,19,19.051)"/>' +
            '<path d="M10.833 11.002C9.71201 11.938 9 13.335 9 14.895C9 17.715 11.325 20.001 14.193 20.001C14.617 20.001 15.03 19.951 15.425 19.856M10.891 12.357L10.891 10.953"/>' +
          '</svg>' +
        '</span>' +
        '<div class="kf-rd-id">' +
          '<div class="kf-rd-titleline"><h2>' + esc(title) + '</h2>' + runLifecycle(m) + '</div>' +
          '<div class="kf-rd-run"><span>Run · </span><code>' + esc(runDisplayId(r)) + '</code><i class="kf-rd-meta-sep" aria-hidden="true"></i>' +
            '<small><i aria-hidden="true">◷</i>' + esc(r.time) + '</small></div>' +
        '</div>' +
      '</div>' +
    '</section>';
  }

  /* one line saying what, if anything, needs attention */
  function verdictLine(r, L) {
    const v = getRunDisplayStatus(r);
    const flags = [];
    if (L) {
      if (L.over) flags.push(L.over + " 个 kernel 内存溢出");
      if (L.atLimit) flags.push(L.atLimit + " 个 kernel 填满 " + spLabel(L.kmem[0].sp) + "（零余量）");
      if (L.actionable) flags.push(L.actionable + ' 条搬运提示可优化（同一个根因）');
      if (L.demotedK.length) flags.push(L.demotedK.length + ' 处流水被降级');
    }
    return '<p class="kf-rd-verdictline is-' + v[1] + '">' +
      '<b>' + v[0] + '</b>' +
      (!L ? '<span>归档记录，无逐项数据。</span>'
        : flags.length
          ? '<span>无 Error · 关注 ' + flags.length + ' 项：' + flags.map(esc).join(' · ') + '</span>'
          : '<span>无 Error · 无待处理项</span>') +
    '</p>';
  }

  const usFmt = (v) => v >= 1000 ? (v / 1000).toFixed(2) + ' ms' : Math.round(v) + ' µs';

  const band = (t, s) => '<div class="kf-rd-band"><b>' + t + '</b><small>' + s + '</small></div>';

  function kpis(r, L) {
    const P = (window.PTO_RUN_TRACE || {}).perf;
    const tiles = [];

    if (P) {
      const aic = P.occ.AIC || { pct: 0, n: 0 }, aiv = P.occ.AIV || { pct: 0, n: 0 };
      const gap = Math.round(aic.pct - aiv.pct);
      const work = Math.max(0, Math.min(100, Math.round(P.chain.workPct)));
      const wait = 100 - work;
      tiles.push(
        { k: 'efficiency', v: Math.round(P.span), u: 'µs', l: '运行效率', t: work < 50 ? 'warn' : 'ok', tag: 'TIME + CHAIN',
          s: P.chain.n + ' 步关键链 · 实测',
          viz: '<div class="kf-rd-kpi-eff" role="img" aria-label="计算 ' + work + '%，等待 ' + wait + '%">' +
            '<div class="kf-rd-kpi-split" style="--p:' + work + '"><i></i><b></b></div>' +
            '<div class="kf-rd-kpi-eff-legend">' +
              '<span class="is-work"><i></i><b>计算 ' + work + '%</b></span>' +
              '<span class="is-wait"><i></i><b>等待 ' + wait + '%</b></span>' +
            '</div></div>' },
        { k: 'occupancy', v: Math.round(aic.pct), u: '%', l: '核占用',
          t: aic.pct >= 60 ? 'warn' : 'ok', tag: 'AIC / AIV',
          s: 'AIV ' + Math.round(aiv.pct) + '% · 差 ' + gap + ' 个百分点',
          viz: '<div class="kf-rd-kpi-ring" style="--p:' + Math.round(aic.pct) + '" aria-hidden="true"><i>' + Math.round(aiv.pct) + '</i></div>' },
      );
    }
    if (!tiles.length) return '';
    return tiles.map(t =>
      '<article class="kf-rd-kpi is-' + t.t + ' is-' + t.k + '" style="grid-column:auto;min-width:0;">' +
        '<header><span>' + esc(t.l) + '</span><em>' + esc(t.tag || '') + '</em></header>' +
        '<div class="kf-rd-kpi-main"><b>' + t.v + '<i>' + t.u + '</i></b>' + t.viz + '</div>' +
        '<small>' + esc(t.s) + '</small>' +
      '</article>').join('');
  }

  /* Historical story Runs keep the same composed overview as a measured Run.
     Their headline tiles come from the immutable Run record rather than the
     currently mounted trace. */
  function historicalKpis(r) {
    const m = getRunModel(r);
    let tiles = [];
    if (r.id === 'run_109') {
      const f = window.PTO_COMPILATION?.numericalFixtures?.compiler_semantic_error;
      tiles = [
        { l: '首个分歧 Pass', v: esc(f?.firstDivergentPass || '未采集'), u: '', t: 'bad', tag: 'COMPILATION', s: 'Host IR 数值校验 · ' + (f?.status || '未采集') },
        { l: '设备执行', v: '未执行', u: '', t: 'dim', tag: 'DEVICE', s: '没有 Runtime / Tensor / Task 执行证据' }
      ];
    } else if (isCompileFailureStory(r)) {
      tiles = [
        { l: '首个失败 Pass', v: 'FAIL', u: '', t: 'bad', tag: 'COMPILATION', s: 'LegalizeIndexing · IR verification' },
        { l: '证据', v: '3', u: '项', t: 'ok', tag: 'AVAILABLE', s: 'IR Validation · Pass Dump · Source Location' }
      ];
    } else if (isCorrectnessFailureStory(r)) {
      tiles = [
        { l: 'First divergence', v: 'T37', u: '', t: 'bad', tag: 'CORRECTNESS', s: 'attention_out max_abs_diff ' + (window.PTO_CORRECTNESS_DIAGNOSTICS?.profiles?.run_106?.tensors?.find(t => t.id === 'attention_out')?.maxAbs ?? '未采集') + ' · 重复运行不稳定' },
        { l: 'Ordering evidence', v: '1', u: '', t: 'warn', tag: 'EXECUTION', s: 'Task #182 → #197 · missing edge' }
      ];
    } else if (isPerformanceWarningStory(r)) {
      tiles = [
        { l: 'Latency', v: '1.82', u: ' ms', t: 'warn', tag: '目标 < 1.50', s: 'Critical Path 1.41 ms · 等待 / 停顿 37%' },
        { l: '最长任务', v: '214', u: ' µs', t: 'warn', tag: 'TASK #182', s: 'reference 147 µs · MTE stall 31%' }
      ];
    } else if (isValidatedOptimizationStory(r)) {
      tiles = [
        { l: 'Latency', v: '1.36', u: ' ms', t: 'ok', tag: '目标已达成', s: 'Critical Path 0.98 ms · 目标 < 1.50 ms' },
        { l: '资源守护', v: '83', u: '%', t: 'ok', tag: 'L0B PEAK', s: '预算内 · 12 / 12 checkpoint 匹配' }
      ];
    }
    return tiles.map(t =>
      '<article class="kf-rd-kpi is-' + t.t + '" style="grid-column:auto;min-width:0;">' +
        '<header><span>' + esc(t.l) + '</span><em>' + esc(t.tag) + '</em></header>' +
        '<div class="kf-rd-kpi-main"><b>' + esc(t.v) + '<i>' + esc(t.u) + '</i></b></div>' +
        '<small>' + esc(t.s) + '</small>' +
      '</article>').join('');
  }

  /* memory water level — one square per kernel, filled by utilisation.
     Squares are grouped by the space that is TIGHTEST for that kernel, because
     only two spaces ever become the bottleneck here (Vec and L0B/Right) and a
     percentage is only comparable against its own limit. Utilisation is binned
     rather than continuous: five bands read faster than a gradient, and the
     legend can then state exactly what each shade means. */
  // IR-level space name first, hardware buffer in parens. Note ir-compile-guard.js
  // uses the reverse order ('UB (Vec)'); keep the two in sync if either changes.
  const SPACE_LABEL = { Vec: 'Vec (UB)', Mat: 'Mat (L1)', Acc: 'Acc (L0C)',
                       Left: 'Left (L0A)', Right: 'Right (L0B)' };
  const spLabel = (s) => SPACE_LABEL[s] || s;

  const BINS = [
    { k: 'b1', lo: 0,  hi: 25,  l: '< 25%' },
    { k: 'b2', lo: 25, hi: 50,  l: '25 – 50%' },
    { k: 'b3', lo: 50, hi: 75,  l: '50 – 75%' },
    { k: 'b4', lo: 75, hi: 100, l: '75 – 100%' },
    { k: 'b5', lo: 100, hi: 100, l: '100% · 零余量' },
    { k: 'b6', lo: 101, hi: Infinity, l: '> 100% · 溢出' }
  ];
  // Binned on the exact byte counts, not the rounded percent: a kernel at
  // 99.6% would round to 100 and be mislabelled as having zero headroom.
  const binOf = (k) => k.u > k.lim ? BINS[5]
                     : k.u === k.lim ? BINS[4]
                     : (BINS.find(b => k.p >= b.lo && k.p < b.hi) || BINS[0]);
  function memBlock(r) {
    if (!r.kmem.length) return '';

    // group by the kernel's own tightest space, hottest space first
    const byspace = {};
    r.kmem.forEach(k => { (byspace[k.sp] = byspace[k.sp] || []).push(k); });
    const groups = Object.keys(byspace)
      .map(sp => ({ sp, ks: byspace[sp].slice().sort((a, b) => b.p - a.p || b.u - a.u) }))
      .sort((a, b) => b.ks[0].p - a.ks[0].p);

    const counts = {};
    r.kmem.forEach(k => { const b = binOf(k).k; counts[b] = (counts[b] || 0) + 1; });
    const legend = '<div class="kf-rd-hleg">' +
      BINS.map(b => '<span><i class="is-' + b.k + '"></i>' + b.l +
        '<em>' + (counts[b.k] || 0) + '</em></span>').join('') +
      '<span class="kf-rd-hlegn">每格 = 1 个 kernel · 最紧空间占用</span></div>';

    const cell = (k) => {
      const b = binOf(k);
      const tip = k.n + ' · ' + k.t + '\n' + spLabel(k.sp) + ' ' + kb(k.u) + ' / ' + kb(k.lim) +
        '（' + k.p + '%）' +
        (k.u > k.lim ? '\n已溢出 ' + kb(k.u - k.lim) + '，必须缩小分块'
         : k.u === k.lim ? '\n分块已放到最大，正好填满；但零余量，改 shape 会溢出'
         : '\n余量 ' + kb(k.lim - k.u)) +
        (k.diag ? '\n' + k.diag + ' 条性能提示' : '') + '\n点击在编译卫士中打开';
      return '<button type="button" class="kf-rd-cell is-' + b.k + '"' +
        ' data-th-kernel="' + esc(k.n) + '" title="' + esc(tip) + '">' +
        '<b>' + esc(k.n) + '</b>' +
        '<i>' + k.p + '<u>%</u></i>' +
        (k.u === k.lim ? '<span class="kf-rd-full">满</span>' : '') +
        (k.u > k.lim ? '<span class="kf-rd-full is-over">溢出</span>' : '') +
        '<small>' + kb(k.u) + ' / ' + kb(k.lim) + '</small>' +
      '</button>';
    };

    const grid = groups.map(g => {
      const peak = g.ks[0].p, full = g.ks.filter(k => k.u === k.lim).length;
      return '<div class="kf-rd-hgrp">' +
        '<div class="kf-rd-hh"><b>' + esc(spLabel(g.sp)) + '</b>' +
          '<span>上限 ' + kb(g.ks[0].lim) + ' · ' + g.ks.length + ' 个 kernel · 峰值 ' + peak + '%' +
          (full ? ' · <em>' + full + ' 个正好填满（零余量）</em>' : '') + '</span></div>' +
        '<div class="kf-rd-heat">' + g.ks.map(cell).join('') + '</div>' +
      '</div>';
    }).join('');

    const none = (r.noMem && r.noMem.length)
      ? '<div class="kf-rd-hgrp is-none">' +
          '<div class="kf-rd-hh"><b>无片上分配</b><span>' + r.noMem.length +
            ' 个 · 编排 / SPMD 壳 / Group，不占片上内存</span></div>' +
          '<div class="kf-rd-heat">' + r.noMem.map(k =>
            '<div class="kf-rd-cell is-none" title="' + esc(k.n + ' · ' + k.t) + '">' +
              '<b>' + esc(k.n) + '</b><small>' + esc(k.t) + '</small></div>').join('') +
          '</div>' +
        '</div>'
      : '';

    return '<section class="kf-rd-sec">' +
      '<div class="kf-rd-h">内存水位<small>最紧空间占用 · report/memory_after_AllocateMemoryAddr.txt</small></div>' +
      legend + grid + none +
    '</section>';
  }

  /* perf hints — a diagnosis, not a log dump.
     report/perf_hints.log emits one row per (line, shape); 16 rows all carrying
     the same check, the same 512 B target and the same space is noise. Read the
     operator source at each line instead and the 16 collapse into 3 root causes
     with 3 different answers — one is a real local source change, one is blocked
     by the paged KV layout, one is not a defect at all. The generator derives
     `cause` from elems vs HEAD_DIM and `scopeName` by walking out to the
     enclosing `with pl.at/spmd`, so nothing here is hand-tagged per line. */
  const FINDINGS = [
    {
      k: 'half', tone: 'warn', verdict: '可改',
      title: 'RoPE 的 lo/hi 半维被拆成了两次搬运',
      what: '每次只搬 HEAD_DIM 的一半（64 个元素）。一行本来是连续的 128 个元素，' +
            '被拆成两次半行读写，每次都只用掉 L2 行宽的一小段。',
      why: 'rot_lo / rot_hi 是两个独立的 tile，分别 assemble 到 [row, 0] 和 [row, 64]；' +
           'cos / sin 也按 lo、hi 两个半维切片加载。RoPE 的旋转天然产生两半，' +
           '源码就顺着这个形状直接写了两次。',
      fix: '先把 lo / hi 拼成一个 128 元素的整行，再 assemble 一次；' +
           'cos / sin 直接切 [0:HEAD_DIM]，把 rotate-half 放进乘法里做。',
      gain: '搬运次数减半，单次宽度翻倍',
      risk: '多一个 128 元素的中间 tile 和一次 UB 内拷贝；宽度仍只到目标的一半，' +
            '净收益要实测'
    },
    {
      k: 'row', tone: 'dim', verdict: '布局锁死',
      title: '行宽已经等于 HEAD_DIM，分页 KV 布局下加不宽',
      what: '这两处一次写满一整行 128 个元素，已经是这个张量能给的最大宽度。',
      why: '张量的最内维就是 HEAD_DIM（128 个 BF16 = 256 B），由模型结构定死。' +
           '再宽只能把相邻的行合并成一次搬运 —— 写 KV cache 的那处不行，' +
           '行号是 (page × NUM_KV_HEADS + ki) × BLOCK_SIZE + offset，' +
           '同一个 batch 的相邻 KV head 物理上相隔 BLOCK_SIZE 行；' +
           '写 Q padding 的那处行本身连续，但合不合并由编译器决定，源码这边没有旋钮。',
      fix: '算子内基本无解。真要动，只能改分页方案本身（head-major → slot-major），' +
           '那是服务运行时的 page 布局约定，不是算子能单方面改的。',
      gain: '—', risk: '—'
    }
  ];

  function hintBlock(r) {
    if (!r.hints.length) return '';
    const by = {};
    r.hints.forEach(h => { (by[h.cause] = by[h.cause] || []).push(h); });
    const scalar = by.scalar || [];
    const want = r.hints[0].want, sp = r.hints[0].mem;
    const nfind = FINDINGS.filter(f => (by[f.k] || []).length).length;

    const cards = FINDINGS.map(f => {
      const g = (by[f.k] || []).slice().sort((a, b) => b.mult - a.mult || a.line - b.line);
      if (!g.length) return '';
      const hits = g.reduce((a, h) => a + (h.count || 1), 0);
      const wides = [...new Set(g.map(h => h.bytes))].sort((a, b) => a - b);
      const mults = [...new Set(g.map(h => h.mult))].sort((a, b) => a - b);
      const scopes = [...new Set(g.map(h => h.scopeName).filter(Boolean))];
      const facts = [
        ['实测宽度', wides.map(b => b + ' B').join(' / ') + '　（目标 ' + want + ' B）'],
        ['距目标', mults.map(m => m + ' ×').join(' / ')],
        ['所在作用域', scopes.join(' · ') || '—'],
        ['涉及源码', g.length + ' 处 · 编译期共触发 ' + hits + ' 次']
      ];
      /* Collapsed by default: the verdict and the four measured facts are the
         card's conclusion, and 现象/为什么/怎么改 is the reasoning behind it.
         Three stacked cards of full prose bury the ranking, so the reasoning
         opens on demand. The raw log rows stay one further click in. */
      const id = 'kfFind-' + f.k;
      return '<article class="kf-rd-find is-' + f.tone + '">' +
        '<button type="button" class="kf-rd-findhd" data-th-find="' + id + '"' +
          ' aria-expanded="false" aria-controls="' + id + '">' +
          '<span class="kf-rd-findtag">' + f.verdict + '</span>' +
          '<b>' + f.title + '</b><span class="kf-rd-findn">' + g.length + ' 条</span>' +
          '<i class="kf-rd-findchev" aria-hidden="true"></i></button>' +
        '<dl class="kf-rd-findfacts">' + facts.map(([k, v]) =>
          '<div><dt>' + k + '</dt><dd>' + esc(v) + '</dd></div>').join('') + '</dl>' +
        '<div class="kf-rd-findmore" id="' + id + '" hidden>' +
        '<div class="kf-rd-findbody">' +
          '<p><b>现象</b>' + f.what + '</p>' +
          '<p><b>为什么</b>' + f.why + '</p>' +
          '<p class="is-fix"><b>怎么改</b>' + f.fix + '</p>' +
          (f.gain !== '—'
            ? '<p class="is-meta"><b>预期</b>' + f.gain + '　<b>代价</b>' + f.risk + '</p>' : '') +
        '</div>' +
        '<details class="kf-rd-findsrc"><summary>展开这 ' + g.length + ' 处源码</summary>' +
          '<div class="kf-rd-tblwrap"><table class="kf-rd-tbl kf-rd-hint">' +
          '<tbody>' + g.map(h =>
            '<tr><td><code>:' + h.line + '</code></td>' +
              '<td>' + esc(h.op) + '</td>' +
              '<td class="is-bad">' + h.elems + ' 元素 · ' + h.bytes + ' B</td>' +
              '<td><code class="kf-rd-src">' + esc(h.src || '') + '</code>' +
                (h.scope ? '<i class="kf-rd-srcnote">编译器把提示挂在了这个作用域头上，' +
                  '实际是它内部的搬运</i>' : '') + '</td></tr>').join('') +
          '</tbody></table></div></details>' +
        '</div>' +
      '</article>';
    }).join('');

    /* The scalar class is dropped rather than shown: a per-row 1/rms or a
       pl.read of one value has an innermost dim of exactly one element by
       construction, so there is nothing to act on. The header counts only what
       is shown, and says how many were excluded so the total still reconciles
       with report/perf_hints.log. */
    return '<section class="kf-rd-sec">' +
      '<div class="kf-rd-h">性能提示 · ' + (r.hints.length - scalar.length) +
        ' 条 / ' + nfind + ' 类根因' +
        '<small>' + esc(spLabel(sp)) + ' · 目标最内维 ≥ ' + want + ' B · report ' +
        r.hints.length + ' 条 · 已排除 ' + scalar.length + ' 条单元素读写</small></div>' +
      cards +
    '</section>';
  }

  /* what the optimiser promised vs. what it delivered */
  function intentBlock(r) {
    const dem = r.demotedK.length
      ? r.demotedK.map(k => '<div class="kf-rd-line is-warn"><code>' + esc(k.n) + '</code>' +
          '<span>声明 ' + k.decl + ' 处流水 · 被 SkewCrossCorePipeline 降级 ' + k.d + ' 处为串行</span></div>').join('')
      : '<div class="kf-rd-line is-ok"><span>声明的流水全部保留</span></div>';
    const l0 = r.l0K.length
      ? '<div class="kf-rd-line is-ok"><span>AutoTileMatmulL0 为 ' + r.l0K.length +
        ' 个 kernel 生成了 L0 分块</span></div>' : '';
    const reuseW = r.reuse.bb ? Math.round(r.reuse.ba / r.reuse.bb * 100) : 0;
    return '<section class="kf-rd-sec">' +
      '<div class="kf-rd-h">调度与复用<small>Pipeline 保留 · MemoryReuse 前后</small></div>' +
      '<div class="kf-rd-two">' +
        '<div class="kf-rd-col">' + dem + l0 +
          '<div class="kf-rd-line is-dim"><span>全算子共声明 ' + r.declared + ' 处 pl.pipeline</span></div>' +
        '</div>' +
        '<div class="kf-rd-col">' +
          '<div class="kf-rd-reuse">' +
            '<div class="kf-rd-rbar"><span>复用前</span><i class="is-before" style="width:100%"></i>' +
              '<b>' + r.reuse.nb + ' 块 · ' + kb(r.reuse.bb) + '</b></div>' +
            '<div class="kf-rd-rbar"><span>复用后</span><i class="is-after" style="width:' + reuseW + '%"></i>' +
              '<b>' + r.reuse.na + ' 块 · ' + kb(r.reuse.ba) + '</b></div>' +
          '</div>' +
          '<div class="kf-rd-line is-dim"><span>' + r.reuse.noGain +
            ' 个 kernel 没有复用空间可省</span></div>' +
        '</div>' +
      '</div>' +
    '</section>';
  }

  /* ---------- right rail inspector ----------------------------------------
     demo-v2.js owns #inspector for the old linear route, and its stage-0 copy
     ("目标契约 / Toolkit 读取") was written when stage 0 was still the recipe
     wizard. Stage 0 is the run detail page now, so while it is showing the
     inspector belongs to the selected run.

     The main column keeps the measured argument — verdict, KPIs, timeline,
     memory, findings. The rail keeps everything that is reference material
     about the run: where it came from, what it produced, where the wall time
     went, how big it is, and what it leaves open. The artifact inventory in
     particular belongs here — it is a list you look things up in, not part of
     the argument, and as a 3-column card grid it was pushing the sections that
     are the argument off the bottom of the main column. */

  const num = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  const dl = (rows) => '<dl>' + rows.map(([k, v]) =>
    '<div><dt>' + esc(k) + '</dt><dd>' + esc(v) + '</dd></div>').join('') + '</dl>';
  const path = (k, v) => '<p class="kf-ri-path"><span>' + esc(k) + '</span>' +
    '<code>' + esc(v) + '</code></p>';

  function riSource(t, r, L) {
    return '<section class="kf-inspector-section">' +
      '<h2 class="kf-inspector-title">运行来源</h2>' +
      dl([['运行', r.id], ['时间', r.time], ['目标', r.target || '—'], ['分支', t.branch]]) +
      (r.dir ? path('产物目录', r.dir) : '') +
      (L ? path('数据来源', 'passes_dump/ · report/ · dfx_outputs/') : '') +
    '</section>';
  }

  /* the run's artifact inventory, restacked for a ~300px rail.
     Same rows, same open/scroll targets as the old main-column grid; a card
     grid needed three columns to breathe, a rail only needs the three lines
     each row already had — name + what you can do with it, what it is, and
     where it sits on disk. */
  function riArts(r) {
    const inv = r.inventory || [];
    if (!inv.length) return '';
    const groups = GROUPS.map(G => [G, inv.filter(i => i.g === G[0])]).filter(x => x[1].length);
    const activeGroup = groups.find(([G]) => G[0] === st.artifactGroup) || groups[0];
    const activeGroupIndex = groups.indexOf(activeGroup);
    const kindLabel = { input: 'IN', compile: 'IR', codegen: 'CG', runtime: 'RT', repro: 'RP' };
    return '<section class="kf-inspector-section">' +
      '<div class="kf-ri-invhead">' +
        '<div><h2 class="kf-inspector-title">产物清单</h2>' +
          '<small>一次运行的输入、编译、生成与运行时证据</small></div>' +
        '<b class="kf-ri-invcount"><strong>' + inv.length + '</strong><span>项产物</span></b>' +
      '</div>' +
      '<div class="kf-ri-flow" role="tablist" aria-label="产物阶段链路">' +
        groups.map(([G], gi) =>
          '<button type="button" role="tab" class="kf-ri-flow-step is-' + G[0] + (G[0] === activeGroup[0][0] ? ' is-active' : '') + '" data-th-art-group="' + G[0] + '" aria-selected="' + (G[0] === activeGroup[0][0]) + '">' +
            '<i>' + String(gi + 1).padStart(2, '0') + '</i><b>' + esc(G[1]) + '</b>' +
          '</button>').join('') +
      '</div>' +
      [activeGroup].map(([G, items]) =>
        '<div class="kf-ri-grp is-' + G[0] + '">' +
          '<div class="kf-ri-glabel"><i class="kf-ri-gstep">' + String(activeGroupIndex + 1).padStart(2, '0') + '</i>' +
            '<b>' + G[1] + '</b><em>' + items.length + '</em>' +
            '<small>' + G[2] + '</small></div>' +
          items.map(i => {
            const o = i.open ? OPEN[i.open] : null;
            const tl = !o && TL_ARTS[i.k] && window.PTO_RUN_TRACE;
            const attr = o
              ? (o.explorer ? ' data-th-view="explorer" data-th-art="source"'
                            : ' data-th-tab="' + o.tab + '" data-th-art="' + o.k + '"')
              : tl ? ' data-th-scroll="timeline"' : '';
            const action = o ? '<i class="kf-ri-art-action is-open">打开 <span>→</span></i>'
              : tl ? '<i class="kf-ri-art-action is-open">时间线 <span>↑</span></i>'
                   : '';
            const live = o || tl;
            return '<' + (live ? 'button type="button"' : 'div') +
              ' class="kf-ri-art kf-ri-art--visual is-' + G[0] + (live ? ' is-live' : ' is-static') +
              (o && o.k === st.artifact ? ' is-sel' : '') + '"' + attr + '>' +
              '<span class="kf-ri-art-mark" aria-hidden="true">' + (kindLabel[G[0]] || '•') + '</span>' +
              '<span class="kf-ri-art-copy"><b>' + esc(i.label) + '</b>' +
                '<small>' + esc(i.meta) + '</small><code>' + esc(i.where) + '</code></span>' + action +
            '</' + (live ? 'button' : 'div') + '>';
          }).join('') +
        '</div>').join('') +
    '</section>';
  }

  /* archived runs carry a recorded artifact list instead of a real inventory */
  function riArchivedArts(r) {
    if (!r.artifacts || !r.artifacts.length) return '';
    const items = r.artifacts;
    return '<section class="kf-inspector-section">' +
      '<div class="kf-ri-invhead">' +
        '<div><h2 class="kf-inspector-title">产物清单</h2>' +
          '<small>这次运行留下的归档证据</small></div>' +
        '<b class="kf-ri-invcount"><strong>' + items.length + '</strong><span>项产物</span></b>' +
      '</div>' +
      '<div class="kf-ri-invmeta"><span><i class="is-stored"></i>归档记录</span>' +
        '<span><i class="is-stored"></i>目录未挂载</span></div>' +
      '<div class="kf-ri-grp is-repro">' +
        '<div class="kf-ri-glabel"><i class="kf-ri-gstep">AR</i><b>归档记录</b><em>' + items.length + '</em>' +
          '<small>目录不在当前工作区</small></div>' +
        items.map(a =>
          '<button type="button" class="kf-ri-art kf-ri-art--visual is-repro' + (a.k === st.artifact ? ' is-sel' : '') + '"' +
            ' data-th-art="' + a.k + '"' +
            (a.explorer ? ' data-th-view="explorer"' : ' data-step="' + a.step + '"') + '>' +
            '<span class="kf-ri-art-mark" aria-hidden="true">AR</span>' +
            '<span class="kf-ri-art-copy"><b>' + esc(a.label) + '</b>' +
              '<small>' + esc(a.meta) + '</small><code>归档运行 · ' + esc(r.id) + '</code></span>' +
            '<i class="kf-ri-art-action is-open">打开 <span>→</span></i>' +
          '</button>').join('') +
      '</div>' +
    '</section>';
  }

  /* where the 993 µs went. The KPI strip in the main column states the three
     headline numbers; this breaks the biggest one down into the individual
     waits, which is the only form a developer can act on. */
  function riTime(P) {
    const c = P.chain, workW = Math.max(0, Math.min(100, c.workPct));
    const waits = c.steps.slice().sort((a, b) => b.wait - a.wait).filter(s => s.wait > 0).slice(0, 4);
    const top = waits.length ? waits[0].wait : 1;
    const share = c.wait ? Math.round(waits[0].wait / c.wait * 100) : 0;

    return '<section class="kf-inspector-section">' +
      '<h2 class="kf-inspector-title">时间去向</h2>' +
      '<div class="kf-ri-split">' +
        '<i class="is-work" style="width:' + workW.toFixed(1) + '%"></i>' +
        '<i class="is-wait" style="width:' + (100 - workW).toFixed(1) + '%"></i>' +
      '</div>' +
      '<div class="kf-ri-splitk">' +
        '<span><i class="is-work"></i>在算 ' + usFmt(c.work) + ' · ' + Math.round(c.workPct) + '%</span>' +
        '<span><i class="is-wait"></i>在等核 ' + usFmt(c.wait) + ' · ' + (100 - Math.round(c.workPct)) + '%</span>' +
      '</div>' +
      '<p class="kf-ri-note">关键链 ' + c.n + ' 步，端到端 ' + Math.round(P.span) +
        ' µs。等的时间比算的多，缩短等待比让单个 kernel 更快更划算。</p>' +
      '<div class="kf-ri-waits">' + waits.map(s =>
        '<div class="kf-ri-wait"><code>' + esc(s.kn) + '</code>' +
          '<i style="width:' + Math.round(s.wait / top * 100) + '%"></i>' +
          '<b>' + usFmt(s.wait) + '</b></div>').join('') + '</div>' +
      '<p class="kf-ri-note is-dim">最长的一次等待发生在 <code>' + esc(waits[0].kn) +
        '</code> 之前，占关键链全部等待时间的 ' + share + '%。</p>' +
    '</section>';
  }

  function riCores(D, P) {
    const kinds = ['AIC', 'AIV', 'AICPU'];
    return '<section class="kf-inspector-section">' +
      '<h2 class="kf-inspector-title">核占用</h2>' +
      '<div class="kf-ri-occ">' + kinds.map(k => {
        const o = P.occ[k]; if (!o) return '';
        return '<div><span>' + k + '<em>' + o.n + ' 核</em></span>' +
          '<i class="is-' + k.toLowerCase() + '" style="width:' + Math.round(o.pct) + '%"></i>' +
          '<b>' + o.pct.toFixed(1) + '%</b></div>';
      }).join('') + '</div>' +
      '<p class="kf-ri-note">全芯片平均忙碌 ' + P.busyPct.toFixed(1) +
        '%。Cube 侧接近 Vector 侧的三倍，负载并不均衡。</p>' +
      '<p class="kf-ri-note is-dim">最耗时的是 <code>' + esc(P.top.kn) + '</code>：累计忙碌 ' +
        usFmt(P.top.busy) + '，占全芯片忙碌时间的 ' + P.top.pct.toFixed(1) +
        '%，铺在 ' + P.top.cores + ' 个核上。</p>' +
    '</section>';
  }

  function riScale(L, D) {
    const rows = [['编译 pass', num(L.passes)], ['生成 Kernel', num(L.kernels)]];
    if (D) rows.push(['逻辑任务', num(D.counts.tasks)], ['下沉 Kernel', num(D.counts.kernels)],
                     ['参与核心', num(D.counts.lanes)], ['依赖边', num(D.counts.edges)],
                     ['张量', num(D.counts.tensors)]);
    const TL = { AIV: 'AIV · Vector', AIC: 'AIC · Cube', Spmd: 'SPMD 壳',
                 Group: 'Group', Orchestration: '编排' };
    const shell = L.kernels - (L.types.AIV || 0) - (L.types.AIC || 0);
    return '<section class="kf-inspector-section">' +
      '<h2 class="kf-inspector-title">运行规模</h2>' + dl(rows) +
      '<div class="kf-ri-types">' + Object.keys(L.types).sort((a, b) => L.types[b] - L.types[a])
        .map(k => '<span>' + esc(TL[k] || k) + '<b>' + L.types[k] + '</b></span>').join('') + '</div>' +
      (D ? '<p class="kf-ri-note is-dim">' + L.kernels + ' 个 Kernel 里有 ' + shell +
        ' 个是编排 / SPMD / Group 壳，不发射到设备 —— 所以 dfx 泳道里只看到 ' +
        D.counts.kernels + ' 个。</p>' : '') +
    '</section>';
  }

  /* an explicit open-items list. The main column reports each finding inside
     the section that owns it; here they are collected so the run can be closed
     out (or handed over) without scrolling the whole page. */
  function riOpen(L) {
    const items = [];
    if (L.over) items.push(['bad', '内存溢出', L.over + ' 个 kernel 超出上限', '必须改']);
    if (L.atLimit) items.push(['warn', '片上缓冲零余量',
      L.atLimit + ' 个 kernel 填满 ' + spLabel(L.kmem[0].sp), '有风险']);
    if (L.actionable) items.push(['warn', '搬运宽度',
      L.actionable + ' 处可改 · 同一个根因', '待优化']);
    if (L.byCause.row) items.push(['dim', '分页布局限制',
      L.byCause.row + ' 处加不宽，算子内无解', '不可解']);
    if (L.demotedK.length) items.push(['warn', '流水降级',
      L.demotedK.map(k => k.n).join(' · '), '待确认']);
    items.push(['idle', '正确性比对', '这次运行没有产出 oracle 输出', '未运行']);
    items.push(['idle', '性能基线', '没有可对照的历史运行', '未建立']);

    return '<section class="kf-inspector-section">' +
      '<h2 class="kf-inspector-title">这次运行留下的</h2>' +
      '<ul class="kf-ri-open">' + items.map(i =>
        '<li class="is-' + i[0] + '"><b>' + esc(i[1]) + '</b><em>' + esc(i[3]) + '</em>' +
          '<small>' + esc(i[2]) + '</small></li>').join('') +
      '</ul>' +
    '</section>';
  }

  function runInspector(t, r) {
    const L = r.live ? liveRun() : null;
    const D = r.live ? window.PTO_RUN_TRACE : null;
    const P = D && D.perf;
    let html = '<div id="kfRunInspector" class="kf-ri">' + riSource(t, r, L);

    if (!L) {
      html += riArchivedArts(r) +
        '<div class="kf-inspector-card"><b>只剩结论</b><p>' +
        esc(r.purged
          ? (r.note || '产物目录已清理。') + '结论保留在记录里，逐项数据无法再核。'
          : '这次运行的产物目录不在当前工作区，页面上的判定来自归档记录，' +
            '无法下钻到 IR、内存水位或运行时 trace。') +
        '</p></div></div>';
      return html;
    }

    html += riArts(Object.assign({ dir: r.dir }, L));
    if (P) html += riTime(P) + riCores(D, P);
    html += riScale(L, D) + riOpen(L);
    return html + '</div>';
  }

  function renderComparePicker() {
    const task = TASKS.find(t => t.id === st.task);
    if (!els.compareBox || !task) return;
    st.compareRuns = [];
    if (task.runs.length < 2) {
      els.compareBox.innerHTML = '<div><b>当前算子暂无两次可比较的运行</b><small>' + esc(task.title) + ' · 至少需要两次执行记录</small></div>';
      return;
    }
    els.compareBox.innerHTML = '<div><b>选择同一算子的两次运行</b><small>' + esc(task.title) + '</small></div>' +
      '<div class="kf-th-compare-options">' + task.runs.map(r => '<label><input type="checkbox" value="' + r.id + '" data-th-compare-run><span>' + esc(r.id) + '</span><em>' + esc(getRunDisplayStatus(r)[0]) + '</em></label>').join('') + '</div>' +
      '<button type="button" class="kf-th-compare-submit" data-th-compare-submit disabled>开始对比</button>';
  }

  function renderRunComparison() {
    const task = TASKS.find(t => t.id === st.task);
    const selected = task ? st.compareRuns.map(id => task.runs.find(r => r.id === id)).filter(Boolean) : [];
    if (selected.length !== 2 || !els.detail) return;
    const run106 = selected.find(r => r.id === 'run_106'), run107 = selected.find(r => r.id === 'run_107'), run108 = selected.find(r => r.id === 'run_108');
    if (run106 && run107) {
      renderCorrectnessRecoveryComparison(task, run106, run107);
      return;
    }
    if (run107 && run108) {
      renderOptimizationComparison(task, run107, run108);
      return;
    }
    const live = liveRun();
    const snapshots = selected.map(r => {
      const measured = task.id === 'task_decode' && live && r.id === live.stamp ? live : null;
      return {
        title: task.title, run: r.id, verdict: getRunDisplayStatus(r)[0],
        model: task.model, target: r.target || '—',
        passes: measured ? measured.passes + ' pass' : '—',
        kernels: measured ? measured.kernels + ' kernel' : '—',
      };
    });
    els.detail.innerHTML = '<section class="kf-rd kf-th-comparison">' +
      '<div class="kf-th-compare-head"><div><span class="kf-eyebrow">COMPARISON</span><h2>运行对比</h2><p>' + esc(task.title) + '</p></div><button type="button" class="kf-th-compare-close" data-th-compare-close>返回运行</button></div>' +
      '<div class="kf-th-compare-grid">' + snapshots.map(s => '<article><h3>' + esc(s.title) + '</h3><code>' + esc(s.run) + '</code><dl>' +
        '<div><dt>状态</dt><dd>' + esc(s.verdict) + '</dd></div><div><dt>模型</dt><dd>' + esc(s.model) + '</dd></div>' +
        '<div><dt>目标</dt><dd>' + esc(s.target) + '</dd></div>' +
        '<div><dt>Pass</dt><dd>' + esc(s.passes) + '</dd></div><div><dt>Kernel</dt><dd>' + esc(s.kernels) + '</dd></div>' +
      '</dl></article>').join('') + '</div></section>';
  }

  /* #106 → #107 is a diagnostic story, not a general-purpose diff. Its goal is
     to make the repair and the evidence for that repair legible before users
     move on to the separate performance diagnosis in #107. */
  function renderCorrectnessRecoveryComparison(task, before, after) {
    const profile = diagnosisViewForRun(before.id);
    const first = profile?.tensors?.find(t => t.id === profile.firstDivergence?.id) || null;
    const firstTensor = first ? first.tid : '—';
    const maxAbs = profile?.result?.maxAbs != null ? String(profile.result.maxAbs) : '—';
    const row = (label, x, y, beforeTone, afterTone) => '<tr><th>' + esc(label) + '</th><td' + (beforeTone === 'bad' ? ' class="is-bad"' : '') + '>' + esc(x) + '</td><td' + (afterTone === 'ok' ? ' class="is-ok"' : '') + '>' + esc(y) + '</td></tr>';
    els.detail.innerHTML = '<section class="kf-rd kf-th-comparison kf-th-comparison--correctness">' +
      '<div class="kf-th-compare-head"><div><span class="kf-eyebrow">正确性修复验证</span><h2>Run ' + esc(runDisplayId(before)) + ' vs Run ' + esc(runDisplayId(after)) + '</h2><p>' + esc(task.title) + ' · 修复是否消除了数值分歧</p></div><button type="button" class="kf-th-compare-close" data-th-compare-close>返回运行</button></div>' +
      '<section class="kf-rd-sec"><div class="kf-rd-h">正确性变化<small>失败 Run 与修复后 Run 的关键证据</small></div><table class="kf-th-compare-outcome"><thead><tr><th></th><th>' + esc(runDisplayId(before)) + '</th><th>' + esc(runDisplayId(after)) + '</th></tr></thead><tbody>' +
        row('正确性', 'FAIL', 'PASS', 'bad', 'ok') +
        row('最大绝对误差', maxAbs, '容差范围内') +
        row('首个分歧', firstTensor, '—', 'bad') +
        row('重复运行', '不稳定', '稳定', null, 'ok') +
        row('Task #182 → #197', '缺少依赖', '已建立排序', null, 'ok') +
      '</tbody></table></section>' +
      '<section class="kf-th-correctness-story" aria-label="Correctness recovery evidence">' +
        '<div><span>修改</span><b>新增 Task #182 → #197 排序依赖</b></div>' +
        '<div><span>证据</span><b>重复运行由不稳定 → 稳定 · ' + esc(firstTensor) + ' 分歧 → 无</b></div>' +
        '<div class="is-ok"><span>结论</span><b>正确性已恢复</b></div>' +
      '</section>' +
      '<p class="kf-rd-note">#106 正确性未通过，因此性能结论为 NOT EVALUATED；#107 正确性通过后才进入性能诊断 —— 性能证据在「执行」中查看。</p>' +
    '</section>';
  }

  function renderOptimizationComparison(task, before, after) {
    const a = getRunModel(before), b = getRunModel(after);
    const ma = a.compareMetrics, mb = b.compareMetrics;
    const sourceChange = (b.changes || []).find(x => x.domain === 'source') || {};
    const compileChange = (b.changes || []).find(x => x.domain === 'compile') || {};
    const row = (label, x, y, delta) => '<tr><th>' + esc(label) + '</th><td>' + esc(x) + '</td><td>' + esc(y) + '</td><td>' + esc(delta) + '</td></tr>';
    const change = (label, beforeValue, afterValue) => '<div class="kf-rd-art is-static"><b>' + esc(label) + '</b><code>修改前 · ' + esc(beforeValue) + '</code><small>修改后 · ' + esc(afterValue) + '</small></div>';
    els.detail.innerHTML = '<section class="kf-rd kf-th-comparison">' +
      '<div class="kf-th-compare-head"><div><span class="kf-eyebrow">COMPARISON</span><h2>Run #107 vs Run #108</h2><p>' + esc(task.title) + ' · 修改 → 证据 → 效果 → 结果</p></div><button type="button" class="kf-th-compare-close" data-th-compare-close>返回运行</button></div>' +
      '<section class="kf-rd-sec"><div class="kf-rd-h">结果<small>优化结果与守护条件</small></div><table class="kf-th-compare-outcome"><thead><tr><th></th><th>#107</th><th>#108</th><th>Δ</th></tr></thead><tbody>' +
        row('正确性', 'PASS', 'PASS', 'unchanged') +
        row('Latency', ma.latency, mb.latency, '-25.3%') +
        row('Critical Path', ma.criticalPath, mb.criticalPath, '-30.5%') +
        row('Task #182', ma.task182, mb.task182, '-31.3%') +
        row('MTE stall', ma.mteStall, mb.mteStall, '-14 pp') +
        row('Transfer width', ma.transferWidth, mb.transferWidth, '+100%') +
        row('L0B peak', ma.l0bPeak, mb.l0bPeak, '+2 pp') +
        row('Kernel count', ma.kernelCount, mb.kernelCount, '—') +
      '</tbody></table><p class="kf-rd-note">未发现正确性回退。Golden / Oracle 未变化 · 12 / 12 checkpoint 匹配。</p></section>' +
      '<section class="kf-rd-sec"><div class="kf-rd-h">修改内容<small>执行结构未缩减；只改变 Task #182 的 memory behavior</small></div><div class="kf-rd-arts">' +
        change('Source · ' + (sourceChange.title || 'RoPE transfer'), sourceChange.before || '—', sourceChange.after || '—') +
        change('Compilation · ' + (compileChange.title || 'Kernel count'), compileChange.before || '45', compileChange.after || '45') +
        change('Execution · Task topology / dependencies', 'unchanged', 'unchanged') +
        change('Resources · L0B', '81%', '83% · within budget') +
      '</div></section>' +
      '<section class="kf-rd-sec"><div class="kf-rd-h">为什么变快<small>同一执行路径中，窄搬运变宽，降低 MTE 等待</small></div><div class="kf-oi-links">' +
        '<span class="kf-oi-evidence">2 × narrow transfer → 1 × wide transfer</span><span class="kf-oi-evidence">↓</span><span class="kf-oi-evidence">256 B → 512 B · MTE stall 31% → 17%</span><span class="kf-oi-evidence">↓</span><span class="kf-oi-evidence">Task #182 · 214 µs → 147 µs</span><span class="kf-oi-evidence">↓</span><span class="kf-oi-evidence">Critical Path · 1.41 ms → 0.98 ms</span><span class="kf-oi-evidence">↓</span><span class="kf-oi-evidence">Latency · 1.82 ms → 1.36 ms</span>' +
      '</div><p class="kf-rd-note">资源峰值增加 2 pp，但仍在预算内；未通过删减 kernel、task 或牺牲正确性换取性能。</p></section>' +
      '<div class="kf-oi-actions"><button type="button" data-ws-set-baseline="run_108">设为可信基线</button></div>' +
    '</section>';
  }

  function compareCurrentRunWith(runId) {
    const task = TASKS.find(t => t.id === st.task);
    if (!task || !task.runs.some(r => r.id === runId) || !task.runs.some(r => r.id === st.run)) return;
    st.compareRuns = [runId, st.run];
    renderRunComparison();
  }

  function setTrustedBaseline(runId) {
    const task = TASKS.find(t => t.id === st.task);
    const run = task && task.runs.find(r => r.id === runId);
    if (!run) return;
    const model = getRunModel(run);
    model.baseline = true;
    model.baselineMeta = {
      id: 'ptok://qwen3-14b/decode-layer@run108', sourceCommit: '8da1bf09', backend: 'Ascend 910B',
      environmentFingerprint: 'env:8da1bf09', inputShape: '[16, 40, 128]', compilerVersion: 'PyPTO 0.8', runId: run.id
    };
    st.run = run.id; st.selection = null; st.tab = 'overview';
    render(); renderDetail();
    const toast = $('#toast');
    if (toast) { toast.textContent = 'Run #108 已设为可信基线'; toast.classList.add('is-visible'); setTimeout(() => toast.classList.remove('is-visible'), 1800); }
  }

  /* Inspector ownership follows the primary Activity rail. Run is the
     workflow activity as a whole; its internal analysis tabs never change
     this layout rule. Project and Model Analysis keep their own inspectors. */
  function runActivityIsActive() {
    const workflow = $('[data-side-view="workflow"]');
    const activity = $('#activityWorkflow');
    return !!((workflow && !workflow.hidden) || (activity && activity.getAttribute('aria-pressed') === 'true'));
  }

  function syncRunInspectorLayout() {
    const toggle = $('#inspectorToggle');
    const split = $('#ideMainSplit');
    if (!split) return;
    const frame = split.closest('[data-ide-frame]');
    const panes = ['explorer', 'editor-preview', 'inspector']
      .map(name => split.querySelector(':scope > [data-ide-pane="' + name + '"]'));

    if (runActivityIsActive()) {
      if (!st.runSplitSession && panes.every(Boolean)) {
        const storageKey = split.dataset.storageKey;
        st.runSplitSession = {
          panes: panes.map(pane => ({
            flex: pane.style.flex,
            flexBasis: pane.style.flexBasis,
            width: pane.style.width
          })),
          storageKey,
          storedSizes: storageKey ? localStorage.getItem(storageKey) : null
        };
        /* Run hides the inspector, so a percentage split would hand the freed
           column to the explorer too and the left pane would land ~200px wider
           than Project. Pin the explorer to whatever width Project is showing
           and let the editor take the rest; the editor keeps a grow value above
           15 so the legacy-damage repair below never fires on a pinned pane. */
        const explorerPx = st.runSplitExplorerPx || panes[0].getBoundingClientRect().width || 0;
        panes[0].style.flex = explorerPx > 0 ? '0 0 ' + explorerPx + 'px' : '30 1 0%';
        panes[0].style.flexBasis = explorerPx > 0 ? explorerPx + 'px' : '0%';
        panes[0].style.width = 'auto';
        panes[1].style.flex = '70 1 0%';
        panes[1].style.flexBasis = '0%';
        panes[1].style.width = 'auto';
      }
      if (toggle) toggle.hidden = true;
      if (frame) frame.dataset.surface = 'solid';
      split.classList.add('kf-run-no-inspector');
      return;
    }

    if (st.runSplitSession && panes.every(Boolean)) {
      const width = panes[0].getBoundingClientRect().width;
      if (width > 0) st.runSplitExplorerPx = width;

      panes.forEach((pane, index) => {
        const saved = st.runSplitSession.panes[index];
        pane.style.flex = saved.flex;
        pane.style.flexBasis = saved.flexBasis;
        pane.style.width = saved.width;
      });
      const { storageKey, storedSizes } = st.runSplitSession;
      if (storageKey) {
        if (storedSizes == null) localStorage.removeItem(storageKey);
        else localStorage.setItem(storageKey, storedSizes);
      }
      st.runSplitSession = null;
    }
    if (!st.runSplitSession && panes.every(Boolean) && toggle?.getAttribute('aria-pressed') === 'true') {
      const weights = panes.map(pane => parseFloat(pane.style.flex) || 0);
      const hasLegacyRunDamage = weights[0] > 80 || weights[1] < 15 || weights[2] < 8;
      if (hasLegacyRunDamage) {
        [22, 51, 27].forEach((size, index) => {
          panes[index].style.flex = size + ' 1 0%';
          panes[index].style.flexBasis = '0%';
          panes[index].style.width = 'auto';
        });
        const storageKey = split.dataset.storageKey;
        if (storageKey) localStorage.setItem(storageKey, JSON.stringify([22, 51, 27]));
      }
    }
    split.classList.remove('kf-run-no-inspector');
    if (frame) frame.removeAttribute('data-surface');
    if (toggle) toggle.hidden = false;
  }

  function traceData() { return window.PTO_RUN_TRACE || null; }
  function objectButton(kind, id, label, sourceTab, selected) {
    return '<button type="button" class="kf-oi-link' + (selected ? ' is-sel' : '') + '" data-ws-select-kind="' + esc(kind) + '" data-ws-select-id="' + esc(id) + '" data-ws-source="' + esc(sourceTab || st.tab) + '"' +
      (selected ? ' aria-pressed="true"' : '') + '>' + esc(label) + '</button>';
  }

  /* 统一 Object Header：Task / Tensor / Kernel / Pass 共用同一个身份头。
     结构固定为「种类 · 标题（身份）· 附带说明」，同一对象跨页后读起来是同一个
     东西，差别只体现在下面的证据区。role 是可选的角色小片（如 Task 的生产者 /
     可疑写入），带色调但不喧宾夺主。 */
  const KIND_LABEL = { op: '算子', tensor: '张量', task: '任务', kernel: 'Kernel',
                        dependency: '依赖', buffer: '缓冲区', pass: 'Pass', timeline: '时间线', finding: '发现' };
  function objectHead(kind, title, meta, role) {
    const eyebrow = KIND_LABEL[kind] || String(kind).toUpperCase();
    return '<div class="kf-oi-head"><span>' + esc(eyebrow) +
      (role ? '<em class="' + esc(role.tone || '') + '">' + esc(role.label) + '</em>' : '') +
      '</span><h3>' + esc(title) + '</h3>' + (meta ? '<small>' + esc(meta) + '</small>' : '') + '</div>';
  }

  function objectInspector(selection) {
    const D = traceData();
    const L = liveRun();
    if (!selection) return '<div id="kfObjectInspector" class="kf-oi is-empty"><p>选择 Task、Tensor、Kernel、依赖、Pass 或 Buffer 查看属性与证据。</p></div>';
    const kind = selection.kind, id = selection.id;
    let title = kind, meta = String(id), body = '', role = null, variantClass = '';
    const section = (label, content) => '<section class="kf-inspector-section"><h2 class="kf-inspector-title">' + label + '</h2>' + content + '</section>';
    const rows = (items) => dl(items);
    const current = TASKS.find(x => x.id === st.task)?.runs.find(x => x.id === st.run);
    if (isPerformanceWarningStory(current) && kind === 'task' && id === '182') {
      title = 'Task #182'; meta = 'attention_incore_2';
      body = section('Kernel', rows([['Kernel', 'attention_incore_2'], ['耗时', '214 µs'], ['参考值', '147 µs'], ['Δ', '+45.6%'], ['Critical Path', 'YES']])) +
        section('正确性', rows([['输入', '正常'], ['输出', '正确'], ['正确性', 'PASS']])) +
        section('动作', '<div class="kf-oi-actions"><button type="button" data-ws-select-kind="kernel" data-ws-select-id="attention_incore_2" data-ws-source="performance">查看 Kernel 性能证据</button></div>');
    } else if (isPerformanceWarningStory(current) && kind === 'kernel' && id === 'attention_incore_2') {
      title = 'attention_incore_2'; meta = 'Kernel 性能证据';
      body = section('搬运证据', rows([['实际宽度', '256 B'], ['目标宽度', '512 B'], ['触发次数', '16'], ['MTE stall', '31% · 参考 17%'], ['PMU coverage', 'partial']])) +
        section('问题', '<p class="kf-ri-note">RoPE lo/hi 半维被拆成两次窄搬运，增加搬运次数，并拉长 Critical Path 上的 Task #182。</p>') +
        section('潜在改动', '<p class="kf-ri-note">合并两次 64-element transfer → 一次 128-element transfer；在 tile 内完成 rotate-half 相关计算。</p>') +
        section('预期效果与风险', '<p class="kf-ri-note">搬运次数 ↓ · 有效宽度 ↑。额外 temporary tile / local copy 可能抵消收益，必须通过下一次 Run 验证。</p>') +
        section('源码映射', '<p class="kf-ri-note"><code>decode_layer.py:728</code> · RoPE block → attention_incore_2</p>') +
        section('动作', '<div class="kf-oi-actions"><button type="button" data-ws-open-source="decode_layer.py:728" data-ws-source-context="RoPE transfer recommendation">定位 Source</button><button type="button" data-ws-optimize-rerun>应用宽搬运方案并重跑</button></div>');
    } else if (usesCorrectnessDiagnosis(current) && kind === 'op' && dgOp(id)) {
      const o = dgOp(id);
      title = o.name; meta = '语义计算 · ' + o.be;
      body = section('动作', '<div class="kf-oi-actions"><button type="button" data-ws-open-source="' + esc(o.src) + '" data-ws-source-context="' + esc(o.name) + '">定位源码</button></div>');
    } else if (usesCorrectnessDiagnosis(current) && kind === 'tensor' && dgTensor(id)) {
      const t = dgTensor(id);
      const rProd = dgRuntimeProducer(t.id);
      const verdict = t.state === 'match' ? '匹配' : t.state === 'unchecked' ? '无参考基准' : '不一致';
      const stateRows = [['状态', verdict], ['首个分歧', t.state === 'first' ? '是' : t.state === 'propagated' ? '否 · 下游传播' : '—']];
      if (t.ref && t.act) stateRows.push(['最大绝对误差', String(t.maxAbs)], ['最大相对误差', String(t.maxRel)]);
      title = t.name; meta = t.tid + ' · ' + t.shape + ' · ' + t.dtype;
      body = section('比对状态', rows(stateRows)) +
        section('动作', '<div class="kf-oi-actions">' +
          (rProd
            ? '<button type="button" data-dg-select-kind="task" data-dg-select-id="' + rProd.id + '">定位运行时生产者</button>'
            : '') +
          '<button type="button" data-dg-' + (dg.runtime ? 'collapse' : 'expand') + '>' + (dg.runtime ? '收起运行时' : '展开运行时') + '</button>' +
        '</div>');
    } else if (usesCorrectnessDiagnosis(current) && kind === 'task' && dgTask(id)) {
      const t = dgTask(id);
      const pred = DG.runtimeChain.filter(x => x.to === t.id).map(x => x.from);
      const succ = DG.runtimeChain.filter(x => x.from === t.id).map(x => x.to);
      const outTensors = t.writes.map(n => {
        const x = dgTensor(n), st = x ? dgStateText(x.state) : '';
        return n + (st ? ' · ' + st : '');
      });
      /* Task tooltip 只保留定位所需的节奏：身份 → 时序 → 数据流 → 依赖证据。 */
      const tlRow = DG.timeline.rows.find(r => r.task === t.id);
      const ov = DG.timeline.overlap;
      const overlapDuration = ov.to - ov.from;
      const isReader = t.id === ov.a;
      const isWriter = t.id === ov.b;
      const isRisk = isReader || isWriter;
      const roleTone = t.role === 'suspicious' ? ' is-warn' : t.role === 'producer' ? ' is-bad' : '';
      const dependencyIn = pred.length
        ? pred.map(x => 'Task #' + x).join(' / ')
        : t.expectsAfter ? '缺失：应在 Task #' + t.expectsAfter + ' 之后' : '无';
      const dependencyOut = succ.length ? succ.map(x => 'Task #' + x).join(' / ') : '无';
      const flowRows = [];
      if (t.reads.length) flowRows.push(['读取', t.reads.join(' · ')]);
      if (outTensors.length) flowRows.push(['写入', outTensors.join(' · ')]);
      if (t.shared) flowRows.push(['共享 Buffer', t.shared.buffer + ' · ' + (t.shared.op === 'read' ? '读取' : '写入')]);
      if (t.semantic) flowRows.push(['语义映射', t.semantic]);

      variantClass = ' kf-oi--task';
      /* 身份头走统一 Object Header：任务 = Task #182，角色作为小片挂在种类旁，
         下面只放证据，不再另起一套头部。 */
      role = { label: t.label, tone: roleTone.trim() };
      title = 'Task #' + t.id;
      meta = t.core + (t.semantic ? ' · ' + t.semantic : '');
      body = (t.note ? '<p class="kf-ri-note">' + esc(t.note) + '</p>' : '') +
        (tlRow ? '<div class="kf-task-tip-metrics" aria-label="时间线摘要">' +
          '<div><span>开始</span><b>' + tlRow.s + ' μs</b></div>' +
          '<div><span>结束</span><b>' + tlRow.e + ' μs</b></div>' +
          '<div><span>耗时</span><b>' + (tlRow.e - tlRow.s) + ' μs</b></div>' +
        '</div>' : '') +
        (flowRows.length ? '<section class="kf-task-tip-section"><h2>数据访问</h2>' +
          '<dl>' + flowRows.map(([label, value]) => '<div><dt>' + esc(label) + '</dt><dd>' + esc(value) + '</dd></div>').join('') + '</dl></section>' : '') +
        '<section class="kf-task-tip-section"><h2>依赖关系</h2><dl>' +
          '<div><dt>前驱</dt><dd>' + esc(dependencyIn) + '</dd></div>' +
          '<div><dt>后继</dt><dd>' + esc(dependencyOut) + '</dd></div>' +
        '</dl></section>' +
        (isRisk ? '<div class="kf-task-tip-finding' + (isWriter ? ' is-warn' : ' is-bad') + '">' +
          '<span>' + (isWriter ? '排序缺口' : '读写重叠') + '</span>' +
          '<b>' + (isWriter
            ? '与 Task #' + esc(ov.a) + ' 重叠 ' + overlapDuration + ' μs，且缺少依赖边'
            : 'Task #' + esc(ov.b) + ' 在本任务读完前开始写入 · ' + overlapDuration + ' μs') + '</b>' +
          '<small>3 次重复运行结果不稳定</small></div>' : '') +
        '<div class="kf-task-tip-actions kf-oi-actions"><button type="button" data-dg-view="timeline">在时间线中查看</button></div>';
    } else if (usesCorrectnessDiagnosis(current) && kind === 'timeline') {
      const T = DG.timeline, ov = T.overlap, a = dgTask(ov.a), b = dgTask(ov.b);
      const dur = ov.to - ov.from;
      title = '时间线证据'; meta = '共享 buffer ' + T.buffer + ' 的读写重叠';
      body = section('重叠窗口', rows([['区间', ov.from + ' → ' + ov.to + ' μs'], ['时长', dur + ' μs']])) +
        section('读写冲突', rows([['读方', 'Task #' + a.id + ' · ' + a.core + ' 读共享 buffer ' + T.buffer],
          ['写方', 'Task #' + b.id + ' · ' + b.core + ' 写共享 buffer ' + T.buffer]])) +
        section('解读', '<p class="kf-ri-note">生产者在读路径结束前，写入方已开始覆盖同一块 buffer，' +
          '可能造成 early overwrite。这与重复运行不稳定的现象一致。</p>') +
        section('动作', '<div class="kf-oi-actions"><button type="button" data-dg-select-kind="dependency" data-dg-select-id="missing_182_197">查看缺失依赖</button></div>');
    } else if (usesCorrectnessDiagnosis(current) && kind === 'buffer' && id === 'B2') {
      title = 'B2'; meta = '共享 buffer';
      body = section('共享 buffer 生命周期', rows([['读取方', 'Task #182 · AICore 7'], ['写入方', 'Task #197 · AICore 5'], ['当前排序', '无依赖边']])) +
        section('证据', '<p class="kf-ri-note">写入方与仍在执行的读取方重叠 48 μs；重复运行产出非确定性结果。</p>') +
        section('动作', '<div class="kf-oi-actions"><button type="button" data-ws-select-kind="dependency" data-ws-select-id="missing_182_197" data-ws-source="execution">查看缺失依赖</button></div>');
    } else if (usesCorrectnessDiagnosis(current) && kind === 'dependency' && id === 'missing_182_197') {
      title = '预期排序'; meta = 'Task #182 → Task #197';
      body = section('关系', rows([['原因', '共享 buffer 生命周期'], ['当前状态', '无依赖边'], ['实际影响', '写入方与仍在执行的读取方重叠']])) +
        section('证据', '<div class="kf-oi-links"><span class="kf-oi-evidence">时间线重叠</span><span class="kf-oi-evidence">共享 buffer</span><span class="kf-oi-evidence">输出非确定性</span></div>') +
        section('源码映射', '<p class="kf-ri-note"><code>decode_layer.py:728</code> → Task #182 读取 → Task #197 覆盖写入 → 排序依赖缺失</p>') +
        section('建议修复', '<p class="kf-ri-note"><code>deps=[reader_tid]</code></p>') +
        section('动作', '<div class="kf-oi-actions"><button type="button" data-ws-open-source="decode_layer.py:728" data-ws-source-context="Runtime task dependency">定位源码</button><button type="button" data-ws-fix-dependency>预览修复</button></div>');
    } else if (kind === 'task' && D && D.tasks[Number(id)]) {
      const i = Number(id), t = D.tasks[i], K = D.kernels[t.k] || {};
      /* 身份 = Task 编号，和 Correctness 的「Task #182」同一种读法；kernel 名与
         核数归到说明行，不再抢占标题。 */
      title = 'Task #' + (t.id != null ? t.id : i);
      meta = (t.kn || K.name || '') + ' · ' + t.c + ' 核';
      const pre = D.edges.filter(e => e[1] === i), suc = D.edges.filter(e => e[0] === i);
      body = section('身份与状态', rows([['任务 ID', t.id], ['时间', usFmt(t.s) + ' → ' + usFmt(t.e)], ['墙上时间', usFmt(t.e - t.s)], ['下沉 Kernel', (t.ks || []).map(x => x.kn).join(' + ')], ['调度域', t.sc || '—'], ['核心', t.c + ' 核']])) +
        section('Tensor 数据流', t.io && t.io.length ? '<div class="kf-oi-links">' + t.io.map(a => objectButton('tensor', a.x, (a.t === 'in' ? '输入 ' : '输出 ') + a.i + ' · ' + a.d + ' [' + a.sh.join('×') + ']', 'execution')).join('') + '</div>' : '<p class="kf-ri-note is-dim">未采集参数表。</p>') +
        section('依赖', '<div class="kf-oi-links">' + pre.slice(0, 8).map(e => objectButton('dependency', e[0] + ':' + e[1] + ':' + e[2], '前驱 · ' + (D.tasks[e[0]] || {}).kn, 'execution')).join('') + suc.slice(0, 8).map(e => objectButton('dependency', e[0] + ':' + e[1] + ':' + e[2], '后继 · ' + (D.tasks[e[1]] || {}).kn, 'execution')).join('') + '</div>') +
        section('动作', '<div class="kf-oi-actions">' + (t.ks || []).map(x => '<button type="button" data-ws-open-kernel="' + esc(x.kn) + '">在「编译」中查看</button>').join('') + '</div>');
    } else if (kind === 'tensor' && D && D.tensors[Number(id)]) {
      const i = Number(id), x = D.tensors[i], refs = D.tasks.filter(t => (t.io || []).some(a => a.x === i));
      title = 'Tensor ' + i; meta = x.d + ' · ' + num(x.n) + ' 元素';
      body = section('属性', rows([['dtype', x.d], ['元素', num(x.n)], ['地址', x.a || '—']])) +
        section('生产与消费', '<div class="kf-oi-links">' + refs.slice(0, 16).map(t => objectButton('task', D.tasks.indexOf(t), t.kn || t.id, 'execution')).join('') + '</div>');
    } else if (kind === 'dependency' && D) {
      const parts = String(id).split(':').map(Number), e = [parts[0], parts[1], parts[2]], a = D.tasks[e[0]], b = D.tasks[e[1]];
      title = 'Dependency'; meta = (a && a.kn || e[0]) + ' → ' + (b && b.kn || e[1]);
      body = section('关系', rows([['前驱', a ? a.kn : String(e[0])], ['后继', b ? b.kn : String(e[1])], ['类型', ['显式顺序', '数据产出', 'TensorMap'][e[2]] || '—']])) +
        section('动作', '<div class="kf-oi-links">' + (a ? objectButton('task', e[0], '查看前驱', 'execution') : '') + (b ? objectButton('task', e[1], '查看后继', 'execution') : '') + '</div>');
    } else if ((kind === 'kernel' || kind === 'buffer') && L) {
      const k = (window.PTO_IR_KERNELS || {}).kernels?.find(x => x.name === id), m = L.kmem.find(x => x.n === id);
      title = id; meta = kind === 'buffer' ? 'Buffer' : 'Kernel';
      body = section('身份与资源', rows([['类型', k ? k.type : '—'], ['最紧空间', m ? spLabel(m.sp) : '—'], ['使用', m ? kb(m.u) + ' / ' + kb(m.lim) + ' · ' + m.p + '%' : '未采集'], ['诊断', m ? m.diag + ' 条' : '—']])) +
        section('动作', '<div class="kf-oi-actions"><button type="button" data-ws-open-kernel="' + esc(id) + '">在「编译」中查看</button></div>');
    } else if (kind === 'pass') {
      if (isCompileFailureStory(current) && id === 'LegalizeIndexing') {
        title = 'LegalizeIndexing'; meta = 'PASS';
        body = section('状态', rows([['状态', 'FAIL'], ['问题', 'IR 校验失败'], ['受影响操作', 'tensor.write[index]'], ['源码', 'decode_layer.py:728'], ['上一个 Pass', 'PASS'], ['下一个 Pass', '未执行']])) +
          section('编译失败已定位', '<p class="kf-ri-note">LegalizeIndexing 将动态 index lowering 为 GM store 地址计算后，违反 index 位宽 / 地址计算约束。</p>') +
          section('动作', '<div class="kf-oi-actions"><button type="button" data-ws-open-source="decode_layer.py:728">打开源码</button><button type="button" data-ws-fix-rerun>应用静态仿射 fallback 并重跑</button></div>');
      } else {
        title = id; meta = 'IR Pass';
        body = section('编译证据', '<p class="kf-ri-note">Pass 河流图与 IR diff 保留在「编译」；选择已同步到该视图。</p>') + section('动作', '<div class="kf-oi-actions"><button type="button" data-ws-open-pass="' + esc(id) + '">打开「编译」</button></div>');
      }
    } else if (kind === 'finding') {
      const f = getFindings(current).find(x => x.id === id) || FINDINGS.find(x => x.k === id);
      title = f ? f.title : 'Finding'; meta = f ? (DOMAIN_LABEL[f.domain] || f.domain || '') : '';
      body = section('结论', '<p class="kf-ri-note">' + esc(f ? (f.summary || f.what) : '—') + '</p>') +
        section('证据', '<div class="kf-oi-links">' + (f && f.evidence || []).map(x => '<span class="kf-oi-evidence">✓ ' + esc(x) + '</span>').join('') + '</div>') +
        section('下一步', '<div class="kf-oi-actions"><button type="button" data-ws-route="' + esc(f?.action?.route || f?.domain || 'overview') + '">' + esc(f?.action?.label || '查看证据') + '</button></div>');
    }
    const head = objectHead(kind, title, meta, role);
    return '<div id="kfObjectInspector" class="kf-oi' + variantClass + '">' + head + body + '</div>';
  }

  function hoveredObject(target) {
    if (!target) return null;
    if (target.dataset.wsSelectKind) return { kind: target.dataset.wsSelectKind, id: target.dataset.wsSelectId, sourceTab: target.dataset.wsSource || st.tab };
    // Correctness graph nodes are wired through data-dg-* so the click can also
    // repaint the graph; hover still goes through this one inspector.
    if (target.dataset.dgSelectKind) return { kind: target.dataset.dgSelectKind, id: target.dataset.dgSelectId, sourceTab: 'correctness' };
    if (target.dataset.wsFinding) return { kind: 'finding', id: target.dataset.wsFinding, sourceTab: 'overview' };
    if (target.matches('[data-kg-p], [data-kg-kp]')) {
      const i = target.dataset.kgP || target.dataset.kgKp;
      return { kind: 'pass', id: ((window.PTO_IR_KERNELS || {}).passNames || [])[Number(i)] || ('Pass ' + i), sourceTab: 'compilation' };
    }
    return null;
  }

  function cancelObjectTooltipShow() {
    if (objectTooltipShowTimer) { clearTimeout(objectTooltipShowTimer); objectTooltipShowTimer = null; }
  }

  function hideObjectTooltip() {
    if (!objectTooltip) return;
    if (objectTooltipTimer) { clearTimeout(objectTooltipTimer); objectTooltipTimer = null; }
    cancelObjectTooltipShow();
    if (objectTooltipTarget) objectTooltipTarget.removeAttribute('aria-describedby');
    objectTooltipTarget = null;
    objectTooltip.classList.remove('is-open');
    objectTooltip.setAttribute('aria-hidden', 'true');
  }

  function scheduleObjectTooltipHide() {
    cancelObjectTooltipShow();
    if (objectTooltipTimer) clearTimeout(objectTooltipTimer);
    objectTooltipTimer = setTimeout(hideObjectTooltip, 120);
  }

  /* 未打开时延时出现；已经打开时直接切换，避免连续浏览时反复等待 */
  function scheduleObjectTooltipShow(target, obj) {
    if (!objectTooltip || !obj || obj.id == null) return;
    if (objectTooltipTimer) { clearTimeout(objectTooltipTimer); objectTooltipTimer = null; }
    if (objectTooltip.classList.contains('is-open')) { cancelObjectTooltipShow(); showObjectTooltip(target, obj); return; }
    cancelObjectTooltipShow();
    objectTooltipShowTimer = setTimeout(() => {
      objectTooltipShowTimer = null;
      if (target.isConnected) showObjectTooltip(target, obj);
    }, TOOLTIP_SHOW_DELAY);
  }

  function showObjectTooltip(target, obj) {
    if (!objectTooltip || !obj || obj.id == null) return;
    cancelObjectTooltipShow();
    if (objectTooltipTimer) { clearTimeout(objectTooltipTimer); objectTooltipTimer = null; }
    if (objectTooltipTarget && objectTooltipTarget !== target) objectTooltipTarget.removeAttribute('aria-describedby');
    objectTooltipTarget = target;
    target.setAttribute('aria-describedby', objectTooltip.id);
    objectTooltip.innerHTML = objectInspector({ kind: obj.kind, id: String(obj.id), sourceTab: obj.sourceTab || st.tab, runId: st.run }).replace(' id="kfObjectInspector"', '');
    objectTooltip.classList.add('is-open');
    objectTooltip.setAttribute('aria-hidden', 'false');
    /* 卡片贴在目标左右两侧，纵向与目标居中对齐 */
    const box = target.getBoundingClientRect();
    const tip = objectTooltip.getBoundingClientRect();
    const maxLeft = window.innerWidth - tip.width - 8;
    const right = box.right + TOOLTIP_GAP;
    const left = box.left - TOOLTIP_GAP - tip.width;
    let x = right <= maxLeft ? right : left;
    if (x < 8) x = Math.max(8, Math.min(right, maxLeft));
    const y = Math.max(8, Math.min(box.top + box.height / 2 - tip.height / 2, window.innerHeight - tip.height - 8));
    objectTooltip.style.left = Math.round(x) + 'px';
    objectTooltip.style.top = Math.round(y) + 'px';
  }

  const HOVER_TARGET = '[data-ws-select-kind], [data-ws-finding], [data-kg-p], [data-kg-kp], [data-dg-select-kind]';

  function bindObjectTooltips() {
    const onOver = e => {
      const target = e.target.closest(HOVER_TARGET);
      if (!target || target.contains(e.relatedTarget)) return;
      scheduleObjectTooltipShow(target, hoveredObject(target));
    };
    const onOut = e => {
      const target = e.target.closest(HOVER_TARGET);
      if (target && !target.contains(e.relatedTarget) && !objectTooltip?.contains(e.relatedTarget)) scheduleObjectTooltipHide();
    };
    [els.root, els.detail].filter(Boolean).forEach(host => {
      host.addEventListener('pointerover', onOver);
      host.addEventListener('pointerout', onOut);
      host.addEventListener('focusin', onOver);
      host.addEventListener('focusout', onOut);
    });
    objectTooltip?.addEventListener('pointerenter', () => { if (objectTooltipTimer) clearTimeout(objectTooltipTimer); });
    objectTooltip?.addEventListener('pointerleave', scheduleObjectTooltipHide);
    objectTooltip?.addEventListener('click', e => {
      const pick = e.target.closest('[data-dg-select-kind]');
      if (pick) { dgSelect(pick.dataset.dgSelectKind, pick.dataset.dgSelectId); hideObjectTooltip(); return; }
      const view = e.target.closest('[data-dg-view]');
      if (view) { dg.view = view.dataset.dgView; dgRender(); hideObjectTooltip(); return; }
      const runtime = e.target.closest('[data-dg-collapse], [data-dg-expand]');
      if (runtime) { dg.runtime = runtime.hasAttribute('data-dg-expand'); dgRender(); hideObjectTooltip(); return; }
      onRunClick(e);
    });
  }

  function renderInspector() {
    // Object detail belongs to the hover tooltip in Run workspace.
  }

  /* ---------- 修复预览 ----------
     #106 → 执行 → 查看缺失依赖 的下一步。这里只做「看」：不写真实 patch、不碰
     源码文件。应用时直接复用既有的 createFollowupRun('dependency') 生成 #107，
     和这条链路原来的行为一致，只是中间多了一次确认。 */
  function fixPreviewHTML() {
    const diff = [
      'with pl.at(',
      '    level=pl.Level.CORE_GROUP,',
      '<span class="kf-rw-fix-add">+   deps=[reader_tid]</span>',
      '):'
    ].join('\n');
    return '<section class="kf-rw-fix-dialog" role="dialog" aria-modal="true" aria-labelledby="fixPreviewTitle">' +
      '<header><div><span class="kf-eyebrow">FIX PREVIEW</span><h2 id="fixPreviewTitle">修复预览</h2></div>' +
        '<button type="button" data-rw-fix-close aria-label="关闭">×</button></header>' +
      '<div class="kf-rw-fix-body">' +
        '<div><h3>运行时问题</h3><p><code>Task #182 → Task #197</code> 缺少排序依赖</p></div>' +
        '<div><h3>源码映射</h3><p><code>decode_layer.py:728</code></p></div>' +
        '<div><h3>建议修改</h3><pre>' + diff + '</pre></div>' +
        '<div><h3>说明</h3><p>Task #182 是当前 Run 的 runtime ID；<br>源码通过 <code>reader_tid</code> 建立 dependency。</p></div>' +
      '</div>' +
      '<footer class="kf-rw-fix-actions"><button type="button" class="btn" data-rw-fix-cancel>取消</button>' +
        '<button type="button" class="btn btn-solid" data-rw-fix-apply>应用修改并重新运行</button></footer>' +
    '</section>';
  }

  function openFixPreview() {
    if (!fixPreview) return;
    hideObjectTooltip();
    fixPreview.innerHTML = fixPreviewHTML();
    fixPreview.hidden = false;
    const cancel = fixPreview.querySelector('[data-rw-fix-cancel]');
    if (cancel && cancel.focus) cancel.focus({ preventScroll: true });
  }

  function closeFixPreview() {
    if (!fixPreview || fixPreview.hidden) return;
    fixPreview.hidden = true;
    fixPreview.innerHTML = '';
  }

  function bindFixPreview() {
    if (!fixPreview) return;
    fixPreview.addEventListener('click', e => {
      if (e.target === fixPreview) { closeFixPreview(); return; }   // 点遮罩关闭
      if (e.target.closest('[data-rw-fix-close], [data-rw-fix-cancel]')) { closeFixPreview(); return; }
      if (e.target.closest('[data-rw-fix-apply]')) {
        closeFixPreview();
        createFollowupRun('dependency');
      }
    });
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeFixPreview(); });
  }

  function clearSelection() { st.selection = null; hideObjectTooltip(); }
  function selectObject(obj) {
    if (!obj || obj.id == null) return;
    st.selection = { kind: obj.kind, id: String(obj.id), sourceTab: obj.sourceTab || st.tab, runId: st.run };
    if (st.tab === 'execution' && obj.kind === 'task') window.PTO_TIMELINE?.selectTask?.(Number(obj.id));
  }

  /* demo-v2.js owns Activity switching. Observe the rail and its workflow pane
     so the right pane is removed for the entire Run activity, independent of
     Overview / Correctness / Execution / Compilation / Performance / Resources. */
  function watchRunInspectorLayout() {
    const retake = () => { syncPanel(); syncRunInspectorLayout(); };
    if (typeof MutationObserver !== 'function') { retake(); return; }
    const watch = (el, opts) => { if (el) new MutationObserver(retake).observe(el, opts); };
    watch($('[data-side-view="workflow"]'), { attributes: true, attributeFilter: ['hidden', 'class'] });
    watch($('#activityWorkflow'), { attributes: true, attributeFilter: ['aria-pressed', 'class'] });
    // Stage observation remains only for borrowed Compilation content.
    $$('.kf-stage').forEach(s => watch(s, { attributes: true, attributeFilter: ['class'] }));
    retake();
  }

  // the main column and the right rail are two views of the same selection
  function renderDetail() {
    syncRunInspectorLayout();
    hideObjectTooltip();
    renderDetailBody();
    renderInspector();
  }

  // Investigation is a view of the current Run, never a second diagnosis store.
  const investigation = { runId: null, controller: null, savedStates: new Map(), activeCaseId: null, entry: null };

  function investigationCases(run) {
    return window.PTO_RUN_INVESTIGATION?.cases?.(run.id) || [];
  }

  function activeInvestigationCase(run) {
    const cases = investigationCases(run);
    if (!cases.length) return null;
    const current = cases.find(item => item.id === investigation.activeCaseId) || cases[0];
    investigation.activeCaseId = current.id;
    return current;
  }

  function investigationContext(cases, active) {
    if (cases.length < 2) return '<p class="kf-investigation__description">' + esc(active.description) + '</p>';
    return '<div class="kf-investigation__cases" role="tablist" aria-label="调查案例">' +
      cases.map(item => '<button type="button" class="btn btn-ghost kf-investigation__case is-' + esc(item.tone || 'warning') + '" role="tab" data-investigation-case="' + esc(item.id) + '"' +
        ' aria-selected="' + String(item.id === active.id) + '" aria-controls="run-investigation-map">' +
        '<span class="kf-investigation__case-marker">' + esc(item.marker || 'WARNING · 调查') + '</span><strong>' + esc(item.title) + '</strong><small>' + esc(item.location || '') + '</small></button>').join('') +
      '</div><p class="kf-investigation__description">' + esc(active.description) + '</p>';
  }

  function releaseInvestigation() {
    if (investigation.controller) {
      const key = investigation.runId + ':' + investigation.activeCaseId;
      investigation.savedStates.set(key, investigation.controller.getState());
      investigation.controller.destroy();
      investigation.controller = null;
    }
    if (investigation.runId !== st.run) {
      investigation.runId = st.run;
      investigation.savedStates.clear();
      investigation.activeCaseId = null;
      investigation.entry = null;
      clearSelection();
    }
  }

  function investigationPanel(run) {
    const active = activeInvestigationCase(run);
    const count = investigationCases(run).length;
    const data = active?.investigationData;
    if (!data) return '';
    return '<section class="kf-investigation" aria-label="算子调查地图">' +
      '<header class="kf-investigation__title so-section-title so-findings-section-title"><h2>诊断发现 <span class="kf-investigation__count">' + count + '</span></h2></header>' +
      '<div data-run-investigation id="run-investigation-map" role="tabpanel"></div>' +
      '<footer class="kf-investigation-verdict"><div><span class="kf-rd-h">当前判断</span><p>' + esc(data.judgment) + '</p>' +
      '<p class="kf-rd-note">' + esc(data.nextStep) + '</p></div>' +
      (data.action ? '<button type="button" class="btn btn-solid" data-investigation-next>检查关键证据 →</button>' : '') +
      '</footer></section>';
  }

  function mountInvestigation(run) {
    const root = $('[data-run-investigation]', els.detail);
    const cases = investigationCases(run);
    const active = activeInvestigationCase(run);
    const data = active?.investigationData;
    if (!root || !data || !window.DiagnosticGraphController) return;
    const stateKey = run.id + ':' + active.id;
    investigation.controller = new window.DiagnosticGraphController(root, data, {
      evidenceMode: 'note', savedState: investigation.savedStates.get(stateKey), contextHTML: investigationContext(cases, active),
      typeTitles: { finding: '现象', reasoning: '缩小范围', entity: '定位对象', cause: '分支排查', hypothesis: '待验证假设', diagnosis: '当前判断' },
      openEvidence: openInvestigationEvidence
    });
  }

  function investigationReturnBar() {
    if (!investigation.entry || investigation.runId !== st.run || st.tab === 'overview') return '';
    return '<div class="kf-investigation-return"><button type="button" class="btn btn-ghost" data-investigation-return>← 返回调查地图</button>' +
      '<span>当前 Run 的调查证据 · ' + esc(investigation.entry.id || '编译校验') + '</span></div>';
  }

  function openInvestigationEvidence(input) {
    const action = window.PTO_RUN_INVESTIGATION?.resolveAction(st.run, input);
    if (!action) return;
    investigation.entry = action;
    dgReset(st.run);
    if (action.id != null) {
      selectObject({ kind: action.kind, id: action.id, sourceTab: 'overview' });
      dg.sel = { kind: action.kind, id: String(action.id) };
    }
    const profile = window.PTO_CORRECTNESS_DIAGNOSTICS.profiles[st.run];
    const options = action.tab === 'execution' ? { executionEntry: {
      from: 'investigation', runId: st.run, finding: '正确性调查', suspectedCause: '运行时排序',
      focusTask: action.taskIds?.[0] || action.id, taskIds: action.taskIds || [],
      timeline: profile.runtime?.timeline, range: action.range
    }} : action.tab === 'compilation' ? { compilationEntry: {
      from: 'investigation', runId: st.run, finding: '正确性调查', intent: '检查编译校验证据',
      numericalValidation: profile.compiler?.numericalValidation
    }} : {};
    toTab(action.tab, options);
    requestAnimationFrame(() => {
      if (st.run !== action.runId || st.tab !== action.tab) return;
      if (action.tab === 'correctness') {
        dgSelect(action.kind, action.id);
        const target = $$('[data-dg-select-id]', els.detail).find(el => el.dataset.dgSelectId === String(action.id));
        target?.scrollIntoView({ block: 'center', behavior: 'smooth' });
        if (target) showObjectTooltip(target, st.selection);
      } else if (action.tab === 'execution') {
        // This profile owns its own timeline; never focus similarly numbered
        // tasks in PTO_RUN_TRACE, which belongs to a different measured Run.
        $('[aria-label="排序证据"]', els.detail)?.scrollIntoView({ block: 'start', behavior: 'smooth' });
      } else if (action.tab === 'compilation' && action.id != null) {
        window.PTO_COMPILATION?.selectPass?.(action.id);
      }
    });
  }

  function overviewPanel(r, L) {
    const m = getRunModel(r);
    const states = DOMAIN_ORDER.map(key => {
      const d = getDomainVerdict(r, key), verdict = DOMAIN_VERDICT[d.verdict] || DOMAIN_VERDICT.unknown;
      /* Verdict 域仍是五个（一次 Run 可以又对又慢，结论必须分开说），但
         Performance 的证据已经并进 Execution，卡片直接指向合并后的页签，
         并在卡上写明去处，免得用户回来找一个不存在的页签。 */
      const where = toTabKey(key);
      return '<button type="button" class="is-' + verdict[1] + '" data-ws-route="' + where + '"><b>' + DOMAIN_LABEL[key] + '</b><em>' + verdict[0] + '</em><small>' + esc(d.summary) + '</small>' + (where === key ? '' : '<i class="kf-rw-route">证据见「' + DOMAIN_LABEL[where] + '」</i>') + '</button>';
    }).join('');
    const profile = isCorrectnessFailureStory(r) ? diagnosisViewForRun(r.id) : null;
    const findings = getFindings(r).map(f => {
      const severity = f.severity === 'critical' ? 'bad' : f.severity === 'warning' ? 'warn' : 'dim';
      /* Numerical Accuracy 卡的信息层级：Domain 在 span，Finding 是标题，
         Scenario（数值精度）与 Localization（首个分歧）在 metadata 行 ——
         一行说「差多少」，一行说「从哪儿开始」，Scenario 不做成第二个 eyebrow。 */
      if (f.domain === 'correctness' && profile?.type === 'numerical') {
        const first = profile.tensors?.find(t => t.id === profile.firstDivergence?.id);
        const firstLabel = [profile.firstDivergence?.id, first?.tid].filter(Boolean).join(' / ') || '—';
        const firstCaption = profile.firstDivergence?.kind === 'pass' ? '首个数值分歧 Pass' : '首个分歧 Tensor';
        const maxAbs = profile.result?.maxAbs != null ? profile.result.maxAbs : '—';
        return '<button type="button" class="kf-rw-finding kf-rw-finding--numerical is-' + severity + '" data-ws-finding="' + esc(f.id) + '">' +
          '<span>' + esc(String(f.severity || 'info').toUpperCase()) + ' · ' + DOMAIN_LABEL.correctness + '</span><div><b>输出不一致</b><small>数值精度 · 最大绝对误差 ' + esc(maxAbs) + ' <mark>' + esc(firstCaption) + ' · ' + esc(firstLabel) + '</mark></small></div><em>查看诊断 →</em></button>';
      }
      return '<button type="button" class="kf-rw-finding is-' + severity + '" data-ws-finding="' + esc(f.id) + '">' +
        '<span>' + esc(String(f.severity || 'info').toUpperCase()) + ' · ' + esc(DOMAIN_LABEL[f.domain] || f.domain) + '</span><b>' + esc(f.title) + '</b><small>' + esc(f.location || f.summary) + ' →</small></button>';
    }).join('') || '<p class="kf-rd-note is-dim">无待处理 Finding。</p>';
    const coverage = getEvidenceCoverage(r);
    const evidenceRows = profile ? correctnessEvidenceRows(profile, coverage) : (isCompileFailureStory(r)
      ? ['ir_validation', 'pass_dump', 'source_location', 'golden_compare', 'runtime_timeline', 'dependency_graph', 'pmu'].map(key => ({ key, entry: coverage[key] }))
      : Object.keys(coverage).map(key => ({ key, entry: coverage[key] })));
    const evidence = evidenceRows.map(({ key, entry: e }) => {
      const s = EVIDENCE_STATUS[e.status] || EVIDENCE_STATUS.unavailable;
      return '<div class="is-' + e.status + '"><b>' + esc(e.label || EVIDENCE_LABEL[key]) + '</b><small>' + esc(e.statusText || (s[0] + ' ' + s[1])) + '</small></div>';
    }).join('');
    const raw = (r.inventory || []).map(i => '<code>' + esc(i.where || i.label) + '</code>').join(' · ') || '<code>无挂载产物目录</code>';
    const lineage = m.derivedFrom
      ? '<p class="kf-rd-note">派生自 ' + esc(m.derivedFrom === 'run_105' ? '#105' : m.derivedFrom === 'run_106' ? '#106' : m.derivedFrom === 'run_107' ? '#107' : m.derivedFrom) + ' · ' + esc(m.change || (m.changes || []).join(' · ')) + '</p>' : '';
    const fixtureNote = m.fixture ? '<p class="kf-rd-note is-dim">场景记录 · ' + esc(m.fixtureLabel || '诊断用例') + '</p>' : '';
    const optimizationActions = isValidatedOptimizationStory(r)
      ? '<div class="kf-oi-actions"><button type="button" data-ws-compare-with="run_107">与 #107 对比</button></div>' : '';
    const baseline = m.baseline && m.baselineMeta
      ? '<p class="kf-rd-note">可信基线 · <code>' + esc(m.baselineMeta.id) + '</code></p>' +
        '<details class="kf-rw-evidence"><summary>可复现性元数据</summary><div class="kf-rw-evidence-body"><p class="kf-rd-note">source commit · ' + esc(m.baselineMeta.sourceCommit) + ' · backend · ' + esc(m.baselineMeta.backend) + ' · environment · ' + esc(m.baselineMeta.environmentFingerprint) + ' · input · ' + esc(m.baselineMeta.inputShape) + ' · compiler · ' + esc(m.baselineMeta.compilerVersion) + ' · Run ID · ' + esc(m.baselineMeta.runId) + '</p></div></details>' : '';
    const map = window.PTO_RUN_INVESTIGATION?.build(r.id);
    /* 分析状态（kf-rw-health）始终保留；有调查地图时在其后追加调查地图，
       不再用调查地图把 domain 结论整段顶掉。 */
    return '<section class="kf-rw-health"><div>' + states + '</div></section>' +
      (map ? investigationPanel(r) : '') +
      '<section class="kf-rd-sec"><div class="kf-rd-h">问题发现<small>用户需要处理的问题</small></div>' + findings + fixtureNote + lineage + baseline + optimizationActions + '</section>' +
      '<details class="kf-rw-evidence"><summary>诊断证据与产物</summary><div class="kf-rw-evidence-body kf-rw-coverage">' + evidence +
        '<p class="kf-rd-note">原始产物 · ' + raw + '</p></div></details>';
  }

  /* Overview keeps run evidence as its source of truth. The two compiler
     checks come from the active correctness profile because they are scoped
     to its Numerical Accuracy diagnosis, rather than being new first-level
     Run domains. Missing profile data remains explicitly not collected. */
  function correctnessEvidenceRows(profile, coverage) {
    const collected = (status, label, statusText) => ({ status: status || 'not_collected', label, statusText });
    const structural = profile.compiler?.structuralVerification;
    const numerical = profile.compiler?.numericalValidation;
    const compilerStatus = item => item?.status === 'pass' ? 'available' : item?.status === 'fail' ? 'partial' : 'not_collected';
    const passText = numerical?.status === 'fail' && numerical?.firstDivergentPass
      ? '✕ FIRST DIVERGENCE · ' + numerical.firstDivergentPass
      : numerical?.passed != null && numerical?.total != null
        ? numerical.passed + ' / ' + numerical.total + (numerical.status === 'pass' ? ' PASS' : '') : null;
    return [
      { key: 'golden_compare', entry: coverage.golden_compare },
      { key: 'structural_verification', entry: collected(compilerStatus(structural), '结构校验', structural?.status === 'pass' ? '✓ PASS' : null) },
      { key: 'numerical_validation', entry: collected(compilerStatus(numerical), '数值校验', passText ? (numerical?.status === 'fail' ? passText : '✓ ' + passText) : null) },
      { key: 'tensor_dump', entry: Object.assign({}, coverage.tensor_dump, { label: 'Tensor Checkpoints / Intermediate Tensor' }) },
      { key: 'args_dump', entry: coverage.args_dump },
      { key: 'dependency_graph', entry: coverage.dependency_graph },
      { key: 'runtime_timeline', entry: coverage.runtime_timeline }
    ];
  }

  function domainUnavailablePanel(r, key) {
    const d = getDomainVerdict(r, key), v = DOMAIN_VERDICT[d.verdict] || DOMAIN_VERDICT.unknown;
    const prompt = key === 'correctness' && d.verdict === 'not_evaluated'
      ? '未采集 Golden / Oracle 对比数据。'
      : key === 'execution' && d.verdict === 'not_evaluated'
        ? '本次运行未进入 runtime，未生成 Timeline 或依赖数据。'
        : key === 'performance' && d.verdict === 'not_evaluated'
          ? '本次运行未产生可用的性能证据。'
          : key === 'resources' && d.verdict === 'unknown'
            ? '本次运行未采集运行期资源数据。'
            : '本次运行未采集此专项的下钻证据。';
    const action = key === 'correctness' && d.verdict === 'not_evaluated'
      ? '<button type="button" data-ws-new-run="correctness">运行正确性校验</button>'
      : key === 'resources' && d.verdict === 'unknown'
        ? '<button type="button" data-ws-new-run="resources">重新运行并采集资源数据</button>' : '';
    return '<section class="kf-rd-sec kf-rw-domain-empty"><div class="kf-rd-h">' + DOMAIN_LABEL[key] + '<small>' + v[0] + ' · ' + esc(d.summary) + '</small></div><p class="kf-rd-note">' + prompt + '</p>' + action + '</section>';
  }

  function notEvaluatedEvidencePanel(r, key) {
    const d = getDomainVerdict(r, key), v = DOMAIN_VERDICT[d.verdict] || DOMAIN_VERDICT.unknown;
    const evidence = getEvidenceCoverage(r);
    const available = name => hasEvidence(r, name) ? '可用' : '未采集';
    const content = {
      compilation: {
        title: '编译证据',
        signals: ['IR Validation · ' + available('ir_validation'), 'Pass Dump · ' + available('pass_dump'), 'Device code · 未生成'],
        art: ['编译 Gate', '编译未产出完整产物。', '先完成编译，再生成后续专项证据。']
      },
      correctness: {
        title: '正确性校验 Gate',
        signals: ['Golden Compare · ' + available('golden_compare'), 'Tensor Checkpoints · ' + available('tensor_dump'), 'Output compare · 被阻塞'],
        art: ['校验状态', '没有产出可供对比的输出。', '编译未完成，正确性验证未启动。']
      },
      execution: {
        title: '执行与性能 Gate',
        signals: ['Device dispatch · 未开始', 'Dependency Graph · ' + available('dependency_graph'),
                  'Runtime Timeline · ' + available('runtime_timeline'), 'End-to-end latency · 未测量',
                  'Critical Path · 未构建', 'PMU · ' + available('pmu')],
        art: ['终止阶段', 'LegalizeIndexing', '编译在设备执行前中止，未创建 Task runtime 记录，也没有延迟与关键链可归因。']
      },
      performance: {
        title: '性能测量 Gate',
        signals: ['End-to-end latency · 未测量', 'Critical Path · 未构建', 'PMU · ' + available('pmu')],
        art: ['测量状态', '无运行时采样', '没有可用于归因的延迟、关键链或硬件计数器。']
      },
      resources: {
        title: '资源证据',
        signals: ['Compile-time allocation · 未到达', 'Runtime Resources · ' + available('runtime_resources'), 'Memory Map · ' + available('memory_map')],
        art: ['采集状态', '无运行时资源采样', '未进入设备执行，无法判断 Heap、TensorMap 或 Ringbuffer 使用情况。']
      }
    }[key] || {
      title: DOMAIN_LABEL[key] + '证据',
      signals: [], art: ['状态', '未评估', '当前 Run 未生成该专项的可下钻结果。']
    };
    const isCorrectnessBlocked = key === 'correctness' && d.verdict === 'not_evaluated';
    const action = isCorrectnessBlocked
      ? '<button type="button" data-ws-new-run="correctness">运行正确性校验</button>'
      : key === 'resources' && d.verdict === 'unknown'
        ? '<button type="button" data-ws-new-run="resources">重新运行并采集资源数据</button>' : '';
    /* 有统一 Domain Header 的页签，结论只在头部说一次；这里退回成一个小节标题。
       资源页没有头部，保持原来的「结论 · 摘要」写法不变。 */
    const head = DOMAIN_HEAD_KEYS[key]
      ? '<div class="kf-rd-h">' + esc(content.title) + '</div>'
      : '<div class="kf-rd-h">' + esc(content.title) + '<small>' + v[0] + ' · ' + esc(d.summary) + '</small></div>';
    return '<section class="kf-rd-sec" aria-label="' + esc(DOMAIN_LABEL[key]) + ' evidence">' +
      head +
      '<div class="kf-oi-links">' + content.signals.map(x => '<span class="kf-oi-evidence">' + esc(x) + '</span>').join('') + '</div>' +
      '<div class="kf-rd-art"><b>' + esc(content.art[0]) + '</b><code>' + esc(content.art[1]) + '</code><small>' + esc(content.art[2]) + '</small></div>' + action +
    '</section>';
  }

  function hasEvidence(r, key) {
    const item = getEvidenceCoverage(r)[key];
    return !!(item && (item.status === 'available' || item.status === 'partial'));
  }

  /* ---------- 统一 Domain Header（编译 / 正确性 / 执行）--------------------
     这三个页签此前各有一套自己的结论头：编译页用 kc-summary，正确性页用
     kf-dg-diagnosis-head，执行页只是一行 kf-rd-h 加几个证据 chip。同一个位置
     三种读法，跨页签切换时像三个产品。这里收敛成一个轻量组件：

       Domain 名称 · PASS / WARNING / FAIL / NOT EVALUATED · 一句话结论 · 关键指标

     它只说分析结论。Run 的运行状态（已完成 / 运行中）只出现在顶部 headline，
     两套词汇不混用 —— 一次「已完成」的运行照样可以正确性 FAIL。
     指标全部沿用各页签原有的证据值，这里只换排版，不引入新数据。 */
  const DOMAIN_HEAD_KEYS = { compilation: 1, correctness: 1, execution: 1 };

  /* 一句话结论：页签里原本写在 kf-rd-note / 校验结论上的结论句搬到这里，
     其余用 domain summary —— 它本来就是「这次 Run 在这个域上得出的结论」。 */
  const DOMAIN_CONCLUSION = {
    compilation: { pass: '编译链已完成，未发现阻塞性 IR 约束。' },
    correctness: { pass: '输出与中间 checkpoint 均未发现回归。' }
  };

  /* 编译失败 Run 的 Pass 序列：Domain Header（首个失败 Pass）和失败故事面板
     共用一份，避免同一条序列在两处各写一遍。 */
  const COMPILE_STORY_PASSES = [
    ['Frontend', 'PASS'], ['InlineFunctions', 'PASS'], ['InferLayout', 'PASS'],
    ['LegalizeIndexing', 'FAIL'], ['AllocateMemory', 'NOT RUN'], ['Codegen', 'NOT RUN']
  ];

  /* 指标 = [标签, 值, 结论色]。色只在值真的是结论时才给，未采集一律保持中性。 */
  const evidenceState = (r, key) => hasEvidence(r, key) ? '可用' : '未采集';
  const stateMetric = (label, r, key, okText) => {
    const ok = hasEvidence(r, key);
    return [label, ok ? okText : '未采集', ok ? 'ok' : null];
  };

  function domainHeadMetrics(key, r) {
    if (key === 'compilation') {
      if (isCompileFailureStory(r)) {
        const failed = COMPILE_STORY_PASSES.find(p => p[1] === 'FAIL');
        return [
          ['首个失败 Pass', failed[0], 'bad'],
          ['IR Validation', 'FAIL', 'bad'],
          ['设备代码', '未生成', 'bad']
        ];
      }
      if (getDomainVerdict(r, 'compilation').verdict !== 'pass') {
        return [
          ['IR Validation', evidenceState(r, 'ir_validation')],
          ['Pass Dump', evidenceState(r, 'pass_dump')],
          ['Source Location', evidenceState(r, 'source_location')]
        ];
      }
      return [
        stateMetric('IR Validation', r, 'ir_validation', 'PASS'),
        ['阻塞错误', '0', 'ok'],
        ['设备代码', '已生成', 'ok']
      ];
    }
    if (key === 'correctness') {
      if (isCorrectnessFailureStory(r)) {
        const profile = diagnosisViewForRun(r.id) || {};
        const first = profile.firstDivergence || {};
        const cells = [
          ['首个分歧', (first.kind === 'pass' ? 'Pass · ' : 'Tensor · ') + (first.id || '—'), 'bad'],
          ['Golden Compare', 'FAIL', 'bad']
        ];
        if (profile.result && profile.result.maxAbs != null) {
          cells.splice(1, 0, ['最大绝对误差', profile.result.maxAbs, 'bad']);
        }
        return cells;
      }
      if (getDomainVerdict(r, 'correctness').verdict === 'pass') {
        return [
          ['Golden Compare', 'PASS', 'ok'],
          ['Oracle', '3 / 3', 'ok'],
          ['Checkpoint', '12 / 12 匹配', 'ok']
        ];
      }
      return [
        ['Golden Compare', evidenceState(r, 'golden_compare')],
        ['Tensor Checkpoints', evidenceState(r, 'tensor_dump')],
        ['Args Dump', evidenceState(r, 'args_dump')]
      ];
    }
    /* execution：Performance 的结论已经并进这个域，它在这里只是同一组运行时
       证据上的一条支撑项，不再单独成域。 */
    const metrics = [
      stateMetric('Dependency Graph', r, 'dependency_graph', '完整'),
      stateMetric('Runtime Timeline', r, 'runtime_timeline', '可用')
    ];
    if (isCorrectnessFailureStory(r)) metrics.push(['Task 排序', '#182 → #197 缺少依赖', 'warn']);
    else if (isPerformanceWarningStory(r) || isValidatedOptimizationStory(r)) metrics.push(['Task 排序', '#182 → #197 已建立', 'warn']);
    return metrics;
  }

  /* 正确性页签：诊断 profile 存在时，把原来的 Gate（参考基准 / 容差 / 语义
     校验）和结果指标（输出 / 误差 / 重复运行）合成同一组列，与 Domain Header
     融为顶部唯一的 summary strip —— 诊断解释交给下方工作区，不在这里展开。 */
  function domainOverviewMetrics(key, r) {
    if (key === 'correctness') {
      /* 注册表对未知 run 会兜底返回失败样例，只有 profile 确实属于这个 run
         时才用它的数据，否则维持各页签通用的指标行。 */
      const profile = diagnosisViewForRun(r.id);
      if (profile && profile.runId === String(r.id) && profile.result) {
        const R = profile.result, cells = [];
        if (R.output != null) cells.push(['输出', R.output]);
        if (profile.reference) cells.push(['参考基准', profile.reference.source || '不可用']);
        if (R.maxAbs != null) cells.push(['最大绝对误差', R.maxAbs, 'bad']);
        if (profile.tolerance) cells.push(['容差', 'rtol ' + (profile.tolerance.rtol || '—') + ' · atol ' + (profile.tolerance.atol || '—')]);
        if (R.maxRel != null) cells.push(['最大相对误差', R.maxRel]);
        if (profile.repeatability) cells.push(['重复运行', profile.repeatability.summary || '—', profile.repeatability.stable ? 'ok' : 'warn']);
        const numerical = profile.compiler?.numericalValidation;
        if (numerical && numerical.total != null) {
          cells.push(['编译语义校验', numerical.status === 'pass'
            ? numerical.passed + ' / ' + numerical.total + ' 匹配'
            : String(numerical.status || 'unknown').toUpperCase(), numerical.status === 'pass' ? 'ok' : numerical.status === 'fail' ? 'bad' : null]);
        }
        if (cells.length) return cells;
      }
    }
    return domainHeadMetrics(key, r);
  }

  function domainHead(key, r) {
    if (!DOMAIN_HEAD_KEYS[key] || !r) return '';
    const d = getDomainVerdict(r, key);
    const v = DOMAIN_VERDICT[d.verdict] || DOMAIN_VERDICT.unknown;
    const byVerdict = DOMAIN_CONCLUSION[key] || {};
    const said = (d.verdict === 'pass' && byVerdict.pass) || d.summary || '无可用结论';
    const profile = key === 'correctness' ? diagnosisViewForRun(r.id) : null;
    const note = (profile && profile.runId === String(r.id) && profile.compiler?.numericalValidation?.summary) || said;
    const icon = { ok: '✓', warn: '!', bad: '×', idle: '○' }[v[1]] || '○';
    return '<header class="kf-dh is-' + v[1] + '" aria-label="' + esc(DOMAIN_LABEL[key]) + '域结论">' +
      '<div class="kf-dh-verdict" aria-hidden="true"><span>' + esc(DOMAIN_LABEL[key]) + '</span><b>' + esc(v[0]) + '</b><i>' + icon + '</i></div>' +
      '<div class="kf-dh-metrics">' + domainOverviewMetrics(key, r).map(m =>
        '<span><i>' + esc(m[0]) + '</i><b' + (m[2] ? ' class="is-' + m[2] + '"' : '') + '>' + esc(m[1]) + '</b></span>').join('') +
      '</div>' +
      '<p class="kf-dh-note">' + esc(note) + '</p>' +
    '</header>';
  }

  function validatedPerformancePanel(r) {
    const m = getRunModel(r), metrics = m.compareMetrics || {};
    return '<section class="kf-rd-sec" aria-label="已验证的性能结果">' +
      '<div class="kf-rd-h">性能<small>PASS · 达到延迟目标</small></div>' +
      '<div class="kf-oi-links"><span class="kf-oi-evidence">Latency · ' + esc(metrics.latency || '1.36 ms') + '</span>' +
        '<span class="kf-oi-evidence">目标 · &lt; 1.50 ms</span><span class="kf-oi-evidence">Critical Path · ' + esc(metrics.criticalPath || '0.98 ms') + '</span>' +
        '<span class="kf-oi-evidence">Task #182 · ' + esc(metrics.task182 || '147 µs') + '</span></div>' +
      '<div class="kf-rd-art"><b>搬运证据</b><code>256 B → ' + esc(metrics.transferWidth || '512 B') + '</code><small>MTE stall 31% → ' + esc(metrics.mteStall || '17%') + ' · 正确性未改变</small></div>' +
    '</section>';
  }

  function resourcesSummaryPanel(r) {
    const d = getDomainVerdict(r, 'resources'), metrics = getRunModel(r).compareMetrics || {};
    const knownPeak = metrics.l0bPeak || (isPerformanceWarningStory(r) ? '81%' : '');
    return '<section class="kf-rd-sec" aria-label="资源结论">' +
      '<div class="kf-rd-h">资源<small>' + esc(d.verdict.toUpperCase()) + ' · ' + esc(d.summary) + '</small></div>' +
      '<div class="kf-oi-links">' + (knownPeak ? '<span class="kf-oi-evidence">L0B peak · ' + esc(knownPeak) + '</span>' : '') +
        (isPerformanceWarningStory(r) ? '<span class="kf-oi-evidence">UB peak · 61%</span>' : '') +
        '<span class="kf-oi-evidence">容量越界 · 无</span><span class="kf-oi-evidence">溢出 · 无</span></div>' +
      '<p class="kf-rd-note">未检测到容量越界。' + (hasEvidence(r, 'scope_stats') ? '' : ' Runtime Scope Stats 未采集，不展示 Heap、TensorMap 或 Ringbuffer 数值。') + '</p>' +
    '</section>';
  }

  function compilationFailureStoryPanel(r) {
    const passes = COMPILE_STORY_PASSES;
    return '<section class="kf-rd-sec" aria-label="编译失败已定位">' +
      '<div class="kf-rd-h">Pass 序列<small>Mock 编译器证据 · 已确认失败 · 首个失败 Pass</small></div>' +
      '<div class="kf-oi-links">' + passes.map(p =>
        '<button type="button" class="kf-oi-link" data-ws-select-kind="pass" data-ws-select-id="' + esc(p[0]) + '" data-ws-source="compilation"' +
          (p[0] === 'LegalizeIndexing' ? ' aria-pressed="true"' : '') + '>' + esc(p[0]) + ' · ' + esc(p[1]) + '</button>').join('') +
      '</div>' +
      '<div class="kf-rd-h">LegalizeIndexing 处的 IR 变化<small>上一个 Pass 合法 → 当前 lowering 产生了非法的 GM store 地址形式</small></div>' +
      '<div class="kf-rd-art"><b>修改前</b><code>tensor.write(cache, value, index = dynamic_index)</code><b>修改后</b><code>gm.store(base + cast_i64(dynamic_index) * stride, value)</code><small>约束冲突 · index 位宽 / 地址计算不一致</small></div>' +
      '<div class="kf-oi-actions"><button type="button" data-ws-select-kind="pass" data-ws-select-id="LegalizeIndexing" data-ws-source="compilation">查看失败 Pass</button></div>' +
    '</section>';
  }

  /* ============================================================
     Correctness diagnosis view — Run #106

     A cross-layer debugger that keeps the semantic computation graph
     on screen while drilling into runtime execution:

       semantic op  →  tensor  →  runtime task  →  dependency / timeline

     Op is the only node. Tensor is an edge label, never a card. Exactly
     one edge carries the error weight: the first divergence. Everything
     downstream of it is a weaker "propagated" state.

     Evidence model (kept honest):
       reference  PyTorch golden checkpoint
       actual     Args Dump of this run
     A tensor shows MATCH / MISMATCH only when both sides exist. Tensors
     with no reference are "no reference", never PASS.

     Values come from the selected diagnostic profile; replacing the source
     with dfx_outputs remains a data-layer change.
     ============================================================ */
  /* Legacy inline DG fixture moved to correctness-diagnostic-data.js.
     Keep a single selected profile reference here so all existing graph,
     runtime expansion, timeline, tooltip and selection interactions remain
     renderer-only behaviour. */
  let DG = null;
  /* Profile data lives in correctness-diagnostic-data.js. */

  const DG_STATE_TONE = { match: 'ok', first: 'bad', propagated: 'warn', unchecked: 'dim' };

  function diagnosisViewForRun(runId) {
    const profile = window.PTO_CORRECTNESS_DIAGNOSTICS?.get?.(runId);
    if (!profile) return null;
    /* The visual grammar predates the shared profile. This adapter keeps that
       grammar stable while making profile fields the only source of content. */
    return Object.assign({}, profile, {
      ops: profile.semanticGraph?.ops || [],
      edges: profile.semanticGraph?.edges || [],
      tasks: profile.runtime?.tasks || [],
      runtimeChain: profile.runtime?.runtimeChain || [],
      timeline: profile.runtime?.timeline || { base: 0, span: 1, tickStep: 1, rows: [] },
      semantics: profile.compiler?.numericalValidation || { status: 'not_collected', passed: 0, total: 0 },
      cause: {
        label: profile.diagnosis?.label || '诊断结论',
        text: profile.diagnosis?.summary || '暂无可归因结论',
        evidence: profile.diagnosis?.evidence || []
      }
    });
  }

  /* 选择状态：跨 tab 切换保留，换 Run 时重置 */
  const dg = { run: null, sel: null, runtime: false, view: 'tasks' };
  let dgRO = null;

  /* DG 只有在 Correctness 页签渲染后才会被 dgReset 载入。用户直接进执行页就
     hover 对象时 DG 仍是 null，这里要兜底，否则 find 会抛 TypeError。 */
  function dgOp(id) { return DG ? DG.ops.find(o => o.id === id) || null : null; }
  function dgTensor(id) {
    if (!DG) return null;
    if (id === '37') id = 'attention_out';            // F106 仍按旧的内部编号选
    if (id === '41') id = 'out';
    return DG.tensors.find(t => t.id === id) || null;
  }
  function dgTask(id) { return DG ? DG.tasks.find(t => t.id === String(id)) || null : null; }
  function dgStateText(state) {
    return state === 'match' ? '匹配'
      : state === 'first' ? '不一致'
      : state === 'propagated' ? '不一致 · 下游传播' : '';
  }
  /* tensor ↔ runtime task 的映射，全部从 runtimeChain 推出来，不额外存字段 */
  function dgRuntimeProducer(tid) {
    const e = DG.runtimeChain.find(x => x.via === tid);
    return e ? dgTask(e.from) : null;
  }
  /* 这一行的色调只由角色决定，避免和 role 两处各写一份判断 */
  function dgTimelineTone(t) {
    return t.role === 'producer' ? 'focus'
      : t.role === 'suspicious' ? 'risk'
      : t.role === 'consumer' ? 'dim' : 'calm';
  }
  /* 时间线里的「访问」：共享 buffer 优先，其次张量读写 */
  function dgAccessText(t) {
    const p = [];
    if (t.shared) p.push((t.shared.op === 'read' ? '读' : '写') + '共享 buffer ' + t.shared.buffer);
    if (t.writes.length) p.push('写 ' + t.writes.join(' / '));
    else if (!t.shared && t.reads.length) p.push('读 ' + t.reads.join(' / '));
    return p.length ? p.join(' · ') : '—';
  }
  function dgRoleText(t) {
    if (t.role === 'producer') return t.label + ' · 产生 ' + t.writes.join(' / ');
    if (t.role === 'consumer') return t.label + ' · 消费 ' + t.reads.join(' / ');
    if (t.role === 'suspicious') return t.label + ' · 覆盖共享 buffer ' + (t.shared ? t.shared.buffer : '—');
    return t.label;
  }

  function dgReset(runId) {
    if (dg.run === runId) return;
    DG = diagnosisViewForRun(runId);
    dg.run = runId; dg.sel = null; dg.runtime = false; dg.view = 'tasks';
  }

  /* ---------- 图的几何：op / tensor 端口 → SVG 路径 ---------- */
  function dgPath(a, b, laneX) {
    const my = (a.y + b.y) / 2;
    if (laneX == null) return 'M' + a.x + ' ' + a.y + ' C' + a.x + ' ' + my + ', ' + b.x + ' ' + my + ', ' + b.x + ' ' + b.y;
    return 'M' + a.x + ' ' + a.y + ' C' + laneX + ' ' + my + ', ' + laneX + ' ' + my + ', ' + b.x + ' ' + b.y;
  }

  function dgDraw(root) {
    const canvas = root.querySelector('[data-dg-canvas]');
    if (!canvas) return;
    const svg = canvas.querySelector('[data-dg-edges]');
    if (!svg) return;
    const box = canvas.getBoundingClientRect();
    if (!box.width) return;

    const nodes = {};
    canvas.querySelectorAll('[data-dg-op]').forEach(el => { nodes['op:' + el.dataset.dgOp] = el; });
    canvas.querySelectorAll('[data-dg-tlabel]').forEach(el => { nodes['t:' + el.dataset.dgTlabel] = el; });
    const port = (el, side) => {
      const r = el.getBoundingClientRect();
      return { x: +(r.left + r.width / 2 - box.left).toFixed(1), y: +((side === 'top' ? r.top : r.bottom) - box.top).toFixed(1) };
    };

    const out = [];
    DG.edges.forEach(e => {
      const src = nodes['op:' + e.from] || nodes['t:' + e.from];
      if (!src) return;
      const laneX = e.lane === 'left' ? 14 : e.lane === 'right' ? box.width - 14 : null;
      const tone = e.via ? (dgTensor(e.via)?.state || 'unchecked') : 'unchecked';
      const cls = ' class="is-' + (DG_STATE_TONE[tone] || 'dim') + '"';
      const start = port(src, 'bottom');
      const mid = e.via ? nodes['t:' + e.via] : null;
      if (mid) {
        const top = port(mid, 'top'), bot = port(mid, 'bottom');
        out.push('<path' + cls + ' d="' + dgPath(start, top, null) + '"/>');
        if (e.to) {
          const dst = nodes['op:' + e.to];
          if (dst) out.push('<path' + cls + ' d="' + dgPath(bot, port(dst, 'top'), laneX) + '"/>');
        }
      } else if (e.to) {
        const dst = nodes['op:' + e.to];
        if (dst) out.push('<path' + cls + ' d="' + dgPath(start, port(dst, 'top'), laneX) + '"/>');
      }
    });
    svg.innerHTML = out.join('');
  }

  /* ---------- 运行时展开区 ---------- */
  function dgRoleChip(t) {
    const tone = t.role === 'producer' ? ' is-bad' : t.role === 'suspicious' ? ' is-warn' : '';
    return '<em class="kf-dg-rt-role' + tone + '">' + esc(t.label) + '</em>';
  }

  /* Task 是这一层的节点，比 Semantic Op 更小更紧凑；本轮不显示 Kernel。 */
  function dgRuntimeTask(t) {
    const on = (dg.sel && dg.sel.kind === 'task' && dg.sel.id === t.id) ||
      (st.executionEntry?.from === 'investigation' && st.executionEntry.taskIds?.includes(t.id));
    return '<button type="button" class="kf-dg-rt-task is-' + t.role + (on ? ' is-sel' : '') +
      '" data-dg-select-kind="task" data-dg-select-id="' + t.id + '"' +
      ' aria-label="' + esc('Task #' + t.id + ' · ' + t.note) + '">' +
      '<code>Task #' + t.id + '</code>' +
      dgRoleChip(t) + '</button>';
  }

  /* Task 之间的 tensor 是边标签，不是节点：节点列居中，标签挂在竖直连线右侧 */
  function dgRuntimeLink(e) {
    const tone = e.via ? (DG_STATE_TONE[dgTensor(e.via)?.state] || 'dim') : 'dim';
    return '<span class="kf-dg-rt-link is-' + tone + '">' +
      '<i class="kf-dg-rt-slug" aria-hidden="true"></i>' +
      '<small>' + esc(e.label) + '</small></span>';
  }

  /* 单列链路：直接嵌在 semantic graph 的数据流里，不是左右分栏。 */
  function dgRuntimeTasks() {
    const chain = DG.runtimeChain || [];
    const expansion = DG.runtime?.expansion || {};
    const ordered = [expansion.entryTask].concat(chain.map(e => e.to)).filter(Boolean)
      .filter((id, index, all) => all.indexOf(id) === index).map(dgTask).filter(Boolean);
    return '<div class="kf-dg-rt-chain">' +
      '<span class="kf-dg-rt-stem is-entry" aria-hidden="true"></span>' +
      ordered.map((task, index) => dgRuntimeTask(task) + (chain[index] ? dgRuntimeLink(chain[index]) : '')).join('') +
      '<span class="kf-dg-rt-stem is-exit" aria-hidden="true"></span>' +
      '</div>';
  }

  /* 每行左侧固定列 + 右侧 time bar。core 与起止时间不再隐藏。 */
  function dgTimelineRow(r, i) {
    const T = DG.timeline, ov = T.overlap, t = dgTask(r.task);
    const pct = v => (v - T.base) / T.span * 100;
    const on = (dg.sel && dg.sel.kind === 'task' && dg.sel.id === t.id) ||
      (st.executionEntry?.from === 'investigation' && st.executionEntry.taskIds?.includes(t.id));
    return '<button type="button" class="kf-dg-tlx-row is-' + dgTimelineTone(t) + (on ? ' is-sel' : '') + '"' +
      ' style="grid-row:' + (3 + i) + '" data-dg-select-kind="task" data-dg-select-id="' + t.id + '"' +
      ' aria-label="' + esc('Task #' + t.id + ' · ' + t.core + ' · ' + r.s + ' → ' + r.e + ' μs · ' + t.note) + '">' +
      '<code class="kf-dg-tlx-id">#' + t.id + '</code>' +
      '<em class="kf-dg-tlx-role">' + esc(t.label) + '</em>' +
      '<span class="kf-dg-tlx-core">' + esc(t.core) + '</span>' +
      '<span class="kf-dg-tlx-range">' + r.s + ' → ' + r.e + '<i> μs</i></span>' +
      '<span class="kf-dg-tlx-dur">' + (r.e - r.s) + ' μs</span>' +
      '<span class="kf-dg-tlx-track">' +
        '<i class="kf-dg-tlx-bar" style="left:' + pct(r.s) + '%;width:' +
          ((r.e - r.s) / T.span * 100) + '%"></i>' +
      '</span></button>';
  }

  function dgTimelineAxis() {
    const T = DG.timeline;
    const n = Math.round(T.span / T.tickStep);
    let ticks = '';
    for (let i = 0; i <= n; i++) ticks += '<i style="left:' + (i / n * 100) + '%">' + (T.base + i * T.tickStep) + '</i>';
    return '<div class="kf-dg-tlx-axis" style="grid-row:2"><span></span><span></span><span></span><span></span><span></span>' +
      '<span class="kf-dg-tlx-ticks">' + ticks + '</span></div>';
  }

  function dgRuntimeTimeline() {
    const T = DG.timeline, ov = T.overlap, a = dgTask(ov.a), b = dgTask(ov.b);
    const ra = T.rows.find(r => r.task === a.id), rb = T.rows.find(r => r.task === b.id);
    const dur = ov.to - ov.from;
    const n = Math.round(T.span / T.tickStep);
    const pct = v => (v - T.base) / T.span * 100;

    /* 所有子项都显式给行号：跨行标注带是显式定位的，会先占住它那两行，
       若其余子项走自动排布就会被挤到后面的空行上、对应行高塌成 0。 */
    const head = '<div class="kf-dg-tlx-head" style="grid-row:1"><span>任务</span><span>角色</span><span>核心</span>' +
      '<span>时间区间 (μs)</span><span>耗时</span><span>时间线</span></div>';

    /* 重叠带跨 #182 / #197 两行：靠 grid-row 定位，高度自动跟随行高与行距，
       不写死像素；画在 bar 之上，否则整段落在两条 bar 内部等于没画。 */
    const ra0 = T.rows.findIndex(r => r.task === ov.a), rb0 = T.rows.findIndex(r => r.task === ov.b);
    const lo = Math.min(ra0, rb0), hi = Math.max(ra0, rb0);
    const span = '<div class="kf-dg-tlx-span" style="grid-row:' + (3 + lo) + ' / ' + (4 + hi) + '">' +
      '<span class="kf-dg-tlx-band" style="left:' + pct(ov.from) + '%;width:' +
      ((ov.to - ov.from) / T.span * 100) + '%"></span></div>';

    /* 尺寸标注与上面那条跨行重叠带共用同一 x 区间，视觉上直接对应 */
    const ovl = '<div class="kf-dg-tlx-ovlrow" style="grid-row:' + (3 + T.rows.length) + '">' + '<span></span>'.repeat(5) +
      '<span class="kf-dg-tlx-ovlcell">' +
        '<button type="button" class="kf-dg-tlx-ovl" data-dg-select-kind="timeline" data-dg-select-id="overlap_182_197"' +
        ' style="left:' + pct(ov.from) + '%;width:' + ((ov.to - ov.from) / T.span * 100) + '%">' +
          '<b>可疑重叠 ' + dur + ' μs</b><em>' + ov.from + ' → ' + ov.to + ' μs</em></button>' +
      '</span></div>';

    const sum = '<div class="kf-dg-tlx-sum">' +
      '<div class="kf-dg-tlx-sum-h"><span>可疑重叠</span><b>' + dur + ' μs</b></div>' +
      '<p>#' + b.id + ' 在 #' + a.id + ' 读完共享 buffer ' + T.buffer + ' 之前 ' + dur +
        ' μs 就开始写入。该重叠与「重复运行不稳定」的现象一致，提示存在排序 / buffer 覆盖写入风险。</p>' +
      '<ul>' +
        '<li>Task #' + a.id + '（' + a.label + '）· ' + a.core + ' · ' + ra.s + ' → ' + ra.e +
          ' μs，重叠窗口内正在读共享 buffer ' + T.buffer + '</li>' +
        '<li>Task #' + b.id + '（' + b.label + '）· ' + b.core + ' · ' + rb.s + ' → ' + rb.e +
          ' μs，提前写同一块 buffer</li>' +
        '<li>重叠窗口 ' + ov.from + ' → ' + ov.to + ' μs · ' + dur + ' μs</li>' +
      '</ul></div>';

    return '<div class="kf-dg-tlx" style="--kf-tlx-n:' + n + '">' + head + dgTimelineAxis() +
      T.rows.map(dgTimelineRow).join('') + span + ovl + '</div>' + sum;
  }

  function dgRuntime() {
    if (!dg.runtime) return '';
    const divergence = DG.firstDivergence || {};
    const chain = DG.runtimeChain || [];
    const focused = chain.find(e => e.via === divergence.id) || chain[0];
    const views = [['tasks', '任务'], ['timeline', '时间线']];
    /* 只有两个视图，任何未知的 dg.view 都回落到任务，标签高亮不会落空 */
    const view = dg.view === 'timeline' ? 'timeline' : 'tasks';
    const body = view === 'timeline' ? dgRuntimeTimeline() : dgRuntimeTasks();
    return '<div class="kf-dg-rt" data-dg-rt>' +
      '<div class="kf-dg-rt-head">' +
        '<span class="kf-dg-rt-title">运行时执行</span>' +
        '<span class="kf-dg-rt-sub">' + esc((divergence.id || 'runtime') + (focused ? ' · Task #' + focused.from + ' → Task #' + focused.to : '')) + '</span>' +
        '<span class="kf-dg-rt-views">' + views.map(v =>
          '<button type="button" class="' + (view === v[0] ? 'is-on' : '') + '" data-dg-view="' + v[0] + '">' + v[1] + '</button>').join('') + '</span>' +
        '<button type="button" class="kf-dg-rt-x" data-dg-collapse title="收起 Runtime">收起</button>' +
      '</div>' +
      '<div class="kf-dg-rt-body" data-dg-rtbody="' + view + '">' + body + '</div>' +
    '</div>';
  }

  /* ---------- 语义图 ---------- */
  function dgOpNode(id) {
    const o = dgOp(id);
    const on = dg.sel && dg.sel.kind === 'op' && dg.sel.id === id;
    return '<button type="button" class="kf-dg-op' + (on ? ' is-sel' : '') + '" data-dg-op="' + o.id + '"' +
      ' data-dg-select-kind="op" data-dg-select-id="' + o.id + '"' +
      ' title="' + esc(o.name + ' · ' + o.be + ' · ' + o.src) + '">' + esc(o.name) + '</button>';
  }

  function dgTensorLabel(id, opts) {
    const t = dgTensor(id);
    if (!t) return '';
    opts = opts || {};
    const tone = DG_STATE_TONE[t.state];
    const on = dg.sel && dg.sel.kind === 'tensor' && dg.sel.id === t.id;
    const state = t.state === 'first' ? '首个分歧'
      : t.state === 'propagated' ? '下游传播'
      : t.state === 'match' ? '✓' : '';
    return '<button type="button" class="kf-dg-tensor is-' + tone + (on ? ' is-sel' : '') + (opts.source ? ' is-source' : '') + '"' +
      ' data-dg-tlabel="' + t.id + '" data-dg-select-kind="tensor" data-dg-select-id="' + t.id + '"' +
      ' title="' + esc(t.name + ' · ' + t.tid + ' · ' + t.shape + ' · ' + t.dtype) + '">' +
      '<span class="kf-dg-tensor-name">' + esc(t.name) + '</span>' +
      (state ? '<em class="kf-dg-tensor-mark">' + state + '</em>' : '') +
      (t.state === 'first' ? '<em class="kf-dg-tensor-diff">最大绝对误差 ' + t.maxAbs + '</em>' : '') +
    '</button>';
  }

  function dgCanvas() {
    const stage = (name, body) => '<div class="kf-dg-stage" data-dg-stage="' + name + '">' + body + '</div>';

    return '<div class="kf-dg-canvas" data-dg-canvas>' +
      '<svg class="kf-dg-edges" data-dg-edges aria-hidden="true"></svg>' +

      stage('source', '<div class="kf-dg-slot is-center">' + dgTensorLabel('hidden_states', { source: true }) + '</div>') +

      stage('proj', '<div class="kf-dg-slot">' + dgOpNode('q_proj') + '</div>' +
                    '<div class="kf-dg-slot">' + dgOpNode('k_proj') + '</div>' +
                    '<div class="kf-dg-slot">' + dgOpNode('v_proj') + '</div>') +
      stage('link-proj', dgTensorLabel('q') + dgTensorLabel('k') + dgTensorLabel('v')) +

      stage('rope', '<div class="kf-dg-slot">' + dgOpNode('rope_q') + '</div>' +
                    '<div class="kf-dg-slot">' + dgOpNode('rope_k') + '</div>' +
                    '<div class="kf-dg-slot is-empty"></div>') +
      stage('link-rope', dgTensorLabel('q_rotated') + dgTensorLabel('k_rotated') + '<span></span>') +

      stage('score', '<div class="kf-dg-slot is-center">' + dgOpNode('score') + '</div>') +
      stage('link-score', '<div class="kf-dg-slot is-center">' + dgTensorLabel('attn_score') + '</div>') +

      stage('softmax', '<div class="kf-dg-slot is-center">' + dgOpNode('softmax') + '</div>') +
      stage('link-softmax', '<div class="kf-dg-slot is-center">' + dgTensorLabel('softmax_p') + '</div>') +

      stage('attention', '<div class="kf-dg-slot is-center">' + dgOpNode('attention') + '</div>') +
      stage('link-attention', '<div class="kf-dg-slot is-center">' + dgTensorLabel('attention_out') + '</div>') +

      dgRuntime() +

      stage('out_proj', '<div class="kf-dg-slot is-center">' + dgOpNode('out_proj') + '</div>') +
      stage('link-out_proj', '<div class="kf-dg-slot is-center">' + dgTensorLabel('projected_out') + '</div>') +

      stage('residual', '<div class="kf-dg-slot is-center">' + dgOpNode('residual') + '</div>') +
      stage('link-residual', '<div class="kf-dg-slot is-center">' + dgTensorLabel('out') + '</div>') +
    '</div>';
  }

  /* ---------- 结果 / 诊断路径 ----------
     结果指标（输出 / 误差 / 重复运行）与诊断 Gate（参考基准 / 容差 / 语义
     校验）都已并入统一 Domain Header 的 summary strip（见 domainOverviewMetrics）。
     诊断解释（现象 / 失效模式 / 根因域）不再在顶部叠大卡片，交给下方
     诊断工作区（dgTrail / dgCanvas / dgCause）表达。 */
  function dgHasCompilerDivergence() {
    const numerical = DG?.compiler?.numericalValidation || {};
    return numerical.status === 'fail' && numerical.firstDivergentPass != null;
  }

  function dgCompilerDivergence() {
    const numerical = DG.compiler.numericalValidation;
    return '<section class="kf-rd-art" aria-label="首个数值分歧 Pass">' +
      '<b>首个数值分歧 Pass</b><code>' + esc(numerical.firstDivergentPass) + '</code>' +
      '<small>该 Pass 后的 Host IR execution 首次偏离 Golden；后续 Pass 持续 MISMATCH。</small>' +
    '</section>';
  }

  function dgTrail() {
    const items = (DG.firstDivergence && DG.firstDivergence.trail) || [];
    return '<nav class="kf-dg-trail" aria-label="诊断路径">' +
      items.map(it => '<button type="button" class="' + (dg.sel && dg.sel.kind === it.kind && dg.sel.id === it.id ? 'is-on' : '') +
        '" data-dg-select-kind="' + it.kind + '" data-dg-select-id="' + it.id + '">' + esc(it.label) + '</button>').join('<i>›</i>') +
      '</nav>';
  }

  function dgCause() {
    const C = DG.cause;
    const numerical = DG.compiler?.numericalValidation || {};
    const action = numerical.status === 'fail' && numerical.firstDivergentPass != null
      ? '<div class="kf-oi-actions"><button type="button" data-dg-route="compilation">在「编译」中查看首个数值分歧 Pass</button></div>'
      : DG.diagnosis?.category === 'runtime_data_error'
        ? '<div class="kf-oi-actions"><button type="button" data-dg-route="execution">在「执行」中查看排序证据</button></div>'
        : '';
    return '<section class="kf-dg-cause" aria-label="可能原因">' +
      '<div><span>' + esc(C.label) + '</span><b>' + esc(C.text) + '</b>' + action + '</div>' +
      '<ul>' + C.evidence.map(e => '<li>' + esc(e) + '</li>').join('') + '</ul>' +
    '</section>';
  }

  /* 对象详情只保留 hover tooltip，画布右侧不再挂常驻容器 */
  function correctnessDiagnosisPanel() {
    if (dgHasCompilerDivergence()) {
      return '<section class="kf-rd-sec kf-dg" aria-label="编译语义正确性诊断">' +
        dgCompilerDivergence() +
        dgCause() +
      '</section>';
    }
    return '<section class="kf-rd-sec kf-dg" aria-label="正确性诊断">' +
      dgTrail() +
      '<div class="kf-dg-layout">' +
        dgCanvas() +
      '</div>' +
      dgCause() +
    '</section>';
  }

  function dgRender() {
    const panel = $('#runTabPanel');
    if (!panel || st.tab !== 'correctness') return;
    hideObjectTooltip();
    panel.innerHTML = correctnessDiagnosisPanel();
    dgDraw(panel);
  }

  /* selection 走同一条 selectObject 链路，只是多刷一次图 */
  function dgSelect(kind, id) {
    if (kind === 'tensor' && id === DG.firstDivergence?.id) dg.runtime = true;
    if (kind === 'task' && DG.tasks.some(task => task.id === String(id))) dg.runtime = true;
    dg.sel = { kind: kind, id: String(id) };
    selectObject({ kind: kind, id: String(id), sourceTab: 'correctness' });
    dgRender();
  }

  function correctnessExecutionEntry() {
    const profile = DG || diagnosisViewForRun(st.run);
    const divergence = profile?.firstDivergence || {};
    const overlap = profile?.runtime?.timeline?.overlap || null;
    return {
      from: 'correctness', runId: st.run,
      finding: '数值精度 · ' + (divergence.id || 'output') + ' 首个分歧',
      suspectedCause: '运行时排序',
      focusTask: overlap?.a || divergence.trail?.find(x => x.kind === 'task')?.id || null,
      timeline: profile?.runtime?.timeline || null
    };
  }

  function correctnessCompilationEntry() {
    const profile = DG || diagnosisViewForRun(st.run);
    const numerical = profile?.compiler?.numericalValidation || {};
    return {
      from: 'correctness', runId: st.run,
      finding: '数值精度 · 输出不一致',
      intent: '正在定位编译语义分歧',
      numericalValidation: numerical
    };
  }

  function focusCorrectnessExecutionEvidence(entry) {
    const overlap = entry?.timeline?.overlap;
    if (!overlap) return;
    window.PTO_TIMELINE?.focusRange?.({ from: overlap.from, to: overlap.to });
    window.PTO_TIMELINE?.clearHighlight?.();
    $('#runTimeline', els.detail)?.scrollIntoView({ block: 'start', behavior: 'smooth' });
  }

  function routeCorrectnessEvidence(route) {
    if (route === 'execution') {
      const entry = correctnessExecutionEntry();
      dg.runtime = true; dg.view = 'timeline';
      /* 跨页尽量保留对象选择：把聚焦的 Task 记进 st.selection，执行页的
         Context Banner 与排序证据里的对象 chip 都会读到它，保持连续性。 */
      if (entry.focusTask) {
        dg.sel = { kind: 'task', id: String(entry.focusTask) };
        selectObject({ kind: 'task', id: String(entry.focusTask), sourceTab: 'correctness' });
      }
      toTab('execution', { executionEntry: entry });
      requestAnimationFrame(() => focusCorrectnessExecutionEvidence(entry));
      return;
    }
    if (route === 'compilation') {
      toTab('compilation', { compilationEntry: correctnessCompilationEntry() });
    }
  }

  function dgWire() {
    if (!els.detail) return;
    els.detail.addEventListener('click', e => {
      const route = e.target.closest('[data-dg-route]');
      if (route) { routeCorrectnessEvidence(route.dataset.dgRoute); return; }
      const pick = e.target.closest('[data-dg-select-kind]');
      if (pick) { dgSelect(pick.dataset.dgSelectKind, pick.dataset.dgSelectId); return; }
      const view = e.target.closest('[data-dg-view]');
      if (view) { dg.view = view.dataset.dgView; dgRender(); return; }
      const rt = e.target.closest('[data-dg-collapse], [data-dg-expand]');
      // data-dg-expand 没有值，dataset 拿到的是空字符串，不能用 !! 判断
      if (rt) { dg.runtime = rt.hasAttribute('data-dg-expand'); dgRender(); return; }
    });
  }

  function dgWatch() {
    const canvas = $('#runTabPanel [data-dg-canvas]');
    if (dgRO) { dgRO.disconnect(); dgRO = null; }
    if (!canvas || typeof ResizeObserver !== 'function') return;
    dgRO = new ResizeObserver(() => { const r = $('#runTabPanel'); if (r) dgDraw(r); });
    dgRO.observe(canvas);
  }

  /* 统一跨域 Context Banner：来源 → 意图 → 当前聚焦对象。
     从一个 Domain 跳到另一个时给读者一个连续的锚点，三行固定读法：
       来自「X」 / 结论 → 正在做什么 / 当前聚焦 · 对象 */
  function contextBanner(from, intentFrom, intentTo, focus) {
    return '<section class="kf-cv-context" aria-label="跨域上下文">' +
      '<span class="kf-cv-context-from">来自「' + esc(from) + '」</span>' +
      '<div class="kf-cv-context-intent"><b>' + esc(intentFrom) + '</b><i>→</i><b>' + esc(intentTo) + '</b></div>' +
      (focus ? '<p class="kf-cv-context-focus">当前聚焦 · <b>' + esc(focus) + '</b></p>' : '') +
    '</section>';
  }

  function correctnessExecutionStoryPanel() {
    const entry = st.executionEntry;
    const enteredFromCorrectness = entry?.from === 'correctness' || entry?.from === 'investigation';
    const profileTimeline = DG?.timeline?.overlap ?
      '<div class="kf-dg-rt-body" data-dg-rtbody="timeline">' + dgRuntimeTimeline() + '</div>' : '';
    const focusedTask = enteredFromCorrectness && entry.focusTask ? 'Task #' + entry.focusTask : '';
    const selTaskId = st.selection && st.selection.kind === 'task' ? String(st.selection.id) : null;
    return '<section class="kf-rd-sec" aria-label="排序证据">' +
      (enteredFromCorrectness ? contextBanner(entry.from === 'investigation' ? '调查地图' : '正确性', entry.finding, '正在验证' + entry.suspectedCause, focusedTask) : '') +
      '<div class="kf-rd-h">排序证据<small>Task 依赖与执行时间线（Timeline）</small></div>' +
      '<div class="kf-oi-links">' +
        objectButton('task', '182', 'Task #182 · 读取方', 'execution', selTaskId === '182') +
        objectButton('buffer', 'B2', 'B2 · 共享 buffer', 'execution') +
        objectButton('task', '197', 'Task #197 · 写入方 / 覆盖', 'execution', selTaskId === '197') +
      '</div>' +
      '<div class="kf-rd-art"><b>缺少预期排序</b><code>Task #182  →  Task #197</code><small>#182 仍在读取 · #197 开始覆盖写入 · 写入方在读取方完成前开始写入</small></div>' +
      profileTimeline +
      '<div class="kf-oi-actions"><button type="button" data-ws-select-kind="dependency" data-ws-select-id="missing_182_197" data-ws-source="execution">查看缺失依赖</button></div>' +
    '</section>';
  }

  function performanceWarningStoryPanel() {
    const chain = ['Task #141', 'Task #178', 'Task #182 · Long pole', 'Task #196', 'Task #201'];
    return '<section class="kf-rd-sec" aria-label="性能关键路径">' +
      '<div class="kf-rd-h">性能<small>时间主要花在哪里？</small></div>' +
      '<div class="kf-oi-links">' +
        '<span class="kf-oi-evidence">Latency · 1.82 ms</span><span class="kf-oi-evidence">目标 · &lt; 1.50 ms</span><span class="kf-oi-evidence">Critical Path · 1.41 ms</span><span class="kf-oi-evidence">等待 / 停顿 · 37%</span>' +
      '</div>' +
      '<div class="kf-rd-h">Critical Path<small>Task #182 是当前的最长任务</small></div>' +
      '<div class="kf-oi-links">' + chain.map(x => x.indexOf('182') >= 0
        ? objectButton('task', '182', x + ' · 214 µs · +45.6%', 'performance')
        : '<span class="kf-oi-evidence">' + esc(x) + '</span>').join(' <span class="kf-oi-evidence">↓</span>') +
      '</div>' +
      '<div class="kf-rd-art"><b>最长任务证据</b><code>Task #182 · 214 µs vs 参考 · 147 µs</code><small>Critical Path · YES · 输入正常 · 输出正确</small></div>' +
      '<div class="kf-oi-actions"><button type="button" data-ws-select-kind="task" data-ws-select-id="182" data-ws-source="performance">查看 Task #182</button></div>' +
    '</section>';
  }

  const rtUs = v => v >= 1000 ? (v / 1000).toFixed(2) + ' ms' : Number(v).toFixed(2) + ' µs';
  function runtimeFindingLabel(id) {
    const labels = {
      mem: '内存容量',
      perf: '性能问题',
      intent: '流水线意图',
      all: '编译问题'
    };
    return labels[id] || id;
  }
  function runtimeIndices(D, kernel) {
    return D.tasks.map((t, i) => t.kn === kernel ? i : -1).filter(i => i >= 0);
  }
  function runtimeFocus(D, ctx, indices) {
    if (indices.indexOf(Number(ctx.taskIndex)) >= 0) return Number(ctx.taskIndex);
    const chained = ((D.perf || {}).chain || {}).steps || [];
    const hit = chained.find(s => indices.indexOf(s.i) >= 0);
    if (hit) return hit.i;
    return indices.slice().sort((a, b) => (D.tasks[b].e - D.tasks[b].s) - (D.tasks[a].e - D.tasks[a].s))[0];
  }
  function runtimeNode(D, i, side) {
    const t = D.tasks[i];
    if (!t) return '';
    const K = D.kernels[t.k] || {};
    return '<button type="button" class="kf-cv-node is-' + side + '" data-runtime-task="' + i + '">' +
      '<code>' + esc(t.kn) + '</code><small>' + esc('Task ' + t.id + ' · ' + rtUs(t.s) + ' → ' + rtUs(t.e) + ' · ' + (K.be || '—')) + '</small></button>';
  }
  function localDependencyHTML(D, focus) {
    const up = D.edges.filter(e => e[1] === focus).map(e => e[0]);
    const down = D.edges.filter(e => e[0] === focus).map(e => e[1]);
    const cap = (list, side) => list.slice(0, 6).map(i => runtimeNode(D, i, side)).join('') +
      (list.length > 6 ? '<span class="kf-cv-more">+' + (list.length - 6) + ' 个同层任务</span>' : '');
    return '<section class="kf-cv-dependency" aria-label="局部依赖上下文">' +
      '<header><div><b>依赖上下文</b><small>当前 Task 的 1-hop 上下游</small></div>' +
      '<button type="button" data-runtime-full>查看完整执行依赖 ↓</button></header>' +
      '<div class="kf-cv-graph"><div class="kf-cv-hop"><span>上游 · ' + up.length + '</span>' + cap(up, 'up') + '</div>' +
      '<i>↓</i>' + runtimeNode(D, focus, 'focus') + '<i>↓</i>' +
      '<div class="kf-cv-hop"><span>下游 · ' + down.length + '</span>' + cap(down, 'down') + '</div></div></section>';
  }
  function renderCompilationRuntime(panel, r, ctx) {
    const D = window.PTO_RUN_TRACE;
    const indices = D ? runtimeIndices(D, ctx.kernelName) : [];
    if (!D || !indices.length) {
      panel.innerHTML = '<section class="kf-rd-sec"><div class="kf-rd-h">运行时证据<small>当前 Run 未找到与该 Kernel 对应的 Task</small></div></section>';
      return;
    }
    const focus = runtimeFocus(D, ctx, indices), task = D.tasks[focus], backend = (D.kernels[task.k] || {}).be || '—';
    const runs = indices.map(i => D.tasks[i]);
    const from = Math.min.apply(null, runs.map(t => t.s)), to = Math.max.apply(null, runs.map(t => t.e));
    const longest = indices.slice().sort((a, b) => (D.tasks[b].e - D.tasks[b].s) - (D.tasks[a].e - D.tasks[a].s))[0];
    const chain = (((D.perf || {}).chain || {}).steps || []).filter(s => indices.indexOf(s.i) >= 0);
    const chainFacts = chain.length ? chain.map(s => '等待 ' + rtUs(s.wait) + ' · 执行 ' + rtUs(s.run)).join(' / ') : '不在当前关键链记录中';
    const selection = indices.length > 1 ? '<nav class="kf-cv-task-list" aria-label="选择运行时任务">' + indices.map(i => {
      const t = D.tasks[i];
      return '<button type="button" class="' + (i === focus ? 'is-on' : '') + '" data-runtime-task="' + i + '"><code>' + esc(t.id) + '</code><span>' + esc(rtUs(t.e - t.s)) + '</span></button>';
    }).join('') + '</nav>' : '';
    const summary = indices.length === 1
      ? '1 个逻辑运行时任务 · ' + rtUs(task.s) + ' → ' + rtUs(task.e) + ' · ' + task.c + ' 个 ' + backend + ' 核'
      : indices.length + ' 个逻辑运行时任务 · 跨度 ' + rtUs(from) + ' → ' + rtUs(to) + ' · 最长 ' + rtUs(D.tasks[longest].e - D.tasks[longest].s);
    panel.innerHTML = '<section class="kf-cv" data-compilation-runtime>' +
      '<header class="kf-cv-context"><span>来自「编译」</span><b>' + esc(runtimeFindingLabel(ctx.findingId)) + ' · ' + esc(ctx.kernelName) + '</b></header>' +
      '<section class="kf-cv-summary"><div><span>运行时证据</span><b>' + esc(ctx.kernelName) + '</b><small>' + esc(summary) + '</small></div>' +
      '<div><span>关键链</span><b>' + esc(chainFacts) + '</b><small>运行时行为是验证证据，不单独判定瓶颈。</small></div></section>' +
      selection + localDependencyHTML(D, focus) +
      '<div class="kf-cv-execution-next"><span>时间线在「执行」中</span><button type="button" data-th-tab="execution">在「执行」中查看已定位的时间线 →</button></div></section>';
  }

  function performanceResourcesStoryPanel() {
    return '<section class="kf-rd-sec" aria-label="资源容量结论">' +
      '<div class="kf-rd-h">资源<small>PASS · 未检测到容量越界</small></div>' +
      '<div class="kf-oi-links"><span class="kf-oi-evidence">L0B peak · 81%</span><span class="kf-oi-evidence">UB peak · 61%</span><span class="kf-oi-evidence">未发现溢出</span></div>' +
      '<p class="kf-rd-note">性能差不等于内存容量不足；当前证据指向关键路径上的窄搬运。</p>' +
    '</section>';
  }

  function renderDetailBody() {
    if (!els.detail) return;
    releaseInvestigation();
    // never innerHTML over nodes on loan from another stage
    releaseBorrowed();
    // 同理：Compilation 的第二个页签借用了 stage 2 的 #kgTrace，
    // 不先还回去，重写 #runTabPanel 会把那个实体节点冲成游离节点。
    window.PTO_COMPILATION?.release?.();
    const t = TASKS.find(x => x.id === st.task);
    const r = t && t.runs.find(x => x.id === st.run);
    if (!t || !r) {
      els.detail.innerHTML = '<p class="kf-rd-empty">选择一次运行查看结果。</p>';
      return;
    }
    if (st.executionEntry?.runId && st.executionEntry.runId !== r.id) st.executionEntry = null;
    if (st.compilationEntry?.runId && st.compilationEntry.runId !== r.id) st.compilationEntry = null;
    const LX = r.live ? liveRun() : null;
    const head = headline(t, r, LX);

    if (r.purged && !r.model) {
      els.detail.innerHTML = head +
        '<p class="kf-rd-note">' + esc(r.note || '产物目录已清理。') + '</p>' +
        '<p class="kf-rd-note is-dim">工件已清理，仅保留结论。</p>';
      return;
    }

    const L = LX;
    /* 统一 Domain Header 放在面板外面：页签切换、借用 stage DOM、编译视图重绘
       都只会重写 #runTabPanel，头部不会被冲掉，也不用参与借用 / 归还流程。 */
    const semanticHandoff = isSemanticCompilationHandoff(r);
    const shell = '<div class="kf-rtpwrap">' + investigationReturnBar() + (semanticHandoff ? semanticCompilationHandoff(t, r) : domainHead(st.tab, r)) +
      '<div class="kf-rtp" id="runTabPanel" role="tabpanel"></div></div>';
    if (L) {
      /* Identity, verdict and the headline numbers stay above the tabs — they
         are true of the run, not of one view of it. Everything below switches. */
      const overview = '<section class="kf-rd-overview" style="display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,.95fr) minmax(0,.85fr);" aria-label="运行快照概览">' +
        '<div class="kf-rd-overview-left">' + head + '</div>' +
        kpis(r, L) +
        '</section>';
      els.detail.innerHTML = overview + tabStrip() + shell;
      const panel = $('#runTabPanel', els.detail);

      if (st.tab === 'overview') {
        panel.innerHTML = overviewPanel(r, L);
        mountInvestigation(r);
      } else if (st.tab === 'execution') {
        renderExecutionTab(panel, r);
      } else if (st.tab === 'compilation') {
        renderCompilationTab(panel, r);
      } else if (st.tab === 'correctness' && usesCorrectnessDiagnosis(r)) {
        dgReset(r.id);
        panel.innerHTML = correctnessDiagnosisPanel();
        dgDraw(panel); dgWatch();
      } else if (st.tab === 'resources') {
        panel.innerHTML = band('编译期内存', '片上水位、复用与调度兑现') + memBlock(L) + intentBlock(L) +
          band('运行时资源', '仅展示已采集信号') +
          '<section class="kf-rw-runtime-empty"><p>本次 Run 未采集运行时资源数据。</p><small>需要 Scope Stats 才能分析 Heap、TensorMap、Ringbuffer。</small><button type="button" data-ws-new-run="resources">重新运行并采集资源数据</button></section>';
      } else {
        syncPanel();
      }
      return;
    }

    // Historical runs retain immutable conclusions even when their directory is gone.
    const sig = (r.signals && r.signals.length)
      ? '<div class="kf-rd-signals">' + r.signals.map(s =>
          '<span class="is-' + s[0] + '"><i></i>' + esc(s[1]) + '</span>').join('') + '</div>'
      : '';
    const storyOverview = semanticHandoff ? '' : (isCompileFailureStory(r) || isCorrectnessFailureStory(r) || isPerformanceWarningStory(r) || isValidatedOptimizationStory(r))
      ? '<section class="kf-rd-overview" style="display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,.95fr) minmax(0,.85fr);" aria-label="运行快照概览">' +
          '<div class="kf-rd-overview-left">' + head + '</div>' + historicalKpis(r) + '</section>'
      : head;
    // its recorded artifact list is in the right rail too (riArchivedArts)
    els.detail.innerHTML = storyOverview + tabStrip() + shell;
    const panel = $('#runTabPanel', els.detail);
    if (st.tab === 'overview') {
      panel.innerHTML = overviewPanel(r, null) + sig;
      mountInvestigation(r);
    } else if (st.tab === 'compilation') {
      if (r.id === 'run_109' || (r.id === 'run_106' && st.compilationEntry?.from === 'investigation')) {
        renderCompilationTab(panel, r);
      } else if (isCompileFailureStory(r)) {
        syncPanel();
        panel.prepend(document.createRange().createContextualFragment(compilationFailureStoryPanel(r)));
      } else if (getDomainVerdict(r, 'compilation').verdict === 'pass') {
        /* 结论已经由统一 Domain Header 承担：模块能画就画模块视图，画不了就
           只留头部，不再叠一层同义的编译摘要。 */
        const view = window.PTO_COMPILATION;
        if (!(view && view.ready && compilationDataMatches(r) && view.render(panel))) {
          panel.innerHTML = '';
        }
      } else panel.innerHTML = notEvaluatedEvidencePanel(r, 'compilation');
    } else if (st.tab === 'correctness') {
      if (isCorrectnessFailureStory(r)) {
        dgReset(r.id);
        panel.innerHTML = correctnessDiagnosisPanel();
        dgDraw(panel); dgWatch();
      }
      else if (getDomainVerdict(r, 'correctness').verdict === 'pass' && hasEvidence(r, 'golden_compare')) panel.innerHTML = '';
      else panel.innerHTML = notEvaluatedEvidencePanel(r, 'correctness');
    } else if (st.tab === 'execution') {
      if (r.id === 'run_106') {
        dgReset(r.id);
        panel.innerHTML = correctnessExecutionStoryPanel(r);
      } else if (hasEvidence(r, 'dependency_graph') || hasEvidence(r, 'runtime_timeline')) {
        renderExecution(panel);
      } else panel.innerHTML = notEvaluatedEvidencePanel(r, 'execution');
    } else if (st.tab === 'performance') {
      if (isPerformanceWarningStory(r)) panel.innerHTML = performanceWarningStoryPanel();
      else if (isValidatedOptimizationStory(r)) panel.innerHTML = validatedPerformancePanel(r);
      else panel.innerHTML = notEvaluatedEvidencePanel(r, 'performance');
    } else if (st.tab === 'resources') {
      if (isPerformanceWarningStory(r)) panel.innerHTML = performanceResourcesStoryPanel();
      else if (getDomainVerdict(r, 'resources').verdict === 'pass') panel.innerHTML = resourcesSummaryPanel(r);
      else panel.innerHTML = notEvaluatedEvidencePanel(r, 'resources');
    }
  }

  function render() {
    const rows = TASKS.filter(matches);
    if (!rows.length) { els.list.innerHTML = '<p class="kf-th-empty">没有符合条件的任务。</p>'; return; }

    els.list.innerHTML = rows.map(t => {
      const open = t.id === st.task;
      const head = latest(t);
      const v = getRunDisplayStatus(head);
      const liveN = t.runs.filter(r => r.live).length;
      const sel = open ? (t.runs.find(r => r.id === st.run) || head) : null;

      return '<div class="kf-th-item' + (open ? ' is-open' : '') + '">' +
        '<button type="button" class="kf-th-row" data-th-task="' + t.id + '" aria-expanded="' + open + '">' +
          '<span class="kf-th-task-icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none"><path d="M20.6604 8.15501L20.6604 15.845C20.6604 16.56 20.2792 17.22 19.6604 17.577L13.0001 21.423C12.3813 21.78 11.6189 21.78 11.0001 21.423L4.33984 17.577C3.72105 17.22 3.33984 16.56 3.33984 15.845L3.33984 8.15501C3.33984 7.44002 3.72105 6.78001 4.33984 6.42301L11.0001 2.57702C11.6189 2.22002 12.3813 2.22002 13.0001 2.57702L19.6604 6.42301C20.2792 6.78001 20.6604 7.44002 20.6604 8.15501Z" fill-rule="evenodd" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1"/><path d="M20.3252 7.19398L13.9506 10.874M9.95794 10.821L3.55273 7.12299" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1"/><path d="M12 21L12 14" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1"/><path d="M8.5023 7.442L9.15181 7.067L11.2499 5.856C11.7499 5.567 12.2499 5.567 12.7499 5.856L14.8479 7.067L15.4975 7.442C15.7861 7.609 15.8471 7.836 15.6805 8.125C15.5138 8.414 15.2861 8.475 14.9975 8.308L14.3479 7.933L12.2499 6.722C12.0832 6.625 11.9165 6.625 11.7499 6.722L9.65181 7.933L9.0023 8.308C8.71361 8.475 8.48591 8.414 8.31931 8.125C8.15261 7.836 8.21361 7.609 8.5023 7.442ZM17.696 11.25L17.696 12L17.696 14.423C17.696 15 17.446 15.433 16.946 15.722L14.8479 16.933L14.1984 17.308C13.9097 17.475 13.6821 17.414 13.5154 17.125C13.3487 16.836 13.4097 16.609 13.6984 16.442L14.3479 16.067L16.446 14.856C16.6127 14.759 16.696 14.615 16.696 14.423L16.696 12L16.696 11.25C16.696 10.917 16.8627 10.75 17.196 10.75C17.5294 10.75 17.696 10.917 17.696 11.25ZM9.80131 17.308L9.15181 16.933L7.05371 15.722C6.55371 15.433 6.30371 15 6.30371 14.423L6.30371 12L6.30371 11.25C6.30371 10.917 6.47041 10.75 6.80371 10.75C7.13701 10.75 7.30371 10.917 7.30371 11.25L7.30371 12L7.30371 14.423C7.30371 14.615 7.38701 14.759 7.55371 14.856L9.65181 16.067L10.3013 16.442C10.59 16.609 10.651 16.836 10.4843 17.125C10.3176 17.414 10.09 17.475 9.80131 17.308ZM15.4701 7.449L15.4975 7.442C15.7399 7.582 15.8205 7.883 15.6805 8.125C15.5405 8.367 15.2399 8.448 14.9975 8.308L14.9901 8.281L15.4701 7.449ZM9.00961 8.281L9.0023 8.308C8.75981 8.448 8.4593 8.367 8.31931 8.125C8.17931 7.883 8.25981 7.582 8.5023 7.442L8.52961 7.449L9.00961 8.281ZM6.32371 11.27L6.30371 11.25C6.30371 10.97 6.5237 10.75 6.80371 10.75C7.08371 10.75 7.30371 10.97 7.30371 11.25L7.28371 11.27L6.32371 11.27ZM10.274 16.449L10.3013 16.442C10.5438 16.582 10.6243 16.883 10.4843 17.125C10.3443 17.367 10.0438 17.448 9.80131 17.308L9.79401 17.281L10.274 16.449ZM14.2057 17.281L14.1984 17.308C13.9559 17.448 13.6554 17.367 13.5154 17.125C13.3754 16.883 13.4559 16.582 13.6984 16.442L13.7257 16.449L14.2057 17.281ZM16.716 11.27L16.696 11.25C16.696 10.97 16.916 10.75 17.196 10.75C17.476 10.75 17.696 10.97 17.696 11.25L17.676 11.27L16.716 11.27Z" fill="currentColor" fill-rule="nonzero"/><circle cx="12" cy="12" r="2" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1"/></svg></span>' +
          '<span class="kf-th-dot is-' + v[1] + '"></span>' +
          '<span class="kf-th-task-copy"><span class="kf-th-title">' + esc(t.title) + '</span>' +
            '<span class="kf-th-task-meta">' + t.runs.length + ' 次运行 · ' + (liveN ? liveN + ' 次产物在库' : '暂无产物在库') + '</span></span>' +
        '</button>' +
        '<button type="button" class="kf-th-compare-icon" data-th-compare-open data-th-compare-task-id="' + t.id + '" aria-label="对比 ' + esc(t.title) + ' 的运行" title="对比此算子的运行"' + (t.runs.length < 2 ? ' disabled' : '') + '>⇄</button>' +
        (open
          ? '<div class="kf-th-runs">' +
              t.runs.map(r => runRow(t, r)).join('') +
            '</div>' +
            ''
          : '') +
      '</div>';
    }).join('');
  }

  /* ---------- events ---------- */
  /* Both the main column and the right rail live outside the side pane and
     carry the same three affordances, so they share one handler. data-step is
     still left to demo-v2's document-level delegation. */
  function toTab(key, entry) {
    const k = toTabKey(key);
    if (st.tab === k) return;
    const options = entry || {};
    const currentProfile = st.tab === 'correctness' ? (DG || diagnosisViewForRun(st.run)) : null;
    const directCompilerEvidence = currentProfile?.compiler?.numericalValidation?.status === 'fail'
      ? currentProfile.compiler.numericalValidation : null;
    st.compilationEntry = k === 'compilation'
      ? (options.compilationEntry || (st.tab === 'correctness' ? {
          from: 'correctness',
          finding: '数值精度 · 输出不一致',
          intent: '正在定位编译语义分歧',
          numericalValidation: directCompilerEvidence
        } : null))
      : null;
    st.executionEntry = k === 'execution' ? (options.executionEntry || null) : null;
    st.tab = k;
    renderDetailBody();
    renderInspector();
  }

  function createFollowupRun(domain, silent) {
    const task = TASKS.find(x => x.id === st.task);
    const previous = task && task.runs.find(x => x.id === st.run);
    if (!task || !previous) return;
    if (previous.id === 'run_105') {
      let next = task.runs.find(x => x.id === 'run_106');
      if (!next) {
        next = {
          id: 'run_106', displayId: '#106', time: '12:48', target: previous.target, duration: '', live: false,
          model: runModel('completed', {
            compilation: { verdict: 'pass', summary: '42 Pass · 45 Kernel · 0 blocking error' },
            correctness: { verdict: 'fail', summary: '输出存在非稳定性数值分歧' },
            execution: { verdict: 'warning', summary: '检测到 1 个可疑的 task 排序问题' },
            performance: { verdict: 'not_evaluated', summary: '正确性未通过，暂不评价性能' },
            resources: { verdict: 'pass', summary: '未发现资源越界' }
          }, {
            golden_compare: { status: 'available' }, tensor_dump: { status: 'partial' }, args_dump: { status: 'available' },
            dependency_graph: { status: 'available' }, runtime_timeline: { status: 'available' }, ir_validation: { status: 'available' },
            pmu: { status: 'not_collected' }, core_trace: { status: 'not_collected' }, scope_stats: { status: 'not_collected', label: 'Scope Stats' }
          }, [{
            id: 'F106', severity: 'critical', domain: 'correctness',
            title: '输出在相同输入下出现非稳定性分歧',
            summary: '重复执行结果不一致，误差明显超过正常浮点累加扰动。', location: 'Tensor T37 · 首个分歧点',
            affectedObjects: [{ kind: 'tensor', id: '37' }],
            evidence: ['与 Golden 比对不一致', 'max_abs_diff = 0.382', '3 次重复运行的 mismatch 区域各不相同'],
            action: { label: '定位首个分歧', route: 'correctness' }
          }, {
            id: 'F106E', severity: 'warning', domain: 'execution',
            title: 'Task #197 在 Task #182 完成读取前覆盖共享缓冲区',
            summary: '当前 Dependency Graph 缺少必要的排序依赖边，导致执行结果随调度时序变化。', location: 'Task #182 → Task #197 · 共享 buffer B2',
            affectedObjects: [{ kind: 'task', id: '182' }, { kind: 'task', id: '197' }, { kind: 'dependency', id: 'missing_182_197' }, { kind: 'buffer', id: 'B2' }],
            evidence: ['涉及同一块 buffer', '#182 读取该 buffer', '#197 覆盖写入该 buffer', '两者之间不存在依赖边', '时间线出现重叠', '重复运行结果不稳定'],
            action: { label: '查看执行', route: 'execution' }
          }], {
            correctnessStory: true, derivedFrom: 'run_105', change: 'dynamic index → affine fallback',
            source: { file: 'decode_layer.py', line: 728 },
            objectMap: { tensor: 'T37', producer: 'Task #182', writer: 'Task #197', dependency: '缺少排序依赖边', buffer: '共享 buffer B2' }
          }),
          artifacts: [Object.assign({}, ART().source, { meta: 'decode_layer.py · 任务排序源码映射', tone: 'warn' }), Object.assign({}, ART().correct, { meta: 'T37 · 首个分歧点', tone: 'bad', primary: true })], signals: []
        };
        task.runs.unshift(next);
      }
      st.run = next.id; st.selection = null; st.tab = 'overview';
      render(); renderDetail();
      const toast = $('#toast');
      if (!silent && toast) { toast.textContent = '已基于 #105 创建新的运行 #106'; toast.classList.add('is-visible'); setTimeout(() => toast.classList.remove('is-visible'), 1800); }
      return;
    }
    if (previous.id === 'run_106') {
      let next = task.runs.find(x => x.id === 'run_107');
      if (!next) {
        next = {
          id: 'run_107', displayId: '#107', time: '12:51', target: previous.target, duration: '', live: false,
          model: runModel('completed', {
            compilation: { verdict: 'pass', summary: '42 Pass · 45 Kernel' },
            correctness: { verdict: 'pass', summary: '3 / 3 Oracle · 12 / 12 checkpoint match' },
            execution: { verdict: 'pass', summary: '428 Task · Dependency Graph 完整' },
            performance: { verdict: 'warning', summary: 'Latency 1.82 ms · 目标 < 1.50 ms' },
            resources: { verdict: 'pass', summary: '未发现溢出 · L0B peak 81%' }
          }, {
            golden_compare: { status: 'available' }, tensor_dump: { status: 'available', label: 'Tensor Checkpoints' },
            dependency_graph: { status: 'available' }, runtime_timeline: { status: 'available' }, critical_path: { status: 'available' },
            pmu: { status: 'partial' }, core_trace: { status: 'not_collected', label: 'Core Swimlane' }, memory_map: { status: 'available' },
            scope_stats: { status: 'not_collected', label: 'Scope Stats' }
          }, [{
            id: 'F107', severity: 'warning', domain: 'performance',
            title: 'Critical Path 超过延迟目标',
            summary: '端到端延迟 1.82 ms，目标 < 1.50 ms；Task #182 是当前主要的最长任务。', location: 'Task #182 · attention_incore_2',
            affectedObjects: [{ kind: 'task', id: '182' }, { kind: 'kernel', id: 'attention_incore_2' }],
            evidence: ['total latency: 1.82 ms', 'target: < 1.50 ms', 'Task #182 duration: 214 µs', 'baseline/reference task duration: 147 µs', 'Task #182 位于 Critical Path 上'],
            action: { label: '在「执行」中查看性能证据', route: 'performance' }
          }, {
            id: 'F107K', severity: 'warning', domain: 'performance',
            title: 'RoPE lo/hi 半维被拆成两次窄搬运',
            summary: '连续的 128-element row 被拆成两个 64-element transfer，增加搬运次数，并拉长 Critical Path 上的 Task #182。', location: 'attention_incore_2 · decode_layer.py / RoPE block',
            affectedObjects: [{ kind: 'kernel', id: 'attention_incore_2' }, { kind: 'source', id: 'decode_layer.py:728' }],
            evidence: ['实际宽度: 256 B', '目标宽度: 512 B', '触发次数: 16', 'Task #182: 耗时 +45.6%'],
            action: { label: '查看 Kernel 性能证据', route: 'performance' }
          }], {
            performanceStory: true, derivedFrom: 'run_106', change: '添加 Task #182 → #197 ordering dependency',
            source: { file: 'decode_layer.py', line: 728 },
            objectMap: { longPole: 'Task #182', kernel: 'attention_incore_2', source: 'decode_layer.py:728', criticalPath: ['141', '178', '182', '196', '201'] },
            compareMetrics: { latency: '1.82 ms', criticalPath: '1.41 ms', task182: '214 µs', mteStall: '31%', transferWidth: '256 B', l0bPeak: '81%', kernelCount: '45' }
          }),
          artifacts: [Object.assign({}, ART().source, { meta: 'decode_layer.py · RoPE transfer recommendation', tone: 'warn' }), Object.assign({}, ART().correct, { meta: '12 / 12 checkpoint match', tone: 'ok' })], signals: []
        };
        task.runs.unshift(next);
      }
      st.run = next.id; st.selection = null; st.tab = 'overview';
      render(); renderDetail();
      const toast = $('#toast');
      if (!silent && toast) { toast.textContent = '已基于 #106 添加 ordering dependency，创建运行 #107'; toast.classList.add('is-visible'); setTimeout(() => toast.classList.remove('is-visible'), 1800); }
      return;
    }
    if (previous.id === 'run_107') {
      let next = task.runs.find(x => x.id === 'run_108');
      if (!next) {
        next = {
          id: 'run_108', displayId: '#108', time: '12:56', target: previous.target, duration: '', live: false,
          model: runModel('completed', {
            compilation: { verdict: 'pass', summary: '42 Pass · 45 Kernel' },
            correctness: { verdict: 'pass', summary: '3 / 3 Oracle · 12 / 12 checkpoint match' },
            execution: { verdict: 'pass', summary: 'Task topology 未变 · 依赖有效' },
            performance: { verdict: 'pass', summary: 'Latency 1.36 ms · 目标 < 1.50 ms' },
            resources: { verdict: 'pass', summary: 'L0B peak 83% · 预算内' }
          }, {
            golden_compare: { status: 'available' }, tensor_dump: { status: 'available', label: 'Tensor Checkpoints' },
            dependency_graph: { status: 'available' }, runtime_timeline: { status: 'available' }, critical_path: { status: 'available' },
            pmu: { status: 'partial' }, memory_map: { status: 'available' }, scope_stats: { status: 'not_collected', label: 'Scope Stats' }
          }, [{
            id: 'F108P', severity: 'info', domain: 'performance', title: '已达到延迟目标',
            summary: '1.36 ms · 目标 < 1.50 ms', location: 'Critical Path 0.98 ms', affectedObjects: [{ kind: 'task', id: '182' }],
            evidence: ['Task #182: 147 µs', 'transfer width: 512 B', 'MTE stall: 17%'], action: { label: '与 #107 对比', route: 'overview' }
          }, {
            id: 'F108V', severity: 'info', domain: 'correctness', title: '优化未引入正确性回退',
            summary: '3 / 3 Oracle · 12 / 12 checkpoint match', location: 'Golden / Oracle unchanged', affectedObjects: [],
            evidence: ['Correctness: PASS', '12 / 12 checkpoints match'], action: { label: '与 #107 对比', route: 'overview' }
          }], {
            optimizationStory: true, derivedFrom: 'run_107', change: 'wide transfer optimization',
            changes: [
              { domain: 'source', title: '合并 RoPE lo / hi 搬运', before: '2 × 64-element transfer', after: '1 × 128-element transfer' },
              { domain: 'compile', title: 'Kernel topology unchanged', before: '45 kernels', after: '45 kernels' }
            ],
            compareMetrics: { latency: '1.36 ms', criticalPath: '0.98 ms', task182: '147 µs', mteStall: '17%', transferWidth: '512 B', l0bPeak: '83%', kernelCount: '45' },
            source: { file: 'decode_layer.py', line: 728 }
          }),
          artifacts: [], signals: []
        };
        task.runs.unshift(next);
      }
      st.run = next.id; st.selection = null; st.tab = 'overview';
      render(); renderDetail();
      const toast = $('#toast');
      if (!silent && toast) { toast.textContent = '优化方案已应用，创建新的运行 #108'; toast.classList.add('is-visible'); setTimeout(() => toast.classList.remove('is-visible'), 1800); }
      return;
    }
    const n = task.runs.filter(x => x.id.indexOf(previous.id + '_rerun') === 0).length + 1;
    const domains = makeDomains({});
    domains.compilation = { verdict: 'not_evaluated', summary: '等待新运行' };
    domains.correctness = { verdict: 'not_evaluated', summary: '等待新运行' };
    domains.execution = { verdict: 'not_evaluated', summary: '等待新运行' };
    domains.performance = { verdict: 'not_evaluated', summary: '等待新运行' };
    domains.resources = { verdict: 'not_evaluated', summary: '等待新运行' };
    const next = {
      id: previous.id + '_rerun' + n, time: '待调度', target: previous.target, duration: '', live: false,
      model: runModel('running', domains, {}, [], { parentRunId: previous.id, requestedDomain: domain }), artifacts: [], signals: []
    };
    task.runs.unshift(next);                 // prior Run remains immutable
    st.run = next.id; st.selection = null; st.tab = 'overview';
    render(); renderDetail();
  }

  function onRunClick(e) {
    const runtimeFull = e.target.closest('[data-runtime-full]');
    if (runtimeFull) {
      const fullGraph = document.getElementById('runFan');
      if (fullGraph && fullGraph.scrollIntoView) fullGraph.scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }
    const runtimeTask = e.target.closest('[data-runtime-task]');
    if (runtimeTask && st.runtimeContext) {
      st.runtimeContext = Object.assign({}, st.runtimeContext, { taskIndex: Number(runtimeTask.dataset.runtimeTask) });
      renderDetailBody();
      return;
    }
    const compareOpen = e.target.closest('[data-th-compare-open]');
    if (compareOpen) {
      if (compareOpen.disabled) return;
      const targetTask = compareOpen.dataset.thCompareTaskId;
      if (targetTask && targetTask !== st.task) {
        st.task = targetTask;
        const target = TASKS.find(t => t.id === targetTask);
        st.run = target ? latest(target).id : null;
        st.compareRuns = [];
        render();
        renderDetail();
      }
      const opening = els.compareBox.hidden;
      els.compareBox.hidden = !opening;
      if (opening) renderComparePicker();
      return;
    }
    const compareRun = e.target.closest('[data-th-compare-run]');
    if (compareRun) {
      st.compareRuns = $$('[data-th-compare-run]:checked', els.compareBox).map(x => x.value);
      if (st.compareRuns.length > 2) {
        compareRun.checked = false;
        st.compareRuns = $$('[data-th-compare-run]:checked', els.compareBox).map(x => x.value);
      }
      const submit = $('[data-th-compare-submit]', els.compareBox);
      if (submit) submit.disabled = st.compareRuns.length !== 2;
      return;
    }
    const compareSubmit = e.target.closest('[data-th-compare-submit]');
    if (compareSubmit && !compareSubmit.disabled) {
      renderRunComparison();
      els.compareBox.hidden = true;
      return;
    }
    if (e.target.closest('[data-investigation-return]')) {
      clearSelection();
      toTab('overview');
      return;
    }
    const caseSwitch = e.target.closest('[data-investigation-case]');
    if (caseSwitch && caseSwitch.dataset.investigationCase !== investigation.activeCaseId) {
      investigation.activeCaseId = caseSwitch.dataset.investigationCase;
      investigation.entry = null;
      clearSelection();
      renderDetail();
      return;
    }
    if (e.target.closest('[data-investigation-next]')) {
      const currentRun = TASKS.find(task => task.id === st.task)?.runs.find(run => run.id === st.run);
      openInvestigationEvidence(currentRun ? activeInvestigationCase(currentRun)?.investigationData?.action : null);
      return;
    }
    const compareClose = e.target.closest('[data-th-compare-close]');
    if (compareClose) {
      renderDetail();
      return;
    }
    const compareWith = e.target.closest('[data-ws-compare-with]');
    if (compareWith) { compareCurrentRunWith(compareWith.dataset.wsCompareWith); return; }
    const setBaseline = e.target.closest('[data-ws-set-baseline]');
    if (setBaseline) { setTrustedBaseline(setBaseline.dataset.wsSetBaseline); return; }
    const fixRerun = e.target.closest('[data-ws-fix-rerun]');
    if (fixRerun) { createFollowupRun('compilation'); return; }
    const fixDependency = e.target.closest('[data-ws-fix-dependency]');
    if (fixDependency) { openFixPreview(); return; }
    const optimizeRerun = e.target.closest('[data-ws-optimize-rerun]');
    if (optimizeRerun) { createFollowupRun('performance'); return; }
    const openSource = e.target.closest('[data-ws-open-source]');
    if (openSource) {
      const parts = openSource.dataset.wsOpenSource.split(':');
      const file = parts[0], line = Number(parts[1]);
      releaseBorrowed();
      const sourceButton = document.querySelector('[data-file="' + file + '"]');
      if (sourceButton) sourceButton.click();
      setTimeout(() => {
        const row = $$('#dslEditor > div').find(x => Number($('i', x)?.textContent) === line);
        if (!row) return;
        row.classList.add('is-changed');
        row.setAttribute('aria-label', file + ':' + line + ' · Source → ' + (openSource.dataset.wsSourceContext || 'IR Pass: LegalizeIndexing'));
        row.scrollIntoView({ block: 'center', behavior: 'smooth' });
      }, 80);
      return;
    }
    const newRun = e.target.closest('[data-ws-new-run]');
    if (newRun) { createFollowupRun(newRun.dataset.wsNewRun); return; }
    // finding cards open in place; no state is kept, so a tab switch re-collapses
    const fh = e.target.closest('[data-th-find]');
    if (fh) {
      const open = fh.getAttribute('aria-expanded') !== 'true';
      fh.setAttribute('aria-expanded', String(open));
      const more = document.getElementById(fh.dataset.thFind);
      if (more) more.hidden = !open;
      return;
    }
    // a hot cell in the memory heatmap drills into that kernel in the compile
    // guard, which is where the buffer detail lives — now one tab over
    const kc = e.target.closest('[data-th-kernel]');
    if (kc) {
      toTab('compilation');
      const name = kc.dataset.thKernel;
      selectObject({ kind: 'buffer', id: name, sourceTab: 'resources' });
      setTimeout(() => { if (window.PTO_GUARD) window.PTO_GUARD.select(name); }, 60);
      return;
    }
    const finding = e.target.closest('[data-ws-finding]');
    if (finding) {
      const task = TASKS.find(x => x.id === st.task);
      const run = task && task.runs.find(x => x.id === st.run);
      const f = getFindings(run).find(x => x.id === finding.dataset.wsFinding);
      selectObject({ kind: 'finding', id: finding.dataset.wsFinding, sourceTab: 'overview' });
      toTab(f?.action?.route || f?.domain || 'overview');
      return;
    }
    const select = e.target.closest('[data-ws-select-kind]');
    if (select) {
      const obj = { kind: select.dataset.wsSelectKind, id: select.dataset.wsSelectId, sourceTab: select.dataset.wsSource };
      selectObject(obj);
      showObjectTooltip(select, obj);
      return;
    }
    /* ① 结论 → ③ 证据：等待条把同名 kernel 的全部任务和关键链上的那一个
       task 一起点亮，其余降透明度，整次 Run 的形状保持可见。再点一次取消。 */
    const xfLocate = e.target.closest('[data-xf-locate]');
    if (xfLocate) {
      const on = xfLocate.getAttribute('aria-pressed') !== 'true';
      $$('[data-xf-locate]', els.detail).forEach(b => b.setAttribute('aria-pressed', 'false'));
      if (on) {
        xfLocate.setAttribute('aria-pressed', 'true');
        window.PTO_TIMELINE?.setHighlight?.(
          { kernels: [xfLocate.dataset.xfLocate], tasks: [Number(xfLocate.dataset.xfTask)] },
          { track: 'both' });
      } else window.PTO_TIMELINE?.clearHighlight?.();
      const tlHost = els.detail && $('#runTimeline', els.detail);
      if (tlHost) tlHost.scrollIntoView({ block: 'center', behavior: 'smooth' });
      return;
    }
    const route = e.target.closest('[data-ws-route]');
    if (route) { toTab(route.dataset.wsRoute); return; }
    const kernel = e.target.closest('[data-ws-open-kernel]');
    if (kernel) {
      const name = kernel.dataset.wsOpenKernel;
      toTab('compilation');
      selectObject({ kind: 'kernel', id: name, sourceTab: 'compilation' });
      // 新的 Compilation 视图在 #runTabPanel 里，先把它选中；stage 2 的
      // Kernel Guard 仍照旧同步，两者互不影响。
      setTimeout(() => {
        window.PTO_COMPILATION?.selectKernel?.(name);
        window.PTO_GUARD?.select?.(name);
      }, 60);
      return;
    }
    const pass = e.target.closest('[data-ws-open-pass]');
    if (pass) { toTab('compilation'); return; }
    const passNode = e.target.closest('[data-kg-p], [data-kg-kp]');
    if (passNode) {
      const i = passNode.dataset.kgP || passNode.dataset.kgKp;
      const name = ((window.PTO_IR_KERNELS || {}).passNames || [])[Number(i)] || ('Pass ' + i);
      selectObject({ kind: 'pass', id: name, sourceTab: 'compilation' });
      return;
    }
    const compilationDiff = e.target.closest('[data-th-compilation-diff]');
    if (compilationDiff) {
      window.PTO_COMPILATION?.selectPass?.(compilationDiff.dataset.thCompilationDiff);
      return;
    }
    const tb = e.target.closest('[data-th-tab]');
    if (tb) {
      const art = tb.dataset.thArt;
      if (art) st.artifact = art;
      toTab(tb.dataset.thTab);
      if (art) render();
      // the timeline artifacts scroll rather than switch, so only tabs return here
      els.detail.scrollIntoView({ block: 'start' });
      return;
    }
    const group = e.target.closest('[data-th-art-group]');
    if (group) {
      st.artifactGroup = group.dataset.thArtGroup;
      renderInspector();
      return;
    }
    const sc = e.target.closest('[data-th-scroll]');
    if (sc) {
      toTab('execution');
      const el = els.detail && $('#runTimeline', els.detail);
      if (el) el.scrollIntoView({ block: 'start', behavior: 'smooth' });
      return;
    }
    const art = e.target.closest('[data-th-art]');
    if (!art) return;
    st.artifact = art.dataset.thArt;
    render(); renderDetail();
    // archived runs have no tab panel to open into, so they keep the old route
    if (art.dataset.thView === 'explorer') { const b = $('#activityExplorer'); if (b) b.click(); }
  }

  function wire() {
    dgWire();
    if (els.root) els.root.addEventListener('click', onRunClick);
    if (els.detail) els.detail.addEventListener('click', onRunClick);
    bindObjectTooltips();
    bindFixPreview();
    window.addEventListener('pto:compilation-runtime', event => {
      const ctx = event.detail || {};
      const task = TASKS.find(t => t.id === st.task);
      const run = task && task.runs.find(r => r.id === st.run);
      if (!run || !compilationDataMatches(run) || String(ctx.runId) !== String(run.id)) return;
      st.runtimeContext = Object.assign({}, ctx, { taskIndex: null });
      st.tab = 'execution';
      renderDetailBody();
      renderInspector();
    });
    // A fix never rewrites the selected historical Run. The linear correctness
    // surface delegates this action to a newly created follow-up Run instead.
    document.addEventListener('click', e => {
      if (!e.target.closest('#fixAndRerun')) return;
      e.preventDefault(); e.stopImmediatePropagation();
      createFollowupRun('correctness');
    }, true);

    els.filters.addEventListener('click', e => {
      const b = e.target.closest('[data-th-filter]'); if (!b) return;
      st.filter = b.dataset.thFilter;
      $$('[data-th-filter]', els.filters).forEach(x => x.setAttribute('aria-pressed', String(x === b)));
      render();
    });

    els.list.addEventListener('click', e => {
      const art = e.target.closest('[data-th-art]');
      if (art) {
        st.artifact = art.dataset.thArt;
        render(); renderDetail();
        // stage 1 belongs to the explorer activity view (EXPLORER_STEP in
        // demo-v2.js), so route source artifacts through the rail button.
        if (art.dataset.thView === 'explorer') { const b = $('#activityExplorer'); if (b) b.click(); }
        return;                       // demo-v2 handles data-step itself
      }
      const run = e.target.closest('[data-th-run]');
      if (run) {
        st.task = run.dataset.thOf; st.run = run.dataset.thRun; st.artifactGroup = 'input'; st.compareRuns = []; st.selection = null; st.tab = 'overview';
        render(); renderDetail(); if (!els.compareBox.hidden) renderComparePicker(); toOverview();
        return;
      }
      const row = e.target.closest('[data-th-task]');
      if (!row) return;
      const id = row.dataset.thTask;
      if (st.task === id) { st.task = null; }
      else { st.task = id; const t = TASKS.find(x => x.id === id); st.run = t ? latest(t).id : null; st.compareRuns = []; st.selection = null; st.tab = 'overview'; }
      render(); renderDetail(); if (!els.compareBox.hidden) renderComparePicker();
      if (st.task) toOverview();
    });

    // keep the artifact highlight in sync when the stage changes elsewhere
    const stages = $$('.kf-stage');
    if (stages.length && typeof MutationObserver === 'function') {
      const mo = new MutationObserver(() => {
        const i = stages.findIndex(s => s.classList.contains('is-active'));
        const t = TASKS.find(x => x.id === st.task);
        if (i < 0 || !t) return;
        const r = t.runs.find(x => x.id === st.run);
        const a = r && r.artifacts && r.artifacts.find(x => x.step === i);
        if (a && a.k !== st.artifact) { st.artifact = a.k; render(); renderDetail(); }
      });
      stages.forEach(s => mo.observe(s, { attributes: true, attributeFilter: ['class'] }));
    }
  }

  // demo-v2.js owns stage switching, so reach stage 0 through its own button.
  function toOverview() { const b = $('#stepNav [data-step="0"]'); if (b) b.click(); }

  // demo-v2.js rewrites the pane header on every activity-view switch with the
  // old route wording; take it back whenever the workflow pane is showing.
  function claimPaneHeader() {
    const title = $('#sidePaneTitle'), meta = $('#sidePaneMeta');
    if (!title || typeof MutationObserver !== 'function') return;
    const fix = () => {
      const showing = !$('[data-side-view="workflow"]').hidden;
      if (!showing) return;
      if (title.textContent !== '任务与运行') title.textContent = '任务与运行';
      const want = '';
      if (meta && meta.textContent !== want) meta.textContent = want;
    };
    new MutationObserver(fix).observe(title, { childList: true, characterData: true, subtree: true });
    if (meta) new MutationObserver(fix).observe(meta, { childList: true, characterData: true, subtree: true });
    new MutationObserver(fix).observe($('[data-side-view="workflow"]'), { attributes: true, attributeFilter: ['hidden'] });
    fix();
  }

  // Stage 0 is the run detail page and carries its own identity header, so the
  // generic pane header ('定义目标 / recipe · decode_layer') is dead chrome
  // there. Hidden by class so demo-v2 keeps owning the .hidden flag.
  function hideStageHeaderOnDetail() {
    const hdr = $('#stageTitle') && $('#stageTitle').closest('.pto-ide-frame__pane-header');
    const stage0 = $('.kf-stage[data-stage="0"]');
    if (!hdr || !stage0 || typeof MutationObserver !== 'function') return;
    const fix = () => hdr.classList.toggle('kf-rd-nohdr', stage0.classList.contains('is-active'));
    new MutationObserver(fix).observe(stage0, { attributes: true, attributeFilter: ['class'] });
    fix();
  }

  /* The four-run sequence is a reviewable fixture, not an interaction-only
     breadcrumb. Keep each immutable Run visible after a page reload while the
     same follow-up commands remain available for future runs. */
  function seedStoryRuns() {
    const decode = TASKS.find(t => t.id === 'task_decode');
    if (!decode || !decode.runs.some(r => r.id === 'run_105')) return;
    st.task = decode.id;
    st.run = 'run_105';
    createFollowupRun('compilation', true);
    createFollowupRun('dependency', true);
    createFollowupRun('performance', true);
  }

  function seedCompilerSemanticFixture() {
    const decode = TASKS.find(t => t.id === 'task_decode');
    if (!decode || decode.runs.some(r => r.id === 'run_109')) return;
    decode.runs.unshift({
      id: 'run_109', displayId: '#109', time: '13:02', target: 'Ascend 910B', duration: '', live: false,
      model: runModel('completed', {
        compilation: { verdict: 'pass', summary: '42 Pass · 结构校验 PASS' },
        correctness: { verdict: 'fail', summary: 'ExpandMixedKernel 首次引入数值语义偏差' },
        execution: { verdict: 'not_evaluated', summary: '编译语义偏差已在设备执行前定位' },
        performance: { verdict: 'not_evaluated', summary: '正确性未通过，暂不评价性能' },
        resources: { verdict: 'unknown', summary: '未采集运行时资源证据' }
      }, {
        golden_compare: { status: 'available' }, ir_validation: { status: 'available' }, pass_dump: { status: 'available' }, source_location: { status: 'available' },
        tensor_dump: { status: 'not_collected' }, args_dump: { status: 'not_collected' }, dependency_graph: { status: 'not_collected' }, runtime_timeline: { status: 'not_collected' },
        pmu: { status: 'not_collected' }, core_trace: { status: 'not_collected' }, scope_stats: { status: 'not_collected' }
      }, [{
        id: 'F109', severity: 'critical', domain: 'correctness',
        title: 'ExpandMixedKernel 引入数值语义偏差',
        summary: '结构校验通过，但该 Pass 后的 Host IR execution 首次偏离 PyTorch Golden。',
        location: 'ExpandMixedKernel · 首个数值分歧 Pass',
        affectedObjects: [{ kind: 'pass', id: 'ExpandMixedKernel' }],
        evidence: ['PyTorch Golden 有效', 'Tolerance 有效', '结构校验: PASS', '数值校验: FAIL', '后续 Pass: MISMATCH'],
        action: { label: '定位首个数值分歧 Pass', route: 'correctness' }
      }], {
        correctnessStory: true, fixture: true, fixtureLabel: '编译器语义错误',
        source: { file: 'decode_layer.py', line: 728 }
      }),
      artifacts: [Object.assign({}, ART().compile, { meta: 'ExpandMixedKernel · 首个数值分歧', tone: 'bad', primary: true }), Object.assign({}, ART().correct, { meta: 'PyTorch Golden · 该 Pass 后 MISMATCH', tone: 'bad' })],
      signals: []
    });
  }

  /* 打开 Run 视图时默认落在磁盘上那次真实运行，而不是演示脚本刚生成的
     follow-up —— 只有它有 passes_dump / dfx_outputs。判定用 live 标记，
     不写死 run id，换一批数据也不会指错。 */
  function selectDefaultRun() {
    const task = TASKS.find(t => t.id === st.task) || TASKS[0];
    st.run = (task.runs.find(r => r.live) || latest(task)).id;
  }

  function boot() {
    if (!mount()) return;
    seedStoryRuns();
    seedCompilerSemanticFixture();
    selectDefaultRun();
    wire(); claimPaneHeader(); hideStageHeaderOnDetail(); watchRunInspectorLayout();
    render(); renderDetail();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
