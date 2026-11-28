/* ==========================================================================
   PyPTO Full-Story 2.0 · Spatial Systems Investigation Workbench (v2.5)
   AI_Infra v1.3.1 Task Contract · Scoped Evidence Model · Official Glyphs
   Repo: hw-native-sys/pypto · Operator: decode_sparse_attn_csa · 910B2
   ========================================================================== */

(function (global) {
  'use strict';

  // 1. Task Contract & Presentation Semantics Contract (ai-infra-investigation)
  const CASE_CONTRACT = Object.freeze({
    caseId: 'pypto-case-32x4096-r910b2',
    sourceKind: 'scenario-record',
    rawCaptureVerified: false,
    declaredRevisionVerified: false,
    scope: 'pypto-case-32x4096-r910b2',
    operator: 'decode_sparse_attn_csa',
    operatorTitle: 'Decode 稀疏注意力',
    repo: 'hw-native-sys/pypto',
    revision: '910b2-a5c8e41d',
    chip: 'Ascend 910B2',
    chipShort: '910B2',
    topology: '24 CG / 48 AIV · TP=2',
    fingerprint: '910b2-cg24-a5c8e41d',
    shape: '[32, 4096]',
    auxShapes: ['[4, 2048]', '[1, 128]', 'valid_shape [+64 尾块]'],
    dtype: 'FP16/BF16 (FP32 Acc)',
    clock: 'device_wall_us',
    sampleCount: 20,
    vecLimitKB: 192,
    matLimitKB: 512,
    baselineId: 'exp-0',
    baselineDisplayId: '#0',
    baselineLatencyUs: 482.0,
    defaultTargetLatencyUs: 220.0,
  });

  const PRESENTATION_CONTRACT = Object.freeze({
    version: '1.3.1',
    alignmentKey: '{scope, id}',
    modes: {
      landscape: {
        id: 'landscape',
        name: '演进地形',
        question: '实验波次推进中性能收益是否越过样本波动包络，片上 Vec 容量是否触顶？',
        xSemantics: '离散并发演进波次 (Wave 0..8)',
        yLenses: ['latency', 'vec', 'speedup'],
      },
      bullseye: {
        id: 'bullseye',
        name: '靶心收敛',
        question: '4 个 PyPTO 调优方向如何向 ≤220μs 门禁圈与 6/6 全合规靶心收敛，哪些实验因单维违例被隔离？',
        radialSemantics: 'device_wall_us 相对 220μs 门禁圈与 482μs 基线环比例',
        quadrantSemantics: 'NW InCore 调优 · NE 内存与 Ring · SE 编排与派发 · SW Host 常驻',
      },
      swimlane: {
        id: 'swimlane',
        name: '分支泳道',
        question: '各策略轨道的派生因果、停滞区间、晋升主干路径及门禁裕度如何随波次演进？',
        xSemantics: '共享波次列分区 (Wave 0..8)',
        ySemantics: '最优主干 + 4 条 PyPTO 原生调优方向泳道 + 共享 X 证据带',
      },
    },
    comparabilityRules: [
      '所有耗时比较使用同一硬件指纹 (910b2-cg24-a5c8e41d) 与同一 device_wall_us 计时口径（20 次热身后的设备端起止差）。',
      '样本线段 [min, max] 表示已观测样本极值包络，用于识别重叠停滞（如 #6/#8/#9），不代表统计置信区间。',
      '契约版本从 v1 演进至 v3 [+64 尾块] 后，仅测主 Shape 的实验（如 #10）保留原始实测值并标注泛化违例，不篡改历史读数。',
      '编译期被静态拦截的实验（#5 Vec 234KB > 192KB）标为 derived，不伪造设备端耗时。',
    ],
  });

  // 2. Official Glyph Paths from ai-infra-glyphs/assets/op-icons.js
  // Source SHA256: a2d430dcf655d38bcb09428f7b2645990e761a3b6abee1b3401b8cc18e804408
  const GLYPH_PATHS = Object.freeze({
    attention: 'M4 5h5v5H4zM15 5h5v5h-5zM9 16h6v5H9zM6 10l6 6M18 10l-6 6',
    router: 'M4 12h7M11 12l5-7h4M11 12h9M11 12l5 7h4',
    matmul: 'M3 4h4M3 4v16h4M21 4h-4M21 4v16h-4M8 7h3v3H8zM14 7h3v3h-3zM8 14h3v3H8zM14 14h3v3h-3z',
    cache: 'M5 4h14v16H5zM5 8h14M9 12h6M9 16h6',
    exchange: 'M3 8h17M16 4l4 4-4 4M21 16H4M8 12l-4 4 4 4',
    spine: 'M4 12h16M12 4l4 8-4 8',
  });

  // 3. Verified PyPTO Native Performance Tuning Categories (hw-native-sys/pypto)
  const AXIS_META = Object.freeze({
    dispatch: {
      id: 'dispatch',
      name: '编排与派发',
      short: '编排派发',
      glyphId: 'router',
      glyphPath: GLYPH_PATHS.router,
      pillClass: 'pill-dispatch',
      desc: '任务粒度、pl.spmd / pl.at 分核映射、mix kernel 与假依赖解除',
      quadrant: 'SE',
      excluded: false,
    },
    incore: {
      id: 'incore',
      name: 'InCore 调优',
      short: 'InCore',
      glyphId: 'matmul',
      glyphPath: GLYPH_PATHS.matmul,
      pillClass: 'pill-incore',
      desc: 'Double Buffer、Tile/Block 切分、小 Shape 直通快路径与指令粒度',
      quadrant: 'NW',
      excluded: false,
    },
    memory: {
      id: 'memory',
      name: '内存与 Ring',
      short: '内存与 Ring',
      glyphId: 'cache',
      glyphPath: GLYPH_PATHS.cache,
      pillClass: 'pill-memory',
      desc: 'Vec / Mat / Left / Right / Acc / Bias 片上放置与 scope-depth ring 复用',
      quadrant: 'NE',
      excluded: false,
    },
    host: {
      id: 'host',
      name: 'Host 常驻',
      short: 'Host 常驻',
      glyphId: 'exchange',
      glyphPath: GLYPH_PATHS.exchange,
      pillClass: 'pill-host',
      desc: '未触发 · 输入 Tensor 与 KV 索引常驻 Device HBM，无 Host 往返瓶颈',
      quadrant: 'SW',
      excluded: true,
    },
  });

  // 6 Two-Character Operator Contract Gates (MUST KEEP EXACT NAMES)
  const GATE_KEYS = ['hardware', 'correctness', 'precision', 'generalization', 'performance', 'engineering'];
  const GATE_LABELS = ['硬件', '正确性', '精度', '泛化', '性能', '规范'];
  const GATE_NUMBERED = ['1. 硬件', '2. 正确性', '3. 精度', '4. 泛化', '5. 性能', '6. 规范'];

  const EVIDENCE_GRADE_META = Object.freeze({
    observed: { id: 'observed', label: '实测 observed', short: '实测', desc: 'Ascend 910B2 设备端 20 次采样实测 (device_wall_us)' },
    derived: { id: 'derived', label: '推导 derived', short: '推导', desc: 'PyPTO 编译期静态 Pass 容量与生命周期上界推导' },
    candidate: { id: 'candidate', label: '待复核 candidate', short: '待复核', desc: 'Agent 依据源码与前序样本派生的候选分支，待真机上板复核' },
  });

  // Natural Language Prompt & Phrase Anchors for 目标与约束
  const DEFAULT_NL_PROMPT =
    '优化 decode_sparse_attn_csa 在 Ascend 910B2 (24 CG / 48 AIV, TP=2) 上的 Decode 稀疏注意力 [32, 4096] FP16/BF16 端到端耗时，目标 ≤220μs (≥2.19×)，通过 Golden 对拍、保持 FP16/BF16→FP32 Acc→BF16 且满足 rtol=1/128，禁止隐式 Fallback。';

  const NL_PHRASES = [
    { key: 'hardware', text: 'Ascend 910B2 (24 CG / 48 AIV, TP=2)' },
    { key: 'scenario', text: 'Decode 稀疏注意力 [32, 4096] FP16/BF16' },
    { key: 'perf', text: '目标 ≤220μs (≥2.19×)' },
    { key: 'correctness', text: '通过 Golden 对拍' },
    { key: 'precision', text: 'FP16/BF16→FP32 Acc→BF16 且满足 rtol=1/128' },
    { key: 'fallback', text: '禁止隐式 Fallback' },
  ];

  const INITIAL_CONTRACT_V1 = {
    version: 'v1',
    epochNum: 1,
    chip: 'Ascend 910B2',
    topology: '24 CG / 48 AIV · TP=2',
    fingerprint: '910b2-cg24-a5c8e41d',
    goldenRef: 'golden_sparse_attn (FP16/BF16→FP32 Acc)',
    atol: 1e-4,
    rtol: '1/128 (7.8e-3)',
    p0TargetLatencyUs: 220.0,
    p0TargetSpeedup: 482.0 / 220.0,
    p1GuardrailEnabled: false,
    p1MaxRegressionPct: 10.0,
    minPrecisionMarginPct: 15,
    maxWorkspaceKB: 128,
    vecLimitKB: 192,
    sparseCoverageMode: '20% 均匀稀疏 (主 Shape)',
    forbidTopologyChange: true,
    requireZeroFallback: true,
    maxRuns: 30,
    maxWorkers: 3,
    plateauThresholdRounds: 3,
    certainty: {
      hardware: 'explicit',
      correctness: 'explicit',
      precision: 'explicit',
      precisionMargin: 'inferred',
      p0Target: 'explicit',
      p1Regression: 'default',
      workspace: 'inferred',
      sparseBucket: 'default',
      topology: 'inferred',
    },
  };

  const CONTRACT_V2_PATCH = {
    version: 'v3',
    epochNum: 3,
    p0TargetLatencyUs: 220.0,
    p0TargetSpeedup: 482.0 / 220.0,
    p1GuardrailEnabled: true,
    p1MaxRegressionPct: 2.0,
    minPrecisionMarginPct: 20,
    maxWorkspaceKB: 64,
    sparseCoverageMode: '5% / 20% / 50% + [+64 尾块]',
    certainty: {
      hardware: 'explicit',
      correctness: 'explicit',
      precision: 'explicit',
      precisionMargin: 'explicit',
      p0Target: 'explicit',
      p1Regression: 'explicit',
      workspace: 'explicit',
      sparseBucket: 'inferred',
      topology: 'inferred',
    },
  };

  // Swimlane Track Metadata (Open-Canvas Rows: Spine + 4 Official PyPTO Categories)
  const SWIMLANE_ROWS = [
    { id: 'SPINE', name: '最优主干', sub: '6/6 合规晋升版本', glyphPath: GLYPH_PATHS.spine, axis: null },
    { id: 'incore', name: 'InCore 调优', sub: '双缓冲 · Tile · 快路径', glyphPath: GLYPH_PATHS.matmul, axis: 'incore' },
    { id: 'memory', name: '内存与 Ring', sub: 'Vec/Mat · Scope Ring', glyphPath: GLYPH_PATHS.cache, axis: 'memory' },
    { id: 'dispatch', name: '编排与派发', sub: 'pl.spmd 24 CG 均衡', glyphPath: GLYPH_PATHS.router, axis: 'dispatch' },
    { id: 'host', name: 'Host 常驻', sub: '未触发 · HBM 常驻', glyphPath: GLYPH_PATHS.exchange, axis: 'host', emptyNote: '输入 Tensor 与稀疏索引表已常驻 Device HBM · 0 次尝试' },
  ];

  const BASE_LANES = [
    {
      id: 'lane-incore',
      slotTag: '轨道 1',
      name: 'InCore 调优',
      axis: 'incore',
      statusByAct: { 0: 'idle', 1: 'plateau', 2: 'pruned', 3: 'pruned', 4: 'converged' },
      budgetMode: 'normal',
      hypothesis: '调整 Double Buffer 与 Mat/Vec Tile 切分掩盖 MTE2 搬运，并为小 Shape 提供直通分支',
      pruneReason: 'Wave 3–5 样本包络重叠 (Δ<1.5%)，主瓶颈在 24 核组分核不均',
    },
    {
      id: 'lane-memory',
      slotTag: '轨道 2',
      name: '内存与 Ring',
      axis: 'memory',
      statusByAct: { 0: 'idle', 1: 'active', 2: 'paused', 3: 'paused', 4: 'archived' },
      budgetMode: 'normal',
      hypothesis: '复用 Vec/Mat/Acc 片上空间与 scope-depth ring 槽位降低 HBM 中间写回',
      pruneReason: null,
    },
    {
      id: 'lane-dispatch',
      slotTag: '轨道 3',
      name: '编排与派发',
      axis: 'dispatch',
      statusByAct: { 0: 'hidden', 1: 'hidden', 2: 'running', 3: 'active', 4: 'archived' },
      budgetMode: 'boost',
      hypothesis: '对 pl.spmd(NUM_QK_CORES) 引入错峰轮转映射，消除 12慢/12快核组同余碰撞',
      pruneReason: null,
    },
    {
      id: 'lane-host',
      slotTag: '轨道 4',
      name: 'Host 常驻',
      axis: 'host',
      statusByAct: { 0: 'idle', 1: 'idle', 2: 'idle', 3: 'idle', 4: 'idle' },
      budgetMode: 'cap',
      hypothesis: '检查 Host Tensor 拷贝与 pinned memory 常驻（当前算子已满足 Device 常驻，无需派单）',
      pruneReason: null,
    },
  ];

  // Enriched Sequential Experiments (#0 .. #11) with Scoped Identity & Evidence Metadata
  const BASE_RUNS = [
    {
      scope: CASE_CONTRACT.scope,
      uid: 'exp-0',
      id: '#0',
      seq: 0,
      round: 0,
      waveSubSlot: 0,
      shortLabel: '基线',
      bullseyeQuadrant: 'NW',
      bullseyeAngleOffset: -45,
      laneId: 'SPINE',
      swimRow: 'SPINE',
      axis: 'dispatch',
      actIntro: 0,
      title: '基线实现 · 静态 stride 24 核组映射',
      p0Latency: 482.0,
      latencyRangeUs: [476.2, 489.4],
      p0MidLatency: 194.2,
      p1Latency: 3.25,
      p1RegressionPct: 0.0,
      tailRegressionPct: 0.0,
      rooflinePct: 34.2,
      vecKB: 96,
      matKB: 128,
      workspaceKB: 0,
      onChipBind: 'Vec 96KB · Mat 128KB · Acc FP32 32KB',
      precisionMarginPct: 86,
      maxRelErr: '1.1e-3',
      compileSec: 18.4,
      coreSkew: 2.50,
      evidenceGrade: 'observed',
      sourceRef: {
        file: 'kernels/attn/decode_sparse_attn_csa.py',
        lines: 'L164-L182',
        revision: CASE_CONTRACT.revision,
      },
      comparability: {
        contractVersion: 'v1 基线',
        shapeScope: '[32, 4096] · [1, 128] · [+64 尾块]',
        comparableToBaseline: true,
        note: '固定提交基线 · 20 次 NPU device_wall_us 同口径采样',
      },
      counters: { aiv_time: 4.47, core_skew: 2.50, vec_resc_cflt: 121, mte2: 7.6, mte3: 9.2, icache_miss: 14.0 },
      buildOk: true,
      testsOk: true,
      topologyOk: true,
      promotedSpine: true,
      summary: '静态 stride 分核 (qk_core + qk_it * 24)，CG 0~11 均值 382μs 而 CG 12~23 均值 153μs（核组极差 2.50×）。',
      codeSnippet: `with pl.spmd(NUM_QK_CORES, name_hint="qk_pv") as qk_tid:\n    qk_core = pl.tile.get_block_idx()\n    for qk_it in pl.range(qk_lane_iters):\n        qk_flat = qk_core + qk_it * NUM_QK_CORES\n        qk_item = pl.cast(pl.read(qk_order, [qk_flat]), pl.INDEX)`,
    },
    {
      scope: CASE_CONTRACT.scope,
      uid: 'exp-1',
      id: '#1',
      seq: 1,
      round: 1,
      waveSubSlot: 0,
      shortLabel: 'QK_M=32 双缓冲',
      bullseyeQuadrant: 'NW',
      bullseyeAngleOffset: -158,
      laneId: 'lane-incore',
      swimRow: 'incore',
      axis: 'incore',
      parentId: '#0',
      edgeKind: 'confirmed',
      actIntro: 1,
      title: 'QK_M_TILE=32 双级切分',
      p0Latency: 436.4,
      latencyRangeUs: [431.0, 442.8],
      p0MidLatency: 176.5,
      p1Latency: 3.24,
      p1RegressionPct: -0.3,
      tailRegressionPct: 0.1,
      rooflinePct: 37.8,
      vecKB: 128,
      matKB: 192,
      workspaceKB: 12,
      onChipBind: 'Vec 128KB · Mat 192KB · Left/Right 双槽',
      precisionMarginPct: 84,
      maxRelErr: '1.2e-3',
      compileSec: 19.1,
      coreSkew: 2.48,
      evidenceGrade: 'observed',
      sourceRef: {
        file: 'kernels/attn/decode_sparse_attn_csa.py',
        lines: 'L190-L214',
        revision: CASE_CONTRACT.revision,
      },
      comparability: {
        contractVersion: 'v1 → v3 可比',
        shapeScope: '[32, 4096] · [1, 128]',
        comparableToBaseline: true,
        note: '与 #0 同计时口径，仅改变 InCore QK_M_TILE 分块粒度',
      },
      counters: { aiv_time: 4.18, core_skew: 2.48, vec_resc_cflt: 114, mte2: 8.8, mte3: 8.9, icache_miss: 13.8 },
      buildOk: true,
      testsOk: true,
      topologyOk: true,
      summary: '扩大 QK_M_TILE 提升 Mat 缓存命中，主 Shape 耗时降至 436.4μs (-9.5%)，但 24 核组极差仍为 2.48×。',
    },
    {
      scope: CASE_CONTRACT.scope,
      uid: 'exp-2',
      id: '#2',
      seq: 2,
      round: 1,
      waveSubSlot: 0,
      shortLabel: 'Ring 槽位复用',
      bullseyeQuadrant: 'NE',
      bullseyeAngleOffset: -64,
      laneId: 'lane-memory',
      swimRow: 'memory',
      axis: 'memory',
      parentId: '#0',
      edgeKind: 'confirmed',
      actIntro: 1,
      title: 'Vec/Mat 缓冲 scope-depth ring 复用',
      p0Latency: 452.0,
      latencyRangeUs: [446.5, 458.2],
      p0MidLatency: 184.0,
      p1Latency: 3.22,
      p1RegressionPct: -0.9,
      tailRegressionPct: -0.4,
      rooflinePct: 36.5,
      vecKB: 88,
      matKB: 128,
      workspaceKB: 8,
      onChipBind: 'Vec 88KB (Ring=2) · Mat 128KB · Acc FP32',
      precisionMarginPct: 85,
      maxRelErr: '1.1e-3',
      compileSec: 18.8,
      coreSkew: 2.49,
      evidenceGrade: 'observed',
      sourceRef: {
        file: 'kernels/attn/decode_sparse_attn_csa.py',
        lines: 'L108-L132',
        revision: CASE_CONTRACT.revision,
      },
      comparability: {
        contractVersion: 'v1 → v3 可比',
        shapeScope: '[32, 4096] · [1, 128]',
        comparableToBaseline: true,
        note: '与 #0 同计时口径，收紧 Vec/Mat 生命周期复用槽位',
      },
      counters: { aiv_time: 4.30, core_skew: 2.49, vec_resc_cflt: 118, mte2: 8.1, mte3: 9.0, icache_miss: 13.7 },
      buildOk: true,
      testsOk: true,
      topologyOk: true,
      summary: '启用 scope-depth ring 消除一次中间 HBM 回写，Vec 降至 88KB，端到端耗时 452.0μs (-6.2%)。',
    },
    {
      scope: CASE_CONTRACT.scope,
      uid: 'exp-3',
      id: '#3',
      seq: 3,
      round: 2,
      waveSubSlot: 0,
      shortLabel: 'K=128 双缓冲预取',
      bullseyeQuadrant: 'NW',
      bullseyeAngleOffset: -142,
      laneId: 'lane-incore',
      swimRow: 'incore',
      axis: 'incore',
      parentId: '#1',
      edgeKind: 'confirmed',
      actIntro: 1,
      title: 'ATTN_K_TILE=128 双缓冲预取',
      p0Latency: 398.0,
      latencyRangeUs: [392.4, 404.1],
      p0MidLatency: 162.8,
      p1Latency: 3.26,
      p1RegressionPct: 0.3,
      tailRegressionPct: 0.4,
      rooflinePct: 41.4,
      vecKB: 148,
      matKB: 256,
      workspaceKB: 24,
      onChipBind: 'Vec 148KB · Mat 256KB (stage=2) · Acc FP32',
      precisionMarginPct: 81,
      maxRelErr: '1.5e-3',
      compileSec: 20.3,
      coreSkew: 2.46,
      evidenceGrade: 'observed',
      sourceRef: {
        file: 'kernels/attn/decode_sparse_attn_csa.py',
        lines: 'L215-L246',
        revision: CASE_CONTRACT.revision,
      },
      comparability: {
        contractVersion: 'v1 → v3 可比',
        shapeScope: '[32, 4096] · [1, 128] · [+64 尾块]',
        comparableToBaseline: true,
        note: '阶段合规最优（未达 ≤220μs 终态目标），晋升主干作为后续分支锚点',
      },
      counters: { aiv_time: 3.86, core_skew: 2.46, vec_resc_cflt: 108, mte2: 10.2, mte3: 8.6, icache_miss: 13.6 },
      buildOk: true,
      testsOk: true,
      topologyOk: true,
      promotedSpine: true,
      summary: '开启 pl.pipeline(stage=2) 与 Mat KV 预取，较 #0 提速 1.21× (398.0μs)，晋升为早期主干锚点。',
      codeSnippet: `for qk_hb in pl.pipeline(H // QK_M_TILE, stage=2):\n    qk_q_tile = q_flat[qk_head_row : qk_head_row + QK_M_TILE, 0 : HEAD_DIM]\n    qk_raw = pl.matmul(qk_q_tile, qk_kv, b_trans=True, out_dtype=pl.FP32)`,
    },
    {
      scope: CASE_CONTRACT.scope,
      uid: 'exp-4',
      id: '#4',
      seq: 4,
      round: 2,
      waveSubSlot: 0,
      shortLabel: 'Acc 降精度越界',
      bullseyeQuadrant: 'NE',
      bullseyeAngleOffset: -42,
      laneId: 'lane-memory',
      swimRow: 'memory',
      axis: 'memory',
      parentId: '#2',
      edgeKind: 'confirmed',
      actIntro: 1,
      title: 'Acc 提前降精度至 BF16（Golden 越界回退）',
      p0Latency: 410.0,
      latencyRangeUs: [404.8, 416.5],
      p0MidLatency: 168.0,
      p1Latency: 3.21,
      p1RegressionPct: -1.2,
      tailRegressionPct: -0.6,
      rooflinePct: 40.2,
      vecKB: 84,
      matKB: 128,
      workspaceKB: 8,
      onChipBind: 'Vec 84KB · Mat 128KB · Acc BF16 (违例)',
      precisionMarginPct: -42,
      maxRelErr: '1.9e-2 (>7.8e-3)',
      compileSec: 18.6,
      coreSkew: 2.48,
      evidenceGrade: 'observed',
      sourceRef: {
        file: 'kernels/attn/decode_sparse_attn_csa.py',
        lines: 'L288-L299',
        revision: CASE_CONTRACT.revision,
      },
      comparability: {
        contractVersion: '正确性/精度门禁拦截',
        shapeScope: '[32, 4096] Golden 对拍失败',
        comparableToBaseline: false,
        note: '设备端虽完成计时 (410.0μs)，但 rtol=1.9e-2 超出 1/128 红线，不纳入有效候选',
      },
      counters: { aiv_time: 3.92, core_skew: 2.48, vec_resc_cflt: 102, mte2: 8.4, mte3: 8.8, icache_miss: 13.5 },
      buildOk: true,
      testsOk: false,
      failReason: 'Golden 对拍 rtol=1.9e-2 > 7.8e-3（长尾指数溢出，自动退回 #2）',
      failShort: '精度越界',
      rollbackTo: '#2',
      topologyOk: true,
      summary: '跳过 FP32 Acc 累加直接截断为 BF16，引发 softmax 分母下溢，触发正确性与精度门禁拦截。',
    },
    {
      scope: CASE_CONTRACT.scope,
      uid: 'exp-5',
      id: '#5',
      seq: 5,
      round: 3,
      waveSubSlot: -1,
      shortLabel: '三级预取 Vec 越界',
      bullseyeQuadrant: 'NW',
      bullseyeAngleOffset: -168,
      laneId: 'lane-incore',
      swimRow: 'incore',
      axis: 'incore',
      parentId: '#3',
      edgeKind: 'confirmed',
      actIntro: 1,
      title: 'stage=3 激进预取（编译期 Vec UB 容量越界）',
      p0Latency: null,
      latencyRangeUs: null,
      p0MidLatency: null,
      p1Latency: null,
      p1RegressionPct: null,
      tailRegressionPct: null,
      rooflinePct: null,
      vecKB: 234,
      matKB: 320,
      workspaceKB: 180,
      onChipBind: 'Vec 234KB / 192KB (+42KB 越界) · 未上板',
      precisionMarginPct: null,
      maxRelErr: '未执行',
      compileSec: 9.2,
      coreSkew: null,
      evidenceGrade: 'derived',
      sourceRef: {
        file: 'kernels/attn/decode_sparse_attn_csa.py',
        lines: 'L215-L231',
        revision: CASE_CONTRACT.revision,
      },
      comparability: {
        contractVersion: '硬件门禁静态拦截',
        shapeScope: '编译期静态缓冲分析 (Pass: CheckUBCapacity)',
        comparableToBaseline: false,
        note: 'PyPTO 编译器静态推导 Vec 峰值驻留 234KB > 910B2 单核 192KB 上限，未生成设备耗时',
      },
      counters: null,
      buildOk: false,
      testsOk: false,
      failReason: '编译期静态推导 Vec (UB) 234KB > 192KB 上限 (+42KB 越界，退回 #3)',
      failShort: 'Vec 234KB 越界',
      rollbackTo: '#3',
      topologyOk: true,
      summary: '三级流水同时驻留 3 份 FP32 得分 Tile 导致 Vec 静态需求达 234KB（超出 192KB 硬件上限 42KB）。',
    },
    {
      scope: CASE_CONTRACT.scope,
      uid: 'exp-6',
      id: '#6',
      seq: 6,
      round: 3,
      waveSubSlot: 1,
      shortLabel: '偏置折叠',
      bullseyeQuadrant: 'NW',
      bullseyeAngleOffset: -124,
      laneId: 'lane-incore',
      swimRow: 'incore',
      axis: 'incore',
      parentId: '#3',
      edgeKind: 'confirmed',
      actIntro: 1,
      title: 'col_expand_add 偏置折叠流水重排',
      p0Latency: 394.6,
      latencyRangeUs: [389.8, 400.2],
      p0MidLatency: 161.2,
      p1Latency: 3.27,
      p1RegressionPct: 0.6,
      tailRegressionPct: 0.7,
      rooflinePct: 41.8,
      vecKB: 152,
      matKB: 256,
      workspaceKB: 24,
      onChipBind: 'Vec 152KB · Mat 256KB · Bias 折叠',
      precisionMarginPct: 80,
      maxRelErr: '1.5e-3',
      compileSec: 20.5,
      coreSkew: 2.47,
      evidenceGrade: 'observed',
      sourceRef: {
        file: 'kernels/attn/decode_sparse_attn_csa.py',
        lines: 'L248-L265',
        revision: CASE_CONTRACT.revision,
      },
      comparability: {
        contractVersion: 'v1 → v3 可比',
        shapeScope: '[32, 4096] · 样本包络 [389.8, 400.2]μs',
        comparableToBaseline: true,
        note: '均值较 #3 仅差 3.4μs (0.85%)，完全落入 #3 [392.4, 404.1]μs 观测波动区间',
      },
      counters: { aiv_time: 3.81, core_skew: 2.47, vec_resc_cflt: 106, mte2: 10.4, mte3: 8.6, icache_miss: 13.7 },
      buildOk: true,
      testsOk: true,
      topologyOk: true,
      inPlateau: true,
      summary: '折叠稀疏掩码偏置加法，耗时 394.6μs（样本 [389.8, 400.2]μs 与 #3 重叠，开始进入局部停滞）。',
    },
    {
      scope: CASE_CONTRACT.scope,
      uid: 'exp-7',
      id: '#7',
      seq: 7,
      round: 3,
      waveSubSlot: 0,
      shortLabel: '原位写回',
      bullseyeQuadrant: 'NE',
      bullseyeAngleOffset: -22,
      laneId: 'lane-memory',
      swimRow: 'memory',
      axis: 'memory',
      parentId: '#2',
      edgeKind: 'confirmed',
      actIntro: 1,
      title: 'inverse-RoPE 片上原位写回',
      p0Latency: 421.5,
      latencyRangeUs: [416.0, 427.6],
      p0MidLatency: 171.8,
      p1Latency: 3.24,
      p1RegressionPct: -0.3,
      tailRegressionPct: 0.2,
      rooflinePct: 39.1,
      vecKB: 80,
      matKB: 128,
      workspaceKB: 16,
      onChipBind: 'Vec 80KB · Mat 128KB · 原位覆盖',
      precisionMarginPct: 16,
      maxRelErr: '6.5e-3',
      compileSec: 19.6,
      coreSkew: 2.45,
      evidenceGrade: 'observed',
      sourceRef: {
        file: 'kernels/attn/decode_sparse_attn_csa.py',
        lines: 'L276-L298',
        revision: CASE_CONTRACT.revision,
      },
      comparability: {
        contractVersion: 'v3 精度裕度违例 (<20%)',
        shapeScope: '[32, 4096] · [1, 128]',
        comparableToBaseline: true,
        note: '虽节省 Vec 至 80KB，但精度裕度降至 16%（低于 v3 ≥20% 要求）',
      },
      counters: { aiv_time: 4.02, core_skew: 2.45, vec_resc_cflt: 111, mte2: 8.7, mte3: 8.4, icache_miss: 13.4 },
      buildOk: true,
      testsOk: true,
      topologyOk: true,
      summary: '压缩 Vec 中转槽位至 80KB (421.5μs)，但提前舍入使极端值精度裕度降至 16%（低于 v3 20% 下限）。',
    },
    {
      scope: CASE_CONTRACT.scope,
      uid: 'exp-8',
      id: '#8',
      seq: 8,
      round: 4,
      waveSubSlot: 0,
      shortLabel: 'Tile 8↔16 回摆',
      bullseyeQuadrant: 'NW',
      bullseyeAngleOffset: -110,
      laneId: 'lane-incore',
      swimRow: 'incore',
      axis: 'incore',
      parentId: '#6',
      edgeKind: 'confirmed',
      actIntro: 1,
      title: 'BIAS_T_TILE 8↔16 参数回摆',
      p0Latency: 396.2,
      latencyRangeUs: [391.2, 401.5],
      p0MidLatency: 162.0,
      p1Latency: 3.28,
      p1RegressionPct: 0.9,
      tailRegressionPct: 1.1,
      rooflinePct: 41.6,
      vecKB: 156,
      matKB: 256,
      workspaceKB: 24,
      onChipBind: 'Vec 156KB · Mat 256KB · BIAS_T_TILE=16',
      precisionMarginPct: 80,
      maxRelErr: '1.5e-3',
      compileSec: 20.6,
      coreSkew: 2.47,
      evidenceGrade: 'observed',
      sourceRef: {
        file: 'kernels/attn/decode_sparse_attn_csa.py',
        lines: 'L250-L270',
        revision: CASE_CONTRACT.revision,
      },
      comparability: {
        contractVersion: 'v1 → v3 可比',
        shapeScope: '[32, 4096] · 样本包络 [391.2, 401.5]μs',
        comparableToBaseline: true,
        note: '与 #6 [389.8, 400.2]μs 包络完全重叠，参数回摆无实质增益',
      },
      counters: { aiv_time: 3.84, core_skew: 2.47, vec_resc_cflt: 107, mte2: 10.3, mte3: 8.6, icache_miss: 13.7 },
      buildOk: true,
      testsOk: true,
      topologyOk: true,
      inPlateau: true,
      oscillatesWith: '#6',
      summary: '在 BIAS_T_TILE=8 与 16 间切换得 396.2μs [391.2, 401.5]μs，与 #6 样本区间重叠。',
    },
    {
      scope: CASE_CONTRACT.scope,
      uid: 'exp-9',
      id: '#9',
      seq: 9,
      round: 5,
      waveSubSlot: 0,
      shortLabel: 'MTE2 预取微调',
      bullseyeQuadrant: 'NW',
      bullseyeAngleOffset: -96,
      laneId: 'lane-incore',
      swimRow: 'incore',
      axis: 'incore',
      parentId: '#8',
      edgeKind: 'confirmed',
      actIntro: 1,
      title: 'MTE2 预取距离微调（连续 3 轮 Δ<1.5%）',
      p0Latency: 393.1,
      latencyRangeUs: [388.6, 399.4],
      p0MidLatency: 160.9,
      p1Latency: 3.28,
      p1RegressionPct: 0.9,
      tailRegressionPct: 1.2,
      rooflinePct: 41.9,
      vecKB: 154,
      matKB: 256,
      workspaceKB: 24,
      onChipBind: 'Vec 154KB · Mat 256KB · MTE2 预取距离=2',
      precisionMarginPct: 79,
      maxRelErr: '1.6e-3',
      compileSec: 20.8,
      coreSkew: 2.46,
      evidenceGrade: 'observed',
      sourceRef: {
        file: 'kernels/attn/decode_sparse_attn_csa.py',
        lines: 'L220-L246',
        revision: CASE_CONTRACT.revision,
      },
      comparability: {
        contractVersion: 'v1 → v3 可比（收益停滞终点）',
        shapeScope: '[32, 4096] · 样本包络 [388.6, 399.4]μs',
        comparableToBaseline: true,
        note: '#6 (395μs) / #8 (396μs) / #9 (393μs) 三轮均值极差 <1.5% 且包络重叠，证实瓶颈不在 InCore',
      },
      counters: { aiv_time: 3.79, core_skew: 2.46, vec_resc_cflt: 105, mte2: 10.6, mte3: 8.6, icache_miss: 13.8 },
      buildOk: true,
      testsOk: true,
      topologyOk: true,
      inPlateau: true,
      plateauTerminal: true,
      summary: '连续 3 轮均值 394.6→396.2→393.1μs 落入同一 [389, 401]μs 波动包络，Perf Lane 揭示 24 CG 极差仍达 2.46×。',
    },
    {
      scope: CASE_CONTRACT.scope,
      uid: 'exp-10',
      id: '#10',
      seq: 10,
      round: 6,
      waveSubSlot: 0,
      shortLabel: '错峰轮转分核',
      bullseyeQuadrant: 'SE',
      bullseyeAngleOffset: 42,
      humanVia: '反证转编排',
      laneId: 'lane-dispatch',
      swimRow: 'dispatch',
      axis: 'dispatch',
      parentId: '#9',
      logicalBaseId: '#3',
      edgeKind: 'refutation',
      actIntro: 3,
      runningInAct: 2,
      title: '24 核组错峰轮转分核（206μs · v3 尾块与小Shape违例）',
      p0Latency: 206.1,
      latencyRangeUs: [202.8, 210.4],
      p0MidLatency: 96.4,
      p1Latency: 3.43,
      p1RegressionPct: 5.5,
      tailRegressionPct: 5.5,
      rooflinePct: 79.6,
      vecKB: 168,
      matKB: 256,
      workspaceKB: 64,
      onChipBind: 'Vec 168KB · Mat 256KB · 全局 qk_order 压实表 (+64KB WS)',
      precisionMarginPct: 21,
      maxRelErr: '6.1e-3',
      compileSec: 27.6,
      coreSkew: 1.08,
      evidenceGrade: 'observed',
      sourceRef: {
        file: 'kernels/attn/decode_sparse_attn_csa.py',
        lines: 'L168-L179',
        revision: CASE_CONTRACT.revision,
      },
      comparability: {
        contractVersion: 'v1 达标 → v3 [+64 尾块] 泛化违例',
        shapeScope: '[32, 4096] 206.1μs ✓ / [1, 128] & [+64 尾块] +5.5% ✗',
        comparableToBaseline: true,
        note: '主 Shape 进入 ≤220μs 门禁圈，但全局压实同步在 [1,128] 与 +64 尾块引入 +5.5% 退化（超 ≤2.0% 上限）',
      },
      counters: { aiv_time: 2.04, core_skew: 1.08, vec_resc_cflt: 84, mte2: 11.8, mte3: 7.9, icache_miss: 13.2 },
      buildOk: true,
      testsOk: true,
      topologyOk: true,
      hasHiddenRisk: true,
      c1Best: true,
      summary: 'rot_core=(qk_core+qk_it)%24 消除双峰碰撞，主耗时骤降至 206.1μs (2.34×)；但 v3 尾块与 [1,128] 退化 +5.5%。',
      codeSnippet: `with pl.spmd(NUM_QK_CORES, name_hint="qk_pv", deps=[qk_plan_tid]) as qk_tid:\n    qk_core = pl.tile.get_block_idx()\n    for qk_it in pl.range(qk_lane_iters):\n        rot_core = (qk_core + qk_it) % NUM_QK_CORES\n        qk_flat = rot_core + qk_it * NUM_QK_CORES\n        qk_item = pl.cast(pl.read(qk_order, [qk_flat]), pl.INDEX)`,
    },
    {
      scope: CASE_CONTRACT.scope,
      uid: 'exp-11',
      id: '#11',
      seq: 11,
      round: 7,
      waveSubSlot: 0,
      shortLabel: '快路径收敛',
      bullseyeQuadrant: 'NW',
      bullseyeAngleOffset: -45,
      humanVia: '契约分支 +快路径',
      laneId: 'lane-incore',
      swimRow: 'incore',
      axis: 'incore',
      parentId: '#10',
      edgeKind: 'contract-fork',
      actIntro: 4,
      runningInAct: 3,
      title: '错峰分核 + t_dim≤4 直通快路径（6/6 全门禁通过）',
      p0Latency: 214.0,
      latencyRangeUs: [210.6, 217.8],
      p0MidLatency: 94.8,
      p1Latency: 3.19,
      p1RegressionPct: -1.8,
      tailRegressionPct: -0.9,
      rooflinePct: 76.8,
      vecKB: 136,
      matKB: 224,
      workspaceKB: 32,
      onChipBind: 'Vec 136KB · Mat 224KB · Ring 复用 WS +32KB · Acc FP32',
      precisionMarginPct: 74,
      maxRelErr: '2.0e-3',
      compileSec: 21.2,
      coreSkew: 1.08,
      evidenceGrade: 'observed',
      sourceRef: {
        file: 'kernels/attn/decode_sparse_attn_csa.py',
        lines: 'L114-L126',
        revision: CASE_CONTRACT.revision,
      },
      comparability: {
        contractVersion: 'v3 [+64 尾块] 6/6 全通过',
        shapeScope: '[32, 4096] · [4, 2048] · [1, 128] · [+64 尾块]',
        comparableToBaseline: true,
        note: '全 Shape 簇与 20 次设备采样同口径验证通过，当前推荐合入基线',
      },
      counters: { aiv_time: 2.11, core_skew: 1.08, vec_resc_cflt: 82, mte2: 11.4, mte3: 7.8, icache_miss: 12.1 },
      buildOk: true,
      testsOk: true,
      topologyOk: true,
      promotedSpine: true,
      summary: '保留 24 核组错峰轮转 (1.08×)，新增 t_dim<=4 直通快路径与尾块掩码保护，[32,4096] 214.0μs (2.25×)，[1,128] -1.8%，6/6 全通过。',
      codeSnippet: `if t_dim <= 4:\n    qk_item = pl.cast(qk_core + qk_it * NUM_QK_CORES, pl.INDEX)\nelse:\n    rot_core = (qk_core + qk_it) % NUM_QK_CORES\n    qk_item = pl.cast(pl.read(qk_order, [rot_core + qk_it * NUM_QK_CORES]), pl.INDEX)`,
    },
  ];

  // 4. Cohesive Solid Decision Cards for Exception Audit Drawer (approved-ui-and-repair-flow.md)
  // One Decision = One Complete #202020 Solid Card (context + stacked options with micro preview + inline 推荐 + bottom primary button)
  const INITIAL_DECISION_CARDS = [
    {
      id: 'cand-incore-stall',
      code: 'CAND-01',
      title: '#6–#9 InCore 收益停滞（样本包络重叠 Δ < 1.5%）',
      targetRunIds: ['#3', '#6', '#8', '#9'],
      focusRunId: '#9',
      evidenceGrade: 'observed',
      sourceAnchor: 'decode_sparse_attn_csa.py:L168-L179 · Perf Lane 24 CG 采集',
      context: 'Wave 3–5 连续 3 轮均值 394.6 → 396.2 → 393.1 μs 完全落入 [388.6, 401.5] μs 样本波动包络；同时 CG 0~11 (382μs) 与 CG 12~23 (153μs) 存在 2.46× 双峰倾斜。',
      selectedOptionId: 'opt-pivot-dispatch',
      status: 'open',
      appliedNote: null,
      options: [
        {
          id: 'opt-pivot-dispatch',
          title: '停止 InCore 微调，转向 pl.spmd 错峰轮转分核',
          recommended: true,
          structure: 'SPMD 轮转 (qk_core+it)%24',
          binding: '24 CG 极差 2.46× → 1.08×',
          outcome: 'PASS · 206.1 μs',
          outcomeKind: 'pass',
        },
        {
          id: 'opt-stage3-prefetch',
          title: '继续加码 InCore 三级流水预取 (stage=3)',
          recommended: false,
          structure: 'pl.pipeline(stage=3)',
          binding: 'Vec 234 KB > 192 KB 上限',
          outcome: '仍 FAIL · UB OOM',
          outcomeKind: 'fail',
        },
      ],
      premise: '依据 20 次设备采样包络与 24 核组 Perf Lane 实测核对',
      actionLabel: '定位并复核编排转向 (#9 → #10)',
    },
    {
      id: 'cand-v3-tail',
      code: 'CAND-02',
      title: '#10 契约 v3 尾块与 [1, 128] 泛化退化 (+5.5%)',
      targetRunIds: ['#10', '#11'],
      focusRunId: '#10',
      evidenceGrade: 'observed',
      sourceAnchor: 'decode_sparse_attn_csa.py:L114-L126 · 多 Shape 矩阵实测',
      context: '#10 将 [32, 4096] 推进至 206.1 μs (≤220μs)，但两趟全局 qk_order 压实同步使 [1, 128] 与 valid_shape [+64 尾块] 退化 +5.5%，超出 v3 ≤2.0% 门禁。',
      selectedOptionId: 'opt-fastpath-ring',
      status: 'open',
      appliedNote: null,
      options: [
        {
          id: 'opt-fastpath-ring',
          title: '引入 t_dim≤4 直通快路径 + Ring 槽位复用 (#11)',
          recommended: true,
          structure: 'if t_dim<=4 免屏障直通',
          binding: 'Vec 136KB · WS 64→32KB',
          outcome: 'PASS · 214μs (6/6)',
          outcomeKind: 'pass',
        },
        {
          id: 'opt-waive-v3',
          title: '保留 #10 全局压实并申请离线大批量豁免',
          recommended: false,
          structure: '维持单一路径无分支',
          binding: '[1,128] 仍退化 +5.5%',
          outcome: '仍 FAIL · 泛化违例',
          outcomeKind: 'fail',
        },
      ],
      premise: '依据 v3 契约 [32,4096] + [1,128] + [+64 尾块] 联合回归实测',
      actionLabel: '采纳快路径方案并核对 #11',
    },
    {
      id: 'cand-contract-pending',
      code: 'CAND-03',
      title: '目标与约束 2 项推断规则待显式确认',
      targetRunIds: ['#10', '#11'],
      focusRunId: '#11',
      evidenceGrade: 'derived',
      sourceAnchor: '目标与约束 · 4. 泛化 / 6. 规范 (certainty: inferred)',
      context: '自然语言输入未显式指定「5% / 20% / 50% 多稀疏度分桶 + [+64 尾块]」与「对外算子签名锁定」，当前作为 Agent 推断规则参评。',
      selectedOptionId: 'opt-confirm-explicit',
      status: 'open',
      appliedNote: null,
      options: [
        {
          id: 'opt-confirm-explicit',
          title: '转为明确锁定规则并保持 v3 全量门禁生效',
          recommended: true,
          structure: '4.泛化 + 6.规范 → 已指定',
          binding: '8 组边界 Fixture 常驻门禁',
          outcome: 'PASS · 契约锁定',
          outcomeKind: 'pass',
        },
        {
          id: 'opt-relax-v1',
          title: '退回 v1 仅校验 20% 均匀稀疏主 Shape',
          recommended: false,
          structure: '移除尾块与偏斜稀疏门禁',
          binding: '#10 尾块退化漏检风险',
          outcome: '仍 FAIL · 覆盖缺口',
          outcomeKind: 'fail',
        },
      ],
      premise: '确认后同步更新「目标与约束」确定性状态与导出契约',
      actionLabel: '一键确认推断约束项',
    },
  ];

  // 5. Semantic Code Diff Hunks
  const CODE_DIFF_HUNKS = [
    {
      id: 'hunk-dispatch',
      axis: 'dispatch',
      title: 'qk_pv 24 核组错峰轮转分核映射',
      file: 'kernels/attn/decode_sparse_attn_csa.py:168-179',
      stats: '+5 / -2',
      impact: '24 CG 极差 2.50× → 1.08× · [32,4096] -57.2%',
      appliesTo: ['#10', '#11'],
      lines: [
        { type: 'ctx', ln: 168, text: 'with pl.spmd(NUM_QK_CORES, name_hint="qk_pv", deps=[qk_plan_tid, cache_ready_dep]) as qk_tid:' },
        { type: 'ctx', ln: 169, text: '    qk_core = pl.tile.get_block_idx()' },
        { type: 'ctx', ln: 170, text: '    for qk_it in pl.range(qk_lane_iters):' },
        { type: 'del', ln: 171, text: '        qk_flat = qk_core + qk_it * NUM_QK_CORES' },
        { type: 'del', ln: 172, text: '        qk_item = pl.cast(pl.read(qk_order, [qk_flat]), pl.INDEX)' },
        { type: 'add', ln: 171, text: '        # 错峰轮转消除 CG 0~11 与 CG 12~23 同余碰撞，均衡稀疏块负载' },
        { type: 'add', ln: 172, text: '        rot_core = (qk_core + qk_it) % NUM_QK_CORES' },
        { type: 'add', ln: 173, text: '        qk_flat = rot_core + qk_it * NUM_QK_CORES' },
        { type: 'add', ln: 174, text: '        qk_item = pl.cast(pl.read(qk_order, [qk_flat]), pl.INDEX)' },
      ],
    },
    {
      id: 'hunk-incore-fastpath',
      axis: 'incore',
      title: 't_dim ≤ 4 小 Shape 直通快路径与 [+64 尾块] 掩码保护',
      file: 'kernels/attn/decode_sparse_attn_csa.py:114-126',
      stats: '+8 / -2',
      impact: '[1,128] +5.5% ✗ → -1.8% ✓ · Vec 168→136KB · WS 64→32KB',
      appliesTo: ['#11'],
      lines: [
        { type: 'ctx', ln: 114, text: '    qk_order = pl.create_tensor([QK_ITEMS], dtype=pl.INT32)' },
        { type: 'del', ln: 115, text: '    # 统一执行两趟压实扫描，小批量与尾块引入固定同步屏障开销' },
        { type: 'del', ln: 116, text: '    sparse_bias = pl.create_tensor([t_dim, PADDED_TOPK], dtype=pl.FP32)' },
        { type: 'add', ln: 115, text: '    # 复用 scope-depth ring 槽位，额外 Workspace 由 64KB 压降至 32KB' },
        { type: 'add', ln: 116, text: '    sparse_bias = pl.create_tensor([t_dim, PADDED_TOPK], dtype=pl.FP32, reuse_slot=True)' },
        { type: 'add', ln: 117, text: '    if t_dim <= 4:  # InCore 小 Shape 直通快路径：免除 qk_plan 屏障与轮转表读' },
        { type: 'add', ln: 118, text: '        qk_plan_tid = pl.system.task_dummy(deps=[])' },
        { type: 'add', ln: 119, text: '        use_direct_spmd = True' },
      ],
    },
    {
      id: 'hunk-memory-precision',
      axis: 'memory',
      title: 'Acc 片上 FP32 累加保护与 rint 显式舍入',
      file: 'kernels/attn/decode_sparse_attn_csa.py:291-299',
      stats: '+3 / -2',
      impact: '精度裕度 21% → 74% · Max Rel Err 2.0e-3',
      appliesTo: ['#11'],
      lines: [
        { type: 'ctx', ln: 291, text: '        n_denom = pl.add(m_li, pl.exp(pl.sub(n_sink_tile, m_mi)))' },
        { type: 'del', ln: 292, text: '        n_full = pl.row_expand_div(pl.cast(m_oi, pl.BF16), n_denom)' },
        { type: 'add', ln: 292, text: '        # 保持 Acc FP32 完成归一化与 inverse-RoPE 后再 rint 转 BF16' },
        { type: 'add', ln: 293, text: '        n_full = pl.row_expand_div(m_oi, n_denom)[0 : H_TILE, 0 : HEAD_DIM]' },
        { type: 'add', ln: 294, text: '        n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")' },
      ],
    },
  ];

  const RISK_BUCKETS_FAST = [
    {
      num: 1,
      id: 'risk-shape',
      sev: 'high',
      title: '小 Shape 与 +64 尾块退化超限',
      metric: '[1,128] +5.5%',
      limit: '上限 ≤2.0%',
      meterType: 'needle',
      valuePct: 86,
      thresholdPct: 32,
      caption: '3.25 → 3.43 μs · 全局 qk_order 压实同步开销',
    },
    {
      num: 2,
      id: 'risk-precision',
      sev: 'med',
      title: '精度裕度贴近下限',
      metric: '裕度 21%',
      limit: '下限 ≥20%',
      meterType: 'gauge',
      valuePct: 21,
      thresholdPct: 20,
      caption: 'Max Rel Err 6.1e-3 (阈值 7.8e-3)',
    },
    {
      num: 3,
      id: 'risk-resource',
      sev: 'med',
      title: 'Workspace 触顶且 Vec 偏高',
      metric: '+64 KB · Vec 168KB',
      limit: '上限 64 KB',
      meterType: 'bar',
      valuePct: 100,
      thresholdPct: 80,
      caption: '编译 18.4s → 27.6s · Ring 槽位触顶',
    },
    {
      num: 4,
      id: 'risk-sparse',
      sev: 'med',
      title: '多稀疏度分桶覆盖不完整',
      metric: '1 / 4 桶已测',
      limit: '需全桶验证',
      meterType: 'dots',
      filledDots: 1,
      totalDots: 4,
      caption: '仅覆盖 20% 均匀稀疏 · 缺 5% / 50% 偏斜与尾块',
    },
  ];

  const COVERAGE_SHAPES = [
    { shape: '[32, 4096]', bucket: '主场景 · 稀疏 20%', status: 'PASS', seeds: '20 samples' },
    { shape: '[4, 2048]', bucket: '中批量 · 稀疏 20%', status: 'PASS', seeds: '20 samples' },
    { shape: '[1, 128]', bucket: '小 Shape 防回退', status: 'PASS', seeds: '20 samples' },
    { shape: 'valid_shape [+64 尾块]', bucket: '非对齐尾块掩码', status: 'PASS', seeds: 'Fixture' },
    { shape: '[16, 8192]', bucket: '长上下文序列', status: 'PASS', seeds: '8 seeds' },
    { shape: '全 -1 Mask', bucket: '空有效块保护', status: 'PASS', seeds: 'Fixture' },
    { shape: '5% / 50% 稀疏', bucket: '偏斜稀疏分桶', status: 'PASS', seeds: '8 seeds' },
    { shape: 'NaN / Inf 边界', bucket: 'NEG_INF 稳定性', status: 'PASS', seeds: 'Fixture' },
  ];

  // 6. Deterministic 6-Gate Contract Evaluation Engine
  function getTargetLatencyUs(contract) {
    if (contract && Number.isFinite(contract.p0TargetLatencyUs)) return contract.p0TargetLatencyUs;
    const sp = (contract && contract.p0TargetSpeedup) || (482.0 / 220.0);
    return 482.0 / sp;
  }

  function evaluateRun(run, contract) {
    const targetUs = getTargetLatencyUs(contract);
    if (run.isRunning || run.evidenceGrade === 'candidate') return {overall:'pending',badge:'候选待采样',dims:Object.fromEntries(GATE_KEYS.map(k=>[k,'pending'])),risks:['尚无编译与设备采样结果'],speedup:null,invalidUnderCurrent:false};
    const vecLimit = (contract && contract.vecLimitKB) || CASE_CONTRACT.vecLimitKB;

    if (!run.buildOk) {
      return {
        overall: 'fail',
        badge: '硬件越界',
        dims: {
          hardware: 'fail',
          correctness: 'pending',
          precision: 'pending',
          generalization: 'pending',
          performance: 'pending',
          engineering: 'pass',
        },
        risks: [run.failReason || `Vec (UB) ${run.vecKB}KB 超出 ${vecLimit}KB 硬件上限`],
        speedup: null,
        invalidUnderCurrent: false,
      };
    }

    if (!run.testsOk) {
      return {
        overall: 'fail',
        badge: '对拍失败',
        dims: {
          hardware: 'pass',
          correctness: 'fail',
          precision: 'fail',
          generalization: 'pending',
          performance: run.p0Latency && run.p0Latency <= targetUs ? 'pass' : 'fail',
          engineering: 'pass',
        },
        risks: [run.failReason || 'Golden 对拍或数值稳定性校验未通过'],
        speedup: run.p0Latency ? 482.0 / run.p0Latency : null,
        invalidUnderCurrent: false,
      };
    }

    const speedup = 482.0 / run.p0Latency;
    const speedPass = run.p0Latency <= targetUs + 1e-6;
    const wsPass = run.workspaceKB <= contract.maxWorkspaceKB;
    const vecPass = (run.vecKB || 0) <= vecLimit;

    const dims = {
      hardware: vecPass && run.workspaceKB <= 160 ? 'pass' : 'fail',
      correctness: run.testsOk ? 'pass' : 'fail',
      precision: 'pass',
      generalization: 'pass',
      performance: speedPass && wsPass ? 'pass' : 'fail',
      engineering: run.topologyOk ? 'pass' : 'fail',
    };
    const risks = [];

    if (!vecPass) {
      risks.push(`片上 Vec ${run.vecKB}KB > ${vecLimit}KB`);
    }

    if (run.precisionMarginPct < 0) {
      dims.precision = 'fail';
      risks.push(`精度超限 (${run.maxRelErr})`);
    } else if (run.precisionMarginPct < contract.minPrecisionMarginPct) {
      dims.precision = contract.epochNum >= 2 ? 'fail' : 'warn';
      risks.push(`精度裕度 ${run.precisionMarginPct}% < ${contract.minPrecisionMarginPct}%`);
    } else if (run.precisionMarginPct <= 22 && run.hasHiddenRisk) {
      dims.precision = 'warn';
      risks.push(`精度裕度贴近下限 (${run.precisionMarginPct}%)`);
    }

    const worstGen = Math.max(run.p1RegressionPct ?? 0, run.tailRegressionPct ?? 0);
    if (worstGen > contract.p1MaxRegressionPct) {
      dims.generalization = 'fail';
      risks.push(`[1,128] 与 +64 尾块退化 +${worstGen.toFixed(1)}% > ≤${contract.p1MaxRegressionPct.toFixed(1)}%`);
    } else if (worstGen > 2.0) {
      dims.generalization = contract.p1GuardrailEnabled ? 'fail' : 'warn';
      risks.push(`小 Shape / 尾块退化 +${worstGen.toFixed(1)}%`);
    }

    if (!wsPass) {
      dims.performance = 'fail';
      risks.push(`Workspace +${run.workspaceKB}KB > ${contract.maxWorkspaceKB}KB`);
    } else if (!speedPass) {
      dims.performance = 'warn';
    }

    const nonPerfFail = ['hardware', 'correctness', 'precision', 'generalization', 'engineering'].some((k) => dims[k] === 'fail') || !wsPass;
    const hasWarn = ['hardware', 'correctness', 'precision', 'generalization', 'engineering'].some((k) => dims[k] === 'warn');

    let overall = 'pass';
    let badge = run.evidenceGrade === 'candidate' ? '候选待复核' : '6/6 全通过';
    if (nonPerfFail) {
      overall = 'invalid';
      badge = `${contract.version} 门禁违例`;
    } else if (!speedPass) {
      overall = 'progress';
      badge = hasWarn ? '未达标 · 预警' : `未达 ${targetUs.toFixed(1)}μs`;
    } else if (hasWarn || run.hasHiddenRisk) {
      overall = 'risk';
      badge = '存在风险';
    }

    return {
      overall,
      badge,
      dims,
      risks,
      speedup,
      targetUs,
      invalidUnderCurrent: nonPerfFail && Boolean(run.c1Best),
    };
  }

  function computeRunGateProfile(run, contract) {
    const ev = evaluateRun(run, contract);
    const targetUs = getTargetLatencyUs(contract);
    const statuses = GATE_KEYS.map((k) => ev.dims[k] || 'pending');
    const passFlags = statuses.map((s) => (s === 'pass' ? 1 : 0));

    const hwScore = !run.buildOk
      ? 0.18
      : Math.max(0.35, Math.min(1.0, 1.0 - Math.max(0, (run.vecKB || 96) - 120) / 140));
    const corrScore = !run.buildOk || !run.testsOk ? 0.18 : 1.0;
    const precScore = run.precisionMarginPct == null
      ? 0.20
      : run.precisionMarginPct < 0
      ? 0.16
      : Math.max(0.28, Math.min(1.0, 0.25 + (run.precisionMarginPct / 90) * 0.75));
    const worstGen = Math.max(run.p1RegressionPct ?? 0, run.tailRegressionPct ?? 0);
    const genScore = run.p1RegressionPct == null
      ? 0.22
      : worstGen <= 0
      ? Math.min(1.0, 0.90 + Math.abs(worstGen) * 0.04)
      : Math.max(0.20, 0.90 - (worstGen / Math.max(3.0, contract.p1MaxRegressionPct * 2.5)) * 0.68);
    const perfScore = !run.p0Latency || !run.buildOk || !run.testsOk
      ? 0.18
      : Math.max(0.30, Math.min(1.0, (targetUs / run.p0Latency) * 0.94));
    const engScore = run.topologyOk ? 1.0 : 0.3;

    return {
      ev,
      keys: GATE_KEYS,
      labels: GATE_LABELS,
      numbered: GATE_NUMBERED,
      statuses,
      passFlags,
      passCount: passFlags.reduce((a, b) => a + b, 0),
      radar: [hwScore, corrScore, precScore, genScore, perfScore, engScore],
    };
  }

  function computeDiagnostics(runs, lanes, contract, act) {
    const completed = runs.filter((r) => r.id !== '#0' && !r.isRunning);
    const coverage = ['dispatch', 'incore', 'memory', 'host'].map((axisId) => {
      const meta = AXIS_META[axisId];
      const axisRuns = completed.filter((r) => r.axis === axisId);
      const validRuns = axisRuns.filter((r) => r.buildOk && r.testsOk && r.p0Latency);
      const best = validRuns.reduce((min, r) => (r.p0Latency < min ? r.p0Latency : min), Infinity);
      return {
        axisId,
        meta,
        count: axisRuns.length,
        bestLatency: best === Infinity ? null : best,
        excluded: meta.excluded,
      };
    });
    const exceptions=[];
    const incore=lanes.find(l=>l.id==='lane-incore');
    if(incore && ['plateau','active'].includes(incore.status) && runs.some(r=>r.id==='#9')) exceptions.push({id:'stall',sev:'warn',signal:'InCore 样本包络重叠',advice:'核查 #6 / #8 / #9 的停滞，决定保留或剪枝路径。',inspectAction:'inspect-stall',inspectLabel:'核查 #9',actionType:'prune-and-dispatch',actionLabel:'剪枝并转向编排'});
    const focus=runs.find(r=>r.id===(act>=4?'#11':act>=3?'#10':'#3'));
    if(focus && evaluateRun(focus,contract).overall==='invalid') exceptions.push({id:'gate-risk',sev:'risk',signal:focus.id+' 存在契约违例',advice:'区分主 Shape 耗时目标与精度、泛化等门禁。',inspectAction:'inspect-run-10',inspectLabel:'核查 #10',actionType:'goto-contract',actionLabel:'调整目标与约束'});
    if(Object.values(contract.certainty||{}).some(v=>v!=='explicit')) exceptions.push({id:'rules',sev:'warn',signal:'存在未明确确认的约束',advice:'确认稀疏度覆盖与接口约束的适用范围。',inspectAction:'goto-contract',inspectLabel:'核查约束',actionType:'goto-contract',actionLabel:'前往确认'});
    return { coverage, exceptions };
  }

  // 7. Single Shared Serializer for Inspector Preview, Copy & JSON Export (evidence-workbench.md §12)
  function buildEvidencePayload(run, contract, questionText, activeScheme) {
    const prof = computeRunGateProfile(run, contract);
    const ev = prof.ev;
    const grade = EVIDENCE_GRADE_META[run.evidenceGrade] || EVIDENCE_GRADE_META.observed;
    const gatesObj = {};
    GATE_KEYS.forEach((k, idx) => {
      gatesObj[`${idx + 1}_${GATE_LABELS[idx]}`] = prof.statuses[idx];
    });

    return {
      schemaVersion: '1.3.1',
      provenance: {sourceKind:'scenario-record', rawCaptureVerified:false, declaredRevisionVerified:false},
      case: {
        scope: CASE_CONTRACT.scope,
        operator: CASE_CONTRACT.operator,
        repo: CASE_CONTRACT.repo,
        revision: CASE_CONTRACT.revision,
        hardware: `${CASE_CONTRACT.chip} (${CASE_CONTRACT.topology})`,
        primaryShape: CASE_CONTRACT.shape,
        dtype: CASE_CONTRACT.dtype,
        clock: CASE_CONTRACT.clock,
        baseline: { id: CASE_CONTRACT.baselineId, displayId: '#0', latencyUs: CASE_CONTRACT.baselineLatencyUs },
        activeContract: contract.version,
        targetLatencyUs: Number(getTargetLatencyUs(contract).toFixed(1)),
      },
      selection: {
        ref: { scope: run.scope || CASE_CONTRACT.scope, id: run.uid || `exp-${run.seq}` },
        displayId: run.id,
        wave: run.round,
        tuningCategory: (AXIS_META[run.axis] || AXIS_META.incore).name,
        parentId: run.parentId || null,
        activeCanvasScheme: activeScheme || 'landscape',
      },
      measurements: {
        evidenceGrade: grade.id,
        evidenceGradeDesc: grade.desc,
        deviceWallMeanUs: run.p0Latency,
        deviceWallRangeUs: run.latencyRangeUs || null,
        speedupVsBaseline: ev.speedup ? Number(ev.speedup.toFixed(3)) : null,
        smallShape1x128DeltaPct: run.p1RegressionPct,
        tailPlus64DeltaPct: run.tailRegressionPct ?? run.p1RegressionPct,
        vecFootprintKB: run.vecKB,
        vecLimitKB: CASE_CONTRACT.vecLimitKB,
        matFootprintKB: run.matKB,
        workspaceKB: run.workspaceKB,
        precisionMarginPct: run.precisionMarginPct,
        coreSkew24CG: run.coreSkew,
      },
      gates: {
        summary: `${prof.passCount}/6`,
        verdict: ev.badge,
        dimensions: gatesObj,
        risks: ev.risks,
      },
      evidence: {
        sourceRef: run.sourceRef || {
          file: 'kernels/attn/decode_sparse_attn_csa.py',
          lines: 'L114-L179',
          revision: CASE_CONTRACT.revision,
        },
        comparability: run.comparability || {
          contractVersion: contract.version,
          shapeScope: CASE_CONTRACT.shape,
          comparableToBaseline: Boolean(run.buildOk && run.testsOk),
          note: '继承父分支测量上下文，待真机 20 次采样复核',
        },
      },
      handoff: {
        question: (questionText || '').trim(),
        exportMode: 'offline-html-evidence-handoff',
      },
    };
  }

  function serializeInvestigationHandoff(run, contract, questionText, activeScheme) {
    const payload = buildEvidencePayload(run, contract, questionText, activeScheme);
    const markdown = [
      `### 继续调查交接 · ${payload.selection.displayId} (${payload.selection.ref.scope} :: ${payload.selection.ref.id})`,
      `- **问题**：${payload.handoff.question || '请核对该实验的分块/分核改动、样本波动包络与 6 项契约门禁证据。'}`,
      `- **来源锚点**：\`${payload.evidence.sourceRef.file}:${payload.evidence.sourceRef.lines}\` @ \`${payload.evidence.sourceRef.revision}\``,
      `- **证据等级**：\`${payload.measurements.evidenceGrade}\` (${payload.measurements.evidenceGradeDesc})`,
      `- **关键读数**：\`device_wall_us\` = ${payload.measurements.deviceWallMeanUs != null ? `${payload.measurements.deviceWallMeanUs.toFixed(1)} μs` : '未上板'} (包络 ${payload.measurements.deviceWallRangeUs ? `[${payload.measurements.deviceWallRangeUs[0]}, ${payload.measurements.deviceWallRangeUs[1]}] μs` : '无'}) · Vec ${payload.measurements.vecFootprintKB}KB / ${payload.measurements.vecLimitKB}KB · 门禁 ${payload.gates.summary} (${payload.gates.verdict})`,
      `- **可比性前提**：${payload.evidence.comparability.note}`,
      '',
      '```json',
      JSON.stringify(payload, null, 2),
      '```',
    ].join('\n');
    return { payload, markdown, json: JSON.stringify(payload, null, 2) };
  }

  const PROPOSAL_HTML = `
    <div class="proposal-md"><h2>三套探索画布 · 当前实现</h2>
    <p>共用全视口 SVG、实验 ID、契约与选择集合；浮层叠加，保持相机与节点位置。深色默认，支持浅色。</p>
    <h3>演进地形</h3><p>离散波次对齐耗时 μs、Vec KB 或相对 #0 加速比。蓝色正常、绿色最终采纳、红色回退或违例，柔黄无边框底色来自 #6 / #8 / #9 的实际坐标。末端引线标注策略，切轴及 Fork 后重新布局。</p>
    <h3>靶心收敛与分支泳道</h3><p>靶心半径严格按耗时 / 482 标定，当前契约目标圈与门禁状态分开。门禁筛选及角落状态轮廓不压缩主图；泳道节点、波次列和遥测共轴，同轨直线，跨轨必要折点。</p>
    <h3>证据边界</h3><p>当前来源为 data.js 场景记录，原始设备采集、硬件参数及声明 revision 尚未核验。Fork 仅生成待采样候选，不制造性能读数或验收结果。源码片段保留用于核查，尚未验证索引映射。</p>
    <p>详细设计、Skill 分工和人工走查记录见同目录 PROPOSAL.md、manifest.json 与 DELIVERY.md。</p></div>`;

  global.PTO_V2_DATA = {
    CASE_CONTRACT,
    PRESENTATION_CONTRACT,
    GLYPH_PATHS,
    AXIS_META,
    GATE_KEYS,
    GATE_LABELS,
    GATE_NUMBERED,
    EVIDENCE_GRADE_META,
    SWIMLANE_ROWS,
    DEFAULT_NL_PROMPT,
    NL_PHRASES,
    INITIAL_CONTRACT_V1,
    CONTRACT_V2_PATCH,
    BASE_LANES,
    BASE_RUNS,
    INITIAL_DECISION_CARDS,
    CODE_DIFF_HUNKS,
    RISK_BUCKETS_FAST,
    COVERAGE_SHAPES,
    PROPOSAL_HTML,
    getTargetLatencyUs,
    evaluateRun,
    computeRunGateProfile,
    computeDiagnostics,
    buildEvidencePayload,
    serializeInvestigationHandoff,
  };
})(window);
