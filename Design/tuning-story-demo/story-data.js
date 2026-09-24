/* Tuning Story — 场景数据
 *
 * 全部数值来自两处真实材料，不做演示性改写：
 *  [本地] Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/   (2026-09-03 DSpark decode_csa 构建)
 *         Data/_jit_decode_fwd_layers_20260625_184941/report/   (6 月构建，提供内存分配报告)
 *  [GH]   pypto-lib#314 / #622 / #628 / #641、pypto#2309、社区泳道调优日志 §2 §6 §8 §14 §19 §20
 *
 * 标注：fact=实测或公开记录；infer=多源推断；design=产品方案建议（不是已有能力）。
 */
window.TS_DATA = (() => {
  'use strict';

  /* ---------------------------------------------------------------- 本地实测 */

  // 来源：merged_swimlane_20260903_010746.json,Worker View(pid=4) 的 AIC_*/AIV_* 轨道。
  // wall = 最早开始→最晚结束 = 4879.8us；整图 AIC 占用 32.2%、AIV 37.5%。
  const WALL = 4879.8;
  const SCOPES = [
    { name: 'qk_pv_aiv_spmd',                  n: 48,   sum: 31626, pct: 25.2, avg: 658.9, cores: 48, start: 2634, span: 886, kind: 'aiv' },
    { name: 'csa_merge_pack_publish_spmd',     n: 48,   sum: 18597, pct: 14.8, avg: 387.4, cores: 48, start: 3019, span: 711, kind: 'aiv' },
    { name: 'qk_pv_aic_spmd',                  n: 24,   sum: 15036, pct: 12.0, avg: 626.5, cores: 24, start: 2634, span: 836, kind: 'aic' },
    { name: 'kv_score_proj_spmd',              n: 512,  sum: 5654,  pct: 4.5,  avg: 11.0,  cores: 24, start: 1431, span: 764, kind: 'aic' },
    { name: 'indexer_topk_group_wave_spmd',    n: 48,   sum: 5420,  pct: 4.3,  avg: 112.9, cores: 48, start: 2193, span: 133, kind: 'aiv' },
    { name: 'indexer_score_leaf_wave_aiv_spmd',n: 48,   sum: 4335,  pct: 3.5,  avg: 90.3,  cores: 48, start: 2037, span: 148, kind: 'aiv' },
    { name: 'qr_rms_norm_quant_spmd',          n: 32,   sum: 3183,  pct: 2.5,  avg: 99.5,  cores: 32, start: 274,  span: 111, kind: 'aiv' },
    { name: 'tp_o_a_spmd',                     n: 128,  sum: 2540,  pct: 2.0,  avg: 19.8,  cores: 24, start: 4460, span: 118, kind: 'aic' },
    { name: 'qproj_dequant_rms_nope_rope_spmd',n: 16,   sum: 2372,  pct: 1.9,  avg: 148.2, cores: 16, start: 542,  span: 151, kind: 'aiv' },
    { name: 'qr_hadamard_quant_spmd',          n: 256,  sum: 2265,  pct: 1.8,  avg: 8.8,   cores: 48, start: 1038, span: 316, kind: 'aiv' },
    { name: 'scatter_softmax_pool_spmd',       n: 64,   sum: 2140,  pct: 1.7,  avg: 33.4,  cores: 48, start: 2211, span: 174, kind: 'aiv', rank: 14 },
  ];

  // 按 500us 窗口统计的核占用率（同一份 trace）。
  const WINDOWS = [
    { from: 0,    to: 500,  aic: 40, aiv: 31 },
    { from: 500,  to: 1000, aic: 19, aiv: 22 },
    { from: 1000, to: 1500, aic: 15, aiv: 12 },
    { from: 1500, to: 2000, aic: 50, aiv: 9  },
    { from: 2000, to: 2500, aic: 28, aiv: 59 },
    { from: 2500, to: 3000, aic: 72, aiv: 73 },
    { from: 3000, to: 3500, aic: 53, aiv: 100 },
    { from: 3500, to: 4000, aic: 0,  aiv: 38 },
    { from: 4000, to: 4500, aic: 7,  aiv: 7  },
    { from: 4500, to: 4880, aic: 30, aiv: 15 },
  ];

  // 来源：dfx_outputs/rank*/d0/host.*.log 的 chip.run.runner_run.device_wall span。
  const RUNS = [
    { rank: 'rank0', inv: 1, deviceWall: 45.19, bind: 187.21, note: '首次调用：bind.prebuilt 187ms' },
    { rank: 'rank0', inv: 2, deviceWall: 5.13,  bind: 0.003,  note: '稳态' },
    { rank: 'rank1', inv: 1, deviceWall: 3.61,  bind: null,   note: '首次并不慢' },
    { rank: 'rank1', inv: 2, deviceWall: 4.02,  bind: null,   note: '稳态' },
  ];

  return {
    meta: {
      build: 'Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/',
      memBuild: 'Data/_jit_decode_fwd_layers_20260625_184941/report/',
      platform: 'a2a3 (Ascend 910B)',
      cores: '24 AIC + 48 AIV',
      clock: '50 MHz',
      wall: WALL,
      utilAic: 32.2,
      utilAiv: 37.5,
      scopeCount: 66,
      taskRecords: 4081,
      deps: { tasks: 86, tensors: 138, edges: 338 },
      passes: 52,
      hints: { total: 230, ph001: 197, phmr001: 33 },
    },
    scopes: SCOPES,
    windows: WINDOWS,
    runs: RUNS,

    phases: [
      { id: 'prepare', name: '准备',      scenes: ['T1'] },
      { id: 'locate',  name: '定位',      scenes: ['T2', 'T3'] },
      { id: 'cause',   name: '归因与修改', scenes: ['T4', 'T5', 'T6', 'T7'] },
      { id: 'verify',  name: '验证',      scenes: ['T8'] },
      { id: 'ship',    name: '交付',      scenes: ['T9'] },
    ],

    scenes: [
      /* ============================================================ T1 */
      {
        id: 'T1', phase: 'prepare', name: '建任务、立基线',
        question: '现在有多快？怎么测才算数？',
        meta: 'compressor 家族 · B=64 / S=1 · 主指标待定',
        emotion: 0.5, emotionLabel: '先立基线，心里有底',
        focus: null,
        views: [
          { id: 'baseline', label: '基线卡', render: 'baseline' },
          { id: 'runs',     label: '调用次序', render: 'runs' },
        ],
        actions: [
          '选定配置 B=64 / S=1，记录 pypto、pypto-lib、simpler、ptoas、CANN 版本',
          '跑一次 <code>--enable-l2-swimlane</code>，抄下 Total Test Time 与 12 个 kernel 的 Exec/Latency',
          '算出 Head OH / Tail OH 分布，在 Issue 正文里写测量协议',
        ],
        pains: [
          '基线是手抄表格，和原始数据之间没有链接，换台机器无法复核',
          '冷启动、rank 差异不可见：本地 rank0 首次 45.19ms、稳态 5.13ms，相差 9 倍',
          '测量协议写在 Issue 正文里，工具不会执行也不会检查',
        ],
        opps: [
          { p: 'P0', t: '基线对象化：引用原始数据，自带环境指纹，此后每次运行自动对比' },
          { p: 'P0', t: '冷启动与稳态分开统计，首次调用默认不进入主指标' },
          { p: 'P1', t: '测量协议做成表单，由 T8 的实验台自动执行' },
        ],
        objs: 'Workload · Objective · Environment · Run · Baseline · Plan',
        proto: '算子调优控制台（门禁先行）',
        sources: [
          { t: 'fact', s: 'pypto-lib#314', d: '基线 394.84 µs / 94 task，Tail OH 727.8 µs 占 28.3%' },
          { t: 'fact', s: '本地 host.*.log', d: 'STRACE 分层耗时，带 inv 调用序号' },
        ],
        terminal: [
          '$ python models/deepseek/v4/compressor_ratio128.py -p a2a3 --enable-l2-swimlane',
          '  Total Test Time      394.84 us      Total Tasks  94',
          '  Total Exec          1787.40 us      Exec/Latency 69.6%',
          '  Tail OH (finish→end) 727.80 us      28.3%  P50 9.5 / P95 13.3 us',
          '$ grep device_wall dfx_outputs/rank0/d0/host.*.log',
          '  inv=1 dur=45193079ns      inv=2 dur=5132800ns      # 冷启动 9x',
        ],
        vizNote: '基线阶段还没有可对比的对象；泳道总览先作为后续每个场景的锚点。',
      },

      /* ============================================================ T2 */
      {
        id: 'T2', phase: 'locate', name: '整图定位热点',
        question: '时间花在哪个 scope？哪段时间核在空转？',
        meta: '66 个 scope · wall 4,879.8 µs · AIC 32.2% / AIV 37.5%',
        emotion: -1, emotionLabel: '3.6 万个事件，只能写脚本',
        focus: 'kv_score_proj_spmd',
        views: [
          { id: 'rank',   label: 'Scope 排行',  render: 'scopeRank' },
          { id: 'window', label: '占用率窗口',  render: 'windows' },
        ],
        actions: [
          '在 Perfetto 打开 merged_swimlane（36,336 个事件）',
          '写临时脚本按 kernel 汇总 core-time，再按时间窗口算占用率',
          'Latency 列缺数据时，从 l2_swimlane_records 重建 wall',
        ],
        pains: [
          '通用 trace 工具不认识 scope、spmd、AIC/AIV 配对，汇总全靠临时脚本',
          'Worker View 与 Scheduler View 两套视图容易重复计数',
          '关键路径工具不可用，只能凭经验判断「在不在关键路径上」',
        ],
        opps: [
          { p: 'P0', t: '以 scope 为单位排行，同时显示关键路径 slack，而不只是 Σdur' },
          { p: 'P0', t: '泳道上叠加占用率色带，低占用窗口直接高亮' },
          { p: 'P1', t: '视图去重并在界面上标注统计口径（Worker / Scheduler）' },
        ],
        objs: 'Task · Scope · Counter（窗口占用率）· Bottleneck',
        proto: 'Tuning Console V2 · Run Overview',
        sources: [
          { t: 'fact', s: '本地 merged_swimlane', d: 'kv_score_proj 512 个 task、平均 11.0 µs、排第 4' },
          { t: 'fact', s: '本地窗口统计', d: '1000–1500 µs 窗口 AIC 15% / AIV 12%' },
          { t: 'infer', s: '对照 §20', d: '6 月调优过的 scatter_softmax_pool 已退到第 14 位，热点发生迁移' },
        ],
        terminal: [
          '$ node tools/scope-rank.js dfx_outputs/rank0/d0/merged_swimlane_*.json',
          '  events 36336   X 9480   flow 12745x2   counters 1128',
          '  worker-view AICore tasks 4081   scopes 66   wall 4879.8us',
          '  #4  kv_score_proj_spmd      n=512  sum=5654us  avg=11.0us  cores=24  span=764us',
          '  #14 scatter_softmax_pool_spmd n=64  sum=2140us  avg=33.4us  cores=48',
        ],
        vizNote: 'kv_score_proj 平均每个 task 只有 11 µs 却切成 512 个，且大半落在 1000–2000 µs 的低占用窗口里。',
      },

      /* ============================================================ T3 */
      {
        id: 'T3', phase: 'locate', name: '诊断调度与依赖',
        question: '是派发开销，还是依赖把并行变成了串行？',
        meta: 'deps.json 86 task / 138 张量 / 338 条边',
        emotion: -2, emotionLabel: '越拆越慢',
        focus: 'scatter_softmax_pool_spmd',
        views: [
          { id: 'diag', label: '调度诊断', render: 'schedDiag' },
          { id: 'exp',  label: '三次拆分实验', render: 'passSplit' },
        ],
        actions: [
          '看 latency/exec 比值与 Head/Tail OH 分布',
          '数一个 scope 实际落在几个核上',
          '用 deps_viewer 把 deps.json 渲染成 HTML，找跨迭代的写依赖',
          '做 3-pass / 4-pass 拆分实验验证假设',
        ],
        pains: [
          '理解 Head OH / Tail OH 需要先懂调度器的派发与完成语义',
          '依赖图和泳道在两个工具里，「哪条边让这两个 task 串行」要人工对照',
          '带宽假设曾被误判：减少重读后 Total 反而从 582 退化到 621 µs',
        ],
        opps: [
          { p: 'P0', t: '一句话诊断：派发型还是依赖串行型，并附判定依据' },
          { p: 'P0', t: '依赖边能直接回到产生它的源码行' },
          { p: 'P1', t: '把用户手写的 deps=[...] 与自动推导的依赖分开显示' },
        ],
        objs: 'DepEdge · Tensor · Bottleneck（派发 / 串行）· Experiment',
        proto: 'Decode Layer 计算图',
        sources: [
          { t: 'fact', s: 'pypto-lib#314', d: 'kv_cache_write 派发 64 次，latency/exec 达 6.4 倍' },
          { t: 'fact', s: '调优日志 §8', d: '中段 1500 µs 内 AIC 仅 1.9%、AIV 仅 5.8%' },
          { t: 'fact', s: '本地 deps.json', d: '边带 source=creator/explicit 与 flags=wait/retain' },
        ],
        terminal: [
          '$ python -m simpler_setup.tools.deps_viewer dfx_outputs/rank0/d0/deps.json',
          '  tasks 86   tensors 138   edges 338',
          '  edge[0] pred=8589934593 succ=8589934594 source=creator flags=[wait,retain]',
          '$ # 三次真机实验（standalone compressor）',
          '  baseline 全融合          1815 us   scatter 5 核 / kv_hadamard 1 核',
          '  3-pass 拆出计算          1866 us   计算解放但不是瓶颈',
          '  4-pass scatter 独立      2593 us   塌成 1 核',
        ],
        vizNote: 'task 多不等于并行：几百个 task 只落在 1–6 个核上，根因在跨迭代的写依赖链。',
      },

      /* ============================================================ T4 */
      {
        id: 'T4', phase: 'cause', name: '修正源码结构',
        question: '是哪一行写法造成的？怎么改？',
        meta: 'decode_indexer_compressor.py · state_scatter_paged',
        emotion: 2, emotionLabel: '外提 pl.at，−43%',
        focus: 'scatter_softmax_pool_spmd',
        views: [
          { id: 'diff', label: '源码改动', render: 'sourceDiff' },
          { id: 'unmix', label: '反直觉：拆开融合', render: 'unmix' },
        ],
        actions: [
          '按 name_hint 搜回源码，读 pl.at 与 for 的嵌套关系',
          '找到对共享 GM 句柄反复 pl.assemble 重新赋值的那几行',
          '把 pl.at 提到外层包住循环，缩短依赖链',
        ],
        pains: [
          '改一行代码要完整编译、上板才知道效果，每轮成本在小时级',
          'pl.spmd 写法曾被编译器 bug 挡住（pypto#1414），分不清是写错还是不支持',
        ],
        opps: [
          { p: 'P0', t: '源码为主界面：选中 scope 块即显示 task 数、占用核数、写依赖' },
          { p: 'P0', t: '估算值（estimated）与实测值（measured）并列且严格区分' },
          { p: 'P1', t: '在造成依赖链的那一行标出伪并行风险' },
        ],
        objs: 'Scope · Tensor · SourceMap · Candidate',
        proto: '算子开发 Copilot · 算子对象工作台',
        sources: [
          { t: 'fact', s: '调优日志 §8', d: '1815 → ~1025 µs（−43%），scatter 512→128 task、5→42 核' },
          { t: 'fact', s: '调优日志 §14', d: '拆开 matmul+dequant 融合，整图 −69 µs' },
        ],
        terminal: [
          '$ python models/deepseek/v4/decode_indexer_compressor.py -p a2a3 --enable-l2-swimlane',
          '  Total 1039.7 us  /  复跑 1012.0 us          # 改前 1815 us',
          '  state_scatter_paged  tasks 512→128   cores 5→42   span 1602→197 us',
          '  precision: uniform + hetero start_pos  ALL PASS',
        ],
        vizNote: 'pl.at 的位置决定了每个 batch 生成 1 个还是 4 个 scope，也就决定了依赖链的长度。',
      },

      /* ============================================================ T5 */
      {
        id: 'T5', phase: 'cause', name: '核对编译过程',
        question: '我写的意图生效了吗？编译器提示了什么？',
        meta: '52 个 pass · 230 条性能提示 · kv_score_proj 五层对照',
        emotion: -2, emotionLabel: '意图被静默忽略',
        focus: 'kv_score_proj_spmd',
        views: [
          { id: 'intent', label: '意图核对', render: 'intent' },
          { id: 'layers', label: '五层对照', render: 'layers' },
          { id: 'hints',  label: '提示分诊', render: 'hints' },
        ],
        actions: [
          '写探针脚本，分别生成 NZ / ND 两个版本的 .pto 做 diff',
          '在 52 个 pass dump 里 grep 函数名，找某个变换发生在哪一步',
          '打开 kernel .cpp 数 set_flag / wait_flag / pipe_barrier',
          '读 230 行 perf_hints.log，按文件找与自己相关的条目',
        ],
        pains: [
          '编译器「没做」的事没有任何信号：NZ 声明生成的 .pto 逐字节相同',
          '230 条提示按编译顺序输出，PH001 就占 197 条，重要的被淹没',
          'pass 编号会随版本漂移（MemoryReuse 从 29 号变成 35 号）',
        ],
        opps: [
          { p: 'P0', t: '意图与生效值并排：请求值 / 实际值 / 证据位置' },
          { p: 'P0', t: '提示与热点 join 后再排序，只把落在热点 scope 上的排到前面' },
          { p: 'P1', t: '按「有变化的 pass」浏览，并用 pass 名而非编号跨版本对齐' },
        ],
        objs: 'Pass · IRNode · CodegenArtifact · Diagnostic · SourceMap · Fence',
        proto: 'Pass Transform Explorer · Pass Atlas',
        sources: [
          { t: 'fact', s: '本地 perf_hints.log', d: 'decode_compressor_ratio4.py:110 有 5 条 PH-MR-001' },
          { t: 'fact', s: '本地 .pto / .cpp', d: 'kv_score_proj：6 TLOAD、24 TEXTRACT、12 次 matmul、33 对 set/wait_flag' },
          { t: 'fact', s: '调优日志 §19 §20', d: 'NZ 声明被静默忽略；col_expand_sub 无 codegen' },
        ],
        terminal: [
          '$ grep -c perf_hint report/perf_hints.log',
          '  230        # PH001 197 条 · PH-MR-001 33 条',
          '$ grep "decode_compressor_ratio4.py:110" report/perf_hints.log | head -1',
          '  [perf_hint PH-MR-001] MemoryReuse: software pipelining requested depth 2 ...',
          '  but only 1 of 2 buffers fit (32768 B per stage, 65536 B free) — stages 1 apart',
          '  share storage and serialize.',
          '$ ls passes_dump | wc -l',
          '  52',
        ],
        vizNote: '源码里写的 pipeline(stage=2)，在 Right（L0B）这一级只放得下 1 份——意图只是部分生效。',
      },

      /* ============================================================ T6 */
      {
        id: 'T6', phase: 'cause', name: '硬件资源预算',
        question: '再加大一档放得下吗？数据是怎么搬的？',
        meta: 'Vec 184KB · Mat 512KB · Right 64KB（编译器实际上限）',
        emotion: -1, emotionLabel: '每加一档都撞墙',
        focus: 'kv_score_proj_spmd',
        views: [
          { id: 'budget', label: '容量预算', render: 'budget' },
          { id: 'walls',  label: '撞墙记录', render: 'walls' },
          { id: 'move',   label: '搬运形态', render: 'movement' },
        ],
        actions: [
          '把 tile 调大一档、加深 pipeline，编译，看报错',
          '心算 3×[128,256] FP32 的占用能不能放进 UB',
          '打开 715 行的内存分配报告，找每个 buffer 的地址与生命周期',
          '看 op-sim 里的 ND2NZ 指令，判断是不是事务过碎',
        ],
        pains: [
          '容量上限只能靠编译试错试出来，报错只给总量不给明细',
          '硬件规格与编译器上限不一致：报告写 184 KB，调优日志常写 192 KB',
          '搬运形态藏在指令名里，判断「短 burst」需要深厚背景知识',
        ],
        opps: [
          { p: 'P0', t: '调参数时实时显示各级占用，编译前就预判越界' },
          { p: 'P0', t: '每个上限都标注来源（编译器 / 硬件手册）与平台' },
          { p: 'P1', t: '越界归因到具体 buffer，并给出可释放空间的候选' },
          { p: 'P1', t: '把 ND2NZ 翻译成「拆成多少笔、每笔多长」，再与带宽上限对照' },
        ],
        objs: 'Constraint（带平台与版本）· MemBlock · Tile · SweepAxis · Counter',
        proto: 'Memory Inspector · Ascend Memory Studio',
        sources: [
          { t: 'fact', s: '本地内存报告', d: 'Vec 上限 184.0 KB；down_proj 的 Right 用满 64 KB（100%）' },
          { t: 'fact', s: '调优日志 §19', d: 'b_trans：busy −15.1%，CSA 整图 wall −12.9%（3 跑中位数）' },
          { t: 'fact', s: '本地 .pto', d: 'Right tile bf16[256,64] row_major/col_major，即 b_trans 后的 ZN 形态' },
        ],
        terminal: [
          '$ sed -n "/--- down_proj ---/,+6p" report/memory_after_AllocateMemoryAddr.txt',
          '  Space  |  Used      |  Limit     |  Usage   |  MemRefs',
          '  Mat    |  260.0 KB  |  512.0 KB  |   50.8%  |  4',
          '  Right  |   64.0 KB  |   64.0 KB  |  100.0%  |  2',
          '$ # 试着把 pipeline 加深到 stage=3',
          '  error: Mat buffer usage 589824 > 524288',
        ],
        vizNote: '容量、流水深度、融合方式互相牵连：一个参数同时影响 L1、L0B 和 L0C。',
      },

      /* ============================================================ T7 */
      {
        id: 'T7', phase: 'cause', name: '核内下钻',
        question: '单个 task 的时间花在搬运还是计算上？',
        meta: 'softmax_pool · op-sim 单 task · veccore span 20.58 µs',
        emotion: 1, emotionLabel: '看清：Vector 在忙搬运',
        focus: 'scatter_softmax_pool_spmd',
        views: [
          { id: 'incore', label: '核内构成', render: 'incore' },
          { id: 'valid',  label: '采样有效性', render: 'traceValid' },
        ],
        actions: [
          '用 incore-profiling skill 调用 msprof op simulator',
          '读 clean.json 里各单元的占比与指令构成',
          '发现 trace 退化后，手动把 position_ids 写成 127 再重跑',
        ],
        pains: [
          '需要单独的工具链，输入要从 golden 生成',
          '按硬件单元罗列周期，要自己把指令归类成搬运 / 计算 / 转置',
          '数据门控让 trace 退化，kernel 看起来「很快」但其实跑了 0 轮',
          'replay 脚本默认输入是随机数、动态维填 1，同样可能触发退化',
        ],
        opps: [
          { p: 'P0', t: '默认按「搬运 / 片上搬移 / 格式转换 / 计算」归类，而不是按硬件单元' },
          { p: 'P0', t: '先显示 trace 是否有效，再显示任何数字' },
          { p: 'P1', t: '每类指令能回到产生它的源码切片' },
        ],
        objs: 'Counter（指令、单元）· Kernel · TensorDump · SourceMap',
        proto: '调试与调优工作台（Operator Lab）',
        sources: [
          { t: 'fact', s: '调优日志 §20', d: 'MTE2 占 69%，由 256 笔 [1,64] 散读组成，每笔约 211 ns' },
          { t: 'fact', s: '调优日志 §20', d: 'R1 块读后 MTE2 cycles −90.3%，笔数 256→32，span −62.4%' },
          { t: 'infer', s: '本地 .cpp 静态计数', d: '可与 op-sim 实测周期并排，帮助判断哪条指令贵' },
        ],
        terminal: [
          '$ msprof op simulator --application=incore_softmax_pool_ratio128',
          '  [warn] degenerate trace: loop executed 0 iterations',
          '  gate: position_ids % 128 >= 126   但 auto-golden 把 position_ids 清零',
          '$ # 手动把 v1.bin 写成 127 后重跑',
          '  veccore0 span 20.58us   MTE2 69%（256 笔 [1,64]）  VECTOR 63%（多为 staging）',
          '  VEXP + VDIV + VCADD 合计 < 1us',
        ],
        vizNote: 'Vector 单元很忙，但忙的是逐行 staging 与两次转置，真正的 softmax 计算不到 1 µs。',
      },

      /* ============================================================ T8 */
      {
        id: 'T8', phase: 'verify', name: '实验与验证',
        question: '真的更快吗？是噪声吗？精度对吗？',
        meta: '同 session A/B · 3 跑 · 锚点 qk_pv',
        emotion: 2, emotionLabel: 'R2 −8.7%，逐位一致',
        focus: 'qk_pv_aiv_spmd',
        views: [
          { id: 'ab',    label: 'A/B 与噪声锚点', render: 'abTable' },
          { id: 'sweep', label: '参数扫描', render: 'sweep' },
        ],
        actions: [
          '手工改常量，扫描 task 数 2 / 4 / 8',
          '同一 session 做 A/B，每个版本跑 3 次取中位数',
          '自己挑一个与改动无关的大 scope 当噪声锚点',
          '跑 golden 比对，手记 build 目录以便追溯',
        ],
        pains: [
          '全程手工：改代码、跑、记录，参数扫描非常耗时',
          '噪声判断靠经验：选哪个锚点、跑几次、看 wall 还是 busy 各不相同',
          '单次运行曾出现 +20% 的假象；R1 的收益大半来自那次跑得偏快',
          '实验记录散落在个人日志、PR 描述和本地目录里',
        ],
        opps: [
          { p: 'P0', t: '自动推荐噪声锚点，并在结果表里显示它的漂移' },
          { p: 'P0', t: '比较资格不合格时，把提升百分比置灰并说明缺哪一步' },
          { p: 'P1', t: '批量实验前显示预计占用的设备时长' },
          { p: 'P1', t: '保留被推翻的假设及其证据' },
        ],
        objs: 'Experiment · Candidate · SweepAxis · Gate · Baseline · Distribution',
        proto: 'Tuning Console（实验台账）',
        sources: [
          { t: 'fact', s: '调优日志 §20', d: '原始 / R1 / R2 三版的整图 wall 与锚点 scope 读数' },
          { t: 'fact', s: '调优日志 §6',  d: 'task 数扫描 2 / 4 / 8 → 621 / 582 / 598 µs，4 是拐点' },
        ],
        terminal: [
          '$ # 同 session A/B，各 3 跑，锚一个无关大 scope 判噪声',
          '  原始   wall 1303.6us   softmax_pool 9495us   qk_pv 22301us',
          '  R1     wall 1276.2us   softmax_pool 3203us   qk_pv 19754us  <- 这次跑得偏快',
          '  R2     wall 1190.1us   softmax_pool 2792us   qk_pv 21918us  <- 锚点匹配',
          '  Δ(R2 vs 原始) = -113.5us (-8.7%)   max_error_ratio = 0.0',
        ],
        vizNote: 'R1 的 wall 收益大半来自那次 session 里无关的 qk_pv 恰好快了 2,500 µs，不是改动带来的。',
      },

      /* ============================================================ T9 */
      {
        id: 'T9', phase: 'ship', name: '交付与沉淀',
        question: '改动会影响谁？经验在什么条件下成立？',
        meta: 'PR #628 / #641 已合入 · 反例 pypto#2309',
        emotion: 1, emotionLabel: '合入并沉淀法则',
        focus: null,
        views: [
          { id: 'impact', label: '影响面', render: 'impact' },
          { id: 'recipe', label: '经验卡', render: 'recipe' },
        ],
        actions: [
          '全局搜索，把权重签名一路改到 4 个文件',
          '排查 CI 报出的 507018，发现分支基于旧 base 后 rebase',
          '写 PR：改前改后数据 + build 目录 + 精度结果',
          '在源码里留下原因注释，把经验写成带条件的法则',
        ],
        pains: [
          '影响面靠全局搜索；漏改的代价是一个难以理解的运行时错误（507018 而不是 shape 错误）',
          '经验散落在源码注释、PR、个人日志和 Issue 中，适用条件往往只写在正文里',
          '本地与 CI 的 pypto 版本偏差会制造假问题',
        ],
        opps: [
          { p: 'P0', t: '经验卡强制写明适用条件与反例，否则不能发布' },
          { p: 'P1', t: '修改签名前先显示影响面，而不是等运行时报错' },
          { p: 'P1', t: '本地与 CI 环境一致性检查' },
          { p: 'P2', t: '按 scope 设置回归看护，而不只盯整图 wall' },
        ],
        objs: 'Lineage · Artifact / Manifest · Recipe（带 Scope 边界）· Regression',
        proto: 'Toolkit Studio',
        sources: [
          { t: 'fact', s: 'pypto-lib#628 / #641', d: '2026-06-29 / 06-30 合入，PR 描述写明 build 目录' },
          { t: 'fact', s: 'pypto#2309', d: '同样的 b_trans 在 Qwen3 TK=64 场景慢了 15.8%–23.2%' },
          { t: 'fact', s: '本地源码注释', d: 'decode_compressor_ratio4.py:122 保留了 b_trans 的原因说明' },
        ],
        terminal: [
          '$ git grep -n "cmp_wkv\\|csa_cmp_wkv" models/',
          '  decode_compressor_ratio4.py   wkv/wgate -> [OUT_DIM, D]',
          '  decode_attention_csa.py       cmp_wkv/cmp_wgate -> [MAIN_OUT_DIM, D]',
          '  decode_layer.py               csa_cmp_wkv/wgate',
          '  decode_fwd.py                 stacked [NUM_LAYERS*MAIN_OUT_DIM, D]',
          '$ # 漏改任一处：运行时 507018（device drain），不是 shape 报错',
        ],
        vizNote: '一条经验如果不写明适用条件，就会被误用——b_trans 在另一个模型上反而慢了 16–23%。',
      },
    ],

    /* ------------------------------------------------- 各视图用到的明细数据 */
    detail: {
      // T1 基线（pypto-lib#314）
      baseline: {
        total: 394.84, tasks: 94, exec: 1787.4, latency: 2567.2, ratio: 69.6,
        headOh: 52.0, tailOh: 727.8, tailPct: 28.3,
        kernels: [
          { name: 'softmax_pool',    n: 8,  exec: 161.18, lat: 163.54, pct: 98.6, note: '#1 热点' },
          { name: 'kv_score_proj',   n: 4,  exec: 70.81,  lat: 75.77,  pct: 93.5, note: '矩阵累加，7 次迭代' },
          { name: 'kv_hadamard',     n: 4,  exec: 8.63,   lat: 9.89,   pct: 87.2, note: '' },
          { name: 'rmsnorm',         n: 1,  exec: 11.58,  lat: 12.78,  pct: 90.6, note: '' },
          { name: 'kv_cache_write',  n: 64, exec: 2.06,   lat: 13.24,  pct: 15.6, note: '64 次细碎派发' },
        ],
      },
      // T3 调度诊断
      sched: {
        rows: [
          { k: 'latency / exec', v: '6.4 倍', s: 'bad', d: 'kv_cache_write：执行 2.06 µs，latency 13.24 µs' },
          { k: '派发次数', v: '64 次', s: 'bad', d: '按 batch 一行派发一次' },
          { k: '中段 AIC 占用', v: '1.9%', s: 'bad', d: '1500 µs 窗口内，占整体 72%' },
          { k: '中段 AIV 占用', v: '5.8%', s: 'bad', d: '同一窗口' },
          { k: '实际占用核数', v: '1–6 核', s: 'bad', d: '每个 scope 有几百个 task' },
          { k: 'pop_hit / pop_miss', v: '1 / 162', s: 'warn', d: '本地调度阶段记录，可解释「完成后等很多轮才被发现」' },
        ],
        splits: [
          { name: 'baseline 全融合单循环', total: 1815, cores: '5–7 核', ok: false, note: '跨 batch 流水把串行散射藏在计算下面' },
          { name: '3-pass 拆出计算',      total: 1866, cores: '24–48 核', ok: false, note: '计算 scope 解放了，但计算本来就不是瓶颈' },
          { name: '4-pass scatter 独立',  total: 2593, cores: '1 核',    ok: false, note: 'scatter 单独成 pass 后塌成 1 核' },
          { name: '3-pass + 折叠散射',    total: 1025, cores: '42 核',   ok: true,  note: 'scatter 必须和 softmax 留在同一个 pass' },
        ],
      },
      // T4 源码
      source: {
        file: 'models/deepseek/v4/decode_indexer_compressor.py',
        before: [
          'for o0 in pl.range(0, OUT_DIM, OUT_CHUNK):',
          '    with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter_paged"):',
          '        for s in pl.range(S):',
          '            ...',
          '            compress_state_flat = pl.assemble(compress_state_flat, kv_tile, [state_row, o0])',
          '            compress_state_flat = pl.assemble(compress_state_flat, score_tile, [state_row, OUT_DIM + o0])',
        ],
        after: [
          'with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter_paged"):',
          '    for o0 in pl.range(0, OUT_DIM, OUT_CHUNK):',
          '        for s in pl.range(S):',
          '            ...',
          '            compress_state_flat = pl.assemble(compress_state_flat, kv_tile, [state_row, o0])',
          '            compress_state_flat = pl.assemble(compress_state_flat, score_tile, [state_row, OUT_DIM + o0])',
        ],
        beforeNote: 'for 在 pl.at 外 → 每 batch 4 个独立 scope，各自重新赋值同一个 GM 句柄，链被拉长',
        afterNote: 'pl.at 包住 for → 每 batch 1 个 scope，链变短，不相交行的 batch 自由铺到 42 核',
        metrics: [
          { k: 'Total',     before: '1815 µs', after: '~1025 µs', delta: '−43%' },
          { k: 'scatter task 数', before: '512', after: '128', delta: '−75%' },
          { k: 'scatter 占用核数', before: '5',  after: '42',  delta: '8.4×' },
          { k: 'scatter span', before: '1602 µs', after: '197 µs', delta: '−88%' },
        ],
      },
      // T5 意图核对
      intents: [
        { intent: 'pipeline stage', req: 'stage=2', act: 'Right 上只放得下 1 份', ok: 'partial',
          ev: 'perf_hints.log · decode_compressor_ratio4.py:110', src: 'fact' },
        { intent: '权重 layout', req: 'b_trans=True（DN2ZN）', act: '已生效：right tile row_major/col_major', ok: 'yes',
          ev: 'ptoas/kv_score_proj.pto · alloc_tile loc=right', src: 'fact' },
        { intent: 'NZ 预打包', req: 'pl.Tensor[..., pl.NZ]', act: '静默忽略：NZ 与 ND 生成的 .pto 逐字节相同', ok: 'no',
          ev: '调优日志 §19 · _tmp_nz_weight_probe.py', src: 'fact' },
        { intent: '列广播减法', req: 'col_expand_sub', act: '报 No codegen registered，改用 col_expand_expdif', ok: 'no',
          ev: '调优日志 §20', src: 'fact' },
      ],
      layers: [
        { layer: '源码', file: 'decode_compressor_ratio4.py:110-136',
          body: 'pl.spmd(..., name_hint="kv_score_proj", deps=[late_dep])\n  for kb in pl.pipeline(0, D // K_TILE, stage=2):\n    pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True)' },
        { layer: 'IR', file: 'passes_dump/35_after_MemoryReuse.py:2229',
          body: 'def kv_score_proj(..., cmp_wkv__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16,\n    pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 8388608)], ...)' },
        { layer: 'PTO', file: 'ptoas/kv_score_proj.pto（50.8 KB）',
          body: 'alloc_tile loc=acc,   dtype=f32,  rows=16,  cols=64   addr=0 / 4096\nalloc_tile loc=mat,   dtype=bf16, rows=64,  cols=512  addr=163840\nalloc_tile loc=right, dtype=bf16, rows=256, cols=64   row_major/col_major' },
        { layer: 'Kernel', file: 'kernels/aic/kv_score_proj.cpp（33 KB）',
          body: 'TLOAD ×6   TEXTRACT ×24   TMATMUL ×2   TMATMUL_ACC ×10   TSTORE ×2\nset_flag / wait_flag ×33 对   pipe_barrier ×7' },
        { layer: '提示', file: 'report/perf_hints.log',
          body: '[PH-MR-001] ×5  Right 请求深度 2，只放得下 1 份（每份 32768 B，空闲 65536 B）\n[PH001]         最内维 256 B < 512 B（a2a3 L2 cache line）' },
      ],
      hints: {
        byFile: [
          { file: 'qkv_proj_rope.py',            n: 62, hot: false },
          { file: 'hc_pre.py',                   n: 41, hot: false },
          { file: 'decode_indexer.py',           n: 29, hot: false },
          { file: 'decode_compressor_ratio4.py', n: 23, hot: true },
          { file: 'decode_indexer_compressor.py',n: 22, hot: true },
          { file: 'decode_csa.py',               n: 17, hot: false },
          { file: 'decode_o_proj.py',            n: 13, hot: false },
          { file: 'decode_sparse_attn_csa.py',   n: 10, hot: false },
          { file: 'rmsnorm.py',                  n: 8,  hot: false },
          { file: 'hc_post.py',                  n: 5,  hot: false },
        ],
      },
      // T6 容量
      budget: [
        { space: 'Vec (UB)', used: 128.0, limit: 184.0, fn: 'dcr_xgamma', refs: 2,
          blocks: [{ n: 'mem_vec_5', kb: 64, range: '[0, 65536)', live: '[6, 13]' }, { n: 'mem_vec_6', kb: 64, range: '[65536, 131072)', live: '[7, 11]' }] },
        { space: 'Mat (L1)', used: 260.0, limit: 512.0, fn: 'down_proj', refs: 4, blocks: [] },
        { space: 'Right (L0B)', used: 64.0, limit: 64.0, fn: 'down_proj', refs: 2, blocks: [] },
      ],
      walls: [
        { want: 'qr_proj 输出 tile 256 → 512', wall: 'UB',    detail: 'qr_acc [T,512] INT32 = 256 KB > 192 KB', fix: '上限定为 256' },
        { want: 'qr_hadamard 跟随 rope 一起加大', wall: 'L0C', detail: '[HEAD_ROWS,128] FP32 在 GRP=4 时已占满 128 KB', fix: '拆成独立参数组' },
        { want: 'kv_score_proj 流水加深到 stage=3', wall: 'L1', detail: 'Mat 576 KB > 512 KB', fix: 'stage=2 已是上限' },
        { want: 'softmax_pool 头维加宽到 256', wall: 'UB',    detail: '3×[128,256] FP32 > 192 KB', fix: '上限定为 128' },
        { want: 'Qwen3 MLP TN=512（旁证）', wall: 'L1',       detail: '655360 > 524288', fix: '编译失败，改走其他路线' },
      ],
      movement: {
        before: { name: 'ND2NZ', desc: '权重 [K, N] 按行存储，读 [K_TILE, OUT_TILE] 时沿 K 是 256 笔 64 宽的碎 burst', gbps: 21 },
        after:  { name: 'DN2ZN', desc: '权重转置存成 [N, K]，读 [OUT_TILE, K_TILE] 时每行 K 个连续 = 长 burst', gbps: null },
        results: [
          { k: 'kv_score_proj busy', before: '6456 µs', after: '5479 µs', delta: '−15.1%' },
          { k: 'compressor 总 wall', before: '294.6 µs', after: '270.2 µs', delta: '−8.3%' },
          { k: 'CSA 整图 wall（3 跑中位数）', before: '4304 µs', after: '3750 µs', delta: '−12.9%' },
        ],
      },
      // T7 核内
      incore: [
        { unit: 'MTE2（GM→片上加载）', cat: '搬运', pct: 69, cyclesBefore: 450249, cyclesAfter: 43474, delta: '−90.3%',
          detail: '256 笔 [1,64] 散读，每笔约 211 ns → 32 笔 [8,64] 块读' },
        { unit: 'VECTOR', cat: '大部分是搬运', pct: 63, cyclesBefore: 473236, cyclesAfter: 91948, delta: '−80.6%',
          detail: '逐行 MOV_UB_TO_UB staging + 两次 VNCHWCONV 转置；真正的 VEXP/VDIV/VCADD 合计 < 1 µs' },
      ],
      incoreSpan: { before: 20.58, after: 7.73, delta: '−62.4%' },
      // T8 实验
      ab: [
        { ver: '原始（逐行 + 转置）', wall: 1303.6, target: 9495, targetN: 512, anchor: 22301, verdict: 'base' },
        { ver: 'R1 块读 HT64',       wall: 1276.2, target: 3203, targetN: 512, anchor: 19754, verdict: 'noisy' },
        { ver: 'R2 块读 + 列归约 HT128', wall: 1190.1, target: 2792, targetN: 256, anchor: 21918, verdict: 'ok' },
      ],
      sweep: [
        { n: 2, tile: 'N=128', total: 621, best: false },
        { n: 4, tile: 'N=64',  total: 582, best: true },
        { n: 8, tile: 'N=32',  total: 598, best: false },
      ],
      // T9 交付
      impact: [
        { file: 'decode_compressor_ratio4.py', change: 'wkv / wgate → [OUT_DIM, D]，matmul 加 b_trans=True', must: true },
        { file: 'decode_attention_csa.py', change: 'cmp_wkv / cmp_wgate → [MAIN_OUT_DIM, D]（含 golden / specs）', must: true },
        { file: 'decode_layer.py', change: 'csa_cmp_wkv / wgate', must: true },
        { file: 'decode_fwd.py', change: 'stacked [NUM_LAYERS*MAIN_OUT_DIM, D]，per-layer slice 偏移', must: true },
        { file: 'compressor_ratio128（HCA 走这条）', change: '不受影响，不要误改', must: false },
        { file: 'indexer_compressor（inner 走这条）', change: '不受影响，不要误改', must: false },
      ],
      recipe: {
        rule: '分页池的 pooling 若 MTE2-bound 且窗口按块对齐，就把逐行 [1,N] 散读换成逐块 [block_size, N] 跨步读',
        scope: ['平台 a2a3（910B）', 'ratio128 / HCA 形态', '窗口起点是 block_size 的倍数', 'pypto-lib @ 2026-06'],
        evidence: 'PR #641：scope core-time −71%，整图 wall −8.7%，max_error_ratio = 0.0',
        counter: 'b_trans 在 Qwen3-14B 浅 K tile（TK=64）反而慢 15.8%–23.2%（pypto#2309）——同一手段，条件不同结论相反',
      },
    },
  };
})();
