/* ──────────────────────────────────────────────────────────────────────────────
 * Hardware-Native Lens · 语义模型
 *
 * 全部结构、常量与约束来自 Design/Toolkit Studio/decode_layer.py
 * (Qwen3-14B fused decode · FP32 跨层 carry)。行号与该文件一致，便于回溯。
 * ────────────────────────────────────────────────────────────────────────────── */
window.HNL = (function () {
  'use strict';

  // ── 由 checkpoint 固定的模型维度 ──
  const NUM_HEADS = 40;
  const INTERMEDIATE = 17408;
  const MAX_SEQ = 4096;

  // ── 由工具链/硬件固定的 tile 常量 ──
  const TK = 256;                 // 内层 K chunk
  const TN = 256;                 // 内层 N sub-tile
  const QKV_N_TILE = 512;         // 外层 N-tile
  const RMSNORM_K_CHUNK = 256;
  const MLP_OUT_CHUNK = 256;      // silu 内层流水子块
  const CUBE_M_FRACTAL = 16;      // matmul L0 内维对齐粒度
  const L0B_CAP = 64 * 1024;      // Right(L0B) 容量

  // ── Vec(UB) 预算：名义容量与实际可用容量不是一个数 ──
  // 抽象自 pto-isa#170：a2a3 的 Vec UB 名义 192 KB，但顶部约 8 KB 被运行时保留，
  // 上游分配器按名义容量放行，落在保留区的 tile 读写垃圾 → 静默 NaN。
  const UB_NOMINAL = 192 * 1024;              // 分配器用的预算
  const UB_RESERVED = 8 * 1024;               // 顶部保留（ReserveBufferOp / pipe init）
  const UB_USABLE = UB_NOMINAL - UB_RESERVED; // 真正能安全使用的上限

  // 一个有长度倾斜的 ragged decode batch（用于块级负载均衡演示）
  const SEQ_LENS = [
    3972, 1184, 2560, 640, 3328, 896, 2048, 1472,
    3712, 512, 2816, 1920, 1088, 3456, 768, 2304,
    3104, 1664, 2176, 832, 3840, 1024, 2432, 1536,
    3584, 704, 2944, 1792, 1216, 3200, 960, 2688
  ];

  // ════════════════════════════════════════════════════════════════════════════
  // 可调旋钮 —— 改动后所有形状 / 布局 / 约束立刻重算（约束前移的演示载体）
  // ════════════════════════════════════════════════════════════════════════════
  const KNOBS = [
    { id: 'BATCH',        label: 'Decode batch',      note: '请求并发度',            options: [4, 8, 16, 32],   def: 16 },
    { id: 'NUM_KV_HEADS', label: 'KV heads',          note: 'GQA 分组',              options: [4, 8, 10],       def: 8 },
    { id: 'HEAD_DIM',     label: 'Head dim',          note: '每头宽度',              options: [64, 128],        def: 128 },
    { id: 'SEQ_TILE',     label: 'SEQ_TILE',          note: 'KV block 的 seq 长度',  options: [64, 128, 256],   def: 128 },
    { id: 'PAGE_SIZE',    label: 'runtime.page_size', note: 'paged KV 物理页',       options: [64, 128, 256],   def: 128 },
    { id: 'QKV_OK',       label: 'QKV split-K',       note: 'atomic-add 切片数',     options: [4, 5, 10],       def: 5 },
    { id: 'DOWN_TN',      label: 'DOWN_TN / MLP_TN',  note: '输出 N-tile 宽度',      options: [512, 1024, 2048], def: 1024 },
    { id: 'GP_SIZE',      label: 'GP_SIZE',           note: '每 work item 的头数',   options: [2, 4, 8],        def: 8 },
    { id: 'FA_STAGE',     label: 'fa_fused stage',    note: '内层流水同时驻留几级；每级各占一份 softmax 中间量，直接决定 Vec(UB) 高水位。实际驻留不会超过循环次数 GP_SIZE。', options: [1, 2, 3, 4], def: 2 },
    { id: 'NUM_CORES',    label: 'fa_fused cores',    note: '常驻 spmd block 数',    options: [20, 24, 32],     def: 24 },
    { id: 'ROPE_CORES',   label: 'rope_qkv cores',    note: '单次 spmd 栅格',        options: [16, 24, 32],     def: 32 }
  ];

  const DEFAULTS = KNOBS.reduce(function (acc, k) { acc[k.id] = k.def; return acc; }, {});

  // ── 格式化 ──
  function bytes(n) {
    if (n >= 1024 * 1024) return (n / 1024 / 1024).toFixed(n >= 10 * 1024 * 1024 ? 0 : 1) + ' MB';
    if (n >= 1024) return (n / 1024).toFixed(n >= 10 * 1024 ? 0 : 1) + ' KB';
    return n + ' B';
  }
  function num(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ','); }
  function isInt(n) { return Number.isInteger(n); }

  // ════════════════════════════════════════════════════════════════════════════
  // 派生 —— 单一真相源：所有展示值都从这里出
  // ════════════════════════════════════════════════════════════════════════════
  function derive(cfg) {
    const HEAD_DIM = cfg.HEAD_DIM;
    const NUM_KV_HEADS = cfg.NUM_KV_HEADS;
    const BATCH = cfg.BATCH;
    const SEQ_TILE = cfg.SEQ_TILE;
    const PAGE_SIZE = cfg.PAGE_SIZE;
    const QKV_OK = cfg.QKV_OK;
    const DOWN_TN = cfg.DOWN_TN;
    const GP_SIZE = cfg.GP_SIZE;
    const NUM_CORES = cfg.NUM_CORES;
    const ROPE_CORES = cfg.ROPE_CORES;

    const HIDDEN = NUM_HEADS * HEAD_DIM;
    const KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM;
    const Q_PER_KV = NUM_HEADS / NUM_KV_HEADS;
    const Q_HEAD_BATCH = Q_PER_KV;
    const Q_HEAD_PAD = Math.ceil(Q_HEAD_BATCH / CUBE_M_FRACTAL) * CUBE_M_FRACTAL;
    const HALF_DIM = HEAD_DIM / 2;

    const MAX_CTX_BLOCKS = Math.ceil(MAX_SEQ / SEQ_TILE);
    const QKV_K_SLICE = HIDDEN / QKV_OK;
    const QKV_K_CHUNKS = QKV_K_SLICE / TK;
    const Q_ON = HIDDEN / QKV_N_TILE;
    const KV_ON = KV_HIDDEN / QKV_N_TILE;
    const DOWN_ON = HIDDEN / DOWN_TN;
    const K_SPLITS = INTERMEDIATE / DOWN_TN;
    const ROPE_ITEMS = NUM_KV_HEADS * BATCH;

    // 块级负载均衡：只把 ragged batch 的真实块压进 work table
    const seq = SEQ_LENS.slice(0, BATCH);
    const blocksPerSeq = seq.map(function (L) { return Math.ceil(L / SEQ_TILE); });
    const REAL_BLOCKS = blocksPerSeq.reduce(function (a, b) { return a + b; }, 0);
    const perCore = new Array(NUM_CORES).fill(0);
    for (let w = 0; w < REAL_BLOCKS; w++) perCore[w % NUM_CORES] += 1;
    const MAX_CORE = Math.max.apply(null, perCore);
    const MIN_CORE = Math.min.apply(null, perCore);
    const SKEW = MAX_CORE / (REAL_BLOCKS / NUM_CORES);

    // 片上占用
    const K_TILE_B = SEQ_TILE * HEAD_DIM * 2;          // BF16 K tile
    const K_TILE_DB = K_TILE_B * 2;                    // 双缓冲
    const Q_TILE_B = Q_HEAD_PAD * HEAD_DIM * 2;
    const ACC_TILE_B = Q_HEAD_PAD * SEQ_TILE * 4;      // FP32 scores
    const PAD_WASTE = (Q_HEAD_PAD - Q_HEAD_BATCH) / Q_HEAD_PAD;

    // ── fa_fused 内层流水的 Vec(UB) 高水位 ──
    // 每个驻留级一份 softmax 链：5 个 [Q_HEAD_PAD, SEQ_TILE] FP32 中间量
    // (scores_scaled / scores_valid / scores / row_expand_sub 的展开 temp / exp_scores)
    // + 1 个同形 BF16 (exp_scores_bf16) + 2 个 [Q_HEAD_PAD, 1] 行向量 (cur_mi / cur_li)。
    // 这是按「每个中间量各占一块」估算；是否复用、是否就地要按编译产物核对。
    const FA_STAGE = cfg.FA_STAGE;
    const FA_STAGE_EFF = Math.min(FA_STAGE, GP_SIZE);   // 流水深度不会超过循环次数
    const VEC_SCORE_B = Q_HEAD_PAD * SEQ_TILE * 4;
    const VEC_SCORE_BF16_B = Q_HEAD_PAD * SEQ_TILE * 2;
    const VEC_ROW_B = Q_HEAD_PAD * 4;
    const VEC_PER_STAGE = VEC_SCORE_B * 5 + VEC_SCORE_BF16_B + VEC_ROW_B * 2;
    // 跨块常驻的 online-softmax carry：o_i 累加器 + m_i / l_i
    const VEC_CARRY = Q_HEAD_PAD * (HEAD_DIM * 4 + 8);
    const VEC_HW = VEC_PER_STAGE * FA_STAGE_EFF + VEC_CARRY;

    const d = {
      NUM_HEADS: NUM_HEADS, INTERMEDIATE: INTERMEDIATE, MAX_SEQ: MAX_SEQ,
      TK: TK, TN: TN, QKV_N_TILE: QKV_N_TILE, RMSNORM_K_CHUNK: RMSNORM_K_CHUNK,
      MLP_OUT_CHUNK: MLP_OUT_CHUNK, CUBE_M_FRACTAL: CUBE_M_FRACTAL, L0B_CAP: L0B_CAP,
      BATCH: BATCH, NUM_KV_HEADS: NUM_KV_HEADS, HEAD_DIM: HEAD_DIM, HALF_DIM: HALF_DIM,
      SEQ_TILE: SEQ_TILE, PAGE_SIZE: PAGE_SIZE, QKV_OK: QKV_OK, DOWN_TN: DOWN_TN,
      GP_SIZE: GP_SIZE, NUM_CORES: NUM_CORES, ROPE_CORES: ROPE_CORES,
      HIDDEN: HIDDEN, KV_HIDDEN: KV_HIDDEN, Q_PER_KV: Q_PER_KV,
      Q_HEAD_BATCH: Q_HEAD_BATCH, Q_HEAD_PAD: Q_HEAD_PAD,
      MAX_CTX_BLOCKS: MAX_CTX_BLOCKS, QKV_K_SLICE: QKV_K_SLICE, QKV_K_CHUNKS: QKV_K_CHUNKS,
      Q_ON: Q_ON, KV_ON: KV_ON, DOWN_ON: DOWN_ON, K_SPLITS: K_SPLITS,
      ROPE_ITEMS: ROPE_ITEMS, ROPE_ITEMS_PER_CORE: ROPE_ITEMS / ROPE_CORES,
      REAL_BLOCKS: REAL_BLOCKS, blocksPerSeq: blocksPerSeq, perCore: perCore,
      MAX_CORE: MAX_CORE, MIN_CORE: MIN_CORE, SKEW: SKEW,
      K_TILE_B: K_TILE_B, K_TILE_DB: K_TILE_DB, Q_TILE_B: Q_TILE_B, ACC_TILE_B: ACC_TILE_B,
      PAD_WASTE: PAD_WASTE,
      FA_STAGE: FA_STAGE, FA_STAGE_EFF: FA_STAGE_EFF,
      VEC_SCORE_B: VEC_SCORE_B, VEC_SCORE_BF16_B: VEC_SCORE_BF16_B, VEC_ROW_B: VEC_ROW_B,
      VEC_PER_STAGE: VEC_PER_STAGE, VEC_CARRY: VEC_CARRY, VEC_HW: VEC_HW,
      UB_NOMINAL: UB_NOMINAL, UB_RESERVED: UB_RESERVED, UB_USABLE: UB_USABLE,
      FA_TASKS: NUM_CORES, Q_TASKS: Q_ON * QKV_OK, KV_TASKS: KV_ON * QKV_OK
    };

    // 展示字符串
    d.HIDDEN_FMT = num(HIDDEN);
    d.ALLQ_ROWS = BATCH * NUM_KV_HEADS * Q_HEAD_PAD;
    d.ALLQ_ROWS_FMT = num(d.ALLQ_ROWS);
    d.ALLQ_BYTES = bytes(d.ALLQ_ROWS * HEAD_DIM * 2);
    d.HIDDEN_BYTES = bytes(BATCH * HIDDEN * 4);
    d.NORMED_BYTES = bytes(BATCH * HIDDEN * 2);
    d.QPROJ_BYTES = bytes(BATCH * HIDDEN * 4);
    d.K_TILE_KB = (K_TILE_B / 1024).toFixed(K_TILE_B % 1024 ? 1 : 0);
    d.K_TILE_DB_KB = (K_TILE_DB / 1024).toFixed(K_TILE_DB % 1024 ? 1 : 0);
    d.PAD_WASTE_PCT = (PAD_WASTE * 100).toFixed(1);
    d.SKEW_FMT = SKEW.toFixed(2);
    d.FA_TABLE_CAP = BATCH * MAX_CTX_BLOCKS;
    d.FA_TABLE_CAP_FMT = num(d.FA_TABLE_CAP);
    d.Q_TASK_TOTAL = Q_ON * QKV_OK;
    d.RMS_CHUNKS = HIDDEN / RMSNORM_K_CHUNK;

    // Vec(UB) 预算的展示值
    function kb(n) { return (n / 1024).toFixed(n % 1024 ? 1 : 0); }
    d.VEC_HW_KB = kb(VEC_HW);
    d.VEC_PER_STAGE_KB = kb(VEC_PER_STAGE);
    d.VEC_CARRY_KB = kb(VEC_CARRY);
    d.UB_NOMINAL_KB = kb(UB_NOMINAL);
    d.UB_USABLE_KB = kb(UB_USABLE);
    d.UB_RESERVED_KB = kb(UB_RESERVED);
    d.VEC_HW_PCT = (VEC_HW / UB_USABLE * 100).toFixed(1);
    d.VEC_OVER_USABLE = Math.max(0, VEC_HW - UB_USABLE);
    d.VEC_OVER_USABLE_FMT = bytes(d.VEC_OVER_USABLE);
    d.VEC_UNDER_NOMINAL = Math.max(0, UB_NOMINAL - VEC_HW);
    d.VEC_UNDER_NOMINAL_FMT = bytes(d.VEC_UNDER_NOMINAL);
    d.FA_STAGE_CLAMPED = FA_STAGE_EFF < FA_STAGE;
    // 高水位的构成，给地址图用（自低地址向高地址排）
    d.vecBuffers = (function () {
      const out = [];
      out.push({ name: 'online carry (o_i / m_i / l_i)', size: VEC_CARRY, kind: 'carry' });
      for (let s = 0; s < FA_STAGE_EFF; s++) {
        out.push({ name: 'stage ' + s + ' · softmax 链', size: VEC_PER_STAGE, kind: 'stage', stage: s });
      }
      return out;
    })();
    d.bytes = bytes;
    d.num = num;
    return d;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 硬件约束 —— 每条都标注「硬件事实」与「若不前移，会在哪一环暴露」
  // ════════════════════════════════════════════════════════════════════════════
  const CONSTRAINTS = [
    {
      id: 'cube-m-fractal',
      lens: 'layout',
      title: 'Q 头打包行数必须对齐 cube M fractal',
      rule: 'Q_HEAD_PAD = ceil(Q_HEAD_BATCH / 16) × 16',
      knobs: ['NUM_KV_HEADS'],
      origin: '本例以 16 行作为 Cube M 方向对齐模型；具体 tile 布局和合法性需按目标芯片、dtype 与编译产物核对。set_validshape 另外标识有效区域。',
      late: '编译期 alloc_tile 报错，或 tile 被静默放大导致 C2V 传输量翻倍。',
      lines: [127, 768, 781],
      eval: function (d) {
        const ok = d.Q_HEAD_PAD % 16 === 0 && d.Q_HEAD_PAD >= d.Q_HEAD_BATCH;
        return {
          status: !ok ? 'error' : (d.PAD_WASTE >= 0.5 ? 'warn' : 'ok'),
          value: d.Q_HEAD_BATCH + ' 真实行 / ' + d.Q_HEAD_PAD + ' 物理行',
          detail: !ok
            ? 'Q_HEAD_PAD 未对齐 16'
            : 'tpush/tpop 搬运整块 ' + d.Q_HEAD_PAD + ' 行，填充占 ' + d.PAD_WASTE_PCT + '%；'
              + 'set_validshape 只把 ' + d.Q_HEAD_BATCH + ' 行标为有效。'
        };
      }
    },
    {
      id: 'page-identity',
      lens: 'layout',
      title: 'SEQ_TILE 必须等于 runtime 的物理页大小',
      rule: 'SEQ_TILE == runtime.page_size',
      knobs: ['SEQ_TILE', 'PAGE_SIZE'],
      origin: 'decode 经 block_table 读 paged KV。一个逻辑块必须正好落在一个物理页上，否则会跨到物理 id 无关的另一页。',
      late: '运行时读到另一个请求的 KV —— 数值错误，且不一定复现。',
      lines: [176, 758, 773],
      eval: function (d) {
        const ok = d.SEQ_TILE === d.PAGE_SIZE;
        return {
          status: ok ? 'ok' : 'error',
          value: 'SEQ_TILE ' + d.SEQ_TILE + ' · page ' + d.PAGE_SIZE,
          detail: ok
            ? '一个逻辑块 = 一个物理页，cache_row 算术成立。'
            : '逻辑块跨 ' + (d.SEQ_TILE / d.PAGE_SIZE).toFixed(2) + ' 个物理页，block_table 映射失效。'
        };
      }
    },
    {
      id: 'l0b-capacity',
      lens: 'tile',
      title: 'K tile 双缓冲的 Right(L0B) 容量估算',
      rule: 'SEQ_TILE × HEAD_DIM × 2B × 2 ≤ 64 KB',
      knobs: ['SEQ_TILE', 'HEAD_DIM'],
      origin: '此估算假设两个完整 K tile 同时驻留 Right，并采用示例中的 64 KiB 容量。实际是否双缓冲、如何切分和驻留需查看目标配置与编译产物。',
      late: 'ptoas 分配失败，或退化为单缓冲 —— AIC 每拍等搬运。',
      lines: [176, 773],
      eval: function (d) {
        const r = d.K_TILE_DB / L0B_CAP;
        return {
          status: r > 0.75 ? 'warn' : 'ok',
          value: d.K_TILE_DB_KB + ' KB / 64 KB',
          detail: r > 1
            ? '按完整 tile 双缓冲估算超出 ' + ((r - 1) * 100).toFixed(0) + '%；实际映射待核实。'
            : '假设完整 tile 双缓冲，占示例容量 ' + (r * 100).toFixed(0) + '%；实际映射待核实。'
        };
      }
    },
    {
      id: 'ub-usable-headroom',
      lens: 'tile',
      title: 'Vec 高水位必须留出 UB 顶部的保留区',
      rule: 'vec_high_water ≤ UB_NOMINAL − UB_RESERVED（184 KB，而非 192 KB）',
      knobs: ['FA_STAGE', 'SEQ_TILE', 'HEAD_DIM', 'NUM_KV_HEADS', 'GP_SIZE'],
      viz: 'ub-budget',
      origin: 'Vec UB 名义 192 KB，但顶部约 8 KB 被运行时保留（ReserveBufferOp / pipe init）。'
        + '上游分配器按名义 192 KB 发预算，所以落在保留区的 tile 能通过编译，运行时读写的是垃圾。'
        + '保留区大小随平台与 CANN 版本变化，需按目标配置核对。',
      late: '最坏的一种：编译不报错、运行不报错，最高地址的那块 tile 出 NaN。'
        + '现象只在高水位越过保留区边界时出现，改小 tile 或把流水降到 stage=1 就消失，'
        + '所以极易被当成数值问题或偶发故障。',
      lines: [764, 775, 782, 784, 786],
      eval: function (d) {
        const hw = d.VEC_HW;
        if (hw > UB_NOMINAL) {
          return {
            status: 'error',
            value: d.VEC_HW_KB + ' KB / ' + d.UB_NOMINAL_KB + ' KB',
            detail: '超出名义容量 ' + bytes(hw - UB_NOMINAL) + '，分配器自己会报错 —— 这反而是能看见的失败。'
          };
        }
        if (hw > UB_USABLE) {
          return {
            status: 'error',
            value: d.VEC_HW_KB + ' KB / ' + d.UB_USABLE_KB + ' KB 可用',
            detail: '越过可用上限 ' + d.VEC_OVER_USABLE_FMT + '，但距名义上限还有 '
              + d.VEC_UNDER_NOMINAL_FMT + '：编译通过、运行不报错，最高地址的 tile 静默出 NaN。'
          };
        }
        return {
          status: hw > UB_USABLE * 0.92 ? 'warn' : 'ok',
          value: d.VEC_HW_KB + ' KB / ' + d.UB_USABLE_KB + ' KB 可用',
          detail: d.FA_STAGE_EFF + ' 级 × ' + d.VEC_PER_STAGE_KB + ' KB + carry '
            + d.VEC_CARRY_KB + ' KB，占可用容量 ' + d.VEC_HW_PCT + '%；'
            + '距保留区边界还有 ' + bytes(UB_USABLE - hw) + '。'
            + (d.FA_STAGE_CLAMPED ? ' stage=' + d.FA_STAGE + ' 被循环次数 GP_SIZE='
              + d.GP_SIZE + ' 限制为 ' + d.FA_STAGE_EFF + ' 级。' : '')
        };
      }
    },
    {
      id: 'slab-1to1',
      lens: 'scope',
      title: 'split-K 切片宽度必须等于上一层 normed slab 宽度',
      rule: 'QKV_K_SLICE == DOWN_TN',
      knobs: ['QKV_OK', 'DOWN_TN', 'HEAD_DIM'],
      origin: 'manual_scope 内 tensormap 关闭，q_proj 只能用 deps=[prev_normed_tids[q_ks]] 精确挂到上一层第 q_ks 个 slab 写者。',
      late: '依赖边挂错 slab：要么整父张量 WAW 串行化（性能塌陷），要么读到未写完的数据。',
      lines: [166, 262, 441],
      eval: function (d) {
        const ok = isInt(d.QKV_K_SLICE) && d.QKV_K_SLICE === d.DOWN_TN;
        return {
          status: ok ? 'ok' : 'error',
          value: 'K_SLICE ' + (isInt(d.QKV_K_SLICE) ? d.QKV_K_SLICE : d.QKV_K_SLICE.toFixed(1)) + ' · DOWN_TN ' + d.DOWN_TN,
          detail: ok
            ? 'prev_normed_tids[q_ks] 与 K 切片 1:1 对应。'
            : '切片与 slab 不同宽，索引 q_ks 不再指向唯一写者。'
        };
      }
    },
    {
      id: 'down-chunk-align',
      lens: 'tile',
      title: '输出 N-tile 必须整除 HIDDEN 且是流水子块的整数倍',
      rule: 'HIDDEN % DOWN_TN == 0 且 DOWN_TN % 256 == 0',
      knobs: ['DOWN_TN', 'HEAD_DIM'],
      origin: 'down/MLP 以 DOWN_TN 为单位 atomic-add 回 HIDDEN；silu 的内层流水以 256 列为子块。',
      late: '尾块形状不规整，assemble 偏移越界或退化成标量搬运。',
      lines: [262, 264],
      eval: function (d) {
        const ok = d.HIDDEN % d.DOWN_TN === 0 && d.DOWN_TN % MLP_OUT_CHUNK === 0;
        return {
          status: ok ? 'ok' : 'error',
          value: 'DOWN_ON ' + (isInt(d.DOWN_ON) ? d.DOWN_ON : d.DOWN_ON.toFixed(2)) + ' × ' + d.DOWN_TN,
          detail: ok
            ? 'HIDDEN 被整除，每个 N-block 都是 ' + (d.DOWN_TN / MLP_OUT_CHUNK) + ' 个 256 列子块。'
            : 'HIDDEN ' + d.HIDDEN + ' 无法被 DOWN_TN ' + d.DOWN_TN + ' 整除。'
        };
      }
    },
    {
      id: 'splitk-divides-k',
      lens: 'scope',
      title: 'split-K 切片数必须整除 K 维',
      rule: 'HIDDEN % QKV_OK == 0',
      knobs: ['QKV_OK', 'HEAD_DIM'],
      origin: 'QKV 投影按 K 切片后 atomic-add 累加；切不齐则最后一片形状不同，无法共用同一 task 体。',
      late: '编译期形状推导失败。',
      lines: [165, 166],
      eval: function (d) {
        const ok = d.HIDDEN % d.QKV_OK === 0;
        return {
          status: ok ? 'ok' : 'error',
          value: d.HIDDEN + ' / ' + d.QKV_OK,
          detail: ok ? '每片 ' + d.QKV_K_SLICE + ' 列，含 ' + d.QKV_K_CHUNKS + ' 个 TK=256 chunk。'
                     : '余 ' + (d.HIDDEN % d.QKV_OK) + ' 列，尾片形状不一致。'
        };
      }
    },
    {
      id: 'gp-head-cover',
      lens: 'parallel',
      title: 'work table 编码 (b, block) 时 GP_SIZE 必须覆盖全部 KV 头',
      rule: 'GP_SIZE == NUM_KV_HEADS',
      knobs: ['GP_SIZE', 'NUM_KV_HEADS'],
      origin: 'fa_work_table[w] 只编码 b × MAX_CTX_BLOCKS + p，没有头维；一个 work item 必须一次处理该块的所有头。',
      late: '部分头永远不被调度 —— 输出静默缺失一部分 attention。',
      lines: [208, 744, 764],
      eval: function (d) {
        const ok = d.GP_SIZE === d.NUM_KV_HEADS;
        return {
          status: ok ? 'ok' : 'error',
          value: 'GP_SIZE ' + d.GP_SIZE + ' / KV heads ' + d.NUM_KV_HEADS,
          detail: ok
            ? 'HEAD_GROUPS = 1，(b, block) 编码足够。'
            : '需要 ' + (d.NUM_KV_HEADS / d.GP_SIZE) + ' 个 head group，work table 缺少头维度。'
        };
      }
    },
    {
      id: 'spmd-even-split',
      lens: 'parallel',
      title: 'spmd 静态栅格必须整除 work item 数',
      rule: '(KV heads × batch) % ROPE_CORES == 0',
      knobs: ['ROPE_CORES', 'BATCH', 'NUM_KV_HEADS'],
      origin: 'rope_qkv 用固定 item/core 的仿射索引，不做 grid-stride 兜底。',
      late: '尾部 item 不被执行，或越界读 inv_rms_states。',
      lines: [234, 615, 628],
      eval: function (d) {
        const ok = d.ROPE_ITEMS % d.ROPE_CORES === 0;
        return {
          status: ok ? 'ok' : 'error',
          value: d.ROPE_ITEMS + ' items / ' + d.ROPE_CORES + ' cores',
          detail: ok
            ? '每核 ' + d.ROPE_ITEMS_PER_CORE + ' 个 item，无尾块。'
            : '余 ' + (d.ROPE_ITEMS % d.ROPE_CORES) + ' 个 item 落在栅格之外。'
        };
      }
    },
    {
      id: 'block-balance',
      lens: 'parallel',
      title: '块级 grid-stride 的整数打包偏差',
      rule: 'ceil(real_blocks / cores) / (real_blocks / cores) ≲ 1.25',
      knobs: ['NUM_CORES', 'BATCH', 'SEQ_TILE'],
      origin: '每个 work item 是一个等价的 SEQ_TILE 块，均衡度只受整数打包限制，与 batch 内长度倾斜无关。',
      late: '上线后才看到核间等待，需要回到 tile 配置重做。',
      lines: [223, 744],
      eval: function (d) {
        return {
          status: d.SKEW > 1.25 ? 'warn' : 'ok',
          value: d.SKEW_FMT + '× 理想',
          detail: d.REAL_BLOCKS + ' 个真实块分到 ' + d.NUM_CORES + ' 核：最忙 ' + d.MAX_CORE
            + ' 块，最闲 ' + d.MIN_CORE + ' 块。'
        };
      }
    }
  ];

  // ════════════════════════════════════════════════════════════════════════════
  // 源码视图 —— 节选自 decode_layer.py，行号保留原值
  // ════════════════════════════════════════════════════════════════════════════
  const SOURCE_BASE = [
    { sec: '硬件可见常量', hint: '这些不是超参，而是硬件事实在代码里的落点' },
    { no: 103, t: 'NUM_HEADS = 40' },
    { no: 104, t: 'NUM_KV_HEADS = {{NUM_KV_HEADS}}' },
    { no: 105, t: 'HEAD_DIM = {{HEAD_DIM}}' },
    { no: 117, t: 'HIDDEN = NUM_HEADS * HEAD_DIM  # {{HIDDEN}}' },
    { no: 119, t: 'Q_PER_KV = NUM_HEADS // NUM_KV_HEADS  # {{Q_PER_KV}} (GQA ratio)' },
    { no: 126, t: 'Q_HEAD_BATCH = Q_PER_KV  # {{Q_HEAD_BATCH}}: 每个 attention lane 打包的 Q 头' },
    {
      no: 127, t: 'Q_HEAD_PAD = ((Q_HEAD_BATCH + 15) // 16) * 16  # {{Q_HEAD_PAD}}',
      focus: 'layout:q-pad',
      inlay: [{ lens: 'layout', text: '物理 {{Q_HEAD_PAD}} 行 · 有效 {{Q_HEAD_BATCH}} 行 · 填充 {{PAD_WASTE_PCT}}%' }]
    },
    { fold: 37 },
    {
      no: 165, t: 'QKV_OK = {{QKV_OK}}  # split-K 切片数(atomic-add)',
      focus: 'layout:split-k'
    },
    {
      no: 166, t: 'QKV_K_SLICE = HIDDEN // QKV_OK  # {{QKV_K_SLICE}}',
      focus: 'layout:split-k',
      inlay: [{ lens: 'layout', text: '必须 1:1 对齐上一层 slab 宽度 DOWN_TN={{DOWN_TN}}' }]
    },
    {
      no: 176, t: 'SEQ_TILE = {{SEQ_TILE}}  # 每个 KV block 的 seq 长度',
      focus: 'layout:paged-kv',
      inlay: [{ lens: 'tile', text: 'K tile {{SEQ_TILE}}×{{HEAD_DIM}} BF16 = {{K_TILE_KB}} KB ×2 buf' }]
    },
    { no: 177, t: 'BLOCK_SIZE = SEQ_TILE  # paged page size (== runtime.page_size)', focus: 'layout:paged-kv' },
    {
      no: 208, t: 'GP_SIZE = {{GP_SIZE}}  # 每个 work item 打包的 KV 头数',
      focus: 'parallel:grid-stride'
    },
    {
      no: 223, t: 'NUM_CORES = {{NUM_CORES}}  # fa_fused 常驻 block 数',
      focus: 'parallel:spmd',
      inlay: [{ lens: 'parallel', text: '{{REAL_BLOCKS}} 真实块 / {{NUM_CORES}} 核 · 偏差 {{SKEW_FMT}}×' }]
    },
    { no: 234, t: 'ROPE_CORES = {{ROPE_CORES}}  # rope_qkv 单次 spmd 栅格', focus: 'parallel:spmd' },
    {
      no: 262, t: 'DOWN_TN = {{DOWN_TN}}  # down/MLP 输出 N-tile(== MLP_TN)',
      focus: 'layout:split-k'
    },
    { no: 264, t: 'DOWN_ON = HIDDEN // DOWN_TN  # {{DOWN_ON}} 个输出 N-block', focus: 'layout:split-k' },
    { fold: 6 },
    { no: 281, t: 'assert NUM_KV_HEADS % GP_SIZE == 0, "GP_SIZE must divide NUM_KV_HEADS"' },
    { no: 283, t: 'assert GP_SIZE == NUM_KV_HEADS, "block-level work table encodes (b, block)"' },
    { no: 291, t: 'assert MLP_TN == DOWN_TN, "silu/down K-slice alignment requires MLP_TN == DOWN_TN"' },
    { fold: 11 },

    { sec: '张量契约', hint: '形状、dtype、所在内存空间都写在签名里' },
    { no: 303, t: '@pl.jit.inline' },
    { no: 304, t: 'def _decode_layer(' },
    {
      no: 305, t: '    hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.FP32],',
      focus: 'tensor:hidden-states',
      inlay: [{ lens: 'tensor', text: '[{{BATCH}}, {{HIDDEN}}] FP32 · GM · {{HIDDEN_BYTES}}' }]
    },
    {
      no: 327, t: '    normed_in: pl.Tensor[[BATCH, HIDDEN], pl.BF16],',
      focus: 'tensor:normed',
      inlay: [{ lens: 'tensor', text: '[{{BATCH}}, {{HIDDEN}}] BF16 · GM · 上一层 dcr_xgamma 产出' }]
    },
    {
      no: 330, t: '    normed_out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],',
      focus: 'tensor:normed',
      inlay: [{ lens: 'tensor', text: '写出给下一层 · 省掉一次 GM 往返' }]
    },
    {
      no: 337, t: '    prev_out_tids: pl.Array[DOWN_ON, pl.TASK_ID],',
      focus: 'scope:manual',
      inlay: [{ lens: 'scope', text: '{{DOWN_ON}} 个跨层依赖句柄 · 按 slab 粒度' }]
    },
    { no: 339, t: ') -> pl.Tensor[[BATCH, HIDDEN], pl.FP32]:' },
    { fold: 30 },
    {
      no: 370, t: '    q_proj = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)',
      focus: 'tensor:q-proj',
      inlay: [{ lens: 'tensor', text: 'GM scratch · {{QPROJ_BYTES}} · split-K atomic 目标' }]
    },
    {
      no: 392, t: '    all_q_padded = pl.create_tensor([BATCH * NUM_KV_HEADS * Q_HEAD_PAD, HEAD_DIM], dtype=pl.BF16)',
      focus: 'layout:q-pad',
      inlay: [
        { lens: 'tensor', text: '{{ALLQ_ROWS_FMT}} 行 × {{HEAD_DIM}} BF16 · {{ALLQ_BYTES}}' },
        { lens: 'layout', text: '行距 Q_HEAD_PAD={{Q_HEAD_PAD}} 而非 {{Q_HEAD_BATCH}}' }
      ]
    },
    {
      no: 407, t: '    fa_work_table = pl.create_tensor([FA_TABLE_CAP, 1], dtype=pl.INT32)',
      focus: 'parallel:grid-stride',
      inlay: [{ lens: 'parallel', text: '容量 {{FA_TABLE_CAP_FMT}} · 实际压入 {{REAL_BLOCKS}} 个真实块' }]
    },

    { sec: 'Scope · 依赖语义的边界', hint: 'manual_scope 内 tensormap 关闭，只有显式 deps 生效' },
    {
      no: 410, t: '    with pl.manual_scope():',
      focus: 'scope:manual',
      scope: { kind: 'manual', label: 'manual_scope · auto-dep 关闭', span: 33 },
      inlay: [{ lens: 'scope', text: 'tensormap 注册被抑制' }]
    },
    {
      no: 411, t: '        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rms_recip",',
      focus: 'scope:at',
      scope: { kind: 'at', label: 'pl.at · CORE_GROUP · 1 task', span: 5 }
    },
    {
      no: 414, t: '                   deps=[prev_out_tids[i] for i in range(DOWN_ON)]) as rms_tid:',
      focus: 'scope:manual',
      inlay: [{ lens: 'scope', text: '{{DOWN_ON}} 条显式边 · 逐 slab 挂' }]
    },
    {
      no: 417, t: '            for kb in pl.pipeline(HIDDEN // RMSNORM_K_CHUNK, stage=4):',
      focus: 'parallel:pipeline',
      inlay: [{ lens: 'parallel', text: '{{RMS_CHUNKS}} 次迭代 · stage=4 软流水' }]
    },
    {
      no: 419, t: '                x_chunk = hidden_states[:, k0 : k0 + RMSNORM_K_CHUNK]',
      focus: 'tile:x-chunk',
      inlay: [{ lens: 'tile', text: 'tile [{{BATCH}}, 256] FP32 → Vec(UB)' }]
    },
    { no: 422, t: '                partial_sq = pl.add(partial_sq, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH]))' },
    { fold: 11 },

    { sec: '并行请求 · pl.parallel + split-K atomic', hint: '并行度与依赖都是代码里的一等公民' },
    {
      no: 434, t: '        for q_nt in pl.parallel(Q_ON):',
      focus: 'parallel:parallel',
      scope: { kind: 'parallel', label: 'pl.parallel × {{Q_ON}}', span: 8 },
      inlay: [{ lens: 'parallel', text: '{{Q_ON}} 个 N-tile 并行请求' }]
    },
    {
      no: 436, t: '            for q_ks in pl.range(QKV_OK):',
      focus: 'layout:split-k',
      inlay: [{ lens: 'parallel', text: '× {{QKV_OK}} split-K → 共 {{Q_TASK_TOTAL}} 个 task' }]
    },
    {
      no: 438, t: '                with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj",',
      focus: 'scope:at',
      scope: { kind: 'at', label: 'pl.at · q_proj', span: 6 }
    },
    {
      no: 441, t: '                           deps=[prev_normed_tids[q_ks], q_seed_tid]) as q_tid:',
      focus: 'scope:manual',
      inlay: [{ lens: 'scope', text: 'q_ks 与 slab 1:1 —— 依赖这条约束成立' }]
    },
    {
      no: 445, t: '                    q_acc = pl.matmul(normed_in[:, q_k_base : q_k_base + TK],',
      focus: 'tile:q-acc',
      inlay: [{ lens: 'tile', text: 'A 逻辑块 [{{BATCH}}, 256] BF16' }]
    },
    {
      no: 447, t: '                                      wq[kk : kk + TK, n0 : n0 + TN], out_dtype=pl.FP32)',
      focus: 'tile:q-acc',
      inlay: [{ lens: 'tile', text: 'B 逻辑块 [256, 256] · C FP32；片上切分待核实' }]
    },
    {
      no: 450, t: '                    for kc in pl.pipeline(1, QKV_K_CHUNKS, stage=2):',
      focus: 'parallel:pipeline',
      inlay: [{ lens: 'parallel', text: '{{QKV_K_CHUNKS}} 个 K chunk · stage=2' }]
    },
    {
      no: 455, t: '                    q_proj = pl.assemble(q_proj, q_acc, [0, n0], atomic=pl.AtomicType.Add)',
      focus: 'layout:split-k'
    },
    { fold: 159 },

    { sec: '并行请求 · spmd 栅格与 grid-stride', hint: '静态栅格 + 设备侧游标，两种并行请求形态' },
    {
      no: 615, t: '        with pl.spmd(ROPE_CORES, name_hint="rope_qkv", deps=[qk_tids..., v_tile_tids...]) as rope_tid:',
      focus: 'parallel:spmd',
      scope: { kind: 'spmd', label: 'pl.spmd × {{ROPE_CORES}} blocks', span: 4 },
      inlay: [{ lens: 'parallel', text: '{{ROPE_CORES}} 核 · 仿射索引按 {{ROPE_ITEMS_PER_CORE}} items/core 计算' }]
    },
    { no: 627, t: '            rope_core = pl.get_block_idx()', focus: 'parallel:spmd' },
    { no: 628, t: '            for it in pl.pipeline(ROPE_ITEMS_PER_CORE, stage=2):', focus: 'parallel:pipeline',
      inlay: [{ lens: 'parallel', text: '源码循环使用 ROPE_GROUP_ITEMS_PER_CORE · stage=2' }] },
    {
      no: 655, t: '                cache_row = layer_cache_base + (wr_slot_block * NUM_KV_HEADS + ki) * BLOCK_SIZE + wr_slot_offset',
      focus: 'layout:paged-kv',
      inlay: [{ lens: 'layout', text: '物理页 × 头 × BLOCK_SIZE + 页内偏移' }]
    },
    { fold: 72 },
    {
      no: 728, t: '        with pl.spmd(NUM_CORES, name_hint="fa_fused",',
      focus: 'parallel:spmd',
      scope: { kind: 'spmd', label: 'pl.spmd × {{NUM_CORES}} · 混合 cube+vec', span: 10 }
    },
    {
      no: 731, t: '                     optimizations=[pl.split(pl.SplitMode.UP_DOWN)], deps=[work_tid, rope_tid]) as fa_tid:',
      focus: 'parallel:spmd',
      inlay: [{ lens: 'parallel', text: 'UP_DOWN 切分 · AIC/AIV 同根' }]
    },
    {
      no: 744, t: '            for fa_w in pl.range(fa_core, fa_total_blocks, NUM_CORES):',
      focus: 'parallel:grid-stride',
      inlay: [{ lens: 'parallel', text: 'grid-stride · core = w % {{NUM_CORES}}' }]
    },
    {
      no: 758, t: '                fa_pbid = pl.cast(pl.tensor.read(block_table, [fa_b * max_blocks_per_seq + sb]), pl.INDEX)',
      focus: 'layout:paged-kv',
      inlay: [{ lens: 'layout', text: '逻辑块 → 物理页 id' }]
    },
    {
      no: 764, t: '                for gp in pl.pipeline(GP_SIZE, stage=2):',
      focus: 'parallel:pipeline',
      scope: { kind: 'pipeline', label: 'pl.pipeline × {{GP_SIZE}} · stage=2', span: 6 },
      inlay: [{ lens: 'parallel', text: 'i+1 的 QK(AIC) 叠 i 的 softmax(AIV)' }]
    },
    {
      no: 768, t: '                    q_padded = all_q_padded[q_pad_row_g : q_pad_row_g + Q_HEAD_PAD, :]',
      focus: 'layout:q-pad',
      inlay: [{ lens: 'tile', text: '[{{Q_HEAD_PAD}}, {{HEAD_DIM}}] BF16 → Left' }]
    },
    {
      no: 773, t: '                    k_tile = k_cache[cache_row : cache_row + SEQ_TILE, :]',
      focus: 'tile:k-tile',
      inlay: [{ lens: 'tile', text: 'K 逻辑块 [{{SEQ_TILE}}, {{HEAD_DIM}}] BF16 · {{K_TILE_KB}} KB' }]
    },
    {
      no: 774, t: '                    raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)',
      focus: 'tile:k-tile',
      inlay: [{ lens: 'tile', text: 'C [{{Q_HEAD_PAD}}, {{SEQ_TILE}}] FP32 → Acc' }]
    },
    {
      no: 781, t: '                    scores_valid = pl.set_validshape(scores_scaled, Q_HEAD_BATCH, valid_len)',
      focus: 'layout:q-pad',
      inlay: [{ lens: 'layout', text: '只标 {{Q_HEAD_BATCH}} 行有效 · tile 仍是 {{Q_HEAD_PAD}} 行' }]
    },
    {
      no: 782, t: '                    scores = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)',
      focus: 'layout:q-pad',
      inlay: [{ lens: 'layout', text: '填充位置写 -inf，不污染 row_max' }]
    }
  ];

  // 合并真实源码中的计算主链路；原有标注与焦点保留，遗漏行数按行号计算。
  const SOURCE = (function () {
    const byLine = {};
    SOURCE_BASE.forEach(function (entry) { if (entry.no) byLine[entry.no] = entry; });
    (window.HN_EXTRA_SOURCE || []).forEach(function (entry) {
      byLine[entry.no] = Object.assign({}, byLine[entry.no] || {}, entry);
    });
    // 在完整源码合并后附加对象标注，保留源码原文和已有标注。
    const additions = {
      429: ['scope:deps', 'scope', 'q_seed 产出清零任务句柄'],
      441: ['scope:deps', 'scope', '等待对应 slab 与 q_seed'],
      455: ['scope:atomic', 'parallel', 'split-K 原子累加至 GM [0, n0]'],
      774: ['parallel:handoff', 'parallel', 'QK · Cube 计算，结果交给 Vector'],
      783: ['layout:q-pad', 'layout', '填充值不应影响有效区域的最大值'],
      786: ['precision:cast', 'precision', 'Softmax 结果转 BF16，供 SV matmul 使用'],
      789: ['tile:v-tile', 'tile', 'V 页与 K 页使用同一物理行范围'],
      790: ['parallel:handoff', 'parallel', 'SV · Cube 计算，FP32 输出'],
      939: ['precision:cast', 'precision', '残差归一化结果转 BF16']
    };
    Object.keys(additions).forEach(function (key) {
      const entry = byLine[key], a = additions[key];
      if (!entry) return;
      entry.focus = a[0];
      entry.inlay = (entry.inlay || []).concat([{ lens: a[1], text: a[2] }]);
    });
    const sections = {
      103: '模型与切分参数', 303: '张量契约', 410: 'RMSNorm 与 Q 投影',
      459: 'K / V 投影', 615: 'RoPE 与 KV 缓存写入', 728: '分块 Attention',
      810: 'Online Softmax 归约', 881: '输出投影与残差', 960: 'MLP · Gate / Up',
      998: 'SwiGLU 与 Down 投影', 1079: '残差输出与跨层传递'
    };
    const rows = [];
    let previous = null;
    Object.keys(byLine).map(Number).sort(function (a, b) { return a - b; }).forEach(function (no) {
      if (previous !== null && no > previous + 1) rows.push({ fold: no - previous - 1 });
      if (sections[no]) rows.push({ sec: sections[no] });
      rows.push(byLine[no]);
      previous = no;
    });
    return rows;
  })();

  // ════════════════════════════════════════════════════════════════════════════
  // 概念词典 —— 代码里的符号 → 可视化面板
  // ════════════════════════════════════════════════════════════════════════════
  const SYMBOLS = {
    'hidden_states': 'tensor:hidden-states',
    'normed_in': 'tensor:normed',
    'normed_out': 'tensor:normed',
    'q_proj': 'tensor:q-proj',
    'all_q_padded': 'layout:q-pad',
    'fa_work_table': 'parallel:grid-stride',
    'k_cache': 'layout:paged-kv',
    'block_table': 'layout:paged-kv',
    'cache_row': 'layout:paged-kv',
    'x_chunk': 'tile:x-chunk',
    'q_acc': 'tile:q-acc',
    'q_padded': 'layout:q-pad',
    'k_tile': 'tile:k-tile',
    'v_tile': 'tile:v-tile',
    'exp_scores_bf16': 'precision:cast',
    'q_seed_tid': 'scope:deps',
    'scores_valid': 'layout:q-pad',
    'Q_HEAD_PAD': 'layout:q-pad',
    'Q_HEAD_BATCH': 'layout:q-pad',
    'SEQ_TILE': 'layout:paged-kv',
    'BLOCK_SIZE': 'layout:paged-kv',
    'QKV_OK': 'layout:split-k',
    'QKV_K_SLICE': 'layout:split-k',
    'DOWN_TN': 'layout:split-k',
    'DOWN_ON': 'layout:split-k',
    'GP_SIZE': 'parallel:grid-stride',
    'NUM_CORES': 'parallel:spmd',
    'ROPE_CORES': 'parallel:spmd',
    'prev_out_tids': 'scope:manual',
    'prev_normed_tids': 'scope:manual',
    'deps': 'scope:deps',
    'pl.manual_scope': 'scope:manual',
    'pl.at': 'scope:at',
    'pl.Level.CORE_GROUP': 'scope:at',
    'pl.parallel': 'parallel:parallel',
    'pl.spmd': 'parallel:spmd',
    'pl.pipeline': 'parallel:pipeline',
    'pl.range': 'parallel:grid-stride',
    'pl.get_block_idx': 'parallel:spmd',
    'pl.set_validshape': 'layout:q-pad',
    'pl.fillpad': 'layout:q-pad',
    'pl.assemble': 'layout:split-k',
    'pl.AtomicType.Add': 'scope:atomic',
    'pl.matmul': 'tile:q-acc',
    'pl.create_tensor': 'tensor:q-proj',
    'pl.Tensor': 'tensor:hidden-states'
  };

  return {
    KNOBS: KNOBS,
    DEFAULTS: DEFAULTS,
    SOURCE: SOURCE,
    CONSTRAINTS: CONSTRAINTS,
    SYMBOLS: SYMBOLS,
    derive: derive,
    bytes: bytes,
    num: num
  };
})();
