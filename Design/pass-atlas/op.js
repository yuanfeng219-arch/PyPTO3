/* 算子解剖：一个 FlashAttention 内层步 qk_pv 走完编译流水线
 * 场景与数字取自 pypto#1820（RFC，open）与 soc.cpp 的真实容量。
 * 交叉验证：#1820 原文 "197632 B > 188416 B"，而 soc.cpp 里 A2/A3 的 Vec 安全上限
 * 正是 184ULL*1024 = 188416 —— Issue 与源码严丝合缝。 */
window.OP = {
  name: 'qk_pv',
  title: 'FlashAttention 内层步：QK^T → online-softmax → PV',
  src: 'pypto#1820（RFC · open）· 蒸馏自 fa_fused decode',
  shape: 'M_TILE=64 · HEAD_DIM=512 · K_TILE=128',
  dsl: [
    '@pl.function',
    'def qk_pv(self, q:  pl.Tensor[[64, 128],  pl.BF16],',
    '                k:  pl.Tensor[[128, 128], pl.BF16],',
    '                v:  pl.Tensor[[128, 512], pl.BF16],',
    '                mi: pl.Tensor[[64, 1],    pl.FP32]) -> None:',
    '    qk  = pl.matmul(q, k, out_dtype=pl.FP32)      # cube',
    '    m   = pl.tile.row_max(qk)                     # vector',
    '    p   = pl.exp(pl.sub(qk, m))                   # vector',
    '    out = pl.matmul(p, v, out_dtype=pl.FP32)      # cube',
    '    mi[0:16] = m[0:16]                            # 跨切分轴的条带写回'
  ],
  /* 每一步：哪个 Pass、它对这个算子做了什么、落到哪块硬件内存 */
  steps: [
    { order: 16, pass: 'AutoTileMatmulL0', title: '选 L0 tile，改写成 K 循环',
      does: '为 QK 与 PV 两个 matmul 各从 backend 的 L0 容量里选一组 (m, n, k)，把调用改写成两级流水的 K 循环，每轮插 Mat→Left/Right 的 extract。',
      picks: [
        { label: 'QK', mnk: 'm=64 · n=128 · k=128', trips: 1 },
        { label: 'PV', mnk: 'm=64 · n=512 · k=128', trips: 1 }
      ],
      mem: [
        { s: 'Left',  bytes: 16384,  slots: 2, why: 'q tile [64,128] bf16 = 16 KB，stage=2 要两份' },
        { s: 'Right', bytes: 32768,  slots: 2, why: 'k tile [128,128] bf16 = 32 KB，两份正好吃满 L0B 64 KB' },
        { s: 'Acc',   bytes: 32768,  slots: 1, why: 'QK 输出 [64,128] fp32 = 32 KB' }
      ],
      note: 'K 循环被标成 ForKind::Pipeline + pipeline_stages=2，交给第 29 步 LowerPipelineLoops 复制成 ping-pong。' },

    { order: 18, pass: 'InferTileMemorySpace', title: '给每个 tile 定住址空间',
      does: 'q / k / v 走 Mat→Left/Right；qk 输出落 Acc；row_max、exp 的中间量落 Vec；PV 的左操作数是 softmax 结果，以 Vec 形态跨越 cube↔vector 边界。',
      mem: [
        { s: 'Mat', bytes: 98304, slots: 1, why: 'q/k/v 的 L1 暂存' },
        { s: 'Vec', bytes: 197632, slots: 1, why: 'softmax 中间量 [64,128]：qk 副本 + m + p + 临时' }
      ],
      note: '**这一步产生的 Vec 需求 197632 B，就是整条链的分水岭**：A2/A3 的安全上限 188416 B 装不下，A5 的 245760 B 装得下。' },

    { order: 22, pass: 'ExpandMixedKernel', title: '拆成 AIC 与 AIV 两个 kernel',
      does: '把这个混合 InCore 函数拆成 qk_pv_aic（两个 matmul）+ qk_pv_aiv（row_max / exp / 条带写回），外面包一层 Group 函数；cube↔vector 的数据经 tpush / tpop 交接。',
      mem: [], note: '拆完之后，Acc / Left / Right / Mat 归 AIC 核，Vec 归 AIV 核 —— 内存预算从此分两本账算。' },

    { order: 24, pass: 'SplitVectorKernel', title: '把向量段切到两个 AIV 子核',
      does: 'Vec 装不下时的逃生门：UP_DOWN 沿行轴对半切，[64,128] 的 softmax 中间量变成每核 [32,128]，Vec 需求减半到 98816 B。',
      mem: [{ s: 'Vec', bytes: 98816, slots: 1, why: '切半之后每个 AIV 子核的占用' }],
      note: '**但这是 #1820 报的第三个 blocker**：这个 Pass 是逐语句的语法半分器，没有 index space 模型 —— 它把 `mi[0:16] = m[0:16]` 这条切分轴上的条带写回的**结果类型**减半成 [8,1]，却没改它的**形状实参**，于是造出 `tstore(subview<16>, partition<8>)`，ptoas 直接拒绝。这个失败模式对 online-softmax 是结构性的。' },

    { order: 29, pass: 'LowerPipelineLoops', title: '复制流水体，实现 ping-pong',
      does: '把 K 循环体复制 stage 份，让 load 与 compute 重叠。复制之后 Left / Right 各需要 stage 份 slot —— 上面第 16 步选 k=128 而不是 k=256，正是被这一步的乘 2 逼出来的。',
      mem: [
        { s: 'Left',  bytes: 32768, slots: 1, why: '16 KB × 2 slot' },
        { s: 'Right', bytes: 65536, slots: 1, why: '32 KB × 2 slot = 64 KB，L0B 刚好打满' }
      ],
      note: '如果容量门满足不了请求的深度，这里会降级，并由第 34 步 MemoryReuse 发出 PH-MR-001 说明降到了几级、代价是什么、怎么改。' },

    { order: 34, pass: 'MemoryReuse', title: '把生命周期不重叠的 buffer 合并',
      does: '按内存空间分组做 largest-first 打包。QK 的 Acc 累加器在 PV 开始前就死了，于是 PV 的累加器可以复用它的地址。',
      mem: [{ s: 'Acc', bytes: 131072, slots: 1, why: 'PV 输出 [64,512] fp32 = 128 KB，复用 QK 让掉的 32 KB 区间' }],
      note: '**这一步同时是机会与风险**：#1805 的全局 largest-first 打包正是为了让 PV 复用 QK 腾出的 L0C 槽（#1820 记录它 fixed 了第一个 blocker）；而同一次改动在 dsv4 上丢掉了一对 PIPE_MTE3↔PIPE_V 同步，造成 17.5% 输出错（#1821）。' },

    { order: 35, pass: 'AllocateMemoryAddr', title: '落到真实地址，撞线就报错',
      does: '在每块内存里按该空间的 align 分配真实地址（Mat/Acc/Vec 对齐 128，Left/Right 对齐 64，Scale 对齐 32），超出容量上限直接抛错。',
      mem: [], note: '**240 KB 这条线就是在这里拦人的**。物理 Vec 是 248 KB，但 soc.cpp 刻意只声明 240 KB，注释写明：顶部约 8 KB 被 PTO-ISA 占用，放 tile 会静默损坏（pto-isa#170）——宁可在编译期报错，也不要在设备上出 NaN。' }
  ],
  /* 同一个算子在两代硬件上的结局 */
  verdict: {
    a5:   { ok: true,  headline: 'Acc 256 KB 与 Vec 240 KB 都装得下，不需要切分逃生门',
            rows: [
              { k: 'L0C 双累加器', v: 'QK 32 KB + PV 128 KB = 160 KB', cap: '256 KB', ok: true },
              { k: 'Vec softmax 中间量', v: '197632 B', cap: '245760 B', ok: true },
              { k: '是否被迫 pl.split(UP_DOWN)', v: '否', cap: '—', ok: true }
            ] },
    a2a3: { ok: false, headline: '两处都撞线，被迫走切分逃生门，然后撞上 SplitVectorKernel 的结构性缺陷',
            rows: [
              { k: 'L0C 双累加器', v: 'QK 32 KB + PV 128 KB = 160 KB', cap: '128 KB', ok: false },
              { k: 'Vec softmax 中间量', v: '197632 B', cap: '188416 B', ok: false },
              { k: '是否被迫 pl.split(UP_DOWN)', v: '是 → ptoas codegen 失败', cap: '—', ok: false }
            ] }
  }
};
