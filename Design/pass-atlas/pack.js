/* MemoryReuse 的 largest-first 打包，逐帧
 * 算法与判据取自 src/ir/transforms/memory_reuse_pass.cpp（can_share 在 2173 行）
 * Acc 场景的两块累加器尺寸与结论在 pypto#1820 里有原文；Vec 场景的 tile 尺寸由 shape 推算，已标注。 */
window.PACK = {
  algoRef: 'src/ir/transforms/memory_reuse_pass.cpp:2173',
  /* 执行序泳道：按 qk_pv 的真实结构还原的指令序列。
     pipe 划分与 Ascend 的搬运/计算单元一致（MTE2 载入 / MTE1 Mat→L0 / M cube / FIX 定点流水 /
     V 向量 / MTE3 写回）；order 是执行序槽位，不是实测周期。
     reads / writes 指出每条指令碰了哪些 tile —— 选中 tile 时据此高亮。 */
  pipes: ['MTE2', 'MTE1', 'M · cube', 'FIX', 'V · vector', 'MTE3'],
  sched: [
    { p: 'MTE2', a: 0,  w: 3, n: 'TLOAD q → Mat',   line: 12, writes: [] },
    { p: 'MTE2', a: 3,  w: 4, n: 'TLOAD k → Mat',   line: 12, writes: [] },
    { p: 'MTE1', a: 7,  w: 2, n: 'TMOV → L0A/L0B',  line: 12, writes: [] },
    { p: 'M · cube', a: 9,  w: 6, n: 'TMATMUL (QK)', line: 12, writes: ['mem_acc_qk'] },
    { p: 'FIX', a: 15, w: 3, n: 'FixPipe Acc→Vec',  line: 14, reads: ['mem_acc_qk'], writes: ['mem_vec_qk'] },
    { p: 'V · vector', a: 18, w: 4, n: 'TROWMAX',    line: 20, reads: ['mem_vec_qk', 'mem_vec_tmp'], writes: ['mem_vec_m'] },
    { p: 'V · vector', a: 22, w: 2, n: 'TSUB',       line: 25, reads: ['mem_vec_qk', 'mem_vec_m'], writes: [] },
    { p: 'V · vector', a: 24, w: 3, n: 'TEXP',       line: 25, writes: ['mem_vec_p'] },
    { p: 'MTE2', a: 27, w: 4, n: 'TLOAD v → Mat',    line: 26, writes: [] },
    { p: 'MTE1', a: 28, w: 2, n: 'TMOV p → L0A',     line: 26, reads: ['mem_vec_p'] },
    { p: 'M · cube', a: 31, w: 9, n: 'TMATMUL (PV)', line: 26, writes: ['mem_acc_pv'] },
    { p: 'MTE3', a: 41, w: 3, n: 'TSTORE mi',        line: 38, reads: ['mem_vec_m', 'mem_acc_pv'] }
  ],
  schedSpan: 46,

  /* 源码清单：行号与各 tile 的 from/to 对齐，供三栏联动定位。
     语句是按 qk_pv 的真实结构写的，行号为演示用锚点。 */
  source: [
    { n: 10, t: '@pl.function' },
    { n: 11, t: 'def qk_pv(self, q, k, v, mi):' },
    { n: 12, t: '    qk = pl.matmul(q, k, out_dtype=pl.FP32)', def: 'mem_acc_qk', note: 'QK 累加器落 Acc' },
    { n: 13, t: '' },
    { n: 14, t: '    qk_v = pl.tile.move(qk, mem=pl.Mem.Vec)', def: 'mem_vec_qk', note: 'Acc → Vec，交给向量核做 softmax' },
    { n: 15, t: '' },
    { n: 16, t: '    # ---- online softmax ----' },
    { n: 18, t: '    tmp = pl.tile.create([64, 128], pl.FP32)', def: 'mem_vec_tmp', note: 'row_max 要求的同 dtype scratch' },
    { n: 20, t: '    m = pl.tile.row_max(qk_v, tmp)', def: 'mem_vec_m', use: 'mem_vec_tmp',
      note: 'tile.row_max 声明 not_inplace_safe：输出不得与 qk_v 或 tmp 共享 buffer' },
    { n: 21, t: '    # qk 在 Acc 上的最后一次读', use: 'mem_acc_qk' },
    { n: 24, t: '    # qk_v 在 Vec 上的最后一次读', use: 'mem_vec_qk' },
    { n: 25, t: '    p = pl.exp(pl.sub(qk_v, m))', def: 'mem_vec_p', note: 'qk_v 已死，p 可以接管它的地址' },
    { n: 26, t: '    out = pl.matmul(p, v, out_dtype=pl.FP32)', def: 'mem_acc_pv', note: 'PV 累加器；QK 的 Acc 槽此时已空' },
    { n: 30, t: '    # m 的最后一次读', use: 'mem_vec_m' },
    { n: 33, t: '    # p 的最后一次读', use: 'mem_vec_p' },
    { n: 38, t: '    mi[0:16] = m[0:16]', use: 'mem_acc_pv', note: '跨切分轴的条带写回' }
  ],

  gates: [
    { id: 'overlap', name: '生命周期重叠', code: 'LifetimesOverlap && overlap_blocks_sharing',
      why: '两块 tile 同时存活就不能共用一块地址。先做粗粒度区间判断，命中了再回落到逐变量精查——这样互斥分支、同值 phi 家族仍有机会共享。' },
    { id: 'hazard', name: '目标冒险', code: 'hazard_blocks(a,b) || hazard_blocks(b,a)',
      why: '后端相关的硬件冒险，双向都要查（打包不再按程序序，所以每一道门都得对称判断）。' },
    { id: 'forbid', name: '别名禁令', code: 'forbid_blocks(a,b) || forbid_blocks(b,a)',
      why: 'ForbidAliasCollector 收集的三类禁令：not_inplace_safe（输出不得与任何输入共享）、forbid_output_alias(i)（不得与指定操作数共享）、加宽 cast（写游标跑赢读游标）。' },
    { id: 'pipeline', name: '流水冲突', code: 'pipeline_blocks(cand, member)',
      why: '同一流水组里相隔 F 级的 slot 不能落在一块 buffer 上，否则 ping-pong 退化成串行。' },
    { id: 'layout', name: 'Vec ND/NZ 兼容', code: '!AreVecNdNzCompatible(...)',
      why: '两块 tile 的 Vec 布局形态不兼容时不能共用。' }
  ],
  cases: [
    {
      id: 'acc', space: 'Acc', title: 'Acc · 两个累加器接力',
      cite: 'pypto#1820 原文：“Fixed by #1805 (global largest-first packing now reuses the dead QK slot)”',
      caps: { a5: 262144, a2a3: 131072 },
      tiles: [
        { n: 'mem_acc_qk', label: 'QK 累加器 [64,128] fp32', bytes: 32768, from: 12, to: 21, cited: true, op: 'tile.matmul', space: 'Acc' },
        { n: 'mem_acc_pv', label: 'PV 累加器 [64,512] fp32', bytes: 131072, from: 26, to: 38, cited: true, op: 'tile.matmul', space: 'Acc' }
      ],
      frames: [
        { t: 'InitMemRef 之后：两块各自独立', act: 'input',
          d: '第 32 步把每个 tile 绑上自己的 MemRef，地址还没分配。此时 Acc 空间的需求是两块之和。',
          place: [{ n: 'mem_acc_qk', buf: 0 }, { n: 'mem_acc_pv', buf: 1 }],
          hw: '160 KB —— A5 的 256 KB 装得下；**A2/A3 的 128 KB 装不下，这就是 #1820 报的第一个 blocker**。' },
        { t: '按内存空间分组，组内尺寸从大到小排序', act: 'sort',
          d: 'by_space 先把区间按 MemorySpace 分桶——复用只在同一空间内发生。桶内按 largest-first 排序，同尺寸按定义序打破平手以保证确定性。',
          order: ['mem_acc_pv', 'mem_acc_qk'],
          hw: 'PV 128 KB 排在前面，QK 32 KB 在后。**顺序决定结果**：#1805 之前是按定义序的贪心，带一个单向尺寸门（source.size >= target.size），所以「小的先定义」的两块永远合不起来。' },
        { t: '放置 PV：最大的先落地，成为代表', act: 'place', focus: 'mem_acc_pv',
          d: '第一块直接开一个新 buffer。buffer 的大小由它第一个（也是最大的）成员决定，之后接纳更小的成员零成本。',
          place: [{ n: 'mem_acc_pv', buf: 0 }],
          hw: 'buffer#0 = 128 KB' },
        { t: '放置 QK：过 can_share 五道门', act: 'gate', focus: 'mem_acc_qk', against: 'mem_acc_pv',
          d: '候选要加入某个 buffer，必须与该 buffer 的**所有**成员都能共享。逐道判定：',
          gates: [
            { id: 'overlap',  pass: true, note: 'QK 在第 21 行最后一次被读，PV 在第 26 行才定义——prev.last_use <= curr.def，不重叠' },
            { id: 'hazard',   pass: true, note: '双向都无后端冒险' },
            { id: 'forbid',   pass: true, note: 'matmul 的输出没有 not_inplace_safe，也没有 forbid_output_alias' },
            { id: 'pipeline', pass: true, note: '不在同一流水组' },
            { id: 'layout',   pass: true, note: 'Acc 空间不涉及 Vec 的 ND/NZ 判定' }
          ],
          place: [{ n: 'mem_acc_pv', buf: 0 }, { n: 'mem_acc_qk', buf: 0, reused: true }],
          hw: '五道全过 → QK 并入 buffer#0，**rebase 到该 buffer 最大成员（PV）的地址上**' },
        { t: '结果：Acc 峰值 160 KB → 128 KB', act: 'done',
          d: '两块合成一块，删掉不再被引用的 alloc。A2/A3 上的 L0C 溢出被解掉了。',
          place: [{ n: 'mem_acc_pv', buf: 0 }, { n: 'mem_acc_qk', buf: 0, reused: true }],
          hw: '**但代价没有被记账**：QK 与 PV 现在住同一块地址，中间必须有同步保证 QK 读完再让 PV 写。这条边由下游 PTOAS InsertSync 补，而 #1821 记录的正是同一次 #1805 改动在 dsv4 上丢掉了一对 PIPE_MTE3↔PIPE_V 同步，造成 17.5% 输出错。',
          risk: true }
      ]
    },
    {
      id: 'vec', space: 'Vec', title: 'Vec · 一次被禁令拦下的复用',
      cite: '判据来自 src/ir/op/tile_ops/reduction.cpp:200-214 —— tile.row_max 注册时声明了 .not_inplace_safe()',
      caps: { a5: 245760, a2a3: 188416 },
      derived: true,
      tiles: [
        { n: 'mem_vec_qk',  label: 'qk 副本 [64,128] fp32', bytes: 32768, from: 14, to: 24, op: 'tile.move', space: 'Vec' },
        { n: 'mem_vec_tmp', label: 'row_max 的 tmp scratch [64,128]', bytes: 32768, from: 18, to: 20, op: 'tile.create', space: 'Vec', forbidden: 'row_max 的第二个输入，同样受 not_inplace_safe 保护' },
        { n: 'mem_vec_p',   label: 'p = exp(qk - m) [64,128] fp32', bytes: 32768, from: 25, to: 33, op: 'tile.exp', space: 'Vec' },
        { n: 'mem_vec_m',   label: 'm = row_max(qk) [64,1] fp32', bytes: 256, from: 20, to: 30, op: 'tile.row_max', space: 'Vec', forbidden: 'not_inplace_safe：输出不得与任何输入共享（reduction.cpp:210）' }
      ],
      frames: [
        { t: '输入：Vec 空间 4 块中间量', act: 'input',
          d: 'online-softmax 的中间量。注意 tile.row_max 有两个输入——被规约的 tile 和一块同 dtype 的 tmp scratch。',
          place: [{ n: 'mem_vec_qk', buf: 0 }, { n: 'mem_vec_tmp', buf: 1 }, { n: 'mem_vec_p', buf: 2 }, { n: 'mem_vec_m', buf: 3 }],
          hw: '未复用：96.25 KB（32768×3 + 256）' },
        { t: 'largest-first：三块 32 KB 并列，按定义序打破平手', act: 'sort',
          order: ['mem_vec_qk', 'mem_vec_tmp', 'mem_vec_p', 'mem_vec_m'],
          d: '同尺寸时按定义序排，保证同样的输入每次得到同样的布局。',
          hw: 'qk → tmp → p → m' },
        { t: '放置 p：与 qk 生命周期不重叠，可以合并', act: 'gate', focus: 'mem_vec_p', against: 'mem_vec_qk',
          d: 'p 在第 25 行定义，qk 在第 24 行最后一次被读。',
          gates: [
            { id: 'overlap', pass: true, note: 'qk.last_use(24) <= p.def(25)，不重叠' },
            { id: 'hazard', pass: true, note: '—' },
            { id: 'forbid', pass: true, note: 'tile.exp / tile.sub 是 in-place safe 的' },
            { id: 'pipeline', pass: true, note: '—' },
            { id: 'layout', pass: true, note: '同为 Vec ND，兼容' }
          ],
          place: [{ n: 'mem_vec_qk', buf: 0 }, { n: 'mem_vec_p', buf: 0, reused: true },
                  { n: 'mem_vec_tmp', buf: 1 }, { n: 'mem_vec_m', buf: 3 }],
          hw: '合并 → 省下 32 KB' },
        { t: '放置 m：第三道门命中，拒绝', act: 'gate', focus: 'mem_vec_m', against: 'mem_vec_qk', reject: true,
          d: 'm 是 row_max 的输出，qk 是它的输入。生命周期上 m(20–30) 与 qk(14–24) 确实重叠，但**即使不重叠，第三道门也会拦住它**：',
          gates: [
            { id: 'overlap', pass: false, note: 'm 在第 20 行定义时 qk 仍存活到第 24 行——重叠' },
            { id: 'forbid', pass: false, note: 'tile.row_max 注册时声明 .not_inplace_safe()：TROW* 在写规约输出的同时要读完整输入行**和** tmp scratch，输出不得与任何一个输入共享 buffer',
              evidence: 'src/ir/op/tile_ops/reduction.cpp:208-210' }
          ],
          place: [{ n: 'mem_vec_qk', buf: 0 }, { n: 'mem_vec_p', buf: 0, reused: true },
                  { n: 'mem_vec_tmp', buf: 1 }, { n: 'mem_vec_m', buf: 3 }],
          hw: '拒绝 → m 保持独立 buffer。**这正是「我为什么没帮你省这块」——今天这条判定只走 LOG_DEBUG，用户看不到。**' },
        { t: '结果：Vec 峰值 96.25 KB → 64.25 KB', act: 'done',
          d: 'qk 与 p 合并，tmp 与 m 各自独立（tmp 同样受 row_max 的禁令保护）。',
          place: [{ n: 'mem_vec_qk', buf: 0 }, { n: 'mem_vec_p', buf: 0, reused: true },
                  { n: 'mem_vec_tmp', buf: 1 }, { n: 'mem_vec_m', buf: 3 }],
          hw: '省下一块 32 KB。**被禁令挡住的 tmp 与 m 合计仍占 32.25 KB** —— 这不是浪费，是 row_max 的正确性要求；但「为了正确性我多花了 32.25 KB」这句话，今天没有任何界面会告诉你。' }
      ]
    }
  ]
};
