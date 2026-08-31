/* Ascend SoC 拓扑与内存层级
 * 全部取自 repo/hw-native-sys/pypto/src/backend/common/soc.cpp
 * 与 include/pypto/backend/{950,910B}/backend_*_handler.h —— 未做任何改写。 */
window.HW = {
  src: 'src/backend/common/soc.cpp · Create950SoC() / Create910BSoC()',

  /* 数据通路拓扑：节点是内存或计算单元，边取自各 SoC 的 mem_graph（未改写）。
     计算单元的位置按 Ascend 的实际数据流插入：L1→L0A/L0B→Cube→L0C→FixPipe。 */
  topo: {
    units: [
      { id: 'DDR',  kind: 'mem',  label: 'GM / DDR', sub: '全局内存', col: 0, row: 1 },
      { id: 'Vec',  kind: 'mem',  label: 'UB',       sub: 'Vec',      col: 1, row: 2, core: 'AIV' },
      { id: 'VALU', kind: 'unit', label: 'Vector ALU', sub: 'SIMD / SIMT', col: 2, row: 2, core: 'AIV' },
      { id: 'Mat',  kind: 'mem',  label: 'L1',       sub: 'Mat',      col: 1, row: 0, core: 'AIC' },
      { id: 'Left', kind: 'mem',  label: 'L0A',      sub: 'Left',     col: 2, row: -0.55, core: 'AIC' },
      { id: 'Right',kind: 'mem',  label: 'L0B',      sub: 'Right',    col: 2, row: 0.15, core: 'AIC' },
      { id: 'Bias', kind: 'mem',  label: 'BT',       sub: 'Bias',     col: 2, row: 0.85, core: 'AIC' },
      { id: 'LeftScale',  kind: 'mem', label: 'L0A scale', sub: 'LeftScale',  col: 2, row: -1.25, core: 'AIC', a5only: true },
      { id: 'RightScale', kind: 'mem', label: 'L0B scale', sub: 'RightScale', col: 3, row: -1.25, core: 'AIC', a5only: true },
      { id: 'CUBE', kind: 'unit', label: 'Cube MAC', sub: '矩阵乘累加', col: 3, row: 0, core: 'AIC' },
      { id: 'Acc',  kind: 'mem',  label: 'L0C',      sub: 'Acc',      col: 4, row: 0, core: 'AIC' },
      { id: 'FIX',  kind: 'unit', label: 'FixPipe',  sub: '定点流水 · 收窄/量化', col: 5, row: 0.6, core: 'AIC' }
    ],
    /* 计算单元相关的边（内存之间的边直接读 mem_graph） */
    unitEdges: [
      { f: 'Left',  t: 'CUBE' }, { f: 'Right', t: 'CUBE' }, { f: 'Bias', t: 'CUBE' },
      { f: 'LeftScale', t: 'CUBE', a5only: true }, { f: 'RightScale', t: 'CUBE', a5only: true },
      { f: 'CUBE',  t: 'Acc' },
      { f: 'Acc',   t: 'FIX' },
      { f: 'Vec',   t: 'VALU' }, { f: 'VALU', t: 'Vec' }
    ],
    /* FixPipe 的去向就是 mem_graph 里 Acc 的出边 */
    fixOut: 'Acc'
  },
  socs: {
    a5: {
      id: 'a5', name: 'Ascend 950', short: 'A5', factory: 'Create950SoC()', line: 161,
      dies: 2, clustersPerDie: 18, clusterShape: '1 AIC + 2 AIV',
      cores: [
        { type: 'CUBE', label: 'AIC · Cube', per: 1, mems: [
          { s: 'Mat',        bytes: 524288, align: 128, role: 'L1 · 矩阵暂存', note: 'tile.load 的落点；Left/Right/Bias/Scale 都从这里取' },
          { s: 'Acc',        bytes: 262144, align: 128, role: 'L0C · 累加器',   note: 'matmul 输出；A5 比 A2/A3 翻倍（256KB vs 128KB）' },
          { s: 'Left',       bytes: 65536,  align: 64,  role: 'L0A · 左操作数', note: 'AutoTileMatmulL0 的 m×k 容量门' },
          { s: 'Right',      bytes: 65536,  align: 64,  role: 'L0B · 右操作数', note: 'AutoTileMatmulL0 的 k×n 容量门' },
          { s: 'Bias',       bytes: 4096,   align: 64,  role: '偏置表',        note: 'A5 是 A2/A3 的 4 倍（4KB vs 1KB）' },
          { s: 'LeftScale',  bytes: 4096,   align: 32,  role: 'L0A scale',     note: 'MX 量化；A2/A3 没有这块' },
          { s: 'RightScale', bytes: 4096,   align: 32,  role: 'L0B scale',     note: 'MX 量化；A2/A3 没有这块' }
        ]},
        { type: 'VECTOR', label: 'AIV · Vector', per: 2, mems: [
          { s: 'Vec', bytes: 245760, align: 128, phys: 253952, role: 'UB · 向量缓冲',
            note: '物理 248KB，安全上限 240KB —— 顶部约 8KB 被 PTO-ISA 占用，放 tile 会静默损坏（pto-isa#170）', danger: true }
        ]}
      ],
      memGraph: {
        DDR: ['Vec', 'Mat'],
        Vec: ['Mat', 'DDR'],
        Mat: ['Left', 'Right', 'Bias', 'LeftScale', 'RightScale'],
        Acc: ['Vec', 'Mat', 'DDR']
      },
      handler: { gmGranularity: 128, l2Line: 512, recommendedInnermost: 128,
                 l0a: 65536, l0b: 65536, l0c: 262144, fractal: 16,
                 accToGm: ['INT32','FP32','FP16'], bf16AtomicAdd: false },
      highlights: [
        'Vec↔Mat 直连：A5 的 mem_graph 里 Vec 可以直接到 Mat，A2/A3 必须绕 DDR',
        'Acc 翻倍到 256KB，两个 FP32 累加器（QK 32KB + PV 128KB）才装得下',
        '新增 LeftScale / RightScale 两块 4KB —— MX 量化专用，InsertMxScaleAddr 就是为它们服务的',
        'Acc→GM 白名单比 A2/A3 窄：没有 BF16。同一份 DSL 在 A2/A3 能过、在 A5 会被 ptoas 拒'
      ]
    },
    a2a3: {
      id: 'a2a3', name: 'Ascend 910B', short: 'A2/A3', factory: 'Create910BSoC()', line: 119,
      dies: 1, clustersPerDie: 72, clusterShape: '24 AIC + 48 AIV（各自独立 cluster）',
      cores: [
        { type: 'CUBE', label: 'AIC · Cube', per: 24, mems: [
          { s: 'Mat',   bytes: 524288, align: 128, role: 'L1 · 矩阵暂存', note: '与 A5 相同' },
          { s: 'Acc',   bytes: 131072, align: 128, role: 'L0C · 累加器',  note: '只有 A5 的一半' },
          { s: 'Left',  bytes: 65536,  align: 64,  role: 'L0A', note: '与 A5 相同' },
          { s: 'Right', bytes: 65536,  align: 64,  role: 'L0B', note: '与 A5 相同' },
          { s: 'Bias',  bytes: 1024,   align: 64,  role: '偏置表', note: '只有 A5 的四分之一' }
        ]},
        { type: 'VECTOR', label: 'AIV · Vector', per: 48, mems: [
          { s: 'Vec', bytes: 188416, align: 128, phys: 196608, role: 'UB · 向量缓冲',
            note: '物理 192KB，安全上限 184KB（同 pto-isa#170）', danger: true }
        ]}
      ],
      memGraph: { DDR: ['Vec','Mat'], Vec: ['DDR'], Mat: ['Left','Right','Bias'], Acc: ['Mat','DDR'] },
      handler: { gmGranularity: 512, l2Line: 512, recommendedInnermost: 512,
                 l0a: 65536, l0b: 65536, l0c: 131072, fractal: 16,
                 accToGm: ['INT32','FP32','FP16','BF16'], bf16AtomicAdd: true },
      highlights: [
        'Vec 只能到 DDR —— 向量结果要进 Mat 必须绕一圈全局内存',
        'Acc 128KB 是 FlashAttention 的硬约束：QK + PV 两个累加器共 160KB 装不下（pypto#1820 记录的第一个 blocker）',
        'GM 访问粒度 512B，是 A5 的 4 倍 —— PH001 的推荐最内维阈值因此也是 512B'
      ]
    }
  },
  /* 哪些 Pass 决定了哪块内存的什么 */
  owners: [
    { space: 'ALL',   order: 18, pass: 'InferTileMemorySpace', what: '决定每个 tile 住哪块内存',
      how: '按算子约束推断；推不出来默认 Vec。生产者与消费者不匹配处插 tile.move。' },
    { space: 'Left',  order: 16, pass: 'AutoTileMatmulL0', what: '决定 L0A 里放多大的 m×k',
      how: 'ChooseL0Tile 用 l0a_bytes=64KB 做容量门，配合 roofline 代价模型选 (m,n,k)。' },
    { space: 'Right', order: 16, pass: 'AutoTileMatmulL0', what: '决定 L0B 里放多大的 k×n',
      how: '同上，l0b_bytes=64KB。ping-pong 时要乘以流水深度，这是最常见的容量瓶颈。' },
    { space: 'Acc',   order: 16, pass: 'AutoTileMatmulL0', what: '决定 L0C 里累加器的 m×n',
      how: 'l0c_bytes：A5 256KB / A2A3 128KB。FP32 累加器 4B/元素，很快吃满。' },
    { space: 'ALL',   order: 32, pass: 'InitMemRef', what: '建立 MemRef，默认空间 UB / load-store 操作数 DDR',
      how: '同时把所有 alloc 提到函数体头部 —— 这个事实后来被 MemoryReuse 的 largest-first 打包直接利用。' },
    { space: 'ALL',   order: 33, pass: 'MaterializeSemanticAliases', what: '强制语义要求的同一块分配',
      how: 'loop-carry 累加器、in-place 算子必须活在同一块 buffer 里。' },
    { space: 'ALL',   order: 34, pass: 'MemoryReuse', what: '决定哪些 tile 共用一块 buffer',
      how: '按内存空间分组，组内 largest-first 打包；can_share 五道门任一命中即拒。' },
    { space: 'ALL',   order: 35, pass: 'AllocateMemoryAddr', what: '在每块内存里分配真实地址',
      how: '按该空间的 align 对齐；超出容量上限直接报错——240KB 这条线就是在这里拦下 pto-isa#170 的。' },
    { space: 'LeftScale', order: 19, pass: 'InsertMxScaleAddr', what: '为 MX matmul 绑 scale 地址',
      how: '要求 Left/LeftScale 与 Right/RightScale 的空间都已确定，所以排在 InferTileMemorySpace 之后。' }
  ]
};
