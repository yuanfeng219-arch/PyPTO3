/* =============================================================
 * Pass Decision Studio · Pass 编译决策可视化 — 场景数据
 *
 * 场景取自真实 Issue，非虚构：
 *   pypto#1475  MemoryReuse 过度复用 → fa_qks 两个 matmul 串行（open）
 *   pypto#1744  AutoDeriveTaskDependencies 把 manual_scope 降级为 AUTO（open）
 *   PH-MR-001   在树的软件流水深度降级 perf hint
 *   PH-AT-007   AutoTileMatmulL0 因 K 非 16 对齐而放弃切分
 *
 * 决策字段模型见 Product_Planning/PyPTO3.0_Pass编译决策可视化设计.md §6
 * ============================================================= */
window.PASS_DECISION_DATA = (function () {
  'use strict';

  /* ---- 四层：颜色与顺序 ---- */
  var LAYERS = [
    { id: 'tiling', name: '自动切分与调度', short: '切分', tone: 'violet' },
    { id: 'memory', name: '自动内存复用', short: '复用', tone: 'amber' },
    { id: 'deps',   name: '自动依赖分析', short: '依赖', tone: 'cyan' },
    { id: 'sync',   name: '自动插入同步', short: '同步', tone: 'rose' }
  ];

  /* ---- 编译阶段分组（严格按执行序） ---- */
  var PHASES = [
    { id: 'front',  name: '前端与规范化',        range: '01–06' },
    { id: 'scope',  name: '作用域外提',          range: '07–09' },
    { id: 'tile',   name: 'Tensor→Tile 与合法化', range: '10–19' },
    { id: 'split',  name: '核拆分与流水调度',     range: '20–29' },
    { id: 'mem',    name: '内存布局与复用',       range: '30–36' },
    { id: 'dep',    name: '任务方向与依赖',       range: '37–39' },
    { id: 'dist',   name: '分布式与通信域',       range: '40–43' },
    { id: 'tail',   name: '运行时 scope 与收尾',  range: '44–47' },
    { id: 'ptoas',  name: 'PTOAS（跨仓 · level2）', range: '—' }
  ];

  /* ---- 全部 47 个 Pass + PTOAS 侧
   * layer: 属于幻灯片四层中的哪一层（仅决策 Pass 有）
   * decisions: 本次编译在该 Pass 上产生的决策条数（0 = 机械改写）
   * why: 这个 Pass 做什么
   * note: 为什么它是 / 不是决策点
   * ---- */
  var PASSES = [
    { idx: 1,  phase: 'front', name: 'InlineFunctions', decisions: 0,
      why: '把 FunctionType.Inline 的函数体原地展开到每一个调用点。',
      note: '输入定则输出唯一，无可选方案。' },
    { idx: 2,  phase: 'front', name: 'UnrollLoops', layer: 'tiling', decisions: 2,
      why: '在编译期展开 ForKind::Unroll 循环，为每个迭代值内联一份循环体。',
      note: '展开与否、展开到什么程度会改变后续所有 Pass 的输入规模，是切分层的第一个决策点。' },
    { idx: 3,  phase: 'front', name: 'CtrlFlowTransform', decisions: 0,
      why: '把 break / continue 改写成结构化控制流（if-else + while），让下游 Pass 与 codegen 不必处理非结构化跳转。',
      note: '纯规范化。' },
    { idx: 4,  phase: 'front', name: 'ConvertToSSA', decisions: 0,
      why: '转成 SSA 形式：变量重命名、插 phi 节点、建立 iter_args。',
      note: '纯规范化。SSA 名字后来成为依赖分析追踪 value-flow 的基础。' },
    { idx: 5,  phase: 'front', name: 'Simplify', decisions: 0,
      why: '用代数重写与区间分析折叠算术表达式、类型里内嵌的 shape 表达式和标量常量绑定。',
      note: '折叠是确定性的；但它曾经因为丢掉承载语义的赋值而引发 bug（#1461）。' },
    { idx: 6,  phase: 'front', name: 'FlattenCallExpr', decisions: 0,
      why: '把嵌套的调用表达式摊平成三地址形式。',
      note: '纯规范化。' },

    { idx: 7,  phase: 'scope', name: 'OutlineHierarchyScopes', decisions: 0,
      why: '把 Hierarchy scope 外提成独立函数，并在签名上带 level / role 元数据。',
      note: '外提规则由 scope 语法唯一确定。' },
    { idx: 8,  phase: 'scope', name: 'OutlineIncoreScopes', decisions: 0,
      why: '把 InCore scope 外提成独立函数。',
      note: '同上。' },
    { idx: 9,  phase: 'scope', name: 'OutlineClusterScopes', decisions: 0,
      why: '把 Cluster scope 外提成 Group 函数，把独立的 Spmd scope 外提成 Spmd 函数。',
      note: '同上。' },

    { idx: 10, phase: 'tile', name: 'ConvertTensorToTileOps', decisions: 0,
      why: '在 InCore 函数里把 tensor 算子转成 tile 算子，并同步更新 orchestration 侧的调用点。',
      note: '映射表驱动。' },
    { idx: 11, phase: 'tile', name: 'OptimizeOrchTensors', decisions: 0,
      why: '消除 orchestration 与 InCore 之间冗余的 tensor 分配，改善数据流。',
      note: '消除规则确定；曾因 out-window 外置漏掉父 SSA 重绑而丢依赖（#1444）。' },
    { idx: 12, phase: 'tile', name: 'LowerCompositeOps', decisions: 0,
      why: '把复合 tile / 分布式算子拆成原语组合（muls / adds / add / sub / mul / maximum / minimum / cast），让 codegen 永远不必处理高层算子。',
      note: '拆解路径固定。' },
    { idx: 13, phase: 'tile', name: 'FlattenTileNdTo2D', decisions: 0,
      why: '把 3D 及以上的 tile 操作摊平成 2D：除最后一维外全部合并。',
      note: '纯形状改写。' },
    { idx: 14, phase: 'tile', name: 'LegalizeTileCast', decisions: 0,
      why: '把当前 pto.tcvt profile 无法一条指令完成的 cast 对，展开成最短的原生 cast 链。',
      note: '「最短链」由 ISA 能力表唯一确定，不是启发式选择。' },
    { idx: 15, phase: 'tile', name: 'AutoTileMatmulL0', layer: 'tiling', decisions: 6,
      why: '为静态 2D 的 matmul / matmul_acc / matmul_bias 从 backend 的 L0 容量里挑一组 (m, n, k)，并把调用改写成两级流水的 K 循环，每轮插 Mat→Left/Right 的 extract。',
      note: '★ 决策点：在多个合法 tile shape 中选一个；选不出来时放弃切分并发 perf hint。在树已有 8 个理由码 PH-AT-003/005/006/007/008/009/010/011，全部是「我为什么没帮你切」。' },
    { idx: 16, phase: 'tile', name: 'CanonicalizeTileSlice', decisions: 0,
      why: '把 tile.slice 下降成规范的 tile.extract 形式，让所有搬运统一走 pto.textract。',
      note: '规范化；但 slice 的 MemRef 编码曾静默产生错数据（#2010）。' },
    { idx: 17, phase: 'tile', name: 'InferTileMemorySpace', layer: 'tiling', decisions: 3,
      why: '为 InCore 里每个 tile 推断片上 MemorySpace，在生产者与消费者约束不匹配处插 tile.move，并让可证明循环不变的 Mat 操作数跨迭代常驻。',
      note: '★ 决策点：空间归属与「要不要常驻」都是选择，且直接决定后面复用与同步的形态。' },
    { idx: 18, phase: 'tile', name: 'InsertMxScaleAddr', decisions: 0,
      why: '在所有操作数的内存空间确定之后，为 MX matmul 的消费者插入编译器生成的 tile.tget_scale_addr 绑定。',
      note: '插入位置由数据流唯一确定。' },
    { idx: 19, phase: 'tile', name: 'ResolveBackendOpLayouts', layer: 'tiling', decisions: 2,
      why: '修复 elementwise 算子的 backend 要求布局：[N,1] 列主向量重塑成 [1,N] 行主视图，其余非行主 tile 走 tile.move 强制转换。',
      note: '★ 决策点：改布局还是插搬运，是有代价的选择；layout 类问题在仓里长期高发。' },

    { idx: 20, phase: 'split', name: 'LowerAutoVectorSplit', layer: 'tiling', decisions: 2,
      why: '把 AUTO 的 pl.split 混合 InCore 函数转成显式的 split_aiv 形式。',
      note: '★ 决策点：AUTO 模式下由编译器推断切分轴与方式。' },
    { idx: 21, phase: 'split', name: 'ExpandMixedKernel', layer: 'tiling', decisions: 3,
      why: '把混合 InCore 函数拆成 AIC（Cube）+ AIV（Vector）两个 kernel，外面包一层 Group 函数；非混合的直接改 FunctionType。',
      note: '★ 决策点：算子归 cube 还是 vector 的分类会出错（#1083 把 tile.create 误判为 VECTOR）。' },
    { idx: 22, phase: 'split', name: 'InjectGMPipeBuffer', decisions: 0,
      why: '为需要经 GM 中转 slot 数据的 backend（当前 Ascend910B）注入 __gm_pipe_buffer workspace 参数。',
      note: '按 backend 能力表注入。' },
    { idx: 23, phase: 'split', name: 'SplitVectorKernel', layer: 'tiling', decisions: 4,
      why: '盖 split 属性，并处理 no-split 的双 AIV 路径。',
      note: '★ 决策点：RFC #1820 的诊断是「逐语句的语法半分器，没有 index space 模型」，要求用显式 pl.split_aiv 整体替换掉它的推断。' },
    { idx: 24, phase: 'split', name: 'StampTfreeSplit', decisions: 0,
      why: '把每个跨核 tpop 的 split 与 pipe id 复制到与之配对的 tfree 上。',
      note: '纯复制。' },
    { idx: 25, phase: 'split', name: 'NormalizeReturnOrder', decisions: 0,
      why: '把每个 InCore 函数的返回元组重排成规范顺序。',
      note: '纯规范化。' },
    { idx: 26, phase: 'split', name: 'SkewCrossCorePipeline', layer: 'tiling', decisions: 2,
      why: '对混合 cube/vector 的跨核 pl.pipeline 循环做软件流水，让两个核重叠执行。',
      note: '★ 决策点：错峰多少、哪些语句进哪一级，都是调度选择。#2130 指出它生成不了 skewed 跨核调度时用户只能回落手工 sync_set/sync_wait。' },
    { idx: 27, phase: 'split', name: 'LowerPipelineToSlots', layer: 'tiling', decisions: 1,
      why: '把 pl.pipeline(N, stage=F) 的循环体在一块分配的 F 个 slot 之间轮转，而不是复制 F 份（memory_planner=PTOAS 路径）。',
      note: '★ 决策点：轮转 vs 复制是两种多缓冲策略。' },
    { idx: 28, phase: 'split', name: 'LowerPipelineLoops', layer: 'tiling', decisions: 3,
      why: '在 tile 层下降 pl.pipeline(N, stage=F)：把循环体复制 F 份以支持 ping-pong 缓冲，外层循环保持顺序。',
      note: '★ 决策点：实际达成的深度可能低于用户请求，见 PH-MR-001。' },
    { idx: 29, phase: 'split', name: 'CanonicalizeIOOrder', layer: 'tiling', decisions: 2,
      why: '在 ForKind::Pipeline 循环体内，按同核硬件单元的阶梯（scalar → load → compute → store）重排语句，受 SSA 依赖图约束。',
      note: '★ 决策点：重排就是调度；把复制体的 load 聚到前面直接决定流水气泡。' },

    { idx: 30, phase: 'mem', name: 'MaterializeTensorStrides', decisions: 0,
      why: '为程序里每一个还没有 stride 的 TensorType / DistributedTensorType 填上该布局的紧致规范 stride。',
      note: '按布局唯一推导。' },
    { idx: 31, phase: 'mem', name: 'InitMemRef', layer: 'memory', decisions: 9,
      why: '为所有变量初始化 MemRef，并创建尚未分配地址的 alloc 操作。',
      note: '★ 决策点：谁跟谁共用一个 MemRef 的初始判断在这里定型；alloc 全部被提到函数体头部。' },
    { idx: 32, phase: 'mem', name: 'MaterializeSemanticAliases', layer: 'memory', decisions: 2,
      why: '把程序语义要求必须是同一块分配的 buffer 强制合并（loop-carry、in-place）。',
      note: '★ 决策点：语义强制，非启发式——但「哪些算语义要求」本身是判断。' },
    { idx: 33, phase: 'mem', name: 'MemoryReuse', layer: 'memory', decisions: 7,
      why: '按生命周期分析找复用机会：同一内存空间内、生命周期不重叠且无别名禁令的 buffer 合并成一块，然后删掉不再被引用的 alloc。全局 largest-first 打包。',
      note: '★ 核心决策点。约束来自 can_share 与 ForbidAliasCollector（not_inplace_safe / forbid_output_alias / 加宽 cast 三个来源）。理由今天只走 LOG_DEBUG；唯一对外发声的是流水深度降级 PH-MR-001。' },
    { idx: 34, phase: 'mem', name: 'AllocateMemoryAddr', layer: 'memory', decisions: 4,
      why: '给已有 alloc 的 MemRef 分配真实地址。',
      note: '★ 决策点：packing 顺序与对齐策略影响 high-water；DSA-RP 模式下这里还要在容量与复用惩罚下联合求解。已有理由码 PH-DSA-001。' },
    { idx: 35, phase: 'mem', name: 'FoldNoOpReshape', decisions: 0,
      why: '折叠既不改变物理形状也不改变分配的 tile.reshape。',
      note: '纯折叠。' },
    { idx: 36, phase: 'mem', name: 'FuseCreateAssembleToSlice', decisions: 0,
      why: '把 tensor.create + tensor.assemble 融成一个 tensor.slice 视图，消掉中间 buffer。',
      note: '模式匹配。' },

    { idx: 37, phase: 'dep', name: 'DeriveCallDirections', layer: 'deps', decisions: 3,
      why: '先把每个 Group/Spmd wrapper 的有效 ParamDirection 落到签名上，再遍历函数体，为每个跨函数 Call 的每个实参推导 ArgDirection。',
      note: '★ 决策点：Input / InOut / Output 的判定直接决定 runtime 能不能看到写者版本——WAR 丢失的源头之一。' },
    { idx: 38, phase: 'dep', name: 'AutoDeriveTaskDependencies', layer: 'deps', decisions: 5,
      why: '在 AUTO scope 内推导保守的任务间依赖边，写进 compiler_manual_dep_edges；用户写的 deps 留在 manual_dep_edges，两者故意分开存以保住 IR dump 里的出处。',
      note: '★ 核心决策点。无法证明用户边覆盖时会触发 dynamic_prior_producer_requires_scope_lift 回退，把 manual_scope 降级成 AUTO（#1744）；WAR 反依赖至今不推（#2058）。' },
    { idx: 39, phase: 'dep', name: 'ExpandManualPhaseFence', layer: 'deps', decisions: 1,
      why: '在 manual scope 里把划算的整数组 TaskId 依赖压缩成相位栅栏。',
      note: '★ 决策点：「划算」是启发式判断。' },

    { idx: 40, phase: 'dist', name: 'SynthesizeAllReduceSignals', decisions: 0,
      why: '把 host 级 allreduce 的可选 signal 规范成显式的内部 signal IR。',
      note: '规范化。' },
    { idx: 41, phase: 'dist', name: 'MaterializeCommDomainScopes', decisions: 0,
      why: '遍历每个 host orchestration 函数体，装配 WindowBuffer 与 CommDomainScopeStmt 包装。',
      note: '按通信域声明装配。' },
    { idx: 42, phase: 'dist', name: 'LowerHostTensorCollectives', decisions: 0,
      why: '把 host 级 tensor 集合通信改写成内部 builtin 的 chip dispatch。',
      note: '一对一改写。' },
    { idx: 43, phase: 'dist', name: 'MaterializeDistTensorCtx', decisions: 0,
      why: '为每个 DistributedTensor 物化一个显式的 CommCtx 参数与实参。',
      note: '按类型唯一确定。' },

    { idx: 44, phase: 'tail', name: 'MaterializeRuntimeScopes', layer: 'deps', decisions: 2,
      why: '往 Orchestration 函数里插入显式的 AUTO RuntimeScopeStmt，使 codegen 能 1:1 发出 PTO2_SCOPE。',
      note: '★ 决策点：消费 Pass 38 留下的标记，决定这一段最终发 AUTO 还是编译器自有的 MANUAL。' },
    { idx: 45, phase: 'tail', name: 'ClassifyIterArgCarry', decisions: 0,
      why: '把 Orchestration 里每个 ForStmt 的 iter_arg 分类成平凡别名或需要物化的重绑 carry。',
      note: '分类规则确定。' },
    { idx: 46, phase: 'tail', name: 'InsertCommFence', layer: 'sync', decisions: 1,
      why: '实现 data-before-signal 的内存一致性契约：在每个发布写与释放它的 pld.system.notify 之间插入整张量 system.cacheinvalid + GM system.fence。',
      note: '★ 决策点：fence 的范围与位置是选择；#1561 就是漏插导致 signal 抢在远端 TSTORE 之前。' },
    { idx: 47, phase: 'tail', name: 'MaterializeValidShapeSymbols', decisions: 0,
      why: '把设备 kernel 绑不上的 valid_shape 符号变成前置的 Scalar[INDEX] 参数，由调用方喂进真实的有效范围。',
      note: '机械提参。' },

    { idx: 90, phase: 'ptoas', name: 'PTOAS · PlanMemory', layer: 'memory', decisions: 0, foreign: true,
      why: 'level2 侧的内存规划：与 pypto 的 MemoryReuse 同类工作，在 tile_buf 世界里做生命周期复用。',
      note: '★ 决策点（本次编译走 pypto planner，故为 0 条）。#934 记录过它把存活 tile 的 UB 区间借给 row_sum tmp 且缺同步。' },
    { idx: 91, phase: 'ptoas', name: 'PTOAS · InsertSync', layer: 'sync', decisions: 8, foreign: true,
      why: '为跨 pipe 的读写冒险自动插入 set_flag / wait_flag / pipe_barrier，用户不必手工管流水同步。',
      note: '★ 核心决策点。漏插→静默错数据；多插→死锁；插得太保守→比手工慢 10%。可解释性数据结构 VFDecisionReason 已在 PR#605 设计好但未合入。' },
    { idx: 92, phase: 'ptoas', name: 'PTOAS · SyncEventIdAllocation', layer: 'sync', decisions: 3, foreign: true,
      why: '为同步 flag 分配 EVENT_ID。',
      note: '★ 决策点：#823 记录过它给重叠 fence 复用同一个全局 event 导致 AIV 死锁；#1789 记录过删掉一个无关算子导致 EVENT_ID 重排后死锁。' },
    { idx: 93, phase: 'ptoas', name: 'PTOAS · RemoveRedundantSync', layer: 'sync', decisions: 0, foreign: true,
      why: '删除可证明多余的同步。',
      note: '★ 决策点（本次无删除）。#646 记录过它没删掉的一条 PIPE_V barrier——手工删掉后正确性不变、长序列 FA 吞吐提到 ~165 TFLOP/s。' }
  ];

  /* ---- 决策记录（decisions.json 的形态） ---- */
  var DECISIONS = [
    {
      id: 'D-38-01', pass: 38, layer: 'deps', level: 'L3',
      kind: 'scope_demotion',
      title: 'pl.manual_scope() 被降级为 AUTO',
      subject: ['fa_qks_scope', 'qkv_proj_scope', 'mlp_band_scope', 'lm_head_scope'],
      trigger: 'dynamic_prior_producer_requires_scope_lift',
      triggerText: '依赖经由 pl.array 的 TaskId 元素（arr[i]）表达，Pass 无法把它静态对应回循环内的 producer，判定「未被用户边覆盖」。',
      protects: '对 acc 共享缓冲的 RAW / WAW 冒险',
      declined: '保留 MANUAL 并信任用户的 deps=[arr[i]] 边',
      cost: { kind: 'serialization', text: '整个 scope 交回 runtime OverlapMap 追踪；写同一缓冲的任务被保守全串行', metric: '并行度 4 → 1' },
      confidence: 'conservative',
      overrideCode: '# 方式一：把 TaskId 摊平成具名变量，Pass 即可静态匹配\ntid0, tid1, tid2, tid3 = prod_tids[0], prod_tids[1], prod_tids[2], prod_tids[3]\nwith pl.at(level=pl.Level.CORE_GROUP, deps=[tid0, tid1, tid2, tid3]):\n    ...\n\n# 方式二：把期望写成断言（L3 契约）——编译器再想降级就报错\nwith pl.manual_scope(assert_manual=True):\n    ...',
      evidence: [
        { type: 'pass_dump', path: 'passes_dump/38_after_AutoDeriveTaskDependencies.py', line: 218, note: 'RuntimeScopeStmt(manual=False)' },
        { type: 'codegen',   path: 'orchestration/decode_layer.cpp', line: 476, note: 'PTO2_SCOPE(PTO2ScopeMode::AUTO)' }
      ],
      span: { file: 'models/qwen3/14b/decode_layer.py', line: 214 },
      issue: 'pypto#1744'
    },
    {
      id: 'D-33-04', pass: 33, layer: 'memory', level: 'L2',
      kind: 'buffer_coalesce',
      title: 'k_tile1 的 Mat buffer 合并到 k_tile0 上',
      subject: ['mem_mat_12', 'mem_mat_17'],
      trigger: 'lifetime_disjoint + can_share',
      triggerText: 'mem_mat_17 的定义点晚于 mem_mat_12 的最后一次使用；同为 Mat 空间；无 not_inplace_safe / forbid_output_alias 约束。全局 largest-first 打包把它并入首个可共享 buffer。',
      protects: '省下 32768 B Mat 空间（Mat 占用 128 KB → 96 KB）',
      declined: '保持两块独立 buffer，两个 K-load 可并行发射',
      cost: { kind: 'serialization', text: '引入一条只因复用而存在的 MTE1→MTE2 WAR 边：matmul1 的 TLOAD 必须等 matmul0 的 TMOV 完成', metric: 'cube 两个 matmul 串行 · 实测 +38 μs / iter' },
      confidence: 'proven',
      confidenceNote: '正确性可证明；性能代价未参与决策——打包只按容量与生命周期评分。',
      overrideCode: '# 局部否决：让这块 K tile 不参与复用打包\nk_tile1 = pl.no_reuse(pl.tile.load(k_gm, ...))\n\n# 或按 buffer 粒度声明预算，让打包器保留并行\nwith pl.memory_policy(mat_reserve="parallel_matmul"):\n    ...',
      evidence: [
        { type: 'pass_dump', path: 'passes_dump/28_after_InitMemRef.py',  line: 96,  note: '4 个 Mat alloc：12 / 13 / 17 / 18' },
        { type: 'pass_dump', path: 'passes_dump/29_after_MemoryReuse.py', line: 142, note: '仅剩 2 个 Mat alloc' },
        { type: 'codegen',   path: 'kernels/aic/fa_qks_aic.cpp', line: 90,  note: 'TMOV(v44, v32) — matmul0 对该 buffer 的最后一次读' },
        { type: 'codegen',   path: 'kernels/aic/fa_qks_aic.cpp', line: 114, note: 'wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1) — matmul1 阻塞在此' }
      ],
      span: { file: 'models/qwen3/14b/decode_layer.py', line: 88 },
      issue: 'pypto#1475'
    },
    {
      id: 'D-33-06', pass: 33, layer: 'memory', level: 'L2',
      kind: 'pipeline_depth_shed',
      title: '软件流水深度 4 只满足到 2',
      subject: ['pipeline group qk_pv', 'Mem.Vec'],
      trigger: 'capacity_gate · slot_bound',
      triggerText: '每 stage 需 49408 B，Vec 空间保留区之后仅剩 98816 B 自由容量；slot × requested = 197632 B 溢出。',
      protects: '不越过 Vec 容量上界',
      declined: '按 pl.pipeline(stage=4) 请求的深度分配 4 份缓冲',
      cost: { kind: 'serialization', text: 'stage k 与 k+2 共享缓冲并重新串行化——正是流水本想避免的假 WAR', metric: '实际深度 2 / 请求 4' },
      confidence: 'proven',
      overrideCode: '# 依据 PH-MR-001 给出的两条具体动作：\n# (a) 把每 stage 的 tile 缩到 <= 24704 B\n# (b) 或把流水深度降到实际可达值，避免误以为拿到了 4\nwith pl.pipeline(n_iter, stage=2):\n    ...',
      evidence: [
        { type: 'perf_hint', path: 'report/perf_hints.log', line: 12, note: '[perf_hint PH-MR-001] software pipelining requested depth 4 …' }
      ],
      span: { file: 'models/qwen3/14b/decode_layer.py', line: 132 },
      issue: 'PH-MR-001（在树）'
    },
    {
      id: 'D-15-02', pass: 15, layer: 'tiling', level: 'L2',
      kind: 'tiling_declined',
      title: 'kv_proj 的 matmul 放弃 L0 切分',
      subject: ['tile.matmul @ kv_proj'],
      trigger: 'PH-AT-007 · K 非 cube fractal 对齐',
      triggerText: 'K=1000 不是 cube fractal 16 的倍数；任何尾块或整 K 块都会产生非 fractal 列，没有合法的 K 切分方案。',
      protects: '不生成非法的 extract',
      declined: '按 (m=64, n=128, k=128) 切分该 matmul',
      cost: { kind: 'lost_opportunity', text: '该 matmul 保持整块执行，L0 利用率偏低；left untouched', metric: 'L0C 利用率 ~41%' },
      confidence: 'proven',
      overrideCode: '# 把 K padding 到 16 的倍数，切分即可生效\nk_padded = pl.pad(k, to_multiple_of=16)   # 1000 -> 1008',
      evidence: [
        { type: 'perf_hint', path: 'report/perf_hints.log', line: 4, note: '[perf_hint PH-AT-007] K=1000 is not a multiple of the cube fractal 16 — left untouched.' }
      ],
      span: { file: 'models/qwen3/14b/decode_layer.py', line: 61 },
      issue: 'PH-AT-007（在树）'
    },
    {
      id: 'D-15-01', pass: 15, layer: 'tiling', level: 'L1',
      kind: 'tiling_applied',
      title: 'fa_qks 的 QK matmul 切到 (64, 128, 128)',
      subject: ['tile.matmul @ fa_qks'],
      trigger: 'ChooseL0Tile · roofline',
      triggerText: '按 backend L0 容量（L0A/L0B/L0C）与 roofline 选出 (m=64, n=128, k=128)。',
      protects: '—',
      declined: '(64, 256, 128)：L0C 需 128 KB，超出容量',
      cost: { kind: 'none', text: '无用户可感代价', metric: '—' },
      confidence: 'proven',
      overrideCode: '# 显式指定 L0 tile（绕过自动选择）\npl.matmul(q, k, l0_tile=(64, 128, 128))',
      evidence: [
        { type: 'pass_dump', path: 'passes_dump/15_after_AutoTileMatmulL0.py', line: 71, note: '切分后的三层循环' }
      ],
      span: { file: 'models/qwen3/14b/decode_layer.py', line: 84 },
      issue: null
    },
    {
      id: 'D-90-03', pass: 91, layer: 'sync', level: 'L2',
      kind: 'sync_insert',
      title: '为复用边界插入 MTE1→MTE2 flag（EVENT_ID1）',
      subject: ['EVENT_ID1', 'mem_mat_12'],
      trigger: '上游 D-33-04 制造的 WAR 冒险',
      triggerText: 'InsertSync 看到同一物理 buffer 上的「读 → 写」次序，必须插 set/wait flag 保护。它无从判断这个冒险是真实数据依赖还是复用副产物。',
      protects: 'matmul0 对 mem_mat_12 的最后一次读',
      declined: '无——依赖既然存在就必须保护',
      cost: { kind: 'serialization', text: 'matmul1 的 K-load 阻塞在 wait_flag 上', metric: 'MTE2 空档 ~38 μs' },
      confidence: 'conservative',
      confidenceNote: '这是一条被动决策：真正该被质疑的是上游 D-33-04，而不是这条同步。',
      overrideCode: '# 无法在同步层解决——依赖存在就必须保护。\n# 出口在上游：撤销复用决策 D-33-04。',
      evidence: [
        { type: 'codegen', path: 'kernels/aic/fa_qks_aic.cpp', line: 91,  note: 'set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1)' },
        { type: 'codegen', path: 'kernels/aic/fa_qks_aic.cpp', line: 114, note: 'wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1)' }
      ],
      span: { file: 'kernels/aic/fa_qks_aic.cpp', line: 114 },
      issue: 'pypto#1475（同一根因）',
      upstream: 'D-33-04'
    }
  ];

  /* ---- 对象时间线：mem_mat_12 穿过流水线 ---- */
  var TIMELINE = {
    object: 'mem_mat_12',
    kind: 'Mat buffer · 32768 B',
    nodes: [
      { pass: 31, name: 'InitMemRef',                 state: '创建 · 32768 B · k_tile0', touch: 'create', decision: null },
      { pass: 32, name: 'MaterializeSemanticAliases', state: '未触碰', touch: 'skip', decision: null },
      { pass: 33, name: 'MemoryReuse',                state: '合并 mem_mat_17（k_tile1）进来', touch: 'decide', decision: 'D-33-04',
        warn: '引入 MTE1→MTE2 WAR 边' },
      { pass: 34, name: 'AllocateMemoryAddr',         state: 'addr = 0', touch: 'assign', decision: null },
      { pass: 91, name: 'PTOAS · InsertSync',         state: 'EVENT_ID1 守护该复用边界', touch: 'decide', decision: 'D-90-03' },
      { pass: 99, name: 'codegen',                    state: 'fa_qks_aic.cpp:90 / :114', touch: 'emit', decision: null }
    ]
  };

  /* ---- Memory Map：Mat 空间，pass 32（前）vs pass 33（后） ---- */
  var MEMMAP = {
    space: 'Mem.Mat',
    capacity: 131072,
    lineRange: [70, 160],
    before: {
      label: '32_after_MaterializeSemanticAliases',
      boxes: [
        { base: 'mem_mat_12', name: 'k_tile0',    addr: 0,     size: 32768, from: 78,  to: 96,  role: 'k' },
        { base: 'mem_mat_13', name: 'q_padded0',  addr: 32768, size: 4096,  from: 80,  to: 94,  role: 'q' },
        { base: 'mem_mat_17', name: 'k_tile1',    addr: 36864, size: 32768, from: 104, to: 128, role: 'k' },
        { base: 'mem_mat_18', name: 'q_padded1',  addr: 69632, size: 4096,  from: 106, to: 126, role: 'q' }
      ],
      edges: []
    },
    after: {
      label: '33_after_MemoryReuse',
      boxes: [
        { base: 'mem_mat_12', name: 'k_tile0',   addr: 0,     size: 32768, from: 78,  to: 96,  role: 'k', seg: 1 },
        { base: 'mem_mat_12', name: 'k_tile1',   addr: 0,     size: 32768, from: 104, to: 128, role: 'k', seg: 2, reused: true },
        { base: 'mem_mat_13', name: 'q_padded0', addr: 32768, size: 4096,  from: 80,  to: 94,  role: 'q', seg: 1 },
        { base: 'mem_mat_13', name: 'q_padded1', addr: 32768, size: 4096,  from: 106, to: 126, role: 'q', seg: 2, reused: true }
      ],
      edges: [
        { from: 'k_tile0', to: 'k_tile1', pipe: 'MTE1→MTE2', event: 'EVENT_ID1', decision: 'D-33-04', label: '复用引入的 WAR 边' },
        { from: 'q_padded0', to: 'q_padded1', pipe: 'MTE1→MTE2', event: 'EVENT_ID2', decision: 'D-33-04', label: '同上（次要路径）' }
      ],
      blocked: [
        { base: 'mem_vec_9', name: 'softmax_tmp', reason: 'tile.row_sum 声明 not_inplace_safe：输出不得与任何输入共享 buffer', addr: 73728, size: 8192, from: 96, to: 124 }
      ]
    }
  };

  /* ---- 同步泳道：auto（复用后）vs 假设（撤销复用） ---- */
  var SWIMLANE = {
    pipes: ['MTE2', 'MTE1', 'M(cube)'],
    span: 200,
    current: {
      label: '当前 · 复用后（D-33-04 生效）',
      total: '112 μs / iter',
      ops: [
        { pipe: 'MTE2', t: 2,   w: 30, name: 'TLOAD k0',  tone: 'k' },
        { pipe: 'MTE1', t: 34,  w: 14, name: 'TMOV v44',  tone: 'mov', line: 90 },
        { pipe: 'M(cube)', t: 50, w: 34, name: 'TMATMUL 0', tone: 'mm' },
        { pipe: 'MTE2', t: 88,  w: 30, name: 'TLOAD k1',  tone: 'k', blocked: true, line: 115 },
        { pipe: 'M(cube)', t: 122, w: 34, name: 'TMATMUL 1', tone: 'mm' }
      ],
      flags: [
        { from: 'MTE1', to: 'MTE2', t: 48, label: 'set/wait EVENT_ID1', decision: 'D-90-03' }
      ],
      gaps: [ { pipe: 'MTE2', t: 32, w: 56, label: 'MTE2 空档 38 μs — 等 matmul0 的 TMOV' } ]
    },
    counterfactual: {
      label: '假设 · 撤销复用（pl.no_reuse）',
      total: '74 μs / iter',
      ops: [
        { pipe: 'MTE2', t: 2,  w: 30, name: 'TLOAD k0',  tone: 'k' },
        { pipe: 'MTE2', t: 34, w: 30, name: 'TLOAD k1',  tone: 'k' },
        { pipe: 'MTE1', t: 34, w: 14, name: 'TMOV v44',  tone: 'mov' },
        { pipe: 'M(cube)', t: 50, w: 34, name: 'TMATMUL 0', tone: 'mm' },
        { pipe: 'M(cube)', t: 86, w: 34, name: 'TMATMUL 1', tone: 'mm' }
      ],
      flags: [],
      gaps: [],
      note: 'Mat 占用 96 KB → 128 KB（仍在容量内）'
    }
  };

  /* ---- 决策 diff：应用 override 前后 ---- */
  var DIFF = {
    left:  { label: 'C-4417 · 当前', sha: 'pypto 21f11ecb' },
    right: { label: 'C-4418 · 应用 pl.no_reuse 后', sha: 'pypto 21f11ecb + override' },
    rows: [
      { kind: 'removed', id: 'D-33-04', layer: 'memory', text: 'MemoryReuse：k_tile1 合并到 mem_mat_12', note: '被 pl.no_reuse 否决' },
      { kind: 'removed', id: 'D-90-03', layer: 'sync',   text: 'InsertSync：MTE1→MTE2 flag（EVENT_ID1）', note: '上游冒险消失，同步随之消失' },
      { kind: 'changed', id: 'D-34-02', layer: 'memory', text: 'AllocateMemoryAddr：Mat high-water 96 KB → 128 KB', note: '仍在 131072 B 容量内' },
      { kind: 'same',    id: 'D-33-06', layer: 'memory', text: 'MemoryReuse：流水深度 4 → 2', note: '未受影响（Vec 空间）' },
      { kind: 'same',    id: 'D-38-01', layer: 'deps',   text: 'AutoDeriveTaskDependencies：manual_scope 降级', note: '未受影响，仍需单独处理' }
    ],
    outcome: { before: '112 μs / iter', after: '74 μs / iter', delta: '−34%', correctness: 'Golden PASS（未变化）' }
  };

  /* ---- 源码 / IR 视图 ---- */
  var SOURCE = {
    file: 'models/qwen3/14b/decode_layer.py',
    lines: [
      { n: 80, t: '    @pl.function' },
      { n: 81, t: '    def fa_qks(self, q: pl.Tensor[[64, 128], pl.BF16],' },
      { n: 82, t: '               k0: pl.Tensor[[128, 128], pl.BF16],' },
      { n: 83, t: '               k1: pl.Tensor[[128, 128], pl.BF16]) -> None:' },
      { n: 84, t: '        qk0 = pl.matmul(q, k0, out_dtype=pl.FP32)', dec: 'D-15-01' },
      { n: 85, t: '' },
      { n: 86, t: '        # 第二个 QK：与上一个在数据上完全独立' },
      { n: 87, t: '        # 期望两个 K-load 在 MTE2 上重叠发射' },
      { n: 88, t: '        qk1 = pl.matmul(q, k1, out_dtype=pl.FP32)', dec: 'D-33-04' },
      { n: 89, t: '' },
      { n: 90, t: '        mi = pl.tile.row_max(pl.concat(qk0, qk1))' },
      { n: 91, t: '        ...' },
      { n: 130, t: '' },
      { n: 131, t: '    # online-softmax 的两级流水' },
      { n: 132, t: '        with pl.pipeline(n_iter, stage=4):', dec: 'D-33-06' },
      { n: 133, t: '            ...' },
      { n: 212, t: '' },
      { n: 213, t: '        # 手工编排：显式 deps，避免 runtime 保守串行' },
      { n: 214, t: '        with pl.manual_scope():', dec: 'D-38-01' },
      { n: 215, t: '            prod_tids = pl.array.create(4, pl.TASK_ID)' },
      { n: 216, t: '            for n in pl.parallel(4):' },
      { n: 217, t: '                with pl.at(level=pl.Level.CORE_GROUP) as p_tid:' },
      { n: 218, t: '                    acc = pl.assemble(acc, c, [0, n * 128], atomic=pl.AtomicType.Add)' },
      { n: 219, t: '                prod_tids[n] = p_tid' }
    ]
  };

  var TREE = [
    { d: 0, t: 'build_output/decode_layer/', kind: 'dir' },
    { d: 1, t: 'report/', kind: 'dir' },
    { d: 2, t: 'decisions.json', kind: 'json', badge: '47' },
    { d: 2, t: 'perf_hints.log', kind: 'log', badge: '11' },
    { d: 1, t: 'passes_dump/', kind: 'dir' },
    { d: 2, t: '28_after_InitMemRef.py', kind: 'ir' },
    { d: 2, t: '29_after_MemoryReuse.py', kind: 'ir', hot: true },
    { d: 2, t: '38_after_AutoDeriveTaskDependencies.py', kind: 'ir' },
    { d: 1, t: 'kernels/aic/', kind: 'dir' },
    { d: 2, t: 'fa_qks_aic.cpp', kind: 'cpp', hot: true },
    { d: 1, t: 'ptoas/', kind: 'dir' },
    { d: 2, t: 'fa_qks.pto', kind: 'ir' }
  ];

  /* ---- 回放步骤 ---- */
  var STEPS = [
    {
      id: 's1', phase: '编译完成', stage: 'spine', focus: null, mode: 'decisions',
      title: '第一眼不是 51 个 Pass，是一句摘要',
      body: '编译结束。界面默认给出的是「本次编译 75 个决策 · 3 个需要复核 · 1 个推翻了你的显式意图」，以及折叠态的决策脊。左栏列出全部 51 个 Pass（pypto 47 + PTOAS 4）并说明各自做什么，但只有 22 个在做决定——决策视图默认只画这 22 个，其余可展开查看。',
      readout: 'decisions.json · 75 条 · 默认开启，零额外开销'
    },
    {
      id: 's2', phase: 'L3 先行', stage: 'spine', focus: 'D-38-01', mode: 'decisions',
      title: 'L3：你写的 manual_scope 没有被执行',
      body: '唯一必须回答的一条被顶置。判据是可机器判定的——你在源码里写了 pl.manual_scope()，编译器最终发出的是 PTO2_SCOPE(AUTO)。它不是猜你会不会在意，而是确认「用户的显式意图被推翻」。',
      readout: 'L3 · 4 个 scope 受影响 · 必须 ack'
    },
    {
      id: 's3', phase: '展开复用层', stage: 'spine', focus: 'D-33-04', mode: 'decisions',
      title: '跨层弧线：性能问题的源头在三个 Pass 之前',
      body: '决策脊按四层着色。注意 33 MemoryReuse 与 PTOAS InsertSync 之间的那条弧线——同步层那条 flag 不是独立决策，它是复用决策的下游产物。没有这条弧线，四层还是四个孤岛。',
      readout: '内存复用层 · 22 个决策 · 2 个 L2'
    },
    {
      id: 's4', phase: '追对象', stage: 'timeline', focus: 'D-33-04', mode: 'decisions',
      title: '用户追的是 buffer，不是 Pass',
      body: 'mem_mat_12 穿过整条流水线的完整履历：31 创建 → 33 被合并（⚠ 引入 WAR 边）→ 34 分配地址 → InsertSync 加锁 → codegen 落到 :90 / :114。issue #1475 的作者手工走了五步才拼出这条线；这里是一屏。',
      readout: '对象时间线 · mem_mat_12 · 6 个节点 · 2 个决策'
    },
    {
      id: 's5', phase: '看地址', stage: 'memmap', focus: 'D-33-04', mode: 'decisions',
      title: 'Memory Map 加三样：分段、复用边、被挡住的复用',
      body: '左右是 pass 32 与 pass 33，地址轴对齐。右侧 mem_mat_12 画成一根竖条上的两个分段，段间的橙色连线就是复用引入的 MTE1→MTE2 WAR 边——现有 memory_map 只显示复用发生了，不显示它的代价。灰色虚线段是被 not_inplace_safe 挡住、没能复用的位置：「我为什么没帮你省这块」同样要可见。',
      readout: 'Mem.Mat · high-water 128 KB → 96 KB · 代价见橙线'
    },
    {
      id: 's6', phase: '看时间', stage: 'swimlane', focus: 'D-90-03', mode: 'decisions',
      title: '双泳道对比是默认视图，不是高级功能',
      body: '上：当前。matmul1 的 TLOAD 阻塞在 wait_flag 上，MTE2 空了 38 μs。下：撤销复用的反事实。PTOAS#226 的作者当年是自己贴 msprof 双时间线截图，另一位工程师从图里读出气泡差在哪——那段对话就是这个视图的需求文档。',
      readout: '112 μs → 74 μs（反事实）· MTE2 空档 38 μs'
    },
    {
      id: 's7', phase: '出口', stage: 'memmap', focus: 'D-33-04', mode: 'decisions', showOverride: true,
      title: '有身份、有标价，还要有出口',
      body: '决策卡上的按钮不叫「应用」，叫「复制 override 代码」。override 必须落到源码里、能被 code review、能进复现包——存成 UI 偏好就会脱离 git 并在下一次环境迁移时静默丢失。',
      readout: 'override · pl.no_reuse(k_tile1) · 待写入源码'
    },
    {
      id: 's8', phase: '复核', stage: 'diff', focus: null, mode: 'decisions',
      title: '决策 diff：两次编译之间，编译器换了哪几个决定',
      body: '重编译后做决策集合差：复用决策消失，它下游的同步决策随之消失，地址分配变化但仍在容量内。这个交互同时服务跨版本排查——PTOAS#1111 那种「算术 opcode 一条没变但结果回归」的问题，今天要逐版本 bisect 读汇编，这里是一屏。',
      readout: '2 removed · 1 changed · 2 same · 正确性未变'
    },
    {
      id: 's9', phase: '回到源码', stage: 'inline', focus: 'D-33-04', mode: 'source',
      title: 'diff 之上叠一层语义，而不是把 diff 做得更好看',
      body: 'ir_trace 已经把行级 diff 做得很细了，它回答「改了哪些行」。决策标注挂在源码与 IR 的改动行旁边，回答「为什么这么改」。两者是两层，不互相替代。',
      readout: '源码 4 行带决策标注 · 点击跳转决策卡'
    }
  ];

  return {
    session: {
      id: 'C-4417',
      title: 'qwen3_14b/decode_layer.py · fa_qks 编译决策回放',
      env: 'a2a3 · Ascend 910B · pypto 21f11ecb · ptoas 0.48',
      total: 75, review: 3, override: 1
    },
    layers: LAYERS, phases: PHASES, passes: PASSES, decisions: DECISIONS,
    timeline: TIMELINE, memmap: MEMMAP, swimlane: SWIMLANE,
    diff: DIFF, source: SOURCE, tree: TREE, steps: STEPS
  };
})();
