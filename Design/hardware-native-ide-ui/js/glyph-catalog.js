/* 图元目录页：把 LogicGlyphs 的每个图元摆成一张卡片。
 * 每张卡片回答三件事 —— 表达什么、必须区分开的两样、相对 path.html 是复用/迭代/新增。
 * 卡片顺序 = 梳理里的 8 类 + 存储放置 + 9 种关系。 */
(function () {
  'use strict';
  const G = window.LogicGlyphs;

  // ── 分组：num / 标题 / 一句话 / 谁决定 ──
  const GROUPS = [
    ['1', '程序与函数边界', '这段逻辑在哪一级运行，参数往哪个方向走。', '用户写；AIC / AIV / Group 由编译器拆出'],
    ['2', '数据对象', '算的是什么数据：Tensor、Tile、Scalar、Array，以及有效区。', '用户写 shape 与 dtype；layout 与 view 多由编译器推导'],
    ['3', '计算', '做什么运算。按 DSL 的操作族归类，不借用模型层语义。', '用户写'],
    ['4', '搬运 · 形状 · 视图', '数据怎么取、怎么放，以及只改元数据的那一类。', 'tensor 级用户写；tile 级搬运多由编译器补齐'],
    ['5', '控制结构', '按什么顺序、重复几次、几份在飞。', '用户写；pipeline 的复制展开由编译器完成'],
    ['6', '执行作用域与并行映射', '映射到多少个核、哪一类核。', '用户写意图；外提与拆核由编译器完成'],
    ['7', '任务依赖与同步', '谁要等谁，跨核怎么交接。', '默认编译器与运行时推导；用户可补边或关掉'],
    ['8', '编译期辅助', '只在解析期起作用，不生成 IR。', '用户写'],
    ['S', '存储放置', '数据落到哪个存储空间、哪段地址。', '全部由编译器决定（InferTileMemorySpace → AllocateMemoryAddr）'],
    ['R', '关系', '图元之间的 9 种连接。线型本身就是语义，不靠颜色区分。', '随对象而来']
  ];

  // ── 卡片 ──
  // g: () => [svg body, 宽, 高]；origin: reuse | iterate | new
  const CARDS = [
    // ───────── 1 函数边界 ─────────
    {
      group: '1', name: '函数 / 核函数', api: 'LogicGlyphs.fnBox',
      what: '一个 Function 的边界。顶栏左边是 FunctionType（Opaque / Orchestration / InCore / AIC / AIV / Group / Spmd），右边是函数名；下面每行一个参数，针脚颜色给出方向。',
      split: ['函数定义', '它的一次派发 —— 后者是第 7 组的任务卡片，不是这个框'],
      origin: 'new', note: 'path.html 的逻辑图从 SPMD 框开始，没有函数边界这一层。针脚沿用 ParamDirection 的三个值。',
      g: () => [G.fnBox({ x: 8, y: 10, w: 250, h: 100, kind: 'INCORE', name: 'fa_fused', params: [{ name: 'q_padded', dir: 'In' }, { name: 'k_cache', dir: 'In' }, { name: 'all_oi_tmp', dir: 'InOut' }] }), 266, 120]
    },
    // ───────── 2 数据对象 ─────────
    {
      group: '2', name: 'Tensor（GM / DDR）', api: 'LogicGlyphs.dataObject',
      what: '片外的一片数据。实心角色色 + 网格纹理；标签写 shape 与 dtype，格子数不编码数量。',
      split: ['逻辑上的一片数据', '它占的那段存储 —— 后者是 MemRef 地址条'],
      origin: 'reuse', note: '直接沿用 PathGridGlyphs.tensor 的实心网格画法，只补上 dtype 与存储空间的角色色分工。',
      g: () => [G.dataObject({ x: 8, y: 14, w: 180, h: 60, label: 'k_cache', shape: '[128, 128]', dtype: 'BF16', space: 'gm' }), 196, 90]
    },
    {
      group: '2', name: 'Tile（片上）', api: 'LogicGlyphs.dataObject · space',
      what: '片上缓冲。角色色区分落点：Vec 用 sky、Mat / Left / Right 用 lavender、Acc 用 amber。',
      split: ['Tile 的形态（shape、dtype）', '它在哪个存储空间 —— 后者是编译器推导的结果'],
      origin: 'iterate', note: 'path.html 的逻辑图里 Q / K / V / out 四个张量用了手写色值（绿、紫、黄）。这里把色值收进角色色 token，和硬件映射共用一套。',
      g: () => [G.dataObject({ x: 8, y: 10, w: 168, h: 52, label: 'scores', shape: '[16, 128]', dtype: 'FP32', space: 'vec' })
        + G.dataObject({ x: 8, y: 72, w: 168, h: 52, label: 'qk_acc', shape: '[16, 128]', dtype: 'FP32', space: 'acc' }), 184, 134]
    },
    {
      group: '2', name: '有效区（valid_shape）', api: 'LogicGlyphs.dataObject · valid',
      what: 'shape 与 valid_shape 的面积比。亮的那块是有效区，其余压暗；不渲染 valid_shape=[...] 这种字段原文。',
      split: ['申请了多大', '真正有效的是哪一块 —— 两者不同是 fillpad 存在的原因'],
      origin: 'new', note: 'path.html 把 set_validshape 画成了一个计算节点，没有表达"面积变了"。有效区在数据对象上表达更准。',
      g: () => [G.dataObject({ x: 8, y: 16, w: 180, h: 64, label: 'scores_valid', shape: '[16, 128]', dtype: 'FP32', space: 'vec', valid: [42 / 128, 5 / 16] })
        + G.text(8, 96, 'shape [16, 128] · valid [5, 42]', 'gl-small'), 196, 108]
    },
    {
      group: '2', name: '标量', api: 'LogicGlyphs.scalar',
      what: '单个值：循环变量、长度、索引、TaskId。小胶囊，不带网格纹理。',
      split: ['一个标量', '一片数据 —— 标量进不了 Vec 预算，别画成同形'],
      origin: 'new', note: 'path.html 把 valid_len 并进了节点的联动行里，没有独立图元。索引关系（第 9 组）需要它当锚点。',
      g: () => [G.scalar({ x: 8, y: 12, w: 132, label: 'valid_len', dtype: 'INDEX' })
        + G.scalar({ x: 8, y: 46, w: 132, label: 'fa_core', dtype: 'INDEX' })
        + G.scalar({ x: 8, y: 80, w: 132, label: 'work_tid', dtype: 'TASK_ID' }), 148, 116]
    },
    {
      group: '2', name: 'Array（片上定长数组）', api: 'LogicGlyphs.array',
      what: '落在标量寄存器 / C 栈的一维定长数组，`pl.array.create(N, dtype)`。常见用途是存一批 TaskId。',
      split: ['Array（标量本地）', 'Tensor / Tile —— Array 不跨函数边界，也不占 Vec'],
      origin: 'new', note: 'path.html 完全没有这一类。手工依赖那条线（`tids[k]`）要用到它。',
      g: () => [G.array({ x: 10, y: 28, n: 6, label: 'grp_tids', dtype: 'TASK_ID' }), 136, 70]
    },
    // ───────── 3 计算 ─────────
    {
      group: '3', name: '算子节点 · 七个操作族', api: 'LogicGlyphs.op',
      what: '圆端胶囊 + 操作族图标。七族：逐元素、一元/激活、归约、广播、矩阵、类型转换、比较/选择。',
      split: ['算什么（族）', '算在什么形状上 —— 形状交给数据对象，不在节点里重复'],
      origin: 'iterate', note: '胶囊沿用 VisualSystem.node(\'op\')；图标分类重做。path.html 用 cache 图标画有效区掩码、reshape 图标画填充和 cast、gate 图标画 ×scale，和 DSL 操作族对不上。matmul / reduce / expand / activation 四个路径仍复用 PtoOpIcons。',
      g: () => {
        const fams = ['elementwise', 'unary', 'reduce', 'broadcast', 'matmul', 'cast', 'compare'];
        const labels = ['mul · add · div', 'exp · rsqrt · relu', 'row_max · row_sum', 'row_expand_sub', 'matmul · gemv', 'cast', 'cmp · sel'];
        return [fams.map((f, i) => G.op({ x: 8, y: 8 + i * 40, w: 230, family: f, label: labels[i] })
          + G.text(246, 29 + i * 40, G.FAMILIES[f].label, 'gl-dim')).join(''), 320, 294];
      }
    },
    // ───────── 4 搬运 / 视图 ─────────
    {
      group: '4', name: '搬运', api: 'LogicGlyphs.move',
      what: '真的动了数据：load / store / move。方角胶囊，右边写源与目的存储空间。',
      split: ['动了数据的搬运', '只改元数据的视图 —— 前者花时间，后者不花'],
      origin: 'new', note: 'path.html 把搬运并进了张量节点（K 块 = slice + load），搬运链在逻辑图里看不见。方角 vs 圆端是这一组和第 3 组的分界。',
      g: () => [G.move({ x: 8, y: 10, w: 230, label: 'load', from: 'DDR', to: 'Vec' })
        + G.move({ x: 8, y: 54, w: 230, label: 'move', from: 'Mat', to: 'Right' })
        + G.move({ x: 8, y: 98, w: 230, label: 'store', from: 'Acc', to: 'DDR' }), 246, 142]
    },
    {
      group: '4', name: '视图 / 仅元数据', api: 'LogicGlyphs.viewOp',
      what: 'slice、view、reshape、set_validshape、fillpad 这一类。空心虚框 + 两侧方括号，内部不填色。',
      split: ['换个看法（不拷贝）', '搬一份出来 —— 画成同形会让人以为多了一块缓冲'],
      origin: 'iterate', note: 'path.html 的 BUFS 已经给 scores_valid / scores 标了 alias:true，但逻辑图仍把它们画成计算节点。这个图元把那个已有判定画出来。',
      g: () => [G.viewOp({ x: 8, y: 10, w: 240, label: 'slice', note: '仅索引' })
        + G.viewOp({ x: 8, y: 54, w: 240, label: 'set_validshape', note: '仅元数据' })
        + G.viewOp({ x: 8, y: 98, w: 240, label: 'fillpad', note: '按 pad 填充' }), 256, 142]
    },
    // ───────── 5 控制结构 ─────────
    {
      group: '5', name: '循环 · 四种 ForKind', api: 'LogicGlyphs.loop',
      what: '虚线框 + 左侧粗条。顺序 1 条、并行 3 条、展开为分段、流水为 1 条加叠影层（层数 = stage）。',
      split: ['迭代次数', '同时在飞的份数 —— 只有 pipeline 的 stage 才是后者'],
      origin: 'iterate', note: 'path.html 用同一种虚线框画 SPMD、grid-stride 和 pipeline。前者是作用域，后两者是循环，`ForKind` 也不同。这里循环归为虚线框，作用域改用第 6 组的角标。叠影层沿用 lg-ghost。',
      g: () => [G.loop({ x: 8, y: 10, w: 250, h: 52, kind: 'sequential', label: 'FOR fa_w' })
        + G.loop({ x: 8, y: 74, w: 250, h: 52, kind: 'parallel', label: 'FOR branch' })
        + G.loop({ x: 8, y: 138, w: 250, h: 52, kind: 'unroll', label: 'FOR i' })
        + G.loop({ x: 8, y: 202, w: 250, h: 56, kind: 'pipeline', stage: 4, label: 'FOR gp' }), 290, 274]
    },
    {
      group: '5', name: '循环携带', api: 'LogicGlyphs.loop · carry',
      what: 'IterArg + yield_ 的回边。绕到框底再折回开头，和前向数据流形状不同。',
      split: ['跨迭代的回边', '同一次迭代内的数据流 —— 回边才是 init_values 存在的理由'],
      origin: 'new', note: 'fa_fused 本身没有循环携带，所以 path.html 没画。换到在线 softmax 的累加、或手工依赖的 prev_tid 就必须有。',
      g: () => [G.loop({ x: 8, y: 10, w: 258, h: 92, kind: 'sequential', label: 'FOR i', meta: 'init_values=(acc,)', carry: true })
        + G.op({ x: 40, y: 42, w: 150, h: 28, family: 'elementwise', label: 'acc + chunk' }), 282, 126]
    },
    // ───────── 6 作用域 ─────────
    {
      group: '6', name: '执行作用域 · 并行映射', api: 'LogicGlyphs.scope',
      what: '四角实线角标 + 层级标签。覆盖 pl.at(CORE_GROUP)、pl.spmd(N)、pl.cluster、HOST、split_aiv、manual_scope 六种 ScopeStmt。',
      split: ['作用域（一次执行，不是控制流）', '循环（会重复） —— 线型就是这个分界'],
      origin: 'iterate', note: '从 path.html 的虚线框拆出来。IR 里 ScopeStmt 对 SSA 透明、不是控制流，和 ForStmt 是两类节点，图元也该分开。',
      g: () => [G.scope({ x: 8, y: 8, w: 268, h: 112, level: 'PL.SPMD × 24', meta: '每个 block 跑一份', split: 'SPLIT · UP_DOWN' })
        + G.scope({ x: 24, y: 40, w: 236, h: 60, level: 'PL.AT · CORE_GROUP', meta: '外提成 InCore' }), 292, 136]
    },
    // ───────── 7 任务与同步 ─────────
    {
      group: '7', name: '任务 · TaskId 端口', api: 'LogicGlyphs.task',
      what: '一次派发（Submit）。右侧端口是 TaskId，是 deps= 唯一能接的位置。',
      split: ['函数定义（第 1 组）', '它的这一次派发 —— 一个定义可以派发多次'],
      origin: 'new', note: 'path.html 的第 732 行写了 deps=[work_tid, rope_grp_tids[0]]，逻辑图里没有对应图元。端口让依赖边有地方落。',
      g: () => [G.task({ x: 8, y: 14, w: 196, kind: 'SUBMIT', label: 'fa_fused', tid: 'fa_tid' })
        + G.task({ x: 8, y: 72, w: 196, kind: 'SPMD_SUBMIT', label: 'rope_grp', tid: 'rope_tid' }), 222, 126]
    },
    {
      group: '7', name: '依赖汇合 / 栅栏', api: 'LogicGlyphs.barrier',
      what: 'task_dummy 或 phase fence：多入一出的扇入点，自己不提交 kernel。',
      split: ['扇入点（不跑计算）', '一个任务 —— 画成同形会让人以为多了一次派发'],
      origin: 'new', note: '新增。手工依赖里的 fan-in 没有图元就只能画成一堆交叉线。',
      g: () => [G.barrier({ x: 14, y: 20, w: 210, fan: 3, label: 'task_dummy · 不提交 kernel' }), 240, 84]
    },
    {
      group: '7', name: '跨核管道', api: 'LogicGlyphs.pipeChannel',
      what: 'AIC 与 AIV 之间的 tpush / tpop / tfree。中线是核边界，格子是 slot —— slot 数 × slot 大小从 Vec 预算里扣。',
      split: ['跨核交接（要握手、要槽位）', '核内的数据流 —— 前者占预算，后者不占'],
      origin: 'new', note: 'path.html 里 split 是 hidden 节点，只连到硬件映射的通道上；逻辑图里 qk→scale、cast→sv 这两处其实就是跨核交接，没有画出来。',
      g: () => [G.pipeChannel({ x: 12, y: 16, w: 240, h: 62, slots: 4, left: 'AIV', right: 'AIC', label: 'tpush_to_aic → tpop_from_aiv' }), 268, 96]
    },
    // ───────── 8 编译期 ─────────
    {
      group: '8', name: '编译期辅助', api: 'LogicGlyphs.staticTag',
      what: 'static_print、static_assert、dump_tag：解析期消费，不生成 IR。整体压暗、虚线边。',
      split: ['解析期的断言与标记', '运行期真的执行的东西 —— 前者在 IR 里查不到'],
      origin: 'new', note: '新增。放进逻辑图时必须和真节点有明显权重差，否则会被读成一步计算。',
      g: () => [G.staticTag({ x: 8, y: 10, w: 220, label: 'static_assert' })
        + G.staticTag({ x: 8, y: 46, w: 220, label: 'static_print' })
        + G.staticTag({ x: 8, y: 82, w: 220, label: 'dump_tag' }), 240, 120]
    },
    // ───────── S 存储 ─────────
    {
      group: 'S', name: '存储空间', api: 'LogicGlyphs.memory',
      what: '一个存储面：DDR、Vec、Mat、Left、Right、Acc、Bias。空心网格 —— 和数据对象的实心网格同形、填充相反。',
      split: ['存储面（容量）', '放在里面的数据（占用） —— 同形不同填充就是这个对照'],
      origin: 'reuse', note: '直接沿用 PathGridGlyphs.memory 的空心网格。实心/空心这组对照本来就是它和 tensor 的设计分工。',
      g: () => [G.memory({ x: 8, y: 12, w: 216, h: 62, label: 'Vec (UB)', detail: '192 KB 名义' })
        + G.memory({ x: 8, y: 86, w: 216, h: 62, label: 'Right (L0B)', detail: '64 KB' }), 240, 162]
    },
    {
      group: 'S', name: 'MemRef · 地址区间', api: 'LogicGlyphs.memref',
      what: '一段地址。AllocateMemoryAddr 之后才有确定的值；之前只有 size。',
      split: ['逻辑上的一块数据', '它拿到的地址区间 —— 多个视图可以指向同一段'],
      origin: 'new', note: 'path.html 的硬件映射里有一条按比例的 Vec 地址条，但那是整条 UB 的占用图，不是单个 MemRef 的身份。',
      g: () => [G.memref({ x: 12, y: 18, w: 230, label: 'MemRef · scores', span: [0.22, 0.52], lo: '0x1a000', hi: '+8 KB' }), 258, 80]
    },
    // ───────── R 关系 ─────────
    ...[
      ['contain', '包含 / 嵌套', '虚线框本身就是这个关系：Program ⊃ Function ⊃ Scope ⊃ Loop ⊃ Stmt。', ['嵌套（谁在谁里面）', '先后（谁在谁之后）'], 'reuse', 'path.html 的虚线框已经承担了这个关系。'],
      ['data', '数据流（定义 → 使用）', '开口箭头，从生产者指向消费者。线长不编码耗时。', ['数据依赖', '执行顺序 —— 并行循环里两者不等价'], 'reuse', '沿用 lg-edge 与 lgArrow 箭头。'],
      ['carry', '循环携带', '回边：绕到下方再折回。lavender 和前向流区分。', ['跨迭代的回边', '同迭代内的前向流'], 'new', '新增。'],
      ['alias', '视图 / 别名', '点线 + 圆端点，不带箭头 —— 没有方向，也没有拷贝。', ['指向同一块存储', '拷了一份出来'], 'new', '圆端点沿用 VisualSystem 的 binding 关系画法。'],
      ['index', '索引 / 寻址', '细点线，从标量指向它参与的偏移。包括 block_table 这种间接寻址。', ['索引值的来路', '被索引的数据本身'], 'new', 'path.html 画了 block_table 节点，但它和下游节点之间是普通数据流线。'],
      ['place', '存储放置', '长点线 + 空心菱形，从数据指向它落的存储面或地址。', ['逻辑上的数据', '物理上的落点 —— 后者是编译器决定的'], 'new', '菱形沿用 VisualSystem 的 mapping 关系画法；这里专指"落在哪"。'],
      ['call', '调用 / 派发', '长虚线箭头。同步调用、异步 submit、SPMD 一次派发 N 份都走这条。', ['调用方指向定义', '数据从一处流到另一处'], 'new', 'path.html 的逻辑图里没有调用关系。'],
      ['dep', '任务依赖', '双线箭头 —— 读作"等它完成"，不是"接它的数据"。双线是为了和单线的数据流在线型上区分，不靠颜色。', ['等它完成（顺序）', '要它的数据（数据流）—— deps= 只管前者'], 'new', '新增。path.html 第 732 行的 deps= 在逻辑图里没有对应画法。'],
      ['xcore', '跨核通信', 'sky 色箭头，必定穿过核边界线。', ['跨核交接（要槽位）', '核内数据流'], 'new', '新增，和第 7 组的管道图元配套。']
    ].map(([kind, name, what, split, origin, note]) => ({
      group: 'R', name, api: `LogicGlyphs.relation('${kind}', …)`, what, split, origin, note,
      g: () => {
        const a = G.rect(8, 26, 54, 26, 'gl-op', 13) + G.text(35, 43, 'A', 'gl-node-text', 'middle');
        const b = G.rect(188, 26, 54, 26, 'gl-op', 13) + G.text(215, 43, 'B', 'gl-node-text', 'middle');
        const line = kind === 'carry' ? G.backEdge(188, 39, 62, 39) : G.relation(kind, 62, 39, 188, 39);
        const extra = kind === 'xcore' ? G.path('M125 10v58', 'gl-core-split') : '';
        return [a + extra + line + b, 250, 78];
      }
    }))
  ];

  // ── 渲染 ──
  function card(c) {
    const [body, w, h] = c.g();
    const label = c.name + ' 图元';
    const originText = { reuse: '复用', iterate: '迭代', new: '新增' }[c.origin];
    return `<article class="gl-card" data-origin="${c.origin}">
      <div class="gl-card-canvas">${G.wrap(body, w, h, label)}</div>
      <div class="gl-card-body">
        <div class="gl-card-head"><h3>${c.name}</h3><span class="gl-tag" data-origin="${c.origin}">${originText}</span></div>
        <p class="gl-api">${c.api}</p>
        <p>${c.what}</p>
        <p class="gl-split"><b>必须区分开：</b>${c.split[0]} ／ ${c.split[1]}</p>
        <p class="gl-origin-note">${c.note}</p>
      </div>
    </article>`;
  }

  function render() {
    const host = document.getElementById('glCatalog');
    host.innerHTML = GROUPS.map(([num, title, blurb, owner]) => {
      const mine = CARDS.filter((c) => c.group === num);
      if (!mine.length) return '';
      return `<section class="gl-group" id="g-${num}" data-group="${num}">
        <div class="gl-group-head">
          <span class="gl-group-no">${num === 'S' ? '存储' : num === 'R' ? '关系' : num}</span>
          <h2>${title}</h2><p>${blurb}</p><span class="gl-owner">${owner}</span>
        </div>
        <div class="gl-grid">${mine.map(card).join('')}</div>
      </section>`;
    }).join('');

    const counts = { reuse: 0, iterate: 0, new: 0 };
    CARDS.forEach((c) => { counts[c.origin]++; });
    document.getElementById('glCount').textContent =
      `${CARDS.length} 个图元 · 复用 ${counts.reuse} · 迭代 ${counts.iterate} · 新增 ${counts.new}`;

    // 来源筛选：按复用 / 迭代 / 新增过滤，空分组整体隐藏
    const buttons = Array.prototype.slice.call(document.querySelectorAll('.gl-filter'));
    buttons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const want = btn.dataset.filter;
        buttons.forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
        document.querySelectorAll('.gl-card').forEach((el) => {
          el.hidden = want !== 'all' && el.dataset.origin !== want;
        });
        document.querySelectorAll('.gl-group').forEach((sec) => {
          sec.hidden = !sec.querySelector('.gl-card:not([hidden])');
        });
      });
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', render);
  else render();
})();
