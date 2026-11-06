/* ──────────────────────────────────────────────────────────────────────────────
 * PyPTO 算子逻辑图元库
 *
 * 覆盖一个算子的 8 类逻辑表达 + 存储放置 + 9 种关系。分类依据 PyPTO 文档：
 *   repo/pto/docs/en/user/01-language_guide.md、02-operation_reference.md
 *   repo/pto/docs/en/dev/language/00-python_syntax.md
 *   repo/pto/docs/en/dev/ir/01-hierarchy.md、02-types.md
 *
 * 复用：PtoOpIcons 的 matmul / reduce / expand / activation 四个图标路径；
 *       path.css 的色板与 lg-* / sv-* 的视觉约定（虚线=作用域、实框=计算段）。
 * 迭代：算子节点的图标按 DSL 操作族归类，不再借用模型层语义；
 *       循环与作用域拆成两套框；视图类操作不再画成计算节点。
 * 新增：搬运、视图/有效区、任务与依赖、跨核管道、编译期辅助、MemRef 地址条。
 *
 * 颜色分工（单一告警红不在本库内使用）：
 *   数据实心角色色 —— GM/DDR lime · Vec sky · Mat/Left/Right lav · Acc amber · 标量灰
 *   结构描边 —— 函数/作用域/循环/任务统一用淡紫灰描边，不与数据争色
 * 所有填色走 glyphs.css 的 class，便于整页改主题。
 * ────────────────────────────────────────────────────────────────────────────── */
(function (root) {
  'use strict';

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const T = (x, y, s, cls, anchor) => `<text x="${x}" y="${y}" class="${cls || 'gl-label'}"${anchor ? ` text-anchor="${anchor}"` : ''}>${esc(s)}</text>`;
  const R = (x, y, w, h, cls, rx) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx == null ? 4 : rx}" class="${cls}"/>`;
  const P = (d, cls) => `<path d="${d}" class="${cls}"/>`;

  // ── 操作族 → 图标；四个来自 PtoOpIcons，三个本库新增 ──
  const FAMILIES = {
    elementwise: { label: '逐元素', path: 'M4 6h4v4H4z M10 6h4v4h-4z M16 6h4v4h-4z M4 14h4v4H4z M10 14h4v4h-4z M16 14h4v4h-4z' },
    unary: { label: '一元 / 激活', reuse: 'activation' },
    reduce: { label: '归约', reuse: 'reduce' },
    broadcast: { label: '广播', reuse: 'expand' },
    matmul: { label: '矩阵', reuse: 'matmul' },
    cast: { label: '类型转换', path: 'M4 6h11 M4 6v5 M4 11h11 M7 15h6 M7 15v4 M7 19h6 M17 9l4 4-4 4' },
    compare: { label: '比较 / 选择', path: 'M4 8h6 M4 16h6 M10 8l6 4-6 4 M16 12h4' }
  };
  function famIcon(family) {
    const d = FAMILIES[family];
    if (!d) return null;
    if (d.path) return d.path;
    const icon = (root.PtoOpIcons || []).filter((i) => i.id === d.reuse)[0];
    return icon ? icon.path : null;
  }
  function icon(d, x, y, size, cls) {
    if (!d) return '';
    return `<path d="${d}" class="${cls || 'gl-icon'}" transform="translate(${x} ${y}) scale(${size / 24})"/>`;
  }

  // ── 网格纹理：只表达"这是一片数据/一块存储"，格子数不编码数量 ──
  function cells(x, y, w, h, cls, rows, pad, gap) {
    pad = pad == null ? 4 : pad; gap = gap == null ? 1.6 : gap; rows = rows || 4;
    const cell = (h - pad * 2 - gap * (rows - 1)) / rows;
    const cols = Math.max(1, Math.floor((w - pad * 2 + gap) / (cell + gap)));
    const gx = cols > 1 ? (w - pad * 2 - cols * cell) / (cols - 1) : 0;
    let s = '';
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        s += `<rect x="${(x + pad + c * (cell + gx)).toFixed(1)}" y="${(y + pad + r * (cell + gap)).toFixed(1)}" width="${cell.toFixed(1)}" height="${cell.toFixed(1)}" class="${cls}"/>`;
      }
    }
    return s;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 1 程序 / 函数边界
  // ════════════════════════════════════════════════════════════════════════════
  // 名称栏 + 函数种类 chip + 参数方向针脚。种类取 IR 的 FunctionType。
  function fnBox(o) {
    const x = o.x || 0, y = o.y || 0, w = o.w || 210, h = o.h || 84;
    let s = R(x, y, w, h, 'gl-fn', 6) + P(`M${x} ${y + 26}h${w}`, 'gl-fn-rule');
    s += T(x + 10, y + 17, o.kind || 'INCORE', 'gl-chip');
    s += T(x + w - 10, y + 17, o.name || 'fa_fused', 'gl-name', 'end');
    (o.params || []).forEach((p, i) => {
      const py = y + 44 + i * 17;
      s += `<circle cx="${x + 14}" cy="${py - 4}" r="3.5" class="gl-pin gl-pin-${p.dir.toLowerCase()}"/>`;
      s += T(x + 25, py, p.name, 'gl-small') + T(x + w - 10, py, p.dir, 'gl-dim', 'end');
    });
    return `<g data-glyph="fn" data-kind="${esc(o.kind || '')}">${s}</g>`;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 2 数据对象
  // ════════════════════════════════════════════════════════════════════════════
  // 实心角色色 + 网格纹理。space 决定角色色；valid 给出有效区占比（shape vs valid_shape）。
  // 纹理画法同 PathGridGlyphs.tensor：底色 → 压暗的内面板 → 满色格子，
  // 格子才有对比度。只在底色上叠半透明同色格子是看不见的。
  function dataObject(o) {
    const x = o.x || 0, y = o.y || 0, w = o.w || 150, h = o.h || 56;
    const role = 'gl-role-' + (o.space || 'gm');
    const gx = x + 2, gy = y + 18, gw = w - 4, gh = h - 20;
    let s = R(x, y, w, h, role + ' gl-data', 5);
    s += R(gx, gy, gw, gh, 'gl-data-panel', 3);
    s += cells(gx, gy, gw, gh, role + ' gl-data-cell');
    if (o.valid) {
      // 压暗的是「无效区」，亮的那块才是有效区；虚线框住有效区的边界
      const vw = Math.max(12, gw * o.valid[0]), vh = Math.max(8, gh * o.valid[1]);
      s += R(gx + vw, gy, gw - vw, gh, 'gl-invalid', 0);
      if (vh < gh) s += R(gx, gy + vh, vw, gh - vh, 'gl-invalid', 0);
      s += R(gx, gy, vw, vh, 'gl-valid-edge', 2);
      s += T(x + w - 7, y + h - 6, '有效区', 'gl-dim', 'end');
    }
    s += T(x + 7, y + 13, o.label || 'scores', 'gl-onrole-strong');
    s += T(x + w - 7, y + 13, (o.shape || '[16, 128]') + (o.dtype ? ' · ' + o.dtype : ''), 'gl-onrole', 'end');
    return `<g data-glyph="data" data-space="${esc(o.space || 'gm')}">${s}</g>`;
  }
  // 标量：索引、长度、TaskId 都是标量，用小胶囊，和一片数据区分开
  function scalar(o) {
    const x = o.x || 0, y = o.y || 0, w = o.w || 108, h = 24;
    return `<g data-glyph="scalar">${R(x, y, w, h, 'gl-scalar', 12)}${T(x + 11, y + 15, o.label || 'valid_len', 'gl-small')}${T(x + w - 11, y + 15, o.dtype || 'INDEX', 'gl-dim', 'end')}</g>`;
  }
  // 片上定长数组：落在标量寄存器/C 栈，不是 Tile，用一行独立格子
  function array(o) {
    const x = o.x || 0, y = o.y || 0, n = o.n || 6, cw = 16, h = 24;
    let s = T(x, y - 5, (o.label || 'tids') + ' · Array[' + n + ', ' + (o.dtype || 'TASK_ID') + ']', 'gl-small');
    for (let i = 0; i < n; i++) s += R(x + i * (cw + 3), y, cw, h, 'gl-arr-cell', 2);
    return `<g data-glyph="array">${s}</g>`;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 3 计算
  // ════════════════════════════════════════════════════════════════════════════
  // 圆端胶囊 + 操作族图标。算子名写全，形状交给数据对象，不在节点里重复。
  function op(o) {
    const x = o.x || 0, y = o.y || 0, w = o.w || 168, h = o.h || 34;
    const d = famIcon(o.family), sz = h * 0.56, ix = x + h * 0.2, iy = y + (h - sz) / 2;
    let s = R(x, y, w, h, 'gl-op', h / 2);
    if (d) s += `<circle cx="${ix + sz / 2}" cy="${y + h / 2}" r="${sz / 2 + 2}" class="gl-op-badge"/>` + icon(d, ix, iy, sz);
    s += T(x + (d ? h * 0.92 : w / 2), y + h / 2 + 4, o.label || 'row_max', 'gl-node-text', d ? null : 'middle');
    return `<g data-glyph="op" data-family="${esc(o.family || '')}">${s}</g>`;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 4 搬运 / 形状 / 视图
  // ════════════════════════════════════════════════════════════════════════════
  // 搬运：方角胶囊 + 源/目的存储空间，和圆端的计算节点一眼区分。
  function move(o) {
    const x = o.x || 0, y = o.y || 0, w = o.w || 168, h = o.h || 34;
    let s = R(x, y, w, h, 'gl-move', 3);
    s += P(`M${x + 12} ${y + h / 2}h10 M${x + 18} ${y + h / 2 - 4}l4 4-4 4`, 'gl-move-arrow');
    s += T(x + 30, y + h / 2 - 1, o.label || 'load', 'gl-node-text');
    s += T(x + w - 10, y + h / 2 - 1, (o.from || 'DDR') + ' → ' + (o.to || 'Vec'), 'gl-dim', 'end');
    return `<g data-glyph="move">${s}</g>`;
  }
  // 视图 / 有效区：只改元数据、不搬数据。空心方括号，内部不填色。
  function viewOp(o) {
    const x = o.x || 0, y = o.y || 0, w = o.w || 168, h = o.h || 34;
    let s = R(x, y, w, h, 'gl-view', 3);
    s += P(`M${x + 9} ${y + 8}h-4v${h - 16}h4 M${x + w - 9} ${y + 8}h4v${h - 16}h-4`, 'gl-view-bracket');
    s += T(x + 20, y + h / 2 - 1, o.label || 'set_validshape', 'gl-node-text');
    // note 传空字符串 = 不要注脚（窄容器里用）；不传则给默认值
    const note = o.note === undefined ? '仅元数据' : o.note;
    if (note) s += T(x + w - 14, y + h / 2 - 1, note, 'gl-dim', 'end');
    return `<g data-glyph="view">${s}</g>`;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 5 控制结构
  // ════════════════════════════════════════════════════════════════════════════
  // 虚线框 + 左侧粗条。粗条的条数/形态区分 ForKind：顺序 1 条、并行 3 条、
  // 展开为分段、流水为左条 + 叠影层（叠影层数 = stage）。
  function loop(o) {
    const x = o.x || 0, y = o.y || 0, w = o.w || 230, h = o.h || 86, kind = o.kind || 'sequential';
    const stage = o.stage || 0;
    let s = '';
    if (kind === 'pipeline') {
      for (let i = Math.max(0, stage - 1); i >= 1; i--) s += R(x + i * 5, y + i * 5, w - 10, h - 10, 'gl-ghost', 4);
    }
    s += R(x, y, w, h, 'gl-loop', 4);
    const bars = kind === 'parallel' ? [0, 5, 10] : [0];
    bars.forEach((d) => { s += P(`M${x + 2 + d} ${y + 3}v${h - 6}`, 'gl-loop-bar'); });
    if (kind === 'unroll') s += P(`M${x + 2} ${y + 3}v${(h - 6) / 3} M${x + 2} ${y + 3 + (h - 6) / 3 * 1.2}v${(h - 6) / 3 * 0.6} M${x + 2} ${y + h - 3 - (h - 6) / 3}v${(h - 6) / 3}`, 'gl-loop-bar');
    s += T(x + 18, y + 17, o.label || 'FOR gp', 'gl-title');
    s += T(x + w - 10, y + 17, o.meta || { sequential: 'pl.range · 顺序', parallel: 'pl.parallel · 可并行', unroll: 'pl.unroll · 编译期展开', pipeline: 'pl.pipeline · stage=' + stage }[kind], 'gl-dim', 'end');
    if (o.carry) s += backEdge(x + w - 26, y + h - 26, x + 22, y + h - 26, '循环携带');
    return `<g data-glyph="loop" data-kind="${esc(kind)}">${s}</g>`;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 6 执行作用域 / 并行映射
  // ════════════════════════════════════════════════════════════════════════════
  // 四角实线角标 + 层级 chip。实线角标是作用域，虚线整框是循环，两者不混用。
  function scope(o) {
    const x = o.x || 0, y = o.y || 0, w = o.w || 250, h = o.h || 96, c = 16;
    let s = R(x, y, w, h, 'gl-scope-fill', 5);
    s += P(`M${x} ${y + c}V${y}h${c} M${x + w - c} ${y}h${c}v${c} M${x + w} ${y + h - c}v${c}h-${c} M${x + c} ${y + h}h-${c}v-${c}`, 'gl-scope-corner');
    s += T(x + 12, y + 19, o.level || 'PL.SPMD × 24', 'gl-title');
    s += T(x + w - 12, y + 19, o.meta || '每个 block 跑一份', 'gl-dim', 'end');
    // 角标占住四角各 16px，split chip 要让开左下角
    if (o.split) s += T(x + c + 14, y + h - 11, o.split, 'gl-chip');
    return `<g data-glyph="scope" data-level="${esc(o.level || '')}">${s}</g>`;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 7 任务依赖与同步
  // ════════════════════════════════════════════════════════════════════════════
  // 任务：圆角卡片 + 右侧 TaskId 端口。端口是 deps= 能接的唯一位置。
  function task(o) {
    const x = o.x || 0, y = o.y || 0, w = o.w || 176, h = o.h || 40;
    let s = R(x, y, w, h, 'gl-task', 8);
    s += T(x + 12, y + 16, o.kind || 'SUBMIT', 'gl-chip');
    s += T(x + 12, y + 32, o.label || 'fa_fused', 'gl-node-text');
    s += `<circle cx="${x + w}" cy="${y + h / 2}" r="5" class="gl-port"/>`;
    s += T(x + w - 12, y + h - 8, o.tid || 'tid', 'gl-dim', 'end');
    return `<g data-glyph="task">${s}</g>`;
  }
  // 依赖汇合：task_dummy / phase fence。多入一出的扇入点，自己不跑 kernel。
  function barrier(o) {
    const x = o.x || 0, y = o.y || 0, w = o.w || 176, fan = o.fan || 3;
    let s = R(x, y + 10, w, 12, 'gl-barrier', 6);
    for (let i = 0; i < fan; i++) {
      const bx = x + 20 + i * ((w - 40) / Math.max(1, fan - 1));
      s += P(`M${bx} ${y}v10`, 'gl-dep-edge');
    }
    s += P(`M${x + w / 2} ${y + 22}v10`, 'gl-dep-edge');
    s += T(x, y + 44, o.label || 'task_dummy · 不提交 kernel', 'gl-small');
    return `<g data-glyph="barrier">${s}</g>`;
  }
  // 跨核管道：AIC / AIV 之间的 tpush / tpop / tfree，槽位是预算，从 Vec 里扣。
  function pipeChannel(o) {
    const x = o.x || 0, y = o.y || 0, w = o.w || 210, h = o.h || 58, slots = o.slots || 4;
    let s = T(x, y + 10, o.left || 'AIV', 'gl-small') + T(x + w, y + 10, o.right || 'AIC', 'gl-small', 'end');
    s += P(`M${x + w / 2} ${y}v${h}`, 'gl-core-split');
    const sw = (w - 24) / slots;
    for (let i = 0; i < slots; i++) s += R(x + 12 + i * sw, y + 20, sw - 4, 14, 'gl-slot', 2);
    s += P(`M${x + 12} ${y + 46}h${w - 24} M${x + w - 24} ${y + 42}l4 4-4 4`, 'gl-pipe-arrow');
    s += T(x, y + h, (o.label || 'tpush_to_aic → tpop_from_aiv') + ' · ' + slots + ' 槽', 'gl-dim');
    return `<g data-glyph="pipe">${s}</g>`;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 8 编译期辅助（不生成 IR）
  // ════════════════════════════════════════════════════════════════════════════
  function staticTag(o) {
    const x = o.x || 0, y = o.y || 0, w = o.w || 190, h = 26;
    let s = R(x, y, w, h, 'gl-static', 13);
    s += P(`M${x + 12} ${y + 8}l5 5-5 5`, 'gl-static-mark');
    s += T(x + 24, y + 17, o.label || 'static_assert', 'gl-dim');
    s += T(x + w - 11, y + 17, '不生成 IR', 'gl-dim', 'end');
    return `<g data-glyph="static">${s}</g>`;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 存储放置（由编译器决定，不是用户写的）
  // ════════════════════════════════════════════════════════════════════════════
  // 空心网格 = 存储面；实心网格 = 数据。两者纹理同形、填充相反。
  function memory(o) {
    const x = o.x || 0, y = o.y || 0, w = o.w || 190, h = o.h || 58;
    let s = R(x, y, w, h, 'gl-mem', 6) + cells(x, y + 20, w, h - 20, 'gl-mem-cell', 3);
    s += T(x + 9, y + 14, o.label || 'Vec (UB)', 'gl-strong');
    s += T(x + w - 9, y + 14, o.detail || '192 KB 名义', 'gl-dim', 'end');
    return `<g data-glyph="memory">${s}</g>`;
  }
  // MemRef：地址区间。一条地址条 + 占位块，地址是编译器分配的结果。
  function memref(o) {
    const x = o.x || 0, y = o.y || 0, w = o.w || 210, h = 30;
    const seg = o.span || [0.18, 0.46];
    let s = R(x, y + 8, w, h - 8, 'gl-addr', 2);
    s += R(x + w * seg[0], y + 9, w * (seg[1] - seg[0]), h - 10, 'gl-addr-block', 1);
    s += T(x, y + 4, o.label || 'MemRef · scores', 'gl-small');
    s += T(x + w * seg[0], y + h + 11, o.lo || '0x1000', 'gl-dim', 'middle');
    s += T(x + w * seg[1], y + h + 11, o.hi || '+8 KB', 'gl-dim', 'middle');
    return `<g data-glyph="memref">${s}</g>`;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 9 种关系
  // ════════════════════════════════════════════════════════════════════════════
  const relStyle = {
    contain: { cls: 'gl-rel-contain', marker: null },
    data: { cls: 'gl-rel-data', marker: 'arrow' },
    carry: { cls: 'gl-rel-carry', marker: 'arrow' },
    alias: { cls: 'gl-rel-alias', marker: 'dot' },
    index: { cls: 'gl-rel-index', marker: 'arrow' },
    place: { cls: 'gl-rel-place', marker: 'diamond' },
    call: { cls: 'gl-rel-call', marker: 'arrow' },
    dep: { cls: 'gl-rel-dep', marker: 'tick' },
    xcore: { cls: 'gl-rel-xcore', marker: 'arrow' }
  };
  function relation(kind, x1, y1, x2, y2, label) {
    const st = relStyle[kind] || relStyle.data;
    const mid = (x1 + x2) / 2;
    const leg = (dy) => (y1 === y2 ? `M${x1} ${y1 + dy}H${x2}` : `M${x1} ${y1 + dy}H${mid}V${y2 + dy}H${x2}`);
    let s;
    if (kind === 'dep') {
      // 任务依赖画成双线：和数据流的单线在线型上就不同，不靠颜色区分
      s = `<path d="${leg(-1.9)}" class="${st.cls}"/><path d="${leg(1.9)}" class="${st.cls}" marker-end="url(#gl-${st.marker})"/>`;
    } else {
      s = `<path d="${leg(0)}" class="${st.cls}"${st.marker ? ` marker-end="url(#gl-${st.marker})"` : ''}/>`;
    }
    if (label) s += T(mid, Math.min(y1, y2) - 7, label, 'gl-dim', 'middle');
    return `<g data-relation="${esc(kind)}">${s}</g>`;
  }
  // 循环携带是唯一的回边：绕到下方再回头，和前向数据流形状不同。
  function backEdge(x1, y1, x2, y2, label) {
    const dy = 14;
    let s = `<path d="M${x1} ${y1}v${dy}H${x2}V${y2}" class="gl-rel-carry" marker-end="url(#gl-arrow)"/>`;
    if (label) s += T((x1 + x2) / 2, y1 + dy + 12, label, 'gl-dim', 'middle');
    return `<g data-relation="carry">${s}</g>`;
  }

  const DEFS = '<defs>'
    + '<marker id="gl-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" markerUnits="userSpaceOnUse" orient="auto"><path d="M1 1 L7 4 L1 7" class="gl-marker"/></marker>'
    + '<marker id="gl-dot" viewBox="0 0 8 8" refX="4" refY="4" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse"><circle cx="4" cy="4" r="3" class="gl-marker-fill"/></marker>'
    + '<marker id="gl-diamond" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto"><path d="M1 5L5 1L9 5L5 9Z" class="gl-marker"/></marker>'
    + '<marker id="gl-tick" viewBox="0 0 8 10" refX="7" refY="5" markerWidth="6" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto"><path d="M1 1 L7 5 L1 9 M4 1v8" class="gl-marker"/></marker>'
    + '</defs>';

  function wrap(body, w, h, label) {
    return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="${esc(label || '图元')}" class="gl-svg">${DEFS}${body}</svg>`;
  }

  root.LogicGlyphs = {
    FAMILIES, famIcon, wrap, cells,
    fnBox, dataObject, scalar, array,
    op, move, viewOp,
    loop, scope,
    task, barrier, pipeChannel,
    staticTag,
    memory, memref,
    relation, backEdge,
    text: T, rect: R, path: P
  };
})(window);
