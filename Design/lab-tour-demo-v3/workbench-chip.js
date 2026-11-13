/* 硅片落点图（#hpSvg，lens-tile.js 渲染）在本 Demo 里的重排与整理，lens-tile.js 不改。
 * lens-tile.js 每次重画（切 stage、UP_DOWN 推算等）都整块替换 SVG，这里画完后用同一批图元重排，联动（.sv-region / .sv-buf）不动。
 *
 * 布局参照昇腾 AI Core 结构图：最左一列 GM；右边 BLOCK 0 —— 上面一行 AIC，下面一行 AIV。
 *   AIC 从左往右：L1（竖条，高地址在上：Q / KV / v2c）→ L0A（Q）/ L0B（KV）→ CUBE → L0C
 *   AIV 从左往右：C2V / V2C 管道 → AIV0 → AIV1；Vec 柱改为横条（同结构图里的 Unified Buffer），地址从左往右
 *   写回：L0C（oi）与 AIV0（mi / li）汇到 AIC 与 AIV 之间的横道，回到 GM 的 mi / li / oi
 * 其余整理：AIV1 斜纹空心与琥珀虚线、默认不点亮、184 线中性灰（workbench-polish.css 第 11 节）；
 *   同一对象同一图元（第 13 节 + unify）；次要说明悬停 / 点击 / “全部标注”显示（第 12 节 + layer）。 */
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const host = document.getElementById('hpSvg');
  if (!host) return;
  const el = function (tag, attrs, parent, before) {
    const e = document.createElementNS(NS, tag);
    Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (parent) parent.insertBefore(e, before || null);
    return e;
  };
  const num = function (e, a) { return Number(e.getAttribute(a)) || 0; };
  const set = function (e, attrs) { Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); }); };
  const region = function (svg, id) { return svg.querySelector('.sv-region[data-region="' + id + '"]'); };
  const textOf = function (svg, re) { return [].filter.call(svg.children, function (c) { return c.tagName === 'text' && re.test(c.textContent.trim()); })[0]; };

  // ── 画布与各块位置（全图坐标）：最左一列 GM；右边三行 —— AIV0 横条 / AIC / AIV1 横条（两个 AIV 夹着 AIC） ──
  //   外边距 12；GM 列与 BLOCK 框上下沿对齐；三行左右边对齐、行间距 14、内边距 24。
  //   连线优先“直连”：每个对象摆在它上下游的正对面，连线尽量是一条直线 ——
  //     GM 卡片与目标同高（qk_order → Vec 横条、q → L1 Q 段、ori / cmp → L1 KV 段、写回卡片 = 写回横道）；
  //     L1 低地址在上：v2c 槽在顶端，正对 AIV0 UB 的 softmax 工作区（V2C 一条竖线）；
  //     L0B / L0A 与 L1 的 KV / Q 段同高，CUBE 同时盖住两条输入线，L0C 与 CUBE 同中线；
  //     L0C 与 AIV0 / AIV1 的 Vector 列上下对齐：C2V 一条竖线向上，AIV1 的握手虚线一条竖线向下。
  const VB_W = 1100, VB_H = 440;
  const FRAME = { x: 190, y: 36, w: 898, h: 384 };
  const GM = { x: 12, y: FRAME.y, w: 158, h: FRAME.h, cardX: 23, cardW: 136, cardH: 30 };   // 张量图元高 30（同计算图张量的比例）
  const ROWX = FRAME.x + 12, ROWW = FRAME.w - 24;                       // 三行共用的左沿与宽度
  const AIV_H = 78, GAP = 14;
  const AIV_Y = [FRAME.y + 12, FRAME.y + 12 + AIV_H + GAP + 178 + GAP]; // AIV0 在上、AIV1 在下
  const AIC = { x: ROWX, y: AIV_Y[0] + AIV_H + GAP, w: ROWW, h: 178 };
  const BAR = { dx: 16, dy: 30, w: 440, h: 30 };                        // Vec 横条（相对 AIV 横条框）
  const BAR_X = ROWX + 24, BAR_MID = AIV_Y[0] + BAR.dy + BAR.h / 2;    // Vec 横条左沿、AIV0 横条中线
  const VEC_X = BAR_X + BAR.w + 36;                                     // Vector 计算块左沿（宽 90）
  const V2C_X = BAR_X + 164;                                            // UB softmax 工作区中线（V2C 竖线）
  const L1 = { x: V2C_X - 28, y: AIC.y + 26, w: 56, h: 130 };          // 竖条：低地址在上，v2c 槽正对 V2C 竖线
  const Y_KV = AIC.y + 89, Y_Q = AIC.y + 145;                          // L1 KV 段 / Q 段里的两条水平线
  const COL = { l0: L1.x + L1.w + 40, cube: 618, l0c: 738 };            // 各列左沿
  const BOX = {                                                          // 目标位置与尺寸 [x, y, w, h]
    l0b: [COL.l0, Y_KV - 24, 130, 48], l0a: [COL.l0, Y_Q - 24, 130, 48],
    cube: [COL.cube, Y_KV - 18, 90, Y_Q - Y_KV + 36], l0c: [COL.l0c, (Y_KV + Y_Q) / 2 - 35, 210, 70]
  };
  const ROW = (Y_KV + Y_Q) / 2;                                         // CUBE → L0C 的水平中线
  const ORIG = { l0a: [36, 218], l0b: [128, 218], cube: [36, 284], l0c: [36, 356] };    // lens-tile.js 原坐标
  const LANE_OUT = AIV_Y[0] + AIV_H + GAP / 2;                          // AIV0 与 AIC 之间的写回横道
  const C2V_X = VEC_X + 83, OI_X = COL.l0c + 16;                        // C2V / 握手竖线（Vector 列内）、oi 写回竖线
  const PIPE_D = [C2V_X - 321, AIC.y + 36 - 302];                       // 管道组平移：两个槽骑在 C2V 竖线上
  const CORES_D = [FRAME.x + FRAME.w - 708, 8 - 86];                    // SPMD 条：右端对齐分组框，放在框上方
  // GM 卡片与目标同高（自上而下：AIV0 横条、写回横道、L1 KV 段 ×2、L1 Q 段），全部水平直连
  const GM_ORDER = ['gm-plan', 'gm-out', 'gm-ori', 'gm-cmp', 'gm-q'];
  const GM_Y = { 'gm-plan': BAR_MID, 'gm-out': LANE_OUT, 'gm-ori': Y_KV - 24, 'gm-cmp': Y_KV + 16, 'gm-q': Y_Q };
  const VBAR = { top: 186, bottom: 440 };                               // lens-tile.js 竖条的 y 范围
  const S = BAR.w / (VBAR.bottom - VBAR.top);
  const MAPX = function (x0, yy) { return x0 + (VBAR.bottom - yy) * S; };   // 竖条 y（地址）→ 横条 x

  function decorate() {
    const svg = host.querySelector(':scope > svg');
    if (!svg || svg.dataset.wb) return;
    svg.dataset.wb = '1';
    svg.classList.add('wb-chip');
    svg.setAttribute('viewBox', '0 0 ' + VB_W + ' ' + VB_H);
    const defs = svg.querySelector('defs') || el('defs', {}, svg, svg.firstChild);
    [['wbChipVecBoth', 'var(--sky-ink)'], ['wbChipQ', 'var(--lime)'], ['wbChipKv', 'var(--lav-ink)'], ['wbChipVec', 'var(--sky-ink)'], ['wbChipAcc', 'var(--amber)'], ['wbChipDim', '#92939a'], ['wbChipOut', '#c6c7cb']].forEach(function (m) {
      const mk = el('marker', { id: m[0], viewBox: '0 0 8 8', refX: '7', refY: '4', markerWidth: '6', markerHeight: '6', orient: m[0] === 'wbChipVecBoth' ? 'auto-start-reverse' : 'auto' }, defs);
      el('path', { d: 'M0 0 L8 4 L0 8 z', style: 'fill:context-stroke' }, mk);
    });

    // ── 先认出 AIV 的散落元素（参考线、刻度、说明文字），按原坐标归属 AIV0 / AIV1 ──
    const loose = { aiv0: [], aiv1: [] };
    [].slice.call(svg.children).forEach(function (c) {
      if (c.tagName !== 'text' && c.tagName !== 'line') return;
      const x = num(c, c.tagName === 'line' ? 'x1' : 'x'), y = num(c, c.tagName === 'line' ? 'y1' : 'y');
      if (y <= 140 || y >= 456) return;
      if (x >= 340 && x < 528) loose.aiv0.push(c); else if (x >= 528) loose.aiv1.push(c);
    });

    // ── GM：最左一列，与 BLOCK 框上下沿对齐；卡片统一宽度、等间距 ──
    const gmBg = [].filter.call(svg.children, function (c) { return c.tagName === 'rect' && c.getAttribute('fill') === '#282828'; })[0];
    const gmTitle = textOf(svg, /^GM · HBM/);
    if (gmBg) set(gmBg, { x: GM.x, y: GM.y, width: GM.w, height: GM.h, rx: 12 });
    if (gmTitle) set(gmTitle, { x: GM.x + 12, y: GM.y + 18 });
    const card = {};
    // GM 里的五份数据都是张量：换成和上方计算图 / Kernel 链完全相同的张量图元（PathGridGlyphs.tensor），
    // 颜色同上方图（Q 绿、KV 紫、其余张量用 Kernel 链里的张量蓝），名字与形状常驻
    const GM_META = {
      'gm-plan': ['qk_order / bias', '[QK_ITEMS] · INT32', '#80b7f7'],
      'gm-out': ['mi / li / oi', '部分结果 · FP32', '#80b7f7'],
      'gm-q': ['q', '[T, 64, 512] · BF16', '#88EC30'],
      'gm-ori': ['ori_kv · 滑窗', '分页 · 32 行/页', '#A459FF'],
      'gm-cmp': ['cmp_kv · 压缩', 'top-k 散读', '#A459FF']
    };
    GM_ORDER.forEach(function (id, i) {
      const g = [].filter.call(svg.querySelectorAll('.sv-region[data-region="' + id + '"]'), function (r) { return r.querySelector('svg'); })[0];
      if (!g) return;
      const inner = g.querySelector(':scope > g'), y = GM_Y[id] - GM.cardH / 2, m = GM_META[id];
      if (window.PathGridGlyphs && m) inner.innerHTML = window.PathGridGlyphs.tensor(id, m[0], m[1], 0, 0, GM.cardW, GM.cardH, m[2]);
      inner.setAttribute('transform', 'translate(' + GM.cardX + ' ' + y + ')');
      card[id] = { r: GM.cardX + GM.cardW, m: y + GM.cardH / 2 };
    });
    svg.querySelectorAll('.sv-region[data-region="gm-ori"] .sv-arrow, .sv-region[data-region="gm-cmp"] .sv-arrow, .sv-zoom').forEach(function (a) { a.style.display = 'none'; });

    // ── BLOCK 分组框、标题、注脚 ──
    const frame = svg.querySelector('.sv-frame');
    if (frame) { frame.classList.add('wb-block'); set(frame, { x: FRAME.x, y: FRAME.y, width: FRAME.w, height: FRAME.h }); }
    const bt = textOf(svg, /^展开 BLOCK 0/);
    if (bt) { set(bt, { x: FRAME.x, y: FRAME.y - 8 }); bt.textContent = 'BLOCK 0'; bt.classList.add('wb-block-k'); }
    el('text', { x: FRAME.x + 62, y: FRAME.y - 8, class: 'sv-label wb-block-sub' }, svg).textContent = '1 AIC + 2 AIV · 其余 23 个同构';
    const foot = textOf(svg, /^数据：pass 36/);
    if (foot) { set(foot, { x: VB_W - 12, y: VB_H - 6, 'text-anchor': 'end' }); foot.classList.add('wb-chip-foot'); foot.textContent = '数据：pass 36 · AllocateMemoryAddr（真实分配）'; }

    // ── AIC 行 ──
    const aicG = region(svg, 'aic');
    if (aicG) {
      set(aicG.querySelector('.sv-box'), { x: AIC.x, y: AIC.y, width: AIC.w, height: AIC.h });
      const t = aicG.querySelector('text'); if (t) set(t, { x: AIC.x + 24, y: AIC.y + 16 });
    }
    const l1Label = textOf(svg, /^L1 · MAT/);
    if (l1Label) set(l1Label, { x: L1.x - 8, y: L1.y + 14, 'text-anchor': 'end' });   // 放在 L1 左侧，让出顶端给 V2C 竖线
    ['l1-slot', 'l1-kv', 'l1-q'].forEach(function (id) {
      const g = region(svg, id), r = g && g.querySelector('.sv-l1'); if (!r) return;
      const lo = (num(r, 'x') - 36) / 246, hi = (num(r, 'x') + num(r, 'width') - 36) / 246;
      const y = L1.y + lo * L1.h, h = (hi - lo) * L1.h;                // 低地址在上
      set(r, { x: L1.x, y: y, width: L1.w, height: h, rx: 4 });
      const t = g.querySelector('text'); if (t) set(t, { x: L1.x + 6, y: y + Math.min(h / 2 + 4, 14) });
    });
    Object.keys(BOX).forEach(function (id) {
      const g = region(svg, id); if (!g) return;
      const b = BOX[id], o = ORIG[id];
      g.setAttribute('transform', 'translate(' + (b[0] - o[0]) + ' ' + (b[1] - o[1]) + ')');
      [].filter.call(g.querySelectorAll('rect'), function (r) { return num(r, 'x') === o[0] && num(r, 'y') === o[1]; }).forEach(function (r) { set(r, { width: b[2], height: b[3] }); });
      g.querySelectorAll('text[text-anchor="end"]').forEach(function (t) { t.setAttribute('x', o[0] + b[2] - 8); });
      if (id === 'l0c') g.querySelectorAll('.sv-fill-amber').forEach(function (r) { r.setAttribute('width', b[2] - 16); });
      if (id === 'l0a' || id === 'l0b') { const half = (b[2] - 16 - 4) / 2; g.querySelectorAll(id === 'l0a' ? '.sv-fill-lime' : '.sv-fill-lav').forEach(function (r, i) { set(r, { x: o[0] + 8 + i * (half + 4), width: half }); }); }
    });
    const pipeG = region(svg, 'pipe');
    if (pipeG) {
      pipeG.setAttribute('transform', 'translate(' + PIPE_D[0] + ' ' + PIPE_D[1] + ')');
      pipeG.querySelectorAll('.sv-arrow').forEach(function (arrow) { arrow.style.display = 'none'; });
      // V2C 的接收槽就在 L1 中，不再在画布右侧重复画一组槽。
      pipeG.querySelectorAll('text').forEach(function (t) { if (/^(C2V|V2C)$/.test(t.textContent.trim())) t.style.display = 'none'; });
      pipeG.querySelectorAll('.sv-slot').forEach(function (r) { r.style.display = 'none'; });
      // C2V 竖线从两个槽中间穿过；名字与槽数放在槽右侧
      pipeG.querySelectorAll('text').forEach(function (t) {
        const txt = t.textContent.trim();
        if (txt === 'C2V') set(t, { x: 345, y: 306, 'text-anchor': 'start' });
        else if (/槽|×/.test(txt)) set(t, { x: 373, y: 306, 'text-anchor': 'start' });
      });
      const hit = pipeG.querySelector('.sv-hit'); if (hit) set(hit, { x: 299, y: 290, width: 96, height: 32 });
    }

    // ── AIV0 / AIV1：细长横条，标题在左上、Vec 横条居左、说明两列在右（每条说明前加同色色块对应分段） ──
    const proj = svg.querySelector('.sv-proj-frame');
    if (proj) set(proj, { x: ROWX - 4, y: AIV_Y[0] - 4, width: ROWW + 8, height: AIV_Y[1] + AIV_H - AIV_Y[0] + 8 });
    const MAP = [[/^c2v 槽/, 0], [/^bias/, 1], [/^临时区/, 2], [/^mi \/ li/, 3], [/^softmax/, 5]];
    const swatches = el('g', { class: 'wb-swatches' }, svg);
    ['aiv0', 'aiv1'].forEach(function (id, n) {
      const X = ROWX, Y = AIV_Y[n], ub = 'ub' + n, g = region(svg, id);
      if (!g) return;
      set(g.querySelector('.sv-box'), { x: X, y: Y, width: ROWW, height: AIV_H });
      const title = g.querySelector('text'); if (title) set(title, { x: X + 24, y: Y + 20 });
      const x0 = X + 24, by = Y + BAR.dy;
      const ubG = region(svg, ub);
      if (ubG) {
        const mem = ubG.querySelector('.sv-mem'); if (mem) set(mem, { x: x0, y: by, width: BAR.w, height: BAR.h });
        const vt = ubG.querySelector('text'); if (vt) set(vt, { x: x0, y: by + BAR.h + 14 });
      }
      const bufs = [].slice.call(svg.querySelectorAll('.sv-ub[data-ub="' + ub + '"] .sv-buf'));
      bufs.forEach(function (b) {
        const y = num(b, 'y'), h = num(b, 'height'), x = MAPX(x0, y + h);
        set(b, { x: x, y: by + 1, width: Math.max(1, h * S), height: BAR.h - 2 });
      });
      // Vector 计算块（参考 hardware-architecture-viewport：UB ↔ Vector），放在 Vec 横条右侧；AIV1 空转时压暗
      const vx = x0 + BAR.w + 36, vy = Y + 14;
      el('path', { d: 'M' + (x0 + BAR.w + 4) + ' ' + (by + BAR.h / 2) + ' H ' + (vx - 4), class: 'wb-vec-link', 'data-owners': id + ' ' + ub, 'marker-start': 'url(#wbChipVecBoth)', 'marker-end': 'url(#wbChipVecBoth)' }, g);
      el('rect', { x: vx, y: vy, width: 90, height: 50, rx: 12, class: 'wb-vector' + (id === 'aiv1' ? ' is-idle' : '') }, g);
      el('text', { x: vx + 45, y: vy + 29, 'text-anchor': 'middle', class: 'wb-compute-k' }, g).textContent = 'Vector';
      const colX = x0 + BAR.w + 154;                                     // 右侧说明区（两列）
      let row = 0;
      loose[id].forEach(function (c) {
        c.dataset.ownerHint = id;
        if (c.tagName === 'line') {
          const x = MAPX(x0, num(c, 'y1'));
          if (num(c, 'x1') < num(c, 'x2') - 10) set(c, { x1: x, x2: x, y1: by - 4, y2: by + BAR.h + 4 });   // 192 / 184 参考线
          else set(c, { x1: x, x2: x, y1: by + BAR.h, y2: by + BAR.h + 6 });                                   // 高水位刻度
          return;
        }
        const txt = c.textContent.trim();
        if (txt === '192') { set(c, { x: x0 + BAR.w + 6, y: by - 4 }); return; }
        if (/^184/.test(txt)) { set(c, { x: MAPX(x0, VBAR.top + 10) - 6, y: by - 4, 'text-anchor': 'end' }); return; }
        if (/^高水位/.test(txt)) { set(c, { x: X + 164, y: Y + 20 }); return; }
        if (txt === '空转') { set(c, { x: X + 164, y: Y + 20 }); return; }
        const cx = colX + (row < 3 ? 0 : 128), cy = by + 8 + (row % 3) * 13;
        set(c, { x: cx + 12, y: cy });
        const hit = id === 'aiv0' && MAP.filter(function (m) { return m[0].test(txt); })[0];
        if (hit && bufs[hit[1]]) {
          const cls = (bufs[hit[1]].getAttribute('class') || '').split(' ').filter(function (k) { return /^(sv-buf|k-|is-idle|is-proj)/.test(k); }).join(' ');
          const sw = el('rect', { x: cx, y: cy - 8, width: 8, height: 8, rx: 1.5, class: cls + ' wb-sw', 'pointer-events': 'none' }, swatches);
          sw.dataset.ownerHint = id;
        }
        row += 1;
      });
    });

    // ── SPMD 条：分组框上方右端 ──
    const coresG = region(svg, 'cores'), core0 = coresG && coresG.querySelector('.sv-core.is-zoom');
    if (coresG) coresG.setAttribute('transform', 'translate(' + CORES_D[0] + ' ' + CORES_D[1] + ')');

    unify(svg);
    restyle(svg, gmBg);

    // ── 连线（全图坐标）：线色跟数据角色，交叉处留缝 ──
    const links = el('g', { class: 'wb-links' }, svg);
    const line = function (d, cls, mk, label, owners) {
      d = window.roundWorkbenchConnector(d);
      el('path', { d: d, class: 'wb-link-gap' }, links);
      el('path', { d: d, class: 'wb-link ' + cls, 'data-owners': owners || '', 'marker-end': mk ? 'url(#' + mk + ')' : '' }, links);
      if (label) el('text', { x: label[0], y: label[1], class: 'wb-link-k ' + cls, 'text-anchor': label[2] || 'start' }, links).textContent = label[3];
    };
    // 每条关系一根线：能直连就直连（0 折），跨行的写回只拐一次
    const gmR = GM.cardX + GM.cardW;                                    // GM 卡片右沿
    const aiv0Bot = AIV_Y[0] + AIV_H, barBot = AIV_Y[0] + BAR.dy + BAR.h;
    const l0R = COL.l0 + BOX.l0a[2], cubeR = BOX.cube[0] + BOX.cube[2];
    const l0cTop = BOX.l0c[1], l0cBot = BOX.l0c[1] + BOX.l0c[3];
    const rowBuf = svg.querySelector('.sv-ub[data-ub="ub0"] .sv-buf.k-row');
    const MI_X = rowBuf ? num(rowBuf, 'x') + 1 : BAR_X + 110;          // UB 里 mi / li 两行的位置
    // GM → AIV0 / L1：卡片与目标同高，水平直连
    if (card['gm-plan']) line('M' + gmR + ' ' + BAR_MID + ' H ' + (BAR_X - 2), 'is-vec', 'wbChipVec', null, 'gm-plan aiv0 ub0');
    if (card['gm-ori']) line('M' + gmR + ' ' + GM_Y['gm-ori'] + ' H ' + (L1.x - 2), 'is-kv', 'wbChipKv', null, 'gm-ori l1-kv');
    if (card['gm-cmp']) line('M' + gmR + ' ' + GM_Y['gm-cmp'] + ' H ' + (L1.x - 2), 'is-kv', 'wbChipKv', null, 'gm-cmp l1-kv');
    if (card['gm-q']) line('M' + gmR + ' ' + Y_Q + ' H ' + (L1.x - 2), 'is-q', 'wbChipQ', null, 'gm-q l1-q');
    // AIC 内部：L1 KV → L0B → CUBE、L1 Q → L0A → CUBE、CUBE → L0C，全是水平直线
    line('M' + (L1.x + L1.w) + ' ' + Y_KV + ' H ' + (COL.l0 - 2), 'is-kv', 'wbChipKv', null, 'l1-kv l0b');
    line('M' + (L1.x + L1.w) + ' ' + Y_Q + ' H ' + (COL.l0 - 2), 'is-q', 'wbChipQ', null, 'l1-q l0a');
    line('M' + l0R + ' ' + Y_KV + ' H ' + (BOX.cube[0] - 2), 'is-kv', 'wbChipKv', null, 'l0b cube');
    line('M' + l0R + ' ' + Y_Q + ' H ' + (BOX.cube[0] - 2), 'is-q', 'wbChipQ', null, 'l0a cube');
    line('M' + cubeR + ' ' + ROW + ' H ' + (COL.l0c - 2), 'is-acc', 'wbChipAcc', null, 'cube l0c');
    // 核间管道：C2V 从 L0C 竖直向上进 AIV0；V2C 从 UB 工作区竖直向下进 L1 顶端的 v2c 槽；
    //   AIV1 与 C2V 同一条竖线向下，虚线 = 只握手、无有效数据
    line('M' + C2V_X + ' ' + l0cTop + ' V ' + (aiv0Bot + 2), 'is-vec', 'wbChipVec', null, 'l0c pipe ub0 aiv0');
    line('M' + V2C_X + ' ' + barBot + ' V ' + (L1.y - 2), 'is-vec', 'wbChipVec', null, 'ub0 aiv0 l1-slot');
    el('text', { x: V2C_X + 6, y: AIC.y + 16, class: 'wb-link-k is-vec' }, links).textContent = 'V2C';
    line('M' + C2V_X + ' ' + l0cBot + ' V ' + (AIV_Y[1] - 2), 'is-dim', 'wbChipDim', [C2V_X - 6, AIV_Y[1] - 4, 'end', 'AIV1 ↔ AIC 只握手（tpop / tfree），无有效数据'], 'pipe aiv1 ub1');
    // 写回：oi 从 L0C 竖直上到写回横道、左拐进 GM（只拐一次）；mi / li 从 UB 竖直落到横道汇入
    if (card['gm-out']) {
      line('M' + OI_X + ' ' + l0cTop + ' V ' + LANE_OUT + ' H ' + (gmR + 2), 'is-out', 'wbChipOut', null, 'l0c gm-out');
      line('M' + MI_X + ' ' + barBot + ' V ' + LANE_OUT, 'is-out', '', null, 'ub0 aiv0 gm-out');
    }
    // 搬运单元：胶囊标在对应连线上（参考 hardware-architecture-viewport 的 MTE / FixPipe），同类上下对齐
    const pillG = el('g', { class: 'wb-pills', 'pointer-events': 'none' }, svg);
    const pill = function (cx, cy, t) {
      const w = t.length * 5.4 + 10;
      el('rect', { x: cx - w / 2, y: cy - 6, width: w, height: 12, rx: 6, class: 'wb-pill' }, pillG);
      el('text', { x: cx, y: cy + 3, 'text-anchor': 'middle', class: 'wb-pill-k' }, pillG).textContent = t;
    };
    const mte2X = (AIC.x + L1.x) / 2 + 18;
    if (card['gm-plan']) pill((FRAME.x + BAR_X) / 2, BAR_MID, 'MTE2');                               // GM → UB
    ['gm-ori', 'gm-cmp', 'gm-q'].forEach(function (id) { if (card[id]) pill(mte2X, GM_Y[id], 'MTE2'); }); // GM → L1
    pill((L1.x + L1.w + COL.l0) / 2, Y_KV, 'MTE1');                                                // L1 → L0B
    pill((L1.x + L1.w + COL.l0) / 2, Y_Q, 'MTE1');                                                 // L1 → L0A
    if (card['gm-out']) {
      pill(OI_X, l0cTop - 22, 'FixPipe');                                                          // L0C → GM
      pill(MI_X, (barBot + LANE_OUT) / 2, 'MTE3');                                                 // UB → GM
    }
    // 放大锥：SPMD 第 1 格 → 分组框上沿；SPMD 条垫底并置顶
    if (core0 && frame) {
      const cx = num(core0, 'x') + CORES_D[0], cy = num(core0, 'y') + num(core0, 'height') + CORES_D[1];
      el('path', { d: 'M' + cx + ' ' + cy + ' L ' + FRAME.x + ' ' + FRAME.y + ' H ' + (FRAME.x + FRAME.w) + ' L ' + (cx + num(core0, 'width')) + ' ' + cy + ' Z', class: 'wb-cone' }, svg, links);
    }
    if (coresG) {
      const hit = coresG.querySelector('.sv-hit');
      if (hit) el('rect', { x: num(hit, 'x') - 4 + CORES_D[0], y: num(hit, 'y') + CORES_D[1], width: num(hit, 'width') + 8, height: num(hit, 'height'), rx: 6, class: 'wb-strip-plate' }, svg);
      svg.appendChild(coresG);
    }
    layer(svg);
  }

  // ── 图形与布局参考 PTO 设计系统 hardware-architecture-viewport；颜色、字体、面板质感沿用本页（与上方计算图一致）──
  //   核面板：实心、无描边（同计算图阶段卡）；存储（L0A / L0B / L0C / UB）：中性底 + 灰点阵，占用条按数据角色上色；
  //   计算（CUBE / Vector）：实心块，按执行单元着色（Cube 紫、Vec 蓝，同计算图角标）；搬运单元：胶囊（同源码栏 GM 搬运标签）
  function restyle(svg, gmBg) {
    const defs = svg.querySelector('defs');
    // 存储：沿用页面里 GM 卡片、L0 的灰色点阵；UB（Vec 横条）也铺一层
    ['l0a', 'l0b', 'l0c'].forEach(function (id) {
      const g = region(svg, id); if (!g) return;
      g.classList.add('wb-buf');
      const k = g.querySelector('text'); if (k) k.classList.add('wb-buf-k');
    });
    ['ub0', 'ub1'].forEach(function (id) {
      const g = region(svg, id), mem = g && g.querySelector('.sv-mem'); if (!mem) return;
      g.classList.add('wb-buf');
      el('rect', { x: mem.getAttribute('x'), y: mem.getAttribute('y'), width: mem.getAttribute('width'), height: mem.getAttribute('height'), rx: 5, fill: 'url(#csGrid)', opacity: '.45', 'pointer-events': 'none' }, g, mem.nextSibling);
    });
    // 计算：CUBE 改成实心青绿块，名字居中；形状与切分说明挪到块下方（悬停显示）
    const cube = region(svg, 'cube');
    if (cube) {
      cube.classList.add('wb-compute');
      const o = ORIG.cube, b = BOX.cube, cx = o[0] + b[2] / 2;
      cube.querySelectorAll('text').forEach(function (t) {
        const txt = t.textContent.trim();
        if (txt === 'CUBE') { set(t, { x: cx, y: o[1] + b[3] / 2 + 5, 'text-anchor': 'middle' }); t.setAttribute('class', 'wb-compute-k'); return; }
        const k = /^QK/.test(txt) ? 0 : /^PV/.test(txt) ? 1 : /^K=/.test(txt) ? 2 : -1;
        if (k >= 0) set(t, { x: cx, y: o[1] - 41 + k * 13, 'text-anchor': 'middle' });   // 放在 CUBE 上方空白处，不压下方的握手虚线说明
      });
    }
  }

  // ── 与上方计算图统一图元：同一对象同一图形、同一颜色（计算图一侧见 workbench-polish.js 的 unifyLogic） ──
  //   L1 上的 Q / KV = 计算图里的 Q 块 / KV(L1) 张量：亮色 + 网格纹理 + 深色字，名字统一为 Q、KV（形状悬停看）
  //   v2c 槽与 C2V / V2C 管道槽 = 计算图 PIPELINE 右上角的“2 槽同时在飞”：同一种亮蓝方块
  //   PL.SPMD × 24 前加计算图同款竖条图标；AIC / AIV 框左侧加执行单元竖条（Cube 紫、Vec 蓝），对应计算图阶段卡上的竖条
  function unify(svg) {
    const defs = svg.querySelector('defs');
    const pat = el('pattern', { id: 'wbTensorGrid', width: '6', height: '6', patternUnits: 'userSpaceOnUse' }, defs);
    el('rect', { x: '.5', y: '.5', width: '5', height: '5', fill: 'none', stroke: 'rgb(0 0 0 / 32%)', 'stroke-width': '.7' }, pat);
    ['l1-q', 'l1-kv'].forEach(function (id) {
      const g = region(svg, id), r = g && g.querySelector('.sv-l1'); if (!r) return;
      g.classList.add('wb-tensor');
      el('rect', { x: r.getAttribute('x'), y: r.getAttribute('y'), width: r.getAttribute('width'), height: r.getAttribute('height'), rx: 4, fill: 'url(#wbTensorGrid)', 'pointer-events': 'none' }, g, r.nextSibling);
      const t = g.querySelector('text');
      if (t && id === 'l1-kv') {
        t.textContent = 'KV';
        // 竖条窄：形状与大小分两行放在段内，悬停显示
        [['[128,512]', 16], ['128 KB', 28]].forEach(function (l) {
          const sec = el('text', { x: num(t, 'x'), y: num(t, 'y') + l[1], class: 'sv-strong wb-sec wb-l1-sec' }, g);
          sec.dataset.owner = 'l1-kv'; sec.textContent = l[0];
        });
      }
    });
    svg.querySelectorAll('.sv-region[data-region="pipe"] text').forEach(function (t) {
      const m = /^(\d) × (\d+K)$/.exec(t.textContent.trim());
      if (m) t.textContent = m[1] + ' 槽 × ' + m[2];
    });
    const coresG = region(svg, 'cores');
    if (coresG) {
      const lab = [].filter.call(coresG.querySelectorAll('text'), function (t) { return /^PL\.SPMD/.test(t.textContent); })[0];
      if (lab) {
        let w = 0; try { w = lab.getComputedTextLength(); } catch (e) { w = 0; }
        if (!w) w = lab.textContent.length * 6.4;
        const ic = el('g', { class: 'wb-chip-ic', transform: 'translate(' + (num(lab, 'x') - w - 16) + ' ' + (num(lab, 'y') - 9) + ')' }, coresG);
        [0, 3, 6, 9].forEach(function (x) { el('rect', { x: x, y: 0, width: 1.8, height: 10, rx: .6 }, ic); });
      }
    }
  }

  // ── 信息分层：常驻只留结论与名字；次要说明悬停区块时显示，点击钉住，或用“全部标注”一次展开 ──
  // 每条次要文字（及 AIV0 的引线）标上 data-owner = 它所属的区块；悬停 / 钉住哪块，就点亮哪块的次要信息。
  // 在左侧源码悬停时，lens-tile.js 联动点亮的区块（.sv-region.is-hot）也会显示，见 workbench-polish.css 第 12 节。
  const KEEP = /^(GM · HBM|PL\.SPMD|BLOCK 0|AIC · CUBE|AIV[01] · SUBBLOCK|L1 · MAT|L0A · LEFT|L0B · RIGHT|CUBE$|L0C · ACC|C2V$|V2C$|高水位|空转$|推算$|数据：pass 36)/;
  // 区块之间的从属：悬停 AIC 大框时连同它里面的 L0A / L0B / CUBE / L0C 一起显示；Vec 柱与 AIV 框互通
  const ALIAS = { aic: ['l0a', 'l0b', 'cube', 'l0c', 'l1-kv', 'l1-q', 'l1-slot'], aiv0: ['ub0'], ub0: ['aiv0'], aiv1: ['ub1'], ub1: ['aiv1'] };
  function layer(svg) {
    const mark = function (n, owner) { n.classList.add('wb-sec'); n.dataset.owner = owner; };
    svg.querySelectorAll('text').forEach(function (t) {
      const txt = t.textContent.trim();
      if (!txt || t.classList.contains('wb-sec')) return;
      if (t.closest('.wb-pills') || /wb-(gm|compute)-k/.test(t.getAttribute('class') || '')) return;   // 胶囊与计算块名字常驻
      const reg = t.closest('.sv-region');
      const gm = reg && /^gm-/.test(reg.dataset.region) ? reg : null;
      if (gm) return;                                                     // GM 张量：名字与形状常驻（同上方图）
      if (reg && reg.dataset.region === 'cores') return;                 // PL.SPMD × 24 常驻
      if (/^l1-/.test(reg && reg.dataset.region || '')) return;          // L1 条上的名字常驻
      if (KEEP.test(txt)) return;
      if (t.classList.contains('wb-link-k')) { mark(t, /^bias/.test(txt) ? 'gm-plan' : /^写回|←/.test(txt) ? 'gm-out' : 'pipe'); return; }
      if (t.classList.contains('wb-block-sub')) { mark(t, 'cores'); return; }
      mark(t, reg ? reg.dataset.region : (t.dataset.ownerHint || 'chip'));
    });
    svg.querySelectorAll('.wb-sw').forEach(function (p) { mark(p, p.dataset.ownerHint || 'aiv0'); });
    // 常驻一句提示：细节在哪看
    const hint = document.createElementNS(NS, 'text');
    hint.setAttribute('x', 12); hint.setAttribute('y', VB_H - 6); hint.setAttribute('class', 'wb-peek-hint');
    hint.textContent = '悬停区块看细节 · 点击钉住';
    svg.appendChild(hint);
    paintPeek(svg);
  }
  let hoverSet = [], pinSet = [], showAll = false;
  function ownersFrom(target) {
    const svg = host.querySelector(':scope > svg');
    if (!svg || !target || !svg.contains(target)) return [];
    const ub = target.closest('.sv-ub');
    let id = ub ? ub.dataset.ub : null;
    if (!id) { const r = target.closest('.sv-region'); id = r ? r.dataset.region : null; }
    if (!id) return [];
    return [id].concat(ALIAS[id] || []);
  }
  function paintPeek(svg) {
    svg = svg || host.querySelector(':scope > svg');
    if (!svg) return;
    svg.classList.toggle('wb-all', showAll);
    const on = hoverSet.concat(pinSet);
    if (document.body.classList.contains('is-focusing') || !document.getElementById('lensDetail').hidden) {
      svg.querySelectorAll('.sv-region.is-hot').forEach(function (r) { on.push(r.dataset.region); });
    }
    svg.querySelectorAll('[data-owners]').forEach(function (link) {
      const active = link.dataset.owners.split(' ').some(function (id) { return on.indexOf(id) >= 0; });
      link.classList.toggle('is-active', active);
    });
    svg.querySelectorAll('.wb-sec').forEach(function (n) { n.classList.toggle('is-peek', on.indexOf(n.dataset.owner) >= 0); });
    svg.querySelectorAll('.sv-region').forEach(function (r) { r.classList.toggle('wb-pinned', pinSet.indexOf(r.dataset.region) >= 0); });
  }
  host.addEventListener('pointerover', function (e) { hoverSet = ownersFrom(e.target); paintPeek(); });
  host.addEventListener('pointerleave', function () { hoverSet = []; paintPeek(); });
  host.addEventListener('click', function (e) {
    const o = ownersFrom(e.target);
    if (!o.length) { pinSet = []; paintPeek(); return; }
    pinSet = pinSet[0] === o[0] ? [] : o;            // 再点同一块取消钉住
    paintPeek();
  });

  // “全部标注”开关 + 顶部资源读数行只留关键两项（Vec、AIV 在算），其余悬停读数行时展开
  function toolbar() {
    const extra = document.querySelector('.wb-chip-extra');
    if (!extra || extra.querySelector('.wb-peek-all')) return;
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'wb-peek-all'; b.setAttribute('aria-pressed', 'false');
    b.textContent = '全部标注';
    b.title = '一次显示图上所有次要说明（讲解时用）';
    b.addEventListener('click', function () { showAll = !showAll; b.setAttribute('aria-pressed', String(showAll)); paintPeek(); });
    const info = extra.querySelector('.pane-info');
    extra.insertBefore(b, info || null);
  }
  function tagStats() {
    const stats = document.getElementById('hpChipStats');
    if (!stats) return;
    stats.querySelectorAll('.cs-stat').forEach(function (s) {
      const k = (s.querySelector('.meter-k') || {}).textContent || '';
      s.classList.toggle('wb-stat-sec', !/^(Vec|AIV)$/.test(k));
    });
  }
  const stats = document.getElementById('hpChipStats');
  if (stats) new MutationObserver(tagStats).observe(stats, { childList: true });
  new MutationObserver(decorate).observe(host, { childList: true });
  new MutationObserver(function (changes) {
    if (changes.some(function (change) { return change.target === document.body || change.target.classList.contains('sv-region'); })) paintPeek();
  }).observe(document.body, { attributes: true, subtree: true, attributeFilter: ['class'] });
  const detail = document.getElementById('lensDetail');
  if (detail) new MutationObserver(function () { paintPeek(); }).observe(detail, { attributes: true, attributeFilter: ['hidden'] });
  decorate(); tagStats();
  setTimeout(toolbar, 0);
})();
