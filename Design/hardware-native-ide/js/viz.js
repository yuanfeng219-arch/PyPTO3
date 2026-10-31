/* ──────────────────────────────────────────────────────────────────────────────
 * Hardware-Native Lens · 可视化
 * 全部为内联 SVG，配色走设计系统 token（见 styles/app.css 的 .hv-* 类）。
 * ────────────────────────────────────────────────────────────────────────────── */
window.HNV = (function () {
  'use strict';

  const P = window.VisualSystem;
  let infraDiagramId = 0;
  function infraSvg(body, width, height, label) {
    const prefix = 'hn-infra-' + (++infraDiagramId) + '-';
    return P.wrap(body, width, height)
      .replace('<svg ', '<svg class="hv infra-diagram" ')
      .replace('aria-label="对象与关系示例"', 'aria-label="' + esc(label) + '"')
      .replace(/id="(arrow|bind|map)"/g, function (_, id) { return 'id="' + prefix + id + '"'; })
      .replace(/url\(#(arrow|bind|map)\)/g, function (_, id) { return 'url(#' + prefix + id + ')'; });
  }

  // ── SVG 原子 ──────────────────────────────────────────────────────────────
  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function svg(w, h, inner) {
    return '<svg class="hv" viewBox="0 0 ' + w + ' ' + h + '" preserveAspectRatio="xMidYMid meet" '
      + 'aria-hidden="true" focusable="false">'
      + '<defs>'
      + '<pattern id="hvPad" width="5" height="5" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">'
      + '<line x1="0" y1="0" x2="0" y2="5" class="hv-hatch"/></pattern>'
      + marker('hvArrow', 'hv-ah') + marker('hvArrowOk', 'hv-ah-ok')
      + marker('hvArrowErr', 'hv-ah-err') + marker('hvArrowFlow', 'hv-ah-flow')
      + '</defs>' + inner + '</svg>';
  }
  function marker(id, cls) {
    return '<marker id="' + id + '" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5.5" markerHeight="5.5"'
      + ' orient="auto"><path d="M0 1 L7 4 L0 7 z" class="' + cls + '"/></marker>';
  }
  function r(x, y, w, h, cls, rx) {
    return '<rect x="' + x + '" y="' + y + '" width="' + Math.max(0, w) + '" height="' + Math.max(0, h)
      + '" rx="' + (rx == null ? 2 : rx) + '" class="' + cls + '"/>';
  }
  function t(x, y, s, cls, anchor) {
    return '<text x="' + x + '" y="' + y + '" class="' + (cls || 'hv-t') + '"'
      + (anchor ? ' text-anchor="' + anchor + '"' : '') + '>' + esc(s) + '</text>';
  }
  function ln(x1, y1, x2, y2, cls) {
    return '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '" class="' + (cls || 'hv-l') + '"/>';
  }
  function arrow(x1, y1, x2, y2, cls) {
    const c = cls || 'hv-l';
    let mk = 'hvArrow';
    if (c.indexOf('ok-stroke') >= 0) mk = 'hvArrowOk';
    else if (c.indexOf('err-stroke') >= 0) mk = 'hvArrowErr';
    else if (c.indexOf('flow') >= 0) mk = 'hvArrowFlow';
    return '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2
      + '" class="' + c + '" marker-end="url(#' + mk + ')"/>';
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 1 · Tensor / Tile —— 整张张量上切出一块，并落到具体片上空间
  // ════════════════════════════════════════════════════════════════════════════
  function tensorTile(opts) {
    const W = 560, PAD = 14;
    const bw = 400, bh = 86, bx = PAD + 46, by = 34;
    const cols = Math.min(opts.cols, 12);
    const truncated = opts.cols > cols;
    const cw = bw / cols;
    let s = '';

    s += t(bx, by - 12, opts.topLabel, 'hv-t hv-dim');
    s += t(bx + bw, by - 12, opts.spaceLabel || 'GM', 'hv-t hv-dim', 'end');
    s += r(bx, by, bw, bh, 'hv-tensor', 3);

    for (let i = 0; i < cols; i++) {
      const x = bx + i * cw;
      if (i > 0) s += ln(x, by, x, by + bh, 'hv-l hv-grid');
      if (i === opts.hot) s += r(x + 1, by + 1, cw - 2, bh - 2, 'hv-tile', 2);
    }
    if (truncated) {
      s += r(bx + bw - cw * 0.5, by + 1, cw * 0.5 - 1, bh - 2, 'hv-elide', 2);
      s += t(bx + bw - cw * 0.25, by + bh / 2 + 3, '…', 'hv-t', 'middle');
    }
    // 行标（M 维）
    s += t(bx - 8, by + bh / 2 + 3, opts.rowsLabel, 'hv-t hv-dim', 'end');
    // 命中 tile 的标注
    const hx = bx + opts.hot * cw + cw / 2;
    s += t(hx, by + bh / 2 + 3, opts.tileLabel, 'hv-t hv-strong', 'middle');

    // 下方：tile 去向
    const dy = by + bh + 30;
    s += arrow(hx, by + bh + 4, hx, dy - 12, 'hv-l hv-flow');
    const dw = 150, dx = Math.min(Math.max(hx - dw / 2, PAD), W - PAD - dw);
    s += r(dx, dy - 10, dw, 30, 'hv-buf', 3);
    s += t(dx + 10, dy + 4, opts.dest, 'hv-t hv-strong');
    s += t(dx + dw - 10, dy + 4, opts.destBytes, 'hv-t hv-dim', 'end');
    s += t(PAD, dy + 36, opts.foot, 'hv-t hv-dim');

    return svg(W, dy + 46, s);
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 2 · 内部布局 —— cube fractal 填充 + set_validshape
  // ════════════════════════════════════════════════════════════════════════════
  function padRows(d) {
    const W = 560, PAD = 14;
    const rowH = 9, gap = 2, bx = PAD + 34, bw = 300;
    const n = d.Q_HEAD_PAD, real = d.Q_HEAD_BATCH;
    const by = 36;
    let s = '';

    s += t(PAD, 18, 'all_q_padded 的一个 lane · 行距 = Q_HEAD_PAD', 'hv-t hv-dim');
    s += t(bx + bw + 10, 18, 'HEAD_DIM = ' + d.HEAD_DIM, 'hv-t hv-dim');

    for (let i = 0; i < n; i++) {
      const y = by + i * (rowH + gap);
      const isReal = i < real;
      s += r(bx, y, bw, rowH, isReal ? 'hv-row-real' : 'hv-row-pad', 1);
      if (!isReal) s += r(bx, y, bw, rowH, 'hv-row-hatch', 1);
      s += t(bx - 8, y + rowH - 1, String(i), 'hv-t hv-micro', 'end');
    }
    const realH = real * (rowH + gap) - gap;
    const allH = n * (rowH + gap) - gap;

    // 有效行括号
    const kx = bx + bw + 8;
    s += ln(kx, by, kx, by + realH, 'hv-l hv-ok-stroke');
    s += ln(kx, by, kx + 5, by, 'hv-l hv-ok-stroke');
    s += ln(kx, by + realH, kx + 5, by + realH, 'hv-l hv-ok-stroke');
    s += t(kx + 10, by + realH / 2 + 3, 'set_validshape → ' + real + ' 行有效', 'hv-t hv-ok-text');

    // 物理 tile 括号
    const px = bx - 24;
    s += ln(px, by, px, by + allH, 'hv-l');
    s += ln(px, by, px + 5, by, 'hv-l');
    s += ln(px, by + allH, px + 5, by + allH, 'hv-l');
    s += t(px - 4, by + allH / 2 + 3, 'tpush ' + n + ' 行', 'hv-t hv-dim', 'end');

    const fy = by + allH + 22;
    s += t(PAD, fy, 'cube M fractal = 16 行 → 物理 tile 不能收缩到 ' + real + ' 行', 'hv-t hv-dim');
    s += t(PAD, fy + 15, 'fillpad 把填充行写成 -inf，row_max / row_sum 不被污染', 'hv-t hv-dim');
    const cls = d.PAD_WASTE >= 0.5 ? 'hv-t hv-warn-text' : 'hv-t hv-ok-text';
    s += t(W - PAD, fy, '填充占比 ' + d.PAD_WASTE_PCT + '%', cls, 'end');
    s += t(W - PAD, fy + 15, '搬运 ' + n + ' 行 / 计算 ' + real + ' 行', 'hv-t hv-dim', 'end');

    return svg(W, fy + 26, s);
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 3 · 内部布局 —— paged KV：逻辑块 → 物理页
  // ════════════════════════════════════════════════════════════════════════════
  function pagedKV(d) {
    const W = 560, PAD = 14, bw = 380, bx = PAD + 70;
    const SHOW = 4;
    const perTok = bw / (SHOW * d.SEQ_TILE);
    const pgW = d.PAGE_SIZE * perTok;
    const blkW = d.SEQ_TILE * perTok;
    const pages = Math.ceil((SHOW * d.SEQ_TILE) / d.PAGE_SIZE);
    const match = d.SEQ_TILE === d.PAGE_SIZE;
    const ya = 40, yb = 128, h = 26;
    let s = '';

    s += t(PAD, 18, '一个请求的逻辑 KV 块 (SEQ_TILE = ' + d.SEQ_TILE + ')', 'hv-t hv-dim');
    s += t(PAD, ya + h / 2 + 3, 'logical', 'hv-t hv-dim');
    for (let i = 0; i < SHOW; i++) {
      const x = bx + i * blkW;
      s += r(x, ya, blkW - 2, h, i === 2 ? 'hv-tile' : 'hv-tensor', 2);
      s += t(x + blkW / 2 - 1, ya + h / 2 + 3, 'blk ' + i, 'hv-t hv-micro', 'middle');
    }

    s += t(PAD, yb + h / 2 + 3, 'physical', 'hv-t hv-dim');
    const pid = [12, 7, 23, 4, 31, 18, 9, 27];
    for (let p = 0; p < pages; p++) {
      const x = bx + p * pgW;
      s += r(x, yb, Math.min(pgW - 2, bx + bw - x - 2), h, 'hv-page', 2);
      s += t(x + Math.min(pgW, bx + bw - x) / 2 - 1, yb + h / 2 + 3, '#' + pid[p % pid.length], 'hv-t hv-micro', 'middle');
    }
    s += t(bx + bw + 6, yb + h / 2 + 3, 'page ' + d.PAGE_SIZE, 'hv-t hv-dim');

    // block_table 中转
    const ym = 84;
    s += r(bx, ym - 11, bw, 22, 'hv-buf', 3);
    s += t(bx + 8, ym + 4, 'block_table[b · max_blocks + blk] → 物理页 id', 'hv-t');

    // 第 2 个逻辑块的映射
    const sx = bx + 2 * blkW + blkW / 2;
    s += arrow(sx, ya + h + 3, sx, ym - 14, match ? 'hv-l hv-flow' : 'hv-l hv-err-stroke');
    if (match) {
      const tx = bx + 2 * pgW + pgW / 2;
      s += arrow(sx, ym + 14, tx, yb - 4, 'hv-l hv-flow');
      s += t(PAD, yb + h + 24, 'SEQ_TILE == page_size：blk 2 整块落在一个物理页上，cache_row 算术成立。', 'hv-t hv-ok-text');
    } else {
      const span0 = (2 * d.SEQ_TILE) / d.PAGE_SIZE;
      const span1 = (3 * d.SEQ_TILE) / d.PAGE_SIZE;
      const x0 = bx + span0 * pgW, x1 = bx + span1 * pgW;
      s += r(x0, yb - 5, Math.min(x1 - x0, bx + bw - x0), h + 10, 'hv-straddle', 2);
      s += arrow(sx, ym + 14, (x0 + Math.min(x1, bx + bw)) / 2, yb - 8, 'hv-l hv-err-stroke');
      s += t(PAD, yb + h + 24, 'blk 2 横跨 ' + (span1 - span0).toFixed(2)
        + ' 个物理页；相邻页的物理 id 无关，读到的是别的请求的 KV。', 'hv-t hv-err-text');
    }
    s += t(PAD, yb + h + 40, 'MAX_CTX_BLOCKS = ceil(' + d.MAX_SEQ + ' / ' + d.SEQ_TILE + ') = '
      + d.MAX_CTX_BLOCKS + ' 个逻辑块 / 请求', 'hv-t hv-dim');

    return svg(W, yb + h + 50, s);
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 4 · 内部布局 —— split-K 切片与上一层 slab 的对齐
  // ════════════════════════════════════════════════════════════════════════════
  function splitK(d) {
    const W = 560, PAD = 14, bw = 420, bx = PAD + 56;
    const ya = 40, yb = 112, h = 26;
    const perCol = bw / d.HIDDEN;
    const nSlice = d.QKV_OK, nSlab = Math.max(1, Math.round(d.DOWN_ON));
    const sliceW = d.QKV_K_SLICE * perCol, slabW = d.DOWN_TN * perCol;
    const match = d.QKV_K_SLICE === d.DOWN_TN;
    let s = '';

    s += t(PAD, 18, 'K 维 = HIDDEN ' + d.HIDDEN + '，两侧各自切分', 'hv-t hv-dim');

    s += t(PAD, ya + h / 2 + 3, 'q_proj', 'hv-t hv-dim');
    for (let i = 0; i < nSlice; i++) {
      const x = bx + i * sliceW;
      s += r(x, ya, sliceW - 2, h, i === 1 ? 'hv-tile' : 'hv-tensor', 2);
      s += t(x + sliceW / 2 - 1, ya + h / 2 + 3, 'k' + i, 'hv-t hv-micro', 'middle');
    }
    s += t(bx + bw + 6, ya + h / 2 + 3, '×' + nSlice + ' split-K', 'hv-t hv-dim');

    s += t(PAD, yb + h / 2 + 3, 'normed_in', 'hv-t hv-dim');
    for (let i = 0; i < nSlab; i++) {
      const x = bx + i * slabW;
      s += r(x, yb, Math.min(slabW - 2, bx + bw - x - 2), h, 'hv-page', 2);
      s += t(x + Math.min(slabW, bx + bw - x) / 2 - 1, yb + h / 2 + 3, 'slab ' + i, 'hv-t hv-micro', 'middle');
    }
    s += t(bx + bw + 6, yb + h / 2 + 3, '×' + nSlab + ' slab', 'hv-t hv-dim');

    // 依赖连线
    const cnt = match ? nSlice : Math.max(nSlice, nSlab);
    for (let i = 0; i < cnt; i++) {
      const x1 = bx + i * sliceW + sliceW / 2;
      const x2 = bx + i * slabW + slabW / 2;
      if (i < nSlice && i < nSlab) {
        s += arrow(x2, yb - 3, x1, ya + h + 4, match ? 'hv-l hv-ok-stroke' : 'hv-l hv-err-stroke hv-dash');
      }
    }
    const fy = yb + h + 26;
    if (match) {
      s += t(PAD, fy, 'QKV_K_SLICE == DOWN_TN：deps=[prev_normed_tids[q_ks]] 精确挂到唯一写者。', 'hv-t hv-ok-text');
      s += t(PAD, fy + 15, 'manual_scope 内没有 tensormap，这条 1:1 是依赖边唯一的来源。', 'hv-t hv-dim');
    } else {
      s += t(PAD, fy, 'K 切片 ' + d.QKV_K_SLICE + ' ≠ slab ' + d.DOWN_TN
        + '：q_ks 不再指向唯一写者。', 'hv-t hv-err-text');
      s += t(PAD, fy + 15, '退路只有整父张量 WAW 串行化 —— split-K 的并行度被吃掉。', 'hv-t hv-dim');
    }
    return svg(W, fy + 26, s);
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 5 · Scope —— manual_scope 内外的依赖语义
  // ════════════════════════════════════════════════════════════════════════════
  function scopeDeps(d) {
    const W = 560, PAD = 14;
    const n = Math.min(Math.max(1, Math.round(d.DOWN_ON)), 6);
    const m = Math.min(d.QKV_OK, 6);
    const bw = 104, bh = 22, gap = 7;
    const lx = PAD + 24, rx = W - PAD - bw - 24;
    let s = '';

    s += t(lx, 18, 'auto-dep 区 · tensormap 自动成边', 'hv-t hv-dim');
    s += t(rx + bw, 18, 'manual_scope · 仅显式 deps', 'hv-t hv-dim', 'end');

    const top = 34;
    for (let i = 0; i < n; i++) {
      const y = top + i * (bh + gap);
      s += r(lx, y, bw, bh, 'hv-page', 3);
      s += t(lx + 8, y + bh / 2 + 3, 'dcr slab ' + i, 'hv-t hv-micro');
    }
    for (let i = 0; i < m; i++) {
      const y = top + i * (bh + gap);
      s += r(rx, y, bw, bh, 'hv-tensor', 3);
      s += t(rx + 8, y + bh / 2 + 3, 'q_proj k' + i, 'hv-t hv-micro');
    }
    const match = d.QKV_K_SLICE === d.DOWN_TN;
    for (let i = 0; i < Math.min(n, m); i++) {
      const y = top + i * (bh + gap) + bh / 2;
      s += arrow(lx + bw + 3, y, rx - 4, y, match ? 'hv-l hv-ok-stroke' : 'hv-l hv-err-stroke hv-dash');
    }
    const rows = Math.max(n, m);
    const fy = top + rows * (bh + gap) + 14;
    s += r(lx, fy - 10, rx + bw - lx, 24, 'hv-buf', 3);
    s += t(lx + 10, fy + 5, 'tensormap 在 manual_scope 内被抑制 —— 这些边必须由 deps= 写出来', 'hv-t');
    s += t(PAD, fy + 32, 'pl.at(level=pl.Level.CORE_GROUP) 决定一次 submit 的粒度；'
      + 'TaskId 是代码里可以存进 pl.Array 的值。', 'hv-t hv-dim');
    s += t(PAD, fy + 47, '跨层 carry 用 ' + Math.round(d.DOWN_ON)
      + ' 个 TaskId 句柄逐 slab 挂边，而不是整张张量一次 fence。', 'hv-t hv-dim');
    return svg(W, fy + 58, s);
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 6 · 并行请求 —— 核上分布
  // ════════════════════════════════════════════════════════════════════════════
  function coreFan(d, opts) {
    const W = 560, PAD = 14;
    const cores = opts.cores, counts = opts.counts;
    const bw = W - PAD * 2 - 40;
    const cw = Math.max(5, bw / cores - 3);
    const maxC = Math.max.apply(null, counts) || 1;
    const unit = Math.min(13, 92 / maxC);
    const base = 40 + maxC * unit;
    let s = '';

    s += t(PAD, 18, opts.title, 'hv-t hv-dim');
    s += t(W - PAD, 18, opts.right, 'hv-t hv-dim', 'end');

    for (let c = 0; c < cores; c++) {
      const x = PAD + 34 + c * (cw + 3);
      for (let k = 0; k < counts[c]; k++) {
        const y = base - (k + 1) * unit;
        const hot = counts[c] === maxC && opts.flagMax;
        s += r(x, y + 1, cw, unit - 2, hot ? 'hv-core-hot' : 'hv-core', 1);
      }
      if (cores <= 32) s += t(x + cw / 2, base + 12, String(c), 'hv-t hv-micro', 'middle');
    }
    s += ln(PAD + 30, base, PAD + 34 + cores * (cw + 3), base, 'hv-l');
    if (maxC >= 2) s += t(PAD + 26, base - maxC * unit + 8, String(maxC), 'hv-t hv-micro', 'end');
    s += t(PAD + 26, base, maxC >= 2 ? '0' : '1', 'hv-t hv-micro', 'end');
    s += t(PAD, base + 30, opts.foot, 'hv-t hv-dim');
    if (opts.foot2) s += t(PAD, base + 45, opts.foot2, 'hv-t hv-dim');
    return svg(W, base + (opts.foot2 ? 56 : 42), s);
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 7 · 并行请求 —— pl.pipeline 的 AIC / AIV 重叠
  // ════════════════════════════════════════════════════════════════════════════
  function pipelineLanes(d, iters, stage) {
    const W = 560, PAD = 14;
    const label = Number.isInteger(iters) ? String(iters) : iters.toFixed(2);
    const n = Math.max(2, Math.min(Math.round(iters), 7));
    const step = (W - PAD * 2 - 70) / (n + 1);
    const ya = 46, yb = 84, h = 22;
    let s = '';

    s += t(PAD, 18, 'pl.pipeline(' + label + ', stage=' + stage + ') · 预期重叠示意', 'hv-t hv-dim');
    s += t(PAD, ya + h / 2 + 3, 'AIC', 'hv-t hv-dim');
    s += t(PAD, yb + h / 2 + 3, 'AIV', 'hv-t hv-dim');

    for (let i = 0; i < n; i++) {
      const xa = PAD + 36 + i * step;
      s += r(xa, ya, step - 4, h, 'hv-aic', 2);
      s += t(xa + (step - 4) / 2, ya + h / 2 + 3, 'QK ' + i, 'hv-t hv-micro', 'middle');
      const xb = xa + step / 2;
      s += r(xb, yb, step - 4, h, 'hv-aiv', 2);
      s += t(xb + (step - 4) / 2, yb + h / 2 + 3, 'sm ' + i, 'hv-t hv-micro', 'middle');
      s += arrow(xa + (step - 4) / 2, ya + h + 2, xb + (step - 4) / 2, yb - 3, 'hv-l hv-flow');
    }
    const fy = yb + h + 24;
    s += r(PAD + 36 + step / 2, ya - 6, step - 4, (yb + h + 5) - (ya - 6), 'hv-overlap', 2);
    s += t(PAD, fy, '预期：第 i+1 次 QK 与第 i 次 softmax 重叠；非实测时间线。', 'hv-t hv-ok-text');
    s += t(PAD, fy + 15, '不需要手工展开 QK0/QK1 → sm0/sm1 —— 流水结构是声明出来的。', 'hv-t hv-dim');
    if (d.K_TILE_DB > d.L0B_CAP) {
      s += t(PAD, fy + 30, '当前 K tile 双缓冲 ' + d.K_TILE_DB_KB + ' KB 超出模型中的 Right 64 KB 限制。', 'hv-t hv-err-text');
      return svg(W, fy + 40, s);
    }
    return svg(W, fy + 26, s);
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 8 · 并行请求 —— work table 压实 + grid-stride
  // ════════════════════════════════════════════════════════════════════════════
  function gridStride(d) {
    const W = 560, PAD = 14;
    const nb = Math.min(d.BATCH, 8);
    const maxBlk = Math.max.apply(null, d.blocksPerSeq.slice(0, nb));
    const cell = Math.min(15, (W - PAD * 2 - 80) / maxBlk);
    const ya = 36, rowH = 11;
    let s = '';

    s += t(PAD, 18, 'ragged batch 的真实块（前 ' + nb + ' 条请求）', 'hv-t hv-dim');
    for (let b = 0; b < nb; b++) {
      const y = ya + b * rowH;
      s += t(PAD + 26, y + 8, 'b' + b, 'hv-t hv-micro', 'end');
      for (let p = 0; p < d.blocksPerSeq[b]; p++) {
        s += r(PAD + 32 + p * cell, y, cell - 2, rowH - 3, 'hv-core', 1);
      }
      const cap = Math.ceil(d.MAX_SEQ / d.SEQ_TILE);
      for (let p = d.blocksPerSeq[b]; p < Math.min(cap, maxBlk); p++) {
        s += r(PAD + 32 + p * cell, y, cell - 2, rowH - 3, 'hv-elide', 1);
      }
    }
    const ym = ya + nb * rowH + 12;
    s += r(PAD + 32, ym, W - PAD * 2 - 32, 22, 'hv-buf', 3);
    s += t(PAD + 42, ym + 15, 'fa_work_build 前缀和压实 → fa_work_table[w] = b × MAX_CTX_BLOCKS + p', 'hv-t');
    const fy = ym + 40;
    s += t(PAD, fy, '压实后 ' + d.REAL_BLOCKS + ' 个等价块，core = w % ' + d.NUM_CORES
      + ' 逐个取走；长度倾斜不再影响均衡。', 'hv-t hv-dim');
    s += t(PAD, fy + 15, '灰色是 padding 槽位 —— 仿射布局会把它们也排进调度，倾斜就从这里来。', 'hv-t hv-dim');
    return svg(W, fy + 26, s);
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 语义视图：分别表达片上空间、任务汇合和提交边界。
  function matmulSpaces(d) {
    let s = P.edge(200, 152, 320, 152) + P.edge(580, 152, 700, 152);
    s += P.edge(450, 108, 450, 128);
    s += P.data({ id: 'a', label: 'A · 输入块', role: 'activation', shape: [d.BATCH, 256], dtype: 'BF16' }, 40, 120);
    s += P.data({ id: 'b', label: 'B · 权重块', role: 'weight', shape: [256, 256], dtype: 'BF16' }, 370, 44);
    s += P.node('op', 'matmul', 'MatMul', 320, 128);
    s += P.data({ id: 'c', label: 'C · 结果块', role: 'activation', shape: [d.BATCH, 256], dtype: 'FP32' }, 700, 120);
    s += t(40, 238, '源码逻辑块 · 实际片上切分需查看下沉 IR', 'hv-t hv-dim');
    return infraSvg(s, 900, 260, 'MatMul 的输入、权重与累加输出');
  }

  function atomicMerge(d) {
    let s = P.edge(220, 78, 375, 134) + P.edge(220, 198, 375, 158) + P.edge(423, 146, 640, 146);
    s += P.data({ id: 'partial-0', label: 'K0 · 部分结果', role: 'activation', shape: [d.BATCH, d.TN], dtype: 'FP32' }, 60, 46);
    s += P.data({ id: 'partial-1', label: 'K1 · 部分结果', role: 'activation', shape: [d.BATCH, d.TN], dtype: 'FP32' }, 60, 166);
    s += P.node('add', 'atomic-add', '', 375, 122);
    s += P.data({ id: 'q-proj', label: 'q_proj · GM', role: 'activation', shape: [d.BATCH, d.HIDDEN], dtype: 'FP32' }, 640, 114);
    s += t(399, 198, 'atomic Add', 'hv-t hv-strong', 'middle');
    s += t(60, 270, '示例显示两个部分结果；每个输出块共有 ' + d.QKV_OK + ' 个 K 分片。', 'hv-t hv-dim');
    return infraSvg(s, 900, 292, 'Split-K 部分结果通过原子累加写入 q_proj');
  }

  function submitBoundary(d) {
    let s = t(18, 20, 'CORE_GROUP · 一次提交', 'hv-t hv-dim');
    s += r(18, 38, 145, 56, 'hv-page', 5);
    s += t(30, 60, 'deps', 'hv-t hv-strong');
    s += t(30, 80, '上游 TaskId', 'hv-t hv-dim');
    s += arrow(168, 66, 200, 66, 'hv-l hv-flow');
    s += r(205, 31, 150, 70, 'hv-aic', 5);
    s += t(217, 56, 'pl.at', 'hv-t hv-strong');
    s += t(217, 77, 'CORE_GROUP', 'hv-t');
    s += arrow(360, 66, 392, 66, 'hv-l hv-flow');
    s += r(397, 38, 145, 56, 'hv-tensor', 5);
    s += t(409, 60, 'TaskId', 'hv-t hv-strong');
    s += t(409, 80, '供下游依赖', 'hv-t hv-dim');
    s += t(18, 128, 'q_proj ' + d.Q_TASK_TOTAL + ' tasks · fa_fused ' + d.NUM_CORES + ' blocks', 'hv-t hv-dim');
    return svg(560, 144, s);
  }

  // 概念面板 —— id → { lens, kind, title, sub, facts, note, viz }
  // ════════════════════════════════════════════════════════════════════════════
  function stages(labels, caption) {
    let s = t(14, 20, caption, 'hv-t hv-dim');
    const width = (530 - (labels.length - 1) * 16) / labels.length;
    labels.forEach(function (label, i) {
      const x = 14 + i * (width + 16);
      s += r(x, 38, width, 36, i % 2 ? 'hv-aiv' : 'hv-aic', 2);
      s += t(x + width / 2, 60, label, 'hv-t hv-micro', 'middle');
      if (i < labels.length - 1) s += arrow(x + width + 2, 56, x + width + 14, 56);
    });
    return svg(560, 90, s);
  }

  // 五个代表对象使用不同的视觉结构，突出各自的硬件关系。
  function tileMap(d) {
    const count = d.QKV_K_CHUNKS;
    if (!Number.isInteger(count) || count < 1) {
      return '<div class="tile-animation-empty">当前 K 切片宽度 ' + esc(d.QKV_K_SLICE)
        + ' 不能整除 chunk 宽度 ' + d.TK + '，请调整配置后查看分块过程。</div>';
    }
    let cells = '';
    for (let i = 0; i < count; i++) cells += '<span data-tile-chunk="' + i + '">' + (i + 1) + '</span>';
    function chunkGlyph(id, role, shape, x, y) {
      let index = 0;
      return P.data({ id: id, label: '', role: role, shape: shape, dtype: 'BF16' }, x, y, { expanded: true })
        .replace(/ data-element="true"/g, function () { return ' data-element="true" data-tile-chunk="' + (index++) + '"'; })
        .replace(/<text[^>]*>[^<]*<\/text>(?=<\/g>$)/, '');
    }
    let graph = P.edge(45 + count * 24, 100, 320, 158)
      + P.edge(69, 244, 320, 178) + P.edge(580, 168, 700, 168);
    graph += chunkGlyph('q-input-chunks', 'activation', [1, count], 45, 88);
    graph += chunkGlyph('q-weight-chunks', 'weight', [count, 1], 45, 232);
    graph += P.node('op', 'q-matmul', 'MatMul / Acc', 320, 144);
    graph += P.data({ id: 'q-output', label: 'C · 部分结果', role: 'activation', shape: [d.BATCH, d.TN], dtype: 'FP32' }, 700, 136);
    graph += t(45, 66, 'A · 输入：沿 K 取列块', 'hv-t hv-strong');
    graph += t(45, 216, 'B · 权重：沿 K 取行块', 'hv-t hv-strong');
    graph += t(320, 222, '首块 MatMul · 后续 MatMul Acc', 'hv-t hv-dim');
    graph += t(45, 400, '方格代表 chunk，不表示张量元素；逻辑块 A [' + d.BATCH + ', ' + d.TK + '] × B [' + d.TK + ', ' + d.TN + ']。', 'hv-t hv-dim');
    const graphSvg = infraSvg(graph, 900, 420, 'Q 投影输入、权重分块和 MatMul 累加示意');
    let slices = '';
    for (let i = 0; i < d.QKV_OK; i++) {
      slices += i === 0
        ? '<div class="tile-map-slice is-selected" aria-label="选中的第一个 Split-K 切片"><div class="tile-map-chunks">' + cells + '</div></div>'
        : '<div class="tile-map-slice" aria-hidden="true"></div>';
    }
    return '<div class="tile-animation" data-tile-animation data-count="' + count + '" data-tk="' + d.TK + '">'
      + '<div class="tile-animation-top"><span>Q PROJECTION · 分块计算</span><span data-tile-counter></span></div>'
      + '<div class="tile-location"><div class="tile-location-heading"><b>当前计算的位置</b><span>示例：q_ks = 0 · q_nt = 0 · n_sub = 0</span></div>'
      + '<div class="tile-axis-heading"><span>K 维 · HIDDEN = ' + d.HIDDEN + '</span><span>' + d.QKV_OK + ' 个 Split-K 切片</span></div>'
      + '<div class="tile-map-slices">' + slices + '</div>'
      + '<div class="tile-axis-ends"><span>0</span><span>' + d.HIDDEN + '</span></div>'
      + '<div class="tile-location-detail"><span>高亮：第 1 个切片，宽 ' + d.QKV_K_SLICE + '</span><span data-tile-location></span></div>'
      + '<div class="tile-axis-heading"><span>N 维 · 输出宽度 ' + d.HIDDEN + '</span><span>一个 N 子块 = ' + d.TN + '</span></div>'
      + '<div class="tile-n-track"><span style="width:' + (d.TN / d.HIDDEN * 100) + '%"></span></div>'
      + '<div class="tile-axis-ends"><span>0 · 当前输出子块</span><span>' + d.HIDDEN + '</span></div>'
      + '<p>下方展示选中 K 切片中的各 chunk，计算结果累加到选中的 N 子块。</p></div>'
      + '<div class="tile-glyph-diagram">' + graphSvg + '</div>'
      + '<div class="tile-glyph-progress"><span>已累加 <b data-tile-acc>0</b> / ' + count + ' 个 chunk</span><div class="tile-acc-track"><i data-tile-progress></i></div></div>'
      + '<div class="tile-animation-step" data-tile-description aria-live="off"></div>'
      + '<div class="tile-animation-controls"><button type="button" data-tile-action="toggle">暂停</button>'
      + '<button type="button" data-tile-action="prev">上一步</button><button type="button" data-tile-action="next">下一步</button>'
      + '<span>逻辑分块示意 · 实际片上切分需查看编译产物</span></div></div>';
  }

  function dependencyMap(d) {
    let s = P.edge(300, 76, 540, 140) + P.edge(300, 204, 540, 140);
    s += P.node('task', 'slab', 'normed slab', 40, 46);
    s += P.node('task', 'seed', 'q_seed', 40, 174);
    s += P.node('task', 'q-proj', 'q_proj', 540, 110);
    s += t(40, 126, 'prev_normed_tids[q_ks]', 'hv-t hv-dim');
    s += t(40, 254, 'q_seed_tid · 清零目标', 'hv-t hv-dim');
    s += t(540, 194, 'deps=[slab tid, seed tid]', 'hv-t hv-dim');
    const aligned = d.QKV_K_SLICE === d.DOWN_TN;
    s += t(40, 294, '切片 ' + d.QKV_K_SLICE + ' / slab ' + d.DOWN_TN + ' · '
      + (aligned ? '一条 slab 依赖对应一个切片' : '宽度不一致，需检查依赖覆盖'), aligned ? 'hv-t hv-ok-text' : 'hv-t hv-err-text');
    return infraSvg(s, 900, 310, 'Q 投影任务的显式依赖');
  }

  function handoffMap() {
    let s = t(16, 20, 'ATTENTION  /  数据交接示意 · 非实测', 'hv-t hv-dim');
    s += t(16, 62, 'CUBE', 'hv-t hv-strong');
    s += t(16, 141, 'VECTOR', 'hv-t hv-strong');
    s += ln(88, 65, 544, 65, 'hv-l hv-dash');
    s += ln(88, 144, 544, 144, 'hv-l hv-dash');
    s += r(98, 39, 126, 53, 'hv-aic', 5);
    s += t(110, 59, '01  Q × K', 'hv-t hv-strong');
    s += t(110, 77, 'scores · FP32', 'hv-t');
    s += r(248, 117, 144, 53, 'hv-aiv', 5);
    s += t(260, 137, '02  Softmax', 'hv-t hv-strong');
    s += t(260, 155, 'max → exp → cast', 'hv-t');
    s += r(416, 39, 128, 53, 'hv-aic', 5);
    s += t(428, 59, '03  exp × V', 'hv-t hv-strong');
    s += t(428, 77, 'oi_tmp · FP32', 'hv-t');
    s += arrow(224, 80, 266, 115, 'hv-l hv-flow');
    s += arrow(390, 125, 430, 94, 'hv-l hv-flow');
    s += t(16, 199, 'C2V：scores 交给 Vector  ·  V2C：exp 交回 Cube', 'hv-t');
    s += t(16, 216, 'stage=2 表达迭代重叠意图；实际执行需以 trace 验证', 'hv-t hv-dim');
    return svg(560, 230, s);
  }

  function validRegionMap(d) {
    let s = t(16, 20, 'SCORES TILE  /  物理范围与有效范围', 'hv-t hv-dim');
    const x = 20, y = 49, w = 260, h = 130;
    const ratio = Math.max(0, Math.min(1, d.Q_HEAD_BATCH / d.Q_HEAD_PAD));
    s += r(x, y, w, h, 'hv-row-pad', 3);
    s += r(x, y, w, h, 'hv-row-hatch', 3);
    s += r(x, y, w * 0.65, h * ratio, 'hv-row-real', 2);
    s += t(x + 8, y + 16, '有效 scores', 'hv-t hv-strong');
    s += t(x + w / 2, y - 9, '物理列：SEQ_TILE = ' + d.SEQ_TILE, 'hv-t hv-dim', 'middle');
    s += t(300, 65, '物理行  ' + d.Q_HEAD_PAD, 'hv-t hv-strong');
    s += t(300, 88, '有效行  ' + d.Q_HEAD_BATCH, 'hv-t hv-ok-text');
    s += t(300, 111, '有效列  valid_len（运行时）', 'hv-t');
    s += t(300, 137, '阴影 = 无效区域', 'hv-t hv-dim');
    s += t(300, 156, '图中列宽为示意', 'hv-t hv-dim');
    s += r(16, 198, 528, 51, 'hv-buf', 4);
    s += t(28, 218, 'set_validshape → fillpad(PadValue.min) → row_max', 'hv-t hv-strong');
    s += t(28, 237, '填充值避免无效位置干扰有效区域的最大值；行填充占 ' + d.PAD_WASTE_PCT + '%', 'hv-t hv-dim');
    return svg(560, 263, s);
  }

  function precisionMap() {
    let s = t(16, 20, 'ATTENTION  /  数据类型随计算变化', 'hv-t hv-dim');
    const steps = [
      ['Q / K', 'BF16', '输入'], ['QK scores', 'FP32', 'matmul 输出'],
      ['exp_scores', 'FP32', 'Softmax'], ['exp_scores', 'BF16', '显式 cast'], ['oi_tmp', 'FP32', 'SV 输出']
    ];
    steps.forEach(function (v, i) {
      const x = 16 + i * 106;
      s += r(x, 40, 96, 94, i === 3 ? 'hv-aiv' : (v[1] === 'FP32' ? 'hv-aic' : 'hv-page'), 5);
      s += t(x + 8, 59, v[0], 'hv-t hv-micro');
      s += t(x + 8, 85, v[1], 'hv-t hv-viz-big');
      s += t(x + 8, 112, v[2], 'hv-t hv-dim');
      if (i < steps.length - 1) s += arrow(x + 97, 87, x + 105, 87, 'hv-l hv-flow');
    });
    s += r(16, 151, 520, 56, 'hv-buf', 4);
    s += t(28, 171, '另一条路径：残差 FP32 → 乘 gamma → cast BF16', 'hv-t hv-strong');
    s += t(28, 190, '转换位置来自源码；精度影响需数值对比验证', 'hv-t hv-dim');
    return svg(560, 221, s);
  }
  const CONCEPTS = {
    'scope:deps': {
      lens: 'scope', kind: '依赖', title: 'deps 与 Task ID', sub: '显式任务依赖',
      facts: function () { return [['前置任务', '对应 normed slab 与 q_seed'], ['依赖表达', 'deps=[prev_normed_tids[q_ks], q_seed_tid]'], ['作用', '等待数据准备和目标清零'], ['前提', 'split-K 切片与 slab 的对应关系成立']]; },
      note: 'manual_scope 中通过 Task ID 明确表达任务间顺序。',
      viz: function (d) { return dependencyMap(d); }
    },
    'scope:atomic': {
      lens: 'parallel', kind: '并发写入', title: 'Split-K 原子累加', sub: '清零 → 并行部分结果 → 汇合',
      facts: function (d) { return [['切片数', String(d.QKV_OK)], ['目标', 'q_proj · FP32'], ['写入', 'pl.assemble(..., atomic=pl.AtomicType.Add)'], ['顺序', 'q_seed 先清零；各 task 依赖 q_seed'], ['注意', '汇合后的消费者仍需等待对应生产任务']]; },
      note: '原子加保护并发累加；任务依赖负责表达清零及后续消费的顺序。',
      viz: function (d) { return atomicMerge(d); }
    },
    'tile:v-tile': {
      lens: 'tile', kind: 'Tile', title: 'v_tile', sub: 'KV 物理页 → SV matmul',
      facts: function (d) { return [['切法', 'v_cache[cache_row : cache_row + SEQ_TILE, :]'], ['形状', '[' + d.SEQ_TILE + ', ' + d.HEAD_DIM + '] BF16'], ['来源', '与 k_tile 使用同一物理行范围'], ['消费者', 'exp_scores_bf16 × v_tile → oi_tmp FP32']]; },
      note: 'K 页用于 QK，V 页用于 SV；二者通过同一个 cache_row 定位。',
      viz: function (d) { return pagedKV(d); }
    },
    'parallel:handoff': {
      lens: 'parallel', kind: '计算单元', title: 'QK → Softmax → SV', sub: 'Cube / Vector 交接与流水意图',
      facts: function () { return [['QK', 'Cube：Q × K，FP32 scores'], ['Softmax', 'Vector：缩放、填充、row_max 与 exp'], ['SV', 'Cube：BF16 exp × V，FP32 输出'], ['交接', 'QK → Vector；exp → Cube'], ['状态', '源码表达预期结构，非真机执行结果']]; },
      note: '这是源码中的阶段与交接意图，实际排布及重叠需要编译产物和设备 trace 验证。',
      viz: function () { return handoffMap(); }
    },
    'precision:cast': {
      lens: 'precision', kind: '数值与精度', title: 'FP32 / BF16 转换边界', sub: '计算与存储的数据类型',
      facts: function () { return [['QK 输出', 'FP32 scores'], ['Softmax 输出', 'exp_scores 转 BF16，供 SV matmul 使用'], ['SV 输出', 'oi_tmp 使用 FP32'], ['残差路径', '残差保留 FP32；归一化后的 MLP 输入转 BF16'], ['验证', '精度影响需以数值对比验证']]; },
      note: '转换位置由源码表达；不应仅凭 dtype 推断精度损失或性能收益。',
      viz: function () {
        return precisionMap();
      }
    },
    'tensor:hidden-states': {
      lens: 'tensor', kind: 'Tensor', title: 'hidden_states', sub: '跨层 FP32 carry',
      facts: function (d) {
        return [
          ['形状', '[' + d.BATCH + ', ' + d.HIDDEN + ']'],
          ['dtype', 'FP32（跨层不做 BF16 round-trip）'],
          ['内存空间', 'GM · ' + d.HIDDEN_BYTES],
          ['写者', 'DOWN_ON = ' + Math.round(d.DOWN_ON) + ' 个 dcr_xgamma slab 写者'],
          ['读者', 'rms_recip · residual · post_rms']
        ];
      },
      note: '形状与 dtype 写在签名里，不是运行时才知道的事。残差流保持 FP32，'
        + '只在首层入口与末层出口各做一次 cast。',
      viz: function (d) {
        return tensorTile({
          cols: d.RMS_CHUNKS, hot: 0,
          topLabel: 'HIDDEN = ' + d.HIDDEN + '（' + d.RMS_CHUNKS + ' × RMSNORM_K_CHUNK 256）',
          rowsLabel: 'BATCH ' + d.BATCH,
          tileLabel: '256',
          spaceLabel: 'GM · FP32',
          dest: 'Vec (UB)',
          destBytes: d.bytes(d.BATCH * 256 * 4),
          foot: 'rms_recip 以 256 列为一拍读入，stage=4 流水累加 sum-of-squares。'
        });
      }
    },
    'tensor:normed': {
      lens: 'tensor', kind: 'Tensor', title: 'normed_in / normed_out', sub: '层间 x·gamma 直传',
      facts: function (d) {
        return [
          ['形状', '[' + d.BATCH + ', ' + d.HIDDEN + '] BF16'],
          ['内存空间', 'GM · ' + d.NORMED_BYTES],
          ['生产者', '上一层 dcr_xgamma（与 out 同一次写出）'],
          ['消费者', '本层 QKV 投影'],
          ['slab 宽度', 'DOWN_TN = ' + d.DOWN_TN]
        ];
      },
      note: '把下一层的 x·gamma 和本层的残差输出放在同一个 task 里产出，'
        + '省掉一次 GM 往返；代价是依赖边必须按 slab 粒度显式写出。',
      viz: function (d) { return splitK(d); }
    },
    'tensor:q-proj': {
      lens: 'tensor', kind: 'Tensor', title: 'q_proj', sub: 'split-K atomic 累加目标',
      facts: function (d) {
        return [
          ['形状', '[' + d.BATCH + ', ' + d.HIDDEN + '] FP32'],
          ['内存空间', 'GM scratch · ' + d.QPROJ_BYTES],
          ['写者', d.Q_TASK_TOTAL + ' 个 task（' + d.Q_ON + ' N-tile × ' + d.QKV_OK + ' split-K）'],
          ['写入方式', 'pl.assemble(..., atomic=AtomicType.Add)'],
          ['种子', 'q_seed 先零填充']
        ];
      },
      note: 'split-K 的分片结果靠硬件 atomic add 在同一块 GM 上汇总，'
        + '所以这些 task 可以真并行，而不是串成归约树。',
      viz: function (d) { return atomicMerge(d); }
    },
    'tile:x-chunk': {
      lens: 'tile', kind: 'Tile', title: 'x_chunk', sub: 'hidden_states 的一列块',
      facts: function (d) {
        return [
          ['切法', 'hidden_states[:, k0 : k0 + 256]'],
          ['形状', '[' + d.BATCH + ', 256] FP32'],
          ['片上空间', 'Vec (UB) · ' + d.bytes(d.BATCH * 256 * 4)],
          ['迭代数', d.RMS_CHUNKS + ' 个 chunk'],
          ['流水', 'stage=4']
        ];
      },
      note: 'Tile 是张量上的切片表达式，不是另一种对象。切出来之后它落在哪个片上空间，'
        + '由消费它的指令决定。',
      viz: function (d) {
        return tensorTile({
          cols: d.RMS_CHUNKS, hot: 3,
          topLabel: 'k0 = kb × 256',
          rowsLabel: 'BATCH ' + d.BATCH,
          tileLabel: '256',
          spaceLabel: 'GM · FP32',
          dest: 'Vec (UB)',
          destBytes: d.bytes(d.BATCH * 256 * 4),
          foot: 'row_sum(mul(x_chunk, x_chunk)) 在 AIV 上做，结果累加到 [1, BATCH] 的 partial_sq。'
        });
      }
    },
    'tile:q-acc': {
      lens: 'tile', kind: 'Tile', title: 'pl.matmul 的三块 tile', sub: '逻辑分块与片上映射',
      facts: function (d) {
        return [
          ['A 逻辑块', '[' + d.BATCH + ', 256] BF16 · ' + d.bytes(d.BATCH * 256 * 2)],
          ['B 逻辑块', '[256, 256] BF16 · ' + d.bytes(256 * 256 * 2)],
          ['C 逻辑块', '[' + d.BATCH + ', 256] FP32 · ' + d.bytes(d.BATCH * 256 * 4)],
          ['K chunk 数', d.QKV_K_CHUNKS + ' × TK 256'],
          ['片上映射', 'Left / Right / Acc；具体切分需查看下沉 IR']
        ];
      },
      note: '图中大小来自源码的逻辑块，不代表完整块同时驻留片上。B 逻辑块为 128 KiB，高于示例 Right 容量 64 KiB；实际切分和驻留须查看目标的下沉 IR 或生成代码。',
      viz: function (d) { return tileMap(d); }
    },
    'tile:k-tile': {
      lens: 'tile', kind: 'Tile', title: 'k_tile', sub: 'paged KV 上的一个物理页',
      facts: function (d) {
        return [
          ['切法', 'k_cache[cache_row : cache_row + SEQ_TILE, :]'],
          ['形状', '[' + d.SEQ_TILE + ', ' + d.HEAD_DIM + '] BF16'],
          ['逻辑块大小', d.K_TILE_KB + ' KB'],
          ['双缓冲估算', d.K_TILE_DB_KB + ' KB / 示例容量 64 KiB'],
          ['来源', 'block_table 查到的物理页']
        ];
      },
      note: 'SEQ_TILE 同时是三件事：KV 块长度、paged 物理页大小、Right buffer 的占用量。'
        + '调它会同时动三条约束 —— 这正是需要前移的地方。',
      viz: function (d) { return pagedKV(d); }
    },
    'scope:manual': {
      lens: 'scope', kind: 'Scope', title: 'pl.manual_scope', sub: '依赖语义的边界',
      facts: function (d) {
        return [
          ['作用', 'tensormap / auto-dep 注册被抑制'],
          ['成边方式', '只有显式 deps=[tid] 生效'],
          ['跨层句柄', 'prev_out_tids / prev_normed_tids · 各 ' + Math.round(d.DOWN_ON) + ' 个'],
          ['换来的', 'dcr slab → x_gamma 的细粒度边'],
          ['代价', '漏写一条 deps 就是数据竞争']
        ];
      },
      note: 'auto-dep 的粒度是整张父张量，会把 slab 级的并行压成 WAW 串行。'
        + 'manual_scope 把成边权交回代码，于是「谁依赖谁」变成可读、可审的一行。',
      viz: function (d) { return scopeDeps(d); }
    },
    'scope:at': {
      lens: 'scope', kind: 'Scope', title: 'pl.at · Level.CORE_GROUP', sub: '一次 submit 的粒度',
      facts: function (d) {
        return [
          ['level', 'CORE_GROUP'],
          ['产出', '一个 TaskId（可存进 pl.Array）'],
          ['name_hint', '在 trace 与 IR dump 里就是这个名字'],
          ['本层 q_proj', d.Q_TASK_TOTAL + ' 个 task'],
          ['本层 fa_fused', d.NUM_CORES + ' 个常驻 block']
        ];
      },
      note: 'task 的边界是写出来的，不是编译器猜的。name_hint 让 profile 里的泳道'
        + '和源码里的 with 一一对应。',
      viz: function (d) { return submitBoundary(d); }
    },
    'parallel:parallel': {
      lens: 'parallel', kind: '并行请求', title: 'pl.parallel', sub: '展开成独立 task',
      facts: function (d) {
        return [
          ['并行度', d.Q_ON + '（HIDDEN / QKV_N_TILE）'],
          ['内层', 'pl.range(' + d.QKV_OK + ') split-K'],
          ['task 总数', String(d.Q_TASK_TOTAL)],
          ['汇总', 'atomic Add 回 q_proj'],
          ['依赖', '每个 task 各自 deps=']
        ];
      },
      note: 'pl.parallel 的每次迭代是一个独立的调度请求；pl.range 留在 task 内部。'
        + '两者的区别直接决定 task 数量与依赖图形状。',
      viz: function (d) {
        const cores = Math.min(d.Q_TASK_TOTAL, 50);
        const counts = new Array(cores).fill(1);
        return coreFan(d, {
          cores: cores, counts: counts,
          title: d.Q_ON + ' N-tile × ' + d.QKV_OK + ' split-K 展开为独立请求',
          right: d.Q_TASK_TOTAL + ' tasks',
          foot: '每个 task 读一片 K、写一个 N-tile，彼此只通过 atomic add 汇合。'
        });
      }
    },
    'parallel:spmd': {
      lens: 'parallel', kind: '并行请求', title: 'pl.spmd', sub: '固定栅格 + 块内取号',
      facts: function (d) {
        return [
          ['fa_fused 栅格', d.NUM_CORES + ' blocks（常驻）'],
          ['rope_qkv 栅格', d.ROPE_CORES + ' blocks'],
          ['rope items', d.ROPE_ITEMS + ' → 每核 ' + d.ROPE_ITEMS_PER_CORE],
          ['取号', 'pl.get_block_idx()'],
          ['切分', 'pl.split(SplitMode.UP_DOWN)']
        ];
      },
      note: 'spmd 把 N 个请求塌缩成 N 个常驻 block，调度开销从「每个 work item 一次」'
        + '变成「每个核一次」。代价是栅格必须整除 item 数，或者自己写 grid-stride。',
      viz: function (d) {
        const counts = new Array(d.ROPE_CORES).fill(Math.floor(d.ROPE_ITEMS / d.ROPE_CORES));
        for (let i = 0; i < d.ROPE_ITEMS % d.ROPE_CORES; i++) counts[i] += 1;
        const even = d.ROPE_ITEMS % d.ROPE_CORES === 0;
        return coreFan(d, {
          cores: d.ROPE_CORES, counts: counts, flagMax: !even,
          title: 'rope_qkv · ' + d.ROPE_ITEMS + ' 个 (head, batch) item',
          right: d.ROPE_CORES + ' blocks',
          foot: even
            ? '整除：每核 ' + d.ROPE_ITEMS_PER_CORE + ' 个 item，仿射索引不需要兜底分支。'
            : '不整除：余 ' + (d.ROPE_ITEMS % d.ROPE_CORES) + ' 个 item 落在仿射索引之外。',
          foot2: 'rope 用的是固定 item/core 的仿射索引，不做 grid-stride。'
        });
      }
    },
    'parallel:pipeline': {
      lens: 'parallel', kind: '并行请求', title: 'pl.pipeline', sub: '声明式软流水',
      facts: function (d) {
        return [
          ['rms_recip', d.RMS_CHUNKS + ' 迭代 · stage=4'],
          ['q_proj K chunk', d.QKV_K_CHUNKS + ' 迭代 · stage=2'],
          ['fa_fused gp', d.GP_SIZE + ' 迭代 · stage=2'],
          ['重叠对象', 'AIC(QK) ↔ AIV(softmax)'],
          ['通道', 'C2V / V2C']
        ];
      },
      note: '代码声明流水重叠意图，实际排布需通过编译产物和设备 trace 验证；本源码注明该版本预期代码生成失败。',
      viz: function (d) { return pipelineLanes(d, d.GP_SIZE, 2); }
    },
    'parallel:grid-stride': {
      lens: 'parallel', kind: '并行请求', title: 'grid-stride 块级均衡', sub: '设备侧游标',
      facts: function (d) {
        return [
          ['work item', '1 个 SEQ_TILE 块 × ' + d.GP_SIZE + ' 头'],
          ['真实块数', String(d.REAL_BLOCKS)],
          ['表容量', d.FA_TABLE_CAP_FMT],
          ['取号', 'core = w % ' + d.NUM_CORES],
          ['均衡偏差', d.SKEW_FMT + '× 理想']
        ];
      },
      note: 'fa_work_build 先把 ragged batch 的真实块压成无空洞的表，fa_fused 再 grid-stride 取号。'
        + '每个 item 等价，于是均衡度只剩整数打包这一项误差。',
      viz: function (d) { return gridStride(d); }
    },
    'layout:q-pad': {
      lens: 'layout', kind: '内部布局', title: 'Q_HEAD_PAD 与 set_validshape', sub: '物理行 vs 有效行',
      facts: function (d) {
        return [
          ['真实 Q 头', String(d.Q_HEAD_BATCH)],
          ['物理行距', String(d.Q_HEAD_PAD)],
          ['填充占比', d.PAD_WASTE_PCT + '%'],
          ['有效标记', 'set_validshape(·, ' + d.Q_HEAD_BATCH + ', valid_len)'],
          ['填充值', 'fillpad(PadValue.min)'],
          ['Softmax', '无效位置使用最小填充值，避免影响有效区域的 row_max']
        ];
      },
      note: 'tile 物理上保持 ' + '16 行以满足 cube M fractal；set_validshape 只把真实行标为有效，'
        + '搬运与计算的差额因此是可见的、可算的，而不是一个隐藏的性能黑洞。',
      viz: function (d) { return validRegionMap(d); }
    },
    'layout:paged-kv': {
      lens: 'layout', kind: '内部布局', title: 'paged KV 的行寻址', sub: '逻辑块 → 物理页',
      facts: function (d) {
        return [
          ['SEQ_TILE', String(d.SEQ_TILE)],
          ['runtime.page_size', String(d.PAGE_SIZE)],
          ['逻辑块 / 请求', String(d.MAX_CTX_BLOCKS)],
          ['行寻址', '(page × KV_HEADS + kvh) × BLOCK_SIZE + offset'],
          ['查表', 'block_table[b × max_blocks + blk]']
        ];
      },
      note: 'cache_row 的算术成立，前提是一个逻辑块正好等于一个物理页。'
        + '这条等式在代码里是一行常量赋值，不满足时问题会跑到运行时才炸。',
      viz: function (d) { return pagedKV(d); }
    },
    'layout:split-k': {
      lens: 'layout', kind: '内部布局', title: 'split-K 切片与 slab 对齐', sub: '依赖边的几何前提',
      facts: function (d) {
        return [
          ['QKV_OK', String(d.QKV_OK)],
          ['QKV_K_SLICE', String(d.QKV_K_SLICE)],
          ['DOWN_TN (slab)', String(d.DOWN_TN)],
          ['DOWN_ON', String(Math.round(d.DOWN_ON))],
          ['对齐', d.QKV_K_SLICE === d.DOWN_TN ? '1:1 成立' : '不成立']
        ];
      },
      note: 'split-K 的切片宽度等于上一层输出 slab 的宽度时，deps=[prev_normed_tids[q_ks]] '
        + '才是一条精确的边。这是一个纯几何条件，所以可以在编码时就判定。',
      viz: function (d) { return splitK(d); }
    }
  };

  return { CONCEPTS: CONCEPTS, esc: esc };
})();
