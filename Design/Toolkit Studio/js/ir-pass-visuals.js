/* Pass visuals — what a pass DOES to the computation, drawn rather than diffed.
   Each renderer returns { svg, caption } and is keyed by pass name. All numbers
   come from the real dump via the ctx passed in; nothing here is decorative. */
(function () {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';
  const W = 520;                       // viewBox width; height varies per visual
  const esc = (s) => String(s).replace(/[&<>"']/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const kb = (b) => b >= 1024 ? (b / 1024).toFixed(b >= 10240 ? 0 : 1) + 'KB' : b + 'B';

  // Palette hooks resolve against the page's design tokens.
  const C = {
    tensor: 'var(--irp-s1)', tile: 'var(--irp-s3)',
    aic: '#CE5622', aiv: '#E09258', group: '#BC7440', orch: '#4E7C90',
    ok: 'var(--success)', warn: 'var(--warning)', bad: 'var(--danger)',
    pri: 'var(--primary)', muted: 'var(--foreground-muted)',
    line: 'var(--border-subtle)', strong: 'var(--border-strong)',
    fg: 'var(--foreground)', fg2: 'var(--foreground-secondary)',
    surf: 'var(--surface-2)', surf3: 'var(--surface-3)'
  };
  const SPACE_C = { Vec: '#4E7C90', Mat: '#A2814F', Acc: '#CE5622', Left: '#7BB088', Right: '#BC7440' };

  const svg = (h, body) =>
    '<svg class="kf-kg-vis" viewBox="0 0 ' + W + ' ' + h + '" role="img" ' +
    'preserveAspectRatio="xMidYMid meet">' + body + '</svg>';

  const t = (x, y, s, o) => {
    o = o || {};
    return '<text x="' + x + '" y="' + y + '" fill="' + (o.fill || C.muted) + '" ' +
      'font-size="' + (o.size || 9) + '" font-family="var(--font-mono)" ' +
      'text-anchor="' + (o.anchor || 'start') + '"' +
      (o.weight ? ' font-weight="' + o.weight + '"' : '') + '>' + esc(s) + '</text>';
  };
  const rect = (x, y, w, h, o) => {
    o = o || {};
    return '<rect x="' + x + '" y="' + y + '" width="' + Math.max(w, 0.6) + '" height="' + Math.max(h, 0.6) + '" ' +
      'fill="' + (o.fill || 'none') + '"' +
      (o.stroke ? ' stroke="' + o.stroke + '" stroke-width="' + (o.sw || 1) + '"' : '') +
      (o.rx ? ' rx="' + o.rx + '"' : '') +
      (o.op ? ' opacity="' + o.op + '"' : '') + '/>';
  };
  const line = (x1, y1, x2, y2, o) => {
    o = o || {};
    return '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '" ' +
      'stroke="' + (o.stroke || C.line) + '" stroke-width="' + (o.sw || 1) + '"' +
      (o.dash ? ' stroke-dasharray="' + o.dash + '"' : '') +
      (o.op ? ' opacity="' + o.op + '"' : '') + '/>';
  };
  const arrow = (x1, y, x2, o) => {
    o = o || {};
    const c = o.stroke || C.strong, d = x2 > x1 ? -4 : 4;
    return line(x1, y, x2, y, o) +
      '<path d="M' + x2 + ' ' + y + ' l' + d + ' -2.6 l0 5.2 z" fill="' + c + '"/>';
  };
  const label = (x, y, s) => t(x, y, s, { size: 8, fill: C.muted });

  /* ---------------- helpers shared by several visuals ---------------- */

  // (address × lifetime) packing chart — the shape MemoryReuse / AllocateMemoryAddr act on
  function packing(bufs, opt) {
    opt = opt || {};
    const H = opt.h || 116, padL = 40, padR = 8, padT = 14, padB = 16;
    if (!bufs.length) return { body: t(W / 2, H / 2, '该 kernel 没有片上分配', { anchor: 'middle' }), h: H };

    const spaces = [...new Set(bufs.map(b => b.sp))];
    const sp = opt.space || spaces.sort((a, b) =>
      Math.max(...bufs.filter(x => x.sp === b).map(x => x.off + x.sz)) -
      Math.max(...bufs.filter(x => x.sp === a).map(x => x.off + x.sz)))[0];
    const rows = bufs.filter(b => b.sp === sp);
    const maxT = Math.max(...rows.map(b => b.off + b.sz), 1);
    const maxE = Math.max(...rows.map(b => b.e), 1);
    const top = opt.limit ? Math.max(maxT, opt.limit) : maxT;

    const gx = (i) => padL + (i / maxE) * (W - padL - padR);
    const gy = (v) => padT + (1 - v / top) * (H - padT - padB);

    let body = '';
    // frame + address ticks
    body += line(padL, gy(0), W - padR, gy(0), { stroke: C.strong });
    body += line(padL, padT, padL, gy(0), { stroke: C.strong });
    [0, 0.5, 1].forEach(f => {
      const v = top * f;
      body += line(padL, gy(v), W - padR, gy(v), { stroke: C.line, dash: '2 3', op: 0.5 });
      body += t(padL - 4, gy(v) + 3, kb(v), { anchor: 'end', size: 7.5 });
    });
    if (opt.limit && opt.limit <= top) {
      body += line(padL, gy(opt.limit), W - padR, gy(opt.limit), { stroke: C.bad, sw: 1.2, dash: '4 3' });
      body += t(W - padR, gy(opt.limit) - 3, '平台上限 ' + kb(opt.limit), { anchor: 'end', size: 7.5, fill: C.bad });
    }
    // buffers
    rows.forEach(b => {
      const x = gx(b.s), w = Math.max(gx(b.e) - gx(b.s), 2);
      const y = gy(b.off + b.sz), h = Math.max(gy(b.off) - y, 1.5);
      body += rect(x, y, w, h, { fill: SPACE_C[b.sp] || C.pri, op: 0.82, rx: 1 });
      body += rect(x, y, w, h, { stroke: 'var(--surface-1)', sw: 0.5 });
    });
    body += t(padL, padT - 5, sp + ' · ' + rows.length + ' 块 · 峰值 ' + kb(maxT), { size: 8, fill: C.fg2 });
    body += t(W - padR, H - 4, '语句序 →', { anchor: 'end', size: 7.5 });
    return { body, h: H, space: sp, count: rows.length, peak: maxT };
  }

  function fanout(n, from, to, colors) {
    const H = 92, cx = 62, cy = H / 2;
    let body = rect(cx - 40, cy - 13, 80, 26, { fill: C.surf, stroke: C.strong, rx: 3 }) +
      t(cx, cy + 3.5, from, { anchor: 'middle', size: 9, fill: C.fg2 });
    const shown = Math.min(n, 14), x0 = 210, gap = (W - x0 - 12) / shown;
    for (let i = 0; i < shown; i += 1) {
      const y = 14 + (H - 30) * (i / Math.max(shown - 1, 1));
      const c = colors ? colors(i) : C.tile;
      body += '<path d="M' + (cx + 42) + ' ' + cy + ' C 150 ' + cy + ', 175 ' + y + ', ' + (x0 - 4) + ' ' + y +
        '" fill="none" stroke="' + C.line + '" stroke-width="0.8" opacity="0.7"/>';
      body += rect(x0, y - 4.5, gap * 0.72 + 26, 9, { fill: c, op: 0.85, rx: 2 });
    }
    if (n > shown) body += t(x0, H - 2, '… 共 ' + n + ' 个', { size: 8 });
    body += t(W - 8, 10, to, { anchor: 'end', size: 9, fill: C.fg2 });
    return { body, h: H };
  }

  /* ---------------- per-pass renderers ---------------- */

  const V = {};

  V.ConvertTensorToTileOps = (ctx) => {
    const a = ctx.prev.o, b = ctx.pass.o, H = 96;
    const bar = (y, te, ti, tag) => {
      const tot = Math.max(te + ti, 1), x0 = 64, w = W - x0 - 68;
      const wt = (te / tot) * w, wi = (ti / tot) * w;
      return t(x0 - 6, y + 10, tag, { anchor: 'end', size: 8.5, fill: C.fg2 }) +
        rect(x0, y, wt, 14, { fill: C.tensor, rx: 1 }) +
        rect(x0 + wt, y, wi, 14, { fill: C.tile, rx: 1 }) +
        (wt > 30 ? t(x0 + 5, y + 10, 'tensor ' + te, { size: 8, fill: '#fff' }) : '') +
        (wi > 30 ? t(x0 + wt + 5, y + 10, 'tile ' + ti, { size: 8, fill: '#fff' }) : '') +
        t(W - 62, y + 10, tot + ' 个算子', { size: 8 });
    };
    let body = bar(20, a.te, a.ti, '这一步前') + bar(52, b.te, b.ti, '这一步后');
    body += arrow(W / 2 - 10, 42, W / 2 + 10, { stroke: C.strong });
    body += t(8, 84, 'InCore 内部整套算子词汇被替换：tensor.slice / assemble → tile.load / store，值的类型从 TensorType 变成 TileType',
      { size: 8, fill: C.muted });
    return { svg: svg(H, body), caption: 'tensor 算子 ' + a.te + ' → ' + b.te + '，tile 算子 ' + a.ti + ' → ' + b.ti };
  };

  V.AutoTileMatmulL0 = (ctx) => {
    const k = ctx.kernels.filter(x => x.intent.l0.length)[0];
    if (!k) return null;
    const tl = k.intent.l0[0];
    const segs = Math.max(1, Math.ceil((tl.to - tl.from) / tl.step));
    const H = 128, y0 = 26, bh = 46;
    const aw = 118, bw = 118, cw = 118, gap = 40;
    const ax = 42, bx = ax + aw + gap, cx = bx + bw + gap;

    let body = '';
    // A[M,K] sliced along K
    body += t(ax, y0 - 7, 'A [M, K=' + tl.to + ']', { size: 8, fill: C.fg2 });
    for (let i = 0; i < segs; i += 1) {
      const w = aw / segs;
      body += rect(ax + i * w, y0, w, bh, { fill: C.tensor, op: i === 0 ? 0.95 : 0.42, rx: 1 });
      body += rect(ax + i * w, y0, w, bh, { stroke: 'var(--surface-1)', sw: 0.8 });
    }
    body += t(ax + aw / 2, y0 + bh + 11, 'K 轴切 ' + segs + ' 段 · 每段 ' + tl.step, { anchor: 'middle', size: 8 });
    body += t(ax - 5, y0 + bh / 2 + 3, 'M', { anchor: 'end', size: 8 });

    // B[K,N] sliced along K (rows)
    body += t(bx, y0 - 7, 'B [K=' + tl.to + ', N]', { size: 8, fill: C.fg2 });
    for (let i = 0; i < segs; i += 1) {
      const h = bh / segs;
      body += rect(bx, y0 + i * h, bw, h, { fill: C.tile, op: i === 0 ? 0.95 : 0.42, rx: 1 });
      body += rect(bx, y0 + i * h, bw, h, { stroke: 'var(--surface-1)', sw: 0.8 });
    }
    body += t(bx + bw / 2, y0 + bh + 11, 'L0B 每次载入 1 段', { anchor: 'middle', size: 8 });

    // C accumulator, stationary
    body += t(cx, y0 - 7, 'C [M, N] · L0C', { size: 8, fill: C.fg2 });
    body += rect(cx, y0, cw, bh, { fill: C.aic, op: 0.9, rx: 1 });
    body += t(cx + cw / 2, y0 + bh / 2 + 3, '常驻累加', { anchor: 'middle', size: 8.5, fill: '#fff' });
    body += t(cx + cw / 2, y0 + bh + 11, segs + ' 次 matmul_acc', { anchor: 'middle', size: 8 });

    body += t(ax + aw + 12, y0 + bh / 2 + 3, '×', { size: 12, fill: C.muted });
    body += t(bx + bw + 12, y0 + bh / 2 + 3, '=', { size: 12, fill: C.muted });

    body += t(8, H - 20, '一条 tile.matmul 被改写成 K 循环：第 1 次 matmul，之后 ' + (segs - 1) + ' 次 matmul_acc 累加进常驻的 C。',
      { size: 8, fill: C.muted });
    body += t(8, H - 8, 'stage=' + tl.stage + ' 双缓冲让载入与计算重叠 —— 这也是 L0B 被占满的原因。',
      { size: 8, fill: C.muted });
    return { svg: svg(H, body), caption: 'K 轴 ' + tl.to + ' 按步长 ' + tl.step + ' 切成 ' + segs + ' 段，stage=' + tl.stage };
  };

  V.ExpandMixedKernel = (ctx) => {
    const f = ctx.pass.f, H = 118;
    const laneY = [30, 78], laneH = 26, x0 = 84, x1 = W - 18;
    let body = '';
    [['AIC · Cube', C.aic, f.aic], ['AIV · Vector', C.aiv, f.aiv]].forEach((L, i) => {
      const y = laneY[i];
      body += t(8, y + laneH / 2 + 3, L[0], { size: 8.5, fill: C.fg2 });
      body += rect(x0, y, x1 - x0, laneH, { fill: C.surf, stroke: C.line, rx: 3 });
      body += t(x1 - 6, y + laneH / 2 + 3, L[2] + ' 个 kernel', { anchor: 'end', size: 8 });
      const ops = i === 0 ? ['matmul', 'matmul_acc'] : ['exp', 'reduce', 'div'];
      ops.forEach((op, j) => {
        const bx = x0 + 10 + j * 66;
        body += rect(bx, y + 5, 58, laneH - 10, { fill: L[1], op: 0.85, rx: 2 });
        body += t(bx + 29, y + laneH / 2 + 3, op, { anchor: 'middle', size: 7.5, fill: '#fff' });
      });
    });
    // tpush / tpop between the lanes
    const mx = x0 + 250;
    body += arrow(mx, laneY[0] + laneH + 4, mx, { stroke: C.ok });
    body += '<path d="M' + mx + ' ' + (laneY[0] + laneH) + ' L' + mx + ' ' + laneY[1] + '" stroke="' + C.ok + '" stroke-width="1.2"/>';
    body += '<path d="M' + mx + ' ' + laneY[1] + ' l-3 -4 l6 0 z" fill="' + C.ok + '"/>';
    body += t(mx + 6, laneY[0] + laneH + 13, 'tpush_to_aiv', { size: 7.5, fill: C.ok });
    const mx2 = mx + 96;
    body += '<path d="M' + mx2 + ' ' + laneY[1] + ' L' + mx2 + ' ' + (laneY[0] + laneH) + '" stroke="' + C.pri + '" stroke-width="1.2"/>';
    body += '<path d="M' + mx2 + ' ' + (laneY[0] + laneH) + ' l-3 4 l6 0 z" fill="' + C.pri + '"/>';
    body += t(mx2 + 6, laneY[1] - 5, 'tpop_from_aiv', { size: 7.5, fill: C.pri });

    body += t(8, H - 6, 'Cube 与 Vector 是两种物理核。混合 kernel 被拆开，中间结果靠 tpush / tpop 传递，外面套一个 Group 统一发射。',
      { size: 8, fill: C.muted });
    return { svg: svg(H, body), caption: f.aic + ' AIC + ' + f.aiv + ' AIV + ' + f.group + ' Group' };
  };

  V.SkewCrossCorePipeline = (ctx) => {
    const H = 122, n = 5, cw = 62, x0 = 76, y0 = 26, rh = 22;
    let body = '';
    const lanes = [['生产者 AIC', C.aic, 0], ['消费者 AIV', C.aiv, 1]];
    lanes.forEach((L, li) => {
      const y = y0 + li * (rh + 12);
      body += t(8, y + rh / 2 + 3, L[0], { size: 8.5, fill: C.fg2 });
      for (let i = 0; i < n; i += 1) {
        const x = x0 + (i + L[2]) * cw;
        if (x + cw - 6 > W - 8) continue;
        body += rect(x, y, cw - 6, rh, { fill: L[1], op: 0.85, rx: 2 });
        body += t(x + (cw - 6) / 2, y + rh / 2 + 3, 'it ' + i, { anchor: 'middle', size: 7.5, fill: '#fff' });
      }
    });
    // phase brackets
    const py = y0 + 2 * (rh + 12) + 4;
    const seg = [['prologue', x0, cw - 6, C.muted], ['steady', x0 + cw, cw * (n - 1), C.ok], ['epilogue', x0 + cw * n, cw - 6, C.muted]];
    seg.forEach(s => {
      if (s[1] + s[2] > W - 8) s[2] = W - 8 - s[1];
      if (s[2] <= 0) return;
      body += line(s[1], py, s[1] + s[2], py, { stroke: s[3], sw: 1.2 });
      body += t(s[1] + s[2] / 2, py + 10, s[0], { anchor: 'middle', size: 7.5, fill: s[3] });
    });
    const dem = ctx.kernels.filter(k => k.intent.demoted > 0).length;
    body += t(8, H - 6,
      dem ? '生产者提前一轮 → 消费者侧的 pl.pipeline 被降级为 Sequential 以保住 FIFO 顺序（本次命中 ' + dem + ' 个 kernel）'
          : '生产者提前一轮，形成 prologue / steady / epilogue 三段',
      { size: 8, fill: dem ? C.warn : C.muted });
    return { svg: svg(H, body), caption: dem ? dem + ' 个 kernel 的流水被降级' : '跨核流水错位' };
  };

  V.LowerPipelineLoops = (ctx) => {
    const H = 112, stage = 2, iters = 4, cw = 46, ch = 20, x0 = 96, y0 = 24;
    let body = t(8, y0 + 12, 'pipeline', { size: 8.5, fill: C.fg2 }) +
      t(8, y0 + 23, 'stage=' + stage, { size: 8 });
    body += rect(x0, y0, cw * 2, ch, { fill: C.tile, op: 0.9, rx: 2 });
    body += t(x0 + cw, y0 + ch / 2 + 3, '循环体', { anchor: 'middle', size: 8, fill: '#fff' });
    body += arrow(x0 + cw * 2 + 8, y0 + ch / 2, x0 + cw * 2 + 30, { stroke: C.strong });

    const gx = x0 + cw * 2 + 40;
    for (let i = 0; i < iters; i += 1) {
      for (let s = 0; s < stage; s += 1) {
        const x = gx + i * (cw + 4), y = y0 + 34 + s * (ch + 3);
        if (x + cw > W - 8) continue;
        body += rect(x, y, cw, ch, { fill: s === 0 ? C.tile : C.aiv, op: 0.85, rx: 2 });
        body += t(x + cw / 2, y + ch / 2 + 3, '克隆 ' + s, { anchor: 'middle', size: 7, fill: '#fff' });
      }
    }
    body += t(gx, y0 + 30, '外层 ' + iters + ' 轮 × ' + stage + ' 份深拷贝克隆', { size: 8, fill: C.fg2 });
    body += t(8, H - 18, '每一份克隆之后会分到独立的 MemRef —— 这正是 ping-pong 双缓冲的前提。', { size: 8, fill: C.muted });
    body += t(8, H - 6, 'tile 算子 ' + ctx.prev.o.ti + ' → ' + ctx.pass.o.ti + '，增量全部来自克隆。', { size: 8, fill: C.muted });
    return { svg: svg(H, body), caption: 'tile 算子 ' + ctx.prev.o.ti + ' → ' + ctx.pass.o.ti };
  };

  V.CanonicalizeIOOrder = () => {
    const H = 100, bw = 30, bh = 17, y0 = 26, y1 = 62;
    const before = ['C', 'L', 'S', 'σ', 'C', 'L', 'S', 'σ', 'C', 'L'];
    const after = ['σ', 'σ', 'L', 'L', 'L', 'C', 'C', 'C', 'S', 'S'];
    const COL = { L: C.tensor, C: C.tile, S: C.aic, 'σ': C.muted };
    let body = t(8, y0 + 12, '重排前', { size: 8.5, fill: C.fg2 }) + t(8, y1 + 12, '重排后', { size: 8.5, fill: C.fg2 });
    [[before, y0], [after, y1]].forEach(row => {
      row[0].forEach((c, i) => {
        const x = 62 + i * (bw + 4);
        body += rect(x, row[1], bw, bh, { fill: COL[c], op: 0.85, rx: 2 });
        body += t(x + bw / 2, row[1] + bh / 2 + 3.5, c, { anchor: 'middle', size: 8, fill: '#fff' });
      });
    });
    body += t(W - 8, y0 + 12, '交错', { anchor: 'end', size: 8 });
    body += t(W - 8, y1 + 12, '分组', { anchor: 'end', size: 8, fill: C.ok });
    body += t(8, H - 16, 'σ 标量 · L load · C 计算 · S store', { size: 8, fill: C.muted });
    body += t(8, H - 5, '优先级感知的稳定拓扑排序：load 尽早、store 尽晚，给硬件流水留出重叠窗口。', { size: 8, fill: C.muted });
    return { svg: svg(H, body), caption: '语句重排为 [标量 · load · 计算 · store]' };
  };

  V.InitMemRef = (ctx) => {
    const k = ctx.focus;
    if (!k || !k.bufs.b30.length) return null;
    const p = packing(k.bufs.b30, { h: 116 });
    let body = p.body;
    body += t(8, 10, k.name + ' · 绑定了 ' + k.bufs.b30.length + ' 块片上 buffer（尚未定址，全部从 0 起）',
      { size: 8, fill: C.fg2 });
    return { svg: svg(p.h + 12, '<g transform="translate(0,12)">' + p.body + '</g>' +
      t(8, 9, k.name + ' · ' + p.space + ' · 绑定 ' + p.count + ' 块 buffer，偏移尚未分配', { size: 8, fill: C.fg2 })),
      caption: '绑定 ' + ctx.pass.o.mr + ' 个 MemRef' };
  };

  V.MemoryReuse = (ctx) => {
    const k = ctx.focus;
    if (!k || !k.bufs.b30.length) return null;
    const H = 96;
    const draw = (bufs, x0, w, tag, tone) => {
      const rows = bufs, maxT = Math.max(...rows.map(b => b.off + b.sz), 1);
      const maxE = Math.max(...rows.map(b => b.e), 1);
      let s = t(x0, 10, tag, { size: 8.5, fill: tone });
      s += rect(x0, 16, w, H - 34, { stroke: C.line });
      rows.forEach(b => {
        const bx = x0 + (b.s / maxE) * w, bw = Math.max(((b.e - b.s) / maxE) * w, 1.5);
        const by = 16 + (1 - (b.off + b.sz) / maxT) * (H - 34);
        const bh = Math.max(((b.sz) / maxT) * (H - 34), 1.2);
        s += rect(bx, by, bw, bh, { fill: SPACE_C[b.sp] || C.pri, op: 0.8, rx: 0.8 });
      });
      s += t(x0, H - 6, rows.length + ' 块 · ' + kb(rows.reduce((a, b) => a + b.sz, 0)), { size: 8, fill: tone });
      return s;
    };
    const cw = (W - 60) / 2;
    let body = draw(k.bufs.b30, 8, cw, '复用前', C.muted);
    body += draw(k.bufs.b32, 8 + cw + 44, cw, '复用后', C.ok);
    body += arrow(8 + cw + 12, H / 2, 8 + cw + 34, { stroke: C.strong });
    const g = k.reuse.before.b ? Math.round((1 - k.reuse.after.b / k.reuse.before.b) * 100) : 0;
    body += t(8 + cw + 22, H / 2 - 8, '省 ' + g + '%', { anchor: 'middle', size: 8, fill: C.ok, weight: 600 });
    return { svg: svg(H, body),
      caption: k.name + '：生命周期不重叠的 buffer 被合并到同一块地址上' };
  };

  V.AllocateMemoryAddr = (ctx) => {
    const k = ctx.focus;
    if (!k || !k.bufs.b32.length) return null;
    const sp = ctx.worstSpace || 'Vec';
    const p = packing(k.bufs.b32, { h: 124, space: sp, limit: ctx.limits[sp] });
    const used = p.peak, lim = ctx.limits[sp];
    return { svg: svg(p.h + 12,
      '<g transform="translate(0,12)">' + p.body + '</g>' +
      t(8, 9, k.name + ' · ' + sp + ' 实际占用 ' + kb(used) + ' / ' + kb(lim) +
        '（' + Math.round(used / lim * 100) + '%）', { size: 8, fill: used / lim >= 0.9 ? C.bad : C.fg2 })),
      caption: '每块 buffer 拿到真实地址，纵轴从全 0 铺开' };
  };

  V.OutlineIncoreScopes = (ctx) => {
    const n = ctx.born || 38;
    const f = fanout(n, '单个编排函数', n + ' 个 InCore kernel', () => C.tile);
    return { svg: svg(f.h + 14, '<g transform="translate(0,14)">' + f.body + '</g>' +
      t(8, 10, '程序结构从一棵语句树变成一张调用图', { size: 8, fill: C.muted })),
      caption: n + ' 个 scope 外提成独立 kernel' };
  };

  V.OutlineClusterScopes = (ctx) => {
    const n = ctx.born || 4;
    const f = fanout(n, '编排函数', n + ' 个 SPMD 分发', () => C.orch);
    return { svg: svg(f.h + 14, '<g transform="translate(0,14)">' + f.body + '</g>' +
      t(8, 10, '每个 SPMD 函数带 core_num，决定发射到多少个物理核', { size: 8, fill: C.muted })),
      caption: n + ' 个 Cluster scope 外提' };
  };

  V.InlineFunctions = () => {
    const H = 92, bw = 128, bh = 22;
    let body = t(8, 14, '内联前', { size: 8.5, fill: C.fg2 });
    body += rect(8, 22, bw, 52, { stroke: C.strong, rx: 3 });
    body += t(14, 36, 'main', { size: 8, fill: C.fg2 });
    body += rect(18, 42, bw - 20, bh, { fill: C.tile, op: 0.5, rx: 2 });
    body += t(26, 56, 'call _decode_layer', { size: 7.5, fill: C.fg2 });
    body += arrow(bw + 20, 48, bw + 48, { stroke: C.strong });
    body += t(bw + 60, 14, '内联后', { size: 8.5, fill: C.ok });
    body += rect(bw + 60, 22, W - bw - 70, 52, { stroke: C.strong, rx: 3 });
    for (let i = 0; i < 5; i += 1) {
      body += rect(bw + 68, 28 + i * 9, (W - bw - 90) * (0.5 + 0.1 * i), 6, { fill: C.tile, op: 0.75, rx: 1 });
    }
    body += t(8, H - 6, '函数体被拼接进每个调用点，局部量做 alpha 重命名（后缀 _inline38 就是这么来的）。', { size: 8, fill: C.muted });
    return { svg: svg(H, body), caption: '调用点被函数体替换' };
  };

  V.ConvertToSSA = () => {
    const H = 104, y = 34;
    let body = t(8, 14, '进入 SSA 前', { size: 8.5, fill: C.fg2 });
    body += rect(8, 22, 150, 22, { fill: C.surf, stroke: C.line, rx: 2 });
    body += t(16, 36, 'x = 0;  loop { x = f(x) }', { size: 7.5, fill: C.fg2 });
    body += arrow(170, y - 1, 196, { stroke: C.strong });
    body += t(206, 14, '进入 SSA 后 · 每个值单次定义', { size: 8.5, fill: C.ok });
    const nodes = [['x_v0', 210], ['iter_v1', 274], ['x_v5', 344], ['yield', 414]];
    nodes.forEach((n, i) => {
      body += rect(n[1], 24, 58, 18, { fill: i === 1 ? C.pri : C.tile, op: 0.85, rx: 2 });
      body += t(n[1] + 29, 36, n[0], { anchor: 'middle', size: 7.5, fill: '#fff' });
      if (i) body += arrow(nodes[i - 1][1] + 58 + 2, 33, n[1] - 2, { stroke: C.line });
    });
    body += '<path d="M' + (414 + 29) + ' 44 C ' + (414 + 29) + ' 64, ' + (274 + 29) + ' 64, ' + (274 + 29) + ' 44" ' +
      'fill="none" stroke="' + C.pri + '" stroke-width="1" stroke-dasharray="3 2"/>';
    body += t(348, 74, '循环进位（iter_arg ← yield）', { size: 7.5, fill: C.pri, anchor: 'middle' });
    body += t(8, H - 6, '循环里的重复赋值变成显式的 iter_arg / yield 链，数据流从此可以被机械地追踪。', { size: 8, fill: C.muted });
    return { svg: svg(H, body), caption: '值变成单次定义，循环进位显式化' };
  };

  V.FlattenCallExpr = () => {
    const H = 100;
    let body = t(8, 14, '嵌套表达式', { size: 8.5, fill: C.fg2 });
    const tree = [[86, 26, 'add'], [46, 46, 'reshape'], [46, 64, 'row_sum'], [46, 82, 'mul']];
    tree.forEach((n, i) => {
      body += rect(n[0], n[1] - 10, 64, 15, { fill: C.tile, op: 0.85 - i * 0.12, rx: 2 });
      body += t(n[0] + 32, n[1] + 0.5, n[2], { anchor: 'middle', size: 7.5, fill: '#fff' });
      if (i) body += line(tree[i - 1][0] + 6, tree[i - 1][1] + 5, n[0] + 30, n[1] - 10, { stroke: C.line });
    });
    body += arrow(200, 52, 228, { stroke: C.strong });
    body += t(244, 14, '一句一调用', { size: 8.5, fill: C.ok });
    ['t0 = mul(x, x)', 't1 = row_sum(t0)', 't2 = reshape(t1)', 'r  = add(acc, t2)'].forEach((s, i) => {
      body += rect(244, 22 + i * 18, W - 254, 14, { fill: C.surf, stroke: C.line, rx: 2 });
      body += t(250, 32 + i * 18, s, { size: 7.5, fill: C.fg2 });
    });
    body += t(8, H - 6, '每条语句只剩一个调用，后面的所有分析都建立在这个前提上。', { size: 8, fill: C.muted });
    return { svg: svg(H, body), caption: '嵌套调用摊平成线性序列' };
  };

  V.DeriveCallDirections = () => {
    const H = 92, y = 32;
    const args = [['down_acc_all', 'input', C.tensor], ['post_norm', 'input', C.tensor],
                  ['next_hidden', 'inout', C.aic], ['rms_weight', 'input', C.tensor], ['next_normed', 'inout', C.aic]];
    let body = t(8, 14, 'dcr_xgamma(…) 的实参方向', { size: 8.5, fill: C.fg2 });
    const cw = (W - 20) / args.length;
    args.forEach((a, i) => {
      const x = 10 + i * cw;
      body += rect(x, y, cw - 8, 20, { fill: a[2], op: 0.82, rx: 2 });
      body += t(x + (cw - 8) / 2, y + 13, a[0].slice(0, 12), { anchor: 'middle', size: 7, fill: '#fff' });
      const dir = a[1] === 'inout' ? '↔' : '→';
      body += t(x + (cw - 8) / 2, y + 34, dir + ' ' + a[1], { anchor: 'middle', size: 7.5, fill: a[1] === 'inout' ? C.aic : C.muted });
    });
    body += t(8, H - 6, 'input 只读、inout 会被就地写回。运行时靠这个信息推导任务之间的真实依赖边。', { size: 8, fill: C.muted });
    return { svg: svg(H, body), caption: '每个实参标注 input / output / inout' };
  };

  V.MaterializeRuntimeScopes = () => {
    const H = 96;
    let body = t(8, 14, '隐式（codegen 内部决定）', { size: 8.5, fill: C.fg2 });
    body += rect(8, 22, 180, 56, { stroke: C.line, dash: '3 2', rx: 3 });
    body += t(16, 40, 'for … :', { size: 7.5, fill: C.fg2 });
    body += t(26, 54, 'submit(...)', { size: 7.5, fill: C.muted });
    body += t(16, 70, '（作用域由 codegen 隐式补）', { size: 7, fill: C.muted });
    body += arrow(200, 50, 226, { stroke: C.strong });
    body += t(242, 14, '显式写进 IR', { size: 8.5, fill: C.ok });
    body += rect(242, 22, W - 252, 56, { stroke: C.ok, rx: 3 });
    body += t(250, 38, 'with pl.scope(AUTO):', { size: 7.5, fill: C.ok });
    body += rect(258, 44, W - 274, 26, { stroke: C.line, rx: 2 });
    body += t(266, 58, 'for … : with pl.scope(AUTO): submit(...)', { size: 7, fill: C.fg2 });
    body += t(8, H - 6, '作用域决策从 codegen 搬进 IR，从此可被检查、可被 diff，也不再是黑盒。', { size: 8, fill: C.muted });
    return { svg: svg(H, body), caption: '隐式 PTO2_SCOPE 变成显式 RuntimeScopeStmt' };
  };

  V.SplitVectorKernel = (ctx) => {
    const H = 96, n = ctx.kernels.filter(k => k.split).length;
    let body = t(8, 14, 'AIV 双核切分 · UP_DOWN', { size: 8.5, fill: C.fg2 });
    const bw = (W - 30) / 2;
    [['上半 tile', 0], ['下半 tile', 1]].forEach((L, i) => {
      const x = 10 + i * (bw + 10);
      body += rect(x, 24, bw, 22, { fill: C.aiv, op: 0.85, rx: 2 });
      body += t(x + bw / 2, 38, L[0], { anchor: 'middle', size: 8, fill: '#fff' });
      body += t(x + bw / 2, 58, 'AIV 核 ' + i, { anchor: 'middle', size: 7.5 });
      body += arrow(x + bw / 2, 46, x + bw / 2, { stroke: C.line });
    });
    body += t(8, H - 18, n + ' 个 AIV kernel 的 tpush / tpop 被打上 split 标记，tpop 形状与 store 偏移同步调整。', { size: 8, fill: C.muted });
    body += t(8, H - 6, '一个逻辑 kernel 同时跑在两个物理 Vector 核上，各算一半。', { size: 8, fill: C.muted });
    return { svg: svg(H, body), caption: n + ' 个 AIV kernel 落上 split 标记' };
  };

  /* ---------------- data-driven visuals ----------------
     These render the actual transformation instances parsed out of this
     program's IR, not a schematic of what the pass does in general. */

  V.UnrollLoops = (ctx) => {
    const f = ctx.facts && ctx.facts.UnrollLoops;
    if (!f || !f.loops.length) return null;

    const maxGrow = Math.max.apply(null, f.loops.map(l => l.grew));
    const rows = f.loops.map((L, i) => {
      const w = Math.round((L.grew / maxGrow) * 100);
      const blocks = Array.from({ length: Math.min(L.trip, 16) }, () =>
        '<i style="flex:1 1 0"></i>').join('');
      return '<button type="button" class="kf-kg-fact' + (i === (ctx.factSel || 0) ? ' is-sel' : '') + '" data-kg-fact="' + i + '">' +
        '<code>' + esc(L.v) + '</code>' +
        '<span class="kf-kg-fact-t">pl.unroll(' + L.trip + ')</span>' +
        '<span class="kf-kg-fact-b" style="width:' + Math.max(w, 6) + '%">' + blocks + '</span>' +
        '<span class="kf-kg-fact-n">' + L.bodyLines + ' 行 × ' + L.trip + ' = <b>' + L.grew + '</b></span>' +
      '</button>';
    }).join('');

    const L = f.loops[ctx.factSel || 0] || f.loops[0];
    let sub = '';
    if (L.subst) {
      sub =
        '<div class="kf-kg-fact-sub">' +
          '<div class="kf-kg-fact-h">循环变量替换 · <code>' + esc(L.v) + '</code> → pl.const(i)</div>' +
          '<div class="kf-kg-fact-line is-before"><span>展开前</span><code>' + esc(L.subst.orig) + '</code></div>' +
          L.subst.after.map((h, i) =>
            '<div class="kf-kg-fact-line is-after"><span>i=' + i + ' · L' + h.line + '</span><code>' + esc(h.s) + '</code></div>'
          ).join('') +
        '</div>';
    } else {
      sub = '<p class="kf-kg-fact-none">这个循环体里没有直接依赖循环变量的赋值，展开后是 ' + L.trip + ' 份等价副本。</p>';
    }

    const usesList = L.uses.length
      ? '<div class="kf-kg-fact-sub"><div class="kf-kg-fact-h">引用循环变量的语句 · 共 ' + L.useCount + ' 条</div>' +
        L.uses.map(u => '<div class="kf-kg-fact-line"><span></span><code>' + esc(u) + '</code></div>').join('') +
        (L.useCount > L.uses.length ? '<div class="kf-kg-fact-more">… 另 ' + (L.useCount - L.uses.length) + ' 条</div>' : '') +
        '</div>'
      : '';

    return {
      html:
        '<div class="kf-kg-facts">' +
          '<div class="kf-kg-fact-h">这份 IR 里的 ' + f.loops.length + ' 个 pl.unroll 循环 · 点击查看替换细节</div>' +
          rows +
          sub + usesList +
          // Derived from the selected loop's actual rewritten lines, not asserted:
          // scalar arithmetic keeps a pl.const(i) node, an index slot is
          // substituted literally.
          (L.subst
            ? '<p class="kf-kg-fact-note">' +
                (L.subst.after.some(h => /pl\.const\(\d+/.test(h.s))
                  ? '循环变量在这里变成 pl.const(i) 节点，常量折叠要等第 05 步 Simplify。'
                  : '循环变量在这里被直接替换成字面下标，没有留下 const 节点。') +
              '</p>'
            : '') +
          '<p class="kf-kg-fact-note">展开后 4 个循环变量在 IR 中残留 ' +
            f.loops.reduce((a, l) => a + l.residual, 0) + ' 处 —— UnrollResolved 成立。' +
            '整体 ' + f.lines.before + ' → ' + f.lines.after + ' 行。</p>' +
        '</div>',
      caption: f.loops.length + ' 个循环展开：' + f.loops.map(l => l.v.replace(/_inline\d+$/, '') + '×' + l.trip).join('、')
    };
  };

  /* ---------------- generic fallback ---------------- */
  function generic(ctx) {
    const p = ctx.pass, pr = ctx.prev;
    if (!pr) return null;
    const metrics = [
      ['IR 行数', pr.l, p.l], ['tensor 算子', pr.o.te, p.o.te],
      ['tile 算子', pr.o.ti, p.o.ti], ['MemRef', pr.o.mr, p.o.mr],
      ['函数', Object.keys(pr.f).reduce((a, x) => a + pr.f[x], 0), Object.keys(p.f).reduce((a, x) => a + p.f[x], 0)]
    ].filter(m => m[1] !== m[2]);
    if (!metrics.length) return null;

    const H = 30 + metrics.length * 20;
    let body = '';
    metrics.forEach((m, i) => {
      const y = 18 + i * 20, max = Math.max(m[1], m[2], 1), x0 = 78, w = W - x0 - 96;
      body += t(x0 - 6, y + 8, m[0], { anchor: 'end', size: 8, fill: C.fg2 });
      body += rect(x0, y, (m[1] / max) * w, 7, { fill: C.muted, op: 0.5, rx: 1 });
      body += rect(x0, y + 8, (m[2] / max) * w, 7, { fill: m[2] > m[1] ? C.tile : C.ok, rx: 1 });
      body += t(W - 90, y + 11, m[1] + ' → ' + m[2], { size: 8 });
      const d = m[2] - m[1];
      body += t(W - 8, y + 11, (d > 0 ? '+' : '') + d, { anchor: 'end', size: 8, fill: d > 0 ? C.tile : C.ok });
    });
    body += t(8, H - 4, '这一步没有专属图示，先看它对 IR 规模的影响。', { size: 7.5, fill: C.muted });
    return { svg: svg(H, body), caption: '规模变化' };
  }

  window.PTO_PASS_VISUAL = {
    has: (name) => !!V[name],
    render: (name, ctx) => {
      try {
        const r = (V[name] && V[name](ctx)) || generic(ctx);
        return r;
      } catch (e) { return null; }
    }
  };
})();
