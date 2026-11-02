/* ============================================================================
 * Hardware-Native Kit · core
 * ----------------------------------------------------------------------------
 * 纯函数层，不碰 DOM。四件事：
 *
 *   resolveAnchors()  manifest 的名字锚点 → 真实源码行号（并报告对不上的锚点）
 *   capacity()        buffer 声明 → 片上空间高水位、地址排布、判定档位
 *   layoutBoxes()     平台区域盒子树 → 具体坐标
 *   layoutGraph()     节点 + 依赖 + 阶段 → 逻辑图坐标与连线
 *
 * manifest 里没有任何像素，也没有任何行号；这两样都在这里算出来。
 * ========================================================================== */
(function (root) {
  'use strict';

  // ── 小工具 ────────────────────────────────────────────────────────────────
  function uniq(list) { return list.filter(function (v, i) { return list.indexOf(v) === i; }); }
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function kb(n) {
    const v = n / 1024;
    return (Math.abs(v - Math.round(v)) < 0.05 ? Math.round(v) : v.toFixed(1)) + ' KB';
  }
  function bytes(n) { return Math.abs(n) < 1024 ? n + ' B' : kb(n); }
  function escapeRe(s) { return String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

  // 估算标签宽度：CJK 按 1.8 个字宽算，用于逻辑图列宽
  function textWidth(s) {
    let units = 0;
    for (let i = 0; i < s.length; i++) units += /[⺀-￯]/.test(s[i]) ? 1.8 : 1;
    return Math.min(116, Math.max(66, Math.round(18 + units * 7.2)));
  }

  // buffer 的键：有赋值目标就用它；没有的（比如内联在表达式里的临时量）用显式 id
  function bufKey(b) { return b.id || b.assigns; }

  const DTYPE = {
    fp32: { bytes: 4, label: 'FP32' },
    bf16: { bytes: 2, label: 'BF16' },
    fp16: { bytes: 2, label: 'FP16' },
    int8: { bytes: 1, label: 'INT8' },
    int32: { bytes: 4, label: 'INT32' }
  };

  // ════════════════════════════════════════════════════════════════════════════
  // 常量
  // ════════════════════════════════════════════════════════════════════════════
  // manifest 给 consts（旋钮当前值 + 固定事实），derive 再补派生量。
  function resolveConsts(kernel, cfg) {
    const base = Object.assign({}, kernel.consts, cfg || {});
    return kernel.derive ? Object.assign(base, kernel.derive(base)) : base;
  }

  function dim(v, consts, diag, where) {
    if (typeof v === 'number') return v;
    const n = consts[v];
    if (typeof n !== 'number') {
      diag.push({ level: 'error', what: where, msg: '形状引用了未定义的常量 ' + v });
      return 0;
    }
    return n;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 锚点解析：名字 → 行号
  // ════════════════════════════════════════════════════════════════════════════
  // manifest 用「赋值目标名 / 调用的 pl.* / scope」定位，不用行号。
  // 这个 DSL 里每个中间量都是单赋值 `x = pl.foo(...)`，所以名字是可靠锚点；
  // 源码上游插几行，这里重跑一次就对，不会像写死行号那样静默错位。
  function assignLines(source, name) {
    const re = new RegExp('^\\s*' + escapeRe(name) + '\\s*(?::[^=]*)?=(?!=)');
    return source.filter(function (r) { return re.test(r.t); }).map(function (r) { return r.no; });
  }
  function callLines(source, callee) {
    const needle = callee + '(';
    return source.filter(function (r) { return r.t.indexOf(needle) >= 0; }).map(function (r) { return r.no; });
  }

  // 一个锚点规格（节点、frame 或 buffer 都用同一套字段）→ 行号集合
  function anchorLines(source, spec, diag) {
    let out = [];
    const names = (spec.assigns ? [spec.assigns] : []).concat(spec.also || []);
    const calls = (spec.call ? [spec.call] : []).concat(spec.calls || []);
    names.forEach(function (n) {
      const hit = assignLines(source, n);
      if (!hit.length) diag.push({ level: 'warn', what: spec.id || n, msg: '找不到赋值 ' + n + '（源码变了？）' });
      out = out.concat(hit);
    });
    calls.forEach(function (c) {
      const hit = callLines(source, c);
      if (!hit.length) diag.push({ level: 'warn', what: spec.id || c, msg: '找不到调用 ' + c });
      out = out.concat(hit);
    });
    if (spec.scope) {
      const hit = callLines(source, spec.scope);
      if (!hit.length) diag.push({ level: 'warn', what: spec.id, msg: '找不到 scope ' + spec.scope });
      out = out.concat(hit);
    }
    // lines 是逃生口：给锚点表达不了的行（比如一个节点还想连到它依赖的某个标量）
    if (spec.lines) out = out.concat(spec.lines);
    return uniq(out).sort(function (a, b) { return a - b; });
  }

  // 节点 / frame / buffer 的行号一次算好，联动直接查
  function resolveAnchors(kernel) {
    const diag = [];
    const src = kernel.source;
    const lines = {};
    kernel.nodes.forEach(function (n) { lines[n.id] = anchorLines(src, n, diag); });
    (kernel.frames || []).forEach(function (f) { lines[f.id] = anchorLines(src, f, diag); });
    (kernel.buffers || []).forEach(function (b) { lines['buf:' + bufKey(b)] = anchorLines(src, b, diag); });
    return { lines: lines, diagnostics: diag };
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 容量模型
  // ════════════════════════════════════════════════════════════════════════════
  function frameOf(kernel, id) {
    return (kernel.frames || []).filter(function (f) { return f.id === id; })[0] || null;
  }

  // 一个 buffer 同时驻留几份：由它所在 frame 的 residency 决定。
  // resident 的（跨迭代 carry）永远一份。
  function bufCount(kernel, b, consts) {
    if (b.resident) return 1;
    const f = frameOf(kernel, b.frame);
    if (!f || f.residency == null) return 1;
    const r = typeof f.residency === 'function' ? f.residency(consts) : f.residency;
    return Math.max(1, Math.round(r));
  }

  function bufSize(b, consts, diag) {
    const d = DTYPE[b.dtype];
    if (!d) { diag.push({ level: 'error', what: bufKey(b), msg: '未知 dtype ' + b.dtype }); return 0; }
    let n = 1;
    (b.shape || []).forEach(function (v) { n *= dim(v, consts, diag, bufKey(b)); });
    return n * d.bytes;
  }

  function verdictOf(sp) {
    if (sp.used > sp.nominal) return 'over';      // 分配器自己会报错 —— 看得见的失败
    if (sp.used > sp.usable) return 'silent';     // 只有 reserved > 0 的空间有这一档
    if (sp.used > sp.usable * 0.92) return 'tight';
    return 'ok';
  }

  const VERDICT_RANK = { ok: 0, tight: 1, silent: 2, over: 3 };

  function capacity(kernel, platform, consts) {
    const diag = [];
    const spaces = {};
    Object.keys(platform.spaces).forEach(function (id) {
      const p = platform.spaces[id];
      const reserved = p.reserved || 0;
      spaces[id] = {
        id: id, label: p.label, nominal: p.nominal, reserved: reserved,
        usable: p.nominal - reserved, used: 0, items: [],
        modeled: p.modeled !== false, region: platform.spaceRegion[id] || id
      };
    });

    if (kernel.compiledAllocations) {
      kernel.compiledAllocations.forEach(function (a) {
        const id = { Mat:'mat', Left:'left', Right:'right', Acc:'acc', Vec:'vec' }[a.space];
        if (!id || !Number.isFinite(a.address)) { diag.push({what:a.id,msg:'未识别空间或地址，未纳入高水位'}); return; }
        const sp=spaces[id]; sp.modeled=true;
        sp.items.push({id:a.worker+':'+a.id,bufId:a.worker+':'+a.id,label:a.id+' · '+a.worker+' · 编译行 '+a.allocLine,from:a.address,to:a.address+a.size,size:a.size,count:1,resident:true});
        sp.used=Math.max(sp.used,a.address+a.size);
      });
      Object.keys(spaces).forEach(function(id) { const sp=spaces[id]; sp.verdict=verdictOf(sp); sp.over=Math.max(0,sp.used-sp.usable); sp.headroom=Math.max(0,sp.usable-sp.used); sp.underNominal=Math.max(0,sp.nominal-sp.used); sp.pct=sp.usable?sp.used/sp.usable:0; sp.offender=sp.items.find(function(i){return i.to>sp.usable;})||null; });
      const worst=Object.values(spaces).reduce(function(w,sp){return sp.items.length && VERDICT_RANK[sp.verdict]>VERDICT_RANK[w]?sp.verdict:w;},'ok');
      return {spaces:spaces,worst:worst,diagnostics:diag};
    }
    // 先把每个 buffer 的大小和份数算出来
    const decls = (kernel.buffers || []).map(function (b) {
      return {
        b: b, size: bufSize(b, consts, diag), count: bufCount(kernel, b, consts),
        space: b.space
      };
    });

    // 地址排布：常驻的在低地址，多份驻留的按「一级一组」往高地址排。
    // 这是估算口径 —— 是否复用、是否就地、真实分配顺序要按编译产物核对。
    Object.keys(spaces).forEach(function (sid) {
      const mine = decls.filter(function (d) { return d.space === sid && d.size > 0; });
      if (!mine.length) return;
      const sp = spaces[sid];
      let at = 0;
      const place = function (d, copy) {
        const from = at;
        at += d.size;
        sp.items.push({
          id: bufKey(d.b) + (d.count > 1 ? '#' + copy : ''),
          bufId: bufKey(d.b),
          label: d.b.label || bufKey(d.b),
          dtype: d.b.dtype, size: d.size, copy: copy, count: d.count,
          resident: !!d.b.resident, from: from, to: at,
          crosses: at > sp.usable && from < sp.usable,
          inReserved: from >= sp.usable
        });
      };
      mine.filter(function (d) { return d.count === 1; }).forEach(function (d) { place(d, 0); });
      const maxCount = mine.reduce(function (m, d) { return Math.max(m, d.count); }, 1);
      for (let c = 0; c < maxCount; c++) {
        mine.filter(function (d) { return d.count > 1 && d.count > c; })
          .forEach(function (d) { place(d, c); });
      }
      sp.used = at;
      sp.verdict = verdictOf(sp);
      sp.over = Math.max(0, sp.used - sp.usable);
      sp.headroom = Math.max(0, sp.usable - sp.used);
      sp.underNominal = Math.max(0, sp.nominal - sp.used);
      sp.pct = sp.usable ? sp.used / sp.usable : 0;
      // 越过可用上限的第一块：预警里要点名的就是它
      sp.offender = sp.items.filter(function (i) { return i.crosses || i.inReserved; })[0] || null;
    });

    // 未建模的空间（容量口径还没核对的）不参与总判定，免得报一个站不住的结论
    let worst = 'ok';
    Object.keys(spaces).forEach(function (sid) {
      if (spaces[sid].modeled === false) return;
      const v = spaces[sid].verdict || 'ok';
      if (VERDICT_RANK[v] > VERDICT_RANK[worst]) worst = v;
    });

    return { spaces: spaces, worst: worst, diagnostics: diag };
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 硅片区域盒子树 → 坐标
  // ════════════════════════════════════════════════════════════════════════════
  // slots 节点的子项由 manifest 的 tensors 填充，区域 id 形如 gm:k_cache
  function expandSlots(node, kernel) {
    const copy = Object.assign({}, node);
    if (node.kind === 'slots') {
      const want = node.space || 'gm';
      copy.children = (kernel.tensors || [])
        .filter(function (t) { return (t.space || 'gm') === want; })
        .map(function (t) {
          return { id: want + ':' + t.id, label: t.label || t.id, kind: 'mem', note: t.note };
        });
      if (!copy.children.length) copy.children = [{ id: want + ':none', label: '—', kind: 'mem' }];
      return copy;
    }
    if (node.children) copy.children = node.children.map(function (c) { return expandSlots(c, kernel); });
    return copy;
  }

  function layoutBoxes(platform, kernel, view) {
    const tree = expandSlots(platform.chip, kernel);
    const out = [];
    walkBox(tree, { x: 0, y: 0, w: view.w, h: view.h }, out, 0);
    const byId = {};
    out.forEach(function (b) { byId[b.id] = b; });
    return { boxes: out, byId: byId, view: view };
  }

  function walkBox(node, rect, out, depth) {
    out.push({
      id: node.id, label: node.label, kind: node.kind, space: node.space,
      note: node.note, depth: depth, rect: rect
    });
    const kids = node.children;
    if (!kids || !kids.length) return;
    const pad = node.pad == null ? 10 : node.pad;
    const gap = node.gap == null ? 8 : node.gap;
    const head = node.label ? (node.headH == null ? 20 : node.headH) : 0;
    const inner = {
      x: rect.x + pad, y: rect.y + pad + head,
      w: Math.max(0, rect.w - pad * 2), h: Math.max(0, rect.h - pad * 2 - head)
    };
    const horiz = node.dir === 'row';
    const total = (horiz ? inner.w : inner.h) - gap * (kids.length - 1);
    let fixed = 0, flexSum = 0;
    kids.forEach(function (c) {
      const f = horiz ? c.w : c.h;
      if (f != null) fixed += f; else flexSum += (c.flex == null ? 1 : c.flex);
    });
    const unit = flexSum > 0 ? Math.max(0, total - fixed) / flexSum : 0;
    let at = horiz ? inner.x : inner.y;
    kids.forEach(function (c) {
      const f = horiz ? c.w : c.h;
      const size = f != null ? f : unit * (c.flex == null ? 1 : c.flex);
      walkBox(c, horiz
        ? { x: at, y: inner.y, w: size, h: inner.h }
        : { x: inner.x, y: at, w: inner.w, h: size }, out, depth + 1);
      at += size + gap;
    });
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 逻辑图自动布局
  // ════════════════════════════════════════════════════════════════════════════
  // 列 = 阶段内的依赖深度；行 = 同深度的堆叠（阶段用 wrap 折行）。
  // manifest 只给 after（依赖）和 phase（语义分段），不给任何坐标。
  const NODE_H = 26, ROW_PITCH = 38, COL_GAP = 14;
  const PHASE_PAD = 12, PHASE_HEAD = 22, PHASE_GAP = 18, FRAME_HEAD = 24;

  function frameDepth(kernel, id, guard) {
    const f = frameOf(kernel, id);
    if (!f || (guard || 0) > 8) return 0;
    return f.parent ? 1 + frameDepth(kernel, f.parent, (guard || 0) + 1) : 1;
  }

  // 阶段内的局部依赖深度：指向阶段外的依赖不计，这样每段都从 0 列开始
  function localRanks(nodes) {
    const ids = nodes.map(function (n) { return n.id; });
    const rank = {};
    nodes.forEach(function (n) { rank[n.id] = 0; });
    for (let pass = 0; pass < nodes.length + 1; pass++) {
      let moved = false;
      nodes.forEach(function (n) {
        (n.after || []).forEach(function (p) {
          if (ids.indexOf(p) < 0) return;
          if (rank[n.id] < rank[p] + 1) { rank[n.id] = rank[p] + 1; moved = true; }
        });
      });
      if (!moved) break;
    }
    return rank;
  }

  function layoutGraph(kernel) {
    const visible = kernel.nodes.filter(function (n) { return !n.hidden; });
    const phases = (kernel.phases || []).length
      ? kernel.phases
      : [{ id: 'all', label: '', nodes: visible.map(function (n) { return n.id; }) }];

    let maxDepth = 0;
    visible.forEach(function (n) { maxDepth = Math.max(maxDepth, frameDepth(kernel, n.frame)); });
    const baseY = maxDepth * FRAME_HEAD + 10;

    const boxes = {};
    const phaseBoxes = [];
    let cursorX = 14;

    phases.forEach(function (ph) {
      const mine = ph.nodes.map(function (id) {
        return visible.filter(function (n) { return n.id === id; })[0];
      }).filter(Boolean);
      if (!mine.length) return;

      const rank = localRanks(mine);
      const wrap = ph.wrap || 4;
      // (band, col) 分格，同格内堆叠
      const stackAt = {};
      const cells = mine.map(function (n) {
        const r = rank[n.id];
        const band = Math.floor(r / wrap), col = r % wrap;
        const key = band + ':' + col;
        const stack = stackAt[key] || 0;
        stackAt[key] = stack + 1;
        return { n: n, band: band, col: col, stack: stack };
      });
      // 每个 band 占几行 → band 的纵向偏移
      const bandRows = {};
      cells.forEach(function (c) { bandRows[c.band] = Math.max(bandRows[c.band] || 0, c.stack + 1); });
      const bandY = {}; let acc = 0;
      Object.keys(bandRows).map(Number).sort(function (a, b) { return a - b; }).forEach(function (b) {
        bandY[b] = acc; acc += bandRows[b] * ROW_PITCH;
      });
      // 每列宽度取该列最宽标签，列内对齐
      const colW = {};
      cells.forEach(function (c) {
        const w = textWidth(c.n.label || c.n.id);
        colW[c.col] = Math.max(colW[c.col] || 0, w);
      });
      const colX = {}; let cx = cursorX + PHASE_PAD;
      Object.keys(colW).map(Number).sort(function (a, b) { return a - b; }).forEach(function (col) {
        colX[col] = cx; cx += colW[col] + COL_GAP;
      });

      const top = baseY + PHASE_HEAD;
      cells.forEach(function (c) {
        boxes[c.n.id] = {
          id: c.n.id, label: c.n.label || c.n.id, node: c.n,
          x: colX[c.col], y: top + bandY[c.band] + c.stack * ROW_PITCH,
          w: colW[c.col], h: NODE_H
        };
      });

      const right = cx - COL_GAP + PHASE_PAD;
      phaseBoxes.push({
        id: ph.id, label: ph.label, nodes: ph.nodes,
        x: cursorX, y: baseY,
        w: right - cursorX, h: PHASE_HEAD + acc - (ROW_PITCH - NODE_H) + PHASE_PAD
      });
      cursorX = right + PHASE_GAP;
    });

    // frame 框：罩住属于它（或它的子 frame）的节点，以及完全在内的阶段框
    const frames = (kernel.frames || []).map(function (f) {
      return { id: f.id, label: f.label || f.id, depth: frameDepth(kernel, f.id), frame: f };
    }).sort(function (a, b) { return b.depth - a.depth; });

    function inFrame(nodeFrameId, frameId, guard) {
      if (!nodeFrameId || (guard || 0) > 8) return false;
      if (nodeFrameId === frameId) return true;
      const f = frameOf(kernel, nodeFrameId);
      return f && f.parent ? inFrame(f.parent, frameId, (guard || 0) + 1) : false;
    }

    frames.forEach(function (fr) {
      const ids = visible.filter(function (n) { return inFrame(n.frame, fr.id); })
        .map(function (n) { return n.id; });
      const rects = ids.map(function (id) { return boxes[id]; }).filter(Boolean);
      phaseBoxes.forEach(function (pb) {
        if (pb.nodes.some(function (id) { return ids.indexOf(id) >= 0; })) rects.push(pb);
      });
      if (!rects.length) { fr.rect = null; return; }
      const pad = (maxDepth - fr.depth + 1) * 7 + 4;
      const box = bbox(rects);
      fr.rect = {
        x: box.x - pad, y: box.y - pad - 18,
        w: box.w + pad * 2, h: box.h + pad * 2 + 18
      };
    });

    const edges = [];
    visible.forEach(function (n) {
      (n.after || []).forEach(function (p) {
        const a = boxes[p], b = boxes[n.id];
        if (a && b) edges.push({ from: p, to: n.id, d: elbow(a, b) });
      });
    });

    // manifest 可为需要精确编排的图提供布局，其余 kernel 继续自动布局。
    const layout = kernel.graphLayout;
    if (layout) {
      Object.keys(layout.nodes || {}).forEach(function (id) {
        const pos = layout.nodes[id], node = boxes[id];
        if (!node) return;
        node.x = pos[0]; node.y = pos[1]; node.w = pos[2]; node.h = 26;
        if (pos[3]) node.label = pos[3];
      });
      frames.forEach(function (frame) {
        const pos = (layout.frames || {})[frame.id];
        if (pos) frame.rect = { x: pos[0], y: pos[1], w: pos[2], h: pos[3] };
      });
      const arranged = phaseBoxes.filter(function (phase) { return (layout.phases || {})[phase.id]; });
      arranged.forEach(function (phase) {
        const pos = layout.phases[phase.id];
        phase.x = pos[0]; phase.y = pos[1]; phase.w = pos[2]; phase.h = pos[3];
        if (pos[4]) phase.label = pos[4];
      });
      return { nodes: boxes, phases: arranged, frames: frames.filter(function (f) { return f.rect; }),
        edges: edges.map(function (edge) { return { from: edge.from, to: edge.to, d: elbow(boxes[edge.from], boxes[edge.to]) }; }),
        viewBox: layout.viewBox };
    }

    const all = Object.keys(boxes).map(function (k) { return boxes[k]; })
      .concat(phaseBoxes)
      .concat(frames.map(function (f) { return f.rect; }).filter(Boolean));
    const b = bbox(all);
    const M = 10;
    return {
      nodes: boxes, phases: phaseBoxes,
      frames: frames.filter(function (f) { return f.rect; }),
      edges: edges,
      viewBox: (b.x - M) + ' ' + (b.y - M) + ' ' + (b.w + M * 2) + ' ' + (b.h + M * 2)
    };
  }

  function bbox(rects) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    rects.forEach(function (r) {
      x0 = Math.min(x0, r.x); y0 = Math.min(y0, r.y);
      x1 = Math.max(x1, r.x + r.w); y1 = Math.max(y1, r.y + r.h);
    });
    if (x0 === Infinity) return { x: 0, y: 0, w: 1, h: 1 };
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  }

  // 三种走线：同带向右、向下换带、折行回左
  function elbow(a, b) {
    const ay = Math.round(a.y + a.h / 2), by = Math.round(b.y + b.h / 2);
    if (b.x >= a.x + a.w) {
      const x1 = a.x + a.w, mid = Math.round((x1 + b.x) / 2);
      return ay === by
        ? 'M' + x1 + ' ' + ay + ' H' + b.x
        : 'M' + x1 + ' ' + ay + ' H' + mid + ' V' + by + ' H' + b.x;
    }
    const ax = Math.round(a.x + a.w / 2), bx = Math.round(b.x + b.w / 2);
    const down = a.y + a.h, lane = Math.round((down + b.y) / 2);
    return 'M' + ax + ' ' + down + ' V' + lane + ' H' + bx + ' V' + b.y;
  }

  root.HNKitCore = {
    uniq: uniq, esc: esc, kb: kb, bytes: bytes, textWidth: textWidth, DTYPE: DTYPE,
    resolveConsts: resolveConsts, resolveAnchors: resolveAnchors,
    capacity: capacity, layoutBoxes: layoutBoxes, layoutGraph: layoutGraph,
    frameOf: frameOf, bufKey: bufKey, VERDICT_RANK: VERDICT_RANK
  };
})(typeof window !== 'undefined' ? window : this);
