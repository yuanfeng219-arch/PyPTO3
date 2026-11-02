/* ============================================================================
 * Hardware-Native Kit · view
 * ----------------------------------------------------------------------------
 * 三个内容区的渲染 + 它们之间的联动。
 *
 *   联动的单一真相源是 manifest 的 nodes 表：
 *   一个节点 = 源码行 + 硅片区域 + 片上 buffer。
 *   悬停任意一区，都反查到这张表，再同时点亮另外两区。
 *
 * 渲染器不认识任何具体 kernel；拿到的都是 core.js 算好的坐标。
 * ========================================================================== */
(function (root) {
  'use strict';

  const C = root.HNKitCore;
  const esc = C.esc, kb = C.kb, bytes = C.bytes;

  // ── 源码着色 ──────────────────────────────────────────────────────────────
  const KW = {
    with: 1, for: 1, in: 1, as: 1, if: 1, else: 1, return: 1, def: 1, assert: 1,
    True: 1, False: 1, None: 1, and: 1, or: 1, not: 1
  };

  function highlight(text) {
    const re = /("[^"]*"|'[^']*')|(#.*$)|(\d+(?:\.\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|(\s+)|([\s\S])/g;
    let out = '', m, prevCh = '';
    while ((m = re.exec(text)) !== null) {
      if (m[1]) out += '<span class="t-str">' + esc(m[1]) + '</span>';
      else if (m[2]) out += '<span class="t-com">' + esc(m[2]) + '</span>';
      else if (m[3]) out += '<span class="t-num">' + esc(m[3]) + '</span>';
      else if (m[4]) {
        const id = m[4];
        const rest = text.slice(re.lastIndex);
        const nextCh = (rest.match(/^\s*(.)/) || [])[1] || '';
        let cls = 't-id';
        if (KW[id]) cls = 't-kw';
        else if (nextCh === '(') cls = 't-fn';
        else if (nextCh === '=' && rest.replace(/^\s*/, '').slice(0, 2) !== '==') cls = 't-kwarg';
        else if (prevCh === '.') cls = 't-attr';
        else if (/^[A-Z][A-Z0-9_]*$/.test(id)) cls = 't-const';
        else if (/^[A-Z]/.test(id)) cls = 't-type';
        out += '<span class="' + cls + '">' + esc(id) + '</span>';
      } else if (m[5]) out += esc(m[5]);
      else { out += esc(m[6]); }
      if (m[0].trim()) prevCh = m[0][m[0].length - 1];
    }
    return out;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 索引：四类对象互相反查
  // ════════════════════════════════════════════════════════════════════════════
  function buildIndex(kernel, platform, anchors) {
    const bufBySpace = {};
    (kernel.buffers || []).forEach(function (b) { bufBySpace[C.bufKey(b)] = b.space; });

    const nodes = kernel.nodes.map(function (n) {
      // 节点携带的 buffer：显式 bufs，否则看它的赋值目标是不是一个 buffer
      let bufs = n.bufs ? n.bufs.slice() : [];
      if (!n.bufs) {
        [n.assigns].concat(n.also || []).forEach(function (name) {
          if (name && bufBySpace[name]) bufs.push(name);
        });
      }
      // 区域：显式 regions + 它的 buffer 所在空间对应的区域
      let regions = (n.regions || []).slice();
      bufs.forEach(function (bid) {
        const r = platform.spaceRegion[bufBySpace[bid]];
        if (r && regions.indexOf(r) < 0) regions.push(r);
      });
      return {
        id: n.id, label: n.label || n.id, node: n, frame: n.frame,
        lines: anchors.lines[n.id] || [], regions: regions, bufs: C.uniq(bufs)
      };
    });

    const frames = (kernel.frames || []).map(function (f) {
      return {
        id: f.id, label: f.label || f.id, frame: f,
        lines: anchors.lines[f.id] || [], regions: (f.regions || []).slice(),
        bufs: (f.bufs === 'all'
          ? (kernel.buffers || []).filter(function (b) { return b.frame === f.id; })
            .map(function (b) { return C.bufKey(b); })
          : (f.bufs || []).slice())
      };
    });

    const all = nodes.concat(frames);
    const byId = {};
    all.forEach(function (o) { byId[o.id] = o; });
    return { nodes: nodes, frames: frames, all: all, byId: byId, bufSpace: bufBySpace };
  }

  // 悬停目标 → 四个要点亮的集合
  function focusFor(idx, hover) {
    const empty = { lines: [], regions: [], bufs: [], nodes: [] };
    if (!hover) return empty;
    let hits = [];
    if (hover.type === 'node') hits = idx.byId[hover.id] ? [idx.byId[hover.id]] : [];
    else if (hover.type === 'line') {
      hits = idx.all.filter(function (o) { return o.lines.indexOf(hover.no) >= 0; });
    } else if (hover.type === 'region') {
      hits = idx.all.filter(function (o) { return o.regions.indexOf(hover.id) >= 0; });
    } else if (hover.type === 'buf') {
      hits = idx.all.filter(function (o) { return o.bufs.indexOf(hover.id) >= 0; });
    }
    const f = { lines: [], regions: [], bufs: [], nodes: [] };
    hits.forEach(function (o) {
      f.nodes.push(o.id);
      f.lines = f.lines.concat(o.lines);
      f.regions = f.regions.concat(o.regions);
      f.bufs = f.bufs.concat(o.bufs);
    });
    if (hover.type === 'region' && f.regions.indexOf(hover.id) < 0) f.regions.push(hover.id);
    if (hover.type === 'buf' && f.bufs.indexOf(hover.id) < 0) f.bufs.push(hover.id);
    if (hover.type === 'line' && f.lines.indexOf(hover.no) < 0) f.lines.push(hover.no);
    f.lines = C.uniq(f.lines); f.regions = C.uniq(f.regions);
    f.bufs = C.uniq(f.bufs); f.nodes = C.uniq(f.nodes);
    return f;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 一区 · 源码
  // ════════════════════════════════════════════════════════════════════════════
  // {{CONST}} 处代入当前参数值；锚点匹配跑在原始文本上，所以代入不影响定位
  function substitute(text, consts) {
    return text.replace(/\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}/g, function (m, name) {
      return consts[name] == null ? m : String(consts[name]);
    });
  }

  function renderSource(kernel, idx, consts) {
    const mapped = {};
    idx.all.forEach(function (o) { o.lines.forEach(function (no) { mapped[no] = true; }); });
    let prev = null, html = '';
    kernel.source.forEach(function (r) {
      if (prev !== null && r.no > prev + 1) {
        html += '<div class="kb-gap"><span>⋯ ' + (r.no - prev - 1) + ' 行省略</span></div>';
      }
      prev = r.no;
      html += '<div class="kb-line' + (mapped[r.no] ? ' is-live' : '') + '" data-line="' + r.no + '"' + (mapped[r.no] ? ' tabindex="0" role="button" aria-label="源码第 ' + r.no + ' 行"' : '') + '>'
        + '<i class="kb-no">' + r.no + '</i>'
        + '<code>' + highlight(substitute(r.t, consts)) + '</code></div>';
    });
    return html;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 二区 · 算子逻辑（坐标全部来自 core.layoutGraph）
  // ════════════════════════════════════════════════════════════════════════════
  function txt(x, y, s, cls, anchor) {
    return '<text x="' + x + '" y="' + y + '" class="' + (cls || '') + '"'
      + (anchor ? ' text-anchor="' + anchor + '"' : '') + '>' + esc(s) + '</text>';
  }

  function renderGraph(g, kernel, cap, consts) {
    const lab = function (s) { return substitute(String(s), consts || {}); };
    // 越过可用上限的 buffer 属于哪个节点 —— 逻辑图上标红的就是它
    const badBufs = [];
    Object.keys(cap.spaces).forEach(function (sid) {
      const o = cap.spaces[sid].offender;
      if (o) badBufs.push(o.bufId);
    });
    const badNodes = kernel.nodes.filter(function (n) {
      const bufs = n.bufs || [n.assigns];
      return bufs.some(function (b) { return badBufs.indexOf(b) >= 0; });
    }).map(function (n) { return n.id; });

    let s = '<svg viewBox="' + g.viewBox + '" role="img" aria-label="'
      + esc(kernel.entry + ' 的循环嵌套与计算链') + '">'
      + '<defs><marker id="kbArrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7"'
      + ' markerHeight="7" orient="auto"><path d="M0 1 L7 4 L0 7 z" class="kb-arrow"/></marker></defs>';

    g.frames.forEach(function (f) {
      const r = f.rect;
      s += '<g class="kb-hit" tabindex="0" role="button" aria-label="' + esc(lab(f.label)) + '" data-node="' + f.id + '">'
        + '<rect x="' + r.x + '" y="' + r.y + '" width="' + r.w + '" height="' + r.h + '" class="kb-frame"/>'
        + txt(r.x + 10, r.y + 16, lab(f.label), 'kb-t-frame') + '</g>';
    });
    g.phases.forEach(function (p) {
      if (!p.label) return;
      s += '<g class="kb-phase-g"><rect x="' + p.x + '" y="' + p.y + '" width="' + p.w
        + '" height="' + p.h + '" class="kb-phase"/>'
        + txt(p.x + 10, p.y + 15, lab(p.label), 'kb-t-phase') + '</g>';
    });
    g.edges.forEach(function (e) {
      s += '<path d="' + e.d + '" class="kb-edge" data-from="' + e.from + '" data-to="' + e.to
        + '" marker-end="url(#kbArrow)"/>';
    });
    Object.keys(g.nodes).forEach(function (id) {
      const n = g.nodes[id], bad = badNodes.indexOf(id) >= 0;
      s += '<g class="kb-hit" tabindex="0" role="button" aria-label="' + esc(lab(n.label)) + '" data-node="' + id + '">'
        + '<rect x="' + n.x + '" y="' + n.y + '" width="' + n.w + '" height="' + n.h
        + '" class="kb-node' + (bad ? ' is-bad' : '') + '"/>'
        + txt(n.x + 8, n.y + 17, lab(n.label), bad ? 'kb-t-node is-bad' : 'kb-t-node') + '</g>';
    });
    return s + '</svg>';
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 三区 · 硅片落点（盒子坐标来自 core.layoutBoxes）
  // ════════════════════════════════════════════════════════════════════════════
  function fillBar(r, sp) {
    const w = r.w - 16, x = r.x + 8, y = r.y + r.h - 16;
    const ratio = sp.nominal ? Math.min(1, sp.used / sp.nominal) : 0;
    const cls = sp.verdict === 'ok' ? 'kb-fill' : 'kb-fill is-' + sp.verdict;
    return '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="6" class="kb-track"/>'
      + '<rect x="' + x + '" y="' + y + '" width="' + (w * ratio) + '" height="6" class="' + cls + '"/>';
  }

  // Vec 那条地址条：竖着画，0 在下。保留带、越界块、两条上限都在上面。
  function addressBar(r, sp) {
    const barX = r.x + 10, barW = 34;
    const top = r.y + 30, bottom = r.y + r.h - 26;
    const scale = Math.max(sp.nominal, sp.used) || 1;
    const yOf = function (a) { return bottom - a / scale * (bottom - top); };
    const labX = barX + barW + 10;
    let s = '<rect x="' + barX + '" y="' + top + '" width="' + barW + '" height="' + (bottom - top)
      + '" class="kb-ub-track"/>';
    if (sp.reserved > 0) {
      const ry = yOf(sp.nominal), rh = yOf(sp.usable) - yOf(sp.nominal);
      s += '<rect x="' + barX + '" y="' + ry + '" width="' + barW + '" height="' + rh
        + '" class="kb-ub-reserved"/>';
    }
    sp.items.forEach(function (it) {
      const y0 = yOf(it.to), h = Math.max(1.2, yOf(it.from) - yOf(it.to));
      const cls = it.inReserved ? 'is-reserved' : it.crosses ? 'is-crosses'
        : it.resident ? 'is-resident' : 'is-copy';
      s += '<g class="kb-buf" tabindex="0" role="button" aria-label="' + esc(it.label) + '" data-buf="' + esc(it.bufId) + '">'
        + '<rect x="' + barX + '" y="' + y0 + '" width="' + barW + '" height="' + h
        + '" class="kb-ub-seg ' + cls + '"><title>' + esc(it.label + ' · ' + bytes(it.size))
        + '</title></rect></g>';
    });
    s += '<line x1="' + (barX - 4) + '" y1="' + yOf(sp.usable) + '" x2="' + (barX + barW + 4)
      + '" y2="' + yOf(sp.usable) + '" class="kb-ub-limit is-usable"/>';
    s += '<line x1="' + (barX - 4) + '" y1="' + yOf(sp.nominal) + '" x2="' + (barX + barW + 4)
      + '" y2="' + yOf(sp.nominal) + '" class="kb-ub-limit is-nominal"/>';
    s += txt(labX, yOf(sp.nominal) + 4, kb(sp.nominal) + ' 名义', 'kb-t-tick');
    s += txt(labX, yOf(sp.usable) + 13, kb(sp.usable) + ' 可用上限', 'kb-t-tick is-usable');
    s += txt(labX, bottom - 2, '0', 'kb-t-tick');
    s += txt(labX, top + 16, '高水位 ' + kb(sp.used), sp.verdict === 'ok' ? 'kb-t-hw' : 'kb-t-hw is-bad');
    if (sp.offender) {
      s += txt(labX, top + 30, '越界 ' + sp.offender.label, 'kb-t-tick is-usable');
    }
    return s;
  }

  // a2a3 工作台示意：坐标与主线画布一致，数据仍来自平台和 kernel。
  function renderA2A3Chip(platform, cap, kernel, consts) {
    const lab = function (s) { return substitute(String(s), consts || {}); };
    const bad = cap.worst === 'silent' || cap.worst === 'over';
    const rect = function (x, y, w, h, cls) { return '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" class="' + cls + '"/>'; };
    const region = function (id, content) { return '<g class="kb-region" tabindex="0" role="button" aria-label="' + esc(id) + '" data-region="' + esc(id) + '">' + content + '</g>'; };
    let s = '<svg viewBox="0 0 720 470" role="img" aria-label="' + esc(kernel.entry + ' 在一个 AI Core block 上的落点') + '">'
      + '<defs><marker id="kbChipArrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#555"/></marker>'
      + '<pattern id="kbHatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="5" height="5" fill="#2a1716"/><line y2="5" stroke="#ff6b5b" stroke-width="1.4" opacity=".55"/></pattern>'
      + '<clipPath id="kbInsetClip"><rect x="560" y="300" width="116" height="112"/></clipPath></defs>';
    s += rect(12, 8, 696, 74, 'kb-chip-frame') + txt(24, 25, 'GM · HBM', 'kb-chip-title');
    const tensors = kernel.tensors || [];
    const count = Math.max(1, tensors.length), gap = 8, left = 24, usable = 672, w = (usable - gap * (count - 1)) / count;
    tensors.forEach(function (t, i) {
      const x = left + i * (w + gap);
      let content = rect(x, 32, w, 42, 'kb-chip-mem') + txt(x + 6, 46, lab(t.label || t.id), 'kb-chip-strong');
      if (t.note) content += txt(x + 6, 64, lab(t.note), 'kb-t-note');
      s += region((t.space || 'gm') + ':' + t.id, content);
    });
    let cores = rect(284, 86, 424, 24, 'kb-chip-hit') + txt(376, 102, 'PL.SPMD × ' + platform.cores, 'kb-chip-title', 'end');
    for (let i = 0; i < platform.cores; i++) cores += rect(384 + i * 13, 93, 10, 10, 'kb-core' + (i === 0 ? ' is-zoom' : '') + (bad ? ' is-bad' : ''));
    s += region('cores', cores) + '<line x1="389" y1="103" x2="389" y2="122" class="kb-chip-zoom"/>';
    s += rect(12, 122, 696, 340, 'kb-chip-frame') + txt(397, 137, '展开 BLOCK 0 · ' + platform.cores + ' 个 block 用同一份片上分配', 'kb-chip-title');
    s += region('aic', rect(24, 144, 262, 308, 'kb-chip-unit') + txt(36, 162, 'AIC · CUBE', 'kb-chip-title'));
    s += region('mat', rect(36, 170, 238, 26, 'kb-chip-mem' + (cap.spaces.mat.items.length ? '' : ' is-dim')) + txt(44, 187, 'L1 · Mat', 'kb-t-frame') + txt(266, 187, cap.spaces.mat.items.length ? kb(cap.spaces.mat.used) + ' 高水位' : '本例未建模', 'kb-t-frame', 'end'));
    function memory(id, space, x, y, width, height, title) {
      const sp = cap.spaces[space];
      let content = rect(x, y, width, height, 'kb-chip-mem') + txt(x + 8, y + 15, title, 'kb-t-frame');
      if (sp && sp.modeled !== false && sp.items.length) {
        content += rect(x + 8, y + 22, width - 16, 7, 'kb-track');
        content += rect(x + 8, y + 22, (width - 16) * Math.min(1, sp.used / sp.nominal), 7, 'kb-fill' + (sp.verdict === 'ok' ? '' : ' is-' + sp.verdict));
        content += txt(x + 8, y + height - 7, kb(sp.used) + ' / ' + kb(sp.nominal) + ' 估算', 'kb-t-note');
      } else content += txt(x + 8, y + height - 7, '本例未建模', 'kb-t-note');
      return region(id, content);
    }
    s += memory('l0a', 'left', 36, 204, 112, 58, 'L0A · LEFT');
    s += memory('l0b', 'right', 156, 204, 118, 58, 'L0B · RIGHT');
    s += region('cube', rect(36, 270, 238, 56, 'kb-chip-mem') + txt(44, 285, 'CUBE', 'kb-t-frame') + txt(44, 305, kernel.entry, 'kb-chip-strong'));
    s += memory('l0c', 'acc', 36, 334, 238, 52, 'L0C · ACC');
    s += region('ch-c2v', rect(288, 288, 60, 24, 'kb-chip-hit') + '<path d="M288 300 H348" class="kb-chip-arrow"/>' + txt(318, 294, 'C2V', 'kb-t-frame', 'middle'));
    s += region('ch-v2c', rect(288, 318, 60, 24, 'kb-chip-hit') + '<path d="M348 330 H288" class="kb-chip-arrow"/>' + txt(318, 344, 'V2C', 'kb-t-frame', 'middle'));
    s += region('aiv', rect(350, 144, 346, 308, 'kb-chip-unit') + txt(362, 162, 'AIV · VECTOR', 'kb-chip-title'));
    const sp = cap.spaces.vec;
    if (sp && sp.nominal) {
      const top = 186, bottom = 440, x = 418, bw = 60;
      const y = function (v) { return bottom - Math.min(sp.nominal, Math.max(0, v)) / sp.nominal * (bottom - top); };
      const insetLo = Math.max(0, sp.nominal - 16 * 1024), insetTop = 300, insetBottom = 412;
      const iy = function (v) { return insetBottom - (v - insetLo) / (sp.nominal - insetLo) * (insetBottom - insetTop); };
      s += region('ub', rect(x, top, bw, bottom - top, 'kb-chip-mem') + txt(x, top - 8, 'VEC · UB', 'kb-t-frame'));
      sp.items.forEach(function (it) {
        const y1 = y(it.to), h = Math.max(0, y(it.from) - y1);
        if (h) s += '<g class="kb-buf" tabindex="0" role="button" aria-label="' + esc(it.label) + '" data-buf="' + esc(it.bufId) + '">' + rect(x + 1, y1, bw - 2, h, 'kb-ub-seg ' + (it.resident ? 'is-resident' : 'is-copy')) + '<title>' + esc(it.label + ' · ' + bytes(it.size)) + '</title></g>';
      });
      if (sp.reserved) s += rect(x, top, bw, y(sp.usable) - top, 'kb-ub-reserved');
      if (sp.used > sp.usable) s += rect(x + 1, y(sp.used), bw - 2, Math.max(1.5, y(sp.usable) - y(sp.used)), 'kb-ub-over');
      s += '<line x1="414" y1="186" x2="492" y2="186" class="kb-ub-limit is-nominal"/>'
        + '<line x1="414" y1="' + y(sp.usable) + '" x2="492" y2="' + y(sp.usable) + '" class="kb-ub-limit is-usable"/>'
        + txt(496, 189, kb(sp.nominal) + ' · 分配器预算', 'kb-t-frame')
        + txt(496, y(sp.usable) + 9, kb(sp.usable) + ' · 实际可用', 'kb-t-tick is-usable');
      const hw = y(sp.used), ubBad = sp.verdict === 'silent' || sp.verdict === 'over';
      s += '<line x1="408" y1="' + hw + '" x2="418" y2="' + hw + '" class="kb-chip-water' + (ubBad ? ' is-bad' : '') + '"/>'
        + txt(406, hw - 2, '高水位', 'kb-t-frame', 'end') + txt(406, hw + 10, kb(sp.used), ubBad ? 'kb-t-hw is-bad' : 'kb-chip-strong', 'end');
      s += '<line x1="478" y1="' + y(sp.nominal) + '" x2="560" y2="300" class="kb-chip-zoom"/>'
        + '<line x1="478" y1="' + y(insetLo) + '" x2="560" y2="412" class="kb-chip-zoom"/>'
        + txt(560, 282, '顶部 16 KB', 'kb-t-frame') + txt(560, 293, '放大', 'kb-t-frame') + rect(560, 300, 116, 112, 'kb-chip-mem') + '<g clip-path="url(#kbInsetClip)">';
      sp.items.forEach(function (it) {
        if (it.to <= insetLo || it.from >= sp.nominal) return;
        const y1 = iy(Math.min(it.to, sp.nominal)), h = iy(Math.max(it.from, insetLo)) - y1;
        s += '<g class="kb-buf" tabindex="0" role="button" aria-label="' + esc(it.label) + '" data-buf="' + esc(it.bufId) + '">' + rect(561, y1, 114, Math.max(1.5, h), 'kb-ub-seg ' + (it.resident ? 'is-resident' : 'is-copy')) + '<title>' + esc(it.label) + '</title></g>';
      });
      if (sp.reserved) s += rect(560, 300, 116, iy(sp.usable) - 300, 'kb-ub-reserved');
      if (sp.used > sp.usable) s += rect(561, iy(Math.min(sp.used, sp.nominal)), 114, iy(sp.usable) - iy(Math.min(sp.used, sp.nominal)), 'kb-ub-over');
      s += '</g><line x1="560" y1="' + iy(sp.usable) + '" x2="676" y2="' + iy(sp.usable) + '" class="kb-ub-limit is-usable"/>'
        + txt(618, iy(sp.usable) - 18, '保留 ' + kb(sp.reserved), 'kb-t-tick is-usable', 'middle')
        + txt(560, 425, kb(insetLo), 'kb-t-frame')
        + txt(560, 440, sp.used > sp.usable ? '越界 ' + bytes(sp.used - sp.usable) : '余量 ' + bytes(sp.usable - sp.used), sp.used > sp.usable ? 'kb-t-tick is-usable' : 'kb-chip-good');
    }
    return s + '</svg>';
  }

  function renderChip(bx, platform, cap, kernel, consts) {
    if (platform.id === 'a2a3') return renderA2A3Chip(platform, cap, kernel, consts);
    const lab = function (s) { return substitute(String(s), consts || {}); };
    let s = '<svg viewBox="0 0 ' + bx.view.w + ' ' + bx.view.h + '" role="img" aria-label="'
      + esc(kernel.entry + ' 在一个 AI Core block 上的落点') + '">'
      + '<defs><pattern id="kbHatch" width="6" height="6" patternUnits="userSpaceOnUse"'
      + ' patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" class="kb-hatch"/>'
      + '</pattern></defs>';
    bx.boxes.forEach(function (b) {
      const r = b.rect, sp = b.space ? cap.spaces[b.space] : null;
      // 没有 kind 的是纯排版容器；slots 容器的位置由展开出来的子项占据
      if (!b.kind || b.kind === 'slots') return;
      let inner = '';
      if (b.kind === 'frame') {
        inner = '<rect x="' + r.x + '" y="' + r.y + '" width="' + r.w + '" height="' + r.h
          + '" class="kb-chip-frame"/>' + (b.label ? txt(r.x + 10, r.y + 15, lab(b.label), 'kb-t-frame') : '');
      } else if (b.kind === 'cores') {
        const n = platform.cores, pitch = Math.min(14, (r.w - 150) / n);
        inner = txt(r.x, r.y + 14, 'PL.SPMD × ' + n, 'kb-t-frame');
        const bad = cap.worst === 'silent' || cap.worst === 'over';
        for (let i = 0; i < n; i++) {
          inner += '<rect x="' + (r.x + 132 + i * pitch) + '" y="' + (r.y + 4)
            + '" width="' + Math.max(4, pitch - 3) + '" height="11" class="kb-core'
            + (bad ? ' is-bad' : '') + '"/>';
        }
      } else if (b.kind === 'bar' && sp) {
        inner = '<rect x="' + r.x + '" y="' + r.y + '" width="' + r.w + '" height="' + r.h
          + '" class="kb-chip-mem"/>' + txt(r.x + 8, r.y + 16, lab(b.label), 'kb-t-mem')
          + addressBar(r, sp);
      } else {
        const cls = b.kind === 'chan' ? 'kb-chip-chan' : b.kind === 'unit' ? 'kb-chip-unit' : 'kb-chip-mem';
        const dim = sp && sp.modeled === false;
        inner = '<rect x="' + r.x + '" y="' + r.y + '" width="' + r.w + '" height="' + r.h
          + '" class="' + cls + (dim ? ' is-dim' : '') + '"/>'
          + (b.label ? txt(r.x + 8, r.y + 15, lab(b.label), 'kb-t-mem') : '');
        if (b.note) inner += txt(r.x + 8, r.y + 29, lab(b.note), 'kb-t-note');
        if (sp && sp.modeled !== false && sp.items.length) {
          inner += txt(r.x + 8, r.y + 30, kb(sp.used) + ' / ' + kb(sp.nominal), 'kb-t-note')
            + fillBar(r, sp);
        } else if (sp && sp.modeled === false) {
          inner += txt(r.x + r.w - 8, r.y + 15, '本例未建模', 'kb-t-note', 'end');
        }
      }
      s += '<g class="kb-region" data-region="' + b.id + '">' + inner + '</g>';
    });
    return s + '</svg>';
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 判定条 + 旋钮 + 说明
  // ════════════════════════════════════════════════════════════════════════════
  const VERDICT_TEXT = {
    ok: '未触发', tight: '接近上限', silent: '分配器放行但越过可用上限', over: '超出名义容量'
  };

  function renderVerdicts(cap, platform) {
    const ids = Object.keys(cap.spaces).filter(function (id) {
      const sp = cap.spaces[id];
      return sp.modeled !== false && sp.items.length;
    });
    if (!ids.length) return '<p class="kb-empty">当前 kernel 没有声明片上 buffer。</p>';
    return ids.map(function (id) {
      const sp = cap.spaces[id];
      let detail;
      if (sp.verdict === 'over') {
        detail = '超出名义容量 ' + bytes(sp.used - sp.nominal) + '，分配器自己会报错。';
      } else if (sp.verdict === 'silent') {
        detail = '越过可用上限 ' + bytes(sp.over) + '，距名义上限还有 ' + bytes(sp.underNominal)
          + '；分配器按名义容量放行，落进保留区的是 ' + (sp.offender ? sp.offender.label : '最高地址那块') + '。';
      } else {
        detail = '占可用容量 ' + Math.round(sp.pct * 100) + '%，距上限还有 ' + bytes(sp.headroom) + '。'
          + (sp.reserved ? ' 可用 = 名义 ' + kb(sp.nominal) + ' − 保留 ' + kb(sp.reserved) + '。' : '');
      }
      return '<button type="button" class="kb-verdict is-' + sp.verdict + '" data-region="'
        + sp.region + '">'
        + '<span class="kb-v-head"><b>' + esc(sp.label) + '</b>'
        + '<em>' + VERDICT_TEXT[sp.verdict] + '</em></span>'
        + '<span class="kb-v-num">' + kb(sp.used) + ' / ' + kb(sp.usable) + ' 可用</span>'
        + '<span class="kb-v-detail">' + esc(detail) + '</span></button>';
    }).join('');
  }

  function renderKnobs(kernel, cfg) {
    return (kernel.knobs || []).map(function (k) {
      return '<label class="kb-knob"><span>' + esc(k.label || k.id) + '</span>'
        + '<select data-knob="' + k.id + '">'
        + k.options.map(function (o) {
          return '<option value="' + o + '"' + (cfg[k.id] === o ? ' selected' : '') + '>' + o + '</option>';
        }).join('') + '</select></label>';
    }).join('');
  }

  function caption(idx, cap, kernel, consts, hover) {
    if (!hover) {
      return '悬停源码行、逻辑节点、硅片区域或地址条上的任意一块 —— 另外两区会同时点亮。';
    }
    const f = focusFor(idx, hover);
    const objs = f.nodes.map(function (id) { return idx.byId[id]; }).filter(Boolean);
    if (!objs.length) return '这一处没有挂到任何逻辑节点。';
    const o = objs[0];
    const parts = [];
    parts.push('<b>' + esc(substitute(String(o.label), consts)) + '</b>');
    if (o.lines.length) parts.push('源码 L' + o.lines.join(' / L'));
    if (o.regions.length) parts.push('落点 ' + o.regions.join(' · '));
    o.bufs.forEach(function (bid) {
      const b = (kernel.buffers || []).filter(function (x) { return C.bufKey(x) === bid; })[0];
      if (!b && kernel.compiledAllocations) {
        const a = kernel.compiledAllocations.find(function (item) { return item.worker + ':' + item.id === bid; });
        if (a) parts.push(esc(a.id) + ' · ' + esc(a.space) + ' · 编译地址 ' + (a.address == null ? '未知' : a.address) + ' · ' + bytes(a.size) + ' · 输出 L' + a.allocLine + '（名称匹配为推断）');
        return;
      }
      if (!b) return;
      const sp = cap.spaces[b.space];
      const it = sp ? sp.items.filter(function (i) { return i.bufId === bid; })[0] : null;
      const shape = (b.shape || []).map(function (v) {
        return typeof v === 'number' ? v : consts[v];
      }).join(' × ');
      parts.push(esc(bid) + ' [' + shape + '] ' + (C.DTYPE[b.dtype] || {}).label
        + (it ? ' · ' + bytes(it.size) + ' × ' + it.count + ' 份' : ''));
    });
    return parts.join('<i>|</i>');
  }

  root.HNKitView = {
    highlight: highlight, buildIndex: buildIndex, focusFor: focusFor,
    renderSource: renderSource, renderGraph: renderGraph, renderChip: renderChip,
    substitute: substitute,
    renderVerdicts: renderVerdicts, renderKnobs: renderKnobs, caption: caption,
    VERDICT_TEXT: VERDICT_TEXT
  };
})(typeof window !== 'undefined' ? window : this);
