/* CSA Operator Lens · 算子卡片的可视化：功能摘要、公式排版、按比例的形状示例、场景矩阵、限制清单。
 * 只产出 HTML / SVG 字符串，交给 lens.js 放进右侧详情；不依赖外部库。 */
(function (root) {
  'use strict';
  const esc = function (s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); };

  // ── 简化 TeX → HTML：\frac \sqrt \text \sum/\max 上下限、上下标、希腊字母与常用运算符 ──
  const SYM = { alpha: 'α', beta: 'β', gamma: 'γ', sigma: 'σ', tau: 'τ', varepsilon: 'ε', epsilon: 'ε', theta: 'θ', lambda: 'λ', mu: 'μ', pi: 'π',
    odot: '⊙', cdot: '·', times: '×', leftarrow: '←', rightarrow: '→', in: '∈', le: '≤', ge: '≥', ne: '≠', approx: '≈', lfloor: '⌊', rfloor: '⌋',
    land: '∧', lor: '∨', top: '⊤', ldots: '…', infty: '∞', bmod: ' mod ', mid: '|', partial: '∂' };
  const BIG = { sum: 'Σ', prod: 'Π' };
  const OPN = { max: 'max', min: 'min', exp: 'exp', log: 'log' };
  function tex(src) {
    let i = 0;
    function group() {
      while (src[i] === ' ') i++;
      if (src[i] === '{') { i++; const h = seq('}'); i++; return h; }
      if (src[i] === '\\') return cmd();
      const c = src[i++] || '';
      return /[A-Za-z]/.test(c) ? '<i>' + c + '</i>' : esc(c);
    }
    function scripts(base) {
      let sub = '', sup = '';
      for (;;) {
        if (src[i] === '_') { i++; sub = group(); continue; }
        if (src[i] === '^') { i++; sup = group(); continue; }
        break;
      }
      if (!sub && !sup) return base;
      if (!sub) return base + '<sup>' + sup + '</sup>';
      if (!sup) return base + '<sub>' + sub + '</sub>';
      return base + '<span class="mx-ss"><sup>' + sup + '</sup><sub>' + sub + '</sub></span>';
    }
    function cmd() {
      i++;
      if (src[i] === '\\') { i++; return '<br>'; }
      if (src[i] === ',') { i++; return '<span class="mx-sp1"></span>'; }
      const m = /^[A-Za-z]+/.exec(src.slice(i)); const name = m ? m[0] : src[i]; i += name.length;
      if (name === 'frac') { const a = group(), b = group(); return '<span class="mx-frac"><span>' + a + '</span><span>' + b + '</span></span>'; }
      if (name === 'sqrt') return '<span class="mx-sqrt"><span class="mx-rad">√</span><span class="mx-sqrt-b">' + group() + '</span></span>';
      if (name === 'text' || name === 'operatorname') return '<span class="mx-t">' + group().replace(/<\/?i>/g, '') + '</span>';
      if (name === 'mathbb') return '<span class="mx-bb">' + ({ R: 'ℝ', N: 'ℕ', Z: 'ℤ' }[group().replace(/<\/?i>/g, '')] || '') + '</span>';
      if (name === 'hat') return '<span class="mx-hat">' + group() + '</span>';
      if (name === 'quad') return '<span class="mx-sp"></span>';
      if (BIG[name]) {
        let sub = '', sup = '';
        for (;;) { if (src[i] === '_') { i++; sub = group(); continue; } if (src[i] === '^') { i++; sup = group(); continue; } break; }
        return '<span class="mx-big"><span class="mx-lim">' + sup + '</span><span class="mx-big-s">' + BIG[name] + '</span><span class="mx-lim">' + sub + '</span></span>';
      }
      if (OPN[name]) {
        let sub = '';
        if (src[i] === '_') { i++; sub = group(); }
        return '<span class="mx-op">' + OPN[name] + (sub ? '<sub>' + sub + '</sub>' : '') + '</span>';
      }
      if (SYM[name]) return '<span class="mx-s">' + SYM[name] + '</span>';
      return esc(name);
    }
    function seq(stop) {
      let h = '';
      while (i < src.length && src[i] !== stop) {
        const c = src[i];
        if (c === '\\') h += scripts(cmd());
        else if (c === '{') { i++; const g = seq('}'); i++; h += scripts(g); }
        else if (/[A-Za-z]/.test(c)) {
          let w = ''; while (i < src.length && /[A-Za-z]/.test(src[i])) w += src[i++];
          h += scripts(w.length > 1 && /^(mean|rms|quant|dequant|softmax|RMSNorm|RoPE|Sinkhorn|ReLU|Top|concat|done|cache|rank|owner|pre|post|comb|cmp|ape|score|idx|ready|bias|sink|kv|cos|sin|bf|full|fn|row|rows|il|qa|qb|kv|gate|ori)$/.test(w) ? '<span class="mx-t">' + w + '</span>' : '<i>' + w + '</i>');
        } else if (/[0-9.]/.test(c)) { let n = ''; while (i < src.length && /[0-9.]/.test(src[i])) n += src[i++]; h += scripts('<span class="mx-n">' + n + '</span>'); }
        else if (c === ' ') { i++; h += ' '; }
        else if (/[=+\-<>]/.test(c)) { i++; h += '<span class="mx-rel">' + esc(c === '-' ? '−' : c) + '</span>'; }
        else if (c === '_' || c === '^') h = scripts(h);
        else { i++; h += scripts(esc(c)); }
      }
      return h;
    }
    return seq(undefined);
  }

  // ── 按比例的形状图：宽 ∝ log₂(列)，高 ∝ log₂(行)，三维画成叠层；分段 = 行的归属 ──
  function blockSize(t) {
    const d = t.dims, cols = d[d.length - 1], rows = d.length > 1 ? d[d.length - 2] : 1, depth = d.length > 2 ? d[0] : 1;
    const f = function (n, a, b, lo, hi) { return Math.max(lo, Math.min(hi, a + b * Math.log2(Math.max(1, n)))); };
    return { w: f(cols, 8, 7.5, 16, 112), h: f(rows, 6, 5.2, 8, 70), depth: depth, rows: rows, cols: cols };
  }
  function shapeSvg(T, ids, title) {
    const W = 300, gap = 14, items = ids.map(function (id) { return { id: id, t: T[id], s: blockSize(T[id]) }; }).filter(function (x) { return x.t; });
    // 简单换行排布
    let x = 0, y = 18, rowH = 0, out = '';
    const placed = [];
    items.forEach(function (it) {
      const lay = it.s.depth > 1 ? 2 : 0, w = it.s.w + lay * 6 + 4, h = it.s.h + lay * 6 + 30;
      const needW = Math.max(w, Math.min(W, it.t.name.length * 6.2 + 8));
      if (x > 0 && x + needW > W) { x = 0; y += rowH + 10; rowH = 0; }
      placed.push({ it: it, x: x, y: y, lay: lay, w: needW });
      x += needW + gap; rowH = Math.max(rowH, h);
    });
    const H = y + rowH + 4;
    placed.forEach(function (p) {
      const t = p.it.t, s = p.it.s, by = p.y + 14;
      out += '<text class="oc-sh-name" x="' + p.x + '" y="' + (p.y + 9) + '">' + esc(t.name.length > 28 ? t.name.slice(0, 27) + '…' : t.name) + '</text>';
      // 三维张量：后面再叠两层，向右上错开，表示第一维（如 4 路 HC、64 个头）
      for (let l = p.lay; l > 0; l--) out += '<rect class="oc-sh-b sc-' + t.sc + ' is-back" x="' + (p.x + l * 6) + '" y="' + (by + p.lay * 5 - l * 5) + '" width="' + s.w + '" height="' + s.h + '" rx="2"/>';
      out += '<rect class="oc-sh-b sc-' + t.sc + '" x="' + (p.x) + '" y="' + (by + p.lay * 5) + '" width="' + s.w + '" height="' + s.h + '" rx="2"/>';
      if (t.seg) {
        let acc = 0;
        t.seg.forEach(function (sg, k) {
          const sy = by + p.lay * 5 + s.h * acc;
          if (k > 0) out += '<line class="oc-sh-seg" x1="' + p.x + '" x2="' + (p.x + s.w) + '" y1="' + sy + '" y2="' + sy + '"/>';
          if (s.h * sg[1] >= 9 && s.w >= 30) out += '<text class="oc-sh-segt" x="' + (p.x + 3) + '" y="' + (sy + Math.min(s.h * sg[1], 12) - 2) + '">' + esc(sg[0]) + '</text>';
          acc += sg[1];
        });
      }
      const dimTxt = t.dims.join(' × ');
      out += '<text class="oc-sh-dim" x="' + p.x + '" y="' + (by + p.lay * 5 + s.h + 11) + '">' + esc(dimTxt) + ' · ' + esc(t.dt) + '</text>';
    });
    return '<div class="oc-sh-k">' + esc(title) + '</div><svg class="oc-shape" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' + esc(title + '：' + items.map(function (i) { return i.t.name + ' ' + i.t.dims.join('×'); }).join('，')) + '">' + out + '</svg>';
  }

  // ── 场景矩阵与限制 ──
  function scenRows(rows, label) {
    if (!rows.length) return '';
    return '<div class="oc-scn">' + (label ? '<div class="oc-scn-cap">' + esc(label) + '</div>' : '') + rows.map(function (r) {
      return '<div class="oc-scn-row"><span class="oc-scn-k">' + esc(r[0]) + '</span><span class="oc-scn-v">' + r[1].map(function (o) {
        return '<span class="oc-opt is-' + o[1] + '"' + (o[2] ? ' title="' + esc(o[2]) + '"' : '') + '>' + (o[1] === 'off' ? '<s>' + esc(o[0]) + '</s>' : esc(o[0])) + (o[1] === 'def' ? '<i>默认</i>' : '') + '</span>';
      }).join('') + '</span>' + (r[2] ? '<span class="oc-scn-n">' + esc(r[2]) + '</span>' : '') + '</div>';
    }).join('') + '</div>';
  }
  const LIM = { fix: ['固定', 'M7 11V8a5 5 0 0 1 10 0v3M5 11h14v10H5z'], div: ['整除', 'M5 12h14M12 6h.01M12 18h.01'], run: ['运行期', 'M20 12a8 8 0 1 1-2.3-5.6M20 4v5h-5'] };
  function limits(list) {
    if (!list.length) return '';
    return '<ul class="oc-lim">' + list.map(function (l) {
      const k = LIM[l.kind];
      return '<li class="is-' + l.kind + '"><span class="oc-lim-i" title="' + k[0] + '"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="' + k[1] + '"/></svg></span>'
        + '<span class="oc-lim-t">' + esc(l.text) + (l.check ? '<span class="oc-lim-c">' + esc(l.check) + '</span>' : '') + '</span>'
        + (l.src ? '<button class="oc-lim-src" type="button" data-lens-src="' + esc(l.src[0]) + ':' + l.src[1] + ':' + l.src[1] + '" title="' + esc(l.src[0] + ' ' + l.src[1] + ' 行') + '">' + l.src[1] + '</button>' : '') + '</li>';
    }).join('') + '</ul>';
  }
  function sec(icon, title, body) {
    return '<section class="oc-sec"><h3 class="oc-h"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="' + icon + '"/></svg>' + esc(title) + '</h3>' + body + '</section>';
  }
  const IC = {
    sum: 'M4 6h16M4 12h10M4 18h7', math: 'M5 6h8l-5 6 5 6H5M15 9l5 6M20 9l-5 6', shape: 'M4 5h7v7H4zM13 5h7v14h-7zM4 14h7v5H4z', scen: 'M4 7h16M4 12h16M4 17h10M18 15l2 2 3-4'
  };

  function html(card, opts) {
    if (!card) return '';
    opts = opts || {};
    const C = root.LensCards, T = C.T;
    let s = sec(IC.sum, '功能摘要', '<p class="oc-lead">' + esc(card.summary) + '</p>');
    if (card.math.length) {
      s += sec(IC.math, '数学表达式', '<div class="oc-math">' + card.math.map(function (m) {
        return '<div class="oc-eq"><span class="oc-eq-k">' + esc(m.k) + '</span><div class="mx">' + tex(m.f) + '</div></div>';
      }).join('') + '</div>' + (card.sym.length ? '<div class="oc-sym">' + card.sym.map(function (y) { return '<span><b class="mx">' + tex(y[0]) + '</b>' + esc(y[1]) + '</span>'; }).join('') + '</div>' : '') + '<p class="oc-src-note">依据：模块的 golden_* torch 参考实现</p>');
    }
    s += sec(IC.shape, '输入输出示例', shapeSvg(T, card.flow.in, '输入') + '<div class="oc-sh-arrow">↓ ' + esc(opts.op || '') + '</div>' + shapeSvg(T, card.flow.out, '输出')
      + '<p class="oc-ex">' + esc(card.ex ? card.ex + ' ' : '') + '<span>默认配置：TP = 2，每 Rank T = 256（32 请求 × 8 token）。块的宽高按维度的 log₂ 比例画。</span></p>');
    s += sec(IC.scen, '支持的场景及限制', scenRows(card.scen, card.scen.length ? '本模块' : '') + scenRows(C.LAYER_SCEN, '整层通用') + limits(card.limits));
    return '<div class="oc">' + s + '</div>';
  }
  root.LensCardView = { html: html, tex: tex };
})(window);
