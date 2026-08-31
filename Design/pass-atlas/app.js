/* PyPTO Pass Atlas —— 编译流水线 × 硬件 × 算子，合并版 */
(function () {
  'use strict';
  var A = window.ATLAS, HW = window.HW, OP = window.OP, PK = window.PACK;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return [].slice.call((r || document).querySelectorAll(s)); };
  var esc = function (s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); };
  var rich = function (s) { return esc(s).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/`([^`]+)`/g, '<code>$1</code>'); };
  var kb = function (b) { return b >= 1024 ? (b / 1024).toFixed(b % 1024 ? 1 : 0) + ' KB' : b + ' B'; };
  var TONE = { tiling: 'var(--l-tiling)', memory: 'var(--l-memory)', deps: 'var(--l-deps)', sync: 'var(--l-sync)' };
  var byName = {};
  A.passes.forEach(function (p) { if (!byName[p.name]) byName[p.name] = p; });

  var state = { view: 'overview', sel: 'MemoryReuse', soc: 'a5', railAll: true, q: '',
                pkCase: 'acc', pkFrame: 0, pkSel: null, hwMode: 'topo' };

  var VIEWS = [
    { id: 'overview', label: '总览',        rail: false, soc: false },
    { id: 'pipe',     label: '流水线全景',  rail: false, soc: false },
    { id: 'pass',     label: 'Pass 详解',   rail: true,  soc: false },
    { id: 'props',    label: '属性依赖图',  rail: false, soc: false },
    { id: 'hw',       label: '硬件映射',    rail: false, soc: true  },
    { id: 'op',       label: '算子解剖',    rail: false, soc: true  },
    { id: 'pack',     label: '内存复用逐帧', rail: false, soc: true }
  ];

  /* ---------- tooltip ---------- */
  var tip = null, tipOwner = null;
  function hideTip() { if (tip) { tip.hidden = true; } tipOwner = null; }
  function bindTip(root) {
    tip = tip || $('#tip');
    $$('[data-tip]', root).forEach(function (el) {
      el.addEventListener('mousemove', function (e) {
        tipOwner = el;
        tip.innerHTML = el.dataset.tip; tip.hidden = false;
        var w = tip.offsetWidth, h = tip.offsetHeight;
        tip.style.left = Math.min(e.clientX + 14, innerWidth - w - 10) + 'px';
        tip.style.top = Math.max(8, Math.min(e.clientY + 14, innerHeight - h - 10)) + 'px';
      });
      el.addEventListener('mouseleave', hideTip);
      el.addEventListener('click', hideTip);
    });
  }
  // 兜底：宿主元素随 innerHTML 一起消失时 mouseleave 不会触发，
  // 所以再挂几个全局收口，并周期性确认 owner 还在文档里。
  function installTipGuards() {
    var v = $('#view');
    v.addEventListener('scroll', hideTip, { passive: true });
    v.addEventListener('mouseleave', hideTip);
    document.addEventListener('mouseleave', hideTip);
    window.addEventListener('blur', hideTip);
    window.addEventListener('resize', hideTip);
    setInterval(function () {
      if (tipOwner && !document.contains(tipOwner)) hideTip();
    }, 400);
  }

  function railList() {
    var q = state.q.toLowerCase();
    return A.passes.filter(function (p) {
      if (!state.railAll && p.role !== 'decide') return false;
      if (!q) return true;
      return (p.name + ' ' + p.brief + ' ' + p.required.join(' ') + ' ' + p.produced.join(' ')).toLowerCase().indexOf(q) >= 0;
    });
  }
  function renderRail() {
    $('#rail-modes').innerHTML =
      '<button type="button" class="rmode' + (state.railAll ? ' is-on' : '') + '" data-a="1">全部 ' + A.passes.length + '</button>' +
      '<button type="button" class="rmode' + (state.railAll ? '' : ' is-on') + '" data-a="0">仅决策点 ' + A.meta.decide + '</button>';
    $$('#rail-modes .rmode').forEach(function (b) {
      b.addEventListener('click', function () { state.railAll = b.dataset.a === '1'; renderRail(); });
    });

    var list = railList(), html = '', shown = 0;
    A.phases.forEach(function (ph) {
      var ps = list.filter(function (p) { return p.phase === ph.id; });
      if (!ps.length) return;
      shown += ps.length;
      html += '<div class="pgroup"><div class="pgroup__head"><span class="pgroup__name">' + esc(ph.name) +
              '</span><span class="pgroup__n">' + ps.length + '</span></div>';
      ps.forEach(function (p) {
        var tone = p.layer ? TONE[p.layer] : 'var(--border-strong)';
        html += '<div class="prow' + (state.sel === p.name ? ' is-on' : '') + (p.role === 'mech' ? ' is-mech' : '') +
                '" data-n="' + p.name + '" style="--tone:' + tone + '" tabindex="0" role="button">' +
                '<span class="prow__i">' + p.order + '</span>' +
                '<span class="prow__n">' + esc(p.name) + '</span>' +
                (p.role === 'decide'
                  ? '<span class="prow__b dec">' + esc(A.layers[p.layer].slice(2)) + '</span>'
                  : '<span class="prow__b mech">机械</span>') +
                '</div>';
      });
      html += '</div>';
    });
    $('#rail').innerHTML = html || '<div style="padding:20px;color:var(--foreground-muted);font:var(--type-body-sm)">没有匹配的 Pass</div>';
    $$('#rail .prow').forEach(function (r) {
      var go = function () { state.sel = r.dataset.n; state.view = 'pass'; render(); };
      r.addEventListener('click', go);
      r.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
    });
    $('#rail-foot').innerHTML = '流水线共 <b style="color:var(--foreground)">' + A.passes.length +
      ' 步</b>（Simplify 跑两次），其中 <b style="color:var(--warning)">' + A.meta.decide +
      ' 个在做决定</b>。顺序取自 <code style="font-size:10px">python/pypto/ir/pass_manager.py</code>。';
  }


  function viewOverview() {
    var h = '<div class="wrap"><h1 class="h1">一次编译，49 步</h1>' +
      '<p class="lead">这份图鉴的每一条都来自源码快照 <code>' + esc(A.meta.src) + '</code>：' +
      '流水线顺序取自 <code>pass_manager.py</code>，每个 Pass 的作用取自 <code>passes.h</code> 的 doxygen 与各自的实现文件，' +
      '<strong>上下游关系不是我画的，是从 <code>pass_properties.h</code> 里每个 Pass 声明的 required / produced / invalidated 算出来的</strong>。</p></div>';

    var mech = A.passes.length - A.meta.decide;
    h += '<div class="stats">' +
      '<div class="stat"><div class="stat__v">' + A.passes.length + '</div><div class="stat__k">流水线步数<br>Simplify 出现两次</div></div>' +
      '<div class="stat"><div class="stat__v" style="color:var(--warning)">' + A.meta.decide + '</div><div class="stat__k">决策点<br>在多个合法方案中选一个</div></div>' +
      '<div class="stat"><div class="stat__v" style="color:var(--foreground-muted)">' + mech + '</div><div class="stat__k">机械改写<br>输入定则输出唯一</div></div>' +
      '<div class="stat"><div class="stat__v">' + Object.keys(A.props).length + '</div><div class="stat__k">IRProperty<br>构成真实的依赖图</div></div>' +
      '</div>';

    h += '<div class="legend">' + Object.keys(A.layers).map(function (k) {
      return '<span class="legend__i" style="--tone:' + TONE[k] + '"><i class="legend__d"></i>' + esc(A.layers[k]) + '</span>';
    }).join('') + '<span class="legend__i" style="--tone:var(--foreground-disabled)"><i class="legend__d"></i>机械改写</span></div>';

    A.phases.forEach(function (ph) {
      var ps = A.passes.filter(function (p) { return p.phase === ph.id; });
      if (!ps.length) return;
      var nd = ps.filter(function (p) { return p.role === 'decide'; }).length;
      h += '<div class="phase"><div class="phase__h"><span class="phase__n">' + esc(ph.name) + '</span>' +
           '<span class="phase__r">' + Math.min.apply(null, ps.map(function (x) { return x.order; })) + '–' +
             Math.max.apply(null, ps.map(function (x) { return x.order; })) + '</span>' +
           '<span class="phase__c">' + ps.length + ' 步 · ' + nd + ' 个决策点</span></div>' +
           '<p class="phase__d">' + esc(ph.desc) + '</p><div class="phase__list">' +
           ps.map(function (p) {
             var tone = p.layer ? TONE[p.layer] : 'var(--border-default)';
             return '<button type="button" class="plink" data-go="' + p.name + '" style="' +
                    (p.role === 'decide' ? 'border-color:' + tone + ';color:' + tone : '') + '">' +
                    p.order + ' ' + esc(p.name) + '</button>';
           }).join('') + '</div></div>';
    });
    return h + '</div>';
  }


  function viewPipe() {
    var P = A.passes, n = P.length;
    var W = Math.max(1240, n * 26 + 200), padL = 176, padR = 30;
    var x = function (i) { return padL + (i + 0.5) / n * (W - padL - padR); };

    /* ---------- Pass 河 ---------- */
    var irY = 22, bandY = 58, bandH = 74, nodeY = bandY + 40, H1 = bandY + bandH + 30;
    var s = '<svg viewBox="0 0 ' + W + ' ' + H1 + '" style="min-width:' + W + 'px" role="img" aria-label="Pass 流水线">';
    s += '<defs>' +
      '<linearGradient id="bandg" x1="0" y1="0" x2="0" y2="1">' +
        '<stop offset="0%" stop-color="var(--foreground)" stop-opacity=".055"/>' +
        '<stop offset="100%" stop-color="var(--foreground)" stop-opacity=".015"/></linearGradient>' +
      '<linearGradient id="spineg" gradientUnits="userSpaceOnUse" x1="' + padL + '" y1="0" x2="' + (W - padR) + '" y2="0">' +
        '<stop offset="0%" stop-color="var(--foreground)" stop-opacity=".05"/>' +
        '<stop offset="12%" stop-color="var(--foreground)" stop-opacity=".2"/>' +
        '<stop offset="88%" stop-color="var(--foreground)" stop-opacity=".2"/>' +
        '<stop offset="100%" stop-color="var(--foreground)" stop-opacity=".05"/></linearGradient>' +
      '</defs>';

    var IR = [
      { at: 0,  t: 'Tensor IR' },
      { at: 7,  t: '编排 / InCore 分家' },
      { at: 10, t: 'Tile 算子' },
      { at: 21, t: 'AIC / AIV kernel' },
      { at: 31, t: 'MemRef' },
      { at: 34, t: '真实地址' },
      { at: 37, t: '任务依赖图' }
    ];
    IR.forEach(function (m, i) {
      var x0 = x(m.at) - 9, x1 = (i + 1 < IR.length ? x(IR[i + 1].at) - 9 : W - padR);
      s += '<rect class="riv__irbox" x="' + x0 + '" y="' + (irY - 11) + '" width="' + Math.max(46, x1 - x0 - 5) + '" height="19" rx="9"/>';
      s += '<text class="riv__ir" x="' + (x0 + 10) + '" y="' + (irY + 3) + '">' + esc(m.t) + '</text>';
    });
    s += '<text class="t-sm" x="' + (padL - 14) + '" y="' + (irY + 3) + '" text-anchor="end">IR 形态</text>';

    A.phases.forEach(function (ph) {
      var idx = P.map(function (p, i) { return p.phase === ph.id ? i : -1; }).filter(function (i) { return i >= 0; });
      if (!idx.length) return;
      var a2 = x(Math.min.apply(null, idx)) - 11, b2 = x(Math.max.apply(null, idx)) + 11;
      s += '<rect class="riv__band" x="' + a2 + '" y="' + bandY + '" width="' + (b2 - a2) + '" height="' + bandH + '" rx="10" fill="url(#bandg)"/>';
      s += '<text class="riv__bandlab" x="' + ((a2 + b2) / 2) + '" y="' + (bandY + 16) + '" text-anchor="middle">' + esc(ph.name) + '</text>';
    });

    s += '<line class="riv__spine" x1="' + padL + '" y1="' + nodeY + '" x2="' + (W - padR) + '" y2="' + nodeY + '" stroke="url(#spineg)"/>';
    s += '<text class="t-sm" x="' + (padL - 14) + '" y="' + (nodeY + 4) + '" text-anchor="end">执行序 →</text>';

    P.forEach(function (p, i) {
      var tone = p.layer ? TONE[p.layer] : 'var(--border-strong)';
      var t = '<b>' + esc(p.order + ' · ' + p.name) + '</b>' + esc(p.brief) +
              '<br><em>' + (p.role === 'decide' ? '★ 决策点 · ' + esc(A.layers[p.layer]) : '机械改写') + '</em>';
      s += '<g class="riv__node" data-tip="' + esc(t) + '" data-go="' + p.name + '">';
      if (p.role === 'decide') {
        s += '<circle class="rglow" cx="' + x(i) + '" cy="' + nodeY + '" r="13" fill="' + tone + '"/>';
        s += '<circle cx="' + x(i) + '" cy="' + nodeY + '" r="9.5" fill="none" stroke="' + tone + '" stroke-width="1" opacity=".38"/>';
        s += '<circle class="rn" cx="' + x(i) + '" cy="' + nodeY + '" r="6.5" fill="' + tone + '" stroke="var(--background)" stroke-width="1.8"/>';
        s += '<text class="t-xs" x="' + x(i) + '" y="' + (nodeY + 25) + '" text-anchor="middle" style="fill:' + tone + '">' + p.order + '</text>';
      } else {
        s += '<circle class="rglow" cx="' + x(i) + '" cy="' + nodeY + '" r="9" fill="var(--foreground)"/>';
        s += '<line class="rn" x1="' + x(i) + '" y1="' + (nodeY - 4.5) + '" x2="' + x(i) + '" y2="' + (nodeY + 4.5) +
             '" stroke="color-mix(in srgb,var(--foreground) 46%,transparent)" stroke-width="2" stroke-linecap="round"/>';
      }
      s += '<title>' + esc(p.order + ' ' + p.name) + '</title></g>';
    });
    s += '</svg>';

    /* ---------- 属性生命线 ---------- */
    var keys = Object.keys(A.props).sort(function (a2, b2) { return A.props[b2].required_by.length - A.props[a2].required_by.length; }).slice(0, 12);
    var rowH = 30, H2 = keys.length * rowH + 34;
    var s2 = '<svg viewBox="0 0 ' + W + ' ' + H2 + '" style="min-width:' + W + 'px" role="img" aria-label="IRProperty 生命线">';
    s2 += '<defs><linearGradient id="lifeg" gradientUnits="userSpaceOnUse" x1="' + padL + '" y1="0" x2="' + (W - padR) + '" y2="0">' +
          '<stop offset="0%" stop-color="var(--success)" stop-opacity=".95"/>' +
          '<stop offset="100%" stop-color="var(--success)" stop-opacity=".45"/></linearGradient></defs>';
    keys.forEach(function (k, r) {
      var y = 22 + r * rowH, v = A.props[k];
      s2 += '<g class="lfrow">';
      s2 += '<rect class="lfbg" x="' + (padL - 170) + '" y="' + (y - 13) + '" width="' + (W - padL + 170 - padR) + '" height="26" rx="7"/>';
      s2 += '<text class="life-lab" x="' + (padL - 14) + '" y="' + (y + 3) + '" text-anchor="end">' + esc(k) + '</text>';
      s2 += '<line class="ln-d" x1="' + padL + '" y1="' + y + '" x2="' + (W - padR) + '" y2="' + y + '"/>';

      var segs = [], open = -1, marks = [];
      P.forEach(function (p, i) {
        var inv = p.invalidated.indexOf(k) >= 0, prod = p.produced.indexOf(k) >= 0;
        if (inv && open >= 0) { segs.push([open, i]); open = -1; marks.push(i); }
        else if (inv && open < 0) { marks.push(i); }
        if (prod && open < 0) open = i;
      });
      if (open >= 0) segs.push([open, P.length - 1]);

      segs.forEach(function (sg, si) {
        var a2 = x(sg[0]), b2 = Math.max(a2 + 3, x(sg[1]));
        var tipTxt = '<b>' + k + '</b>第 ' + P[sg[0]].order + ' 步 ' + P[sg[0]].name + ' 建立' +
          (sg[1] < P.length - 1 ? '，第 ' + P[sg[1]].order + ' 步 ' + P[sg[1]].name + ' 失效' : '，此后一直有效') +
          '<br><em>共 ' + v.required_by.length + ' 个 Pass 要求它' + (segs.length > 1 ? ' · 第 ' + (si + 1) + '/' + segs.length + ' 段' : '') + '</em>';
        s2 += '<line class="lifeline" x1="' + a2 + '" y1="' + y + '" x2="' + b2 + '" y2="' + y +
              '" stroke="url(#lifeg)" data-tip="' + esc(tipTxt) + '"/>';
        s2 += '<circle class="life-dot" cx="' + a2 + '" cy="' + y + '" r="4.5"/>';
      });
      v.required_by.forEach(function (nm) {
        var i = P.findIndex(function (p) { return p.name === nm; });
        if (i < 0) return;
        var inside = segs.some(function (sg) { return i >= sg[0] && i <= sg[1]; });
        s2 += '<circle cx="' + x(i) + '" cy="' + y + '" r="2.4" fill="' + (inside ? 'var(--background)' : 'none') +
              '" stroke="' + (inside ? 'color-mix(in srgb,var(--foreground) 46%,transparent)' : 'var(--foreground-disabled)') + '" stroke-width="1.1"/>';
      });
      marks.forEach(function (i) {
        s2 += '<g data-tip="' + esc('<b>第 ' + P[i].order + ' 步 ' + P[i].name + ' 破坏了 ' + k + '</b>' +
              (k === 'SSAForm' ? '变量绑上物理 MemRef 之后，「一个名字只赋值一次」不再成立。' :
               '声明里同时 invalidate 与 produce，意思是强制在这一步重新验证一次。')) + '">' +
              '<circle cx="' + x(i) + '" cy="' + y + '" r="7.5" fill="var(--background)" stroke="color-mix(in srgb,var(--danger) 48%,transparent)"/>' +
              '<line class="life-x" x1="' + (x(i) - 3.4) + '" y1="' + (y - 3.4) + '" x2="' + (x(i) + 3.4) + '" y2="' + (y + 3.4) + '"/>' +
              '<line class="life-x" x1="' + (x(i) - 3.4) + '" y1="' + (y + 3.4) + '" x2="' + (x(i) + 3.4) + '" y2="' + (y - 3.4) + '"/></g>';
      });
      s2 += '</g>';
    });
    s2 += '</svg>';

    return '<div class="wrap">' +
      '<h1 class="h1">一次编译，49 步</h1>' +
      '<p class="lead">上图是流水线本身：<strong>实心圆点是决策点</strong>（在多个合法方案里选一个），竖线是机械改写（输入定则输出唯一）；' +
      '顶部那条带子标出 IR 在每个阶段的形态。下图是同一条时间轴上的 <strong>IRProperty 生命线</strong>——' +
      '绿点是属性被建立的位置，渐隐的绿线是它有效的区间，线上的小圈是每一个要求它的 Pass，' +
      '<strong>红圈叉是它被破坏的位置</strong>。整张图由 <code>pass_properties.h</code> 的声明算出，没有人工排布。</p>' +
      '<div class="legend">' +
      Object.keys(A.layers).map(function (k2) {
        return '<span class="legend__i" style="--tone:' + TONE[k2] + '"><i class="legend__d"></i>' + esc(A.layers[k2]) + '</span>';
      }).join('') +
      '<span class="legend__i" style="--tone:var(--foreground-disabled)"><i class="legend__d"></i>机械改写</span></div>' +
      '<div class="card"><div class="card__h"><span class="card__t">Pass 河</span>' +
      '<span class="card__m">python/pypto/ir/pass_manager.py</span></div><div class="scroll">' + s + '</div></div>' +
      '<div class="card"><div class="card__h"><span class="card__t">IRProperty 生命线 · 被消费最多的 12 个</span>' +
      '<span class="card__m">include/pypto/ir/transforms/pass_properties.h</span></div><div class="scroll">' + s2 + '</div>' +
      '<p class="n">看第二行 <code>SSAForm</code>：第 4 步 ConvertToSSA 建立，被 21 个 Pass 消费，' +
      '<strong>在第 32 步 InitMemRef 被一个红叉终结</strong>——变量一旦绑上物理 MemRef，SSA 就不成立了。' +
      '所有内存相关的 Pass 都排在这条线断掉之后，这不是巧合，是排序的结果。' +
      '再看最后一行 <code>AivSplitValid</code>：它是<strong>三段</strong>——两次「先失效再重建」，' +
      '为的是等内存侧变得可观测之后强制重验一次。</p></div></div>';
  }

  function viewAtlas() {
    var p = byName[state.sel] || A.passes[0];
    var tone = p.layer ? TONE[p.layer] : 'var(--foreground-muted)';
    var h = '<div class="wrap" style="--tone:' + tone + '">';

    h += '<div class="d__top"><span class="d__ord">第 ' + p.order + ' 步</span>' +
         (p.role === 'decide'
           ? '<span class="d__badge dec">★ 决策点 · ' + esc(A.layers[p.layer]) + '</span>'
           : '<span class="d__badge mech">机械改写</span>') + '</div>';
    h += '<h1 class="d__h">' + esc(p.name) + '</h1>';
    h += '<p class="d__brief">' + rich(p.brief) + '</p>';
    h += '<div class="d__file"><b>源码</b>' + esc(p.file) + (p.lines ? '　<b>' + p.lines + ' 行</b>' : '') + '</div>';

    h += '<div class="sec"><h2 class="sec__t">它做什么</h2><p class="sec__p">' + rich(p.detail) + '</p></div>';

    if (p.watch) {
      h += '<div class="sec"><h2 class="sec__t">值得注意</h2><div class="watch"><p class="sec__p">' + rich(p.watch) + '</p></div></div>';
    }

    if (p.factory) {
      h += '<div class="sec"><h2 class="sec__t">Pass 工厂函数</h2><div class="snip">' +
           '<div class="code__cap"><b>Pass ' + esc(p.name) + '()</b><span>' + esc(p.factoryRef || p.file) + '</span></div>' +
           '<pre class="code">' + esc(p.factory) + '</pre></div></div>';
    }

    if (p.snippets && p.snippets.length) {
      h += '<div class="sec"><h2 class="sec__t">核心代码</h2>' + p.snippets.map(function (s) {
        return '<div class="snip"><div class="code__cap"><b>' + esc(s.label) + '</b>' +
               '<span>' + esc(s.file) + ':' + s.from + '–' + s.to + '</span></div>' +
               '<pre class="code">' + esc(s.code) + '</pre></div>';
      }).join('') + '</div>';
    }

    // 属性
    var box = function (cls, title, arr, empty) {
      return '<div class="pbox pbox--' + cls + '"><div class="pbox__t">' + title + '</div>' +
        (arr.length ? arr.map(function (x) { return '<span class="chip" data-prop="' + x + '">' + esc(x) + '</span>'; }).join('')
                    : '<span class="chip none">' + empty + '</span>') + '</div>';
    };
    h += '<div class="sec"><h2 class="sec__t">IR 属性契约</h2><div class="props">' +
      box('req', 'REQUIRED 入口要求', p.required, '无要求') +
      box('prod', 'PRODUCED 出口保证', p.produced, '不产生新属性') +
      box('inv', 'INVALIDATED 出口失效', p.invalidated, '不破坏任何属性') +
      '</div></div>';

    // 上游
    var ups = Object.keys(p.origin || {});
    h += '<div class="sec"><h2 class="sec__t">上游 · 我要求的属性由谁最先建立</h2><div class="flow">';
    h += ups.length ? ups.map(function (k) {
      var src = p.origin[k];
      return '<div class="frow"><span class="frow__k">' + esc(k) + '</span><span class="frow__a">←</span>' +
        '<span class="frow__v">' + (src ? '<button type="button" class="plink" data-go="' + src + '">' +
        (byName[src] ? byName[src].order + ' ' : '') + esc(src) + '</button>' : '<span class="plink dim">流水线入口即成立</span>') + '</span></div>';
    }).join('') : '<div class="frow"><span class="frow__k">—</span><span class="frow__a"></span><span class="frow__v"><span class="plink dim">无入口要求，可在任意位置运行</span></span></div>';
    h += '</div></div>';

    // 下游
    var downs = Object.keys(p.downstream || {});
    h += '<div class="sec"><h2 class="sec__t">下游 · 我产生的属性被谁消费</h2><div class="flow">';
    h += downs.length ? downs.map(function (k) {
      var cs = p.downstream[k] || [];
      return '<div class="frow"><span class="frow__k">' + esc(k) + '</span><span class="frow__a">→</span>' +
        '<span class="frow__v">' + (cs.length ? cs.map(function (c) {
          return '<button type="button" class="plink" data-go="' + c + '">' +
                 (byName[c] ? byName[c].order + ' ' : '') + esc(c) + '</button>';
        }).join('') : '<span class="plink dim">此后无人再要求（属性到此为止或被后续 Pass 重建）</span>') + '</span></div>';
    }).join('') : '<div class="frow"><span class="frow__k">—</span><span class="frow__a"></span><span class="frow__v"><span class="plink dim">不产生属性，下游无结构性依赖</span></span></div>';
    h += '</div></div>';

    // 相邻步
    var i = A.passes.indexOf(p);
    var prev = A.passes[i - 1], next = A.passes[i + 1];
    h += '<div class="sec"><h2 class="sec__t">流水线相邻步</h2><div class="frow">' +
      '<span class="frow__k">上一步</span><span class="frow__a">·</span><span class="frow__v">' +
      (prev ? '<button type="button" class="plink" data-go="' + prev.name + '">' + prev.order + ' ' + esc(prev.name) + '</button>' : '<span class="plink dim">流水线起点</span>') +
      '</span></div><div class="frow" style="margin-top:6px">' +
      '<span class="frow__k">下一步</span><span class="frow__a">·</span><span class="frow__v">' +
      (next ? '<button type="button" class="plink" data-go="' + next.name + '">' + next.order + ' ' + esc(next.name) + '</button>' : '<span class="plink dim">流水线终点，进入 codegen</span>') +
      '</span></div></div>';

    return h + '</div>';
  }


  function viewProps() {
    var keys = Object.keys(A.props).sort(function (a, b) {
      return A.props[b].required_by.length - A.props[a].required_by.length;
    });
    var h = '<div class="wrap"><h1 class="h1">IRProperty 依赖图</h1>' +
      '<p class="lead">Pass 之间真正的上下游关系不写在文档里，写在 <code>include/pypto/ir/transforms/pass_properties.h</code>：' +
      '每个 Pass 声明它 <strong>要求</strong>、<strong>产生</strong>、<strong>破坏</strong> 哪些 IRProperty，' +
      'PassPipeline 在每个 Pass 边界据此自动验证。下表按被消费次数排序——<strong>越靠上的属性，越是整条流水线的地基</strong>。</p></div>';

    h += '<table class="ptable"><thead><tr>' +
      '<th>属性</th><th style="width:70px">被要求</th><th>由谁建立 / 保持</th><th>被谁破坏</th></tr></thead><tbody>';
    keys.forEach(function (k) {
      var v = A.props[k];
      var link = function (n) {
        return '<button type="button" class="plink" data-go="' + n + '">' +
               (byName[n] ? byName[n].order + ' ' : '') + esc(n) + '</button>';
      };
      var producers = v.produced_by.slice(0, 3).map(link).join('') +
        (v.produced_by.length > 3 ? '<span class="plink dim">+' + (v.produced_by.length - 3) + ' 个 Pass 保持</span>' : '');
      h += '<tr><td><span class="pname">' + esc(k) + '</span></td>' +
        '<td><span class="pcount">' + v.required_by.length + '</span></td>' +
        '<td><div class="frow__v">' + (producers || '<span class="plink dim">—</span>') + '</div></td>' +
        '<td><div class="frow__v">' + (v.invalidated_by.length ? v.invalidated_by.map(link).join('') : '<span class="plink dim">从不失效</span>') + '</div></td></tr>';
    });
    h += '</tbody></table>';
    return h + '</div>';
  }



  /* 数据通路拓扑图 */
  function hwTopo(soc) {
    var T = HW.topo, cap = {};
    soc.cores.forEach(function (c) { c.mems.forEach(function (m) { cap[m.s] = m; }); });
    var has = function (id) { return id === 'DDR' || !!cap[id] || T.units.filter(function (u) { return u.id === id && u.kind === 'unit'; }).length; };
    var live = T.units.filter(function (u) { return !(u.a5only && soc.id !== 'a5'); });

    var W = 1080, H = 420, cw = 168, ch = 54;
    var px = function (u) { return 42 + u.col * cw; };
    var py = function (u) { return H / 2 - 40 + u.row * 74; };

    var s2 = '<svg viewBox="0 0 ' + W + ' ' + H + '" style="min-width:' + W + 'px" role="img" aria-label="数据通路拓扑">';
    s2 += '<defs><marker id="tarw" viewBox="0 0 9 9" refX="8" refY="4.5" markerWidth="6" markerHeight="6" orient="auto">' +
          '<path d="M0 0 L9 4.5 L0 9 z" fill="color-mix(in srgb,var(--foreground) 42%,transparent)"/></marker>' +
          '<marker id="tarw-hot" viewBox="0 0 9 9" refX="8" refY="4.5" markerWidth="6" markerHeight="6" orient="auto">' +
          '<path d="M0 0 L9 4.5 L0 9 z" fill="var(--warning)"/></marker></defs>';

    // 核心分区底框
    [['AIC', 'cube', 1, 4.55], ['AIV', 'vec', 0.62, 2.5]].forEach(function (g) {
      var us = live.filter(function (u) { return u.core === g[0]; });
      if (!us.length) return;
      var x0 = Math.min.apply(null, us.map(px)) - 26, x1 = Math.max.apply(null, us.map(px)) + cw - 44;
      var y0 = Math.min.apply(null, us.map(py)) - 20, y1 = Math.max.apply(null, us.map(py)) + ch + 16;
      s2 += '<rect class="corecard corecard--' + g[1] + '" x="' + x0 + '" y="' + y0 + '" width="' + (x1 - x0) + '" height="' + (y1 - y0) + '" rx="12" opacity=".5"/>';
      s2 += '<text class="t-xs" x="' + (x0 + 10) + '" y="' + (y0 + 15) + '" style="fill:' + (g[0] === 'AIC' ? 'var(--l-tiling)' : 'var(--success)') + '">' +
            g[0] + (g[0] === 'AIC' ? ' · Cube ×1' : ' · Vector ×' + soc.cores.filter(function(c){return c.type==='VECTOR';})[0].per) + '</text>';
    });

    // 边：内存图 + 计算单元
    var byId = {}; live.forEach(function (u) { byId[u.id] = u; });
    var edge = function (f, t, hot) {
      var A2 = byId[f], B2 = byId[t];
      if (!A2 || !B2) return '';
      var x1 = px(A2) + cw - 46, y1 = py(A2) + ch / 2, x2 = px(B2), y2 = py(B2) + ch / 2;
      if (px(B2) < px(A2)) { x1 = px(A2); x2 = px(B2) + cw - 46; }
      var mx = (x1 + x2) / 2;
      return '<path d="M' + x1 + ' ' + y1 + ' C ' + mx + ' ' + y1 + ', ' + mx + ' ' + y2 + ', ' + x2 + ' ' + y2 +
             '" fill="none" stroke="' + (hot ? 'var(--warning)' : 'color-mix(in srgb,var(--foreground) 26%,transparent)') +
             '" stroke-width="' + (hot ? 2 : 1.3) + '" marker-end="url(#' + (hot ? 'tarw-hot' : 'tarw') + ')"/>';
    };
    Object.keys(soc.memGraph).forEach(function (f) {
      soc.memGraph[f].forEach(function (t) {
        if (f === 'Acc') return;                       // Acc 的出边走 FixPipe
        var hot = (f === 'Vec' && t === 'Mat');
        s2 += edge(f, t, hot);
      });
    });
    (soc.memGraph.Acc || []).forEach(function (t) { s2 += edge('FIX', t, t === 'Vec'); });
    T.unitEdges.forEach(function (e) {
      if (e.a5only && soc.id !== 'a5') return;
      s2 += edge(e.f, e.t, false);
    });

    // 节点
    live.forEach(function (u) {
      var m = cap[u.id], x0 = px(u), y0 = py(u);
      var isUnit = u.kind === 'unit';
      var tip = isUnit ? '<b>' + u.label + '</b>' + u.sub + '<br><em>计算单元</em>'
                       : '<b>' + u.label + ' · ' + u.sub + '</b>' + (m ? esc(m.note) : '片外全局内存') +
                         (m ? '<br><em>' + kb(m.bytes) + (m.phys ? ' 安全 / ' + kb(m.phys) + ' 物理' : '') + ' · align ' + m.align + '</em>' : '');
      s2 += '<g class="tn ' + (isUnit ? 'tn--unit' : 'tn--mem') + (m && m.danger ? ' tn--danger' : '') + '" data-tip="' + esc(tip) + '">' +
        '<rect x="' + x0 + '" y="' + y0 + '" width="' + (cw - 46) + '" height="' + ch + '" rx="9"/>' +
        '<text class="t-lg" x="' + (x0 + 11) + '" y="' + (y0 + 21) + '">' + esc(u.label) + '</text>' +
        '<text class="t-xs" x="' + (x0 + 11) + '" y="' + (y0 + 35) + '">' + esc(u.sub) + '</text>' +
        (m ? '<text class="t-xs" x="' + (x0 + cw - 57) + '" y="' + (y0 + 47) + '" text-anchor="end" style="fill:var(--foreground-secondary)">' + kb(m.bytes) + '</text>' : '') +
        '</g>';
    });
    s2 += '</svg>';
    return s2;
  }

  function viewHW() {
    var soc = HW.socs[state.soc], other = HW.socs[state.soc === 'a5' ? 'a2a3' : 'a5'];
    var cube = soc.cores.filter(function (c) { return c.type === 'CUBE'; })[0];
    var vec = soc.cores.filter(function (c) { return c.type === 'VECTOR'; })[0];
    var maxB = Math.max.apply(null, cube.mems.concat(vec.mems).map(function (m) { return m.phys || m.bytes; }));

    var W = 1120, padL = 16, colW = 500, rowH = 42, hdrH = 34;
    var coreTop = 12;
    var coreH = hdrH + Math.max(cube.mems.length, vec.mems.length) * rowH + 20;
    var gy = coreTop + coreH + 108;
    var H = gy + 128;

    var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" style="min-width:' + W + 'px" role="img" aria-label="' + esc(soc.name) + ' 内存">';
    s += '<defs>' +
      '<marker id="arw" viewBox="0 0 9 9" refX="8" refY="4.5" markerWidth="6.5" markerHeight="6.5" orient="auto">' +
        '<path d="M0 0 L9 4.5 L0 9 z" fill="color-mix(in srgb,var(--foreground) 40%,transparent)"/></marker>' +
      '<marker id="arw-hot" viewBox="0 0 9 9" refX="8" refY="4.5" markerWidth="6.5" markerHeight="6.5" orient="auto">' +
        '<path d="M0 0 L9 4.5 L0 9 z" fill="var(--warning)"/></marker>' +
      '<pattern id="hatch" width="7" height="7" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">' +
        '<rect width="7" height="7" fill="color-mix(in srgb,var(--danger) 12%,transparent)"/>' +
        '<line x1="0" y1="0" x2="0" y2="7" stroke="var(--danger)" stroke-width="2" opacity=".45"/></pattern>' +
      '</defs>';

    function drawCore(core, x0, w) {
      var isCube = core.type === 'CUBE';
      var g = '<rect class="corecard corecard--' + (isCube ? 'cube' : 'vec') + '" x="' + x0 + '" y="' + coreTop +
              '" width="' + w + '" height="' + coreH + '" rx="12"/>';
      g += '<path class="corehdr--' + (isCube ? 'cube' : 'vec') + '" d="M' + (x0 + 12) + ' ' + coreTop +
           ' h' + (w - 24) + ' a12 12 0 0 1 12 12 v' + (hdrH - 12) + ' h-' + w + ' v-' + (hdrH - 12) +
           ' a12 12 0 0 1 12 -12 z" opacity=".55"/>';
      g += '<circle cx="' + (x0 + 18) + '" cy="' + (coreTop + hdrH / 2) + '" r="4" fill="' +
           (isCube ? 'var(--l-tiling)' : 'var(--success)') + '"/>';
      g += '<text class="t-lg" x="' + (x0 + 30) + '" y="' + (coreTop + hdrH / 2 + 4) + '">' + esc(core.label) + '</text>';
      g += '<text class="t-xs" x="' + (x0 + w - 14) + '" y="' + (coreTop + hdrH / 2 + 4) + '" text-anchor="end">每 cluster ×' + core.per + '</text>';

      var trackX = x0 + 108, trackW = w - 108 - 108;
      core.mems.forEach(function (m, i) {
        var y = coreTop + hdrH + 10 + i * rowH, full = m.phys || m.bytes;
        var fullW = (full / maxB) * trackW;
        var safeW = (m.bytes / full) * fullW;
        var t = '<b>' + m.s + ' · ' + m.role + '</b>' + esc(m.note) + '<br><em>' + kb(m.bytes) +
                (m.phys ? ' 安全 / ' + kb(m.phys) + ' 物理' : '') + ' · align ' + m.align + '</em>';
        g += '<g class="mem" data-tip="' + esc(t) + '">';
        g += '<rect class="memtrack" x="' + trackX + '" y="' + (y + 3) + '" width="' + trackW + '" height="24" rx="6"/>';
        if (m.phys && m.phys > m.bytes) {
          g += '<rect class="hatch-danger" x="' + trackX + '" y="' + (y + 3) + '" width="' + fullW + '" height="24" rx="6"/>';
          // 危险区只占 ~3% 宽，太容易被忽略，拉一条引线点名
          var dx = trackX + safeW + (fullW - safeW) / 2;
          g += '<path d="M' + dx + ' ' + (y + 3) + ' L' + dx + ' ' + (y - 12) + ' L' + (dx + 30) + ' ' + (y - 12) +
               '" fill="none" stroke="var(--danger)" stroke-width="1" opacity=".8"/>';
          g += '<text class="t-xs" x="' + (dx + 35) + '" y="' + (y - 9) + '" style="fill:var(--danger)">' +
               kb(m.phys - m.bytes) + ' 危险区 · pto-isa#170</text>';
        }
        g += '<rect class="fillbar" x="' + trackX + '" y="' + (y + 3) + '" width="' + Math.max(10, safeW) + '" height="24" rx="6" fill="' +
             memColor(m.s) + '" stroke="' + memStroke(m.s) + '" stroke-width="1.2"/>';
        g += '<text class="t-md" x="' + (trackX - 10) + '" y="' + (y + 19) + '" text-anchor="end">' + esc(m.s) + '</text>';
        g += '<text class="t-xs" x="' + (trackX + fullW + 10) + '" y="' + (y + 19) + '">' + kb(m.bytes) +
             (m.phys ? ' / ' + kb(m.phys) : '') + '</text></g>';
      });
      return g;
    }
    function memColor(sp) {
      var c = { Mat: 'var(--primary)', Acc: 'var(--warning)', Left: 'var(--l-tiling)', Right: 'var(--l-tiling)',
                Bias: 'var(--foreground-muted)', LeftScale: 'var(--success)', RightScale: 'var(--success)',
                Vec: 'var(--success)' }[sp] || 'var(--primary)';
      return 'color-mix(in srgb,' + c + ' 42%, var(--surface-1))';
    }
    function memStroke(sp) {
      return { Mat: 'var(--primary)', Acc: 'var(--warning)', Left: 'var(--l-tiling)', Right: 'var(--l-tiling)',
               Bias: 'var(--border-strong)', LeftScale: 'var(--success)', RightScale: 'var(--success)',
               Vec: 'var(--success)' }[sp] || 'var(--primary)';
    }

    s += drawCore(cube, padL, colW);
    s += drawCore(vec, padL + colW + 32, colW - 40);

    // ---- 搬运图 ----
    var spots = { DDR: [92, gy], Vec: [268, gy], Mat: [470, gy], Left: [690, gy - 46],
                  Right: [690, gy], Bias: [690, gy + 46], LeftScale: [910, gy - 46],
                  RightScale: [910, gy], Acc: [470, gy + 92] };
    s += '<line class="ln-d" x1="' + padL + '" y1="' + (coreTop + coreH + 30) + '" x2="' + (W - 20) + '" y2="' + (coreTop + coreH + 30) + '"/>';
    s += '<text class="t-sm" x="' + padL + '" y="' + (coreTop + coreH + 58) + '">合法搬运路径　·　soc.cpp 的 mem_graph　·　<tspan style="fill:var(--warning)">橙色是 ' + esc(soc.short) + ' 独有的直连</tspan></text>';
    Object.keys(soc.memGraph).forEach(function (from) {
      soc.memGraph[from].forEach(function (to) {
        var a2 = spots[from], b2 = spots[to];
        if (!a2 || !b2) return;
        var hot = (from === 'Vec' && to === 'Mat') || (from === 'Acc' && to === 'Vec');
        var mx = (a2[0] + b2[0]) / 2, my = (a2[1] + b2[1]) / 2 - 26;
        s += '<path class="flow-arrow' + (hot ? ' hot' : '') + '" d="M' + a2[0] + ' ' + a2[1] + ' Q ' + mx + ' ' + my + ' ' + b2[0] + ' ' + b2[1] + '"/>';
      });
    });
    Object.keys(spots).forEach(function (k) {
      var has = k === 'DDR' || cube.mems.concat(vec.mems).some(function (m) { return m.s === k; });
      var p2 = spots[k];
      s += '<g class="' + (has ? '' : 'flow-node--off') + '">' +
           '<rect class="flow-node" x="' + (p2[0] - 32) + '" y="' + (p2[1] - 13) + '" width="64" height="26" rx="13"/>' +
           '<text class="t-md" x="' + p2[0] + '" y="' + (p2[1] + 4) + '" text-anchor="middle">' + esc(k) + '</text></g>';
    });
    s += '</svg>';

    var owners = HW.owners.map(function (o) {
      var p2 = A.passes.filter(function (x2) { return x2.name === o.pass; })[0];
      return '<div class="crow"><div><button type="button" class="plink" data-go="' + esc(o.pass) + '">' + esc(o.pass) + '</button>' +
        '<div class="crow__k" style="margin-top:5px">' + esc(o.what) + '</div>' +
        '<div class="n" style="margin-top:5px;font-size:12px">' + rich(o.how) + '</div></div>' +
        '<span class="crow__c">' + (o.space === 'ALL' ? '全部空间' : o.space) + '</span>' +
        '<span class="crow__c">第 ' + (p2 ? p2.order : o.order) + ' 步</span></div>';
    }).join('');
    var hl = soc.highlights.map(function (t) { return '<li style="margin-bottom:8px">' + rich(t) + '</li>'; }).join('');
    var hd = soc.handler;

    return '<div class="wrap">' +
      '<h1 class="h1">' + esc(soc.name) + ' 的内存长什么样，谁在决定用它</h1>' +
      '<p class="lead">下面每一个数字都来自 <code>' + esc(HW.src) + '</code>，没有改写。' +
      '<strong>色块宽度按真实容量成比例</strong>；<code>Vec</code> 上那片斜纹是物理容量与安全容量之间的差额。</p>' +
      '<div class="card"><div class="card__h"><span class="card__t">' + esc(soc.name) + '　' + esc(soc.factory) + '</span>' +
      '<div class="modesw">' +
        '<button type="button" class="modebtn' + (state.hwMode === 'topo' ? ' is-on' : '') + '" data-hwmode="topo">架构拓扑</button>' +
        '<button type="button" class="modebtn' + (state.hwMode === 'cap' ? ' is-on' : '') + '" data-hwmode="cap">容量比例</button>' +
      '</div>' +
      '<span class="card__m">' + soc.dies + ' die × ' + soc.clustersPerDie + ' cluster × (' + esc(soc.clusterShape) + ')</span></div>' +
      '<div class="scroll">' + (state.hwMode === 'topo' ? hwTopo(soc) : s) + '</div>' +
      (state.hwMode === 'topo'
        ? '<p class="n">节点是内存或计算单元，<strong>连线全部来自 <code>soc.cpp</code> 的 <code>mem_graph</code></strong>，没有一条是画上去的；' +
          '计算单元按 Ascend 的实际数据流插在中间：<code>L1 → L0A/L0B/BT → Cube MAC → L0C → FixPipe</code>。' +
          '橙色是 ' + esc(soc.short) + ' 独有的直连' + (soc.id === 'a5' ? '（<code>Vec→Mat</code> 与 <code>FixPipe→Vec</code>，A2/A3 都得绕 DDR）' : '') + '。' +
          (soc.id === 'a5' ? '灰底两块 <code>L0A/L0B scale</code> 是 A5 新增的 MX 量化通路，<code>InsertMxScaleAddr</code> 就是为它们服务的。' : '') + '</p>'
        : '') +
      '<p class="n" ' + (state.hwMode === 'topo' ? 'hidden' : '') + '><strong>看 Vec 上那片斜纹</strong>：物理 ' + kb(vec.mems[0].phys) + '，代码里只声明 ' + kb(vec.mems[0].bytes) +
      '。<code>soc.cpp</code> 的注释原话是——顶部约 8KB 被 PTO-ISA 占用，放 tile 会静默损坏，' +
      '所以宁可让 <code>AllocateMemoryAddr</code> 在编译期报错，也不要在设备上产出 NaN（pto-isa#170，TODO 还挂着）。' +
      '<strong>这是用容量声明兜住一个 ISA bug</strong>——一条只有 8KB 宽的产品决策。</p></div>' +

      '<div class="card"><div class="card__h"><span class="card__t">' + esc(soc.short) + ' 的特点</span>' +
      '<span class="card__m">相对 ' + esc(other.short) + '</span></div>' +
      '<ul class="n" style="margin:0 0 16px;padding-left:20px">' + hl + '</ul>' +
      '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(142px,1fr));gap:10px">' +
      [['GM 访问粒度', hd.gmGranularity + ' B'], ['L2 cache line', hd.l2Line + ' B'],
       ['推荐最内维', hd.recommendedInnermost + ' B'], ['L0A / L0B', kb(hd.l0a)],
       ['L0C', kb(hd.l0c)], ['fractal 对齐', String(hd.fractal)],
       ['Acc→GM 白名单', hd.accToGm.join(' / ')], ['BF16 atomic add', hd.bf16AtomicAdd ? '支持' : '不支持']]
       .map(function (r) { return '<div style="padding:11px 13px;border-radius:var(--radius-md);background:var(--surface-3);border:1px solid var(--border-subtle)">' +
         '<div class="crow__k">' + esc(r[0]) + '</div><div class="crow__v" style="margin-top:5px">' + esc(r[1]) + '</div></div>'; }).join('') +
      '</div></div>' +

      '<div class="card"><div class="card__h"><span class="card__t">哪个 Pass 在决定这些内存怎么用</span>' +
      '<span class="card__m">点 Pass 名跳到详解</span></div>' + owners + '</div></div>';
  }

  function viewOp() {
    var soc = HW.socs[state.soc];
    var capOf = function (sp) {
      var all = [].concat.apply([], soc.cores.map(function (c) { return c.mems; }));
      var m = all.filter(function (x) { return x.s === sp; })[0];
      return m ? m.bytes : 0;
    };
    var COLOR = { Mat: 'var(--primary)', Acc: 'var(--warning)', Left: 'var(--l-tiling)',
                  Right: 'var(--l-tiling)', Vec: 'var(--success)', Bias: 'var(--border-strong)' };

    var steps = OP.steps.map(function (st) {
      var p = A.passes.filter(function (x) { return x.name === st.pass; })[0];
      var bars = (st.mem || []).map(function (m) {
        var cap = capOf(m.s), total = m.bytes * (m.slots || 1);
        var pct = cap ? Math.min(100, total / cap * 100) : 0;
        var over = cap && total > cap;
        return '<div class="bar"><span class="bar__k">' + esc(m.s) + (m.slots > 1 ? ' ×' + m.slots : '') + '</span>' +
          '<span class="bar__t' + (over ? ' bar__over' : '') + '" data-tip="' + esc('<b>' + m.s + '</b>' + m.why + '<br><em>' + kb(total) + ' / 容量 ' + kb(cap) + '</em>') + '">' +
          '<i class="bar__f" style="width:' + pct.toFixed(1) + '%;background:' +
          (over ? 'var(--danger)' : 'color-mix(in srgb,' + (COLOR[m.s] || 'var(--primary)') + ' 62%, transparent)') + '"></i></span>' +
          '<span class="bar__v">' + kb(total) + ' / ' + kb(cap) + (over ? ' ⚠ 超' : '') + '</span></div>';
      }).join('');
      var picks = (st.picks || []).map(function (k) {
        return '<span class="pick">' + esc(k.label) + '　' + esc(k.mnk) + '</span>';
      }).join('');
      return '<div class="step"><div class="step__i"><span class="step__n">' + st.order + '</span>' +
        '<div class="step__p">' + esc(p ? A.phases.filter(function (f) { return f.id === p.phase; })[0].name : '') + '</div></div>' +
        '<div><div class="step__t">' + esc(st.title) + '</div>' +
        '<div class="step__pass">' + esc(st.pass) + (p ? '　·　' + esc(p.file) : '') + '</div>' +
        '<p class="step__d">' + rich(st.does) + '</p>' +
        (picks ? '<div class="picks">' + picks + '</div>' : '') +
        (bars ? '<div class="bars">' + bars + '</div>' : '') +
        (st.note ? '<div class="warnbox"><p class="n" style="margin:0">' + rich(st.note) + '</p></div>' : '') +
        '</div></div>';
    }).join('');

    var vd = function (id) {
      var v = OP.verdict[id], soc2 = HW.socs[id];
      return '<div class="cmp__side ' + (v.ok ? 'ok' : 'bad') + '"><div class="cmp__t">' + esc(soc2.name) + '</div>' +
        '<div class="cmp__h">' + esc(v.headline) + '</div>' +
        v.rows.map(function (r) {
          return '<div class="crow ' + (r.ok ? 'ok' : 'bad') + '"><span class="crow__k">' + esc(r.k) + '</span>' +
            '<span class="crow__v">' + esc(r.v) + '</span><span class="crow__c">' + esc(r.cap) + '</span></div>';
        }).join('') + '</div>';
    };

    return '<div class="wrap">' +
      '<h1 class="h1">' + esc(OP.title) + '</h1>' +
      '<p class="lead">同一个算子、同一份 DSL，在两代硬件上走同一条流水线，结局完全不同。' +
      '下面每一步的内存条都按 <strong>' + esc(soc.name) + '</strong> 的真实容量画（顶栏可切芯片）。' +
      '场景与数字取自 <code>' + esc(OP.src) + '</code>。</p>' +

      '<div class="card"><div class="card__h"><span class="card__t">源码</span>' +
      '<span class="card__m">' + esc(OP.shape) + '</span></div>' +
      '<pre class="code">' + esc(OP.dsl.join('\n')) + '</pre></div>' +

      '<div class="card"><div class="card__h"><span class="card__t">两代硬件上的结局</span></div>' +
      '<div class="cmp">' + vd('a5') + vd('a2a3') + '</div>' +
      '<p class="n"><strong>这两个数字可以互相验证</strong>：#1820 原文写 <code>197632 B > 188416 B</code>，' +
      '而 <code>soc.cpp</code> 里 A2/A3 的 Vec 安全上限正是 <code>184ULL * 1024 = 188416</code>——' +
      'Issue 里的报错和源码里的常量严丝合缝。</p></div>' +

      '<h2 class="h1" style="font-size:19px;margin:26px 0 12px">这个算子在流水线上经历了什么</h2>' +
      '<div class="steps">' + steps + '</div></div>';
  }


  /* =========================================================
   * ⑦ 内存复用逐帧
   * ======================================================= */
  function viewPack() {
    var C = PK.cases.filter(function (c) { return c.id === state.pkCase; })[0];
    var soc = HW.socs[state.soc];
    var cap = C.caps[state.soc];
    var fi = Math.max(0, Math.min(state.pkFrame, C.frames.length - 1));
    var F = C.frames[fi];
    var gateById = {}; PK.gates.forEach(function (g) { gateById[g.id] = g; });
    var tileOf = function (n) { return C.tiles.filter(function (t) { return t.n === n; })[0]; };

    // 选中对象：优先用户点选，否则跟随当前帧的焦点
    var sel = state.pkSel && tileOf(state.pkSel) ? state.pkSel : (F.focus || null);

    var place = F.place || [];
    var byBuf = {};
    place.forEach(function (p) { (byBuf[p.buf] = byBuf[p.buf] || []).push(p); });
    var bufIds = Object.keys(byBuf).sort(function (x, y) { return x - y; });
    var bufSize = function (id) { return Math.max.apply(null, byBuf[id].map(function (p) { return tileOf(p.n).bytes; })); };
    var total = bufIds.reduce(function (a2, id) { return a2 + bufSize(id); }, 0);
    var naive = C.tiles.reduce(function (a2, t) { return a2 + t.bytes; }, 0);
    var rejected = (F.gates || []).some(function (g) { return !g.pass; }) ? 1 : 0;
    var reuseEdges = place.filter(function (p) { return p.reused; }).length;

    /* ---------- KPI ---------- */
    var kpi = [
      { k: '当前占用', v: kb(total), tone: total > cap ? 'var(--danger)' : 'var(--foreground)' },
      { k: C.space + ' 容量（' + soc.short + '）', v: kb(cap), tone: 'var(--foreground-muted)' },
      { k: '利用率', v: (total / cap * 100).toFixed(1) + '%', tone: total > cap ? 'var(--danger)' : 'var(--success)' },
      { k: 'buffer 数', v: bufIds.length + ' / ' + C.tiles.length + ' 块 tile', tone: 'var(--foreground)' },
      { k: '相对未复用', v: total < naive ? '省 ' + kb(naive - total) : '—', tone: 'var(--success)' },
      { k: '本帧被禁令拦下', v: rejected ? '1 处' : '0', tone: rejected ? 'var(--danger)' : 'var(--foreground-muted)' }
    ].map(function (m) {
      return '<div class="kpi"><div class="kpi__k">' + esc(m.k) + '</div>' +
             '<div class="kpi__v" style="color:' + m.tone + '">' + esc(m.v) + '</div></div>';
    }).join('');

    /* ---------- 左：源码 ---------- */
    var activeLines = {};
    C.tiles.forEach(function (t) { activeLines[t.from] = 1; activeLines[t.to] = 1; });
    var code = PK.source.map(function (l) {
      var owns = (l.def && tileOf(l.def)) ? l.def : ((l.use && tileOf(l.use)) ? l.use : null);
      var isSel = owns && owns === sel;
      var isActive = owns && place.some(function (p) { return p.n === owns; });
      var cls = 'cl' + (isSel ? ' is-sel' : '') + (owns ? ' has-tile' : '') + (isActive && !isSel ? ' is-active' : '');
      return '<div class="' + cls + '"' + (owns ? ' data-tile="' + owns + '"' : '') + '>' +
        '<span class="cl__n">' + l.n + '</span>' +
        '<span class="cl__t">' + esc(l.t || '') + '</span>' +
        (l.def ? '<span class="cl__b def">def</span>' : (l.use ? '<span class="cl__b use">last use</span>' : '')) +
        '</div>';
    }).join('');

    /* ---------- 中：地址 × 生命周期 ---------- */
    var W = 560, padL = 46, padT = 34, padB = 26, Hh = 330;
    var lo = Math.min.apply(null, C.tiles.map(function (t) { return t.from; })) - 2;
    var hi = Math.max.apply(null, C.tiles.map(function (t) { return t.to; })) + 2;
    var span = Math.max(total, cap * 0.4);
    var ax = function (b2) { return padL + b2 / span * (W - padL - 16); };
    var ay = function (l) { return padT + (l - lo) / (hi - lo) * (Hh - padT - padB); };

    var s = '<svg viewBox="0 0 ' + W + ' ' + Hh + '" role="img" aria-label="地址 × 生命周期">';
    s += '<text class="t-xs" x="' + padL + '" y="12">地址 →</text>';
    s += '<text class="t-xs" x="' + (padL - 8) + '" y="' + (padT - 6) + '" text-anchor="end">行</text>';
    // 行刻度
    for (var ln = lo; ln <= hi; ln += 4) {
      s += '<line class="ln-d" x1="' + padL + '" y1="' + ay(ln) + '" x2="' + (W - 16) + '" y2="' + ay(ln) + '"/>';
      s += '<text class="t-xs" x="' + (padL - 8) + '" y="' + (ay(ln) + 3) + '" text-anchor="end">' + ln + '</text>';
    }
    var off = 0, pos = {};
    bufIds.forEach(function (id) {
      var w = bufSize(id), x0 = ax(off), x1 = ax(off + w);
      s += '<rect x="' + x0 + '" y="' + (padT - 10) + '" width="' + (x1 - x0) + '" height="' + (Hh - padT - padB + 18) +
           '" rx="7" fill="color-mix(in srgb,var(--foreground) 3.5%,transparent)" stroke="var(--border-default)" stroke-dasharray="3 3"/>';
      s += '<text class="t-xs" x="' + ((x0 + x1) / 2) + '" y="' + (padT - 16) + '" text-anchor="middle">#' + id + ' · ' + kb(w) + '</text>';
      byBuf[id].forEach(function (p) {
        var t = tileOf(p.n), tw = ax(off + t.bytes) - x0;
        var y0 = ay(t.from), y1 = ay(t.to);
        var isSel = p.n === sel;
        var col = p.reused ? 'var(--warning)' : 'var(--primary)';
        pos[p.n] = { x: x0 + Math.max(16, tw) / 2, y0: y0, y1: y1 };
        s += '<g class="tl' + (isSel ? ' is-sel' : '') + '" data-tile="' + p.n + '" data-tip="' +
             esc('<b>' + t.n + '</b>' + t.label + '<br><em>' + kb(t.bytes) + ' · 行 ' + t.from + '–' + t.to + ' · ' + (t.op || '') + '</em>') + '">' +
             '<rect x="' + x0 + '" y="' + y0 + '" width="' + Math.max(16, tw) + '" height="' + Math.max(14, y1 - y0) +
             '" rx="5" fill="color-mix(in srgb,' + col + ' ' + (isSel ? '46' : '26') + '%,var(--surface-1))" stroke="' +
             (isSel ? 'var(--foreground)' : col) + '" stroke-width="' + (isSel ? 2.2 : 1.2) + '"' +
             (p.reused ? ' stroke-dasharray="' + (isSel ? 'none' : '4 2') + '"' : '') + '/>' +
             '<text class="t-xs" x="' + (x0 + 6) + '" y="' + (y0 + 13) + '" style="fill:var(--foreground)">' +
             esc(t.n.replace('mem_', '')) + '</text></g>';
      });
      off += w;
    });
    // 复用接力连线
    bufIds.forEach(function (id) {
      var ms = byBuf[id].slice().sort(function (p, q) { return tileOf(p.n).from - tileOf(q.n).from; });
      for (var i = 1; i < ms.length; i++) {
        var A2 = pos[ms[i - 1].n], B2 = pos[ms[i].n];
        if (!A2 || !B2) continue;
        s += '<path d="M' + A2.x + ' ' + A2.y1 + ' C ' + (A2.x + 26) + ' ' + (A2.y1 + 10) + ', ' + (B2.x + 26) + ' ' + (B2.y0 - 10) + ', ' + B2.x + ' ' + B2.y0 +
             '" fill="none" stroke="var(--warning)" stroke-width="1.6" marker-end="url(#pkarw)" opacity=".9"/>';
      }
    });
    s += '<defs><marker id="pkarw" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">' +
         '<path d="M0 0 L8 4 L0 8 z" fill="var(--warning)"/></marker></defs>';
    var cx = ax(cap);
    if (cx < W - 16) {
      s += '<line x1="' + cx + '" y1="' + (padT - 18) + '" x2="' + cx + '" y2="' + (Hh - 10) + '" stroke="var(--danger)" stroke-dasharray="4 3"/>' +
           '<text class="t-xs" x="' + (cx + 5) + '" y="' + (Hh - 12) + '" style="fill:var(--danger)">容量 ' + kb(cap) + '</text>';
    }
    s += '</svg>';

    /* ---------- 右：详情 ---------- */
    var detail;
    if (sel && tileOf(sel)) {
      var t = tileOf(sel);
      var myBuf = place.filter(function (p) { return p.n === sel; })[0];
      var mates = myBuf ? byBuf[myBuf.buf].filter(function (p) { return p.n !== sel; }) : [];
      detail = '<div class="dt__h">' + esc(t.n) + (t.cited ? '<span class="dt__cite">尺寸见 #1820 原文</span>' : '') + '</div>' +
        '<div class="dt__sub">' + esc(t.label) + '</div>' +
        [['占用', kb(t.bytes) + '（' + t.bytes + ' B）'],
         ['内存空间', t.space || C.space],
         ['产生它的算子', t.op || '—'],
         ['生命周期', '行 ' + t.from + ' → ' + t.to + '（' + (t.to - t.from) + ' 条语句）'],
         ['当前所在 buffer', myBuf ? ('#' + myBuf.buf + (myBuf.reused ? '（复用接管）' : '（代表）')) : '未放置'],
         ['同 buffer 的邻居', mates.length ? mates.map(function (p) { return p.n; }).join('、') : '独占']
        ].map(function (r) {
          return '<div class="dt__r"><span class="dt__k">' + esc(r[0]) + '</span><span class="dt__v">' + esc(r[1]) + '</span></div>';
        }).join('') +
        (t.forbidden ? '<div class="dt__warn"><b>别名禁令</b>' + rich(t.forbidden) + '</div>' : '') +
        '<div class="dt__act"><button type="button" class="btn btn-sm" data-clearsel="1">清除选中</button></div>';
    } else {
      detail = '<div class="dt__empty">点左侧源码行、或中间任一色块，<br>三栏会同时定位到那个 tile。</div>';
    }


    /* ---------- 执行序泳道：与内存图共享选中态 ---------- */
    var SW = 1000, sPadL = 92, sRowH = 26, sTop = 26;
    var SH = sTop + PK.pipes.length * sRowH + 34;
    var sx = function (o) { return sPadL + o / PK.schedSpan * (SW - sPadL - 20); };
    var lane = function (p) { return sTop + PK.pipes.indexOf(p) * sRowH; };
    var touches = function (ins) { return (ins.reads || []).concat(ins.writes || []); };

    // 选中 tile 的存活区间 = 第一条写它的指令 → 最后一条读它的指令
    var live = null;
    if (sel) {
      var hit = PK.sched.filter(function (i) { return touches(i).indexOf(sel) >= 0; });
      if (hit.length) live = { a: hit[0].a, b: hit[hit.length - 1].a + hit[hit.length - 1].w };
    }

    var sv = '<svg viewBox="0 0 ' + SW + ' ' + SH + '" style="min-width:' + SW + 'px" role="img" aria-label="执行序泳道">';
    if (live) {
      sv += '<rect x="' + sx(live.a) + '" y="' + (sTop - 8) + '" width="' + (sx(live.b) - sx(live.a)) +
            '" height="' + (PK.pipes.length * sRowH + 12) + '" rx="7" fill="color-mix(in srgb,var(--primary) 13%,transparent)" ' +
            'stroke="color-mix(in srgb,var(--primary) 45%,transparent)" stroke-dasharray="4 3"/>';
      sv += '<text class="t-xs" x="' + (sx(live.a) + 6) + '" y="' + (sTop - 13) + '" style="fill:var(--primary)">' +
            esc(sel.replace('mem_', '')) + ' 存活区间</text>';
    }
    PK.pipes.forEach(function (p) {
      var y = lane(p);
      sv += '<text class="t-md" x="' + (sPadL - 12) + '" y="' + (y + sRowH / 2 + 3) + '" text-anchor="end">' + esc(p) + '</text>';
      sv += '<line class="ln-d" x1="' + sPadL + '" y1="' + (y + sRowH - 4) + '" x2="' + (SW - 20) + '" y2="' + (y + sRowH - 4) + '"/>';
    });
    PK.sched.forEach(function (ins) {
      var y = lane(ins.p), x0 = sx(ins.a), w2 = sx(ins.a + ins.w) - x0;
      var tt = touches(ins);
      var isRel = sel && tt.indexOf(sel) >= 0;
      var isW = sel && (ins.writes || []).indexOf(sel) >= 0;
      var dim = sel && !isRel;
      var col = isW ? 'var(--success)' : (isRel ? 'var(--primary)' : 'var(--foreground-muted)');
      sv += '<g class="ins' + (isRel ? ' is-rel' : '') + '" data-tip="' +
        esc('<b>' + ins.n + '</b>源码行 ' + ins.line + '　pipe ' + ins.p +
            (tt.length ? '<br><em>' + ((ins.writes || []).length ? '写 ' + ins.writes.join('、') + '　' : '') +
             ((ins.reads || []).length ? '读 ' + ins.reads.join('、') : '') + '</em>' : '')) + '"' +
        (ins.line ? ' data-line="' + ins.line + '"' : '') + ' opacity="' + (dim ? '.3' : '1') + '">' +
        '<rect x="' + x0 + '" y="' + (y + 3) + '" width="' + w2 + '" height="' + (sRowH - 10) + '" rx="4" fill="color-mix(in srgb,' +
        col + ' ' + (isRel ? '40' : '20') + '%,var(--surface-1))" stroke="' + col + '" stroke-width="' + (isRel ? 1.8 : 1) + '"/>' +
        '<text class="t-xs" x="' + (x0 + 5) + '" y="' + (y + sRowH / 2 + 1) + '" style="fill:var(--foreground)">' + esc(ins.n) + '</text></g>';
      if (isW) {
        sv += '<circle cx="' + x0 + '" cy="' + (y + 3) + '" r="3.5" fill="var(--success)" stroke="var(--background)" stroke-width="1.2"/>';
      }
    });
    sv += '<text class="t-xs" x="' + sPadL + '" y="' + (SH - 10) + '">执行序 →　' +
          (sel ? '<tspan style="fill:var(--success)">● 写入</tspan>　<tspan style="fill:var(--primary)">▮ 读取</tspan>　灰=与选中无关' : '选中一个 tile 看它的存活区间') +
          '</text></svg>';

    /* ---------- 帧内容 ---------- */
    var extra = '';
    if (F.order) {
      extra = '<div class="picks">' + F.order.map(function (n, i) {
        var t2 = tileOf(n);
        return '<span class="pick' + (n === sel ? ' is-sel' : '') + '" data-tile="' + n + '">' + (i + 1) + '. ' +
               esc(n.replace('mem_', '')) + '　' + kb(t2.bytes) + '</span>';
      }).join('<span class="picksep">›</span>') + '</div>';
    }
    if (F.gates) {
      extra = '<div class="gates">' + F.gates.map(function (g) {
        var def = gateById[g.id];
        return '<div class="gate ' + (g.pass ? 'pass' : 'fail') + '"><span class="gate__i">' + (g.pass ? '✓' : '✕') + '</span>' +
          '<span><span class="gate__n">' + esc(def.name) + '</span><code class="gate__c">' + esc(def.code) + '</code></span>' +
          '<span class="gate__w">' + rich(g.note) + (g.evidence ? '<br><span class="gate__e">' + esc(g.evidence) + '</span>' : '') + '</span></div>';
      }).join('') + '</div>';
    }

    var nav = '<div class="pk__bar">' +
      PK.cases.map(function (c) {
        return '<button type="button" class="pk__case' + (c.id === state.pkCase ? ' is-on' : '') + '" data-case="' + c.id + '">' + esc(c.title) + '</button>';
      }).join('') +
      '<div class="pk__nav"><div class="pk__frames">' +
      C.frames.map(function (f, i) {
        return '<button type="button" class="pk__f' + (i === fi ? ' is-on' : (i < fi ? ' is-done' : '')) + '" data-frame="' + i + '" title="' + esc(f.t) + '">' + (i + 1) + '</button>';
      }).join('') + '</div>' +
      '<button type="button" class="btn" id="pk-prev"' + (fi === 0 ? ' disabled' : '') + '>◀</button>' +
      '<span class="pk__step">' + (fi + 1) + ' / ' + C.frames.length + '</span>' +
      '<button type="button" class="btn" id="pk-next"' + (fi === C.frames.length - 1 ? ' disabled' : '') + '>▶</button></div></div>';

    return '<div class="wrap">' +
      '<h1 class="h1">MemoryReuse 工作台</h1>' +
      '<p class="lead">第 34 步 <code>MemoryReuse</code> 按内存空间分组、组内 <strong>largest-first</strong> 打包，' +
      '每个候选要加入某个 buffer 必须与它的<strong>所有</strong>成员都过 <code>can_share</code> 的五道门。' +
      '<strong>三栏联动</strong>：点源码行或色块，源码 / 布局 / 详情同时定位。算法与判据取自 <code>' + esc(PK.algoRef) + '</code>。' +
      (C.derived ? '<span class="derived">本场景 tile 尺寸由 shape 推算</span>' : '') + '</p>' +
      '<div class="kpis">' + kpi + '</div>' +
      nav +
      '<div class="wb">' +
      '  <div class="wb__pane"><div class="wb__h">源码<span>' + (sel ? '定位 ' + esc(sel) : 'qk_pv') + '</span></div>' +
      '    <div class="wb__b code3">' + code + '</div></div>' +
      '  <div class="wb__pane"><div class="wb__h">地址 × 生命周期<span>帧 ' + (fi + 1) + '：' + esc(F.t) + '</span></div>' +
      '    <div class="wb__b">' + s + '</div></div>' +
      '  <div class="wb__pane"><div class="wb__h">对象详情<span>' + (sel ? 'selected' : '未选中') + '</span></div>' +
      '    <div class="wb__b dt">' + detail + '</div></div>' +
      '</div>' +
      '<div class="card"><div class="card__h"><span class="card__t">执行序泳道</span>' +
      '<span class="card__m">' + (sel ? esc(sel) + ' 的存活区间已框出' : '与上面三栏共享选中态') + '</span></div>' +
      '<div class="scroll">' + sv + '</div>' +
      '<p class="n">泳道按 Ascend 的搬运与计算单元划分。<strong>绿点是写入、蓝框是读取</strong>，' +
      '与选中 tile 无关的指令会淡出。虚线框就是这块 buffer 真正被占住的区间——' +
      '<strong>MemoryReuse 判「生命周期重叠」判的就是这段</strong>。指令序列按 qk_pv 的结构还原，不是实测周期。</p></div>' +
      '<div class="card"><div class="card__h"><span class="card__t">帧 ' + (fi + 1) + '　' + esc(F.t) + '</span>' +
      '<span class="card__m">' + esc(C.cite) + '</span></div>' +
      '<p class="pk__d">' + rich(F.d) + '</p>' + extra +
      '<div class="pk__hw' + (F.risk ? ' risk' : '') + '"><p>' + rich(F.hw) + '</p></div></div>' +
      '<div class="card"><div class="card__h"><span class="card__t">can_share 的五道门</span>' +
      '<span class="card__m">' + esc(PK.algoRef) + '</span></div>' +
      PK.gates.map(function (g) {
        return '<div class="gate"><span class="gate__i" style="color:var(--foreground-disabled)">·</span>' +
          '<span><span class="gate__n">' + esc(g.name) + '</span><code class="gate__c">' + esc(g.code) + '</code></span>' +
          '<span class="gate__w">' + rich(g.why) + '</span></div>';
      }).join('') + '</div></div>';
  }


  /* ---------- 渲染 ---------- */
  function currentView() {
    var v = VIEWS.filter(function (x) { return x.id === state.view; })[0];
    if (!v) { state.view = VIEWS[0].id; v = VIEWS[0]; }   // 未知 view id 时兜底，避免整页卡死
    return v;
  }

  function render() {
    hideTip();
    var cv = currentView();
    $('#tabs').innerHTML = VIEWS.map(function (v) {
      return '<button type="button" class="tab' + (state.view === v.id ? ' is-on' : '') + '" data-v="' + v.id + '">' + esc(v.label) + '</button>';
    }).join('');
    $$('#tabs .tab').forEach(function (b) { b.addEventListener('click', function () { state.view = b.dataset.v; render(); }); });

    $('#socsw').innerHTML = ['a5', 'a2a3'].map(function (k) {
      return '<button type="button" class="socbtn' + (state.soc === k ? ' is-on' : '') + '" data-s="' + k + '">' + esc(HW.socs[k].name) + '</button>';
    }).join('');
    $$('#socsw .socbtn').forEach(function (b) { b.addEventListener('click', function () { state.soc = b.dataset.s; render(); }); });
    $('#socsw').style.display = cv.soc ? '' : 'none';
    $('#search').hidden = !cv.rail;
    $('#rail-pane').hidden = !cv.rail;
    $('#meta').textContent = A.passes.length + ' 步 · ' + A.meta.decide + ' 个决策点 · 快照 ' + A.meta.snapshot;

    if (cv.rail) renderRail();

    var v = $('#view');
    v.innerHTML = state.view === 'overview' ? viewOverview()
                : state.view === 'pipe'     ? viewPipe()
                : state.view === 'pass'     ? viewAtlas()
                : state.view === 'props'    ? viewProps()
                : state.view === 'hw'       ? viewHW()
                : state.view === 'op'       ? viewOp()
                :                              viewPack();
    v.scrollTop = 0;
    bindTip(v);
    $$('[data-go]', v).forEach(function (b) {
      b.addEventListener('click', function () { state.sel = b.dataset.go; state.view = 'pass'; render(); });
    });
    $$('[data-prop]', v).forEach(function (b) {
      b.addEventListener('click', function () { state.q = b.dataset.prop; $('#search').value = state.q; state.view = 'props'; render(); });
    });
    $$('[data-hwmode]', v).forEach(function (b) {
      b.addEventListener('click', function () { state.hwMode = b.dataset.hwmode; render(); });
    });
    $$('[data-case]', v).forEach(function (b) {
      b.addEventListener('click', function () { state.pkCase = b.dataset.case; state.pkFrame = 0; state.pkSel = null; render(); });
    });
    $$('[data-frame]', v).forEach(function (b) {
      b.addEventListener('click', function () { state.pkFrame = Number(b.dataset.frame); render(); });
    });
    $$('[data-tile]', v).forEach(function (b) {
      b.addEventListener('click', function () {
        state.pkSel = (state.pkSel === b.dataset.tile) ? null : b.dataset.tile;   // 再点一次取消
        render();
      });
    });
    if ($('[data-clearsel]', v)) $('[data-clearsel]', v).addEventListener('click', function () { state.pkSel = null; render(); });
    if ($('#pk-prev', v)) $('#pk-prev', v).addEventListener('click', function () { state.pkFrame = Math.max(0, state.pkFrame - 1); render(); });
    if ($('#pk-next', v)) $('#pk-next', v).addEventListener('click', function () { state.pkFrame++; render(); });
  }

  function init() {
    installTipGuards();
    $('#search').addEventListener('input', function (e) { state.q = e.target.value.trim(); if (currentView().rail) renderRail(); });
    document.addEventListener('keydown', function (e) {
      if (/^(INPUT|TEXTAREA)$/.test(e.target.tagName)) return;
      if (state.view === 'pack') {
        var C = PK.cases.filter(function (c) { return c.id === state.pkCase; })[0];
        if (e.key === 'ArrowRight' && state.pkFrame < C.frames.length - 1) { state.pkFrame++; render(); e.preventDefault(); }
        if (e.key === 'ArrowLeft' && state.pkFrame > 0) { state.pkFrame--; render(); e.preventDefault(); }
        return;
      }
      if (state.view === 'pass') {
        var list = A.passes, i = list.findIndex(function (p) { return p.name === state.sel; });
        if (e.key === 'ArrowDown' && i < list.length - 1) { state.sel = list[i + 1].name; render(); e.preventDefault(); }
        if (e.key === 'ArrowUp' && i > 0) { state.sel = list[i - 1].name; render(); e.preventDefault(); }
      }
    });
    render();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
