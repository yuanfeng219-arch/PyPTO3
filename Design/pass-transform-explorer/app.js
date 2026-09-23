/* Pass Transform Explorer
 *
 * The timeline and the evidence cards come precomputed from build.mjs, so the
 * app is interactive immediately. Diffs and graphs are computed live in the
 * browser from the IR snapshots themselves (lib/bundle.js is the same parser
 * the build uses), which is what lets any function, any lens and any pass be
 * inspected at full fidelity without shipping a giant precomputed blob.
 */
(function () {
  'use strict';

  var LIB = window.PTXLib;
  var INDEX = window.PTX_INDEX;

  // ── snapshot loading ──────────────────────────────────────────────────
  var SRC = Object.create(null);
  var WAITING = Object.create(null);

  window.PTX = {
    src: function (runId, idx, text) {
      var key = runId + ':' + idx;
      SRC[key] = text;
      (WAITING[key] || []).forEach(function (fn) { fn(text); });
      delete WAITING[key];
    },
  };

  /**
   * Snapshots arrive one of two ways: as `data/<run>/NN.js` next to the page,
   * or - in the single-file demo - gzip+base64 inside the document itself.
   * Either way they are fetched one at a time, only when a view needs them.
   */
  function loadSnapshot(runId, idx) {
    var key = runId + ':' + idx;
    if (SRC[key]) return Promise.resolve(SRC[key]);

    if (window.PTX_EMBEDDED) {
      var packed = window.PTX_EMBEDDED[key];
      if (!packed) return Promise.reject(new Error('这份 demo 未内嵌快照 ' + key));
      return gunzipBase64(packed).then(function (text) {
        SRC[key] = text;
        return text;
      });
    }

    return new Promise(function (resolve, reject) {
      if (WAITING[key]) { WAITING[key].push(resolve); return; }
      WAITING[key] = [resolve];
      var s = document.createElement('script');
      s.src = 'data/' + runId + '/' + String(idx).padStart(2, '0') + '.js';
      s.onerror = function () {
        delete WAITING[key];
        reject(new Error('无法加载快照 ' + s.src + '（若通过 file:// 打开，请改用本地静态服务器）'));
      };
      document.head.appendChild(s);
    });
  }

  function gunzipBase64(b64) {
    if (typeof DecompressionStream === 'undefined') {
      return Promise.reject(new Error('这个浏览器不支持 DecompressionStream，无法解压内嵌快照。请改用 Chrome 80+ / Firefox 113+ / Safari 16.4+。'));
    }
    var bin = atob(b64);
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    var stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Response(stream).text();
  }

  // Parsing a snapshot costs ~50ms, so a handful are kept around; stepping
  // back and forth along the timeline then stays instant.
  var ANALYSIS = new Map();
  var ANALYSIS_CAP = 8;

  function analyze(runId, idx) {
    var key = runId + ':' + idx;
    if (ANALYSIS.has(key)) {
      var hit = ANALYSIS.get(key);
      ANALYSIS.delete(key);
      ANALYSIS.set(key, hit);
      return Promise.resolve(hit);
    }
    return loadSnapshot(runId, idx).then(function (text) {
      var lines = text.split(/\r?\n/);
      var an = LIB.analyzeProgram(LIB.parseDump(text, String(idx)), lines);
      an.lines = lines;
      an.text = text;
      ANALYSIS.set(key, an);
      while (ANALYSIS.size > ANALYSIS_CAP) ANALYSIS.delete(ANALYSIS.keys().next().value);
      return an;
    });
  }

  // ── tiny DOM helpers ──────────────────────────────────────────────────
  function $(id) { return document.getElementById(id); }
  var esc = LIB.escapeHtml;
  var mdInline = LIB.mdInline;
  var md = LIB.md;
  function fmt(n) { return typeof n === 'number' ? n.toLocaleString('en-US') : n; }
  function bytes(n) { return LIB.fmtBytes(n); }

  function toast(msg) {
    var t = $('toast');
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(function () { t.hidden = true; }, 4200);
  }

  // ── state ─────────────────────────────────────────────────────────────
  var state = {
    runId: INDEX.runs[0].id,
    passIdx: 1,
    tab: 'overview',
    fn: null,
    lens: null,
    diffMode: 'split',
    onlyChanged: false,
    jumpLine: null,
    filter: '',
    docOpen: true,
  };

  function run() { return INDEX.runs.find(function (r) { return r.id === state.runId; }); }
  function pass() { return run().passes.find(function (p) { return p.idx === state.passIdx; }); }
  function phase(id) { return INDEX.phases.find(function (p) { return p.id === id; }) || { label: id, hint: '' }; }

  // ══════════════════════════════════════════════════════════════════════
  // Timeline
  // ══════════════════════════════════════════════════════════════════════

  function renderRail() {
    var r = run();
    var changedCount = r.passes.filter(function (p) { return p.idx > 0 && p.changed; }).length;
    var total = r.passes.length - 1;
    $('railSummary').innerHTML = '<strong>' + changedCount + '</strong> / ' + total
      + ' 个 Pass 改动了 IR<span class="ptx-rail__sub">' + (total - changedCount) + ' 个对本算子为空操作</span>';

    var maxChurn = 1;
    r.passes.forEach(function (p) { maxChurn = Math.max(maxChurn, p.diff.add + p.diff.del); });

    var filter = state.filter.toLowerCase();
    var html = '';
    var lastPhase = null;

    r.passes.forEach(function (p) {
      if (state.onlyChanged && p.idx > 0 && !p.changed) return;
      if (filter && p.name.toLowerCase().indexOf(filter) < 0) return;

      if (p.phase !== lastPhase) {
        var ph = phase(p.phase);
        html += '<div class="ptx-phasehead" title="' + esc(ph.hint) + '">' + esc(ph.label) + '</div>';
        lastPhase = p.phase;
      }

      var churn = p.diff.add + p.diff.del;
      var w = churn ? Math.max(3, Math.round((churn / maxChurn) * 100)) : 0;
      var addW = churn ? Math.round((p.diff.add / churn) * w) : 0;

      html += '<button class="ptx-pass' + (p.idx === state.passIdx ? ' is-active' : '')
        + (p.idx > 0 && !p.changed ? ' is-noop' : '') + '" data-idx="' + p.idx + '">'
        + '<span class="ptx-pass__idx">' + String(p.idx).padStart(2, '0') + '</span>'
        + '<span class="ptx-pass__body">'
        + '<span class="ptx-pass__name">' + esc(p.name === 'frontend' ? '前端 IR' : p.name) + '</span>'
        + '<span class="ptx-pass__bar">'
        + '<i class="ptx-pass__bar-add" style="width:' + addW + '%"></i>'
        + '<i class="ptx-pass__bar-del" style="width:' + (w - addW) + '%"></i>'
        + '</span></span>'
        + '<span class="ptx-pass__churn">' + (churn ? fmt(churn) : '—') + '</span>'
        + '</button>';
    });

    $('passList').innerHTML = html || '<p class="ptx-empty">没有匹配的 Pass。</p>';
    var active = $('passList').querySelector('.is-active');
    if (active) active.scrollIntoView({ block: 'nearest' });
  }

  // ══════════════════════════════════════════════════════════════════════
  // Pass header
  // ══════════════════════════════════════════════════════════════════════

  function renderHeader() {
    var p = pass();
    var ph = phase(p.phase);
    $('passPhase').textContent = ph.label;
    $('passPhase').title = ph.hint;
    $('passName').textContent = p.name === 'frontend' ? '前端 IR（Pass 流水线输入）' : p.name;
    $('passHeadline').textContent = p.headline;

    var d = p.diff;
    $('passDelta').innerHTML = p.idx === 0 ? '<span class="ptx-muted">流水线起点</span>'
      : (d.add || d.del)
        ? '<b class="ptx-add">+' + fmt(d.add) + '</b> <b class="ptx-del">−' + fmt(d.del) + '</b> 行'
        : '<span class="ptx-noop">未改动 IR</span>';

    // In the single-file demo the repo is not alongside the page, so the path
    // is shown but not offered as a link that would 404.
    var link = $('passSource');
    if (p.source) {
      link.hidden = false;
      link.textContent = p.source.split('/').pop();
      if (window.PTX_STANDALONE) {
        link.removeAttribute('href');
        link.title = '仓库内路径：' + p.source;
        link.classList.add('is-inert');
      } else {
        link.href = '../../' + p.source;
        link.title = p.source;
      }
    } else {
      link.hidden = true;
    }

    document.querySelectorAll('.ptx-tab').forEach(function (b) {
      b.classList.toggle('is-active', b.dataset.tab === state.tab);
    });
    ['overview', 'diff', 'graph'].forEach(function (t) {
      $('view' + t[0].toUpperCase() + t.slice(1)).hidden = state.tab !== t;
    });
  }

  // ══════════════════════════════════════════════════════════════════════
  // Overview: evidence cards + function change table
  // ══════════════════════════════════════════════════════════════════════

  var TONE_CLASS = { add: 'is-add', remove: 'is-del', change: 'is-chg', neutral: '' };

  function renderOverview() {
    var p = pass();
    var m = p.metrics;
    var prev = run().passes.find(function (x) { return x.idx === p.idx - 1; });
    var pm = prev ? prev.metrics : null;

    var html = '<div class="ptx-metrics">' + [
      ['语句', m.stmts], ['函数', m.functions], ['循环', m.loops],
      ['Tile 值', m.tiles], ['Tensor 值', m.tensors],
      ['tile.alloc', m.allocs], ['片上内存', m.allocBytes, 'bytes'],
      ['任务', m.tasks], ['最大嵌套', m.maxNest],
    ].map(function (row) {
      var v = row[2] === 'bytes' ? bytes(row[1]) : fmt(row[1]);
      var d = pm ? row[1] - pm[metricKey(row[0])] : 0;
      return '<div class="ptx-metric"><span class="ptx-metric__label">' + row[0] + '</span>'
        + '<span class="ptx-metric__value">' + v + '</span>'
        + (d ? '<span class="ptx-metric__delta ' + (d > 0 ? 'ptx-add' : 'ptx-del') + '">'
          + (d > 0 ? '+' : '−') + (row[2] === 'bytes' ? bytes(Math.abs(d)) : fmt(Math.abs(d))) + '</span>' : '')
        + '</div>';
    }).join('') + '</div>';

    if (!p.evidence.length) {
      html += p.idx === 0
        ? '<p class="ptx-note">这是 Pass 流水线的输入快照。切到「结构图」可以先看清算子本身的结构，再沿时间线逐个 Pass 往下走。</p>'
        : '<p class="ptx-note ptx-note--noop">本 Pass 在这个算子上是空操作：逐行比对前后快照完全一致。这本身是有用的结论——它说明该 Pass 的触发条件没有在这段 IR 上命中。</p>';
    }

    p.evidence.forEach(function (card) {
      html += '<section class="ptx-card ' + (TONE_CLASS[card.tone] || '') + '">'
        + '<h3>' + esc(card.title) + '</h3>'
        + '<p class="ptx-card__headline">' + mdInline(card.headline) + '</p>';

      if (card.rows && card.rows.length) {
        html += '<table class="ptx-table"><thead><tr><th></th><th>之前</th><th>之后</th><th>Δ</th></tr></thead><tbody>';
        card.rows.forEach(function (r) {
          var d = r.after - r.before;
          var f = r.fmt === 'bytes' ? bytes : fmt;
          html += '<tr><td>' + esc(r.label) + '</td><td>' + f(r.before) + '</td><td>' + f(r.after) + '</td>'
            + '<td class="' + (d > 0 ? 'ptx-add' : d < 0 ? 'ptx-del' : 'ptx-muted') + '">'
            + (d > 0 ? '+' : d < 0 ? '−' : '') + (d ? f(Math.abs(d)) : '0') + '</td></tr>';
        });
        html += '</tbody></table>';
      }

      if (card.items && card.items.length) {
        html += '<ul class="ptx-items">' + card.items.map(function (it) {
          return '<li class="' + (TONE_CLASS[it.tone] || '') + '"><code>' + esc(it.label) + '</code>'
            + '<span>' + esc(it.note) + '</span>'
            + (it.delta ? '<b>' + (it.delta > 0 ? '+' : '−') + Math.abs(it.delta) + '</b>' : '') + '</li>';
        }).join('') + '</ul>';
        if (card.more) html += '<p class="ptx-more">另有 ' + card.more + " 项，见「代码 Diff」</p>";
      }

      if (card.subs && card.subs.length) {
        html += '<table class="ptx-table ptx-table--subs"><thead><tr><th>次数</th><th>改写前</th><th>改写后</th><th>示例</th></tr></thead><tbody>';
        card.subs.forEach(function (s) {
          html += '<tr><td class="ptx-num">' + s.count + '</td>'
            + '<td><code class="ptx-del-code">' + esc(trunc(s.from, 46)) + '</code></td>'
            + '<td><code class="ptx-add-code">' + esc(trunc(s.to, 46)) + '</code></td>'
            + '<td class="ptx-ex" title="' + esc(s.example.after) + '"><code>' + esc(trunc(s.example.after, 70)) + '</code></td></tr>';
        });
        html += '</tbody></table>'
          + '<p class="ptx-more">' + card.stats.paired + ' 行成对改写 · '
          + card.stats.pureAdd + ' 行纯新增 · ' + card.stats.pureDel + ' 行纯删除 · '
          + card.stats.distinct + ' 种不同改写</p>';
      }
      html += '</section>';
    });

    if (p.changedFunctions && p.changedFunctions.length) {
      html += '<section class="ptx-card"><h3>受影响的函数</h3>'
        + '<p class="ptx-card__headline">' + p.changedFunctions.length + ' 个函数被改动，'
        + p.sameFunctions + ' 个完全未变</p>'
        + '<table class="ptx-table ptx-table--fns"><thead><tr><th>函数</th><th>状态</th><th>语句</th><th>行数</th><th></th></tr></thead><tbody>';
      p.changedFunctions.slice(0, 60).forEach(function (f) {
        html += '<tr><td><code>' + esc(f.name) + '</code>'
          + (f.kind ? '<em>' + esc(f.kind) + '</em>' : '') + '</td>'
          + '<td><span class="ptx-status ptx-status--' + f.status + '">' + statusLabel(f.status) + '</span></td>'
          + '<td>' + deltaCell(f.stmtsBefore, f.stmtsAfter) + '</td>'
          + '<td>' + deltaCell(f.linesBefore, f.linesAfter) + '</td>'
          + '<td><button class="ptx-linkbtn" data-openfn="' + esc(f.name) + '">查看 Diff</button></td></tr>';
      });
      html += '</tbody></table>';
      if (p.changedFunctions.length > 60) html += '<p class="ptx-more">仅列出前 60 个</p>';
      html += '</section>';
    }

    $('viewOverview').innerHTML = html;
  }

  function metricKey(label) {
    return { 语句: 'stmts', 函数: 'functions', 循环: 'loops', 'Tile 值': 'tiles', 'Tensor 值': 'tensors', 'tile.alloc': 'allocs', 片上内存: 'allocBytes', 任务: 'tasks', 最大嵌套: 'maxNest' }[label];
  }
  function statusLabel(s) { return { added: '新增', removed: '移除', changed: '改写', same: '未变' }[s] || s; }
  function deltaCell(before, after) {
    if (before === after) return fmt(after);
    return '<span class="ptx-muted">' + fmt(before) + '</span> → <b>' + fmt(after) + '</b>';
  }
  function trunc(s, n) { s = String(s); return s.length > n ? s.slice(0, n - 1) + '…' : s; }

  // ══════════════════════════════════════════════════════════════════════
  // Code diff
  // ══════════════════════════════════════════════════════════════════════

  function renderDiff() {
    var p = pass();
    if (p.idx === 0) {
      $('diffFnChips').innerHTML = '';
      $('diffBody').innerHTML = '<p class="ptx-empty">流水线起点没有可比较的上一版本。</p>';
      return;
    }
    if (!p.changed) {
      $('diffFnChips').innerHTML = '';
      $('diffBody').innerHTML = '<p class="ptx-empty">本 Pass 未改动任何一行 IR。</p>';
      return;
    }

    var fns = p.changedFunctions;
    if (!state.fn || !fns.some(function (f) { return f.name === state.fn; })) state.fn = fns[0].name;

    $('diffFnChips').innerHTML = fns.map(function (f) {
      var churn = (f.add || 0) + (f.del || 0);
      return '<button class="ptx-chip ptx-chip--' + f.status + (f.name === state.fn ? ' is-active' : '')
        + '" data-fn="' + esc(f.name) + '" title="' + esc(f.name) + ' · ' + statusLabel(f.status) + '">'
        + esc(f.name)
        + (churn ? '<b>' + fmt(churn) + '</b>' : '<b>' + statusLabel(f.status) + '</b>')
        + '</button>';
    }).join('');

    $('diffBody').innerHTML = '<p class="ptx-loading">正在解析前后快照…</p>';
    var token = ++renderDiff._token;

    Promise.all([analyze(state.runId, p.idx - 1), analyze(state.runId, p.idx)])
      .then(function (pair) {
        if (token !== renderDiff._token) return;
        paintDiff(pair[0], pair[1]);
      })
      .catch(function (err) {
        $('diffBody').innerHTML = '<p class="ptx-empty">' + esc(err.message) + '</p>';
        toast(err.message);
      });
  }
  renderDiff._token = 0;

  function paintDiff(before, after) {
    var name = state.fn;
    var fa = before.byName.get(name);
    var fb = after.byName.get(name);
    var srcA = fa ? fa.src : [];
    var srcB = fb ? fb.src : [];
    var baseA = fa ? fa.decoLine : 1;
    var baseB = fb ? fb.decoLine : 1;

    var rows = LIB.diffLines(srcA, srcB);
    var hunks = LIB.toHunks(rows, 3, baseA, baseB);

    if (!hunks.length) {
      $('diffBody').innerHTML = '<p class="ptx-empty">该函数在本 Pass 中未发生变化。</p>';
      return;
    }

    var total = LIB.countChanges(rows);
    var head = '<div class="ptx-diffhead"><code>' + esc(name) + '</code>'
      + '<span class="ptx-add">+' + total.add + '</span><span class="ptx-del">−' + total.del + '</span>'
      + '<span class="ptx-muted">' + hunks.length + ' 处变更 · 前 ' + srcA.length + ' 行 / 后 ' + srcB.length + ' 行</span></div>';

    $('diffBody').innerHTML = head
      + (state.diffMode === 'split' ? splitView(hunks) : unifiedView(hunks));

    if (state.jumpLine) {
      var target = null;
      var cells = $('diffBody').querySelectorAll('.ptx-ln');
      for (var i = 0; i < cells.length; i++) {
        if (Number(cells[i].textContent) === state.jumpLine) { target = cells[i].parentNode; break; }
      }
      if (target) {
        target.classList.add('is-jump');
        target.scrollIntoView({ block: 'center' });
      } else {
        toast('源行 ' + state.jumpLine + ' 不在本 Pass 的变更范围内');
      }
      state.jumpLine = null;
    }
  }

  /** Pair adjacent -/+ runs so replaced lines line up and can be word-diffed. */
  function pairHunkRows(rows) {
    var out = [];
    var i = 0;
    while (i < rows.length) {
      var r = rows[i];
      if (r[0] === '=') { out.push({ kind: '=', a: r, b: r }); i++; continue; }
      var dels = [];
      var adds = [];
      while (i < rows.length && rows[i][0] === '-') { dels.push(rows[i]); i++; }
      while (i < rows.length && rows[i][0] === '+') { adds.push(rows[i]); i++; }
      var n = Math.max(dels.length, adds.length);
      for (var k = 0; k < n; k++) {
        out.push({ kind: dels[k] && adds[k] ? '~' : dels[k] ? '-' : '+', a: dels[k] || null, b: adds[k] || null });
      }
    }
    return out;
  }

  function inlineMarked(runs) {
    return runs.map(function (r) {
      return r[0] ? '<mark>' + esc(r[1]) + '</mark>' : esc(r[1]);
    }).join('');
  }

  function splitView(hunks) {
    return hunks.map(function (h) {
      var body = pairHunkRows(h.rows).map(function (pr) {
        var la = '';
        var lb = '';
        var ca = '';
        var cb = '';
        if (pr.kind === '=') {
          la = pr.a[1]; lb = pr.a[2]; ca = esc(pr.a[3]); cb = esc(pr.a[3]);
        } else if (pr.kind === '~') {
          var w = LIB.wordDiff(pr.a[3], pr.b[3]);
          la = pr.a[1]; lb = pr.b[2];
          ca = inlineMarked(w.left); cb = inlineMarked(w.right);
        } else if (pr.kind === '-') {
          la = pr.a[1]; ca = esc(pr.a[3]);
        } else {
          lb = pr.b[2]; cb = esc(pr.b[3]);
        }
        var cls = pr.kind === '=' ? '' : pr.kind === '~' ? 'is-chg' : pr.kind === '-' ? 'is-del' : 'is-add';
        return '<tr class="' + cls + '">'
          + '<td class="ptx-ln">' + (la || '') + '</td>'
          + '<td class="ptx-code ' + (pr.kind === '+' ? 'is-blank' : '') + '"><pre>' + ca + '</pre></td>'
          + '<td class="ptx-ln">' + (lb || '') + '</td>'
          + '<td class="ptx-code ' + (pr.kind === '-' ? 'is-blank' : '') + '"><pre>' + cb + '</pre></td></tr>';
      }).join('');
      return '<div class="ptx-hunk"><div class="ptx-hunk__head">@@ 前 ' + h.aStart + ' · 后 ' + h.bStart
        + ' @@ <span class="ptx-add">+' + h.add + '</span> <span class="ptx-del">−' + h.del + '</span></div>'
        + '<table class="ptx-difftable ptx-difftable--split"><tbody>' + body + '</tbody></table></div>';
    }).join('');
  }

  function unifiedView(hunks) {
    return hunks.map(function (h) {
      var body = h.rows.map(function (r) {
        var cls = r[0] === '+' ? 'is-add' : r[0] === '-' ? 'is-del' : '';
        var sign = r[0] === '=' ? ' ' : r[0];
        return '<tr class="' + cls + '">'
          + '<td class="ptx-ln">' + (r[1] || '') + '</td>'
          + '<td class="ptx-ln">' + (r[2] || '') + '</td>'
          + '<td class="ptx-sign">' + sign + '</td>'
          + '<td class="ptx-code"><pre>' + esc(r[3]) + '</pre></td></tr>';
      }).join('');
      return '<div class="ptx-hunk"><div class="ptx-hunk__head">@@ 前 ' + h.aStart + ' · 后 ' + h.bStart
        + ' @@ <span class="ptx-add">+' + h.add + '</span> <span class="ptx-del">−' + h.del + '</span></div>'
        + '<table class="ptx-difftable"><tbody>' + body + '</tbody></table></div>';
    }).join('');
  }

  // ══════════════════════════════════════════════════════════════════════
  // Structural graphs
  // ══════════════════════════════════════════════════════════════════════

  var LENSES = [
    { id: 'call', label: '调用 / 作用域', scope: 'program', hint: '函数与调用关系。内联、外提、核拆分在这里最直观。' },
    { id: 'control', label: '控制流', scope: 'fn', hint: '循环 / 分支 / 作用域的嵌套骨架。展开、流水线下降在这里最直观。' },
    { id: 'dataflow', label: '数据流', scope: 'fn', hint: 'SSA def-use 图。算子替换、类型与内存空间变化在这里最直观。' },
    { id: 'task', label: '任务 DAG', scope: 'fn', hint: 'submit / pl.at 任务及其依赖边。依赖推导与通信在这里最直观。' },
    { id: 'memory', label: '内存布局', scope: 'fn', hint: '缓冲区大小、空间归属与地址。复用与分配在这里最直观。' },
  ];

  function renderGraph() {
    var p = pass();
    if (!state.lens) state.lens = p.lens;

    $('lensPicker').innerHTML = LENSES.map(function (l) {
      return '<button data-lens="' + l.id + '" class="' + (l.id === state.lens ? 'is-active' : '')
        + (l.id === p.lens ? ' is-suggested' : '') + '" title="' + esc(l.hint) + '">' + l.label
        + (l.id === p.lens ? '<i>推荐</i>' : '') + '</button>';
    }).join('');

    $('graphBody').innerHTML = '<p class="ptx-loading">正在解析前后快照…</p>';
    var token = ++renderGraph._token;
    var needPrev = p.idx > 0;

    Promise.all([needPrev ? analyze(state.runId, p.idx - 1) : Promise.resolve(null), analyze(state.runId, p.idx)])
      .then(function (pair) {
        if (token !== renderGraph._token) return;
        paintGraph(pair[0], pair[1]);
      })
      .catch(function (err) {
        $('graphBody').innerHTML = '<p class="ptx-empty">' + esc(err.message) + '</p>';
        toast(err.message);
      });
  }
  renderGraph._token = 0;

  function paintGraph(before, after) {
    var lens = LENSES.find(function (l) { return l.id === state.lens; });
    var sel = $('graphFn');

    if (lens.scope === 'program') {
      sel.hidden = true;
      var ga = before ? LIB.callGraph(before) : { nodes: [], edges: [] };
      var gb = LIB.callGraph(after);
      drawGraph(merge(ga, gb, function (n) { return n.kind + '|' + n.level + '|' + n.role + '|' + n.stmts; }),
        { hint: lens.hint, kind: 'call' });
      return;
    }

    // Function-scoped lenses. Prefer a function this pass actually changed,
    // but never land on one that is empty under the current lens - opening the
    // recommended lens onto a blank canvas is the worst possible default.
    sel.hidden = false;
    var p = pass();
    var changed = {};
    (p.changedFunctions || []).forEach(function (f) { changed[f.name] = f.status; });

    var cand = after.functions.map(function (f) {
      return { name: f.name, weight: lensWeight(f, state.lens), status: changed[f.name] || 'same' };
    });
    var best = cand.slice().sort(function (a, b) {
      var ca = a.status !== 'same' ? 1 : 0;
      var cb = b.status !== 'same' ? 1 : 0;
      if ((a.weight > 0) !== (b.weight > 0)) return a.weight > 0 ? -1 : 1;
      if (ca !== cb) return cb - ca;
      return b.weight - a.weight;
    })[0];

    var current = cand.find(function (c) { return c.name === state.fn; });
    if (!current || (current.weight === 0 && best && best.weight > 0)) {
      state.fn = best ? best.name : (cand[0] && cand[0].name);
      current = cand.find(function (c) { return c.name === state.fn; });
    }

    sel.innerHTML = cand.map(function (c) {
      return '<option value="' + esc(c.name) + '"' + (c.name === state.fn ? ' selected' : '') + '>'
        + esc(c.name)
        + (c.status !== 'same' ? ' · ' + statusLabel(c.status) : '')
        + (c.weight ? ' · ' + c.weight + lensUnit(state.lens) : '')
        + '</option>';
    }).join('');

    var fb = after.byName.get(state.fn);
    var fa = before ? before.byName.get(state.fn) : null;

    if (state.lens === 'memory') { drawMemory(fa, fb, lens.hint); return; }

    var build = state.lens === 'control' ? LIB.controlTree
      : state.lens === 'dataflow' ? LIB.dataflowGraph
        : LIB.taskGraph;
    var sig = state.lens === 'control'
      ? function (n) { return n.label + '|' + n.detail; }
      : state.lens === 'dataflow'
        ? function (n) { return n.op + '|' + (n.shape || []).join('x') + '|' + n.dtype + '|' + n.space + '|' + n.buffer; }
        : function (n) { return n.label + '|' + n.level + '|' + n.type; };

    var GA = fa ? build(fa) : { nodes: [], edges: [] };
    var GB = fb ? build(fb) : { nodes: [], edges: [] };
    drawGraph(merge(GA, GB, sig), { hint: lens.hint, kind: state.lens, truncated: GA.truncated || GB.truncated });
  }

  /**
   * Union the before and after graphs and tag every node/edge with its status,
   * then lay the union out once. Laying out each side separately would move
   * every node whenever one is inserted, which buries the real change in
   * layout noise.
   */
  function merge(ga, gb, sig) {
    var A = new Map(ga.nodes.map(function (n) { return [n.id, n]; }));
    var B = new Map(gb.nodes.map(function (n) { return [n.id, n]; }));
    var nodes = [];
    var ids = new Set([].concat(ga.nodes.map(function (n) { return n.id; }), gb.nodes.map(function (n) { return n.id; })));
    ids.forEach(function (id) {
      var a = A.get(id);
      var b = B.get(id);
      var status = !a ? 'add' : !b ? 'del' : (sig(a) !== sig(b) ? 'chg' : 'same');
      var node = Object.assign({}, b || a, { status: status, before: a || null, after: b || null });
      nodes.push(node);
    });

    var ekey = function (e) { return JSON.stringify([e.from, e.to]); };
    var EA = new Set(ga.edges.map(ekey));
    var EB = new Set(gb.edges.map(ekey));
    var edges = [];
    var seen = new Set();
    [].concat(ga.edges, gb.edges).forEach(function (e) {
      var k = ekey(e);
      if (seen.has(k)) return;
      seen.add(k);
      edges.push(Object.assign({}, e, { status: !EA.has(k) ? 'add' : !EB.has(k) ? 'del' : 'same' }));
    });

    return { nodes: nodes, edges: edges, counts: countStatus(nodes, edges) };
  }

  function countStatus(nodes, edges) {
    var c = { add: 0, del: 0, chg: 0, same: 0, edgeAdd: 0, edgeDel: 0 };
    nodes.forEach(function (n) { c[n.status]++; });
    edges.forEach(function (e) { if (e.status === 'add') c.edgeAdd++; else if (e.status === 'del') c.edgeDel++; });
    return c;
  }

  // ── layered DAG layout ────────────────────────────────────────────────
  function layout(nodes, edges) {
    var byId = new Map(nodes.map(function (n) { return [n.id, n]; }));
    var out = new Map();
    var indeg = new Map();
    nodes.forEach(function (n) { out.set(n.id, []); indeg.set(n.id, 0); });
    edges.forEach(function (e) {
      if (!byId.has(e.from) || !byId.has(e.to) || e.from === e.to) return;
      out.get(e.from).push(e.to);
      indeg.set(e.to, indeg.get(e.to) + 1);
    });

    // Longest-path ranking over a topological order; nodes left over by a cycle
    // keep rank 0 rather than blocking the layout.
    var rank = new Map();
    nodes.forEach(function (n) { rank.set(n.id, 0); });
    var queue = nodes.filter(function (n) { return indeg.get(n.id) === 0; }).map(function (n) { return n.id; });
    var deg = new Map(indeg);
    var seen = 0;
    while (queue.length) {
      var id = queue.shift();
      seen++;
      out.get(id).forEach(function (to) {
        rank.set(to, Math.max(rank.get(to), rank.get(id) + 1));
        deg.set(to, deg.get(to) - 1);
        if (deg.get(to) === 0) queue.push(to);
      });
    }

    var layers = [];
    nodes.forEach(function (n) {
      var r = rank.get(n.id);
      (layers[r] = layers[r] || []).push(n);
    });

    // Barycenter ordering, a few sweeps, to cut edge crossings.
    var pos = new Map();
    layers.forEach(function (layer) { layer.forEach(function (n, i) { pos.set(n.id, i); }); });
    var preds = new Map(nodes.map(function (n) { return [n.id, []]; }));
    edges.forEach(function (e) { if (preds.has(e.to) && byId.has(e.from)) preds.get(e.to).push(e.from); });
    for (var sweep = 0; sweep < 4; sweep++) {
      for (var r2 = 1; r2 < layers.length; r2++) {
        layers[r2].sort(function (a, b) { return bary(a) - bary(b); });
        layers[r2].forEach(function (n, i) { pos.set(n.id, i); });
      }
    }
    function bary(n) {
      var ps = preds.get(n.id).filter(function (p) { return pos.has(p); });
      if (!ps.length) return pos.get(n.id);
      return ps.reduce(function (s, p) { return s + pos.get(p); }, 0) / ps.length;
    }

    var NW = 176;
    var NH = 40;
    var GX = 74;
    var GY = 16;
    var maxRows = Math.max.apply(null, layers.map(function (l) { return l.length; }).concat([1]));
    layers.forEach(function (layer, r) {
      var h = layer.length * (NH + GY) - GY;
      var top = (maxRows * (NH + GY) - GY - h) / 2;
      layer.forEach(function (n, i) {
        n.x = 24 + r * (NW + GX);
        n.y = 24 + top + i * (NH + GY);
        n.w = NW;
        n.h = NH;
      });
    });

    return {
      width: 48 + layers.length * (NW + GX),
      height: 48 + maxRows * (NH + GY),
      cyclic: seen < nodes.length,
    };
  }

  var STATUS_LABEL = { add: '新增', del: '删除', chg: '属性改变', same: '未变' };

  function drawGraph(g, opts) {
    if (!g.nodes.length) {
      $('graphBody').innerHTML = '<p class="ptx-empty">这个视角下没有可显示的节点。</p>';
      return;
    }
    var box = layout(g.nodes, g.edges);
    var byId = new Map(g.nodes.map(function (n) { return [n.id, n]; }));

    var edgeSvg = g.edges.map(function (e) {
      var a = byId.get(e.from);
      var b = byId.get(e.to);
      if (!a || !b) return '';
      var x1 = a.x + a.w;
      var y1 = a.y + a.h / 2;
      var x2 = b.x;
      var y2 = b.y + b.h / 2;
      var mx = (x1 + x2) / 2;
      return '<path class="ptx-edge ptx-edge--' + e.status + '" d="M' + x1 + ',' + y1
        + ' C' + mx + ',' + y1 + ' ' + mx + ',' + y2 + ' ' + x2 + ',' + y2 + '"/>';
    }).join('');

    var nodeSvg = g.nodes.map(function (n) {
      var sub = nodeSubtitle(n, opts.kind);
      return '<g class="ptx-node ptx-node--' + n.status + '" transform="translate(' + n.x + ',' + n.y + ')"'
        + ' data-line="' + (n.line || '') + '" data-id="' + esc(n.id) + '" tabindex="0">'
        + '<rect width="' + n.w + '" height="' + n.h + '" rx="7"/>'
        + '<text class="ptx-node__label" x="10" y="17">' + esc(trunc(n.label, 22)) + '</text>'
        + '<text class="ptx-node__sub" x="10" y="31">' + esc(trunc(sub, 26)) + '</text>'
        + '<title>' + esc(n.label + '\n' + sub + '\n' + STATUS_LABEL[n.status]
          + (n.line ? '\n源行 ' + n.line : '')) + '</title>'
        + '</g>';
    }).join('');

    var c = g.counts;
    var summary = '<div class="ptx-graphsummary">' + esc(opts.hint)
      + '<span class="ptx-graphstats">节点 ' + g.nodes.length
      + ' · <b class="ptx-add">+' + c.add + '</b> <b class="ptx-del">−' + c.del + '</b>'
      + ' · 属性改变 ' + c.chg + ' · 未变 ' + c.same
      + ' · 边 <b class="ptx-add">+' + c.edgeAdd + '</b> <b class="ptx-del">−' + c.edgeDel + '</b>'
      + (box.cyclic ? ' · 图中存在环' : '')
      + (opts.truncated ? ' · <b class="ptx-warn">节点过多，已截断</b>' : '')
      + '</span></div>';

    $('graphBody').innerHTML = summary
      + '<div class="ptx-canvas" id="canvas"><svg width="' + box.width + '" height="' + box.height + '">'
      + '<g id="viewport">' + edgeSvg + nodeSvg + '</g></svg></div>';

    enablePanZoom($('canvas'));
  }

  /** How much a function has to show under a given lens. */
  function lensWeight(f, lens) {
    if (lens === 'task') return f.tasks.length;
    if (lens === 'memory') return f.allocs.length + f.buffers.length;
    if (lens === 'control') return f.loops.length;
    return f.stmtCount;
  }
  function lensUnit(lens) {
    return { task: ' 任务', memory: ' 缓冲', control: ' 循环' }[lens] || ' 语句';
  }

  function nodeSubtitle(n, kind) {
    if (kind === 'call') return [n.kind, n.level, n.stmts != null ? n.stmts + ' 语句' : ''].filter(Boolean).join(' · ');
    if (kind === 'control') return n.detail || (n.weight != null ? n.weight + ' 语句' : '');
    if (kind === 'dataflow') {
      return [n.op, n.shape ? '[' + n.shape.join('×') + ']' : '', n.dtype, n.space].filter(Boolean).join(' ');
    }
    return [n.type, n.level, n.weight ? n.weight + ' 语句' : ''].filter(Boolean).join(' · ');
  }

  function enablePanZoom(host) {
    var svg = host.querySelector('svg');
    var vp = host.querySelector('#viewport');
    var scale = 1;
    var tx = 0;
    var ty = 0;
    var dragging = false;
    var sx = 0;
    var sy = 0;

    function apply() { vp.setAttribute('transform', 'translate(' + tx + ',' + ty + ') scale(' + scale + ')'); }

    host.addEventListener('wheel', function (e) {
      e.preventDefault();
      var rect = svg.getBoundingClientRect();
      var mx = e.clientX - rect.left;
      var my = e.clientY - rect.top;
      var k = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      var next = Math.min(2.6, Math.max(0.18, scale * k));
      tx = mx - (mx - tx) * (next / scale);
      ty = my - (my - ty) * (next / scale);
      scale = next;
      apply();
    }, { passive: false });

    host.addEventListener('pointerdown', function (e) {
      if (e.target.closest('.ptx-node')) return;
      dragging = true;
      sx = e.clientX - tx;
      sy = e.clientY - ty;
      host.setPointerCapture(e.pointerId);
      host.classList.add('is-dragging');
    });
    host.addEventListener('pointermove', function (e) {
      if (!dragging) return;
      tx = e.clientX - sx;
      ty = e.clientY - sy;
      apply();
    });
    host.addEventListener('pointerup', function () { dragging = false; host.classList.remove('is-dragging'); });

    host.addEventListener('click', function (e) {
      var node = e.target.closest('.ptx-node');
      if (!node) return;
      // In the call lens a node *is* a function, so clicking one opens that
      // function's diff; in the others it maps to a line inside the function
      // already on screen.
      if (state.lens === 'call') {
        var fnName = node.getAttribute('data-id');
        var changed = (pass().changedFunctions || []).some(function (f) { return f.name === fnName; });
        if (!changed) { toast(fnName + ' 在本 Pass 中没有变化'); return; }
        state.fn = fnName;
      }
      state.jumpLine = Number(node.getAttribute('data-line')) || null;
      state.tab = 'diff';
      render();
    });
  }

  // ── memory lens ───────────────────────────────────────────────────────
  function drawMemory(fa, fb, hint) {
    var before = fa ? LIB.memoryView(fa) : [];
    var after = fb ? LIB.memoryView(fb) : [];
    var spaces = [];
    [].concat(before, after).forEach(function (s) { if (spaces.indexOf(s.space) < 0) spaces.push(s.space); });

    if (!spaces.length) {
      $('graphBody').innerHTML = '<p class="ptx-empty">该函数在此阶段还没有具体的内存分配信息。'
        + '内存要到 <code>InitMemRef</code> / <code>AllocateMemoryAddr</code> 之后才会落实。</p>';
      return;
    }

    var totalB = before.reduce(function (s, x) { return s + x.bytes; }, 0);
    var totalA = after.reduce(function (s, x) { return s + x.bytes; }, 0);
    var max = 1;
    spaces.forEach(function (sp) {
      var b = before.find(function (x) { return x.space === sp; });
      var a = after.find(function (x) { return x.space === sp; });
      max = Math.max(max, b ? b.bytes : 0, a ? a.bytes : 0);
    });

    var html = '<div class="ptx-graphsummary">' + esc(hint)
      + '<span class="ptx-graphstats">合计 ' + bytes(totalB) + ' → <b>' + bytes(totalA) + '</b>'
      + (totalA !== totalB ? ' <b class="' + (totalA < totalB ? 'ptx-add' : 'ptx-del') + '">'
        + (totalA < totalB ? '−' : '+') + bytes(Math.abs(totalA - totalB)) + '</b>' : '')
      + '</span></div><div class="ptx-mem">';

    spaces.forEach(function (sp) {
      var b = before.find(function (x) { return x.space === sp; }) || { bytes: 0, items: [] };
      var a = after.find(function (x) { return x.space === sp; }) || { bytes: 0, items: [] };
      var names = new Set();
      b.items.forEach(function (i) { names.add(i.name); });
      a.items.forEach(function (i) { names.add(i.name); });

      html += '<section class="ptx-memspace"><h4>' + esc(sp)
        + '<span>' + bytes(b.bytes) + ' → <b>' + bytes(a.bytes) + '</b>'
        + ' · ' + b.items.length + ' → ' + a.items.length + ' 个缓冲'
        + (a.dynamicCount ? ' · 其中 ' + a.dynamicCount + ' 个为动态 shape，字节数编译期未定' : '')
        + '</span></h4>'
        + '<div class="ptx-membars">'
        + memBar('之前', b, max) + memBar('之后', a, max)
        + '</div><table class="ptx-table ptx-table--mem"><thead><tr><th>缓冲</th><th>之前</th><th>之后</th><th>偏移</th><th></th></tr></thead><tbody>';

      var rows = [];
      names.forEach(function (n) {
        var bi = b.items.find(function (i) { return i.name === n; });
        var ai = a.items.find(function (i) { return i.name === n; });
        rows.push({ name: n, b: bi, a: ai });
      });
      rows.sort(function (x, y) { return ((y.a && y.a.size) || (y.b && y.b.size) || 0) - ((x.a && x.a.size) || (x.b && x.b.size) || 0); });
      rows.slice(0, 40).forEach(function (r) {
        var status = !r.b ? 'add' : !r.a ? 'del' : (r.b.size !== r.a.size ? 'chg' : 'same');
        var cell = function (it) { return !it ? '—' : it.dynamic ? '<i class="ptx-dyn">动态</i>' : bytes(it.size); };
        html += '<tr class="ptx-mem--' + status + '"><td><code>' + esc(r.name) + '</code></td>'
          + '<td>' + cell(r.b) + '</td>'
          + '<td>' + cell(r.a) + '</td>'
          + '<td class="ptx-offsets">' + (r.a && r.a.offsets && r.a.offsets.length
            ? esc(r.a.offsets.slice(0, 6).join(', ')) + (r.a.offsets.length > 6 ? ' …' : '') : '—') + '</td>'
          + '<td><span class="ptx-status ptx-status--' + ({ add: 'added', del: 'removed', chg: 'changed', same: 'same' })[status] + '">'
          + STATUS_LABEL[status] + '</span></td></tr>';
      });
      html += '</tbody></table>';
      if (rows.length > 40) html += '<p class="ptx-more">共 ' + rows.length + ' 个缓冲，仅列出最大的 40 个</p>';
      html += '</section>';
    });

    $('graphBody').innerHTML = html + '</div>';
  }

  function memBar(label, group, max) {
    var segs = group.items.slice(0, 60).map(function (it) {
      var w = (it.size || 0) / max * 100;
      return '<i style="width:' + w.toFixed(3) + '%" title="' + esc(it.name + ' · ' + bytes(it.size)) + '"></i>';
    }).join('');
    return '<div class="ptx-membar"><span>' + label + '</span><div class="ptx-membar__track">' + segs + '</div>'
      + '<b>' + bytes(group.bytes) + '</b></div>';
  }

  // ══════════════════════════════════════════════════════════════════════
  // Pass documentation
  // ══════════════════════════════════════════════════════════════════════

  var docsLoading = false;

  function renderDoc() {
    var pane = $('docPane');
    pane.hidden = !state.docOpen;
    if (!state.docOpen) return;

    var p = pass();
    if (!p.doc) {
      $('docOrigin').textContent = '';
      $('docBody').innerHTML = '<p class="ptx-empty">本仓库的 <code>repo/pto</code> 镜像里没有这个 Pass 的文档。'
        + '左侧的实测证据仍然完全可用——它直接来自两份快照的比对。</p>';
      return;
    }
    if (!window.PTX_DOCS) {
      if (!docsLoading) {
        docsLoading = true;
        var s = document.createElement('script');
        s.src = 'data/docs.js';
        s.onload = function () { docsLoading = false; renderDoc(); };
        s.onerror = function () { docsLoading = false; $('docBody').innerHTML = '<p class="ptx-empty">文档数据加载失败。</p>'; };
        document.head.appendChild(s);
      }
      $('docBody').innerHTML = '<p class="ptx-loading">正在加载 Pass 文档…</p>';
      return;
    }

    var doc = window.PTX_DOCS[p.doc];
    if (!doc) { $('docBody').innerHTML = '<p class="ptx-empty">未找到该 Pass 的文档。</p>'; return; }

    $('docOrigin').innerHTML = window.PTX_STANDALONE
      ? '来自 <span class="is-inert" title="仓库内路径：' + esc(doc.file) + '">' + esc(doc.file.split('/').pop()) + '</span>'
      : '来自 <a href="../../' + esc(doc.file) + '" target="_blank" rel="noreferrer">' + esc(doc.file.split('/').pop()) + '</a>';

    var html = '<h2>' + esc(doc.title) + '</h2>';
    if (doc.tagline) html += '<p class="ptx-doc__tagline">' + mdInline(doc.tagline) + '</p>';
    if (doc.timing) html += '<p class="ptx-doc__timing"><b>使用时机</b>' + mdInline(doc.timing) + '</p>';
    (doc.blocks || []).forEach(function (b, i) {
      html += '<details' + (i === 0 ? ' open' : '') + '><summary>' + esc(b.heading) + '</summary>'
        + md(b.body) + '</details>';
    });
    $('docBody').innerHTML = html;
  }

  // ══════════════════════════════════════════════════════════════════════
  // Wiring
  // ══════════════════════════════════════════════════════════════════════

  function render() {
    renderRail();
    renderHeader();
    if (state.tab === 'overview') renderOverview();
    if (state.tab === 'diff') renderDiff();
    if (state.tab === 'graph') renderGraph();
    renderDoc();
    writeHash();
  }

  function selectPass(idx) {
    var r = run();
    idx = Math.max(0, Math.min(r.passes.length - 1, idx));
    if (idx === state.passIdx) return;
    state.passIdx = idx;
    state.fn = null;
    state.lens = r.passes[idx].lens;
    render();
  }

  function writeHash() {
    var h = '#' + state.runId + '/' + state.passIdx + '/' + state.tab
      + (state.fn ? '/' + encodeURIComponent(state.fn) : '');
    if (location.hash !== h) history.replaceState(null, '', h);
  }

  function readHash() {
    var parts = location.hash.replace(/^#/, '').split('/');
    if (!parts[0]) return;
    if (INDEX.runs.some(function (r) { return r.id === parts[0]; })) state.runId = parts[0];
    var idx = Number(parts[1]);
    if (!Number.isNaN(idx)) state.passIdx = idx;
    if (['overview', 'diff', 'graph'].indexOf(parts[2]) >= 0) state.tab = parts[2];
    if (parts[3]) state.fn = decodeURIComponent(parts[3]);
  }

  function boot() {
    $('runSelect').innerHTML = INDEX.runs.map(function (r) {
      return '<option value="' + r.id + '">' + esc(r.title) + '</option>';
    }).join('');

    readHash();
    $('runSelect').value = state.runId;
    state.lens = state.lens || pass().lens;
    updateRunMeta();

    $('runSelect').addEventListener('change', function (e) {
      state.runId = e.target.value;
      state.passIdx = Math.min(state.passIdx, run().passes.length - 1);
      state.fn = null;
      updateRunMeta();
      render();
    });

    $('passList').addEventListener('click', function (e) {
      var b = e.target.closest('.ptx-pass');
      if (b) selectPass(Number(b.dataset.idx));
    });

    $('onlyChanged').addEventListener('change', function (e) {
      state.onlyChanged = e.target.checked;
      renderRail();
    });
    $('passFilter').addEventListener('input', function (e) {
      state.filter = e.target.value.trim();
      renderRail();
    });

    document.querySelector('.ptx-tabs').addEventListener('click', function (e) {
      var b = e.target.closest('.ptx-tab');
      if (!b) return;
      state.tab = b.dataset.tab;
      render();
    });

    $('viewOverview').addEventListener('click', function (e) {
      var b = e.target.closest('[data-openfn]');
      if (!b) return;
      state.fn = b.dataset.openfn;
      state.tab = 'diff';
      render();
    });

    $('diffFnChips').addEventListener('click', function (e) {
      var b = e.target.closest('.ptx-chip');
      if (!b) return;
      state.fn = b.dataset.fn;
      renderDiff();
      writeHash();
    });

    $('diffMode').addEventListener('click', function (e) {
      var b = e.target.closest('button');
      if (!b) return;
      state.diffMode = b.dataset.mode;
      $('diffMode').querySelectorAll('button').forEach(function (x) { x.classList.toggle('is-active', x === b); });
      renderDiff();
    });

    $('lensPicker').addEventListener('click', function (e) {
      var b = e.target.closest('button');
      if (!b) return;
      state.lens = b.dataset.lens;
      renderGraph();
      writeHash();
    });

    $('graphFn').addEventListener('change', function (e) {
      state.fn = e.target.value;
      renderGraph();
      writeHash();
    });

    $('prevPass').addEventListener('click', function () { selectPass(state.passIdx - 1); });
    $('nextPass').addEventListener('click', function () { selectPass(state.passIdx + 1); });

    $('docToggle').addEventListener('click', function () {
      state.docOpen = !state.docOpen;
      document.body.classList.toggle('ptx--nodoc', !state.docOpen);
      renderDoc();
    });

    $('themeToggle').addEventListener('click', function () {
      var root = document.documentElement;
      root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    });

    document.addEventListener('keydown', function (e) {
      if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
      if (e.key === 'ArrowLeft') { selectPass(state.passIdx - 1); e.preventDefault(); }
      if (e.key === 'ArrowRight') { selectPass(state.passIdx + 1); e.preventDefault(); }
      if (e.key === '1') { state.tab = 'overview'; render(); }
      if (e.key === '2') { state.tab = 'diff'; render(); }
      if (e.key === '3') { state.tab = 'graph'; render(); }
    });

    render();
  }

  function updateRunMeta() {
    var r = run();
    var last = r.passes[r.passes.length - 1];
    $('runMeta').innerHTML = esc(r.subtitle) + ' · <code>' + esc(r.dir) + '</code> · '
      + (r.passes.length - 1) + ' 个 Pass · 终态 ' + last.functions.length + ' 个函数';
  }

  boot();
})();
