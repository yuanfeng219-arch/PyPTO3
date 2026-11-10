/* CSA Operator Lens · 从子模块源码扫描 Kernel，补齐 L1 每个计算 / 通信模块的 L2。
 * 规则：在模块调用的函数体内找 pl.spmd(...) / pl.at(...) 派发语句，读出 name_hint、派发数表达式、deps、as 捕获的 TaskId。
 * 只读源码快照，不推断运行规模；Lane 视图只给手工建模过的 3 个 Kernel（qk_pv / KV Writeback / Merge · Publish）。
 * 运行在 lens-data.js 之后，原地补充 LensData.KERNELS / CHAINS / TAGS 与 MODULES[].kernels。 */
(function (root) {
  'use strict';
  const D = root.LensData, SRC = root.LENS_SOURCE;
  if (!D || !SRC) return;

  // 每个模块调用的函数（按 decode_csa.py 里的调用顺序）
  const MODULE_FNS = {
    pre: [['hc_pre.py', 'hc_pre'], ['rmsnorm.py', 'rms_norm']],
    gather: [['decode_cp_token_allgather.py', 'decode_cp_token_allgather_step']],
    q: [['qkv_proj_rope.py', 'rope_prepare'], ['qkv_proj_rope.py', 'q_proj_rope']],
    kv: [['qkv_proj_rope.py', 'rope_prepare'], ['qkv_proj_rope.py', 'kv_proj_rope']],
    main: [['decode_compressor_ratio4.py', 'compressor_ratio4']],
    idxcomp: [['decode_indexer_compressor.py', 'indexer_compressor']],
    indexer: [['decode_indexer.py', 'indexer'], ['decode_indexer.py', 'indexer_score_topk_forest']],
    a2a: [['decode_o_proj.py', 'o_group_a2a']],
    oproj: [['decode_o_proj.py', 'o_proj_reduce_scatter']],
    post: [['hc_post.py', 'hc_post']]
  };
  // pass 36（36_after_AllocateMemoryAddr.py）里出现的 Kernel 名：用来标“编译产物中可见”
  const PASS36 = ('comb_sinkhorn compress_state_commit cp_token_allgather_payload_wait cp_token_allgather_push cp_token_allgather_readback cp_token_allgather_readback_wait '
    + 'cp_token_allgather_retire csa_cache_writeback csa_merge_pack_publish csa_rope_interleave csa_slots_build_valid_qk_plan indexer_topk_group_wave indexer_topk_query_merge '
    + 'hc_post hc_pre_linear_reduce hc_pre_linear hc_pre_rms idx_qr_proj_dequant idx_qr_proj_matmul indexer_score_leaf_wave kv_and_cache_write kv_hadamard kv_proj_matmul '
    + 'kv_proj_seed kv_rms_norm_rope kv_score_proj kv_touch mix_x o_group_a2a_complete o_group_a2a_gather o_group_a2a_wait q_rope_prepare qk_pv qproj_dequant_rms_nope_rope '
    + 'qproj_matmul qr_hadamard_matmul qr_hadamard_quant qr_proj_matmul qr_proj_seed qr_rms_norm_quant qr_rope qr_rope_swap_idx rms_norm rmsnorm_rope_cache_write rmsnorm_rope '
    + 'rope_cs scatter_softmax_pool split_pre_post tp_o_a_quant tp_o_a tp_o_b_dequant tp_o_b_publish tp_o_b tp_o_rs_complete tp_o_rs_reduce tp_o_rs_wait weights_proj_reduce weights_proj').split(' ');

  const linesCache = {};
  const lines = function (file) { return linesCache[file] || (linesCache[file] = SRC.files[file].text.split('\n')); };
  const indent = function (t) { return t.match(/^ */)[0].length; };

  function fnRange(file, name) {
    const L = lines(file), re = new RegExp('^def ' + name + '\\(');
    let a = -1;
    for (let i = 0; i < L.length; i++) if (re.test(L[i])) { a = i; break; }
    if (a < 0) return null;
    let b = L.length - 1;
    for (let i = a + 1; i < L.length; i++) if (/^(@|def |class )/.test(L[i])) { b = i - 1; break; }
    while (b > a && !L[b].trim()) b--;
    return [a + 1, b + 1];
  }
  // 从 pl.spmd( / pl.at( 起，按括号配平取出整个调用文本（可能跨行）
  function callText(L, i, col) {
    let depth = 0, out = '', end = i;
    for (let r = i; r < L.length; r++) {
      const s = r === i ? L[r].slice(col) : L[r];
      for (let k = 0; k < s.length; k++) {
        const ch = s[k];
        out += ch;
        if (ch === '(' || ch === '[') depth++;
        else if (ch === ')' || ch === ']') { depth--; if (depth === 0) return { text: out, end: r, rest: s.slice(k + 1) }; }
      }
      out += ' ';
      end = r;
    }
    return { text: out, end: end, rest: '' };
  }
  function firstArg(text) {
    const body = text.slice(text.indexOf('(') + 1);
    let depth = 0;
    for (let k = 0; k < body.length; k++) {
      const ch = body[k];
      if (ch === '(' || ch === '[') depth++;
      else if (ch === ')' || ch === ']') { if (depth === 0) return body.slice(0, k).trim(); depth--; }
      else if (ch === ',' && depth === 0) return body.slice(0, k).trim();
    }
    return body.trim();
  }
  // deps=[...]：按括号配平取列表，避免吞掉后面的 optimizations=[...] 等参数
  function depsOf(text) {
    const at = text.indexOf('deps=[');
    if (at < 0) return [];
    let depth = 0, body = '';
    for (let k = at + 5; k < text.length; k++) {
      const ch = text[k];
      if (ch === '[' || ch === '(') { depth++; if (depth === 1) continue; }
      else if (ch === ']' || ch === ')') { depth--; if (depth === 0) break; }
      body += ch;
    }
    return body.split(',').map(function (x) { return x.trim(); }).filter(Boolean);
  }
  function scan(mod) {
    const out = [];
    MODULE_FNS[mod].forEach(function (ref) {
      const file = ref[0], fn = fnRange(file, ref[1]);
      if (!fn) return;
      const L = lines(file);
      for (let i = fn[0] - 1; i < fn[1]; i++) {
        const m = /pl\.(spmd|at)\(/.exec(L[i]);
        if (!m || /^\s*#/.test(L[i])) continue;
        const call = callText(L, i, m.index);
        const hint = /name_hint="([^"]+)"/.exec(call.text);
        if (!hint) continue;
        const asTid = /^\s*as\s+(\w+)/.exec(call.rest);
        const deps = depsOf(call.text);
        // Kernel 体：派发语句之后、缩进回到派发行或更浅之前
        const base = indent(L[i]);
        let j = call.end + 1;
        while (j < fn[1] && (!L[j].trim() || indent(L[j]) > base)) j++;
        while (j - 1 > call.end && !L[j - 1].trim()) j--;
        out.push({
          name: hint[1], file: file, fnName: ref[1], fn: fn, line: i + 1, range: [i + 1, Math.max(j, call.end + 1)],
          dispatch: m[1], loop: /^\s*for\s+\w+\s+in\s+pl\.spmd/.test(L[i]),
          expr: m[1] === 'spmd' ? firstArg(call.text) : null,
          level: (/level=pl\.Level\.(\w+)/.exec(call.text) || [])[1] || null,
          deps: deps, produces: asTid && asTid[1] !== '_' ? asTid[1] : null
        });
        i = call.end;
      }
    });
    return out;
  }
  const short = function (s, n) { s = String(s).replace(/\s+/g, ' '); return s.length > n ? s.slice(0, n - 1) + '…' : s; };

  Object.keys(MODULE_FNS).forEach(function (mod) {
    const found = scan(mod);
    if (!found.length) return;
    const ids = [];
    found.forEach(function (k) {
      const id = mod + '.' + k.name;
      if (D.KERNELS[id]) return;
      ids.push(id);
      const what = k.dispatch === 'spmd'
        ? (k.loop ? 'for … in pl.spmd(…)：循环体被外提成一个 InCore Kernel，按派发数生成逻辑 block，每个 block 用循环变量取自己的那份。' : 'with pl.spmd(…)：块内联体被外提成一个 InCore Kernel，各 block 用 get_block_idx() 分工。')
        : 'pl.at(' + (k.level || 'CORE_GROUP') + ')：整段作为一个任务在一个 core group 上执行，不做 SPMD 拆分。';
      D.KERNELS[id] = {
        id: id, name: k.name, module: mod, file: k.file, line: k.line, range: k.range, fn: k.fn, fnName: k.fnName,
        dispatch: k.dispatch, lanes: k.expr, laneExpr: k.expr, produces: k.produces, deps: k.deps, auto: true,
        compiled: PASS36.indexOf(k.name) >= 0, what: what,
        loops: k.dispatch === 'at' ? '函数 ' + k.fnName + '()，' + k.file + ' ' + k.range[0] + '–' + k.range[1] + ' 行。' : null
      };
      const tags = D.TAGS[k.file] = D.TAGS[k.file] || {};
      if (!tags[k.line]) tags[k.line] = [k.dispatch === 'spmd' ? 'SPMD × ' + short(k.expr, 22) : 'AT · ' + (k.level || 'CORE_GROUP'), 'fw',
        k.name + '：' + (k.dispatch === 'spmd' ? '派发数 ' + k.expr : '单任务') + (k.deps.length ? '；等待 ' + k.deps.join(', ') : '') + (k.produces ? '；产生 ' + k.produces : '')];
    });
    const produced = {};
    ids.forEach(function (id) { const K = D.KERNELS[id]; if (K.produces) produced[K.produces] = id; });
    const edges = [], external = [];
    ids.forEach(function (id) {
      D.KERNELS[id].deps.forEach(function (d) {
        if (produced[d]) edges.push([produced[d], id, d]);
        else external.push([d, id]);
      });
    });
    D.CHAINS[mod] = { order: ids, edges: edges, external: external, auto: true };
    const m = D.MODULES.filter(function (x) { return x.id === mod; })[0];
    if (m) m.kernels = ids;
  });

  // 手工建模的 Kernel 也补上函数范围，L2 源码栏按所选 Kernel 的函数显示
  Object.keys(D.KERNELS).forEach(function (id) {
    const K = D.KERNELS[id];
    if (!K.fn) K.fn = K.file === D.FILES.SPARSE ? [105, 358] : [151, 499];
    if (K.compiled === undefined) K.compiled = PASS36.indexOf(K.name) >= 0;
  });

  // ── Kernel 之间的数据依赖：从源码里张量的读写推出 ──
  // 张量 = 函数参数 + 函数体里 pl.create_tensor / pl.reshape 出来的名字（Kernel 内部的 Tile 不跨 Kernel，不算）。
  // 写：X[...] = …、pl.write(X, …)、pl.store(…, X)、pld.tensor.put(dst=X, …)；读：Kernel 体里出现、但不是写目标的张量名。
  // A → B：B 读到的张量，在链里最近一次由排在它前面的 A 写过。只是“谁写谁读”，不推断耗时或重叠。
  const tensorCache = {};
  function fnTensors(file, fn) {
    const key = file + ':' + fn[0];
    if (tensorCache[key]) return tensorCache[key];
    const L = lines(file), set = {};
    let inSig = true;
    for (let i = fn[0] - 1; i < fn[1]; i++) {
      const t = L[i];
      if (inSig) { const p = /^\s+(\w+)\s*:/.exec(t); if (p) set[p[1]] = 1; if (/\)\s*(->.*)?:\s*$/.test(t)) inSig = false; continue; }
      const a = /^\s*(\w+)\s*=\s*pl\.(create_tensor|reshape)\(/.exec(t);
      if (a) set[a[1]] = 1;
    }
    return (tensorCache[key] = set);
  }
  function lastArg(text) {
    let depth = 0, cut = 0;
    for (let k = 0; k < text.length; k++) {
      const ch = text[k];
      if (ch === '(' || ch === '[') depth++;
      else if (ch === ')' || ch === ']') depth--;
      else if (ch === ',' && depth === 1) cut = k + 1;
    }
    const m = /^\s*(\w+)/.exec(text.slice(cut));
    return m ? m[1] : null;
  }
  function flowOf(K) {
    const L = lines(K.file), T = fnTensors(K.file, K.fn), reads = {}, writes = {};
    const a = K.range[0] - 1, b = K.range[1];
    for (let i = a; i < b; i++) {
      let t = L[i].replace(/#.*$/, '');
      const sub = /^\s*(\w+)\s*\[[^\n]*?\]\s*=(?!=)/.exec(t);
      if (sub && T[sub[1]]) { writes[sub[1]] = 1; t = t.slice(sub[0].length); }
      [/pl\.write\(\s*(\w+)/g, /dst\s*=\s*(\w+)/g].forEach(function (re) {
        let m; while ((m = re.exec(t))) if (T[m[1]]) { writes[m[1]] = 1; t = t.slice(0, m.index) + t.slice(m.index + m[0].length); re.lastIndex = 0; }
      });
      const st = t.indexOf('pl.store(');
      if (st >= 0) { const c = callText(L, i, L[i].indexOf('pl.store(')); const w = lastArg(c.text); if (w && T[w]) writes[w] = 1; }
      (t.match(/[A-Za-z_]\w*/g) || []).forEach(function (n) { if (T[n] && !(sub && n === sub[1])) reads[n] = 1; });
    }
    Object.keys(writes).forEach(function (n) { if (reads[n] && !new RegExp('(^|[^\\w])' + n + '\\s*\\[[^\\]]*\\]\\s*[^=\\s]').test(L.slice(a, b).join('\n'))) delete reads[n]; });
    return { reads: Object.keys(reads), writes: Object.keys(writes) };
  }
  Object.keys(D.CHAINS).forEach(function (mod) {
    const ch = D.CHAINS[mod], flows = {};
    ch.order.forEach(function (id) { flows[id] = flowOf(D.KERNELS[id]); D.KERNELS[id].flow = flows[id]; });
    const pairs = {}, control = {};
    ch.edges.forEach(function (e) { control[e[0] + '>' + e[1]] = 1; });
    ch.order.forEach(function (id, j) {
      flows[id].reads.forEach(function (tn) {
        for (let i = j - 1; i >= 0; i--) {
          const src = ch.order[i];
          if (flows[src].writes.indexOf(tn) >= 0) { (pairs[src + '>' + id] = pairs[src + '>' + id] || [src, id, []])[2].push(tn); break; }
        }
      });
    });
    ch.flow = Object.keys(pairs).filter(function (k) { return !control[k]; }).map(function (k) { return pairs[k]; });
  });
})(window);
