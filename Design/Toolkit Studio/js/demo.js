(function () {
  const recipes = [
    { id: 'prefill', label: 'Prefill', meta: 'dense attention' },
    { id: 'decode', label: 'Decode', meta: 'single token' },
    { id: 'decode_layer', label: 'Decode Layer', meta: 'selected · Qwen3-14B' },
    { id: 'rmsrope', label: 'RMSNorm + RoPE', meta: 'fused recipe' },
    { id: 'moe', label: 'MoE Expert', meta: 'grouped GEMM' },
    { id: 'lm_head', label: 'LM Head', meta: 'vocab parallel' }
  ];
  const passes = ['Semantic Lowering', 'Layout Planning', 'Parallel Mapping', 'Memory Scheduling', 'ISA Emission'];
  const guards = ['Op legality', 'Dependencies', 'Manual scope', 'Liveness', 'Paged layout', 'Index width', 'ISA capacity', 'FP32 carry'];
  const state = { step: 0, workflowStep: 0, activityView: 'explorer', editorTab: 'source', activeFile: 'decode_layer.py', hardwareFlowLine: 0, hardwareFlowPinned: false, productMode: 'ide', selectedRecipe: 'decode_layer', fixed: false, compiled: false, verified: false, soloFollow: true, soloRunning: false, soloPaused: false, soloComplete: false, soloStep: -1, soloTool: 'context', currentRun: 'run_8f2c', runActionTab: 'cmd', selectedEvidence: 'tensor', intentTab: 'shape', passesGraphMode: 'single', rmsNormFunction: 'input', rmsNormTab: 'overview', rmsNormFlowStep: 'load', attentionTab: 'overview', attentionFocus: 'position', qwenDecodeTab: 'overview', qwenDecodeFocus: 'scope1', pagedAttentionTab: 'overview', pagedAttentionFocus: 'paging', pagedAttentionOverlay: 'precision', pagedAttentionExpandedNode: null, sourceCache: {} };
  const EXPLORER_STEP = 1;
  const WORKFLOW_STEPS = [0, 2, 3, 4];
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  function setEditorTab(tab) {
    state.editorTab = tab;
    $$('[data-editor-tab]').forEach((button) => button.classList.toggle('is-active', button.dataset.editorTab === tab));
    $$('[data-editor-panel]').forEach((panel) => { panel.hidden = panel.dataset.editorPanel !== tab; });
  }

  const intentSourceLines = { 176: 'layout', 223: 'resource', 305: 'shape', 410: 'scope', 732: 'deps' };
  const matmulSource = `@pl.jit.incore
def mm(
    a: pl.Tensor[[32, 32], pl.FP16],
    b: pl.Tensor[[32, 32], pl.FP16],
    out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
):
    a_l1 = pl.load(a, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
    b_l1 = pl.load(b, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
    a_l0a = pl.move(a_l1, target_memory=pl.Mem.Left)
    b_l0b = pl.move(b_l1, target_memory=pl.Mem.Right)
    c_acc = pl.matmul(a_l0a, b_l0b)      # 落在 Acc
    pl.store(c_acc, [0, 0], out)         # Acc -> DDR
    return out`;
  const matmulHardwarePreset = {
    id: 'matmul-aic-ddr',
    name: 'Matmul AIC + DDR Memory Path',
    rails: [{
      key: 'DDR',
      label: 'DDR',
      tone: 'memory-shell',
      grid: { rows: 24, cols: 4, cellSize: 12, gap: 4, shape: 'hex' },
    }],
    cores: [{
      id: 'matmul-aic-core',
      kind: 'aic',
      title: 'AIC',
      presetKey: 'aicDraftV1',
    }],
    routes: [
      {
        id: 'matmul-load-a',
        label: 'load A / B',
        tone: 'transport',
        from: '[data-mem950-node="rail:DDR"]',
        to: '#matmul-aic-core [data-aic-node="buffer:L1"]',
        fromSide: 'right',
        toSide: 'left',
        style: 'lane-h-target',
        labelDy: -18,
      },
      {
        id: 'matmul-store-out',
        label: 'store out',
        tone: 'directReturn',
        from: '#matmul-aic-core [data-aic-node="buffer:L0C"]',
        to: '[data-mem950-node="rail:DDR"]',
        fromSide: 'left',
        toSide: 'right',
        style: 'lane-h-source',
        labelDy: 18,
      },
    ],
    hoverTips: {
      'rail:DDR': { title: 'DDR', body: '算子输入 A、B 的来源，以及 FP32 输出 out 的写回位置。' },
      'core:AIC': { title: 'AIC', body: '完成 Mat、Left、Right、Acc 层级中的矩阵乘与累加。' },
    },
  };
  const rmsNormHardwarePreset = {
    id: 'rmsnorm-aiv-ddr',
    name: 'RMSNorm Ascend AIV Data Path',
    rails: [{
      key: 'DDR',
      label: 'DDR / GM',
      tone: 'memory-shell',
      grid: { rows: 24, cols: 4, cellSize: 12, gap: 4, shape: 'hex' },
    }],
    cores: [{
      id: 'rmsnorm-aiv-core',
      kind: 'aiv',
      title: 'AIV',
      presetKey: 'aivOfficialV1',
    }],
    routes: [
      {
        id: 'rmsnorm-load',
        label: 'MTE2 · load',
        tone: 'transport',
        from: '[data-mem950-node="rail:DDR"]',
        to: '#rmsnorm-aiv-core [data-aiv-node="cache:ND-DMA Cache"]',
        fromSide: 'right',
        toSide: 'left',
        style: 'lane-h-target',
        labelDy: -17,
      },
      {
        id: 'rmsnorm-store',
        label: 'MTE3 · store',
        tone: 'directReturn',
        from: '#rmsnorm-aiv-core [data-aiv-node="buffer:UB"]',
        to: '[data-mem950-node="rail:DDR"]',
        fromSide: 'left',
        toSide: 'right',
        style: 'lane-h-source',
        labelDy: 17,
      },
    ],
    hoverTips: {
      'rail:DDR': { title: 'DDR / GM', body: '承载输入、Gamma 与 BF16 输出；两遍 RMSNorm 会重复读取输入。' },
      'core:AIV': { title: 'AIV', body: 'RMSNorm 的 cast、逐元素计算、行归约与归一化在 Vector 路径完成。' },
      'cache:ND-DMA Cache': { title: 'ND-DMA Cache', body: 'MTE2 从 GM 搬入当前 Chunk，进入 UB 前经过片上搬运路径。' },
      'buffer:UB': { title: 'Unified Buffer', body: '保存当前输入 Chunk、FP32 部分和、Gamma 与待写回输出。' },
      'vector:Vector': { title: 'Vector', body: '执行 cast、square、row_sum、sqrt、recip 与 expand_mul。' },
    },
  };
  const matmulLineFlows = {
    1: { label: 'JIT in-core：算子将在 AIC 内执行', selectors: ['#matmul-aic-core'] },
    2: { label: 'mm 契约：DDR 张量进入 AIC 计算', selectors: ['[data-mem950-node="rail:DDR"]', '#matmul-aic-core'], routes: ['matmul-load-a'] },
    3: { label: '输入 a：FP16 张量位于 DDR', selectors: ['[data-mem950-node="rail:DDR"]'] },
    4: { label: '输入 b：FP16 张量位于 DDR', selectors: ['[data-mem950-node="rail:DDR"]'] },
    5: { label: '输出 out：FP32 张量写回 DDR', selectors: ['[data-mem950-node="rail:DDR"]'] },
    6: { label: '签名完成：建立 DDR ⇄ AIC 数据边界', selectors: ['[data-mem950-node="rail:DDR"]', '#matmul-aic-core'], routes: ['matmul-load-a', 'matmul-store-out'] },
    7: { label: 'load a：DDR → L1（Mat）', routes: ['matmul-load-a'] },
    8: { label: 'load b：DDR → L1（Mat）', routes: ['matmul-load-a'] },
    9: { label: 'move a：L1 → L0A（Left）', selectors: ['#matmul-aic-core [data-aic-node="buffer:L1"]', '#matmul-aic-core [data-aic-node="buffer:L0A"]'] },
    10: { label: 'move b：L1 → L0B（Right）', selectors: ['#matmul-aic-core [data-aic-node="buffer:L1"]', '#matmul-aic-core [data-aic-node="buffer:L0B"]'] },
    11: { label: 'matmul：L0A + L0B → CUBE → L0C（Acc）', selectors: ['#matmul-aic-core [data-aic-node="buffer:L0A"]', '#matmul-aic-core [data-aic-node="buffer:L0B"]', '#matmul-aic-core [data-aic-node="cube:CUBE"]', '#matmul-aic-core [data-aic-node="buffer:L0C"]'] },
    12: { label: 'store：L0C（Acc）→ DDR', routes: ['matmul-store-out'] },
    13: { label: 'return out：结果驻留 DDR', selectors: ['[data-mem950-node="rail:DDR"]'] },
  };
  let matmulHardwareGraphInstance = null;
  let rmsNormHardwareGraphInstance = null;
  let attentionGraphController = null;
  let qwenDecodeGraphController = null;
  let pagedAttentionGraphController = null;
  let passesGraphInstance = null;

  // Minimal Python syntax highlighter — stateful across lines so triple-quoted
  // docstrings that span multiple rows stay a single string token. Returns one
  // HTML string per source line (token spans styled by `.kf-code .tok-*`).
  const PY_KEYWORDS = new Set([
    'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def',
    'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if',
    'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise',
    'return', 'try', 'while', 'with', 'yield', 'match', 'case'
  ]);

  function highlightPythonLines(source) {
    const esc = (s) => s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
    const span = (cls, s) => `<span class="${cls}">${esc(s)}</span>`;
    const lines = source.split('\n');
    const out = [];
    let triple = null; // active multi-line string delimiter: '"""' or "'''"

    for (let li = 0; li < lines.length; li++) {
      const line = lines[li];
      let html = '';
      let i = 0;

      if (triple) {
        const close = line.indexOf(triple);
        if (close === -1) { out.push(span('tok-str', line)); continue; }
        html += span('tok-str', line.slice(0, close + 3));
        i = close + 3;
        triple = null;
      }

      let expectName = null; // 'func' | 'class' — next identifier is a definition name
      let plain = '';
      const flush = () => { if (plain) { html += esc(plain); plain = ''; } };

      while (i < line.length) {
        const rest = line.slice(i);
        const ch = line[i];

        // comment
        if (ch === '#') { flush(); html += span('tok-com', line.slice(i)); i = line.length; break; }

        // string (optional r/b/u/f prefix)
        const sm = /^([rRbBuUfF]{0,2})("""|'''|"|')/.exec(rest);
        if (sm) {
          flush();
          const q = sm[2];
          const startQuote = i + sm[1].length;
          if (q.length === 3) {
            const after = line.indexOf(q, startQuote + 3);
            if (after === -1) { html += span('tok-str', line.slice(i)); triple = q; i = line.length; break; }
            html += span('tok-str', line.slice(i, after + 3));
            i = after + 3;
          } else {
            let j = startQuote + 1;
            while (j < line.length) {
              if (line[j] === '\\') { j += 2; continue; }
              if (line[j] === q) { j++; break; }
              j++;
            }
            html += span('tok-str', line.slice(i, j));
            i = j;
          }
          continue;
        }

        // number
        const nm = /^(0[xXoObB][0-9a-fA-F_]+|(?:\d[\d_]*\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[jJ]?)/.exec(rest);
        if (nm && /\d/.test(nm[0])) { flush(); html += span('tok-num', nm[0]); i += nm[0].length; continue; }

        // decorator (only at the visual start of a line)
        if (ch === '@') {
          const dm = /^@[A-Za-z_][\w.]*/.exec(rest);
          if (dm && line.slice(0, i).trim() === '') { flush(); html += span('tok-dec', dm[0]); i += dm[0].length; continue; }
        }

        // identifier / keyword
        const im = /^[A-Za-z_]\w*/.exec(rest);
        if (im) {
          const word = im[0];
          flush();
          let cls = null;
          if (expectName) { cls = expectName === 'class' ? 'tok-cls' : 'tok-fn'; expectName = null; }
          else if (word === 'def') { cls = 'tok-kw'; expectName = 'func'; }
          else if (word === 'class') { cls = 'tok-kw'; expectName = 'class'; }
          else if (PY_KEYWORDS.has(word)) cls = 'tok-kw';
          else if (word === 'True' || word === 'False' || word === 'None') cls = 'tok-const';
          else if (word === 'self' || word === 'cls') cls = 'tok-self';
          else if (/^\s*\(/.test(line.slice(i + word.length))) cls = 'tok-fn';
          if (cls) html += span(cls, word); else plain += word;
          i += word.length;
          continue;
        }

        plain += ch;
        i++;
      }
      flush();
      out.push(html);
    }
    return out;
  }

  function resolveSource(file) {
    const passes = window.PTO_PASSES_DUMP_SOURCES;
    if (passes && Object.prototype.hasOwnProperty.call(passes, file)) return passes[file];
    if (file === 'matmul.py') return matmulSource;
    return state.sourceCache[file] || window.PTO_DECODE_LAYER_SOURCE || '';
  }

  async function loadSource(file) {
    if (state.sourceCache[file]) return state.sourceCache[file];
    const sourceFile = isPagedAttentionFile(file) ? PAGED_ATTENTION_FILE : file;
    if (state.sourceCache[sourceFile]) {
      state.sourceCache[file] = state.sourceCache[sourceFile];
      return state.sourceCache[file];
    }
    const bundled = window.PTO_EXAMPLES_SOURCES?.[sourceFile];
    if (bundled && sourceFile !== PAGED_ATTENTION_FILE) {
      state.sourceCache[file] = bundled;
      return bundled;
    }
    if (!sourceFile.startsWith('examples/')) return resolveSource(sourceFile);
    try {
      const url = new URL(`../../repo/pto/${sourceFile}`, document.baseURI);
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Unable to load ${sourceFile}: ${response.status}`);
      state.sourceCache[sourceFile] = await response.text();
    } catch (error) {
      // Direct file previews and packaged demos cannot fetch outside the page tree.
      // Use the bundled snapshot so both tree entries still open with complete source.
      if (!bundled) throw error;
      state.sourceCache[sourceFile] = bundled;
    }
    state.sourceCache[file] = state.sourceCache[sourceFile];
    return state.sourceCache[file];
  }

  function isPassesDumpFile(file) {
    const passes = window.PTO_PASSES_DUMP_SOURCES;
    return !!(passes && Object.prototype.hasOwnProperty.call(passes, file));
  }

  function renderFullSource() {
    const isPasses = isPassesDumpFile(state.activeFile);
    const source = resolveSource(state.activeFile);
    const editor = $('#dslEditor');
    const highlighted = highlightPythonLines(source);
    const fragment = document.createDocumentFragment();
    highlighted.forEach((lineHtml, index) => {
      const lineNumber = index + 1;
      const row = document.createElement('div');
      const gutter = document.createElement('i');
      const code = document.createElement('code');
      gutter.textContent = lineNumber;
      code.innerHTML = lineHtml || ' ';
      if (!isPasses && intentSourceLines[lineNumber]) row.dataset.intentLine = intentSourceLines[lineNumber];
      if (isPasses) { row.dataset.passesLine = String(lineNumber); row.tabIndex = 0; }
      if (state.activeFile === 'matmul.py') {
        row.dataset.hardwareLine = String(lineNumber);
        row.tabIndex = 0;
        row.title = matmulLineFlows[lineNumber]?.label || `第 ${lineNumber} 行硬件映射`;
      }
      if (state.activeFile === 'examples/models/qwen3_jit/kernels/rmsnorm.py' && lineNumber >= 26) {
        row.dataset.rmsLine = String(lineNumber);
        row.dataset.rmsFunction = lineNumber < 56 ? 'input' : 'post';
        row.tabIndex = 0;
        row.title = lineNumber < 56 ? 'input_rmsnorm · 点击同步右侧分析' : 'post_rmsnorm · 点击同步右侧分析';
      }
      if (state.activeFile === 'examples/models/qwen3_jit/kernels/attention.py' && lineNumber >= 39) {
        const focus = lineNumber < 61 ? 'position' : lineNumber < 81 ? 'kv' : lineNumber < 107 ? 'q' : 'contract';
        row.dataset.attentionLine = String(lineNumber);
        row.dataset.attentionFocus = focus;
        row.tabIndex = 0;
        row.title = `${{ position: '位置与 RoPE 表', kv: 'K/V 旋转与缓存写入', q: 'Q 旋转与 Padding', contract: 'Out 参数写回契约' }[focus]} · 点击同步右侧分析`;
      }
      if (state.activeFile === 'examples/models/qwen3_jit/qwen3_decode.py' && lineNumber >= 49) {
        const focus = lineNumber < 68 ? 'signature' : lineNumber < 79 ? 'scope1' : lineNumber < 101 ? 'scope2' : lineNumber < 115 ? 'scope3' : 'smoke';
        row.dataset.qwenDecodeLine = String(lineNumber);
        row.dataset.qwenDecodeFocus = focus;
        row.tabIndex = 0;
        row.title = `${{ signature: 'JIT 入口契约', scope1: 'Scope 1 · RMSNorm + QKV', scope2: 'Scope 2 · RoPE + KV Cache', scope3: 'Scope 3 · Output + MLP', smoke: '编译 Smoke Test' }[focus]} · 点击同步右侧分析`;
      }
      if (isPagedAttentionFile(state.activeFile) && lineNumber >= 35) {
        const focus = lineNumber < 49 ? 'dynamic' : lineNumber < 93 ? 'builder' : lineNumber < 109 ? 'qk' : lineNumber < 136 ? 'softmax' : lineNumber < 151 ? 'pv' : lineNumber < 237 ? 'online' : lineNumber < 289 ? 'orchestration' : lineNumber < 368 ? 'paging' : lineNumber < 457 ? 'golden' : 'runtime';
        row.dataset.pagedAttentionLine = String(lineNumber);
        row.dataset.pagedAttentionFocus = focus;
        row.tabIndex = 0;
        row.title = `${{ dynamic: '动态 Shape 声明', builder: 'Program Builder 与 Init', qk: 'QK Matmul · Cube', softmax: 'Softmax Prepare · Vector', pv: 'PV Matmul · Cube', online: 'Online Update · Vector', orchestration: '动态维度推导', paging: 'Paged KV 编排', golden: 'Torch Golden', runtime: '运行配置与验证' }[focus]} · 点击同步右侧分析`;
      }
      row.append(gutter, code);
      fragment.append(row);
    });
    editor.replaceChildren(fragment);
    if (state.activeFile === RMSNORM_FILE) {
      $$('#dslEditor [data-rms-function]').forEach(row => row.classList.toggle('is-rms-function-active', row.dataset.rmsFunction === state.rmsNormFunction));
    }
    if (state.activeFile === ATTENTION_FILE) {
      $$('#dslEditor [data-attention-focus]').forEach(row => row.classList.toggle('is-attention-line-active', row.dataset.attentionFocus === state.attentionFocus));
    }
    if (state.activeFile === QWEN_DECODE_FILE) {
      $$('#dslEditor [data-qwen-decode-focus]').forEach(row => row.classList.toggle('is-qwen-decode-line-active', row.dataset.qwenDecodeFocus === state.qwenDecodeFocus));
    }
    if (isPagedAttentionFile(state.activeFile)) {
      $$('#dslEditor [data-paged-attention-focus]').forEach(row => row.classList.toggle('is-paged-attention-line-active', row.dataset.pagedAttentionFocus === state.pagedAttentionFocus));
    }
    $('[data-editor-tab="source"]').textContent = state.activeFile;
    editor.setAttribute('aria-label', `${state.activeFile} 全量源码`);
    editor.closest('[data-stage="1"]').setAttribute('aria-label', `${state.activeFile} 全量源码`);
  }

  async function renderSelectedSource(file) {
    const editor = $('#dslEditor');
    $('[data-editor-tab="source"]').textContent = file;
    editor.closest('[data-stage="1"]').setAttribute('aria-label', `${file} 全量源码`);
    editor.innerHTML = '<div><i>…</i><code>正在加载源码…</code></div>';
    try {
      await loadSource(file);
      if (state.activeFile === file) renderFullSource();
    } catch (error) {
      if (state.activeFile !== file) return;
      editor.innerHTML = `<div><i>!</i><code>无法加载 ${escapeHtml(file)}：${escapeHtml(error.message)}</code></div>`;
      toast(`无法读取 ${file}`);
    }
  }

  function renderRecipes() {
    $('#recipeGrid').innerHTML = recipes.map((r, index) => `<button class="kf-recipe${r.id === state.selectedRecipe ? ' is-active' : ''}" data-recipe="${r.id}"><span>0${index + 1}</span><b>${r.label}</b><small>${r.meta}</small></button>`).join('');
  }

  function renderPasses() {
    $('#passStrip').innerHTML = passes.map((name, index) => `<button class="kf-pass" data-pass="${index}"><span>PASS 0${index + 1}</span><b>${name}</b></button>`).join('');
    $('#guardGrid').innerHTML = guards.map(name => `<div class="kf-guard"><i>·</i>${name}</div>`).join('');
  }

  function renderOracles() {
    const cards = [
      ['CPU', 'Torch golden_decode_layer', 'argmax · B16', 'MATCH', false],
      ['HOST', 'FP32 carry reference', 'ratio tolerance', 'MATCH', false],
      ['PTO', 'PyPTO device', state.verified ? '16 / 16 argmax' : 'codegen blocked', state.verified ? 'MATCH' : 'BLOCKED', !state.verified]
    ];
    $('#oracleCards').innerHTML = cards.map(c => `<article class="kf-oracle${c[4] ? ' is-fail' : ''}"><span>${c[0]}</span><div><b>${c[1]}</b><small>${c[2]}</small></div><em>${c[3]}</em></article>`).join('');
  }

  function renderTensorCompare() {
    const values = ['1081', '431', '982', '77', '1532', '94', '611', '128', '205', '731', '44', '899', '1304', '62', '540', '311'];
    const tensor = (title, blocked) => `<section class="kf-tensor"><header><span>${title}</span><span>argmax · batch 16</span></header><div class="kf-tensor-grid">${values.map((v, i) => `<span class="${blocked && i === 10 ? 'diff' : ''}">${blocked && i === 10 ? '—' : v}</span>`).join('')}</div></section>`;
    $('#tensorCompare').innerHTML = tensor('Torch golden argmax', false) + tensor('PyPTO device argmax', !state.verified);
  }

  function renderGraph() {
    const mount = $('#irGraph');
    mount.innerHTML = '';
    const helper = window.PtoPassIrGraphNodePattern;
    if (helper) {
      const cards = [
        { type: 'tensor', data: { symbol: 'hidden_states', shape: [16, 5120], rawShape: [16, 5120], dtype: 'fp32', format: 'ND' } },
        { type: 'op', data: { opType: 'RMSNorm + QKV', stage: 'scope_1', latency: 'pending', outShape: [16, 5120], subgraphId: 1 }, accent: '#4369EF' },
        { type: 'op', data: { opType: 'Paged FA', stage: 'fa_fused', latency: 'pending', outShape: [16, 5120], subgraphId: 2 }, accent: '#9B60AA' },
        { type: 'op', data: { opType: 'MLP + dcr_xgamma', stage: 'scope_3', latency: 'pending', outShape: [16, 5120], subgraphId: 3 }, accent: '#2F9E7A' },
        { type: 'outcast', data: { name: 'out', shape: [16, 5120], rawShape: [16, 5120], dtype: 'fp32', format: 'ND', slotIdx: 0 } }
      ];
      cards.forEach(card => mount.appendChild(helper.buildNodeCardElement(card, { compact: true })));
    } else {
      mount.innerHTML = '<code>hidden_states → RMS/QKV → fa_fused → online_softmax → MLP → dcr_xgamma</code>';
    }
  }

  // ---- Unified run detail data (design spec §6.1) ----
  const gateMeta = { compile: '编译', correctness: '正确性', resource: '资源', perf: '性能' };
  const gateOrder = ['compile', 'correctness', 'resource', 'perf'];
  const gateSymbol = { pass: '✓', warn: '!', fail: '✕', idle: '·' };
  const gateTag = { pass: 'PASS', warn: 'WARN', fail: 'FAIL', idle: 'IDLE' };
  const evidenceMeta = {
    source: ['SRC', '源码'], ir: ['IR', 'IR / Pass'], trace: ['TRC', '设备 Trace'], tensor: ['TSR', 'Tensor'], metric: ['MTR', '指标']
  };

  const runs = [
    {
      id: 'run_8f2c', token: 'ptok://qwen3-14b/decode-layer@run_8f2c', title: 'Decode Layer · FP32 carry',
      verdict: 'blocked', verdictLabel: '被阻塞', subtitle: 'qwen3-14b', branch: 'kernel/decode-layer', time: '08/07 10:24', duration: '4m12s',
      gates: {
        env: ['pass', '指纹一致', 'env:8da1bf09'],
        compile: ['fail', 'Codegen 阻塞', 'INDEX / i64'],
        correctness: ['idle', '尚未运行', '被 codegen 阻塞'],
        resource: ['warn', '24 + 48 cores', '混合 Cube / Vector'],
        perf: ['idle', '未评估', '待编译通过']
      },
      conclusions: [
        ['high', '动态 work-table 索引阻塞 codegen', '<code>cursor + wp</code> 由设备侧读取驱动并进入 GM store offset，当前工具链触发 <code>GetOrCreateTensorView / index vs i64</code>。'],
        ['med', 'manual_scope 依赖必须显式保持', '<code>fa_work_build → fa_fused → online_softmax</code> 以及 <code>down_tids → dcr_xgamma</code> 依赖不能由 tensormap 自动补全。'],
        ['low', 'FP32 跨层传递改变数值基线', 'hidden_states / out 在层间保持 FP32，仅在输入和 LM Head 边界转为 BF16，应使用 argmax 与比例容差验证。']
      ],
      impact: [
        ['errValue', '错值风险', 'med', '待验证', 'FP32 carry 更精确，但与 BF16 旧基线不再逐位一致。'],
        ['repro', '复现风险', 'low', '低', '输入 seed、平台和编译参数已固化。'],
        ['perf', '性能影响', 'med', '待测', '目标是减少跨层 GM round-trip 并改善 ragged decode 负载均衡。']
      ],
      next: [
        ['cmd', '运行编译烟测', 'python decode_layer.py --smoke', '验证 parser 与 Pass 链并保留 codegen 证据'],
        ['fix', '切换静态仿射 work-table fallback', 'kernels/decode_layer.py:514', '绕过数据依赖 store offset 限制'],
        ['exp', '建议实验：block-level vs affine', 'pypto exp queue --schedule dense,affine', '比较负载均衡收益与编译可行性']
      ],
      evidence: { source: '1 span', ir: '5 pass', trace: '1 store', tensor: '12 ckpt', metric: '4 层' }
    },
    {
      id: 'run_d9a1', token: 'ptok://qwen3-32b/l14/rmsnorm-rope@run_d9a1', title: 'RMSNorm + RoPE 融合内核',
      verdict: 'trusted', verdictLabel: '可信基线', subtitle: 'qwen3-32b', branch: 'kernel/rmsnorm-rope', time: '08/01 10:07', duration: '1m52s',
      gates: {
        env: ['pass', '指纹一致', 'env:8da1bf09'],
        compile: ['pass', '4 / 4 Pass', '7 约束通过'],
        correctness: ['pass', '3 / 3 oracle', '16 / 16 match'],
        resource: ['pass', 'UB 61%', '预算内'],
        perf: ['pass', '+8% vs 基线', 'fusion 收益']
      },
      conclusions: [
        ['low', '融合内核已可信并优于基线', '三路 oracle 一致，且 RMSNorm 与 RoPE 融合较分开执行减少一次 GM round trip，端到端 +8%。']
      ],
      impact: [
        ['errValue', '错值风险', 'low', '无', '16 / 16 checkpoint 一致，最大绝对误差 0.0004883。'],
        ['repro', '复现风险', 'low', '低', '证据包已封存，指纹与工件哈希齐备。'],
        ['perf', '性能收益', 'low', '+8%', '相对未融合基线，收益可归因至减少的 GM 往返。']
      ],
      next: [
        ['cmd', '一键复现该基线', 'pypto trust replay ptok://qwen3-32b/l14/rmsnorm-rope@7c31e2a', '在任意锁定环境中重放'],
        ['exp', '基于此基线开始优化', 'pypto opt start --from 7c31e2a', '正确性契约将自动随实验比对']
      ],
      evidence: { source: '1 span', ir: '4 pass', trace: '2 store', tensor: '16 ckpt', metric: '4 层' }
    },
    {
      id: 'run-0729-m', token: 'ptok://qwen3-32b/moe/grouped-gemm@b8160fd', title: 'MoE Expert Grouped GEMM',
      verdict: 'trusted', verdictLabel: '可信基线', subtitle: 'qwen3-32b', branch: 'kernel/moe-expert', time: '07/29 16:41', duration: '3m04s',
      gates: {
        env: ['pass', '指纹一致', 'env:8da1bf09'],
        compile: ['pass', '5 / 5 Pass', '8 约束通过'],
        correctness: ['pass', '3 / 3 oracle', '24 / 24 match'],
        resource: ['pass', 'UB 73%', '预算内'],
        perf: ['pass', 'skip-empty 生效', '-19% 冗余计算']
      },
      conclusions: [
        ['low', 'Grouped GEMM 已可信', 'dispatch predicate 与 skip-empty-expert 正确表达，空专家被跳过，正确性与资源均在预算内。']
      ],
      impact: [
        ['errValue', '错值风险', 'low', '无', '24 / 24 checkpoint 一致。'],
        ['repro', '复现风险', 'low', '低', '证据包已封存。'],
        ['perf', '性能收益', 'low', '-19%', '跳过空专家减少冗余 grouped GEMM 计算。']
      ],
      next: [
        ['cmd', '一键复现该基线', 'pypto trust replay ptok://qwen3-32b/moe/grouped-gemm@b8160fd', '在锁定环境中重放']
      ],
      evidence: { source: '2 span', ir: '5 pass', trace: '3 store', tensor: '24 ckpt', metric: '4 层' }
    },
    {
      id: 'run-0726-d', token: 'ptok://qwen3-32b/decode-attn@run-0726-d', title: 'Decode Attention 延迟回归',
      verdict: 'stopped', verdictLabel: '已中止', subtitle: 'qwen3-32b', branch: 'perf/decode-attn', time: '07/26 09:18', duration: '0m47s',
      gates: {
        env: ['pass', '指纹一致', 'env:8da1bf09'],
        compile: ['pass', '5 / 5 Pass', '8 约束通过'],
        correctness: ['warn', '首个分歧待处理', 'step 128 logits'],
        resource: ['pass', 'UB 58%', '预算内'],
        perf: ['fail', '-31% 回退', 'slot wait 激增']
      },
      conclusions: [
        ['high', 'Decode 延迟相对基线回退 31%', 'Runtime Timeline 显示 slot wait 激增，dispatch 排队时间占比升至 44%，疑似 continuous batching 调度参数变更引入。'],
        ['med', 'step 128 出现首个 logits 分歧', '采样路径下 step 128 的 logits 与 reference 偏离，需先区分采样噪声与系统错误。']
      ],
      impact: [
        ['errValue', '错值风险', 'med', '中', 'logits 分歧可能改变停止条件，需 delta debugging 裁剪确认。'],
        ['repro', '复现风险', 'low', '低', '已生成脱敏复现包，含 task graph 与 trace。'],
        ['perf', '性能损失', 'high', '-31%', 'TPOT 相对可信基线明显回退，已中止以避免污染基线。']
      ],
      next: [
        ['cmd', '与可信基线做因果 diff', 'pypto diff run-0726-d ptok://…moe/grouped-gemm@b8160fd', '定位调度参数与 sync 变化'],
        ['exp', '回滚 batching 参数复测', 'pypto exp queue --batching continuous:prev', '验证回退是否来自调度变更']
      ],
      evidence: { source: '1 span', ir: '5 pass', trace: '1 timeline', tensor: '8 ckpt', metric: '4 层' }
    }
  ];

  const getRun = () => runs.find(r => r.id === state.currentRun) || runs[0];

  function renderRunList() {
    $('#runList').innerHTML = runs.map(run => `
      <button class="kf-run-item verdict-${run.verdict}${run.id === state.currentRun ? ' is-selected' : ''}" type="button" role="option" aria-selected="${run.id === state.currentRun}" data-run="${run.id}">
        <i></i><span><b>${escapeHtml(run.id)}</b><small>${escapeHtml(run.subtitle)} · ${escapeHtml(run.title)}</small><em>${escapeHtml(run.time)} · ${escapeHtml(run.duration)}</em></span><time>${escapeHtml(run.verdictLabel)}</time>
      </button>`).join('');
  }

  function renderRunDetail() {
    const run = getRun();
    const gatesHtml = gateOrder.map(key => {
      const [status, headline, detail] = run.gates[key];
      return `<button class="kf-gate ${status}" type="button" data-gate="${key}"><span class="kf-gate-icon">${gateSymbol[status]}</span><b>${gateMeta[key]}</b><span class="kf-gate-status">${gateTag[status]} · ${escapeHtml(headline)}</span><code>${escapeHtml(detail)}</code><span class="kf-gate-chevron">›</span></button>`;
    }).join('');
    const conclusionsHtml = run.conclusions.map((c, i) => {
      const sevLabel = { high: '阻塞', med: '风险', low: '提示' }[c[0]];
      return `<article class="kf-conclusion sev-${c[0]}"><span class="kf-conclusion-rank">${i + 1}</span><div><b>${escapeHtml(c[1])}</b><p>${c[2]}</p></div><span class="kf-conclusion-sev">${sevLabel}</span></article>`;
    }).join('');
    const impactLevelTag = { high: 'HIGH', med: 'MED', low: 'LOW' };
    const impactHtml = run.impact.map(im => `<div class="kf-impact"><span>${escapeHtml(im[1])}</span><b class="${im[2]}">${escapeHtml(im[3])}</b><small>${escapeHtml(im[4])}</small></div>`).join('');
    const selectedIndex = Math.max(0, run.next.findIndex(n => n[0] === state.runActionTab));
    const selectedNext = run.next[selectedIndex] || run.next[0];
    const nextTabs = run.next.map(n => `<button type="button" class="${n[0] === selectedNext[0] ? 'is-active' : ''}" data-run-action-tab="${n[0]}">${({ cmd: '执行命令', fix: '源码修复建议', exp: '实验验证' })[n[0]]}</button>`).join('');
    const evidenceHtml = Object.keys(evidenceMeta).map(key => {
      const [badge, label] = evidenceMeta[key];
      return `<button class="kf-evidence-node${state.selectedEvidence === key ? ' is-selected' : ''}" type="button" data-evidence="${key}"><span>${key}</span><b>${label}</b><em>${escapeHtml(run.evidence[key] || '—')}</em></button>`;
    }).join('');

    $('#runDetail').innerHTML = `
      <header class="kf-run-head">
        <div class="kf-run-head-main">
          <div class="kf-run-title-line"><span class="kf-run-verdict ${run.verdict}">${escapeHtml(run.verdictLabel)}</span><h1>${escapeHtml(run.id)}</h1><button type="button" id="copyRunToken">复制链接</button></div>
          <div class="kf-run-meta"><span><code>${escapeHtml(run.subtitle)}</code></span><span>${escapeHtml(run.title)}</span><span>分支 <code>${escapeHtml(run.branch)}</code></span><span>${escapeHtml(run.time)}</span><span>耗时 ${escapeHtml(run.duration)}</span></div>
        </div>
        <div class="kf-run-head-actions">
          <button type="button" id="runShare">分享</button>
          <button type="button" id="runCompare2" class="is-primary">对比运行</button>
        </div>
      </header>

      <button class="kf-run-baseline" type="button" id="baselinePicker"><span>对比基线</span><b>run_d9a1 · trusted</b><em>更换</em></button>

      <section class="kf-run-section kf-gates-section"><header><h2>四项运行门禁</h2><span class="kf-eyebrow">环境由右上角全局环境统一管理</span></header><div class="kf-run-gates">${gatesHtml}</div></section>

      <div class="kf-run-summary-grid">
        <div class="kf-run-summary-column">
          <section class="kf-run-section kf-conclusion-section"><header><h2>主要阻塞</h2><span class="kf-eyebrow">按影响排序</span></header><div class="kf-conclusion-list">${conclusionsHtml}</div></section>
          <section class="kf-run-section kf-impact-section"><header><h2>影响评估</h2></header><div class="kf-impact-grid">${impactHtml}</div></section>
        </div>
        <section class="kf-run-section kf-recommendation">
          <header><h2>推荐下一步</h2></header>
          <div class="kf-action-tabs">${nextTabs}</div>
          <div class="kf-action-workspace"><span class="kf-eyebrow">优先处理正确性失败位置</span><b>${escapeHtml(selectedNext[1])}</b><code>${escapeHtml(selectedNext[2])}</code><small>${escapeHtml(selectedNext[3])}</small><div class="kf-patch-preview" aria-label="源码修复预览"><span>184</span><del>out = fused_attention(q, k, v, mask)</del><span>185</span><ins>out = fused_attention(q.contiguous(), k.contiguous(), v.contiguous(), mask)</ins></div></div>
          <div class="kf-experiment-row"><button type="button">禁用融合验证</button><button type="button">切换 Kernel 版本</button><button type="button">调整并行度</button></div>
          <button class="kf-execute" type="button" data-next-action="${selectedNext[0]}" data-next-index="${selectedIndex}">执行所选建议</button>
        </section>
      </div>

      <section class="kf-run-section kf-evidence-section"><header><h2>证据链</h2><span class="kf-eyebrow">可逐层钻取</span></header><div class="kf-evidence-chain">${evidenceHtml}</div></section>`;
  }

  function updateRunInspector() {
    const run = getRun();
    $('#inspectorTitle').textContent = '证据检查器';
    $('#inspector').innerHTML = `
      <section class="kf-inspector-section kf-compare-explain"><h2 class="kf-inspector-title">为什么会变化</h2><p>IR 融合策略调整后，输出张量 stride 与基线不一致，数值误差随之放大；Kernel 调度变化同时造成吞吐下降。</p></section>
      <section class="kf-inspector-section"><h2 class="kf-inspector-title">关键指标对比</h2><div class="kf-metric-table"><div><span>指标</span><span>本次</span><span>基线</span></div><div><b>max_abs_error</b><em>2.7e-2</em><small>9.3e-8</small></div><div><b>mean_abs_error</b><em>4.1e-3</em><small>1.2e-8</small></div><div><b>吞吐 (tok/s)</b><em>8,432</em><small>10,454</small></div><div><b>HBM (GB)</b><em>9.72</em><small>9.11</small></div></div></section>
      <section class="kf-inspector-section"><h2 class="kf-inspector-title">复现信息</h2><dl><div><dt>确定性级别</dt><dd>非确定性</dd></div><div><dt>复现概率</dt><dd>~62%</dd></div><div><dt>相关性</dt><dd>高</dd></div><div><dt>受影响用例</dt><dd>3 / 12</dd></div><div><dt>首次出现</dt><dd>2026-08-06 14:32</dd></div></dl></section>
      <section class="kf-inspector-section"><h2 class="kf-inspector-title">当前证据</h2><div class="kf-run-inspector-hero ${run.verdict}"><b>${evidenceMeta[state.selectedEvidence][1]}</b><small>${escapeHtml(run.evidence[state.selectedEvidence] || '—')} · ${escapeHtml(run.id)}</small></div></section>`;
    $('#inspectorMeta').textContent = 'vs run_d9a1';
  }

  const inspectorContent = [
    `<section class="kf-inspector-section"><h2 class="kf-inspector-title">目标契约</h2><dl><div><dt>Source</dt><dd>Qwen3-14B · 40 layers</dd></div><div><dt>Recipe</dt><dd>decode_layer</dd></div><div><dt>Target</dt><dd>Ascend A2/A3/A5</dd></div><div><dt>Precision</dt><dd>FP32 carry · BF16 edge</dd></div></dl></section><section class="kf-inspector-section"><h2 class="kf-inspector-title">Toolkit 读取</h2><div class="kf-evidence-list"><div class="kf-evidence"><span>✓</span><b>29 inputs</b><small>signature</small></div><div class="kf-evidence"><span>✓</span><b>4 scopes</b><small>schedule</small></div><div class="kf-evidence"><span>✓</span><b>explicit TaskIds</b><small>deps</small></div></div></section><div class="kf-inspector-card"><b>为什么从契约开始？</b><p>Decode Layer 同时跨越 RMSNorm、QKV、Paged Attention、MLP 与层间 carry，任何局部修改都必须保持整条依赖链。</p></div>`,
    `<section class="kf-inspector-section"><h2 class="kf-inspector-title">语义意图</h2><dl><div><dt>Compute</dt><dd>RMS → QKV → FA → MLP</dd></div><div><dt>Carry</dt><dd>FP32 inter-layer</dd></div><div><dt>Schedule</dt><dd>dense block-level</dd></div><div><dt>Output</dt><dd>out + normed_out</dd></div></dl></section><section class="kf-inspector-section"><h2 class="kf-inspector-title">即时诊断</h2><div class="kf-inspector-card"><b style="color:var(--warning)">PTO-CODEGEN-INDEX</b><p id="inspectorDiagnostic">设备侧动态索引进入 GM store offset；当前工具链可能出现 INDEX / i64 类型冲突。</p></div></section>`,
    `<section class="kf-inspector-section"><h2 class="kf-inspector-title">卫士覆盖</h2><div class="kf-evidence-list">${guards.map(g => `<div class="kf-evidence"><span>○</span><b>${g}</b><small>pending</small></div>`).join('')}</div></section><div class="kf-inspector-card"><b>验证粒度</b><p>卫士在每个 Pass 之后运行。失败时保留前后 IR、约束快照与最小复现入口。</p></div>`,
    `<section class="kf-inspector-section"><h2 class="kf-inspector-title">阻塞证据</h2><dl><div><dt>Operator</dt><dd>decode_layer</dd></div><div><dt>Task</dt><dd>fa_work_build</dd></div><div><dt>Tensor</dt><dd>fa_work_table</dd></div><div><dt>Offset</dt><dd>cursor + wp</dd></div></dl></section><section class="kf-inspector-section"><h2 class="kf-inspector-title">关联证据</h2><div class="kf-evidence-list"><div class="kf-evidence"><span>↗</span><b>Source line 520</b><small>dynamic index</small></div><div class="kf-evidence"><span>↗</span><b>Lowering Pass</b><small>INDEX / i64</small></div><div class="kf-evidence"><span>↗</span><b>Codegen log</b><small>TensorView</small></div></div></section>`,
    `<section class="kf-inspector-section"><h2 class="kf-inspector-title">可信状态</h2><div class="kf-inspector-card"><b style="color:var(--success)">可用于性能优化</b><p>此基线冻结 correctness 契约。之后的 tile、pipeline 或内存优化都可与它自动比对。</p></div></section><section class="kf-inspector-section"><h2 class="kf-inspector-title">签名摘要</h2><dl><div><dt>Evidence</dt><dd>sha256:91b4…0e2c</dd></div><div><dt>Environment</dt><dd>sha256:8da1…bf09</dd></div><div><dt>Artifact</dt><dd>sha256:13fe…8c71</dd></div></dl></section>`
  ];

  const intentPreview = {
    shape: {
      label: 'Shape', meta: 'Qwen3-14B · contracted',
      rows: [['hidden / out', '[16, 5120] · FP32'], ['Q / KV hidden', '5120 / 1024'], ['Heads', '40 Q / 8 KV · dim 128'], ['Layer carry', 'out + normed_out']],
      note: '层间 hidden 与 residual 保持 FP32；仅外部 embedding 输入和最终 LM Head 边界进行 BF16 转换。'
    },
    layout: {
      label: 'Layout', meta: 'paged · split-K · tiled',
      rows: [['Paged KV', 'SEQ_TILE 128'], ['Q head batch', '5 real / 16 padded'], ['QKV tile', 'TM16 · TN256 · TK256'], ['MLP tile', 'TN1024 · chunk256']],
      note: 'SEQ_TILE 与 serving page_size 绑定；每个 dense work item 只处理一个真实 sequence block。'
    },
    scope: {
      label: 'Scope', meta: 'manual · auto-dep boundary',
      rows: [['Scope 1', 'RMSNorm + Q/K/V'], ['Scope 2', 'Paged FA + online softmax'], ['Scope 3', 'out_proj + MLP'], ['Boundary', 'dcr_xgamma outside manual']],
      note: 'manual_scope 内 tensormap 自动依赖被抑制，跨 scope 的任务顺序必须通过显式 TaskId 传递。'
    },
    deps: {
      label: '依赖', meta: 'explicit TaskId chain',
      rows: [['prev_out_tids', 'rms_recip'], ['work_tid', 'fa_fused'], ['fa_tid', 'online_softmax'], ['down_tids', 'dcr_xgamma']],
      note: '核心链路是 fa_work_build → fa_fused → online_softmax；层间 carry 由 dcr_xgamma 的单次 SPMD dispatch 完成。'
    },
    resource: {
      label: '资源', meta: 'declared scheduling intent',
      rows: [['FA grid', '24 persistent cores'], ['Softmax grid', '48 vector blocks'], ['dcr_xgamma', '5 parallel slabs'], ['FA table cap', 'BATCH × MAX_CTX_BLOCKS']],
      note: '资源意图优先平衡 ragged decode；A2/A3 的 Cube↔Vector 边界仍会经过 GM pipe buffer。'
    }
  };

  const ATTENTION_FILE = 'examples/models/qwen3_jit/kernels/attention.py';
  const QWEN_DECODE_FILE = 'examples/models/qwen3_jit/qwen3_decode.py';
  const PAGED_ATTENTION_FILE = 'examples/models/06_paged_attention_dynamic.py';
  const PAGED_ATTENTION_ROOT_FILE = 'paged_attention_dynamic.py';
  const isPagedAttentionFile = (file) => file === PAGED_ATTENTION_FILE || file === PAGED_ATTENTION_ROOT_FILE;
  const RMSNORM_FILE = 'examples/models/qwen3_jit/kernels/rmsnorm.py';
  const attentionFocusMeta = {
    position: { label: '位置索引', lines: '52–60', detail: '读取 seq_lens，定位当前 token，并保留 RoPE 行维度' },
    kv: { label: 'K/V Cache', lines: '62–79', detail: '按 8 个 KV Head 旋转 K，并把 K/V 写入当前 cache_row' },
    q: { label: 'Q 旋转与补齐', lines: '81–106', detail: '8 个真实 Q Head 旋转后补齐到 16 行，供后续 GQA 使用' },
    contract: { label: '写回契约', lines: '107–111', detail: '三个 Out 参数原位更新，仅返回 k_cache 的 SSA 句柄' },
  };
  const attentionComputationGraph = {
    width: 620,
    height: 690,
    nodes: [
      { id: 'attn-q', label: 'q_proj', typeLabel: '[16,8192] · FP32', kind: 'tensor', x: 110, y: 65, width: 176, height: 50, colorKey: 'io:activation' },
      { id: 'attn-k', label: 'k_proj', typeLabel: '[16,1024] · FP32', kind: 'tensor', x: 310, y: 65, width: 176, height: 50, colorKey: 'io:activation' },
      { id: 'attn-v', label: 'v_proj', typeLabel: '[16,1024] · FP32', kind: 'tensor', x: 510, y: 65, width: 176, height: 50, colorKey: 'io:activation' },
      { id: 'attn-seq', label: 'seq_lens', typeLabel: '[16] · INT32', kind: 'tensor', x: 110, y: 185, width: 158, height: 46, colorKey: 'io:state' },
      { id: 'attn-position', label: 'pos = ctx_len − 1', typeLabel: 'Current token index', kind: 'op', x: 110, y: 285, width: 202, height: 54, colorKey: 'sem:linear' },
      { id: 'attn-rope-table', label: 'RoPE cos / sin', typeLabel: '[1,64] halves · FP32', kind: 'state', state_type: 'constant', x: 310, y: 285, width: 198, height: 50, colorKey: 'io:constant' },
      { id: 'attn-q-rotate', label: 'Q RoPE rotate', typeLabel: '8 heads × 128 · FP32', kind: 'op', x: 110, y: 440, width: 194, height: 56, colorKey: 'sem:rope' },
      { id: 'attn-k-rotate', label: 'K RoPE rotate', typeLabel: '1 KV head × 128 · FP32', kind: 'op', x: 310, y: 440, width: 204, height: 56, colorKey: 'sem:rope' },
      { id: 'attn-v-cast', label: 'V cast', typeLabel: 'FP32 → BF16', kind: 'op', x: 510, y: 440, width: 154, height: 54, colorKey: 'sem:linear' },
      { id: 'attn-q-pad', label: 'Q pad + assemble', typeLabel: '8 real + 8 zero · BF16', kind: 'op', x: 110, y: 605, width: 198, height: 56, colorKey: 'sem:comm' },
      { id: 'attn-k-cache', label: 'K Cache', typeLabel: 'Current row write · BF16', kind: 'state', x: 310, y: 605, width: 184, height: 50, colorKey: 'io:state' },
      { id: 'attn-v-cache', label: 'V Cache', typeLabel: 'Current row write · BF16', kind: 'state', x: 510, y: 605, width: 184, height: 50, colorKey: 'io:state' },
    ],
    edges: [
      { source: 'attn-q', target: 'attn-q-rotate', tag: 'Q block' },
      { source: 'attn-k', target: 'attn-k-rotate', tag: 'K lo / hi' },
      { source: 'attn-v', target: 'attn-v-cast', tag: 'V row' },
      { source: 'attn-seq', target: 'attn-position', tag: 'ctx_len' },
      { source: 'attn-position', target: 'attn-rope-table', dashed: true, tag: 'pos' },
      { source: 'attn-rope-table', target: 'attn-q-rotate', dashed: true, tag: 'cos / sin' },
      { source: 'attn-rope-table', target: 'attn-k-rotate', dashed: true, tag: 'cos / sin' },
      { source: 'attn-q-rotate', target: 'attn-q-pad', tag: 'cast BF16' },
      { source: 'attn-k-rotate', target: 'attn-k-cache', tag: 'assemble' },
      { source: 'attn-v-cast', target: 'attn-v-cache', tag: 'assemble' },
    ],
  };
  const attentionGraphFocus = {
    'attn-seq': 'position', 'attn-position': 'position', 'attn-rope-table': 'position',
    'attn-k': 'kv', 'attn-v': 'kv', 'attn-k-rotate': 'kv', 'attn-v-cast': 'kv', 'attn-k-cache': 'kv', 'attn-v-cache': 'kv',
    'attn-q': 'q', 'attn-q-rotate': 'q', 'attn-q-pad': 'q',
  };

  function attentionOverview() {
    return `
      <section class="kf-attn-context"><span>Q / K / V projection</span><i>→</i><b>RoPE + KV Cache Update</b><i>→</i><span>Grouped-query attention</span></section>
      <section class="kf-inspector-section kf-attn-contract"><header><h2 class="kf-inspector-title">算子契约</h2><span>Qwen3-32B · decode</span></header><div class="kf-attn-contract-grid"><div><span>Q projection</span><b>[16, 8192]</b><em>FP32</em></div><div><span>K / V projection</span><b>[16, 1024] × 2</b><em>FP32</em></div><div><span>RoPE cos / sin</span><b>[4096, 128] × 2</b><em>FP32</em></div><div><span>K / V cache</span><b>[524288, 128] × 2</b><em>BF16</em></div><div><span>Padded Q</span><b>[2048, 128]</b><em>BF16</em></div><div><span>Scope</span><b>16 batch × 8 KV heads</b><em>CORE_GROUP</em></div></div></section>
      <section class="kf-inspector-section kf-attn-computation"><header><h2 class="kf-inspector-title">算子计算图</h2><span>设计系统 · Model Graphviz</span></header><div class="pto-model-graphviz-pattern-page pto-model-graphviz-stage kf-attn-computation__stage" id="attentionComputationGraph" aria-label="RoPE 与 KV Cache 更新计算图"></div><footer id="attentionGraphStatus">点击节点可联动对应源码阶段 · 支持拖拽与缩放</footer></section>
      <section class="kf-inspector-section kf-attn-coverage"><header><h2 class="kf-inspector-title">Attention 覆盖范围</h2><span>当前文件并非完整 Attention</span></header><div class="kf-attn-stage-line"><span class="is-done">RoPE</span><span class="is-done">KV 写入</span><span class="is-done">Q Padding</span><span>QK Matmul</span><span>Mask</span><span>Softmax</span><span>SV Matmul</span></div><p>当前仅实现 Scope 2 的前置子阶段。完整 grouped-query attention 的 QK、Softmax、SV 和 online accumulation 仍未在此文件中实现。</p></section>
      <div class="kf-inspector-card kf-attn-insight"><b>Agent 结论</b><p>这个函数的主要产物不是 Attention 输出，而是当前 token 的 K/V Cache 增量，以及供后续 GQA 消费的 BF16 padded Q。</p></div>`;
  }

  function attentionData() {
    return `
      <section class="kf-inspector-section kf-attn-precision"><header><h2 class="kf-inspector-title">数据与精度流</h2><span>源码事实</span></header><div class="kf-attn-data-flow"><div><span>Q / K / V</span><b>FP32</b></div><i>＋ FP32 RoPE table</i><div><span>RoPE rotate</span><b>FP32 compute</b></div><i>cast before assemble</i><div><span>K / V Cache · Padded Q</span><b>BF16</b></div></div></section>
      <section class="kf-inspector-section kf-attn-memory"><header><h2 class="kf-inspector-title">逻辑数据规模</h2><span>EST. · shape × dtype</span></header><dl><div><dt>单个 K Cache</dt><dd>128 MiB · BF16</dd></div><div><dt>单个 V Cache</dt><dd>128 MiB · BF16</dd></div><div><dt>Padded Q</dt><dd>512 KiB · BF16</dd></div><div><dt>每 Batch Q 输入</dt><dd>32 KiB · FP32</dd></div><div><dt>每 Batch K + V 输入</dt><dd>8 KiB · FP32</dd></div><div><dt>每 Batch Cache 增量</dt><dd>4 KiB · BF16</dd></div></dl></section>
      <section class="kf-attn-pad"><div><span>8 real Q heads</span><b>8 × 128</b></div><i>pad</i><div><span>8 zero rows</span><b>8 × 128</b></div><em>→ 16 × 128 BF16 / KV head</em></section>
      <div class="kf-inspector-card kf-rms-estimate"><b>可信边界</b><p>字节数是逻辑规模；实际搬运次数、片上占用和 Cache 写合并方式需要结合 Pass 后 IR 与设备指令确认。</p></div>`;
  }

  function attentionMapping() {
    const active = attentionFocusMeta[state.attentionFocus] || attentionFocusMeta.position;
    return `
      <section class="kf-inspector-section kf-attn-mapping"><header><h2 class="kf-inspector-title">并行与地址映射</h2><span>16 Batch lanes</span></header><div class="kf-attn-lanes">${Array.from({ length: 16 }, (_, index) => `<i>${index}</i>`).join('')}</div><dl><div><dt>外层并行</dt><dd><code>pl.parallel(16)</code></dd></div><div><dt>每 Lane KV 循环</dt><dd><code>pl.range(8)</code></dd></div><div><dt>K/V cache_row</dt><dd><code>b × 8 × 4096 + ki × 4096 + pos</code></dd></div><div><dt>Q pad_row0</dt><dd><code>b × 8 × 16 + ki × 16</code></dd></div></dl></section>
      <section class="kf-inspector-section kf-attn-source-map"><header><h2 class="kf-inspector-title">源码阶段</h2><span>点击与源码联动</span></header><div>${Object.entries(attentionFocusMeta).map(([key, item]) => `<button type="button" class="${key === state.attentionFocus ? 'is-active' : ''}" data-attention-focus="${key}"><i>${item.lines}</i><span><b>${item.label}</b><small>${item.detail}</small></span></button>`).join('')}</div></section>
      <div class="kf-inspector-card kf-attn-insight"><b>${active.label}</b><p>${active.detail}。当前选中源码第 ${active.lines} 行。</p></div>`;
  }

  function attentionValidation() {
    return `
      <section class="kf-inspector-section kf-rms-validation"><header><h2 class="kf-inspector-title">当前证据</h2><span>结构 ≠ 数值</span></header><div class="kf-rms-proof"><div class="is-pass"><i>✓</i><p><b>Qwen3 JIT 全管线可编译</b><small>tests/ut/jit/test_qwen3_decode.py</small></p><em>已验证</em></div><div class="is-pass"><i>✓</i><p><b>rope_kv_cache scope 被 outline</b><small>name_hint = rope_kv_cache</small></p><em>已验证</em></div><div><i>○</i><p><b>RoPE 数值 Golden</b><small>Q / K rotation · position edge</small></p><em>缺失</em></div><div><i>○</i><p><b>Cache 地址与增量写入</b><small>pos = 0 / MAX_SEQ - 1</small></p><em>缺失</em></div><div><i>○</i><p><b>Q Padding 内容验证</b><small>8 real + 8 zero rows</small></p><em>缺失</em></div><div><i>○</i><p><b>完整 Attention 数值链路</b><small>QK · mask · softmax · SV</small></p><em>未实现</em></div></div></section>
      <section class="kf-inspector-section kf-attn-risks"><header><h2 class="kf-inspector-title">编码风险</h2><span>需要显式守卫</span></header><ul><li><b>位置边界</b><span><code>ctx_len</code> 必须位于 1…4096，否则 <code>pos</code> 越界。</span></li><li><b>Rank 约束</b><span>RoPE 表必须保留 [1, 64] 行维，供 <code>col_expand_mul</code> 使用。</span></li><li><b>Out 契约</b><span>V Cache 与 padded Q 依赖原位写回，不能只从返回值判断产物。</span></li></ul></section>
      <button class="kf-rms-action" type="button" data-attention-action="golden">＋ 生成 RoPE / Cache / Padding 数值测试</button>`;
  }

  function renderAttentionInspector({ scrollToFocus = false } = {}) {
    attentionGraphController?.destroy?.();
    attentionGraphController = null;
    const tabs = { overview: '概览', data: '数据与精度', mapping: '并行与地址', validation: '验证' };
    const content = state.attentionTab === 'data' ? attentionData() : state.attentionTab === 'mapping' ? attentionMapping() : state.attentionTab === 'validation' ? attentionValidation() : attentionOverview();
    $('#inspectorTitle').textContent = 'Attention 分析';
    $('#inspectorMeta').textContent = 'rope_kv_cache_update · static';
    $('#inspector').innerHTML = `
      <section class="kf-attn-hero"><span class="kf-eyebrow">CODING AGENT · SOURCE ANALYSIS</span><div><b>RoPE + KV Cache Update</b><em>PARTIAL ATTENTION</em></div><small>Qwen3-32B decode · Scope 2 · current-token update</small></section>
      <div class="kf-attn-tabs" role="tablist" aria-label="Attention 分析视图">${Object.entries(tabs).map(([key, label]) => `<button type="button" class="${key === state.attentionTab ? 'is-active' : ''}" data-attention-tab="${key}">${label}</button>`).join('')}</div>
      <div class="kf-attn-view">${content}</div>
      <footer class="kf-rms-provenance"><span><i class="fact"></i>源码事实</span><span><i class="resolved"></i>调用点解析</span><span><i class="estimated"></i>静态估算</span></footer>`;
    $$('#dslEditor [data-attention-focus]').forEach(row => row.classList.toggle('is-attention-line-active', row.dataset.attentionFocus === state.attentionFocus));
    if (scrollToFocus) $(`#dslEditor [data-attention-focus="${state.attentionFocus}"]`)?.scrollIntoView({ block: 'center', behavior: 'smooth' });
    if (state.attentionTab === 'overview') renderAttentionComputationGraph();
  }

  function renderAttentionComputationGraph() {
    const pattern = window.PtoModelGraphvizPattern;
    const stage = $('#attentionComputationGraph');
    const status = $('#attentionGraphStatus');
    if (!pattern || !stage) return;
    attentionGraphController = pattern.renderController(stage, attentionComputationGraph, {
      ariaLabel: 'RoPE rotation, KV cache update and Q padding computation graph',
      colormap: pattern.modelArchitectureColormap(attentionComputationGraph),
      fitMode: 'full', viewportPadding: 18, autoFit: true,
      interaction: { panZoom: true, selectableClusters: false },
      overlays: { edgeTags: true },
      onSelect: ({ nodeId }) => {
        const focus = attentionGraphFocus[nodeId];
        if (!focus) return;
        state.attentionFocus = focus;
        $$('#dslEditor [data-attention-focus]').forEach(row => row.classList.toggle('is-attention-line-active', row.dataset.attentionFocus === focus));
        const meta = attentionFocusMeta[focus];
        if (status && meta) status.textContent = `${meta.label} · 源码第 ${meta.lines} 行 · ${meta.detail}`;
      },
    });
  }

  const qwenDecodeFocusMeta = {
    signature: { label: '入口契约', lines: '49–67', detail: '单层 decode 的输入、权重、KV Cache 与 BF16 输出契约' },
    scope1: { label: 'Scope 1 · QKV', lines: '68–77', detail: 'Input RMSNorm 后并行生成 FP32 Q、K、V 投影' },
    scope2: { label: 'Scope 2 · Attention', lines: '79–99', detail: '完成 RoPE 与 KV Cache 更新；完整 GQA 尚未实现，attn_out 是占位张量' },
    scope3: { label: 'Scope 3 · MLP', lines: '101–110', detail: '输出投影残差、Post RMSNorm、MLP 与 Down Projection 残差' },
    smoke: { label: '编译冒烟', lines: '115–144', detail: '构造静态输入并调用 compile_for_test，仅验证编译链路' },
  };
  const qwenDecodeComputationGraph = {
    width: 640,
    height: 980,
    nodes: [
      { id: 'decode-hidden', label: 'hidden_states', typeLabel: '[16,8192] · BF16', kind: 'tensor', x: 320, y: 45, width: 190, height: 50, colorKey: 'io:activation' },
      { id: 'decode-input-rms', label: 'input_rmsnorm', typeLabel: 'Scope 1 · BF16', kind: 'op', x: 320, y: 135, width: 194, height: 54, colorKey: 'sem:norm' },
      { id: 'decode-q', label: 'q_projection', typeLabel: '[16,8192] · FP32', kind: 'op', x: 105, y: 245, width: 176, height: 54, colorKey: 'sem:linear' },
      { id: 'decode-k', label: 'k_projection', typeLabel: '[16,1024] · FP32', kind: 'op', x: 320, y: 245, width: 176, height: 54, colorKey: 'sem:linear' },
      { id: 'decode-v', label: 'v_projection', typeLabel: '[16,1024] · FP32', kind: 'op', x: 535, y: 245, width: 176, height: 54, colorKey: 'sem:linear' },
      { id: 'decode-rope', label: 'RoPE + KV Cache Update', typeLabel: 'Scope 2 · BF16 Q/K/V', kind: 'op', x: 320, y: 370, width: 238, height: 58, colorKey: 'sem:rope' },
      { id: 'decode-cache', label: 'K / V Cache', typeLabel: '2 × 128 MiB · BF16', kind: 'state', x: 535, y: 475, width: 184, height: 50, colorKey: 'io:state' },
      { id: 'decode-gap', label: 'Full Grouped-query Attention', typeLabel: 'NOT IMPLEMENTED', kind: 'op', x: 320, y: 500, width: 240, height: 58, colorKey: 'io:constant' },
      { id: 'decode-attn-out', label: 'attn_out placeholder', typeLabel: '[16,8192] · BF16', kind: 'tensor', x: 320, y: 605, width: 210, height: 52, colorKey: 'io:constant' },
      { id: 'decode-out-proj', label: 'out_projection + residual', typeLabel: '[16,8192] · FP32', kind: 'op', x: 320, y: 700, width: 228, height: 56, colorKey: 'sem:linear' },
      { id: 'decode-post-rms', label: 'post_rmsnorm', typeLabel: 'BF16', kind: 'op', x: 320, y: 790, width: 176, height: 52, colorKey: 'sem:norm' },
      { id: 'decode-mlp', label: 'mlp_block', typeLabel: '[16,25600] · BF16', kind: 'op', x: 320, y: 875, width: 188, height: 54, colorKey: 'sem:mlp' },
      { id: 'decode-out', label: 'down_projection + residual', typeLabel: 'out · [16,8192] · BF16', kind: 'op', x: 320, y: 960, width: 238, height: 56, colorKey: 'io:output' },
    ],
    edges: [
      { source: 'decode-hidden', target: 'decode-input-rms', tag: 'BF16' },
      { source: 'decode-input-rms', target: 'decode-q', tag: 'normed' },
      { source: 'decode-input-rms', target: 'decode-k', tag: 'normed' },
      { source: 'decode-input-rms', target: 'decode-v', tag: 'normed' },
      { source: 'decode-q', target: 'decode-rope', tag: 'Q' },
      { source: 'decode-k', target: 'decode-rope', tag: 'K' },
      { source: 'decode-v', target: 'decode-rope', tag: 'V' },
      { source: 'decode-rope', target: 'decode-cache', tag: 'write state' },
      { source: 'decode-rope', target: 'decode-gap', dashed: true, tag: 'padded Q' },
      { source: 'decode-cache', target: 'decode-gap', dashed: true, tag: 'read history' },
      { source: 'decode-gap', target: 'decode-attn-out', dashed: true, tag: 'missing producer' },
      { source: 'decode-attn-out', target: 'decode-out-proj', tag: 'placeholder' },
      { source: 'decode-hidden', target: 'decode-out-proj', dashed: true, tag: 'residual' },
      { source: 'decode-out-proj', target: 'decode-post-rms', tag: 'resid1' },
      { source: 'decode-post-rms', target: 'decode-mlp', tag: 'normed' },
      { source: 'decode-mlp', target: 'decode-out', tag: 'gated MLP' },
      { source: 'decode-out-proj', target: 'decode-out', dashed: true, tag: 'residual' },
    ],
  };
  const qwenDecodeGraphFocus = {
    'decode-hidden': 'signature',
    'decode-input-rms': 'scope1', 'decode-q': 'scope1', 'decode-k': 'scope1', 'decode-v': 'scope1',
    'decode-rope': 'scope2', 'decode-cache': 'scope2', 'decode-gap': 'scope2', 'decode-attn-out': 'scope2',
    'decode-out-proj': 'scope3', 'decode-post-rms': 'scope3', 'decode-mlp': 'scope3', 'decode-out': 'scope3',
  };

  function qwenDecodeOverview() {
    return `
      <section class="kf-qwen-decode-context"><span>9 inline utilities</span><i>→</i><b>3 manual scopes</b><i>→</i><span>1 orchestration entry</span></section>
      <section class="kf-inspector-section kf-qwen-decode-contract"><header><h2 class="kf-inspector-title">单层 Decode 契约</h2><span>Qwen3-32B · batch 16</span></header><div class="kf-attn-contract-grid"><div><span>Hidden / Output</span><b>[16, 8192]</b><em>BF16</em></div><div><span>Q / KV hidden</span><b>8192 / 1024</b><em>FP32 projection</em></div><div><span>Intermediate</span><b>[16, 25600]</b><em>BF16</em></div><div><span>Head dim</span><b>128</b><em>8 KV groups</em></div><div><span>K / V Cache</span><b>[524288, 128] × 2</b><em>BF16</em></div><div><span>Max sequence</span><b>4096</b><em>static</em></div></div></section>
      <section class="kf-qwen-decode-scopes" aria-label="源码执行范围"><button type="button" data-qwen-decode-focus="scope1"><i>01</i><span><b>RMSNorm + QKV</b><small>BF16 norm → FP32 projections</small></span></button><button type="button" data-qwen-decode-focus="scope2"><i>02</i><span><b>RoPE + KV Cache</b><small>Attention 主体仍为空缺</small></span></button><button type="button" data-qwen-decode-focus="scope3"><i>03</i><span><b>Output + MLP</b><small>FP32 residual carry → BF16 out</small></span></button></section>
      <section class="kf-inspector-section kf-qwen-decode-computation"><header><h2 class="kf-inspector-title">算子计算图</h2><span>设计系统 · Model Graphviz</span></header><div class="pto-model-graphviz-pattern-page pto-model-graphviz-stage kf-qwen-decode-computation__stage" id="qwenDecodeComputationGraph" aria-label="Qwen3 单层 Decode 计算图"></div><footer id="qwenDecodeGraphStatus">点击节点可定位 Scope · 虚线表示状态或尚未闭合的数据依赖</footer></section>
      <div class="kf-qwen-decode-gap"><i>!</i><div><b>计算图存在真实断点</b><p>源码只更新 RoPE / KV Cache，未实现完整 grouped-query attention；<code>attn_out</code> 是没有生产者的占位张量，不能把当前函数视为可数值执行的完整 Decoder Layer。</p></div></div>`;
  }

  function qwenDecodeData() {
    return `
      <section class="kf-inspector-section kf-qwen-decode-precision"><header><h2 class="kf-inspector-title">精度传递</h2><span>按源码调用边界</span></header><div class="kf-qwen-decode-data-chain"><div><b>hidden_states</b><em>BF16</em></div><i>RMSNorm</i><div><b>Q / K / V</b><em>FP32</em></div><i>RoPE + assemble</i><div><b>attn_out</b><em>BF16 · placeholder</em></div><i>residual</i><div><b>resid1_tile</b><em>FP32</em></div><i>Post RMS + MLP</i><div><b>out</b><em>BF16</em></div></div></section>
      <section class="kf-inspector-section kf-attn-memory"><header><h2 class="kf-inspector-title">逻辑数据规模</h2><span>EST. · 不含临时 Tile</span></header><dl><div><dt>Hidden / Normed / Output</dt><dd>各 256 KiB · BF16</dd></div><div><dt>Q projection</dt><dd>512 KiB · FP32</dd></div><div><dt>K / V projection</dt><dd>各 64 KiB · FP32</dd></div><div><dt>Padded Q</dt><dd>512 KiB · BF16</dd></div><div><dt>Residual carry</dt><dd>512 KiB · FP32</dd></div><div><dt>MLP intermediate</dt><dd>800 KiB · BF16</dd></div><div><dt>K / V Cache</dt><dd>各 128 MiB · BF16</dd></div></dl></section>
      <div class="kf-inspector-card kf-rms-estimate"><b>Agent 观察</b><p>FP32 主要承载投影与残差累加，BF16 用于 scope 间激活和最终输出。<code>attn_out</code> 的 dtype 虽已声明，但内容并无有效数值来源。</p></div>`;
  }

  function qwenDecodeOrchestration() {
    const active = qwenDecodeFocusMeta[state.qwenDecodeFocus] || qwenDecodeFocusMeta.scope1;
    const structure = qwenDecodeStructure();
    return `
      ${structure}
      <section class="kf-inspector-section kf-qwen-decode-deps"><header><h2 class="kf-inspector-title">跨文件组合</h2><span>9 utilities · 4 source files</span></header><div><article><b>rmsnorm.py</b><p>input_rmsnorm · post_rmsnorm</p></article><article><b>projection.py</b><p>Q / K / V · Out residual · Down residual</p></article><article><b>attention.py</b><p>rope_kv_cache_update</p></article><article><b>mlp.py</b><p>mlp_block</p></article></div></section>
      <section class="kf-qwen-decode-pass"><span>InlineFunctions</span><i>→</i><span>OutlineIncoreScopes</span><i>→</i><b>qwen3_decode · Orchestration</b></section>
      <section class="kf-inspector-section kf-attn-source-map kf-qwen-decode-source-map"><header><h2 class="kf-inspector-title">源码阶段</h2><span>点击与源码联动</span></header><div>${Object.entries(qwenDecodeFocusMeta).map(([key, item]) => `<button type="button" class="${key === state.qwenDecodeFocus ? 'is-active' : ''}" data-qwen-decode-focus="${key}"><i>${item.lines}</i><span><b>${item.label}</b><small>${item.detail}</small></span></button>`).join('')}</div></section>
      <div class="kf-inspector-card kf-attn-insight"><b>${active.label}</b><p>${active.detail}。当前选中源码第 ${active.lines} 行。</p></div>`;
  }

  function qwenDecodeValidation() {
    return `
      <section class="kf-inspector-section kf-rms-validation"><header><h2 class="kf-inspector-title">当前证据</h2><span>编译通过 ≠ 功能完整</span></header><div class="kf-rms-proof"><div class="is-pass"><i>✓</i><p><b>完整 JIT 管线可编译</b><small>test_qwen3_decode_full_pipeline</small></p><em>已验证</em></div><div class="is-pass"><i>✓</i><p><b>Inline 节点全部消除</b><small>入口保留为 Orchestration</small></p><em>已验证</em></div><div class="is-pass"><i>✓</i><p><b>11 个预期 scope hints 存在</b><small>RMS · Projection · MLP · RoPE</small></p><em>已验证</em></div><div><i>○</i><p><b>单层数值 Golden</b><small>当前 Attention 数据链未闭合</small></p><em>不可验证</em></div><div><i>○</i><p><b>昇腾设备实跑</b><small>输出误差 · Cache 增量 · 边界位置</small></p><em>缺失</em></div><div><i>○</i><p><b>端到端性能基线</b><small>scope latency · bandwidth · overlap</small></p><em>缺失</em></div></div></section>
      <section class="kf-inspector-section kf-attn-risks"><header><h2 class="kf-inspector-title">完成 Decode 前的阻塞项</h2><span>高优先级</span></header><ul><li><b>补齐 Grouped-query Attention</b><span>连接 padded Q、历史 K/V Cache 到合法的 <code>attn_out</code> 生产者。</span></li><li><b>建立数值 Oracle</b><span>覆盖短序列、最大位置、KV Cache 增量与 BF16 容差。</span></li><li><b>设备侧验证</b><span>编译测试不包含实际昇腾执行与性能数据。</span></li></ul></section>
      <button class="kf-rms-action" type="button" data-qwen-decode-action="test">＋ 生成单层 Decode 测试清单</button>`;
  }

  function qwenDecodeStructure() {
    const scopes = [
      ['scope1', '01', 'RMSNorm + QKV', 'input_rmsnorm → q / k / v projection', 'rmsnorm.py · projection.py'],
      ['scope2', '02', 'RoPE + KV Cache', 'rope_kv_cache_update → q_pad + cache writes', 'attention.py'],
      ['scope3', '03', 'Output + MLP', 'out_projection → post_rmsnorm → mlp_block', 'projection.py · rmsnorm.py · mlp.py'],
    ];
    const files = [['rmsnorm.py', '2 functions', 'input_rmsnorm · post_rmsnorm', 'scope1'], ['projection.py', '5 functions', 'QKV · out · down projection', 'scope3'], ['attention.py', '1 function', 'rope_kv_cache_update', 'scope2'], ['mlp.py', '1 function', 'mlp_block · SiLU gate', 'scope3']];
    const calls = [['input_rmsnorm', 'rmsnorm.py', 'row_sum · sqrt · recip · col_expand_mul', 'scope1'], ['q_projection / k_projection / v_projection', 'projection.py', 'matmul · matmul_acc · assemble', 'scope1'], ['rope_kv_cache_update', 'attention.py', 'slice · col_expand_mul · sub · add · cast · assemble', 'scope2'], ['mlp_block', 'mlp.py', 'matmul · silu · mul · assemble', 'scope3'], ['out_projection / down_projection', 'projection.py', 'matmul · matmul_acc · add · assemble', 'scope3']];
    return `
      <section class="kf-structure-hero"><div><span class="kf-eyebrow">CODE STRUCTURE · STATIC CALL MAP</span><h2>qwen3_decode.py</h2><p>1 个入口 · 3 个执行 Scope · 4 个 kernel 文件 · 26 个细粒度算子调用</p></div><span class="kf-structure-status"><i></i>已解析</span></section>
      <section class="kf-structure-flow" aria-label="qwen3_decode 调用结构"><div class="kf-structure-column kf-structure-entry"><span class="kf-structure-column-label">入口</span><button type="button" data-qwen-structure-focus="signature"><b>_decode_layer</b><small>@pl.jit.inline · line 49</small></button><i class="kf-structure-connector"></i></div><div class="kf-structure-column"><span class="kf-structure-column-label">执行 Scope</span>${scopes.map(([key, no, title, meta, file]) => `<button type="button" class="kf-structure-node ${key}" data-qwen-structure-focus="${key}"><span>${no}</span><b>${title}</b><small>${meta}</small><em>${file}</em></button>`).join('')}</div><div class="kf-structure-column kf-structure-kernels"><span class="kf-structure-column-label">kernels/ 文件映射</span>${files.map(([file, count, detail, focus]) => `<button type="button" class="kf-structure-file" data-qwen-structure-focus="${focus}"><span class="kf-file-icon py">Py</span><span><b>${file}</b><small>${count} · ${detail}</small></span><i>↗</i></button>`).join('')}</div></section>
      <section class="kf-structure-operators"><header><div><h2 class="kf-inspector-title">细粒度算子调用</h2><span>按 kernel 文件归组 · 点击定位 Scope</span></div><code>pl.*</code></header><div class="kf-operator-list">${calls.map(([name, file, ops, focus], index) => `<button type="button" data-qwen-structure-focus="${focus}"><span class="kf-operator-index">0${index + 1}</span><span><b>${name}</b><small>${file}</small></span><em>${ops}</em><i>›</i></button>`).join('')}</div></section>
      <section class="kf-structure-legend"><span><i class="entry"></i>入口</span><span><i class="scope"></i>Scope</span><span><i class="kernel"></i>kernel 文件</span><span><i class="op"></i>细粒度算子</span><b>虚线依赖：Scope 2 的 attn_out 仍是待补齐生产者</b></section>`;
  }

  function renderQwenDecodeInspector({ scrollToFocus = false } = {}) {
    qwenDecodeGraphController?.destroy?.();
    qwenDecodeGraphController = null;
    pagedAttentionGraphController?.destroy?.();
    pagedAttentionGraphController = null;
    const tabs = { overview: '概览', data: '数据与精度', orchestration: '编排与依赖', validation: '验证' };
    const content = state.qwenDecodeTab === 'data' ? qwenDecodeData() : state.qwenDecodeTab === 'orchestration' ? qwenDecodeOrchestration() : state.qwenDecodeTab === 'validation' ? qwenDecodeValidation() : qwenDecodeOverview();
    $('#inspectorTitle').textContent = 'Decode Layer 分析';
    $('#inspectorMeta').textContent = 'qwen3_decode · orchestration';
    $('#inspector').innerHTML = `
      <section class="kf-qwen-decode-hero"><span class="kf-eyebrow">CODING AGENT · SOURCE ANALYSIS</span><div><b>qwen3_decode</b><em>PARTIAL DECODE</em></div><small>Qwen3-32B · single layer · JIT orchestration entry</small></section>
      <div class="kf-qwen-decode-tabs" role="tablist" aria-label="Qwen3 Decode 分析视图">${Object.entries(tabs).map(([key, label]) => `<button type="button" class="${key === state.qwenDecodeTab ? 'is-active' : ''}" data-qwen-decode-tab="${key}">${label}</button>`).join('')}</div>
      <div class="kf-qwen-decode-view">${content}</div>
      <footer class="kf-rms-provenance"><span><i class="fact"></i>源码事实</span><span><i class="resolved"></i>跨文件解析</span><span><i class="estimated"></i>静态估算</span></footer>`;
    $$('#dslEditor [data-qwen-decode-focus]').forEach(row => row.classList.toggle('is-qwen-decode-line-active', row.dataset.qwenDecodeFocus === state.qwenDecodeFocus));
    if (scrollToFocus) $(`#dslEditor [data-qwen-decode-focus="${state.qwenDecodeFocus}"]`)?.scrollIntoView({ block: 'center', behavior: 'smooth' });
    if (state.qwenDecodeTab === 'overview') renderQwenDecodeComputationGraph();
  }

  function renderQwenDecodeComputationGraph() {
    const pattern = window.PtoModelGraphvizPattern;
    const stage = $('#qwenDecodeComputationGraph');
    const status = $('#qwenDecodeGraphStatus');
    if (!pattern || !stage) return;
    qwenDecodeGraphController = pattern.renderController(stage, qwenDecodeComputationGraph, {
      ariaLabel: 'Qwen3 single layer decode orchestration computation graph with incomplete attention stage',
      colormap: pattern.modelArchitectureColormap(qwenDecodeComputationGraph),
      fitMode: 'full', viewportPadding: 18, autoFit: true,
      interaction: { panZoom: true, selectableClusters: false },
      overlays: { edgeTags: true },
      onSelect: ({ nodeId }) => {
        const focus = qwenDecodeGraphFocus[nodeId];
        if (!focus) return;
        state.qwenDecodeFocus = focus;
        $$('#dslEditor [data-qwen-decode-focus]').forEach(row => row.classList.toggle('is-qwen-decode-line-active', row.dataset.qwenDecodeFocus === focus));
        const meta = qwenDecodeFocusMeta[focus];
        if (status && meta) status.textContent = `${meta.label} · 源码第 ${meta.lines} 行 · ${meta.detail}`;
      },
    });
  }

  const pagedAttentionFocusMeta = {
    dynamic: { label: '动态 Shape 声明', lines: '35–41', detail: '7 个 pl.dynamic 符号描述 Batch、Head、Block 与扁平缓存规模' },
    builder: { label: 'Builder 与闭包参数', lines: '49–92', detail: 'q_tile、head_dim、block_size 固化为 load 的 Tile 尺寸，Tensor 标注保持动态' },
    qk: { label: 'QK Matmul', lines: '93–108', detail: 'Cube 路径计算 qi × kjᵀ，FP32 累加输出 sij' },
    softmax: { label: 'Softmax Prepare', lines: '109–135', detail: 'Vector 完成 scale、row_max、exp、BF16 概率与 FP32 row_sum' },
    pv: { label: 'PV Matmul', lines: '136–150', detail: 'Cube 路径计算 pij × vj，得到 FP32 block output' },
    online: { label: 'Online Update', lines: '151–236', detail: 'Vector 合并跨 Block 的 mi、li、oi，并在末块归一化写回' },
    orchestration: { label: '动态维度推导', lines: '237–288', detail: '运行时从 Tensor.dim 推导 batch、head、block_size、block_num 与 q_loop' },
    paging: { label: 'Paged KV 编排', lines: '289–367', detail: 'block_table 将逻辑 KV Block 映射到物理 Cache Row，并处理末块 valid_len' },
    golden: { label: 'Torch Golden', lines: '368–456', detail: '参考实现复现分页寻址、Padding Mask、BF16 概率与 Online Softmax' },
    runtime: { label: '运行与门禁', lines: '457–543', detail: 'A2/A3 · Ascend910B，64 Batch、8192 Context，rtol/atol 2e-2' },
  };
  const pagedAttentionComputationGraph = {
    width: 650,
    height: 890,
    nodes: [
      { id: 'pa-query', label: 'query', typeLabel: '[B×H, D] · BF16', kind: 'tensor', x: 105, y: 55, width: 174, height: 50, colorKey: 'io:activation' },
      { id: 'pa-context', label: 'context_lens', typeLabel: '[B] · INT32', kind: 'tensor', x: 320, y: 55, width: 212, height: 50, colorKey: 'io:state' },
      { id: 'pa-table', label: 'block_table', typeLabel: '[B×MaxBlocks] · INT32', kind: 'tensor', x: 535, y: 55, width: 208, height: 50, colorKey: 'io:state' },
      { id: 'pa-page', label: 'Logical → Physical', typeLabel: 'row = block_id × block_size', kind: 'op', x: 430, y: 165, width: 238, height: 56, colorKey: 'sem:comm' },
      { id: 'pa-kv', label: 'K / V Cache Block', typeLabel: '[Block, D] × 2 · BF16', kind: 'state', x: 535, y: 275, width: 206, height: 52, colorKey: 'io:state' },
      { id: 'pa-qk', label: 'QK Matmul', typeLabel: '[QTile, Block] · FP32', kind: 'op', x: 215, y: 285, width: 220, height: 62, colorKey: 'sem:linear' },
      { id: 'pa-mask', label: 'valid_len Slice', typeLabel: 'Last block padding mask', kind: 'op', x: 215, y: 395, width: 196, height: 52, colorKey: 'sem:comm' },
      { id: 'pa-softmax', label: 'Softmax Prepare', typeLabel: 'pij BF16 · mi/li FP32', kind: 'op', x: 320, y: 500, width: 238, height: 64, colorKey: 'sem:softmax' },
      { id: 'pa-pv', label: 'PV Matmul', typeLabel: '[QTile, D] · FP32', kind: 'op', x: 430, y: 610, width: 214, height: 62, colorKey: 'sem:linear' },
      { id: 'pa-online', label: 'Online Update', typeLabel: 'mi / li / oi · FP32', kind: 'op', x: 320, y: 715, width: 246, height: 64, colorKey: 'sem:softmax' },
      { id: 'pa-out', label: 'out', typeLabel: '[B×H, D] · FP32', kind: 'tensor', x: 320, y: 825, width: 220, height: 54, colorKey: 'io:output' },
    ],
    edges: [
      { source: 'pa-context', target: 'pa-page', tag: 'valid blocks' },
      { source: 'pa-table', target: 'pa-page', tag: 'block id' },
      { source: 'pa-page', target: 'pa-kv', tag: 'physical row' },
      { source: 'pa-query', target: 'pa-qk', tag: 'Q tile' },
      { source: 'pa-kv', target: 'pa-qk', tag: 'K block' },
      { source: 'pa-qk', target: 'pa-mask', tag: 'sij' },
      { source: 'pa-context', target: 'pa-mask', dashed: true, tag: 'valid_len' },
      { source: 'pa-mask', target: 'pa-softmax', tag: 'valid scores' },
      { source: 'pa-softmax', target: 'pa-pv', tag: 'pij BF16' },
      { source: 'pa-kv', target: 'pa-pv', tag: 'V block' },
      { source: 'pa-pv', target: 'pa-online', tag: 'oi_new' },
      { source: 'pa-softmax', target: 'pa-online', tag: 'mi / li' },
      { source: 'pa-online', target: 'pa-online', dashed: true, tag: 'next block state' },
      { source: 'pa-online', target: 'pa-out', tag: 'last block · oi/li' },
    ],
  };
  const pagedAttentionGraphFocus = {
    'pa-query': 'orchestration', 'pa-context': 'orchestration', 'pa-table': 'paging', 'pa-page': 'paging', 'pa-kv': 'paging',
    'pa-qk': 'qk', 'pa-mask': 'paging', 'pa-softmax': 'softmax', 'pa-pv': 'pv', 'pa-online': 'online', 'pa-out': 'online',
  };
  const pagedAttentionDrilldowns = {
    'pa-qk': {
      focus: 'qk',
      children: [
        { key: 'load', label: 'Load Q / K', precision: 'BF16', shape: '[Q,D] / [B,D]', hardware: 'GM → L1' },
        { key: 'view', label: 'Transpose K', precision: 'BF16 view', shape: '[B,D] → [D,B]', hardware: 'L1 view' },
        { key: 'matmul', label: 'Matmul', precision: 'FP32 accum', shape: '[Q,D]×[D,B]', hardware: 'Cube · L0' },
      ],
    },
    'pa-softmax': {
      focus: 'softmax',
      children: [
        { key: 'scale', label: 'Scale + Slice', precision: 'FP32', shape: '[Q,valid]', hardware: 'Vector · UB' },
        { key: 'exp', label: 'Row Max + Exp', precision: 'FP32', shape: 'row-wise', hardware: 'Vector · UB' },
        { key: 'sum', label: 'Sum + Cast', precision: 'FP32 → BF16', shape: '[Q,1] / [Q,B]', hardware: 'Vector · UB' },
      ],
    },
    'pa-pv': {
      focus: 'pv',
      children: [
        { key: 'load', label: 'Load P / V', precision: 'BF16', shape: '[Q,B] / [B,D]', hardware: 'GM → L1' },
        { key: 'move', label: 'Move to L0', precision: 'BF16', shape: 'tile view', hardware: 'L0A / L0B' },
        { key: 'matmul', label: 'Matmul', precision: 'FP32 accum', shape: '[Q,B]×[B,D]', hardware: 'Cube · L0C' },
      ],
    },
    'pa-online': {
      focus: 'online',
      children: [
        { key: 'rescale', label: 'Max + Rescale', precision: 'FP32', shape: '[Q,1]', hardware: 'Vector · UB' },
        { key: 'accum', label: 'Accumulate', precision: 'FP32 state', shape: 'li [Q,1] · oi [Q,D]', hardware: 'Vector · UB' },
        { key: 'store', label: 'Normalize + Store', precision: 'FP32', shape: 'out [Q,D]', hardware: 'UB → GM' },
      ],
    },
  };

  function pagedAttentionOverview() {
    const overlayMeta = {
      precision: { label: '精度', legend: '<i class="bf16"></i>BF16 输入 / 概率　<i class="fp32"></i>FP32 计算 / 状态　<i class="index"></i>INT32 / INDEX' },
      shape: { label: 'Shape', legend: '<i class="tensor"></i>Tensor / Tile Shape　<i class="dynamic"></i>动态有效区与循环边界' },
      hardware: { label: '硬件', legend: '<i class="cube"></i>Cube　<i class="vector"></i>Vector　<i class="memory"></i>GM / 编排' },
    }[state.pagedAttentionOverlay];
    return `
      <section class="kf-pa-summary-strip"><div><span>动态维度</span><b>B · H · D · Block</b></div><div><span>四阶段构成</span><b title="QK、PV 为 Cube 阶段；Softmax、Online Update 为 Vector 阶段">C×2 / V×2</b></div><div><span>状态</span><b>FP32 Online</b></div></section>
      <section class="kf-inspector-section kf-pa-computation"><header class="kf-pa-graph-head"><div><h2 class="kf-inspector-title">融合计算图</h2><span>点击节点联动源码</span></div><div class="kf-pa-overlay-switch" role="group" aria-label="计算图叠加信息">${[['precision','精度'],['shape','Shape'],['hardware','硬件']].map(([key,label]) => `<button type="button" class="${key === state.pagedAttentionOverlay ? 'is-active' : ''}" data-pa-overlay="${key}">${label}</button>`).join('')}</div></header><div class="kf-pa-overlay-legend" data-overlay="${state.pagedAttentionOverlay}"><b>${overlayMeta.label}叠加</b><span>${overlayMeta.legend}</span></div><div class="pto-model-graphviz-pattern-page pto-model-graphviz-stage kf-pa-computation__stage" id="pagedAttentionComputationGraph" aria-label="动态 Paged Attention 融合计算图"></div><footer id="pagedAttentionGraphStatus">当前显示${overlayMeta.label}信息 · 带 + 节点可展开 · 虚线表示跨 Block 状态</footer></section>
      <section class="kf-pa-insight-grid"><button type="button" data-pa-go-tab="data"><i>01</i><span><b>精度断点</b><small>Softmax 概率显式降为 BF16</small></span></button><button type="button" data-pa-go-tab="schedule"><i>02</i><span><b>动态边界</b><small>Tensor 动态，load Tile 由闭包固定</small></span></button><button type="button" data-pa-go-tab="validation"><i>!</i><span><b>首要风险</b><small>Q Head 尾 Tile 尚无有效 Shape</small></span></button></section>`;
  }

  function pagedAttentionDataExecution() {
    return `
      <section class="kf-pa-data-group"><h3>数据流与精度</h3><div class="kf-pa-execution-band" aria-label="Paged Attention 数据与精度流"><div class="source"><em>GM · BF16</em><b>Q [16,128]</b><small>4 KiB</small></div><i>load</i><button type="button" class="cube" data-paged-attention-focus="qk"><em>CUBE · L1/L0</em><b>QK Matmul</b><small>BF16 × BF16 → FP32 sij [16,128]</small></button><i>store / load</i><button type="button" class="vector" data-paged-attention-focus="softmax"><em>VECTOR · UB</em><b>Mask + Softmax</b><small>FP32 exp → BF16 pij [16,128]</small></button><i>store / load</i><button type="button" class="cube" data-paged-attention-focus="pv"><em>CUBE · L1/L0</em><b>PV Matmul</b><small>BF16 × BF16 → FP32 oi_new [16,128]</small></button><i>store / load</i><button type="button" class="vector" data-paged-attention-focus="online"><em>VECTOR · UB</em><b>Online Update</b><small>FP32 mi / li / oi → FP32 out</small></button><i>store</i><div class="source"><em>GM · FP32</em><b>Output [B×H,D]</b><small>512 KiB / example</small></div></div><div class="kf-inspector-card"><b>精度敏感点</b><p><code>exp</code> 后概率先转 BF16；<code>mi / li / oi</code> 跨 Block 合并保持 FP32。Scale 与末 Block Mask 仍需结合真实模型和边界输入确认。</p></div></section>
      <section class="kf-pa-data-group"><h3>数据形状与布局</h3><section class="kf-inspector-section kf-pa-layout"><header><h2 class="kf-inspector-title">Shape / Layout 变换</h2><span>Shape · View · Memory</span></header><div class="kf-pa-layout-flow"><div><i>Query</i><b>[QTile,D]</b><small>BF16 · natural</small></div><span>×</span><div><i>K natural</i><b>[Block,D]</b><small>BF16 · L1</small></div><span>transpose_view</span><div><i>Kᵀ view</i><b>[D,Block]</b><small>零拷贝视图</small></div><span>→</span><div><i>Score</i><b>[QTile,Block]</b><small>FP32 · L0C</small></div></div></section></section>
      <section class="kf-pa-data-group"><h3>有效数据与边界</h3><section class="kf-inspector-section kf-pa-validshape"><header><h2 class="kf-inspector-title">有效区与 Padding</h2><span>Block128 · valid_len dynamic</span></header><div><span class="is-valid"><b>有效 Token 列</b><small>进入 row_max / exp / row_sum</small></span><span class="is-pad"><b>Padding</b><small>末块排除</small></span></div></section></section>
      <section class="kf-pa-data-group"><h3>资源规模</h3><section class="kf-inspector-section kf-pa-memory"><header><h2 class="kf-inspector-title">单 Block 工作集</h2><span>QTile16 · Block128 · D128</span></header><div class="kf-pa-working-set"><span><b>Q</b><em>4 KiB · BF16</em></span><span><b>K + V</b><em>64 KiB · BF16</em></span><span><b>sij</b><em>8 KiB · FP32</em></span><span><b>pij</b><em>4 KiB · BF16</em></span><span><b>oi state</b><em>8 KiB · FP32</em></span></div></section><div class="kf-inspector-card kf-rms-estimate"><b>硬件可信边界</b><p>执行带同时展示数据类型与执行节点；真实 GM 往返、Buffer 地址和重叠程度仍需读取 Pass IR、Swimlane 与 PMU。</p></div></section>`;
  }

  function pagedAttentionDataExecutionLegacy() {
    return `
      <section class="kf-pa-execution-band" aria-label="Paged Attention 数据与硬件执行带"><div class="source"><em>GM · BF16</em><b>Q [16,128]</b><small>4 KiB</small></div><i>load</i><button type="button" class="cube" data-paged-attention-focus="qk"><em>CUBE · L1/L0</em><b>QK Matmul</b><small>BF16 × BF16 → FP32 sij [16,128]</small></button><i>store / load</i><button type="button" class="vector" data-paged-attention-focus="softmax"><em>VECTOR · UB</em><b>Mask + Softmax</b><small>FP32 exp → BF16 pij [16,128]</small></button><i>store / load</i><button type="button" class="cube" data-paged-attention-focus="pv"><em>CUBE · L1/L0</em><b>PV Matmul</b><small>BF16 × BF16 → FP32 oi_new [16,128]</small></button><i>store / load</i><button type="button" class="vector" data-paged-attention-focus="online"><em>VECTOR · UB</em><b>Online Update</b><small>FP32 mi / li / oi → FP32 out</small></button><i>store</i><div class="source"><em>GM · FP32</em><b>Output [B×H,D]</b><small>512 KiB / example</small></div></section>
      <section class="kf-inspector-section kf-pa-layout"><header><h2 class="kf-inspector-title">Layout 叠加</h2><span>Shape · View · Memory</span></header><div class="kf-pa-layout-flow"><div><i>Query</i><b>[QTile,D]</b><small>BF16 · natural</small></div><span>×</span><div><i>K natural</i><b>[Block,D]</b><small>BF16 · L1</small></div><span>transpose_view</span><div><i>Kᵀ view</i><b>[D,Block]</b><small>零拷贝视图</small></div><span>→</span><div><i>Score</i><b>[QTile,Block]</b><small>FP32 · L0C</small></div></div></section>
      <section class="kf-inspector-section kf-pa-validshape"><header><h2 class="kf-inspector-title">有效区与数据规模</h2><span>Block128 · valid_len dynamic</span></header><div><span class="is-valid"><b>有效 Token 列</b><small>进入 row_max / exp / row_sum</small></span><span class="is-pad"><b>Padding</b><small>末块排除</small></span></div><div class="kf-pa-working-set"><span><b>Q</b><em>4 KiB</em></span><span><b>K + V</b><em>64 KiB</em></span><span><b>sij</b><em>8 KiB FP32</em></span><span><b>pij</b><em>4 KiB BF16</em></span><span><b>oi state</b><em>8 KiB FP32</em></span></div></section>
      <div class="kf-inspector-card kf-rms-estimate"><b>硬件可信边界</b><p>执行带把 MemorySpace 和算子语义叠加显示；A2/A3 上 Cube↔Vector 的真实 GM 往返、Buffer 地址和重叠程度仍需读取 Pass IR、Swimlane 与 PMU。</p></div>`;
  }

  function pagedAttentionSchedule() {
    const blocks = Array.from({ length: 16 }, (_, index) => `<i class="${index < 4 ? 'is-hot' : ''}">${index}</i>`).join('');
    return `
      <section class="kf-pa-schedule-canvas"><div class="kf-pa-loop-rail"><div><i>B</i><span><b>64 Batch</b><small>pl.range</small></span></div><div><i>Q</i><span><b>1 Head Tile</b><small>16 heads ÷ QTile16</small></span></div><div><i>K</i><span><b>64 KV Blocks</b><small>8192 ÷ Block128</small></span></div></div><div class="kf-pa-schedule-main"><div class="kf-pa-tile-row"><button type="button" data-paged-attention-focus="qk"><b>QK</b><small>16×128×128</small></button><i>→</i><button type="button" data-paged-attention-focus="softmax"><b>Softmax</b><small>16×valid_len</small></button><i>→</i><button type="button" data-paged-attention-focus="pv"><b>PV</b><small>16×128×128</small></button><i>→</i><button type="button" data-paged-attention-focus="online"><b>Update</b><small>FP32 carry</small></button></div><div class="kf-pa-block-mini">${blocks}</div><div class="kf-pa-page-equation"><span>logical <b>bn</b></span><i>table[b × block_num + bn]</i><span>physical <b>block_id</b></span><i>× block_size</i><span>cache <b>row</b></span></div></div></section>
      <section class="kf-inspector-section kf-pa-scope"><header><h2 class="kf-inspector-title">Scope 与依赖叠加</h2><span>16,448 InCore calls · example</span></header><div class="kf-pa-compact-scope"><div><i>O</i><span><b>Orchestration</b><small>动态维度 · 分页寻址 · 三层循环</small></span></div><i>dispatch</i><div><i>C</i><span><b>QK</b><small>Cube</small></span></div><i>→</i><div><i>V</i><span><b>Softmax</b><small>Vector</small></span></div><i>→</i><div><i>C</i><span><b>PV</b><small>Cube</small></span></div><i>→</i><div><i>V</i><span><b>Update</b><small>Vector</small></span></div><i class="carry">↺ mi / li / oi carry to next Block</i></div></section>
      <section class="kf-pa-schedule-notes"><article><b>可并行</b><p>Batch 与 Q Tile 数据相互独立，但当前使用 <code>pl.range</code>，未显式声明并行。</p></article><article><b>必须串行</b><p>KV Block 之间通过 FP32 <code>mi/li/oi</code> 状态 Carry 形成循环依赖。</p></article><article><b>边界风险</b><p>KV 末块有 <code>valid_len</code>；Q Head 尾 Tile 尚缺对应有效 Shape。</p></article></section>
      <section class="kf-inspector-section kf-attn-source-map kf-pa-source-map"><header><h2 class="kf-inspector-title">源码阶段</h2><span>点击联动</span></header><div>${Object.entries(pagedAttentionFocusMeta).map(([key, item]) => `<button type="button" class="${key === state.pagedAttentionFocus ? 'is-active' : ''}" data-paged-attention-focus="${key}"><i>${item.lines}</i><span><b>${item.label}</b><small>${item.detail}</small></span></button>`).join('')}</div></section>`;
  }

  function pagedAttentionDynamic() {
    const active = pagedAttentionFocusMeta[state.pagedAttentionFocus] || pagedAttentionFocusMeta.paging;
    return `
      <section class="kf-inspector-section kf-pa-shape"><header><h2 class="kf-inspector-title">动态 Shape 推导</h2><span>Tensor.dim · runtime</span></header><div class="kf-pa-formulas"><div><span>batch</span><b>context_lens.dim(0)</b></div><div><span>num_heads</span><b>query.rows ÷ batch</b></div><div><span>block_size</span><b>value_cache.rows ÷ block_table.size</b></div><div><span>blocks / request</span><b>block_table.size ÷ batch</b></div><div><span>Q loops</span><b>ceil(num_heads ÷ q_tile)</b></div><div><span>KV loops</span><b>ceil(context_len ÷ block_size)</b></div></div></section>
      <section class="kf-inspector-section kf-pa-address"><header><h2 class="kf-inspector-title">Paged KV 地址映射</h2><span>logical block → physical row</span></header><div class="kf-pa-page-map"><div><small>Request b</small><b>logical block bn</b></div><i>table[b × block_num + bn]</i><div><small>Physical Block</small><b>cur_block_idx</b></div><i>× block_size</i><div><small>Cache Row</small><b>kv_block_row</b></div></div><p><code>valid_len = min(block_size, context_len − bn × block_size)</code>，末 Block 只让有效列进入 Softmax。</p></section>
      <section class="kf-inspector-section kf-attn-source-map kf-pa-source-map"><header><h2 class="kf-inspector-title">源码阶段</h2><span>点击与源码联动</span></header><div>${Object.entries(pagedAttentionFocusMeta).map(([key, item]) => `<button type="button" class="${key === state.pagedAttentionFocus ? 'is-active' : ''}" data-paged-attention-focus="${key}"><i>${item.lines}</i><span><b>${item.label}</b><small>${item.detail}</small></span></button>`).join('')}</div></section>
      <div class="kf-inspector-card kf-attn-insight"><b>${active.label}</b><p>${active.detail}。当前选中源码第 ${active.lines} 行。</p></div>`;
  }

  function pagedAttentionContractLayout() {
    return `
      <section class="kf-inspector-section kf-pa-contract"><header><h2 class="kf-inspector-title">Tensor 契约与方向</h2><span>B / H / D / Block 均运行时解析</span></header><div class="kf-pa-tensor-table"><div class="head"><span>Tensor</span><b>Shape</b><em>方向 · DType</em></div><button type="button" data-paged-attention-focus="orchestration"><span>query</span><b>[B×H, D]</b><em>In · BF16</em></button><button type="button" data-paged-attention-focus="paging"><span>key_cache</span><b>[KVRows, D]</b><em>In · BF16</em></button><button type="button" data-paged-attention-focus="paging"><span>value_cache</span><b>[KVRows, D]</b><em>In · BF16</em></button><button type="button" data-paged-attention-focus="paging"><span>block_table</span><b>[B×MaxBlocks]</b><em>In · INT32</em></button><button type="button" data-paged-attention-focus="orchestration"><span>context_lens</span><b>[B]</b><em>In · INT32</em></button><button type="button" data-paged-attention-focus="online"><span>out</span><b>[B×H, D]</b><em>Out · FP32</em></button></div></section>
      <section class="kf-inspector-section kf-pa-layout"><header><h2 class="kf-inspector-title">Shape / Layout 变换</h2><span>runtime row-major Tensor → on-chip Tile</span></header><div class="kf-pa-layout-flow"><div><i>Query view</i><b>[QTile, D]</b><small>BF16 · natural</small></div><span>×</span><div><i>K natural</i><b>[Block, D]</b><small>BF16 · L1/Mat</small></div><span>transpose_view</span><div><i>Kᵀ view</i><b>[D, Block]</b><small>no data copy</small></div><span>→</span><div><i>Score</i><b>[QTile, Block]</b><small>FP32 · L0C</small></div></div><div class="kf-pa-layout-flow is-pv"><div><i>Probability</i><b>[QTile, Block]</b><small>BF16</small></div><span>×</span><div><i>V natural</i><b>[Block, D]</b><small>BF16</small></div><span>→</span><div><i>Block output</i><b>[QTile, D]</b><small>FP32</small></div></div></section>
      <section class="kf-inspector-section kf-pa-validshape"><header><h2 class="kf-inspector-title">有效 Shape 与 Padding</h2><span>动态边界</span></header><div><span class="is-valid" style="--valid:78%"><b>valid_len</b><small>进入 Softmax 的有效 Token 列</small></span><span class="is-pad"><b>padding</b><small>末 Block 不应参与 row_max / row_sum</small></span></div><p>KV Slice 仍取完整 <code>[block_size, D]</code>，Score 通过 <code>sij_valid = slice(..., valid_len)</code> 收窄。Q Head 尾 Tile 则没有同等明确的 valid shape，是需要补测的接口边界。</p></section>
      <section class="kf-inspector-section kf-pa-memory"><header><h2 class="kf-inspector-title">示例逻辑规模</h2><span>B64 · H16 · D128 · Block128</span></header><dl><div><dt>Query</dt><dd>256 KiB · BF16</dd></div><div><dt>单个 K / V Cache</dt><dd>512 MiB · BF16</dd></div><div><dt>Block Table</dt><dd>64 KiB · INT32</dd></div><div><dt>Context Lengths</dt><dd>256 B · INT32</dd></div><div><dt>Output</dt><dd>512 KiB · FP32</dd></div></dl></section>`;
  }

  function pagedAttentionPrecision() {
    return `
      <section class="kf-inspector-section kf-pa-precision"><header><h2 class="kf-inspector-title">端到端精度流</h2><span>cast 与累加边界</span></header><div class="kf-pa-precision-path"><button type="button" data-paged-attention-focus="qk"><span>Q / K</span><b>BF16</b><small>Cube input</small></button><i>matmul accumulate</i><button type="button" data-paged-attention-focus="softmax"><span>sij / exp</span><b>FP32</b><small>Vector compute</small></button><i>explicit cast</i><button type="button" data-paged-attention-focus="pv"><span>pij</span><b>BF16</b><small>PV input</small></button><i>matmul accumulate</i><button type="button" data-paged-attention-focus="online"><span>oi_new</span><b>FP32</b><small>block result</small></button><i>online merge</i><button type="button" data-paged-attention-focus="online"><span>mi / li / oi / out</span><b>FP32</b><small>cross-block state</small></button></div></section>
      <section class="kf-inspector-section kf-pa-precision"><header><h2 class="kf-inspector-title">精度敏感点</h2><span>Agent review</span></header><div class="kf-pa-sensitivity"><article><i>01</i><div><b>Softmax 概率降精度</b><p><code>exp</code> 后先转 BF16，再转回 FP32 求和；Golden 已显式复现这一量化点。</p></div></article><article><i>02</i><div><b>Online 状态保持 FP32</b><p><code>mi/li/oi</code> 跨 Block 合并，避免长上下文累计完全落在 BF16。</p></div></article><article><i>03</i><div><b>Scale 固定为 1.0</b><p>当前实现与 Golden 一致，但不是常见的 <code>1/sqrt(D)</code>；集成真实模型时必须确认上游是否已缩放。</p></div></article><article><i>04</i><div><b>末 Block Mask</b><p>Padding 进入 exp/row_sum 会系统性污染分母，必须覆盖 <code>context_len % block_size ≠ 0</code>。</p></div></article></div></section>
      <section class="kf-inspector-section kf-pa-memory"><header><h2 class="kf-inspector-title">单 Block 工作集</h2><span>QTile16 · Block128 · D128</span></header><dl><div><dt>Q Tile</dt><dd>4 KiB · BF16</dd></div><div><dt>K / V Block</dt><dd>各 32 KiB · BF16</dd></div><div><dt>Score sij</dt><dd>8 KiB · FP32</dd></div><div><dt>Probability pij</dt><dd>4 KiB · BF16</dd></div><div><dt>oi / oi_new</dt><dd>各 8 KiB · FP32</dd></div><div><dt>mi + li</dt><dd>128 B · FP32</dd></div></dl></section>`;
  }

  function pagedAttentionTiling() {
    const blocks = Array.from({ length: 16 }, (_, index) => `<i class="${index < 4 ? 'is-hot' : ''}">${index}</i>`).join('');
    return `
      <section class="kf-inspector-section kf-pa-loop-nest"><header><h2 class="kf-inspector-title">循环与 Tile 映射</h2><span>main() 示例实例化</span></header><div class="kf-pa-loop-tree"><div><i>B</i><span><b>Batch loop</b><small>64 requests · <code>pl.range(batch_cfg)</code></small></span><em>64</em></div><div class="depth-1"><i>Q</i><span><b>Head Tile loop</b><small>ceil(16 heads ÷ QTile16)</small></span><em>1 / request</em></div><div class="depth-2"><i>K</i><span><b>KV Block loop</b><small>ceil(8192 context ÷ Block128)</small></span><em>64 / Q tile</em></div><div class="depth-3"><i>5</i><span><b>InCore chain</b><small>init once；QK → Softmax → PV → Update per block</small></span><em>16,448 calls</em></div></div></section>
      <section class="kf-inspector-section kf-pa-block-strip"><header><h2 class="kf-inspector-title">Paged Block 扫描</h2><span>64 used blocks / request</span></header><div>${blocks}</div><small>为便于阅读仅画 16 个区段；高亮区表示当前可视窗口，实际逐个 logical block 通过 block_table 映射到物理 Cache。</small></section>
      <section class="kf-pa-tile-matrix"><button type="button" data-paged-attention-focus="qk"><span>QK</span><b>16 × 128 × 128</b><small>M=QTile · N=Block · K=D</small></button><i>→</i><button type="button" data-paged-attention-focus="softmax"><span>Softmax</span><b>16 × valid_len</b><small>Vector row-wise</small></button><i>→</i><button type="button" data-paged-attention-focus="pv"><span>PV</span><b>16 × 128 × 128</b><small>M=QTile · N=D · K=Block</small></button></section>
      <section class="kf-inspector-section kf-pa-tail"><header><h2 class="kf-inspector-title">尾块与整除守卫</h2><span>coding-time checks</span></header><div><article class="is-pass"><b>KV 末 Block</b><span><code>valid_len</code> 已显式裁剪</span><em>有处理</em></article><article><b>Q Head 尾 Tile</b><span>ceil-div 后仍固定 slice q_tile</span><em>需补处理</em></article><article class="is-pass"><b>示例 Heads</b><span>16 % QTile16 = 0</span><em>安全</em></article><article><b>空 Context</b><span>bn loop 为 0，输出语义需定义</span><em>需补测试</em></article></div></section>`;
  }

  function pagedAttentionOrchestration() {
    return `
      <section class="kf-inspector-section kf-pa-scope"><header><h2 class="kf-inspector-title">Scope 层级</h2><span>1 Program · 1 Orchestration · 5 InCore</span></header><div class="kf-pa-scope-tree"><div><i>P</i><span><b>DynamicPagedAttentionProgram</b><small>Builder 返回的 @pl.program</small></span></div><div class="depth-1"><i>O</i><span><b>paged_attention</b><small>运行时维度、分页寻址、三层循环</small></span></div>${[['builder','I','init_inplace','动态形状绑定'],['qk','C','qk_matmul','Cube'],['softmax','V','softmax_prepare','Vector'],['pv','C','pv_matmul','Cube'],['online','V','online_update','Vector']].map(([focus,mark,name,role]) => `<button type="button" class="depth-2" data-paged-attention-focus="${focus}"><i>${mark}</i><span><b>${name}</b><small>${role} · InCore</small></span></button>`).join('')}</div></section>
      <section class="kf-inspector-section kf-pa-dependency"><header><h2 class="kf-inspector-title">数据依赖与状态 Carry</h2><span>Tensor-derived ordering</span></header><div class="kf-pa-dep-flow"><div><b>QK</b><small>produces sij</small></div><i>→</i><div><b>Softmax</b><small>pij · mi · li</small></div><i>→</i><div><b>PV</b><small>oi_new</small></div><i>→</i><div><b>Online Update</b><small>mi_update · li_update · oi</small></div><i class="loop">↺ next bn</i></div><p>源码没有显式 <code>pl.submit(..., deps=...)</code>，依赖主要由 Call 的 Tensor 生产/消费和 InOut 状态推导。需要在 Pass 后依赖图确认最终 Task 顺序。</p></section>
      <section class="kf-inspector-section kf-pa-parallel"><header><h2 class="kf-inspector-title">并行意图</h2><span>当前源码事实</span></header><div class="kf-pa-parallel-grid"><div><span>Batch</span><b>pl.range</b><em>未显式 parallel</em></div><div><span>Q Tile</span><b>pl.range</b><em>未显式 parallel</em></div><div><span>KV Block</span><b>pl.range</b><em>状态依赖串行</em></div><div><span>Pipeline</span><b>未声明</b><em>无 pl.pipeline</em></div></div><p>Online Softmax 的 <code>mi/li/oi</code> 形成 loop-carried dependency，因此 KV Block 不能简单并行。Batch 与 Q Tile 理论上有独立性，但当前源码未显式表达并行调度。</p></section>
      <section class="kf-inspector-section kf-attn-source-map kf-pa-source-map"><header><h2 class="kf-inspector-title">源码阶段</h2><span>点击与源码联动</span></header><div>${Object.entries(pagedAttentionFocusMeta).map(([key, item]) => `<button type="button" class="${key === state.pagedAttentionFocus ? 'is-active' : ''}" data-paged-attention-focus="${key}"><i>${item.lines}</i><span><b>${item.label}</b><small>${item.detail}</small></span></button>`).join('')}</div></section>`;
  }

  function pagedAttentionHardware() {
    return `
      <section class="kf-inspector-section kf-pa-hardware"><header><h2 class="kf-inspector-title">昇腾执行与精度路径</h2><span>A2/A3 · semantic mapping</span></header><div class="kf-pa-hw-lanes"><div class="memory"><em>GM</em><b>Query · Paged K/V · State</b><small>BF16 inputs / FP32 accumulators</small></div><i>load</i><button type="button" data-paged-attention-focus="qk"><em>CUBE</em><b>QK Matmul</b><small>L1 → L0A/L0B → L0C · FP32</small></button><i>store/load</i><button type="button" data-paged-attention-focus="softmax"><em>VECTOR</em><b>Softmax Prepare</b><small>UB · FP32 exp/sum → BF16 pij</small></button><i>store/load</i><button type="button" data-paged-attention-focus="pv"><em>CUBE</em><b>PV Matmul</b><small>BF16 inputs · FP32 oi_new</small></button><i>store/load</i><button type="button" data-paged-attention-focus="online"><em>VECTOR</em><b>Online Update</b><small>FP32 mi/li/oi · normalize output</small></button><i>store</i><div class="memory"><em>GM</em><b>Attention Output</b><small>[B × Heads, D] · FP32</small></div></div></section>
      <section class="kf-inspector-section kf-pa-precision"><header><h2 class="kf-inspector-title">关键精度边界</h2><span>source facts</span></header><div class="kf-pa-precision-grid"><div><span>Q / K / V</span><b>BF16</b><small>Matmul input</small></div><div><span>sij</span><b>FP32</b><small>QK accumulation</small></div><div><span>pij</span><b>BF16</b><small>exp 后显式 cast</small></div><div><span>mi / li / oi</span><b>FP32</b><small>online state</small></div><div><span>out</span><b>FP32</b><small>oi ÷ li</small></div></div></section>
      <section class="kf-inspector-section kf-attn-risks"><header><h2 class="kf-inspector-title">Coding 风险</h2><span>需要显式验证</span></header><ul><li><b>Q Head 尾块</b><span><code>q_loop</code> 使用 ceil-div，但 slice 仍固定为 <code>q_tile</code>；num_heads 不能整除 q_tile 时需确认有效 Shape 处理。</span></li><li><b>动态标注 ≠ 动态 Tile</b><span>InCore 类型使用 <code>pl.dynamic</code>，load 尺寸仍来自 Builder 闭包常量。</span></li><li><b>跨核数据往返</b><span>当前 5-stage InCore 管线在 A2/A3 上可能经过 GM；真实流量与重叠需结合 Pass IR、Swimlane 和 PMU。</span></li></ul></section>
      <div class="kf-inspector-card kf-rms-estimate"><b>可信边界</b><p>此图是依据 MemorySpace 与 Kernel 语义的静态映射，不代表最终指令时序和真实 Buffer 地址。</p></div>`;
  }

  function pagedAttentionValidation() {
    return `
      <section class="kf-inspector-section kf-pa-capability"><header><h2 class="kf-inspector-title">目标能力 Lens</h2><span>A2/A3 · Ascend910B</span></header><div><article class="is-supported"><i>✓</i><span><b>动态 Tensor 标注</b><small>pl.dynamic · Tensor.dim</small></span><em>源码采用</em></article><article class="is-supported"><i>✓</i><span><b>Cube Matmul</b><small>BF16 input · FP32 accumulate</small></span><em>源码采用</em></article><article class="is-supported"><i>✓</i><span><b>Vector Softmax primitives</b><small>row_max · exp · row_sum</small></span><em>源码采用</em></article><article class="is-caution"><i>!</i><span><b>动态有效宽度</b><small>sij_valid uses runtime valid_len</small></span><em>重点验证</em></article><article class="is-caution"><i>!</i><span><b>动态 Head 尾 Tile</b><small>fixed q_tile load/slice</small></span><em>能力缺口</em></article><article><i>○</i><span><b>Cube↔Vector 片上交接</b><small>A2/A3 可能经 GM Buffer</small></span><em>需 Pass/实测</em></article></div></section>
      <section class="kf-inspector-section kf-rms-validation"><header><h2 class="kf-inspector-title">当前验证设计</h2><span>源码自带 Golden</span></header><div class="kf-rms-proof"><div class="is-pass"><i>✓</i><p><b>Torch Golden 已实现</b><small>复现分页寻址、Mask 与 Online Softmax</small></p><em>直接证据</em></div><div class="is-pass"><i>✓</i><p><b>概率精度行为已对齐</b><small>pij 模拟 BF16 cast 后再转 FP32</small></p><em>直接证据</em></div><div class="is-pass"><i>✓</i><p><b>运行后执行 allclose</b><small>rtol = atol = 2e-2</small></p><em>源码门禁</em></div><div><i>○</i><p><b>动态 Shape 参数矩阵</b><small>Batch · Heads · D · Block · Context</small></p><em>缺失</em></div><div><i>○</i><p><b>Q Head 尾 Tile</b><small>num_heads % q_tile ≠ 0</small></p><em>高风险缺口</em></div><div><i>○</i><p><b>末 Block 与空 Context</b><small>valid_len · context_len 0/1/boundary</small></p><em>缺失</em></div></div></section>
      <section class="kf-inspector-section kf-attn-risks"><header><h2 class="kf-inspector-title">风险与守卫</h2><span>G · capability & risk</span></header><ul><li><b>Shape 可除性</b><span><code>query.rows % batch == 0</code>、<code>cache.rows % table.size == 0</code>、<code>table.size % batch == 0</code> 应成为显式守卫。</span></li><li><b>Page Table 合法性</b><span><code>cur_block_idx</code> 必须处于物理 Block 池范围内，否则 KV Slice 越界。</span></li><li><b>Scale 语义</b><span>固定 1.0 需要与模型调用点对齐，避免遗漏 Attention Scale。</span></li><li><b>资源与后端</b><span>片上工作集是静态估算；最终地址、GM Round Trip 和执行重叠必须读取 Pass IR、Swimlane 与 PMU。</span></li></ul></section>
      <section class="kf-inspector-section kf-pa-run"><header><h2 class="kf-inspector-title">示例运行画像</h2><span>main()</span></header><dl><div><dt>Platform / Backend</dt><dd>A2/A3 · Ascend910B</dd></div><div><dt>Batch / Heads</dt><dd>64 / 16</dd></div><div><dt>Head / Block</dt><dd>128 / 128</dd></div><div><dt>Context / Max model</dt><dd>8192 / 32768</dd></div><div><dt>Blocks / Request</dt><dd>64 used / 256 max</dd></div><div><dt>Optional evidence</dt><dd>L2 Swimlane</dd></div></dl></section>
      <button class="kf-rms-action" type="button" data-paged-attention-action="tests">＋ 生成动态 Shape 与分页边界测试</button>`;
  }

  function renderPagedAttentionInspector({ scrollToFocus = false } = {}) {
    pagedAttentionGraphController?.destroy?.();
    pagedAttentionGraphController = null;
    const tabs = { overview: '概览', data: '数据&精度', schedule: '分块与编排', validation: '风险与验证' };
    const content = state.pagedAttentionTab === 'data' ? pagedAttentionDataExecution() : state.pagedAttentionTab === 'schedule' ? pagedAttentionSchedule() : state.pagedAttentionTab === 'validation' ? pagedAttentionValidation() : pagedAttentionOverview();
    $('#inspectorTitle').textContent = 'Paged Attention 分析';
    $('#inspectorMeta').textContent = 'dynamic · online softmax';
    $('#inspector').innerHTML = `
      <section class="kf-pa-hero"><span class="kf-eyebrow">CODING AGENT · SOURCE ANALYSIS</span><div><b>paged_attention_dynamic</b><em>DYNAMIC SHAPE</em></div><small>5-stage InCore pipeline · Paged KV · online softmax</small></section>
      <div class="kf-pa-tabs" role="tablist" aria-label="动态 Paged Attention 分析视图">${Object.entries(tabs).map(([key, label]) => `<button type="button" class="${key === state.pagedAttentionTab ? 'is-active' : ''}" data-paged-attention-tab="${key}">${label}</button>`).join('')}</div>
      <div class="kf-pa-view">${content}</div>
      <footer class="kf-rms-provenance"><span><i class="fact"></i>源码事实</span><span><i class="resolved"></i>运行配置解析</span><span><i class="estimated"></i>硬件静态映射</span></footer>`;
    $$('#dslEditor [data-paged-attention-focus]').forEach(row => row.classList.toggle('is-paged-attention-line-active', row.dataset.pagedAttentionFocus === state.pagedAttentionFocus));
    if (scrollToFocus) $(`#dslEditor [data-paged-attention-focus="${state.pagedAttentionFocus}"]`)?.scrollIntoView({ block: 'center', behavior: 'smooth' });
    if (state.pagedAttentionTab === 'overview') renderPagedAttentionComputationGraph();
  }

  function renderPagedAttentionComputationGraph() {
    const pattern = window.PtoModelGraphvizPattern;
    const stage = $('#pagedAttentionComputationGraph');
    const status = $('#pagedAttentionGraphStatus');
    if (!pattern || !stage) return;
    const overlays = {
      precision: {
        'pa-query': ['BF16 · input', 'io:activation'], 'pa-context': ['INT32 · control', 'io:state'], 'pa-table': ['INT32 · index', 'io:state'], 'pa-page': ['INDEX · address math', 'io:state'], 'pa-kv': ['BF16 · input', 'io:activation'], 'pa-qk': ['BF16 × BF16 → FP32', 'sem:linear'], 'pa-mask': ['FP32 · valid width', 'sem:comm'], 'pa-softmax': ['FP32 compute → BF16 pij', 'sem:softmax'], 'pa-pv': ['BF16 × BF16 → FP32', 'sem:linear'], 'pa-online': ['FP32 mi / li / oi', 'sem:softmax'], 'pa-out': ['FP32 · output', 'io:output'],
      },
      shape: {
        'pa-query': ['[B×H, D]', 'io:activation'], 'pa-context': ['[B]', 'io:state'], 'pa-table': ['[B×MaxBlocks]', 'io:state'], 'pa-page': ['scalar block_id → row', 'sem:comm'], 'pa-kv': ['[Block, D] × 2', 'io:state'], 'pa-qk': ['[QTile, Block]', 'sem:linear'], 'pa-mask': ['[QTile, valid_len]', 'sem:comm'], 'pa-softmax': ['pij [Q,B] · state [Q,1]', 'sem:softmax'], 'pa-pv': ['[QTile, D]', 'sem:linear'], 'pa-online': ['state [QTile,1/D]', 'sem:softmax'], 'pa-out': ['[B×H, D]', 'io:output'],
      },
      hardware: {
        'pa-query': ['GM · load', 'io:state'], 'pa-context': ['Orchestration · scalar read', 'io:state'], 'pa-table': ['Orchestration · scalar read', 'io:state'], 'pa-page': ['Orchestration · address', 'sem:comm'], 'pa-kv': ['GM · paged block', 'io:state'], 'pa-qk': ['CUBE · L1 → L0', 'sem:linear'], 'pa-mask': ['Tensor slice · GM view', 'sem:comm'], 'pa-softmax': ['VECTOR · UB', 'sem:softmax'], 'pa-pv': ['CUBE · L1 → L0', 'sem:linear'], 'pa-online': ['VECTOR · UB', 'sem:softmax'], 'pa-out': ['GM · store', 'io:output'],
      },
    };
    const overlay = overlays[state.pagedAttentionOverlay] || overlays.precision;
    const tensorOverlayLabels = {
      precision: { 'pa-query': 'BF16', 'pa-context': 'INT32', 'pa-table': 'INT32', 'pa-out': 'FP32' },
      shape: { 'pa-query': '[B×H,D]', 'pa-context': '[B]', 'pa-table': '[B×M]', 'pa-out': '[B×H,D]' },
      hardware: { 'pa-query': 'GM', 'pa-context': 'ORCH', 'pa-table': 'ORCH', 'pa-out': 'GM' },
    }[state.pagedAttentionOverlay] || {};
    const expandedId = pagedAttentionDrilldowns[state.pagedAttentionExpandedNode] ? state.pagedAttentionExpandedNode : null;
    const expandedSpec = expandedId ? pagedAttentionDrilldowns[expandedId] : null;
    const expandedBaseNode = expandedId ? pagedAttentionComputationGraph.nodes.find((node) => node.id === expandedId) : null;
    const expansionShift = expandedId ? 180 : 0;
    const expandableIds = new Set(Object.keys(pagedAttentionDrilldowns));
    const baseNodes = pagedAttentionComputationGraph.nodes
      .filter((node) => node.id !== expandedId)
      .map((node) => ({
        ...node,
        y: expandedBaseNode && node.y > expandedBaseNode.y ? node.y + expansionShift : node.y,
        height: node.kind === 'tensor' ? node.height : Math.max(72, node.height),
        collapsed: expandableIds.has(node.id),
        label: node.kind === 'tensor' && tensorOverlayLabels[node.id] ? `${node.label} · ${tensorOverlayLabels[node.id]}` : node.label,
        typeLabel: overlay[node.id]?.[0] || node.typeLabel,
        colorKey: overlay[node.id]?.[1] || node.colorKey,
      }));
    const drillNodes = [];
    const drillEdges = [];
    const drillClusters = [];
    const childFocusMap = new Map();
    let firstDrillNodeId = null;
    let lastDrillNodeId = null;
    if (expandedId && expandedSpec && expandedBaseNode) {
      const clusterId = `${expandedId}-detail`;
      const parentColor = overlay[expandedId]?.[1] || expandedBaseNode.colorKey;
      expandedSpec.children.forEach((child, index) => {
        const childId = `${expandedId}-${child.key}`;
        if (!firstDrillNodeId) firstDrillNodeId = childId;
        childFocusMap.set(childId, expandedSpec.focus);
        drillNodes.push({
          id: childId,
          label: child.label,
          typeLabel: child[state.pagedAttentionOverlay] || child.precision,
          kind: 'op',
          x: expandedBaseNode.x,
          y: expandedBaseNode.y + index * 82,
          width: 244,
          height: 66,
          colorKey: parentColor,
          overlayKind: 'drilldown',
          parent: clusterId,
        });
        if (index > 0) {
          drillEdges.push({
            source: `${expandedId}-${expandedSpec.children[index - 1].key}`,
            target: childId,
            tag: null,
          });
        }
        lastDrillNodeId = childId;
      });
      drillClusters.push({
        id: clusterId,
        label: `${expandedBaseNode.label} · 细粒度`,
        x: expandedBaseNode.x - 150,
        y: expandedBaseNode.y - 52,
        width: 300,
        height: 270,
        colorKey: parentColor,
        nodes: drillNodes.map((node) => node.id),
      });
    }
    const graphNodes = [...baseNodes, ...drillNodes];
    const graphEdges = pagedAttentionComputationGraph.edges.map((edge) => {
      if (!expandedId || !firstDrillNodeId || !lastDrillNodeId) return { ...edge };
      return {
        ...edge,
        source: edge.source === expandedId ? lastDrillNodeId : edge.source,
        target: edge.target === expandedId ? firstDrillNodeId : edge.target,
      };
    }).concat(drillEdges);
    const nodeMap = new Map(graphNodes.map((node) => [node.id, node]));
    const orthogonalEdges = graphEdges.map((edge) => {
      const source = nodeMap.get(edge.source);
      const target = nodeMap.get(edge.target);
      if (!source || !target) return { ...edge };
      if (source.id === target.id) {
        const loopX = source.x + source.width / 2 + 50;
        const loopY = source.y + source.height / 2 + 42;
        return {
          ...edge,
          sourceAnchor: 'right',
          targetAnchor: 'bottom',
          waypoints: [{ x: loopX, y: source.y }, { x: loopX, y: loopY }, { x: source.x, y: loopY }],
          cornerRadius: 10,
        };
      }
      const vertical = Math.abs(source.y - target.y) >= Math.abs(source.x - target.x);
      if (vertical) {
        const downward = source.y < target.y;
        const startY = source.y + (downward ? source.height / 2 : -source.height / 2);
        const endY = target.y + (downward ? -target.height / 2 : target.height / 2);
        const midY = (startY + endY) / 2;
        return {
          ...edge,
          sourceAnchor: downward ? 'bottom' : 'top',
          targetAnchor: downward ? 'top' : 'bottom',
          waypoints: [{ x: source.x, y: midY }, { x: target.x, y: midY }],
          cornerRadius: 10,
        };
      }
      const rightward = source.x < target.x;
      const startX = source.x + (rightward ? source.width / 2 : -source.width / 2);
      const endX = target.x + (rightward ? -target.width / 2 : target.width / 2);
      const midX = (startX + endX) / 2;
      return {
        ...edge,
        sourceAnchor: rightward ? 'right' : 'left',
        targetAnchor: rightward ? 'left' : 'right',
        waypoints: [{ x: midX, y: source.y }, { x: midX, y: target.y }],
        cornerRadius: 10,
      };
    });
    const graph = {
      ...pagedAttentionComputationGraph,
      height: pagedAttentionComputationGraph.height + expansionShift,
      clusters: drillClusters,
      nodes: graphNodes,
      edges: orthogonalEdges,
    };
    stage.classList.toggle('is-expanded', Boolean(expandedId));
    pagedAttentionGraphController = pattern.renderController(stage, graph, {
      ariaLabel: 'Dynamic paged attention with page lookup, QK, softmax, PV and online update',
      colormap: pattern.modelArchitectureColormap(graph),
      fitMode: 'full', viewportPadding: 18, autoFit: true,
      interaction: { panZoom: true, selectableClusters: false },
      overlays: { edgeTags: true },
      onSelect: ({ nodeId }) => {
        if (pagedAttentionDrilldowns[nodeId]) {
          state.pagedAttentionExpandedNode = state.pagedAttentionExpandedNode === nodeId ? null : nodeId;
          state.pagedAttentionFocus = pagedAttentionDrilldowns[nodeId].focus;
          pagedAttentionGraphController?.destroy?.();
          pagedAttentionGraphController = null;
          renderPagedAttentionComputationGraph();
          return;
        }
        const focus = pagedAttentionGraphFocus[nodeId] || childFocusMap.get(nodeId);
        if (!focus) return;
        state.pagedAttentionFocus = focus;
        $$('#dslEditor [data-paged-attention-focus]').forEach(row => row.classList.toggle('is-paged-attention-line-active', row.dataset.pagedAttentionFocus === focus));
        const meta = pagedAttentionFocusMeta[focus];
        if (status && meta) status.textContent = `${meta.label} · 源码第 ${meta.lines} 行 · ${meta.detail}`;
      },
    });
    if (expandedId) {
      const detailCluster = stage.querySelector(`[data-cluster-id="${expandedId}-detail"]`);
      detailCluster?.addEventListener('click', (event) => {
        if (event.target.closest('.pto-model-graphviz-node')) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        state.pagedAttentionExpandedNode = null;
        pagedAttentionGraphController?.destroy?.();
        pagedAttentionGraphController = null;
        renderPagedAttentionComputationGraph();
      }, true);
      if (status) status.textContent = `${pagedAttentionFocusMeta[expandedSpec.focus].label} 已展开 · 点击 − 收起 · 当前显示${state.pagedAttentionOverlay === 'precision' ? '精度' : state.pagedAttentionOverlay === 'shape' ? 'Shape' : '硬件'}信息`;
    }
  }
  const rmsNormProfiles = {
    input: {
      id: 'input', name: 'input_rmsnorm', role: 'Attention 前', scope: 'CORE_GROUP · rmsnorm', source: 'hidden_states', sourceType: 'BF16', weight: 'input_rms_weight', output: 'normed_states', chunk: 512, chunks: 16, stage: 4,
      cast: 'BF16 → FP32 → BF16', chunkBytes: '32 KiB', inputBytes: '256 KiB', scanBytes: '512 KiB', line: 27,
      upstream: 'hidden_states', downstream: 'Q / K / V projection', note: '输入为 BF16；两遍都先将当前 chunk 转为 FP32，再完成平方和与归一化。'
    },
    post: {
      id: 'post', name: 'post_rmsnorm', role: 'Attention 后 · MLP 前', scope: 'CORE_GROUP · post_rmsnorm', source: 'resid', sourceType: 'FP32', weight: 'post_rms_weight', output: 'post_norm_tile', chunk: 128, chunks: 64, stage: 2,
      cast: 'FP32 → BF16', chunkBytes: '8 KiB', inputBytes: '512 KiB', scanBytes: '1 MiB', line: 57,
      upstream: 'out_projection_residual', downstream: 'mlp_block', note: '残差流已经是 FP32，因此两遍扫描都不需要输入 cast；只在 assemble 前转为 BF16。'
    }
  };
  const rmsNormExecutionSteps = {
    input: [
      { id: 'load', index: '01', title: '载入并升精度', detail: 'DDR · BF16 → UB · FP32', lines: [38, 49, 50], selectors: ['[data-mem950-node="rail:DDR"]', '#rmsnorm-aiv-core [data-aiv-node="cache:ND-DMA Cache"]', '#rmsnorm-aiv-core [data-aiv-node="buffer:UB"]'], routes: ['rmsnorm-load'] },
      { id: 'reduce', index: '02', title: '平方与行归约', detail: 'Vector · FP32 accumulate', lines: [39, 40, 41, 42], selectors: ['#rmsnorm-aiv-core [data-aiv-node="buffer:UB"]', '#rmsnorm-aiv-core [data-aiv-node="vector:Vector"]'] },
      { id: 'normalize', index: '03', title: '计算 inv_rms 并归一化', detail: 'Vector · FP32', lines: [44, 45, 51], selectors: ['#rmsnorm-aiv-core [data-aiv-node="buffer:UB"]', '#rmsnorm-aiv-core [data-aiv-node="vector:Vector"]'] },
      { id: 'store', index: '04', title: '降精度并写回', detail: 'UB · BF16 → DDR', lines: [52], selectors: ['#rmsnorm-aiv-core [data-aiv-node="buffer:UB"]', '[data-mem950-node="rail:DDR"]'], routes: ['rmsnorm-store'] },
    ],
    post: [
      { id: 'load', index: '01', title: '载入残差与 Gamma', detail: 'DDR · FP32 → UB · FP32', lines: [74, 85, 86], selectors: ['[data-mem950-node="rail:DDR"]', '#rmsnorm-aiv-core [data-aiv-node="cache:ND-DMA Cache"]', '#rmsnorm-aiv-core [data-aiv-node="buffer:UB"]'], routes: ['rmsnorm-load'] },
      { id: 'reduce', index: '02', title: '平方与行归约', detail: 'Vector · FP32 accumulate', lines: [75, 76, 77, 78], selectors: ['#rmsnorm-aiv-core [data-aiv-node="buffer:UB"]', '#rmsnorm-aiv-core [data-aiv-node="vector:Vector"]'] },
      { id: 'normalize', index: '03', title: '计算 inv_rms 并归一化', detail: 'Vector · FP32', lines: [80, 81, 87], selectors: ['#rmsnorm-aiv-core [data-aiv-node="buffer:UB"]', '#rmsnorm-aiv-core [data-aiv-node="vector:Vector"]'] },
      { id: 'store', index: '04', title: '降精度并写回', detail: 'UB · BF16 → DDR', lines: [88], selectors: ['#rmsnorm-aiv-core [data-aiv-node="buffer:UB"]', '[data-mem950-node="rail:DDR"]'], routes: ['rmsnorm-store'] },
    ],
  };

  function rmsNormOverview(profile) {
    return `
      <section class="kf-rms-flow" aria-label="模型上下文"><span>${profile.upstream}</span><i>→</i><b>${profile.name}</b><i>→</i><span>${profile.downstream}</span></section>
      <section class="kf-inspector-section kf-intent-detail kf-rms-contract"><header><h2>算子契约</h2><span>调用点解析</span></header><dl><div><dt>${profile.source}</dt><dd>[16, 8192] · ${profile.sourceType}</dd></div><div><dt>${profile.weight}</dt><dd>[1, 8192] · FP32</dd></div><div><dt>${profile.output}</dt><dd>[16, 8192] · BF16</dd></div><div><dt>执行形态</dt><dd>${profile.scope}</dd></div></dl></section>
      <section class="kf-inspector-section kf-rms-compare"><header><h2 class="kf-inspector-title">双变体对比</h2><span>同语义 · 不同调度</span></header><div class="kf-rms-compare-grid"><div class="head"><span></span><b>Input</b><b>Post</b></div><div><span>Chunk</span><b class="${profile.id === 'input' ? 'is-current' : ''}">512</b><b class="${profile.id === 'post' ? 'is-current' : ''}">128</b></div><div><span>Chunks / pass</span><b class="${profile.id === 'input' ? 'is-current' : ''}">16</b><b class="${profile.id === 'post' ? 'is-current' : ''}">64</b></div><div><span>Pipeline stage</span><b class="${profile.id === 'input' ? 'is-current' : ''}">4</b><b class="${profile.id === 'post' ? 'is-current' : ''}">2</b></div><div><span>Input cast</span><b>BF16→FP32</b><b>无</b></div></div></section>
      <div class="kf-inspector-card kf-rms-insight"><b>Agent 结论</b><p>两个函数共享两遍 RMSNorm 数学结构，但 chunk、stage 和输入精度不同。修改归约逻辑时应同步检查两处，不应直接统一调度参数。</p></div>`;
  }

  function rmsNormPrecision(profile) {
    const firstCast = profile.id === 'input' ? '<span>BF16 chunk</span><i>cast</i><span>FP32 [16,512]</span>' : '<span>FP32 chunk</span><i>直接计算</i><span>FP32 [16,128]</span>';
    const steps = rmsNormExecutionSteps[profile.id];
    return `
      <section class="kf-inspector-section kf-rms-precision"><header><h2 class="kf-inspector-title">精度流</h2><span>源码事实</span></header><div class="kf-rms-precision-flow">${firstCast}<i>square + row_sum</i><span>FP32 [1,16]</span><i>recip(sqrt)</i><span>FP32 inv_rms</span><i>× gamma · cast</i><span>BF16 output</span></div></section>
      <section class="kf-rms-hardware" aria-labelledby="rmsNormHardwareTitle">
        <header><div><b id="rmsNormHardwareTitle">昇腾执行路径</b><span>DDR → UB ⇄ Vector → DDR</span></div><div class="kf-rms-hardware__tools" data-no-pan><button type="button" data-rms-fit aria-pressed="true">最佳视图</button><button type="button" data-rms-zoom="out" aria-label="缩小硬件图">−</button><span data-rms-zoom-readout>—</span><button type="button" data-rms-zoom="in" aria-label="放大硬件图">＋</button></div></header>
        <div class="pto-memory-architecture-viewport kf-rms-hardware__viewport" id="rmsNormHardwareViewport" data-pto-mem-arch-viewport><div class="pto-memory-architecture-sizer" id="rmsNormHardwareSizer" data-pto-mem-arch-sizer><div class="pto-memory-architecture-canvas" id="rmsNormHardwareGraph" data-pto-mem-arch-canvas></div></div></div>
        <div class="kf-rms-hardware__steps" role="list" aria-label="RMSNorm 执行阶段">${steps.map((step) => `<button type="button" role="listitem" data-rms-flow-step="${step.id}"><i>${step.index}</i><span><b>${step.title}</b><small>${step.detail}</small></span></button>`).join('')}</div>
        <footer id="rmsNormFlowStatus"><span><i></i>点击阶段查看数据路径与对应源码</span></footer>
      </section>
      <section class="kf-inspector-section kf-intent-detail"><header><h2>逻辑工作集</h2><span>EST. · 静态 shape</span></header><dl><div><dt>单个计算 Chunk</dt><dd>${profile.chunkBytes} · FP32</dd></div><div><dt>完整输入</dt><dd>${profile.inputBytes} · ${profile.sourceType}</dd></div><div><dt>Gamma</dt><dd>32 KiB · FP32</dd></div><div><dt>输出</dt><dd>256 KiB · BF16</dd></div><div><dt>两遍输入读取</dt><dd>${profile.scanBytes}</dd></div></dl></section>
      <div class="kf-inspector-card kf-rms-estimate"><b>可信边界</b><p>硬件路径由源码算子语义静态映射，用于解释数据流转，不代表实际指令时序；逻辑字节数不包含 Tile 对齐、临时缓冲与后端地址分配。真实片上占用和搬运指令需读取 Pass 后 IR。</p></div>`;
  }

  function rmsNormTiling(profile) {
    const blocks = Array.from({ length: profile.id === 'input' ? 16 : 32 }, () => '<i></i>').join('');
    return `
      <section class="kf-inspector-section kf-rms-tiling"><header><h2 class="kf-inspector-title">Hidden 分块</h2><span>8192 = ${profile.chunks} × ${profile.chunk}</span></header><div class="kf-rms-blocks ${profile.id === 'post' ? 'is-dense' : ''}" title="${profile.chunks} chunks / pass">${blocks}</div><small>${profile.id === 'post' ? '每 2 个可视块代表 4 个 128-element chunks' : '每个可视块代表 1 个 512-element chunk'}</small></section>
      <section class="kf-inspector-section kf-intent-detail"><header><h2>流水摘要</h2><span>两遍扫描</span></header><dl><div><dt>Chunk</dt><dd>${profile.chunk}</dd></div><div><dt>迭代 / pass</dt><dd>${profile.chunks}</dd></div><div><dt>Chunk 处理总次数</dt><dd>${profile.chunks * 2}</dd></div><div><dt>Pipeline stage</dt><dd>${profile.stage}</dd></div><div><dt>尾块</dt><dd class="kf-rms-good">✓ 无</dd></div></dl></section>
      <section class="kf-rms-two-pass"><div><span>PASS A</span><b>平方 · 行归约 · 累加</b></div><i>inv_rms</i><div><span>PASS B</span><b>归一化 · gamma · assemble</b></div></section>
      <div class="kf-inspector-card kf-rms-insight"><b>整除性守卫</b><p>✓ 8192 % ${profile.chunk} = 0　✓ assemble offset 与 chunk 对齐。若 HIDDEN 改为不可整除值，需要补 valid_shape 或尾块处理。</p></div>`;
  }

  function rmsNormValidation() {
    return `
      <section class="kf-inspector-section kf-rms-validation"><header><h2 class="kf-inspector-title">当前证据</h2><span>结构 ≠ 数值</span></header><div class="kf-rms-proof"><div class="is-pass"><i>✓</i><p><b>Qwen3 JIT 完整管线可编译</b><small>tests/ut/jit/test_qwen3_decode.py</small></p><em>已验证</em></div><div class="is-pass"><i>✓</i><p><b>两个 scope 均被 outline</b><small>rmsnorm · post_rmsnorm</small></p><em>已验证</em></div><div class="is-related"><i>≈</i><p><b>通用 RMSNorm 数值 ST</b><small>不同 shape / EPS / kernel</small></p><em>间接证据</em></div><div><i>○</i><p><b>当前两个函数的 Torch golden</b><small>Qwen3 shape · BF16 edge</small></p><em>缺失</em></div><div><i>○</i><p><b>逐 Pass 数值验证</b><small>定位首个语义偏差</small></p><em>缺失</em></div><div><i>○</i><p><b>Chunk / stage 性能对比</b><small>benchmark · PMU · trace</small></p><em>缺失</em></div></div></section>
      <div class="kf-inspector-card kf-rms-estimate"><b>可信边界</b><p>现有 Qwen3 测试证明 inline 与 outline 结构成立；通用 RMSNorm ST 不能直接证明这里的两个 Qwen3 实现数值正确。</p></div>
      <button class="kf-rms-action" type="button" data-rms-action="golden">＋ 生成当前 Kernel 数值测试</button>`;
  }

  function renderRmsNormHardwareGraph(profile) {
    const memoryPattern = window.PtoMemoryArchitecturePattern;
    const canvas = $('#rmsNormHardwareGraph');
    const viewport = $('#rmsNormHardwareViewport');
    const sizer = $('#rmsNormHardwareSizer');
    const host = $('.kf-rms-hardware');
    const status = $('#rmsNormFlowStatus');
    const fitButton = $('[data-rms-fit]', host);
    const readout = $('[data-rms-zoom-readout]', host);
    const steps = rmsNormExecutionSteps[profile.id] || [];
    if (!memoryPattern || !canvas || !viewport || !sizer) return;

    memoryPattern.renderArchitecture(canvas, rmsNormHardwarePreset);
    memoryPattern.setBufferBlocks?.(canvas, [
      { core: 'rmsnorm-aiv-core', buffer: 'UB', label: `${profile.source} · ${profile.sourceType}`, state: 'enqueued', tone: 'input', cellRange: [0, 23], sourceTile: `${profile.source}[:, k0:k0+${profile.chunk}]` },
      { core: 'rmsnorm-aiv-core', buffer: 'UB', label: 'partial_sq · FP32', state: 'accumulating', tone: 'accumulator', cellRange: [26, 35], sourceTile: '[1, 16]' },
      { core: 'rmsnorm-aiv-core', buffer: 'UB', label: 'output · BF16', state: 'committed', tone: 'output', cellRange: [40, 55], sourceTile: `${profile.output}[:, k0:k0+${profile.chunk}]` },
    ]);
    const routes = memoryPattern.createRouteOverlay(canvas, rmsNormHardwarePreset);
    const hover = memoryPattern.attachHoverInteractions(canvas, rmsNormHardwarePreset, {
      selector: '[data-mem950-node="rail:DDR"], #rmsnorm-aiv-core, #rmsnorm-aiv-core [data-aiv-node]',
    });
    let fitZoom = 0;
    const syncFitState = (currentZoom) => {
      const isFit = fitZoom > 0 && Math.abs(currentZoom - fitZoom) < 0.006;
      fitButton?.classList.toggle('is-active', isFit);
      fitButton?.setAttribute('aria-pressed', String(isFit));
    };
    const zoom = memoryPattern.createZoomController({
      root: $('#inspector'), viewport, sizer, canvas,
      defaultZoom: 0.36, min: 0.16, max: 1.2, step: 0.08,
      pan: true, wheelZoom: false, centerTarget: '.pto-mem950__layout',
      outButton: '[data-rms-zoom="out"]', inButton: '[data-rms-zoom="in"]', readout,
      onZoom: ({ zoom: currentZoom }) => syncFitState(currentZoom),
    });
    const fit = () => {
      const graph = canvas.querySelector('.pto-mem950');
      if (!graph) return;
      const widthScale = (viewport.clientWidth - 16) / Math.max(graph.scrollWidth, 1);
      const heightScale = (viewport.clientHeight - 16) / Math.max(graph.scrollHeight, 1);
      fitZoom = Math.max(0.16, Math.min(1.05, widthScale, heightScale));
      zoom?.setZoom(fitZoom);
      zoom?.center();
      routes?.render();
      syncFitState(zoom?.getZoom() || fitZoom);
    };
    const activateStep = (stepId, { scroll = false } = {}) => {
      const step = steps.find((item) => item.id === stepId);
      if (!step) return;
      state.rmsNormFlowStep = step.id;
      memoryPattern.setPathFocus(canvas, rmsNormHardwarePreset, step);
      host?.classList.add('is-code-flowing');
      $$('[data-rms-flow-step]', host).forEach((button) => button.classList.toggle('is-active', button.dataset.rmsFlowStep === step.id));
      $$('#dslEditor [data-rms-line]').forEach((row) => row.classList.toggle('is-rms-execution-line', step.lines.includes(Number(row.dataset.rmsLine))));
      if (status) status.innerHTML = `<span><i></i>${step.title} · 源码第 ${step.lines.join('、')} 行</span>`;
      if (scroll) $(`#dslEditor [data-rms-line="${step.lines[0]}"]`)?.scrollIntoView({ block: 'center', behavior: 'smooth' });
    };
    const onStepClick = (event) => {
      const button = event.target.closest('[data-rms-flow-step]');
      if (button) activateStep(button.dataset.rmsFlowStep, { scroll: true });
    };
    fitButton?.addEventListener('click', fit);
    host?.addEventListener('click', onStepClick);
    const fitObserver = typeof ResizeObserver === 'function' ? new ResizeObserver(fit) : null;
    fitObserver?.observe(viewport);
    requestAnimationFrame(() => { fit(); activateStep(state.rmsNormFlowStep); });

    rmsNormHardwareGraphInstance = {
      activateStep,
      destroy() {
        fitButton?.removeEventListener('click', fit);
        host?.removeEventListener('click', onStepClick);
        fitObserver?.disconnect();
        routes?.destroy?.();
        hover?.destroy?.();
        zoom?.destroy?.();
        $$('#dslEditor [data-rms-line]').forEach((row) => row.classList.remove('is-rms-execution-line'));
      },
    };
  }

  function renderRmsNormInspector({ scrollToFunction = false } = {}) {
    rmsNormHardwareGraphInstance?.destroy?.();
    rmsNormHardwareGraphInstance = null;
    const profile = rmsNormProfiles[state.rmsNormFunction] || rmsNormProfiles.input;
    const tabLabels = { overview: '概览', precision: '数据与精度', tiling: '分块流水', validation: '验证' };
    const content = state.rmsNormTab === 'precision' ? rmsNormPrecision(profile) : state.rmsNormTab === 'tiling' ? rmsNormTiling(profile) : state.rmsNormTab === 'validation' ? rmsNormValidation() : rmsNormOverview(profile);
    $('#inspectorTitle').textContent = 'RMSNorm 分析';
    $('#inspectorMeta').textContent = `${profile.name} · static`;
    $('#inspector').innerHTML = `
      <section class="kf-rms-hero"><span class="kf-eyebrow">CODING AGENT · SOURCE ANALYSIS</span><b>RMSNorm</b><small>x / sqrt(mean(x²) + 1e-6) × gamma</small><div class="kf-rms-function-switch" role="group" aria-label="RMSNorm 函数">${Object.values(rmsNormProfiles).map(item => `<button type="button" class="${item.id === profile.id ? 'is-active' : ''}" data-rms-function="${item.id}"><b>${item.name}</b><small>${item.role}</small></button>`).join('')}</div></section>
      <div class="kf-rms-tabs" role="tablist" aria-label="RMSNorm 分析视图">${Object.entries(tabLabels).map(([key, label]) => `<button type="button" class="${key === state.rmsNormTab ? 'is-active' : ''}" data-rms-tab="${key}">${label}</button>`).join('')}</div>
      <div class="kf-rms-view">${content}</div>
      <footer class="kf-rms-provenance"><span><i class="fact"></i>源码事实</span><span><i class="resolved"></i>跨文件解析</span><span><i class="estimated"></i>静态估算</span></footer>`;
    $$('#dslEditor [data-rms-function]').forEach(row => row.classList.toggle('is-rms-function-active', row.dataset.rmsFunction === profile.id));
    if (scrollToFunction) $(`#dslEditor [data-rms-line="${profile.line}"]`)?.scrollIntoView({ block: 'center', behavior: 'smooth' });
    if (state.rmsNormTab === 'precision') renderRmsNormHardwareGraph(profile);
  }

  function renderIntentInspector() {
    matmulHardwareGraphInstance?.destroy?.();
    matmulHardwareGraphInstance = null;
    rmsNormHardwareGraphInstance?.destroy?.();
    rmsNormHardwareGraphInstance = null;
    attentionGraphController?.destroy?.();
    attentionGraphController = null;
    qwenDecodeGraphController?.destroy?.();
    qwenDecodeGraphController = null;
    passesGraphInstance?.destroy?.();
    passesGraphInstance = null;
    if (isPassesDumpFile(state.activeFile)) {
      renderPassesGraphInspector();
      return;
    }
    if (state.activeFile === ATTENTION_FILE) {
      renderAttentionInspector();
      return;
    }
    if (state.activeFile === QWEN_DECODE_FILE) {
      renderQwenDecodeInspector();
      return;
    }
    if (isPagedAttentionFile(state.activeFile)) {
      renderPagedAttentionInspector();
      return;
    }
    if (state.activeFile === RMSNORM_FILE) {
      renderRmsNormInspector();
      return;
    }
    if (state.activeFile === 'matmul.py') {
      $('#inspectorTitle').textContent = '意图预览';
      $('#inspectorMeta').textContent = 'matmul.py';
      $('#inspector').innerHTML = `
        <section class="kf-intent-hero"><span class="kf-eyebrow">CURRENT OPERATOR</span><b>mm</b><small>kernels/matmul.py · in-core matrix multiply</small></section>
        <section class="kf-inspector-section kf-intent-detail"><header><h2>Shape</h2><span>32 × 32 · contracted</span></header><dl><div><dt>Input A</dt><dd>[32, 32] · FP16</dd></div><div><dt>Input B</dt><dd>[32, 32] · FP16</dd></div><div><dt>Output</dt><dd>[32, 32] · FP32</dd></div></dl></section>
        <section class="kf-matmul-hardware" aria-labelledby="matmulHardwareTitle">
          <header><b id="matmulHardwareTitle">内存路径</b><div class="kf-matmul-hardware__tools" data-no-pan><button type="button" data-matmul-fit aria-pressed="true">最佳视图</button><button type="button" data-matmul-actual aria-pressed="false">100%</button><span data-matmul-zoom-readout>—</span></div></header>
          <div class="pto-memory-architecture-viewport kf-matmul-hardware__viewport" id="matmulHardwareViewport" data-pto-mem-arch-viewport>
            <div class="pto-memory-architecture-sizer" id="matmulHardwareSizer" data-pto-mem-arch-sizer>
              <div class="pto-memory-architecture-canvas" id="matmulHardwareGraph" data-pto-mem-arch-canvas></div>
            </div>
          </div>
          <footer id="matmulFlowStatus"><span><i></i>悬停源码行查看数据流，点击可锁定</span></footer>
        </section>
         `;
      renderMatmulHardwareGraph();
      return;
    }
    const active = intentPreview[state.intentTab] || intentPreview.shape;
    const tabs = Object.entries(intentPreview).map(([key, item]) => `<button type="button" class="${key === state.intentTab ? 'is-active' : ''}" data-intent-tab="${key}">${item.label}</button>`).join('');
    $('#inspectorTitle').textContent = '意图预览';
    $('#inspectorMeta').textContent = 'decode_layer.py';
    $('#inspector').innerHTML = `
      <section class="kf-intent-hero"><div class="kf-intent-title-row"><b>decode_layer</b><span class="kf-inference-badge">推理</span><span class="kf-megakernel-badge">megakernel</span></div></section>
      <div class="kf-intent-tabs" role="tablist" aria-label="算子意图类型">${tabs}</div>
      <section class="kf-inspector-section kf-intent-detail"><header><h2>${active.label}</h2><span>${active.meta}</span></header><dl>${active.rows.map(row => `<div><dt>${row[0]}</dt><dd>${row[1]}</dd></div>`).join('')}</dl></section>
       `;
  }

  // Ordered passes_dump files (for the evolution/diff timeline).
  function passesDumpList() {
    const passes = window.PTO_PASSES_DUMP_SOURCES || {};
    return Object.keys(passes).map((name) => ({ name, text: passes[name] }));
  }

  function renderPassesGraphInspector() {
    const file = state.activeFile;
    const mode = state.passesGraphMode;
    const compare = mode === 'compare';
    $('#inspectorTitle').textContent = compare ? '计算图演进对比' : '计算图';
    $('#inspectorMeta').textContent = compare ? 'passes_dump · 全部' : file;
    const sectionSub = compare ? 'Pass 间增删变化' : 'name_hint · deps 链';
    const note = compare
      ? '对比 <b>各 Pass</b> 后计算图的结构变化：<span class="kf-cg-add-txt">绿色</span>为新增，<span class="kf-cg-del-txt">红色虚线</span>为被删除并即将消失的节点/边。点击步骤或“播放”查看动态演进。'
      : '每个节点对应一个 <code>pl.at</code> 任务（name_hint），连线表示 <code>deps</code> 依赖；层级按最长依赖路径排布，<em>上一层输出</em>为跨层 carry 输入。';
    $('#inspector').innerHTML = `
      <section class="kf-intent-hero"><span class="kf-eyebrow">COMPUTATION GRAPH</span><b>_jit_decode_fwd_layers</b><small>passes_dump/${compare ? '00 → 01 → 02' : file} · 由 IR 任务依赖推导</small></section>
      <section class="kf-inspector-section kf-cg-section"><header class="kf-cg-head"><h2 class="kf-inspector-title">任务依赖图</h2><span class="kf-cg-mode" role="group" aria-label="计算图模式"><button type="button" data-cg-mode="single"${!compare ? ' class="is-active"' : ''}>单图</button><button type="button" data-cg-mode="compare"${compare ? ' class="is-active"' : ''}>对比</button></span></header><div class="kf-cg-mount" id="passesGraphMount"></div></section>
      <div class="kf-inspector-card kf-intent-note"><b>如何解读</b><p>${note}</p></div>`;
    const graphApi = window.PtoPassesGraph;
    const mount = $('#passesGraphMount');
    if (!graphApi || !mount) { if (mount) mount.innerHTML = '<code>计算图渲染模块未加载</code>'; return; }

    if (compare) {
      const list = passesDumpList();
      // Default the timeline to the transition that produced the active file.
      const activeIdx = Math.max(0, list.findIndex((p) => p.name === file));
      passesGraphInstance = graphApi.buildAndCompare(mount, list, { startStep: activeIdx });
    } else {
      passesGraphInstance = graphApi.buildAndRender(mount, resolveSource(file), {
        onSelect: (node) => highlightSourceLine(node.line, { scroll: true }),
        onClear: () => clearSourceLineHighlight(),
        onHover: (line) => hoverSourceLine(line)
      });
      bindSourceToGraph(passesGraphInstance);
    }
  }

  function setPassesGraphMode(mode) {
    if (state.passesGraphMode === mode) return;
    state.passesGraphMode = mode;
    clearSourceLineHighlight();
    passesGraphInstance?.destroy?.();
    passesGraphInstance = null;
    renderPassesGraphInspector();
  }

  // ---- source ⇄ computation-graph line mapping ----
  function editorRowByLine(line) {
    if (!line) return null;
    return $(`#dslEditor [data-passes-line="${line}"]`);
  }

  function clearSourceLineHighlight() {
    $$('#dslEditor .is-cg-active').forEach((row) => row.classList.remove('is-cg-active'));
  }

  function hoverSourceLine(line) {
    $$('#dslEditor .is-cg-hover').forEach((row) => row.classList.remove('is-cg-hover'));
    const row = editorRowByLine(line);
    if (row) row.classList.add('is-cg-hover');
  }

  function highlightSourceLine(line, { scroll } = {}) {
    clearSourceLineHighlight();
    const row = editorRowByLine(line);
    if (!row) return;
    row.classList.add('is-cg-active');
    if (scroll) row.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }

  // Mark which source rows map to a graph node and route their clicks/hovers.
  function bindSourceToGraph(instance) {
    const editor = $('#dslEditor');
    if (!editor || !instance) return;
    const lineByNode = instance.lineByNode || {};
    const mappedLines = new Set(Object.values(lineByNode).map(Number));
    $$('#dslEditor [data-passes-line]').forEach((row) => {
      const isMapped = mappedLines.has(Number(row.dataset.passesLine));
      row.classList.toggle('is-cg-mapped', isMapped);
    });
    if (editor.dataset.cgBound === 'true') return; // delegate once
    editor.dataset.cgBound = 'true';
    editor.addEventListener('click', (event) => {
      if (!isPassesDumpFile(state.activeFile) || !passesGraphInstance) return;
      const row = event.target.closest('[data-passes-line]');
      if (!row) return;
      const line = Number(row.dataset.passesLine);
      if (passesGraphInstance.selectByLine(line, true)) {
        highlightSourceLine(line, { scroll: false });
      }
    });
    editor.addEventListener('mouseover', (event) => {
      if (!isPassesDumpFile(state.activeFile)) return;
      const row = event.target.closest('[data-passes-line].is-cg-mapped');
      if (row) row.classList.add('is-cg-hover');
    });
    editor.addEventListener('mouseout', (event) => {
      const row = event.target.closest('[data-passes-line]');
      if (row) row.classList.remove('is-cg-hover');
    });
  }

  function renderMatmulHardwareGraph() {
    const memoryPattern = window.PtoMemoryArchitecturePattern;
    const canvas = $('#matmulHardwareGraph');
    const viewport = $('#matmulHardwareViewport');
    const sizer = $('#matmulHardwareSizer');
    const host = $('.kf-matmul-hardware');
    const fitButton = $('[data-matmul-fit]');
    const actualButton = $('[data-matmul-actual]');
    const readout = $('[data-matmul-zoom-readout]');
    const flowStatus = $('#matmulFlowStatus');
    if (!memoryPattern || !canvas || !viewport || !sizer) return;

    memoryPattern.renderArchitecture(canvas, matmulHardwarePreset);
    const routes = memoryPattern.createRouteOverlay(canvas, matmulHardwarePreset);
    const hover = memoryPattern.attachHoverInteractions(canvas, matmulHardwarePreset, {
      selector: '[data-mem950-node="rail:DDR"], #matmul-aic-core',
    });
    let viewMode = 'fit';
    const syncViewMode = () => {
      fitButton?.classList.toggle('is-active', viewMode === 'fit');
      actualButton?.classList.toggle('is-active', viewMode === 'actual');
      fitButton?.setAttribute('aria-pressed', String(viewMode === 'fit'));
      actualButton?.setAttribute('aria-pressed', String(viewMode === 'actual'));
    };
    const zoom = memoryPattern.createZoomController({
      root: $('#inspector'),
      viewport,
      sizer,
      canvas,
      defaultZoom: 0.25,
      min: 0.16,
      max: 1.2,
      step: 0.08,
      pan: true,
      wheelZoom: false,
      centerTarget: '.pto-mem950__layout',
      readout,
      onZoom: () => syncViewMode(),
    });

    const fit = () => {
      const graph = canvas.querySelector('.pto-mem950');
      if (!graph) return;
      const widthScale = (viewport.clientWidth - 10) / Math.max(graph.scrollWidth, 1);
      const heightScale = (viewport.clientHeight - 10) / Math.max(graph.scrollHeight, 1);
      viewMode = 'fit';
      zoom?.setZoom(Math.max(0.16, Math.min(1, widthScale, heightScale)));
      zoom?.center();
      routes?.render();
      syncViewMode();
    };
    const actual = () => {
      viewMode = 'actual';
      zoom?.setZoom(1);
      zoom?.center();
      routes?.render();
      syncViewMode();
    };
    const onWheel = (event) => {
      event.preventDefault();
      viewMode = 'custom';
      const direction = event.deltaY > 0 ? -1 : 1;
      const magnitude = Math.min(3, Math.max(1, Math.abs(event.deltaY) / 120));
      zoom?.zoomAtPoint(zoom.getZoom() + 0.08 * direction * magnitude, event.clientX, event.clientY);
      routes?.render();
      syncViewMode();
    };
    const activateFlow = (lineNumber) => {
      const flow = matmulLineFlows[lineNumber];
      if (!flow) return;
      memoryPattern.setPathFocus(canvas, matmulHardwarePreset, flow);
      host?.classList.add('is-code-flowing');
      $$('#dslEditor [data-hardware-line]').forEach((row) => row.classList.toggle('is-hardware-selected', Number(row.dataset.hardwareLine) === lineNumber));
      if (flowStatus) flowStatus.innerHTML = `<span><i></i>第 ${lineNumber} 行 · ${flow.label}</span>`;
      state.hardwareFlowLine = lineNumber;
    };
    const clearFlow = () => {
      memoryPattern.clearPathFocus(canvas);
      host?.classList.remove('is-code-flowing');
      $$('#dslEditor [data-hardware-line]').forEach((row) => row.classList.remove('is-hardware-selected'));
      if (flowStatus) flowStatus.innerHTML = '<span><i></i>悬停源码行查看数据流，点击可锁定</span>';
      state.hardwareFlowLine = 0;
    };
    fitButton?.addEventListener('click', fit);
    actualButton?.addEventListener('click', actual);
    viewport.addEventListener('wheel', onWheel, { passive: false });
    const fitObserver = typeof ResizeObserver === 'function' ? new ResizeObserver(() => { if (viewMode === 'fit') fit(); }) : null;
    fitObserver?.observe(viewport);
    requestAnimationFrame(fit);

    matmulHardwareGraphInstance = {
      activateFlow,
      clearFlow,
      fit,
      actual,
      destroy() {
        fitButton?.removeEventListener('click', fit);
        actualButton?.removeEventListener('click', actual);
        viewport.removeEventListener('wheel', onWheel);
        fitObserver?.disconnect();
        routes?.destroy?.();
        hover?.destroy?.();
        zoom?.destroy?.();
      },
    };
  }

  function updateInspector() {
    if (state.step === 1) {
      renderIntentInspector();
      if (state.fixed) $('.kf-intent-note p').textContent = '已切换到静态仿射 work-table fallback；动态 GM store offset 已移除。';
      return;
    }
    $('#inspectorTitle').textContent = ['上下文预览', '意图预览', '编译约束', '正确性定位', '可信摘要'][state.step];
    $('#inspector').innerHTML = inspectorContent[state.step];
    $('#inspectorMeta').textContent = state.step === 4 ? 'sealed' : 'live';
    if (state.step === 1 && state.fixed) {
      $('#inspectorDiagnostic').textContent = '动态索引已替换为静态仿射 slot；codegen 限制已绕过。';
    }
  }

  function setActivityView(view) {
    state.activityView = view;
    const isRuns = view === 'runs';
    const isModel = view === 'model';
    if (view === 'explorer' && state.step !== EXPLORER_STEP) renderStage(EXPLORER_STEP);
    if (view === 'workflow' && state.step === EXPLORER_STEP) renderStage(state.workflowStep);
    $$('[data-side-view]').forEach((panel) => {
      const active = panel.dataset.sideView === view;
      panel.hidden = !active;
      panel.classList.toggle('is-active', active);
    });
    $$('[data-activity-view]').forEach((button) => {
      const active = button.dataset.activityView === view || (view === 'runs' && button.dataset.activityView === 'workflow');
      button.classList.toggle('is-selected', active);
      button.setAttribute('aria-pressed', String(active));
      button.setAttribute('aria-expanded', String(active));
    });
    $('#ideMainSplit').hidden = isModel;
    $('#ideStatusStrip').hidden = isModel;
    $('#modelArchitectureView').hidden = !isModel;
    if (isModel) {
      $('.kf-command').textContent = 'MODEL · Qwen3 14B 架构可视化';
      window.PtoQwen3ModelViz?.show();
      return;
    }
    $('.kf-main-body').classList.toggle('is-runs', isRuns);
    $('.kf-main-body').classList.toggle('is-explorer', view === 'explorer');
    $('.kf-main-body').classList.toggle('is-workflow', view === 'workflow');
    $('#tabs').parentElement.hidden = view !== 'explorer';
    $('#stageTitle').closest('.pto-ide-frame__pane-header').hidden = view === 'explorer';
    $('.kf-command').textContent = isRuns ? 'RUNS · 统一运行详情' : '⌘ K　搜索命令、tensor 或 pass';
    const sideTitle = { explorer: '资源管理器', workflow: '任务路线', runs: '运行列表' }[view] || '资源管理器';
    $('#sidePaneTitle').textContent = sideTitle;
    const workflowPosition = Math.max(0, WORKFLOW_STEPS.indexOf(state.step));
    $('#sidePaneMeta').textContent = view === 'explorer' ? 'workspace' : isRuns ? `${runs.length} runs` : `${workflowPosition + 1} / ${WORKFLOW_STEPS.length}`;
    if (isRuns) {
      renderRunList();
      renderRunDetail();
      $('#stageTitle').textContent = '统一运行详情页';
      $('#stageMeta').textContent = getRun().id;
      updateRunInspector();
    } else {
      $('#stageTitle').textContent = titles[state.step][0];
      $('#stageMeta').textContent = titles[state.step][1];
      updateInspector();
    }
    if (view === 'explorer') {
      $$('[data-file]').forEach((item) => item.classList.toggle('is-selected', item.dataset.file === state.activeFile));
      setEditorTab(state.editorTab);
    }
  }

  function toggleTreeGroup(name, expanded) {
    const group = $(`[data-tree-group="${name}"]`);
    const toggle = $(`[data-tree-toggle="${name}"]`);
    if (!group || !toggle) return;
    group.hidden = !expanded;
    toggle.setAttribute('aria-expanded', String(expanded));
    const caret = $('.kf-caret', toggle);
    if (caret) caret.textContent = '›';
  }

  const titles = [
    ['定义目标', 'recipe · decode_layer'],
    ['编写 Kernel', 'kernels/decode_layer.py'],
    ['编译卫士', '5 passes · 8 guards'],
    ['Correctness Lab', '3 oracles · tensor checkpoints'],
    ['可信基线', 'ptok · signed evidence']
  ];

  function renderStage(step) {
    state.step = Math.max(0, Math.min(4, step));
    if (state.step !== EXPLORER_STEP) state.workflowStep = state.step;
    $$('.kf-stage').forEach((el, i) => el.classList.toggle('is-active', i === state.step));
    const workflowPosition = WORKFLOW_STEPS.indexOf(state.step);
    $$('#stepNav [data-step]').forEach((button) => {
      const buttonStep = Number(button.dataset.step);
      if (buttonStep === EXPLORER_STEP) {
        button.classList.remove('is-active', 'is-complete');
        return;
      }
      const buttonPosition = WORKFLOW_STEPS.indexOf(buttonStep);
      button.classList.toggle('is-active', buttonStep === state.step);
      button.classList.toggle('is-complete', buttonPosition < workflowPosition || (buttonStep === 3 && state.verified));
    });
    $('#progressBar').style.width = `${((workflowPosition + 1) / WORKFLOW_STEPS.length) * 100}%`;
    if (state.activityView === 'workflow') $('#sidePaneMeta').textContent = `${workflowPosition + 1} / ${WORKFLOW_STEPS.length}`;
    $('#stageTitle').textContent = titles[state.step][0];
    $('#stageMeta').textContent = titles[state.step][1];
    $('#statusText').textContent = ['目标契约已就绪', 'DSL 即时诊断运行中', 'Pass 不变量验证', 'Oracle 三角比对', '可信基线已签发'][state.step];
    updateInspector();
  }

  function goTo(step) {
    const targetStep = Math.max(0, Math.min(4, step));
    const targetView = targetStep === EXPLORER_STEP ? 'explorer' : 'workflow';
    if (state.activityView !== targetView) setActivityView(targetView);
    renderStage(targetStep);
  }

  function toast(message) {
    const el = $('#toast');
    el.textContent = message;
    el.classList.add('is-visible');
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => el.classList.remove('is-visible'), 1800);
  }

  function setEnvironmentPanel(open) {
    const control = $('#envControl');
    const panel = $('#envFingerprintPanel');
    panel.hidden = !open;
    control.setAttribute('aria-expanded', String(open));
  }

  const soloToolNames = { context: '工程上下文', editor: 'Kernel Editor', guard: 'Compiler Guard', lab: 'Correctness Lab' };
  const soloToolStatus = { context: 'Context indexed', editor: 'Editing decode_layer.py', guard: 'Validating pass invariants', lab: 'Comparing three oracles' };
  const soloRunSteps = [
    { tool: 'context', title: '上下文与目标契约已锁定', detail: '读取 12 个 tensor contract、Ascend 950B 容量约束和 BF16 精度目标。' },
    { tool: 'editor', title: 'Decode Layer 源码已解析', detail: '识别 FP32 carry、manual_scope 与动态 work-table 索引，并准备静态仿射 fallback。' },
    { tool: 'guard', title: '所有编译 Pass 不变量成立', detail: 'Semantic、Layout、Parallel、Memory 与 ISA 五个 Pass 的 8 类卫士全部通过。' },
    { tool: 'lab', title: '三路 Oracle 已完成交叉验证', detail: '定位并消除首个 tensor 分歧；12 / 12 checkpoint 满足 rtol 1e-3。' },
    { tool: 'lab', title: '可信基线已签发', detail: '生成环境指纹 env:8da1bf09、证据链和可复现命令，基线已封存。' }
  ];

  function setProductMode(mode) {
    state.productMode = mode;
    const solo = mode === 'solo';
    $('[data-ide-frame]').dataset.productMode = mode;
    $('#ideActivityRail').hidden = solo;
    $('#ideWorkarea').hidden = solo;
    $('#soloWorkarea').hidden = !solo;
    $$('.kf-mode-switch [data-product-mode]').forEach((button) => {
      const active = button.dataset.productMode === mode;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    $('.kf-command').textContent = solo ? 'SOLO 正在编排 Context · Editor · Guard · Lab' : '⌘ K　搜索命令、tensor 或 pass';
    setEnvironmentPanel(false);
  }

  function setSoloTaskModal(open) {
    const modal = $('#soloTaskModal');
    modal.hidden = !open;
    $('#soloNewTaskTrigger').setAttribute('aria-expanded', String(open));
    if (open) requestAnimationFrame(() => $('#soloNewTaskGoal').focus());
  }

  function setAgentTeamDrawer(open) {
    const drawer = $('#agentTeamDrawer');
    const toggle = $('#agentTeamToggle');
    drawer.hidden = !open;
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-label', open ? '收起 Agent Team 成员' : '展开 Agent Team 成员');
  }

  function escapeHtml(value) {
    return value.replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
  }

  function setSoloFollow(active) {
    state.soloFollow = active;
    $('#soloFollow').classList.toggle('is-active', active);
    $('#soloFollow').setAttribute('aria-pressed', String(active));
  }

  function showSoloTool(tool, fromAgent = false) {
    state.soloTool = tool;
    $$('[data-solo-tool]').forEach((button) => {
      const active = button.dataset.soloTool === tool;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-selected', String(active));
    });
    $$('[data-solo-tool-panel]').forEach((panel) => {
      const active = panel.dataset.soloToolPanel === tool;
      panel.hidden = !active;
      panel.classList.toggle('is-active', active);
    });
    $('#soloToolTitle').textContent = soloToolNames[tool];
    $('#soloToolStatus').innerHTML = `<i></i> ${soloToolStatus[tool]}`;
    if (!fromAgent) setSoloFollow(false);
  }

  function appendSoloEvent(step, complete = false) {
    const event = document.createElement('article');
    event.className = `kf-solo-event${complete ? ' is-complete' : ''}`;
    event.innerHTML = `<header><b>${step.title}</b><span>${complete ? 'COMPLETE' : 'DONE'}</span></header><p>${step.detail}</p>`;
    $('#soloFeed').appendChild(event);
    $('#soloFeed').scrollTop = $('#soloFeed').scrollHeight;
  }

  async function soloDelay(ms) {
    let elapsed = 0;
    while (elapsed < ms) {
      if (!state.soloPaused) elapsed += 80;
      await new Promise(resolve => setTimeout(resolve, 80));
    }
  }

  function setSoloTask(index, status) {
    const item = $(`[data-solo-task="${index}"]`);
    item.classList.toggle('is-active', status === 'active');
    item.classList.toggle('is-done', status === 'done');
    if (status === 'done') $('i', item).textContent = '✓';
  }

  function selectAgentMember(member, followTool = false) {
    $$('.kf-agent-member').forEach(agent => agent.classList.toggle('is-selected', agent === member));
    const detail = $('#agentTeamDetail');
    $('b', detail).textContent = member.dataset.agentName;
    $('span', detail).textContent = member.dataset.agentDetail;
    $('em', detail).textContent = member.classList.contains('is-active') ? '正在协作' : member.classList.contains('is-complete') ? '产物已交付' : '等待接力';
    if (followTool) showSoloTool(member.dataset.agentTool, false);
  }

  function setAgentTeamStep(stepIndex) {
    const activeAgent = [0, 1, 2, 3, 0][stepIndex];
    const completedBeforeStep = [[], [0], [0, 1], [0, 1, 2], [1, 2, 3]][stepIndex];
    $$('.kf-agent-member').forEach((member, index) => {
      const active = index === activeAgent;
      const complete = completedBeforeStep.includes(index);
      member.classList.toggle('is-active', active);
      member.classList.toggle('is-complete', complete);
      $('.kf-agent-member-copy em', member).textContent = active ? (stepIndex === 4 ? 'SEALING' : 'WORKING') : complete ? 'DONE' : 'STANDBY';
    });
    const member = $(`[data-agent-index="${activeAgent}"]`);
    selectAgentMember(member);
  }

  function completeAgentTeam() {
    $$('.kf-agent-member').forEach(member => {
      member.classList.remove('is-active');
      member.classList.add('is-complete');
      $('.kf-agent-member-copy em', member).textContent = 'DONE';
    });
    const detail = $('#agentTeamDetail');
    $('b', detail).textContent = 'Kernel Alpha Team';
    $('span', detail).textContent = '四个 Agent 的产物已汇入同一条可信证据链。';
    $('em', detail).textContent = '协作完成';
  }

  async function runSolo() {
    if (state.soloRunning || state.soloComplete) return;
    state.soloRunning = true;
    state.soloPaused = false;
    $('#soloReady').hidden = true;
    $('#soloPause').disabled = false;
    $('#soloRunStatus').className = 'is-running';
    $('#soloRunStatusText').textContent = '自主执行中';

    for (let index = 0; index < soloRunSteps.length; index += 1) {
      const step = soloRunSteps[index];
      state.soloStep = index;
      setAgentTeamStep(index);
      setSoloTask(index, 'active');
      $('#soloProgress').textContent = `${index} / 5`;
      if (state.soloFollow) showSoloTool(step.tool, true);

      if (index === 1) $('#soloEditorState').textContent = 'Agent editing';
      if (index === 2) {
        $('#soloGuardState').textContent = '运行中';
        const passRows = $$('#soloGuardPasses > span');
        for (const row of passRows) {
          row.classList.add('is-active');
          await soloDelay(150);
          row.classList.remove('is-active');
          row.classList.add('is-done');
        }
        $('.kf-solo-guard-matrix').classList.add('is-pass');
        $('#soloGuardState').textContent = '5 / 5 PASS';
      } else if (index === 3) {
        $('#soloLabResult').className = 'kf-solo-lab-result is-running';
        $('#soloLabResult').innerHTML = '<span class="kf-solo-spinner"></span><h3>正在比对 Decode Layer argmax</h3><p>Torch golden · FP32 carry reference · PyPTO device</p>';
        await soloDelay(650);
        $('#soloDeviceResult').textContent = 'PASS';
        $('#soloLabResult').className = 'kf-solo-lab-result is-pass';
        $('#soloLabResult').innerHTML = '<span class="kf-solo-spinner"></span><h3>3 / 3 Oracle 一致</h3><p>12 / 12 checkpoints match · max diff 0.0009766</p>';
      } else {
        await soloDelay(620);
      }

      if (index === 1) $('#soloEditorState').textContent = 'Saved · diagnostic cleared';
      setSoloTask(index, 'done');
      $('#soloProgress').textContent = `${index + 1} / 5`;
      appendSoloEvent(step, index === soloRunSteps.length - 1);
    }

    state.soloRunning = false;
    state.soloComplete = true;
    $$('.kf-pass').forEach(item => item.classList.add('is-pass'));
    $$('.kf-guard').forEach(item => { item.classList.add('is-pass'); $('i', item).textContent = '✓'; });
    $('#compileStatus').textContent = '5 / 5 Pass 通过';
    $('#compileStatus').className = 'kf-state-chip good';
    $('#guardSummary').textContent = '8 / 8 约束通过';
    $('#runCompile').hidden = true;
    $('#toLab').hidden = false;
    state.compiled = true;
    verifyAndFinish();
    $('#soloPause').disabled = true;
    $('#soloRunStatus').className = 'is-complete';
    $('#soloRunStatusText').textContent = 'Agent Team 已完成可信基线';
    completeAgentTeam();
    $('#soloToolStatus').innerHTML = '<i></i> Baseline sealed · 9f2a71c';
    toast('SOLO 已完成：首个可信 Kernel 基线已签发');
  }

  function applyDslFix() {
    state.fixed = true;
    updateInspector();
    toast('已应用仿射 fallback：动态索引已移除');
  }

  async function runCompile() {
    const button = $('#runCompile');
    button.disabled = true;
    $('#compileStatus').textContent = '正在验证…';
    const passEls = $$('.kf-pass');
    for (let i = 0; i < passEls.length; i += 1) {
      passEls[i].classList.add('is-running');
      $('#activePassName').textContent = passes[i];
      $('#guardSummary').textContent = `Pass ${i + 1} / 5 · 验证 8 项约束`;
      await new Promise(resolve => setTimeout(resolve, 260));
      passEls[i].classList.remove('is-running');
      passEls[i].classList.add('is-pass');
    }
    $$('.kf-guard').forEach((el) => { el.classList.add('is-pass'); $('i', el).textContent = '✓'; });
    state.compiled = true;
    $('#compileStatus').textContent = '5 / 5 Pass 通过';
    $('#compileStatus').className = 'kf-state-chip good';
    $('#guardSummary').textContent = '8 / 8 约束通过';
    button.hidden = true;
    $('#toLab').hidden = false;
    toast('编译完成：所有 Pass 不变量成立');
  }

  function verifyAndFinish() {
    applyDslFix();
    state.verified = true;
    renderOracles();
    $('#labStatus').textContent = '3 / 3 oracle 一致';
    $('#labStatus').className = 'kf-state-chip good';
    $('.kf-divergence').style.opacity = '.42';
    $('.kf-root-cause').innerHTML = '<span style="color:var(--success)">✓</span><div><b style="color:var(--success)">Fallback 已验证</b><p>16 / 16 batch argmax 一致；FP32 carry reference 的比例容差满足预期。</p></div><button class="btn btn-solid" id="issueBaseline">签发可信基线 →</button>';
    $('#issueBaseline').addEventListener('click', () => goTo(4));
    toast('复验通过：首个分歧已消除');
  }

  renderRecipes();
  renderPasses();
  renderOracles();
  renderTensorCompare();
  renderGraph();
  renderFullSource();
  goTo(1);
  setProductMode('ide');

  $$('[data-activity-view]').forEach((button) => button.addEventListener('click', (event) => {
    const isExplorer = button.dataset.activityView === 'explorer';
    const returningToExplorer = isExplorer && state.activityView !== 'explorer';
    const explorerHidden = $('#kf-explorer')?.hidden;
    if (returningToExplorer && !explorerHidden) event.stopImmediatePropagation();
    setActivityView(button.dataset.activityView);
  }, true));
  setActivityView('explorer');
  $('[data-file="decode_layer.py"]')?.classList.add('is-selected');

  document.addEventListener('click', (event) => {
    if (!event.target.closest('#envControl') && !event.target.closest('#envFingerprintPanel')) setEnvironmentPanel(false);
    const recipe = event.target.closest('[data-recipe]');
    if (recipe) { state.selectedRecipe = recipe.dataset.recipe; renderRecipes(); toast(`已选择 ${$('b', recipe).textContent}`); }
    const step = event.target.closest('[data-step]');
    if (step) { setActivityView('workflow'); goTo(Number(step.dataset.step)); }
    if (event.target.closest('[data-open-runs]')) setActivityView('runs');
    if (event.target.closest('[data-back-workflow]')) setActivityView('workflow');
    const treeToggle = event.target.closest('[data-tree-toggle]');
    if (treeToggle) toggleTreeGroup(treeToggle.dataset.treeToggle, treeToggle.getAttribute('aria-expanded') !== 'true');
    const file = event.target.closest('[data-file]');
    if (file) {
      $$('[data-file]').forEach(item => item.classList.remove('is-selected'));
      file.classList.add('is-selected');
      const filePath = file.dataset.file;
      const isFolder = filePath.endsWith('/');
      if (!isFolder) {
        state.activeFile = filePath;
        if (filePath === RMSNORM_FILE) {
          state.rmsNormFunction = 'input';
          state.rmsNormTab = 'overview';
          state.rmsNormFlowStep = 'load';
        }
        if (filePath === ATTENTION_FILE) {
          state.attentionTab = 'overview';
          state.attentionFocus = 'position';
        }
        if (filePath === QWEN_DECODE_FILE) {
          state.qwenDecodeTab = 'overview';
          state.qwenDecodeFocus = 'scope1';
        }
        if (isPagedAttentionFile(filePath)) {
          state.pagedAttentionTab = 'overview';
          state.pagedAttentionFocus = 'paging';
          state.pagedAttentionOverlay = 'precision';
          state.pagedAttentionExpandedNode = null;
        }
        state.hardwareFlowLine = 0;
        state.hardwareFlowPinned = false;
        renderSelectedSource(state.activeFile);
      }
      const openStep = file.dataset.openStep;
      if (openStep == null && !isFolder) goTo(EXPLORER_STEP);
      if (openStep != null) {
        goTo(Number(openStep));
        const isPasses = file.dataset.passesDump === 'true';
        $('#stageMeta').textContent = isPasses
          ? `passes_dump/${state.activeFile}`
          : `kernels/${state.activeFile}`;
        $('[data-editor-tab="source"]').textContent = state.activeFile;
        toast(isPasses
          ? `已打开 ${file.dataset.file} · passes_dump 中间代码`
          : `已打开 ${file.dataset.file} · 定位到${titles[Number(openStep)][0]}`);
      } else {
        $('[data-editor-tab="source"]').textContent = state.activeFile;
        $('#stageMeta').textContent = state.activeFile;
        toast(`已选择 ${file.dataset.file}`);
      }
    }
    const rmsFunction = event.target.closest('.kf-rms-function-switch [data-rms-function]');
    if (rmsFunction) {
      state.rmsNormFunction = rmsFunction.dataset.rmsFunction;
      state.rmsNormFlowStep = 'load';
      renderRmsNormInspector({ scrollToFunction: true });
    }
    const rmsTab = event.target.closest('[data-rms-tab]');
    if (rmsTab) {
      state.rmsNormTab = rmsTab.dataset.rmsTab;
      renderRmsNormInspector();
    }
    const rmsLine = event.target.closest('#dslEditor [data-rms-line]');
    if (rmsLine && state.activeFile === RMSNORM_FILE) {
      state.rmsNormFunction = rmsLine.dataset.rmsFunction;
      const sourceLine = Number(rmsLine.dataset.rmsLine);
      const matchingStep = (rmsNormExecutionSteps[state.rmsNormFunction] || []).find((item) => item.lines.includes(sourceLine));
      if (matchingStep) state.rmsNormFlowStep = matchingStep.id;
      renderRmsNormInspector();
    }
    if (event.target.closest('[data-rms-action="golden"]')) toast('已生成测试草案：Qwen3 shape · Torch golden · BF16 输出容差');
    const attentionTab = event.target.closest('[data-attention-tab]');
    if (attentionTab) {
      state.attentionTab = attentionTab.dataset.attentionTab;
      renderAttentionInspector();
    }
    const attentionFocus = event.target.closest('.kf-attn-source-map [data-attention-focus]');
    if (attentionFocus) {
      state.attentionFocus = attentionFocus.dataset.attentionFocus;
      renderAttentionInspector({ scrollToFocus: true });
    }
    const attentionLine = event.target.closest('#dslEditor [data-attention-line]');
    if (attentionLine && state.activeFile === ATTENTION_FILE) {
      state.attentionFocus = attentionLine.dataset.attentionFocus;
      state.attentionTab = 'mapping';
      renderAttentionInspector();
    }
    if (event.target.closest('[data-attention-action="golden"]')) toast('已生成测试草案：RoPE 位置边界 · K/V Cache 写入 · Q Padding');
    const qwenDecodeTab = event.target.closest('[data-qwen-decode-tab]');
    if (qwenDecodeTab) {
      state.qwenDecodeTab = qwenDecodeTab.dataset.qwenDecodeTab;
      renderQwenDecodeInspector();
    }
    const qwenStructureFocus = event.target.closest('[data-qwen-structure-focus]');
    if (qwenStructureFocus) {
      state.qwenDecodeFocus = qwenStructureFocus.dataset.qwenStructureFocus;
      state.qwenDecodeTab = 'orchestration';
      renderQwenDecodeInspector({ scrollToFocus: true });
    }
    const qwenDecodeFocus = event.target.closest('[data-qwen-decode-focus]');
    if (qwenDecodeFocus && !qwenDecodeFocus.closest('#dslEditor')) {
      state.qwenDecodeFocus = qwenDecodeFocus.dataset.qwenDecodeFocus;
      renderQwenDecodeInspector({ scrollToFocus: true });
    }
    const qwenDecodeLine = event.target.closest('#dslEditor [data-qwen-decode-line]');
    if (qwenDecodeLine && state.activeFile === QWEN_DECODE_FILE) {
      state.qwenDecodeFocus = qwenDecodeLine.dataset.qwenDecodeFocus;
      state.qwenDecodeTab = 'orchestration';
      renderQwenDecodeInspector();
    }
    if (event.target.closest('[data-qwen-decode-action="test"]')) toast('已生成测试清单：编译结构 · Attention 数据链 · Cache 增量 · BF16 数值 · 昇腾实跑');
    const pagedAttentionTab = event.target.closest('[data-paged-attention-tab]');
    if (pagedAttentionTab) {
      state.pagedAttentionTab = pagedAttentionTab.dataset.pagedAttentionTab;
      renderPagedAttentionInspector();
    }
    const pagedAttentionGoTab = event.target.closest('[data-pa-go-tab]');
    if (pagedAttentionGoTab) {
      state.pagedAttentionTab = pagedAttentionGoTab.dataset.paGoTab;
      renderPagedAttentionInspector();
    }
    const pagedAttentionOverlay = event.target.closest('[data-pa-overlay]');
    if (pagedAttentionOverlay) {
      state.pagedAttentionOverlay = pagedAttentionOverlay.dataset.paOverlay;
      renderPagedAttentionInspector();
    }
    const pagedAttentionFocus = event.target.closest('[data-paged-attention-focus]');
    if (pagedAttentionFocus && !pagedAttentionFocus.closest('#dslEditor')) {
      state.pagedAttentionFocus = pagedAttentionFocus.dataset.pagedAttentionFocus;
      renderPagedAttentionInspector({ scrollToFocus: true });
    }
    const pagedAttentionLine = event.target.closest('#dslEditor [data-paged-attention-line]');
    if (pagedAttentionLine && isPagedAttentionFile(state.activeFile)) {
      state.pagedAttentionFocus = pagedAttentionLine.dataset.pagedAttentionFocus;
      state.pagedAttentionTab = 'schedule';
      renderPagedAttentionInspector();
    }
    if (event.target.closest('[data-paged-attention-action="tests"]')) toast('已生成测试清单：动态 Shape 组合 · Q Head 尾 Tile · KV 末块 · 空/短 Context · BF16 Online Softmax');
    if (event.target.closest('[data-next]')) goTo(state.step + 1);
    if (event.target.closest('[data-prev]')) goTo(state.step - 1);

    const runItem = event.target.closest('[data-run]');
    if (runItem) {
      state.currentRun = runItem.dataset.run;
      renderRunList();
      renderRunDetail();
      updateRunInspector();
      $('#stageMeta').textContent = state.currentRun;
      toast(`已打开运行详情：${getRun().title}`);
    }
    const nextAction = event.target.closest('[data-next-action]');
    if (nextAction) {
      const run = getRun();
      const entry = run.next[Number(nextAction.dataset.nextIndex)];
      if (nextAction.dataset.nextAction === 'fix') {
        setActivityView('workflow');
        goTo(1);
        toast(`已在 IDE 中打开 ${entry[2]}`);
      } else {
        navigator.clipboard?.writeText(entry[2]);
        toast(nextAction.dataset.nextAction === 'exp' ? `已加入实验队列：${entry[2]}` : `已复制命令：${entry[2]}`);
      }
    }
    const evidenceNode = event.target.closest('[data-evidence]');
    if (evidenceNode) {
      const run = getRun();
      const key = evidenceNode.dataset.evidence;
      state.selectedEvidence = key;
      renderRunDetail();
      updateRunInspector();
      toast(`下钻 ${evidenceMeta[key][1]} 证据 · ${run.evidence[key] || '—'} · ${run.id}`);
    }
    const runActionTab = event.target.closest('[data-run-action-tab]');
    if (runActionTab) {
      state.runActionTab = runActionTab.dataset.runActionTab;
      renderRunDetail();
    }
    const intentTab = event.target.closest('[data-intent-tab]');
    if (intentTab) {
      state.intentTab = intentTab.dataset.intentTab;
      renderIntentInspector();
      toast(`意图预览已切换到 ${intentPreview[state.intentTab].label}`);
    }
    const cgMode = event.target.closest('[data-cg-mode]');
    if (cgMode) {
      setPassesGraphMode(cgMode.dataset.cgMode);
      toast(cgMode.dataset.cgMode === 'compare' ? '已切换到计算图演进对比' : '已切换到单图视图');
    }
    const intentLine = event.target.closest('[data-intent-line]');
    if (intentLine) {
      state.intentTab = intentLine.dataset.intentLine;
      $$('[data-intent-line]').forEach(line => line.classList.toggle('is-intent-selected', line === intentLine));
      renderIntentInspector();
      toast(`第 ${$('i', intentLine).textContent} 行 · ${intentPreview[state.intentTab].label} 意图`);
    }
    const hardwareLine = event.target.closest('[data-hardware-line]');
    if (hardwareLine) {
      const lineNumber = Number(hardwareLine.dataset.hardwareLine);
      const isSamePinnedLine = state.hardwareFlowPinned && state.hardwareFlowLine === lineNumber;
      state.hardwareFlowPinned = !isSamePinnedLine;
      if (isSamePinnedLine) matmulHardwareGraphInstance?.clearFlow?.();
      else matmulHardwareGraphInstance?.activateFlow?.(lineNumber);
      toast(isSamePinnedLine ? '已取消硬件路径锁定' : `已锁定第 ${lineNumber} 行硬件数据流`);
    }
    if (event.target.closest('#baselinePicker')) toast('已打开可信基线选择器 · 当前 run_d9a1');
    if (event.target.closest('#copyRunToken')) { navigator.clipboard?.writeText(getRun().token); toast('运行链接已复制，可共享或跨 Run diff'); }
    if (event.target.closest('#runShare')) { navigator.clipboard?.writeText(getRun().token); toast('已生成可共享运行详情链接'); }
    if (event.target.closest('#runCompare2') || event.target.closest('#compareRuns')) {
      const trusted = runs.find(r => r.verdict === 'trusted');
      toast(`对比 ${getRun().id} ↔ ${trusted ? trusted.id : '可信基线'} · 因果 diff 已就绪`);
    }
  });
  document.addEventListener('pointerover', (event) => {
    const line = event.target.closest?.('[data-hardware-line]');
    if (!line || line.contains(event.relatedTarget) || state.hardwareFlowPinned) return;
    matmulHardwareGraphInstance?.activateFlow?.(Number(line.dataset.hardwareLine));
  });
  document.addEventListener('pointerout', (event) => {
    const line = event.target.closest?.('[data-hardware-line]');
    if (!line || line.contains(event.relatedTarget) || state.hardwareFlowPinned) return;
    matmulHardwareGraphInstance?.clearFlow?.();
  });
  document.addEventListener('focusin', (event) => {
    const line = event.target.closest?.('[data-hardware-line]');
    if (!line || state.hardwareFlowPinned) return;
    matmulHardwareGraphInstance?.activateFlow?.(Number(line.dataset.hardwareLine));
  });
  document.addEventListener('focusout', (event) => {
    if (!event.target.closest?.('[data-hardware-line]') || state.hardwareFlowPinned) return;
    matmulHardwareGraphInstance?.clearFlow?.();
  });
  document.addEventListener('keydown', (event) => {
    const line = event.target.closest?.('[data-hardware-line]');
    if (!line || (event.key !== 'Enter' && event.key !== ' ')) return;
    event.preventDefault();
    line.click();
  });
  $('#applyFix')?.addEventListener('click', applyDslFix);
  $$('[data-product-mode]').forEach((button) => button.addEventListener('click', () => {
    const mode = button.dataset.productMode;
    if (mode === 'ide' && state.soloRunning) {
      state.soloPaused = true;
      $('#soloPause').textContent = '继续';
    }
    setProductMode(mode);
    if (mode === 'ide' && state.soloStep >= 0) {
      setActivityView('workflow');
      goTo(state.soloComplete ? 4 : state.soloStep);
    }
  }));
  $('#soloNewTaskTrigger').setAttribute('aria-haspopup', 'dialog');
  $('#soloNewTaskTrigger').setAttribute('aria-expanded', 'false');
  $('#soloNewTaskTrigger').addEventListener('click', () => setSoloTaskModal(true));
  $('#soloNewTaskClose').addEventListener('click', () => setSoloTaskModal(false));
  $('#soloNewTaskCancel').addEventListener('click', () => setSoloTaskModal(false));
  $('#soloTaskModal').addEventListener('click', (event) => {
    if (event.target === event.currentTarget) setSoloTaskModal(false);
  });
  $('#soloNewTaskForm').addEventListener('submit', (event) => {
    event.preventDefault();
    const goal = $('#soloNewTaskGoal').value.trim();
    if (!goal) return;
    const recipe = $('#soloNewTaskRecipe').value;
    const target = $('#soloNewTaskTarget').value;
    const item = document.createElement('button');
    item.className = 'kf-solo-history-item is-selected';
    item.type = 'button';
    item.dataset.historyTask = goal;
    item.innerHTML = `<span class="kf-solo-history-icon is-queued">↗</span><span><b>${escapeHtml(goal)}</b><small>排队中 · ${escapeHtml(recipe)} · ${escapeHtml(target)}</small></span><time>刚刚</time>`;
    $$('[data-history-task]').forEach(historyItem => historyItem.classList.remove('is-selected'));
    $('#soloHistoryList').prepend(item);
    $('#soloHistoryCount').textContent = `${$$('[data-history-task]').length} 项`;
    event.currentTarget.reset();
    setSoloTaskModal(false);
    toast('新任务已创建，并加入 SOLO 任务队列');
  });
  $('#soloHistoryList').addEventListener('click', (event) => {
    const item = event.target.closest('[data-history-task]');
    if (!item) return;
    $$('[data-history-task]').forEach(historyItem => historyItem.classList.toggle('is-selected', historyItem === item));
    toast(`已打开任务摘要：${item.dataset.historyTask}`);
  });
  $('#agentTeamToggle').addEventListener('click', () => {
    const open = $('#agentTeamToggle').getAttribute('aria-expanded') !== 'true';
    setAgentTeamDrawer(open);
  });
  $('#soloStart').addEventListener('click', runSolo);
  $('#soloPause').addEventListener('click', () => {
    state.soloPaused = !state.soloPaused;
    $('#soloPause').textContent = state.soloPaused ? '继续' : '暂停';
    $('#soloRunStatusText').textContent = state.soloPaused ? '已暂停 · 等待接管' : '自主执行中';
    toast(state.soloPaused ? 'SOLO 已暂停' : 'SOLO 已继续执行');
  });
  $('#soloTakeover').addEventListener('click', () => {
    state.soloPaused = true;
    setProductMode('ide');
    setActivityView('workflow');
    goTo(Math.max(0, state.soloStep));
    toast(`已切换到 IDE · 定位到${titles[Math.max(0, state.soloStep)][0]}`);
  });
  $('#soloFollow').addEventListener('click', () => {
    setSoloFollow(!state.soloFollow);
    if (state.soloFollow && state.soloStep >= 0) showSoloTool(soloRunSteps[state.soloStep].tool, true);
  });
  $('.kf-agent-team-grid').addEventListener('click', (event) => {
    const member = event.target.closest('.kf-agent-member');
    if (!member) return;
    selectAgentMember(member, true);
    toast(`${member.dataset.agentName} · 已打开对应工作现场`);
  });
  $$('[data-solo-tool]').forEach((button) => button.addEventListener('click', () => showSoloTool(button.dataset.soloTool, false)));
  $('#soloOpenTool').addEventListener('click', () => {
    const toolStep = { context: 0, editor: 1, guard: 2, lab: 3 }[state.soloTool];
    setProductMode('ide');
    setActivityView('workflow');
    goTo(toolStep);
    toast(`已在 IDE 中打开 ${soloToolNames[state.soloTool]}`);
  });
  $('#soloComposer').addEventListener('submit', (event) => {
    event.preventDefault();
    const prompt = $('#soloPrompt').value.trim();
    if (!prompt) return;
    const message = document.createElement('article');
    message.className = 'kf-solo-message is-user';
    message.innerHTML = `<span>你</span><div><p>${prompt.replace(/[&<>]/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[char])}</p><time>刚刚</time></div>`;
    $('#soloFeed').appendChild(message);
    $('#soloPrompt').value = '';
    $('#soloFeed').scrollTop = $('#soloFeed').scrollHeight;
    toast('约束已加入 SOLO 当前上下文');
  });
  $('#envControl').addEventListener('click', (event) => {
    event.preventDefault();
    event.stopPropagation();
    const open = event.currentTarget.getAttribute('aria-expanded') !== 'true';
    setEnvironmentPanel(open);
  });
  $('#envFingerprintPanel').addEventListener('click', (event) => event.stopPropagation());
  $('#copyFingerprint').addEventListener('click', () => { navigator.clipboard?.writeText('env:8da1bf09'); toast('环境指纹已复制'); });
  $('#runCompile').addEventListener('click', runCompile);
  $('#toLab').addEventListener('click', () => goTo(3));
  $('#fixAndRerun').addEventListener('click', verifyAndFinish);
  $('#copyBaseline').addEventListener('click', () => { navigator.clipboard?.writeText('ptok://qwen3-14b/decode-layer@9f2a71c'); toast('基线 ID 已复制'); });
  $('#copyRepro').addEventListener('click', () => { navigator.clipboard?.writeText('pypto trust replay ptok://qwen3-14b/decode-layer@9f2a71c'); toast('复现命令已复制'); });
  $('#viewEvidence').addEventListener('click', () => toast('证据包：24 项事实 · 3 个 oracle · 5 个 Pass 快照'));
  $('#newBaseline').addEventListener('click', () => toast('已创建调度优化分支：opt/decode-layer-from-9f2a71c'));
  $('#resetDemo').addEventListener('click', () => window.location.reload());
  $('#collapseTree').addEventListener('click', () => {
    $$('[data-tree-toggle]').forEach(toggle => toggleTreeGroup(toggle.dataset.treeToggle, false));
    toast('工程目录已折叠');
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !$('#soloTaskModal').hidden) {
      setSoloTaskModal(false);
      $('#soloNewTaskTrigger').focus();
      return;
    }
    if (event.key === 'Escape' && $('#agentTeamToggle').getAttribute('aria-expanded') === 'true') {
      setAgentTeamDrawer(false);
      $('#agentTeamToggle').focus();
      return;
    }
    if (event.key === 'Escape' && $('#envControl').getAttribute('aria-expanded') === 'true') {
      setEnvironmentPanel(false);
      $('#envControl').focus();
    }
  });
  $$('[data-editor-tab]').forEach((button) => button.addEventListener('click', () => {
    goTo(EXPLORER_STEP);
    setEditorTab(button.dataset.editorTab);
  }));

  // Product interactions are bound before the shared frame initializes so a
  // non-critical resize/chrome failure can never disable the workbench UI.
  try {
    window.PtoIdeFrame?.initAll();
  } catch (error) {
    console.warn('IDE frame enhancement unavailable; core interactions remain active.', error);
  }
  setActivityView(state.activityView);
  try {
    window.kernelForgeSoloSplit = window.PtoWorkbenchShell?.initResizablePanes({
      root: $('#soloWorkarea'),
      panes: ['#soloPlanPane', '#soloAgentPane', '#soloToolsPane'],
      direction: 'horizontal',
      sizes: [24, 42, 34],
      minSize: [210, 360, 300],
      gutterSize: 8,
      keyboardStep: 24,
      storageKey: 'pypto-studio-solo-split-v1',
      gutterLabel: '调整 Solo 相邻栏宽度',
    });
  } catch (error) {
    console.warn('SOLO pane resizing unavailable; default layout remains active.', error);
  }
  document.documentElement.dataset.kernelForgeReady = 'true';
})();
