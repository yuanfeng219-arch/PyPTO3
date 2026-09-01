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
  const state = { step: 0, workflowStep: 0, activityView: 'explorer', editorTab: 'source', activeFile: 'decode_layer.py', hardwareFlowLine: 0, hardwareFlowPinned: false, productMode: 'ide', selectedRecipe: 'decode_layer', fixed: false, compiled: false, verified: false, soloFollow: true, soloRunning: false, soloPaused: false, soloComplete: false, soloStep: -1, soloTool: 'context', currentRun: 'run_8f2c', runActionTab: 'cmd', selectedEvidence: 'tensor', intentTab: 'shape', intentGraphNode: null, passesGraphMode: 'single', rmsNormFunction: 'input', rmsNormTab: 'overview', rmsNormFlowStep: 'load', rmsNormPlan: {}, attentionTab: 'overview', attentionFocus: 'position', qwenDecodeTab: 'overview', qwenDecodeFocus: 'scope1', pagedAttentionTab: 'graph', pagedAttentionFocus: 'paging', pagedAttentionOverlay: 'data', pagedAttentionExpandedNode: null, pagedAttentionNode: 'orch', pagedAttentionTask: 'qk', pagedAttentionDep: 'sij', pagedAttentionPipeKernel: 'qk', pagedAttentionLine: null, pagedAttentionDetailOpen: false, pto3LabTab: 'loops', pto3LabFocus: 'matmul', sourceCache: {} };
  const EXPLORER_STEP = 1;
  const WORKFLOW_STEPS = [0, 2, 3, 4];
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  function setEditorTab(tab) {
    state.editorTab = tab;
    $$('[data-editor-tab]').forEach((button) => button.classList.toggle('is-active', button.dataset.editorTab === tab));
    $$('[data-editor-panel]').forEach((panel) => { panel.hidden = panel.dataset.editorPanel !== tab; });
  }

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
  const PTO3_TILE_LAB_FILE = 'pto3_tile_loop_lab.py';
  const pto3TileLabSource = `@pl.jit.incore
def matmul_tiled(a, b, out, M, K, N):
    # PTO 3.0: 显式表达 M / N / K 的 Tile 循环
    for m0 in pl.range(0, M, 128):
        for n0 in pl.range(0, N, 128):
            acc = pl.zeros([128, 128], dtype=pl.FP32, target_memory=pl.Mem.Left)
            for k0 in pl.range(0, K, 128):
                a_tile = pl.load(a, [m0, k0], [128, 128], target_memory=pl.Mem.Mat)
                b_tile = pl.load(b, [k0, n0], [128, 128], target_memory=pl.Mem.Mat)
                acc += pl.matmul(a_tile, b_tile)
            pl.store(acc, [m0, n0], out)

def rmsnorm_large_h(x, gamma, out, H=32768):
    # 大 H 维：每次只把安全的 H_TILE 放入 UB
    H_TILE = 4096
    for h0 in pl.range(0, H, H_TILE):
        h1 = min(h0 + H_TILE, H)
        x_tile = pl.load(x, [h0], [h1 - h0], target_memory=pl.Mem.Vec)
        ss = pl.sum(x_tile * x_tile, axis=0)
        inv = pl.rsqrt(ss / H + 1e-6)
        pl.store(x_tile * inv * gamma[h0:h1], [h0], out)`;
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
  let decodeLayerGraphController = null;
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
    if (file === PTO3_TILE_LAB_FILE) return pto3TileLabSource;
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
    const sourceIntentLines = !isPasses ? resolveIntentSourceLines(source) : {};
    const fragment = document.createDocumentFragment();
    highlighted.forEach((lineHtml, index) => {
      const lineNumber = index + 1;
      const row = document.createElement('div');
      const gutter = document.createElement('i');
      const code = document.createElement('code');
      let sourceTag = null;
      gutter.textContent = lineNumber;
      code.innerHTML = lineHtml || ' ';
      if (!isPasses && sourceIntentLines[lineNumber]) row.dataset.intentLine = sourceIntentLines[lineNumber];
      if (isPasses) { row.dataset.passesLine = String(lineNumber); row.tabIndex = 0; }
      if (state.activeFile === 'matmul.py') {
        row.dataset.hardwareLine = String(lineNumber);
        row.tabIndex = 0;
        row.title = matmulLineFlows[lineNumber]?.label || `第 ${lineNumber} 行硬件映射`;
      }
      if (state.activeFile === PTO3_TILE_LAB_FILE) {
        const focus = lineNumber >= 14 ? 'rmsnorm' : 'matmul';
        row.dataset.pto3LabLine = String(lineNumber);
        row.dataset.pto3LabFocus = focus;
        row.tabIndex = 0;
        row.title = `${focus === 'rmsnorm' ? 'RMSNorm 大 H 分块' : 'Matmul M/N/K 分块'} · 点击查看循环建议`;
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
      if (isPagedAttentionFile(state.activeFile)) {
        const focus = pagedAttentionFocusForLine(lineNumber);
        sourceTag = pagedAttentionSourceTags[lineNumber] || null;
        row.dataset.pagedAttentionLine = String(lineNumber);
        row.dataset.pagedAttentionFocus = focus;
        row.classList.add('kf-pa-source-row');
        row.tabIndex = 0;
        row.title = `${{ dynamic: '动态 Shape 声明', builder: 'Program Builder 与 Init', qk: 'QK Matmul · Cube', softmax: 'Softmax Prepare · Vector', pv: 'PV Matmul · Cube', online: 'Online Update · Vector', orchestration: '动态维度推导', paging: 'Paged KV 编排', golden: 'Torch Golden', runtime: '运行配置与验证' }[focus]} · 点击同步计算图与源码`;
      }
      row.append(gutter, code);
      if (sourceTag) {
        const tag = document.createElement('span');
        tag.className = `kf-pa-source-tag is-${sourceTag.focus}`;
        tag.textContent = sourceTag.label;
        tag.title = `${pagedAttentionFocusMeta[sourceTag.focus].detail} · 点击同步计算图`;
        tag.setAttribute('aria-label', `${sourceTag.label} 源码阶段标签`);
        row.append(tag);
      }
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
      if (state.pagedAttentionLine) markPagedAttentionTargetLine(state.pagedAttentionLine);
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

  function parseDecodeNumber(expression, constants) {
    let value = String(expression || '').replace(/#.*/, '').trim();
    value = value.replace(/int\(os\.environ\.get\([^,]+,\s*["'](\d+)["']\)\)/g, '$1');
    value = value.replace(/\b(?:int|float)\(([^()]+)\)/g, '($1)');
    for (let pass = 0; pass < 8; pass += 1) {
      value = value.replace(/\b[A-Z][A-Z0-9_]*\b/g, (name) => (
        Object.prototype.hasOwnProperty.call(constants, name) ? `${constants[name]}` : name
      ));
    }
    const operand = '(?:\\([^()]*\\)|(?:\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?))';
    for (let pass = 0; pass < 8 && value.includes('//'); pass += 1) {
      value = value.replace(new RegExp(`(${operand})\\s*//\\s*(${operand})`, 'g'), 'Math.floor($1 / $2)');
    }
    if (!/^[0-9eE+*/%().,\s-Mathfloor]+$/.test(value)) return null;
    try {
      const result = Number(Function(`"use strict"; return (${value});`)());
      return Number.isFinite(result) ? result : null;
    } catch {
      return null;
    }
  }

  function parseDecodeLayerSource(source) {
    const text = source || '';
    const lines = text.split(/\r?\n/);
    const constants = {};
    const assignment = /^([A-Z][A-Z0-9_]*)\s*=\s*(.+?)(?:\s+#.*)?$/gm;
    for (let pass = 0; pass < 8; pass += 1) {
      assignment.lastIndex = 0;
      let match;
      while ((match = assignment.exec(text))) {
        if (!Object.prototype.hasOwnProperty.call(constants, match[1])) {
          const value = parseDecodeNumber(match[2], constants);
          if (value !== null) constants[match[1]] = value;
        }
      }
    }

    const lineOf = (pattern) => {
      const index = lines.findIndex((line) => pattern.test(line));
      return index < 0 ? null : index + 1;
    };
    const number = (name, fallback = null) => (
      Number.isFinite(constants[name]) ? constants[name] : fallback
    );
    const integer = (name, fallback = null) => {
      const value = number(name, fallback);
      return value === null ? null : Math.round(value);
    };
    const taskRecords = [];
    const readBracketed = (input, start) => {
      let depth = 0;
      let quote = null;
      for (let index = start; index < input.length; index += 1) {
        const char = input[index];
        if (quote) {
          if (char === quote && input[index - 1] !== '\\') quote = null;
          continue;
        }
        if (char === '"' || char === "'") { quote = char; continue; }
        if (char === '[') depth += 1;
        if (char === ']') {
          depth -= 1;
          if (depth === 0) return input.slice(start + 1, index);
        }
      }
      return '';
    };
    const taskPattern = /with\s+pl\.(at|spmd)\(([\s\S]*?)\)\s+as\s+([A-Za-z_]\w*)\s*:/g;
    let taskMatch;
    while ((taskMatch = taskPattern.exec(text))) {
      const args = taskMatch[2];
      const nameMatch = args.match(/name_hint\s*=\s*["']([^"']+)["']/);
      if (!nameMatch) continue;
      const depsStart = args.search(/\bdeps\s*=\s*\[/);
      const depsOpen = depsStart < 0 ? -1 : args.indexOf('[', depsStart);
      const depsText = depsOpen < 0 ? '' : readBracketed(args, depsOpen).replace(/\s+/g, ' ').trim();
      const depCount = !depsText ? 0 : depsText.includes(' for ') ? 1 : depsText.split(',').length;
      const line = text.slice(0, taskMatch.index).split(/\r?\n/).length;
      taskRecords.push({
        name: nameMatch[1],
        kind: taskMatch[1],
        alias: taskMatch[3],
        line,
        deps: depsText || '自动依赖 / 无显式 deps',
        depCount,
        dispatch: args.split(',')[0].replace(/\s+/g, ' ').trim(),
      });
    }

    const taskNames = [...new Set(taskRecords.map((task) => task.name))];
    const taskByName = (name) => taskRecords.find((task) => task.name === name) || null;
    const names = (items) => items.filter((name) => taskNames.includes(name)).join(' · ') || '源码未声明';
    const manualLine = lineOf(/with\s+pl\.manual_scope\(\)/);
    const explicitTasks = taskRecords.filter((task) => task.depCount > 0);
    const sourceFacts = {
      lines: lines.length,
      manualLine,
      dynamicIndex: /fa_total_blocks|cursor\s*\+\s*wp|g_base\s*\+\s*sb/.test(text),
      compileBlocked: /does\s+NOT\s+compile/i.test(text) && /index\s+vs\s+i64/i.test(text),
      markers: {
        shape: lineOf(/# ── Model architecture/),
        layout: lineOf(/SEQ_TILE\s*=/),
        scope: manualLine,
        deps: taskRecords.find((task) => task.depCount > 0)?.line || lineOf(/deps\s*=\s*\[/),
        resource: lineOf(/NUM_CORES\s*=/),
      },
    };
    return {
      constants,
      integer,
      number,
      names,
      taskRecords,
      taskNames,
      taskByName,
      explicitTasks,
      sourceFacts,
    };
  }

  function resolveIntentSourceLines(source) {
    const parsed = parseDecodeLayerSource(source);
    const map = {};
    Object.entries(parsed.sourceFacts.markers).forEach(([tab, line]) => {
      if (line) map[line] = tab === 'layout' ? 'shape' : tab;
    });
    return map;
  }

  function buildDecodeLayerIntent(source) {
    const parsed = parseDecodeLayerSource(source);
    const { integer, number, names, taskRecords, taskNames, taskByName, explicitTasks, sourceFacts } = parsed;
    const c = (name, fallback = '—') => integer(name, fallback);
    const value = (name, fallback = '—') => number(name, fallback);
    const line = (tab) => sourceFacts.markers[tab] ? `源码第 ${sourceFacts.markers[tab]} 行` : '源码锚点未找到';
    const task = (name) => taskByName(name);
    const dep = (name) => task(name)?.deps || '源码未声明';
    const qOn = c('Q_ON');
    const kvOn = c('KV_ON');
    const qkvOk = c('QKV_OK');
    const mlpOn = c('MLP_ON');
    const kSplitsMlp = c('K_SPLITS_MLP');
    const downOn = c('DOWN_ON');
    const kSplits = c('K_SPLITS');
    const nSplitsOut = c('N_SPLITS_OUT');
    const kSplitsOut = c('K_SPLITS_OUT');
    const maxCtxBlocks = c('MAX_CTX_BLOCKS');
    const batch = c('BATCH');
    const hidden = c('HIDDEN');
    const kvHidden = c('KV_HIDDEN');
    const headDim = c('HEAD_DIM');
    const intermediate = c('INTERMEDIATE');
    const qHeadBatch = c('Q_HEAD_BATCH');
    const qHeadPad = c('Q_HEAD_PAD');
    const cacheRows = c('CACHE_ROWS');
    const phase1 = names(['rms_recip', 'q_seed', 'q_proj', 'k_seed', 'k_proj', 'v_seed', 'v_proj', 'qk_norm']);
    const phase2 = names(['fa_work_build', 'rope_qkv', 'fa_fused', 'online_softmax']);
    const phase3 = names(['out_seed', 'out_proj', 'residual_rms_cast', 'post_rms_reduce', 'gate_seed', 'gate_proj', 'up_seed', 'up_proj', 'silu', 'down_proj']);

    return {
      shape: {
        label: 'Shape & Layout',
        meta: `源码解析 · ${line('shape')} · ${line('layout')}`,
        rows: [
          ['Shape · hidden_states', `[${batch}, ${hidden}] · BF16 → cur FP32`],
          ['Shape · Q / K / V', `Q [${batch}, ${hidden}] · K/V [${batch}, ${kvHidden}] · FP32`],
          ['Shape · Attention heads', `${c('NUM_HEADS')} Q / ${c('NUM_KV_HEADS')} KV · head_dim ${headDim} · GQA ${c('Q_PER_KV')}:1`],
          ['Shape · Paged K/V cache', `[${cacheRows}, ${headDim}] × 2 · BF16 · paged GM`],
          ['Shape · Attention output', `[${batch}, ${hidden}] · BF16 · attn_out`],
          ['Shape · MLP intermediate', `[${batch}, ${intermediate}] · BF16 · gate/up/SiLU`],
          ['Shape · Layer outputs', `out [${batch}, ${hidden}] FP32 · normed_out [${batch}, ${hidden}] BF16`],
          ['Shape · Precision boundary', 'BF16 at input/cache/activation edges; FP32 inter-layer residual carry'],
          ['Layout · Paged KV page', `SEQ_TILE=${c('SEQ_TILE')} · BLOCK_SIZE=${c('BLOCK_SIZE')} · MAX_CTX_BLOCKS=${maxCtxBlocks}`],
          ['Layout · Q head tile', `${qHeadBatch} real rows → ${qHeadPad} physical rows · valid_shape 保留真实行`],
          ['Layout · QKV projection', `M=${c('TM')} · inner N=${c('TN')} · inner K=${c('TK')} · outer N=${c('QKV_N_TILE')}`],
          ['Layout · QKV split-K', `${qkvOk} K slices · slice ${c('QKV_K_SLICE')} · ${c('QKV_K_CHUNKS')} inner K chunks`],
          ['Layout · Paged address', 'block_table[b, block] → physical page；slot_mapping → current-token row'],
          ['Layout · FA work item', `${c('TOKENS_PER_SPLIT')} tokens / ${c('BLOCKS_PER_SPLIT')} KV block · GP_SIZE=${c('GP_SIZE')} KV heads`],
          ['Layout · Out projection', `N split ${nSplitsOut} × K split ${kSplitsOut} · OUT_TN=${c('OUT_TN')} · OUT_TK=${c('OUT_TK')}`],
          ['Layout · MLP / Down', `MLP_TN=${c('MLP_TN')} · inner K=${c('MLP_INNER_TK')} · chunk=${c('MLP_OUT_CHUNK')} · Down K=${c('DOWN_TK')}`],
        ],
        visual: {
          kpis: [
            ['Batch', batch, 'real sequences'],
            ['Hidden', hidden, 'model width'],
            ['Q / KV', `${c('NUM_HEADS')} / ${c('NUM_KV_HEADS')}`, `head_dim ${headDim}`],
            ['Q rows', `${qHeadBatch} → ${qHeadPad}`, 'valid_shape'],
          ],
          flow: [
            { kicker: 'INPUT', title: 'hidden_states', detail: `[${batch}, ${hidden}] · BF16 → FP32`, tone: 'source' },
            { kicker: 'PROJECTION', title: 'Q / K / V', detail: `Q [${batch}, ${hidden}] · K/V [${batch}, ${kvHidden}] · FP32`, tone: 'compute' },
            { kicker: 'ATTENTION', title: 'Paged KV + FA', detail: `[${cacheRows}, ${headDim}] × 2 · BF16`, tone: 'attention' },
            { kicker: 'OUTPUT', title: 'out + normed_out', detail: `[${batch}, ${hidden}] · FP32 carry`, tone: 'output' },
          ],
          layout: [
            ['PAGED KV', `SEQ_TILE ${c('SEQ_TILE')} · BLOCK ${c('BLOCK_SIZE')}`, `MAX_CTX_BLOCKS ${maxCtxBlocks}`],
            ['TILE', `M ${c('TM')} · N ${c('TN')} · K ${c('TK')}`, `outer N ${c('QKV_N_TILE')}`],
            ['SPLIT-K', `${qkvOk} slices · ${c('QKV_K_CHUNKS')} chunks`, `slice ${c('QKV_K_SLICE')}`],
            ['WORK ITEM', `${c('TOKENS_PER_SPLIT')} tokens · ${c('BLOCKS_PER_SPLIT')} blocks`, `GP_SIZE ${c('GP_SIZE')}`],
          ],
        },
        note: `Shape 与 Layout 已合并展示：前半段是源码常量/函数签名，后半段是分页、padding、Tile 和 Split-K 参数。它描述物理切分意图，不等同于编译后最终 stride descriptor。${sourceFacts.compileBlocked ? '当前源码头部还标注了动态索引 codegen 阻塞。' : ''}`,
      },
      graph: {
        label: '计算图',
        meta: '源码任务 · Scope / TaskId / 资源一体化',
        rows: [],
        note: '计算图把源码中带 name_hint 的任务、显式 TaskId 边、逻辑 Scope 和调度资源放到同一张图中。',
      },
      scope: {
        label: 'Scope',
        meta: `manual boundary · ${line('scope')}`,
        rows: [
          ['Runtime boundary', `1 个 pl.manual_scope · ${line('scope')}`],
          ['Logical Scope 1', phase1],
          ['Logical Scope 2', phase2],
          ['Logical Scope 3', phase3],
          ['Manual dependency rule', 'manual_scope 内关闭 tensormap 自动依赖；显式 TaskId 负责跨任务顺序'],
          ['Outside boundary', taskNames.includes('dcr_xgamma') ? 'dcr_xgamma → out + normed_out → next layer' : '源码未找到 dcr_xgamma'],
          ['Parallel constructs', `${taskRecords.filter((task) => task.kind === 'spmd').length} SPMD · ${taskRecords.filter((task) => task.kind === 'at').length} pl.at declarations`],
        ],
        note: `Scope 1/2/3 是源码中的逻辑阶段；真正的 Runtime 边界由 manual_scope 决定。当前解析到 ${taskNames.length} 个带 name_hint 的任务声明。`,
      },
      deps: {
        label: '依赖',
        meta: `TaskId + source scan · ${line('deps')}`,
        rows: [
          ['Task declarations', `${taskNames.length} named tasks · ${explicitTasks.length} have explicit deps`],
          ['Layer carry → RMS', `prev_out_tids → rms_recip · ${dep('rms_recip')}`],
          ['Normed input → QKV', `prev_normed_tids → q_proj/k_proj/v_proj · q=${dep('q_proj')}`],
          ['Work table → FA', `fa_work_build → fa_fused · ${dep('fa_fused')}`],
          ['RoPE → FA', `rope_qkv → fa_fused · ${dep('fa_fused')}`],
          ['FA → online reduce', `fa_fused → online_softmax · ${dep('online_softmax')}`],
          ['Attention → OutProj', `attn_done_tid → out_proj · ${dep('out_proj')}`],
          ['Down → carry', `down_tids → dcr_xgamma · ${dep('dcr_xgamma')}`],
        ],
        note: `当前面板列出源码中解析到的关键显式边；Tensor 自动依赖、循环 carry、WAR/WAW 推导仍需编译后 IR 才能最终确认。`,
      },
      resource: {
        label: '资源',
        meta: `declared scheduling intent · ${line('resource')}`,
        rows: [
          ['FA fused grid', `NUM_CORES=${c('NUM_CORES')} persistent blocks · grid-stride over real blocks`],
          ['RoPE grid', `ROPE_CORES=${c('ROPE_CORES')} · ${c('NUM_KV_HEADS')} × ${batch} head/batch items`],
          ['Online softmax', `NUM_CORES × 2 = ${value('NUM_CORES', 0) * 2} Vector blocks · work=${c('OS_WORK')}`],
          ['Q / K / V projection', `Q ${qOn * qkvOk} · K ${kvOn * qkvOk} · V ${kvOn * qkvOk} split tasks`],
          ['Out projection', `${nSplitsOut * kSplitsOut} atomic tasks · ${nSplitsOut} N tiles × ${kSplitsOut} K slices`],
          ['MLP projection', `gate/up ${mlpOn * kSplitsMlp} each · down ${downOn * kSplits} · FP32 atomic accum`],
          ['Layer-tail carry', `dcr_xgamma ${downOn}-way SPMD · out + normed_out written together`],
          ['Work table capacity', `FA_TABLE_CAP=${c('FA_TABLE_CAP')} entries · BATCH × MAX_CTX_BLOCKS`],
        ],
        note: `这些是源码声明的调度资源意图，不是设备实测 occupancy 或耗时。${sourceFacts.compileBlocked ? '源码头部明确提示 dynamic INDEX/i64 store offset 当前可能阻塞 codegen。' : ''}`,
      },
    };
  }

  function buildDecodeLayerGraph(source) {
    const parsed = parseDecodeLayerSource(source);
    const { integer, number, taskRecords, taskNames, taskByName } = parsed;
    const c = (name, fallback = '—') => integer(name, fallback);
    const value = (name, fallback = '—') => number(name, fallback);
    const graphWidth = 920;
    const nodeWidth = 176;
    const nodeHeight = 62;
    const centerX = 460;
    const details = {};
    const nodes = [];
    const clusters = [];
    const taskResources = (name) => {
      if (name === 'fa_fused') return `${c('NUM_CORES')} cores`;
      if (name === 'online_softmax') return `${value('NUM_CORES', 0) * 2} Vector`;
      if (name === 'rope_qkv') return `${c('ROPE_CORES')} cores`;
      if (name === 'q_proj') return `${c('Q_ON')}×${c('QKV_OK')} splits`;
      if (name === 'k_proj' || name === 'v_proj') return `${c('KV_ON')}×${c('QKV_OK')} splits`;
      if (name === 'out_proj') return `${c('N_SPLITS_OUT')}×${c('K_SPLITS_OUT')} atomic`;
      if (name === 'gate_proj' || name === 'up_proj') return `${c('MLP_ON')}×${c('K_SPLITS_MLP')} atomic`;
      if (name === 'down_proj') return `${c('DOWN_ON')}×${c('K_SPLITS')} atomic`;
      if (name === 'dcr_xgamma') return `${c('DOWN_ON')}-way SPMD`;
      if (name.endsWith('_seed')) return 'zero / seed';
      return '源码任务';
    };
    const taskRecord = (name) => taskByName(name) || { name, kind: 'at', alias: '', line: null, deps: '', depCount: 0 };

    const existing = (names) => names.filter((name) => taskNames.includes(name));
    const recordsFor = (names) => existing(names).map(taskRecord);
    const taskText = (names) => existing(names).join(' · ') || '源码任务未找到';
    const dependencyText = (records) => {
      const explicit = records.filter((record) => record.depCount > 0);
      if (!explicit.length) return '自动依赖 / 无显式 TaskId';
      return explicit.map((record) => `${record.alias || record.name}: ${record.deps}`).join('；');
    };
    const aggregate = ({ id, label, phase, kind = '计算节点', names = [], resource, colorKey, x, y, width = nodeWidth, height = nodeHeight, parent }) => {
      const records = recordsFor(names);
      const explicitCount = records.filter((record) => record.depCount > 0).length;
      const lineList = records.map((record) => record.line).filter(Boolean);
      const typeLabel = records.length
        ? `${kind} · ${records.length} tasks · TaskId ${explicitCount}`
        : `${kind} · ${resource || '源码数据'}`;
      const node = {
        id, label,
        typeLabel,
        kind: kind === 'Tensor' || kind === 'State' ? kind.toLowerCase() : 'op',
        x, y, width, height, colorKey, parent,
      };
      nodes.push(node);
      details[id] = {
        title: label,
        phase,
        kind,
        line: lineList.length ? lineList.join('、') : null,
        resource: resource || records.map((record) => `${record.name}: ${taskResources(record.name)}`).join(' · ') || '源码任务',
        deps: dependencyText(records),
        alias: records.map((record) => record.alias).filter(Boolean).join(' · ') || '—',
        tasks: existing(names).length ? taskText(names) : '',
      };
      return id;
    };

    const inputId = aggregate({ id: 'decode-input-hidden', label: 'hidden_states', phase: '入口边界', kind: 'Tensor', names: ['copy_hidden'], resource: `[${c('BATCH')}, ${c('HIDDEN')}] · BF16 → FP32`, colorKey: 'io:activation', x: centerX, y: 45, width: 210, height: 54 });
    const layerInputId = aggregate({ id: 'decode-layer-input', label: 'layer_input', phase: 'Layer boundary → Scope 1', kind: 'Tensor', names: ['x_gamma0'], resource: `cur FP32 + x×γ BF16 · [${c('BATCH')}, ${c('HIDDEN')}]`, colorKey: 'io:activation', x: centerX, y: 140, width: 210, height: 54 });
    const qkvId = aggregate({ id: 'decode-qkv-proj', label: 'RMSNorm + Q / K / V', phase: 'Scope 1 · RMSNorm + Q / K / V', names: ['rms_recip', 'q_seed', 'q_proj', 'k_seed', 'k_proj', 'v_seed', 'v_proj'], resource: `${c('Q_ON')}×${c('QKV_OK')} Q · ${c('KV_ON')}×${c('QKV_OK')} K/V`, colorKey: 'sem:norm', x: 150, y: 260, width: 190, height: 64 });
    const qkNormId = aggregate({ id: 'decode-qk-norm', label: 'qk_norm', phase: 'Scope 1 · RMSNorm + Q / K / V', names: ['qk_norm'], resource: 'Q/K fused norm · gamma + inv_rms', colorKey: 'sem:norm', x: 150, y: 350 });
    const workId = aggregate({ id: 'decode-fa-work-build', label: 'fa_work_build', phase: 'Scope 2 · Paged FA + online softmax', names: ['fa_work_build'], resource: `AIV prep · ${c('MAX_CTX_BLOCKS')} block table`, colorKey: 'sem:attention', x: 350, y: 260 });
    const ropeId = aggregate({ id: 'decode-rope-qkv', label: 'rope_qkv', phase: 'Scope 2 · Paged FA + online softmax', names: ['rope_qkv'], resource: `${c('ROPE_CORES')} cores · Q/K rotate + K/V write`, colorKey: 'sem:attention', x: 470, y: 350 });
    const cacheId = aggregate({ id: 'decode-paged-kv-cache', label: 'paged K / V cache', phase: 'Scope 2 · 状态', kind: 'State', resource: `[${c('CACHE_ROWS')}, ${c('HEAD_DIM')}] × 2 · BF16 · GM paged`, colorKey: 'io:state', x: 570, y: 260, width: 190, height: 54 });
    const faId = aggregate({ id: 'decode-fa-fused', label: 'fa_fused', phase: 'Scope 2 · Paged FA + online softmax', names: ['fa_fused'], resource: `${c('NUM_CORES')} cores · QK → softmax → SV`, colorKey: 'sem:attention', x: 470, y: 450 });
    const onlineId = aggregate({ id: 'decode-online-softmax', label: 'online_softmax', phase: 'Scope 2 · Paged FA + online softmax', names: ['online_softmax'], resource: `${value('NUM_CORES', 0) * 2} Vector · block partial reduce`, colorKey: 'sem:attention', x: 470, y: 550 });
    const attnOutId = aggregate({ id: 'decode-attn-out', label: 'attn_out', phase: 'Scope 2 → Scope 3', kind: 'Tensor', resource: `[${c('BATCH')}, ${c('HIDDEN')}] · BF16`, colorKey: 'io:activation', x: 570, y: 650, width: 170, height: 54 });
    const outProjId = aggregate({ id: 'decode-out-proj', label: 'out_proj', phase: 'Scope 3 · out_proj + MLP', names: ['out_seed', 'out_proj'], resource: `${c('N_SPLITS_OUT')}×${c('K_SPLITS_OUT')} atomic`, colorKey: 'sem:mlp', x: 760, y: 650 });
    const residualId = aggregate({ id: 'decode-residual-rms', label: 'residual_rms_cast', phase: 'Scope 3 · residual / RMS', names: ['residual_rms_cast'], resource: 'out_proj + residual · FP32', colorKey: 'sem:mlp', x: 700, y: 750 });
    const postRmsId = aggregate({ id: 'decode-post-rms', label: 'post_rms_reduce', phase: 'Scope 3 · residual / RMS', names: ['post_rms_reduce'], resource: 'FP32 reduce · inv_rms', colorKey: 'sem:norm', x: 820, y: 850 });
    const gateUpId = aggregate({ id: 'decode-gate-up-proj', label: 'Gate / Up projection', phase: 'Scope 3 · out_proj + MLP', names: ['gate_seed', 'gate_proj', 'up_seed', 'up_proj'], resource: `${c('MLP_ON')}×${c('K_SPLITS_MLP')} atomic`, colorKey: 'sem:mlp', x: 760, y: 950, width: 190, height: 64 });
    const siluId = aggregate({ id: 'decode-silu', label: 'SiLU', phase: 'Scope 3 · out_proj + MLP', names: ['silu'], resource: 'gate × silu · elementwise', colorKey: 'sem:mlp', x: 760, y: 1050 });
    const downId = aggregate({ id: 'decode-down-proj', label: 'down_proj', phase: 'Scope 3 · out_proj + MLP', names: ['down_seed', 'down_proj'], resource: `${c('DOWN_ON')}×${c('K_SPLITS')} atomic`, colorKey: 'sem:mlp', x: 760, y: 1150 });
    const dcrId = aggregate({ id: 'decode-dcr-xgamma', label: 'dcr_xgamma', phase: '层间边界 · fused output', names: ['dcr_xgamma'], resource: `${c('DOWN_ON')}-way SPMD · out + normed_out`, colorKey: 'sem:comm', x: 760, y: 1250 });
    const outputId = aggregate({ id: 'decode-output', label: 'out + normed_out', phase: '层间边界', kind: 'Tensor', names: ['cast_lmhead_in'], resource: `[${c('BATCH')}, ${c('HIDDEN')}] · FP32 / BF16`, colorKey: 'io:output', x: centerX, y: 1360, width: 220, height: 54 });

    const cluster = (id, label, x, y, width, height, colorKey, nodeList) => clusters.push({ id, label, x, y, width, height, colorKey, nodes: nodeList });
    cluster('decode-boundary-in', 'Layer boundary · runtime entry / carry', 30, 15, 860, 170, 'sem:comm', [inputId, layerInputId]);
    cluster('decode-scope1', 'Scope 1 · RMSNorm + Q / K / V', 45, 205, 220, 270, 'sem:norm', [qkvId, qkNormId]);
    cluster('decode-scope2', 'Scope 2 · Paged FA + online softmax', 285, 205, 360, 580, 'sem:attention', [workId, ropeId, cacheId, faId, onlineId, attnOutId]);
    cluster('decode-scope3', 'Scope 3 · out_proj + MLP', 650, 605, 245, 700, 'sem:mlp', [outProjId, residualId, postRmsId, gateUpId, siluId, downId, dcrId]);
    cluster('decode-boundary-out', 'Layer boundary · output / next layer', 30, 1320, 860, 120, 'sem:comm', [outputId]);

    const edges = [];
    const edgeMap = new Map();
    const addEdge = (source, target, tag, options = {}) => {
      if (!source || !target || source === target) return;
      const key = `${source}>${target}`;
      const existing = edgeMap.get(key);
      if (existing) {
        if (tag && !String(existing.tag || '').includes(tag)) existing.tag = `${existing.tag ? `${existing.tag} · ` : ''}${tag}`;
        return;
      }
      const edge = { source, target, tag, ...options };
      edgeMap.set(key, edge);
      edges.push(edge);
    };
    const semanticEdges = [
      [inputId, layerInputId, 'FP32 carry'],
      [layerInputId, qkvId, 'cur FP32 · TaskId prev_out_tids'], [qkvId, qkNormId, 'Q / K · inv_rms'],
      [qkNormId, ropeId, 'Q / K'], [layerInputId, workId, 'seq_lens', { dashed: true, relation: 'control' }],
      [workId, faId, 'dense blocks'], [ropeId, faId, 'Q padded'], [ropeId, cacheId, 'paged write', { dashed: true, relation: 'state' }], [cacheId, faId, 'paged read', { dashed: true, relation: 'state' }],
      [faId, onlineId, 'block partials'], [onlineId, attnOutId, 'reduce'], [attnOutId, outProjId, 'attn_out'],
      [outProjId, residualId, 'atomic out'], [outProjId, postRmsId, 'atomic out'], [layerInputId, residualId, 'residual FP32', { dashed: true, relation: 'state' }], [layerInputId, postRmsId, 'residual FP32', { dashed: true, relation: 'state' }], [residualId, postRmsId, 'residual'],
      [postRmsId, gateUpId, 'normed_in · TaskId'], [gateUpId, siluId, 'gate / up'], [siluId, downId, 'activation'], [downId, dcrId, 'down partials'], [residualId, dcrId, 'post_norm_partial', { dashed: true, relation: 'state' }],
      [dcrId, outputId, 'out + normed_out'], [outputId, layerInputId, 'next layer ×40', { dashed: true, relation: 'carry' }],
    ];
    semanticEdges.forEach(([source, target, tag, options]) => addEdge(source, target, tag, options));

    return {
      graph: routeDecodeLayerGraph({ width: graphWidth, height: 1460, nodes, clusters, edges }),
      details,
      summary: {
        named: taskNames.length,
        at: taskRecords.filter((task) => task.kind === 'at').length,
        spmd: taskRecords.filter((task) => task.kind === 'spmd').length,
        explicit: parsed.explicitTasks.length,
        manualLine: parsed.sourceFacts.manualLine,
        compileBlocked: parsed.sourceFacts.compileBlocked,
        resources: [`FA ${c('NUM_CORES')} cores`, `Online ${value('NUM_CORES', 0) * 2} Vector`, `Q/K/V ${c('Q_ON')}×${c('QKV_OK')} / ${c('KV_ON')}×${c('QKV_OK')}`, `MLP ${c('MLP_ON')}×${c('K_SPLITS_MLP')}`, `dcr ${c('DOWN_ON')}-way`],
      },
    };
  }

  // 右侧图沿用“模型结构”图的路由原则：长距离/回流关系走侧向 lane，
  // 同一节点的多条输入输出使用分散 port，避免所有关系挤在一个锚点上。
  function routeDecodeLayerGraph(graph) {
    const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
    const outgoing = new Map();
    const incoming = new Map();
    graph.edges.forEach((edge) => {
      if (!outgoing.has(edge.source)) outgoing.set(edge.source, []);
      if (!incoming.has(edge.target)) incoming.set(edge.target, []);
      outgoing.get(edge.source).push(edge);
      incoming.get(edge.target).push(edge);
    });
    outgoing.forEach((edges) => edges.sort((a, b) => (nodeById.get(a.target)?.y || 0) - (nodeById.get(b.target)?.y || 0)));
    incoming.forEach((edges) => edges.sort((a, b) => (nodeById.get(a.source)?.y || 0) - (nodeById.get(b.source)?.y || 0)));

    const routeHints = {
      'decode-input-hidden>decode-layer-input': { side: 'left', lane: 104 },
      'decode-layer-input>decode-fa-work-build': { side: 'left', lane: 300 },
      'decode-layer-input>decode-residual-rms': { side: 'right', lane: 885 },
      'decode-layer-input>decode-post-rms': { side: 'right', lane: 900 },
      'decode-residual-rms>decode-dcr-xgamma': { side: 'right', lane: 900 },
      'decode-output>decode-layer-input': { side: 'left', lane: 80 },
    };

    function portOffset(edge, collection, node, axis) {
      const list = collection.get(axis === 'source' ? edge.source : edge.target) || [];
      if (list.length < 2) return 0;
      const index = list.indexOf(edge);
      const raw = (index - (list.length - 1) / 2) * 18;
      return Math.max(-(node.width / 2 - 20), Math.min(node.width / 2 - 20, raw));
    }

    const edges = graph.edges.map((edge) => {
      const source = nodeById.get(edge.source);
      const target = nodeById.get(edge.target);
      if (!source || !target) return { ...edge };
      const routeKey = edge.routeKey || `${edge.source}>${edge.target}`;
      const hint = routeHints[routeKey];
      const routed = { ...edge, routeKey, waypoints: undefined, curve: undefined, route: 'rounded', cornerRadius: 14 };
      const dx = target.x - source.x;
      const dy = target.y - source.y;

      if (hint) {
        routed.sourceAnchor = { side: hint.side, dy: 0 };
        routed.targetAnchor = { side: hint.side, dy: 0 };
        routed.waypoints = [{ x: hint.lane, y: source.y }, { x: hint.lane, y: target.y }];
        routed.routeClass = 'side-lane';
        return routed;
      }

      if (dy < -70 || Math.abs(dy) > 360) {
        const useLeft = (source.x + target.x) / 2 < graph.width / 2;
        const lane = useLeft ? 90 : graph.width - 90;
        const side = useLeft ? 'left' : 'right';
        routed.sourceAnchor = side;
        routed.targetAnchor = side;
        routed.waypoints = [{ x: lane, y: source.y }, { x: lane, y: target.y }];
        routed.routeClass = 'side-lane';
        return routed;
      }

      if (Math.abs(dy) <= 54 && Math.abs(dx) > 40) {
        const sourceSide = dx > 0 ? 'right' : 'left';
        const targetSide = dx > 0 ? 'left' : 'right';
        routed.sourceAnchor = sourceSide;
        routed.targetAnchor = targetSide;
        routed.waypoints = [{ x: (source.x + target.x) / 2, y: source.y }, { x: (source.x + target.x) / 2, y: target.y }];
        routed.routeClass = 'horizontal-lane';
        return routed;
      }

      const sourceDx = portOffset(edge, outgoing, source, 'source');
      const targetDx = portOffset(edge, incoming, target, 'target');
      const forward = dy >= 0;
      const startY = source.y + (forward ? source.height / 2 : -source.height / 2);
      const endY = target.y + (forward ? -target.height / 2 : target.height / 2);
      const midY = (startY + endY) / 2;
      routed.sourceAnchor = { side: forward ? 'bottom' : 'top', dx: sourceDx };
      routed.targetAnchor = { side: forward ? 'top' : 'bottom', dx: targetDx };
      routed.waypoints = [{ x: source.x + sourceDx, y: midY }, { x: target.x + targetDx, y: midY }];
      routed.routeClass = Math.abs(dx) < 32 ? 'spine-lane' : 'branch-lane';
      return routed;
    });
    return { ...graph, edges };
  }

  const intentSource = window.PTO_DECODE_LAYER_SOURCE || '';
  const intentPreview = buildDecodeLayerIntent(intentSource);
  const decodeLayerGraph = buildDecodeLayerGraph(intentSource);

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
  const pagedAttentionSourceTags = {
    35: { focus: 'dynamic', label: '动态 Shape' },
    49: { focus: 'builder', label: 'Builder / Init' },
    93: { focus: 'qk', label: 'QK Matmul' },
    109: { focus: 'softmax', label: 'Softmax Prepare' },
    136: { focus: 'pv', label: 'PV Matmul' },
    151: { focus: 'online', label: 'Online Update' },
    237: { focus: 'orchestration', label: 'Orchestration' },
    289: { focus: 'paging', label: 'Paged KV' },
    368: { focus: 'golden', label: 'Torch Golden' },
    457: { focus: 'runtime', label: 'Runtime / Verify' },
  };
  const pagedAttentionObjectDetails = {
    dynamic: { object: 'pl.dynamic · 运行时维度', code: 'B, H, D, block_size = pl.dynamic(...)', semantic: '声明编译时未知、运行时绑定的维度；让同一 Kernel 覆盖不同 Batch、Context 与分页大小。', io: [['输入', 'Tensor.dim / 调用点实参'], ['输出', '符号维度 B、H、D、M'], ['Shape · dtype · layout', '[B×H,D] · BF16 · row-major'], ['Memory space', 'Host metadata → GM Tensor view']], deps: '被 Builder、Paged KV 与 q_loop 共同消费；不产生数据依赖边。', sync: '不产生 sync；只在编译时形成 Shape guard。', pass: 'Semantic Lowering（已解析）→ Layout Planning（待确认动态 stride）', isa: '当前无法从源码确认具体 ISA；预计落入 scalar address / shape guard。', impact: '动态性减少重编译，但保守边界检查会增加少量标量开销。', evidence: 'source' },
    builder: { object: 'Program Builder · Init', code: 'program = pl.program(...)  ·  fa_work_build(...)', semantic: '创建程序上下文并预分配工作表；把 q_tile、head_dim、block_size 固化为 Tile 级契约。', io: [['输入', '动态 Shape + page table'], ['输出', 'fa_work_table、初始化 Task'], ['Shape · dtype · layout', '[B×M] · INT32 · contiguous；work state · FP32'], ['Memory space', 'GM 持久化工作表；Task metadata 在编排侧']], deps: '初始化完成后，QK / Softmax / PV / Online 才可消费工作表。', sync: '需要 init → compute 的一次前置 event，避免未初始化工作表被读取。', pass: 'Semantic Lowering → Parallel Mapping（初始化 Task 与 compute Task 分离）', isa: 'AICPU / orchestration 生成 init dispatch；具体指令待 ISA Emission。', impact: '一次性初始化成本换取持久化 workspace；影响 GM 占用与首 token 延迟。', evidence: 'infer' },
    qk: { object: 'QK Matmul · Cube', code: 'sij = pl.matmul(q_tile, k_block.T)', semantic: '对当前 Query Tile 与一个 KV Block 做点积，得到未归一化 attention scores。', io: [['输入', 'q_tile / k_block'], ['输出', 'sij scores'], ['Shape · dtype · layout', '[QTile, D] × [D, Block] → [QTile, Block] · FP32 accum · tile'], ['Memory space', 'GM → L1 → L0A/L0B；结果 L0C / UB']], deps: 'RAW: q_tile、paged K block → sij → valid_len mask；跨 Block 无状态依赖。', sync: 'CopyIn 完成后才可发 Cube；通常由 event/fence 连接 GM→L1 与 Matmul。', pass: 'Layout Planning（转置 K view）→ Parallel Mapping（Q head × block）→ Memory Scheduling', isa: '预计 Cube MatMul + Load2D / DataCopy；真实 opcode 待 ISA Emission。', impact: '主计算热点；Block 越大 Cube 利用率越高，但 L1/L0 工作集与尾块浪费增加。', evidence: 'infer' },
    softmax: { object: 'Softmax Prepare · Vector', code: 'pij, mi, li = softmax_prepare(sij, valid_len)', semantic: '应用 scale 与末块 mask，计算 row max / exp / row sum，并把概率压到 BF16。', io: [['输入', 'sij + valid_len'], ['输出', 'pij、mi、li'], ['Shape · dtype · layout', '[QTile, valid] → [QTile,Block] · FP32→BF16 · row-major'], ['Memory space', 'L0C/GM → UB；state 留在 UB']], deps: 'RAW(sij)、RAW(valid_len) → pij；mi/li 将依赖传给 Online Update。', sync: 'Cube 写 sij 后需要 event；同一 UB region 上的 mask/exp 需要顺序 fence。', pass: 'Semantic Lowering → Layout Planning（mask view）→ Memory Scheduling（UB liveness）', isa: 'Vector exp/max/add/mul + cast；精确向量指令待 ISA Emission。', impact: 'Vector 受限于 exp 与 UB 带宽；BF16 概率降低带宽，但增加 cast。', evidence: 'infer' },
    pv: { object: 'PV Matmul · Cube', code: 'oi_tmp = pl.matmul(pij, v_block)', semantic: '用归一化概率加权当前 KV Block 的 V，形成一个可合并的输出块。', io: [['输入', 'pij + v_block'], ['输出', 'oi_tmp'], ['Shape · dtype · layout', '[QTile, Block] × [Block,D] → [QTile,D] · FP32 accum · tile'], ['Memory space', 'UB/L1 → L0A/L0B → L0C']], deps: 'RAW(pij)、读—读(v_block) → oi_tmp；oi_tmp → Online Update。', sync: 'pij cast/store 与 V CopyIn 完成后才能发 Cube；event 保护 L0 buffer 复用。', pass: 'Parallel Mapping → Memory Scheduling（L0A/L0B 双缓冲候选）', isa: 'Cube MatMul；Load2D / Move；精确流水组合待编译后确认。', impact: '与 QK 类似的 Cube 热点；双缓冲可隐藏搬运，但会增加 L1/L0 资源。', evidence: 'infer' },
    online: { object: 'Online Update · Vector', code: 'mi, li, oi = online_update(mi, li, oi, mi_new, li_new, oi_tmp)', semantic: '跨 KV Block 合并 running max、sum 与 output，末块完成归一化并写回。', io: [['输入', 'mi/li/oi state + mi_new/li_new/oi_tmp'], ['输出', 'loop-carried state；out'], ['Shape · dtype · layout', 'mi/li [QTile,1]、oi [QTile,D] · FP32 · row-major'], ['Memory space', 'UB loop-carried；末块 UB → GM']], deps: 'loop-carried RAW + WAW：第 n Block 的状态必须先于第 n+1 Block 更新。', sync: '必须插入 loop-carried fence；否则向量更新会读到旧 state，末块 store 还需 write-complete event。', pass: 'Dependency Analysis → Memory Scheduling（liveness）→ ISA Emission', isa: 'Vector max/sub/exp/mul/add/div + DataCopy GM store。', impact: '串行状态链限制 Block 间并行度；UB 保持 state 可减少 GM round-trip。', evidence: 'infer' },
    orchestration: { object: 'Runtime Orchestration · Host', code: 'q_loop = pl.range(query.rows // (B * H))', semantic: '从 Tensor.dim 和 page table 推导运行时循环边界，并把逻辑 Block 映射到物理 Cache Row。', io: [['输入', 'query.rows、context_lens、block_table'], ['输出', 'block_id、valid_len、物理 row'], ['Shape · dtype · layout', '标量 INT32 / INDEX；不改变 Tensor layout'], ['Memory space', 'AICPU / host 编排侧读取 GM metadata']], deps: '为每个 InCore Task 提供 index；与 Tensor 数据流是辅助依赖。', sync: 'index 计算完成后用 dispatch dependency 约束对应 block 的 CopyIn。', pass: 'Semantic Lowering → Parallel Mapping（batch/head/block）', isa: 'AICPU scalar address arithmetic；设备 opcode 待运行采集。', impact: '动态索引可增加编排开销；错误的 block/page 映射会直接破坏 cache locality。', evidence: 'source' },
    paging: { object: 'Paged KV · Address Mapping', code: 'row = block_table[b, block_id] * block_size + offset', semantic: '将逻辑 KV Block 映射到物理 Cache Row，并对最后一个 Block 施加 valid_len。', io: [['输入', 'block_table、context_lens、K/V cache'], ['输出', 'k_block、v_block view + valid_len'], ['Shape · dtype · layout', '[Block,D] × 2 · BF16 · paged / strided'], ['Memory space', 'GM paged KV；CopyIn 目标 L1']], deps: '读—读 page metadata；寻址结果控制 K/V CopyIn，valid_len 控制 mask。', sync: 'page lookup → CopyIn 的 dispatch dependency；末块 mask 前需确保 block 数据可见。', pass: 'Layout Planning（paged stride）→ Memory Scheduling（GM→L1）', isa: 'AddressGen + DataCopy / Load2D；真实 layout descriptor 待编译确认。', impact: '提升 KV 复用并降低连续内存需求；随机 page 会增加 GM 访问延迟。', evidence: 'source' },
    golden: { object: 'Torch Golden · Reference', code: 'torch_paged_attention(...)', semantic: '以 CPU/PyTorch 复现分页寻址、mask、BF16 cast 与 online softmax，作为结果对照。', io: [['输入', '同一组 query / cache / page table'], ['输出', 'FP32 reference out'], ['Shape · dtype · layout', '[B×H,D] · FP32 · contiguous'], ['Memory space', 'Host DRAM / Torch tensor']], deps: '不进入设备 Task DAG；只在 run 后与 device output 对比。', sync: '无设备 sync；对比阶段等待 device run complete。', pass: '不参与设备 Pass；属于 Correctness Lab 证据链。', isa: '无硬件指令；CPU reference。', impact: '增加验证时间与 host 内存，不影响 device kernel 性能。', evidence: 'source' },
    runtime: { object: 'Runtime / Verify · Host', code: 'run(program, inputs)  ·  torch.allclose(...)', semantic: '提交编译后的程序并以 Golden 校验输出；这里能把静态推断升级为实测证据。', io: [['输入', 'program + runtime tensors'], ['输出', 'TaskId、状态、timestamps、校验结果'], ['Shape · dtype · layout', '运行配置：B=64 · D=128 · BF16/FP32'], ['Memory space', '设备 GM / L1 / UB + Host result']], deps: '等待所有 Task complete，再执行 allclose；失败时回溯首个分歧。', sync: 'run completion fence 是验证边界；当前静态页面不伪造 TaskId 或耗时。', pass: 'ISA Emission 后的 Runtime 实测入口；状态待运行采集。', isa: '最终指令流待编译 / 设备 trace；当前仅显示预测路径。', impact: '可获得真实 occupancy、带宽、event 等指标；是性能结论的证据入口。', evidence: 'runtime' },
  };

  function pagedAttentionObjectDetail() {
    const focus = state.pagedAttentionFocus || 'paging';
    const detail = pagedAttentionObjectDetails[focus] || pagedAttentionObjectDetails.paging;
    const evidence = EVIDENCE_LEVELS[detail.evidence] || EVIDENCE_LEVELS.infer;
    return `<section class="kf-pa-object-detail" aria-label="选中对象详情">
      <header><div><span class="kf-eyebrow">SELECTED OBJECT · ${evidence.label}</span><h2>${detail.object}</h2><code>${detail.code}</code></div><span class="kf-pa-detail-line">源码 ${pagedAttentionFocusMeta[focus]?.lines || '—'}</span></header>
      <p class="kf-pa-detail-semantic">${detail.semantic}</p>
      <div class="kf-pa-detail-grid">${detail.io.map(([label, value]) => `<div><span>${label}</span><b>${value}</b></div>`).join('')}</div>
      <div class="kf-pa-detail-facts"><article><span>自动依赖</span><p>${detail.deps}</p></article><article><span>Sync / Event / Fence</span><p>${detail.sync}</p></article><article><span>Pass 修改点</span><p>${detail.pass}</p></article><article><span>最终硬件指令</span><p>${detail.isa}</p></article><article><span>性能与资源影响</span><p>${detail.impact}</p></article></div>
      <footer><i class="${evidence.cls}">${evidence.short}</i><span>源码选择会同步下方计算图、调度与依赖视图</span></footer>
    </section>`;
  }

  function pagedAttentionDetailView() {
    return `<div class="kf-pa-detail-view">
      <button type="button" class="kf-pa-detail-back" data-pa-detail-back aria-label="返回 Paged Attention 分析">← 返回分析</button>
      <div class="kf-pa-detail-intro"><span class="kf-eyebrow">ON-DEMAND INSPECTOR</span><h1>对象详情</h1><p>当前详情来自你选中的源码行或计算图节点。</p></div>
      ${pagedAttentionObjectDetail()}
    </div>`;
  }
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

  /* ──────────────────────────────────────────────────────────────────────────
   * V2 · Coding 阶段内容
   * 依据《PTO 调度与执行：开发者作业辅助内容规划》§3 证据分层、§5 Coding 阶段、
   * §9.1 计算图页签、§10 交互原则、§12 Agent 输出建议。
   * 全部事实取自 examples/models/06_paged_attention_dynamic.py。
   * ────────────────────────────────────────────────────────────────────────── */

  const EVIDENCE_LEVELS = {
    source: { label: '源码事实', short: '源码', cls: 'is-source' },
    infer: { label: '静态推断', short: '推断', cls: 'is-infer' },
    compile: { label: '编译事实 · 待编译', short: '待编译', cls: 'is-compile' },
    runtime: { label: 'Runtime 实测 · 待运行', short: '待运行', cls: 'is-runtime' },
    device: { label: '硬件实测 · 待采集', short: '待采集', cls: 'is-device' },
  };
  const ev = (level, text) => {
    const meta = EVIDENCE_LEVELS[level] || EVIDENCE_LEVELS.infer;
    return `<em class="kf-ev ${meta.cls}" title="${meta.label}">${text || meta.short}</em>`;
  };

  // 源码行 → 关注区间（renderFullSource 与卡片定位共用同一份映射）
  function pagedAttentionFocusForLine(lineNumber) {
    return lineNumber < 49 ? 'dynamic'
      : lineNumber < 93 ? 'builder'
      : lineNumber < 109 ? 'qk'
      : lineNumber < 136 ? 'softmax'
      : lineNumber < 151 ? 'pv'
      : lineNumber < 237 ? 'online'
      : lineNumber < 289 ? 'orchestration'
      : lineNumber < 368 ? 'paging'
      : lineNumber < 457 ? 'golden'
      : 'runtime';
  }

  // 只在源码面板内部滚动，不带动整页
  function scrollEditorRowIntoView(row) {
    let box = row.parentElement;
    while (box && box !== document.body && box.scrollHeight <= box.clientHeight + 1) box = box.parentElement;
    if (!box || box === document.body || box === document.documentElement) {
      row.scrollIntoView({ block: 'center', behavior: 'smooth' });
      return;
    }
    const settle = () => {
      const rowRect = row.getBoundingClientRect();
      const boxRect = box.getBoundingClientRect();
      const delta = (rowRect.top + rowRect.height / 2) - (boxRect.top + boxRect.height / 2);
      if (Math.abs(delta) < 2) return;
      box.scrollTo({ top: Math.max(0, box.scrollTop + delta), behavior: 'auto' });
    };
    settle();
    // 计算图等异步渲染会改变容器高度，下一帧再校正一次落点
    requestAnimationFrame(settle);
  }

  function markPagedAttentionTargetLine(line) {
    const row = $(`#dslEditor [data-paged-attention-line="${line}"]`);
    $$('#dslEditor .is-paged-attention-line-target').forEach((el) => el.classList.remove('is-paged-attention-line-target'));
    if (row) row.classList.add('is-paged-attention-line-target');
    return row;
  }

  // 卡片 → 源码：精确定位到该声明 / 该 Kernel 的定义行
  function revealPagedAttentionLine(line) {
    const target = Number(line);
    if (!target || !isPagedAttentionFile(state.activeFile)) return;
    state.pagedAttentionLine = target;
    const row = markPagedAttentionTargetLine(target);
    if (row) scrollEditorRowIntoView(row);
  }

  // §5.1 JIT 入口与函数层级
  // 唯一的结构视图 + 唯一的选择器：Host 侧 → 特化侧 → 编译侧，JIT 边界显式画出。
  // 每个节点自带详情（decl / role / detail），选中后由下方详情卡展开。
  const pagedAttentionLayerMap = [
    {
      id: 'host', tone: 'host', title: 'Host 侧', hint: '普通 Python · 逐行执行 · 不产生 Task',
      nodes: [
        {
          id: 'main', name: 'main()', line: 487, focus: 'runtime', badge: 'Runner', tone: 'py',
          note: '下面三步都由它按顺序调用',
          decl: 'def main()', role: 'Host Runner · 最外层调用者', evidence: 'source',
          detail: '按 ① 取 Program → ② 造 Tensor → ③ run() 提交 → ④ golden 比对 的顺序执行。选 platform="a2a3"（Ascend910B）、strategy=Default，自身不生成任何 Task。',
        },
        {
          id: 'tensors', name: 'build_tensors()', line: 457, focus: 'runtime', badge: 'Tensor', tone: 'py',
          step: '②', callLine: 518, note: '真实 Shape 在这里才确定',
          decl: 'def build_tensors(params)', role: 'Tensor 构造 · 动态维在此定形', evidence: 'source',
          detail: 'query / key_cache / value_cache / block_table / out 五个 Tensor 的具体维度在这里才确定，运行时正是按它们解析 BATCH_DYN 等 pl.dynamic 符号。在 main() 第 518 行调用。',
        },
        {
          id: 'golden', name: 'golden(tensors)', line: 368, focus: 'golden', badge: 'Oracle', tone: 'py',
          step: '④', callLine: 546, note: 'run() 之后才比对，torch 复现同一算法',
          decl: 'def golden(tensors, params)', role: 'Torch 参考实现 · 精度 Oracle', evidence: 'source',
          detail: '用 torch 复现同一套 paged attention + online softmax，作为 rtol / atol = 2e-2 比对的基准。在 main() 第 546 行、run() 之后调用。它与编译路径完全独立，只能验证最终结果，覆盖不到中间 Tile 的边界情况。',
        },
      ],
    },
    {
      id: 'spec', tone: 'spec', title: '特化侧', hint: '实参固化为闭包常量 · 决定编译缓存键',
      nodes: [
        {
          id: 'builder', name: 'build_dynamic_paged_attention_program(…)', line: 49, focus: 'builder', badge: 'Builder', tone: 'builder',
          note: 'q_tile / head_dim / block_size → _Q_TILE / _HEAD_DIM / _BLOCK_SIZE',
          decl: '普通 Python 函数（非 JIT）', role: 'Host Builder · 编译特化入口', evidence: 'source',
          detail: '把 q_tile / head_dim / block_size 固化为闭包常量 _Q_TILE / _HEAD_DIM / _BLOCK_SIZE（第 67–69 行），供 5 个 InCore 的 pl.load / pl.slice 当字面量使用；Tensor 标注仍保持 pl.dynamic。每次以不同实参调用都会得到一个独立的 Program 与独立的编译缓存键。',
        },
      ],
    },
    {
      id: 'jit', tone: 'jit', title: '编译侧 · 任务图', hint: '1 个 Orchestration 根 + 5 个 InCore 叶',
      container: {
        id: 'program', name: 'DynamicPagedAttentionProgram', line: 237, focus: 'orchestration', badge: '@pl.program', tone: 'container',
        note: 'Program 容器 · Builder 的返回值',
        decl: '@pl.program', role: 'Program 容器 · 编译单元边界', evidence: 'source',
        detail: '定义在 Builder 闭包内部并作为返回值。它是编译单元的边界：run() 接收的就是这个类，JIT 缓存也以它为单位。',
      },
      root: {
        id: 'orch', name: 'paged_attention', line: 249, focus: 'orchestration', badge: 'Orchestration', tone: 'orch',
        note: 'batch → q_tile → kv_block 三层 pl.range',
        decl: '@pl.function(type=pl.FunctionType.Orchestration)', role: '芯片级 Orchestration 入口 · 任务图唯一根', evidence: 'source',
        detail: '从 Tensor.dim 解析动态维度（第 270–277 行），用 ceil-div 算 Q Tile 循环次数（第 281 行），经 block_table 做分页寻址（第 303–307 行），并驱动 batch → q_tile → kv_block 三层 pl.range。所有 Task 都从这里发出。',
      },
      leaves: [
        {
          id: 'init', name: 'dyn_kernel_init_inplace', line: 76, focus: 'builder', badge: 'AIV', tone: 'aiv',
          note: '绑定动态 Shape，无实际计算',
          decl: '@pl.function(type=pl.FunctionType.InCore)', role: 'AIV / Vector · 无实际计算', evidence: 'infer',
          detail: 'pl.create_tensor 已完成零初始化，这个 Kernel 只在调用点（第 296 行）把具体 Shape 绑定到 pl.dynamic 标注上，把 oi / li / mi 三个累积缓冲的类型定下来。',
        },
        {
          id: 'qk', name: 'dyn_kernel_qk_matmul', line: 93, focus: 'qk', badge: 'AIC', tone: 'aic',
          note: 'sij = qi @ kjᵀ',
          decl: '@pl.function(type=pl.FunctionType.InCore)', role: 'AIC / Cube · sij = qi @ kjᵀ', evidence: 'source',
          detail: 'GM → L1(Mat) → L0A / L0B → L0C → GM。kj 经 pl.tile.transpose_view 做零拷贝转置后进 Right。调用点第 316 行，输出 sij [q_tile, block_size] FP32。',
        },
        {
          id: 'softmax', name: 'dyn_kernel_softmax_prepare', line: 109, focus: 'softmax', badge: 'AIV', tone: 'aiv',
          note: 'mi / li / pij 在线统计',
          decl: '@pl.function(type=pl.FunctionType.InCore)', role: 'AIV / Vector · online softmax 统计', evidence: 'source',
          detail: '按 valid_len（第 305 行）裁掉 KV 末块的无效列，行内求 max / exp / sum，产出 pij 与 mi / li 增量。调用点第 326 行。',
        },
        {
          id: 'pv', name: 'dyn_kernel_pv_matmul', line: 136, focus: 'pv', badge: 'AIC', tone: 'aic',
          note: 'oi_tmp = pij @ vj',
          decl: '@pl.function(type=pl.FunctionType.InCore)', role: 'AIC / Cube · oi_tmp = pij @ vj', evidence: 'source',
          detail: '第二个 Cube 段，M = q_tile、N = head_dim、K = block_size。调用点第 336 行，结果写入 oi_tmp 供下一步累积。',
        },
        {
          id: 'online', name: 'dyn_kernel_online_update', line: 151, focus: 'online', badge: 'AIV', tone: 'aiv',
          note: 'InOut 累积状态，形成 loop-carried 依赖',
          decl: '@pl.function(type=pl.FunctionType.InCore)', role: 'AIV / Vector · InOut 累积状态', evidence: 'source',
          detail: '把 oi_tmp 与 mi / li 增量并入累积器，参数标注为 pl.InOut，因此在 kv_block 循环上形成 loop-carried RAW + WAW 依赖 —— 这是整个 kernel 无法做搬运 / 计算重叠的根因。调用点第 351 行。',
        },
      ],
    },
  ];

  // 跨泳道的两次调用。步骤号 ①③ 在此，②④ 挂在 Host 泳道内不跨层的节点上，每个号只出现一次。
  const pagedAttentionFlows = {
    hostToSpec: {
      step: '①', label: 'build_…program(q_tile=16, …)', line: 512, focus: 'runtime',
      hint: 'main() 第 512 行 · 关键字调用 Builder 取得 Program · 点击跳转',
    },
    specToJit: {
      step: '③', label: 'run(program, …, RunConfig)', line: 526, focus: 'runtime',
      hint: 'main() 第 526 行 · 提交执行，跨过 JIT 边界 · 点击跳转',
    },
  };

  // 扁平索引：节点 id → 节点，供详情卡与联动查表
  const pagedAttentionNodeIndex = pagedAttentionLayerMap.reduce((acc, layer) => {
    [layer.container, layer.root, ...(layer.leaves || []), ...(layer.nodes || [])]
      .filter(Boolean)
      .forEach((node) => { acc[node.id] = node; });
    return acc;
  }, {});

  // 关注区间 → 代表节点，使别处（任务卡 / 依赖卡 / 源码行）改变选择时地图与详情卡同步
  const pagedAttentionFocusToNode = {
    dynamic: 'builder', builder: 'builder', qk: 'qk', softmax: 'softmax', pv: 'pv', online: 'online',
    orchestration: 'orch', paging: 'orch', golden: 'golden', runtime: 'main',
  };

  // §10.1 调用链，回答「我在哪一层」。节点级优先，精确到每个声明。
  const pagedAttentionNodeChain = {
    main: ['main()', 'run(program, tensors)', '→ 交给编译器'],
    tensors: ['main()', 'build_tensors(params)', '决定动态维实参'],
    golden: ['main()', 'golden(tensors)', 'torch · 不编译'],
    builder: ['main()', 'build_…program(16, 128, 128)', 'return Program'],
    program: ['build_…program()', '@pl.program', 'DynamicPagedAttentionProgram'],
    orch: ['run()', 'DynamicPagedAttentionProgram', 'paged_attention'],
    init: ['paged_attention', 'b_idx → q_idx', 'dyn_kernel_init_inplace'],
    qk: ['paged_attention', 'b_idx → q_idx → bn', 'dyn_kernel_qk_matmul'],
    softmax: ['paged_attention', 'b_idx → q_idx → bn', 'dyn_kernel_softmax_prepare'],
    pv: ['paged_attention', 'b_idx → q_idx → bn', 'dyn_kernel_pv_matmul'],
    online: ['paged_attention', 'b_idx → q_idx → bn', 'dyn_kernel_online_update'],
  };

  // 区间级兜底：源码里没有对应声明节点的关注区间（paging / dynamic 等）
  const pagedAttentionCallChain = {
    runtime: ['main()', 'run(program, tensors)', '→ 交给编译器'],
    golden: ['main()', 'golden(tensors)', 'torch · 不编译'],
    builder: ['main()', 'build_…program(16, 128, 128)', 'DynamicPagedAttentionProgram'],
    orchestration: ['run()', 'paged_attention', 'AICPU 编排'],
    dynamic: ['module 第 35–41 行', '7 × pl.dynamic', '被 5 个 InCore 标注引用'],
    paging: ['paged_attention', 'block_table 间接寻址', 'kv_block_row'],
    qk: ['paged_attention', 'b_idx → q_idx → bn', 'dyn_kernel_qk_matmul'],
    softmax: ['paged_attention', 'b_idx → q_idx → bn', 'dyn_kernel_softmax_prepare'],
    pv: ['paged_attention', 'b_idx → q_idx → bn', 'dyn_kernel_pv_matmul'],
    online: ['paged_attention', 'b_idx → q_idx → bn', 'dyn_kernel_online_update'],
  };

  // 本次编译条：读整个 section 的前提。
  // 「特化」一词只落在真正构成 JIT 特化键的那一组上（对齐 pypto/jit/specializer.py 与规划文档 §6.3）。
  // 前两组的配色与下方固 / 动两张卡一致。
  const pagedAttentionCurrentSpec = [
    {
      id: 'const', label: '特化键',
      hint: '闭包常量 · 构成 JIT 特化键，改动换一个 Program 与新的编译缓存',
      items: [
        { label: 'q_tile', value: '16' },
        { label: 'head_dim', value: '128' },
        { label: 'block_size', value: '128' },
      ],
    },
    {
      id: 'dyn', label: '动态维',
      hint: 'pl.dynamic 维 · 运行时按 Tensor.dim 解析，不参与特化，不触发重编译',
      items: [
        { label: 'batch', value: '64' },
        { label: 'heads', value: '16' },
        { label: 'ctx', value: '8192' },
      ],
    },
    {
      id: 'target', label: '目标',
      hint: 'run() 参数 · 改动会换编译产物，但不走特化机制',
      items: [
        { label: 'platform', value: 'a2a3 · 910B' },
        { label: 'strategy', value: 'Default' },
      ],
    },
  ];

  // 改动影响：哪些改动换 Program，哪些改动复用同一份代码
  const pagedAttentionSpecialization = [
    {
      kind: 'const', mark: '固', title: '闭包常量 · 编译期固化', effect: '改动 → 新 Program + 新缓存键 + 重新编译',
      items: ['_Q_TILE = 16', '_HEAD_DIM = 128', '_BLOCK_SIZE = 128'],
      note: '被 5 个 InCore 的 pl.load / pl.slice 直接当作字面量使用（第 67–69 行）。',
    },
    {
      kind: 'dyn', mark: '动', title: 'pl.dynamic 维 · 运行时解析', effect: '改动 → 复用同一份编译产物',
      items: ['BATCH_DYN', 'QUERY_ROWS_DYN', 'KEY_CACHE_ROWS_DYN', 'BLOCK_TABLE_FLAT_DYN'],
      note: '由 Tensor.dim 在 Orchestration 内解析（第 270–277 行），batch 变化不必然触发重编译。',
    },
  ];

  // §5.2 任务图预览 · 任务节点执行含义
  const pagedAttentionTasks = {
    init: {
      label: 'init_inplace', fn: 'dyn_kernel_init_inplace', fnType: 'InCore', focus: 'builder',
      defLine: 76, callLine: 296, invoke: 'Call', taskId: '不产生 · Call 不返回 TaskId',
      inputs: ['oi_buf [q_tile, D] FP32 · Out', 'li_buf [q_tile, 1] FP32 · Out', 'mi_buf [q_tile, 1] FP32 · Out'],
      outputs: ['oi / li_update / mi_update · 原值透传'],
      hardware: 'AIV / Vector（无实际计算）', memory: '不搬运 Tile，仅绑定类型',
      spmd: '普通 Kernel · 未声明 core_num / block',
      siblings: '每个 (b_idx, q_idx) 各调用一次，迭代之间数据独立',
      scope: 'AUTO', confidence: 'infer',
      why: 'pl.create_tensor 已完成零初始化；该 Kernel 只在调用点把具体 Shape 绑定到 pl.dynamic 标注（Q_HEADS / HEAD_DIM_DYN）上。',
    },
    qk: {
      label: 'QK Matmul', fn: 'dyn_kernel_qk_matmul', fnType: 'InCore', focus: 'qk',
      defLine: 93, callLine: 316, invoke: 'Call', taskId: '不产生 · Call 不返回 TaskId',
      inputs: ['qi [Q_HEADS, HEAD_DIM_DYN] BF16 · In', 'kj [BLOCK_SIZE_DYN, HEAD_DIM_DYN] BF16 · In', 'output [Q_HEADS, BLOCK_SIZE_DYN] FP32 · Out'],
      outputs: ['sij [q_tile, block_size] FP32'],
      hardware: 'AIC / Cube', memory: 'GM → L1(Mat) → L0A / L0B → L0C → GM',
      spmd: '普通 Kernel · 未声明 core_num / block',
      siblings: '同一 bn 内无可并行兄弟；跨 (b_idx, q_idx) 理论独立',
      scope: 'AUTO', confidence: 'infer',
      why: 'kj 以自然布局载入 L1 后用 pl.tile.transpose_view 取零拷贝转置视图，因此 Cube 实际算的是 qi × kjᵀ，FP32 累加落在 L0C。',
    },
    softmax: {
      label: 'Softmax Prepare', fn: 'dyn_kernel_softmax_prepare', fnType: 'InCore', focus: 'softmax',
      defLine: 109, callLine: 326, invoke: 'Call', taskId: '不产生 · Call 不返回 TaskId',
      inputs: ['sij_valid [q_tile, valid_len] FP32 · In', 'scale = 1.0 · Scalar[FP32]', 'out_pij BF16 · Out', 'out_mi / out_li FP32 · Out'],
      outputs: ['pij_f16 [q_tile, block_size] BF16', 'mi / li [q_tile, 1] FP32'],
      hardware: 'AIV / Vector', memory: 'GM → UB(Vec) → GM',
      spmd: '普通 Kernel · 未声明 core_num / block',
      siblings: '无 · 位于 Block 串行链中间',
      scope: 'AUTO', confidence: 'infer',
      why: 'row_max → row_expand_sub → exp 之后显式 cast 到 BF16 再转回 FP32 求 row_sum，这个降精度点是刻意的，golden 也复现了它。',
    },
    pv: {
      label: 'PV Matmul', fn: 'dyn_kernel_pv_matmul', fnType: 'InCore', focus: 'pv',
      defLine: 136, callLine: 336, invoke: 'Call', taskId: '不产生 · Call 不返回 TaskId',
      inputs: ['pij [Q_HEADS, BLOCK_SIZE_DYN] BF16 · In', 'vj [BLOCK_SIZE_DYN, HEAD_DIM_DYN] BF16 · In', 'output [Q_HEADS, HEAD_DIM_DYN] FP32 · Out'],
      outputs: ['oi_tmp [q_tile, D] FP32'],
      hardware: 'AIC / Cube', memory: 'GM → L1(Mat) → L0A / L0B → L0C → GM',
      spmd: '普通 Kernel · 未声明 core_num / block',
      siblings: '无 · 依赖 pij 与同 Block 的 V Slice',
      scope: 'AUTO', confidence: 'infer',
      why: 'pij 与 vj 都以自然布局载入，无 transpose_view；输出是本 Block 的未归一化部分和，等待 online_update 合并。',
    },
    online: {
      label: 'Online Update', fn: 'dyn_kernel_online_update', fnType: 'InCore', focus: 'online',
      defLine: 151, callLine: 351, invoke: 'Call', taskId: '不产生 · Call 不返回 TaskId',
      inputs: ['mij / lij [q_tile,1] FP32 · In', 'oi_new [q_tile,D] FP32 · In', 'mi / li / oi · InOut', 'dst = slice(out, ...) · Out', 'is_first / is_last · Scalar[BOOL]'],
      outputs: ['mi_out / li_out / oi_out · 跨 Block 状态', 'dst_out · 末块写归一化结果，否则写 0'],
      hardware: 'AIV / Vector', memory: '6 × load GM → UB · 4 × store UB → GM',
      spmd: '普通 Kernel · 未声明 core_num / block',
      siblings: '无 · InOut 状态构成 loop-carried 依赖',
      scope: 'AUTO', confidence: 'infer',
      why: 'mi / li / oi 声明为 pl.InOut，因此该任务既读又写同一 Tensor：这是把 KV Block 循环钉成串行的根本原因。',
    },
  };

  // §5.3 数据流与依赖叠加 · 选中连线后的依赖解释
  const pagedAttentionDeps = {
    sij: {
      label: 'sij_buf → sij_valid', type: 'RAW', producer: 'QK Matmul', consumer: 'Softmax Prepare', focus: 'softmax',
      region: '[q_tile, block_size] → slice [q_tile, valid_len]', dtype: 'FP32', origin: 'Tensor 自动依赖', confidence: 'source',
      reason: 'dyn_kernel_qk_matmul 把结果写入 sij_buf（第 316 行）；dyn_kernel_softmax_prepare 通过 sij_valid 读取同一 Region 的前 valid_len 列（第 320、326 行）。',
      evidence: '源码事实：第 315–332 行。最终 Task 顺序需 AutoDeriveTaskDependencies 后确认。',
    },
    pij: {
      label: 'pij_f16', type: 'RAW', producer: 'Softmax Prepare', consumer: 'PV Matmul', focus: 'pv',
      region: '[q_tile, block_size] 全区', dtype: 'BF16', origin: 'Tensor 自动依赖', confidence: 'source',
      reason: 'softmax_prepare 的 out_pij 是 pv_matmul 的 pij 实参，同一 SSA 值直接传递（第 326、336 行）。',
      evidence: '源码事实：第 323–336 行。编译期可确认（同一 SSA 值，无别名歧义）。',
    },
    oitmp: {
      label: 'oi_tmp', type: 'RAW', producer: 'PV Matmul', consumer: 'Online Update', focus: 'online',
      region: '[q_tile, head_dim] 全区', dtype: 'FP32', origin: 'Tensor 自动依赖', confidence: 'source',
      reason: 'pv_matmul 写 oi_tmp_buf，随后作为 online_update 的 oi_new 参数被读取（第 335–336、351 行）。',
      evidence: '源码事实：第 334–361 行。编译期可确认。',
    },
    mili: {
      label: 'mi / li（本 Block）', type: 'RAW', producer: 'Softmax Prepare', consumer: 'Online Update', focus: 'online',
      region: '[q_tile, 1] × 2', dtype: 'FP32', origin: 'Tensor 自动依赖', confidence: 'source',
      reason: 'softmax_prepare 产出的 mi / li 作为 online_update 的 mij / lij 传入，用于计算 alpha / beta 重标定因子。',
      evidence: '源码事实：第 326–332、351–361 行。',
    },
    carry: {
      label: 'mi / li / oi（跨 Block 状态）', type: 'RAW + WAW · loop-carried', producer: 'Online Update (bn)', consumer: 'Online Update (bn+1)', focus: 'online',
      region: 'mi/li [q_tile,1] · oi [q_tile,D]', dtype: 'FP32', origin: 'InOut 参数方向', confidence: 'source',
      reason: 'mi / li / oi 声明为 pl.InOut：同一任务先读旧值再写新值，下一次 bn 迭代又读这些值，形成循环携带依赖。',
      evidence: '源码事实：第 156–158（InOut 声明）、351–361（迭代内回写）行。这是 KV Block 无法并行的原因。',
    },
    kvread: {
      label: 'key_cache / value_cache', type: '读—读 · 无依赖', producer: '（外部输入，本入口不写）', consumer: 'QK / PV Matmul', focus: 'paging',
      region: 'slice [block_size, head_dim] @ block_id × block_size', dtype: 'BF16', origin: '同为 In 方向', confidence: 'source',
      reason: '两个 Cache 在本 Orchestration 内只被 pl.slice 读取，没有任何写入方，因此多个读取者之间不产生依赖。',
      evidence: '源码事实：第 253–254（In 声明）、310–312（slice 读取）行。',
    },
    outwrite: {
      label: 'out · slice(cur_offset)', type: 'WAW · 潜在保守', producer: 'Online Update (b, q)', consumer: 'Online Update (b′, q′)', focus: 'online',
      region: 'slice [q_tile, head_dim] @ cur_offset = b_idx × q_head_num + q_idx × q_tile', dtype: 'FP32', origin: 'Out 方向 + 动态偏移', confidence: 'infer',
      reason: '不同 (b_idx, q_idx) 写入 out 的行段在数学上互不相交，但 cur_offset 是运行时标量；若编译期无法证明区间不相交，会按潜在重叠建立 WAW，把本可并行的 tile 串行化。',
      evidence: '静态推断：第 289、350 行。需读 AutoDeriveTaskDependencies 后 IR 与 OverlapMap 才能确认是否真的串行。',
    },
  };

  // §5.4 AUTO / MANUAL Scope
  const pagedAttentionScopeFacts = {
    mode: 'AUTO',
    facts: [
      ['pl.manual_scope', '未使用', 'ok'],
      ['pl.submit(..., deps=[...])', '未使用 · 全部为 Call', 'ok'],
      ['no_dep / manual_dep', '未使用', 'ok'],
      ['显式 TaskId 集合', '空', 'ok'],
      ['最终依赖', '自动依赖 ∪ 显式依赖 = 自动依赖', 'ok'],
    ],
    risks: [
      {
        title: '过度串行风险 · Paged 间接寻址',
        body: 'key_cache / value_cache 的 Slice 起点来自 block_table 读出的运行时标量（第 303、307、310 行）。编译期无法证明不同 bn 的 Region 不相交；一旦按保守重叠处理，读—读本应无依赖的路径也可能被排成串行。',
        level: 'infer',
      },
      {
        title: '过度串行风险 · out 写回',
        body: 'out 的写回视图 slice(out, [q_tile, head_dim], [cur_offset, 0]) 使用动态 cur_offset（第 350 行）。行段实际不相交，但保守 WAW 会让不同 (b_idx, q_idx) 的 tile 失去并行机会。',
        level: 'infer',
      },
      {
        title: '未覆盖依赖风险 · 无',
        body: '本文件不存在 MANUAL Scope，因此没有"读了别人写的 Tensor 但 deps 里缺 TaskId"这类漏依赖。想显式接管顺序时才需要切换到 manual_scope + pl.submit。',
        level: 'source',
      },
    ],
  };

  // §5.5 硬件执行映射
  const pagedAttentionHardwareMap = [
    { name: 'paged_attention', type: 'Orchestration', core: 'AICPU', path: '标量运算 · 循环驱动 · Kernel 派发', level: 'infer' },
    { name: 'dyn_kernel_init_inplace', type: 'InCore', core: 'AIV / Vector', path: '仅类型绑定 · 无搬运', level: 'infer' },
    { name: 'dyn_kernel_qk_matmul', type: 'InCore', core: 'AIC / Cube', path: 'GM → L1 → L0A / L0B → L0C → GM', level: 'infer' },
    { name: 'dyn_kernel_softmax_prepare', type: 'InCore', core: 'AIV / Vector', path: 'GM → UB → GM', level: 'infer' },
    { name: 'dyn_kernel_pv_matmul', type: 'InCore', core: 'AIC / Cube', path: 'GM → L1 → L0A / L0B → L0C → GM', level: 'infer' },
    { name: 'dyn_kernel_online_update', type: 'InCore', core: 'AIV / Vector', path: 'GM → UB → GM', level: 'infer' },
  ];
  const pagedAttentionHardwareAbsent = [
    ['Group / 混合核（1C2V）', '未使用 · 5 个 Kernel 各自独立声明'],
    ['SPMD · core_num / block_idx', '未声明 · 对照 09_paged_attention_spmd.py'],
    ['sync_start / allow_early_resolve', '未声明'],
    ['TPUSH / TPOP 跨核 FIFO', '未使用 · Cube↔Vector 可能经 GM 往返'],
  ];

  // §5.6 核内 Tile 流水
  const pagedAttentionTilePipelines = {
    qk: {
      label: 'dyn_kernel_qk_matmul', core: 'AIC · Cube', lines: '93–108',
      steps: [
        { op: 'pl.load(qi, [_Q_TILE, _HEAD_DIM], Mat)', from: 'GM', to: 'L1', kind: 'copyin', note: 'Tile 尺寸来自 Builder 闭包常量' },
        { op: 'pl.load(kj, [_BLOCK_SIZE, _HEAD_DIM], Mat)', from: 'GM', to: 'L1', kind: 'copyin', note: 'K 以自然布局载入' },
        { op: 'pl.tile.transpose_view(kj_nat)', from: 'L1', to: 'L1 view', kind: 'view', note: '零拷贝转置视图，不产生搬运' },
        { op: 'pl.move(qi_l1, Left)', from: 'L1', to: 'L0A', kind: 'move', note: '' },
        { op: 'pl.move(kj_l1, Right)', from: 'L1', to: 'L0B', kind: 'move', note: '' },
        { op: 'pl.matmul(qi_l0a, kj_l0b)', from: 'L0A × L0B', to: 'L0C', kind: 'compute', note: 'BF16 输入 · FP32 累加' },
        { op: 'pl.store(sij_l0c, output)', from: 'L0C', to: 'GM', kind: 'copyout', note: '' },
      ],
    },
    softmax: {
      label: 'dyn_kernel_softmax_prepare', core: 'AIV · Vector', lines: '109–135',
      steps: [
        { op: 'pl.load(sij, [_Q_TILE, _BLOCK_SIZE], Vec)', from: 'GM', to: 'UB', kind: 'copyin', note: '' },
        { op: 'pl.mul(s_tile, scale)', from: 'UB', to: 'UB', kind: 'compute', note: 'scale 固定 1.0' },
        { op: 'pl.row_max → pl.row_expand_sub → pl.exp', from: 'UB', to: 'UB', kind: 'compute', note: 'FP32 逐行归约与指数' },
        { op: 'pl.cast(exp_tile, BF16) → pl.cast(..., FP32)', from: 'UB', to: 'UB', kind: 'compute', note: '刻意的降精度点' },
        { op: 'pl.row_sum(pij_tile, tmp_tile)', from: 'UB', to: 'UB', kind: 'compute', note: '' },
        { op: 'pl.store × 3（pij / mi / li）', from: 'UB', to: 'GM', kind: 'copyout', note: '' },
      ],
    },
    pv: {
      label: 'dyn_kernel_pv_matmul', core: 'AIC · Cube', lines: '136–150',
      steps: [
        { op: 'pl.load(pij, Mat) · pl.load(vj, Mat)', from: 'GM', to: 'L1', kind: 'copyin', note: '两者均为自然布局' },
        { op: 'pl.move(pij_l1, Left) · pl.move(vj_l1, Right)', from: 'L1', to: 'L0A / L0B', kind: 'move', note: '' },
        { op: 'pl.matmul(pij_l0a, vj_l0b)', from: 'L0A × L0B', to: 'L0C', kind: 'compute', note: 'BF16 输入 · FP32 累加' },
        { op: 'pl.store(oi_l0c, output)', from: 'L0C', to: 'GM', kind: 'copyout', note: '' },
      ],
    },
    online: {
      label: 'dyn_kernel_online_update', core: 'AIV · Vector', lines: '151–236',
      steps: [
        { op: 'pl.load × 6（mij / lij / oi_new / mi / li / oi）', from: 'GM', to: 'UB', kind: 'copyin', note: '其中 mi / li / oi 为 InOut' },
        { op: 'is_first 分支：直接赋值累加器', from: 'UB', to: 'UB', kind: 'compute', note: '首块无需重标定' },
        { op: 'pl.maximum → pl.sub → pl.exp（alpha / beta）', from: 'UB', to: 'UB', kind: 'compute', note: '' },
        { op: 'pl.reshape ↔ [1,Q] / [Q,1] · row_expand_mul', from: 'UB', to: 'UB', kind: 'compute', note: '广播轴与逐元素运算的布局要求不同' },
        { op: 'is_last 分支：row_expand_div(oi, li)', from: 'UB', to: 'UB', kind: 'compute', note: '仅末块归一化' },
        { op: 'pl.store × 4（mi / li / oi / dst）', from: 'UB', to: 'GM', kind: 'copyout', note: '非末块 dst 写 0' },
      ],
    },
  };
  const pagedAttentionLoopSemantics = [
    ['for b_idx in pl.range(batch_cfg)', '第 283 行', 'range · 未声明 parallel', '迭代间数据独立'],
    ['for q_idx in pl.range(q_loop_cfg)', '第 287 行', 'range · 未声明 parallel', '迭代间数据独立'],
    ['for bn in pl.range(bn_this_batch)', '第 298 行', 'range', 'mi / li / oi 状态 Carry · 必须串行'],
    ['pl.pipeline(stage=F)', '未出现', '未声明', '搬运与计算不表达重叠'],
    ['unroll', '未出现', '未声明', '—'],
  ];

  // §12 Agent 输出建议 · 结论 / 原因 / 证据 / 影响 / 建议 / 验证
  const pagedAttentionAgentFinding = {
    title: 'Q Head 尾 Tile 在动态 num_heads 下缺少有效 Shape 保护',
    severity: '正确性',
    sections: [
      { key: '结论', level: 'infer', body: 'q_loop_cfg 用 ceil-div 计算需要多少个 q_tile 覆盖全部 Q Head，但取 Q 的 slice 始终按固定 q_tile 取行。当 num_heads 不是 q_tile 的整数倍时，最后一个 q_idx 会越过本 request 的 Q 行段。' },
      { key: '原因', level: 'source', body: '第 281 行 q_loop_cfg = (q_head_num + q_tile - 1) // q_tile；第 289 行 cur_offset = b_idx * q_head_num + q_idx * q_tile；第 300 行 qi = pl.slice(query, [q_tile, head_dim_cfg], [cur_offset, 0])。三者组合下，尾 Tile 的 [cur_offset, cur_offset + q_tile) 会跨过 request 边界。' },
      { key: '证据', level: 'source', body: '对照 KV 侧：末 Block 已用 valid_len = pl.min(block_size_cfg, cur_seq - bn * block_size_cfg)（第 305 行）并通过 sij_valid 收窄（第 320 行）。Q 侧没有等价处理。main() 取 num_heads = 16、q_tile = 16 恰好整除（第 504、509 行），因此自带 golden 覆盖不到这条路径。' },
      { key: '影响', level: 'infer', body: '跨 request 的 Q 数据污染，属正确性问题而非性能问题；同时输出视图 slice(out, ..., [cur_offset, 0]) 也会写到相邻 request 的行段上。仅在 num_heads % q_tile ≠ 0 时触发。' },
      { key: '建议', level: 'infer', body: '为 Q 侧补一个与 valid_len 对称的量：valid_q = pl.min(q_tile, q_head_num - q_idx * q_tile)，并把 Q Tile 与 out 写回视图都收窄到 valid_q；InCore 的 load 尺寸仍可保持 _Q_TILE，只需让参与计算与写回的行数正确。' },
      { key: '验证', level: 'runtime', body: '新增参数化用例 num_heads ∈ {8, 17, 24} × q_tile = 16，与 torch golden 比 allclose(rtol = atol = 2e-2)；并单独断言 out 中相邻 request 的行段未被覆写。' },
    ],
  };

  function pagedAttentionMapNode(node, extraClass) {
    const active = node.id === state.pagedAttentionNode ? ' is-active' : '';
    const inFocus = !active && node.focus === state.pagedAttentionFocus ? ' is-in-focus' : '';
    const target = state.pagedAttentionLine === node.line ? ' is-line-target' : '';
    const step = node.step ? `<span class="kf-pa2-node-step" title="在 main() 第 ${node.callLine} 行调用">${node.step}</span>` : '';
    return `<button type="button" class="kf-pa2-node is-${node.tone}${active}${inFocus}${target}${extraClass ? ' ' + extraClass : ''}" data-paged-attention-focus="${node.focus}" data-pa2-node="${node.id}" data-pa2-line="${node.line}" title="跳到第 ${node.line} 行">
      <span class="kf-pa2-node-top">${step}<b>${node.name}</b><em>${node.badge}</em></span>
      <span class="kf-pa2-node-foot"><small>${node.note}</small><i>${node.line}</i></span>
    </button>`;
  }

  // 泳道之间的调用箭头：真实调用点，可点击跳转，与节点一致的行为
  function pagedAttentionFlowArrow(flow) {
    const target = state.pagedAttentionLine === flow.line ? ' is-line-target' : '';
    return `<button type="button" class="kf-pa2-flow${target}" data-paged-attention-focus="${flow.focus}" data-pa2-line="${flow.line}" title="${flow.hint}">
      <span><i>${flow.step}</i><code>${flow.label}</code><em>${flow.line}</em></span>
    </button>`;
  }

  function pagedAttentionCallChainBar() {
    const node = pagedAttentionNodeIndex[state.pagedAttentionNode];
    // 节点与当前关注区间一致时用节点链；否则（如源码点到 paging 区间）退回区间链
    const chain = (node && node.focus === state.pagedAttentionFocus && pagedAttentionNodeChain[node.id])
      || pagedAttentionCallChain[state.pagedAttentionFocus]
      || pagedAttentionCallChain.orchestration;
    return `<div class="kf-pa2-chain" aria-label="当前对象调用链"><span>调用链</span>${chain.map((hop) => `<code>${hop}</code>`).join('<i>›</i>')}</div>`;
  }

  function pagedAttentionEntrySection() {
    const active = pagedAttentionNodeIndex[state.pagedAttentionNode] || pagedAttentionNodeIndex.orch;
    const lane = (layer) => `
      <div class="kf-pa2-lane is-${layer.tone}">
        <header><b>${layer.title}</b><small>${layer.hint}</small></header>
        ${layer.container ? pagedAttentionMapNode(layer.container) : ''}
        ${layer.root ? pagedAttentionMapNode(layer.root, 'is-root-node') : ''}
        ${layer.leaves ? `<div class="kf-pa2-branch">${layer.leaves.map((node) => pagedAttentionMapNode(node)).join('')}</div>` : ''}
        ${layer.nodes ? layer.nodes.map((node) => pagedAttentionMapNode(node)).join('') : ''}
      </div>`;
    return `
      <section class="kf-inspector-section kf-pa2-entry"><header><h2 class="kf-inspector-title">JIT 入口与函数层级</h2></header>
        <div class="kf-pa2-current" aria-label="本次编译的实参与配置">
          <header><span>本次编译</span>${ev('source')}</header>
          ${pagedAttentionCurrentSpec.map((group) => `<div class="kf-pa2-current-row is-${group.id}"><i title="${group.hint}">${group.label}</i>${group.items.map((item) => `<b title="${group.hint}"><em>${item.label}</em>${item.value}</b>`).join('')}</div>`).join('')}
        </div>
        <div class="kf-pa2-map" role="tree" aria-label="编译分层地图">
          ${lane(pagedAttentionLayerMap.find((layer) => layer.id === 'host'))}
          ${pagedAttentionFlowArrow(pagedAttentionFlows.hostToSpec)}
          ${lane(pagedAttentionLayerMap.find((layer) => layer.id === 'spec'))}
          ${pagedAttentionFlowArrow(pagedAttentionFlows.specToJit)}
          <div class="kf-pa2-boundary"><span>JIT 边界</span><small>以下由编译器接管 · 才开始产生 Task</small></div>
          ${lane(pagedAttentionLayerMap.find((layer) => layer.id === 'jit'))}
        </div>
        <div class="kf-pa2-entry-detail"><div class="kf-pa2-entry-id"><b>${active.name}</b><code>${active.decl}</code></div><div><span>调度含义</span><b>${active.role}</b></div><p>${active.detail}</p><footer><code>第 ${active.line} 行</code>${ev(active.evidence)}</footer></div>
        ${pagedAttentionCallChainBar()}
        <div class="kf-pa2-spec"><header><span>改动影响 · 什么会触发重编译</span>${ev('source')}</header>
          <div class="kf-pa2-spec-split">${pagedAttentionSpecialization.map((group) => `
            <div class="is-${group.kind}"><header><i>${group.mark}</i><b>${group.title}</b></header>
              <div class="kf-pa2-spec-items">${group.items.map((item) => `<code>${item}</code>`).join('')}</div>
              <strong>${group.effect}</strong><small>${group.note}</small></div>`).join('')}
          </div></div>
      </section>`;
  }

  function pagedAttentionTaskSection() {
    const task = pagedAttentionTasks[state.pagedAttentionTask] || pagedAttentionTasks.qk;
    const order = ['init', 'qk', 'softmax', 'pv', 'online'];
    return `
      <section class="kf-inspector-section kf-pa2-tasks"><header><h2 class="kf-inspector-title">任务节点执行含义</h2><span>5 个 InCore Call</span></header>
        <div class="kf-pa2-task-rail" role="group" aria-label="任务列表">${order.map((key, index) => `<button type="button" class="${key === state.pagedAttentionTask ? 'is-active' : ''}" data-pa2-task="${key}"><i>${index}</i><span><b>${pagedAttentionTasks[key].label}</b><small>${pagedAttentionTasks[key].hardware.split(' ')[0]}</small></span></button>`).join('')}</div>
        <div class="kf-pa2-task-card">
          <header><div><b>${task.label}</b><code>${task.fn}</code></div><span>${task.fnType}</span></header>
          <dl>
            <div><dt>调用方式</dt><dd>${task.invoke}${ev('source', '第 ' + task.callLine + ' 行')}</dd></div>
            <div><dt>TaskId</dt><dd>${task.taskId}</dd></div>
            <div><dt>预计硬件</dt><dd>${task.hardware}${ev(task.confidence)}</dd></div>
            <div><dt>内存路径</dt><dd>${task.memory}</dd></div>
            <div><dt>SPMD / Group</dt><dd>${task.spmd}</dd></div>
            <div><dt>可并行兄弟</dt><dd>${task.siblings}</dd></div>
            <div><dt>所属 Scope</dt><dd>${task.scope} Scope · 自动依赖生效</dd></div>
          </dl>
          <div class="kf-pa2-io"><section><span>输入 / 输出参数</span><ul>${task.inputs.map((row) => `<li class="is-in">${row}</li>`).join('')}</ul></section><section><span>产出</span><ul>${task.outputs.map((row) => `<li class="is-out">${row}</li>`).join('')}</ul></section></div>
          <footer><i>↳</i><p>${task.why}</p></footer>
        </div>
      </section>`;
  }

  function pagedAttentionDepSection() {
    const dep = pagedAttentionDeps[state.pagedAttentionDep] || pagedAttentionDeps.sij;
    const depType = (value) => value.startsWith('RAW') ? 'is-raw' : value.startsWith('WAW') ? 'is-waw' : value.startsWith('WAR') ? 'is-war' : 'is-none';
    return `
      <section class="kf-inspector-section kf-pa2-deps"><header><h2 class="kf-inspector-title">依赖与数据流</h2></header>
        <div class="kf-pa2-dep-list">${Object.entries(pagedAttentionDeps).map(([key, item]) => `<button type="button" class="${key === state.pagedAttentionDep ? 'is-active' : ''}" data-pa2-dep="${key}"><span class="kf-pa2-dep-tensor">${item.label}</span><em class="${depType(item.type)}">${item.type}</em></button>`).join('')}</div>
        <div class="kf-pa2-dep-card">
          <div class="kf-pa2-dep-flow"><b>${dep.producer}</b><i><span>${dep.label}</span><small>${dep.type}</small></i><b>${dep.consumer}</b></div>
          <dl><div><dt>依赖类型</dt><dd>${dep.type}</dd></div><div><dt>Region</dt><dd>${dep.region}</dd></div><div><dt>DType</dt><dd>${dep.dtype}</dd></div><div><dt>来源</dt><dd>${dep.origin}</dd></div></dl>
          <p><b>原因</b>${dep.reason}</p>
          <footer>${ev(dep.confidence)}<span>${dep.evidence}</span></footer>
        </div>
      </section>`;
  }

  function pagedAttentionScopeSection() {
    const scope = pagedAttentionScopeFacts;
    return `
      <section class="kf-inspector-section kf-pa2-scope"><header><h2 class="kf-inspector-title">Scope 与依赖治理</h2><span>全程 ${scope.mode}</span></header>
        <div class="kf-pa2-scope-banner" data-mode="${scope.mode}"><b>${scope.mode} Scope</b><span>Tensor 生产 / 消费关系自动推导任务顺序，无需显式 TaskId</span>${ev('source')}</div>
        <div class="kf-pa2-scope-facts">${scope.facts.map(([name, value]) => `<div><span>${name}</span><b>${value}</b></div>`).join('')}</div>
        <div class="kf-pa2-scope-risks">${scope.risks.map((risk) => `<article class="is-${risk.level}"><header><b>${risk.title}</b>${ev(risk.level)}</header><p>${risk.body}</p></article>`).join('')}</div>
      </section>`;
  }

  function pagedAttentionHardwareSection() {
    return `
      <section class="kf-inspector-section kf-pa2-hardware"><header><h2 class="kf-inspector-title">硬件执行映射</h2></header>
        <div class="kf-pa2-hw-table"><div class="head"><span>函数</span><b>核 / 单元</b><em>数据路径</em></div>${pagedAttentionHardwareMap.map((row) => `<div><span>${row.name}<small>${row.type}</small></span><b>${row.core}</b><em>${row.path}</em></div>`).join('')}</div>
        <div class="kf-pa2-hw-absent"><header><span>未采用的调度能力</span>${ev('source')}</header>${pagedAttentionHardwareAbsent.map(([name, value]) => `<div><b>${name}</b><span>${value}</span></div>`).join('')}</div>
      </section>`;
  }

  function pagedAttentionEvidenceLegend() {
    return `<div class="kf-pa2-evidence-legend"><span>证据分层</span>${Object.entries(EVIDENCE_LEVELS).map(([key, meta]) => `<i class="${meta.cls}">${meta.label}</i>`).join('')}</div>`;
  }

  function pagedAttentionExecutionGraph() {
    const layerMeta = {
      data: { label: '数据', legend: '<i class="tensor"></i>Tensor Shape / 方向　<i class="dynamic"></i>运行时解析的动态维' },
      dep: { label: '依赖', legend: '<i class="raw"></i>RAW　<i class="waw"></i>WAW / loop-carried　<i class="none"></i>读—读无依赖' },
      hardware: { label: '硬件', legend: '<i class="cube"></i>AIC / Cube　<i class="vector"></i>AIV / Vector　<i class="memory"></i>GM / AICPU 编排' },
      precision: { label: '精度', legend: '<i class="bf16"></i>BF16 输入 / 概率　<i class="fp32"></i>FP32 计算 / 状态　<i class="index"></i>INT32 / INDEX' },
      runtime: { label: '运行状态', legend: '<i class="locked"></i>需要编译并运行后才有 TaskId、状态与时间戳' },
    }[state.pagedAttentionOverlay] || { label: '数据', legend: '' };
    const locked = state.pagedAttentionOverlay === 'runtime';
    return `
      ${pagedAttentionEvidenceLegend()}
      <section class="kf-inspector-section kf-pa-computation"><header class="kf-pa-graph-head"><div><h2 class="kf-inspector-title">任务计算图</h2></div></header>
        <div class="kf-pa2-layer-switch" role="group" aria-label="计算图信息图层">${[['data','数据'],['dep','依赖'],['hardware','硬件'],['precision','精度'],['runtime','运行状态']].map(([key,label]) => `<button type="button" class="${key === state.pagedAttentionOverlay ? 'is-active' : ''}${key === 'runtime' ? ' is-locked' : ''}" data-pa-overlay="${key}">${label}</button>`).join('')}</div>
        <div class="kf-pa-overlay-legend" data-overlay="${state.pagedAttentionOverlay}"><b>${layerMeta.label}图层</b><span>${layerMeta.legend}</span></div>
        ${locked ? '<div class="kf-pa2-locked"><i>○</i><div><b>运行状态图层尚无数据</b><p>TaskId、Ready / Running / Blocked / Complete、未满足依赖数与时间戳属于 Runtime 实测证据。Coding 阶段先建立静态任务图，编译并运行后同一批节点会切换为动态状态图。</p></div></div>' : ''}
        <div class="pto-model-graphviz-pattern-page pto-model-graphviz-stage kf-pa-computation__stage" id="pagedAttentionComputationGraph" aria-label="动态 Paged Attention 任务计算图"></div>
        <footer id="pagedAttentionGraphStatus">当前叠加${layerMeta.label}图层 · 带 ＋ 节点可下钻到核内步骤 · 虚线表示跨 Block 状态 Carry</footer>
      </section>`;
  }

  function pagedAttentionTilePipelineSection() {
    const pipe = pagedAttentionTilePipelines[state.pagedAttentionPipeKernel] || pagedAttentionTilePipelines.qk;
    const kindLabel = { copyin: 'CopyIn', view: 'View', move: 'Move', compute: 'Compute', copyout: 'CopyOut' };
    return `
      <section class="kf-inspector-section kf-pa2-tile"><header><h2 class="kf-inspector-title">核内 Tile 流水</h2></header>
        <div class="kf-pa2-tile-switch" role="group" aria-label="InCore Kernel 选择">${Object.entries(pagedAttentionTilePipelines).map(([key, item]) => `<button type="button" class="${key === state.pagedAttentionPipeKernel ? 'is-active' : ''}" data-pa2-pipe="${key}"><b>${item.label.replace('dyn_kernel_', '')}</b><small>${item.core}</small></button>`).join('')}</div>
        <div class="kf-pa2-tile-head"><b>${pipe.label}</b><span>${pipe.core}</span><code>第 ${pipe.lines} 行</code></div>
        <ol class="kf-pa2-tile-steps">${pipe.steps.map((step) => `<li class="is-${step.kind}"><em>${kindLabel[step.kind]}</em><div><code>${step.op}</code><span>${step.from} → ${step.to}</span>${step.note ? `<small>${step.note}</small>` : ''}</div></li>`).join('')}</ol>
        <div class="kf-pa2-loop-table"><header><span>循环与流水语义</span>${ev('source')}</header><div class="head"><span>语句</span><b>位置</b><em>语义</em><i>调度含义</i></div>${pagedAttentionLoopSemantics.map(([stmt, line, sem, meaning]) => `<div><span><code>${stmt}</code></span><b>${line}</b><em>${sem}</em><i>${meaning}</i></div>`).join('')}</div>
        <div class="kf-pa2-pipe-chart"><header><b>如果声明 pl.pipeline 会发生什么</b>${ev('infer')}</header>
          <div class="kf-pa2-pipe-legend"><span><i class="copyin"></i>CopyIn K / V Block</span><span><i class="compute"></i>QK · Softmax · PV</span><span><i class="copyout"></i>Online Update</span></div>
          <div class="kf-pa2-pipe-lanes"><span class="axis">时间 →</span>
            <div><i>bn 0</i><em class="copyin" style="--s:1;--n:1" title="CopyIn K/V Block 0"></em><em class="compute" style="--s:2;--n:3" title="QK · Softmax · PV">QK·SM·PV</em><em class="copyout" style="--s:5;--n:1" title="Online Update"></em></div>
            <div><i>bn 1</i><em class="copyin" style="--s:2;--n:1" title="CopyIn K/V Block 1 · 可与前一 Block 的计算重叠"></em><em class="compute" style="--s:5;--n:3" title="QK · Softmax · PV">QK·SM·PV</em><em class="copyout" style="--s:8;--n:1" title="Online Update"></em></div>
            <div><i>bn 2</i><em class="copyin" style="--s:3;--n:1" title="CopyIn K/V Block 2 · 可与前面计算重叠"></em><em class="compute" style="--s:8;--n:3" title="QK · Softmax · PV">QK·SM·PV</em><em class="copyout" style="--s:11;--n:1" title="Online Update"></em></div>
          </div>
          <p>当前源码没有 <code>pl.pipeline(stage=F)</code>。由于 <code>mi / li / oi</code> 是 loop-carried 状态，<b>计算段无法跨 bn 重叠</b>；能重叠的只有 K / V Block 的 CopyIn。上图是静态推断的可达形态，不是实测时序。</p>
        </div>
        <div class="kf-pa2-tile-note"><i>↳</i><div><b>一个 Coding 阶段可见的搬运冗余</b><p><code>qi = pl.slice(query, [q_tile, head_dim_cfg], [cur_offset, 0])</code> 位于 <code>bn</code> 循环内部（第 300 行），而 Q Tile 在整个 KV Block 循环中并不变化。每个 Block 都重新取一次 Q，意味着同一份 Q 会被反复搬入；把它提到 <code>bn</code> 循环外是编译期就能确认的改法。</p></div>${ev('source')}</div>
      </section>`;
  }

  function pagedAttentionAgentSection() {
    const finding = pagedAttentionAgentFinding;
    return `
      <section class="kf-inspector-section kf-pa2-agent"><header><h2 class="kf-inspector-title">Agent 结论</h2></header>
        <div class="kf-pa2-agent-head"><span>${finding.severity}</span><b>${finding.title}</b></div>
        <dl class="kf-pa2-agent-body">${finding.sections.map((item) => `<div><dt>${item.key}${ev(item.level)}</dt><dd>${item.body}</dd></div>`).join('')}</dl>
      </section>`;
  }

  function pagedAttentionDataExecution() {
    return `
      <section class="kf-pa-execution-band" aria-label="Paged Attention 数据与硬件执行带"><div class="source"><em>GM · BF16</em><b>Q [16,128]</b><small>4 KiB</small></div><i>load</i><button type="button" class="cube" data-paged-attention-focus="qk"><em>CUBE · L1/L0</em><b>QK Matmul</b><small>BF16 × BF16 → FP32 sij [16,128]</small></button><i>store / load</i><button type="button" class="vector" data-paged-attention-focus="softmax"><em>VECTOR · UB</em><b>Mask + Softmax</b><small>FP32 exp → BF16 pij [16,128]</small></button><i>store / load</i><button type="button" class="cube" data-paged-attention-focus="pv"><em>CUBE · L1/L0</em><b>PV Matmul</b><small>BF16 × BF16 → FP32 oi_new [16,128]</small></button><i>store / load</i><button type="button" class="vector" data-paged-attention-focus="online"><em>VECTOR · UB</em><b>Online Update</b><small>FP32 mi / li / oi → FP32 out</small></button><i>store</i><div class="source"><em>GM · FP32</em><b>Output [B×H,D]</b><small>512 KiB / example</small></div></section>
      ${pagedAttentionTaskSection()}
      <section class="kf-inspector-section kf-pa-layout"><header><h2 class="kf-inspector-title">Layout 叠加</h2><span>Shape · View · Memory</span></header><div class="kf-pa-layout-flow"><div><i>Query</i><b>[QTile,D]</b><small>BF16 · natural</small></div><span>×</span><div><i>K natural</i><b>[Block,D]</b><small>BF16 · L1</small></div><span>transpose_view</span><div><i>Kᵀ view</i><b>[D,Block]</b><small>零拷贝视图</small></div><span>→</span><div><i>Score</i><b>[QTile,Block]</b><small>FP32 · L0C</small></div></div></section>
      <section class="kf-inspector-section kf-pa-validshape"><header><h2 class="kf-inspector-title">有效区与数据规模</h2><span>Block128 · valid_len dynamic</span></header><div><span class="is-valid"><b>有效 Token 列</b><small>进入 row_max / exp / row_sum</small></span><span class="is-pad"><b>Padding</b><small>末块排除</small></span></div><div class="kf-pa-working-set"><span><b>Q</b><em>4 KiB</em></span><span><b>K + V</b><em>64 KiB</em></span><span><b>sij</b><em>8 KiB FP32</em></span><span><b>pij</b><em>4 KiB BF16</em></span><span><b>oi state</b><em>8 KiB FP32</em></span></div></section>
      ${pagedAttentionHardwareSection()}
      <div class="kf-inspector-card kf-rms-estimate"><b>硬件可信边界</b><p>执行带把 MemorySpace 和算子语义叠加显示；A2/A3 上 Cube↔Vector 的真实 GM 往返、Buffer 地址和重叠程度仍需读取 Pass IR、Swimlane 与 PMU。</p></div>`;
  }

  function pagedAttentionSchedule() {
    const blocks = Array.from({ length: 16 }, (_, index) => `<i class="${index < 4 ? 'is-hot' : ''}">${index}</i>`).join('');
    return `
      <section class="kf-pa-summary-strip"><div><span>根入口</span><b>paged_attention</b></div><div><span>任务节点</span><b>5 × InCore Call</b></div><div><span>依赖治理</span><b>AUTO Scope</b></div></section>
      ${pagedAttentionEntrySection()}
      <section class="kf-pa-schedule-canvas"><div class="kf-pa-loop-rail"><div><i>B</i><span><b>64 Batch</b><small>pl.range</small></span></div><div><i>Q</i><span><b>1 Head Tile</b><small>16 heads ÷ QTile16</small></span></div><div><i>K</i><span><b>64 KV Blocks</b><small>8192 ÷ Block128</small></span></div></div><div class="kf-pa-schedule-main"><div class="kf-pa-tile-row"><button type="button" data-paged-attention-focus="qk"><b>QK</b><small>16×128×128</small></button><i>→</i><button type="button" data-paged-attention-focus="softmax"><b>Softmax</b><small>16×valid_len</small></button><i>→</i><button type="button" data-paged-attention-focus="pv"><b>PV</b><small>16×128×128</small></button><i>→</i><button type="button" data-paged-attention-focus="online"><b>Update</b><small>FP32 carry</small></button></div><div class="kf-pa-block-mini">${blocks}</div><div class="kf-pa-page-equation"><span>logical <b>bn</b></span><i>table[b × block_num + bn]</i><span>physical <b>block_id</b></span><i>× block_size</i><span>cache <b>row</b></span></div></div></section>
      ${pagedAttentionDepSection()}
      ${pagedAttentionScopeSection()}
      <section class="kf-pa-schedule-notes"><article><b>可并行</b><p>Batch 与 Q Tile 数据相互独立，但当前使用 <code>pl.range</code>，未显式声明并行。</p></article><article><b>必须串行</b><p>KV Block 之间通过 FP32 <code>mi/li/oi</code> 状态 Carry 形成循环依赖。</p></article><article><b>边界风险</b><p>KV 末块有 <code>valid_len</code>；Q Head 尾 Tile 尚缺对应有效 Shape。</p></article></section>
      ${pagedAttentionTilePipelineSection()}`;
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
      ${pagedAttentionAgentSection()}
      <section class="kf-inspector-section kf-pa-capability"><header><h2 class="kf-inspector-title">目标能力 Lens</h2><span>A2/A3 · Ascend910B</span></header><div><article class="is-supported"><i>✓</i><span><b>动态 Tensor 标注</b><small>pl.dynamic · Tensor.dim</small></span><em>源码采用</em></article><article class="is-supported"><i>✓</i><span><b>Cube Matmul</b><small>BF16 input · FP32 accumulate</small></span><em>源码采用</em></article><article class="is-supported"><i>✓</i><span><b>Vector Softmax primitives</b><small>row_max · exp · row_sum</small></span><em>源码采用</em></article><article class="is-caution"><i>!</i><span><b>动态有效宽度</b><small>sij_valid uses runtime valid_len</small></span><em>重点验证</em></article><article class="is-caution"><i>!</i><span><b>动态 Head 尾 Tile</b><small>fixed q_tile load/slice</small></span><em>能力缺口</em></article><article><i>○</i><span><b>Cube↔Vector 片上交接</b><small>A2/A3 可能经 GM Buffer</small></span><em>需 Pass/实测</em></article></div></section>
      <section class="kf-inspector-section kf-rms-validation"><header><h2 class="kf-inspector-title">当前验证设计</h2><span>源码自带 Golden</span></header><div class="kf-rms-proof"><div class="is-pass"><i>✓</i><p><b>Torch Golden 已实现</b><small>复现分页寻址、Mask 与 Online Softmax</small></p><em>直接证据</em></div><div class="is-pass"><i>✓</i><p><b>概率精度行为已对齐</b><small>pij 模拟 BF16 cast 后再转 FP32</small></p><em>直接证据</em></div><div class="is-pass"><i>✓</i><p><b>运行后执行 allclose</b><small>rtol = atol = 2e-2</small></p><em>源码门禁</em></div><div><i>○</i><p><b>动态 Shape 参数矩阵</b><small>Batch · Heads · D · Block · Context</small></p><em>缺失</em></div><div><i>○</i><p><b>Q Head 尾 Tile</b><small>num_heads % q_tile ≠ 0</small></p><em>高风险缺口</em></div><div><i>○</i><p><b>末 Block 与空 Context</b><small>valid_len · context_len 0/1/boundary</small></p><em>缺失</em></div></div></section>
      <section class="kf-inspector-section kf-attn-risks"><header><h2 class="kf-inspector-title">风险与守卫</h2><span>G · capability & risk</span></header><ul><li><b>Shape 可除性</b><span><code>query.rows % batch == 0</code>、<code>cache.rows % table.size == 0</code>、<code>table.size % batch == 0</code> 应成为显式守卫。</span></li><li><b>Page Table 合法性</b><span><code>cur_block_idx</code> 必须处于物理 Block 池范围内，否则 KV Slice 越界。</span></li><li><b>Scale 语义</b><span>固定 1.0 需要与模型调用点对齐，避免遗漏 Attention Scale。</span></li><li><b>资源与后端</b><span>片上工作集是静态估算；最终地址、GM Round Trip 和执行重叠必须读取 Pass IR、Swimlane 与 PMU。</span></li></ul></section>
      <section class="kf-inspector-section kf-pa-run"><header><h2 class="kf-inspector-title">示例运行画像</h2><span>main()</span></header><dl><div><dt>Platform / Backend</dt><dd>A2/A3 · Ascend910B</dd></div><div><dt>Batch / Heads</dt><dd>64 / 16</dd></div><div><dt>Head / Block</dt><dd>128 / 128</dd></div><div><dt>Context / Max model</dt><dd>8192 / 32768</dd></div><div><dt>Blocks / Request</dt><dd>64 used / 256 max</dd></div><div><dt>Optional evidence</dt><dd>L2 Swimlane</dd></div></dl></section>
      <button class="kf-rms-action" type="button" data-paged-attention-action="tests">＋ 生成动态 Shape 与分页边界测试</button>`;
  }

  function renderPagedAttentionInspector({ scrollToFocus = false } = {}) {
    pagedAttentionGraphController?.destroy?.();
    pagedAttentionGraphController = null;
    const tabs = { graph: '计算图', schedule: '调度', execution: '执行', validation: '风险与验证' };
    if (state.pagedAttentionDetailOpen) {
      $('#inspectorTitle').textContent = '对象详情';
      $('#inspectorMeta').textContent = '按需查看 · source ↔ node';
      $('#inspector').innerHTML = pagedAttentionDetailView();
      return;
    }
    const content = state.pagedAttentionTab === 'execution' ? pagedAttentionDataExecution()
      : state.pagedAttentionTab === 'schedule' ? pagedAttentionSchedule()
      : state.pagedAttentionTab === 'validation' ? pagedAttentionValidation()
      : pagedAttentionExecutionGraph();
    $('#inspectorTitle').textContent = 'Paged Attention 分析';
    $('#inspectorMeta').textContent = `coding · ${tabs[state.pagedAttentionTab] || tabs.graph}`;
    $('#inspector').innerHTML = `
      <section class="kf-pa-hero"><span class="kf-eyebrow">CODING AGENT · 代码将如何执行</span><div><b>paged_attention_dynamic</b><em>DYNAMIC SHAPE</em></div><small>1 Orchestration 入口 · 5 InCore Kernel · AUTO Scope · Paged KV · online softmax</small></section>
      <div class="kf-pa-tabs" role="tablist" aria-label="动态 Paged Attention 分析视图">${Object.entries(tabs).map(([key, label]) => `<button type="button" class="${key === state.pagedAttentionTab ? 'is-active' : ''}" data-paged-attention-tab="${key}">${label}</button>`).join('')}</div>
      <div class="kf-pa-view">${content}</div>
      <footer class="kf-rms-provenance"><span><i class="fact"></i>源码事实</span><span><i class="estimated"></i>静态推断</span><span><i class="resolved"></i>编译 / Runtime / 硬件证据待补</span></footer>`;
    $$('#dslEditor [data-paged-attention-focus]').forEach(row => row.classList.toggle('is-paged-attention-line-active', row.dataset.pagedAttentionFocus === state.pagedAttentionFocus));
    if (scrollToFocus) $(`#dslEditor [data-paged-attention-focus="${state.pagedAttentionFocus}"]`)?.scrollIntoView({ block: 'center', behavior: 'smooth' });
    if (state.pagedAttentionTab === 'graph') renderPagedAttentionComputationGraph();
  }

  function renderPagedAttentionComputationGraph() {
    const pattern = window.PtoModelGraphvizPattern;
    const stage = $('#pagedAttentionComputationGraph');
    const status = $('#pagedAttentionGraphStatus');
    if (!pattern || !stage) return;
    const overlays = {
      data: {
        'pa-query': ['[B×H, D] · In', 'io:activation'], 'pa-context': ['[B] · In', 'io:state'], 'pa-table': ['[B×MaxBlocks] · In', 'io:state'], 'pa-page': ['scalar block_id → row', 'sem:comm'], 'pa-kv': ['slice [Block, D] × 2 · In', 'io:state'], 'pa-qk': ['sij [q_tile, block_size] · Out', 'sem:linear'], 'pa-mask': ['[q_tile, valid_len] · view', 'sem:comm'], 'pa-softmax': ['pij [q,B] · mi/li [q,1] · Out', 'sem:softmax'], 'pa-pv': ['oi_tmp [q_tile, D] · Out', 'sem:linear'], 'pa-online': ['mi/li/oi · InOut · dst · Out', 'sem:softmax'], 'pa-out': ['[B×H, D] · Out', 'io:output'],
      },
      dep: {
        'pa-query': ['读—读 · 无依赖', 'io:activation'], 'pa-context': ['读—读 · 无依赖', 'io:state'], 'pa-table': ['读—读 · 无依赖', 'io:state'], 'pa-page': ['标量寻址 · 无 Tensor 依赖', 'sem:comm'], 'pa-kv': ['读—读 · 无依赖', 'io:state'], 'pa-qk': ['RAW → Softmax(sij)', 'sem:linear'], 'pa-mask': ['视图收窄 · 同 Region', 'sem:comm'], 'pa-softmax': ['RAW → PV(pij) · RAW → Update(mi/li)', 'sem:softmax'], 'pa-pv': ['RAW → Update(oi_tmp)', 'sem:linear'], 'pa-online': ['loop-carried RAW + WAW', 'sem:softmax'], 'pa-out': ['WAW · 潜在保守', 'io:output'],
      },
      runtime: {
        'pa-query': ['待运行', 'io:state'], 'pa-context': ['待运行', 'io:state'], 'pa-table': ['待运行', 'io:state'], 'pa-page': ['待运行', 'io:state'], 'pa-kv': ['待运行', 'io:state'], 'pa-qk': ['TaskId 待编译 · 状态待运行', 'io:state'], 'pa-mask': ['待运行', 'io:state'], 'pa-softmax': ['TaskId 待编译 · 状态待运行', 'io:state'], 'pa-pv': ['TaskId 待编译 · 状态待运行', 'io:state'], 'pa-online': ['TaskId 待编译 · 状态待运行', 'io:state'], 'pa-out': ['待运行', 'io:state'],
      },
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
    const overlay = overlays[state.pagedAttentionOverlay] || overlays.data;
    const tensorOverlayLabels = {
      data: { 'pa-query': 'In', 'pa-context': 'In', 'pa-table': 'In', 'pa-out': 'Out' },
      dep: { 'pa-query': 'read', 'pa-context': 'read', 'pa-table': 'read', 'pa-out': 'write' },
      runtime: { 'pa-query': '—', 'pa-context': '—', 'pa-table': '—', 'pa-out': '—' },
      precision: { 'pa-query': 'BF16', 'pa-context': 'INT32', 'pa-table': 'INT32', 'pa-out': 'FP32' },
      shape: { 'pa-query': '[B×H,D]', 'pa-context': '[B]', 'pa-table': '[B×M]', 'pa-out': '[B×H,D]' },
      hardware: { 'pa-query': 'GM', 'pa-context': 'ORCH', 'pa-table': 'ORCH', 'pa-out': 'GM' },
    }[state.pagedAttentionOverlay] || {};
    const layerName = { data: '数据', dep: '依赖', hardware: '硬件', precision: '精度', runtime: '运行状态' }[state.pagedAttentionOverlay] || '数据';
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
          typeLabel: child[state.pagedAttentionOverlay] || child[{ data: 'shape', dep: 'precision', runtime: 'hardware' }[state.pagedAttentionOverlay]] || child.precision,
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
    const depEdgeTags = {
      'pa-context>pa-page': '读—读', 'pa-table>pa-page': '读—读', 'pa-page>pa-kv': '寻址',
      'pa-query>pa-qk': '读—读', 'pa-kv>pa-qk': '读—读', 'pa-qk>pa-mask': 'RAW · sij',
      'pa-context>pa-mask': 'valid_len', 'pa-mask>pa-softmax': 'RAW · 同 Region',
      'pa-softmax>pa-pv': 'RAW · pij', 'pa-kv>pa-pv': '读—读', 'pa-pv>pa-online': 'RAW · oi_tmp',
      'pa-softmax>pa-online': 'RAW · mi / li', 'pa-online>pa-online': 'loop-carried RAW+WAW',
      'pa-online>pa-out': 'WAW · 潜在保守',
    };
    const layerEdgeTag = (edge) => {
      if (state.pagedAttentionOverlay === 'dep') return depEdgeTags[`${edge.source}>${edge.target}`] || edge.tag;
      if (state.pagedAttentionOverlay === 'runtime') return null;
      return edge.tag;
    };
    const graphEdges = pagedAttentionComputationGraph.edges.map((edge) => {
      edge = { ...edge, tag: layerEdgeTag(edge) };
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
        if (pagedAttentionGraphFocus[nodeId]) state.pagedAttentionNode = nodeId;
        syncPagedAttentionSelection(focus);
        state.pagedAttentionDetailOpen = true;
        $$('#dslEditor [data-paged-attention-focus]').forEach(row => row.classList.toggle('is-paged-attention-line-active', row.dataset.pagedAttentionFocus === focus));
        const meta = pagedAttentionFocusMeta[focus];
        if (status && meta) status.textContent = `${meta.label} · 源码第 ${meta.lines} 行 · ${meta.detail}`;
        renderPagedAttentionInspector();
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
      if (status) status.textContent = `${pagedAttentionFocusMeta[expandedSpec.focus].label} 已展开 · 点击 − 收起 · 当前叠加${layerName}图层`;
    }
  }

  // §10.1 统一选择上下文：当前 Function → 当前 Task → 当前 Tensor / Dependency 保持联动
  const pagedAttentionFocusToTask = { builder: 'init', dynamic: 'init', qk: 'qk', softmax: 'softmax', pv: 'pv', online: 'online' };
  const pagedAttentionFocusToDep = { qk: 'sij', softmax: 'pij', pv: 'oitmp', online: 'carry', paging: 'kvread', orchestration: 'outwrite', dynamic: 'kvread', builder: 'sij', golden: 'sij', runtime: 'outwrite' };
  function syncPagedAttentionSelection(focus) {
    state.pagedAttentionFocus = focus;
    const task = pagedAttentionFocusToTask[focus];
    if (task) state.pagedAttentionTask = task;
    if (pagedAttentionTilePipelines[focus]) state.pagedAttentionPipeKernel = focus;
    const dep = pagedAttentionFocusToDep[focus];
    if (dep) state.pagedAttentionDep = dep;
    const node = pagedAttentionFocusToNode[focus];
    if (node) state.pagedAttentionNode = node;
  }
  const rmsNormProfiles = {
    input: {
      id: 'input', name: 'input_rmsnorm', role: 'Attention 前', scope: 'CORE_GROUP · rmsnorm', source: 'hidden_states', sourceType: 'BF16', weight: 'input_rms_weight', output: 'normed_states', chunk: 512, chunks: 16, stage: 4,
      cast: 'BF16 → FP32 → BF16', chunkBytes: '32 KiB', inputBytes: '256 KiB', scanBytes: '512 KiB', line: 27,
      upstream: 'hidden_states', downstream: 'Q / K / V projection', note: '输入为 BF16；两遍都先将当前 chunk 转为 FP32，再完成平方和与归一化。',
      chunkConst: 'RMSNORM_K_CHUNK', scopeLine: 33, loopLines: [36, 47], nameHint: 'rmsnorm', sharedWith: null
    },
    post: {
      id: 'post', name: 'post_rmsnorm', role: 'Attention 后 · MLP 前', scope: 'CORE_GROUP · post_rmsnorm', source: 'resid', sourceType: 'FP32', weight: 'post_rms_weight', output: 'post_norm_tile', chunk: 128, chunks: 64, stage: 2,
      cast: 'FP32 → BF16', chunkBytes: '8 KiB', inputBytes: '512 KiB', scanBytes: '1 MiB', line: 57,
      upstream: 'out_projection_residual', downstream: 'mlp_block', note: '残差流已经是 FP32，因此两遍扫描都不需要输入 cast；只在 assemble 前转为 BF16。',
      chunkConst: 'K_CHUNK', scopeLine: 69, loopLines: [72, 83], nameHint: 'post_rmsnorm', sharedWith: 'mlp_block'
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

  const RMS_UB_CAPACITY = 192 * 1024;
  const RMS_HIDDEN = 8192;
  const RMS_ROWS = 16;
  const RMS_BIG_HIDDEN = 32768;
  const RMS_CHUNK_CANDIDATES = [128, 256, 512, 1024, 2048];
  const RMS_STAGE_CANDIDATES = [2, 3, 4, 8];
  const rmsKiB = (bytes) => (bytes < 1024 ? `${bytes} B` : bytes >= 10240 ? `${Math.round(bytes / 1024)} KiB` : `${(bytes / 1024).toFixed(1)} KiB`);

  // 静态容量模型：pl.load / pl.assemble 产生的搬运 tile 受 stage 多缓冲，FP32 计算临时量与归约标量不随 stage 增长。
  function rmsUbPlan(profile, chunk, stage) {
    const srcBytes = RMS_ROWS * chunk * (profile.sourceType === 'BF16' ? 2 : 4);
    const gammaBytes = chunk * 4;
    const outBytes = RMS_ROWS * chunk * 2;
    const tempBytes = RMS_ROWS * chunk * 4;
    const reduceBytes = RMS_ROWS * 4 * 2;
    const staged = (srcBytes + gammaBytes + outBytes) * stage;
    const peak = staged + tempBytes + reduceBytes;
    const iters = Math.ceil(RMS_HIDDEN / chunk);
    const ratio = peak / RMS_UB_CAPACITY;
    return {
      chunk, stage, srcBytes, gammaBytes, outBytes, tempBytes, reduceBytes, staged, peak, iters, ratio,
      tail: RMS_HIDDEN % chunk, stageTail: iters % stage, safe: peak <= RMS_UB_CAPACITY, deep: iters >= stage * 2,
      verdict: peak > RMS_UB_CAPACITY ? 'over' : ratio > 0.9 ? 'tight' : 'safe',
    };
  }

  function rmsCurrentPlan(profile) {
    const saved = state.rmsNormPlan[profile.id] || {};
    return rmsUbPlan(profile, saved.chunk || profile.chunk, saved.stage || profile.stage);
  }

  // 求解器：在不溢出、两道整除守卫都通过、流水可填满的前提下，选择迭代次数最少的组合。
  function rmsBestPlan(profile) {
    let best = null;
    RMS_CHUNK_CANDIDATES.forEach((chunk) => RMS_STAGE_CANDIDATES.forEach((stage) => {
      const plan = rmsUbPlan(profile, chunk, stage);
      if (!plan.safe || plan.tail !== 0 || plan.stageTail !== 0 || !plan.deep || plan.ratio > 0.9) return;
      if (!best || plan.iters < best.iters || (plan.iters === best.iters && plan.stage > best.stage)) best = plan;
    }));
    return best;
  }

  // chip 状态：溢出（红）与「能跑但要补尾块 / 流水填不满」（琥珀）分开标注。
  function rmsChipState(probe) {
    if (!probe.safe) return { cls: ' is-over', note: '溢出 UB' };
    if (probe.tail !== 0) return { cls: ' is-tail', note: `末片仅 ${probe.tail} 列，需显式尾块` };
    if (probe.stageTail !== 0) return { cls: ' is-tail', note: `trip ${probe.iters} % stage ${probe.stage} ≠ 0，编译器补 tail dispatch` };
    if (!probe.deep) return { cls: ' is-tail', note: `仅 ${probe.iters} 次迭代，流水填不满` };
    return { cls: '', note: '两道守卫通过' };
  }

  function rmsLoopSkeleton(profile, plan) {
    const read = profile.id === 'input'
      ? `x = pl.cast(${profile.source}[:, k0 : k0 + CHUNK], target_type=pl.FP32)`
      : `x = ${profile.source}[:, k0 : k0 + CHUNK]`;
    return [
      `CHUNK = ${plan.chunk}                    # 求解器给出的安全 tile`,
      `STEPS = HIDDEN // CHUNK         # ${plan.iters} 次 / pass · ${plan.tail || plan.stageTail ? '需尾块处理' : '两道守卫通过'}`,
      '',
      `with pl.at(level=pl.Level.CORE_GROUP, name_hint="${profile.nameHint}"):`,
      '    acc = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)',
      '',
      `    for kb in pl.pipeline(STEPS, stage=${plan.stage}):     # Pass A · 归约`,
      '        k0 = kb * CHUNK',
      `        ${read}`,
      '        acc = pl.add(acc, pl.reshape(pl.row_sum(pl.mul(x, x)), [1, BATCH]))',
      '',
      '    inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(acc, HIDDEN_INV), EPS)))',
      '',
      `    for kb in pl.pipeline(STEPS, stage=${plan.stage}):     # Pass B · 归一化`,
      '        k0 = kb * CHUNK',
      `        ${read}`,
      '        y = pl.col_expand_mul(pl.row_expand_mul(x, inv_rms), gamma[:, k0 : k0 + CHUNK])',
      `        ${profile.output} = pl.assemble(${profile.output}, pl.cast(y, target_type=pl.BF16), [0, k0])`,
    ].join('\n');
  }

  function rmsNormLoops(profile) {
    const plan = rmsCurrentPlan(profile);
    const best = rmsBestPlan(profile);
    const isSourceValue = plan.chunk === profile.chunk && plan.stage === profile.stage;
    const bestIsSource = best && best.chunk === profile.chunk && best.stage === profile.stage;
    const barPct = Math.min(100, Math.round(plan.ratio * 100));
    const verdictLabel = { safe: 'SAFE', tight: '临界', over: '溢出' }[plan.verdict];

    const migrateRows = [
      ['tile shape 声明', `${profile.chunkConst} = ${profile.chunk} + 显式切片 [:, k0 : k0+CHUNK]`, 'config.py', null],
      ['自动展开切分循环', `pl.pipeline(HIDDEN // ${profile.chunkConst}) × 2 遍`, `L${profile.loopLines[0]} / L${profile.loopLines[1]}`, profile.loopLines[0]],
      ['自动 double buffer', `stage=${profile.stage} 显式参数`, `L${profile.loopLines[0]}`, profile.loopLines[0]],
      ['自动尾块处理', `HIDDEN % ${profile.chunk} = 0 · trip ${profile.chunks} % stage ${profile.stage} = 0`, '静态检查', null],
      ['自动 UB 分配', '峰值需自行核对 ≤ 192 KiB', '↓ 求解器', null],
    ];

    const nestRows = [
      ['pl.at · CORE_GROUP', `name_hint="${profile.nameHint}"`, '单个片上执行域，两遍扫描共享 UB', profile.scopeLine],
      ['Pass A · pl.pipeline', `kb: 0 → ${plan.iters} · stage=${plan.stage}`, '平方和跨 chunk 累加到 partial_sq', profile.loopLines[0]],
      ['Pass B · pl.pipeline', `kb: 0 → ${plan.iters} · stage=${plan.stage}`, '归一化 × gamma → assemble 写回', profile.loopLines[1]],
    ];

    const chunkChips = RMS_CHUNK_CANDIDATES.map((chunk) => {
      const probe = rmsUbPlan(profile, chunk, plan.stage);
      const chip = rmsChipState(probe);
      return `<button type="button" class="${chunk === plan.chunk ? 'is-active' : ''}${chip.cls}" data-rms-plan-chunk="${chunk}" title="chunk ${chunk} × stage ${plan.stage} · 峰值 ${rmsKiB(probe.peak)} · ${chip.note}">${chunk}</button>`;
    }).join('');
    const stageChips = RMS_STAGE_CANDIDATES.map((stage) => {
      const probe = rmsUbPlan(profile, plan.chunk, stage);
      const chip = rmsChipState(probe);
      return `<button type="button" class="${stage === plan.stage ? 'is-active' : ''}${chip.cls}" data-rms-plan-stage="${stage}" title="chunk ${plan.chunk} × stage ${stage} · 峰值 ${rmsKiB(probe.peak)} · ${chip.note}">${stage}</button>`;
    }).join('');

    const budgetRows = [
      ['输入 tile', plan.srcBytes * plan.stage, `[${RMS_ROWS}, ${plan.chunk}] · ${profile.sourceType} · ${rmsKiB(plan.srcBytes)} × stage ${plan.stage}`],
      ['gamma tile', plan.gammaBytes * plan.stage, `[1, ${plan.chunk}] · FP32 · ${rmsKiB(plan.gammaBytes)} × stage ${plan.stage}`],
      ['输出 tile', plan.outBytes * plan.stage, `[${RMS_ROWS}, ${plan.chunk}] · BF16 · ${rmsKiB(plan.outBytes)} × stage ${plan.stage}`],
      ['FP32 计算临时量', plan.tempBytes, `[${RMS_ROWS}, ${plan.chunk}] · FP32 · 不随 stage 增长`],
      ['归约标量', plan.reduceBytes, 'partial_sq · inv_rms · 常驻'],
    ];

    const advice = bestIsSource && isSourceValue
      ? '<b>当前取值已是最优</b><p>再增大 chunk 或加深 stage 都会越过 192 KiB。溢出风险来自参数而非结构，不需要为容量重构循环。</p>'
      : best
        ? `<b>推荐 chunk ${best.chunk} × stage ${best.stage}</b><p>峰值 ${rmsKiB(best.peak)}（${Math.round(best.ratio * 100)}%）· 迭代 ${plan.iters === best.iters ? `${best.iters}` : `${plan.iters} → ${best.iters}`} 次 / pass。${profile.sharedWith ? `<code>${profile.chunkConst}</code> 与 ${profile.sharedWith} 共用，应新增独立常量而不是就地改 config。` : ''}</p>`
        : '<b>无安全候选</b><p>该 shape 下所有 chunk × stage 组合都超过 UB，需要先切 BATCH 维再谈 H 维分块。</p>';

    const guardRows = [
      [plan.tail === 0, `HIDDEN % chunk = ${plan.tail}`, plan.tail === 0 ? `8192 % ${plan.chunk} · 切片覆盖完整 H，无残余列` : `8192 % ${plan.chunk} · 末片不足 chunk，需 valid_shape 或显式尾块`],
      [plan.stageTail === 0, `trip % stage = ${plan.stageTail}`, plan.stageTail === 0 ? `${plan.iters} % ${plan.stage} · 外层按 stage × step 推进，无需 tail dispatch` : `${plan.iters} % ${plan.stage} · LowerPipelineLoops 会额外补一次 tail dispatch`],
    ];
    const blocks = Array.from({ length: Math.min(plan.iters, 64) }, () => '<i></i>').join('');
    const bigIters = RMS_BIG_HIDDEN / plan.chunk;
    const wholeRowBytes = RMS_ROWS * RMS_BIG_HIDDEN * 4;
    const overflowX = (wholeRowBytes / RMS_UB_CAPACITY).toFixed(1);

    return `
      <section class="kf-inspector-section kf-rmsl-migrate"><header><h2 class="kf-inspector-title">切分循环由谁生成</h2><span>PTO 2.0 → 3.0</span></header>
        <div class="kf-rmsl-pair">
          <article class="is-old"><span>PTO 2.0</span><code>tile_shape = [${RMS_ROWS}, ${profile.chunk}]<br>框架自动展开 H 维循环</code></article>
          <article class="is-new"><span>PTO 3.0 · 本文件</span><code>for kb in pl.pipeline(<br>&nbsp;&nbsp;HIDDEN // ${profile.chunkConst}, stage=${profile.stage}):</code></article>
        </div>
        <div class="kf-rmsl-map">${migrateRows.map(([was, now, where, line]) => `<div><span>${was}</span>${line ? `<button type="button" data-rms-goto-line="${line}">${where}</button>` : `<em>${where}</em>`}<b>${now}</b></div>`).join('')}</div>
      </section>

      <section class="kf-inspector-section kf-rmsl-nest"><header><h2 class="kf-inspector-title">循环层级 · 源码实测</h2><span>2 层平铺 · 已最小</span></header>
        <ol class="kf-pto3-loop-tree kf-rmsl-tree">${nestRows.map(([name, range, why, line], index) => `<li><i>0${index + 1}</i><div><b>${name}</b><code>${range}</code><small>${why}</small></div><button type="button" data-rms-goto-line="${line}">L${line}</button></li>`).join('')}</ol>
        <div class="kf-pto3-reason"><b>还需要再加一层循环吗 —— 不需要</b><p>层级数由「H 维切分 + 两遍归约」决定：归约必须扫完整个 H 才能得到 inv_rms，所以是两个平铺循环而非嵌套；BATCH=${RMS_ROWS} 行整行常驻，无需再切 M 维。UB 峰值只由 chunk × stage 决定，<u>增加嵌套层级不会降低峰值</u>——溢出应当调参数，而不是重构结构。</p></div>
      </section>

      <section class="kf-inspector-section kf-rmsl-solver"><header><h2 class="kf-inspector-title">UB 峰值求解器</h2><span>192 KiB / AIV</span></header>
        <div class="kf-rmsl-axes">
          <div><span>chunk</span><div class="kf-rmsl-chips">${chunkChips}</div></div>
          <div><span>stage</span><div class="kf-rmsl-chips">${stageChips}</div></div>
        </div>
        <div class="kf-rmsl-gauge is-${plan.verdict}"><i style="width:${barPct}%"></i><b>${rmsKiB(plan.peak)} / 192 KiB</b><em>${Math.round(plan.ratio * 100)}% · ${verdictLabel}</em></div>
        <div class="kf-rmsl-budget">${budgetRows.map(([name, bytes, detail]) => `<div><span>${name}</span><code>${rmsKiB(bytes)}</code><b>${detail}</b></div>`).join('')}<div class="is-total"><span>循环内峰值</span><code>${rmsKiB(plan.peak)}</code><b>搬运 ${rmsKiB(plan.staged)} + 计算临时量 ${rmsKiB(plan.tempBytes + plan.reduceBytes)}</b></div></div>
        <div class="kf-rmsl-advice ${bestIsSource && isSourceValue ? 'is-good' : 'is-tune'}">${advice}</div>
        <div class="kf-rmsl-iters"><header><b>8192 = ${plan.iters} × ${plan.chunk}</b><span>${isSourceValue ? '与源码一致' : `源码为 ${profile.chunks} × ${profile.chunk}`}</span></header><div class="kf-rms-blocks">${blocks}</div><div class="kf-rmsl-guards">${guardRows.map(([ok, name, why]) => `<div class="${ok ? 'is-ok' : 'is-warn'}"><i>${ok ? '✓' : '!'}</i><b>${name}</b><small>${why}</small></div>`).join('')}</div><small>每块 = 1 次 pipeline 迭代 · 两遍扫描共 ${plan.iters * 2} 次</small></div>
      </section>

      <section class="kf-inspector-section kf-rmsl-skeleton"><header><h2 class="kf-inspector-title">循环骨架</h2><span>按当前求解结果生成</span></header>
        <pre class="kf-rmsl-code">${escapeHtml(rmsLoopSkeleton(profile, plan))}</pre>
        <button class="kf-rms-action" type="button" data-rms-action="skeleton">＋ 写入 ${profile.name} 循环骨架</button>
      </section>

      <div class="kf-inspector-card kf-rms-estimate kf-rmsl-bigh"><b>大 H 维推演 · HIDDEN 8192 → ${RMS_BIG_HIDDEN}</b>
        <div class="kf-rmsl-bigh-grid"><div><span>UB 峰值</span><b class="is-good">${rmsKiB(plan.peak)} → ${rmsKiB(plan.peak)}</b></div><div><span>迭代 / pass</span><b>${plan.iters} → ${bigIters}</b></div><div><span>循环层级</span><b class="is-good">2 → 2</b></div><div><span>整行载入</span><b class="is-bad">${rmsKiB(wholeRowBytes)} · ${overflowX}× UB</b></div></div>
        <p>H 变大只改变循环次数与常量，不改变循环层级；只有沿用 PTO 2.0「一次 load 整个 H」的心智才会在这里溢出。容量为静态估算，不含 Tile 对齐与后端临时分配，编译后需以 Pass IR 校准。</p>
      </div>`;
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
    const tabLabels = { overview: '概览', precision: '数据与精度', loops: '循环·UB', validation: '验证' };
    const content = state.rmsNormTab === 'precision' ? rmsNormPrecision(profile) : state.rmsNormTab === 'loops' ? rmsNormLoops(profile) : state.rmsNormTab === 'validation' ? rmsNormValidation() : rmsNormOverview(profile);
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

  function renderPto3TileLabInspector() {
    const isRms = state.pto3LabFocus === 'rmsnorm';
    const tab = state.pto3LabTab;
    const loopRows = isRms
      ? [['H outer', '0 → 32768 · step 4096', '控制大 H 维的分块边界'], ['H inner', 'h0 → min(h0 + 4096, H)', '单个归约 Tile，保证 UB 安全'], ['归约', 'sum(x²) → rsqrt → scale', '在 Tile 内完成，避免整行进入 UB']]
      : [['M loop', '0 → M · step 128', '输出行块并行维度'], ['N loop', '0 → N · step 128', '输出列块，约束单个结果 Tile'], ['K loop', '0 → K · step 128', '跨 K block 累加 FP32 accumulator']];
    const budget = isRms
      ? [['x_tile', '4096 × BF16', '8 KB'], ['square / acc', '4096 × FP32', '16 KB'], ['归一化临时量', '4096 × FP32', '16 KB'], ['预计 UB 使用', '—', '40 KB / 64 KB · SAFE']]
      : [['A tile', '128 × 128 × BF16', '32 KB'], ['B tile', '128 × 128 × BF16', '32 KB'], ['acc', '128 × 128 × FP32', '64 KB · L0C'], ['循环内 UB', 'A/B 双缓冲', '≤ 64 KB · SAFE']];
    const diff = isRms
      ? '<div class="kf-pto3-diff"><div class="remove">- x = pl.load(x, [0], [H], target_memory=pl.Mem.Vec)</div><div class="add">+ for h0 in pl.range(0, H, H_TILE):</div><div class="add">+     x_tile = pl.load(x, [h0], [h1 - h0], target_memory=pl.Mem.Vec)</div><div class="add">+     ss = pl.sum(x_tile * x_tile, axis=0)</div></div>'
      : '<div class="kf-pto3-diff"><div class="add">+ for m0 in pl.range(0, M, 128):</div><div class="add">+     for n0 in pl.range(0, N, 128):</div><div class="add">+         for k0 in pl.range(0, K, 128):</div><div class="add">+             acc += pl.matmul(a_tile, b_tile)</div></div>';
    const body = tab === 'budget' ? `<section class="kf-pto3-section"><header><h2>UB 容量预算</h2><span>${isRms ? 'RMSNorm · H = 32768' : 'Matmul · 128 × 128 Tile'}</span></header><div class="kf-pto3-budget">${budget.map(row => `<div><span>${row[0]}</span><b>${row[1]}</b><em>${row[2]}</em></div>`).join('')}</div><div class="kf-pto3-safe"><b>SAFE</b><span>${isRms ? 'H_TILE = 4096，尾块使用 min()，不会把完整 H 维装入 UB。' : 'A/B Tile 在循环内复用，K 循环只延长 accumulator 生命周期。'}</span></div></section>`
      : tab === 'refactor' ? `<section class="kf-pto3-section"><header><h2>结构化重构建议</h2><span>可逐段接受</span></header><p class="kf-pto3-explain">${isRms ? '把大 H 维拆成 outer loop + UB-sized inner Tile；归约、归一化和写回都留在 Tile 范围内。' : '把 M、N、K 三个切分维度显式展开；acc 只在 N block 内创建，并在 K loop 中复用。'}</p>${diff}<button type="button" class="kf-pto3-action" data-pto3-action="apply">应用推荐骨架</button></section>`
      : `<section class="kf-pto3-section"><header><h2>显式循环结构</h2><span>${isRms ? 'Vector / reduction' : 'Cube / accumulate'}</span></header><ol class="kf-pto3-loop-tree">${loopRows.map((row, index) => `<li><i>0${index + 1}</i><div><b>${row[0]}</b><code>${row[1]}</code><small>${row[2]}</small></div></li>`).join('')}</ol><div class="kf-pto3-reason"><b>为什么需要这些层级？</b><p>${isRms ? 'H 维超过 UB 可容纳范围，必须以安全 Tile 分段归约；最后一个 Tile 允许小于 H_TILE。' : '每层循环对应一个硬件可管理的 Tile 维度：M/N 决定输出块，K 决定跨块累加。'}</p></div></section>`;
    $('#inspectorTitle').textContent = 'PTO 3.0 · 循环与 Tile';
    $('#inspectorMeta').textContent = `${isRms ? 'RMSNorm 大 H' : 'Matmul M / N / K'} · static`; 
    $('#inspector').innerHTML = `<section class="kf-pto3-hero"><span class="kf-eyebrow">PTO 3.0 OPERATOR LAB</span><h1>${isRms ? 'RMSNorm · 大 H 维分块' : 'Matmul · 显式 M / N / K 切分'}</h1><p>选择源码中的循环或算子，查看它如何映射到 Tile、UB 和归约边界。</p><div class="kf-pto3-switch" role="tablist">${[['loops','循环结构'],['budget','UB 预算'],['refactor','重构建议']].map(([key,label]) => `<button type="button" class="${key === tab ? 'is-active' : ''}" data-pto3-tab="${key}">${label}</button>`).join('')}</div></section>${body}<footer class="kf-rms-provenance"><span><i class="fact"></i>源码结构</span><span><i class="estimated"></i>静态容量推断</span><span><i class="resolved"></i>编译后可继续校准</span></footer>`;
  }

  function renderDecodeGraphDetail(nodeId) {
    const detail = decodeLayerGraph.details[nodeId] || decodeLayerGraph.details['decode-input-hidden'];
    if (!detail) return '<b>尚未选择节点</b><p>点击计算图中的任务或 Tensor，查看 Scope、TaskId、资源和源码行。</p>';
    return `<b>${escapeHtml(detail.title)}</b><p><span class="kf-decode-graph-detail__phase">${escapeHtml(detail.phase)}</span> · ${escapeHtml(detail.kind)}${detail.line ? ` · 源码第 ${detail.line} 行` : ''}</p><dl>${detail.tasks ? `<div><dt>源码任务</dt><dd>${escapeHtml(detail.tasks)}</dd></div>` : ''}<div><dt>资源</dt><dd>${escapeHtml(detail.resource)}</dd></div><div><dt>依赖</dt><dd>${escapeHtml(detail.deps)}</dd></div><div><dt>TaskId alias</dt><dd>${escapeHtml(detail.alias || '—')}</dd></div></dl>`;
  }

  function renderDecodeLayerGraph() {
    const pattern = window.PtoModelGraphvizPattern;
    const stage = $('#decodeLayerComputationGraph');
    const status = $('#decodeGraphStatus');
    const detail = $('#decodeGraphDetail');
    if (!pattern || !stage) return;
    decodeLayerGraphController = pattern.renderController(stage, decodeLayerGraph.graph, {
      ariaLabel: 'decode_layer.py complete computation graph with scopes, dependencies and resources',
      colormap: pattern.modelArchitectureColormap(decodeLayerGraph.graph),
      fitMode: 'full', viewportPadding: 20, autoFit: true,
      interaction: { panZoom: true, selectableClusters: false },
      overlays: { edgeTags: true },
      onSelect: ({ nodeId }) => {
        state.intentGraphNode = nodeId;
        if (detail) detail.innerHTML = renderDecodeGraphDetail(nodeId);
        const selected = decodeLayerGraph.details[nodeId];
        if (status && selected) status.textContent = `${selected.title} · ${selected.phase}${selected.line ? ` · 源码第 ${selected.line} 行` : ''}`;
      },
      onHover: (nodeId) => {
        const hovered = decodeLayerGraph.details[nodeId];
        if (status && hovered) status.textContent = `${hovered.title} · ${hovered.resource}`;
      },
    });
    if (detail) detail.innerHTML = renderDecodeGraphDetail(state.intentGraphNode || 'decode-input-hidden');
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
    decodeLayerGraphController?.destroy?.();
    decodeLayerGraphController = null;
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
    if (state.activeFile === PTO3_TILE_LAB_FILE) {
      renderPto3TileLabInspector();
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
    const intentHero = '<section class="kf-intent-hero"><div class="kf-intent-title-row"><b>decode_layer</b><span class="kf-inference-badge">推理</span><span class="kf-megakernel-badge">megakernel</span></div></section>';
    $('#inspectorTitle').textContent = '意图预览';
    $('#inspectorMeta').textContent = 'decode_layer.py';
    if (state.intentTab === 'graph') {
      const summary = decodeLayerGraph.summary;
      $('#inspector').innerHTML = `
        ${intentHero}
        <div class="kf-intent-tabs" role="tablist" aria-label="算子意图类型">${tabs}</div>
        <section class="kf-decode-graph-summary" aria-label="计算图摘要"><div><b>${summary.named}</b><span>named tasks</span></div><div><b>${summary.explicit}</b><span>explicit deps</span></div><div><b>${summary.spmd}</b><span>SPMD grids</span></div><div><b>${summary.at}</b><span>pl.at declarations</span></div></section>
        <div class="kf-decode-graph-legend" aria-label="计算图关系图例"><span><i class="is-data"></i>数据流</span><span><i class="is-task"></i>TaskId / 状态 / 控制</span><span><i class="is-cluster"></i>Scope 边界</span></div>
        <section class="kf-inspector-section kf-decode-graph-section"><header><h2 class="kf-inspector-title">完整计算过程</h2><span>源码第 ${summary.manualLine || '—'} 行 manual_scope</span></header><div class="pto-model-graphviz-pattern-page pto-model-graphviz-stage kf-decode-graph-stage" id="decodeLayerComputationGraph" aria-label="decode_layer.py 完整计算图"></div><footer id="decodeGraphStatus">点击节点查看 Scope、显式依赖、资源和源码位置 · 拖拽 / 缩放查看全图</footer></section>
        <section class="kf-decode-graph-resources" aria-label="计算图资源摘要"><span>资源映射</span>${summary.resources.map((item) => `<b>${item}</b>`).join('')}</section>
        <div class="kf-inspector-card kf-decode-graph-detail" id="decodeGraphDetail"></div>
        <div class="kf-inspector-card kf-intent-note"><b>图的边界</b><p>${summary.compileBlocked ? '源码头部明确标注 dynamic INDEX / i64 store offset 当前会阻塞 codegen；图展示的是源码声明的目标结构。' : '图展示源码声明的目标结构。'} Tensor 自动依赖、WAR/WAW 和最终运行状态仍需编译后 IR / Runtime Trace 继续确认。</p></div>`;
      renderDecodeLayerGraph();
      return;
    }
    const shapeLayoutVisual = active.visual ? `
      <section class="kf-shape-layout-visual" aria-label="Shape 与 Layout 可视化摘要">
        <header class="kf-shape-layout-visual__head"><div><span class="kf-eyebrow">PARSED TENSOR CONTRACT</span><h2>形状与布局总览</h2></div><span class="kf-source-chip">源码驱动</span></header>
        <div class="kf-shape-layout-kpis">${active.visual.kpis.map(([label, value, detail]) => `<div><span>${escapeHtml(label)}</span><b>${escapeHtml(String(value))}</b><small>${escapeHtml(detail)}</small></div>`).join('')}</div>
        <div class="kf-shape-layout-flow" aria-label="张量形状主链">${active.visual.flow.map((item, index) => `${index ? '<i class="kf-shape-layout-flow__arrow" aria-hidden="true">↓</i>' : ''}<div class="kf-shape-layout-flow__node is-${item.tone}"><span>${escapeHtml(item.kicker)}</span><b>${escapeHtml(item.title)}</b><small>${escapeHtml(item.detail)}</small></div>`).join('')}</div>
        <div class="kf-shape-layout-layout"><header><b>Layout 约束</b><span>physical mapping</span></header><div>${active.visual.layout.map(([label, value, detail]) => `<article><span>${escapeHtml(label)}</span><b>${escapeHtml(value)}</b><small>${escapeHtml(detail)}</small></article>`).join('')}</div></div>
        <div class="kf-shape-layout-boundary"><i></i><span><b>精度边界</b> BF16 输入 / Cache / 激活边缘，FP32 层间 residual carry</span></div>
      </section>` : '';
    $('#inspector').innerHTML = `
      ${intentHero}
      <div class="kf-intent-tabs" role="tablist" aria-label="算子意图类型">${tabs}</div>
      ${shapeLayoutVisual}
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
      const activeModel = window.PtoModelArchitectureState?.active || 'qwen3';
      if (activeModel.startsWith('deepseek-v4-flash')) {
        window.PtoDeepSeekV4ModelViz?.show();
      } else {
        window.PtoQwen3ModelViz?.show();
      }
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

  function closeModelSelector() {
    const selector = $('[data-model-selector]');
    const trigger = $('[data-model-selector-trigger]');
    const menu = $('[data-model-selector-menu]');
    if (!selector || !trigger || !menu) return;
    menu.hidden = true;
    trigger.setAttribute('aria-expanded', 'false');
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
    const selector = event.target.closest('[data-model-selector]');
    if (!selector) closeModelSelector();
    const selectorTrigger = event.target.closest('[data-model-selector-trigger]');
    if (selectorTrigger) {
      const menu = selectorTrigger.closest('[data-model-selector]')?.querySelector('[data-model-selector-menu]');
      const open = menu && menu.hidden;
      if (menu) menu.hidden = !open;
      selectorTrigger.setAttribute('aria-expanded', String(Boolean(open)));
      return;
    }
    const recipe = event.target.closest('[data-recipe]');
    if (recipe) { state.selectedRecipe = recipe.dataset.recipe; renderRecipes(); toast(`已选择 ${$('b', recipe).textContent}`); }
    const step = event.target.closest('[data-step]');
    if (step) { setActivityView('workflow'); goTo(Number(step.dataset.step)); }
    if (event.target.closest('[data-open-runs]')) setActivityView('runs');
    if (event.target.closest('[data-back-workflow]')) setActivityView('workflow');
    const modelOption = event.target.closest('[data-model-id]');
    if (modelOption) {
      window.PtoModelArchitectureState = { active: modelOption.dataset.modelId };
      closeModelSelector();
      setActivityView('model');
      return;
    }
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
          state.pagedAttentionTab = 'graph';
          state.pagedAttentionDetailOpen = false;
          state.pagedAttentionOverlay = 'data';
          state.pagedAttentionNode = 'orch';
          state.pagedAttentionExpandedNode = null;
          state.pagedAttentionFocus = 'orchestration';
          state.pagedAttentionTask = 'qk';
          state.pagedAttentionDep = 'sij';
          state.pagedAttentionPipeKernel = 'qk';
          state.pagedAttentionLine = null;
        }
        if (filePath === PTO3_TILE_LAB_FILE) {
          state.pto3LabTab = 'loops';
          state.pto3LabFocus = 'matmul';
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
    const rmsPlanChunk = event.target.closest('[data-rms-plan-chunk]');
    const rmsPlanStage = event.target.closest('[data-rms-plan-stage]');
    if (rmsPlanChunk || rmsPlanStage) {
      const profile = rmsNormProfiles[state.rmsNormFunction] || rmsNormProfiles.input;
      const current = rmsCurrentPlan(profile);
      const chunk = rmsPlanChunk ? Number(rmsPlanChunk.dataset.rmsPlanChunk) : current.chunk;
      const stage = rmsPlanStage ? Number(rmsPlanStage.dataset.rmsPlanStage) : current.stage;
      state.rmsNormPlan[profile.id] = { chunk, stage };
      const next = rmsUbPlan(profile, chunk, stage);
      renderRmsNormInspector();
      toast(next.safe
        ? `chunk ${chunk} × stage ${stage} · UB 峰值 ${rmsKiB(next.peak)}（${Math.round(next.ratio * 100)}%）`
        : `chunk ${chunk} × stage ${stage} 会溢出 UB：峰值 ${rmsKiB(next.peak)} > 192 KiB`);
    }
    const rmsGoto = event.target.closest('[data-rms-goto-line]');
    if (rmsGoto && state.activeFile === RMSNORM_FILE) {
      const target = $(`#dslEditor [data-rms-line="${rmsGoto.dataset.rmsGotoLine}"]`);
      if (target) {
        target.scrollIntoView({ block: 'center', behavior: 'smooth' });
        $$('#dslEditor .is-rms-goto-line').forEach((row) => row.classList.remove('is-rms-goto-line'));
        target.classList.add('is-rms-goto-line');
      }
    }
    if (event.target.closest('[data-rms-action="skeleton"]')) {
      const profile = rmsNormProfiles[state.rmsNormFunction] || rmsNormProfiles.input;
      const plan = rmsCurrentPlan(profile);
      toast(`已生成 ${profile.name} 循环骨架草案：chunk ${plan.chunk} · stage ${plan.stage} · ${plan.iters} 次 / pass`);
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
    const pagedAttentionDetailBack = event.target.closest('[data-pa-detail-back]');
    if (pagedAttentionDetailBack) {
      state.pagedAttentionDetailOpen = false;
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
    const pagedAttentionTaskPick = event.target.closest('[data-pa2-task]');
    if (pagedAttentionTaskPick) {
      state.pagedAttentionTask = pagedAttentionTaskPick.dataset.pa2Task;
      const task = pagedAttentionTasks[state.pagedAttentionTask];
      if (task) syncPagedAttentionSelection(task.focus);
      renderPagedAttentionInspector({ scrollToFocus: true });
    }
    const pagedAttentionDepPick = event.target.closest('[data-pa2-dep]');
    if (pagedAttentionDepPick) {
      state.pagedAttentionDep = pagedAttentionDepPick.dataset.pa2Dep;
      const dep = pagedAttentionDeps[state.pagedAttentionDep];
      if (dep) {
        state.pagedAttentionFocus = dep.focus;
        const task = pagedAttentionFocusToTask[dep.focus];
        if (task) state.pagedAttentionTask = task;
      }
      renderPagedAttentionInspector({ scrollToFocus: true });
    }
    const pagedAttentionPipe = event.target.closest('[data-pa2-pipe]');
    if (pagedAttentionPipe) {
      state.pagedAttentionPipeKernel = pagedAttentionPipe.dataset.pa2Pipe;
      syncPagedAttentionSelection(state.pagedAttentionPipeKernel);
      renderPagedAttentionInspector({ scrollToFocus: true });
    }
    const pagedAttentionFocus = event.target.closest('[data-paged-attention-focus]');
    if (pagedAttentionFocus && !pagedAttentionFocus.closest('#dslEditor')) {
      const line = Number(pagedAttentionFocus.dataset.pa2Line);
      const nodeId = pagedAttentionFocus.dataset.pa2Node;
      syncPagedAttentionSelection(pagedAttentionFocus.dataset.pagedAttentionFocus);
      // 地图节点是精确选择，覆盖 focus 推出的代表节点
      if (nodeId) state.pagedAttentionNode = nodeId;
      if (line) state.pagedAttentionLine = line;
      state.pagedAttentionDetailOpen = true;
      renderPagedAttentionInspector({ scrollToFocus: !line });
      if (line) revealPagedAttentionLine(line);
    }
    const pagedAttentionLine = event.target.closest('#dslEditor [data-paged-attention-line]');
    if (pagedAttentionLine && isPagedAttentionFile(state.activeFile)) {
      syncPagedAttentionSelection(pagedAttentionLine.dataset.pagedAttentionFocus);
      state.pagedAttentionLine = Number(pagedAttentionLine.dataset.pagedAttentionLine);
      state.pagedAttentionDetailOpen = true;
      if (state.pagedAttentionTab !== 'graph') state.pagedAttentionTab = 'graph';
      markPagedAttentionTargetLine(state.pagedAttentionLine);
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
    const pto3Tab = event.target.closest('[data-pto3-tab]');
    if (pto3Tab && state.activeFile === PTO3_TILE_LAB_FILE) {
      state.pto3LabTab = pto3Tab.dataset.pto3Tab;
      renderPto3TileLabInspector();
    }
    const pto3Line = event.target.closest('#dslEditor [data-pto3-lab-line]');
    if (pto3Line && state.activeFile === PTO3_TILE_LAB_FILE) {
      state.pto3LabFocus = pto3Line.dataset.pto3LabFocus;
      state.pto3LabTab = 'loops';
      $$('#dslEditor [data-pto3-lab-focus]').forEach(row => row.classList.toggle('is-pto3-lab-active', row === pto3Line));
      renderPto3TileLabInspector();
    }
    if (event.target.closest('[data-pto3-action="apply"]')) toast('已生成 PTO 3.0 推荐循环骨架，可在源码中逐段接受');
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

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') closeModelSelector();
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
  $('#copyFingerprint')?.addEventListener('click', () => { navigator.clipboard?.writeText('env:8da1bf09'); toast('配置快照 ID 已复制'); });
  $('#recheckEnvironment')?.addEventListener('click', () => {
    const button = $('#recheckEnvironment');
    const control = $('#envControl');
    const stateChip = $('#envStateChip');
    const progress = $('.kf-env-progress span');
    const evidence = $('.kf-env-evidence small');
    const checkedAt = $('#envCheckedAt');
    const gates = ['target', 'imports', 'pins', 'smoke'].map(key => document.querySelector(`[data-env-gate="${key}"]`));
    const mockResults = {
      target: { className: 'is-ready', title: '目标已配置', detail: 'Ascend 950B · Simulator · 无需设备分配', label: '已验证' },
      imports: { className: 'is-ready', title: '框架与 harness 可导入', detail: 'PyPTO 3.0.0-dev · torch · golden harness', label: '已验证' },
      pins: { className: 'is-ready', title: '工具链 pin 已对齐', detail: 'PTOAS 0.8.4 · CANN 9.0.RC1 · Tile-ISA 8f31c2a', label: '已验证' },
      smoke: { className: 'is-ready', title: 'Smoke run 已验证', detail: 'compile · input · golden · runtime · validation 全部通过', label: '已通过' },
    };
    const updateGate = (gate, key, result) => {
      const copy = gate?.querySelector('div');
      if (!gate || !copy) return;
      gate.className = result.className;
      copy.querySelector('b').textContent = result.title;
      copy.querySelector('small').textContent = result.detail;
      gate.querySelector('em').textContent = result.label;
      gate.dataset.envState = key;
    };
    button.disabled = true;
    button.textContent = '检查中…';
    if (control) control.querySelector('small').textContent = '正在读取…';
    if (stateChip) { stateChip.className = 'kf-state-chip neutral'; stateChip.textContent = '检查中…'; }
    if (evidence) evidence.textContent = '正在读取 Python、framework、assembler 和 smoke run 证据…';
    gates.forEach(gate => gate?.classList.add('is-checking'));
    toast('正在模拟 setup-and-run：读取环境并逐项验证 gate');
    gates.forEach((gate, index) => setTimeout(() => {
      const key = gate?.dataset.envGate;
      if (!key) return;
      const result = mockResults[key];
      gate.classList.remove('is-checking');
      updateGate(gate, key, result);
      const passed = index + 1 <= 2;
      if (progress) progress.style.width = `${Math.round(((index + 1) / gates.length) * 50 + (passed ? 0 : 0))}%`;
      if (index === gates.length - 1) {
        const now = new Date();
        const time = now.toLocaleTimeString('zh-CN', { hour12: false });
        const passedCount = 4;
        button.disabled = false;
        button.textContent = '重新检查';
        if (control) { control.querySelector('i').classList.remove('is-warn'); control.querySelector('span:nth-of-type(2)').textContent = `${passedCount} / 4 已通过`; }
        if (stateChip) { stateChip.className = 'kf-state-chip good'; stateChip.textContent = '4 / 4 已通过'; }
        if (progress) { progress.style.width = '100%'; progress.style.background = 'var(--success)'; }
        if (checkedAt) checkedAt.textContent = `最近检查 ${time}`;
        if (evidence) {
          const evidenceCard = evidence.closest('.kf-env-evidence');
          evidence.textContent = 'Mock 结果：版本 pin 一致，且 smoke run 的五个阶段均通过。';
          evidenceCard?.classList.add('is-healthy');
          const evidenceIcon = evidenceCard?.querySelector('.kf-env-evidence-icon');
          const evidenceTitle = evidenceCard?.querySelector('b');
          if (evidenceIcon) evidenceIcon.textContent = '✓';
          if (evidenceTitle) evidenceTitle.textContent = 'Validated run';
        }
        toast('检查完成：4 项 gate 全部通过，可进入模型阶梯');
      }
    }, 450 * (index + 1)));
  });
  $('#openRunAdmission')?.addEventListener('click', () => {
    setEnvironmentPanel(false);
    setActivityView('workflow');
    goTo(0);
    toast('已打开工作流');
  });
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
