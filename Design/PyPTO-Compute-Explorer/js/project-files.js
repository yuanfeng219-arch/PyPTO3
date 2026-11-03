// Project file registry for the PyPTO Compute Explorer demo.
// Owns: file identity (id / name / sourcePath / scene), the Project Explorer tree,
// and a minimal static import → symbol index used for cross-file navigation.
// Content always comes from the existing local source snapshots; nothing is fetched at runtime.
(function () {
  'use strict';

  const CSA_DIR = 'Data/DeepseekV4/deepseek_v4_flash_dspark';
  const CSA_SOURCE = window.PTO_CSA_SOURCE || {};
  const CSA_DEPS = CSA_SOURCE.dependencies || {};

  // Order and role labels follow the real import order in decode_csa.py.
  const CSA_ORDER = [
    ['decode_csa.py', '主入口'],
    ['config.py', '配置'],
    ['hc_pre.py', '计算模块'],
    ['rmsnorm.py', '计算模块'],
    ['qkv_proj_rope.py', '计算模块'],
    ['decode_cp_token_allgather.py', '计算/通信模块'],
    ['decode_compressor_ratio4.py', '计算模块'],
    ['decode_indexer_compressor.py', '计算模块'],
    ['decode_indexer.py', '计算模块'],
    ['decode_sparse_attn_csa.py', '计算模块'],
    ['decode_o_proj.py', '计算/通信模块'],
    ['hc_post.py', '计算模块'],
    ['utils.py', '辅助模块'],
  ];

  const escape = (value) => String(value).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));

  // --- content resolution (existing snapshots only) -------------------------

  function csaContent(name) {
    if (name === 'decode_csa.py') return typeof CSA_SOURCE.text === 'string' ? CSA_SOURCE.text : '';
    const dependency = CSA_DEPS[name];
    if (typeof dependency === 'string') return dependency;
    // utils.py is not part of PTO_CSA_SOURCE.dependencies; it has its own snapshot file.
    if (name === 'utils.py' && typeof window.PTO_CSA_UTILS_SOURCE === 'string') return window.PTO_CSA_UTILS_SOURCE;
    return '';
  }

  function csaOrigin(name) {
    if (name === 'decode_csa.py') return 'PTO_CSA_SOURCE.text';
    if (typeof CSA_DEPS[name] === 'string') return 'PTO_CSA_SOURCE.dependencies';
    return 'PTO_CSA_UTILS_SOURCE';
  }

  // --- file registry -------------------------------------------------------

  const files = [];
  const byName = new Map();
  const byId = new Map();

  function add(entry) {
    if (!entry.content) {
      console.warn('[project-files] missing source snapshot for', entry.name);
      return;
    }
    entry.lineCount = entry.content.split('\n').length;
    entry.regions = deriveRegions(entry.content);
    files.push(entry);
    byName.set(entry.name, entry);
    byId.set(entry.id, entry);
  }

  const pagedSource = window.PTO_PAGED_ATTENTION_PYPTO_SOURCE || '';
  add({
    id: 'paged-attention/paged_attention_pypto.py',
    name: 'paged_attention_pypto.py',
    scene: 'paged-attention',
    sourcePath: 'paged_attention_pypto.py · 本地快照',
    origin: 'PTO_PAGED_ATTENTION_PYPTO_SOURCE',
    role: '主入口',
    content: pagedSource,
  });

  CSA_ORDER.forEach(([name, role]) => {
    add({
      id: 'csa/' + name,
      name,
      scene: 'csa',
      sourcePath: CSA_DIR + '/' + name,
      origin: csaOrigin(name),
      role,
      content: csaContent(name),
    });
  });

  const groups = [
    { key: 'paged', label: 'Paged Attention', files: files.filter((f) => f.scene === 'paged-attention') },
    { key: 'csa', label: 'DeepSeek V4 / Decode CSA', files: files.filter((f) => f.scene === 'csa') },
  ];

  // --- static import scan --------------------------------------------------

  // Only `from x import (a, b as c)` / `from x import a, b` / `import x` at column 0.
  // Not a Python parser: it exists to locate symbols in known local modules.
  function scanImports(content) {
    const lines = content.split('\n');
    const out = [];
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const from = /^from\s+([A-Za-z_][\w.]*)\s+import\s+(.*)$/.exec(line);
      if (from) {
        const module = from[1];
        if (module.startsWith('.')) continue;
        const rest = from[2].trim();
        let body = rest;
        let end = i;
        if (rest.startsWith('(')) {
          const chunks = [rest.slice(1)];
          while (!/\)/.test(chunks[chunks.length - 1]) && end + 1 < lines.length) {
            end += 1;
            chunks.push(lines[end]);
          }
          body = chunks.join(' ').replace(/[()]/g, ' ');
        }
        const names = body
          .split('#')[0]
          .split(',')
          .map((piece) => piece.trim())
          .filter(Boolean)
          .map((piece) => piece.split(/\s+as\s+/))
          .filter((parts) => /^[A-Za-z_]\w*$/.test(parts[0] || ''))
          .map((parts) => ({ symbol: parts[0], name: (parts[1] || parts[0]).trim() }));
        out.push({ module, names, start: i + 1, end: end + 1 });
        i = end;
        continue;
      }
      const plain = /^import\s+([A-Za-z_][\w.]*)(?:\s+as\s+([A-Za-z_]\w*))?\s*$/.exec(line);
      if (plain && !plain[1].includes('.')) {
        out.push({ module: plain[1], names: [], start: i + 1, end: i + 1, alias: plain[2] || plain[1] });
      }
    }
    return out;
  }

  // module-level definition index: first `def` / `class` / `NAME =` at column 0.
  function indexDefs(content) {
    const map = new Map();
    content.split('\n').forEach((line, i) => {
      let match = /^(?:async\s+)?(def|class)\s+([A-Za-z_]\w*)/.exec(line);
      if (match) {
        if (!map.has(match[2])) map.set(match[2], { line: i + 1, kind: match[1] });
        return;
      }
      match = /^([A-Za-z_]\w*)\s*(?:=|:|\()/.exec(line);
      if (match && !map.has(match[1])) map.set(match[1], { line: i + 1, kind: 'assign' });
    });
    return map;
  }

  const defIndex = new Map(files.map((file) => [file.id, indexDefs(file.content)]));
  const linkCache = new Map();

  files.forEach((file) => {
    const symbols = {};
    const modules = {};
    const importLines = [];
    scanImports(file.content).forEach((statement) => {
      importLines.push([statement.start, statement.end]);
      const target = byName.get(statement.module + '.py');
      if (!target) return;
      modules[statement.module] = { file: target.id, line: 1, kind: 'file' };
      if (statement.alias) modules[statement.alias] = { file: target.id, line: 1, kind: 'file' };
      const targetDefs = defIndex.get(target.id) || new Map();
      statement.names.forEach(({ symbol, name }) => {
        const hit = targetDefs.get(symbol);
        if (!hit) return;
        const link = { file: target.id, line: hit.line, kind: hit.kind, symbol };
        symbols[symbol] = link;
        if (name !== symbol) symbols[name] = link;
      });
    });
    linkCache.set(file.id, { symbols, modules, importLines });
  });

  // --- source regions (auto, from real top-level definitions) ---------------

  function deriveRegions(content) {
    const lines = content.split('\n');
    const defs = [];
    lines.forEach((line, i) => {
      const match = /^(?:async\s+)?(def|class)\s+([A-Za-z_]\w*)/.exec(line);
      if (match) defs.push({ kind: match[1], name: match[2], line: i + 1 });
    });
    const regions = {};
    defs.forEach((def, index) => {
      const end = index + 1 < defs.length ? defs[index + 1].line - 1 : lines.length;
      regions[def.kind + '-' + def.line] = {
        label: (def.kind === 'class' ? 'class ' : 'def ') + def.name,
        lines: def.line + '–' + end,
        detail: '模块级' + (def.kind === 'class' ? '类' : '函数') + ' · 共 ' + (end - def.line + 1) + ' 行',
      };
    });
    return regions;
  }

  // --- file tree -----------------------------------------------------------

  function fileRow(file) {
    const title = file.sourcePath + ' · ' + file.lineCount + ' 行 · ' + file.role;
    return '<div class="kf-tree-row depth-2"><button type="button" data-file="' + escape(file.name) +
      '" data-file-id="' + escape(file.id) + '" title="' + escape(title) + '">' +
      '<span class="kf-file-icon py">Py</span><span>' + escape(file.name) + '</span></button></div>';
  }

  function renderTree(host) {
    const root = host || document.getElementById('projectTree');
    if (!root) return;
    const groupHtml = groups.map((group) => {
      const rows = group.files.map(fileRow).join('');
      return '<div class="kf-tree-row depth-1"><button type="button" data-tree-toggle="' + group.key +
        '" aria-expanded="true" aria-controls="pf-tree-' + group.key + '">' +
        '<span class="kf-caret">›</span><span class="kf-folder-icon"></span>' +
        '<span>' + escape(group.label) + '</span><em>' + group.files.length + '</em></button></div>' +
        '<div class="kf-tree-group" data-tree-group="' + group.key + '" id="pf-tree-' + group.key + '">' + rows + '</div>';
    }).join('');
    root.innerHTML = '<div class="kf-tree-row is-root"><button type="button" data-tree-toggle="project"' +
      ' aria-expanded="true" aria-controls="pf-tree-project"><span class="kf-caret">›</span>' +
      '<span class="kf-folder-icon"></span><b>PyPTO Compute Explorer</b></button></div>' +
      '<div class="kf-tree-group" data-tree-group="project" id="pf-tree-project">' + groupHtml + '</div>';
  }

  // --- public surface ------------------------------------------------------

  const api = {
    files,
    groups,
    byName,
    byId,
    get(key) {
      if (!key) return null;
      return byName.get(key) || byId.get(key) || null;
    },
    linksFor(id) {
      return linkCache.get(id) || { symbols: {}, modules: {}, importLines: [] };
    },
    deriveRegions,
    renderTree,
    stats() {
      return files.map((file) => ({
        id: file.id,
        name: file.name,
        scene: file.scene,
        origin: file.origin,
        lines: file.lineCount,
        symbols: Object.keys((linkCache.get(file.id) || {}).symbols || {}).length,
      }));
    },
  };

  window.PtoProjectFiles = Object.freeze(api);

  // The tree host exists before this script runs (scripts sit at the end of <body>).
  if (document.getElementById('projectTree')) renderTree();
})();
