(function initCsaMappingExplorer() {
  'use strict';

  const registry = window.PtoCsaMappingRegistry;
  const csaImplementation = registry.implementationEntries.l1_csa;
  const params = new URLSearchParams(window.location.search);
  let theme = params.get('theme') === 'light' ? 'light' : 'dark';
  let currentLevel = 'l3';
  let modelLevel = params.get('modelLevel') || (params.get('level') === 'l2' ? 'l2' : 'l1');
  if (modelLevel === 'root') modelLevel = 'l0'; // Legacy links restore the useful overview.
  let modelScope = 'model';
  let implementationOpened = params.get('implementation') === 'csa';
  let modelReady = false;
  const initialRootFolded = modelLevel === 'l0';
  let currentSourceTab = 'call';
  let selectedNodeId = 'l1_csa';
  let mappingDiffEnabled = false;
  const diffCategories = new Set(['expanded', 'state', 'deploy']);
  let semanticAutoExpand = false;
  let sourceRequest = 0;
  let sourceRefIndex = null;
  let focusedIds = [];
  let incrementsEnabled = false;
  const sourceCache = new Map();

  const semanticFrame = document.querySelector('[data-csa-semantic-frame]');
  const implementationEmpty = document.querySelector('[data-csa-implementation-empty]');
  const modelTitle = document.querySelector('[data-csa-model-title]');
  const modelDropdown = document.querySelector('[data-csa-model-dropdown]');
  const incrementsToggle = document.querySelector('[data-csa-increments]');
  const evidenceMenu = document.querySelector('[data-csa-evidence-menu]');
  const operatorFrame = document.querySelector('[data-csa-operator-frame]');
  const l4Surface = document.querySelector('[data-csa-l4]');
  const l4Flow = document.querySelector('[data-csa-l4-flow]');
  const artifactView = document.querySelector('[data-csa-artifact-view]');
  let sourceSurface = document.querySelector('[data-csa-source]');
  const sourceBody = sourceSurface.parentElement;
  const fileTabs = new Map();
  let activeFileKey = null;
  let sourceEntityId = 'l1_csa';
  const sourceLoading = document.querySelector('[data-csa-source-loading]');
  const sourceMeta = document.querySelector('[data-csa-source-meta]');
  const sourceTitle = document.querySelector('[data-csa-source-title]');
  const levelTitle = document.querySelector('[data-csa-level-title]');
  const levelDropdown = document.querySelector('[data-csa-level-dropdown]');
  const autoExpandToggle = document.querySelector('[data-csa-auto-expand]');
  const diffToggle = document.querySelector('[data-csa-diff-toggle]');
  const diffFilterToggle = document.querySelector('[data-csa-diff-filter]');
  const diffLegend = document.querySelector('[data-csa-diff-legend]');
  const mappingCard = document.querySelector('[data-csa-mapping-card]');
  const mappingCardCode = document.querySelector('[data-csa-mapping-code]');
  const mappingCardLabel = document.querySelector('[data-csa-mapping-label]');
  const mappingCardReason = document.querySelector('[data-csa-mapping-reason]');
  const legendToggle = document.querySelector('[data-csa-legend-toggle]');
  const legendPanel = document.querySelector('[data-csa-legend-panel]');
  const themeToggle = document.querySelector('[data-csa-theme-toggle]');
  const themeLabel = document.querySelector('[data-csa-theme-label]');

  const PYTHON_KEYWORDS = new Set(['and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'False', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'None', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'True', 'try', 'while', 'with', 'yield']);
  const PYTHON_BUILTINS = new Set(['bool', 'dict', 'enumerate', 'float', 'int', 'len', 'list', 'map', 'max', 'min', 'range', 'set', 'str', 'sum', 'tuple', 'type', 'ValueError', 'zip']);

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
  }

  function highlightPythonLine(line) {
    const commentAt = line.indexOf('#');
    const code = commentAt === -1 ? line : line.slice(0, commentAt);
    const comment = commentAt === -1 ? '' : line.slice(commentAt);
    const highlighted = escapeHtml(code).replace(/\b[A-Za-z_]\w*|\b\d+(?:\.\d+)?\b/g, token => {
      if (PYTHON_KEYWORDS.has(token)) return `<span class="tk-keyword">${token}</span>`;
      if (PYTHON_BUILTINS.has(token) || /^[A-Z][A-Z0-9_]*$/.test(token)) return `<span class="tk-type">${token}</span>`;
      if (/^\d/.test(token)) return `<span class="tk-number">${token}</span>`;
      return token;
    });
    return `${highlighted}${comment ? `<span class="tk-comment">${escapeHtml(comment)}</span>` : ''}` || '&nbsp;';
  }

  function codeRow(lineNumber, html, nodeIds = [], className = '') {
    const row = document.createElement('div');
    row.className = `pto-ide-frame__code-line${className ? ` ${className}` : ''}`;
    row.dataset.line = String(lineNumber || '');
    if (nodeIds.length) row.dataset.csaNodeIds = nodeIds.join(',');
    row.innerHTML = `<span class="pto-ide-frame__code-line-number">${lineNumber || ''}</span><span class="pto-ide-frame__code-line-code">${html}</span>`;
    return row;
  }

  function fileTabFor(item, reference) {
    const origin = item.origin === 'official-source' ? '官方' : 'PyPTO';
    const key = `${origin}:${reference.file}:${reference.hash || 'unavailable'}`;
    if (!fileTabs.has(key)) {
      const surface = fileTabs.size ? sourceSurface.cloneNode(false) : sourceSurface;
      surface.hidden = true;
      sourceBody.appendChild(surface);
      const button = document.createElement('button');
      button.type = 'button'; button.setAttribute('role', 'tab');
      button.textContent = `${origin} · ${reference.file.split('/').pop()}`;
      button.title = `${reference.file}\n${reference.hash || '版本未验证'}`;
      button.dataset.sourceFile = reference.file;
      const tab = { key, item, reference, surface, button, loaded: false, scrollTop: 0, scrollLeft: 0 };
      fileTabs.set(key, tab);
      document.querySelector('[data-csa-file-tabs]').appendChild(button);
      button.addEventListener('click', () => activateFileTab(tab));
    }
    return fileTabs.get(key);
  }

  function activateFileTab(tab) {
    ++sourceRequest; // Invalidate a late fetch from the previously active file.
    const previous = fileTabs.get(activeFileKey);
    if (previous) {
      previous.scrollTop = previous.surface.scrollTop;
      previous.scrollLeft = previous.surface.scrollLeft;
      const selection = window.getSelection();
      if (selection.rangeCount && previous.surface.contains(selection.anchorNode)) previous.selection = selection.getRangeAt(0).cloneRange();
    }
    activeFileKey = tab.key;
    sourceEntityId = tab.item.id;
    sourceSurface = tab.surface;
    fileTabs.forEach(item => {
      const active = item === tab;
      item.surface.hidden = !active || !item.loaded;
      item.button.classList.toggle('is-active', active);
      item.button.setAttribute('aria-selected', String(active));
      item.button.tabIndex = active ? 0 : -1;
    });
    sourceLoading.hidden = tab.loaded;
    if (!tab.loaded) return showSource(tab.item.id, { reference: tab.reference, updateCard: false });
    sourceTitle.title = tab.reference.file;
    sourceMeta.textContent = `L${tab.reference.start}–${tab.reference.end}`;
    sourceMeta.dataset.snapshotStatus = tab.snapshotStatus;
    sourceMeta.title = tab.metaTitle;
    sourceSurface.scrollTop = tab.scrollTop;
    sourceSurface.scrollLeft = tab.scrollLeft;
    if (tab.selection) {
      const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(tab.selection);
    }
    renderEvidenceMenu(tab.item);
  }

  function renderSource(source, reference, item) {
    const fragment = document.createDocumentFragment();
    const sourceLines = source.split('\n');
    const annotations = new Map();
    const groups = [[]];
    registry.annotationsFor(item.id).forEach(line => {
      if (line.startsWith('# [Explorer ·')) groups.push([]);
      groups.at(-1).push(line);
    });
    const relations = registry.relationsFor(item.id);
    groups.forEach((lines, index) => {
      const relation = relations[index - 1];
      const relatedNodes = relation
        ? [...relation.sourceIds, ...relation.targetIds].map(id => registry.getNode(id)).filter(Boolean)
        : [item];
      const locations = relatedNodes.flatMap(node => node.sourceRefs
        .filter(ref => ref.file === reference.file && ref.start <= reference.end && ref.end >= reference.start)
        .map(ref => ({ node, ref })));
      const anchor = locations.length ? Math.min(reference.end, Math.max(...locations.map(({ ref }) => ref.end))) : reference.end;
      const indent = sourceLines[anchor - 1]?.match(/^\s*/)?.[0] || '';
      const explanation = relation ? `# ${relation.explanation.change}` : lines.find(line => !line.startsWith('# [Explorer')) || '# 当前范围暂无独立映射说明。';
      const annotation = document.createElement('details');
      annotation.dataset.explorerAnnotation = 'true';
      annotation.className = 'csa-workbench__annotation-group is-related';
      const summary = document.createElement('summary');
      summary.appendChild(codeRow('', highlightPythonLine(indent + explanation), [], 'csa-workbench__annotation csa-workbench__annotation-heading'));
      annotation.appendChild(summary);
      lines.filter(line => !line.startsWith('# [Explorer') && line !== explanation && line !== `# 变化：${relation?.explanation.change}`)
        .forEach(line => annotation.appendChild(codeRow('', highlightPythonLine(indent + line), [], 'csa-workbench__annotation')));
      if (!annotations.has(anchor)) annotations.set(anchor, []);
      annotations.get(anchor).push(annotation);
    });
    sourceLines.forEach((line, index) => {
      const lineNumber = index + 1;
      const active = lineNumber >= reference.start && lineNumber <= reference.end;
      const ids = registry.nodes.filter(node => node.sourceRefs.some(ref => ref.file === reference.file && lineNumber >= ref.start && lineNumber <= ref.end)).map(node => node.id);
      fragment.appendChild(codeRow(lineNumber, highlightPythonLine(line), ids, active ? 'is-related' : ''));
      annotations.get(lineNumber)?.forEach(annotation => fragment.appendChild(annotation));
    });
    sourceSurface.replaceChildren(fragment);
    sourceSurface.hidden = false;
    sourceLoading.hidden = true;
    const surface = sourceSurface;
    requestAnimationFrame(() => {
      if (!surface.hidden) surface.querySelector(`[data-line="${reference.start}"]`)?.scrollIntoView({ block: 'center' });
    });
  }

  function sourceNodeFor(id) {
    return registry.getNode(id);
  }

  function sourceReferenceFor(id, tab) {
    const item = sourceNodeFor(id) || registry.getNode('l1_csa');
    const refs = item.sourceRefs || [];
    if (sourceRefIndex !== null) return refs[sourceRefIndex] || refs[0];
    if (tab === 'call') return refs.find(ref => ref.role === 'call') || refs[0];
    return refs.find(ref => ref.role === 'definition') || refs.at(-1) || refs[0];
  }

  async function showSource(id, { updateCard = true, reference: explicitReference } = {}) {
    const item = sourceNodeFor(id) || registry.getNode('l1_csa');
    const reference = explicitReference || sourceReferenceFor(id, currentSourceTab);
    if (!reference) return;
    const tab = fileTabFor(item, reference);
    // Activation and navigation differ: file switching restores, graph selection relocates.
    if (activeFileKey !== tab.key) {
      const old = fileTabs.get(activeFileKey);
      if (old) { old.scrollTop = old.surface.scrollTop; old.scrollLeft = old.surface.scrollLeft; }
    }
    activeFileKey = tab.key;
    sourceEntityId = id;
    sourceSurface = tab.surface;
    tab.item = item; tab.reference = reference; tab.selection = null; tab.loaded = false;
    fileTabs.forEach(entry => {
      entry.surface.hidden = true;
      entry.button.classList.toggle('is-active', entry === tab);
      entry.button.setAttribute('aria-selected', String(entry === tab));
      entry.button.tabIndex = entry === tab ? 0 : -1;
    });
    const request = ++sourceRequest;
    sourceLoading.hidden = false;
    sourceSurface.hidden = true;
    sourceLoading.querySelector('span').textContent = '正在读取证据源码…';
    try {
      let source = sourceCache.get(reference.file);
      if (!source) {
        const response = await fetch(reference.file, { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        source = await response.text();
        sourceCache.set(reference.file, source);
      }
      if (request !== sourceRequest) return;
      const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(source));
      const actualHash = [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, '0')).join('');
      if (request !== sourceRequest) return;
      const matched = reference.hash && reference.hash === actualHash;
      renderSource(source, reference, item);
      const filename = reference.file.split('/').pop();
      sourceMeta.textContent = `L${reference.start}–${reference.end}`;
      sourceMeta.dataset.snapshotStatus = matched ? 'verified' : 'unavailable';
      sourceMeta.title = `${reference.symbol} · SHA-256 ${actualHash}\n源码定位不等于数值等价或编译血缘已验证`;
      tab.loaded = true;
      tab.snapshotStatus = matched ? 'verified' : 'unavailable';
      tab.metaTitle = sourceMeta.title;
      sourceTitle.title = `${reference.file} · L${reference.start}–${reference.end} · ${matched ? '快照匹配' : '版本未验证'}`;
      const relations = registry.relationsFor(id);
      if (updateCard && relations.length) {
        mappingCard.hidden = false;
        mappingCardCode.dataset.code = item.mappingType?.toLowerCase() || '';
        const types = [...new Set(relations.flatMap(entry => entry.transformTypes))];
        const primary = types.includes('GROUPED') ? 'GROUPED' : types[0];
        mappingCardCode.textContent = primary + (types.length > 1 ? ` +${types.length - 1}` : '');
        mappingCardCode.title = types.join(' / ');
        mappingCardLabel.textContent = item.label;
        const diff = registry.diffFor(item.id);
        mappingCardReason.textContent = `${item.level === 'L3' ? diff.reason + ' L2 来源：' + (diff.sourceIds.map(id => registry.getNode(id).label).join('、') || 'CSA scope / 模型上下文') + '。 ' : ''}${relations.length} 条映射 · ${focusedIds.length} 个关联实体。 ${relations[0].explanation.change} 全部标签、变换原因与限制见证据菜单 / Explorer 注释；数值等价尚未验证。`;
      } else {
        mappingCard.hidden = true;
      }
      renderEvidenceMenu(item);
    } catch (error) {
      if (request !== sourceRequest) return;
      sourceLoading.querySelector('span').textContent = `${reference.file.split('/').pop()} 加载失败 · ${error.message}`;
      sourceMeta.textContent = 'unavailable';
    }
  }

  function syncThemeToggle() {
    const isLight = theme === 'light';
    const target = isLight ? '深色模式' : '浅色模式';
    themeToggle.setAttribute('aria-pressed', String(isLight));
    themeToggle.setAttribute('aria-label', `切换为${target}`);
    themeLabel.textContent = target;
  }

  function setTheme(nextTheme, notifyFrame = true) {
    theme = nextTheme === 'light' ? 'light' : 'dark';
    document.documentElement.dataset.theme = theme;
    syncThemeToggle();
    const url = new URL(window.location.href);
    url.searchParams.set('theme', theme);
    history.replaceState(null, '', url);
    if (notifyFrame) [operatorFrame, semanticFrame].forEach(frame => frame.contentWindow?.postMessage({ type: 'pto-preview-theme', theme }, '*'));
  }

  function renderL4() {
    l4Flow.hidden = true;
    l4Flow.style.display = 'none';
    document.querySelector('[aria-label="L4 编译证据"]').style.display = 'none';
    l4Flow.replaceChildren();
    artifactView.replaceChildren();
    const message = document.createElement('p');
    message.className = 'csa-workbench__evidence-note';
    message.textContent = '暂无已验证的 Kernel 映射，请先编译并导入 Pass Dump';
    artifactView.appendChild(message);
  }

  function renderEvidenceMenu(item) {
    evidenceMenu.replaceChildren();
    const status = document.createElement('span');
    status.className = 'csa-workbench__level-group';
    status.textContent = sourceMeta.dataset.snapshotStatus === 'verified' ? '源码快照匹配 · 映射仍待验证' : '源码版本未验证';
    evidenceMenu.appendChild(status);
    const add = (label, action) => {
      const button = document.createElement('button');
      button.type = 'button'; button.setAttribute('role', 'menuitem');
      button.textContent = label;
      button.addEventListener('click', () => { action(); document.querySelector('[data-csa-evidence-dropdown]').open = false; });
      evidenceMenu.appendChild(button);
    };
    add('查看 Explorer 变换解释', () => sourceSurface.querySelector('[data-explorer-annotation]')?.scrollIntoView({ block: 'start' }));
    ['call', 'definition'].forEach(role => add(role === 'call' ? '跳转调用位置' : '跳转实现定义', () => {
      currentSourceTab = role; sourceRefIndex = null; showSource(item.id, { updateCard: false });
    }));
    item.sourceRefs.forEach((reference, index) => add(`${reference.role} · ${reference.file.split('/').pop()}:${reference.start}–${reference.end}`, () => {
      sourceRefIndex = index; showSource(item.id);
    }));
    const relatedIds = [...new Set(registry.relationsFor(item.id).flatMap(entry => [...entry.sourceIds, ...entry.targetIds]))];
    relatedIds.filter(id => id !== item.id).forEach(id => {
      const related = registry.getNode(id);
      add(`${related.origin === 'official-source' ? '官方' : 'PyPTO'} · ${related.label}`, () => selectEntity(id));
    });
  }

  function setMappingDiff(enabled, { notifyFrame = true } = {}) {
    mappingDiffEnabled = currentLevel === 'l3' && Boolean(enabled);
    diffToggle.setAttribute('aria-checked', String(mappingDiffEnabled));
    diffToggle.title = mappingDiffEnabled ? '关闭差异染色' : '启用差异染色';
    diffFilterToggle.hidden = !mappingDiffEnabled;
    if (!mappingDiffEnabled) closeDiffMenu();
    if (notifyFrame) {
      operatorFrame.contentWindow?.postMessage({ type: 'csa-mapping-diff', enabled: mappingDiffEnabled, categories: [...diffCategories] }, '*');
    }
  }

  const send = (frame, message) => frame.contentWindow?.postMessage(message, '*');

  function syncUrl() {
    const url = new URL(location.href);
    url.searchParams.set('level', currentLevel);
    url.searchParams.set('modelLevel', modelLevel);
    url.searchParams.set('scope', modelScope);
    if (implementationOpened) url.searchParams.set('implementation', 'csa');
    else url.searchParams.delete('implementation');
    history.replaceState(null, '', url);
  }

  function setLevel(level) {
    currentLevel = level === 'l4' ? 'l4' : 'l3';
    if (currentLevel === 'l4') setMappingDiff(false);
    levelTitle.title = implementationOpened ? `${csaImplementation.sourceContext} → ${csaImplementation.targetContext}` : '';
    operatorFrame.dataset.sourceScope = implementationOpened ? csaImplementation.sourceScopeId : '';
    operatorFrame.dataset.targetScope = implementationOpened ? csaImplementation.targetScopeId : '';
    if (levelDropdown) levelDropdown.hidden = !implementationOpened;
    implementationEmpty.hidden = implementationOpened;
    document.querySelectorAll('[data-csa-level]').forEach(button => {
      const active = button.dataset.csaLevel === currentLevel;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-checked', String(active));
    });
    if (levelDropdown) levelDropdown.open = false;
    diffToggle.hidden = !implementationOpened || currentLevel !== 'l3';
    if (incrementsToggle) incrementsToggle.disabled = currentLevel !== 'l3';
    operatorFrame.hidden = !implementationOpened || currentLevel !== 'l3';
    l4Surface.hidden = !implementationOpened || currentLevel !== 'l4';
    mappingCard.hidden = true;
    if (currentLevel === 'l4') renderL4();
    // Keep the L3 iframe alive: no reload, no resetting focus or pan/zoom.
    syncUrl();
  }

  function updateModelNavigation(label) {
    modelTitle.title = '整网内 CSA 语义展开；既有非 CSA 模板仍使用 Pro 参数，未校准为 Flash，不作为已验证的 Flash 整网证据。';
    document.querySelector('[data-csa-scope-label]').textContent = '整网内展开 · CSA；其他模块待校准';
    document.querySelectorAll('[data-csa-model-level]').forEach(button => {
      const active = button.dataset.csaModelLevel === modelLevel;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-checked', String(active));
    });
    syncUrl();
  }

  function setModelLevel(level) {
    modelLevel = ['l0', 'l1', 'l2'].includes(level) ? level : 'l1';
    modelDropdown.open = false;
    if (level === 'l0') {
      send(semanticFrame, { type: 'csa-root-fold', folded: true });
    } else {
      send(semanticFrame, { type: level === 'l2' ? 'csa-semantic-expand-all' : 'csa-semantic-collapse-all' });
    }
    updateModelNavigation();
  }

  function selectEntity(id) {
    const item = registry.getNode(id);
    if (!item) return;
    selectedNodeId = id;
    sourceRefIndex = null;
    const official = item.origin === 'official-source';
    if (official && registry.implementationEntries[id] && !implementationOpened) openImplementation();
    currentSourceTab = 'call';
    focusedIds = registry.mappedIds(id, official ? 'L3' : 'L2');
    if (official) {
      send(semanticFrame, { type: 'csa-lineage-focus', nodeIds: [id] });
      if (implementationOpened) send(operatorFrame, { type: 'csa-lineage-focus', nodeIds: focusedIds });
    } else {
      send(operatorFrame, { type: 'csa-lineage-focus', nodeIds: [id] });
      send(semanticFrame, { type: 'csa-lineage-focus', nodeIds: focusedIds, autoExpand: semanticAutoExpand });
    }
    showSource(id, { updateCard: implementationOpened && currentLevel === 'l3' });
  }

  function openImplementation({ restore = false } = {}) {
    implementationOpened = true;
    if (!operatorFrame.getAttribute('src')) operatorFrame.src = `./index.html?theme=${theme}&embed=1&surface=white&level=l3&rev=lineage-005`;
    setLevel(restore ? currentLevel : 'l3');
    if (!restore) {
      currentSourceTab = 'call'; sourceRefIndex = null;
      showSource(csaImplementation.sourceNodeId, { updateCard: false });
    }
  }

  setTheme(theme, false);
  semanticFrame.src = `./index.html?theme=${theme}&embed=1&surface=white&modelGraph=1&level=${modelLevel === 'l2' ? 'l2' : 'l1'}&rev=lineage-005`;
  fileTabFor(registry.getNode('l1_csa'), registry.getNode('l1_csa').sourceRefs[0]);
  fileTabFor(registry.getNode(csaImplementation.sourceNodeId), registry.getNode(csaImplementation.sourceNodeId).sourceRefs.find(ref => ref.role === 'call'));
  setLevel(currentLevel);
  if (implementationOpened) openImplementation({ restore: true });
  updateModelNavigation();
  showSource('l1_csa', { updateCard: false });
  window.PtoIdeFrame?.init(document.querySelector('[data-ide-frame]'));

  document.querySelectorAll('[data-csa-level]').forEach(button => button.addEventListener('click', () => setLevel(button.dataset.csaLevel)));
  document.querySelectorAll('[data-csa-programs]').forEach(button => button.addEventListener('click', () => {
    if (!implementationOpened || currentLevel !== 'l3') return;
    send(operatorFrame, { type: 'csa-programs-expand', expanded: button.dataset.csaPrograms === 'expand' });
    levelDropdown.open = false;
  }));
  document.querySelectorAll('[data-csa-model-level]').forEach(button => button.addEventListener('click', () => setModelLevel(button.dataset.csaModelLevel)));
  autoExpandToggle.addEventListener('click', () => {
    semanticAutoExpand = !semanticAutoExpand;
    autoExpandToggle.setAttribute('aria-pressed', String(semanticAutoExpand));
    autoExpandToggle.setAttribute('aria-checked', String(semanticAutoExpand));
    autoExpandToggle.classList.toggle('is-active', semanticAutoExpand);
  });
  incrementsToggle?.addEventListener('click', () => {
    incrementsEnabled = !incrementsEnabled;
    incrementsToggle.setAttribute('aria-pressed', String(incrementsEnabled));
    incrementsToggle.setAttribute('aria-checked', String(incrementsEnabled));
    send(operatorFrame, { type: 'csa-increments', enabled: incrementsEnabled });
    levelDropdown.open = false;
  });
  diffToggle.addEventListener('click', () => {
    setMappingDiff(!mappingDiffEnabled);
  });
  function closeDiffMenu() {
    diffLegend.hidden = true;
    diffFilterToggle.setAttribute('aria-expanded', 'false');
  }
  diffFilterToggle.addEventListener('click', event => {
    event.stopPropagation();
    const open = diffLegend.hidden;
    diffLegend.hidden = !open;
    diffFilterToggle.setAttribute('aria-expanded', String(open));
  });
  document.querySelectorAll('[data-csa-diff-category]').forEach(input => input.addEventListener('change', () => {
    if (input.checked) diffCategories.add(input.dataset.csaDiffCategory);
    else diffCategories.delete(input.dataset.csaDiffCategory);
    setMappingDiff(mappingDiffEnabled);
  }));
  diffLegend.addEventListener('click', event => event.stopPropagation());
  document.addEventListener('click', event => {
    if (!diffLegend.hidden && !diffFilterToggle.contains(event.target)) closeDiffMenu();
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !diffLegend.hidden) { closeDiffMenu(); diffFilterToggle.focus(); }
  });
  document.querySelector('[data-csa-file-tabs]').addEventListener('keydown', event => {
    const tabs = [...fileTabs.values()];
    const index = tabs.findIndex(tab => tab.button === document.activeElement);
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    tabs[next].button.focus(); activateFileTab(tabs[next]);
  });
  themeToggle.addEventListener('click', () => setTheme(theme === 'light' ? 'dark' : 'light'));
  legendToggle.addEventListener('click', event => {
    event.stopPropagation();
    legendPanel.hidden = !legendPanel.hidden;
    legendToggle.setAttribute('aria-expanded', String(!legendPanel.hidden));
  });
  legendPanel.addEventListener('click', event => event.stopPropagation());
  document.addEventListener('click', event => {
    document.querySelectorAll('.csa-workbench__level-dropdown[open]').forEach(dropdown => {
      if (!dropdown.contains(event.target)) dropdown.open = false;
    });
    if (!legendPanel.hidden) { legendPanel.hidden = true; legendToggle.setAttribute('aria-expanded', 'false'); }
  });
  document.querySelectorAll('.csa-workbench__level-dropdown').forEach(dropdown => {
    dropdown.addEventListener('keydown', event => {
      const buttons = [...dropdown.querySelectorAll('button:not(:disabled)')];
      if (event.key === 'Escape') { dropdown.open = false; dropdown.querySelector('summary').focus(); }
      if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
        event.preventDefault(); dropdown.open = true;
        const index = buttons.indexOf(document.activeElement);
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (index + (event.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length;
        buttons[next]?.focus();
      }
    });
  });
  sourceBody.addEventListener('click', event => {
    const row = event.target.closest('[data-csa-node-ids]');
    if (!row) return;
    const ids = row.dataset.csaNodeIds.split(',').filter(id => registry.getNode(id));
    // Prefer the current entity when multiple source spans overlap.
    const id = ids.includes(sourceEntityId) ? sourceEntityId : ids.find(key => registry.getNode(key).level === 'L2') || ids[0];
    if (id) selectEntity(id);
  });

  window.addEventListener('message', event => {
    const fromSemantic = event.source === semanticFrame.contentWindow;
    const fromOperator = event.source === operatorFrame.contentWindow;
    if (!fromSemantic && !fromOperator) return;
    const data = event.data || {};
    if (fromOperator && data.type === 'csa-diff-counts') {
      document.querySelectorAll('[data-csa-diff-count]').forEach(element => { element.textContent = data.counts[element.dataset.csaDiffCount] || 0; });
    }
    if ((fromSemantic || fromOperator) && data.type === 'csa-operator-select') {
      if (fromOperator && (!implementationOpened || currentLevel !== 'l3')) return;
      if (fromSemantic && !registry.getNode(data.nodeId)) return;
      selectEntity(data.nodeId);
    }
    if (fromSemantic && data.type === 'csa-semantic-state') {
      if (!modelReady && initialRootFolded) return;
      if (data.rootFolded) { modelLevel = 'l0'; updateModelNavigation(); return; }
      modelLevel = data.state === 'l1' ? 'l1' : 'l2';
      updateModelNavigation(data.label);
    }
    if (data.type === 'csa-operator-ready') {
      if (fromSemantic) {
        modelReady = true;
        if (initialRootFolded) send(semanticFrame, { type: 'csa-root-fold', folded: true });
      }
      if (fromOperator) { setMappingDiff(mappingDiffEnabled); send(operatorFrame, { type: 'csa-increments', enabled: incrementsEnabled }); }
      if (fromSemantic && implementationOpened) {
        send(semanticFrame, { type: 'csa-source-focus', nodeId: 'l1_csa' });
      }
    }
    if (data.type === 'csa-operator-clear') {
      send(fromSemantic ? operatorFrame : semanticFrame, { type: 'csa-source-clear' });
      mappingCard.hidden = true;
    }
  });
})();
