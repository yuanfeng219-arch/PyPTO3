(function () {
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const relationClass = relation => `dg-edge--${relation}`;
  const typeTitle = {finding: '01 / Finding', reasoning: '02 / Abnormal dimension', entity: '03 / Promoted object', cause: 'Cause exploration', hypothesis: 'Hypothesis / pending validation', diagnosis: 'Diagnosis'};

  function svg(tag, attrs = {}) { const el = document.createElementNS(SVG_NS, tag); Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value)); return el; }
  function el(tag, className, text) { const node = document.createElement(tag); if (className) node.className = className; if (text !== undefined) node.textContent = text; return node; }
  function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }

  function roundedOrthogonalPath(points, radius = 8) {
    const compact = points.filter((point, index) => index === 0 || point.x !== points[index - 1].x || point.y !== points[index - 1].y);
    if (compact.length < 2) return '';
    let path = `M ${compact[0].x} ${compact[0].y}`;
    for (let index = 1; index < compact.length - 1; index += 1) {
      const previous = compact[index - 1]; const current = compact[index]; const next = compact[index + 1];
      const incoming = Math.hypot(current.x - previous.x, current.y - previous.y);
      const outgoing = Math.hypot(next.x - current.x, next.y - current.y);
      const corner = Math.min(radius, incoming / 2, outgoing / 2);
      const before = {x: current.x + (previous.x - current.x) * corner / incoming, y: current.y + (previous.y - current.y) * corner / incoming};
      const after = {x: current.x + (next.x - current.x) * corner / outgoing, y: current.y + (next.y - current.y) * corner / outgoing};
      path += ` L ${before.x} ${before.y} Q ${current.x} ${current.y} ${after.x} ${after.y}`;
    }
    const last = compact[compact.length - 1];
    return `${path} L ${last.x} ${last.y}`;
  }

  function chartIcon() {
    const icon = svg('svg', {viewBox: '0 0 16 16', 'aria-hidden': 'true'});
    icon.append(svg('path', {d: 'M2 13.5h12M4 11V7M8 11V3M12 11V5', fill: 'none', stroke: 'currentColor', 'stroke-width': '1.5', 'stroke-linecap': 'square'}));
    return icon;
  }

  function makeEvidenceAttachment(node, id, detail) {
    const attachment = el('div', 'dg-evidence-attachment'); attachment.dataset.dgEvidenceAttachment = `${node.id}:${id}`;
    const title = el('span', 'dg-evidence-attachment__title', detail.attachmentTitle || detail.title);
    const trigger = el('button', 'dg-evidence-attachment__trigger'); trigger.type = 'button'; trigger.dataset.dgEvidence = id; trigger.setAttribute('aria-label', `查看 ${detail.title} 图表`); trigger.append(chartIcon());
    attachment.append(title, trigger);
    return attachment;
  }

  function renderNote(detail, anchor, overlay, viewport, onClose) {
    const note = el('section', 'dg-note'); note.dataset.dgNote = 'true';
    const close = el('button', 'dg-note__close', '×'); close.type = 'button'; close.setAttribute('aria-label', '关闭证据便利贴'); close.addEventListener('click', () => { note.remove(); onClose?.(); });
    note.append(close, el('p', 'dg-note__summary', detail.note), el('h3', '', detail.title));
    if (detail.type === 'bars') {
      const bars = el('div', 'dg-note__bars'); const max = Math.max(...detail.values);
      detail.values.forEach((value, index) => {
        const row = el('div', `dg-note__bar${detail.hotIndex === index || detail.hotIndices?.includes(index) ? ' is-hot' : ''}`);
        const bar = el('i'); bar.style.setProperty('--w', `${Math.round(value / max * 100)}%`);
        row.append(el('span', '', detail.labels[index]), bar, el('strong', '', Number.isInteger(value) ? String(value) : `${value.toFixed(2)}×`)); bars.append(row);
      }); note.append(bars);
    } else {
      const table = el('div', 'dg-note__table'); table.append(el('span', '', ''), el('span', '', '基线'), el('span', '', '当前'));
      detail.rows.forEach(row => row.forEach((cell, index) => table.append(el(index === 0 ? 'strong' : 'span', '', cell)))); note.append(table);
    }
    overlay.replaceChildren(note);
    const viewportRect = viewport.getBoundingClientRect(); const anchorRect = anchor.getBoundingClientRect();
    note.style.maxHeight = `${Math.max(160, viewport.clientHeight - 24)}px`;
    const width = note.offsetWidth; const height = note.offsetHeight; let left = anchorRect.right - viewportRect.left + 14; let top = anchorRect.top - viewportRect.top - 20;
    if (left + width > viewport.clientWidth - 12) left = anchorRect.left - viewportRect.left - width - 14;
    if (top + height > viewport.clientHeight - 12) top = viewport.clientHeight - height - 12;
    if (top < 12) top = 12; if (left < 12) left = 12;
    note.style.left = `${left}px`; note.style.top = `${top}px`; note.style.setProperty('--dg-note-angle', `${(anchorRect.left % 3) - 1}deg`);
  }

  class DiagnosticGraphController {
    constructor(root, data, actions = {}) {
      this.root = root; this.data = data; this.actions = actions; this.nodesById = new Map(data.nodes.map(node => [node.id, node]));
      this.positions = new Map(); this.camera = {x: 0, y: 0, k: 1}; this.selectedId = null; this.noteEvidenceId = null; this.resizeObserver = null;
      this.renderShell(); this.renderNodes(); this.layout(); this.bind(); requestAnimationFrame(() => this.fit());
    }
    renderShell() {
      this.root.className = 'diagnostic-graph'; this.root.innerHTML = '';
      this.toolbar = el('div', 'diagnostic-graph__toolbar'); this.toolbar.setAttribute('aria-label', '诊断图画布控制');
      const controls = [['−', '缩小画布', 'zoom-out'], ['+', '放大画布', 'zoom-in'], ['Fit', '适应画布', 'fit'], ['100%', '恢复 100% 比例', 'hundred']];
      controls.forEach(([text, label, action]) => { const button = el('button', action === 'fit' || action === 'hundred' ? 'btn btn-ghost btn-compact' : 'btn btn-ghost btn-icon', text); button.type = 'button'; button.dataset.dgAction = action; button.setAttribute('aria-label', label); this.toolbar.append(button); });
      this.zoomOutput = el('output', '', '100%'); this.zoomOutput.setAttribute('aria-live', 'polite'); this.toolbar.insertBefore(this.zoomOutput, this.toolbar.querySelector('[data-dg-action="fit"]'));
      this.viewport = el('div', 'diagnostic-graph__viewport'); this.viewport.tabIndex = 0; this.viewport.setAttribute('aria-label', '诊断推理画布');
      this.world = el('div', 'diagnostic-graph__world'); this.edgeSvg = svg('svg', {class: 'diagnostic-graph__edges', 'aria-hidden': 'true'}); this.edgeSvg.append(this.markerDefs()); this.nodeLayer = el('div', 'diagnostic-graph__nodes'); this.world.append(this.edgeSvg, this.nodeLayer); this.overlay = el('div', 'diagnostic-graph__overlay');
      this.hint = el('div', 'dg-canvas-hint'); this.hint.innerHTML = '拖动空白处平移 · <kbd>Ctrl</kbd> + 滚轮缩放'; this.viewport.append(this.world, this.overlay, this.hint); this.root.append(this.toolbar, this.viewport);
    }
    markerDefs() { const defs = svg('defs'); const main = svg('marker', {id: 'dg-arrow', markerWidth: 8, markerHeight: 8, refX: 6, refY: 4, orient: 'auto'}); main.append(svg('path', {d: 'M1 1 L7 4 L1 7', fill: 'none', stroke: '#c29570', 'stroke-width': 1.2})); const muted = svg('marker', {id: 'dg-arrow-muted', markerWidth: 8, markerHeight: 8, refX: 6, refY: 4, orient: 'auto'}); muted.append(svg('path', {d: 'M1 1 L7 4 L1 7', fill: 'none', stroke: '#7d887e', 'stroke-width': 1.1})); defs.append(main, muted); return defs; }
    renderNodes() {
      this.nodeElements = new Map(); this.evidenceElements = new Map(); this.nodeLayer.replaceChildren();
      this.data.nodes.forEach(node => { const isAbnormal = node.kind === 'reasoning' || node.kind === 'entity' || node.state === 'abnormal'; const classes = isAbnormal ? 'dg-node is-abnormal' : `dg-node dg-node--${node.kind}${node.state ? ` is-${node.state}` : ''}`; const item = el('article', classes); item.id = `dg-node-${node.id}`; item.dataset.dgNode = node.id; item.tabIndex = 0; item.setAttribute('aria-label', node.action ? `${node.title}。${node.signal}。${node.summary}。下一步：${node.action.title}。${node.action.summary}` : `${node.title} ${node.signal}`);
        if (node.kind === 'diagnosis') {
          const head = el('div', 'dg-node__head'); head.append(el('span', 'dg-node__eyebrow', typeTitle[node.kind]), el('strong', 'dg-node__signal', node.signal));
          item.append(head, el('h3', 'dg-node__title', node.title));
        } else item.append(el('span', 'dg-node__eyebrow', typeTitle[node.kind]), el('h3', 'dg-node__title', node.title), el('strong', 'dg-node__signal', node.signal), el('p', 'dg-node__summary', node.summary));
        if (node.evidenceRefs?.length) { const rail = el('div', 'dg-evidence-rail'); node.evidenceRefs.forEach(evidenceId => { const attachment = makeEvidenceAttachment(node, evidenceId, this.data.evidenceById[evidenceId]); rail.append(attachment); this.evidenceElements.set(`${node.id}:${evidenceId}`, attachment); }); item.append(rail); }
        if (node.action) {
          const actionSection = el('div', 'dg-node__action'); actionSection.title = node.action.summary;
          const actionCopy = el('div', 'dg-node__action-copy'); actionCopy.append(el('span', 'dg-node__action-eyebrow', node.action.eyebrow), el('strong', 'dg-node__action-title', node.action.title));
          const action = el('button', 'btn btn-solid btn-compact', node.action.label); action.type = 'button'; action.dataset.dgScenario = 'true';
          actionSection.append(actionCopy, action); item.append(actionSection);
        }
        this.nodeLayer.append(item); this.nodeElements.set(node.id, item); });
    }
    measure() { return new Map([...this.nodeElements].map(([id, item]) => { const rail = item.querySelector('.dg-evidence-rail'); const width = item.offsetWidth; const height = item.offsetHeight; const railWidth = rail?.offsetWidth || 0; const railHeight = rail?.offsetHeight || 0; return [id, {width, height, railWidth, railHeight, occupiedWidth: width, occupiedHeight: height + (railHeight ? railHeight + 12 : 0)}]; })); }
    ranks() {
      const ranks = new Map(this.data.nodes.map(node => [node.id, 0]));
      const incoming = new Map(this.data.nodes.map(node => [node.id, 0]));
      const outgoing = new Map(this.data.nodes.map(node => [node.id, []]));
      this.data.edges.forEach(edge => {
        if (!outgoing.has(edge.source) || !incoming.has(edge.target)) throw new Error(`Unknown graph edge: ${edge.id}`);
        outgoing.get(edge.source).push(edge); incoming.set(edge.target, incoming.get(edge.target) + 1);
      });
      const queue = this.data.nodes.filter(node => incoming.get(node.id) === 0).map(node => node.id);
      while (queue.length) {
        const source = queue.shift();
        outgoing.get(source).forEach(edge => {
          ranks.set(edge.target, Math.max(ranks.get(edge.target), ranks.get(source) + 1));
          incoming.set(edge.target, incoming.get(edge.target) - 1);
          if (incoming.get(edge.target) === 0) queue.push(edge.target);
        });
      }
      return ranks;
    }
    primaryPath() {
      const incoming = new Map(this.data.nodes.map(node => [node.id, 0]));
      const outgoing = new Map(this.data.nodes.map(node => [node.id, []]));
      this.data.edges.forEach(edge => { incoming.set(edge.target, (incoming.get(edge.target) || 0) + 1); outgoing.get(edge.source)?.push(edge); });
      const preferredRelations = ['locate', 'promote', 'continue', 'support', 'diagnose'];
      let current = this.data.nodes.find(node => incoming.get(node.id) === 0 && node.priority === 'anchor') || this.data.nodes.find(node => incoming.get(node.id) === 0);
      const path = []; const visited = new Set();
      while (current && !visited.has(current.id)) {
        path.push(current.id); visited.add(current.id);
        const candidates = outgoing.get(current.id) || [];
        const next = candidates.slice().sort((a, b) => preferredRelations.indexOf(a.relation) - preferredRelations.indexOf(b.relation)).find(edge => preferredRelations.includes(edge.relation));
        current = next ? this.nodesById.get(next.target) : null;
      }
      return path;
    }
    layout() {
      const sizes = this.measure(); const ranks = this.ranks(); const layers = new Map(); this.data.nodes.forEach(node => { const rank = ranks.get(node.id) ?? 0; if (!layers.has(rank)) layers.set(rank, []); layers.get(rank).push(node); });
      const layerWidths = [...layers.entries()].sort(([a], [b]) => a - b).map(([rank, nodes]) => [rank, Math.max(...nodes.map(node => Math.max(sizes.get(node.id).occupiedWidth, sizes.get(node.id).railWidth)))]); let x = 72; const xByRank = new Map(); layerWidths.forEach(([rank, width]) => { xByRank.set(rank, x); x += width + 112; });
      const primary = this.primaryPath(); const primarySet = new Set(primary); const mainCenterY = 420; const branchGap = 42; const groupGap = 58;
      this.positions.clear();
      const setPosition = (node, y) => { const size = sizes.get(node.id); this.positions.set(node.id, {x: xByRank.get(ranks.get(node.id)), y, width: size.width, height: size.height, railWidth: size.railWidth, railHeight: size.railHeight, occupiedWidth: size.occupiedWidth, occupiedHeight: size.occupiedHeight}); };
      layers.forEach(nodes => {
        const spineIndex = nodes.findIndex(node => primarySet.has(node.id));
        if (spineIndex < 0) {
          const total = nodes.reduce((sum, node) => sum + sizes.get(node.id).occupiedHeight, 0) + Math.max(0, nodes.length - 1) * branchGap;
          let y = mainCenterY - total / 2; nodes.forEach(node => { setPosition(node, y); y += sizes.get(node.id).occupiedHeight + branchGap; }); return;
        }
        const spine = nodes[spineIndex]; const spineSize = sizes.get(spine.id); const spineY = mainCenterY - spineSize.height / 2; setPosition(spine, spineY);
        let upperCursor = spineY - groupGap;
        nodes.slice(0, spineIndex).reverse().forEach(node => { const size = sizes.get(node.id); upperCursor -= size.occupiedHeight; setPosition(node, upperCursor); upperCursor -= branchGap; });
        let lowerCursor = spineY + spineSize.occupiedHeight + groupGap;
        nodes.slice(spineIndex + 1).forEach(node => { const size = sizes.get(node.id); setPosition(node, lowerCursor); lowerCursor += size.occupiedHeight + branchGap; });
      });
      const firstBounds = [...this.positions.values()].reduce((box, pos) => ({left: Math.min(box.left, pos.x), top: Math.min(box.top, pos.y), right: Math.max(box.right, pos.x + Math.max(pos.width, pos.railWidth)), bottom: Math.max(box.bottom, pos.y + pos.occupiedHeight)}), {left: Infinity, top: Infinity, right: 0, bottom: 0});
      const shiftX = 72 - firstBounds.left; const shiftY = 72 - firstBounds.top;
      this.positions.forEach(pos => { pos.x += shiftX; pos.y += shiftY; });
      const bounds = [...this.positions.values()].reduce((box, pos) => ({right: Math.max(box.right, pos.x + Math.max(pos.width, pos.railWidth)), bottom: Math.max(box.bottom, pos.y + pos.height + (pos.railHeight ? pos.railHeight + 12 : 0))}), {right: 0, bottom: 0});
      this.world.style.setProperty('--dg-world-width', `${bounds.right + 72}px`); this.world.style.setProperty('--dg-world-height', `${Math.max(620, bounds.bottom + 72)}px`); this.renderGeometry();
    }
    renderGeometry() {
      this.nodeElements.forEach((item, id) => { const pos = this.positions.get(id); item.style.transform = `translate(${pos.x}px,${pos.y}px)`; }); this.renderEdges(); this.applyCamera();
    }
    renderEdges() {
      const defs = this.edgeSvg.querySelector('defs'); this.edgeSvg.replaceChildren(defs);
      const outgoing = new Map();
      this.data.edges.forEach(edge => { if (!outgoing.has(edge.source)) outgoing.set(edge.source, []); outgoing.get(edge.source).push(edge); });
      outgoing.forEach(edges => edges.sort((a, b) => { const aPos = this.positions.get(a.target); const bPos = this.positions.get(b.target); return (aPos.y + aPos.height / 2) - (bPos.y + bPos.height / 2); }));
      this.data.edges.forEach(edge => {
        const from = this.positions.get(edge.source); const to = this.positions.get(edge.target); if (!from || !to) return;
        const siblings = outgoing.get(edge.source) || [edge]; const portIndex = siblings.indexOf(edge); const sourceY = siblings.length === 1 ? from.y + from.height / 2 : from.y + 18 + portIndex * ((from.height - 36) / Math.max(1, siblings.length - 1));
        const start = {x: from.x + from.width, y: sourceY}; const end = {x: to.x, y: to.y + to.height / 2};
        const isStraight = Math.abs(start.y - end.y) < 1; const channel = Math.min(end.x - 22, start.x + 30 + portIndex * 13);
        const points = isStraight ? [start, end] : [start, {x: channel, y: start.y}, {x: channel, y: end.y}, end];
        const path = svg('path', {class: `dg-edge ${relationClass(edge.relation)}`, d: roundedOrthogonalPath(points)}); path.dataset.dgEdge = edge.id; path.append(svg('title')); path.querySelector('title').textContent = edge.label || edge.relation; this.edgeSvg.append(path);
        if (edge.label) {
          const group = svg('g', {class: 'dg-edge-label'}); const width = Math.max(60, edge.label.length * 7 + 16);
          const horizontalSegments = points.slice(0, -1).map((point, index) => ({from: point, to: points[index + 1], length: Math.abs(points[index + 1].x - point.x)})).filter(segment => segment.from.y === segment.to.y).sort((a, b) => b.length - a.length);
          const segment = horizontalSegments[0] || {from: start, to: end}; const labelX = (segment.from.x + segment.to.x) / 2; const labelY = segment.from.y - 15;
          group.setAttribute('transform', `translate(${labelX} ${labelY})`); group.append(svg('rect', {x: -width / 2, y: -10, width, height: 20, rx: 4}), svg('text', {y: 4})); group.lastChild.textContent = edge.label; this.edgeSvg.append(group);
        }
      }); this.updateFocus();
    }
    applyCamera() { this.world.style.transform = `translate(${this.camera.x}px,${this.camera.y}px) scale(${this.camera.k})`; this.zoomOutput.value = `${Math.round(this.camera.k * 100)}%`; this.zoomOutput.textContent = this.zoomOutput.value; }
    fit() { const width = parseFloat(this.world.style.getPropertyValue('--dg-world-width')) || 1600; const height = parseFloat(this.world.style.getPropertyValue('--dg-world-height')) || 900; const pad = 34; const k = Math.min((this.viewport.clientWidth - pad * 2) / width, (this.viewport.clientHeight - pad * 2) / height); this.camera.k = Math.max(.08, k); this.camera.x = (this.viewport.clientWidth - width * this.camera.k) / 2; this.camera.y = (this.viewport.clientHeight - height * this.camera.k) / 2; this.applyCamera(); }
    setScale(next, point) { const old = this.camera.k; const k = clamp(next, .1, 3); const anchor = point || {x: this.viewport.clientWidth / 2, y: this.viewport.clientHeight / 2}; this.camera.x = anchor.x - (anchor.x - this.camera.x) * k / old; this.camera.y = anchor.y - (anchor.y - this.camera.y) * k / old; this.camera.k = k; this.overlay.replaceChildren(); this.noteEvidenceId = null; this.applyCamera(); }
    semanticEdges() { return this.data.edges; }
    focusFor(id) { return new Set([id, ...this.semanticEdges().filter(edge => edge.source === id || edge.target === id).flatMap(edge => [edge.source, edge.target])]); }
    updateFocus() { const focused = this.selectedId ? this.focusFor(this.selectedId) : null; const edgesById = new Map(this.semanticEdges().map(edge => [edge.id, edge])); this.nodeElements.forEach((item, id) => { item.classList.toggle('is-selected', id === this.selectedId); item.classList.toggle('is-dimmed', Boolean(focused && !focused.has(id))); }); this.evidenceElements.forEach((item, id) => { const [nodeId] = id.split(':'); item.classList.toggle('is-dimmed', Boolean(focused && !focused.has(nodeId))); }); this.edgeSvg.querySelectorAll('.dg-edge').forEach(path => { const edge = edgesById.get(path.dataset.dgEdge); const related = !focused || (focused.has(edge.source) && focused.has(edge.target)); path.classList.toggle('is-related', Boolean(focused && related)); path.classList.toggle('is-dimmed', Boolean(focused && !related)); }); this.edgeSvg.querySelectorAll('.dg-edge-label').forEach(label => label.classList.toggle('is-dimmed', Boolean(focused && !label.previousElementSibling.classList.contains('is-related')))); }
    select(id) { this.selectedId = this.selectedId === id ? null : id; this.updateFocus(); }
    bind() {
      this.toolbar.addEventListener('click', event => { const action = event.target.closest('[data-dg-action]')?.dataset.dgAction; if (!action) return; if (action === 'zoom-in') this.setScale(this.camera.k + .12); if (action === 'zoom-out') this.setScale(this.camera.k - .12); if (action === 'fit') this.fit(); if (action === 'hundred') { this.camera = {x: 42, y: 42, k: 1}; this.applyCamera(); } });
      this.viewport.addEventListener('wheel', event => { event.preventDefault(); const rect = this.viewport.getBoundingClientRect(); const point = {x: event.clientX - rect.left, y: event.clientY - rect.top}; if (event.ctrlKey || event.metaKey) this.setScale(this.camera.k * (event.deltaY > 0 ? .9 : 1.1), point); else { this.camera.x -= event.deltaX; this.camera.y -= event.deltaY; this.overlay.replaceChildren(); this.noteEvidenceId = null; this.applyCamera(); } }, {passive: false});
      this.viewport.addEventListener('pointerdown', event => { const node = event.target.closest('[data-dg-node]'); const thumb = event.target.closest('[data-dg-evidence]'); if (thumb || node) return; this.pan(event); });
      this.viewport.addEventListener('click', event => { if (event.target.closest('[data-dg-evidence]')) return; const node = event.target.closest('[data-dg-node]'); if (node) this.select(node.dataset.dgNode); else { this.selectedId = null; this.overlay.replaceChildren(); this.noteEvidenceId = null; this.updateFocus(); } });
      this.nodeLayer.addEventListener('click', event => {
        const scenario = event.target.closest('[data-dg-scenario]');
        if (scenario) { event.stopPropagation(); this.actions.openScenario?.(); return; }
        const thumb = event.target.closest('[data-dg-evidence]'); if (!thumb) return; event.stopPropagation(); const id = thumb.dataset.dgEvidence; if (this.noteEvidenceId === id) { this.overlay.replaceChildren(); this.noteEvidenceId = null; return; } this.noteEvidenceId = id; renderNote(this.data.evidenceById[id], thumb, this.overlay, this.viewport, () => { this.noteEvidenceId = null; });
      });
      this.viewport.addEventListener('keydown', event => { if (event.key === 'Escape') { if (this.noteEvidenceId) { this.overlay.replaceChildren(); this.noteEvidenceId = null; } else if (this.selectedId) { this.selectedId = null; this.updateFocus(); } } if (event.key === ' ' && event.target === this.viewport) event.preventDefault(); });
      this.resizeObserver = new ResizeObserver(() => { if (!this.userCameraMoved) this.fit(); });
      this.resizeObserver.observe(this.viewport);
    }
    pan(event) { const start = {x: event.clientX, y: event.clientY, cameraX: this.camera.x, cameraY: this.camera.y}; this.viewport.dataset.dgPanning = 'true'; this.viewport.setPointerCapture(event.pointerId); const move = moveEvent => { this.userCameraMoved = true; this.camera.x = start.cameraX + moveEvent.clientX - start.x; this.camera.y = start.cameraY + moveEvent.clientY - start.y; this.overlay.replaceChildren(); this.noteEvidenceId = null; this.applyCamera(); }; const end = endEvent => { this.viewport.dataset.dgPanning = 'false'; this.viewport.removeEventListener('pointermove', move); this.viewport.removeEventListener('pointerup', end); this.viewport.removeEventListener('pointercancel', end); if (this.viewport.hasPointerCapture(endEvent.pointerId)) this.viewport.releasePointerCapture(endEvent.pointerId); }; this.viewport.addEventListener('pointermove', move); this.viewport.addEventListener('pointerup', end); this.viewport.addEventListener('pointercancel', end); }
    ensureLayout() { if (!this.positions.size) { this.layout(); this.fit(); } }
    resize() { if (!this.userCameraMoved) this.fit(); }
    destroy() { this.resizeObserver?.disconnect(); this.resizeObserver = null; this.root.replaceChildren(); }
  }
  window.DiagnosticGraphController = DiagnosticGraphController;
}());
