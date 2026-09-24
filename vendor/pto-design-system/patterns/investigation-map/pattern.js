(function registerPtoInvestigationMap(global) {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';
  const defaults = {
    finding: '01 / Finding',
    reasoning: '02 / Abnormal dimension',
    entity: '03 / Promoted object',
    cause: 'Cause exploration',
    hypothesis: 'Hypothesis / pending validation',
    diagnosis: 'Diagnosis'
  };

  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const svg = (tag, attrs = {}) => {
    const node = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    return node;
  };
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const relationClass = relation => `dg-edge--${relation}`;

  /* Rounded orthogonal path through the given waypoints. */
  function pathFor(points, radius = 8) {
    const compact = points.filter((point, index) => !index || point.x !== points[index - 1].x || point.y !== points[index - 1].y);
    if (compact.length < 2) return '';
    let result = `M ${compact[0].x} ${compact[0].y}`;
    for (let index = 1; index < compact.length - 1; index += 1) {
      const previous = compact[index - 1];
      const current = compact[index];
      const next = compact[index + 1];
      const incoming = Math.hypot(current.x - previous.x, current.y - previous.y);
      const outgoing = Math.hypot(next.x - current.x, next.y - current.y);
      const corner = Math.min(radius, incoming / 2, outgoing / 2);
      const before = {
        x: current.x + (previous.x - current.x) * corner / incoming,
        y: current.y + (previous.y - current.y) * corner / incoming
      };
      const after = {
        x: current.x + (next.x - current.x) * corner / outgoing,
        y: current.y + (next.y - current.y) * corner / outgoing
      };
      result += ` L ${before.x} ${before.y} Q ${current.x} ${current.y} ${after.x} ${after.y}`;
    }
    const last = compact[compact.length - 1];
    return `${result} L ${last.x} ${last.y}`;
  }

  function chartIcon() {
    const icon = svg('svg', { viewBox: '0 0 16 16', 'aria-hidden': 'true' });
    icon.append(svg('path', {
      d: 'M2 13.5h12M4 11V7M8 11V3M12 11V5',
      fill: 'none', stroke: 'currentColor', 'stroke-width': '1.5', 'stroke-linecap': 'square'
    }));
    return icon;
  }

  /* Evidence "sticky note" — shared visual language; supports bars / sequence /
     table plus optional source, scope/missing facts and a detail link. */
  function note(detail, anchor, controller) {
    const overlay = controller.overlay;
    const viewport = controller.viewport;
    const paper = el('section', 'dg-note');
    paper.dataset.dgNote = 'true';
    paper.tabIndex = -1;
    paper.addEventListener('pointerdown', event => event.stopPropagation());
    paper.addEventListener('wheel', event => event.stopPropagation(), { passive: true });
    paper.addEventListener('click', event => event.stopPropagation());

    const close = el('button', 'dg-note__close', '×');
    close.type = 'button';
    close.setAttribute('aria-label', '关闭证据便利贴');
    close.addEventListener('click', () => controller.closeNote(true));
    paper.append(close, el('p', 'dg-note__summary', detail.note || detail.summary || ''), el('h3', '', detail.title || '证据'));

    if (detail.type === 'bars' && detail.values) {
      const bars = el('div', 'dg-note__bars');
      const max = Math.max(1, ...detail.values);
      if (detail.labelWidth) bars.style.setProperty('--dg-note-label-width', detail.labelWidth);
      detail.values.forEach((value, index) => {
        const row = el('div', `dg-note__bar${detail.hotIndex === index || detail.hotIndices?.includes(index) ? ' is-hot' : ''}`);
        const bar = el('i');
        bar.style.setProperty('--w', `${Math.round(value / max * 100)}%`);
        const valueText = detail.valueSuffix
          ? `${Number(value.toFixed(2))}${detail.valueSuffix}`
          : (Number.isInteger(value) ? String(value) : `${value.toFixed(2)}×`);
        row.append(el('span', '', detail.labels?.[index] || ''), bar, el('strong', '', valueText));
        bars.append(row);
      });
      paper.append(bars);
    } else if (detail.type === 'sequence' && detail.steps) {
      const sequence = el('div', 'dg-note__sequence');
      detail.steps.forEach(step => {
        sequence.append(el('strong', 'dg-note__sequence-op', step[0]));
        if (step[1]) sequence.append(el('span', 'dg-note__sequence-link', `↓ ${step[1]}`));
      });
      paper.append(sequence);
      if (detail.facts) {
        const facts = el('div', 'dg-note__facts');
        detail.facts.forEach(fact => facts.append(el('strong', '', fact[0]), el('span', '', fact[1])));
        paper.append(facts);
      }
    } else if (detail.rows) {
      const table = el('div', 'dg-note__table');
      const columns = detail.columns || ['', '基线', '当前'];
      table.style.gridTemplateColumns = detail.columnTemplate || `repeat(${columns.length}, minmax(0, 1fr))`;
      columns.forEach(column => table.append(el('span', '', column)));
      detail.rows.forEach(row => row.forEach((cell, index) => table.append(el(index === 0 ? 'strong' : 'span', '', cell))));
      paper.append(table);
    }

    if (detail.source) paper.append(el('p', 'dg-source-note', detail.source));
    if (detail.scope || detail.missing) {
      const facts = el('div', 'dg-note__facts');
      if (detail.scope) facts.append(el('strong', '', '范围'), el('span', '', detail.scope));
      if (detail.missing) facts.append(el('strong', '', '缺失'), el('span', '', detail.missing));
      paper.append(facts);
    }
    const action = detail.action || detail.tab;
    if (action) {
      const button = el('button', 'btn dg-evidence-detail-link', detail.action?.label || '打开原始详情页 →');
      button.type = 'button';
      button.addEventListener('click', event => {
        event.stopPropagation();
        controller.actions.openEvidence?.(action);
        if (!controller.actions.openEvidence && detail.tab) controller.actions.openDetail?.(detail.tab);
      });
      paper.append(button);
    }

    overlay.replaceChildren(paper);
    const bounds = viewport.getBoundingClientRect();
    const anchorBounds = anchor?.getBoundingClientRect?.() || bounds;
    const left = clamp(anchorBounds.left - bounds.left + 18, 12, Math.max(12, viewport.clientWidth - 402));
    const top = clamp(anchorBounds.bottom - bounds.top + 12, 12, Math.max(12, viewport.clientHeight - paper.offsetHeight - 12));
    paper.style.left = `${left}px`;
    paper.style.top = `${top}px`;
    paper.style.setProperty('--dg-note-angle', `${(anchorBounds.left % 3) - 1}deg`);
    paper.focus({ preventScroll: true });
  }

  class Controller {
    constructor(root, data, actions = {}) {
      this.root = root;
      this.data = data || { nodes: [], edges: [], evidenceById: {} };
      this.actions = actions;
      this.nodesById = new Map(this.data.nodes.map(node => [node.id, node]));
      this.positions = new Map();
      this.rankOf = new Map();
      this.camera = { x: 0, y: 0, k: 1 };
      this.selectedId = actions.savedState?.selectedId || null;
      this.expanded = new Set(actions.savedState?.expanded || []);
      this.noteEvidenceId = actions.savedState?.noteEvidenceId || null;
      this.resizeObserver = null;
      this.renderShell();
      this.renderNodes();
      this.expanded.forEach(key => this.renderInlineEvidence(key));
      this.layout();
      this.bind();
      if (actions.savedState?.camera) {
        this.camera = { ...actions.savedState.camera };
        this.userCameraMoved = true;
        this.applyCamera();
      }
      this.initialFrame = requestAnimationFrame(() => {
        if (!actions.savedState && root.isConnected) this.fit();
        else if (this.noteEvidenceId) this.openEvidence(this.noteEvidenceId, false);
      });
    }

    renderShell() {
      this.root.className = ['diagnostic-graph', this.actions.className].filter(Boolean).join(' ');
      this.root.replaceChildren();
      this.toolbar = el('div', 'diagnostic-graph__toolbar');
      this.toolbar.setAttribute('aria-label', this.actions.toolbarLabel || '诊断图画布控制');
      [['−', '缩小画布', 'zoom-out'], ['+', '放大画布', 'zoom-in'], ['Fit', '适应画布', 'fit'], ['100%', '恢复 100% 比例', 'hundred']].forEach(([text, label, action]) => {
        const button = el('button', action === 'fit' || action === 'hundred' ? 'btn btn-ghost btn-compact' : 'btn btn-ghost btn-icon', text);
        button.type = 'button';
        button.dataset.dgAction = action;
        button.setAttribute('aria-label', label);
        this.toolbar.append(button);
      });
      this.zoomOutput = el('output', '', '100%');
      this.zoomOutput.setAttribute('aria-live', 'polite');
      this.toolbar.insertBefore(this.zoomOutput, this.toolbar.querySelector('[data-dg-action="fit"]'));

      this.viewport = el('div', 'diagnostic-graph__viewport');
      this.viewport.tabIndex = 0;
      this.viewport.setAttribute('aria-label', this.actions.ariaLabel || '诊断推理画布');
      this.world = el('div', 'diagnostic-graph__world');
      this.edgeSvg = svg('svg', { class: 'diagnostic-graph__edges', 'aria-hidden': 'true' });
      this.edgeSvg.append(this.markerDefs());
      this.nodeLayer = el('div', 'diagnostic-graph__nodes');
      this.world.append(this.edgeSvg, this.nodeLayer);
      this.overlay = el('div', 'diagnostic-graph__overlay');
      this.viewport.append(this.world, this.overlay);
      this.root.append(this.toolbar, this.viewport);

      if (this.actions.showInfo !== false) {
        const info = el('details', 'dg-context-info');
        info.innerHTML = '<summary aria-label="画布说明" title="画布说明">ⓘ</summary><div><strong>诊断图谱</strong><p>从症状、异常维度追溯到根因与验证建议。节点中的证据可展开查看。</p><p>拖动空白处平移；Ctrl / ⌘ + 滚轮缩放；Esc 取消选中。</p></div>';
        this.root.append(info);
      }
    }

    markerDefs() {
      const defs = svg('defs');
      [['dg-arrow', '#c29570', 1.2], ['dg-arrow-muted', '#7d887e', 1.1]].forEach(([id, stroke, width]) => {
        const marker = svg('marker', { id, markerWidth: 8, markerHeight: 8, refX: 6, refY: 4, orient: 'auto' });
        marker.append(svg('path', { d: 'M1 1 L7 4 L1 7', fill: 'none', stroke, 'stroke-width': width }));
        defs.append(marker);
      });
      return defs;
    }

    renderNodes() {
      this.nodeElements = new Map();
      this.evidenceElements = new Map();
      this.nodeLayer.replaceChildren();
      const titles = { ...defaults, ...(this.actions.typeTitles || {}) };
      const isNote = this.actions.evidenceMode === 'note';

      this.data.nodes.forEach(node => {
        /* Structure is decided by `kind`, emphasis by `state` — never by whether
           the signal happens to look numeric. */
        const isAbnormal = node.kind === 'reasoning' || node.kind === 'entity' || node.state === 'abnormal';
        const classes = isAbnormal
          ? 'dg-node is-abnormal'
          : `dg-node dg-node--${node.kind}${node.state ? ` is-${node.state}` : ''}`;
        const item = el('article', classes);
        item.id = `dg-node-${node.id}`;
        item.dataset.dgNode = node.id;
        item.tabIndex = 0;
        item.setAttribute('aria-label', node.action
          ? `${node.title}。${node.signal || ''}。${node.summary || ''}。下一步：${node.action.title || node.action.label || ''}`
          : `${node.title} ${node.signal || ''}`);

        if (node.kind === 'diagnosis') {
          const head = el('div', 'dg-node__head');
          head.append(el('span', 'dg-node__eyebrow', titles[node.kind]), el('strong', 'dg-node__signal', node.signal || ''));
          item.append(head, el('h3', 'dg-node__title', node.title));
          if (node.summary) item.append(el('p', 'dg-node__summary', node.summary));
        } else {
          item.append(
            el('span', 'dg-node__eyebrow', titles[node.kind]),
            el('h3', 'dg-node__title', node.title),
            el('strong', 'dg-node__signal', node.signal || ''),
            el('p', 'dg-node__summary', node.summary || '')
          );
        }

        if (node.evidenceRefs?.length) {
          const rail = el('div', 'dg-evidence-rail');
          node.evidenceRefs.forEach(id => {
            const detail = this.data.evidenceById?.[id] || { title: id };
            const attachment = el('div', 'dg-evidence-attachment');
            attachment.dataset.dgEvidenceAttachment = `${node.id}:${id}`;
            attachment.append(el('span', 'dg-evidence-attachment__title', detail.attachmentTitle || detail.title));
            if (isNote) {
              const trigger = el('button', 'dg-evidence-attachment__trigger');
              trigger.type = 'button';
              trigger.dataset.dgEvidence = id;
              trigger.setAttribute('aria-label', `查看 ${detail.title} 图表`);
              trigger.setAttribute('aria-expanded', 'false');
              trigger.append(chartIcon());
              attachment.append(trigger);
            } else {
              const trigger = el('button', 'btn dg-evidence-attachment__toggle', '展开图表');
              trigger.type = 'button';
              trigger.dataset.dgEvidence = id;
              trigger.setAttribute('aria-expanded', 'false');
              attachment.append(trigger);
            }
            rail.append(attachment);
            this.evidenceElements.set(`${node.id}:${id}`, attachment);
          });
          item.append(rail);
        }

        if (node.action) {
          const section = el('div', 'dg-node__action');
          section.title = node.action.summary || '';
          const copy = el('div', 'dg-node__action-copy');
          copy.append(el('span', 'dg-node__action-eyebrow', node.action.eyebrow), el('strong', 'dg-node__action-title', node.action.title));
          const button = el('button', 'btn btn-solid btn-compact', node.action.label);
          button.type = 'button';
          button.dataset.dgScenario = 'true';
          section.append(copy, button);
          item.append(section);
        }

        this.nodeLayer.append(item);
        this.nodeElements.set(node.id, item);
      });
    }

    renderInlineEvidence(key) {
      const attachment = this.evidenceElements.get(key);
      if (!attachment) return;
      const trigger = attachment.querySelector('[data-dg-evidence]');
      const open = this.expanded.has(key);
      const detail = this.data.evidenceById[trigger.dataset.dgEvidence];
      trigger.setAttribute('aria-expanded', String(open));
      trigger.textContent = open ? '收起图表' : '展开图表';
      attachment.querySelector('.dg-inline-evidence')?.remove();
      if (open) {
        const region = el('div', 'dg-inline-evidence');
        attachment.append(region);
        const shared = this.actions.renderEvidence?.(trigger.dataset.dgEvidence, detail);
        if (shared) {
          region.classList.add('dg-shared-chart');
          region.innerHTML = shared;
          region.addEventListener('click', event => {
            event.stopPropagation();
            if (event.target.closest('.dg-evidence-detail-link') && detail.tab) this.actions.openDetail?.(detail.tab);
          });
          if (detail.tab) {
            const link = el('button', 'btn dg-evidence-detail-link', '打开原始详情页 →');
            link.type = 'button';
            region.append(link);
          }
        } else {
          note(detail, trigger, { overlay: region, viewport: this.viewport, actions: this.actions, closeNote: () => {} });
        }
      }
      attachment.closest('[data-dg-node]').classList.toggle('has-expanded-evidence', open);
    }

    toggleEvidence(trigger) {
      const key = trigger.closest('[data-dg-evidence-attachment]').dataset.dgEvidenceAttachment;
      const node = trigger.closest('[data-dg-node]').dataset.dgNode;
      if (this.actions.evidenceMode === 'note') return this.openEvidence(trigger.dataset.dgEvidence, true, trigger);
      const before = this.positions.get(node);
      if (this.expanded.has(key)) this.expanded.delete(key);
      else this.expanded.add(key);
      this.selectedId = node;
      this.userCameraMoved = true;
      this.renderInlineEvidence(key);
      this.layout();
      this.actions.afterEvidenceLayout?.();
      const after = this.positions.get(node);
      if (before && after) {
        this.camera.x += (before.x - after.x) * this.camera.k;
        this.camera.y += (before.y - after.y) * this.camera.k;
      }
      this.applyCamera();
      this.updateFocus();
    }

    openEvidence(id, focus = true, anchor) {
      const detail = this.data.evidenceById?.[id];
      if (!detail) return;
      this.noteEvidenceId = id;
      this.noteFocus = anchor || this.nodeElements.get(this.data.nodes.find(node => node.evidenceRefs?.includes(id))?.id) || this.viewport;
      note(detail, this.noteFocus, this);
      if (focus) this.selectedId = this.noteFocus.closest?.('[data-dg-node]')?.dataset.dgNode || this.selectedId;
      this.updateFocus();
    }

    closeNote(restore) {
      const previous = this.noteFocus;
      this.overlay.replaceChildren();
      this.noteEvidenceId = null;
      if (restore) previous?.focus?.({ preventScroll: true });
    }

    measure() {
      return new Map([...this.nodeElements].map(([id, item]) => {
        const rail = item.querySelector('.dg-evidence-rail');
        const width = item.offsetWidth;
        const height = item.offsetHeight;
        const railWidth = rail?.offsetWidth || 0;
        const railHeight = rail?.offsetHeight || 0;
        return [id, {
          width,
          height,
          railWidth,
          railHeight,
          occupiedWidth: width,
          occupiedHeight: height + (railHeight ? railHeight + 12 : 0)
        }];
      }));
    }

    ranks() {
      const ranks = new Map(this.data.nodes.map(node => [node.id, 0]));
      const incoming = new Map(this.data.nodes.map(node => [node.id, 0]));
      const outgoing = new Map(this.data.nodes.map(node => [node.id, []]));
      this.data.edges.forEach(edge => {
        if (!outgoing.has(edge.source) || !incoming.has(edge.target)) throw new Error(`Unknown graph edge: ${edge.id}`);
        outgoing.get(edge.source).push(edge);
        incoming.set(edge.target, incoming.get(edge.target) + 1);
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
      this.data.edges.forEach(edge => {
        incoming.set(edge.target, (incoming.get(edge.target) || 0) + 1);
        outgoing.get(edge.source)?.push(edge);
      });
      const preferred = ['locate', 'promote', 'continue', 'support', 'diagnose'];
      let current = this.data.nodes.find(node => !incoming.get(node.id) && node.priority === 'anchor')
        || this.data.nodes.find(node => !incoming.get(node.id));
      const path = [];
      const seen = new Set();
      while (current && !seen.has(current.id)) {
        path.push(current.id);
        seen.add(current.id);
        const next = (outgoing.get(current.id) || [])
          .slice()
          .sort((a, b) => preferred.indexOf(a.relation) - preferred.indexOf(b.relation))
          .find(edge => preferred.includes(edge.relation));
        current = next ? this.nodesById.get(next.target) : null;
      }
      return path;
    }

    layout() {
      const sizes = this.measure();
      const ranks = this.ranks();
      this.rankOf = ranks;
      const layers = new Map();
      this.data.nodes.forEach(node => {
        const rank = ranks.get(node.id) ?? 0;
        if (!layers.has(rank)) layers.set(rank, []);
        layers.get(rank).push(node);
      });

      let x = 72;
      const xByRank = new Map();
      [...layers.entries()].sort(([a], [b]) => a - b).forEach(([rank, nodes]) => {
        xByRank.set(rank, x);
        x += Math.max(...nodes.map(node => Math.max(sizes.get(node.id).occupiedWidth, sizes.get(node.id).railWidth))) + 112;
      });

      const primary = new Set(this.primaryPath());
      const center = 420;
      const branchGap = 42;
      const groupGap = 58;
      this.positions.clear();

      const place = (node, y) => {
        const size = sizes.get(node.id);
        this.positions.set(node.id, {
          x: xByRank.get(ranks.get(node.id)),
          y,
          width: size.width,
          height: size.height,
          railWidth: size.railWidth,
          railHeight: size.railHeight,
          occupiedWidth: size.occupiedWidth,
          occupiedHeight: size.occupiedHeight
        });
      };

      layers.forEach(nodes => {
        const spineIndex = nodes.findIndex(node => primary.has(node.id));
        if (spineIndex < 0) {
          const total = nodes.reduce((sum, node) => sum + sizes.get(node.id).occupiedHeight, 0) + Math.max(0, nodes.length - 1) * branchGap;
          let y = center - total / 2;
          nodes.forEach(node => { place(node, y); y += sizes.get(node.id).occupiedHeight + branchGap; });
          return;
        }
        const spine = nodes[spineIndex];
        const spineSize = sizes.get(spine.id);
        const spineY = center - spineSize.height / 2;
        place(spine, spineY);
        let upper = spineY - groupGap;
        nodes.slice(0, spineIndex).reverse().forEach(node => {
          upper -= sizes.get(node.id).occupiedHeight;
          place(node, upper);
          upper -= branchGap;
        });
        let lower = spineY + spineSize.occupiedHeight + groupGap;
        nodes.slice(spineIndex + 1).forEach(node => {
          place(node, lower);
          lower += sizes.get(node.id).occupiedHeight + branchGap;
        });
      });

      const first = [...this.positions.values()].reduce(
        (box, pos) => ({ left: Math.min(box.left, pos.x), top: Math.min(box.top, pos.y) }),
        { left: Infinity, top: Infinity }
      );
      this.positions.forEach(pos => { pos.x += 72 - first.left; pos.y += 72 - first.top; });

      const bounds = [...this.positions.values()].reduce(
        (box, pos) => ({
          right: Math.max(box.right, pos.x + Math.max(pos.width, pos.railWidth)),
          bottom: Math.max(box.bottom, pos.y + pos.occupiedHeight)
        }),
        { right: 0, bottom: 0 }
      );

      /* Reserve a lower peripheral lane for cross-level (rank-skipping) edges so
         they route around the whole graph instead of through intermediate nodes. */
      const crossCount = this.data.edges.filter(edge => ranks.get(edge.target) - ranks.get(edge.source) >= 2).length;
      const worldWidth = bounds.right + 72;
      const worldHeight = Math.max(620, bounds.bottom + 72 + crossCount * 22);
      this.world.style.setProperty('--dg-world-width', `${worldWidth}px`);
      this.world.style.setProperty('--dg-world-height', `${worldHeight}px`);
      this.renderGeometry();
    }

    renderGeometry() {
      this.nodeElements.forEach((item, id) => {
        const pos = this.positions.get(id);
        item.style.transform = `translate(${pos.x}px, ${pos.y}px)`;
      });
      this.renderEdges();
      this.applyCamera();
    }

    renderEdges() {
      const defs = this.edgeSvg.querySelector('defs');
      this.edgeSvg.replaceChildren(defs);

      const outgoing = new Map();
      const incoming = new Map();
      this.data.edges.forEach(edge => {
        if (!outgoing.has(edge.source)) outgoing.set(edge.source, []);
        outgoing.get(edge.source).push(edge);
        if (!incoming.has(edge.target)) incoming.set(edge.target, []);
        incoming.get(edge.target).push(edge);
      });
      /* Distribute ports by the vertical order of the far node so a fan-out or
         fan-in does not collapse onto a single point. */
      outgoing.forEach(edges => edges.sort((a, b) => {
        const ap = this.positions.get(a.target);
        const bp = this.positions.get(b.target);
        return (ap.y + ap.height / 2) - (bp.y + bp.height / 2);
      }));
      incoming.forEach(edges => edges.sort((a, b) => {
        const ap = this.positions.get(a.source);
        const bp = this.positions.get(b.source);
        return (ap.y + ap.height / 2) - (bp.y + bp.height / 2);
      }));

      const bottomMax = Math.max(...[...this.positions.values()].map(pos => pos.y + pos.occupiedHeight));
      let peripheralLane = bottomMax + 40;

      const port = (index, count, height) => count === 1 ? height / 2 : 18 + index * ((height - 36) / Math.max(1, count - 1));

      this.data.edges.forEach(edge => {
        const from = this.positions.get(edge.source);
        const to = this.positions.get(edge.target);
        if (!from || !to) return;

        const outSiblings = outgoing.get(edge.source) || [edge];
        const outIndex = outSiblings.indexOf(edge);
        const inSiblings = incoming.get(edge.target) || [edge];
        const inIndex = inSiblings.indexOf(edge);

        const start = { x: from.x + from.width, y: from.y + port(outIndex, outSiblings.length, from.height) };
        const end = { x: to.x, y: to.y + port(inIndex, inSiblings.length, to.height) };

        const fromRank = this.rankOf.get(edge.source);
        const toRank = this.rankOf.get(edge.target);
        const skipsRanks = toRank - fromRank >= 2;

        let points;
        if (skipsRanks) {
          const laneY = peripheralLane;
          peripheralLane += 22;
          const exitX = start.x + 26 + outIndex * 12;
          const entryX = end.x - 26;
          points = [start, { x: exitX, y: start.y }, { x: exitX, y: laneY }, { x: entryX, y: laneY }, { x: entryX, y: end.y }, end];
        } else if (Math.abs(start.y - end.y) < 1) {
          points = [start, end];
        } else {
          const channel = Math.min(end.x - 24, start.x + 30 + outIndex * 12);
          points = [start, { x: channel, y: start.y }, { x: channel, y: end.y }, end];
        }

        const path = svg('path', { class: `dg-edge ${relationClass(edge.relation)}`, d: pathFor(points) });
        path.dataset.dgEdge = edge.id;
        path.append(svg('title'));
        path.querySelector('title').textContent = edge.label || edge.relation;
        this.edgeSvg.append(path);

        if (edge.label) {
          const group = svg('g', { class: 'dg-edge-label' });
          const width = Math.max(60, edge.label.length * 7 + 16);
          const horizontalSegments = points.slice(0, -1)
            .map((point, index) => ({ from: point, to: points[index + 1], length: Math.abs(points[index + 1].x - point.x) }))
            .filter(segment => segment.from.y === segment.to.y)
            .sort((a, b) => b.length - a.length);
          const segment = horizontalSegments[0] || { from: start, to: end };
          const labelX = (segment.from.x + segment.to.x) / 2;
          const labelY = segment.from.y - 15;
          group.setAttribute('transform', `translate(${labelX} ${labelY})`);
          group.append(svg('rect', { x: -width / 2, y: -10, width, height: 20, rx: 4 }), svg('text', { y: 4 }));
          group.lastChild.textContent = edge.label;
          this.edgeSvg.append(group);
        }
      });

      this.updateFocus();
    }

    applyCamera() {
      this.world.style.transform = `translate(${this.camera.x}px, ${this.camera.y}px) scale(${this.camera.k})`;
      this.zoomOutput.value = `${Math.round(this.camera.k * 100)}%`;
      this.zoomOutput.textContent = this.zoomOutput.value;
    }

    fit() {
      const width = parseFloat(this.world.style.getPropertyValue('--dg-world-width')) || 1600;
      const height = parseFloat(this.world.style.getPropertyValue('--dg-world-height')) || 900;
      const pad = 34;
      const k = Math.min((this.viewport.clientWidth - pad * 2) / width, (this.viewport.clientHeight - pad * 2) / height);
      this.camera.k = Math.max(.08, k);
      this.camera.x = (this.viewport.clientWidth - width * this.camera.k) / 2;
      this.camera.y = (this.viewport.clientHeight - height * this.camera.k) / 2;
      this.applyCamera();
    }

    setScale(next, point) {
      this.userCameraMoved = true;
      const old = this.camera.k;
      const k = clamp(next, .01, 3);
      const anchor = point || { x: this.viewport.clientWidth / 2, y: this.viewport.clientHeight / 2 };
      this.camera.x = anchor.x - (anchor.x - this.camera.x) * k / old;
      this.camera.y = anchor.y - (anchor.y - this.camera.y) * k / old;
      this.camera.k = k;
      this.closeNote(false);
      this.applyCamera();
    }

    semanticEdges() { return this.data.edges; }

    focusFor(id) {
      return new Set([id, ...this.semanticEdges()
        .filter(edge => edge.source === id || edge.target === id)
        .flatMap(edge => [edge.source, edge.target])]);
    }

    updateFocus() {
      const focused = this.selectedId ? this.focusFor(this.selectedId) : null;
      const edgesById = new Map(this.semanticEdges().map(edge => [edge.id, edge]));
      this.nodeElements.forEach((item, id) => {
        item.classList.toggle('is-selected', id === this.selectedId);
        item.classList.toggle('is-dimmed', Boolean(focused && !focused.has(id)));
      });
      this.evidenceElements.forEach((item, key) => {
        const nodeId = key.split(':')[0];
        item.classList.toggle('is-dimmed', Boolean(focused && !focused.has(nodeId)));
      });
      this.edgeSvg.querySelectorAll('.dg-edge').forEach(path => {
        const edge = edgesById.get(path.dataset.dgEdge);
        const related = !focused || (focused.has(edge.source) && focused.has(edge.target));
        path.classList.toggle('is-related', Boolean(focused && related));
        path.classList.toggle('is-dimmed', Boolean(focused && !related));
      });
      this.edgeSvg.querySelectorAll('.dg-edge-label').forEach(label => {
        const dimmed = Boolean(focused && !label.previousElementSibling?.classList.contains('is-related'));
        label.classList.toggle('is-dimmed', dimmed);
      });
    }

    select(id) {
      this.selectedId = this.selectedId === id ? null : id;
      this.updateFocus();
    }

    bind() {
      this.nodeLayer.addEventListener('keydown', event => {
        if ((event.key === 'Enter' || event.key === ' ') && event.target.matches('[data-dg-node]')) {
          event.preventDefault();
          const node = this.nodesById.get(event.target.dataset.dgNode);
          if (this.actions.evidenceMode === 'note' && node.evidenceRefs?.[0]) this.openEvidence(node.evidenceRefs[0], true, event.target);
          else this.select(node.id);
        }
      });
      this.toolbar.addEventListener('click', event => {
        const action = event.target.closest('[data-dg-action]')?.dataset.dgAction;
        if (!action) return;
        if (action === 'zoom-in') this.setScale(this.camera.k + .12);
        if (action === 'zoom-out') this.setScale(this.camera.k - .12);
        if (action === 'fit') { this.userCameraMoved = true; this.fit(); }
        if (action === 'hundred') { this.userCameraMoved = true; this.camera = { x: 42, y: 42, k: 1 }; this.applyCamera(); }
      });
      this.viewport.addEventListener('wheel', event => {
        if (event.target.closest('[data-dg-note]')) return;
        event.preventDefault();
        const rect = this.viewport.getBoundingClientRect();
        const point = { x: event.clientX - rect.left, y: event.clientY - rect.top };
        if (event.ctrlKey || event.metaKey) this.setScale(this.camera.k * (event.deltaY > 0 ? .9 : 1.1), point);
        else {
          this.userCameraMoved = true;
          this.camera.x -= event.deltaX;
          this.camera.y -= event.deltaY;
          this.closeNote(false);
          this.applyCamera();
        }
      }, { passive: false });
      this.viewport.addEventListener('pointerdown', event => {
        if (event.target.closest('[data-dg-node],[data-dg-evidence],[data-dg-note]')) return;
        this.pan(event);
      });
      this.viewport.addEventListener('click', event => {
        if (this.suppressNextClick) { this.suppressNextClick = false; return; }
        if (event.target.closest('[data-dg-evidence],[data-dg-note]')) return;
        const node = event.target.closest('[data-dg-node]');
        if (node) {
          const item = this.nodesById.get(node.dataset.dgNode);
          if (this.actions.evidenceMode === 'note' && item.evidenceRefs?.[0]) this.openEvidence(item.evidenceRefs[0], true, node);
          else this.select(item.id);
        } else {
          this.selectedId = null;
          this.closeNote(false);
          this.updateFocus();
        }
      });
      this.nodeLayer.addEventListener('click', event => {
        const scenario = event.target.closest('[data-dg-scenario]');
        if (scenario) { event.stopPropagation(); this.actions.openScenario?.(); return; }
        const thumb = event.target.closest('[data-dg-evidence]');
        if (!thumb) return;
        event.stopPropagation();
        this.toggleEvidence(thumb);
      });
      this.viewport.addEventListener('keydown', event => {
        if (event.key === 'Escape') {
          if (this.noteEvidenceId) this.closeNote(false);
          else if (this.selectedId) { this.selectedId = null; this.updateFocus(); }
        }
        if (event.key === ' ' && event.target === this.viewport) event.preventDefault();
      });
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(this.viewport);
    }

    pan(event) {
      const start = { x: event.clientX, y: event.clientY, cameraX: this.camera.x, cameraY: this.camera.y };
      this.viewport.dataset.dgPanning = 'true';
      this.viewport.setPointerCapture(event.pointerId);
      const move = moveEvent => {
        this.userCameraMoved = true;
        this.suppressNextClick = true;
        this.camera.x = start.cameraX + moveEvent.clientX - start.x;
        this.camera.y = start.cameraY + moveEvent.clientY - start.y;
        this.closeNote(false);
        this.applyCamera();
      };
      const end = endEvent => {
        this.viewport.dataset.dgPanning = 'false';
        this.viewport.removeEventListener('pointermove', move);
        this.viewport.removeEventListener('pointerup', end);
        this.viewport.removeEventListener('pointercancel', end);
        if (this.viewport.hasPointerCapture(endEvent.pointerId)) this.viewport.releasePointerCapture(endEvent.pointerId);
      };
      this.viewport.addEventListener('pointermove', move);
      this.viewport.addEventListener('pointerup', end);
      this.viewport.addEventListener('pointercancel', end);
    }

    resize() { if (!this.userCameraMoved) this.fit(); }

    getState() {
      return { camera: { ...this.camera }, selectedId: this.selectedId, expanded: [...this.expanded], noteEvidenceId: this.noteEvidenceId };
    }

    destroy() {
      cancelAnimationFrame(this.initialFrame);
      this.actions.saveState?.(this.getState());
      this.resizeObserver?.disconnect();
      this.resizeObserver = null;
      this.root.replaceChildren();
    }
  }

  global.PtoInvestigationMap = { Controller, create(root, data, actions) { return new Controller(root, data, actions); } };
  global.DiagnosticGraphController = Controller;
})(window);
