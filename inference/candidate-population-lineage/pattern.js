(function registerPtoCandidatePopulationLineage(global) {
  'use strict';

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const MIN_WIDTH = 720;
  const MIN_HEIGHT = 164;
  const BASE_HEIGHT = 220;
  const DEFAULT_MAX_VISIBLE_CANDIDATES = 240;
  let instanceSeed = 0;

  function svgNode(tag, attributes = {}, text = '') {
    const node = document.createElementNS(SVG_NS, tag);
    Object.entries(attributes).forEach(([name, value]) => {
      if (value !== undefined && value !== null) node.setAttribute(name, String(value));
    });
    if (text !== '') node.textContent = text;
    return node;
  }

  function integerCount(value, field) {
    const count = Number(value);
    if (!Number.isInteger(count) || count < 0) {
      throw new Error(`candidate-population-lineage: ${field} must be a non-negative integer`);
    }
    return count;
  }

  function normalizeItems(items, field, fallbackCount) {
    if (!Array.isArray(items) || items.length === 0) {
      return fallbackCount > 0 ? [{ label: field === 'branches' ? 'Expanded' : 'Rejected', count: fallbackCount, tone: 1 }] : [];
    }
    return items.map((item, index) => {
      const source = Array.isArray(item) ? { label: item[0], count: item[1] } : item;
      if (!source || typeof source !== 'object') {
        throw new Error(`candidate-population-lineage: ${field}[${index}] must be an object or [label, count] tuple`);
      }
      return {
        ...source,
        label: String(source.label ?? `${field === 'branches' ? 'Branch' : 'Rejected'} ${index + 1}`),
        count: integerCount(source.count, `${field}[${index}].count`),
        tone: Math.max(1, Math.min(6, Number(source.tone) || index + 1))
      };
    });
  }

  function normalizeStage(stage, index) {
    if (!stage || typeof stage !== 'object') {
      throw new Error(`candidate-population-lineage: stages[${index}] must be an object`);
    }
    const id = String(stage.id ?? `stage-${index + 1}`);
    const count = integerCount(stage.count, `stages[${index}].count`);
    const rawSegments = Array.isArray(stage.segments) ? stage.segments : [];
    const segments = rawSegments.map((segment, segmentIndex) => {
      const source = typeof segment === 'number' ? { count: segment } : segment;
      return {
        ...source,
        count: integerCount(source?.count, `stages[${index}].segments[${segmentIndex}].count`),
        tone: Math.max(1, Math.min(3, Number(source?.tone) || segmentIndex + 1))
      };
    });
    if (segments.length && segments.reduce((sum, segment) => sum + segment.count, 0) !== count) {
      throw new Error(`candidate-population-lineage: stage "${id}" segments must sum to ${count}`);
    }
    return {
      ...stage,
      id,
      label: String(stage.label ?? id),
      kind: String(stage.kind ?? 'candidate'),
      count,
      summary: String(stage.summary ?? ''),
      segments,
      candidateLabels: Array.isArray(stage.candidateLabels) ? stage.candidateLabels.map(String) : [],
      layout: { ...(stage.layout || {}) }
    };
  }

  function normalizeTransition(rawNext, source, target, index) {
    if (!rawNext || typeof rawNext !== 'object') {
      throw new Error(`candidate-population-lineage: stage "${source.id}" requires a next operation`);
    }
    const rawType = String(rawNext.type || '').toLowerCase();
    const type = rawType === 'expand' ? 'diverge' : rawType === 'filter' ? 'converge' : rawType;
    if (type !== 'diverge' && type !== 'converge') {
      throw new Error(`candidate-population-lineage: stage "${source.id}" next.type must be diverge or converge`);
    }
    const id = String(rawNext.id ?? `${source.id}-to-${target.id}`);
    const difference = Math.abs(target.count - source.count);

    if (type === 'diverge' && target.count < source.count) {
      throw new Error(`candidate-population-lineage: diverge "${id}" cannot reduce ${source.count} to ${target.count}`);
    }
    if (type === 'converge' && target.count > source.count) {
      throw new Error(`candidate-population-lineage: converge "${id}" cannot expand ${source.count} to ${target.count}`);
    }

    const branches = type === 'diverge'
      ? normalizeItems(rawNext.branches, 'branches', target.count)
      : [];
    const rejected = type === 'converge'
      ? normalizeItems(rawNext.rejected ?? rawNext.categories, 'rejected', difference)
      : [];
    const expectedTotal = type === 'diverge' ? target.count : difference;
    const actualTotal = (type === 'diverge' ? branches : rejected).reduce((sum, item) => sum + item.count, 0);
    if (actualTotal !== expectedTotal) {
      const subject = type === 'diverge' ? 'branch counts' : 'rejected counts';
      throw new Error(`candidate-population-lineage: ${subject} for "${id}" must sum to ${expectedTotal}, received ${actualTotal}`);
    }

    const defaultValue = type === 'diverge'
      ? `×${source.count === 0 ? '—' : Number((target.count / source.count).toFixed(1))}`
      : `−${difference}`;
    return {
      ...rawNext,
      id,
      type,
      label: String(rawNext.label ?? (type === 'diverge' ? `EXPAND ${defaultValue}` : `REJECT ${defaultValue}`)),
      summary: String(rawNext.summary ?? ''),
      from: source.count,
      to: target.count,
      sourceId: source.id,
      targetId: target.id,
      branches,
      rejected,
      valueLabel: String(rawNext.valueLabel ?? defaultValue),
      index
    };
  }

  function normalizeData(data) {
    if (!data || !Array.isArray(data.stages) || data.stages.length < 2) {
      throw new Error('candidate-population-lineage: data.stages must contain at least two stages');
    }
    const stages = data.stages.map(normalizeStage);
    const ids = new Set();
    stages.forEach((stage) => {
      if (ids.has(stage.id)) throw new Error(`candidate-population-lineage: duplicate stage id "${stage.id}"`);
      ids.add(stage.id);
    });
    const transitions = stages.slice(0, -1).map((stage, index) => {
      const transition = normalizeTransition(stage.next, stage, stages[index + 1], index);
      if (ids.has(transition.id)) throw new Error(`candidate-population-lineage: duplicate selectable id "${transition.id}"`);
      ids.add(transition.id);
      return transition;
    });
    if (stages[stages.length - 1].next) {
      throw new Error('candidate-population-lineage: the final stage must not define next');
    }
    return { ...data, stages, transitions };
  }

  function createRibbonPath({ startX, startTop, startBottom, endX, endTop, endBottom }) {
    const forwardControl = startX + (endX - startX) * .42;
    const backwardControl = startX + (endX - startX) * .72;
    return [
      `M ${startX} ${startTop}`,
      `C ${forwardControl} ${startTop}, ${backwardControl} ${endTop}, ${endX} ${endTop}`,
      `L ${endX} ${endBottom}`,
      `C ${backwardControl} ${endBottom}, ${forwardControl} ${startBottom}, ${startX} ${startBottom}`,
      'Z'
    ].join(' ');
  }

  function autoCloudLayout(stage, maxVisibleCandidates) {
    const visibleCount = Math.min(stage.count, maxVisibleCandidates);
    const preferredColumns = visibleCount <= 1 ? 1 : Math.ceil(Math.sqrt(visibleCount * .58));
    const columns = Math.max(1, Number(stage.layout.columns) || preferredColumns);
    const rows = Math.max(1, Number(stage.layout.rows) || Math.ceil(Math.max(1, visibleCount) / columns));
    const width = stage.layout.width ?? (columns === 1 ? 0 : Math.min(62, (columns - 1) * 6));
    const height = stage.layout.height ?? Math.max(18, Math.min(88, (rows - 1) * 5));
    return { visibleCount, columns, rows, width: Number(width), baseHeight: Number(height) };
  }

  function layoutChart(svg, stages, maxVisibleCandidates) {
    const width = Math.max(MIN_WIDTH, Math.round(svg.clientWidth || MIN_WIDTH));
    const height = Math.max(MIN_HEIGHT, Math.round(svg.clientHeight || MIN_HEIGHT));
    const horizontalInset = Math.max(48, Math.min(64, width * .05));
    const step = (width - horizontalInset * 2) / Math.max(1, stages.length - 1);
    const verticalScale = height / BASE_HEIGHT;
    const centerY = 96 * verticalScale;
    const laidOutStages = stages.map((stage, index) => {
      const cloud = autoCloudLayout(stage, maxVisibleCandidates);
      return {
        ...stage,
        ...cloud,
        x: horizontalInset + step * index,
        height: cloud.baseHeight * verticalScale
      };
    });
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    return { width, height, verticalScale, centerY, stages: laidOutStages };
  }

  function stageBounds(stage, centerY) {
    return {
      left: stage.x - stage.width / 2,
      right: stage.x + stage.width / 2,
      top: centerY - stage.height / 2,
      bottom: centerY + stage.height / 2
    };
  }

  function appendStageLabel(group, { centerX, dotY, label }) {
    const dotGap = 6;
    const dotRadius = 2.6;
    const horizontalPadding = 12;
    const plateHeight = 20;
    const minPlateWidth = 72;
    const shell = svgNode('g', { class: 'stage-label-shell' });
    const content = svgNode('g', { class: 'stage-label-content' });
    const plate = svgNode('rect', { class: 'stage-label-plate', y: dotY - plateHeight / 2, height: plateHeight, rx: 10 });
    const dot = svgNode('circle', { class: 'stage-state-dot', cy: dotY, r: dotRadius });
    const text = svgNode('text', { class: 'stage-label', y: 0, 'text-anchor': 'middle' }, label);
    content.append(dot, text);
    shell.append(plate, content);
    group.append(shell);

    let labelWidth = Math.max(1, label.length * 5.8);
    let verticalOffset = 3;
    try {
      labelWidth = text.getComputedTextLength() || labelWidth;
      const bounds = text.getBBox();
      verticalOffset = -(bounds.y + bounds.height / 2);
    } catch (error) {
      // Detached or initially hidden SVGs use the stable text-size fallback above.
    }
    const decoratedHalfWidth = labelWidth / 2 + dotGap + dotRadius * 2;
    const plateWidth = Math.max(minPlateWidth, (decoratedHalfWidth + horizontalPadding) * 2);
    plate.setAttribute('x', centerX - plateWidth / 2);
    plate.setAttribute('width', plateWidth);
    dot.setAttribute('cx', centerX - labelWidth / 2 - dotGap - dotRadius);
    text.setAttribute('x', centerX);
    text.setAttribute('y', dotY + verticalOffset);
  }

  function dotTone(stage, visibleIndex) {
    if (!stage.segments.length || stage.visibleCount === 0) return '';
    const logicalIndex = (visibleIndex + .5) * stage.count / stage.visibleCount;
    let cursor = 0;
    const segmentIndex = stage.segments.findIndex((segment) => {
      cursor += segment.count;
      return logicalIndex <= cursor;
    });
    return segmentIndex >= 0 ? ` tone-${stage.segments[segmentIndex].tone}` : '';
  }

  function renderStage(svg, stage, stageIndex, chart, selectionLookup) {
    const bounds = stageBounds(stage, chart.centerY);
    const isFinal = stageIndex === chart.stages.length - 1;
    const group = svgNode('g', {
      class: 'lineage-stage-group',
      role: 'button',
      tabindex: '0',
      'aria-label': `查看 ${stage.label} 阶段：${stage.count} 个${stage.kind}`,
      'aria-pressed': 'false',
      'data-candidate-lineage-select': stage.id,
      'data-lineage-stage': stage.id,
      'data-candidate-count': stage.count
    });
    selectionLookup.set(stage.id, { id: stage.id, kind: 'stage', item: stage, stageIndex });

    group.append(svgNode('rect', {
      class: 'stage-plane',
      x: bounds.left - 10,
      y: bounds.top - 12,
      width: Math.max(40, stage.width + 20),
      height: stage.height + 24,
      rx: 12
    }));
    appendStageLabel(group, { centerX: stage.x, dotY: 28 * chart.verticalScale, label: stage.label });
    group.append(svgNode('text', { class: 'stage-count', x: stage.x, y: bounds.top - 16, 'text-anchor': 'middle' }, stage.count));

    for (let index = 0; index < stage.visibleCount; index += 1) {
      const column = index % stage.columns;
      const row = Math.floor(index / stage.columns);
      const x = stage.columns === 1 ? stage.x : bounds.left + (column / (stage.columns - 1)) * stage.width;
      const y = stage.rows === 1 ? chart.centerY : bounds.top + (row / (stage.rows - 1)) * stage.height;
      group.append(svgNode('circle', {
        class: `candidate-dot${dotTone(stage, index)}${isFinal ? ' is-final' : ''}`,
        cx: x,
        cy: y,
        r: stageIndex < 2 ? 1.35 : stageIndex < 4 ? 1.65 : 2.05
      }));
      if (stage.visibleCount === stage.count && stage.candidateLabels[index]) {
        group.append(svgNode('text', { class: 'candidate-name', x: x + 10, y: y + 3 }, stage.candidateLabels[index]));
      }
    }
    if (stage.count > stage.visibleCount) {
      group.append(svgNode('text', {
        class: 'lineage-aggregate-label',
        x: stage.x,
        y: bounds.bottom + 22,
        'text-anchor': 'middle'
      }, `${stage.visibleCount} dots · ${stage.count} candidates`));
    }
    group.append(svgNode('title', {}, `${stage.label}: ${stage.count} ${stage.kind}`));
    svg.append(group);
  }

  function renderDivergence(group, transition, source, target, chart) {
    const startX = source.right + 9;
    const endX = target.left - 9;
    const sourceHeight = source.bottom - source.top;
    const targetHeight = target.bottom - target.top;
    const labelStartY = Math.max(154 * chart.verticalScale, target.bottom + 22);
    let sourceCursor = source.top;
    let targetCursor = target.top;
    transition.branches.forEach((branch, branchIndex) => {
      const sourceThickness = sourceHeight * (branch.count / Math.max(1, transition.to));
      const targetThickness = targetHeight * (branch.count / Math.max(1, transition.to));
      const sourceTop = sourceCursor;
      const sourceBottom = sourceTop + sourceThickness;
      const targetTop = targetCursor;
      const targetBottom = targetTop + targetThickness;
      group.append(svgNode('path', {
        class: `lineage-ribbon tone-${branch.tone}`,
        d: createRibbonPath({ startX, startTop: sourceTop, startBottom: sourceBottom, endX, endTop: targetTop, endBottom: targetBottom }),
        'data-flow-count': branch.count
      }));
      group.append(svgNode('text', {
        class: 'lineage-branch-label',
        x: target.left,
        y: labelStartY + branchIndex * 11,
        'text-anchor': 'start'
      }, branch.label));
      sourceCursor = sourceBottom;
      targetCursor = targetBottom;
    });
  }

  function renderConvergence(group, transition, source, target, chart, gradientId, isLastTransition) {
    const startX = source.right + 9;
    const endX = target.left - 9;
    const sourceHeight = source.bottom - source.top;
    const passThickness = transition.from === 0 ? 0 : sourceHeight * (transition.to / transition.from);
    const passTop = source.top;
    const passBottom = passTop + passThickness;
    if (transition.to > 0) {
      group.append(svgNode('path', {
        class: 'lineage-ribbon is-pass',
        fill: `url(#${gradientId})`,
        d: createRibbonPath({ startX, startTop: passTop, startBottom: passBottom, endX, endTop: target.top, endBottom: target.bottom }),
        'data-flow-count': transition.to
      }));
    }

    let sourceCursor = passBottom;
    const endpointX = endX - (isLastTransition ? 32 : 4);
    transition.rejected.forEach((rejection, categoryIndex) => {
      const sourceThickness = sourceHeight * (rejection.count / Math.max(1, transition.from));
      const endpointThickness = Math.max(3, sourceThickness * .34);
      const endpointTop = 154 * chart.verticalScale + categoryIndex * 11;
      const endpointBottom = endpointTop + endpointThickness;
      group.append(svgNode('path', {
        class: `lineage-ribbon is-reject tone-${rejection.tone}`,
        d: createRibbonPath({
          startX,
          startTop: sourceCursor,
          startBottom: sourceCursor + sourceThickness,
          endX: endpointX,
          endTop: endpointTop,
          endBottom: endpointBottom
        }),
        'data-flow-count': rejection.count,
        'data-rejection-category': rejection.label
      }));
      group.append(svgNode('text', {
        class: 'lineage-reject-label',
        x: endpointX + 7,
        y: endpointTop + endpointThickness / 2 + 3,
        'text-anchor': 'start'
      }, rejection.label));
      sourceCursor += sourceThickness;
    });
  }

  function renderTransition(svg, transition, transitionIndex, chart, gradientId, selectionLookup) {
    const sourceStage = chart.stages[transitionIndex];
    const targetStage = chart.stages[transitionIndex + 1];
    const source = stageBounds(sourceStage, chart.centerY);
    const target = stageBounds(targetStage, chart.centerY);
    const midpoint = (sourceStage.x + targetStage.x) / 2;
    const stateClass = transition.type === 'diverge' ? 'is-diverge' : 'is-converge';
    const group = svgNode('g', {
      class: 'lineage-edge-group',
      role: 'button',
      tabindex: '0',
      'aria-label': `查看 ${transition.label}：${transition.from} → ${transition.to}`,
      'aria-pressed': 'false',
      'data-candidate-lineage-select': transition.id,
      'data-lineage-edge': transition.id,
      'data-input-count': transition.from,
      'data-output-count': transition.to
    });
    selectionLookup.set(transition.id, {
      id: transition.id,
      kind: 'transition',
      item: transition,
      transitionIndex,
      sourceStage: chart.stages[transitionIndex],
      targetStage: chart.stages[transitionIndex + 1]
    });

    if (transition.type === 'diverge') {
      renderDivergence(group, transition, source, target, chart);
    } else {
      renderConvergence(group, transition, source, target, chart, gradientId, transitionIndex === chart.stages.length - 2);
    }
    group.append(svgNode('circle', { class: `lineage-transition-orb ${stateClass}`, cx: midpoint, cy: chart.centerY, r: 14 }));
    group.append(svgNode('text', {
      class: `lineage-transition-value ${stateClass}`,
      x: midpoint,
      y: chart.centerY + 4,
      'text-anchor': 'middle'
    }, transition.valueLabel));
    group.append(svgNode('title', {}, `${transition.label}: ${transition.from} → ${transition.to}`));
    svg.append(group);
  }

  function render(container, options = {}) {
    if (!container) throw new Error('candidate-population-lineage: a container is required');
    let data = normalizeData(options.data);
    let selectedId = options.initialSelection ?? null;
    let destroyed = false;
    const maxVisibleCandidates = Math.max(24, Number(options.maxVisibleCandidates) || DEFAULT_MAX_VISIBLE_CANDIDATES);
    const selectionLookup = new Map();
    const instanceId = ++instanceSeed;
    const gradientId = `pto-candidate-lineage-pass-${instanceId}`;

    const root = document.createElement('div');
    root.className = 'pto-candidate-lineage';
    const scroll = document.createElement('div');
    scroll.className = 'pto-candidate-lineage__scroll';
    const svg = svgNode('svg', {
      class: 'pto-candidate-lineage__svg',
      role: 'img',
      'aria-label': data.title || 'Candidate population lineage'
    });
    scroll.append(svg);
    root.append(scroll);
    container.replaceChildren(root);

    function applySelection() {
      svg.querySelectorAll('[data-candidate-lineage-select]').forEach((control) => {
        const selected = control.getAttribute('data-candidate-lineage-select') === selectedId;
        control.classList.toggle('is-selected', selected);
        control.setAttribute('aria-pressed', String(selected));
      });
    }

    function draw() {
      if (destroyed) return;
      selectionLookup.clear();
      const chart = layoutChart(svg, data.stages, maxVisibleCandidates);
      svg.replaceChildren();
      svg.append(svgNode('title', {}, data.title || 'Candidate lineage map: expansion, rejection and convergence'));
      const defs = svgNode('defs');
      const passGradient = svgNode('linearGradient', { id: gradientId, x1: '0%', y1: '0%', x2: '100%', y2: '0%' });
      passGradient.append(
        svgNode('stop', { offset: '0%', 'stop-color': 'var(--foreground)', 'stop-opacity': '.12' }),
        svgNode('stop', { offset: '60%', 'stop-color': 'var(--primary-hover)', 'stop-opacity': '.22' }),
        svgNode('stop', { offset: '100%', 'stop-color': 'var(--success)', 'stop-opacity': '.28' })
      );
      defs.append(passGradient);
      svg.append(defs);
      [46 * chart.verticalScale, chart.centerY, 152 * chart.verticalScale].forEach((y) => {
        svg.append(svgNode('line', { class: 'chart-guide', x1: 26, y1: y, x2: chart.width - 26, y2: y }));
      });
      data.transitions.forEach((transition, index) => renderTransition(svg, transition, index, chart, gradientId, selectionLookup));
      chart.stages.forEach((stage, index) => renderStage(svg, stage, index, chart, selectionLookup));
      if (selectedId && !selectionLookup.has(selectedId)) selectedId = null;
      applySelection();
    }

    function selectionPayload(id) {
      const payload = selectionLookup.get(id);
      return payload ? { ...payload, data } : null;
    }

    function select(id, selectOptions = {}) {
      if (id === null || id === undefined) return clearSelection(selectOptions);
      const key = String(id);
      if (!selectionLookup.has(key)) throw new Error(`candidate-population-lineage: unknown selection "${key}"`);
      selectedId = key;
      applySelection();
      const payload = selectionPayload(key);
      if (selectOptions.emit !== false) {
        options.onSelect?.(payload);
        root.dispatchEvent(new CustomEvent('candidate-lineage-select', { detail: payload }));
      }
      return payload;
    }

    function clearSelection(clearOptions = {}) {
      selectedId = null;
      applySelection();
      if (clearOptions.emit !== false) {
        options.onSelect?.(null);
        root.dispatchEvent(new CustomEvent('candidate-lineage-select', { detail: null }));
      }
      return null;
    }

    function activate(event) {
      const control = event.target instanceof Element
        ? event.target.closest('[data-candidate-lineage-select]')
        : null;
      if (!control || !svg.contains(control)) return;
      select(control.getAttribute('data-candidate-lineage-select'));
    }

    function keyActivate(event) {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      const control = event.target instanceof Element
        ? event.target.closest('[data-candidate-lineage-select]')
        : null;
      if (!control || !svg.contains(control)) return;
      event.preventDefault();
      select(control.getAttribute('data-candidate-lineage-select'));
    }

    svg.addEventListener('click', activate);
    svg.addEventListener('keydown', keyActivate);
    const resizeObserver = typeof ResizeObserver === 'function' ? new ResizeObserver(draw) : null;
    resizeObserver?.observe(root);
    draw();

    return {
      root,
      setData(nextData) {
        data = normalizeData(nextData);
        draw();
        return data;
      },
      select,
      clearSelection,
      resize: draw,
      getSelection() { return selectionPayload(selectedId); },
      getData() { return data; },
      destroy() {
        if (destroyed) return;
        destroyed = true;
        resizeObserver?.disconnect();
        svg.removeEventListener('click', activate);
        svg.removeEventListener('keydown', keyActivate);
        root.remove();
        selectionLookup.clear();
      }
    };
  }

  global.PtoCandidatePopulationLineage = Object.freeze({ render, normalizeData });
})(window);
