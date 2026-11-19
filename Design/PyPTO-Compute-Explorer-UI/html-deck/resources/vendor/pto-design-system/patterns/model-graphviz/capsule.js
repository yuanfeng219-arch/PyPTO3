(function registerPtoModelGraphvizCapsule(global) {
  'use strict';

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const capsuleScriptUrl = document.currentScript?.src || '';
  if (typeof global.PTO_PASS_IR_GRAPH_NODE_ASSET_PREFIX !== 'string' && capsuleScriptUrl) {
    global.PTO_PASS_IR_GRAPH_NODE_ASSET_PREFIX = new URL('../../', capsuleScriptUrl).href;
  }
  const SHARED_SELECTOR = '.pto-model-graphviz-node';
  const LEGACY_DEEPSEEK_SELECTOR = '.node_8738e552-e764-4a57-839a-6321bc3ae7d3';
  const LEGACY_DEEPSEEK_EDGE_SELECTOR = '.edge-group_8738e552-e764-4a57-839a-6321bc3ae7d3 > path.edge_8738e552-e764-4a57-839a-6321bc3ae7d3';
  const DEFAULT_SELECTOR = `${SHARED_SELECTOR}, ${LEGACY_DEEPSEEK_SELECTOR}`;
  const OPENPANGU_CAPSULE_FRAME = Object.freeze({
    minWidth: 210,
    minHeight: 46,
    legacyWidth: 365,
    legacyHeight: 46,
  });
  let observer = null;
  let scheduled = false;

  function directChild(group, selector) {
    return Array.from(group.children || []).find((child) => child.matches?.(selector)) || null;
  }

  function boundNode(group) {
    const value = group.__data__;
    if (Array.isArray(value)) return { id: value[0], ...(value[1] || {}) };
    return value && typeof value === 'object' ? value : {};
  }

  function textFrom(group, selectors) {
    for (const selector of selectors) {
      const value = group.querySelector(selector)?.textContent?.trim();
      if (value) return value;
    }
    return '';
  }

  function nodeInfo(group, graphNodes) {
    const bound = boundNode(group);
    const hintedId = group.dataset.nodeId || bound.id || '';
    const graphNode = graphNodes?.get(String(hintedId)) || {};
    const id = String(graphNode.id || hintedId || `capsule-${Math.random().toString(36).slice(2)}`);
    const label = String(
      graphNode.label ||
      textFrom(group, ['.pto-model-graphviz-node-label', '[class*="node-label_"]']) ||
      id
    );
    const legacyTypeLabel = Array.from(group.classList).some((name) => name.includes('module-node_'))
      ? 'Module'
      : Array.from(group.classList).some((name) => name.includes('op-node_')) ? 'Op' : '';
    const typeLabel = String(
      graphNode.typeLabel ||
      legacyTypeLabel ||
      textFrom(group, ['.pto-model-graphviz-node-type', '[class*="type-indicator_"]']) ||
      ''
    );
    return { ...bound, ...graphNode, id, label, typeLabel };
  }

  function isTensorLike(group, node) {
    const kind = String(node.kind || node.type || '').toLowerCase();
    return group.classList.contains('is-tensor') ||
      Array.from(group.classList).some((name) => name.includes('tensor-node_')) ||
      ['tensor', 'state', 'buffer', 'parameter', 'constant', 'input', 'output'].includes(kind);
  }

  function isGlyphLike(group, node) {
    return group.classList.contains('is-glyph') || Boolean(node.glyph);
  }

  function isCollapsible(group) {
    return Boolean(
      group.querySelector('.pto-model-graphviz-toggle') ||
      Array.from(group.classList).some((name) => name.includes('collapsed-module_'))
    );
  }

  function accentFor(rect) {
    const declared = rect.getAttribute('fill') || rect.style.fill;
    if (declared && declared !== 'none' && !declared.startsWith('url(')) return declared;
    const computed = global.getComputedStyle?.(rect).fill;
    if (computed && computed !== 'none' && !computed.startsWith('url(')) return computed;
    return '#3577F6';
  }

  function numericAttr(node, name, fallback) {
    const value = Number(node.getAttribute(name));
    return Number.isFinite(value) && value > 0 ? value : fallback;
  }

  function enhanceGroup(group, graphNodes) {
    if (!group || group.dataset.ptoCapsuleEnhanced === '1') return false;
    const rect = directChild(group, 'rect');
    if (!rect) return false;

    const node = nodeInfo(group, graphNodes);
    if (isTensorLike(group, node) || isGlyphLike(group, node)) return false;

    const helper = global.PtoPassIrGraphNodePattern;
    if (!helper?.buildNodeCardElement) return false;

    const sourceWidth = Number(node.width) || numericAttr(rect, 'width', 180);
    const sourceHeight = Number(node.height) || numericAttr(rect, 'height', 48);
    const sourceX = Number.isFinite(Number(rect.getAttribute('x'))) ? Number(rect.getAttribute('x')) : -sourceWidth / 2;
    const sourceY = Number.isFinite(Number(rect.getAttribute('y'))) ? Number(rect.getAttribute('y')) : -sourceHeight / 2;
    const isLegacyDeepSeek = group.matches(LEGACY_DEEPSEEK_SELECTOR);
    const width = isLegacyDeepSeek
      ? OPENPANGU_CAPSULE_FRAME.legacyWidth
      : Math.max(OPENPANGU_CAPSULE_FRAME.minWidth, sourceWidth);
    const height = isLegacyDeepSeek
      ? OPENPANGU_CAPSULE_FRAME.legacyHeight
      : Math.max(OPENPANGU_CAPSULE_FRAME.minHeight, sourceHeight);
    const x = sourceX + (sourceWidth - width) / 2;
    const y = sourceY + (sourceHeight - height) / 2;
    const accent = accentFor(rect);
    const card = helper.buildNodeCardElement({
      id: node.id,
      type: 'op',
      data: {
        semanticLabel: node.label,
        latency: node.typeLabel ? 'meta' : null,
      },
    }, {
      compact: true,
      accent,
    });

    const meta = card.querySelector('.op-pill-latency');
    if (meta) meta.textContent = node.typeLabel;
    card.setAttribute('aria-hidden', 'true');

    const host = document.createElement('div');
    host.className = 'opv-pass-ir-capsule-host';
    host.appendChild(card);

    const foreignObject = document.createElementNS(SVG_NS, 'foreignObject');
    foreignObject.setAttribute('class', 'opv-pass-ir-capsule-foreign-object');
    foreignObject.setAttribute('x', String(x));
    foreignObject.setAttribute('y', String(y));
    foreignObject.setAttribute('width', String(width));
    foreignObject.setAttribute('height', String(height));
    foreignObject.setAttribute('aria-hidden', 'true');
    foreignObject.appendChild(host);

    group.dataset.ptoCapsuleEnhanced = '1';
    group.dataset.ptoCapsuleWidth = String(width);
    if (!group.dataset.nodeId) group.dataset.nodeId = node.id;
    group.classList.add('opv-pass-ir-capsule-node');
    if (isCollapsible(group)) group.classList.add('opv-pass-ir-capsule-collapsible');
    rect.classList.add('opv-pass-ir-capsule-hit');
    if (isLegacyDeepSeek) {
      rect.setAttribute('x', String(x));
      rect.setAttribute('y', String(y));
      rect.setAttribute('width', String(width));
      rect.setAttribute('height', String(height));
    }
    group.appendChild(foreignObject);
    return true;
  }

  function legacyBezierPath(points) {
    const path = Array.isArray(points)
      ? points.map((point) => ({ x: Number(point?.x), y: Number(point?.y) }))
        .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
      : [];
    if (path.length < 4) return '';

    // Graphviz emits an explicit arrow tip before the spline control points.
    // The legacy renderer appended that tip with an `L`, leaving every edge
    // with a straight tail. Replace the last spline endpoint with the tip so
    // the complete visible edge remains a chain of cubic Bézier segments.
    const hasArrowTip = path.length >= 5 && (path.length - 2) % 3 === 0;
    const arrowTip = hasArrowTip ? path.at(-1) : null;
    const spline = hasArrowTip ? path.slice(0, -1) : path;
    if ((spline.length - 1) % 3 !== 0) return '';

    const start = spline[0];
    const end = arrowTip || spline.at(-1);
    if (spline.length === 4) {
      // The renderer already supplies semantic control points: vertical-flow
      // edges leave/enter through top-bottom anchors, while side inputs use
      // left-right anchors. Preserve those tangents instead of guessing the
      // direction from endpoint distance (wide branches are still vertical).
      return `M${start.x},${start.y} C${spline[1].x},${spline[1].y} ${spline[2].x},${spline[2].y} ${end.x},${end.y}`;
    }

    const commands = [`M${start.x},${start.y}`];
    for (let index = 1; index + 2 < spline.length; index += 3) {
      const isLastSegment = index + 3 >= spline.length;
      const end = isLastSegment && arrowTip ? arrowTip : spline[index + 2];
      commands.push(
        `C${spline[index].x},${spline[index].y} ${spline[index + 1].x},${spline[index + 1].y} ${end.x},${end.y}`
      );
    }
    return commands.join(' ');
  }

  function normalizeLegacyEdges(root = document) {
    let count = 0;
    root.querySelectorAll(LEGACY_DEEPSEEK_EDGE_SELECTOR).forEach((edge) => {
      if (edge.dataset.ptoCapsuleBezier === '1') return;
      const d = legacyBezierPath(edge.__data__?.path);
      if (!d) return;
      edge.setAttribute('d', d);
      edge.dataset.ptoCapsuleBezier = '1';
      count += 1;
    });
    return count;
  }

  function apply(root = document, graph, options = {}) {
    const selector = options.selector || DEFAULT_SELECTOR;
    const graphNodes = new Map((graph?.nodes || []).map((node) => [String(node.id), node]));
    let count = 0;
    root.querySelectorAll(selector).forEach((group) => {
      if (enhanceGroup(group, graphNodes)) count += 1;
    });
    normalizeLegacyEdges(root);
    return count;
  }

  function scheduleApply() {
    if (scheduled) return;
    scheduled = true;
    global.requestAnimationFrame(() => {
      scheduled = false;
      apply(document);
    });
  }

  function observe(root = document) {
    if (observer) observer.disconnect();
    scheduleApply();
    observer = new MutationObserver(scheduleApply);
    observer.observe(root.body || root.documentElement || root, { childList: true, subtree: true });
    return observer;
  }

  function disconnect() {
    observer?.disconnect();
    observer = null;
  }

  global.PtoModelGraphvizCapsule = { apply, normalizeLegacyEdges, observe, disconnect };

  function autoStart() {
    observe(document);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', autoStart, { once: true });
  else autoStart();
})(window);
