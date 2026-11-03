(function(){'use strict';
const pypaEscape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function applyPypaExecution(scene, nodes, graph, data) {
    scene.querySelectorAll('[data-pypa-execution-count], [data-pypa-execution-stack]').forEach((decoration) => decoration.remove());
    scene.querySelectorAll('[data-pypa-node]').forEach((element) => {
      const entry = data?.nodes[element.dataset.pypaNode];
      const node = nodes.find((item) => item.id === element.dataset.pypaNode);
      element.classList.remove('has-execution-badge', 'has-execution-title');
      if (node) element.setAttribute('aria-label', `${node.badge}，${node.title}，${node.sub}${entry ? `，整个 Task ${data.task}：${entry.label}` : ''}`);
      if (!entry || !node || !window.PtoPagedAttentionExecution.decoration(node,data)) return;
      element.classList.add('has-execution-badge');
      const badge = document.createElement('span');
      badge.dataset.pypaExecutionCount = node.id;
      badge.className = 'kf-pypa-graph__execution-count';
      badge.textContent = `×${entry.count}`;
      badge.title = `${entry.unit} · ${entry.formula || entry.label} · 查看逻辑实例`;
      badge.setAttribute('role','button'); badge.setAttribute('tabindex','0');
      badge.setAttribute('aria-label',`${node.title} ×${entry.count}，查看逻辑实例`);
      element.appendChild(badge);
      if (entry.count <= 1) return;
      const stack = document.createElement('div');
      stack.className = 'kf-pypa-graph__execution-stack';
      stack.dataset.pypaExecutionStack = node.id;
      stack.setAttribute('role','button');stack.setAttribute('tabindex','0');
      stack.setAttribute('aria-label',`${node.title}，${entry.count} 个静态逻辑实例`);
      stack.style.cssText = `left:${node.x}px;top:${node.y}px;width:${node.w}px;height:${node.h}px`;
      // Read the rendered node so composite and primitive silhouettes stay identical.
      const nodeShape = getComputedStyle(element);
      for (let instance = entry.count; instance >= 2; instance--) {
        const layer = document.createElement('span');
        layer.className = 'kf-pypa-graph__execution-layer';
        for (const corner of ['borderTopLeftRadius','borderTopRightRadius','borderBottomRightRadius','borderBottomLeftRadius']) {
          layer.style[corner] = nodeShape[corner];
        }
        layer.style.clipPath = nodeShape.clipPath;
        layer.dataset.pypaExecutionInstance = String(instance);
        const offset=window.PtoPagedAttentionExecution.deckOffset(instance);
        layer.style.setProperty('--execution-offset', `${offset.right}px`);
        layer.style.setProperty('--execution-offset-y', `${offset.bottom}px`);
        stack.appendChild(layer);
      }
      stack.classList.toggle('is-journey-path', element.classList.contains('is-journey-path'));
      stack.classList.toggle('is-selected', element.classList.contains('is-selected'));
      stack.classList.toggle('is-context-muted', element.classList.contains('is-context-muted'));
      scene.insertBefore(stack, element);
    });
    const executionForEdge = (edge) => {
      const entries = (edge.sourceEdgeIds || [edge.id]).map((id) => data?.edges[id]).filter(Boolean);
      if (!entries.length) return null;
      return { width: Math.max(...entries.map((entry) => entry.width)),
        label: [...new Set(entries.map((entry) => entry.label))].join(' → ') };
    };
    scene.querySelectorAll('[data-pypa-edge]').forEach((path) => {
      const edge = graph.edges.find((item) => item.id === path.dataset.pypaEdge);
      const entry = executionForEdge(edge);
      if (entry) path.style.strokeWidth = String(entry.width);
      else path.style.removeProperty('stroke-width');
    });
    scene.querySelectorAll('[data-pypa-edge-label]').forEach((label) => {
      const edge = graph.edges.find((item) => item.id === label.dataset.pypaEdgeLabel);
      const entry = executionForEdge(edge);
      label.textContent = edge.label;
      if (entry) {
        const count = document.createElementNS('http://www.w3.org/2000/svg', 'tspan');
        count.setAttribute('x', label.getAttribute('x'));
        count.setAttribute('dy', '1.3em');
        count.textContent = entry.label;
        label.appendChild(count);
      }
    });

  }
  function renderPypaGraphScene(scene, graph, options = {}) {
    const { view = 'compute', selectedNodeId = null, expandedStages = new Set(),
      modelGraphviz = false, compactNodes = false, markerPrefix = 'pypa', cardHtml = pypaComputeCardHtml } = options;
    const placedNodes = graph.nodes.map((node) => ({
      ...node, expandable: options.allowExpand === false ? false : node.expandable,
      y: view === 'overview' ? node.y + (node.y >= 600 ? 70 : 0) + (node.y >= 760 ? 60 : 0) : node.y,
    }));
    const syncBridges = view === 'compute' && options.synchronizations !== false ? window.PTO_COMPUTATION_LAYOUT.synchronizationBridges(graph) : [];
    const sceneWidth = Math.max(graph.width || options.fallbackWidth || 2260, ...syncBridges.map((bridge) => bridge.maxX + 32));
    const sceneHeight = graph.height || options.fallbackHeight || 1480;
    scene.style.width = `${sceneWidth}px`;
    scene.style.height = `${sceneHeight}px`;
    const byId = new Map(placedNodes.map((node) => [node.id, node]));
    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.classList.add('kf-pypa-graph__edges');
    svg.setAttribute('viewBox', `0 0 ${sceneWidth} ${sceneHeight}`);
    svg.style.width = `${sceneWidth}px`;
    svg.style.height = `${sceneHeight}px`;
    svg.setAttribute('aria-hidden', 'true');
    const defs = document.createElementNS(svgNS, 'defs');
    /* 数据依赖用开口箭头（compiler-objects 的关系约定），没有闭合三角的实心填充；
       其余关系沿用闭合箭头以保持既有读法。 */
    const openArrow = modelGraphviz;
    ['dep', 'data', 'sync', 'loop', 'state', 'detail', 'computation', 'control', 'feedback'].forEach((type) => {
      const marker = document.createElementNS(svgNS, 'marker');
      marker.setAttribute('id', `${markerPrefix}-arrow-${type}`);
      marker.setAttribute('viewBox', '0 0 10 10');
      marker.setAttribute('refX', openArrow ? '8' : '9'); marker.setAttribute('refY', '5');
      marker.setAttribute('markerWidth', '7'); marker.setAttribute('markerHeight', '7');
      if (!openArrow && type === 'data') marker.setAttribute('markerUnits', 'userSpaceOnUse');
      marker.setAttribute('orient', 'auto-start-reverse');
      const arrow = document.createElementNS(svgNS, 'path');
      if (openArrow) {
        arrow.setAttribute('d', 'M 2 1.5 L 8 5 L 2 8.5');
        arrow.setAttribute('class', `kf-pypa-graph__arrow is-open is-${type}`);
      } else {
        arrow.setAttribute('d', 'M 0 0 L 10 5 L 0 10 z');
        arrow.setAttribute('class', `kf-pypa-graph__arrow is-${type}`);
      }
      marker.appendChild(arrow); defs.appendChild(marker);
    });
    svg.appendChild(defs);
    graph.edges.forEach((edge, index) => {
      const source = byId.get(edge.s);
      const target = byId.get(edge.t);
      if (!source || !target) return;
      const routed = view === 'compute' ? graph.edgeRoutes?.[index] : null;
      if (routed) {
        const line = document.createElementNS(svgNS, 'path');
        line.setAttribute('d', routed.path);
        line.dataset.pypaEdge = edge.id; line.dataset.edgeType = edge.type;
        line.dataset.edgeSource = edge.s; line.dataset.edgeTarget = edge.t;
        line.setAttribute('class', `kf-pypa-graph__edge is-${edge.type}`);
        line.setAttribute('marker-end', `url(#${markerPrefix}-arrow-${edge.type})`);
        svg.appendChild(line);
        if (edge.type === 'state' || edge.type === 'feedback') {
          const hit = document.createElementNS(svgNS, 'path');
          hit.setAttribute('d', routed.path);
          hit.setAttribute('class', 'kf-pypa-graph__edge-hit');
          hit.dataset.pypaEdgeHit = edge.id;
          svg.appendChild(hit);
        }
        if (edge.label) {
          const label = document.createElementNS(svgNS, 'text');
          label.setAttribute('class', 'kf-pypa-graph__edge-label');
          label.dataset.pypaEdgeLabel = edge.id; label.dataset.edgeType = edge.type;
          label.textContent = edge.label;
          if (modelGraphviz) {
            /* 标签按仓库 edge-tag 的规格落成「底板 + 文本」：不透明底板压在线段上，
               宽度由布局侧的 labelW 给出，文本 12px / 800，颜色取 foreground-secondary。 */
            const width = routed.labelW || Math.max(56, edge.label.length * 7.2 + 12);
            const height = routed.labelH || 18;
            const anchor = routed.labelAnchor || 'middle';
            const left = anchor === 'middle' ? routed.labelX - width / 2
              : anchor === 'end' ? routed.labelX - width : routed.labelX;
            const tag = document.createElementNS(svgNS, 'g');
            tag.setAttribute('class', `kf-pypa-graph__edge-tag is-${edge.type}`);
            const plate = document.createElementNS(svgNS, 'rect');
            plate.setAttribute('x', String(left));
            plate.setAttribute('y', String(routed.labelY - height / 2));
            plate.setAttribute('width', String(width));
            plate.setAttribute('height', String(height));
            plate.setAttribute('rx', '4'); plate.setAttribute('ry', '4');
            label.setAttribute('x', String(left + width / 2));
            label.setAttribute('y', String(routed.labelY));
            label.setAttribute('text-anchor', 'middle');
            label.setAttribute('dominant-baseline', 'middle');
            tag.append(plate, label);
            svg.appendChild(tag);
          } else {
            label.setAttribute('x', String(routed.labelX));
            label.setAttribute('y', String(routed.labelY));
            label.setAttribute('text-anchor', routed.labelAnchor || 'middle');
            svg.appendChild(label);
          }
        }
        return;
      }
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const vertical = Math.abs(dy) > Math.abs(dx) * 0.95;
      let sx, sy, tx, ty, path;
      if (edge.route === 'state') {
        sx = source.x + source.w / 2; sy = source.y + source.h;
        tx = target.x + target.w / 2; ty = target.y + target.h;
        const routeY = Math.max(source.y + source.h, target.y + target.h) + 115;
        path = `M ${sx} ${sy} C ${sx} ${routeY}, ${tx} ${routeY}, ${tx} ${ty}`;
      } else if (edge.route === 'feedback') {
        sx = source.x + source.w; sy = source.y + source.h / 2;
        tx = target.x + target.w; ty = target.y + target.h / 2;
        const routeX = Math.max(sx, tx) + 32;
        path = `M ${sx} ${sy} C ${routeX} ${sy}, ${routeX} ${ty}, ${tx} ${ty}`;
      } else if (vertical) {
        const down = dy > 0;
        sx = source.x + source.w / 2; sy = source.y + (down ? source.h : 0);
        tx = target.x + target.w / 2; ty = target.y + (down ? 0 : target.h);
        const mid = (sy + ty) / 2;
        path = `M ${sx} ${sy} C ${sx} ${mid}, ${tx} ${mid}, ${tx} ${ty}`;
      } else {
        const right = dx > 0;
        sx = source.x + (right ? source.w : 0); sy = source.y + source.h / 2;
        tx = target.x + (right ? 0 : target.w); ty = target.y + target.h / 2;
        const mid = (sx + tx) / 2;
        path = `M ${sx} ${sy} C ${mid} ${sy}, ${mid} ${ty}, ${tx} ${ty}`;
      }
      const line = document.createElementNS(svgNS, 'path');
      line.setAttribute('d', path);
      line.dataset.pypaEdge = edge.id; line.dataset.edgeType = edge.type;
      line.dataset.edgeSource = edge.s; line.dataset.edgeTarget = edge.t;
      line.setAttribute('class', `kf-pypa-graph__edge is-${edge.route || edge.type || 'dep'}`);
      line.setAttribute('marker-end', `url(#${markerPrefix}-arrow-${edge.route || edge.type || 'dep'})`);
      svg.appendChild(line);
      if (view === 'compute' && edge.label) {
        const label = document.createElementNS(svgNS, 'text');
        const labelX = edge.route === 'feedback' ? Math.max(sx, tx) + 28 : (sx + tx) / 2;
        const labelY = (sy + ty) / 2 - 7;
        label.setAttribute('x', String(labelX));
        label.setAttribute('y', String(labelY));
        label.setAttribute('class', 'kf-pypa-graph__edge-label');
        label.dataset.pypaEdgeLabel = edge.id; label.dataset.edgeType = edge.type;
        label.textContent = edge.label;
        svg.appendChild(label);
      }
    });
    scene.replaceChildren();
    graph.groups?.forEach((group) => {
      const box = document.createElement('div');
      box.className = 'kf-pypa-graph__group';
      box.dataset.pypaRegion = group.id;
      box.style.cssText = `left:${group.x}px;top:${group.y}px;width:${group.w}px;height:${group.h}px`;
      box.innerHTML = `<span>${pypaEscape(group.label)}</span>`;
      scene.appendChild(box);
    });
    placedNodes.filter((node) => node.expandedGroup).forEach((node) => {
      const box = document.createElement('div');
      box.className = `kf-pypa-graph__expanded-group${node.id === selectedNodeId ? ' is-selected' : ''}`;
      box.dataset.pypaGroup = node.id;
      box.style.cssText = `left:${node.x - 3}px;top:${node.y - 8}px;width:${node.groupWidth + 6}px;height:${node.groupHeight}px`;
      scene.appendChild(box);
    });
    syncBridges.forEach((bridge) => {
      const line = document.createElementNS(svgNS, 'path');
      line.setAttribute('d', bridge.path);
      line.setAttribute('class', 'kf-pypa-graph__sync-path');
      line.setAttribute('marker-end', `url(#${markerPrefix}-arrow-sync)`);
      line.dataset.pypaSyncPath = bridge.event.id;
      svg.appendChild(line);
    });
    scene.appendChild(svg);
    syncBridges.forEach((bridge) => {
      const event = bridge.event;
      const marker = document.createElement('button');
      marker.type = 'button';
      marker.className = 'btn btn-ghost kf-pypa-graph__sync-marker';
      marker.dataset.pypaSync = event.id;
      marker.style.cssText = `left:${bridge.x}px;top:${bridge.y}px;width:${bridge.w}px;height:${bridge.h}px`;
      marker.innerHTML = `<b>${pypaEscape(event.name)} · #${event.eventId}</b><small>${event.producer.coreType} SET · ${event.producer.pipe} → ${event.consumer.coreType} WAIT · ${event.consumer.pipe}</small>`;
      marker.title = `${event.buffer} · ${event.dtype} · 点击查看同步与源码依据`;
      scene.appendChild(marker);
    });
    placedNodes.forEach((node) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `kf-pypa-graph__node is-${node.kind}${view === 'compute' ? ' is-computation' : ''}${node.expandedGroup ? ' is-expanded-head' : ''}${node.childOperation ? ' is-detail-node' : ''}${node.id === selectedNodeId ? ' is-selected' : ''}`;
      button.dataset.pypaNode = node.id;
      if (node.sem) button.dataset.sem = node.sem;
      /* model-graphviz 按 kind 决定外形：tensor 圆角矩形、op 胶囊、折叠的 composite 走 module。 */
      button.dataset.nodeKind = node.type === 'composite' ? 'module' : node.kind === 'data' ? 'tensor' : 'op';
      button.title = node.title;
      if (node.expandable) {
        button.dataset.pypaExpand = node.id;
        button.setAttribute('aria-expanded', String(expandedStages.has(node.id)));
      }
      button.style.left = `${node.x}px`;
      button.style.top = `${node.y}px`;
      button.style.width = `${node.w}px`;
      button.style.minHeight = `${node.h}px`;
      button.setAttribute('aria-label', `${node.badge}，${node.title}，${node.sub}${node.meta?.length ? '，' + node.meta.join('，') : ''}`);
      if (compactNodes) {
        // Native text shares the viewport labels' density switch; geometry
        // remains identical to the dimensions used by layout and routing.
        button.classList.add('is-compact-node');
        button.style.height = `${node.h}px`;
        button.style.padding = '0';
        button.innerHTML = `<span class="kf-pypa-graph__node-head viz2-graph-compact-title"><b>${pypaEscape(node.title)}</b>${node.expandable ? `<span class="kf-pypa-graph__node-expand" data-pypa-expand-toggle="${pypaEscape(node.id)}" aria-hidden="true">${expandedStages.has(node.id) ? '−' : '+'}</span>` : ''}</span>`;
        scene.appendChild(button);
        return;
      }
      if (modelGraphviz) {
        /* 节点密度按 model-graphviz 还原：op / tensor 只有一行居中标签，
           不再画 badge、Shape、Valid、粒度/精度/内存；这些仍在 title、
           aria-label 与右侧 Inspector 里，图面只留可读的名字。 */
        const expandToggle = node.expandable
          ? `<span class="kf-pypa-graph__node-expand" data-pypa-expand-toggle="${pypaEscape(node.id)}" aria-hidden="true">${expandedStages.has(node.id) ? '−' : '+'}</span>` : '';
        const kindLine = node.type === 'composite' && !node.expandedGroup
          ? '<span class="kf-pypa-graph__node-kind">Composite</span>' : '';
        button.innerHTML = `<span class="kf-pypa-graph__node-label">${pypaEscape(node.title)}</span>${kindLine}${expandToggle}`;
      } else {
        button.innerHTML = `<span class="kf-pypa-graph__node-head">${view !== 'compute' && (node.expandedGroup || (node.childOperation && node.type === 'operation')) ? '' : `<em>${pypaEscape(node.badge)}</em>`}<b>${pypaEscape(node.title)}</b>${node.expandable ? `<span class="kf-pypa-graph__node-expand" data-pypa-expand-toggle="${pypaEscape(node.id)}" aria-hidden="true">${expandedStages.has(node.id) ? '−' : '+'}</span>` : ''}</span>${node.expandedGroup ? '' : (view === 'compute' ? cardHtml(node) : `<small>${pypaEscape(node.sub)}</small>`)}${node.childOperation ? `<span class="kf-pypa-graph__detail-source">源码 ${pypaEscape(node.lines)}</span>` : ''}${node.meta?.length ? `<span class="kf-pypa-graph__node-meta">${node.meta.map((item) => `<i>${pypaEscape(item)}</i>`).join('')}</span>` : ''}`;
      }
      scene.appendChild(button);
    });
    return { placedNodes, sceneWidth, sceneHeight };
  }
  function pypaComputeCardHtml(node, options = {}) {
    const layers = options.layers || { shape: true, precision: true, memory: false, control: false };
    const fields = node.overlay || window.PTO_COMPUTATION_PRESENTATION.overlays(node, { model: options.model || window.PTO_PYPA_SEMANTIC_GRAPH, qualifiedId: node.id });
    const shape = layers.shape && node.card ? fields.shape
      ? `<span class="kf-pypa-graph__node-shape" title="${pypaEscape(fields.shapeTitle)}">${pypaEscape(fields.shape).replace(/,\s+/g, ',<wbr> ')}</span>`
      : '<span class="kf-pypa-graph__node-shape is-undeclared">Shape 未声明</span>' : '';
    const valid = layers.shape && fields.valid ? `<span class="kf-pypa-graph__node-valid" title="Valid ${pypaEscape(fields.valid)}">Valid ${pypaEscape(fields.valid)}</span>` : '';
    const meta = [
      layers.shape && fields.granularity ? `<span data-overlay-field="granularity">${pypaEscape(fields.granularity)}</span>` : '',
      layers.precision && fields.precision ? `<span data-overlay-field="precision" title="${pypaEscape(fields.precision)}">${pypaEscape(fields.precision)}</span>` : '',
      layers.memory && fields.memory ? `<span class="kf-pypa-graph__memory" data-overlay-field="memory">${pypaEscape(fields.memory)}</span>` : '',
      layers.memory && fields.executionUnit ? `<span class="kf-pypa-graph__execution-unit" data-overlay-field="execution-unit">${pypaEscape(fields.executionUnit)}</span>` : '',
    ].filter(Boolean).join('');
    const core = !node.card ? `<small class="kf-pypa-graph__core-operation" title="${pypaEscape(node.api)}">${pypaEscape(node.semantic?.[node.type]?.category || node.sub)}</small>` : '';
    return `<span data-pypa-card>${core}${shape}${valid}${meta ? `<small class="kf-pypa-graph__node-meta-line">${meta}</small>` : ''}</span>`;
  }

window.PtoStudioComputationGraph=Object.freeze({renderScene:renderPypaGraphScene,applyExecution:applyPypaExecution,cardHtml:(node,model,layers)=>pypaComputeCardHtml(node,{model,layers})});
})();
