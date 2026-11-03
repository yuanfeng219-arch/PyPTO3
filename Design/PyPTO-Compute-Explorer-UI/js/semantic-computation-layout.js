/* Deterministic top-to-bottom layout for typed hierarchical computation graphs.
   It reads semantic nodes, edges and regions; it never adds or deletes them. */
(function (root) {
  'use strict';

  const BASE = { rootX: 72, rootTop: 76, childX: 34, childTop: 28,
    columnGap: 72, layerGap: 112, childColumnGap: 60, childLayerGap: 82,
    regionSide: 28, regionTop: 55, regionBottom: 28,
    compositeSide: 22, compositeHeader: 62, compositeBottom: 24 };
  const fullId = (prefix, local) => prefix ? `${prefix}/${local}` : local;
  const forward = (edge) => edge.type !== 'feedback';

  function nodeSize(node, child, options) {
    const overview = options.overviewLayout && !child;
    const fallbackWidth = child ? 194 : overview ? (['data', 'state'].includes(node.type) ? 178 : 228) : ['data', 'state'].includes(node.type) ? 170 : 188;
    return { w: node.w || fallbackWidth, h: options.compactGlyphs ? (['data','state'].includes(node.type) ? 64 : 48) : node.h || (child ? 80 : overview ? 52 : 88) };
  }

  function rankNodes(nodes, edges, scope) {
    const byId = new Map(nodes.map((item) => [item.node.id, item]));
    const incoming = new Map(nodes.map((item) => [item.node.id, []]));
    const outgoing = new Map(nodes.map((item) => [item.node.id, []]));
    edges.filter(forward).forEach((edge) => {
      incoming.get(edge.t).push(edge.s);
      outgoing.get(edge.s).push(edge.t);
    });
    const degree = new Map(nodes.map((item) => [item.node.id, incoming.get(item.node.id).length]));
    const ready = nodes.filter((item) => !degree.get(item.node.id)).map((item) => item.node.id);
    let visited = 0;
    while (ready.length) {
      const id = ready.shift();
      const item = byId.get(id);
      item.rank = Math.max(0, ...incoming.get(id).map((source) => byId.get(source).rank + 1));
      visited += 1;
      outgoing.get(id).forEach((target) => {
        degree.set(target, degree.get(target) - 1);
        if (!degree.get(target)) { ready.push(target); ready.sort((a, b) => byId.get(a).index - byId.get(b).index); }
      });
    }
    if (visited !== nodes.length) throw new Error(`${scope}: forward graph has a cycle; mark feedback edges explicitly`);
    return { incoming, outgoing };
  }

  function layoutScope(graph, prefix, expanded, childScope, options) {
    const nodes = graph.nodes.map((node, index) => {
      const id = fullId(prefix, node.id);
      const inner = node.composite?.subgraph && expanded.has(id)
        ? layoutScope(node.composite.subgraph, id, expanded, true, options) : null;
      const base = nodeSize(node, childScope, options);
      const footprint = inner ? {right:0,bottom:0} : options.executionFootprints?.[id] || {right:0,bottom:0};
      return { node, id, inner, index,
        footprint, frontW:base.w, frontH:base.h,
        boxW: inner ? Math.max(base.w, inner.width + BASE.compositeSide * 2) : base.w + footprint.right,
        boxH: inner ? BASE.compositeHeader + inner.height + BASE.compositeBottom : base.h + footprint.bottom };
    });
    const byId = new Map(nodes.map((item) => [item.node.id, item]));
    if (!nodes.length) return { graph, prefix, nodes, regions: [], width: 120, height: 90 };
    const { incoming, outgoing } = rankNodes(nodes, graph.edges, prefix || graph.id || 'graph');
    const membership = new Map();
    (graph.groups || []).forEach((group, index) => group.members.forEach((id) => membership.set(id, index)));
    const regionKey = (item) => membership.get(item.node.id) ?? (graph.groups || []).length;
    const ranks = new Map();
    nodes.forEach((item) => {
      if (!ranks.has(item.rank)) ranks.set(item.rank, []);
      ranks.get(item.rank).push(item);
    });
    const levels = [...ranks.keys()].sort((a, b) => a - b);
    const order = new Map();
    levels.forEach((rank) => {
      const row = ranks.get(rank);
      row.sort((a, b) => {
        const region = regionKey(a) - regionKey(b);
        if (region) return region;
        const barycenter = (item) => {
          const predecessors = incoming.get(item.node.id).filter((id) => order.has(id));
          return predecessors.length ? predecessors.reduce((sum, id) => sum + order.get(id), 0) / predecessors.length : Infinity;
        };
        return barycenter(a) - barycenter(b) || a.index - b.index;
      });
      row.forEach((item, index) => order.set(item.node.id, index));
    });
    // A reverse sweep reduces crossings where one producer fans out to several rows.
    [...levels].reverse().forEach((rank) => {
      const row = ranks.get(rank);
      row.sort((a, b) => {
        const region = regionKey(a) - regionKey(b);
        if (region) return region;
        const avg = (item) => {
          const successors = outgoing.get(item.node.id).filter((id) => order.has(id));
          return successors.length ? successors.reduce((sum, id) => sum + order.get(id), 0) / successors.length : Infinity;
        };
        return avg(a) - avg(b) || order.get(a.node.id) - order.get(b.node.id);
      });
      row.forEach((item, index) => order.set(item.node.id, index));
    });

    const side = childScope ? BASE.childX : BASE.rootX;
    const top = childScope ? BASE.childTop : options.overviewLayout ? 58 : BASE.rootTop;
    const columnGap = childScope ? BASE.childColumnGap : BASE.columnGap;
    const layerGap = childScope ? (options.childLayerGap ?? BASE.childLayerGap) : options.overviewLayout ? 16 : (options.rootLayerGap ?? BASE.layerGap);
    const rowWidth = (row) => row.reduce((sum, item) => sum + item.boxW, 0) + Math.max(0, row.length - 1) * columnGap;
    const widest = Math.max(0, ...levels.map((rank) => rowWidth(ranks.get(rank))));
    let width = widest + side * 2;
    let cursorY = top;
    if (childScope && graph.stateEvolution && options.overviewLayout) {
      const phases = graph.stateEvolution;
      const phaseRows = ['previous', 'current', 'updated'].flatMap((phase) => {
        const members = (phases[phase] || []).map((id) => byId.get(id)).filter(Boolean);
        return members.map((item) => ({ phase, members: [item] }));
      });
      // Reserve room for the unscaled viewport labels at the overview's typical
      // fit scale; the semantic phases stay in one column at any operator size.
      const phaseWidth = Math.max(520, ...phaseRows.map((row) => rowWidth(row.members)));
      width = phaseWidth + side * 2 + 32;
      phaseRows.forEach((row, index) => {
        let x = side + (phaseWidth - rowWidth(row.members)) / 2;
        row.members.forEach((item) => { item.x = x; item.y = cursorY; x += item.boxW + columnGap; });
        cursorY += Math.max(...row.members.map((item) => item.boxH)) +
          (phaseRows[index + 1]?.phase === row.phase ? 28 : 68);
      });
    } else if (options.flowLayout) {
      // Find a readable computation spine from graph topology, not operator names.
      // Feedback and state/control links remain semantic edges, but do not pull the spine sideways.
      const score = new Map(), previous = new Map();
      const ordered = nodes.slice().sort((a, b) => a.rank - b.rank || a.index - b.index);
      ordered.forEach((item) => {
        const candidates = (incoming.get(item.node.id) || []).map((id) => {
          const edge = graph.edges.find((entry) => entry.s === id && entry.t === item.node.id && ['data', 'state'].includes(entry.type));
          return edge ? { id, value: score.get(id) || 0 } : null;
        }).filter(Boolean).sort((a, b) => b.value - a.value);
        const weight = item.node.type === 'operation' || item.node.type === 'composite' ? 4 : item.node.type === 'data' ? 2 : 0;
        score.set(item.node.id, weight + (candidates[0]?.value || 0));
        if (candidates[0]) previous.set(item.node.id, candidates[0].id);
      });
      const end = ordered.filter((item) => item.node.type !== 'state').sort((a, b) =>
        (score.get(b.node.id) || 0) - (score.get(a.node.id) || 0) || b.rank - a.rank)[0];
      const spine = new Set();
      for (let id = end?.node.id; id; id = previous.get(id)) {
        if (byId.get(id).node.type !== 'state') spine.add(id);
      }
      const isStateRail = (item) => item.node.type === 'state' &&
        (item.node.state?.feedback || graph.edges.some((edge) => edge.type === 'feedback' && (edge.s === item.node.id || edge.t === item.node.id)));
      const sideRows = levels.map((rank) => ranks.get(rank).filter((item) => !spine.has(item.node.id) && !isStateRail(item)));
      const split = (row) => ({ left: row.filter((_, index) => index % 2 === 0), right: row.filter((_, index) => index % 2 === 1) });
      const leftWidth = Math.max(0, ...sideRows.map((row) => rowWidth(options.overviewLayout ? split(row).left : row)));
      const rightWidth = options.overviewLayout ? Math.max(0, ...sideRows.map((row) => rowWidth(split(row).right))) : 0;
      const spineWidth = Math.max(188, ...nodes.filter((item) => spine.has(item.node.id)).map((item) => item.boxW));
      const center = side + leftWidth + columnGap + spineWidth / 2;
      const stateX = center + spineWidth / 2 + columnGap * 2;
      levels.forEach((rank) => {
        const row = ranks.get(rank), main = row.find((item) => spine.has(item.node.id));
        const rail = row.filter(isStateRail);
        const others = row.filter((item) => item !== main && !rail.includes(item));
        const sides = options.overviewLayout ? split(others) : { left: others, right: [] };
        let leftX = center - spineWidth / 2 - columnGap - rowWidth(sides.left);
        sides.left.forEach((item) => { item.x = leftX; item.y = cursorY; leftX += item.boxW + columnGap; });
        if (main) { main.x = center - main.boxW / 2; main.y = cursorY; }
        let rightX = center + spineWidth / 2 + columnGap;
        sides.right.forEach((item) => { item.x = rightX; item.y = cursorY; rightX += item.boxW + columnGap; });
        rail.forEach((item, index) => { item.x = stateX + index * (item.boxW + columnGap); item.y = cursorY; });
        cursorY += Math.max(...row.map((item) => item.boxH)) + layerGap;
      });
      width = Math.max(width, ...nodes.map(item=>item.x+item.boxW+side), center + spineWidth / 2 + columnGap + rightWidth + side,
        stateX + Math.max(0, ...nodes.filter(isStateRail).map((item) => item.boxW)) + side,
        center + spineWidth / 2 + side);
    } else {
      levels.forEach((rank) => {
        const row = ranks.get(rank);
        let cursorX = side + (widest - rowWidth(row)) / 2;
        row.forEach((item) => { item.x = cursorX; item.y = cursorY; cursorX += item.boxW + columnGap; });
        cursorY += Math.max(...row.map((item) => item.boxH)) + layerGap;
      });
    }
    const regionTop = options.overviewLayout && !childScope ? 68 : BASE.regionTop;
    const regionBottom = BASE.regionBottom;
    if (options.overviewLayout && !childScope) {
      // Region frames are layout objects: reserve their header and a gap before
      // the next region instead of drawing overlapping boxes after placement.
      let previousBottom = -Infinity;
      (graph.groups || []).forEach((group, index) => {
        const members = group.members.map((id) => byId.get(id)).filter(Boolean);
        if (!members.length) return;
        const firstTop = Math.min(...members.map((item) => item.y)) - regionTop;
        const shift = Math.max(0, previousBottom + 32 - firstTop);
        if (shift) {
          const later = new Set(graph.groups.slice(index).flatMap((entry) => entry.members));
          nodes.filter((item) => later.has(item.node.id)).forEach((item) => { item.y += shift; });
        }
        previousBottom = Math.max(...members.map((item) => item.y + item.boxH)) + regionBottom;
      });
    }
    const regions = (graph.groups || []).map((group) => {
      const members = group.members.map((id) => byId.get(id));
      return { ...group,
        label: group.name,
        id: fullId(prefix, group.id),
        x: Math.min(...members.map((item) => item.x)) - BASE.regionSide,
        y: Math.min(...members.map((item) => item.y)) - regionTop,
        w: Math.max(...members.map((item) => item.x + item.boxW)) - Math.min(...members.map((item) => item.x)) + BASE.regionSide * 2,
        h: Math.max(...members.map((item) => item.y + item.boxH)) - Math.min(...members.map((item) => item.y)) + regionTop + regionBottom,
      };
    });
    const left = Math.min(0, ...regions.map((group) => group.x));
    if (left < 0) {
      const shift = -left + 16;
      nodes.forEach((item) => { item.x += shift; });
      regions.forEach((group) => { group.x += shift; });
      width += shift;
    }
    const bottom = Math.max(cursorY - layerGap + (childScope ? 34 : 56), ...regions.map((group) => group.y + group.h + 28));
    width = Math.max(width, ...regions.map((group) => group.x + group.w + 28));
    return { graph, prefix, nodes, regions, width, height: bottom };
  }

  function layout(model, expandedIds = new Set(), options = {}) {
    const expanded = new Set(expandedIds);
    const hierarchy = layoutScope(model, '', expanded, false, options);
    const nodes = [], groups = [], rawEdges = [], opened = new Set();
    const visit = (scope, offsetX, offsetY) => {
      scope.regions.forEach((group) => groups.push({ ...group, x: offsetX + group.x, y: offsetY + group.y }));
      scope.nodes.forEach((item) => {
        const node = item.node;
        const id = item.id;
        // 投影需要 model 才能解析常量与运行时维度；几何仍只由语义节点决定。
        nodes.push({ ...root.PTO_COMPUTATION_PRESENTATION.node(node, { id, parentId: scope.prefix || undefined, model }),
          x: offsetX + item.x, y: offsetY + item.y, w: item.inner ? item.boxW : item.frontW,
          h: item.inner ? 42 : item.frontH, executionFootprint:item.footprint,
          childOperation: !!scope.prefix, expandable: !!node.composite?.subgraph,
          expandedGroup: !!item.inner, groupWidth: item.inner ? item.boxW : undefined,
          groupHeight: item.inner ? item.boxH : undefined,
        });
        if (item.inner) {
          opened.add(id);
          visit(item.inner, offsetX + item.x + BASE.compositeSide, offsetY + item.y + BASE.compositeHeader);
        }
      });
      scope.graph.edges.forEach((edge) => rawEdges.push(root.PTO_COMPUTATION_PRESENTATION.edge(edge, {
        id: fullId(scope.prefix, edge.id), sourceId: fullId(scope.prefix, edge.s), targetId: fullId(scope.prefix, edge.t),
        internal: !!scope.prefix })));
    };
    visit(hierarchy, 0, 0);
    const byId = new Map(nodes.map((node) => [node.id, node]));
    const endpoint = (id, port, side) => {
      let current = byId.get(id), name = port || 'default';
      while (current?.expandedGroup) {
        const mapping = current.composite.portMappings[side]?.[name];
        if (!mapping) throw new Error(`${current.id}: missing ${side} port ${name}`);
        current = byId.get(`${current.id}/${mapping.node}`);
        name = mapping.port;
      }
      return { id: current?.id, port: name };
    };
    const edges = rawEdges.map((edge) => {
      const source = endpoint(edge.s, edge.sourcePort, 'outputs'), target = endpoint(edge.t, edge.targetPort, 'inputs');
      return { ...edge, semanticSource: edge.s, semanticTarget: edge.t,
        s: source.id, t: target.id, sourcePort: source.port, targetPort: target.port };
    });
    const right = Math.max(hierarchy.width, ...nodes.map((node) => node.x + node.w), ...groups.map((group) => group.x + group.w));
    const bottom = Math.max(hierarchy.height, ...nodes.map((node) => node.y + (node.groupHeight || node.h)), ...groups.map((group) => group.y + group.h));
    const geometry = routeEdges(nodes, edges, right, options);
    return { ...model, semantic: model, nodes, edges, groups, opened, flowLayout: !!options.flowLayout, edgeRoutes: geometry,
      width: Math.max(right + 88, ...geometry.map((route) => route.maxX + 32)), height: bottom + 100 };
  }

  /* 正交折线的圆角拐点，与 pto-design-system/patterns/model-graphviz 的
     roundedRoutePath 同构：每个拐点用二次贝塞尔过渡，半径统一 12px
     （pattern.js edgeCornerRadius 默认值），过长段自动退让到半段。 */
  const EDGE_CORNER_RADIUS = 12;
  function roundedPath(points, radius) {
    const clean = points.filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
    if (clean.length < 2) return '';
    const limit = Math.max(0, radius);
    const parts = [`M ${clean[0].x} ${clean[0].y}`];
    if (clean.length === 2 || !limit) {
      clean.slice(1).forEach((point) => parts.push(`L ${point.x} ${point.y}`));
      return parts.join(' ');
    }
    const along = (from, to, distance) => {
      const dx = to.x - from.x, dy = to.y - from.y, length = Math.hypot(dx, dy);
      if (!length) return { ...from };
      const ratio = Math.min(1, Math.max(0, distance / length));
      return { x: from.x + dx * ratio, y: from.y + dy * ratio };
    };
    for (let index = 1; index < clean.length - 1; index += 1) {
      const previous = clean[index - 1], corner = clean[index], next = clean[index + 1];
      const size = Math.min(limit, Math.hypot(corner.x - previous.x, corner.y - previous.y) / 2,
        Math.hypot(next.x - corner.x, next.y - corner.y) / 2);
      if (size <= 0.5) { parts.push(`L ${corner.x} ${corner.y}`); continue; }
      const before = along(corner, previous, size), after = along(corner, next, size);
      parts.push(`L ${before.x} ${before.y}`);
      parts.push(`Q ${corner.x} ${corner.y} ${after.x} ${after.y}`);
    }
    const last = clean[clean.length - 1];
    parts.push(`L ${last.x} ${last.y}`);
    return parts.join(' ');
  }

  /* 边标签按仓库 edge-tag 的规格占位：12px 字号，CJK 12px/字、其余 6.8px/字，
     左右各 6px 内边距，高度 18px。渲染层直接用这个宽度画底板，不再二次估算。 */
  const EDGE_TAG_HEIGHT = 18;
  const labelWidth = (text) => [...String(text)]
    .reduce((sum, char) => sum + (/[\u2e80-\u9fff\uff00-\uffef]/.test(char) ? 12 : 6.8), 0) + 12;

  function routeFlowEdges(nodes, edges, rightBound, options = {}) {
    const byId = new Map(nodes.map((node) => [node.id, node]));
    const obstacles = nodes.filter((node) => !node.expandedGroup).map(node=>({...node,w:node.w+(node.executionFootprint?.right||0),h:node.h+(node.executionFootprint?.bottom||0)}));
    const clearance = options.overviewLayout ? 3 : 10;
    const same = (a, b) => Math.abs(a - b) < 0.5;
    const clear = (points, source, target) => points.slice(1).every((end, index) => {
      const start = points[index];
      return obstacles.every((node) => {
        if (node.id === source.id || node.id === target.id) return true;
        const left = node.x - clearance, right = node.x + node.w + clearance;
        const top = node.y - clearance, bottom = node.y + node.h + clearance;
        if (same(start.x, end.x)) return start.x <= left || start.x >= right ||
          Math.max(start.y, end.y) <= top || Math.min(start.y, end.y) >= bottom;
        return start.y <= top || start.y >= bottom ||
          Math.max(start.x, end.x) <= left || Math.min(start.x, end.x) >= right;
      });
    });
    let returnLane = 0;
    return edges.map((edge) => {
      const source = byId.get(edge.s), target = byId.get(edge.t);
      const sx = source.x + source.w / 2, sy = source.y + source.h;
      const tx = target.x + target.w / 2, ty = target.y;
      if (edge.type === 'feedback' || ty <= sy + (options.overviewLayout ? 4 : 22)) {
        if(source.y > target.y + target.h + 20){
          const laneY=(target.y + target.h + source.y)/2;
          const points=[{x:sx,y:source.y},{x:sx,y:laneY},{x:tx,y:laneY},{x:tx,y:target.y+target.h}];
          if(clear(points,source,target))return {points,path:roundedPath(points,EDGE_CORNER_RADIUS),labelX:tx+10,labelY:laneY-8,labelAnchor:'start',maxX:Math.max(sx,tx)};
        }
        if(edge.type==='feedback'&&source.y>target.y){
          const exitY=source.y-24,lanes=[target.x-24,target.x+target.w+24,source.x-24,source.x+source.w+24,rightBound+32+returnLane*18,Math.min(...nodes.map(node=>node.x))-32];
          const candidates=lanes.map(lane=>{const end={x:lane<tx?target.x:target.x+target.w,y:target.y+target.h/2};return [{x:sx,y:source.y},{x:sx,y:exitY},{x:lane,y:exitY},{x:lane,y:end.y},end];});
          const valid=candidates.filter(points=>clear(points,source,target));
          if(valid.length){const points=valid.sort((a,b)=>Math.abs(a[2].x-sx)-Math.abs(b[2].x-sx))[0];returnLane++;return {points,path:roundedPath(points,EDGE_CORNER_RADIUS),labelX:points[2].x-10,labelY:(exitY+points[4].y)/2,labelAnchor:'end',maxX:Math.max(...points.map(point=>point.x))};}
        }
        const parentId = source.parentId && source.parentId === target.parentId ? source.parentId : null;
        const frame = parentId && byId.get(parentId);
        const left = frame?.expandedGroup && target.x - frame.x < frame.x + frame.groupWidth - target.x - target.w;
        const lane = frame?.expandedGroup ? left ? frame.x + 10 : frame.x + frame.groupWidth - 10 : rightBound + 32 + returnLane++ * 18;
        const start = { x: left ? source.x : source.x + source.w, y: source.y + source.h / 2 };
        const end = { x: left ? target.x : target.x + target.w, y: target.y + target.h / 2 };
        const points = [start, { x: lane, y: start.y }, { x: lane, y: end.y }, end];
        return { points, path: roundedPath(points, EDGE_CORNER_RADIUS), labelX: lane - 10,
          labelY: (start.y + end.y) / 2, labelAnchor: 'end', maxX: lane };
      }
      const middle = (sy + ty) / 2;
      const direct = [{ x: sx, y: sy }, { x: sx, y: middle }, { x: tx, y: middle }, { x: tx, y: ty }];
      const candidates = [direct];
      const bend = options.overviewLayout ? 6 : 28;
      const upper = sy + Math.min(bend, (ty - sy) / 3);
      const lower = ty - Math.min(bend, (ty - sy) / 3);
      const corridor = obstacles.filter((node) => node.id !== source.id && node.id !== target.id &&
        node.y < lower && node.y + node.h > upper);
      const lanes = [sx, tx, ...corridor.flatMap((node) => [node.x - 24, node.x + node.w + 24]),
        Math.min(source.x, target.x) - 32, Math.max(source.x + source.w, target.x + target.w) + 32];
      [...new Set(lanes)].forEach((lane) => candidates.push([
        { x: sx, y: sy }, { x: sx, y: upper }, { x: lane, y: upper },
        { x: lane, y: lower }, { x: tx, y: lower }, { x: tx, y: ty },
      ]));
      const valid = candidates.filter((points) => clear(points, source, target));
      const cost = (points) => points.slice(1).reduce((sum, point, index) =>
        sum + Math.abs(point.x - points[index].x) + Math.abs(point.y - points[index].y), 0) + points.length * 12;
      const points = (valid.length ? valid : candidates.slice(-2)).sort((a, b) => cost(a) - cost(b))[0];
      const labelY = points.length === 4 ? middle : lower;
      return { points, path: roundedPath(points, EDGE_CORNER_RADIUS), labelX: tx + 10,
        labelY: labelY - 8, labelAnchor: 'start', maxX: Math.max(...points.map((point) => point.x)) };
    });
  }

  function routeEdges(nodes, edges, rightBound, options = {}) {
    if (options.flowLayout) return routeFlowEdges(nodes, edges, rightBound, options);
    /* 圆角拐点与带底板标签只在显式开启时生效（matmul 走 model-graphviz 契约），
       其余图保持原有的直角折线与无底板标签读法。 */
    const corner = options.modelGraphviz ? EDGE_CORNER_RADIUS : 0;
    const byId = new Map(nodes.map((node) => [node.id, node]));
    const incoming = new Map(nodes.map((node) => [node.id, []]));
    const outgoing = new Map(nodes.map((node) => [node.id, []]));
    edges.forEach((edge, index) => { incoming.get(edge.t).push(index); outgoing.get(edge.s).push(index); });
    incoming.forEach((list) => list.sort((a, b) => byId.get(edges[a].s).x - byId.get(edges[b].s).x || a - b));
    outgoing.forEach((list) => list.sort((a, b) => byId.get(edges[a].t).x - byId.get(edges[b].t).x || a - b));
    const port = (node, list, index) => node.x + node.w * (list.indexOf(index) + 1) / (list.length + 1);
    let sideLane = 0;
    const routes = edges.map((edge, index) => {
      const source = byId.get(edge.s), target = byId.get(edge.t);
      const sx = port(source, outgoing.get(edge.s), index), tx = port(target, incoming.get(edge.t), index);
      const sy = source.y + source.h, ty = target.y;
      if (edge.route === 'feedback') {
        const frames = nodes.filter((node) => node.expandedGroup &&
          source.x >= node.x && target.x >= node.x &&
          source.x + source.w <= node.x + node.groupWidth && target.x + target.w <= node.x + node.groupWidth &&
          source.y >= node.y && target.y >= node.y &&
          source.y + source.h <= node.y + node.groupHeight && target.y + target.h <= node.y + node.groupHeight);
        frames.sort((a, b) => a.groupWidth * a.groupHeight - b.groupWidth * b.groupHeight);
        const left = frames.length && target.x + target.w / 2 < frames[0].x + frames[0].groupWidth / 2;
        const lane = frames.length ? left ? frames[0].x + 10 : frames[0].x + frames[0].groupWidth - 10 : rightBound + 42;
        const x1 = left ? source.x : source.x + source.w;
        const x2 = left ? target.x : target.x + target.w;
        const y1 = source.y + source.h / 2, y2 = target.y + target.h / 2;
        return { path: roundedPath([{ x: x1, y: y1 }, { x: lane, y: y1 }, { x: lane, y: y2 }, { x: x2, y: y2 }], corner),
          labelX: lane + (left ? 10 : -10), labelY: (y1 + y2) / 2, labelAnchor: left ? 'start' : 'end',
          labelAxis: 'y', labelMin: Math.min(y1, y2), labelMax: Math.max(y1, y2), maxX: Math.max(lane, x1, x2) };
      }
      // Contracted DATA paths may pass directly through the now-empty card space.
      // Keep the side lane when a retained operation still occupies that corridor.
      const blocked = !edge.sourceEdgeIds || nodes.some((node) => node.id !== edge.s && node.id !== edge.t && !node.expandedGroup
        && node.y < ty && node.y + node.h > sy
        && node.x < Math.max(sx, tx) + 12 && node.x + node.w > Math.min(sx, tx) - 12);
      if (ty - sy > 190 && blocked) {
        const leftBound = Math.min(...nodes.map((node) => node.x));
        const useLeft = (source.x + target.x) / 2 < rightBound / 2;
        const lane = useLeft ? Math.max(22, leftBound - 42 - sideLane * 14) : rightBound + 42 + sideLane * 14;
        sideLane = (sideLane + 1) % 4;
        return { path: roundedPath([{ x: sx, y: sy }, { x: sx, y: sy + 28 }, { x: lane, y: sy + 28 },
          { x: lane, y: ty - 28 }, { x: tx, y: ty - 28 }, { x: tx, y: ty }], corner),
          labelX: lane + (useLeft ? 10 : -10), labelY: (sy + ty) / 2, labelAnchor: useLeft ? 'start' : 'end',
          labelAxis: 'y', labelMin: sy + 28, labelMax: ty - 28, maxX: Math.max(lane, sx, tx) };
      }
      const middle = (sy + ty) / 2;
      /* 水平段落上方的标签按中间对齐；两侧太窄时改成从目标节点右侧起排，避免压住竖线。 */
      const inline = Math.abs(sx - tx) > 38;
      return { path: roundedPath([{ x: sx, y: sy }, { x: sx, y: middle }, { x: tx, y: middle }, { x: tx, y: ty }], corner),
        labelX: inline ? (sx + tx) / 2 : tx + 12,
        /* 标签底板是不透明的：压在线段正中即为仓库的 edge-tag 读法。 */
        labelY: inline ? (corner ? middle : middle - 7) : middle - 7, labelAnchor: inline ? 'middle' : 'start', labelAxis: inline ? 'x' : 'y',
        labelMin: inline ? Math.min(sx, tx) : sy,
        labelMax: inline ? Math.max(sx, tx) : ty,
        maxX: Math.max(sx, tx) };
    });
    const placed = [];
    /* 展开的父容器也是一道边界：跨层标签压在外框上比压在线段上更难读，
       所以容器外框也参与避让；完全落在某个容器内的标签不受该容器约束。 */
    const frames = nodes.filter((node) => node.expandedGroup)
      .map((node) => ({ x: node.x - 3, y: node.y - 8, w: node.groupWidth + 6, h: node.groupHeight }));
    const contains = (box, frame) => box.x >= frame.x && box.x + box.w <= frame.x + frame.w &&
      box.y >= frame.y && box.y + box.h <= frame.y + frame.h;
    const crossesFrame = (box) => frames.some((frame) => !contains(box, frame) &&
      box.x < frame.x + frame.w && box.x + box.w > frame.x && box.y < frame.y + frame.h && box.y + box.h > frame.y);
    routes.forEach((route, index) => {
      if (!edges[index].label) return;
      const width = labelWidth(edges[index].label);
      route.labelW = width;
      route.labelH = EDGE_TAG_HEIGHT;
      const anchor = route.labelAnchor || 'middle';
      /* 侧边车道上的标签沿垂直方向排，所以避让要先纵向让开容器、再横向让开车道；
         水平标签则沿垂直方向挪。盒子左边按锚点折算，避免 start/end 时错当成居中。 */
      const boxLeft = (x) => anchor === 'middle' ? x - width / 2 : anchor === 'end' ? x - width : x;
      const candidates = route.labelAxis === 'x'
        ? [[0, 0], [0, 22], [0, -22], [0, 44], [0, -44], [0, 66], [0, -66]]
        : [[0, 0], [0, -42], [0, 42], [0, -88], [0, 88], [18, 0], [-18, 0]];
      for (const [dx, dy] of candidates) {
        const x = route.labelX + dx;
        const y = route.labelY + dy;
        const box = { x: boxLeft(x), y: y - EDGE_TAG_HEIGHT / 2, w: width, h: EDGE_TAG_HEIGHT };
        const intersects = (other) => box.x < other.x + other.w && box.x + box.w > other.x && box.y < other.y + other.h && box.y + box.h > other.y;
        if (placed.some(intersects) || nodes.filter((node) => !node.expandedGroup).some(intersects)) continue;
        if (crossesFrame(box)) continue;
        route.labelX = x; route.labelY = y; placed.push(box); return;
      }
      placed.push({ x: boxLeft(route.labelX), y: route.labelY - EDGE_TAG_HEIGHT / 2, w: width, h: EDGE_TAG_HEIGHT });
    });
    return routes;
  }

  // Project static synchronization metadata after layout; never rank it as data flow.
  function synchronizationBridges(graph) {
    const byId = new Map(graph.nodes.map((node) => [node.id, node]));
    const right = Math.max(...graph.nodes.map((node) => node.x + (node.groupWidth || node.w)));
    const resolve = (end) => byId.get(end.anchor) || byId.get(end.boundary || end.node);
    return (graph.synchronizations || []).map((event) => {
      const producer = resolve(event.producer), consumer = resolve(event.consumer);
      if (!producer || !consumer) throw new Error(`${event.name}: unresolved synchronization boundary`);
      const sx = producer.x + producer.w / 2, sy = producer.y + producer.h;
      const tx = consumer.x + consumer.w / 2, ty = consumer.y;
      if (graph.flowLayout) {
        const route = routeFlowEdges(graph.nodes, [{ s: producer.id, t: consumer.id, type: 'sync' }], right)[0];
        const x = (route.points[3]?.x ?? producer.x + producer.w) + 16;
        return { event, producer: producer.id, consumer: consumer.id,
          x, y: sy + 18, w: 254, h: 48, path: route.path, maxX: Math.max(route.maxX, x + 254) };
      }
      const lane = right + 24 + event.eventId * 12;
      return { event, producer: producer.id, consumer: consumer.id,
        x: producer.x + producer.w + 24, y: sy + 22, w: 254, h: 48,
        path: roundedPath([{ x: sx, y: sy }, { x: sx, y: sy + 12 }, { x: lane, y: sy + 12 },
          { x: lane, y: ty - 20 }, { x: tx, y: ty - 20 }, { x: tx, y: ty }], 0),
        maxX: Math.max(lane, producer.x + producer.w + 278) };
    });
  }

  root.PTO_COMPUTATION_LAYOUT = { layout, routeEdges, synchronizationBridges };
})(typeof window === 'undefined' ? globalThis : window);
