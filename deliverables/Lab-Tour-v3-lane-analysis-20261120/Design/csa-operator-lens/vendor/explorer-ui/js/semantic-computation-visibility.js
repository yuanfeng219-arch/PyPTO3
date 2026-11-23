/* View-only contraction of DATA cards. Original semantic objects and positions survive. */
(function (root) {
  'use strict';
  function withoutData(graph, options) {
    const hidden = new Set(graph.nodes.filter((node) => (node.kind === 'data' || ['data','state'].includes(node.type))).map((node) => node.id));
    const nodes = graph.nodes.filter((node) => !hidden.has(node.id));
    const outgoing = new Map(graph.nodes.map((node) => [node.id, []]));
    graph.edges.forEach((edge) => outgoing.get(edge.s).push(edge));
    const edges = [];
    function follow(chain, seen) {
      const first = chain[0], last = chain.at(-1);
      if (!hidden.has(last.t)) {
        if (chain.length === 1) { edges.push(first); return; }
        const sourceEdgeIds = chain.flatMap((edge) => edge.sourceEdgeIds || [edge.id]);
        const type = chain.every((edge) => edge.type === 'data') ? 'data'
          : chain.find((edge) => edge.type !== 'data').type;
        edges.push({ ...first, id: 'data-hidden:' + JSON.stringify(sourceEdgeIds),
          t: last.t, targetPort: last.targetPort, semanticTarget: last.semanticTarget,
          type, route: type, sourceEdgeIds,
          label: [...new Set(chain.map((edge) => edge.label).filter(Boolean))].join(' → ') });
        return;
      }
      if (seen.has(last.t)) return;
      const nextSeen = new Set(seen).add(last.t);
      for (const next of outgoing.get(last.t)) follow([...chain, next], nextSeen);
    }
    for (const edge of graph.edges) if (!hidden.has(edge.s)) follow([edge], new Set([edge.s]));
    // Compact the visible ranks after contraction; hidden tensors leave no empty ranks.
    const ranks=[...new Set(nodes.map(n=>n.y))].sort((a,b)=>a-b), positions=[];
    ranks.forEach((y,i)=>positions.push(i ? positions[i-1]+Math.max(...nodes.filter(n=>n.y===ranks[i-1]).map(n=>n.h))+56 : 76));
    const project=y=>{
      if(!ranks.length)return y;
      if(y<=ranks[0])return positions[0]+y-ranks[0];
      for(let i=1;i<ranks.length;i++)if(y<=ranks[i])return positions[i-1]+(y-ranks[i-1])/(ranks[i]-ranks[i-1])*(positions[i]-positions[i-1]);
      return positions.at(-1)+y-ranks.at(-1);
    };
    const compactNodes=nodes.map(n=>({...n,y:project(n.y),...(n.groupHeight?{groupHeight:project(n.y+n.groupHeight)-project(n.y)}:{})}));
    const groups=(graph.groups||[]).map(g=>({...g,y:project(g.y),h:project(g.y+g.h)-project(g.y)}));
    const height=Math.max(0,...compactNodes.map(n=>n.y+n.h),...groups.map(g=>g.y+g.h))+40;
    return { ...graph, nodes:compactNodes, groups, edges, height,
      edgeRoutes: root.PTO_COMPUTATION_LAYOUT.routeEdges(compactNodes, edges, graph.width - 88, options) };
  }
  // Level 1 keeps the operator contract visible while contracting transient
  // values. Each replacement edge records the semantic edges it represents.
  function overview(model) {
    const hidden = new Set(model.nodes.filter((node) => node.type === 'data' &&
      node.data.role === 'intermediate').map((node) => node.id));
    const outgoing = new Map(model.nodes.map((node) => [node.id, []]));
    model.edges.forEach((edge) => outgoing.get(edge.s).push(edge));
    const edges = [];
    function follow(chain, seen) {
      const first = chain[0], last = chain.at(-1);
      if (!hidden.has(last.t)) {
        if (chain.length === 1) { edges.push(first); return; }
        edges.push({ ...first, id: `overview:${chain.map((edge) => edge.id).join('|')}`,
          t: last.t, targetPort: last.targetPort, type: chain.every((edge) => edge.type === 'data') ? 'data' : last.type,
          label: last.label || first.label, sourceEdgeIds: chain.map((edge) => edge.id) });
        return;
      }
      if (seen.has(last.t)) return;
      for (const next of outgoing.get(last.t)) follow([...chain, next], new Set(seen).add(last.t));
    }
    model.edges.filter((edge) => !hidden.has(edge.s)).forEach((edge) => follow([edge], new Set([edge.s])));
    return { ...model, nodes: model.nodes.filter((node) => !hidden.has(node.id)), edges,
      groups: model.groups.map((group) => ({ ...group, members: group.members.filter((id) => !hidden.has(id)) })) };
  }
  root.PtoComputationVisibility = { withoutData, overview };
})(typeof window === 'undefined' ? globalThis : window);
