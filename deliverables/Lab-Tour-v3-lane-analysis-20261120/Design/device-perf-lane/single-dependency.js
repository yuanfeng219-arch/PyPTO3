(function (root) {
  function findChains(edges, workerTaskIds) {
    const next = new Map(), prev = new Map();
    const workers = new Set(Array.from(workerTaskIds || [], String));
    for (const edge of edges || []) {
      if (edge.pred == null || edge.succ == null) continue;
      const a = String(edge.pred), b = String(edge.succ);
      if (!workers.has(a) || !workers.has(b)) continue;
      if (!next.has(a)) next.set(a, new Set());
      if (!prev.has(b)) prev.set(b, new Set());
      next.get(a).add(b); prev.get(b).add(a);
    }
    const links = new Map(), incoming = new Set();
    for (const [a, targets] of next) {
      if (targets.size !== 1) continue;
      const b = targets.values().next().value;
      if (prev.get(b)?.size === 1 && a !== b) { links.set(a, b); incoming.add(b); }
    }
    const chains = [], visited = new Set();
    for (const start of links.keys()) {
      if (incoming.has(start) || visited.has(start)) continue;
      const chain = [start]; let node = start;
      while (links.has(node) && !visited.has(node)) {
        visited.add(node); node = links.get(node); chain.push(node);
      }
      if (chain.length > 1) chains.push(chain);
    }
    return chains.sort((a, b) => {
      if (/^\d+$/.test(a[0]) && /^\d+$/.test(b[0])) {
        const x = BigInt(a[0]), y = BigInt(b[0]);
        return x < y ? -1 : x > y ? 1 : 0;
      }
      return a[0].localeCompare(b[0]);
    });
  }
  function focusRange(chain, launches, windowUs) {
    if (!Number.isFinite(windowUs) || windowUs <= 0) return null;
    const ids = new Set((chain || []).map(String));
    const tasks = (launches || []).filter(task => ids.has(String(task.taskId)) &&
      Number.isFinite(task.startUs) && Number.isFinite(task.endUs) && task.endUs >= task.startUs);
    if (!tasks.length) return null;
    const start = Math.min(...tasks.map(task => task.startUs));
    const end = Math.max(...tasks.map(task => task.endUs));
    const span = Math.min(windowUs, Math.max(windowUs / 256, (end - start) * 1.24));
    if (span >= windowUs - 1e-6) return null;
    const left = Math.max(0, Math.min(windowUs - span, (start + end - span) / 2));
    return [left, left + span];
  }
  function blockPhases(block, viewStart, viewEnd) {
    const start = block.startUs, end = block.endUs;
    if (![start, end, block.setupUs, block.computeUs, viewStart, viewEnd].every(Number.isFinite) || end <= start || viewEnd <= viewStart) return [];
    const computeStart = Math.max(start, Math.min(end, end - Math.max(0, block.computeUs)));
    const setupEnd = Math.max(start, Math.min(computeStart, start + Math.max(0, block.setupUs)));
    return [
      {kind:'setup', start, end:setupEnd},
      {kind:'other', start:setupEnd, end:computeStart},
      {kind:'compute', start:computeStart, end},
    ].map(part => ({kind:part.kind, start:Math.max(viewStart, part.start), end:Math.min(viewEnd, part.end)}))
      .filter(part => part.end > part.start);
  }
  const api = { findChains, focusRange, blockPhases };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.SingleDependency = api;
})(typeof globalThis === 'undefined' ? this : globalThis);
