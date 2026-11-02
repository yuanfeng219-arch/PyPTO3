import fs from 'node:fs/promises';

// CSA qk_pv: extract evidence without executing either input file.
export function parseCsaQkPv(sourceText, dumpText) {
  const sourceLines = sourceText.split(/\r?\n/);
  const start = sourceLines.findIndex(line => /\bpl\.spmd\s*\(.*name_hint\s*=\s*["']qk_pv["']/.test(line));
  if (start < 0) throw new Error('源码中没有找到 name_hint="qk_pv" 的 pl.spmd');
  const indent = sourceLines[start].match(/^\s*/)[0].length;
  let end = start + 1;
  while (end < sourceLines.length && (!sourceLines[end].trim() || sourceLines[end].match(/^\s*/)[0].length > indent)) end++;
  const source = sourceLines.slice(start, end).map((t, i) => ({ no: start + i + 1, t }));
  const dumpLines = dumpText.split(/\r?\n/);
  const blocks = {};
  for (const kind of ['aic', 'aiv']) {
    const at = dumpLines.findIndex(line => new RegExp('^\\s*def qk_pv_' + kind + '\\(').test(line));
    if (at < 0) throw new Error('编译输出中缺少 qk_pv_' + kind);
    const level = dumpLines[at].match(/^\s*/)[0].length;
    let last = at + 1;
    while (last < dumpLines.length && (!dumpLines[last].trim() || dumpLines[last].match(/^\s*/)[0].length > level)) last++;
    blocks[kind] = { start: at + 1, end: last, lines: dumpLines.slice(at, last) };
  }
  const allocations = [];
  for (const [worker, block] of Object.entries(blocks)) {
    const known = new Map();
    block.lines.forEach((line, index) => {
      const alloc = line.match(/(mem_\w+)\s*:\s*pl\.Ptr\s*=\s*pl\.tile\.alloc\(pl\.Mem\.(\w+),\s*(\d+)\)/);
      if (alloc) known.set(alloc[1], { id: alloc[1], space: alloc[2], size: Number(alloc[3]), worker, allocLine: block.start + index, refs: [] });
    });
    block.lines.forEach((line, index) => {
      const m = line.match(/pl\.MemRef\((mem_\w+),\s*pl\.const\((\d+),\s*pl\.INT64\),\s*(\d+)\)/);
      if (!m || !known.has(m[1])) return;
      const item = known.get(m[1]);
      const address = Number(m[2]);
      if (item.address !== undefined && item.address !== address) throw new Error('同一分配出现多个地址: ' + item.id);
      item.address = address;
      item.refs.push({ line: block.start + index, size: Number(m[3]), text: line.trim() });
    });
    block.lines.forEach((line, index) => {
      const m = line.match(/(\w+)\s*:.*pl\.system\.reserve_buffer\(name=['"][^'"]+['"],\s*size=(\d+),\s*base=(\d+)\)/);
      if (m) allocations.push({ id: m[1], worker, space: worker === 'aiv' ? 'Vec' : 'Mat', size: Number(m[2]), address: Number(m[3]), allocLine: block.start + index, refs: [], kind: 'reserve_buffer' });
    });
    allocations.push(...known.values());
  }
  if (!allocations.length) throw new Error('未发现支持的片上分配记录');
  return { entry: 'qk_pv', source, allocations, blocks: Object.fromEntries(Object.entries(blocks).map(([k,v]) => [k, { start:v.start, end:v.end }])), evidence: '编译输出中的地址和大小；未建立源码到分配的自动对应关系' };
}

export async function importCsaQkPv(sourcePath, dumpPath) {
  const [source, dump] = await Promise.all([fs.readFile(sourcePath, 'utf8'), fs.readFile(dumpPath, 'utf8')]);
  return parseCsaQkPv(source, dump);
}

// Build a conservative graph from source assignment/call sites. Name matches to
// compiler symbols are heuristic; address ranges remain compiler evidence.
export function toCsaManifest(parsed, sourcePath, dumpPath) {
  const wanted = ['qk_kv','qk_q_tile','qk_raw','qk_scaled','qk_scores','qk_mi','qk_exp','qk_li','qk_exp_bf16','qk_oi'];
  const nodes = [];
  for (const name of wanted) {
    const lines = parsed.source.filter(r => new RegExp('^\\s*' + name + '\\s*=(?!=)').test(r.t));
    if (!lines.length) continue;
    const matches = parsed.allocations.filter(a => a.refs.some(r => new RegExp('\\b' + name + '(?:_inline|__|_tile)').test(r.text)));
    nodes.push({ id:name, assigns:name, label:name, frame:'qk', bufs:matches.map(a=>a.worker+':'+a.id), regions:[...new Set(matches.map(a=>({Mat:'mat',Left:'l0a',Right:'l0b',Acc:'l0c',Vec:'ub'})[a.space]).filter(Boolean))] });
  }
  nodes.forEach((n,i) => { if(i) n.after=[nodes[i-1].id]; });
  return { id:'csa_qk_pv', entry:'qk_pv', file:sourcePath, platform:'a2a3', consts:{}, knobs:[], source:parsed.source, tensors:[],
    frames:[{id:'qk',label:'qk_pv · 源码顺序（依赖为推断）',scope:'pl.spmd'}],
    phases:[{id:'ops',label:'源码操作 · 顺序示意',nodes:nodes.map(n=>n.id),wrap:4}],nodes, buffers:[],
    compiledAllocations:parsed.allocations, dumpFile:dumpPath, note:'地址与大小来自编译输出；名称匹配和节点顺序为推断，未匹配项不建立联动。' };
}
