import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';
import { renderPythonModelGraph } from './python_model_parser.mjs';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const MANIFEST = path.join(ROOT, 'assets', 'manifest.json');
const ARTIFACTS = path.join(ROOT, 'artifacts');
const send = (id, result, error) => {
  const message = { jsonrpc: '2.0', id };
  if (error) message.error = error; else message.result = result;
  process.stdout.write(`${JSON.stringify(message)}\n`);
};
const tools = [
  { name: 'search_visual_assets', description: 'Search visualization assets by keyword, category, or framework.', inputSchema: { type: 'object', properties: { query: { type: 'string' }, category: { type: 'string' }, framework: { type: 'string' } } } },
  { name: 'render_model_graph', description: 'Render a model JSON into an HTML graph and return a structural summary.', inputSchema: { type: 'object', properties: { model_path: { type: 'string' }, output_name: { type: 'string' } }, required: ['model_path'] } },
  { name: 'render_swimlane', description: 'Render a Chrome Trace JSON traceEvents file into an interactive HTML swimlane timeline.', inputSchema: { type: 'object', properties: { trace_path: { type: 'string' }, output_name: { type: 'string' }, max_events: { type: 'number' } }, required: ['trace_path'] } },
  { name: 'render_python_model_graph', description: 'Statically inspect a Python/PyTorch model source file and render its whole-network class and submodule structure as interactive HTML. Does not execute the source.', inputSchema: { type: 'object', properties: { source_path: { type: 'string' }, output_name: { type: 'string' } }, required: ['source_path'] } },
  { name: 'inspect_operator', description: 'Inspect an operator contract and provide static visualization and validation guidance.', inputSchema: { type: 'object', properties: { operator: { type: 'string' }, framework: { type: 'string' }, input_shapes: { type: 'array' }, dtype: { type: 'string' } }, required: ['operator'] } },
  { name: 'generate_operator_scaffold', description: 'Generate a reviewable operator development scaffold with tests and visualization metadata.', inputSchema: { type: 'object', properties: { operator_name: { type: 'string' }, framework: { type: 'string' }, output_dir: { type: 'string' } }, required: ['operator_name'] } },
];
const load = (file) => JSON.parse(fs.readFileSync(file, 'utf8'));
const resolve = (value) => path.resolve(process.cwd(), value);
const wrapped = (data) => ({ content: [{ type: 'text', text: JSON.stringify(data, null, 2) }], structuredContent: data });

function call(name, args) {
  if (name === 'search_visual_assets') {
    const q = String(args.query || '').toLowerCase(); const category = String(args.category || '').toLowerCase(); const framework = String(args.framework || '').toLowerCase();
    const assets = load(MANIFEST).assets.filter((asset) => (!q || JSON.stringify(asset).toLowerCase().includes(q)) && (!category || asset.category.toLowerCase() === category) && (!framework || asset.frameworks.map((item) => item.toLowerCase()).includes(framework)));
    return { count: assets.length, assets };
  }
  if (name === 'inspect_operator') {
    const operator = String(args.operator); const dtype = String(args.dtype || 'unknown'); const risks = [];
    if (['float16', 'bfloat16', 'bf16', 'fp16'].includes(dtype)) risks.push('混合精度算子应检查累加精度、cast 位置和容差。');
    if (['multiheadattention', 'softmax', 'layernorm', 'rmsnorm'].includes(operator.toLowerCase())) risks.push('该类算子通常包含归约或非线性路径，建议优先查看 Tensor 精度流和边界 shape。');
    return { operator, framework: args.framework || 'unknown', contract: { input_shapes: args.input_shapes || [], dtype }, static_inference: { recommended_assets: ['operator-card', 'tensor-contract'], risks: risks.length ? risks : ['需要结合实现源码和真实编译结果进一步确认。'] }, evidence_boundary: '这是基于输入参数的静态分析，不代表真实设备性能或编译通过。' };
  }
  if (name === 'render_model_graph') {
    const modelPath = resolve(String(args.model_path)); if (!fs.existsSync(modelPath)) throw new Error(`model_path does not exist: ${modelPath}`);
    const model = load(modelPath); const nodes = model.nodes || []; const outputName = String(args.output_name || path.basename(modelPath, '.json')); fs.mkdirSync(ARTIFACTS, { recursive: true }); const output = path.join(ARTIFACTS, `${outputName}.html`);
    const cards = nodes.map((node, index) => `<article class="node"><span>${String(index + 1).padStart(2, '0')}</span><h2>${node.op || 'Unknown'}</h2><p>${node.id || ''}</p><code>${(node.shape || []).join(' × ')} · ${node.dtype || 'unknown'}</code></article>`).join('<div class="arrow">↓</div>');
    const page = `<!doctype html><meta charset="utf-8"><title>${model.name || outputName}</title><style>body{font:15px system-ui;background:#10141d;color:#eaf0f6;padding:32px}.graph{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.node{background:#1c2635;border:1px solid #4e6b8c;border-radius:12px;padding:16px;min-width:160px;box-shadow:0 6px 18px #0004}h2{margin:0 0 8px;color:#8dd5ff}p{color:#9dacbb}code{color:#ffdc8a}.arrow{color:#78b8d9;font-size:28px}</style><h1>${model.name || outputName}</h1><p>Framework: ${model.framework || 'unknown'} · Nodes: ${nodes.length}</p><div class="graph">${cards}</div>`;
    fs.writeFileSync(output, page); return { summary: { model: model.name || outputName, framework: model.framework || 'unknown', node_count: nodes.length, input_count: (model.inputs || []).length }, artifacts: [{ type: 'html', path: output }], nodes };
  }
  if (name === 'render_swimlane') {
    const tracePath = resolve(String(args.trace_path)); if (!fs.existsSync(tracePath)) throw new Error(`trace_path does not exist: ${tracePath}`);
    const trace = load(tracePath); const raw = (trace.traceEvents || []).filter((event) => event.ph === 'X' && Number.isFinite(event.ts) && Number.isFinite(event.dur) && event.dur >= 0);
    if (!raw.length) throw new Error('trace contains no complete duration events with ph="X"');
    const maxEvents = Number(args.max_events || 5000); const events = raw.slice(0, maxEvents); const names = new Map();
    for (const event of trace.traceEvents || []) if (event.ph === 'M' && event.name === 'thread_name' && event.args?.name) names.set(`${event.pid}:${event.tid}`, event.args.name);
    const lanes = [...new Set(events.map((event) => names.get(`${event.pid}:${event.tid}`) || `pid ${event.pid} / tid ${event.tid}`))]; const starts = events.map((event) => event.ts); const ends = events.map((event) => event.ts + event.dur); const start = Math.min(...starts); const end = Math.max(...ends); const duration = Math.max(1, end - start);
    const safeEvents = events.map((event) => ({ lane: names.get(`${event.pid}:${event.tid}`) || `pid ${event.pid} / tid ${event.tid}`, name: String(event.name || event.args?.name || 'event'), cat: String(event.cat || ''), ts: event.ts - start, dur: event.dur, pid: event.pid, tid: event.tid, args: event.args || {} }));
    const outputName = String(args.output_name || path.basename(tracePath, '.json')); fs.mkdirSync(ARTIFACTS, { recursive: true }); const output = path.join(ARTIFACTS, `${outputName}.html`);
    const data = JSON.stringify({ lanes, events: safeEvents, start, end, duration }).replace(/</g, '\\u003c');
    const page = `<!doctype html><meta charset="utf-8"><title>Swimlane · ${outputName}</title><style>:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0f141d;color:#e8eef5;font:13px system-ui,Segoe UI,sans-serif}.top{height:64px;display:flex;align-items:center;gap:18px;padding:0 22px;border-bottom:1px solid #273345;background:#151c27}.title{font-size:16px;font-weight:700}.meta{color:#91a0b2;font-size:11px}.controls{margin-left:auto;display:flex;gap:8px}.controls input{width:220px;padding:8px 10px;border:1px solid #34445a;border-radius:8px;background:#0f141d;color:#e8eef5}.viewport{height:calc(100vh - 64px);overflow:auto}.grid{min-width:1100px;display:grid;grid-template-columns:190px 1fr}.lane-label{position:sticky;left:0;z-index:2;background:#151c27;border-right:1px solid #273345;border-bottom:1px solid #202b3b;padding:9px 12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.timeline{position:relative;background:#0f141d;border-bottom:1px solid #202b3b}.row{height:34px;position:relative;border-bottom:1px solid #1b2635}.bar{position:absolute;top:6px;height:22px;border:1px solid #6e91bb;border-radius:4px;background:linear-gradient(180deg,#37658e,#234565);overflow:hidden;white-space:nowrap;text-overflow:ellipsis;padding:3px 6px;font-size:10px;color:#eaf3ff;cursor:pointer}.bar:hover{filter:brightness(1.25);z-index:3}.ruler{height:38px;position:sticky;top:0;z-index:4;background:#151c27;border-bottom:1px solid #34445a}.tick{position:absolute;top:0;bottom:0;border-left:1px solid #34445a;color:#91a0b2;padding:5px 0 0 5px;font-size:10px}.detail{position:fixed;right:18px;bottom:18px;width:320px;max-height:340px;overflow:auto;padding:14px;border:1px solid #49627f;border-radius:10px;background:#182332;box-shadow:0 14px 40px #0008;display:none}.detail.show{display:block}.detail h3{margin:0 0 10px;color:#8dd5ff}.detail pre{white-space:pre-wrap;color:#c1cfde;font-size:11px}</style><header class="top"><div class="title">${outputName}</div><div class="meta">${safeEvents.length.toLocaleString()} events · ${lanes.length} lanes · ${duration.toFixed(2)} μs</div><div class="controls"><input id="filter" placeholder="筛选泳道或事件名称" /></div></header><main class="viewport"><section class="grid" id="grid"><div class="lane-label ruler">Lane</div><div class="timeline ruler" id="ruler"></div></section></main><aside class="detail" id="detail"><h3 id="detailTitle"></h3><pre id="detailBody"></pre></aside><script>const DATA=${data};const grid=document.getElementById('grid'),ruler=document.getElementById('ruler'),detail=document.getElementById('detail');const width=Math.max(1200,Math.min(9000,DATA.duration*4));const color=(s)=>{let h=0;for(const c of s)h=(h*31+c.charCodeAt(0))%360;return 'hsl('+h+' 52% 42%)'};for(let i=0;i<=10;i++){const t=document.createElement('span');t.className='tick';t.style.left=(i*10)+'%';t.textContent=(DATA.duration*i/10).toFixed(0)+' μs';ruler.appendChild(t)}const groups=new Map(DATA.lanes.map((lane,i)=>[lane,i]));const rows=DATA.lanes.map((lane,i)=>{const label=document.createElement('div');label.className='lane-label';label.dataset.lane=lane;label.textContent=lane;const track=document.createElement('div');track.className='timeline row';track.style.width=width+'px';track.dataset.lane=lane;grid.append(label,track);return track});for(const event of DATA.events){const row=rows[groups.get(event.lane)];const bar=document.createElement('button');bar.className='bar';bar.style.left=(event.ts/DATA.duration*width)+'px';bar.style.width=Math.max(2,event.dur/DATA.duration*width)+'px';bar.style.background=color(event.name);bar.title=event.name+' · '+event.dur.toFixed(2)+' μs';bar.textContent=event.dur>DATA.duration*.025?event.name:'';bar.onclick=()=>{detail.classList.add('show');document.getElementById('detailTitle').textContent=event.name;document.getElementById('detailBody').textContent=JSON.stringify({lane:event.lane,start_us:event.ts,duration_us:event.dur,category:event.cat,pid:event.pid,tid:event.tid,args:event.args},null,2)};row.appendChild(bar)}document.getElementById('filter').oninput=(e)=>{const q=e.target.value.toLowerCase();for(const row of rows){const visible=row.dataset.lane.toLowerCase().includes(q)||[...row.querySelectorAll('.bar')].some((bar)=>bar.textContent.toLowerCase().includes(q)||bar.title.toLowerCase().includes(q));row.previousElementSibling.style.display=visible?'':'none';row.style.display=visible?'':'none'}};</script>`;
    fs.writeFileSync(output, page); return { summary: { source: tracePath, raw_duration_events: raw.length, rendered_events: events.length, lane_count: lanes.length, start_us: start, end_us: end, duration_us: duration }, artifacts: [{ type: 'html', path: output }] };
  }
  if (name === 'render_python_model_graph') {
    const sourcePath = resolve(String(args.source_path)); if (!fs.existsSync(sourcePath)) throw new Error(`source_path does not exist: ${sourcePath}`);
    const outputName = String(args.output_name || path.basename(sourcePath, '.py')); fs.mkdirSync(ARTIFACTS, { recursive: true });
    return renderPythonModelGraph(sourcePath, path.join(ARTIFACTS, `${outputName}.html`));
  }
  if (name === 'generate_operator_scaffold') {
    const operator = String(args.operator_name); const safe = operator.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '') || 'custom_operator'; const outputDir = resolve(String(args.output_dir || path.join(ARTIFACTS, safe))); fs.mkdirSync(outputDir, { recursive: true });
    const files = { [`${safe}.py`]: `"""${operator} scaffold generated by Model Visualization Kit."""\n\ndef ${safe}(x):\n    raise NotImplementedError("Implement ${operator} after reviewing the contract.")\n`, [`test_${safe}.py`]: `def test_${safe}_placeholder():\n    # Replace with shape, dtype, boundary, and numerical golden tests.\n    assert True\n`, 'visualization.json': JSON.stringify({ operator, assets: ['operator-card', 'tensor-contract'], checks: ['shape', 'dtype', 'layout', 'precision', 'boundary'] }, null, 2) };
    for (const [filename, content] of Object.entries(files)) fs.writeFileSync(path.join(outputDir, filename), content); return { operator, framework: args.framework || 'unknown', output_dir: outputDir, files: Object.keys(files).map((file) => path.join(outputDir, file)), next_steps: ['补充输入输出契约', '实现正确性 Golden', '运行最小编译检查', '再进行真实性能 profiling'] };
  }
  throw new Error(`unknown tool: ${name}`);
}

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on('line', (line) => { if (!line.trim()) return; let request; try { request = JSON.parse(line); } catch (error) { return send(null, null, { code: -32700, message: `invalid JSON: ${error.message}` }); } try {
  if (request.method === 'initialize') return send(request.id, { protocolVersion: '2024-11-05', capabilities: { tools: { listChanged: false } }, serverInfo: { name: 'model-visualization-kit', version: '0.1.0' } });
  if (request.method === 'notifications/initialized') return;
  if (request.method === 'tools/list') return send(request.id, { tools });
  if (request.method === 'tools/call') return send(request.id, wrapped(call(request.params.name, request.params.arguments || {})));
  return send(request.id, null, { code: -32601, message: `method not found: ${request.method}` });
} catch (error) { send(request.id, null, { code: -32000, message: error.message }); } });
