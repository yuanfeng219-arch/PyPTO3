const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const state = { mode: 'overview', validated: false, diff: 'pto', node: 'tensor', toastTimer: null };

const inspectorData = {
  correctness: { title: 'Correctness', tone: 'danger', summary: 'Failed · max_abs_diff', details: [['状态','Failed'],['指标','max_abs_diff'],['值','1.23e−02'],['阈值','1.00e−05'],['采集','Task 142 post-op']], preview: '<div>reference: 0.18750</div><div>compiled:  0.19922</div><div class="err">abs diff: 1.17e−02</div>', actions: ['查看 Tensor diff','定位首个 Task'] },
  tensor: { title: 'First divergent tensor', summary: 'layers.23.mlp.fc1_out', details: [['名称','layers.23.mlp.fc1_out'],['Shape','[1, 32, 4096]'],['Dtype','bf16'],['First diff idx','[0, 17, 2048]'],['max_abs_diff','1.23e−02'],['采集于','Task 142 (post-op)']], preview: '<div>idx                 ref       compiled</div><div>[0,17,2047]       0.03125    0.03125</div><div>[0,17,2048]       0.18750    0.19922 <span class="err">←</span></div><div>[0,17,2049]      −0.12500   −0.11328</div>', actions: ['Replay from divergence','Compare tensors','Trace to source'] },
  task: { title: 'Task 142', summary: 'decode_layer · post-op', details: [['Task ID','142'],['类型','fwd'],['阶段','decode_fwd_layers'],['开始','1,071.88 ms'],['耗时','7.54 ms'],['状态','关联异常']], preview: '<div>inputs: 2 tensors</div><div>outputs: 1 tensor</div><div>successor: Task 143</div>', actions: ['打开 Task 依赖图','查看输入输出'] },
  kernel: { title: 'kernel_aiv_07.cpp', summary: 'line 318 · write_global', details: [['Kernel ID','K-07-aiv'],['Backend','AIV'],['Device','Ascend 910B'],['Source','kernels/aiv/'],['位置','318'],['调用自','Task 142']], preview: '<div>316  // fused attention + mlp epilogue</div><div>317  auto o = rms_norm(accum, gamma);</div><div class="err">318  write_global(o, out_ptr);</div>', actions: ['打开源码定位','查看编译前 PTO'] },
  memory: { title: 'Memory hint', summary: 'Bank conflict · Medium', details: [['类型','Bank conflict'],['等级','Medium'],['位置','smem[0x3f80]'],['详情','32-way conflict'],['置信度','0.72']], preview: '<div>SM 0  compute ▮▮▮ memory ▮▮</div><div>SM 1  compute ▮▮ amber ▮▮</div><div>hint: apply swizzle</div>', actions: ['查看 Swimlane','应用 hint 并复现'] },
  artifacts: { title: 'Generated artifacts', summary: 'fc1_fwd_142.*', details: [['PTO','fc1_fwd_142.pto'],['C++','fc1_fwd_142.cpp'],['Headers','fc1_fwd_142.h'],['Built','2026-06-25 18:49:45'],['Hash','4c8f21…']], preview: '<div>ptoas/ fc1_fwd_142.pto</div><div>kernels/aiv/ kernel_aiv_07.cpp</div><div>orchestration/ module.so</div>', actions: ['打开产物目录','复制路径'] },
  trace: { title: 'Memory / Swimlane', summary: 'bank conflict at 1,079 ms', details: [['来源','merged_swimlane_*.json'],['事件','50,556'],['时间点','1,079.42 ms'],['关联','Task 142 → AIV 07'],['状态','已对齐']], preview: '<div>time →  1,078  1,079  1,080</div><div>AIV07   ▮▮ amber ▮▮ ▮▮</div><div>AICPU   ▮ idle ▮ compute</div>', actions: ['打开时间线','导出事件片段'] }
};

function showToast(message) {
  const toast = $('#toast');
  toast.textContent = message; toast.classList.add('show');
  clearTimeout(state.toastTimer); state.toastTimer = setTimeout(() => toast.classList.remove('show'), 2800);
}

function setMode(mode) {
  state.mode = mode;
  $$('.nav-item[data-mode]').forEach(btn => btn.classList.toggle('active', btn.dataset.mode === mode));
  $$('.mode-panel').forEach(panel => panel.classList.toggle('active', panel.dataset.panel === mode));
  window.location.hash = mode;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function updateValidated() {
  state.validated = true;
  $('#run-status').textContent = '发现 1 处首个偏差';
  $('#correctness-step').textContent = 'Failed · max_abs_diff 1.23e−02';
  $('#correctness-copy').textContent = '首个偏差位于 layers.23.mlp.fc1_out / Task 142';
  $('#correctness-label').textContent = '1 处偏差';
  $('#correctness-gate').classList.remove('pending'); $('#correctness-gate').classList.add('warn');
  $('#correctness-gate .gate-mark').textContent = '!';
  $('#gate-summary').textContent = '需分析 1 处偏差';
  $('#gate-summary').style.color = 'var(--red)';
  $('#gate-summary').style.borderColor = '#75373b';
  $$('[data-action="validate"]').forEach(button => {
    if (button.classList.contains('button')) button.textContent = '进入异常分析';
    const title = button.querySelector('strong');
    const copy = button.querySelector('small');
    if (title) title.textContent = '进入异常分析';
    if (copy) copy.textContent = '从首个偏差下钻到 Tensor、Task 和 Kernel';
  });
  showToast('正确性检查完成：发现首个偏差，已准备好分析上下文');
}

function renderInspector(key = state.node) {
  const data = inspectorData[key] || inspectorData.tensor;
  state.node = key;
  $('#inspector-title').textContent = data.title;
  $('#inspector-content').innerHTML = `<div class="inspector-block"><h3>${data.summary}</h3><dl class="detail-list">${data.details.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl></div><div class="inspector-block"><h3>Evidence preview</h3><div class="tensor-preview">${data.preview}</div><div class="inspector-actions">${data.actions.map((a, i) => `<button class="button ${i === 0 ? 'primary' : 'secondary'}" data-action="inspector-action">${a} ↗</button>`).join('')}</div></div>`;
  $$('.evidence-node').forEach(node => node.classList.toggle('selected', node.dataset.node === key));
}

const diffs = {
  pto: [['112',' @stage decode_fwd {',''],['113',' // load q, k, v',''],['114',' q <- global.load<float16>(Q_ptr, idx_q)',''],['115',' k <- global.load<float16>(K_ptr, idx_k)',''],['116',' v <- global.load<float16>(V_ptr, idx_v)','del'],['116','+ (q, k) <- global.load.v2<float16>(QK_ptr, idx_qk)','add'],['117','+ v <- global.async_copy<float16>(V_ptr, idx_v)','add'],['118',' score <- dot(q, k)',''],['119',' score <- score * scale',''],['120',' score <- softmax(score)','del'],['120','+ score <- online_softmax(score)','add'],['121','+ out <- fused_matmul_softmax_v(score, v)','add'],['122',' store(out, Out_ptr, idx_o)','']],
  cpp: [['316',' // fused attention + mlp epilogue',''],['317',' auto o = rms_norm(accum, gamma);',''],['318',' write_global(o, out_ptr);','del'],['318','+ write_global_swizzled(o, out_ptr, smem);','add'],['319',' barrier();',''],['320',' return;',''],['',' // candidate adds async copy + swizzle','note']],
  hint: [['PH001',' [PH001] memory bank conflict detected','del'],['PH001','+ [PH001] reduced after swizzle candidate','add'],['',' location: smem[0x3f80] · AIV07',''],['',' severity: medium · confidence: 0.72',''],['PH002',' [PH002] right memory space reaches 100%','del'],['PH002','+ [PH002] peak reduced from 18.62 GB to 17.21 GB','add'],['',' action: validate with Task 142 replay','note']]
};

function renderDiff(type = state.diff) {
  state.diff = type;
  $('#code-diff').innerHTML = (diffs[type] || diffs.pto).map(([line, text, kind]) => `<div class="code-line ${kind}"><span class="ln">${line}</span><span class="sign">${kind === 'add' ? '+' : kind === 'del' ? '−' : ' '}</span><span>${text}</span></div>`).join('');
  $$('.diff-tab').forEach(tab => tab.classList.toggle('active', tab.dataset.diff === type));
}

function renderExplorer(type = 'tasks') {
  $$('.explorer-tab').forEach(tab => tab.classList.toggle('active', tab.dataset.explorer === type));
  const body = $('#explorer-body');
  if (!body) return;
  if (type === 'tasks') {
    body.innerHTML = `<div class="trace-table"><div class="trace-header"><span>Task / Kernel</span><span>阶段</span><span>开始</span><span>耗时</span><span>状态</span></div><button class="trace-row selected" data-task="8589934600"><span><b>Task 8589934600</b><small>q_proj · func_id 4 · AIC</small></span><span>decode_fwd</span><span>00:00:00.412</span><span>0.48 ms</span><em class="trace-state pass">完成</em></button><button class="trace-row" data-task="8589934682"><span><b>Task 8589934682</b><small>fa_work_build · func_id 9 · AIV</small></span><span>attention</span><span>00:00:00.763</span><span>1.04 ms</span><em class="trace-state pass">完成</em></button><button class="trace-row" data-task="8589934925"><span><b>Task 8589934925</b><small>online_softmax · func_id 24 · AIV</small></span><span>attention</span><span>00:00:01.017</span><span>2.13 ms</span><em class="trace-state warn">提示</em></button><button class="trace-row danger-row" data-task="8589935016"><span><b>Task 8589935016</b><small>down_proj · func_id 36 · AIC</small></span><span>mlp</span><span>00:00:01.079</span><span>7.54 ms</span><em class="trace-state fail">首个偏差</em></button></div><div class="mini-lanes"><div class="lane-head"><span>Core swimlane</span><small>merged_swimlane_20260625_185006.json</small></div><div class="lane"><label>AIC 00</label><div class="lane-track"><i style="left:3%;width:16%"></i><i style="left:27%;width:23%"></i><i class="hot" style="left:68%;width:24%"></i></div></div><div class="lane"><label>AIC 01</label><div class="lane-track"><i style="left:11%;width:25%"></i><i style="left:46%;width:14%"></i><i style="left:72%;width:18%"></i></div></div><div class="lane"><label>AIV 07</label><div class="lane-track"><i style="left:6%;width:22%"></i><i style="left:36%;width:19%"></i><i class="hot" style="left:67%;width:27%"></i></div></div><div class="lane"><label>AICPU</label><div class="lane-track"><i class="cpu" style="left:18%;width:16%"></i><i class="cpu" style="left:49%;width:10%"></i><i class="cpu" style="left:79%;width:13%"></i></div></div><div class="lane-axis"><span>0 ms</span><span>500</span><span>1,000</span><span>1,500</span><span>2,000</span></div></div>`;
  } else if (type === 'kernels') {
    const kernelRows = [{name:'copy_hidden',core:'AIV',dir:'aiv',calls:1,ms:.22},{name:'q_proj',core:'AIC',dir:'aic',calls:16,ms:1.82},{name:'fa_fused_aic',core:'AIC',dir:'aic',calls:1,ms:4.76},{name:'fa_fused_aiv',core:'AIV',dir:'aiv',calls:1,ms:5.13},{name:'online_softmax',core:'AIV',dir:'aiv',calls:16,ms:2.13},{name:'down_proj',core:'AIC',dir:'aic',calls:16,ms:7.54},{name:'copy_out',core:'AIV',dir:'aiv',calls:1,ms:.31}];
    body.innerHTML = `<div class="trace-table"><div class="trace-header"><span>Kernel</span><span>Core</span><span>调用次数</span><span>平均耗时</span><span>状态</span></div>${kernelRows.map(row=>`<button class="trace-row" data-task="kernel-${row.name}"><span><b>${row.name}</b><small>kernels/${row.dir}/${row.name}.cpp</small></span><span>${row.core}</span><span>${row.calls}</span><span>${row.ms} ms</span><em class="trace-state ${row.name==='down_proj'?'fail':'pass'}">${row.name==='down_proj'?'关注':'稳定'}</em></button>`).join('')}</div><div class="mini-lanes"><div class="lane-head"><span>Kernel occupancy</span><small>func_id → callable name</small></div><div class="occupancy-list">${['AIC · fa_fused_aic','AIV · fa_fused_aiv','AIV · online_softmax','AIC · down_proj'].map((name,i)=>`<div class="occupancy-row"><span>${name}</span><b><i style="width:${[74,81,58,92][i]}%"></i></b><em>${[74,81,58,92][i]}%</em></div>`).join('')}</div></div>`;
  } else {
    body.innerHTML = `<div class="trace-table"><div class="trace-header"><span>Pass snapshot</span><span>变换</span><span>输入</span><span>输出</span><span>状态</span></div>${['00_frontend.py','01_after_InlineFunctions.py','02_after_UnrollLoops.py','03_after_CtrlFlowTransform.py','04_after_ConvertToSSA.py','18_after_LowerTile.py'].map((name,i)=>`<button class="trace-row" data-task="pass-${i}"><span><b>${name}</b><small>passes_dump/${name}</small></span><span>${['frontend','inline','unroll','control-flow','ssa','lowering'][i]}</span><span>${[421,408,391,360,342,256][i]} nodes</span><span>${[521,508,490,462,431,388][i]} nodes</span><em class="trace-state pass">已落盘</em></button>`).join('')}</div><div class="mini-lanes"><div class="lane-head"><span>IR size over passes</span><small>derived from pass snapshots</small></div><div class="pass-bars">${[100,98,94,89,82,74,61].map((v,i)=>`<div><label>${String(i).padStart(2,'0')}</label><b style="width:${v}%"></b><em>${v}%</em></div>`).join('')}</div></div>`;
  }
}

function updateChart(type) {
  const isThroughput = type === 'throughput';
  $$('.chart-toggle').forEach(btn => btn.classList.toggle('active', btn.dataset.chart === type));
  $('#chart-summary').textContent = isThroughput ? '吞吐 +16.50%' : 'P95 −15.25%';
  const base = $$('.spark-line.baseline b'); const cand = $$('.spark-line.candidate b');
  const bVals = isThroughput ? [62,58,64,56,60,54,59,52,57,50,55,51] : [42,48,44,58,52,63,55,68,61,72,66,70];
  const cVals = isThroughput ? [71,76,74,79,77,84,80,87,83,90,86,89] : [31,36,34,39,36,45,40,48,43,52,47,50];
  base.forEach((el,i)=>el.style.height=`${bVals[i]}%`); cand.forEach((el,i)=>el.style.height=`${cVals[i]}%`);
}

document.addEventListener('click', event => {
  const modeButton = event.target.closest('[data-mode]');
  if (modeButton) { event.preventDefault(); setMode(modeButton.dataset.mode); return; }
  const scrollButton = event.target.closest('[data-scroll]');
  if (scrollButton) { $('#artifacts').scrollIntoView({ behavior: 'smooth', block: 'center' }); return; }
  const diffButton = event.target.closest('[data-diff]');
  if (diffButton) { renderDiff(diffButton.dataset.diff); return; }
  const explorerButton = event.target.closest('[data-explorer]');
  if (explorerButton) { renderExplorer(explorerButton.dataset.explorer); return; }
  const chartButton = event.target.closest('[data-chart]');
  if (chartButton) { updateChart(chartButton.dataset.chart); return; }
  const detailButton = event.target.closest('[data-detail]');
  if (detailButton) { $$('.detail-tab').forEach(btn => btn.classList.toggle('active', btn.dataset.detail === detailButton.dataset.detail)); showToast(`${detailButton.textContent} 视图已切换`); return; }
  const taskRow = event.target.closest('[data-task]');
  if (taskRow) { $$('.trace-row').forEach(row => row.classList.toggle('selected', row === taskRow)); showToast(`已选中 ${taskRow.dataset.task}，可继续下钻到关联证据`); return; }
  const nodeButton = event.target.closest('[data-node]');
  if (nodeButton) { renderInspector(nodeButton.dataset.node); return; }
  const action = event.target.closest('[data-action]');
  if (!action) return;
  const type = action.dataset.action;
  if (type === 'validate') { if (!state.validated) updateValidated(); else setMode('analysis'); }
  if (type === 'reproduce') showToast('已复制复现环境：seed 13371337 · Ascend 910B · PyPTO 3.2.1');
  if (type === 'replay') showToast('已创建 Replay：将从 Task 142 重新采集 Tensor 与 Swimlane');
  if (type === 'promote') showToast('Candidate 已标记为待签发，建议先确认编译时间回归');
  if (type === 'rollback') showToast('已保留 Baseline，Candidate 仍可从实验记录恢复');
  if (type === 'export' || type === 'manifest') showToast(type === 'manifest' ? 'Manifest 生成中：身份、哈希和关系索引将被保留' : '报告已准备导出（Demo）');
  if (type === 'artifact') showToast(`${action.dataset.title}：已定位到产物索引`);
  if (type === 'download-slice') showToast('已导出 4 个关联 Task 与 1,079 ms 时间窗口（Demo）');
  if (type === 'pin') showToast('已将首个偏差证据固定到调试报告');
  if (type === 'copy') showToast('复现命令已复制到剪贴板（Demo）');
  if (type === 'reset') { renderInspector('tensor'); showToast('已重置到首个偏差节点'); }
  if (type === 'inspector-action') showToast('已打开关联证据（Demo）');
});

$('#task-search')?.addEventListener('input', event => {
  const term = event.target.value.trim().toLowerCase();
  $$('#explorer-body .trace-row').forEach(row => row.hidden = term && !row.textContent.toLowerCase().includes(term));
});

window.addEventListener('hashchange', () => { const mode = window.location.hash.slice(1); if (['overview', 'compare', 'analysis'].includes(mode)) setMode(mode); });

renderDiff('pto'); renderInspector('tensor');
const initialMode = window.location.hash.slice(1);
if (['overview', 'compare', 'analysis'].includes(initialMode)) setMode(initialMode);
