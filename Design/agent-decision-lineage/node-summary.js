/* Editorial summaries are projections of the existing evidence model, never new run facts. */
window.NodeSummary=(()=>{
const summaries={
'e1':['数值稳定行归约','为二维 FP16 输入生成沿最后一轴归约的 Softmax。',['动态 B∈[1,1024]、N∈[1,8192]。','输入输出形状与 dtype 保持一致。'],'[B,N] → [B,N]'],
'd-kb':['知识选择','识别逐行归约、尾块处理和 Buffer 生命周期知识，供后续设计使用。',['行归约知识支持 Vector / VF；Buffer 生命周期支持双缓冲设计。','Online Softmax 经容量评估后未触发跨 Tile 方案。']],
'd-api':['VF API 适配','将精度与寄存器覆盖约束转化为可实现的 VF 设计。',['FP16 reduce_sum 按 FP16 累加，长轴存在精度风险，因此采用 FP32 sum。','ZERO / ONE 各覆盖半个寄存器，组合后覆盖完整输入。']],
'd-constraint':['A5 / 950 · 硬件约束','在初始设备假设下评估 UB 容量、精度与尾块约束。',['A5 / 950 最初来自默认假设，并非用户确认。','约束分析与备选讨论不等同于编译或性能实验。']],
'd-vf':['Vector / VF','采用单 Vector Module，在同一 Kernel 中完成三趟行归约。',['每行独立，无跨核归约或 Cube 计算。','整行可装入 UB，无须跨 Tile 保留归约状态。'],'Max → Exp+Sum → Exp+Divide'],
'd-precision':['Mixed Precision','保留 FP16 输入与 max，以 FP32 完成 exp、sum 和 div，再缩窄输出。',['max 是输入值之一，可保留原 FP16 表示。','长序列求和采用 FP32，降低 FP16 累加风险。'],'FP16 → FP32 → FP16'],
'd-tiling':['Single Tile · 3×8192','整行容纳于一个 UB Tile，未触发跨 Tile 流式归约。',['TILE_ROWS=3，MAX_N=8192。','每槽 48 KiB，四槽共 192 KiB，未超过 248 KiB。'],'3 行 × 8192 列 / Tile'],
'd-buffer':['UB Buffer','以四个 48 KiB Slot 支持输入、输出各双槽轮转。',['192 / 248 KiB，使用率 77.4%，剩余 56 KiB。','双缓冲是资源设计，不能据此断言实际发生重叠执行。'],'memory'],
'd-schedule':['Scheduling','按行块跨核 Striding，在单次 Kernel launch 中完成计算。',['block_dim=min(vector_core_num, ceil(B/3))。','同一 VF 内完成三趟计算，m/s 不跨行、不跨核。'],'Core k → Tile k, k+block_dim, …'],
'i-kernel':['PyPTO Kernel','生成单 Vector Kernel，完成三趟 Softmax 计算。',['softmax_rows_vf 承载归约与精度转换。','wrapper 组织 Load、VF 调用和 Store；这是静态结构。'],'Load → Max → Exp+Sum → Exp+Divide → Store'],
'x-verify':['Correctness PASS','报告记录 8 个用例通过正确性验证。',['相对 CPU FP32 golden，atol=1e−3 / rtol=1e−3。','覆盖多组形状与尾块；未在本机重测，也不代表穷尽合法输入。'],'8 / 8 PASS'],
'x-base':['Baseline','正式采集 round_001 为优化前的性能基线。',['[32,4096] Kernel Duration 为 5.926 μs。','后续对比采用相同 Shape 的正式采集结果。'],'5.926 μs · [32,4096]'],
'f-mask':['Finding · 重复 Mask','完整寄存器循环内反复生成 Mask，被识别为潜在开销来源。',['N=4096 时每行有 32 个完整寄存器块。','静态源码与 Profiling 信号支持优化假设，不是逐指令因果测量。'],'循环内 update_mask → 潜在重复开销'],
'h-mask':['Mask Lifting 假设','将可复用的 Full Mask 提升到循环外，减少重复操作。',['Full-register mask 在寄存器迭代间保持一致。','收益需要由实现与正式性能采集检验。']],
'x-mask':['Mask Lifting','将 Full Mask 创建移到循环外复用，减少重复操作。',['保持冻结三趟结构；尾寄存器由 Empty-tail-loop 处理。','[32,4096] 从 5.926 降至 4.946 μs；小用例存在回退。'],'Before：每轮创建 Mask\nAfter：预先创建并复用'],
'x-if':['Runtime if','尾寄存器实现尝试 Runtime if，遇到 C++ 作用域/类型错误。',['报告记录 __inline_0_tail_mreg_fp16_2 类型不匹配。','它是 Mask Lifting 的尾处理尝试，随后被空尾循环替代；无独立性能收益。'],'Runtime if → 编译失败 → Empty-tail-loop'],
'x-tail':['Empty-tail-loop','用空尾循环替代 Runtime if，处理尾寄存器。',['N%128=0 时范围为空；存在尾部时处理一个寄存器。','Accepted，属于 Mask Lifting；没有独立性能归因。'],'尾部不存在：0 次循环\n尾部存在：1 次循环'],
'x-final':['Final','正式采集显示大用例获益，小用例出现回退。',['[32,4096] Kernel Duration：5.926 → 4.946 μs。','[1,128]：3.250 → 3.428 μs；Golden E2E 目标比值不等于本次加速比。'],'performance'],
'd-stop':['Target / Stop','正确性和默认性能目标满足，因此本轮优化停止。',['报告记录 8/8 正确性通过、P0 默认目标满足。','停止不代表全局最优，仍有三个方向未尝试。'],'目标满足 · 保留未探索方向'],
'x-rows':['TILE_ROWS Adjustment','现有设计内可以继续调参，但本轮性能已满足目标。',['原始状态：Not Attempted；原因分类“目标已达成”为产品侧归纳。','继续条件：在 DESIGN §9 允许范围另立实验，重做正确性和性能采集。'],'当前性能达标，未继续调参'],
'x-reuse':['Exp Result Reuse','复用 Exp 结果需要改变当前冻结的三趟计算结构。',['原始状态：Not Attempted；原因分类“需要改设计”为产品侧归纳。','继续条件：重新评估 Stage 3 设计与 FP32 UB staging，再实现并测量。'],'Pass 2 → FP32 UB staging → Pass 3'],
'x-fusion':['Pass Fusion','融合计算需要改变现有算法，不能在冻结设计下直接尝试。',['原始状态：Not Attempted；原因分类“需要改设计”为产品侧归纳。','继续条件：重新评估 DESIGN §0 的三趟结构、Module 语义和精度，再进行实验。'],'尚未测试 · 无性能收益结论']
};
const deep=new Set(['d-vf','d-precision','d-tiling','d-buffer','d-schedule','i-kernel','x-verify','x-base','f-mask','h-mask','x-mask','x-if','x-tail','x-final','d-stop','x-rows','x-reuse','x-fusion']);
const make=(tag,text,cls)=>{const n=document.createElement(tag);if(text)n.textContent=text;if(cls)n.className=cls;return n};
function render(id){const node=LINEAGE_DATA.nodes.find(n=>n.id===id),config=summaries[id]||[node.title,node.reason,[]];
 const host=document.getElementById('node-summary');host.replaceChildren();document.getElementById('inspector-title').textContent=config[0];
 const status={adopted:'Accepted · 已采用',attempted_failed:'Rejected · 尝试失败',not_attempted:'Not Attempted · 未尝试',validated:'报告记录 · 已验证',recorded:'历史资料记录',considered_not_adopted:'已评估 · 未采用',completed:'阶段完成'}[node.status]||LINEAGE_DATA.statusLabels[node.status];
 host.append(make('p','Stage '+node.stage+' · '+status,'summary-identity'),make('p',config[1],'summary-takeaway'));
 const list=make('ul',null,'summary-reasons');for(const reason of config[2])list.append(make('li',reason));host.append(list);
 if(config[3]){const preview=make('div',null,'summary-preview');if(config[3]==='memory'){preview.append(make('strong','192 / 248 KiB · 77.4%'));const slots=make('div',null,'summary-slots');for(const label of ['In 0','In 1','Out 0','Out 1'])slots.append(make('span',label+' · 48 KiB'));preview.append(slots)}else if(config[3]==='performance'){preview.append(make('strong','5.926 → 4.946 μs'),make('p','−16.54% · 1.198× · [32,4096]'))}else preview.textContent=config[3];host.append(preview)}

 const refs=node.evidenceRefs.map(ref=>({...ref,support:config[2][0]||node.reason}));for(const relation of window.EXPLORE_DESIGN_RELATIONS.filter(r=>r.sourceId===id||r.targetId===id))for(const ref of relation.evidenceRefs)if(!refs.some(r=>r.sourceId===ref.sourceId&&r.location===ref.location))refs.push({...ref,support:relation.reason});
 const details=make('details',null,'summary-evidence');details.append(make('summary','依据 · '+[...new Set(refs.map(r=>r.sourceId))].slice(0,2).join(' / ')+' · '+refs.length+' 项证据'));
 for(const ref of refs){const item=make('div',null,'summary-evidence-item');item.append(make('strong',({'DESIGN.md':'设计选择与资源预算','PERFORMANCE_REPORT.md':'正式报告与候选记录','KB_SELECTION.json':'知识选择记录','EXPLORE_REPORT.md':'探索结论与约束','test_softmax.py':'实现映射'}[ref.sourceId]||'关联依据')),make('p',ref.support||node.reason),make('small',ref.sourceId+' · '+ref.location));const source=LINEAGE_DATA.sources.find(s=>s.id===ref.sourceId);const a=make('a','查看原始资料');a.href=source?.path||'evidence/'+ref.sourceId;a.target='_blank';a.rel='noopener';item.append(a);details.append(item)}host.append(details);
}
return {render};
})();

window.NodeTooltip=(()=>{
 let timer;
 const tip=()=>document.getElementById('node-tooltip');
 function hide(){clearTimeout(timer);tip().hidden=true}
 function delay(){clearTimeout(timer);timer=setTimeout(hide,160)}
 function show(item){clearTimeout(timer);const id=item.dataset.nodeId;if(!LINEAGE_DATA.nodes.some(n=>n.id===id))return;NodeSummary.render(id);const t=tip();t.hidden=false;const r=item.getBoundingClientRect();const x=Math.min(innerWidth-t.offsetWidth-12,Math.max(12,r.right+12));const y=Math.max(12,Math.min(innerHeight-t.offsetHeight-12,r.top));t.style.left=x+'px';t.style.top=y+'px';item.setAttribute('aria-describedby','node-tooltip')}
 function attach(svg){hide();for(const item of svg.querySelectorAll('[data-node-id]')){item.addEventListener('pointerenter',()=>show(item));item.addEventListener('pointerleave',delay);item.addEventListener('focus',()=>show(item));item.addEventListener('blur',delay)}}
 document.addEventListener('DOMContentLoaded',()=>{tip().addEventListener('pointerenter',()=>clearTimeout(timer));tip().addEventListener('pointerleave',delay)});
 window.addEventListener('resize',hide);document.addEventListener('scroll',event=>{if(!tip().contains(event.target))hide()},true);
 return {attach,hide};
})();
