/* Shared diagnostic evidence, without a second timeline. Main workarea owns all navigation. */
window.InvestigationReport=function(A){
'use strict';
const U=InvestigationUI,{esc,icon,badge,card,table,empty,notice,code}=U,D=INVESTIGATION_DATA;
const $=id=>document.getElementById(id);
const STEPS=[['识别异常','activity'],['定位 Task','search'],['直接原因','clock'],['编译变化','branch'],['调优方案','sliders'],['实验对比','flask']];
const state=new Proxy({},{
 get:(_,key)=>key==='caseKey'?A.activeCase()?.preset:key==='step'?A.activeCase()?.step:key==='tab'?A.activeCase()?.tab||'lifecycle':A.activeCase()?.report?.[key],
 set:(_,key,value)=>{(A.activeCase().report??={})[key]=value;return true;}
});
const meta=()=>{const m=A.workflow.meta[state.caseKey];return {...m,id:'INV–0'+m.number,waitIndex:m.wait,label:D.cases[state.caseKey].ranks[0].tasks[m.wait].name,desc:state.caseKey==='a2a'?'publish 并行度':'通信分块粒度',hypothesis:m.mechanism};};
const data=()=>D.cases[state.caseKey],tasks=(rank=0)=>data().ranks[rank].tasks,wait=(rank=0)=>tasks(rank)[meta().waitIndex],span=t=>t.end-t.start;
const num=(n,d=2)=>Number(n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}),us=n=>n==null?'未采集':num(n)+' µs';
const tone=s=>({CONFIRMED:'success',SUPPORTED:'info',CANDIDATE:'warning',UNKNOWN:'neutral'}[s]||'neutral');
const statusBadge=s=>s?badge(s,tone(s)):'';
const button=(label,action,classes='',attrs='')=>U.button(label,({rank:'slice-rank',align:'slice-align',cores:'slice-cores',task:'report-task','goto-cause':'report-cause','create-experiment':'experiment'}[action]||action),classes,attrs);
const sourceDisclosure=(key,label='关联源码')=>{const s=D.sources[key];return s?`<details class="source-disclosure"><summary>${icon('file')} ${esc(label)} · ${esc(s.file.split('/').at(-1))}:${s.line}</summary><div class="source-disclosure__meta"><code>${esc(s.file)}:${s.line}</code><span>归档快照</span></div>${code(s)}</details>`:'';};
const showTask=(index,rank,logical=true)=>A.selectEvidenceTask(rank,tasks(rank)[index],logical);
let tooltipHandle=null;
function heading(title,desc,status=''){return `<div class="step-heading">${icon(STEPS[state.step-1][1])}<div><h2>${title}</h2><p>${desc}</p></div>${statusBadge(status)}</div>`;}
function answer(title,text,status='SUPPORTED'){return `<div class="answer-card"><div class="answer-title">${statusBadge(status)}${title}</div><p>${text}</p></div>`;}
function metric(label,value,sub){return `<div class="card-demo ui-card metric"><span>${label}</span><strong>${value}</strong><p>${sub}</p></div>`;}
function stepOne(){
 const w=wait(),w1=wait(1);
 return heading('先确认现象和影响范围','Rank 0 的等待事件明显长于 Rank 1；空白区间为何形成仍需继续排查。','CONFIRMED')+
 `<div class="metric-grid metric-grid-2">${metric('Rank 0 · 等待事件',num(span(w))+' <small>µs</small>','Worker View 原始事件')}${metric('Rank 1 · 同阶段',num(span(w1))+' <small>µs</small>','仅作跨 Rank 观测对照')}</div>`+
 card('调查范围',`<p>Rank ${A.activeCase().rank} · ${A.workflow.us(A.activeCase().time_range[0])} — ${A.workflow.us(A.activeCase().time_range[1])} · ${A.activeCase().selected_lanes.length} 条泳道</p><p>口径：等待事件跨度与 AIC 空闲时长分别统计。跨 Rank 数据使用各自 Run 内时间。</p>${button('定位调查范围','focus-case','btn-sm')}${button('跨 Rank 对照','compare-ranks','btn-ghost btn-sm')}`)+notice('建议检查同步等待和 Rank 长尾','当前 Trace 未记录 Ready 状态。先检查任务生命周期，再判断空白区间内是否存在可运行任务。','info',button('检查直接原因 →','goto-cause','btn-ghost'));
}
function stepTwo(){
 return heading('找到异常前后的任务','依据 Task ID、Trace 中记录的前后依赖和源码位置建立关联。','SUPPORTED')+
 card('相关 Task',table(['任务 / ID','Rank 0 跨度','Rank 1 跨度','关系证据'],tasks().map((t,i)=>[
 button(esc(t.name)+'<br><span class="muted">'+t.id+'</span>','task','btn-ghost task-link',`data-task="${i}" data-task-rank="0"`),button(us(span(t)),'task','btn-ghost btn-sm',`data-task="${i}" data-task-rank="0"`),button(us(span(tasks(1)[i])),'task','btn-ghost btn-sm',`data-task="${i}" data-task-rank="1"`),esc(t.fanin)
 ]),'多核任务跨度：首个 Worker 开始至最后一个 Worker 完成。'))+
 notice('Execute / Block Graph 暂不可用','当前 Trace 未记录 rootHash、callOpMagic 和 leafHash。现阶段可定位 Task 与源码 scope。','info')+sourceDisclosure(meta().source,'关联源码');
}
function lifecycle(){
 const target=tasks()[5],first=target.event;
 const stages=[['创建',null],['依赖就绪',null],['进入队列',null],['下发',target.dispatch],['实际开始',first.ts],['执行完成',first.ts+first.dur]];
 return card('下游 Task 生命周期',`<div class="row-between"><code>${esc(target.name)} · ${target.id}</code>${badge('首个启动 Worker','neutral')}</div><div class="lifecycle">${stages.map(([a,v])=>`<div class="lifecycle-event ${v!=null?'is-known':''}"><span class="lifecycle-circle">${v==null?'?':'✓'}</span><strong>${a}</strong><code class="${v==null?'muted':'success'}">${v==null?'未采集':num(v)}</code></div>`).join('')}</div>`+
 notice('还不能判断任务是否已经就绪','Trace 没有记录任务创建、依赖就绪和进入队列的时间；目前只能看到下发、开始和完成。')+
 `<div class="row-between"><span class="muted">时间单位 µs · 完成时间对应同一 Worker</span>${button('查看此 Task 的事件','task','btn-ghost btn-sm','data-task="5" data-task-rank="0"')}</div>`,badge('Q1 · UNKNOWN'));
}
function resources(){
 const items=[['上游依赖 / 同步','CANDIDATE','Trace 和源码都出现了等待操作；仍需补充其他 Rank 的到达时间和依赖解除时间。'],['AIC / AIV 核资源','UNKNOWN','没有记录当时哪些核可用、哪些核被占用以及并发上限。'],['内存 / DMA 资源','UNKNOWN','没有记录 UB、L1、L0 占用和 DMA 队列状态。'],['调度队列','UNKNOWN','没有记录就绪队列、优先级和未运行原因。']];
 return card('依赖与资源检查',`<div class="resource-grid">${items.map(([a,b,c])=>card(a,'<p>'+c+'</p>',statusBadge(b))).join('')}</div>`+notice('当前判断','“等待其他 Rank 的数据”值得优先检查，但还不能排除核、内存或调度资源不足。','info'));
}
function upstream(){
 const w=wait(),total=span(w),r1=wait(1);
 return card('上游耗时分解',answer('只知道等待事件的总时长','Trace 记录了内核执行时长和本地准备时长，但没有把生产者耗时拆成计算、搬运、同步与排队。','UNKNOWN')+
 `<div class="bar-breakdown" aria-label="Wait kernel 与 local setup"><span style="width:${w.kernel/total*100}%;background:var(--warning)"></span><span style="width:${w.setup/total*100}%;background:var(--primary)"></span></div>`+
 table(['已记录区间','Rank 0','Rank 1','解释边界'],[
 ['等待任务总时长',us(total),us(span(r1)),'直接来自 Worker 事件'],
 ['内核执行时长',us(w.kernel),us(r1.kernel),'不能视为纯同步时间'],
 ['本地准备时长',us(w.setup),us(r1.setup),'不能视为调度排队时间'],
 ['生产者的计算 / 搬运 / 同步','未采集','未采集','需要记录每个数据分块由谁、何时送达']
 ])+notice('下一步要补什么','记录每个 Rank、Worker 和数据分块的发布、通知与到达时间，找出最后完成的生产者。'),badge('Q3 · UNKNOWN'));
}
function stepThree(){
 return heading('空白期间任务为什么没有运行？','先看任务何时就绪，再区分依赖等待、资源不足和上游过慢。')+
 `<div class="tab-control diagnosis-tabs" role="tablist" aria-label="直接原因分析">${[['lifecycle','任务生命周期'],['resources','依赖与资源'],['upstream','上游耗时'],['slice','最小因果切片']].map(([id,label])=>`<button class="tab-control-item ${state.tab===id?'is-selected':''}" role="tab" aria-selected="${state.tab===id}" data-cause-tab="${id}">${label}</button>`).join('')}</div><div role="tabpanel">`+
 (state.tab==='resources'?resources():state.tab==='upstream'?upstream():state.tab==='slice'?card('最小依赖切片',notice('这张图能说明什么','只显示 Trace 已记录的直接前驱和下游任务；它不能补出任务何时就绪、在等谁或何时解除等待。','info')+'<div id="graph" class="graph-stage"></div>'):lifecycle())+'</div>';
}
function stepFour(){
 const rank=A.activeCase().rank,w=wait(rank),predecessors=new Set(w.fanin?.match(/r\d+t\d+/g)||[]);
 const related=tasks(rank).map((task,index)=>({task,index})).filter(({task})=>predecessors.has(task.id));
 // A scope-name match or a saved Pass choice is not an investigation target.
 // The archive has no runtime blocker -> compiled-node mapping for either case.
 return heading('尚未定位到相关的编译变化','已找到等待任务，但还不知道它在等哪个 Rank、哪个 Worker 的数据。需要先确定拖慢它的上游任务，再追查该任务的编译过程。')+
 card('当前追查到的任务',table(['Trace 记录','任务','在主泳道中定位'],[
  ['等待事件',`<code>${esc(w.name)}</code><br>${esc(w.id)} · Rank ${rank} · ${us(span(w))}`,button('定位等待任务','task','btn-sm',`data-task="${meta().waitIndex}" data-task-rank="${rank}"`)],
  ...related.map(({task,index})=>['已记录的直接前驱',`<code>${esc(task.name)}</code><br>${esc(task.id)} · Rank ${rank}`,button('定位前驱任务','task','btn-sm',`data-task="${index}" data-task-rank="${rank}"`)])
 ])+'<p>前驱关系来自 Trace。它能帮助缩小排查范围，但没有指出是哪一个生产者导致了这段等待。</p>')+
 card('继续追查需要什么',`<p><strong>上游到达记录：</strong>各 Rank、Worker 的数据发布时间、到达时间，以及对应的等待事件，用来找出最后到达的数据。</p><p><strong>任务与编译子图的对应关系：</strong>将这个生产任务关联到具体节点，再检查影响该节点的 Pass 变化。当前记录缺少这项映射。</p><div class="inline">${button('返回 03 · 排查上游耗时','report-upstream','btn-solid')}${button('查看补采集清单','collect','btn-ghost')}</div>`);
}
function recommendations(){
 const m=meta();
 return [
 {name:m.param,value:m.value,values:m.values,source:m.control,type:m.desc,hypothesis:m.hypothesis,mechanism:state.caseKey==='a2a'?'调整负责发布数据的 Worker 数量，观察最慢生产者是否更早完成，从而缩短等待':'调整每次通信处理的数据行数，观察各数据分块是否更均匀到达，从而缩短等待',direct:'等待时长、Worker 和 Rank 长尾',risk:m.risk}
 ];
}
function stepFive(){
 const r=recommendations()[0];
 return heading('先验证哪一项修改？','当前证据只能提出一个可检验的方向，尚不能把它当作确定修复。','CANDIDATE')+
 notice('为什么只能作为待验证方案','缺少每个 Rank 和 Worker 的数据到达时序，因此可以说明参数可能影响等待，却不能预测能缩短多少。','info')+
 card(`首选验证参数 · ${r.type}`,`<div class="recommendation"><div><h3><code>${esc(r.name)}</code></h3><dl><dt>当前 / 候选</dt><dd><code>${esc(r.value)}</code> → ${r.values.join(' / ')}</dd><dt>为什么可能有效</dt><dd>${r.mechanism}</dd><dt>重点观察</dt><dd>${r.direct}；端到端耗时</dd><dt>需要防范</dt><dd>${r.risk}</dd></dl>${sourceDisclosure(r.source,'参数定义')}<div class="inline">${button(icon('flask')+' 创建单变量实验','create-experiment','btn-solid','data-recommendation="0"')}</div></div></div>`,badge('HYPOTHESIS','warning'))+
 card('暂不建议修改的其他参数',table(['参数类别','暂不建议的原因'],[['TileShape / 融合范围','没有发现 UB 溢出、Copy 放大或融合边界成本证据'],['Loop / OoO / 调度参数','没有记录任务从就绪到下发的排队时间或调度阻塞原因'],['Partition / Fusion Pass 参数','没有编译决策记录，也没有对应的可调参数映射']]));
}
function mountGraph(){
 const el=$('graph');if(!el)return;
 const list=A.workflow.related();
 const edges=list.flatMap(t=>list.filter(p=>t.fanin.includes(p.shortId)).map(p=>[p.shortId,t.shortId]));
 el.innerHTML='<div class="graph-flow"></div>'+table(['已记录前驱','下游'],edges);
 for(const t of list){
  const wrap=document.createElement('div');wrap.className='graph-node-wrap';
  const node=PtoPassIrGraphNodePattern.buildNodeCardElement({id:t.id,type:'op',frame:{width:210},data:{semanticLabel:t.label}},{compact:true});
  node.tabIndex=0;node.setAttribute('role','button');node.setAttribute('aria-label',t.rawName);
  node.addEventListener('click',()=>A.revealTask(t,{logical:true}));node.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();A.revealTask(t,{logical:true});}});
  const label=document.createElement('small');label.textContent=t.shortId+' · '+us(t.duration);
  wrap.append(label,node);el.querySelector('.graph-flow').append(wrap);
 }
}
function mount(){tooltipHandle?.destroy();tooltipHandle=null;if(!A.activeCase()?.preset)return;if((A.simulation?.active()&&state.step>=3)||(state.step===4&&A.simulation?.supported())){A.simulation.mount();return;}mountGraph();}
function renderStep(){tooltipHandle?.destroy();tooltipHandle=null;if((A.simulation?.active()&&state.step>=3)||(state.step===4&&A.simulation?.supported()))return A.simulation.renderStep(state.step);return [stepOne,stepTwo,stepThree,stepFour,stepFive,()=>A.experiments.render()][state.step-1]();}
function action(el){
 switch(el.dataset.action){
  case 'compare-ranks':A.workarea.compareRanks();return true;
  case 'report-cause':A.activeCase().step=3;A.activeCase().tab='lifecycle';break;
  case 'report-upstream':A.activeCase().step=3;A.activeCase().tab='upstream';break;
  case 'report-task':showTask(Number(el.dataset.task),Number(el.dataset.taskRank||0),el.dataset.logical!=='false');return true;
  default:return false;
 }
 A.renderCase();A.persist();return true;
}
return {renderStep,mount,action,meta};
};
