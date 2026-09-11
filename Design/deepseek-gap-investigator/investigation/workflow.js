/* Case workflow. Runtime evidence is never promoted to missing lifecycle or Pass reasons. */
window.InvestigationWorkflow=function(A){
 const D=window.INVESTIGATION_DATA,{esc,icon,badge,button,table,empty,notice,code}=InvestigationUI;
 const steps=['确认异常','定位任务','排查原因','追溯编译变化','制定调优方案','运行实验与对比'];
 const meta={
  a2a:{number:'01',title:'输出通信等待与 AIC 空洞',short:'Output All-to-All',wait:2,source:'oWait',control:'oConst',param:'ATTENTION_PUBLISH_WORKERS',value:48,values:[24,32,64],mechanism:'调整负责发布数据的 Worker 数量，观察最慢生产者是否更早完成，从而缩短等待',risk:'减少 Worker 可能增加单个 Worker 的工作量；修改时还要确认信号计数与生产者数量一致。'},
  token:{number:'02',title:'Token All-Gather 等待长尾',short:'Token All-Gather',wait:1,source:'tokenWait',control:'tokenConst',param:'COMM_ROW_TILE',value:8,values:[4,16,32],mechanism:'调整每次通信处理的数据行数，观察各数据分块是否更均匀到达，从而缩短等待',risk:'可能增加任务数、数据搬运或片上内存占用；需要检查 Shape 整除、尾块处理和结果正确性。'}
 };
 const questions=[['是否存在 Ready Task 未及时下发？',3,'lifecycle','UNKNOWN'],['依赖未满足，还是资源不可用？',3,'resources','UNKNOWN'],['上游慢在计算、搬运、同步还是调度？',3,'upstream','UNKNOWN'],['为什么这样切分或融合？',4,'','UNKNOWN'],['哪个 Pass 做出了决定？',4,'','UNKNOWN'],['应该修改哪个控制入口？',5,'','CANDIDATE'],['预计和实际改善了什么？',6,'','待验证']];
 const num=(v,d=2)=>Number(v).toLocaleString('en-US',{maximumFractionDigits:d,minimumFractionDigits:d});
 const us=v=>v==null?'未采集':num(v)+' µs';
 const section=(title,body,status='',type='neutral')=>`<section class="inspector-section"><div class="inspector-section-head"><h3 class="inspector-section-title">${title}</h3>${status?badge(status,type):''}</div>${body}</section>`;
 const sourceDisclosure=(key,label='关联源码')=>{const s=D.sources[key];return s?`<details class="source-disclosure"><summary>${icon('code')} ${esc(label)} · ${esc(s.file.split('/').at(-1))}:${s.line}</summary><div class="source-disclosure__meta"><code>${esc(s.file)}:${s.line}</code><span>归档快照</span></div>${code(s)}</details>`:'';};
 const c=()=>A.activeCase(),m=()=>meta[c()?.preset],slice=()=>c()?.preset?D.cases[c().preset].ranks[c().rank]:null;
 const wait=()=>slice()?.tasks[m().wait];
 function related(){
  const model=A.caseModel();if(!model)return [];
  if(c().preset)return slice().tasks.map(s=>model.tasks.find(t=>t.taskId===s.taskId&&t.pid===s.event.pid&&t.tid===s.event.tid)).filter(Boolean);
  const unique=list=>[...new Map(list.map(t=>[t.shortId,t])).values()].slice(0,6);
  const lanes=model.lanes.filter(l=>c().selected_lanes.includes(l.id));const pool=lanes.flatMap(l=>l.tasks),[start,end]=c().time_range;
  return [...new Map([...unique(pool.filter(t=>t.end<=start).sort((a,b)=>b.end-a.end)),...unique(pool.filter(t=>t.start>=end).sort((a,b)=>a.start-b.start)),...unique(model.tasks.filter(t=>t.start<end&&t.end>start&&t.process==='Worker View'))].map(t=>[t.id,t])).values()];
 }
 function heading(q,answer,status=''){return `<div class="step-heading">${status?`<div class="inline">${badge(status,status==='CANDIDATE'?'warning':status==='CONFIRMED'?'success':'neutral')}</div>`:''}<h2>${q}</h2><p>${answer}</p></div>`;}
 function first(){
  const w=wait(),model=A.caseModel(),lanes=model.lanes.filter(l=>c().selected_lanes.includes(l.id)),[a,b]=c().time_range;
  const busy=lanes.length?lanes.reduce((n,l)=>n+TraceModel.occupied(l.tasks,a,b),0)/lanes.length:0;
  return heading('这段时间发生了什么？','先确认观测范围和事件，再确定调查方向。','CONFIRMED')+
   section('固定调查范围',`<dl class="definition"><dt>Rank / Lane</dt><dd>Rank ${c().rank} · ${lanes.length} 条泳道</dd><dt>时间窗口</dt><dd>${us(a)} → ${us(b)}</dd><dt>窗口长度</dt><dd>${us(b-a)}</dd><dt>平均空闲</dt><dd>${lanes.length?us(Math.max(0,b-a-busy)):'未选择泳道'}</dd></dl><p>空闲按选中泳道内 X 事件的时间并集统计；具名 wait 事件也占用执行时间。</p>${button('回到调查范围','focus-case','btn-sm')}`,'DERIVED','info')+
   (w?section('具名等待事件 · 跨 Rank 对照',`<p><code>${esc(w.name)}</code></p><div class="metric-pair">${[0,1].map(rank=>{const t=D.cases[c().preset].ranks[rank].tasks[m().wait];return `<div><span>Rank ${rank}</span><strong>${us(t.end-t.start)}</strong></div>`;}).join('')}</div><p>口径：各 Rank 使用自身 Run 内时间；等待事件跨度与 AIC 空闲时长分别统计。</p>`,'RUNTIME','info'):section('手动框选记录','<p>用户定义的调查范围。当前未关联源码和 Pass 数据。</p>','用户标注'))+
   section('建议',notice('检查相邻任务和生命周期','当前 Trace 未记录 Ready 状态，空白区间内是否存在可运行任务尚待确认。'));
 }
 function second(){
  const [a,b]=c().time_range;
  return heading('哪些任务与这个窗口相关？','时间邻近关系、重叠关系和依赖提示分别保留。','SUPPORTED')+section('前驱、下游与重叠任务',`<div class="task-list">${related().map(t=>button(`${esc(t.rawName)}<small>${t.end<=a?'窗口前完成':t.start>=b?'窗口后启动':'与窗口重叠'} · ${esc(t.laneName)} · ${us(t.duration)}</small>`,'task','btn-ghost',`data-task-id="${esc(t.id)}" data-logical="true"`)).join('')}</div>`,'RUNTIME','info')+
   section('执行图与 Block 图',notice('映射数据缺失','当前 Trace 未记录 rootHash、callOpMagic 和 leafHash。现阶段可关联 Task 名称与源码 scope。','info')+button('查看局部任务关系','graph','btn-sm')+(m()?sourceDisclosure(m().source):''))+
   section('映射要求',`<p>Execute / Block 跳转需要稳定关联键；当前没有记录这些字段。</p>`);
 }
 function third(){
  const tab=c().tab||'lifecycle';let body='';
  if(tab==='lifecycle'){
   const target=slice()?.tasks[5],e=target?.event;
   body=section('下游任务生命周期',`<p>${target?esc(target.name)+' · '+esc(target.id):'尚未确认具体下游 Task'}</p><div class="lifecycle">${[['created_at','创建',null],['dependencies_ready_at','依赖就绪',null],['enqueued_at','进入队列',null],['dispatched_at','下发',target?.dispatch],['started_at','实际开始',e?.ts],['finished_at','执行完成',e?e.ts+e.dur:null]].map(([key,label,v])=>`<div title="${key}"><span>${label}</span><code class="${v==null?'muted':'success'}">${v==null?'未采集':us(v)}</code></div>`).join('')}</div><p>已知时间来自同一 Task 的 Scheduler 与首个 Worker 记录。</p>`)+section('当前无法回答',notice('没有就绪任务，还是就绪后未及时下发？','Trace 没有记录创建、依赖就绪、进入队列和完整就绪队列，因此不能判断任务是否更早就在排队。'),'Q1 · UNKNOWN');
  }else if(tab==='resources')body=section('依赖与资源排查',table(['可能原因','当前判断','还需补充'],[['上游依赖 / 同步',badge(m()?'CANDIDATE':'UNKNOWN'), '依赖解除与其他 Rank 的数据到达时间'],['AIC / AIV','暂无法判断','可调度核、亲和性和并发上限'],['内存 / DMA','暂无法判断','UB、L1、L0 占用和 DMA 队列'],['调度队列','暂无法判断','就绪队列、优先级和未运行原因']]))+section('当前判断',notice('还不能排除任何一类原因','等待事件使同步方向值得优先检查，但没有等待原因记录和资源快照。'),'Q2 · UNKNOWN');
  else {const w=wait();body=section('上游耗时分解',table(['已记录区间','当前观测'],[['等待事件总时长',w?us(w.end-w.start):'未采集'],['内核执行时长',us(w?.kernel)],['本地准备时长',us(w?.setup)],['计算 / 搬运 / 同步 / 排队','未采集']]))+section('下一步要补什么',notice('找出最后完成的生产者','记录每个 Rank、Worker 和数据分块的发布、通知与到达时间，才能判断上游具体慢在哪里。'),'Q3 · UNKNOWN');}
  return heading('分析空白区间','当前 Trace 未记录 Ready 和队列事件，暂无法判断任务尚未就绪还是等待下发。')+`<div class="tab-control diagnosis-tabs" role="tablist" aria-label="直接原因">${[['lifecycle','生命周期'],['resources','依赖与资源'],['upstream','上游耗时']].map(([key,label])=>`<button class="tab-control-item ${tab===key?'is-selected':''}" role="tab" aria-selected="${tab===key}" data-cause-tab="${key}">${label}</button>`).join('')}</div><div role="tabpanel">${body}</div>`+section('建议',`<div class="inline">${button('查看相关依赖链','graph','btn-sm')}${button('生成补采集清单','collect','btn-ghost btn-sm')}</div><p>相关依赖链只显示已有 fanin/fanout 记录。</p>`);
 }
 function fourth(){return heading('先确定要追查的任务','这个空闲区间尚未关联到导致等待的具体任务，当前没有可展示的相关 Pass 对照。')+section('继续定位',`<p>先检查空闲区间前后的任务和依赖，再将相关任务对应到编译子图。</p>${button('返回 02 · 定位任务','question','btn-solid','data-target-step="2"')}`);}
 function fifth(){const v=m();return heading('先验证哪一项修改？','有证据时只提出可检验的修改方向；证据不足时先补采集。',v?'CANDIDATE':'UNKNOWN')+(v?section('待验证参数',`<h3><code>${esc(v.param)}</code></h3><dl class="definition"><dt>当前 / 候选</dt><dd>${v.value} → ${v.values.join(' / ')}</dd><dt>为什么可能有效</dt><dd>${v.mechanism}</dd><dt>重点观察</dt><dd>等待时长、Worker 与 Rank 长尾</dd><dt>最终判断</dt><dd>正确性与重复运行的端到端耗时</dd><dt>需要防范</dt><dd>${v.risk}</dd></dl>${sourceDisclosure(v.control,'参数定义')}${sourceDisclosure(v.source,'关联源码')}<p>源码来自归档快照；运行前需要在开发工程中核对同一参数和当前值。</p>`,'HYPOTHESIS','warning'):section('暂无候选参数',notice('还缺哪些数据','补充源码、编译结构和运行时等待原因后再提出修改建议。')))+section('暂不建议修改的其他参数',table(['参数类别','暂不建议的原因'],[['TileShape / 融合范围','没有 UB 溢出、Copy 放大或融合边界成本证据'],['Loop / OoO / 调度','没有任务从就绪到下发的排队时间和阻塞原因'],['Partition / Fusion','没有编译决策记录和对应的可调参数映射']]))+section('下一步',v?button('创建一次只改一个参数的实验','experiment','btn-solid'):button('导出补采集清单','collect','btn-solid'));}

 const report=InvestigationReport(A);
 function render(){
  const current=c();if(!current)return;
  A.simulation?.syncLocation();
  const v=m(),w=wait(),simulation=A.simulation?.active();
  const subtitle=v?`${esc(w.name)} · ${esc(w.id)} · R0 / R1 ${num((D.cases[current.preset].ranks[0].tasks[v.wait].end-D.cases[current.preset].ranks[0].tasks[v.wait].start)/(D.cases[current.preset].ranks[1].tasks[v.wait].end-D.cases[current.preset].ranks[1].tasks[v.wait].start),1)}×`:`Rank ${current.rank} · ${us(current.time_range[0])} — ${us(current.time_range[1])}`;
  document.getElementById('investigationTitle').textContent='异常调查';
  document.getElementById('investigationContent').innerHTML=`<header class="case-header"><div><div class="eyebrow">${v?'DEEPSEEK V4 / DECODE CSA':'LOCAL TRACE'} / ${esc(current.case_id)}</div><h1>${esc(current.title)}</h1><p>${subtitle}</p>${A.simulation?.context()||''}</div><div class="header-actions">${simulation?'':button('诊断问题 · 7','questions','btn-ghost')}${button(simulation?'真实数据来源':'证据'+(v?' · 3':''),'evidence')}</div></header><nav class="workflow" aria-label="六步调查">${steps.map((s,i)=>{const step=i+1,status=step<current.step?'is-complete':step===current.step?'is-selected':'is-upcoming';return `<button class="cp-btn ${status}" data-step="${step}" ${step===current.step?'aria-current="step"':''}><span class="cp-btn-icon">0${step}</span><span class="cp-btn-label">${s}</span></button>`;}).join('')}</nav><section class="step-scroll" aria-label="当前调查步骤"><div id="stepContent" class="${v?'report-step':'custom-step'}">${v?report.renderStep():[first,second,third,fourth,fifth,()=>A.experiments.render()][current.step-1]()}</div></section>`;
  document.getElementById('investigationFooter').innerHTML=`<span class="small-meta">0${current.step} / 06${simulation?' · 模拟补全':''}</span><div class="inline">${button('← 上一步','prev','btn-ghost btn-sm',current.step===1?'disabled':'')}${simulation?'':button('补采集','collect','btn-sm')}${current.step<6?button(steps[current.step]+' →','next','btn-solid btn-sm'):simulation?'':button('检查工程与实验','project-default','btn-solid btn-sm')}</div>`;
  requestAnimationFrame(()=>{
   report.mount();
   const workflow=document.querySelector('#investigationContent .workflow'),active=workflow?.querySelector('[aria-current="step"]');
   if(workflow&&active&&workflow.scrollWidth>workflow.clientWidth){const rail=workflow.getBoundingClientRect(),item=active.getBoundingClientRect();workflow.scrollLeft=Math.max(0,workflow.scrollLeft+item.left-rail.left-(workflow.clientWidth-item.width)/2);}
  });
 }
 function graph(){
  const list=related(),edges=[];for(const t of list)for(const producer of list)if(t.fanin.includes(producer.shortId))edges.push([producer.shortId,t.shortId]);
  A.workarea.content('graph','任务依赖',`<p>来源：Trace fanin-hint。${edges.length} 条已记录的局部依赖；缺少 Ready / blocked_reason，不构成完整因果证明。</p><div id="graph" class="graph-flow"></div>${table(['前驱','下游'],edges.map(e=>e.map(esc)))}`);
  const mount=document.getElementById('graph');for(const t of list){const wrap=document.createElement('div');wrap.className='graph-node-wrap';const node=PtoPassIrGraphNodePattern.buildNodeCardElement({id:t.id,type:'op',frame:{width:210},data:{semanticLabel:t.label}},{compact:true});node.tabIndex=0;node.setAttribute('role','button');node.setAttribute('aria-label',t.rawName);node.addEventListener('click',()=>A.revealTask(t,{logical:true}));node.addEventListener('keydown',e=>{if(e.key==='Enter')A.revealTask(t,{logical:true});});wrap.append(node);const label=document.createElement('small');label.textContent=t.shortId+' · '+us(t.duration);wrap.append(label);mount.append(wrap);}
  A.renderer.related=new Set(list.map(t=>t.id));A.renderer.draw();
 }
 function collect(){return {schema_version:2,case_id:c().case_id,run_id:c().run_id,rank_ids:[c().rank],time_range:c().time_range,selected_lanes:c().selected_lanes,required_events:['created_at','dependencies_ready_at','enqueued_at','dispatched_at','started_at','finished_at','blocked_reason','unresolved_dependencies','queue_id'],resource_snapshots:['AIC/AIV occupied and schedulable cores','UB/L1/L0 allocation','DMA queue','Ready Queue/Task Ring/Dependency Pool'],peer_events:['peer_id','worker_id','chunk_id','publish_at','notify_at','arrival_at'],compiler_decisions:['pass_name','before_graph_ref','after_graph_ref','affected_node_ids','reason_code','reason_detail','source_refs'],note:'采集需求协议，需采集端适配，不是已接通的 Runtime 配置。'};}
 function evidence(){A.dialog('调查证据与数据缺口',section('Trace 记录',`<p>调查窗口 ${us(c().time_range[0])} — ${us(c().time_range[1])}，Rank ${c().rank}。</p>${button('定位相关任务','evidence-tasks','btn-sm')}`)+section('源码与编译产物',m()?sourceDisclosure(m().source)+sourceDisclosure(m().control,'参数定义')+sourceDisclosure(c().preset+'Pass0','最初的编译快照'):'<p>未关联。</p>')+section('还缺哪些数据','<p>任务就绪时间、排队时间、资源快照、最慢生产者和 Pass 决策原因尚未采集。</p>')+`<label class="ui-field">调查备注<textarea id="caseNote" class="ui-input note-area">${esc(c().note||'')}</textarea></label>${button('保存备注','save-note','btn-sm')}${button('导出调查上下文','export-case','btn-ghost btn-sm')}`);}
 function showQuestions(){A.dialog('七个诊断问题',table(['问题','当前结论','查看'],questions.map(([q,step,tab,status],i)=>['Q'+(i+1)+' · '+q,badge(i===5&&!m()?'UNKNOWN':i===6&&c().experiment?.result?c().experiment.result.verdict:status),button('查看','question','btn-ghost btn-sm',`data-target-step="${step}" data-target-tab="${tab}"`)])));}
 return {meta,steps,num,us,section,heading,render,graph,collect,evidence,showQuestions,related,report};
};
