(function(){
 'use strict';
 const {esc,icon,badge,button,table,empty,notice}=InvestigationUI,D=INVESTIGATION_DATA,$=id=>document.getElementById(id);
 const base='../../Data/DeepseekV4/',tracePaths=[0,1].map(rank=>D.cases.a2a.ranks[rank].trace);
 const models=new Map(),state={rank:0,model:null,caseId:null,task:null,filter:'worker',kind:'all',laneQuery:'',hiddenLanes:new Set(),loading:false,diagnosticMode:true,simulationMode:new URLSearchParams(location.search).get('demo')==='compiler'};
 let saved={};try{saved=JSON.parse(localStorage.getItem('pto-gap-v2')||'{}');}catch{}
 const cases=Array.isArray(saved.cases)?saved.cases.filter(c=>c&&typeof c.case_id==='string'&&Array.isArray(c.time_range)&&c.time_range.length===2&&c.time_range.every(Number.isFinite)&&Array.isArray(c.selected_lanes)&&Number.isInteger(c.step)&&c.step>=1&&c.step<=6):[];
 let toastTimer,loadSequence=0,dialogFocus=null,sheetFocus=null,controller,desktopToggle=false,inspectorHasObject=false;
 const A={state,cases,models,activeCase:()=>cases.find(c=>c.case_id===state.caseId),caseModel:()=>{const c=A.activeCase();return c?models.get(c.model_key):null;},toast,dialog,closeDialog,bottom,persist,renderMarkers,renderCase,selectTask,selectEvidenceTask,revealTask,rangeFromEvidence,download,focusCase,showWorkflow,closeInvestigation,openInvestigation,inspectSimulationTask,loadDefault};
 A.workflow=InvestigationWorkflow(A);A.experiments=InvestigationExperiments(A);A.workarea=InvestigationWorkarea(A);A.project=InvestigationProject(A);const W=A.workflow;
 A.simulation=InvestigationSimulation(A);
 document.querySelectorAll('[data-icon]').forEach(el=>el.innerHTML=icon(el.dataset.icon));
 window.PTO_PASS_IR_GRAPH_NODE_ASSET_PREFIX='../../vendor/pto-design-system/';
 controller=PtoIdeFrame.init($('frame'));
 const drawerResize=PtoWorkbenchShell.initResizablePanes({root:$('investigationOverlay'),panes:[$('investigationClearance'),$('investigationDrawer')],direction:'vertical',sizes:[24,76],minSize:[100,220],storageKey:'gap-v5-drawer-height',gutterLabel:'调整异常调查抽屉高度',onGutterCreate:gutter=>{gutter.style.pointerEvents='auto';}});
 A.renderer=new TraceRenderer($('traceCanvas'),{onTask:selectTask,onMarker:showMarker,onRange:offerRange,onView:()=>renderMarkerPositions()});
 // Exposed read-only-by-convention state for reproducible browser smoke and evidence audits.
 window.GapApp=A;
 function persist(){try{localStorage.setItem('pto-gap-v2',JSON.stringify({cases}));}catch{toast('无法保存到浏览器；请导出调查。');}}
 function toast(message){clearTimeout(toastTimer);$('toast').textContent=message;$('toast').hidden=false;toastTimer=setTimeout(()=>$('toast').hidden=true,3500);}
 function dialog(title,body){A.renderer?.hideTip();if(!$('detailDialog').open)dialogFocus=document.activeElement;$('dialogTitle').textContent=title;$('dialogBody').innerHTML=body;if(!$('detailDialog').open)$('detailDialog').showModal();}
 function closeDialog(){$('detailDialog').close();}
 $('detailDialog').addEventListener('close',()=>{if(dialogFocus?.isConnected&&!dialogFocus.closest('[hidden]'))dialogFocus.focus();});
 function download(name,value){const blob=new Blob([JSON.stringify(value,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('已导出 '+name);}
 function narrow(){return innerWidth<=760;}
 function toggleDesktopInspector(){desktopToggle=true;$('inspectorToggle').click();desktopToggle=false;}
 function inspectorOpen(){return $('mobileSheet').open||$('frame').dataset.inspectorCollapsed==='false';}
 function resetInspectorWidth(){
  if(narrow()||$('frame').dataset.inspectorCollapsed!=='false')return;
  const split=$('mainSplit'),mins=String(split.dataset.minSize||'').split(',').map(Number),minimum=Number.isFinite(mins[1])?mins[1]:360,gutter=split.querySelector(':scope > .pto-workbench-shell__split-gutter'),available=Math.max(1,split.clientWidth-(gutter?.offsetWidth||7)),width=Math.min(minimum,Math.max(0,available-(Number.isFinite(mins[0])?mins[0]:220))),ratio=width/available*100;
  controller?.resizeInstances?.[0]?.setSizes([100-ratio,ratio]);controller?.refresh();
 }
 function openInspector(){
  if(narrow()){
   if($('frame').dataset.inspectorCollapsed==='false')toggleDesktopInspector();
   if(!$('mobileSheet').open){sheetFocus=document.activeElement;$('mobileSheet').append($('inspector'));$('inspector').hidden=false;$('inspector').setAttribute('aria-hidden','false');$('mobileSheet').showModal();}
  }else if($('frame').dataset.inspectorCollapsed==='true')$('inspectorToggle').click();
  A.renderer.draw();
 }
 function closeInspector(){if($('mobileSheet').open)$('mobileSheet').close();else if($('frame').dataset.inspectorCollapsed==='false')toggleDesktopInspector();A.renderer.hideTip();}
 $('mobileSheet').addEventListener('close',()=>{$('inspector').hidden=true;$('inspector').setAttribute('aria-hidden','true');$('mainSplit').append($('inspector'));$('inspectorToggle').setAttribute('aria-expanded','false');if(sheetFocus?.isConnected)sheetFocus.focus();controller?.refresh();A.renderer.draw();});
 $('inspectorToggle').addEventListener('click',e=>{if(narrow()&&!desktopToggle){e.stopImmediatePropagation();if(inspectorOpen())closeInspector();else{if(!inspectorHasObject)renderInspectorOverview();openInspector();}}},true);
 $('inspectorToggle').addEventListener('click',()=>{if(!inspectorHasObject)renderInspectorOverview();resetInspectorWidth();A.renderer.draw();});
 let lastNarrow=narrow();window.addEventListener('resize',()=>{const next=narrow();if(next!==lastNarrow){if($('mobileSheet').open)$('mobileSheet').close();if(next&&$('frame').dataset.inspectorCollapsed==='false')toggleDesktopInspector();lastNarrow=next;}A.renderer.draw();});
 function setDiagnosticMode(enabled){state.diagnosticMode=Boolean(enabled);A.renderer.diagnosticMode=state.diagnosticMode;const toggle=$('investigationToggle');toggle.classList.toggle('is-selected',state.diagnosticMode);toggle.setAttribute('aria-pressed',String(state.diagnosticMode));$('markerActions').hidden=!state.diagnosticMode||!$('mainEvidence').hidden;if(!state.diagnosticMode)closeInvestigation();A.renderer.draw();}
 function openInvestigation(){A.renderer.hideTip();if(!state.diagnosticMode)setDiagnosticMode(true);$('investigationOverlay').hidden=false;drawerResize.refresh();A.renderer.draw();}
 function closeInvestigation(){const hadFocus=$('investigationOverlay').contains(document.activeElement);$('investigationOverlay').hidden=true;if(hadFocus)$('investigationToggle').focus();A.renderer.draw();}
 function showWorkflow(){$('bottomDock').hidden=true;$('investigationContent').hidden=false;$('investigationFooter').hidden=!A.activeCase();}
 function bottom(title,html){closeDialog();$('bottomTitle').textContent=title;$('bottomContent').innerHTML=html;$('bottomDock').hidden=false;$('investigationContent').hidden=true;$('investigationFooter').hidden=true;openInvestigation();}
 function visibleLanes(){return state.model?.lanes.filter(l=>(state.filter==='all'||(state.filter==='worker'?l.worker:l.process===state.filter))&&(state.kind==='all'||l.kind===state.kind)&&!state.hiddenLanes.has(l.id))||[];}
 function renderInspectorOverview(lanes=visibleLanes()){
  inspectorHasObject=false;$('inspectorTitle').textContent='泳道概览';$('inspectorFooter').innerHTML='';
  const model=state.model;if(!model){$('inspectorContent').innerHTML=empty('正在加载 Trace','加载完成后显示当前泳道汇总。');return;}
  const events=lanes.reduce((sum,lane)=>sum+lane.tasks.length,0),average=lanes.length?lanes.reduce((sum,lane)=>sum+lane.utilization,0)/lanes.length:0,scope=state.filter==='worker'?'Worker':state.filter==='all'?'所有进程':state.filter,kinds=['AIC','AIV','Other'].map(kind=>[kind,lanes.filter(lane=>lane.kind===kind).length]).filter(([,count])=>count),active=[...lanes].sort((a,b)=>b.utilization-a.utilization||b.tasks.length-a.tasks.length).slice(0,5);
  $('inspectorContent').innerHTML=`<div class="case-context"><div class="eyebrow">${model.annotated?'DEEPSEEK V4 / DECODE CSA':'LOCAL TRACE'} · RANK ${model.rank}</div><h2>当前泳道</h2><p>${esc(scope)}${state.kind==='all'?'':' · '+esc(state.kind)} · 未选择事件</p></div>${W.section('当前显示',`<dl class="definition"><dt>可见泳道</dt><dd>${lanes.length} 条${kinds.length?' · '+kinds.map(([kind,count])=>`${kind} ${count}`).join(' / '):''}</dd><dt>事件</dt><dd>${events.toLocaleString('en-US')} 个 X 事件</dd><dt>Run 时长</dt><dd>${W.us(model.duration)}</dd><dt>平均占用率</dt><dd>${W.num(average)}%</dd></dl><p>占用率按每条可见泳道的执行区间并集统计。</p>`)}${W.section('最活跃泳道',active.length?table(['泳道','事件','占用率'],active.map(lane=>[esc(lane.name),lane.tasks.length.toLocaleString('en-US'),W.num(lane.utilization)+'%'])):'<p>当前筛选没有可见泳道。</p>')}`;
 }
 function renderEmptyInvestigation(){$('investigationTitle').textContent='异常调查';$('investigationContent').innerHTML=empty('选择调查点，开始追溯','具名调查、规则候选和框选记录统一显示在列表中。选择后查看六步调查；点击泳道事件可同时查看详情。');$('investigationFooter').hidden=true;showWorkflow();}
 function setTraceState(title,description,error=false){$('traceState').hidden=false;$('traceState').innerHTML=empty(title,description,error?button('加载默认 Trace','default','btn-sm')+button('选择本地 JSON','load','btn-ghost btn-sm'):'<progress aria-label="加载 Trace"></progress>');}
 async function loadDefault(rank=0,preserveCase=false){
  const key='default-'+rank,seq=++loadSequence;if(models.has(key)){state.loading=false;activate(models.get(key),key,preserveCase);return;}
  state.loading=true;setTraceState('加载 Rank '+rank+' 完整 Trace','正在读取原始事件并建立泳道索引…');
  try{const response=await fetch(base+tracePaths[rank]);if(!response.ok)throw Error('HTTP '+response.status);const raw=await response.json();if(seq!==loadSequence)return;const model=TraceModel.parse(raw,{runId:D.run,rank,name:'decode_csa',source:base+tracePaths[rank],annotated:true});models.set(key,model);activate(model,key,preserveCase);}catch(e){if(seq===loadSequence){setTraceState('Trace 加载失败',esc(e.message),true);$('status').textContent='加载失败 · 可选择本地 JSON';}}finally{if(seq===loadSequence)state.loading=false;}
 }
 async function loadLocal(file){if(!file)return;const seq=++loadSequence;state.loading=true;setTraceState('解析 '+file.name,'建立独立 Run 与泳道索引…');try{const text=await file.text(),raw=JSON.parse(text),digest=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text));if(seq!==loadSequence)return;const runId='local-'+Array.from(new Uint8Array(digest)).map(v=>v.toString(16).padStart(2,'0')).join('').slice(0,24),key=runId+'-R'+state.rank;const model=TraceModel.parse(raw,{runId,rank:state.rank,name:file.name,source:'用户选择的本地文件',annotated:false});models.set(key,model);activate(model,key);closeDialog();}catch(e){setTraceState('无法读取 Trace',esc(e.message),true);}finally{state.loading=false;$('traceImport').value='';}}
 function activate(model,key,preserveCase=false){
  if(!preserveCase)A.workarea.reset();else A.workarea.show('trace');$('selectionAction').hidden=true;
  state.model=model;state.modelKey=key;state.rank=model.rank;state.filter=model.lanes.some(l=>l.worker)?'worker':'all';state.kind='all';state.hiddenLanes.clear();if(!preserveCase)state.task=null;
  $('rankSelect').value=String(model.rank);$('rankSelect').disabled=!model.annotated;$('traceState').hidden=true;
  A.renderer.setModel(model);A.renderer.caseRange=null;
  if(!preserveCase){closeInspector();renderInspectorOverview();}
  if(!preserveCase&&A.activeCase()?.model_key!==key){state.caseId=null;closeInvestigation();renderEmptyInvestigation();}else {A.renderer.caseRange=A.activeCase()?.model_key===key?A.activeCase().time_range:null;if(A.activeCase())renderCase();}
  updateLanes();syncLaneScope();renderMarkers();$('traceMeta').textContent=model.annotated?'DeepSeek V4 ▾':model.name+' ▾';$('traceMeta').title=model.name+' · '+model.tasks.length.toLocaleString()+' events · '+(model.duration/1000).toFixed(2)+' ms · 打开 Trace';
  $('status').textContent=`${model.annotated?'DeepSeek V4':'本地 Trace'} · Rank ${model.rank} · ${model.lanes.length} lanes · µs · ${model.warnings.length?model.warnings.join('；'):'索引就绪'}`;
 }
 function updateLanes(){if(!state.model)return;const lanes=visibleLanes();A.renderer.setLanes(lanes);if(!inspectorHasObject)renderInspectorOverview(lanes);}
 function syncLaneScope(){const value=[state.filter,state.kind].join('|');$('laneScopeSelect').value=['worker|all','worker|AIC','worker|AIV','all|all'].includes(value)?value:'custom';}
 function presetBinding(key){const model=state.model,m=W.meta[key],record=D.cases[key].ranks[state.rank].tasks[m.wait],event=record.event||{},task=model.tasks.find(t=>t.taskId===String(record.taskId)&&t.pid===event.pid&&t.tid===event.tid),coreNames=new Set((record.cores||[]).map(core=>core.lane)),lanes=task?[task.laneId]:model.lanes.filter(l=>coreNames.has(l.name)).map(l=>l.id);return {record,lanes};}
 function presetCase(key){const model=state.model,m=W.meta[key],binding=presetBinding(key),task=binding.record,originalId='INV-'+m.number+'-R'+state.rank;let c=cases.find(c=>c.preset===key&&c.rank===state.rank&&c.model_key===state.modelKey);const id=cases.some(c=>c.case_id===originalId)?originalId+'-'+crypto.randomUUID().slice(0,5):originalId;if(!c){c={case_id:id,title:m.title,preset:key,run_id:model.runId,rank:state.rank,model_key:state.modelKey,time_range:[task.start-model.origin,task.end-model.origin],selected_lanes:binding.lanes,viewport_state:A.renderer.snapshot(),anchor_task_ids:D.cases[key].ranks[state.rank].tasks.map(t=>t.taskId),step:1,tab:'lifecycle',symptom_type:'等待相关调查',evidence_refs:[model.source]};cases.push(c);}return c;}
 function openCase(id){const current=cases.find(c=>c.case_id===id);if(!current)return;const model=models.get(current.model_key);if(!model){if(current.model_key.startsWith('default-')){loadDefault(current.rank).then(()=>openCase(id));return;}toast('此 Case 对应本地文件，请重新加载原始 Trace 后调查。');return;}
  if(state.caseId!==id)A.workarea.reset();if(state.model!==model)activate(model,current.model_key);state.caseId=id;A.renderer.caseRange=current.time_range.slice();A.renderer.related.clear();renderCase();renderMarkers();openInvestigation();persist();
 }
 function renderCase(){W.render();showWorkflow();A.renderer.draw();}
 function investigationPoints(){
  const model=state.model;if(!model)return [];
  const points=model.annotated?Object.entries(W.meta).map(([key,m])=>{const binding=presetBinding(key),t=binding.record;return {key:'preset-'+key,preset:key,kind:'established',title:m.short,source:'具名调查 · 已绑定 wait Task',range:[t.start-model.origin,t.end-model.origin],lanes:binding.lanes};}):[];
  // Candidate scope is fixed per Run / Rank, independent of display filters.
  const lanes=model.lanes.filter(l=>l.worker&&l.kind==='AIC');
  for(const range of lanes.length?TraceModel.gaps(lanes,0,model.duration,80):[])points.push({key:'gap-'+range.join('-'),kind:'candidate',title:'AIC 共同空闲',source:'规则候选 · 24 个 AIC · ≥ 80 µs',range,lanes:lanes.map(l=>l.id)});
  const local=cases.filter(c=>c.model_key===state.modelKey&&c.rank===state.rank);
  for(const point of points)point.caseId=local.find(c=>point.preset?c.preset===point.preset:c.point_key===point.key)?.case_id;
  for(const c of local.filter(c=>!c.preset&&!points.some(p=>p.caseId===c.case_id)))points.push({key:c.case_id,caseId:c.case_id,kind:'manual',title:c.title,source:'手动框选',range:c.time_range,lanes:c.selected_lanes});
  return points.map((p,i)=>({...p,number:String(i+1).padStart(2,'0')}));
 }
 function showMarker(key){
  const point=investigationPoints().find(p=>p.key===key);if(!point)return;
  let id=point.caseId;
  if(!id&&point.preset)id=presetCase(point.preset).case_id;
  if(id&&point.preset){const current=cases.find(c=>c.case_id===id);if(current)current.selected_lanes=point.lanes.slice();}
 if(!id){id='CASE-'+crypto.randomUUID().slice(0,8);cases.push({case_id:id,point_key:point.key,title:point.title+' · '+W.us(point.range[1]-point.range[0]),symptom_type:'共同空闲候选',run_id:state.model.runId,rank:state.rank,model_key:state.modelKey,time_range:point.range.slice(),selected_lanes:point.lanes.slice(),viewport_state:A.renderer.snapshot(),anchor_task_ids:[],evidence_refs:[state.model.source],step:1,tab:'lifecycle'});}
  // Selecting an anomaly is a non-navigating action. The explicit
  // “回到调查范围” control inside the drawer owns viewport navigation.
  openCase(id);
 }
 function renderMarkers(){
  const points=investigationPoints(),activePoint=points.find(p=>p.caseId===state.caseId);A.renderer.markers=points;A.renderer.activeMarkerKey=activePoint?.key||null;
  $('investigationToggle').textContent='诊断模式 · '+String(points.length).padStart(2,'0');
  $('markerActions').hidden=!state.diagnosticMode||!$('mainEvidence').hidden;
  $('markerActions').innerHTML=points.map(p=>`<div class="diagnostic-range${activePoint&&p.key!==activePoint.key?' is-muted':''}${p.key===activePoint?.key?' is-selected':''}" data-point="${esc(p.key)}" data-kind="${p.kind}"><button class="diagnostic-range__tag" data-action="marker" data-point="${esc(p.key)}" title="${esc(p.title+' · '+p.source+' · '+W.us(p.range[0])+' — '+W.us(p.range[1]))}" aria-label="调查此区间：${esc(p.title)} · ${esc(p.source)}">${p.number} ${esc(p.title)}</button></div>`).join('')+'<button id="selectedMarkerJump" class="selected-marker-jump" data-action="focus-case" hidden></button>';
  $('caseList').innerHTML='<div class="case-list-heading"><span>调查列表</span>'+badge(String(points.length).padStart(2,'0'))+'</div>'+points.map(p=>`<button class="cp-btn ${state.caseId===p.caseId?'is-selected':''}" data-action="marker" data-point="${esc(p.key)}" ${p.preset?`data-preset="${p.preset}"`:''} aria-current="${state.caseId===p.caseId?'true':'false'}"><span class="cp-btn-text case-item"><span class="cp-btn-sub">${p.preset?'INV–0'+p.number:p.number} · Rank ${state.rank}</span><span class="case-item-main"><span class="cp-btn-label">${esc(p.title)}</span><span class="case-metric">${W.num(p.range[1]-p.range[0])} <small>µs</small></span></span></span></button>`).join('')+(points.length?'':'<p>暂无调查点，可在泳道中框选建立调查。</p>')+'<details class="case-list-note"><summary>来源与检测范围</summary><p>具名调查已绑定具体 wait Task，但根因仍待取证；规则候选只证明完整 Worker AIC 在该窗口共同无 X 事件（含 Run 两端，阈值 ≥ 80 µs）。</p></details>';A.renderer.draw();
 }
 function renderMarkerPositions(){
  const r=A.renderer;if(!r?.model)return;
  const points=r.markers,entries=[],jump=$('selectedMarkerJump');if(jump)jump.hidden=true;document.querySelectorAll('#markerActions > .diagnostic-range').forEach(el=>{
   const p=points.find(p=>p.key===el.dataset.point);if(!p)return;
   const rect=r.markerRects(p).sort((a,b)=>b.w*b.h-a.w*a.h)[0],tag=el.querySelector('.diagnostic-range__tag');
   el.hidden=!rect;tag.hidden=!rect;if(!rect)return;Object.assign(el.style,{left:rect.x+'px',top:rect.y+'px',width:rect.w+'px',height:rect.h+'px'});entries.push({el,tag,p,left:rect.x,right:rect.x+rect.w,top:rect.y,bottom:rect.y+rect.h});
  });
  const occupied=[];for(const entry of entries.sort((a,b)=>(a.p.key===r.activeMarkerKey?0:1)-(b.p.key===r.activeMarkerKey?0:1)||(a.p.preset?0:1)-(b.p.preset?0:1)||(b.right-b.left)-(a.right-a.left))){entry.tag.hidden=false;entry.tag.style.left='50%';entry.tag.style.top='50%';const bounds=entry.tag.getBoundingClientRect(),width=bounds.width,height=bounds.height,plotLeft=r.labelWidth+4,plotRight=r.canvas.clientWidth-16,plotTop=r.header+2,plotBottom=r.usableHeight()-2,centerX=Math.max(plotLeft+width/2,Math.min(plotRight-width/2,(entry.left+entry.right)/2)),centerY=Math.max(plotTop+height/2,Math.min(plotBottom-height/2,(entry.top+entry.bottom)/2)),box={left:centerX-width/2,right:centerX+width/2,top:centerY-height/2,bottom:centerY+height/2};if(occupied.some(other=>box.left<other.right+8&&box.right>other.left-8&&box.top<other.bottom+4&&box.bottom>other.top-4)){entry.tag.hidden=true;continue;}entry.tag.style.left=(centerX-entry.left)+'px';entry.tag.style.top=(centerY-entry.top)+'px';occupied.push(box);}
  const selected=points.find(p=>p.key===r.activeMarkerKey),selectedEntry=entries.find(entry=>entry.p.key===r.activeMarkerKey);if(!selected||!jump||selectedEntry&&!selectedEntry.tag.hidden)return;
  const laneIndices=r.lanes.map((lane,index)=>selected.lanes.includes(lane.id)?index:-1).filter(index=>index>=0),usable=r.usableHeight(),first=laneIndices.length?Math.min(...laneIndices):0,last=laneIndices.length?Math.max(...laneIndices):r.lanes.length-1,targetTop=r.header+first*r.row-r.scroll,targetBottom=r.header+(last+1)*r.row-r.scroll,isAbove=laneIndices.length&&targetBottom<=r.header,direction=isAbove?'up':'down',arrow=isAbove?'↑':'↓';
  jump.hidden=false;jump.dataset.direction=direction;jump.textContent=`${arrow} 跳到 ${selected.number}`;jump.setAttribute('aria-label',`${isAbove?'向上':'向下'}跳到异常 ${selected.number}：${selected.title}`);jump.title=laneIndices.length?(isAbove?'选中异常位于当前可见泳道上方':'选中异常位于当前可见泳道下方或被调查抽屉遮挡'):'选中异常的泳道当前被筛选隐藏；点击后恢复并定位';
  const targetX=Math.max(r.labelWidth+60,Math.min(r.canvas.clientWidth-72,(r.x(selected.range[0])+r.x(selected.range[1]))/2));Object.assign(jump.style,{left:targetX+'px',top:(isAbove?r.header+6:Math.max(r.header+6,usable-38))+'px'});
 }
 async function focusCase(){const c=A.activeCase();if(!c)return;if(A.caseModel()!==state.model)activate(A.caseModel(),c.model_key,true);A.workarea.show('trace');state.filter='all';state.kind='all';state.hiddenLanes.clear();updateLanes();syncLaneScope();A.renderer.focus(c.time_range,c.selected_lanes[0]);}
 function revealTask(t,{logical=false}={}){
  const model=[...models.values()].find(m=>m.byId.get(t.id)===t);if(!model){toast('事件所属 Run 未加载。');return;}
  if(state.model!==model)activate(model,[...models].find(([,m])=>m===model)[0],true);
  A.workarea.show('trace');state.filter='all';state.kind='all';state.hiddenLanes.clear();updateLanes();syncLaneScope();
  const instances=logical?model.tasks.filter(other=>other.process===t.process&&other.taskId===t.taskId&&other.shortId===t.shortId&&other.laneKind===t.laneKind&&(t.funcId!=null?other.funcId===t.funcId:other.rawName===t.rawName)):[t];
  A.renderer.related=new Set(instances.map(x=>x.id));
  const range=[Math.min(...instances.map(x=>x.start)),Math.max(...instances.map(x=>x.end))];
  selectTask(t);A.renderer.focus(range,t.laneId);
  // Scroll target to the first visible row: it remains above even the tallest drawer.
  A.renderer.scroll=Math.max(0,A.renderer.lanes.findIndex(l=>l.id===t.laneId)*A.renderer.row);
  if(logical){$('inspectorTitle').textContent='逻辑 Task · '+t.shortId;$('inspectorContent').insertAdjacentHTML('afterbegin',W.section('全部 Worker 实例',`<p>${instances.length} 个实例已高亮。跨度 ${W.us(range[1]-range[0])}，不等于单核计算时间。</p><label class="ui-field">精确定位实例<select id="instanceSelect" class="ui-input">${instances.map(x=>`<option value="${esc(x.id)}">${esc(x.laneName)} · pid ${x.pid} / tid ${x.tid} · ${W.us(x.duration)}</option>`).join('')}</select></label>`));$('instanceSelect').addEventListener('change',e=>revealTask(model.byId.get(e.target.value),{remember:false}));}
  if(logical&&$('instanceSelect'))$('instanceSelect').value=t.id;
  A.renderer.draw();
 }
 async function selectEvidenceTask(rank,record,logical=false){
  const caseId=state.caseId,key='default-'+rank;
  try{
   if(!models.has(key)){const response=await fetch(base+tracePaths[rank]);if(!response.ok)throw Error('HTTP '+response.status);models.set(key,TraceModel.parse(await response.json(),{runId:D.run,rank,name:'decode_csa',source:base+tracePaths[rank],annotated:true}));}
   if(state.caseId!==caseId)return;
   const model=models.get(key),task=model.tasks.find(t=>t.taskId===record.taskId&&t.pid===record.event.pid&&t.tid===record.event.tid);
   if(!task)throw Error('未找到对应原始 Worker 事件');revealTask(task,{logical});
  }catch(error){toast('事件读取失败：'+error.message);}
 }
 function offerRange(range,lanes){
  if(!lanes.length||range[1]<=range[0])return;
  A.renderer.brush=false;$('brushButton').setAttribute('aria-pressed','false');$('brushButton').classList.remove('is-selected');
  A.renderer.selectionRange=range;A.renderer.draw();
  const el=$('selectionAction');el.hidden=false;el.innerHTML=`<span>${W.us(range[1]-range[0])} · ${lanes.length} 条泳道</span>${button('调查此选区','confirm-selection','btn-sm')}${button('取消','cancel-selection','btn-ghost btn-sm')}`;
  el.querySelector('[data-action="confirm-selection"]').onclick=()=>{el.hidden=true;createRange(range,lanes);};
  el.querySelector('[data-action="cancel-selection"]').onclick=()=>{el.hidden=true;A.renderer.selectionRange=null;A.renderer.draw();};
 }
 function inspectSimulationTask(t){
  inspectorHasObject=true;$('inspectorTitle').textContent='模拟事件';$('inspectorContent').innerHTML=`<div class="case-context">${badge('模拟数据','warning')}<h2>${esc(t.label)}</h2><p>用于演示编译变化的影响，非设备采集记录。</p></div>`+W.section('模拟时间',`<dl class="definition"><dt>泳道</dt><dd>${esc(t.laneName)}</dd><dt>开始 / 结束</dt><dd>${W.us(t.start)} / ${W.us(t.end)}</dd><dt>持续时间</dt><dd>${W.us(t.duration)}</dd></dl>`);$('inspectorFooter').innerHTML='';openInspector();
 }
 function rangeFromEvidence(rank,range){
  if(rank!==state.rank){toast('这是 Rank '+rank+' 的对照切片。请先切换主 Trace 到该 Rank，再建立调查。');return;}
  const lanes=state.model.lanes.filter(l=>l.worker&&l.kind==='AIC').map(l=>l.id);
  createRange(range.map(t=>t-state.model.origin),lanes);
 }
 function createRange(range,lanes){
  if(!lanes.length||range[1]<=range[0])return;A.renderer.brush=false;$('brushButton').setAttribute('aria-pressed','false');$('brushButton').classList.remove('is-selected');
  const view=A.renderer.snapshot();dialog('创建调查',`<form id="caseForm" class="stack"><p>Rank ${state.rank} · ${lanes.length} 条泳道 · ${W.us(range[0])} — ${W.us(range[1])}</p><label class="ui-field">调查名称<input class="ui-input" name="title" value="选区异常调查" required maxlength="120"></label><label class="ui-field">现象类型<select class="ui-input" name="symptom"><option>待判断</option><option>空洞</option><option>细碎任务</option><option>长尾</option><option>核间不均衡</option></select></label><p>浏览、平移和缩放不会改变此调查范围。</p><div class="inline"><button class="btn btn-solid" type="submit">创建调查</button>${A.activeCase()?button('用此选区更新当前调查','update-range','btn-ghost'):''}</div></form>`);
  const save=()=>{const f=new FormData($('caseForm')),id='CASE-'+crypto.randomUUID().slice(0,8);cases.push({case_id:id,title:String(f.get('title')),symptom_type:String(f.get('symptom')),run_id:state.model.runId,rank:state.rank,model_key:state.modelKey,time_range:range.slice(),selected_lanes:lanes.slice(),viewport_state:view,anchor_task_ids:[],evidence_refs:[state.model.source],step:1,tab:'lifecycle'});closeDialog();A.renderer.selectionRange=null;openCase(id);};
  $('caseForm').addEventListener('submit',e=>{e.preventDefault();save();});
  $('caseForm').querySelector('[data-action="update-range"]')?.addEventListener('click',()=>{const c=A.activeCase();c.time_range=range.slice();c.selected_lanes=lanes.slice();c.viewport_state=view;c.preset=null;delete c.point_key;c.experiment=null;c.step=1;c.title='更新后的选区调查';A.renderer.caseRange=range.slice();closeDialog();persist();renderCase();renderMarkers();openInvestigation();});
 }
 function selectTask(t){inspectorHasObject=true;state.task=t;A.renderer.selected=t.id;A.renderer.draw();$('inspectorTitle').textContent='Task · '+t.shortId;$('inspectorContent').innerHTML=`<div class="case-context">${badge('RUNTIME','info')}<h2>${esc(t.label)}</h2><code>${esc(t.shortId)} · ${esc(t.laneName)}</code></div>`+W.section('运行事件',`<dl class="definition"><dt>进程 / Rank</dt><dd>${esc(t.process)} / ${t.rank}</dd><dt>pid / tid</dt><dd>${t.pid} / ${t.tid}</dd><dt>开始 / 结束</dt><dd>${W.us(t.start)} / ${W.us(t.end)}</dd><dt>持续时间</dt><dd>${W.us(t.duration)}</dd><dt>来源</dt><dd>X event · ${esc(t.taskId)}</dd></dl>`)+W.section('编译映射',table(['关联键','值'],['rootHash','callOpMagic','leafHash'].map(k=>[k,esc(t[k]??'未记录')]))+notice('缺少图映射','Execute / Block 跳转需要稳定关联键。当前只能定位到运行 Task。','info'))+W.section('依赖提示',`<p>前驱：${esc(t.fanin.join(', ')||'未记录')}<br>下游：${esc(t.fanout.join(', ')||'未记录')}</p>${button('高亮已记录的关联任务','task-deps','btn-sm')}`)+W.section('原始事件',`<details><summary>事件参数</summary><div class="code-view"><pre>${esc(JSON.stringify(t.raw,null,2))}</pre></div></details>`);$('inspectorFooter').innerHTML=button('以此任务区间建立调查','task-case','btn-solid btn-sm');openInspector();}
 function showSearch(){dialog('搜索当前 Run',`<label class="ui-field">Task、Lane 或 ID<input id="searchInput" class="ui-input" type="search" placeholder="例如 o_group、AIC_0、r2t56" autocomplete="off" role="combobox" aria-expanded="true" aria-controls="searchResults"></label><div id="searchCount" class="small-meta"></div><div id="searchResults" class="ui-command-list" role="listbox" aria-label="搜索结果"></div>`);const input=$('searchInput');let hits=[],index=0;const render=()=>{const q=input.value.toLowerCase().trim();hits=q?(state.model?.tasks||[]).filter(t=>[t.rawName,t.shortId,t.taskId,t.laneName,t.process].some(s=>s.toLowerCase().includes(q))).slice(0,60):[];index=0;$('searchCount').textContent=q?(hits.length===60?'前 60 个匹配结果':hits.length+' 个结果'):'输入名称或 ID 定位完整 Trace';$('searchResults').innerHTML=hits.length?hits.map((t,i)=>`<button id="search-${i}" class="btn btn-ghost" type="button" role="option" aria-selected="${i===0}" data-search-index="${i}">${esc(t.rawName)}<small>Task · Rank ${t.rank} · ${esc(t.process)} / ${esc(t.laneName)} · ${W.us(t.start)}</small></button>`).join(''):q?'<p>没有匹配结果。</p>':'';if(hits.length)input.setAttribute('aria-activedescendant','search-0');else input.removeAttribute('aria-activedescendant');};const choose=i=>{const t=hits[i];if(!t)return;state.filter='all';state.kind='all';state.hiddenLanes.clear();updateLanes();closeDialog();revealTask(t);};input.addEventListener('input',render);input.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();choose(index);}if(['ArrowDown','ArrowUp'].includes(e.key)&&hits.length){e.preventDefault();index=(index+(e.key==='ArrowDown'?1:-1)+hits.length)%hits.length;$('searchResults').querySelectorAll('[role="option"]').forEach((el,i)=>{el.setAttribute('aria-selected',String(i===index));el.classList.toggle('is-selected',i===index);});input.setAttribute('aria-activedescendant','search-'+index);$('search-'+index).scrollIntoView({block:'nearest'});}});$('searchResults').addEventListener('click',e=>{const el=e.target.closest('[data-search-index]');if(el)choose(Number(el.dataset.searchIndex));});render();input.focus();}
 function showLanes(){if(!state.model)return;const processes=[...new Set(state.model.lanes.map(l=>l.process))];dialog('泳道显示',`<label class="ui-field">任务配色<select id="colorSelect" class="ui-input"><option value="semantic">语义配色</option><option value="engine">执行单元</option></select></label><label class="ui-field">进程范围<select class="ui-input" id="processFilter"><option value="worker">Worker 执行事件（默认）</option><option value="all">所有进程（分别显示）</option>${processes.map(p=>`<option value="${esc(p)}">${esc(p)}</option>`).join('')}</select></label><p>统计按 run / rank / pid / tid 分开。Worker 利用率为执行区间并集 ÷ 全 Run 时长。</p><div class="ui-command-list">${state.model.lanes.map((l,i)=>`<label class="inline small-meta"><input type="checkbox" data-lane-index="${i}" ${state.hiddenLanes.has(l.id)?'':'checked'}>${esc(l.process)} / ${esc(l.name)} · pid ${l.pid} / tid ${l.tid}</label>`).join('')}</div>${button('完成','close-dialog','btn-solid btn-sm')}`);$('colorSelect').value=A.renderer.color;$('processFilter').value=state.filter;$('processFilter').addEventListener('change',e=>{state.filter=e.target.value;updateLanes();syncLaneScope();});$('dialogBody').querySelectorAll('[data-lane-index]').forEach(el=>el.addEventListener('change',()=>{const id=state.model.lanes[Number(el.dataset.laneIndex)].id;if(el.checked)state.hiddenLanes.delete(id);else state.hiddenLanes.add(id);updateLanes();}));}
 function showDataMenu(){const model=state.model;dialog('数据源与工程',`<p>当前：<strong>${esc(model?.annotated?'DeepSeek V4 · decode_csa':model?.name||'加载中')}</strong></p><code>${esc(model?.runId||D.run)}</code><div class="inline">${button('恢复默认 Trace','default','btn-sm')}${button('选择 JSON','load','btn-sm')}${button('选择目录','load-folder','btn-ghost btn-sm')}</div><div class="inline">${button('读取当前 DeepSeek 工程','project-default','btn-sm')}${button('选择开发工程目录','project-import','btn-sm')}</div><p>默认数据来自仓库 Trace；没有采集的字段会明确标为“未记录”。</p>`);}
 document.addEventListener('click',e=>{
  const step=e.target.closest('[data-step]');if(step&&A.activeCase()){A.activeCase().step=Number(step.dataset.step);renderCase();persist();return;}
  const tab=e.target.closest('[data-cause-tab]');if(tab&&A.activeCase()){A.activeCase().tab=tab.dataset.causeTab;renderCase();persist();return;}
  const el=e.target.closest('[data-action]');if(!el)return;const c=A.activeCase();
  if(A.simulation.action(el))return;
  if(c?.preset&&W.report.action(el))return;
  switch(el.dataset.action){
   case 'close-dialog':closeDialog();break;case 'theme':document.documentElement.dataset.theme=document.documentElement.dataset.theme==='dark'?'light':'dark';A.renderer.draw();break;
   case 'load-menu':showDataMenu();break;case 'load':$('traceImport').click();break;case 'load-folder':$('folderImport').click();break;
   case 'default':closeDialog();loadDefault(0);break;case 'rank':closeDialog();loadDefault(Number(el.dataset.rank));break;case 'lanes':showLanes();break;
   case 'search':showSearch();break;case 'fit':A.renderer.fit();break;case 'zoom-in':A.renderer.zoom(1.6);break;case 'zoom-out':A.renderer.zoom(1/1.6);break;
   case 'brush':A.renderer.brush=!A.renderer.brush;el.setAttribute('aria-pressed',String(A.renderer.brush));el.classList.toggle('is-selected',A.renderer.brush);toast(A.renderer.brush?'在泳道上拖动，框选时间和泳道。':'已切回平移模式。');break;
   case 'investigations':setDiagnosticMode(!state.diagnosticMode);break;
   case 'close-investigation':closeInvestigation();break;case 'marker':showMarker(el.dataset.point);break;
   case 'focus-case':focusCase();break;
   case 'work-view':A.workarea.show(el.dataset.view);break;
   case 'project-default':A.project.preload();break;case 'project-import':$('projectImport').click();break;
   case 'candidate-import':$('candidateImport').click();break;
   case 'close-inspector':closeInspector();break;case 'task':{const t=A.caseModel()?.byId.get(el.dataset.taskId);if(t){state.filter='all';state.kind='all';state.hiddenLanes.clear();updateLanes();syncLaneScope();revealTask(t,{logical:el.dataset.logical==='true'});}break;}
   case 'task-case':if(state.task){const task=state.task;if(!state.model.byId.has(task.id))revealTask(task);createRange([task.start,task.end],[task.laneId]);}break;
   case 'task-deps':if(state.task){const t=state.task;A.renderer.related=new Set(state.model.tasks.filter(other=>other.process===t.process&&(t.fanin.includes(other.shortId)||t.fanout.includes(other.shortId))).map(t=>t.id));A.renderer.draw();toast('高亮 '+A.renderer.related.size+' 个匹配执行事件；仅基于 fanin / fanout 提示。');}break;
   case 'prev':case 'next':if(c){c.step=Math.max(1,Math.min(6,c.step+(el.dataset.action==='next'?1:-1)));renderCase();persist();}break;
   case 'questions':W.showQuestions();break;case 'question':c.step=Number(el.dataset.targetStep);c.tab=el.dataset.targetTab||'lifecycle';closeDialog();renderCase();persist();break;
   case 'evidence':W.evidence();break;case 'evidence-tasks':closeDialog();c.step=2;renderCase();break;case 'save-note':c.note=$('caseNote').value;persist();toast('调查备注已保存。');break;
   case 'mapping':dialog('为什么不能跳转到执行图',notice('缺少稳定关联键','需要 rootHash 关联 Execute、callOpMagic 关联调用、leafHash 关联 Block。当前 Trace 的 taskId 只能标识运行任务。','info'));break;
   case 'graph':W.graph();break;
   case 'collect':dialog('补采集要求',`<p>这是采集端需要实现的字段协议，不是可直接运行的配置。</p><div class="code-view"><pre>${esc(JSON.stringify(W.collect(),null,2))}</pre></div>${button('导出采集清单','export-collection','btn-solid btn-sm')}`);break;
   case 'export-collection':download(c.case_id+'-collection.json',W.collect());break;case 'export-case':download(c.case_id+'.json',c);break;
   case 'controls':c.step=5;renderCase();break;case 'experiment':A.experiments.form();break;
   case 'export-experiment':if(c?.experiment)download(c.experiment.experiment_id+'.json',c.experiment);else A.experiments.form();break;
   case 'result-template':if(c?.experiment)download(c.experiment.experiment_id+'-result-template.json',A.experiments.template());break;
   case 'import-result':$('resultImport').click();break;case 'run-diff':A.experiments.diff();break;case 'trace-diff':A.workarea.diff();break;
   case 'close-bottom':showWorkflow();break;
  }
 });
 $('rankSelect').addEventListener('change',e=>loadDefault(Number(e.target.value),Boolean(A.activeCase())));$('laneScopeSelect').addEventListener('change',e=>{if(e.target.value==='custom')return;[state.filter,state.kind]=e.target.value.split('|');state.hiddenLanes.clear();updateLanes();});
 document.addEventListener('change',e=>{if(e.target.id==='colorSelect'){A.renderer.color=e.target.value;A.renderer.draw();}});
 $('traceImport').addEventListener('change',e=>loadLocal(e.target.files[0]));$('resultImport').addEventListener('change',e=>A.experiments.importFile(e.target.files[0]));
 $('folderImport').addEventListener('change',e=>{const files=[...e.target.files].filter(f=>f.name.endsWith('.json'));dialog('选择目录中的 Trace',`<p>按文件名列出候选，不会把目录里的其他 JSON 当作 Trace。</p><div class="ui-command-list">${files.map((f,i)=>button(esc(f.webkitRelativePath),'folder-file','btn-ghost',`data-file-index="${i}"`)).join('')||'<p>没有 JSON 文件。</p>'}</div>`);$('dialogBody').querySelectorAll('[data-file-index]').forEach(el=>el.addEventListener('click',()=>loadLocal(files[Number(el.dataset.fileIndex)])));});
 $('traceSurface').addEventListener('dragover',e=>e.preventDefault());$('traceSurface').addEventListener('drop',e=>{e.preventDefault();loadLocal(e.dataTransfer.files[0]);});
 document.addEventListener('keydown',e=>{if(e.key==='/'&&!e.target.matches('input,textarea,select')&&!$('detailDialog').open){e.preventDefault();showSearch();}if(e.key==='Escape'&&!$('detailDialog').open&&!$('mobileSheet').open){A.renderer.brush=false;$('brushButton').setAttribute('aria-pressed','false');$('brushButton').classList.remove('is-selected');if($('investigationOverlay').contains(document.activeElement)||!inspectorOpen())closeInvestigation();else closeInspector();}});
 $('projectImport').addEventListener('change',e=>A.project.importFiles([...e.target.files]));$('candidateImport').addEventListener('change',e=>A.workarea.importCandidate(e.target.files[0]));
 A.renderer.diagnosticMode=state.diagnosticMode;renderInspectorOverview();renderEmptyInvestigation();renderMarkers();
 const file=new URLSearchParams(location.search).get('file');
 if(file){(async()=>{try{const url=new URL(file,location.href);if(url.origin!==location.origin)throw Error('仅支持当前服务中的本地 Trace URL。');const r=await fetch(url);if(!r.ok)throw Error('HTTP '+r.status);const raw=await r.json();const model=TraceModel.parse(raw,{runId:'url-'+crypto.randomUUID(),rank:0,name:url.pathname.split('/').at(-1),source:url.pathname});models.set(model.runId,model);activate(model,model.runId);}catch(e){setTraceState('Trace 加载失败',esc(e.message),true);}})();}else loadDefault(0).then(()=>{if(state.simulationMode&&state.model?.annotated){showMarker('preset-a2a');A.activeCase().step=4;renderCase();}});
})();
