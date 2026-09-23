/* Domain coordination. Layout is the unchanged PTO Pass IR layout engine;
 * cards and trace bars use the shared Pattern APIs. No sample graphs/runs. */
window.InvestigationWorkarea=function(A){
 const U=InvestigationUI,{esc,button,notice,table,badge}=U,$=id=>document.getElementById(id);
 const views=new Map(),renders=[];let mode='trace',candidate=null;
 const cleanup=()=>{for(const r of renders.splice(0))r.destroy?.();};
 function reset(){cleanup();views.clear();candidate=null;$('mainEvidence').innerHTML='';$('mainEvidence').dataset.view='';show('trace');}
 function tabs(){const el=$('workTabs');el.hidden=!views.size;el.innerHTML=[['trace','Trace'],...views].map(([id,v])=>`<button class="tab-control-item ${id===mode?'is-selected':''}" data-action="work-view" data-view="${id}" aria-pressed="${id===mode}">${esc(typeof v==='string'?v:v.title)}</button>`).join('');}
 function show(id){
  mode=id;tabs();const trace=id==='trace';$('mainEvidence').hidden=trace;$('traceCanvas').style.visibility=trace?'visible':'hidden';$('markerActions').hidden=!trace||!A.state.diagnosticMode;
  document.querySelectorAll('.trace-toolbar select,.trace-view-actions button,#brushButton').forEach(el=>{el.disabled=!trace;});
  if(trace){$('rankSelect').disabled=!A.state.model?.annotated;A.renderer?.draw();}else if(views.has(id)){const v=views.get(id);if($('mainEvidence').dataset.view!==id){cleanup();$('mainEvidence').innerHTML=v.html;$('mainEvidence').dataset.view=id;v.mount?.();}}
 }
 function content(id,title,html,mount){views.set(id,{title,html,mount});$('mainEvidence').dataset.view='';show(id);}
 function drawGraph(el,graph,other,changeKind,onScale){
  if(!graph.nodes.length){el.innerHTML=notice('没有可映射的局部图',esc(graph.warnings.join('；')));return;}
  const layout=computeLayout(graph,graph.simulation?{compact:true,nodeWidth:230,hStep:258,vGap:28,nodeHeightsCompact:{incast:98,outcast:98,tensor:98,op:98,group:148}}:{compact:true,nodeWidth:225});
  el.classList.toggle('is-simulation-ir',Boolean(graph.simulation));
  if(graph.simulation){
   // Trim empty layout margins so the focused Pass slice stays legible in the drawer.
   const positions=[...layout.positions.values()],left=Math.min(...positions.map(p=>p.x)),top=Math.min(...positions.map(p=>p.y));
   for(const p of positions){p.x+=12-left;p.y+=12-top;}
   // A diff is easier to scan when stable nodes keep the same baseline.
   // Added branches opt into another row without moving the unchanged chain.
   for(const n of graph.nodes)layout.positions.get(n.id).y=12+(n.layoutRow||0)*124;
   layout.canvasW=Math.max(...positions.map(p=>p.x+p.w))+12;layout.canvasH=Math.max(...positions.map(p=>p.y+p.h))+12;
  }
  const stage=document.createElement('div');stage.className='ir-stage';stage.style.width=layout.canvasW+'px';stage.style.height=layout.canvasH+'px';
  const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('width',layout.canvasW);svg.setAttribute('height',layout.canvasH);
  for(const edge of graph.edges){const p=layout.positions.get(edge.source),q=layout.positions.get(edge.target);if(!p||!q)continue;const path=document.createElementNS(ns,'path');const x=p.x+p.w,y=p.y+p.h/2,end=q.x,ey=q.y+q.h/2;path.setAttribute('d',`M${x},${y} C${x+30},${y} ${end-30},${ey} ${end},${ey} l-7,-4 m7,4 l-7,4`);path.setAttribute('class',edge.kind==='control'?'ir-control-edge':'ir-data-edge');path.dataset.source=edge.source;path.dataset.target=edge.target;const title=document.createElementNS(ns,'title');const relation=edge.symbol==='源码顺序'?'源码顺序':edge.kind==='control'?'控制关系':'数据依赖';title.textContent=`${relation}${edge.symbol&&edge.symbol!=='源码顺序'?' · '+edge.symbol:''} · ${edge.source_ref.simulation?'模拟依赖':'L'+edge.source_ref.line}`;path.append(title);svg.append(path);}stage.append(svg);
  const signatures=new Set(other.nodes.map(n=>n.signature)),nodeCards=new Map();
  for(const n of graph.nodes){const p=layout.positions.get(n.id),wrap=document.createElement('div');wrap.className='ir-node';wrap.dataset.nodeId=n.id;Object.assign(wrap.style,{left:p.x+'px',top:p.y+'px',width:p.w+'px'});
   const same=signatures.has(n.signature),frame=graph.simulation?{width:p.w,height:98,minHeight:98}:{width:p.w},node=PtoPassIrGraphNodePattern.buildNodeCardElement({id:n.id,type:n.type,frame,data:{semanticLabel:n.label,symbol:n.symbol,opcode:n.label}},{compact:true});
   nodeCards.set(n.id,node);
   const status=same?'未变化':changeKind==='added'?'新增':'移除';wrap.classList.add(same?'is-unchanged':'is-'+changeKind);
   const provenance=n.source.simulation?(n.source.line?`模拟 · 参考 L${n.source.line}`:'模拟节点'):'L'+n.source.line;
   node.tabIndex=0;node.title=n.label;node.setAttribute('aria-label',`${n.label} · ${status} · ${provenance}；聚焦查看详情`);
   const label=document.createElement('div');label.className='small-meta';label.textContent=n.source.simulation?`${status} · L${n.source.line}`:`${status} · ${provenance}`;wrap.append(node,label);stage.append(wrap);
   const show=(event)=>{tip.hidden=false;const sourceDetail=n.source.simulation&&n.source.file?`${n.source.file}:L${n.source.line}`:`${n.source.function} · ${provenance}`;tip.innerHTML=`<strong>${esc(n.label)}</strong><dl><dt>变化</dt><dd>${status}</dd><dt>节点类型</dt><dd>${esc(n.type==='incast'?'外部输入':n.type==='outcast'?'任务结果':n.type==='tensor'?'张量':'运算')}</dd><dt>${n.source.simulation?'参考':'源码位置'}</dt><dd>${esc(sourceDetail)}</dd>${n.source.basis?`<dt>数据口径</dt><dd>${esc(n.source.basis)}</dd>`:''}${n.symbol?`<dt>结果</dt><dd>${esc(n.symbol)}</dd>`:''}</dl><pre>${esc(n.text)}</pre>`;const box=el.getBoundingClientRect(),x=event?.clientX??box.left+el.clientWidth/2,y=event?.clientY??box.top+el.clientHeight/2;requestAnimationFrame(()=>{const left=Math.max(el.scrollLeft+8,Math.min(el.scrollLeft+el.clientWidth-tip.offsetWidth-8,el.scrollLeft+x-box.left+12)),top=Math.max(el.scrollTop+8,Math.min(el.scrollTop+el.clientHeight-tip.offsetHeight-8,el.scrollTop+y-box.top+12));tip.style.left=left+'px';tip.style.top=top+'px';});};
   node.addEventListener('pointerenter',show);node.addEventListener('pointermove',show);node.addEventListener('pointerleave',()=>tip.hidden=true);node.addEventListener('focus',show);node.addEventListener('blur',()=>tip.hidden=true);
  }
  const scene=document.createElement('div');scene.className='ir-scene';scene.append(stage);const tip=document.createElement('div');tip.className='ir-node-tooltip panel-shell';tip.setAttribute('role','tooltip');tip.hidden=true;el.append(scene,tip);
  let scale=1;const applyScale=value=>{scale=Math.max(.15,Math.min(2.5,value));stage.style.transform=`scale(${scale})`;stage.style.transformOrigin='0 0';scene.style.width=layout.canvasW*scale+'px';scene.style.height=layout.canvasH*scale+'px';};const fitScale=()=>Math.min(1,(el.clientWidth-16)/layout.canvasW,(el.clientHeight-16)/layout.canvasH);const fit=()=>applyScale(fitScale());requestAnimationFrame(()=>{
   for(const path of svg.querySelectorAll('path')){const p=layout.positions.get(path.dataset.source),q=layout.positions.get(path.dataset.target),x=p.x+p.w,y=p.y+nodeCards.get(path.dataset.source).offsetHeight/2,end=q.x,ey=q.y+nodeCards.get(path.dataset.target).offsetHeight/2;path.setAttribute('d',`M${x},${y} C${x+25},${y} ${end-25},${ey} ${end},${ey} l-7,-4 m7,4 l-7,4`);}
   if(graph.simulation){layout.canvasH=Math.max(...[...nodeCards].map(([id,card])=>layout.positions.get(id).y+card.parentElement.offsetHeight))+12;stage.style.height=layout.canvasH+'px';svg.setAttribute('height',layout.canvasH);}
   fit();
  });
  el.addEventListener('wheel',e=>{if(!e.ctrlKey&&!e.metaKey)return;e.preventDefault();const next=scale*Math.exp(-e.deltaY*.005);onScale?onScale(next):applyScale(next);},{passive:false});
  return {applyScale,fitScale};
 }
 // Rendering primitive only. Choosing a diagnostic Pass pair requires an
 // upstream investigation result; this renderer never selects by preset/name.
 function renderPassComparison(root,{before,after,a,b}){
  const added=b.nodes.filter(n=>!a.nodes.some(v=>v.signature===n.signature)).length,removed=a.nodes.filter(n=>!b.nodes.some(v=>v.signature===n.signature)).length;
  const passLabel=p=>esc(p.label||`${String(p.index).padStart(2,'0')} · ${p.name}`);
  root.innerHTML=`${a.simulation?'':`<header class="pass-comparison-header"><div><div class="inline"><strong>${passLabel(before)} → ${passLabel(after)}</strong>${badge(added+removed?`新增节点 ${added} · 移除节点 ${removed}`:'结构未变化','neutral')}</div><p class="small-meta">函数：${esc(a.scope||b.scope)} · 实线为数据依赖，虚线为控制关系或源码顺序</p></div></header>`}<div class="ir-pair">${[[before,'变换前','before'],[after,'变换后','after']].map(([p,label,side])=>`<section><div class="row-between"><h3>${label}</h3><span class="small-meta">${passLabel(p)}</span></div><div class="ir-viewport" data-ir-side="${side}" tabindex="0" aria-label="${label} IR 图；悬停或聚焦节点查看详情，Ctrl 或 Command 加滚轮缩放"></div></section>`).join('')}</div>`;
  const graphViews=[],syncScale=value=>graphViews.forEach(view=>view?.applyScale(value));
  graphViews.push(drawGraph(root.querySelector('[data-ir-side="before"]'),a,b,'removed',syncScale),drawGraph(root.querySelector('[data-ir-side="after"]'),b,a,'added',syncScale));
  requestAnimationFrame(()=>requestAnimationFrame(()=>syncScale(Math.min(...graphViews.filter(Boolean).map(view=>view.fitScale())))));
 }
 function compareModels(left,right,title,rankCompare=false,simulation=false){
  const paired=left.lanes.filter(l=>l.worker),rightLanes=right.lanes.filter(l=>l.worker);
  // Exact process + lane name is a lane browsing correspondence, not a Task identity proof.
  const lanes=paired.map(l=>({a:l,b:rightLanes.find(r=>r.process===l.process&&r.name===l.name)}));
  const unknown=rightLanes.filter(l=>!lanes.some(p=>p.b===l));
  const emptyLane=(lane,side)=>({...lane,id:'unmatched-'+side+'-'+lane.id,name:simulation?lane.name+' · 无搬运':'未匹配 · '+lane.name,tasks:[],busy:0,utilization:0,worker:false});
  const displayLanes=[lanes.map(p=>p.a).concat(unknown.map(l=>emptyLane(l,'baseline'))),lanes.map(p=>p.b||emptyLane(p.a,'candidate')).concat(unknown)];
  content(simulation?'simulation':rankCompare?'ranks':'diff',title,`<header class="row-between"><strong>${esc(title)}</strong>${simulation?button('返回模拟调查','simulation-return','btn-sm'):''}<label class="compact-field">对齐<select id="alignment" class="ui-input compact-input"><option value="run">按每次运行的起点</option><option value="selection">按两侧选中的事件</option></select></label></header><p>${simulation?'两侧全部为模拟事件，使用统一相对时钟。右侧去掉 GM 往返搬运后，发布、等待结束与下游启动均提前 320 µs；这不是实测优化结果。':rankCompare?'这里比较同一次运行中的两个 Rank，不是优化前后对比；当前也没有证据证明两个 Rank 的时钟完全同步。':'这里比较修改前后的两次独立 Trace；泳道仅按进程和名称排列，尚未证明两次编译中的 Task 可以逐一对应。'}</p><p class="small-meta">${simulation?'模拟参数：DEMO_PUBLISH_TILE_ROWS 64 → 32 · 主 Trace 原始事件未修改':`${esc(left.runId)} / R${left.rank} → ${esc(right.runId)} / R${right.rank} · Worker 事件 ${paired.reduce((n,l)=>n+l.tasks.length,0)} → ${rightLanes.reduce((n,l)=>n+l.tasks.length,0)} · 未匹配泳道 ${lanes.filter(p=>!p.b).length} / ${unknown.length} · Copy 字节和生命周期分段未采集`}</p><div class="trace-pair">${[left,right].map((m,i)=>`<section><h3>${i?(rankCompare?'Rank '+m.rank:'修改后'):(rankCompare?'Rank '+m.rank:'修改前')} · ${esc(m.name)}</h3><div class="compare-trace"><canvas id="compare-${i}" tabindex="0" aria-label="${simulation?'模拟':''}${i?'候选或对照':'基线'}泳道"></canvas></div></section>`).join('')}</div><p>${simulation?'本示例只展示局部场景。实际正确性和端到端收益尚未验证。':'两侧时间范围和纵向滚动会联动。点击事件可查看原始记录；泳道名称相同不代表 Task 已稳定匹配。端到端收益还需要正确性验证和重复测量。'}</p>`,()=>{
   let syncing=false;const rr=[],selected=[null,null],offsets=[0,0];
   [left,right].forEach((m,i)=>{const r=new TraceRenderer($('compare-'+i),{onTask:t=>{selected[i]=t;r.selected=t.id;r.draw();simulation?A.inspectSimulationTask(t):A.selectTask(t);},onMarker:()=>{},onRange:()=>{},onView:s=>{if(syncing||rr.length<2)return;const target=rr[1-i],view=s.time_range.map(t=>t-offsets[i]+offsets[1-i]);if(target.view.every((t,j)=>Math.abs(t-view[j])<.001)&&target.scroll===s.scroll)return;syncing=true;target.view=view;target.scroll=s.scroll;target.paint();syncing=false;}});r.setModel(m);r.setLanes(displayLanes[i]);rr.push(r);renders.push(r);});
   const duration=Math.max(left.duration,right.duration);rr.forEach(r=>{r.view=[0,duration];r.draw();});
   const observer=new ResizeObserver(()=>rr.forEach(r=>r.draw()));observer.observe($('mainEvidence'));renders.push({destroy:()=>observer.disconnect()});
   $('alignment').onchange=e=>{if(e.target.value==='selection'&&selected.some(t=>!t)){e.target.value='run';return A.toast('请先在两侧分别选择真实事件作为锚点。');}offsets.splice(0,2,...(e.target.value==='selection'?selected.map(t=>t.start):[0,0]));rr[0].view=[offsets[0]-100,offsets[0]+Math.min(1000,duration)];rr[0].draw();};
  });
 }
 async function compareRanks(){const c=A.activeCase();if(!c?.preset)return;const id=c.case_id;try{const ranks=[];for(const rank of [0,1]){const key='default-'+rank;if(!A.models.has(key)){const source='../../Data/DeepseekV4/'+INVESTIGATION_DATA.cases[c.preset].ranks[rank].trace;const response=await fetch(source);if(!response.ok)throw Error('HTTP '+response.status);A.models.set(key,TraceModel.parse(await response.json(),{runId:c.run_id,rank,name:'decode_csa',source,annotated:true}));}ranks.push(A.models.get(key));}if(A.activeCase()?.case_id===id)compareModels(...ranks,'跨 Rank 对照',true);}catch(e){A.toast(e.message);}}
 async function importCandidate(file){if(!file)return;const ex=A.activeCase()?.experiment,id=ex?.experiment_id;if(!ex)return A.toast('先定义实验。');try{
  const text=await file.text(),raw=JSON.parse(text);const runId=raw.run_manifest?.run_id||raw.run_id;
  if(typeof runId!=='string'||!runId.trim()||runId===ex.baseline_run)throw Error('需要独立 Run ID；同一 Run 的两个 Rank 不能作为优化 Diff。');
  if(ex.result&&ex.result.candidate_run!==runId)throw Error('候选 Trace 与已导入指标的 candidate_run 不一致，不能合并为同一实验结果。');
  const manifest=raw.run_manifest||raw;if(manifest.experiment_id!==id||manifest.case_id!==ex.case_id)throw Error('候选 Trace 的 run_manifest 必须关联当前 experiment_id 与 case_id。');
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text)))).map(n=>n.toString(16).padStart(2,'0')).join('');
  const model=TraceModel.parse(raw,{runId,rank:Number(manifest.rank??ex.baseline_rank),name:file.name,source:file.name+' · sha256:'+hash});
  const baseline=A.caseModel();if(model.rank!==baseline.rank)throw Error('候选 Rank 与调查基线不同，需明确 Rank 对应关系。');
  if(A.activeCase()?.experiment?.experiment_id!==id)return;
  A.models.set('candidate-'+id,model);candidate={experimentId:id,model};ex.candidate_trace={run_id:runId,sha256:hash,file:file.name};A.persist();A.renderCase();compareModels(baseline,model,'泳道对比');
 }catch(e){A.dialog('候选 Trace 未接入',notice('请检查产物身份与结构',esc(e.message)));}finally{$('candidateImport').value='';}}
 function diff(){if(candidate?.experimentId===A.activeCase()?.experiment?.experiment_id)return compareModels(A.caseModel(),candidate.model,'泳道对比');content('diff','实验对比',notice('尚无候选 Trace','先加载真实候选 Trace。已有指标可以单独对照，但不等于完成泳道 Diff。','info')+button('导入候选 Trace','candidate-import','btn-sm'));}
 function clearSimulation(){views.delete('simulation');if(mode==='simulation'){cleanup();$('mainEvidence').innerHTML='';$('mainEvidence').dataset.view='';show('trace');}else tabs();}
 return {show,content,renderPassComparison,compareRanks,importCandidate,diff,reset,clearSimulation,compareSimulation:(left,right)=>compareModels(left,right,'模拟泳道对比',false,true)};
};
