/* Task compute graph. Counts are source-level invocations, never runtime cost. */
(function(global){
'use strict';
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function createModel({anatomy,source,preLaunch}){
 const lines=source.split('\n'),line=s=>lines.findIndex(l=>l.includes(s))+1;
 const tasks=anatomy.tasks.map(t=>{
  const P=t.pages.length,S=t.stacks.length,T=S+preLaunch;
  const n=(id,title,role,count,unit,formula,needle,detail)=>({id,title,role,count,unit,formula,line:line(needle),detail});
  const nodes=[
   n('q','Q Tile','tensor',1,'AIC / Task','1','q_tile = pl.load',`BF16 · Mat · ${t.q.shape.join('×')} / valid ${t.q.validShape.join('×')}`),
   n('k','K Cache Pages','tensor',P,'AIC / Task','ceil(seq_len / BLOCK_SIZE)','qk_k_page = pl.load',`BF16 · GM → Mat · Page ${anatomy.pageTokens}×${anatomy.headDim} · physical ID unknown`),
   n('qk','QK Matmul','operation',P,'AIC / Task','page_count','qk_score_page = pl.matmul','每个有效逻辑 Page 一次 Matmul；尾页仍加载完整 Page。'),
   n('scores','Scores','tensor',P,'AIC stores / Task','page_count','score_transfer,',`score_transfer · FP32 · GM · slot tile ${anatomy.rowTile}×${anatomy.stackTokens}`),
   n('softmax','Online Softmax','operation',S,'每 AIV Lane / Task','stack_count; tick < stack_count','score_scaled = pl.tile.muls','Scale → valid_cols Mask → online max/sum → BF16 Probability。两条 Lane 分别处理 2 / 3 Q Heads。'),
   n('probability','Probability','tensor',S,'每 AIV Lane stores / Task','stack_count','probability_valid = pl.set_validshape',`probability_transfer · BF16 · GM · slot ${anatomy.rowTile}×${anatomy.stackTokens}`),
   n('v','V Cache Pages','tensor',P,'AIC / Task','page_count','value_cache_bsnd,',`BF16 · GM → Mat · ${anatomy.pageTokens}×${anatomy.headDim} / Page`),
   n('pv','PV Matmul','operation',P,'AIC / Task','Σ pages per Stack = page_count','pv0 = pl.matmul','每 Stack 首 Page 为 Matmul，其余 Page 为 Matmul Accumulate。'),
   n('partial','Partial Context','tensor',S,'AIC stores / Task','stack_count','pl.store(pv3,',`pv_transfer · FP32 · GM · ${anatomy.rowTile}×${anatomy.headDim}`),
   n('accumulate','Context Accumulate','operation',S,'每 AIV Lane / Task','stack_count; tick >= PRE_LAUNCH','pl.add(pl.row_expand_mul(o_iter','o_next = o_previous × delayed_rescale + partial_context。'),
   n('normalize','Output Normalize','operation',1,'每 AIV Lane / Task','1, after loop','ctx = pl.cast','完成所有 Stack 后 o / l_sum，再 Cast BF16。'),
   n('output','BF16 Output','tensor',1,'每 AIV Lane store / Task','1','                ctx,',`out2d · GM · valid ${t.output.shape.join('×')} · 两条 Lane 分别写 2 / 3 行`),
   n('m','m · Max','state',S,'每 AIV Lane / Task','next_m = max(local_m, m_iter)','next_m = pl.maximum','FP32 · Vec · per Lane 8×1 / valid 2或3×1；m[-1] = −3e38；m[s] 参与下一 Stack 的 online max。'),
   n('l','l_sum','state',S,'每 AIV Lane / Task','l[s] = exp(m[s−1]−m[s]) × l[s−1] + sum(p[s])','next_l = pl.add','FP32 · Vec · per Lane 8×1 / valid 2或3×1；l[-1] = 0；最终 l_sum 用于 Output Normalize。'),
   n('rescale','Rescale FIFO','state',T,'每 AIV Lane / Task','FIFO shift = stack_count + PRE_LAUNCH; rescale_after produced = stack_count','pending0_after = pl.tile.move','初始化 (1,1)。有效生产分支产生 rescale_after ×stack_count；pending0/pending1 每 Tick 移位；消费分支读取延迟两拍的 pending0。'),
   n('o','o · Context','state',S,'每 AIV Lane / Task','o[s] = o[s−1] × rescale[s] + pv[s]','o_after = pl.yield_','FP32 · Vec · per Lane 8×128 / valid 2或3×128；o[-1] = 0；跨 Stack 累积 FP32 Context，最终归一化。')];
  const task={...t,nodes,pageCount:P,stackCount:S,tickCount:T,preLaunch,stateSteps:t.stacks.map((stack,index)=>({index,previous:index?`S${index-1}`:'初始状态',next:`S${index}`,produceTick:index,consumeTick:index+preLaunch,pages:stack.pages}))};
  task.graph=semanticModel(task);return task;
 });return {tasks,source,anatomy};
}

// Adapt Task facts to the same semantic schema used by the outer graph. All
// composite subgraphs, their ports and source evidence come from the source model.
const aliases={k:'k-cache',v:'v-cache',accumulate:'context-acc',normalize:'output-norm'};
const semanticId=id=>aliases[id]||id;
const localId=id=>Object.keys(aliases).find(k=>aliases[k]===id)||id;
function semanticModel(task){
 const base=global.PTO_PYPA_SEMANTIC_GRAPH,copy=v=>JSON.parse(JSON.stringify(v));
 const ids=['q','k-cache','qk','scores','scale-mask','softmax','probability','v-cache','pv','context-acc','output-norm','output'];
 const nodes=ids.map(id=>copy(base.nodes.find(n=>n.id===id)));
 // PV's output tile is also a first-class root tensor; its subgraph retains the
 // Mat-memory partial. No physical page IDs or addresses are synthesized.
 const partial=copy(base.nodes.find(n=>n.id==='pv').composite.subgraph.nodes.find(n=>n.id==='partial'));
 partial.id='partial';partial.name='Partial Context · pv_transfer';
 partial.data={role:'intermediate',granularity:'stack',entity:'context.partial',symbol:'pv_transfer',dtype:'FP32',shape:{allocated:['TRANSFER_ROWS','HEAD_DIM']},storage:{space:'GM'}};
 partial.provenance=[{kind:'resolved',origin:'source',fields:Object.keys(partial.data).map(k=>'data.'+k),note:'FP32 producer and transfer-row addressing in paged_attention_pypto.py.'}];
 nodes.splice(nodes.findIndex(n=>n.id==='context-acc'),0,partial);
 for(const id of ['m','l','rescale','o']){
  const fact=task.nodes.find(n=>n.id===id);
  const state={dtype:'FP32',granularity:'stack',initial:id==='m'?'−3e38':id==='rescale'?'pending0 = 1; pending1 = 1':'0',update:fact.formula,feedback:fact.detail};
  nodes.push({id,type:'state',name:fact.title,description:fact.detail,source:{file:'paged_attention_pypto.py',ranges:[{start:fact.line,end:fact.line}],symbols:[id==='l'?'l_sum':id]},ports:{inputs:{default:{}},outputs:{default:{}}},state,provenance:[{kind:'resolved',origin:'source',fields:Object.keys(state).map(k=>'state.'+k),note:'Loop-carried state from pl.range init_values and pl.yield_.'}]});
 }
 const byId=new Map(nodes.map(n=>[n.id,n]));
 // Extra semantic state ports resolve through the ordinary composite mechanism.
 const bind=(id,side,name,node,port='default')=>{const n=byId.get(id);n.ports[side][name]={};n.composite.portMappings[side][name]={node,port,relation:'control'};};
 bind('softmax','inputs','previous','previous');
 bind('output-norm','inputs','final','normalize');
 const edges=copy(base.edges.filter(e=>ids.includes(e.s)&&ids.includes(e.t)&&!['context-acc','output-norm'].includes(e.t)&&!['softmax>context-acc'].includes(e.s+'>'+e.t)));
 const edge=(s,t,label,type='data',sourcePort='default',targetPort='default',id)=>edges.push({id:id||`${type}:${s}:${t}`,s,t,label,type,sourcePort,targetPort});
 edge('pv','partial','pv_transfer','data','partial');edge('partial','context-acc','Partial Context','data','default','partial');
 edge('softmax','m','next_m','state','state');edge('m','softmax','previous Stack · m','feedback','default','previous');
 edge('softmax','l','next_l','state','state');edge('l','softmax','previous Stack · l_sum','feedback','default','previous');
 edge('softmax','rescale','rescale_after','state','state');edge('rescale','rescale','FIFO shift / Tick','feedback');
 edge('rescale','context-acc','pending0 · delayed rescale','state','default','state');
 edge('context-acc','o','o_next','state','Context');edge('o','context-acc','previous Stack · o','feedback','default','state');
 edge('l','output-norm','final l_sum','data','default','final');edge('o','output-norm','final o','data','default','final');
 const region=(id,name,members)=>({id,type:'region',name,description:name,members,source:{file:'paged_attention_pypto.py',ranges:[{start:759,end:873}],symbols:[]},provenance:[{kind:'resolved',origin:'source',fields:['members'],note:'Task-local semantic regions; not runtime timing.'}]});
 const graph={schemaVersion:2,id:`attention-task-${task.id}`,sourceFiles:base.sourceFiles,constants:base.constants,runtimeDims:base.runtimeDims,nodes,edges,
  groups:[region('task-inputs','Task Inputs',['q','k-cache','v-cache']),region('task-compute',`${task.stackCount} Stacks · ${task.pageCount} Pages total`,['qk','scores','scale-mask','softmax','probability','pv','partial','context-acc','m','l','rescale','o']),region('task-output','Task Output',['output-norm','output'])],
  synchronizations:copy(base.synchronizations)};
 // The root Q is the current Task tile, not the full Q projection tensor.
 const q=byId.get('q');q.name='Q Tile';q.data.shape={allocated:task.q.shape,valid:task.q.validShape};
 return graph;
}
// A Stack is a projection of the existing static dependency model, not a
// separate graph or a sample of runtime tensor values.
function stackEvolution(task,index){
 if(!Number.isInteger(index)||index<0||index>=task.stackCount)return null;
 const pipeline=global.PtoPagedAttentionIntro.model.pipelineFor(task.seqLen);
 const dependencies=global.PtoPagedAttentionStackDependencyModel.build(pipeline,task.graph.synchronizations);
 const node=stage=>dependencies.nodes.find(n=>n.id===`s${index}:${stage}`);
 const fifo=dependencies.edges.find(e=>e.id===`state:rescale:s${index}`);
 const previous=index?`S${index-1}`:'初始状态',next=index+1<task.stackCount?`S${index+1}`:null;
 return {index,previous,next,produceTick:node('softmax').tick,consumeTick:node('accumulate').tick,
  states:{m:{previous:index?`m${index-1}`:'m初始 = −3e38',updated:`m${index}`},l:{previous:index?`l${index-1}`:'l初始 = 0',updated:`l${index}`},o:{previous:index?`o${index-1}`:'o初始 = 0',updated:`o${index}`}},
  fifo:{versions:fifo.versions,source:fifo.source,explanation:fifo.explanation},
  stateEdges:dependencies.edges.filter(e=>e.kind==='state'&&(e.to===`s${index}:softmax`||e.to===`s${index}:accumulate`)),
  dataEdges:dependencies.edges.filter(e=>e.kind==='data'&&e.id.startsWith(`data:s${index}:`))};
}
// Project existing logical Pages; no runtime values or physical IDs are synthesized.
function pageInstances(task,nodeId){
 const root=nodeId?.split('/')[0];
 if(!['qk','pv'].includes(root))return [];
 return task.stacks.map(stack=>({index:stack.index,pages:stack.pages.map((id,offset)=>{
  const page=task.pages[id];
  const full=stack.pages.length===(task.pagesPerStack||4),sourceSymbol=root==='qk'?(full?'qk_score_page':`qk_tail${offset}`):(full?`pv${offset}`:stack.pages.length===1?'pv_tail1':`pv_tail${stack.pages.length}_${offset}`);
  return {index:page.index,stack:stack.index,tokens:page.tokens,sourceSymbol,api:root==='pv'&&offset>0?'pl.matmul_acc':'pl.matmul',input:root==='qk'?'Q Tile + K Cache Page':'Probability Page + V Cache Page',output:root==='qk'?'FP32 Scores 的当前 Page 区间':'当前 Stack 的 FP32 Partial Context'};
 })}));
}
function evolutionMarkup(e){
 if(!e)return '';
 const s=e.states,v=e.fifo.versions;
 return `<section class="viz2-stack-evolution" aria-label="S${e.index} 状态演化"><div class="viz2-stack-evolution__lane"><b>Previous · ${esc(e.previous)}</b><span>${esc(s.m.previous)} · ${esc(s.l.previous)} · ${esc(s.o.previous)}</span></div><div class="viz2-stack-evolution__lane"><b>Current · S${e.index}</b><span>Tick ${e.produceTick}: Scores → Online Softmax → ${esc(s.m.updated)}, ${esc(s.l.updated)}</span><small>m_new = max(m_old, row_max(Scores)); l_new = rescale × l_old + sum(exp)</small><span>Tick ${e.consumeTick}: PV Partial + delayed Rescale + ${esc(s.o.previous)} → ${esc(s.o.updated)}</span><small>o_new = delayed_rescale × o_old + PV Partial</small></div><div class="viz2-stack-evolution__lane"><b>Updated / Next</b><span>${esc(s.m.updated)} · ${esc(s.l.updated)} · ${esc(s.o.updated)}${e.next?` → ${esc(e.next)} Previous`:' → Output Normalize'}</span></div><div class="viz2-stack-evolution__fifo"><b>Rescale FIFO · 静态逻辑 Tick</b><span>produce → ${esc(v[0])} → ${esc(v[1])} → ${esc(v[2])}</span><small>Softmax 先生产；Context 在 ${e.consumeTick-e.produceTick} Tick 后读取 pending0，补偿此前 m 更新造成的缩放。</small></div></section>`;
}
function mount(host,{model,onTask,onFocus,onView,onNavigate}){
 host.classList.add('viz2-compute');
 host.innerHTML='<div class="viz2-anatomy-taskbar" data-compute-toolbar></div><div class="viz2-compute-canvas"></div>';
 const toolbar=host.querySelector('[data-compute-toolbar]');
 toolbar.innerHTML=`<div class="segmented-control segmented-control-muted" role="group" aria-label="Graph 显示模式"><button class="btn btn-sm" data-compute-mode="structure">Structure</button><button class="btn btn-sm" data-compute-mode="weight">Execution Weight</button></div><label>Task <select data-compute-task aria-label="Graph Task">${model.tasks.map(t=>`<option value="${t.id}">Task ${t.id}</option>`).join('')}</select></label><span data-compute-context></span><div data-compute-stacks role="group" aria-label="Stack 焦点"></div>`;
 let task=model.tasks[85]||model.tasks[0],focus,view={},active=false,hover=null;
 const canvas=host.querySelector('.viz2-compute-canvas');
 const stackIndex=()=>Number.isInteger(focus?.stackIndex)?focus.stackIndex:['stack','stage','event'].includes(focus?.kind)?focus.index:focus?.kind==='page'?task.pages[focus.index]?.stack:null;
 const currentNode=()=>focus?.node?semanticId(focus.node):focus?.kind==='stage'?semanticId(focus.stage):focus?.kind==='event'?({QK_READY_EVENT:'qk',SOFTMAX_READY_EVENT:'softmax',PV_READY_EVENT:'pv'}[focus.event]||null):focus?.kind==='output'?'output':null;
 const edgeFor=(id,graph)=>graph.edges.find(e=>e.id===id||e.semantic?.id===id);
 const syncFor=id=>task.graph.synchronizations.find(s=>s.id===id||s.name+'_EVENT'===id);
 const relatedSyncs=id=>task.graph.synchronizations.filter(s=>[s.producer.node,s.consumer.node,s.consumer.boundary].includes(id?.split('/')[0]));
 const syncHtml=sync=>{
  const detail=global.PTO_COMPUTATION_PRESENTATION.synchronization(sync,task.graph);
  const index=stackIndex()??0,dependency=global.PtoPagedAttentionStackDependencyModel.build(global.PtoPagedAttentionIntro.model.pipelineFor(task.seqLen),task.graph.synchronizations).edges.find(e=>e.id===`sync:s${index}:${sync.id}`);
  const waitRanges=dependency?.consumerSource?.ranges||sync.consumer.source.ranges;
  const sources=[...sync.producer.source.ranges.map((r,i)=>({end:'producer',range:i,label:`SET · :${r.start}–${r.end}`})),...waitRanges.map((r,i)=>({end:'consumer',range:i,label:`WAIT · :${r.start}–${r.end}`}))];
  return `<div class="viz2-ready-source-links">${sources.map(r=>`<button class="btn btn-ghost btn-sm" data-compute-sync-source="${sync.id}" data-compute-sync-end="${r.end}" data-compute-sync-range="${r.range}">${esc(r.label)}</button>`).join('')}</div><button class="btn btn-sm" data-compute-pipeline-ready>在 Pipeline 中查看</button><pre data-compute-code hidden></pre><details class="viz2-inspector-details"><summary>同步证据</summary><dl>${detail.rows.map(r=>`<dt>${esc(r.label)}</dt><dd>${esc(r.value)}</dd>`).join('')}</dl></details>`;
 };
 function executionData(graph){
  const data=global.PtoPagedAttentionExecution.derive(task.graph,global.PtoPagedAttentionIntro.model,{task:task.id,seqLen:task.seqLen});
  if(!data)return null;
  const entry=(count,unit,formula,source)=>({count,unit,label:`×${count} · ${unit}`,formula,source});
  task.nodes.forEach(n=>{data.nodes[semanticId(n.id)]=entry(n.count,n.unit,n.formula,{file:'paged_attention_pypto.py',ranges:[[n.line,n.line]]});});
  data.nodes['qk/q-tile']=entry(1,'AIC load / Task','once before tick loop');
  data.nodes['qk/matmul']=entry(task.pageCount,'AIC / Task','page_count');
  data.nodes['pv/matmul']=entry(task.pageCount,'Page 贡献 / AIC / Task',`matmul = ${task.stackCount}; matmul_acc = ${task.pageCount-task.stackCount}`);
  data.nodes['scale-mask']=entry(task.stackCount,'每 AIV Lane / Task','stack_count');
  // Descendant counts follow source loop granularity. PV Matmul is a semantic
  // page contribution: distinguish first-page matmul from matmul_acc in Inspector.
  graph.nodes.forEach(n=>{
   if(data.nodes[n.id]||!n.parentId)return;
   const root=n.id.split('/')[0];
   const count=['qk','pv'].includes(root)?(n.semantic?.data?.granularity==='stack'?task.stackCount:task.pageCount):['output-norm'].includes(root)?1:task.stackCount;
   data.nodes[n.id]=entry(count,['qk','pv'].includes(root)?'AIC / Task':'每 AIV Lane / Task','source loop granularity',n.source);
  });
  graph.edges.forEach(e=>{const root=(e.semanticSource||e.s).split('/')[0],producer=data.nodes[e.semanticSource||e.s]||data.nodes[root];if(!producer)return;const count=producer.count,unit=producer.unit.replace(/^(AIC|每 AIV Lane)( stores| store| load)? \/ Task$/,'$1 / Task');data.edges[e.id]={...entry(count,`逻辑传输 · ${unit}`,`生产者 ${e.semanticSource||e.s}：${producer.formula}`,producer.source),width:global.PtoPagedAttentionExecution.width(count)};});
  return data;
 }
 function resolveFocus(graph,selection,v){
  const f=hover||focus,id=f?.node?semanticId(f.node):hover?null:currentNode();
  const edge=f?.edge?edgeFor(f.edge,graph):null;
  const sync=task.graph.synchronizations.find(s=>s.id===f?.syncId||s.name===f?.event?.replace('_EVENT',''));
  const related=new Set();
  if(id){related.add(id);graph.edges.filter(e=>[e.s,e.t,e.semanticSource,e.semanticTarget].includes(id)).forEach(e=>{related.add(e.s);related.add(e.t);});}
  if(edge){related.add(edge.s);related.add(edge.t);}
  if(sync){related.add(sync.producer.boundary||sync.producer.node);related.add(sync.consumer.boundary||sync.consumer.node);}
  const s=stackIndex(),stateFocus=s!=null&&!hover;
  const nodeRoles={},edgeRoles={};
  if(stateFocus){
   for(const root of ['m','l','o']){nodeRoles[root]='previous';related.add(root);}
   for(const root of ['scores','softmax','probability','pv','partial','context-acc']){nodeRoles[root]='operation';related.add(root);}
   nodeRoles.rescale='next';related.add('rescale');
   graph.edges.forEach(e=>{if(['feedback','state'].includes(e.type)&&['m','l','rescale','o','softmax','context-acc'].includes(e.s)&&['m','l','rescale','o','softmax','context-acc'].includes(e.t)){edgeRoles[e.id]=e.type==='feedback'?'previous':'next';related.add(e.s);related.add(e.t);}});
  }
  if(f?.kind==='page'){['k-cache','qk','scores','probability','v-cache','pv','partial'].forEach(root=>related.add(root));}
  return {nodeId:graph.nodes.some(n=>n.id===id)?id:graph.nodes.find(n=>n.id===id?.split('/')[0])?.id,related:[...related],edgeId:edge?.id,nodeRoles,edgeRoles,kind:'task'};
 }
 function description(node,sync,graph,edgeId){
  const e=edgeFor(edgeId||focus?.edge,graph),s=stackIndex(),stack=task.stacks[s],page=focus?.kind==='page'?task.pages[focus.index]:null,step=stackEvolution(task,s);
  const stateRows=()=>[['状态路径',step?`${step.previous} → S${s}${step.next?` → ${step.next}`:' → Output Normalize'}`:'S(n−1) → S(n)'],['Previous m / l_sum / o',step?`${step.states.m.previous}；${step.states.l.previous}；${step.states.o.previous}`:'上一 Stack 的 m / l_sum / o'],['Softmax · m',step?`tick ${step.produceTick} · ${step.states.m.updated} = max(${step.states.m.previous}, max(Scores))`:'m_new = max(m_old, max(Scores))'],['Softmax · l_sum',step?`tick ${step.produceTick} · ${step.states.l.updated} = exp(m_old−m_new) × l_old + sum(P)`:'l_new = exp(m_old−m_new) × l_old + sum(P)'],['Context Accumulate · o',step?`tick ${step.consumeTick} · ${step.states.o.updated} = delayed_rescale × ${step.states.o.previous} + PV Partial`:'o_new = delayed_rescale × o_old + PV Partial'],['FIFO 延迟',step?step.fifo.versions.join(' → '):'produce → pending1 → pending0 → consume'],['FIFO 原因','Softmax 更新 m 后，旧 o 必须按相同尺度重标定；软件流水线使该 rescale 延迟到 Context 消费。'],['初始状态','m = −3e38；l_sum = 0；o = 0；pending0 = pending1 = 1'],['源码','Softmax :789–823；FIFO :848–854；Context :827–846；均为静态依赖，无运行时数值']];
  let title='Task '+task.id,rows=[],line=759;
  if(sync){title=sync.name;rows=[['Task / Stack',`${task.id} / S${s??0}`],['Buffer',`${sync.buffer} · ${sync.dtype} · producer 写 / consumer 读`],['生产者',`${sync.producer.node} · ${sync.producer.coreType} SET · ${sync.producer.pipe}`],['消费者',`${sync.consumer.boundary||sync.consumer.node} · ${sync.consumer.coreType} WAIT · ${sync.consumer.pipe}`],['原因',sync.explanation]];line=sync.producer.source?.ranges?.[0]?.start||759;}
  else if(node){
   const root=node.id.split('/')[0],n=task.nodes.find(n=>semanticId(n.id)===root),data=executionData(graph),weight=data?.nodes[node.id];
   title=node.title;line=node.sourceLine||n?.line||759;
   rows=[['类型',node.type.toUpperCase()],['静态执行次数',weight?.label||'未推导'],['公式 / 循环边界',weight?.formula||n?.formula||'源码分支'],['语义',node.detail],...(n?[['Task 形状 / 内存',n.detail]]:[]),['生产者',graph.edges.filter(e=>e.t===node.id).map(e=>e.s).join(', ')||'Task input'],['消费者',graph.edges.filter(e=>e.s===node.id).map(e=>e.t).join(', ')||'Task output']];
   if(root==='pv')rows.push(['PV 统计口径',`每 Task / AIC：首 Page matmul ×${task.stackCount}；matmul_acc ×${task.pageCount-task.stackCount}；合计 ${task.pageCount} 次 Page 贡献`]);
   if(['m','l','rescale','o','softmax','context-acc'].includes(root))rows.push(...stateRows());
   if(root==='rescale')rows.push(['有效分支',`rescale_after 产生 ×${task.stackCount}；FIFO 更新 ×${task.tickCount}；Context 消费 ×${task.stackCount}，均为每 AIV Lane / Task`]);
  }else if(e){title=e.label||`${e.s} → ${e.t}`;line=e.s==='rescale'?848:e.s==='o'||e.t==='o'?842:e.s==='m'||e.t==='m'?790:e.s==='l'||e.t==='l'?798:graph.nodes.find(n=>n.id===e.t)?.sourceLine||759;rows=[['依赖类型',e.type],['生产者 → 消费者',`${e.s} → ${e.t}`],['状态关系',e.type==='feedback'?`${s===0?'初始化 → S0':s!=null?`S${s-1} → S${s}`:'S(s−1) → S(s)'}；m/l/o 初始 −3e38 / 0 / 0；rescale FIFO 初始 (1,1)`:'按源码数据流与端口映射'],['次数口径','逻辑传输次数；不推断 Bytes 或实测耗时']];}
  else {title=page?`Page ${page.index}`:stack?`Stack ${s}`:title;rows=[['Task',`Block ${task.block} / Batch ${task.batch} / KV Head ${task.kvHead}`],['有效 Pages',stack?stack.pages.map(p=>`P${p} (${task.pages[p].tokens}/${task.pageTokens})`).join(', '):task.pageCount],['有效 Tokens',page?`${page.tokens}/${task.pageTokens}`:stack?`${stack.tokens}/${task.stackTokens}`:task.seqLen],['计算实例',stack?`QK/PV ${stack.pages.length} Pages；Softmax/Accumulate 各 Lane 1 次`:`${task.stackCount} Stacks / ${task.tickCount} logical ticks`],['物理 Page','未提供实际 block_table，未知']];}
  if(e&&['feedback','state'].includes(e.type))rows.push(...stateRows());
  if(!node&&!e&&!sync&&stack)rows.push(...stateRows());
  if(e){const weight=executionData(graph)?.edges[e.id];if(weight)rows.push(['逻辑传输',`${weight.label}；${weight.formula}`]);}
  return {title,line,rows};
 }
 function instancesMarkup(node){
  if(!focus?.instances||!node)return '';
  const groups=pageInstances(task,node.id),root=node.id.split('/')[0];
  if(!groups.length)return `<section><h3>静态逻辑实例</h3><p>统计口径：${esc(task.nodes.find(item=>semanticId(item.id)===root)?.unit||'源码循环粒度')}。各 AIV Lane 独立执行；不同单位不表示统一硬件成本。</p>${task.stacks.map(stack=>`<button class="btn btn-ghost btn-sm" data-compute-instance-stack="${stack.index}">S${stack.index}</button>`).join('')}</section>`;
  const selected=groups.flatMap(group=>group.pages).find(page=>page.index===focus.instancePage);
  const needle=selected?`${selected.sourceSymbol} = ${selected.api}`:root==='qk'?'qk_score_page = pl.matmul':'pv0 = pl.matmul';
  const line=model.source.split('\n').findIndex(value=>value.includes(needle))+1;
  return `<section class="viz2-page-instances"><h3>Page Instances · ${task.pageCount}</h3><p>Per Task / AIC · 每 Page 一次逻辑贡献；源码静态推导，物理 Cache Page ID 未知。</p>${groups.map(group=>`<div class="viz2-page-instance-group"><b>S${group.index}</b><div>${group.pages.map(page=>`<button class="btn btn-ghost btn-sm ${selected?.index===page.index?'is-selected':''}" data-compute-instance-page="${page.index}" aria-pressed="${selected?.index===page.index}" title="${esc(page.api)}">P${page.index}${root==='pv'?` · ${page.api.replace('pl.','')}`:''}</button>`).join('')}</div></div>`).join('')}${selected?`<section><h3>P${selected.index} · S${selected.stack}</h3><dl>${[['Page Index',selected.index],['有效 Token',`${selected.tokens} / ${task.pageTokens}`],['输入',selected.input],['计算',selected.api],['输出',selected.output]].map(([key,value])=>`<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join('')}</dl><button class="btn btn-ghost btn-sm" data-compute-source="${line}">PyPTO 源码 :${line}</button><pre data-compute-code hidden></pre></section>`:''}</section>`;
 }
 const shared=global.PtoVisualization2Graph.mount(canvas,{onView:patch=>{if(patch.overlays){const {execution,...overlays}=patch.overlays;onView({intraOverlays:overlays,intraMode:execution?'weight':'structure'});return;}const mapped={};for(const [k,v] of Object.entries(patch))mapped[{camera:'intraCamera',expanded:'intraExpanded',overlays:'intraOverlays',inspectorOpen:'intraInspectorOpen',relationMode:'intraRelationMode'}[k]||k]=v;onView(mapped);},adapter:{
  getModel:()=>task.graph,resolveFocus,executionData,showExecutionCount:node=>node.type==='operation'||node.type==='composite',
  stageLabel:g=>`<b>${esc(g.label)}</b>`,
  context:()=>view.intraMode!=='structure'?'Static Execution Weight · AIC / 每 AIV Lane 分别统计 · 非耗时':'Data / State / Feedback / READY · 源码静态计算关系',
  hasInspector:()=>!!focus,
  inspector(node,sync,graph){
   const d=description(node,sync,graph),semantic=node&&!sync?global.PTO_COMPUTATION_PRESENTATION.inspector(node,{model:task.graph,qualifiedId:node.id}):[],related=node&&!sync?relatedSyncs(node.id):[];
   const s=stackIndex(),step=stackEvolution(task,s),root=node?.id.split('/')[0],fact=task.nodes.find(item=>semanticId(item.id)===root);
   const incoming=node?graph.edges.filter(edge=>edge.t===node.id&&edge.type==='data').map(edge=>graph.nodes.find(item=>item.id===edge.s)?.title||edge.s):[];
   const outgoing=node?graph.edges.filter(edge=>edge.s===node.id&&edge.type==='data').map(edge=>graph.nodes.find(item=>item.id===edge.t)?.title||edge.t):[];
   const names=items=>[...new Set(items)].join('、')||'当前 Task 输入';
   let what=node?`${node.type==='composite'?'复合计算':node.type==='operation'?'计算操作':node.type==='state'?'循环状态':'数据'} · ${fact?.unit||node.semantic?.data?.granularity||'按语义节点'}`:`Task ${task.id} · Block ${task.block} · Batch ${task.batch} · KV ${task.kvHead}`;
   let input=node?names(incoming):'当前 Task 的 Q、K/V Cache 与上一 Stack 状态';
   let compute=node?(node.detail||fact?.detail||fact?.formula||'按源码定义的数据依赖计算'):`${task.stackCount} 个 Stack 的静态计算依赖`;
   let output=node?names(outgoing):'更新状态传入下一 Stack；最终输出 BF16 Context';
   if(root==='softmax'){input=`Scores + ${step?`${step.states.m.previous}、${step.states.l.previous}`:'m_old、l_old'}`;compute='m_new = max(m_old, row_max(Scores))；l_new = exp(m_old−m_new) × l_old + sum(exp(Scores−m_new))';output=`${step?`${step.states.m.updated}、${step.states.l.updated}`:'m_new、l_new'} → 下一 Stack；Probability → PV Matmul`;}
   if(root==='context-acc'){input=`${step?step.states.o.previous:'o_old'} + PV Partial + Delayed Rescale（pending0）`;compute='o_new = delayed_rescale × o_old + PV Partial；Context 在延迟的逻辑 Tick 消费 Softmax 产生的缩放';output=`${step?step.states.o.updated:'o_new'} → 下一 Stack / Output Normalize`;}
   if(root==='rescale'){input='Online Softmax 产生 rescale_after；FIFO 初始 pending0 = pending1 = 1';compute=`produce → pending1 → pending0 → consume${step?`；生产 Tick ${step.produceTick}，消费 Tick ${step.consumeTick}`:''}`;output='Delayed Rescale → Context Accumulate，重标定历史 o';}
   if(sync){what=`${sync.name} · READY 同步 · Task ${task.id} / S${s??0}`;input=`${sync.producer.node} · ${sync.producer.coreType} SET；保护 ${sync.buffer} 的写入`;compute=sync.explanation;output=`${sync.consumer.boundary||sync.consumer.node} · ${sync.consumer.coreType} WAIT 后读取 ${sync.buffer}`;}
   const core=[['是什么',what],['输入什么',input],['计算什么',compute],['输出什么',output]].map(([label,value])=>`<section class="viz2-inspector-core"><h3>${label}</h3><p>${esc(value)}</p></section>`).join('');
   const evolution=['softmax','context-acc','rescale'].includes(root)||!node&&!sync&&s!=null?`<details class="viz2-inspector-details" data-compute-state-explore><summary>查看跨 Stack 演化</summary>${step?evolutionMarkup(step):'<p>选择具体 Stack 后查看静态状态版本与逻辑 Tick；当前为符号化依赖。</p>'}</details>`:'';
   const ready=related.length?`<section class="viz2-inspector-related"><h3>关联 READY</h3>${related.map(item=>`<button class="btn btn-ghost btn-sm" data-compute-ready="${item.id}">${esc(item.name)}</button>`).join('')}</section>`:'';
   const detailRows=d.rows.map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
   const detailSections=semantic.map(section=>`<section><h3>${esc(section.title)}</h3><dl>${section.rows.map(r=>`<dt>${esc(r.label)}</dt><dd>${esc(r.value)}</dd>`).join('')}</dl></section>`).join('');
   return {title:d.title,html:`${instancesMarkup(node)}${core}${sync?syncHtml(sync):''}${ready}${evolution}<details class="viz2-inspector-details"><summary>详细信息</summary><dl>${detailRows}</dl>${detailSections}<section><button class="btn btn-ghost btn-sm" data-compute-source="${d.line}">PyPTO 源码 :${d.line}</button><pre data-compute-code hidden></pre></section></details>`};
  },
  onInstances(id){onFocus({kind:'graph',node:id,instances:true,index:stackIndex()??0,stackIndex:stackIndex()});},
  onNode(id){const root=localId(id.split('/')[0]),stage=['qk','softmax','pv','accumulate'].includes(root)?root:null;onFocus({kind:stage?'stage':root==='output'?'output':'graph',node:id,index:stackIndex()??0,stackIndex:stackIndex(),...(stage?{stage}:{})});},
  onEdge(id,graph){onFocus({kind:'graph-edge',edge:id,index:stackIndex()??0,stackIndex:stackIndex()});},
  onSync(id){const sync=syncFor(id),index=stackIndex()??0;onFocus({kind:'event',syncId:sync.id,event:sync.name+'_EVENT',index,stackIndex:index});},
  onHover(node,edge,syncId){hover=node?{node}:edge?{edge}:syncId?{syncId}:null;},
  tooltip(id,edge,syncId,graph){const d=description(graph.nodes.find(n=>n.id===id),task.graph.synchronizations.find(s=>s.id===syncId),graph,edge),state=['m','l','rescale','o'].includes(id?.split('/')[0])||['feedback','state'].includes(edgeFor(edge,graph)?.type),rows=state?d.rows.filter(r=>['状态路径','Softmax · m','Softmax · rescale_after','Softmax · l_sum','FIFO 延迟','Context Accumulate · o'].includes(r[0])):d.rows.slice(0,3);return `${d.title} · ${rows.map(r=>r.join(': ')).join(' · ')} · source:${d.line}`;},
  decorate(scene,nodes,graph,data){scene.dataset.executionWeight=String(!!data);scene.querySelectorAll('[data-stack-container-layer]').forEach(el=>el.remove());const container=scene.querySelector('[data-pypa-region="task-compute"]');if(data&&container){for(let i=task.stackCount-1;i>0;i--){const layer=document.createElement('div');layer.dataset.stackContainerLayer=String(i);layer.className='kf-pypa-graph__group execution-container-layer';layer.style.cssText=container.style.cssText;layer.style.transform=`translate(${i*10}px,${i*8}px)`;layer.setAttribute('aria-hidden','true');scene.insertBefore(layer,container);}}nodes.forEach(n=>{const el=scene.querySelector(`[data-pypa-node="${n.id}"]`),entry=data?.nodes[n.id];if(el)el.style.setProperty('--execution-weight',entry&&n.type==='operation'?String(Math.min(1,entry.count/Math.max(task.pageCount,task.tickCount))):'0');});}
 }});
 host.addEventListener('change',e=>{if(e.target.matches('[data-compute-task]'))onTask(Number(e.target.value));});
 canvas.addEventListener('click',e=>{const button=e.target.closest('[data-compute-instance-page],[data-compute-instance-stack]');if(!button)return;e.stopPropagation();const page=button.dataset.computeInstancePage!==undefined?Number(button.dataset.computeInstancePage):undefined,index=page!==undefined?task.pages[page].stack:Number(button.dataset.computeInstanceStack);onFocus({...focus,instancePage:page,index,stackIndex:index});},true);
 toolbar.addEventListener('click',e=>{const b=e.target.closest('[data-compute-mode],[data-compute-stack],[data-compute-page]');if(!b)return;if(b.dataset.computeMode)onView({intraMode:b.dataset.computeMode});else if(b.dataset.computeStack!==undefined){const index=Number(b.dataset.computeStack);onFocus(focus?.node||focus?.edge?{...focus,index,stackIndex:index,instancePage:undefined}:{kind:'stack',index,stackIndex:index});}else onFocus({kind:'page',index:Number(b.dataset.computePage)});});
 // Shared controller stops legacy document-level graph handlers; source actions
 // are handled in capture so their Inspector remains local to this Task.
 canvas.addEventListener('click',e=>{const ready=e.target.closest('[data-compute-ready]');if(ready){e.stopPropagation();const sync=syncFor(ready.dataset.computeReady),index=stackIndex()??0;onFocus({kind:'event',syncId:sync.id,event:sync.name+'_EVENT',index,stackIndex:index});return;}if(e.target.closest('[data-compute-pipeline-ready]')){e.stopPropagation();onNavigate?.();return;}const b=e.target.closest('[data-compute-source],[data-compute-sync-source]');if(!b)return;e.stopPropagation();let line=Number(b.dataset.computeSource);if(b.dataset.computeSyncSource){const sync=syncFor(b.dataset.computeSyncSource),end=b.dataset.computeSyncEnd;const edge=global.PtoPagedAttentionStackDependencyModel.build(global.PtoPagedAttentionIntro.model.pipelineFor(task.seqLen),task.graph.synchronizations).edges.find(item=>item.id===`sync:s${stackIndex()??0}:${sync.id}`);const ranges=end==='consumer'?(edge?.consumerSource?.ranges||sync.consumer.source.ranges):sync.producer.source.ranges;line=ranges[Number(b.dataset.computeSyncRange)]?.start;}const code=b.closest('section')?.querySelector('[data-compute-code]')||canvas.querySelector('[data-compute-code]');if(!code||!line)return;code.hidden=false;code.textContent=model.source.split('\n').slice(Math.max(0,line-3),line+12).map((v,i)=>`${Math.max(1,line-2)+i}  ${v}`).join('\n');},true);
 return {update(id,f,v,{active:next}={}){
  if(model.tasks[id])task=model.tasks[id];focus=f;view=v;active=!!next;hover=null;
  if(active){toolbar.querySelector('[data-compute-task]').value=task.id;toolbar.querySelector('[data-compute-context]').textContent=`Batch ${task.batch} · KV ${task.kvHead} · ${task.seqLen} tokens · ${task.pageCount} Pages / ${task.stackCount} Stacks / ${task.tickCount} logical ticks`;
   const s=stackIndex();toolbar.querySelector('[data-compute-stacks]').innerHTML=task.stacks.map(stack=>`${stack.index?'<span aria-hidden="true">→</span>':''}<button class="btn btn-ghost btn-sm ${s===stack.index?'is-selected':''}" data-compute-stack="${stack.index}" aria-pressed="${s===stack.index}">S${stack.index}</button>`).join('');
   toolbar.querySelectorAll('[data-compute-mode]').forEach(b=>{const selected=b.dataset.computeMode===(v.intraMode||'weight');b.classList.toggle('is-selected',selected);b.setAttribute('aria-pressed',String(selected));});
  }
  const sync=task.graph.synchronizations.find(s=>s.id===f?.syncId||s.name===f?.event?.replace('_EVENT',''));
  shared.update({kind:'task',task:task.id,block:task.block,batch:task.batch,kvHead:task.kvHead,subobject:f},{camera:v.intraCamera,nodeId:currentNode(),syncId:sync?.id,expanded:v.intraExpanded||[],overlays:{control:true,...v.intraOverlays,sync:!!sync||!!v.intraOverlays?.sync,execution:v.intraMode!=='structure'},relationMode:v.intraRelationMode,inspectorOpen:v.intraInspectorOpen},{active});
 },hideTooltip:shared.hideTooltip,getTransitionRect:shared.getTransitionRect};
}
global.PtoVisualization2GraphIntra=Object.freeze({createModel,stackEvolution,pageInstances,mount});
})(window);
