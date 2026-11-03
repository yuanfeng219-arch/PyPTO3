/* CSA scene adapter. The shared graph pattern owns node rendering, selection and camera input. */
(function (global) {
  'use strict';
  const model=global.PtoCsaModel, objects=model.objects;
  const qkPvId='sparse/attention/qk-pv';
  // 两个视图分别保存展示阶段；Shared Focus 只记录明确选中的计算对象。
  const state={open:false,tab:'assignment',focus:null,rank:0,assignmentStage:qkPvId,pipelineStage:qkPvId,
    views:{assignment:{scope:'overall',topic:'gather',camera:null},pipeline:{scope:'overall',topic:'gather',camera:null},graph:{scope:'overall',camera:null,expanded:[],overlays:{data:true,control:true},nodeId:null,edgeId:null,inspectorOpen:false,projectedFocusKey:null}}};
  const tabs=['assignment','pipeline','graph'];
  let root, controller, semanticController, qkController, qkPipelineController, trigger, previousInert=[], graph;
  const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const titles={assignment:'Rank / Data Assignment',pipeline:'Distributed Execution',graph:'CSA Graph'};
  const alias={rows0:'full',rows1:'full',group0:'publish',group1:'publish',mainstate:'main',innerstate:'idxcomp',gather:'push',local:'pre',full:'gather',original:'write',compressed:'main',indexcache:'idxcomp',window:'publish',group:'publish',push:'gather',payload:'gather',readback:'gather',rope:'ready'};
  // Explicit adapters between the older Rank diagrams and semantic graph IDs.
  const objectToGraph={pre:'input/hc-pre',gather:'input/allgather',q:'input/q',kv:'input/kv',main:'cache/main',idxcomp:'cache/idxcomp',write:'cache/write',indexer:'cache/indexer',attention:'sparse/attention',publish:'redistribute/publish',a2a:'redistribute/a2a',oproj:'output/oproj',rs:'output/oproj/rs',post:'output/post',local:'input/local',full:'input/full',original:'cache/original',compressed:'cache/compressed',indexcache:'cache/indexcache',ready:'cache/ready'};
  const graphToObject=Object.fromEntries(Object.entries(objectToGraph).map(([object,id])=>[id,object]));
  const semantic=global.PTO_CSA_SEMANTIC_GRAPH;
  const semanticNodes=new Map();
  function indexSemantic(nodes,parentId=null){for(const n of nodes){const id=parentId?parentId+'/'+n.id:n.id;semanticNodes.set(id,{...n,parentId});if(n.composite?.subgraph)indexSemantic(n.composite.subgraph.nodes,id);}}
  indexSemantic(semantic.nodes);
  function nearestObject(id){while(id){if(graphToObject[id])return graphToObject[id];id=id.slice(0,id.lastIndexOf('/'));}return null;}
  function isQkFocus(focus){return !!focus&&(['request','query','work-item','lane'].includes(focus.kind)||focus.semanticId===qkPvId);}
  function setFocus(focus){state.focus=focus;state.rank=focus?.rank??state.rank;if(focus){const stage=stageForAssignment(focus);state.assignmentStage=stage;state.pipelineStage=stage;}state.views.graph.nodeId=null;state.views.graph.edgeId=null;state.views.graph.inspectorOpen=false;if(qkController)qkController.setSelection(qkSelection());}
  function qkSelection(){const f=state.focus;return f?.kind==='request'?{type:'request',requestIndex:f.requestIndex}:f?.kind==='query'?{type:'query',tokenIndex:f.tokenIndex,requestIndex:f.requestIndex}:f?.kind==='work-item'?{type:'item',id:f.workItemId}:f?.kind==='lane'?{type:'block',block:f.blockId}:null;}
  // Pipeline 工具栏的 Rank 下拉：只改 Assignment 的 Scenario 状态（qkController.setRank），
  // 由它的 onContext 回写 Shared Focus 的 rank，再按新 Context / Scenario 重渲染当前 Tab。
  function qkSetRank(rank){
    if(rank==null)return;
    if(qkController)qkController.setRank(rank);
    else{state.rank=rank;setFocus({kind:'module',rank,scenarioId:null,semanticId:qkPvId,moduleId:qkPvId,objectId:'attention'});}
    render();
  }
  function graphIdForFocus(){const f=state.focus;if(!f||f.kind==='rank')return null;if(['request','query','work-item','lane'].includes(f.kind))return qkPvId;if(f.semanticId)return f.semanticId;return objectToGraph[f.objectId]||null;}
  function focusKey(){const f=state.focus;return f?JSON.stringify([f.kind,f.semanticId,f.objectId,f.requestIndex,f.tokenIndex,f.workItemId,f.blockId,f.rank,f.scenarioId]):null;}
  function objectForFocus(){const f=state.focus;if(!f)return null;if(f.kind==='rank')return 'rank';if(['request','query','work-item','lane'].includes(f.kind))return 'attention';return f.objectId||nearestObject(f.semanticId)||nearestObject(f.moduleId);}
  function graphFocus(id){const n=semanticNodes.get(id);if(!n)return null;const parent=n.parentId&&semanticNodes.get(n.parentId);const moduleId=n.composite?.category==='module'?id:parent?.composite?.category==='module'?n.parentId:null;return {kind:n.composite?.category==='stage'?'stage':n.type==='composite'?'module':n.type==='operation'?'operation':n.type==='data'?'data':n.type==='state'?'state':'module',rank:null,semanticId:id,moduleId,objectId:nearestObject(id)};}
  function legacyFocus(object,rank){return object==='rank'?{kind:'rank',rank}:{kind:['tensor','state'].includes(objects[object]?.kind)?'data':'module',rank,objectId:object,moduleId:objectToGraph[object]||null};}
  function focusCrumbs(){const f=state.focus,items=[{id:'',label:'Decode CSA'}];
    // 非 qk_pv 阶段只展示阶段名，不把其他 Tab 的 Work Item 当作当前页面标题。
    const visibleStage=state.tab==='assignment'?state.assignmentStage:state.tab==='pipeline'?state.pipelineStage:null;
    if(visibleStage&&visibleStage!==qkPvId){const s=assignmentStages.find(x=>x.id===visibleStage);return [...items,{label:s?s.full:visibleStage}];}
    if(visibleStage===qkPvId&&!isQkFocus(f))return [...items,{label:assignmentStages.find(x=>x.id===qkPvId).full}];
    if(!f)return items;if(f.kind==='rank')return [...items,{label:'Rank '+f.rank}];if(f.rank!=null)items.push({label:'Rank '+f.rank});const id=graphIdForFocus(),path=id?.split('/')||[];path.forEach((_,index)=>{const key=path.slice(0,index+1).join('/'),node=semanticNodes.get(key);if(node&&!(key==='sparse/attention'))items.push({id:key,label:node.name});});if(['request','query','work-item'].includes(f.kind)&&f.requestIndex!=null)items.push({label:'Request '+f.requestIndex});if(['query','work-item'].includes(f.kind)&&f.tokenIndex!=null)items.push({label:'Q'+f.tokenIndex});if(f.kind==='work-item')items.push({label:'Work Item '+f.workItemId});if(f.kind==='lane')items.push({label:'SPMD Lane '+f.blockId});if(path.length===0&&f.objectId)items.push({label:objects[f.objectId]?.label||f.objectId});return items;}
  function node(g,object,rank,x,y,label){const o=objects[object],id='r'+rank+':'+object;
    g.nodes.push({id,object,rank,x,y,width:242,height:62,label:label||o.label,kind:['tensor','state'].includes(o.kind)?'tensor':'module',typeLabel:model.kindLabels[o.kind],colorKey:o.sem,parent:'rank-'+rank});
    return id;
  }
  function edge(g,a,b,type='data'){g.edges.push({source:a,target:b,type:type==='sync'?'control':type==='comm'?'communication':'activation',dashed:type==='sync',color:type==='comm'?'var(--primary)':undefined});}
  function link(g,r,a,b,type){edge(g,'r'+r+':'+a,'r'+r+':'+b,type);}
  function rank(g,r,x,y,w,h,label){g.clusters.push({id:'rank-'+r,label:label||'Rank '+r+' · device '+r,x,y,width:w,height:h,colorKey:'module:decoder',collapsible:false});}
  function blank(w,h){return {width:w,height:h,nodes:[],edges:[],clusters:[]};}
  function assignment(topic){
    const g=blank(1500,topic==='cache'?850:700);
    for(let r=0;r<2;r++){
      const x=30+r*750;rank(g,r,x,25,710,g.height-50,'Rank '+r+' · '+(topic==='output'?'O Groups '+(r?'4–7':'0–3'):'本地 token '+(r?'R1':'R0')));
      if(topic==='gather'){
        node(g,'local',r,x+355,100,'x_normed_t · [T,4096]');
        node(g,'gather',r,x+355,260);
        node(g,'full',r,x+355,410,'x_normed_full · [2T,4096]');
        node(g,'q',r,x+145,570,'Q · 本地 T 行');
        node(g,'kv',r,x+545,570,'KV · 整组 2T 行');
        link(g,r,'local','gather');link(g,r,'gather','full');link(g,r,'local','q');link(g,r,'full','kv');
      } else if(topic==='cache'){
        node(g,'full',r,x+355,105);
        [['kv',145],['main',355],['idxcomp',565]].forEach(([o,c])=>{node(g,o,r,x+c,280);link(g,r,'full',o);});
        [['original',145],['compressed',355],['indexcache',565]].forEach(([o,c])=>node(g,o,r,x+c,460));
        link(g,r,'kv','original');link(g,r,'main','compressed');link(g,r,'idxcomp','indexcache');
        node(g,'attention',r,x+245,690);node(g,'indexer',r,x+555,690);
        link(g,r,'original','attention');link(g,r,'compressed','attention');link(g,r,'indexcache','indexer');link(g,r,'indexer','attention');
      } else {
        node(g,'group',r,x+355,105,'本地 T tokens × 全部 8 Groups');
        node(g,'publish',r,x+355,250);
        node(g,'window',r,x+355,420,'4 Groups × 来自 R0 / R1 的行');
        node(g,'a2a',r,x+355,585,'attention_local_groups');
        link(g,r,'group','publish');link(g,r,'publish','window','comm');link(g,r,'window','a2a','sync');
      }
    }
    if(topic==='gather'){edge(g,'r0:local','r1:gather','comm');edge(g,'r1:local','r0:gather','comm');}
    if(topic==='output'){edge(g,'r0:publish','r1:window','comm');edge(g,'r1:publish','r0:window','comm');}
    return g;
  }
  function pipeline(){
    const g=blank(2070,1040);
    for(let r=0;r<2;r++){
      const y=25+r*515;rank(g,r,25,y,2020,490);
      const positions={pre:[155,120],gather:[440,120],q:[740,70],kv:[740,190],main:[740,310],idxcomp:[740,420],write:[1025,190],indexer:[1025,420],ready:[1290,290],attention:[1510,100],publish:[1800,100],a2a:[1800,230],oproj:[1800,345],rs:[1510,345],post:[1510,450]};
      Object.entries(positions).forEach(([o,[x,dy]])=>node(g,o,r,x,y+dy));
      model.relations.pipeline.forEach(([a,b,t])=>link(g,r,a,b,t));
    }
    edge(g,'r0:gather','r1:gather','comm');edge(g,'r1:gather','r0:gather','comm');
    edge(g,'r0:publish','r1:a2a','comm');edge(g,'r1:publish','r0:a2a','comm');
    edge(g,'r0:oproj','r1:rs','comm');edge(g,'r1:oproj','r0:rs','comm');
    return g;
  }
  function localGraph(topic){
    if(topic==='gather'){
      const g=blank(1500,800);
      for(let r=0;r<2;r++){const x=30+r*750;rank(g,r,x,25,710,750);
        ['local','push','payload','readback','full'].forEach((o,i)=>node(g,o,r,x+355,110+i*145));
        [['local','push'],['push','payload','sync'],['payload','readback','sync'],['readback','full']].forEach(([a,b,t])=>link(g,r,a,b,t));}
      edge(g,'r0:push','r1:payload','comm');edge(g,'r1:push','r0:payload','comm');return g;
    }
    if(topic==='output'){
      const g=blank(1500,1080);
      for(let r=0;r<2;r++){const x=30+r*750;rank(g,r,x,25,710,1030);
        ['attention','publish','window','a2a','oproj','rs','post'].forEach((o,i)=>node(g,o,r,x+355,105+i*145));
        [['attention','publish','sync'],['publish','window','comm'],['window','a2a','sync'],['a2a','oproj'],['oproj','rs','comm'],['rs','post']].forEach(([a,b,t])=>link(g,r,a,b,t));}
      edge(g,'r0:publish','r1:window','comm');edge(g,'r1:publish','r0:window','comm');edge(g,'r0:oproj','r1:rs','comm');edge(g,'r1:oproj','r0:rs','comm');return g;
    }
    const r=state.rank,g=blank(1450,1060);rank(g,r,25,25,1400,1005,'Rank '+r+' · 三条 Cache 分支与两种就绪路径');
    const pos={full:[730,100],local:[160,100],rope:[1150,100],q:[160,305],kv:[460,305],main:[760,305],idxcomp:[1150,305],write:[460,500],compressed:[760,500],indexcache:[1150,500],original:[420,690],ready:[730,690],indexer:[1150,690],attention:[730,930]};
    Object.entries(pos).forEach(([o,[x,y]])=>node(g,o,r,x,y));
    model.relations.cache.forEach(([a,b,t])=>link(g,r,a,b,t));return g;
  }

  function anatomy(topic){
    const r=state.rank;
    if(topic==='gather'){
      const g=blank(1500,800);
      rank(g,0,25,35,440,310,'Source Rank 0 · 本地行');
      rank(g,1,25,430,440,310,'Source Rank 1 · 本地行');
      g.clusters.push({id:'target-rank',label:'目标 Rank '+r+' · 整组副本',x:850,y:35,width:625,height:705,collapsible:false});
      node(g,'local',0,245,185,'R0 · x_normed_t [T,D]');node(g,'local',1,245,580,'R1 · x_normed_t [T,D]');
      node(g,'gather',r,650,380);
      node(g,'full',r,1165,200,'x_normed_full · [2T,D]');
      node(g,'rows0',r,1030,370);node(g,'rows1',r,1300,370);
      node(g,'kv',r,1165,600,'KV / Compressor · 读取整组行');
      edge(g,'r0:local','r'+r+':gather','comm');edge(g,'r1:local','r'+r+':gather','comm');
      link(g,r,'gather','full');link(g,r,'full','kv');
      // Two row-segment marks are a structural decomposition, not two extra copy Tasks.
      g.nodes.filter(n=>['full','rows0','rows1','kv'].includes(n.object)).forEach(n=>n.parent='target-rank');
      return g;
    }
    if(topic==='output'){
      const g=blank(1500,840);
      g.clusters.push({id:'source-attention',label:'Source Rank '+r+' · 本地 T tokens 的全部 8 个 Groups',x:30,y:25,width:1440,height:295,collapsible:false});
      node(g,'group',r,750,105,'64 Heads → 8 O Groups');
      node(g,'group0',r,390,245);node(g,'group1',r,1110,245);
      link(g,r,'group','group0');link(g,r,'group','group1');
      for(let target=0;target<2;target++){const x=30+target*750;rank(g,target,x,405,690,405,'Destination Rank '+target+' · Groups '+(target?'4–7':'0–3'));
        node(g,'window',target,x+345,505,'Window · 4 Groups × 2T_PAD');
        node(g,'a2a',target,x+345,700,'每 Group · [R0 行 | R1 行]');
        link(g,target,'window','a2a');
        edge(g,'r'+r+':group'+target,'r'+target+':window','comm');
      }
      return g;
    }
    const g=blank(1500,900);rank(g,r,25,25,1450,850,'Rank '+r+' · 三种独立 Cache；逻辑 Shape 示意');
    const pos={kv:[260,130],main:[750,130],idxcomp:[1240,130],original:[260,405],compressed:[750,405],indexcache:[1240,405],attention:[500,720],indexer:[1240,720]};
    Object.entries(pos).forEach(([o,[x,y]])=>node(g,o,r,x,y,
      ({original:'Original · [N_ori,128,1,512]',compressed:'Compressed · [N_cmp,128,1,512]',indexcache:'Indexer · INT8 + FP32 scale'})[o]));
    [['kv','original'],['main','compressed'],['idxcomp','indexcache'],['original','attention'],['compressed','attention'],['indexcache','indexer'],['indexer','attention']].forEach(([a,b])=>link(g,r,a,b));
    return g;
  }

  function mount(){
    root=document.createElement('section');root.id='csaExplorer';root.className='pto-ide-frame__pane viz2 csa-explorer';root.hidden=true;
    root.setAttribute('aria-label','DeepSeek V4 CSA Compute Explorer');
    root.innerHTML='<header class="pto-ide-frame__pane-header viz2-header"><button class="btn btn-ghost btn-sm" data-csa-back>← 返回代码</button><span class="pto-ide-frame__pane-title">Compute Explorer</span><span class="pto-ide-frame__pane-meta">decode_csa.py · TP=2</span></header>'+
    '<div class="pto-ide-frame__tabstrip viz2-tabs"><div class="tab-control" role="tablist" aria-label="CSA 可视化视图">'+tabs.map(id=>'<button class="tab-control-item" role="tab" id="csa-tab-'+id+'" data-csa-tab="'+id+'" aria-controls="csa-panel"><span class="viz2-tab-icon viz2-tab-icon--'+id+'"></span>'+id[0].toUpperCase()+id.slice(1)+'</button>').join('')+'</div></div>'+
    '<div class="csa-toolbar"><nav data-csa-breadcrumb aria-label="当前对象"></nav><button class="btn btn-sm csa-navreturn" data-csa-navreturn hidden></button><button class="btn btn-ghost btn-sm" data-csa-return hidden>← 返回上级</button></div>'+
    '<div class="csa-stagebar" hidden>'+stageBarHtml()+'</div>'+
    '<div class="csa-focusbar" aria-label="关键路径焦点"></div>'+
    '<div class="csa-workspace" id="csa-panel" role="tabpanel"><div class="csa-canvas-column"><div class="csa-caption"></div><div class="csa-stage pto-model-graphviz-pattern-page" role="group" aria-label="CSA 数据与任务关系，拖拽平移，按住 Command 或 Ctrl 滚轮缩放"></div><footer class="csa-footer"><div class="csa-legend"><span>中性矩形 · Tensor / Cache</span><span>彩色模块 · 计算 / 通信 / 同步</span><span>实线 · 数据</span><span class="csa-comm-legend">蓝线 · 跨 Rank</span><span>虚线 · 控制 / 同步</span></div><button class="btn btn-ghost btn-sm" data-csa-fit>全览</button><button class="btn btn-ghost btn-sm" data-csa-read>阅读</button><button class="btn btn-ghost btn-sm" data-csa-zoom="0.8" aria-label="缩小 CSA">−</button><button class="btn btn-ghost btn-sm" data-csa-zoom="1.25" aria-label="放大 CSA">＋</button></footer></div>'+
    '<aside class="pto-ide-frame__pane csa-inspector" aria-label="CSA 对象详情"><header class="pto-ide-frame__pane-header"><span class="pto-ide-frame__pane-title">对象详情</span><button class="btn btn-ghost btn-sm" data-csa-clear>清除选择</button></header><div class="pto-ide-frame__pane-body csa-detail"></div></aside></div>'+
    '<div class="csa-stage-hold" hidden></div><div class="csa-qk-host" hidden></div><div class="csa-qk-pipeline-host" hidden></div>';
    document.getElementById('ideWorkarea').append(root);
    root.addEventListener('click',event=>{
      const b=event.target.closest('button');if(!b)return;
      if(b.hasAttribute('data-csa-back'))return close();
      if(b.hasAttribute('data-csa-navreturn')){returnFromNavigation();return;}
      if(b.dataset.csaStage){selectStage(b.dataset.csaStage);return;}
      if(b.hasAttribute('data-csa-crumb')){const id=b.dataset.csaCrumb;setFocus(id?graphFocus(id):null);const v=state.views.graph;v.expanded=id?id.split('/').slice(0,-1).map((_,i)=>id.split('/').slice(0,i+1).join('/')):[];v.scope=v.expanded.length?'local':'overall';v.camera=null;render();return;}
      if(b.dataset.csaTab){saveCamera();state.tab=b.dataset.csaTab;if(state.tab!=='graph')navigationReturn=null;if(state.tab==='graph'){const v=state.views.graph,key=focusKey();if(key&&key!==v.projectedFocusKey){const path=graphIdForFocus()?.split('/')||[],ancestors=path.slice(0,-1).map((_,i)=>path.slice(0,i+1).join('/'));v.expanded=[...new Set([...v.expanded,...ancestors])];v.scope=v.expanded.length?'local':'overall';v.projectedFocusKey=key;}}render();return;}
      if(b.hasAttribute('data-csa-return')){const v=state.views.graph,focusId=graphIdForFocus()||'';const branch=v.expanded.filter(id=>focusId.startsWith(id)).sort((a,b)=>b.length-a.length);const last=branch[0]||v.expanded.at(-1);if(last){v.expanded=v.expanded.filter(id=>id!==last&&!id.startsWith(last+'/'));v.scope=v.expanded.length?'local':'overall';v.camera=null;}render();return;}
      if(b.dataset.csaFocus){const topic=b.dataset.csaFocus,v=state.views[state.tab];v.topic=topic;v.camera=null;setFocus(legacyFocus({gather:'gather',cache:'ready',output:'publish'}[topic],state.rank));render();return;}
      if(b.hasAttribute('data-csa-clear')){setFocus(null);state.views.graph.nodeId=null;controller?.clearSelection();details();chrome();return;}
      if(b.hasAttribute('data-csa-drill')){const v=state.views[state.tab];v.scope='local';v.topic=selectedTopic(v.topic);v.camera=null;render();return;}
      if(b.hasAttribute('data-csa-fit')){controller?.fit();return;}if(b.hasAttribute('data-csa-read')){controller?.setTransform({zoom:.85,tx:24,ty:24});return;}
      if(b.dataset.csaZoom){const t=controller?.getTransform();if(t)controller.setTransform({...t,zoom:t.zoom*Number(b.dataset.csaZoom)});return;}
      if(b.dataset.csaSourceFile){const [f,a,z]=b.dataset.csaSourceFile.split(':');global.PtoComputeIDE.openSource({file:f,start:Number(a),end:Number(z)});return;}
      if(b.dataset.csaSource){const [a,z]=b.dataset.csaSource.split(':').map(Number);global.PtoComputeIDE.openSource({file:'decode_csa.py',start:a,end:z});}
    });
    root.querySelector('[role="tablist"]').addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const at=tabs.indexOf(state.tab),i=e.key==='Home'?0:e.key==='End'?2:(at+(e.key==='ArrowRight'?1:2))%3;root.querySelector('[data-csa-tab="'+tabs[i]+'"]').click();root.querySelector('[data-csa-tab="'+tabs[i]+'"]').focus();});
  }
  function selectedTopic(fallback){const object=objectForFocus();return object==='rank'?(fallback||'gather'):objects[object]?.topic||(fallback||'gather');}
  // Assignment 计算阶段：顺序体现计算链路，ID 复用 csa-semantic-graph.js 的语义节点。
  // 只有 qk_pv 已完成 Work Assignment Canvas，其余阶段在同一框架下显示建设中占位。
  const assignmentStages=[
    {id:'input/q',label:'Q Proj',full:'Q Projection'},
    {id:'input/kv',label:'KV Proj',full:'KV Projection'},
    {id:'cache/main',label:'Compressor',full:'Main Compressor'},
    {id:'cache/indexer',label:'Indexer',full:'Indexer / Top-K'},
    {id:'sparse/attention/plan',label:'qk_plan',full:'Candidate Planning · qk_plan'},
    {id:'sparse/attention/qk-pv',label:'qk_pv',full:'Sparse Attention · qk_pv',ready:true},
    {id:'redistribute/publish',label:'Merge',full:'Merge / Pack / Publish'},
    {id:'output/oproj',label:'O Proj',full:'Output Projection'}
  ];
  // 落在阶段之间但属于该阶段的节点（Cache、Partial、A2A 等）映射回所属阶段；未命中退回 qk_pv。
  const stageAlias={input:'input/q','input/hc-pre':'input/q','input/local':'input/q',
    'input/full':'input/kv','input/allgather':'input/kv','cache/write':'input/kv','cache/original':'input/kv',
    cache:'cache/main','cache/compressed':'cache/main','cache/ready':'cache/main',
    'cache/idxcomp':'cache/indexer','cache/indexcache':'cache/indexer','cache/topk-positions':'cache/indexer',
    redistribute:'redistribute/publish','redistribute/a2a':'redistribute/publish',
    output:'output/oproj','output/post':'output/oproj'};
  // Focus 决定 Assignment 应该展示哪个阶段；null 或 qk 对象统一落到 qk_pv。
  function stageForAssignment(focus){
    if(!focus||isQkFocus(focus))return qkPvId;
    const ids=[focus.semanticId,focus.moduleId,objectToGraph[focus.objectId]].filter(Boolean);
    for(const raw of ids){
      let id=raw;
      while(id){
        if(stageAlias[id])return stageAlias[id];
        const hit=assignmentStages.find(stage=>id===stage.id||id.startsWith(stage.id+'/'));
        if(hit)return hit.id;
        id=id.includes('/')?id.slice(0,id.lastIndexOf('/')):null;
      }
    }
    return qkPvId;
  }
  function stageBarHtml(){
    return '<div class="segmented-control segmented-control-muted" role="group" aria-label="计算阶段">'+assignmentStages.map(stage=>
      '<button type="button" class="btn btn-sm'+(stage.ready?'':' is-pending')+'" data-csa-stage="'+stage.id+'" aria-pressed="false" title="'+esc(stage.full)+(stage.ready?' · 可查看计算画布':' · 建设中')+'">'+esc(stage.label)+(stage.ready?'':'<span class="csa-stage-todo">建设中</span>')+'</button>').join('')+'</div>';
  }
  // Stage Bar 是各 Tab 的展示选择；它不改写 Shared Focus。
  function selectStage(id){
    const key=state.tab==='assignment'?'assignmentStage':state.tab==='pipeline'?'pipelineStage':null;
    if(!key||id===state[key]||!assignmentStages.some(stage=>stage.id===id))return;
    state[key]=id;
    render();
  }
  function stageHoldView(){
    const pipeline=state.tab==='pipeline',id=pipeline?state.pipelineStage:state.assignmentStage,stage=assignmentStages.find(item=>item.id===id);
    root.querySelector('.csa-stage-hold').innerHTML='<div class="csa-unadapted"><span class="csa-eyebrow">DEEPSEEK V4 · DECODE CSA · TP=2</span><h2>'+esc(stage?stage.full:id)+'</h2><p>'+(pipeline?'Pipeline 建设中。当前仅支持 qk_pv Pipeline 可视化。':'该计算阶段的 Work Assignment 尚未建设。qk_pv 已完成 Work Item 与 SPMD Lane 分配可视化，可切换到 qk_pv 查看。')+'</p></div>';
  }
  // P5：跨模块探索只保存对象身份，Work Item / Scenario 数据仍由 QK Work Controller 派生。
  let navigationReturn=null;
  const ancestorsOf=id=>{const parts=id.split('/');return parts.slice(0,-1).map((_,i)=>parts.slice(0,i+1).join('/'));};
  const returnTabs={assignment:'Assignment',pipeline:'Pipeline',graph:'Graph'};
  function returnLabel(focus){
    if(focus?.kind==='work-item')return '← 返回 Work Item '+focus.workItemId;
    if(focus?.kind==='lane')return '← 返回 Lane '+focus.blockId;
    if(focus?.kind==='query')return '← 返回 Q'+(focus.tokenIndex??'');
    if(focus?.kind==='request')return '← 返回 Request '+focus.requestIndex;
    return '← 返回 '+(returnTabs[navigationReturn?.tab]||'Pipeline');
  }
  function navigateToGraph(targetNodeId){
    if(!targetNodeId||!semanticNodes.has(targetNodeId))return false;
    if(state.tab!=='graph')navigationReturn={tab:state.tab,focus:state.focus?{...state.focus}:null};
    saveCamera();
    setFocus(graphFocus(targetNodeId));
    // 只展开目标路径的祖先，避免整图展开；setFocus 会清空 Graph 选中态，因此在其后写回。
    const v=state.views.graph;
    v.expanded=ancestorsOf(targetNodeId);
    v.scope=v.expanded.length?'local':'overall';
    v.camera=null;v.nodeId=targetNodeId;v.edgeId=null;v.bookmarkedNodeId=null;v.inspectorOpen=true;v.projectedFocusKey=focusKey();
    state.tab='graph';
    render();
    return true;
  }
  function returnFromNavigation(){
    const back=navigationReturn;if(!back)return;
    navigationReturn=null;
    const scenarioId=qkController?.getScenario?.()?.items?.[0]?.scenarioId;
    let focus=back.focus?{...back.focus}:null;
    // Scenario 变化后相同数字 ID 不再指同一个 Work Item，退回 qk_pv 而不按 ID 强行恢复。
    const detached=focus&&['request','query','work-item','lane'].includes(focus.kind)&&scenarioId&&focus.scenarioId&&focus.scenarioId!==scenarioId;
    if(detached)focus={kind:'module',rank:focus.rank??state.rank,semanticId:qkPvId,moduleId:qkPvId,objectId:'attention'};
    state.tab=back.tab||'pipeline';
    // setFocus 会按恢复的对象重新推导 Assignment 阶段，无需另存层级。
    setFocus(focus);
    render();
  }
  function saveCamera(){if(controller&&state.tab!=='graph')state.views[state.tab].camera=controller.getTransform();}
  function chrome(){
    root.querySelectorAll('[data-csa-tab]').forEach(b=>{const active=b.dataset.csaTab===state.tab;b.classList.toggle('is-selected',active);b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1;});
    root.querySelector('#csa-panel').setAttribute('aria-labelledby','csa-tab-'+state.tab);
    const stagebar=root.querySelector('.csa-stagebar');
    if(stagebar)stagebar.hidden=state.tab==='graph';
    const visibleStage=state.tab==='pipeline'?state.pipelineStage:state.assignmentStage;
    root.querySelectorAll('[data-csa-stage]').forEach(b=>{const active=b.dataset.csaStage===visibleStage;b.classList.toggle('is-selected',active);b.setAttribute('aria-pressed',String(active));});
    // Graph 层级完全由 expanded 推导：没有展开节点 = 整体，存在展开节点 = 局部。
    const scope=state.tab==='graph'?(state.views.graph.expanded.length?'local':'overall'):state.views[state.tab].scope;
    if(state.tab==='graph')state.views.graph.scope=scope;
    const focusbar=root.querySelector('.csa-focusbar');if(focusbar)focusbar.hidden=true;
    root.querySelector('[data-csa-return]').hidden=state.tab!=='graph'||scope!=='local';root.querySelector('[data-csa-return]').textContent='← 返回上级';
    const navReturn=root.querySelector('[data-csa-navreturn]');
    if(navReturn){navReturn.hidden=!navigationReturn;if(navigationReturn)navReturn.textContent=returnLabel(navigationReturn.focus);}
    const crumbs=state.tab==='pipeline'&&state.pipelineStage===qkPvId&&state.focus?.kind==='work-item'?focusCrumbs().filter(item=>!/^Request |^Q\d+$|^Work Item /.test(item.label)):focusCrumbs(),scopeId=state.views.graph.expanded.at(-1),scopeName=semanticNodes.get(scopeId)?.name;
    root.querySelector('[data-csa-breadcrumb]').innerHTML=crumbs.map((item,index)=>'<span class="csa-crumb">'+(index?'<span class="csa-crumb-separator" aria-hidden="true">/</span>':'')+(item.id!==undefined&&index<crumbs.length-1?'<button type="button" data-csa-crumb="'+esc(item.id)+'">'+esc(item.label)+'</button>':'<span aria-current="location">'+esc(item.label)+'</span>')+'</span>').join('')+(state.tab==='graph'&&scopeName?'<span class="csa-scope-context">视图层级：'+esc(scopeName)+'</span>':'');
    const drill=root.querySelector('[data-csa-drill]');if(drill)drill.disabled=state.tab==='graph';
  }
  // Jumping to a source line only reads correctly once the explorer releases the workarea,
  // so leave the local scene first and then scroll the editor to the range.
  function qkSource(file,a,z){
    // 始终按文件定位：主文件也可能不是当前打开的文件。
    const go=()=>global.PtoComputeIDE.openSource({file,start:a,end:z});
    if(state.open){close();requestAnimationFrame(go);}else go();
  }
  function details(){
    const host=root.querySelector('.csa-detail'),selection=state.focus,o=selection&&objects[objectForFocus()];
    if(!o){host.innerHTML='<span class="csa-eyebrow">DEEPSEEK V4 · CSA</span><h2>从数据归属理解计算</h2><p>选择 Rank、Tensor 或任务查看解释；点击“进入局部”查看它所属的关键关系。</p><dl><dt>Assignment</dt><dd>本地输入 → 全组 KV → 按 O Group 重分布</dd><dt>Pipeline</dt><dd>计算分支、Cache 就绪与跨 Rank 通信</dd><dt>默认容量</dt><dd>TP=2，T=256，整组 512 token 行</dd></dl><p class="csa-note">运行时 T 可变。图中两个 Rank 是同一程序的两个设备实例，源码中的 host rank 循环不表示设备串行执行。</p>';host.scrollTop=0;return;}
    host.innerHTML='<span class="csa-eyebrow">'+(selection.rank==null?'跨 Rank':'RANK '+selection.rank)+' · '+model.kindLabels[o.kind]+'</span><h2>'+esc(o.label)+'</h2><dl>'+[['What',o.what],['Why · 静态解释',o.why],['Data · 源码 Shape',o.data],['How · 源码组织',o.how]].map(([k,v])=>'<dt>'+k+'</dt><dd>'+esc(v)+'</dd>').join('')+'</dl><button class="btn btn-sm" data-csa-drill>进入局部 →</button><section class="csa-sources"><h3>Source · 本地源码</h3>'+o.source.map(([f,a,z])=>{
      if(f==='decode_csa.py')return '<button class="btn btn-ghost btn-sm" data-csa-source="'+a+':'+z+'">'+f+' : '+a+'–'+z+' ↗</button>';
      // 子模块现在可以真正打开，用与主文件一致的入口，不再内联一段只读摘录。
      return '<button class="btn btn-ghost btn-sm" data-csa-source-file="'+esc(f)+':'+a+':'+z+'">'+esc(f)+' : '+a+'–'+z+' ↗</button>';
    }).join('')+'</section><p class="csa-note">源码事实与静态依赖示意。布局为简化示意，不代表实际调度、耗时或通信带宽。</p>';host.scrollTop=0;
  }
  function render(){
    controller?.destroy();controller=null;semanticController?.destroy();semanticController=null;const stage=root.querySelector('.csa-stage'),view=state.views[state.tab];stage.replaceChildren();
    const qkHost=root.querySelector('.csa-qk-host'),qkActive=state.tab==='assignment'&&state.assignmentStage===qkPvId;
    const stageHoldHost=root.querySelector('.csa-stage-hold'),stageHoldActive=state.tab!=='graph'&&(state.tab==='assignment'?state.assignmentStage:state.pipelineStage)!==qkPvId;
    const qkPipelineHost=root.querySelector('.csa-qk-pipeline-host'),qkPipelineActive=state.tab==='pipeline'&&state.pipelineStage===qkPvId;
    if(!qkActive)qkController?.pause();
    root.querySelector('.csa-workspace').hidden=state.tab!=='graph';qkHost.hidden=!qkActive;stageHoldHost.hidden=!stageHoldActive;qkPipelineHost.hidden=!qkPipelineActive;
    if(stageHoldActive){stageHoldView();chrome();return;}
    if(qkActive){
      if(qkController){qkController.setSelection(qkSelection());qkController.render();}
      else qkController=global.PtoCsaQkWork.create(qkHost,{onSource:qkSource,rank:state.focus?.rank??state.rank,selection:qkSelection(),onContext:context=>{
        setFocus({kind:'module',rank:context.rank,scenarioId:context.scenarioId,semanticId:qkPvId,moduleId:qkPvId,objectId:'attention'});chrome();
      },onSelection:selection=>{
        navigationReturn=null;
        if(selection.type==='item')setFocus({kind:'work-item',rank:selection.rank,scenarioId:selection.scenarioId,moduleId:qkPvId,operationId:qkPvId+'/dispatch',workItemId:selection.id,requestIndex:selection.requestIndex,tokenIndex:selection.tokenIndex,positionId:selection.positionId,sparseBlockIndex:selection.sparseBlockIndex,valid:selection.valid,validKvCount:selection.validKvCount,queuePosition:selection.queuePosition,laneId:selection.laneId,laneIteration:selection.laneIteration});
        else if(selection.type==='request')setFocus({kind:'request',rank:selection.rank,scenarioId:selection.scenarioId,moduleId:qkPvId,requestIndex:selection.requestIndex});
        else if(selection.type==='query')setFocus({kind:'query',rank:selection.rank,scenarioId:selection.scenarioId,moduleId:qkPvId,requestIndex:selection.requestIndex,tokenIndex:selection.tokenIndex});
        else setFocus({kind:'lane',rank:selection.rank,scenarioId:selection.scenarioId,moduleId:qkPvId,operationId:qkPvId+'/dispatch',blockId:selection.block,laneId:selection.laneId});
        chrome();
      }});
      chrome();return;
    }
    if(qkPipelineActive){
      const context=qkController?.getState()||{mode:'source',workload:'mixed',position:8192,rank:state.focus?.rank??state.rank};
      if(!qkPipelineController)qkPipelineController=global.PtoCsaQkPipeline.create(qkPipelineHost,{onFocus:focus=>{navigationReturn=null;setFocus(focus);chrome();},onSource:qkSource,onNavigate:navigateToGraph,onRankChange:qkSetRank,moduleId:qkPvId});
      qkPipelineController.update(isQkFocus(state.focus)?state.focus:null,context,qkController?.getScenario());
      root.querySelector('.csa-focusbar').innerHTML='';chrome();return;
    }
    // Decode CSA 的 Assignment / Pipeline 均在上方返回；旧版 Distributed Execution 不再进入渲染分支。
    if(state.tab!=='graph'){chrome();return;}
    root.querySelector('.csa-focusbar').innerHTML=state.tab==='graph'?'':Object.entries(model.topics).map(([key,label])=>'<button class="btn btn-ghost btn-sm '+(view.topic===key?'is-selected':'')+'" data-csa-focus="'+key+'" aria-pressed="'+(view.topic===key)+'">'+label+'</button>').join('');
    if(state.tab==='graph'){
      root.querySelector('.csa-footer').hidden=true;
      root.querySelector('.csa-inspector').hidden=true;
      root.querySelector('.csa-caption').innerHTML='<b>CSA Compute Graph</b><span>Stage 为语义分组；Module 可用 + 逐层展开。Tensor / Cache 表示数据与状态；连线区分数据与 Task / Control 依赖，不表示执行时长。</span>';
      stage.classList.add('csa-semantic-stage');
      semanticController=global.PtoVisualization2Graph.mount(stage,{onSelect:()=>{},onView:patch=>{
        if(patch.expanded){const opened=new Set(patch.expanded);patch.expanded=patch.expanded.filter(id=>id.split('/').slice(0,-1).every((_,i)=>opened.has(id.split('/').slice(0,i+1).join('/'))));}
        if(patch.expanded){delete patch.nodeId;delete patch.inspectorOpen;}
        if(patch.nodeId)state.views.graph.edgeId=null;
        Object.assign(state.views.graph,patch);
        if(!semanticController)return;
        if(patch.expanded)state.views.graph.scope=patch.expanded.length?'local':'overall';
        semanticController.update(null,state.views.graph,{active:true});chrome();
      },adapter:{
        getModel:()=>semantic,executionData:()=>null,fitOnEnter:true,layoutOptions:{rootLayerGap:44},labelMinWidth:96,expandedScaleFloor:node=>node.parentId?.62:0,expandedXOffset:node=>node.id==='cache'?140:0,focusExpandedTop:true,stageLabel:g=>'<b>'+esc(g.label)+'</b>',
        context:()=> '源码静态数据、状态和 Task ID 依赖；Stage 不表示严格串行。',
        resolveFocus:(g,selection,v)=>{const path=graphIdForFocus()?.split('/')||[];let nodeId=null;while(path.length&&!nodeId){const candidate=path.join('/');if(g.nodes.some(n=>n.id===candidate))nodeId=candidate;else path.pop();}return {nodeId,edgeId:v.edgeId,related:nodeId?[nodeId]:[],kind:'node'};},
        hasInspector:()=>!!(state.views.graph.nodeId||state.views.graph.edgeId),
        onNode:id=>{setFocus(graphFocus(id));state.views.graph.projectedFocusKey=focusKey();state.views.graph.nodeId=id;state.views.graph.bookmarkedNodeId=null;state.views.graph.edgeId=null;state.views.graph.inspectorOpen=true;semanticController.update(null,state.views.graph,{active:true});chrome();},
        onEdge:id=>{setFocus(null);state.views.graph.edgeId=id;state.views.graph.bookmarkedNodeId=null;state.views.graph.inspectorOpen=true;semanticController.update(null,state.views.graph,{active:true});chrome();},
        onSync:()=>{},
        onSource:range=>{const [file,a,z]=range.split(':');global.PtoComputeIDE.showSourceFile(file,Number(a),Number(z));},
        inspector:(node,sync,g)=>{
          const selectedEdge=g.edges.find(edge=>edge.id===state.views.graph.edgeId);
          const semanticNode=node?.semantic||semantic.nodes.find(item=>item.id===node?.id?.split('/')[0]);
          const byId=new Map(g.nodes.map(item=>[item.id,item]));
          const label=id=>byId.get(id)?.title||id;
          const semanticLabel=id=>g.nodes.find(item=>item.id===id||item.id.endsWith('/'+id))?.title||id;
          const kind=edge=>edge.communication?'跨 Rank Communication':({data:'Data',state:'State',feedback:'跨 Decode Step State',control:'Control',computation:'Computation'}[edge.type]||edge.type);
          const connection=edge=>`${label(edge.s)} → ${label(edge.t)} · ${kind(edge)}${edge.label?' · '+edge.label:''}`;
          const section=(title,rows)=>rows.length?'<section><h3>'+esc(title)+'</h3><dl>'+rows.map(([key,value])=>'<dt>'+esc(key)+'</dt><dd>'+esc(value)+'</dd>').join('')+'</dl></section>':'';
          if(selectedEdge){
            const sourceNode=byId.get(selectedEdge.s)?.semantic,targetNode=byId.get(selectedEdge.t)?.semantic;
            const html=section('Dependency',[["Type",kind(selectedEdge)],["What",selectedEdge.label||'源码静态关系'],["Producer",label(selectedEdge.s)],["Consumer",label(selectedEdge.t)]])+
              section('Source',[["Producer",sourceNode?.source?.file+' : '+(sourceNode?.source?.ranges||[]).map(range=>range.start+'–'+range.end).join('、')],["Consumer",targetNode?.source?.file+' : '+(targetNode?.source?.ranges||[]).map(range=>range.start+'–'+range.end).join('、')]].filter(([,value])=>value&&!value.startsWith('undefined')));
            return {title:selectedEdge.label||'依赖',html};
          }
          const item=semanticNode,payload=item?.[item.type]||{},incoming=node?g.edges.filter(edge=>edge.t===node.id):[],outgoing=node?g.edges.filter(edge=>edge.s===node.id):[];
          const loc=item?.source?.ranges?.[0],file=item?.source?.file;
          const shapeView=global.PTO_COMPUTATION_PRESENTATION.shapeLine(global.PTO_COMPUTATION_PRESENTATION.shapeAxes(semantic,payload));
          const shape=shapeView?.text&&(shapeView.symbol!==shapeView.text?shapeView.symbol+' = '+shapeView.text:shapeView.text);
          const facts=(item?.provenance||[]).filter(entry=>entry.kind==='fact');
          const resolved=(item?.provenance||[]).filter(entry=>entry.kind==='resolved');
          const rows=[['What',item?.description||node?.detail||'源码静态关系'],['Kind',item?.type||node?.type]];
          const io=section('Inputs / Outputs',[
            ['Inputs',incoming.length?incoming.map(connection).join('；'):'外部输入或当前展开层级未显示'],
            ['Outputs',outgoing.length?outgoing.map(connection).join('；'):'当前展开层级未显示下游'],
            ...(payload.producer?[['Producer',semanticLabel(payload.producer)]]:[]),...(payload.consumers?.length?[['Consumers',payload.consumers.map(semanticLabel).join('、')]]:[])
          ]);
          const metrics=section('Shape / Dtype / Storage',[
            ...(shape?[['Shape',shape]]:[]),...(payload.dtype?[['Dtype',payload.dtype]]:[]),
            ...(payload.storage?.space?[['Storage',payload.storage.space]]:[]),...(payload.symbol?[['Symbol',payload.symbol]]:[])
          ]);
          const dependencies=section('Dependencies',[...incoming,...outgoing].filter(edge=>['state','feedback','control'].includes(edge.type)||edge.communication).map(edge=>[kind(edge),connection(edge)]));
          const lifecycle=item?.type==='state'?section('State lifecycle',[
            ...(payload.initial?[['跨 Step / 调用前',payload.initial]]:[]),
            ...(payload.update?[['当前调用更新',payload.update]]:[]),
            ...(payload.feedback?[['后续 Decode Step',payload.feedback]]:[])
          ]):'';
          const sourceText=file==='decode_csa.py'?global.PTO_CSA_SOURCE.text:(global.PTO_CSA_SOURCE.dependencies[file]||global.PtoProjectFiles?.get?.(file)?.content);
          const excerpt=loc&&sourceText?sourceText.split('\n').slice(loc.start-1,loc.end).map((line,index)=>`${loc.start+index}  ${line}`).join('\n'):'';
          const sourceHtml=loc?'<section><h3>Source</h3><button class="btn btn-ghost btn-sm" data-viz2-graph-source="'+esc(file)+':'+loc.start+':'+loc.end+'">'+esc(file)+' : '+loc.start+'–'+loc.end+' ↗</button>'+
            (excerpt?'<details><summary>查看本地源码行段</summary><pre>'+esc(excerpt)+'</pre></details>':'')+'</section>':'';
          const provenance=section('Evidence',[
            ...facts.map(entry=>['源码事实',entry.origin+(entry.fields.length?' · '+entry.fields.join('、'): '')]),
            ...resolved.map(entry=>['静态推导',entry.origin+(entry.note?' · '+entry.note:'')])
          ]);
          return {title:node?.title||item?.name||'CSA',html:section('What',rows)+io+metrics+lifecycle+dependencies+sourceHtml+provenance};
        },tooltip:(id,edgeId,syncId,g)=>id?(g.nodes.find(n=>n.id===id)?.detail||id):(g.edges.find(e=>e.id===edgeId)?.label||'依赖')
      }});
      semanticController.update(null,view,{active:true});
    }else{
      stage.classList.remove('csa-semantic-stage','viz2-graph');
      root.querySelector('.csa-inspector').hidden=false;
      root.querySelector('.csa-footer').hidden=false;
      graph=view.scope==='local'?(state.tab==='assignment'?anatomy(view.topic):localGraph(view.topic)):state.tab==='assignment'?assignment(view.topic):pipeline();
      const qkContext=graphIdForFocus()?.startsWith(qkPvId);
      root.querySelector('.csa-caption').innerHTML='<b>'+esc(qkContext?'qk_pv · '+titles[state.tab]:view.scope==='local'?(state.tab==='assignment'?'Tensor / Cache Anatomy · ':'Task / Communication Dependency · ')+model.topics[view.topic]:titles[state.tab])+'</b><span>'+esc(qkContext?(state.tab==='assignment'?'当前图显示 Sparse Attention 所属 Rank 与数据归属；通过上方入口查看 qk_order、Work Item 和 SPMD Block 分配。':'当前图以 Sparse Attention 定位 qk_pv 所属静态依赖；qk_plan 与 cache_ready_dep 是派发前置条件，图形位置不表示执行时间。'):state.tab==='assignment'?'T 为本地 token 行；数据块仅示意归属与分组。':'主要静态依赖 · 分支不保证同时执行 · 拖拽平移 / Ctrl 或 ⌘ 滚轮缩放')+'</span>';
      const object=objectForFocus(),focusRank=state.focus?.rank??state.rank;let selectedId=object&&object!=='rank'?'r'+focusRank+':'+object:null;
      if(selectedId&&!graph.nodes.some(n=>n.id===selectedId)){let mapped=object;for(let i=0;i<4&&!graph.nodes.some(n=>n.id==='r'+focusRank+':'+mapped);i++)mapped=alias[mapped];selectedId=mapped?'r'+focusRank+':'+mapped:null;}
      if(state.focus?.kind==='rank')selectedId='rank-'+state.focus.rank;
      controller=global.PtoModelGraphvizPattern.renderController(stage,graph,{fitMode:'full',viewportPadding:24,reportOverlays:false,overlays:{evidence:false,edgeTags:false,edgeHover:false},activeNodeId:selectedId,initialTransform:view.camera||undefined,
        onSelect:({nodeId,source})=>{if(source==='init')return;const n=graph.nodes.find(x=>x.id===nodeId);if(n)setFocus(legacyFocus(n.object,n.rank));else if(nodeId.startsWith('rank-'))setFocus(legacyFocus('rank',Number(nodeId.split('-')[1])));details();chrome();},
        onClearSelection:()=>{setFocus(null);details();chrome();}});
    }
    if(controller){controller.svg.ptoModelGraphviz.metadata.clusterEntries.forEach(({el,cluster})=>{if(cluster.id.startsWith('rank-')){const label=el.querySelector('.pto-model-graphviz-cluster-label');if(label){label.style.pointerEvents='auto';label.style.cursor='pointer';}}});controller.svg.ptoModelGraphviz.metadata.edgeEntries.forEach(({el,edge})=>{if(edge.type==='communication')el.style.stroke='var(--primary)';});}
    if(state.tab!=='graph')details();chrome();
  }
  function open(button){if(!root)mount();if(state.open)return;trigger=button;state.open=true;root.hidden=false;const area=document.getElementById('ideWorkarea');area.dataset.visualization2='open';previousInert=[...area.children].filter(el=>el!==root).map(el=>[el,el.inert]);previousInert.forEach(([el])=>el.inert=true);render();root.querySelector('[data-csa-tab="'+state.tab+'"]').focus();}
  function close(){if(!state.open)return;saveCamera();navigationReturn=null;qkController?.pause();semanticController?.destroy();semanticController=null;state.open=false;root.hidden=true;delete document.getElementById('ideWorkarea').dataset.visualization2;previousInert.forEach(([el,inert])=>el.inert=inert);previousInert=[];trigger?.isConnected&&trigger.focus({preventScroll:true});}
  document.addEventListener('click',e=>{const b=e.target.closest('[data-visualization2-open]');if(b&&global.PtoComputeIDE.getScene()==='csa')open(b);});
  global.PtoCsaExplorer=Object.freeze({open,close,getState:()=>structuredClone(state)});
})(window);
