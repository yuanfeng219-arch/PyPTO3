/* qk_pv Pipeline projection. Scenario and qk_order belong to Assignment. */
(function(global){
  'use strict';
  const work=global.PtoCsaQkWork,subSource='decode_sparse_attn_csa.py',mainSource='decode_csa.py';
  const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const evidence={
    mask:{title:'valid_block_mask',fact:'每个 (Query, Sparse Block) 读取有效性；有效分支执行 Gather 与矩阵计算，无效分支写中性 Partial。',file:subSource,range:[223,226]},
    query:{title:'Query',fact:'同一 Query 的 64 Heads；Head Batch 0 读取 Heads 0–31，Batch 1 读取 Heads 32–63。',file:subSource,range:[278,283]},
    gather:{title:'KV Gather',fact:'SB0 从 Original KV 的 Page 连续区间 Gather；SB1–4 依据压缩索引逐行 Gather。无效槽位用有限值填充并由 sparse_bias 屏蔽。',file:subSource,range:[225,276]},
    kv:{title:'L1 KV Tile',fact:'一个 BF16 [128,512] KV Tile 供两个 Head Batch 共用；QK 使用 Kᵀ，PV 使用 V。',file:subSource,range:[225,296]},
    qk:{title:'QK Matmul',fact:'Q [32,512] × Kᵀ [512,128] → Scores FP32 [32,128]。32 个 Head 分别对应 128 个 KV 位置。',file:subSource,range:[280,284]},
    stats:{title:'Scale / Bias / Softmax Statistics',fact:'Scores → Scale → Bias → Row Max(m [32,1]) → Exp(exp [32,128]) → Row Sum(l [32,1])。当前仅是单个 Sparse Block 的局部统计。',file:subSource,range:[284,294]},
    pv:{title:'PV Matmul',fact:'exp 转为 BF16 [32,128] 后与 V [128,512] 相乘，产生 Partial O FP32 [32,512]；m、l 不进入 PV。',file:subSource,range:[294,296]},
    write:{title:'Partial Write',fact:'每批 32 Heads 的 m [32,1]、l [32,1]、o [32,512] 分成两个 H_TILE=16 的存储分段；两批共四段。',file:subSource,range:[297,304]},
    neutral:{title:'Neutral Partial Write',fact:'Invalid 分支跳过 Gather、QK、Softmax、PV，逐 Head 写 mi=-3.0e38、li=0、oi=0。',file:subSource,range:[305,313]},
    partial:{title:'Partial Results',fact:'该 Work Item 写出当前 Sparse Block 的 m/l/o。两批共四个 16 Head 分段；同一 Query 的其他 Sparse Blocks 也写各自 Partial。',file:subSource,range:[204,206]},
    merge:{title:'csa_merge_pack_publish',fact:'TP=2 主流程中，该后续 Task 依赖 qk_tid 与 attn_rope_tid；合并同一 Query 多个 Sparse Blocks 的 Partial m/l/o，随后 Normalize、Attention Sink、Inverse RoPE、Pack、Publish。它不在 qk_pv SPMD 循环内。',file:mainSource,range:[395,477]}
  };
  const sourceVersions={
    [subSource]:'sha256:7e4afa5eefa4eebabe421cda6da6c32c49f5e200c1dc74dcc63b4c3b6aa8800b',
    [mainSource]:'sha256:5376b4dbcda62b90f31c734ed533cecd4442f770cd7994f08350f08dfca69a0f'
  };
  function create(host,options){
    let focus=null,context=null,scenario=null,lane=0,level='lane',selection=null,camera=null,drag=null,suppressBlankClick=false,mapMode='path',lastItemId=null,mapCamera=null,mapDrag=null;
    const bounds={x:0,y:0,width:1600,height:650},padding=24;
    const moduleId=options.moduleId;
    const scenarioId=()=>scenario?.items[0]?.scenarioId;
    const detail=item=>work.getWorkItemDetails(item.id,scenario.queue,scenario.blockCount);
    function itemFocus(item){const d=detail(item);return {kind:'work-item',rank:context.rank,scenarioId:scenarioId(),moduleId,operationId:moduleId+'/dispatch',workItemId:item.id,requestIndex:item.requestIndex,tokenIndex:item.token,positionId:item.positionId,sparseBlockIndex:item.sparseBlock,valid:item.valid,validKvCount:item.validKvCount,queuePosition:d.queuePosition,laneId:d.block,laneIteration:d.iteration};}
    const laneFocus=index=>({kind:'lane',rank:context.rank,scenarioId:scenarioId(),moduleId,operationId:moduleId+'/dispatch',blockId:index,laneId:index});
    const currentItem=()=>focus?.kind==='work-item'&&focus.scenarioId===scenarioId()&&focus.rank===context.rank?scenario.byId.get(focus.workItemId):null;
    function pickItem(id){const item=scenario.byId.get(id);if(!item)return;if(focus?.workItemId!==id)camera=null;focus=itemFocus(item);lane=focus.laneId;lastItemId=id;level='item';selection=null;options.onFocus(focus);render();}
    function pickLane(index){if(index<0||index>=scenario.blockCount)return;if(level!=='item')lastItemId=null;lane=index;level='lane';selection=null;focus=laneFocus(index);options.onFocus(focus);render();}
    function pickRequest(index){lastItemId=null;focus=index<0?{kind:'module',rank:context.rank,semanticId:moduleId,moduleId}:{kind:'request',rank:context.rank,scenarioId:scenarioId(),moduleId,requestIndex:index};level='lane';options.onFocus(focus);render();}
    function pickQuery(index){lastItemId=null;focus={kind:'query',rank:context.rank,scenarioId:scenarioId(),moduleId,requestIndex:Math.floor(index/8),tokenIndex:index};level='lane';options.onFocus(focus);render();}
    function mapSvg(){return host.querySelector('.csa-qp-map-svg');}
    function applyMapCamera(){const svg=mapSvg();if(!svg)return;const fullWidth=Number(svg.dataset.mapWidth),fullHeight=Number(svg.dataset.mapHeight);if(!mapCamera)mapCamera={x:0,y:0,width:fullWidth,height:fullHeight};mapCamera.width=Math.min(fullWidth,mapCamera.width);mapCamera.height=Math.min(fullHeight,mapCamera.height);mapCamera.x=Math.max(0,Math.min(fullWidth-mapCamera.width,mapCamera.x));mapCamera.y=Math.max(0,Math.min(fullHeight-mapCamera.height,mapCamera.y));svg.setAttribute('viewBox',[mapCamera.x,mapCamera.y,mapCamera.width,mapCamera.height].join(' '));}
    function zoomMap(factor){const svg=mapSvg();if(!svg)return;applyMapCamera();const fullWidth=Number(svg.dataset.mapWidth),fullHeight=Number(svg.dataset.mapHeight),scale=Math.max(1,Math.min(6,fullWidth/mapCamera.width*factor)),cx=mapCamera.x+mapCamera.width/2,cy=mapCamera.y+mapCamera.height/2;mapCamera.width=fullWidth/scale;mapCamera.height=fullHeight/scale;mapCamera.x=cx-mapCamera.width/2;mapCamera.y=cy-mapCamera.height/2;applyMapCamera();}
    function viewport(){return host.querySelector('.csa-qp-stage-inner');}
    function cameraSize(){const view=viewport();return {width:view?.clientWidth||0,height:view?.clientHeight||0};}
    function clampCamera(){if(!camera)return;const visible=80;for(const [axis,size] of [['x','width'],['y','height']]){const low=bounds[axis]-camera[size]+visible,high=bounds[axis]+bounds[size]-visible;camera[axis]=Math.max(low,Math.min(high,camera[axis]));}}
    function applyCamera(){const svg=host.querySelector('.csa-qp-svg');if(!svg||!camera)return;const size=cameraSize();camera.viewportWidth=size.width;camera.viewportHeight=size.height;svg.setAttribute('viewBox',[camera.x,camera.y,camera.width,camera.height].join(' '));const output=host.querySelector('[data-qp-scale]');if(output&&size.width)output.textContent=Math.round(size.width/camera.width*100)+'%';}
    function fitCamera(){const size=cameraSize();if(size.width<1||size.height<1)return;const scale=Math.min((size.width-2*padding)/bounds.width,(size.height-2*padding)/bounds.height);if(!(scale>0))return;camera={x:bounds.x+(bounds.width-size.width/scale)/2,y:bounds.y+(bounds.height-size.height/scale)/2,width:size.width/scale,height:size.height/scale,mode:'fit'};applyCamera();}
    function zoomCamera(factor,clientX,clientY){const view=viewport();if(!view||!camera)return;const rect=view.getBoundingClientRect(),px=clientX==null?.5:Math.max(0,Math.min(1,(clientX-rect.left)/rect.width)),py=clientY==null?.5:Math.max(0,Math.min(1,(clientY-rect.top)/rect.height));const fitScale=Math.min((rect.width-2*padding)/bounds.width,(rect.height-2*padding)/bounds.height),currentScale=rect.width/camera.width,nextScale=Math.max(fitScale*.75,Math.min(fitScale*6,currentScale*factor)),width=rect.width/nextScale,height=rect.height/nextScale;camera={x:camera.x+camera.width*px-width*px,y:camera.y+camera.height*py-height*py,width,height,mode:'manual'};clampCamera();applyCamera();}
    function focusNode(operation,batch){const svg=host.querySelector('.csa-qp-svg');if(!svg||!camera||!operation)return;const target=svg.querySelector('[data-qp-operation="'+operation+'"]'+(batch==null?'':'[data-qp-batch="'+batch+'"]'));if(!target)return;const box=target.getBBox();camera.x=box.x+box.width/2-camera.width/2;camera.y=box.y+box.height/2-camera.height/2;camera.mode='manual';clampCamera();applyCamera();}
    function revealNode(operation,batch){const svg=host.querySelector('.csa-qp-svg'),node=svg?.querySelector('[data-qp-operation="'+operation+'"]'+(batch==null?'':'[data-qp-batch="'+batch+'"]'));if(!node||!camera)return;const box=node.getBBox(),margin=24;if(box.x<camera.x+margin||box.x+box.width>camera.x+camera.width-margin||box.y<camera.y+margin||box.y+box.height>camera.y+camera.height-margin)focusNode(operation,batch);}
    function resizeCamera(){if(!camera||!viewport())return;if(camera.mode==='fit')return fitCamera();const size=cameraSize(),scale=camera.viewportWidth/camera.width;if(!(size.width>0&&size.height>0&&scale>0))return;const cx=camera.x+camera.width/2,cy=camera.y+camera.height/2;camera.width=size.width/scale;camera.height=size.height/scale;camera.x=cx-camera.width/2;camera.y=cy-camera.height/2;clampCamera();applyCamera();}
    function syncSelection(preview=selection){
      const svg=host.querySelector('.csa-qp-svg');if(!svg)return;
      svg.dataset.selectedOperation=preview?.operation||'';svg.dataset.selectedBatch=preview?.batchIndex??'';
      const links=[...svg.querySelectorAll('[data-qp-from]')],nodes=[...svg.querySelectorAll('[data-qp-operation]')];
      const matches=(node,operation,batch)=>node.dataset.qpOperation===operation&&(node.dataset.qpBatch==null?batch==null:Number(node.dataset.qpBatch)===batch);
      const activeLinks=links.filter(link=>{
        const linkBatch=link.dataset.qpLinkBatch==null?null:Number(link.dataset.qpLinkBatch);
        if(!preview?.operation)return preview?.batchIndex!=null&&linkBatch===preview.batchIndex;
        return (link.dataset.qpFrom===preview.operation||link.dataset.qpTo===preview.operation)&&(linkBatch==null||preview.batchIndex==null||linkBatch===preview.batchIndex)&&
          (preview.batchIndex==null||linkBatch===preview.batchIndex||link.dataset.qpFrom===preview.operation&&linkBatch==null);
      });
      const related=new Set();
      activeLinks.forEach(link=>{
        const linkBatch=link.dataset.qpLinkBatch==null?null:Number(link.dataset.qpLinkBatch);
        const sourceBatch=link.dataset.qpFrom==='qk'||link.dataset.qpFrom==='stats'||link.dataset.qpFrom==='pv'||link.dataset.qpFrom==='write'?linkBatch:null;
        const targetBatch=link.dataset.qpTo==='qk'||link.dataset.qpTo==='stats'||link.dataset.qpTo==='pv'||link.dataset.qpTo==='write'?linkBatch:null;
        nodes.forEach(node=>{if(matches(node,link.dataset.qpFrom,sourceBatch)||matches(node,link.dataset.qpTo,targetBatch))related.add(node);});
      });
      svg.querySelectorAll('[data-qp-batch-row]').forEach(el=>el.classList.toggle('is-dimmed',preview?.batchIndex!=null&&Number(el.dataset.qpBatchRow)!==preview.batchIndex));
      nodes.forEach(node=>{const batch=node.dataset.qpBatch==null?null:Number(node.dataset.qpBatch),selected=!!preview?.operation&&matches(node,preview.operation,preview.batchIndex);
        node.classList.toggle('is-selected',selected);node.classList.toggle('is-related',(related.has(node)||!preview?.operation&&preview?.batchIndex!=null&&batch===preview.batchIndex)&&!selected);
        node.classList.toggle('is-dimmed',!!preview?.operation&&!selected&&!related.has(node)||!preview?.operation&&preview?.batchIndex!=null&&batch!=null&&batch!==preview.batchIndex);
      });
      links.forEach(link=>{const active=activeLinks.includes(link);link.classList.toggle('is-related',active);link.classList.toggle('is-dimmed',!!preview&&!active);});
      svg.querySelectorAll('[data-qp-pick-batch]').forEach(el=>el.classList.toggle('is-selected',preview?.batchIndex!=null&&Number(el.dataset.qpPickBatch)===preview.batchIndex));
      const summary=host.querySelector('[data-qp-dependency-summary]');if(summary){
        const upstream=[...new Set(activeLinks.filter(link=>link.dataset.qpTo===preview?.operation).map(link=>link.dataset.qpFrom))];
        const downstream=[...new Set(activeLinks.filter(link=>link.dataset.qpFrom===preview?.operation).map(link=>link.dataset.qpTo))];
        summary.textContent=preview?.operation?`${preview.batchIndex==null?'Work Item '+focus.workItemId:'HB'+preview.batchIndex} · ${evidence[preview.operation]?.title||preview.operation} | ↑ ${upstream.join(', ')||'输入'} | ↓ ${downstream.join(', ')||'输出'}`:'';
      }
      const aside=host.querySelector('.csa-qp-inspector');if(aside){const wasHidden=aside.hidden;aside.hidden=!selection?.operation;if(!aside.hidden&&aside.dataset.selectionKey!==`${selection.workItemId}:${selection.batchIndex}:${selection.operation}`){aside.innerHTML=inspector();aside.dataset.selectionKey=`${selection.workItemId}:${selection.batchIndex}:${selection.operation}`;}if(wasHidden!==aside.hidden)requestAnimationFrame(()=>camera?.mode==='fit'?fitCamera():resizeCamera());}
    }
    function select(batch,operation){selection={workItemId:focus.workItemId,batchIndex:batch,operation};syncSelection();}
    function navActions(){const item=currentItem(),operation=selection?.operation;if(!item||!operation)return '';
      const nav=(id,label,title)=>'<button class="btn btn-ghost btn-sm csa-qp-nav" data-qp-nav="'+id+'" title="'+esc(title||label)+'">'+esc(label)+'</button>';
      let body='';
      // P5：只发出跳转请求，Tab 与 Focus 由 csa-explorer 统一管理。
      if(operation==='query')body=nav('input/q-tensor','查看 Q 来源 · Q heads','Q heads 由 Q Projection 生成');
      else if(operation==='gather')body=item.sparseBlock===0
        ? nav('cache/original','查看上游 · Original KV Cache','SB0 从 Original KV 的 Page 连续区间 Gather')
        : nav('cache/compressed','查看上游 · Compressed KV Cache','QK/PV 实际读取的压缩 KV')+nav('cache/topk-positions','位置来源 · Top-K Positions','这些压缩位置由 Indexer 的相关性筛选产生');
      else if(operation==='partial')body=nav('redistribute/publish','查看下游 · Merge / Pack / Publish','csa_merge_pack_publish 合并多个 Sparse Block 的 partial')+'<span class="csa-qp-nav-label">Partial Tensor · sparse_blk_mi / li / oi</span>'+nav('sparse/partial-mi','mi','sparse_blk_mi')+nav('sparse/partial-li','li','sparse_blk_li')+nav('sparse/partial-oi','oi','sparse_blk_oi');
      else if(operation==='merge')body=nav('redistribute/publish','在 Graph 中查看下游 Task','csa_merge_pack_publish 汇合多个 Sparse Block 的 Partial');
      else if(operation==='mask')body=nav('sparse/attention/valid-block-mask','查看上游 · Valid Block Mask','每个 (Query, Sparse Block) 的有效性');
      return body?'<div class="csa-qp-nav-group">'+body+'</div>':'';
    }
    function directRelations(op,batch,valid){
      const target=(operation,batchIndex,label)=>({operation,batchIndex,label:label||evidence[operation]?.title||operation});
      if(!valid)return {upstream:{neutral:[target('mask',null)],partial:[target('neutral',null)]}[op]||[],downstream:{mask:[target('neutral',null)],neutral:[target('partial',null)]}[op]||[],resource:[]};
      const batches=[0,1],each=operation=>batches.map(index=>target(operation,index,`HB${index} · ${evidence[operation].title}`));
      const upstream={gather:[],kv:[target('gather',null)],qk:[target('query',null),target('kv',null,'共享 L1 KV Tile')],stats:[target('qk',batch)],pv:[target('stats',batch,'exp · Softmax'),target('kv',null,'共享 V Tile')],write:[target('stats',batch,'m / l · Softmax'),target('pv',batch,'Partial O · PV')],partial:each('write')}[op]||[];
      const downstream={query:each('qk'),gather:[target('kv',null)],kv:[...each('qk'),...each('pv')],qk:[target('stats',batch)],stats:[target('pv',batch,'exp → PV'),target('write',batch,'m / l → Partial Write')],pv:[target('write',batch)],write:[target('partial',null)]}[op]||[];
      return {upstream,downstream,resource:op==='kv'?['同一 L1 KV Tile 被两个 Head Batch 共用；各自计算 QK / PV']:['qk','pv'].includes(op)?['共享已准备好的 L1 KV Tile；虚线表示资源关系']:[]};
    }
    function operationDescriptor(operation=selection?.operation,batch=selection?.batchIndex){const item=currentItem(),info=evidence[operation];if(!item||!info)return null;
      const step=item.valid?({query:0,gather:0,kv:1,qk:batch===1?6:2,stats:batch===1?7:3,pv:batch===1?8:4,write:batch===1?9:5,partial:10})[operation]:({mask:0,neutral:1,partial:2})[operation];
      const unit=`csa/${scenarioId()}/rank/${context.rank}/work-item/${item.id}`,iteration=batch==null?null:`${unit}/head-batch/${batch}`,relations=directRelations(operation,batch,item.valid);
      const tensor=(name,dtype,shape)=>({name,kind:'tensor',...(dtype?{dtype}:{}),...(shape?{shape}:{} )});
      const data={mask:{inputs:[tensor('valid_block_mask')],outputs:[tensor('Validity')]},query:{inputs:[tensor('Q heads','BF16')],outputs:[tensor('Q','BF16',[32,512])]},gather:{inputs:[tensor(item.sparseBlock===0?'Original KV Cache':'Compressed KV Cache','BF16')],outputs:[tensor('Gathered KV','BF16',[128,512])]},kv:{inputs:[tensor('Gathered KV','BF16',[128,512])],outputs:[tensor('L1 KV Tile','BF16',[128,512])]},qk:{inputs:[tensor('Q','BF16',[32,512]),tensor('Kᵀ','BF16',[512,128])],outputs:[tensor('Scores','FP32',[32,128])]},stats:{inputs:[tensor('Scores','FP32',[32,128])],outputs:[tensor('m','FP32',[32,1]),tensor('exp','FP32',[32,128]),tensor('l','FP32',[32,1])]},pv:{inputs:[tensor('exp','BF16',[32,128]),tensor('V','BF16',[128,512])],outputs:[tensor('Partial O','FP32',[32,512])]},write:{inputs:[tensor('m','FP32',[32,1]),tensor('l','FP32',[32,1]),tensor('Partial O','FP32',[32,512])],outputs:[tensor('Partial m','FP32',[16,1]),tensor('Partial l','FP32',[16,1]),tensor('Partial o','FP32',[16,512])]},neutral:{inputs:[tensor('valid_block_mask')],outputs:[tensor('mi','FP32'),tensor('li','FP32'),tensor('oi','FP32')]},partial:{inputs:item.valid?[tensor('Partial m','FP32'),tensor('Partial l','FP32'),tensor('Partial o','FP32')]:[tensor('mi','FP32'),tensor('li','FP32'),tensor('oi','FP32')],outputs:[tensor('sparse_blk_mi','FP32'),tensor('sparse_blk_li','FP32'),tensor('sparse_blk_oi','FP32')]}};
      const dependencies=relations.upstream.map(r=>({kind:'data',from:`${r.batchIndex==null?unit:unit+'/head-batch/'+r.batchIndex}/${r.operation}`,to:`${iteration||unit}/${operation}`,...(r.operation==='kv'?{tensor:'L1 KV Tile'}:{})}));
      if(['qk','pv'].includes(operation))dependencies.push({kind:'resource-reuse',from:`${unit}/kv`,to:`${iteration}/${operation}`,resource:`${unit}/l1-kv-tile`});
      return {computeUnitId:unit,iterationId:iteration,operationId:`${iteration||unit}/${operation}`,operationType:operation,logicalPosition:step==null?null:{kind:'step',index:step},inputs:data[operation]?.inputs||[],outputs:data[operation]?.outputs||[],dependencies,resourceRefs:['kv','qk','pv'].includes(operation)?[`${unit}/l1-kv-tile`]:[],evidence:{computation:{level:'source-confirmed',source:{file:info.file,range:info.range,version:sourceVersions[info.file]}},shapeAndDtype:['qk','stats','pv','write','partial','neutral'].includes(operation)?{level:'source-confirmed',source:{file:subSource,range:info.range,version:sourceVersions[subSource]}}:null,logicalPosition:step==null?null:{level:'presentation-only',rule:'源码操作顺序按固定 Step 间距排列；stage=2 不给出真实调度位置'},headBatch:batch==null?null:{level:'model-derived',rule:'qk_h0 = qk_hb * QK_M_TILE',inputs:{qkHb:batch,qkMTile:32},source:{file:subSource,range:[278,304],version:sourceVersions[subSource]}},runtimeDuration:{level:'unverified'}}};
    }
    function inspector(){if(!selection?.operation)return '';const info=evidence[selection.operation]||evidence.query,item=currentItem(),batch=selection.batchIndex,op=selection.operation;
      const descriptor=operationDescriptor(op,batch);
      const logicalStep=item.valid?({query:0,gather:0,kv:1,qk:batch===1?6:2,stats:batch===1?7:3,pv:batch===1?8:4,write:batch===1?9:5,partial:10})[op]:({mask:0,neutral:1,partial:2})[op];
      const inputs={mask:'valid_block_mask',query:'Q heads',gather:item.sparseBlock===0?'Original KV Cache / Window':'Compressed KV Cache / Top-K indices',kv:'KV Gather',qk:'Q BF16 [32,512] · Kᵀ BF16 [512,128]',stats:'Scores FP32 [32,128] · From: HB'+batch+' / QK Matmul',pv:'exp BF16 [32,128] · V BF16 [128,512]',write:'m FP32 [32,1] · l FP32 [32,1] · Partial O FP32 [32,512]',neutral:'valid_block_mask',partial:'四个 H_TILE=16 分段'};
      const outputs={mask:'Valid / Invalid 分支',query:'Q Tile',gather:'KV Tile BF16 [128,512]',kv:'共享 L1 KV Tile',qk:'Scores FP32 [32,128] → Softmax',stats:'m FP32 [32,1] → Partial Write · exp [32,128] → PV · l FP32 [32,1] → Partial Write',pv:'Partial O FP32 [32,512] → Partial Write',write:'两个 16-Head Partial 分段 → Partial Results',neutral:'mi = −3.0e38 · li = 0 · oi = 0',partial:'sparse_blk_mi / li / oi → 下游 Merge Task'};
      const computation={qk:'Q × Kᵀ → Scores',stats:'Scale → Bias → Row Max → Exp → Row Sum',pv:'exp × V → Partial O',write:'写入 m / l / o 的 H_TILE 分段',partial:'汇集当前 Work Item 的四个 Partial 分段',neutral:'Invalid 分支写入中性 Partial'};
      const relation=directRelations(op,batch,item.valid);
      const section=(title,body)=>body?'<section><h3>'+title+'</h3><p>'+esc(body)+'</p></section>':'';
      const links=(title,items)=>items.length?'<div class="csa-qp-relation"><b>'+title+'</b>'+items.map(target=>'<button class="btn btn-ghost btn-sm" data-qp-jump="'+target.operation+'" data-qp-jump-batch="'+(target.batchIndex??'')+'" data-qp-jump-item="'+item.id+'">'+esc(target.label)+' ↗</button>').join('')+'</div>':'';
      return '<header><div><b>'+esc(info.title)+'</b><small>Work Item '+item.id+(batch!=null?' · HB'+batch:'')+(logicalStep!=null?' · 展示 Step '+logicalStep:'')+'</small></div><button class="btn btn-ghost btn-sm" data-qp-close aria-label="关闭详情">×</button></header>'+
        section('Inputs',inputs[op])+section('Computation',computation[op]||info.title)+section('Outputs',outputs[op])+
        '<section><h3>Dependencies</h3>'+links('Upstream',relation.upstream)+links('Downstream',relation.downstream)+(relation.resource.length?'<p>Resource / Synchronization · '+esc(relation.resource.join('；'))+'</p>':'')+'</section>'+
        '<section><h3>Source Evidence</h3><button class="btn btn-ghost btn-sm" data-qp-source="'+info.file+':'+info.range.join(':')+'">'+esc(info.file)+' · L'+info.range.join('–')+' ↗</button><p>计算关系：Source-confirmed；'+(descriptor?.logicalPosition?'Step 位置：Presentation-only，依据源码操作顺序排列；':'')+'Compiler schedule not verified。</p></section>'+navActions();
    }
    function mapScale(){return scenario?scenario.blockCount+' Lanes × '+Math.max(...scenario.blocks.map(block=>block.entries.length))+' Iterations':'';}
    function mapToolbar(){if(!scenario)return '';
      const active=focus?.scenarioId===scenarioId()&&focus?.rank===context.rank?focus:null;
      const request=active?.kind==='request'?active.requestIndex:null,query=active?.kind==='query'?active.tokenIndex:null,requestValue=request??(query!=null?Math.floor(query/8):'');
      const requestOptions=Array.from({length:Math.ceil(scenario.items.length/40)},(_,i)=>'<option value="'+i+'"'+(requestValue===i?' selected':'')+'>Request '+i+'</option>').join('');
      const queryOptions=Array.from({length:scenario.items.length/5},(_,i)=>'<option value="'+i+'"'+(query===i?' selected':'')+'>Q'+i+'</option>').join('');
      const rankOptions=[0,1].map(r=>'<option value="'+r+'"'+(context.rank===r?' selected':'')+'>'+r+'</option>').join('');
      return '<label>Rank <select data-qp-rank>'+rankOptions+'</select></label><label>Request <select data-qp-map-request><option value="">全部</option>'+requestOptions+'</select></label><label>Query <select data-qp-map-query><option value="">全部</option>'+queryOptions+'</select></label><label class="csa-qp-map-mode'+(mapMode==='rows'?' is-on':'')+'"><input type="checkbox" data-qp-map-rows'+(mapMode==='rows'?' checked':'')+'><span>Valid KV Rows</span></label>';
    }
    function contextBar(){const itemMode=level==='item'&&currentItem(),title='<b class="csa-qp-context-title">qk_pv</b>';
      return itemMode?'':'<div class="csa-qp-context">'+title+'<span class="csa-qp-context-meta">'+mapScale()+'</span><div class="csa-qp-context-tools"><span class="csa-qp-context-sep" aria-hidden="true">|</span>'+mapToolbar()+'</div></div>';}
    function laneView(){
      const columns=Math.max(...scenario.blocks.map(block=>block.entries.length)),cell=19,gap=3,row=25,left=73,top=43,summaryX=left+columns*(cell+gap)+9,width=summaryX+150,height=top+scenario.blockCount*row+27;
      const active=focus?.scenarioId===scenarioId()&&focus?.rank===context.rank?focus:null;
      const request=active?.kind==='request'?active.requestIndex:null,query=active?.kind==='query'?active.tokenIndex:null,selectedLane=active?.kind==='lane'?active.laneId:null;
      const ticks=Array.from({length:columns},(_,i)=>i%5===0||i===columns-1?'<text x="'+(left+i*(cell+gap)+cell/2)+'" y="27" text-anchor="middle">'+i+'</text>':'').join('');
      const rows=scenario.blocks.map(block=>{
        let original=0,compressed=0,invalid=0;
        const cells=block.entries.map(entry=>{
          const item=scenario.byId.get(entry.id);if(!item)return '';
          const type=!item.valid?'invalid':item.sparseBlock===0?'original':'compressed';
          if(type==='invalid')invalid++;else if(type==='original')original++;else compressed++;
          const related=request!=null?item.requestIndex===request:query!=null?item.token===query:selectedLane!=null?block.index===selectedLane:true;
          const note='Work Item '+item.id+' · Q'+item.token+' / SB'+item.sparseBlock+' · qk_order['+entry.queuePosition+'] · Lane '+block.index+' / iteration '+entry.iteration+' · '+(item.valid?type==='original'?'Original KV':'Compressed KV':'Invalid Skip')+' · '+item.validKvCount+' / 128 valid rows';
          const intensity=item.valid?Math.max(.18,Math.min(1,item.validKvCount/128)):0;
          return '<rect class="csa-qp-map-cell is-'+type+(related?'':' is-dimmed')+(item.id===lastItemId?' is-last':'')+'" x="'+(left+entry.iteration*(cell+gap))+'" y="'+(top+block.index*row)+'" width="'+cell+'" height="18" rx="2" data-qp-item="'+item.id+'" role="button" tabindex="0" aria-label="'+esc(note)+'" style="--kv-intensity:'+intensity+'"><title>'+esc(note)+'</title></rect>';
        }).join('');
        const y=top+block.index*row;
        return '<g class="csa-qp-map-row'+(selectedLane!=null&&selectedLane!==block.index?' is-dimmed':'')+'"><text class="csa-qp-map-lane" x="5" y="'+(y+14)+'" data-qp-lane="'+block.index+'" role="button" tabindex="0">Lane '+String(block.index).padStart(2,'0')+'</text>'+cells+'<text class="csa-qp-map-count" x="'+summaryX+'" y="'+(y+14)+'">'+original+' O · '+compressed+' C · '+invalid+' I</text></g>';
      }).join('');
      const block=selectedLane==null?null:scenario.blocks[selectedLane];
      const summary=block?'Lane '+selectedLane+' · '+block.entries.length+' Work Items · '+block.validCount+' Valid / '+block.invalidCount+' Invalid · qk_order['+selectedLane+'] + iteration × '+scenario.blockCount:request!=null?'Request '+request+' · '+scenario.items.filter(item=>item.requestIndex===request).length+' Work Items':query!=null?'Q'+query+' · 5 Sparse Block Work Items':'按 qk_order 的 Lane 迭代位置排列；列不表示时间或跨 Lane 等待';
      return '<div class="csa-qp-map"><div class="csa-qp-map-viewport" tabindex="0" role="group" aria-label="Lane Workload Map，拖拽平移，选择单元格查看 Work Item"><svg class="csa-qp-map-svg mode-'+mapMode+'" viewBox="0 0 '+width+' '+height+'" data-map-width="'+width+'" data-map-height="'+height+'" aria-label="SPMD Lane 与 Work Item 迭代矩阵"><text class="csa-qp-map-axis" x="'+left+'" y="14">Iteration →</text>'+ticks+rows+'</svg></div><div class="csa-qp-map-footer"><span>'+esc(summary)+'</span><span class="csa-qp-map-legend"><i class="is-original"></i>Original KV <i class="is-compressed"></i>Compressed KV <i class="is-invalid"></i>Invalid Skip'+(mapMode==='rows'?' · 颜色强度 = 有效 KV 行数 / 128':'')+'</span><div><button class="btn btn-ghost btn-sm" data-qp-map-zoom="0.8">−</button><button class="btn btn-ghost btn-sm" data-qp-map-fit>适应</button><button class="btn btn-ghost btn-sm" data-qp-map-zoom="1.25">＋</button></div></div></div>';
    }
    const label=(x,y,text,cls='label')=>'<text class="csa-qp-'+cls+'" x="'+x+'" y="'+y+'">'+esc(text)+'</text>';
    function node(op,batch,x,y,w,h,title,sub,extra=''){const active=selection?.operation===op&&selection?.batchIndex===batch,dim=selection?.batchIndex!=null&&batch!=null&&selection.batchIndex!==batch;return '<g class="csa-qp-node'+(active?' is-selected':'')+(dim?' is-dimmed':'')+' '+extra+'" data-qp-operation="'+op+'"'+(batch!=null?' data-qp-batch="'+batch+'"':'')+' role="button" tabindex="0" aria-label="'+esc((batch!=null?'Batch '+batch+' ':'')+title)+'"><rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="8"/>'+label(x+12,y+23,title,'title')+label(x+12,y+44,sub,'sub')+'</g>';}
    // Source-derived logical steps; stage=2 does not supply a compiler schedule.
    const stepX=index=>130+index*120;
    function timelineAxis(count){return label(130,28,'Logical Step →','section')+['Input / Gather','QK','Softmax','PV','Output / Write','Partial Results'].map((name,index)=>label(12,[112,212,312,412,512,602][index],name,'row-label')).join('')+Array.from({length:count},(_,index)=>'<g class="csa-qp-tick"><path d="M'+(stepX(index)-9)+' 48 V625"/>'+label(stepX(index)+55,63,String(index),'tick-label')+'</g>').join('');}
    function downstreamBoundary(x){return '<path class="csa-qp-task-boundary" d="M'+x+' 50 V630"/>'+label(x+12,91,'Task Boundary','boundary-label')+label(x+12,113,'Downstream Task','boundary-label');}
    function timelineLink(from,to,x1,y1,x2,y2,batch=null,kind='data'){
      const bend=(x1+x2)/2;
      return '<path class="csa-qp-timeline-'+kind+'" data-qp-from="'+from+'" data-qp-to="'+to+'"'+(batch!=null?' data-qp-link-batch="'+batch+'"':'')+' d="M'+x1+' '+y1+' C'+bend+' '+y1+' '+bend+' '+y2+' '+x2+' '+y2+'"'+(kind==='reuse'?'':' marker-end="url(#csa-qp-arrow)"')+'/>';
    }
    function timelineBatch(index){
      const start=index===0?2:6,x=stepX(start),qkY=180,softY=280,pvY=380,writeY=480;
      return '<g class="csa-qp-batch" data-qp-batch-row="'+index+'" style="--slot-color:var('+(index?'--success':'--primary')+')"><g class="csa-qp-batch-header" data-qp-pick-batch="'+index+'" role="button" tabindex="0" aria-label="Head Batch '+index+' · Heads '+(index*32)+'–'+(index*32+31)+'"><rect x="'+x+'" y="145" width="205" height="25" rx="5"/>'+label(x+8,163,'Head Batch '+index+' · Heads '+(index*32)+'–'+(index*32+31),'batch-title')+'</g>'+
        node('qk',index,x,qkY,110,56,'QK Matmul','Scores [32,128]')+
        node('stats',index,stepX(start+1),softY,110,56,'Softmax','m / exp / l')+
        node('pv',index,stepX(start+2),pvY,110,56,'PV Matmul','Partial O [32,512]')+
        node('write',index,stepX(start+3),writeY,110,56,'Partial Write','2 × H_TILE 16')+
        timelineLink('qk','stats',x+110,qkY+28,stepX(start+1),softY+28,index)+
        timelineLink('stats','pv',stepX(start+1)+110,softY+28,stepX(start+2),pvY+28,index)+
        timelineLink('pv','write',stepX(start+2)+110,pvY+28,stepX(start+3),writeY+28,index)+
        timelineLink('stats','write',stepX(start+1)+55,softY+56,stepX(start+3)+55,writeY,index)+'</g>';
    }
    function validTimeline(item){
      const gather=item.sparseBlock===0?'Original KV':'Compressed KV';
      return timelineAxis(11)+node('query',null,stepX(0),180,110,56,'Query Q'+item.token,'64 Heads × 512')+
        node('gather',null,stepX(0),80,110,56,gather,'SB'+item.sparseBlock+' · '+item.validKvCount+'/128')+
        node('kv',null,stepX(1),80,110,56,'L1 KV Tile','BF16 [128,512]')+
        timelineLink('gather','kv',stepX(0)+110,108,stepX(1),108)+
        [0,1].map(index=>{const start=index===0?2:6;const queryPath=index===0?timelineLink('query','qk',stepX(0)+110,208,stepX(start),208,index):'<path class="csa-qp-timeline-data" data-qp-from="query" data-qp-to="qk" data-qp-link-batch="1" d="M'+(stepX(0)+110)+' 208 V250 H'+(stepX(start)-10)+' V208 H'+stepX(start)+'" marker-end="url(#csa-qp-arrow)"/>';return queryPath+timelineLink('kv','qk',stepX(1)+55,136,stepX(start)+55,180,index,'reuse')+timelineLink('kv','pv',stepX(1)+110,108,stepX(start+2),408,index,'reuse');}).join('')+
        timelineBatch(0)+timelineBatch(1)+
        node('partial',null,stepX(10),570,110,56,'Partial m/l/o','4 × H_TILE 16')+
        downstreamBoundary(stepX(11)-15)+node('merge',null,stepX(11)+15,570,120,56,'Merge / Publish','多个 SB 汇合')+
        timelineLink('write','partial',stepX(5)+110,508,stepX(10),598,0)+
        timelineLink('write','partial',stepX(9)+110,508,stepX(10),598,1)+
        label(stepX(11)+15,551,'qk_tid + attn_rope_tid','boundary-note');
    }
    function invalidTimeline(item){return timelineAxis(3)+
      node('mask',null,stepX(0),80,110,56,'Validity Check','Q'+item.token+' / SB'+item.sparseBlock)+
      node('neutral',null,stepX(1),480,110,56,'Neutral Write','mi / li / oi')+
      node('partial',null,stepX(2),570,110,56,'Partial m/l/o','4 neutral segments')+
      downstreamBoundary(stepX(3)-15)+node('merge',null,stepX(3)+15,570,120,56,'Merge / Publish','多个 SB 汇合')+
      timelineLink('mask','neutral',stepX(0)+110,108,stepX(1),508)+
      timelineLink('neutral','partial',stepX(1)+110,508,stepX(2),598)+
      label(stepX(3)+15,551,'qk_tid + attn_rope_tid','boundary-note');}
    function itemView(){const item=currentItem();if(!item)return laneView();bounds.width=item.valid?1640:680;const d=detail(item),graph=(item.valid?validTimeline(item):invalidTimeline(item)),selectedBatch=selection?.batchIndex,entries=scenario.blocks[lane].entries,position=entries.findIndex(entry=>entry.id===item.id),windowStart=Math.max(0,Math.min(position-2,entries.length-5)),nearby=entries.slice(windowStart,windowStart+5);
      const itemNav=nearby.map(entry=>'<button class="btn btn-sm'+(entry.id===item.id?' is-selected':'')+'" data-qp-goto-item="'+entry.id+'" aria-current="'+(entry.id===item.id?'true':'false')+'">Item '+entry.id+'</button>').join('');
      return '<div class="csa-qp-unit-nav"><b>Lane '+String(lane).padStart(2,'0')+'</b><nav aria-label="Lane 工作项导航">'+itemNav+'</nav><button class="btn btn-ghost btn-sm" data-qp-back>← 返回 Lane Map</button></div>'+
      '<div class="csa-qp-item-top"><div><b>Work Item '+item.id+'</b><span class="csa-qp-item-meta">Request '+item.requestIndex+' · Q'+item.token+' · SB'+item.sparseBlock+' · '+(item.valid?'Valid':'Invalid')+' · '+item.validKvCount+'/128 KV Rows</span></div><details class="csa-qp-item-details"><summary>工作项详情</summary><div class="csa-qp-item-details-body"><dl><dt>Rank</dt><dd>'+context.rank+'</dd><dt>Lane ID</dt><dd>'+lane+'</dd><dt>Lane Iteration</dt><dd>'+d.iteration+'</dd><dt>Work Item ID</dt><dd>'+item.id+'</dd><dt>Request / Query</dt><dd>'+item.requestIndex+' / Q'+item.token+'</dd><dt>Sparse Block</dt><dd>SB'+item.sparseBlock+'</dd><dt>qk_order Queue Position</dt><dd>'+d.queuePosition+'</dd><dt>Validity</dt><dd>'+(item.valid?'Valid':'Invalid')+'</dd><dt>Valid KV Rows</dt><dd>'+item.validKvCount+'/128</dd></dl><button class="btn btn-ghost btn-sm" data-qp-nav="sparse/attention/qk-order">查看 qk_order ↗</button><button class="btn btn-ghost btn-sm" data-qp-nav="sparse/attention/valid-block-mask">查看 valid_block_mask ↗</button></div></details></div>'+
      '<div class="csa-qp-stage"><div class="csa-qp-stage-inner" tabindex="0" role="group" aria-label="计算画布，点击节点高亮依赖，拖拽或触控板平移，方向键移动，Ctrl 或 Command 滚轮缩放"><svg class="csa-qp-svg" viewBox="0 0 '+bounds.width+' '+bounds.height+'" data-selected-operation="'+esc(selection?.operation||'')+'" data-selected-batch="'+(selectedBatch??'')+'" aria-label="Work Item Logical Pipeline"><defs><marker id="csa-qp-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z"/></marker></defs>'+graph+'</svg></div><aside class="csa-qp-inspector" aria-label="计算详情" hidden></aside></div><div class="csa-qp-bottom"><span>实线：数据依赖 · 虚线：共享 KV Tile · 竖线：Task Boundary <small data-qp-dependency-summary aria-live="polite"></small></span><span>Source-derived logical order · Schedule not verified；<code>stage=2</code> 为软件流水声明</span><div class="csa-qp-footer-tools"><button class="btn btn-ghost btn-sm" data-qp-prev aria-label="上一个 Work Item" '+(position<=0?'disabled':'')+'>←</button><output>'+(position+1)+' / '+entries.length+'</output><button class="btn btn-ghost btn-sm" data-qp-next aria-label="下一个 Work Item" '+(position>=entries.length-1?'disabled':'')+'>→</button><button class="btn btn-ghost btn-sm" data-qp-zoom="0.8">−</button><output data-qp-scale></output><button class="btn btn-ghost btn-sm" data-qp-zoom="1.25">＋</button><button class="btn btn-ghost btn-sm" data-qp-fit>适应</button></div></div>';}
    function render(){if(!scenario)return;const itemMode=level==='item'&&currentItem();host.innerHTML='<div class="csa-qp-shell">'+contextBar()+(itemMode?itemView():laneView())+'</div>';if(itemMode)requestAnimationFrame(()=>{if(camera?.mode==='manual')resizeCamera();else fitCamera();syncSelection();});else applyMapCamera();}
    function update(nextFocus,nextContext,sharedScenario){const previous=focus,previousScenario=scenarioId(),previousRank=context?.rank;context={mode:nextContext.mode||'source',workload:nextContext.workload||'mixed',position:nextContext.position??8192,rank:nextContext.rank??nextFocus?.rank??0};scenario=sharedScenario||work.deriveScenario(context);const rankChanged=previousRank!=null&&previousRank!==context.rank;if(previousScenario!==scenarioId()||rankChanged){mapCamera=null;lastItemId=null;}const same=nextFocus?.kind==='work-item'&&nextFocus.scenarioId===scenarioId()&&nextFocus.rank===context.rank&&scenario.byId.has(nextFocus.workItemId);focus=same?itemFocus(scenario.byId.get(nextFocus.workItemId)):nextFocus;if(same){lane=focus.laneId;lastItemId=focus.workItemId;level='item';}else if(nextFocus?.kind==='lane'&&nextFocus.scenarioId===scenarioId()&&nextFocus.rank===context.rank){lane=nextFocus.laneId;level='lane';}else{lane=Math.min(lane,scenario.blockCount-1);level='lane';}if(rankChanged||previous?.workItemId!==focus?.workItemId||previous?.scenarioId!==focus?.scenarioId){selection=null;camera=null;}render();}
    host.addEventListener('click',event=>{const target=event.target.closest('button,[data-qp-operation],[data-qp-pick-batch],[data-qp-item],[data-qp-lane]');if(!target)return;
      if(target.dataset.qpLane!=null)return pickLane(Number(target.dataset.qpLane));
      if(target.dataset.qpItem!=null)return pickItem(Number(target.dataset.qpItem));
      if(target.dataset.qpMapZoom){zoomMap(Number(target.dataset.qpMapZoom));return;}
      if(target.dataset.qpJump){if(Number(target.dataset.qpJumpItem)!==focus?.workItemId)return;const batch=target.dataset.qpJumpBatch===''?null:Number(target.dataset.qpJumpBatch);select(batch,target.dataset.qpJump);requestAnimationFrame(()=>revealNode(target.dataset.qpJump,batch));return;}
      if(target.hasAttribute('data-qp-map-fit')){mapCamera=null;applyMapCamera();return;}
      if(target.dataset.qpNav)return options.onNavigate?.(target.dataset.qpNav);
      if(target.hasAttribute('data-qp-back'))return pickLane(lane);
      if(target.hasAttribute('data-qp-prev')||target.hasAttribute('data-qp-next')){const index=focus.laneIteration+(target.hasAttribute('data-qp-next')?1:-1),entry=scenario.blocks[lane].entries[index];if(entry)pickItem(entry.id);return;}
      if(target.dataset.qpGotoItem!=null)return pickItem(Number(target.dataset.qpGotoItem));
      if(target.dataset.qpPickBatch!=null){const batch=Number(target.dataset.qpPickBatch);selection=selection?.batchIndex===batch&&!selection.operation?null:{workItemId:focus.workItemId,batchIndex:batch,operation:null};syncSelection();return;}
      if(target.dataset.qpOperation){if(target.dataset.qpOperation==='merge')return options.onNavigate?.('redistribute/publish');const batch=target.dataset.qpBatch==null?null:Number(target.dataset.qpBatch);select(batch,target.dataset.qpOperation);return;}
      if(target.hasAttribute('data-qp-close')){selection=null;syncSelection();return;}
      if(target.dataset.qpSource){const [file,a,z]=target.dataset.qpSource.split(':');options.onSource(file,Number(a),Number(z));return;}
      if(target.dataset.qpZoom){zoomCamera(Number(target.dataset.qpZoom));return;}
      if(target.hasAttribute('data-qp-fit'))fitCamera();
    });
    host.addEventListener('click',event=>{if(suppressBlankClick){suppressBlankClick=false;return;}if(event.target.closest('.csa-qp-stage-inner')&&!event.target.closest('[data-qp-operation],[data-qp-pick-batch]')&&selection){selection=null;syncSelection();}});
    host.addEventListener('pointerover',event=>{const node=event.target.closest('[data-qp-operation]');if(node&&node.dataset.qpOperation!=='merge'&&host.contains(node))syncSelection({workItemId:focus?.workItemId,operation:node.dataset.qpOperation,batchIndex:node.dataset.qpBatch==null?null:Number(node.dataset.qpBatch)});});
    host.addEventListener('pointerout',event=>{if(event.target.closest('[data-qp-operation]')&&!event.relatedTarget?.closest?.('[data-qp-operation]'))syncSelection();});
    host.addEventListener('change',event=>{if(event.target.matches('[data-qp-rank]')){const rank=Number(event.target.value);if(rank!==context.rank)options.onRankChange?.(rank);}else if(event.target.matches('[data-qp-map-request]')){const value=event.target.value;value===''?pickRequest(-1):pickRequest(Number(value));}else if(event.target.matches('[data-qp-map-query]')){const value=event.target.value;value===''?pickRequest(-1):pickQuery(Number(value));}else if(event.target.matches('[data-qp-map-rows]')){mapMode=event.target.checked?'rows':'path';render();}});
    host.addEventListener('keydown',event=>{const target=event.target;if(target.classList.contains('csa-qp-stage-inner')&&camera&&['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)){event.preventDefault();camera.x+=(event.key==='ArrowRight'?1:event.key==='ArrowLeft'?-1:0)*camera.width*.12;camera.y+=(event.key==='ArrowDown'?1:event.key==='ArrowUp'?-1:0)*camera.height*.12;camera.mode='manual';clampCamera();applyCamera();return;}if(event.key!=='Enter'&&event.key!==' ')return;if(target.dataset.qpItem!=null||target.dataset.qpLane!=null){event.preventDefault();target.click();}else if(target.dataset.qpOperation){event.preventDefault();target.dataset.qpOperation==='merge'?options.onNavigate?.('redistribute/publish'):select(target.dataset.qpBatch==null?null:Number(target.dataset.qpBatch),target.dataset.qpOperation);}else if(target.dataset.qpPickBatch!=null){event.preventDefault();target.click();}});
    host.addEventListener('pointerdown',event=>{const view=event.target.closest('.csa-qp-map-viewport');if(event.button!==0||!view||event.target.closest('[data-qp-item],[data-qp-lane]'))return;applyMapCamera();mapDrag={x:event.clientX,y:event.clientY,camera:{...mapCamera},pointerId:event.pointerId};view.setPointerCapture(event.pointerId);view.classList.add('is-panning');});
    host.addEventListener('pointermove',event=>{if(!mapDrag||!mapCamera)return;const view=host.querySelector('.csa-qp-map-viewport'),box=view.getBoundingClientRect();if(!box.width||!box.height)return;mapCamera.x=mapDrag.camera.x-(event.clientX-mapDrag.x)*mapDrag.camera.width/box.width;mapCamera.y=mapDrag.camera.y-(event.clientY-mapDrag.y)*mapDrag.camera.height/box.height;applyMapCamera();});
    const endMapPan=()=>{mapDrag=null;host.querySelector('.csa-qp-map-viewport')?.classList.remove('is-panning');};host.addEventListener('pointerup',endMapPan);host.addEventListener('pointercancel',endMapPan);
    host.addEventListener('pointerdown',event=>{const view=event.target.closest('.csa-qp-stage-inner');if(event.button!==0||!view||event.target.closest('[data-qp-operation],[data-qp-pick-batch]')||!camera)return;drag={x:event.clientX,y:event.clientY,camera:{...camera},pointerId:event.pointerId};view.setPointerCapture(event.pointerId);view.classList.add('is-panning');});
    host.addEventListener('pointermove',event=>{if(!drag||!camera)return;const size=cameraSize();if(!size.width||!size.height)return;if(Math.abs(event.clientX-drag.x)+Math.abs(event.clientY-drag.y)>4)drag.moved=true;camera.x=drag.camera.x-(event.clientX-drag.x)*drag.camera.width/size.width;camera.y=drag.camera.y-(event.clientY-drag.y)*drag.camera.height/size.height;camera.mode='manual';clampCamera();applyCamera();});
    const endPan=()=>{suppressBlankClick=!!drag?.moved;drag=null;viewport()?.classList.remove('is-panning');};host.addEventListener('pointerup',endPan);host.addEventListener('pointercancel',endPan);
    host.addEventListener('wheel',event=>{if(!event.target.closest('.csa-qp-stage-inner')||!camera)return;event.preventDefault();if(event.ctrlKey||event.metaKey){zoomCamera(Math.exp(-event.deltaY*.003),event.clientX,event.clientY);}else{const size=cameraSize();camera.x+=event.deltaX*camera.width/size.width;camera.y+=event.deltaY*camera.height/size.height;camera.mode='manual';clampCamera();applyCamera();}},{passive:false});
    if(global.ResizeObserver)new global.ResizeObserver(()=>resizeCamera()).observe(host);
    return {update,getLocalSelection:()=>selection?{...selection}:null,getOperationDescriptor:()=>operationDescriptor()};
  }
  global.PtoCsaQkPipeline=Object.freeze({create});
})(window);
