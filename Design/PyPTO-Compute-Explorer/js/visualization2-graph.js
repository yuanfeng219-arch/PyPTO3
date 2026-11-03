/* Block-level mathematical graph: share the original semantic model, layout and
 * scene renderer. Runtime ownership and mathematical node identity stay distinct.
 */
(function (global) {
  'use strict';
  const escape = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  function resolveFocus(model, selection, nodeId) {
    const requested = selection?.graphNodeId || nodeId;
    const explicit=model.nodes.some(n=>n.id===requested)?requested:requested?.split('/')[0];
    if (model.nodes.some(node => node.id === explicit)) return { nodeId: explicit, related: [explicit], kind: 'node' };
    if (selection?.kind === 'task') {
      const regions = model.groups.filter(group => ['paged-attention', 'output-region'].includes(group.id));
      return { nodeId: null, related: [...new Set(regions.flatMap(group => group.members))].filter(id => id !== 'cache-append'), kind: 'task' };
    }
    if (selection?.kind === 'lane' && selection.ropeCore < model.constants.ROPE_CORES.value) {
      const prep = model.groups.find(group => group.id === 'qkv-preparation');
      return { nodeId: null, related: [...(prep?.members || []).filter(id => model.nodes.find(node=>node.id===id)?.type === 'composite'), 'cache-append'], kind: 'lane' };
    }
    return { nodeId: null, related: [], kind: 'none' };
  }
  // An optional semantic adapter supplies Task facts and selection/Inspector mapping.
  // Scene layout, cards, labels, expansion, camera and gestures have one owner.
  function mount(host, { onSelect, onView, execution, onEnter, onBlockEnter, adapter = null }) {
    let model = adapter?.getModel() || global.PTO_PYPA_SEMANTIC_GRAPH;
    let graph,rendered,nodes,nodeMap,nodeLabels,stageLabels,labels;
    let expanded=new Set(),layers={data:!!adapter,shape:false,precision:false,memory:false,control:false,sync:false,execution:true},layoutKey='',executionKey='';
    const layerNames={data:adapter?'Data 节点':'中间数据',shape:'Shape',precision:'精度',memory:'存储及执行',control:'状态 / 控制',execution:'执行次数',sync:'同步'};
    host.classList.add('viz2-graph');
    host.innerHTML = `<div class="viz2-graph-body"><div class="viz2-graph-stage">
      <div class="kf-pypa-graph__viewport viz2-graph-viewport" role="group" tabindex="0" aria-label="${adapter?'Block 内计算图':'Block 间完整计算图'}，滚轮缩放，拖拽平移"><div class="kf-pypa-graph__scene viz2-graph-scene" data-viz2-graph-scene></div><div class="kf-pa-map__tooltip" role="tooltip" hidden></div></div>
      <footer class="viz2-graph-footer"><span data-viz2-graph-context>完整计算图</span><div class="viz2-graph-footer__controls"><button class="btn btn-ghost btn-sm" data-viz2-graph-layer="data" aria-pressed="${layers.data}">${layerNames.data}</button><details class="viz2-graph-layer-menu"><summary>图层</summary><div class="viz2-graph-layer-menu__content" role="group" aria-label="信息叠加">${Object.entries(layerNames).filter(([key])=>key!=='data').map(([key,label])=>`<button class="btn btn-ghost btn-sm" data-viz2-graph-layer="${key}" aria-pressed="${layers[key]}">${label}</button>`).join('')}</div></details><div class="kf-pa-map__tools"><button class="btn btn-ghost btn-sm" type="button" data-viz2-graph-zoom="0.8" aria-label="缩小 Graph">−</button><output data-viz2-graph-scale>100%</output><button class="btn btn-ghost btn-sm" type="button" data-viz2-graph-zoom="1.25" aria-label="放大 Graph">＋</button><button class="btn btn-ghost btn-sm" type="button" data-viz2-graph-fit>Fit to View</button></div></div></footer>
      </div><aside class="pto-ide-frame__pane viz2-graph-inspector" aria-label="Graph 节点详情" hidden><header class="pto-ide-frame__pane-header"><b data-viz2-graph-title></b><button class="btn btn-ghost btn-sm" type="button" data-viz2-graph-close aria-label="关闭 Graph 详情">×</button></header><div class="pto-ide-frame__pane-body" data-viz2-graph-detail></div></aside></div>`;
    const viewport = host.querySelector('.viz2-graph-viewport'), scene = host.querySelector('[data-viz2-graph-scene]');
    const events=new AbortController();
    // Reuse only the original node skin; no legacy pane chrome is mounted.
    scene.classList.add('has-semantic-overlays');
    scene.dataset.overlayControl='false';
    const inspector = host.querySelector('aside'), tooltip = host.querySelector('[role="tooltip"]');
    function staticExecutionData(g){
      if(adapter?.executionData)return adapter.executionData(g);
      const scenario=selection?.task!=null?{task:selection.task,seqLen:execution?.seqLens?.[Math.floor(selection.task/model.constants.NUM_KV_HEADS.value)]}:global.PtoPagedAttentionContext?.scenario;
      return global.PtoPagedAttentionExecution?.derive(model,global.PtoPagedAttentionIntro?.model,scenario);
    }
    function rebuild(){
      scene.innerHTML='';labels?.remove();
      const overview=!adapter && !layers.data;
      const visibleModel=overview?global.PtoComputationVisibility.overview(model):model;
      graph=global.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(visibleModel,expanded,{flowLayout:true,overviewLayout:overview,...adapter?.layoutOptions});
      // Reserve the same visual footprint in Structure and Weight so switching
      // decoration never moves nodes. Front card dimensions remain unchanged.
      const counts=staticExecutionData(graph);
      const executionFootprints=Object.fromEntries(graph.nodes.flatMap(node=>{
        const entry=global.PtoPagedAttentionExecution.decoration(node,counts);
        return entry?[[node.id,global.PtoPagedAttentionExecution.deckOffset(entry.count)]]:[];
      }));
      const layoutOptions={flowLayout:true,overviewLayout:overview,executionFootprints,...adapter?.layoutOptions};
      graph=global.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(visibleModel,expanded,layoutOptions);
      scene.dataset.stateExpanded=String([...expanded].some(id=>visibleModel.nodes.find(node=>node.id===id)?.composite?.subgraph?.stateEvolution));
      if(adapter&&!layers.data)graph=global.PtoComputationVisibility.withoutData(graph,layoutOptions);
      Object.entries(layers).forEach(([key,value])=>{scene.dataset[`overlay${key[0].toUpperCase()+key.slice(1)}`]=String(value);});
    rendered = global.PtoStudioComputationGraph.renderScene(scene, graph, {
      view:'compute', expandedStages:expanded, synchronizations:true, compactNodes:!adapter, markerPrefix:'viz2-graph',
      cardHtml:node=>global.PtoStudioComputationGraph.cardHtml(node,model,layers),
    });
    nodes = rendered.placedNodes; nodeMap = new Map(nodes.map(node=>[node.id,node]));
    // Labels live in viewport coordinates, independently of scene scale. At low
    // zoom, callouts avoid overlaps without changing the shared graph geometry.
    labels = document.createElement('div');labels.className='viz2-graph-labels';
    labels.dataset.nodeRendering=adapter?'card':'compact';
    viewport.append(labels);
    nodeLabels=nodes.map(node=>{
      const label=document.createElement('span');label.className='viz2-graph-node-title';
      label.dataset.viz2GraphLabel=node.id;
      const name=document.createElement('span');name.dataset.viz2GraphName=node.id;name.textContent=node.title;label.append(name);
      if(node.expandable){const toggle=document.createElement('button');toggle.type='button';toggle.className='btn btn-ghost btn-sm';toggle.dataset.pypaExpandToggle=node.id;toggle.textContent=expanded.has(node.id)?'−':'+';toggle.setAttribute('aria-label',`${expanded.has(node.id)?'收起':'展开'} ${node.title}`);toggle.setAttribute('aria-expanded',String(expanded.has(node.id)));label.append(toggle);}
      labels.append(label);
      return {node,label};
    });
    const prepBlocks=[...new Set((execution?.phase0Model?.lanes || []).filter(lane=>lane.groups.length).map(lane=>lane.block))];
    const attentionBlocks=Array.from({length:Math.min(execution?.blocks || 0,execution?.taskCount || 0)},(_,id)=>id);
    stageLabels=graph.groups.filter(group=>model.groups.some(g=>g.id===group.id)).map(group=>{
      const semantic=model.groups.find(item=>item.id===group.id);
      const preparation=semantic?.members.includes('q-prep'), attention=semantic?.members.includes('qk');
      const blocks=preparation?prepBlocks:attentionBlocks;
      const label=document.createElement('div');label.className='viz2-graph-stage-title';
      label.dataset.viz2StageBlocks=blocks.join(',');label.dataset.viz2Stage=group.id;
      label.innerHTML=adapter?.stageLabel?adapter.stageLabel(group):`<b>${escape(group.label)}</b>${preparation?'<small>Projection · 外部输入</small>':attention?'<small>Cache Append · Attention / Output</small>':''}`;
      labels.append(label);
      let blockRail=null;
      if(!adapter&&(preparation||attention)){
        blockRail=document.createElement('div');blockRail.className='viz2-graph-block-rail';
        blockRail.dataset.viz2Stage=group.id;
        blockRail.setAttribute('role','group');blockRail.setAttribute('aria-label',`${group.label} 的 Blocks`);
        blockRail.innerHTML=`<span class="viz2-graph-block-rail__title">Blocks <small>${blocks.length}</small></span><div class="viz2-graph-block-rail__grid">${blocks.map(id=>`<button type="button" data-viz2-graph-block="${id}" aria-label="进入 Block ${String(id).padStart(2,'0')} 的 Task 视图" aria-pressed="false">${String(id).padStart(2,'0')}</button>`).join('')}</div>`;
        labels.append(blockRail);
      }
      return {group,label,blockRail};
    });
    }
    function paintExecution(){
      const data=layers.execution?staticExecutionData(graph):null;
      global.PtoStudioComputationGraph.applyExecution(scene,nodes,graph,data);
      adapter?.decorate?.(scene,nodes,graph,data);
      nodeLabels.forEach(({node,label})=>{const entry=global.PtoPagedAttentionExecution.decoration(node,data),name=label.querySelector('[data-viz2-graph-name]');name.dataset.fullName=node.title;name.textContent=node.title;label.querySelector('[data-viz2-execution-instances]')?.remove();if(entry){const badge=document.createElement('button');badge.type='button';badge.className='viz2-graph-floating-count';badge.dataset.viz2ExecutionInstances=node.id;badge.textContent=`×${entry.count}`;badge.title=`${entry.unit} · 查看逻辑实例`;label.append(badge);}});
    }

    function paintLabels(){
      const scale=camera.scale, width=viewport.clientWidth,height=viewport.clientHeight;
      const project=box=>({x:camera.x+box.x*scale,y:camera.y+box.y*scale,w:box.w*scale,h:box.h*scale});
      stageLabels.forEach(({group,label,blockRail})=>{
        const box=project(group),visible=box.x+box.w>0&&box.y+box.h>0&&box.x<width&&box.y<height;
        label.style.visibility=visible?'visible':'hidden';
        label.style.transform=`translate(${box.x+10}px,${box.y+2}px)`;
        if(blockRail){
          blockRail.style.visibility=visible?'visible':'hidden';
          blockRail.style.transform=`translate(${box.x-blockRail.offsetWidth-12}px,${box.y+8}px)`;
        }
      });
      nodeLabels.forEach(({node,label})=>{
        const box=project(node),visible=box.x+box.w>0&&box.y+box.h>0&&box.x<width&&box.y<height;
        label.style.visibility=visible?'visible':'hidden';
        if(!visible)return;
        const name=label.querySelector('[data-viz2-graph-name]');
        name.textContent=name.dataset.fullName||node.title;
        const labelWidth=Math.min(300,Math.max(adapter?.labelMinWidth ?? 196,box.w-8));
        label.style.maxWidth=`${labelWidth}px`;
        label.classList.toggle('is-compact',scale<.55);
        label.classList.toggle('is-selected',node.id===(selection?.graphNodeId||view.nodeId));
        label.style.transform=`translate(${box.x+box.w/2}px,${box.y+box.h/2}px) translate(-50%,-50%)`;
      });
    }

    let selection = null, view = {}, active = false, camera = null, first = true, lastObjectKey, lastFocusKey = '', inspectorKey = '', lastVisibleNodeId = null;
    let hoverNodeId = null;
    let lastNodeClick=null;
    let drag = null, suppressClick = false, lastSize = null, pendingFocus = false, skipFocusOnce = false;
    const bounds = ids => {
      const items=ids.map(id=>nodeMap.get(id)).filter(Boolean);
      if(!items.length)return null;
      const minX=Math.min(...items.map(n=>n.x)),minY=Math.min(...items.map(n=>n.y));
      return {x:minX,y:minY,w:Math.max(...items.map(n=>n.x+n.w))-minX,h:Math.max(...items.map(n=>n.y+n.h))-minY};
    };
    const fitted = () => {
      const box=bounds(nodes.filter(node=>!node.expandedGroup).map(node=>node.id));
      if(!box)return {scale:1,x:0,y:0};
      const scale=Math.max(.01,Math.min((viewport.clientWidth-80)/box.w,(viewport.clientHeight-72)/box.h,1));
      return {scale,x:(viewport.clientWidth-box.w*scale)/2-box.x*scale,y:(viewport.clientHeight-box.h*scale)/2-box.y*scale};
    };
    const initialCamera = () => {
      if(!adapter||adapter.fitOnEnter)return fitted();
      const rail=!adapter?184:0,available=Math.max(120,viewport.clientWidth-rail);
      const scale=Math.max(.35,Math.min(.82,(available-48)/rendered.sceneWidth));
      return {scale,x:rail+(available-rendered.sceneWidth*scale)/2,y:16};
    };
    function fit() {if(viewport.clientWidth&&viewport.clientHeight)onView({camera:fitted()});}
    function paintCamera() {
      if(!camera)return;
      scene.style.transform=`translate(${camera.x}px, ${camera.y}px) scale(${camera.scale})`;
      // Shared compact-density rules hide secondary labels, retaining original node colors/shapes.
      scene.dataset.density=camera.scale<.8?'compact':'comfortable';
      labels.dataset.density=scene.dataset.density;
      paintLabels();
      host.querySelector('[data-viz2-graph-scale]').textContent=`${Math.round(camera.scale*100)}%`;
    }
    function focusRelated(focus) {
      const box=bounds(focus.related);if(!box||!camera)return;
      // Fit remains explicit. On a later cross-view entry, frame only the mapped objects.
      const scale=Math.max(.01,Math.min((viewport.clientWidth-80)/(box.w+80),(viewport.clientHeight-80)/(box.h+80),.9));
      onView({camera:{scale,x:viewport.clientWidth/2-(box.x+box.w/2)*scale,y:viewport.clientHeight/2-(box.y+box.h/2)*scale}});
    }
    function keepSelectedVisible(node) {
      if(!node||!camera)return false;
      const pad=24, left=camera.x+node.x*camera.scale,right=left+node.w*camera.scale;
      const top=camera.y+node.y*camera.scale,bottom=top+node.h*camera.scale;
      if(left>=pad&&right<=viewport.clientWidth-pad&&top>=pad&&bottom<=viewport.clientHeight-pad)return false;
      onView({camera:{...camera,x:viewport.clientWidth/2-(node.x+node.w/2)*camera.scale,y:viewport.clientHeight/2-(node.y+node.h/2)*camera.scale}});
      return true;
    }
    function inspectorHtml(node) {
      const sections=global.PTO_COMPUTATION_PRESENTATION.inspector(node,{model,qualifiedId:node.id});
      const primaryOrder=node.type==='state'?['语义','状态演化','依赖','源码']:['语义','属性','依赖','源码'];
      const primary=primaryOrder.map(title=>sections.find(section=>section.title===title)).filter(Boolean);
      const advanced=sections.filter(section=>!primary.includes(section));
      const sectionHtml=section=>`<section><h3>${escape(section.title)}</h3><dl>${section.rows.map(row=>`<dt>${escape(row.label)}</dt><dd>${escape(row.value)}</dd>`).join('')}</dl></section>`;
      let instances='';
      if(view.instancesNode===node.id){
        const data=staticExecutionData(graph),root=node.id.split('/')[0];
        if(data&&['qk','pv'].includes(root))instances=`<section><h3>Page Instances · ${data.pages}</h3><p>Task ${data.task} · Per AIC / Task · 源码静态推导；物理 Cache Page ID 未知。</p>${data.stacksDetail.map(stack=>`<div class="viz2-page-instance-group"><b>S${stack.id}</b><div>${stack.pages.map((page,index)=>`<button class="btn btn-ghost btn-sm" data-viz2-instance-page="${page.id}">P${page.id}</button>`).join('')}</div></div>`).join('')}${data.stacksDetail.flatMap(stack=>stack.pages.map((page,index)=>({page,stack,index}))).filter(item=>item.page.id===view.instancePage).map(({page,stack,index})=>`<p>P${page.id} · S${stack.id} · ${page.validTokens}/${data.pageTokens} Tokens · ${root==='pv'&&index?'pl.matmul_acc':'pl.matmul'}<br>输入：${root==='qk'?'Q Tile + K Cache Page':'Probability + V Cache Page'}<br>输出：${root==='qk'?'FP32 Scores Page':'FP32 Partial Context'}</p>`).join('')}</section>`;
        else instances='<section><h3>静态逻辑实例</h3><p>执行单位与源码依据见下方；不同统计单位不能直接相加为硬件成本。</p></section>';
      }
      const executionEntry=staticExecutionData(graph)?.nodes[node.id],root=node.id.split('/')[0];
      const scope=executionEntry?`<section><h3>静态 Execution Weight</h3><p>×${executionEntry.count} · ${['qk','pv'].includes(root)?'Per Page / Per AIC / Per Task':root==='output-norm'?'Per AIV Lane / Per Task':'Per Stack / Per AIV Lane / Per Task'}${root==='pv'?' · 每 Stack 首 Page matmul，其余 matmul_acc；合计为 Page 贡献。':''}</p><p>依据：${escape(executionEntry.formula||executionEntry.label)}。不同统计单位不可直接比较为硬件执行成本。</p></section>`:'';
      return instances+scope+primary.map(sectionHtml).join('')+(advanced.length?`<details><summary>更多属性</summary>${advanced.map(sectionHtml).join('')}</details>`:'');
    }
    function relatedPath(id,oneHop=false){
      if(!id)return [];
      const found=new Set([id]),frontier=[id];
      while(frontier.length){
        const current=frontier.shift();
        graph.edges.filter(edge=>edge.type==='data'&&(edge.s===current||edge.t===current)).forEach(edge=>{
          const next=edge.s===current?edge.t:edge.s;
          if(!found.has(next)){found.add(next);if(!oneHop)frontier.push(next);}
        });
      }
      return [...found];
    }
    function paintFocus(focus){
      const related=adapter?focus.related:hoverNodeId?relatedPath(hoverNodeId,true):focus.nodeId?relatedPath(focus.nodeId):focus.related;
      scene.dataset.focused=String(!adapter&&!!(hoverNodeId||focus.nodeId));
      scene.querySelectorAll('[data-pypa-node]').forEach(el=>{
        const selected=el.dataset.pypaNode===focus.nodeId,isRelated=related.includes(el.dataset.pypaNode);
        el.classList.toggle('is-selected',selected);el.classList.toggle('is-viz2-related',isRelated&&!selected);el.setAttribute('aria-pressed',String(selected));
        if(adapter){const role=focus.nodeRoles?.[el.dataset.pypaNode];if(role)el.dataset.stateRole=role;else delete el.dataset.stateRole;}
      });
      nodeLabels.forEach(({node,label})=>label.classList.toggle('is-viz2-related',related.includes(node.id)));
      scene.querySelectorAll('[data-pypa-edge]').forEach(el=>{
        const edge=graph.edges.find(edge=>edge.id===el.dataset.pypaEdge);
        el.classList.toggle('is-viz2-related',!!edge&&(view.edgeId===edge.id||focus.edgeId===edge.id||related.includes(edge.s)&&related.includes(edge.t)));
        if(!adapter&&['state','feedback'].includes(edge?.type))el.style.pointerEvents='stroke';
        if(adapter)el.style.pointerEvents='stroke';
        if(adapter){const role=focus.edgeRoles?.[el.dataset.pypaEdge];if(role)el.dataset.stateRole=role;else delete el.dataset.stateRole;}
      });
      scene.querySelectorAll('[data-pypa-sync],[data-pypa-sync-path]').forEach(el=>{
        const id=el.dataset.pypaSync||el.dataset.pypaSyncPath,selected=id===view.syncId;
        el.style.display=view.syncId&&!selected?'none':'';
        if(el.dataset.pypaSync)el.setAttribute('aria-pressed',String(selected));
      });
    }
    function update(nextSelection,nextView,options={}) {
      const entering=!!options.active&&!active;
      const nextModel=adapter?.getModel() || model;
      if(nextModel!==model){model=nextModel;layoutKey='';executionKey='';first=true;}
      selection=nextSelection;view=nextView;active=!!options.active;camera=view.camera;
      const nextLayers={data:!!adapter,shape:false,precision:false,memory:false,control:false,sync:false,execution:true,...view.overlays};
      const nextExpanded=new Set(view.expanded||[]),nextKey=JSON.stringify([{...nextLayers,execution:undefined},[...nextExpanded],selection?.task,global.PtoPagedAttentionContext?.scenario?.seqLen]);
      if(layers.execution!==nextLayers.execution)executionKey='';
      layers=nextLayers;scene.dataset.overlayExecution=String(layers.execution);
      if(nextKey!==layoutKey){layoutKey=nextKey;layers=nextLayers;expanded=nextExpanded;rebuild();inspectorKey='';}
      host.querySelectorAll('[data-viz2-graph-block]').forEach(button=>{
        const selected=Number(button.dataset.viz2GraphBlock)===selection?.block;
        button.classList.toggle('is-selected',selected);button.setAttribute('aria-pressed',String(selected));
      });
      host.querySelectorAll('[data-viz2-graph-layer]').forEach(b=>{const enabled=layers[b.dataset.viz2GraphLayer];b.setAttribute('aria-pressed',String(enabled));b.classList.toggle('is-selected',enabled);});
      const countKey=JSON.stringify([layoutKey,selection?.task,global.PtoPagedAttentionContext?.scenario]);if(countKey!==executionKey){executionKey=countKey;paintExecution();}
      const objectKey=JSON.stringify(selection ? [selection.kind,selection.block,selection.task,selection.lane] : null);
      const changed=lastObjectKey!==undefined&&objectKey!==lastObjectKey;lastObjectKey=objectKey;
      if(!adapter&&changed&&!selection?.graphNodeId&&view.nodeId){onView({nodeId:null,inspectorOpen:false});return;}
      const focus=adapter?.resolveFocus?.(graph,selection,view) || resolveFocus({...model,nodes},selection,view.nodeId);focus.related=nodes.filter(n=>focus.related.includes(n.id)||focus.related.includes(n.id.split('/')[0])).map(n=>n.id);
      const key=JSON.stringify([focus,objectKey]);
      if(options.restore){pendingFocus=false;lastFocusKey=key;}
      if(options.enter){tooltip.hidden=true;pendingFocus=true;}
      if(key!==lastFocusKey){lastFocusKey=key;pendingFocus=!skipFocusOnce;}
      skipFocusOnce=false;
      paintFocus(focus);
      const node=nodeMap.get(focus.nodeId),edge=graph.edges.find(e=>e.id===view.edgeId&&['state','feedback'].includes(e.type));
      const sync=model.synchronizations?.find(s=>s.id===view.syncId),show=!!(node||edge||sync||adapter?.hasInspector?.())&&view.inspectorOpen===true,inspectorWasHidden=inspector.hidden;inspector.hidden=!show;
      if(options.restore)lastSize={width:viewport.clientWidth,height:viewport.clientHeight};
      const ikey=show?(adapter?JSON.stringify([selection,view.nodeId,view.syncId]):edge?'edge:'+edge.id:sync?'sync:'+sync.id:JSON.stringify([node.id,view.instancesNode,view.instancePage])):'';
      if(ikey!==inspectorKey){const detailBody=host.querySelector('[data-viz2-graph-detail]'),previousTitle=host.querySelector('[data-viz2-graph-title]').textContent,openDetails=[...detailBody.querySelectorAll('details[open]')].map(el=>el.querySelector('summary')?.textContent);inspectorKey=ikey;if(adapter&&show){const detail=adapter.inspector(node,sync,graph);host.querySelector('[data-viz2-graph-title]').textContent=detail.title;detailBody.innerHTML=detail.html;if(previousTitle===detail.title)detailBody.querySelectorAll('details').forEach(el=>{if(openDetails.includes(el.querySelector('summary')?.textContent))el.open=true;});}else if(edge&&show){host.querySelector('[data-viz2-graph-title]').textContent=edge.type==='feedback'?'跨 Stack 反馈':'状态依赖';detailBody.innerHTML=global.PTO_COMPUTATION_PRESENTATION.inspectorEdge(edge,{model}).map(section=>`<section><h3>${escape(section.title)}</h3><dl>${section.rows.map(row=>`<dt>${escape(row.label)}</dt><dd>${escape(row.value)}</dd>`).join('')}</dl></section>`).join('');}else if(sync&&show){host.querySelector('[data-viz2-graph-title]').textContent=sync.name;detailBody.innerHTML=`<dl>${[['Buffer',sync.buffer],['Dtype',sync.dtype],['Producer',`${sync.producer.coreType} SET · ${sync.producer.pipe}`],['Consumer',`${sync.consumer.coreType} WAIT · ${sync.consumer.pipe}`]].map(([k,v])=>`<dt>${escape(k)}</dt><dd>${escape(v)}</dd>`).join('')}</dl><p>源码静态 READY 关系，不表示运行耗时。</p>`;}else if(node&&show){host.querySelector('[data-viz2-graph-title]').textContent=node.title;detailBody.innerHTML=inspectorHtml(node);}}
      const context=selection?.kind==='task'?`Task ${selection.task} · Batch ${selection.batch} · KV Head ${selection.kvHead}`:selection?.block!=null?`Block ${String(selection.block).padStart(2,'0')}${selection.kind==='lane'?` · AIV Lane ${selection.lane}`:''}`:'完整计算图';
      host.querySelector('[data-viz2-graph-context]').textContent=layers.execution?'线宽 = 静态逻辑传输 / 贡献频次 · 非数据量、带宽或耗时':adapter?.context?.() || context;
      if(!active||!viewport.clientWidth||!viewport.clientHeight)return;
      if(first||(entering&&adapter&&!camera)){first=false;pendingFocus=false;if(!adapter||!camera){onView({camera:initialCamera()});return;}}
      if(!camera){onView({camera:initialCamera()});return;}
      if(pendingFocus){pendingFocus=false;if(!adapter&&focus.related.length){focusRelated(focus);return;}}
      const ensureSelected=show&&node&&(inspectorWasHidden||lastVisibleNodeId!==node.id);
      lastVisibleNodeId=show?node?.id:null;
      if(ensureSelected&&keepSelectedVisible(node))return;
      paintCamera();
    }
    function zoom(factor,point={x:viewport.clientWidth/2,y:viewport.clientHeight/2}) {
      if(!camera)return;const scale=Math.max(.01,Math.min(3,camera.scale*factor)),ratio=scale/camera.scale;
      onView({camera:{scale,x:point.x-(point.x-camera.x)*ratio,y:point.y-(point.y-camera.y)*ratio}});
    }
    function toggleExpanded(id, select=false) {
      const before=camera,old=nodeMap.get(id),point=old&&camera?{x:camera.x+(old.x+old.w/2)*camera.scale,y:camera.y+(old.y+old.h/2)*camera.scale}:null;
      const next=new Set(expanded);next.has(id)?next.delete(id):next.add(id);
      if(select)skipFocusOnce=true;
      onView({expanded:[...next],...(select?{nodeId:id,syncId:null,inspectorOpen:true}:{})});
      const node=nodeMap.get(id);
      if(node&&next.has(id)&&node.expandedGroup){
        const fitScale=Math.min((viewport.clientWidth-72)/node.groupWidth,(viewport.clientHeight-72)/node.groupHeight);
        const floor=typeof adapter?.expandedScaleFloor==='function'?adapter.expandedScaleFloor(node):adapter?.expandedScaleFloor;
        const scale=floor?Math.max(floor,Math.min(fitScale,.9)):Math.min(before?.scale||1,fitScale*(adapter?.expandedFitInset?.(node)||1));
        onView({camera:{scale,x:(adapter?.focusExpandedTop&&node.groupWidth*scale>viewport.clientWidth-48?24-node.x*scale:viewport.clientWidth/2-(node.x+node.groupWidth/2)*scale)+(adapter?.expandedXOffset?.(node)||0),
          y:adapter?.focusExpandedTop?24-node.y*scale:viewport.clientHeight/2-(node.y+node.groupHeight/2)*scale}});
      }else if(point&&node)onView({camera:{...before,x:point.x-(node.x+node.w/2)*before.scale,y:point.y-(node.y+node.h/2)*before.scale}});
    }
    host.addEventListener('click',event=>{
      // Graph owns these node events; the legacy page has document-level selectors.
      event.stopPropagation(); tooltip.hidden=true;
      const sourceButton=event.target.closest('[data-viz2-graph-source]');
      if(sourceButton&&adapter?.onSource){adapter.onSource(sourceButton.dataset.viz2GraphSource);return;}
      const layer=event.target.closest('[data-viz2-graph-layer]');
      if(layer){const key=layer.dataset.viz2GraphLayer;onView({overlays:{...layers,[key]:!layers[key]}});return;}
      const instances=event.target.closest('[data-pypa-execution-count],[data-pypa-execution-stack],[data-viz2-execution-instances]');
      if(instances){const id=instances.dataset.pypaExecutionCount||instances.dataset.pypaExecutionStack||instances.dataset.viz2ExecutionInstances;if(adapter)adapter.onInstances?.(id);else onView({nodeId:id,instancesNode:id,inspectorOpen:true});return;}
      const page=event.target.closest('[data-viz2-instance-page]');if(page){onView({instancePage:Number(page.dataset.viz2InstancePage)});return;}
      const block=event.target.closest('[data-viz2-graph-block]');
      if(block){const id=Number(block.dataset.viz2GraphBlock);if(onBlockEnter)onBlockEnter(id);else onSelect({kind:'block',block:id,task:null});return;}
      const toggle=event.target.closest('[data-pypa-expand-toggle]')||(event.detail===0?event.target.closest('[data-pypa-expand]'):null);
      if(toggle){toggleExpanded(toggle.dataset.pypaExpandToggle||toggle.dataset.pypaExpand,true);return;}
      const sync=event.target.closest('[data-pypa-sync]');if(sync){if(adapter)adapter.onSync(sync.dataset.pypaSync);else onView({syncId:sync.dataset.pypaSync,inspectorOpen:true});return;}
      if(event.target.closest('[data-viz2-graph-fit]')){fit();return;}
      if(event.target.closest('[data-viz2-graph-close]')){onView({inspectorOpen:false});return;}
      const zoomButton=event.target.closest('[data-viz2-graph-zoom]');if(zoomButton){zoom(Number(zoomButton.dataset.viz2GraphZoom));return;}
      if(suppressClick){suppressClick=false;return;}
      const edge=event.target.closest('[data-pypa-edge],[data-pypa-edge-hit],[data-pypa-edge-label]');
      if(edge){const id=edge.dataset.pypaEdge||edge.dataset.pypaEdgeHit||edge.dataset.pypaEdgeLabel;if(adapter){adapter.onEdge(id,graph);return;}if(['state','feedback'].includes(graph.edges.find(item=>item.id===id)?.type)){skipFocusOnce=true;onView({edgeId:id,nodeId:null,syncId:null,inspectorOpen:true});return;}}
      const target=event.target.closest('[data-pypa-node], [data-viz2-graph-label]');if(!target)return;
      const id=target.dataset.pypaNode||target.dataset.viz2GraphLabel;
      if(event.detail===1)lastNodeClick={id,time:performance.now(),view:{camera:view.camera,nodeId:view.nodeId,inspectorOpen:view.inspectorOpen}};
      if(adapter){adapter.onNode(id);return;}
      skipFocusOnce=true;onView({nodeId:id,edgeId:null,syncId:null,instancesNode:null,instancePage:null,inspectorOpen:true});
      if(selection?.block!=null)onSelect({...selection,graphNodeId:id});
    },{signal:events.signal});
    host.addEventListener('keydown',event=>{if(!['Enter',' '].includes(event.key))return;const target=event.target.closest('[data-pypa-execution-count],[data-pypa-execution-stack]');if(target){event.preventDefault();event.stopPropagation();target.click();}},{signal:events.signal});
    viewport.addEventListener('wheel',event=>{
      event.preventDefault();tooltip.hidden=true;const rect=viewport.getBoundingClientRect(),unit=event.deltaMode===1?16:event.deltaMode===2?rect.height:1;
      zoom(Math.exp(Math.max(-.22,Math.min(.22,-event.deltaY*unit*.003))),{x:event.clientX-rect.left,y:event.clientY-rect.top});
    },{passive:false,signal:events.signal});
    viewport.addEventListener('pointerdown',event=>{if(event.button!==0)return;suppressClick=false;drag={id:event.pointerId,x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY,moved:false};},{signal:events.signal});
    viewport.addEventListener('pointermove',event=>{
      if(drag){if(!drag.moved&&Math.hypot(event.clientX-drag.startX,event.clientY-drag.startY)<=4)return;drag.moved=true;viewport.setPointerCapture(drag.id);tooltip.hidden=true;const c=camera||fitted();onView({camera:{...c,x:c.x+event.clientX-drag.x,y:c.y+event.clientY-drag.y}});drag.x=event.clientX;drag.y=event.clientY;return;}
      const target=event.target.closest('[data-pypa-node], [data-viz2-graph-label]'),node=target?nodeMap.get(target.dataset.pypaNode||target.dataset.viz2GraphLabel):null;tooltip.hidden=!node;
      if(!adapter&&hoverNodeId!==node?.id){hoverNodeId=node?.id||null;paintFocus(resolveFocus({...model,nodes},selection,view.nodeId));}
      if(adapter){const el=event.target.closest('[data-pypa-edge],[data-pypa-edge-label],[data-pypa-sync]');adapter.onHover?.(node?.id,el?.dataset.pypaEdge||el?.dataset.pypaEdgeLabel,el?.dataset.pypaSync,graph);paintFocus(adapter.resolveFocus(graph,selection,view));if(!node&&el){tooltip.hidden=false;tooltip.textContent=adapter.tooltip?.(null,el.dataset.pypaEdge||el.dataset.pypaEdgeLabel,el.dataset.pypaSync,graph)||'';const r=viewport.getBoundingClientRect();tooltip.style.left=`${Math.max(8,Math.min(event.clientX-r.left+14,r.width-tooltip.offsetWidth-8))}px`;tooltip.style.top=`${Math.max(8,Math.min(event.clientY-r.top+16,r.height-tooltip.offsetHeight-8))}px`;}}
      if(node){tooltip.textContent=adapter?.tooltip?.(node.id,null,null,graph)||`${node.title} · ${node.detail.length>100?node.detail.slice(0,100)+'…':node.detail}`;const rect=viewport.getBoundingClientRect();tooltip.style.left=`${Math.max(8,Math.min(event.clientX-rect.left+14,rect.width-tooltip.offsetWidth-8))}px`;tooltip.style.top=`${Math.max(8,Math.min(event.clientY-rect.top+16,rect.height-tooltip.offsetHeight-8))}px`;}
    },{signal:events.signal});
    const endDrag=()=>{if(drag?.moved)suppressClick=true;drag=null;};['pointerup','pointercancel','lostpointercapture'].forEach(type=>viewport.addEventListener(type,endDrag,{signal:events.signal}));
    viewport.addEventListener('pointerleave',()=>{tooltip.hidden=true;hoverNodeId=null;adapter?.onHover?.(null,null,null,graph);paintFocus(adapter?adapter.resolveFocus(graph,selection,view):resolveFocus({...model,nodes},selection,view.nodeId));},{signal:events.signal});
    viewport.addEventListener('dblclick',event=>{if(adapter){if(!event.target.closest('[data-pypa-node],[data-viz2-graph-label]'))fit();return;}const target=event.target.closest('[data-pypa-node],[data-viz2-graph-label]');const id=lastNodeClick&&performance.now()-lastNodeClick.time<700?lastNodeClick.id:target?.dataset.pypaNode||target?.dataset.viz2GraphLabel;const region=model.groups.find(g=>g.id==='paged-attention');if(region?.members.includes(id?.split('/')[0])||event.target.closest('[data-pypa-group="paged-attention"]')){event.stopPropagation();onEnter?.(lastNodeClick?.view);}else if(!target)fit();},{signal:events.signal});
    viewport.addEventListener('keydown',event=>{if(event.key==='Escape'){onView({inspectorOpen:false});viewport.focus();}else if(['+','=','-','0'].includes(event.key)){event.preventDefault();if(event.key==='0')fit();else zoom(event.key === '-' ? .8 : 1.25);}},{signal:events.signal});
    const resizeObserver=new ResizeObserver(()=>{
      if(!active||!viewport.clientWidth||!viewport.clientHeight)return;
      const width=viewport.clientWidth,height=viewport.clientHeight;
      if(first||!camera){first=false;onView({camera:initialCamera()});}
      else if(lastSize&&(lastSize.width!==width||lastSize.height!==height)){
        onView({camera:{...camera,x:camera.x+(width-lastSize.width)/2,y:camera.y+(height-lastSize.height)/2}});
      }
      lastSize={width,height};
    });resizeObserver.observe(viewport);
    return {update,toggleExpanded,destroy:()=>{events.abort();resizeObserver.disconnect();labels?.remove();},getTransitionRect:()=>viewport.getBoundingClientRect(),getBlockTransitionSnapshot:()=>({
      clip:viewport.getBoundingClientRect(),
      blocks:[...host.querySelectorAll('[data-viz2-graph-block]')].filter(button=>button.getBoundingClientRect().width&&button.closest('.viz2-graph-block-rail')?.style.visibility!=='hidden').map(button=>({
        id:Number(button.dataset.viz2GraphBlock),stage:button.closest('[data-viz2-stage]')?.dataset.viz2Stage,
        rect:button.getBoundingClientRect()
      }))
    }),getAttentionRect:()=>{const ids=model.groups.find(g=>g.id==='paged-attention')?.members||[];const boxes=ids.map(id=>scene.querySelector(`[data-pypa-node="${id}"]`)?.getBoundingClientRect()).filter(Boolean);if(!boxes.length)return viewport.getBoundingClientRect();const left=Math.min(...boxes.map(r=>r.left)),top=Math.min(...boxes.map(r=>r.top));return {left,top,width:Math.max(...boxes.map(r=>r.right))-left,height:Math.max(...boxes.map(r=>r.bottom))-top};},hideTooltip:()=>{tooltip.hidden=true;},get graph(){return graph;},get model(){return model;}};
  }
  global.PtoVisualization2Graph=Object.freeze({mount,resolveFocus});
})(window);
