(function attachPtoModelParallelRankDeck(global){
  'use strict';

  const AXES=['tp','pp','ep','edp'];
  const GROUPING_ORDER=['pp','tp','ep','edp'];
  const SCENE_MODES=new Set(['packed','exploded','rank']);
  const TOPOLOGY_MODES=new Set(['standard','pp','tp','ep','edp','replica']);
  const MODE_TO_AXIS={pp:'pp',tp:'tp',ep:'ep',edp:'edp',replica:'edp'};
  const AXIS_TO_MODE={pp:'pp',tp:'tp',ep:'ep',edp:'replica'};
  const THEMES=new Set(['dark','light']);
  const ISOMETRIC_POSE={rx:-35.264,ry:45};
  const DEFAULT_TOPOLOGY={
    worldSize:128,
    dimensions:{tp:2,pp:4,cp:1,ep:8,edp:2},
    rankOrder:['tp','pp','cp','ep','edp'],
    sourceLevel:'demo'
  };
  const TP_SHARDED=new Set([
    'embedding','q_a_proj','kv_a_proj','q_b_proj','kv_b_proj','query_tensor','key_tensor','attention_core',
    'o_proj','dense_gate_up','dense_down','lm_head','mtp_eh_proj','mtp_decoder_layer','mtp_shared_head'
  ]);
  const TP_COLLECTIVE=new Set(['attn_all_gather','attn_reduce_scatter','moe_all_gather','moe_reduce_scatter','logits_allgather']);
  const EP_SHARDED=new Set(['expert_pool','a2a_dispatch','a2a_combine']);
  const OP_COLORS={
    embedding:'#14b8a6',norm:'#38bdf8',attention:'#3b82f6',linear:'#4f46e5',head:'#7c3aed',
    mlp:'#a855f7',act:'#8b5cf6',gate:'#f59e0b',moe:'#ea580c',comm:'#06b6d4',decoder:'#0d9488',
    input:'#a855f7',output:'#38bdf8',parameter:'#3b82f6',state:'#8b5cf6','mhc-state':'#8b5cf6',add:'#94a3b8'
  };
  const EDGE_COLORS={activation:'#94a3b8',comm:'#22d3ee',communication:'#22d3ee',parameter:'#60a5fa',residual:'#c084fc','state-spine':'#a78bfa','model-spine-depth':'#14b8a6','model-spine-front':'#14b8a6'};
  const PP_COLORS=['#38bdf8','#a855f7','#14b8a6','#f59e0b'];
  const TP_COLORS=['#38bdf8','#a855f7'];
  const EP_COLORS=['#f97316','#fb7185','#f59e0b','#eab308','#84cc16','#14b8a6','#06b6d4','#8b5cf6'];
  const EDP_COLORS=['#8b5cf6','#ec4899'];
  const GHOST_COLOR='#8ecbff';
  const TOPOLOGY_COLORS={tp:'#38bdf8',pp:'#38bdf8',ep:'#f97316',edp:'#8b5cf6'};
  // A Layer is a YZ plane whose deck depth runs on X. PP stages use that same
  // X axis, so their four Layer ranges read as one continuous model in side view.
  const RANK_SIZE={x:6.2,y:13.8,z:8.6};
  const RANK_STEP={x:7.6,y:15.8,z:10.5};
  const LAYER_GAP=.43;
  const SOURCE_SCALE=.0108;
  const NODE_DEPTH=.12;
  const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
  const esc=(value)=>String(value??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const qs=(value,root=document)=>typeof value==='string'?root.querySelector(value):value;
  const copy=(value)=>JSON.parse(JSON.stringify(value));
  const finiteInt=(value,label,min=1)=>{const next=Number(value);if(!Number.isInteger(next)||next<min)throw new TypeError(`${label} must be an integer >= ${min}`);return next;};
  const normalizeGroupingAxes=(value)=>{const selected=new Set((Array.isArray(value)?value:String(value||'').split(',')).map(item=>MODE_TO_AXIS[String(item).trim().toLowerCase()]).filter(axis=>AXES.includes(axis)));return GROUPING_ORDER.filter(axis=>selected.has(axis));};
  const toggleGroupingAxis=(axesInput,mode)=>{const axes=normalizeGroupingAxes(axesInput),axis=MODE_TO_AXIS[String(mode||'').toLowerCase()];if(!axis)return axes;return normalizeGroupingAxes(axes.includes(axis)?axes.filter(item=>item!==axis):[...axes,axis]);};
  const topologyModeForAxes=(axes)=>axes.length?AXIS_TO_MODE[axes[axes.length-1]]:'standard';

  function normalizeTopology(input={}){
    const dimensions={...DEFAULT_TOPOLOGY.dimensions,...(input.dimensions||{})};
    Object.keys(dimensions).forEach(key=>{dimensions[key]=finiteInt(dimensions[key],`dimensions.${key}`);});
    const computed=dimensions.tp*dimensions.pp*dimensions.cp*dimensions.ep*dimensions.edp;
    const worldSize=input.worldSize==null?computed:finiteInt(input.worldSize,'worldSize');
    if(worldSize!==computed)throw new RangeError(`worldSize ${worldSize} does not equal TP×PP×CP×EP×EDP (${computed})`);
    return{...DEFAULT_TOPOLOGY,...input,worldSize,dimensions,sourceLevel:input.sourceLevel||'demo'};
  }

  function enumerateRanks(topologyInput={}){
    const topology=normalizeTopology(topologyInput),d=topology.dimensions,coordinates=[];
    for(let edp=0;edp<d.edp;edp+=1)for(let ep=0;ep<d.ep;ep+=1)for(let cp=0;cp<d.cp;cp+=1)for(let pp=0;pp<d.pp;pp+=1)for(let tp=0;tp<d.tp;tp+=1){
      const rank=(((((edp*d.ep)+ep)*d.cp+cp)*d.pp+pp)*d.tp+tp);
      coordinates.push({rank,tp,pp,cp,ep,edp});
    }
    if(coordinates.length!==topology.worldSize)throw new Error('Rank enumeration did not produce worldSize coordinates');
    return coordinates;
  }

  function groupKey(coordinate,axis){
    return ['tp','pp','cp','ep','edp'].filter(key=>key!==axis).map(key=>`${key}${coordinate[key]}`).join('-');
  }

  function buildGroups(coordinates){
    const groups=[];
    AXES.forEach(axis=>{
      const buckets=new Map();
      coordinates.forEach(coordinate=>{const key=groupKey(coordinate,axis);if(!buckets.has(key))buckets.set(key,[]);buckets.get(key).push(coordinate);});
      buckets.forEach((members,key)=>groups.push({
        id:`${axis}:${key}`,axis,label:`${axis.toUpperCase()} group · ${key}`,
        ranks:members.sort((a,b)=>a[axis]-b[axis]).map(item=>item.rank),sourceLevel:'derived'
      }));
    });
    return groups;
  }

  function partitionModel(modelInput={},topologyInput=DEFAULT_TOPOLOGY,rules={}){
    const topology=normalizeTopology(topologyInput),model={...modelInput},d=topology.dimensions;
    const layerCount=finiteInt(model.layerCount,'model.layerCount');
    const stageRanges=(rules.stageRanges||model.stageRanges||[]).map(range=>range.map(Number));
    if(stageRanges.length!==d.pp)throw new RangeError(`PP ${d.pp} requires ${d.pp} explicit stageRanges`);
    const covered=[];
    stageRanges.forEach(([lo,hi],stage)=>{
      if(!Number.isInteger(lo)||!Number.isInteger(hi)||lo<0||hi<lo||hi>=layerCount)throw new RangeError(`Invalid PP${stage} range ${lo}-${hi}`);
      for(let layer=lo;layer<=hi;layer+=1)covered.push(layer);
    });
    if(covered.length!==layerCount||new Set(covered).size!==layerCount||Math.min(...covered)!==0||Math.max(...covered)!==layerCount-1)throw new Error('PP stageRanges must cover every physical Layer exactly once');
    const routedExperts=finiteInt(model.routedExperts||256,'model.routedExperts');
    if(routedExperts%d.ep!==0)throw new RangeError(`routedExperts ${routedExperts} must be divisible by EP ${d.ep}`);
    const expertsPerRank=routedExperts/d.ep,coordinates=enumerateRanks(topology),groups=buildGroups(coordinates);
    const groupByAxisRank=Object.fromEntries(AXES.map(axis=>[axis,new Map()]));
    groups.forEach(group=>group.ranks.forEach(rank=>groupByAxisRank[group.axis].get(rank)?.push(group)||groupByAxisRank[group.axis].set(rank,[group])));
    const manifests=coordinates.map(coordinate=>{
      const [lo,hi]=stageRanges[coordinate.pp],expertStart=coordinate.ep*expertsPerRank,expertEnd=expertStart+expertsPerRank-1;
      const staticRoles=[];if(coordinate.pp===0)staticRoles.push('input');if(coordinate.pp===d.pp-1)staticRoles.push('output','mtp');
      const nodeOwnership={};
      TP_SHARDED.forEach(id=>{nodeOwnership[id]={kind:'sharded',axis:'tp',index:coordinate.tp,count:d.tp};});
      TP_COLLECTIVE.forEach(id=>{nodeOwnership[id]={kind:'sharded',axis:'tp',index:coordinate.tp,count:d.tp};});
      EP_SHARDED.forEach(id=>{nodeOwnership[id]={kind:'sharded',axis:'ep',index:coordinate.ep,count:d.ep};});
      const communicationGroups={};
      AXES.forEach(axis=>{communicationGroups[axis]=(groupByAxisRank[axis].get(coordinate.rank)||[]).map(group=>group.id);});
      return{
        rank:coordinate.rank,coordinate:{...coordinate},modelId:model.id||'openpangu-flash',
        layerSegments:[{start:lo,end:hi,stage:coordinate.pp}],staticRoles,nodeOwnership,
        expertOwnership:{kind:'sharded',axis:'ep',index:coordinate.ep,count:d.ep,start:expertStart,end:expertEnd,ids:Array.from({length:expertsPerRank},(_,index)=>expertStart+index),sourceLevel:'demo'},
        communicationGroups,
        payloadSignature:[model.id||'openpangu-flash',`pp${coordinate.pp}`,`tp${coordinate.tp}`,`cp${coordinate.cp}`,`ep${coordinate.ep}`,`L${lo}-${hi}`,`E${expertStart}-${expertEnd}`].join('|'),
        sourceLevel:topology.sourceLevel||'demo'
      };
    });
    return{
      model,topology,coordinates,manifests,groups,
      manifestByRank:new Map(manifests.map(manifest=>[manifest.rank,manifest])),
      groupById:new Map(groups.map(group=>[group.id,group])),groupByAxisRank,
      stats:{rankVolumes:coordinates.length,layerInstances:manifests.reduce((sum,item)=>sum+item.layerSegments.reduce((count,segment)=>count+segment.end-segment.start+1,0),0),staticInstances:manifests.reduce((sum,item)=>sum+(item.staticRoles.includes('input')?1:0)+(item.staticRoles.includes('output')?1:0),0)}
    };
  }

  function ownershipForNode(nodeId,manifest){return manifest.nodeOwnership[nodeId]||{kind:'replicated',replicaAxis:'edp'};}

  function layoutPosition(coordinate,topologyInput=DEFAULT_TOPOLOGY,mode='exploded',topologyMode='standard'){
    const topology=normalizeTopology(topologyInput),d=topology.dimensions,multiplier=mode==='exploded'?1.18:1,rep=coordinate.edp*d.ep+coordinate.ep,cTp=(d.tp-1)/2,cPp=(d.pp-1)/2,cEp=(d.ep-1)/2,cEdp=(d.edp-1)/2,cRep=(d.ep*d.edp-1)/2;
    if(topologyMode==='pp')return{x:(coordinate.pp-cPp)*RANK_STEP.x*multiplier,y:(cTp-coordinate.tp)*RANK_STEP.y*.78,z:(cRep-rep)*RANK_STEP.z*multiplier};
    if(topologyMode==='tp')return{x:(coordinate.pp-cPp)*RANK_STEP.x*multiplier,y:(cTp-coordinate.tp)*RANK_STEP.y*1.42,z:(cRep-rep)*RANK_STEP.z*multiplier};
    if(topologyMode==='ep')return{x:(coordinate.pp-cPp)*RANK_STEP.x*multiplier,y:(coordinate.ep-cEp)*RANK_STEP.y*1.34+(coordinate.tp-cTp)*RANK_STEP.y*.66,z:(cEdp-coordinate.edp)*RANK_STEP.z*2.2};
    if(topologyMode==='replica')return{x:(coordinate.pp-cPp)*RANK_STEP.x*multiplier,y:(coordinate.edp-cEdp)*RANK_STEP.y*1.7+(coordinate.tp-cTp)*RANK_STEP.y*.66,z:(cEp-coordinate.ep)*RANK_STEP.z*multiplier};
    return{x:(coordinate.rank-(topology.worldSize-1)/2)*RANK_STEP.x*multiplier,y:0,z:0};
  }

  function buildHierarchicalRankLayout(manifestsInput=[],axesInput=[],mode='exploded',topologyInput=DEFAULT_TOPOLOGY){
    const manifests=[...manifestsInput],axes=normalizeGroupingAxes(axesInput),topology=normalizeTopology(topologyInput),d=topology.dimensions,multiplier=mode==='exploded'?1.18:1,positions=new Map(),frames=[];
    if(!axes.length)return{positions,frames,width:0,height:0,depth:0};
    const first=axes[0],secondaryAxes=new Set(axes.slice(1)),cTp=(d.tp-1)/2,cPp=(d.pp-1)/2,cEp=(d.ep-1)/2,cEdp=(d.edp-1)/2,cRep=(d.ep*d.edp-1)/2,epFirst=axes.includes('ep');
    manifests.forEach(manifest=>{
      const coordinate=manifest.coordinate,depthIndex=epFirst?coordinate.ep*d.edp+coordinate.edp:coordinate.edp*d.ep+coordinate.ep;let position;
      if(first==='pp')position={x:(coordinate.pp-cPp)*RANK_STEP.x*multiplier,y:(cTp-coordinate.tp)*RANK_STEP.y*.78,z:(cRep-depthIndex)*RANK_STEP.z*multiplier};
      else if(first==='tp')position={x:(coordinate.pp-cPp)*RANK_STEP.x*multiplier,y:(cTp-coordinate.tp)*RANK_STEP.y*1.42,z:(cRep-depthIndex)*RANK_STEP.z*multiplier};
      else if(first==='ep')position={x:(coordinate.pp-cPp)*RANK_STEP.x*multiplier,y:(coordinate.ep-cEp)*RANK_STEP.y*1.34+(coordinate.tp-cTp)*RANK_STEP.y*.66,z:(cEdp-coordinate.edp)*RANK_STEP.z*2.2};
      else position={x:(coordinate.pp-cPp)*RANK_STEP.x*multiplier,y:(coordinate.edp-cEdp)*RANK_STEP.y*1.7+(coordinate.tp-cTp)*RANK_STEP.y*.66,z:(cEp-coordinate.ep)*RANK_STEP.z*multiplier};
      if(secondaryAxes.has('tp')){
        if(first==='pp')position.y+=(cTp-coordinate.tp)*RANK_STEP.y*.62*multiplier;
        else position.y+=(cTp-coordinate.tp)*RANK_STEP.y*.62*multiplier;
      }
      if(secondaryAxes.has('edp'))position.z+=(cEdp-coordinate.edp)*RANK_STEP.z*(epFirst?.72:1.5)*multiplier;
      positions.set(manifest.rank,position);
    });
    axes.forEach((axis,level)=>{
      const prefix=axes.slice(0,level+1),buckets=new Map();
      manifests.forEach(manifest=>{const key=prefix.map(keyAxis=>`${keyAxis}:${manifest.coordinate[keyAxis]}`).join('|');if(!buckets.has(key))buckets.set(key,[]);buckets.get(key).push(manifest);});
      buckets.forEach(members=>{
        const bounds={minX:Infinity,minY:Infinity,minZ:Infinity,maxX:-Infinity,maxY:-Infinity,maxZ:-Infinity};
        members.forEach(manifest=>{const point=positions.get(manifest.rank);bounds.minX=Math.min(bounds.minX,point.x-RANK_SIZE.x/2);bounds.maxX=Math.max(bounds.maxX,point.x+RANK_SIZE.x/2);bounds.minY=Math.min(bounds.minY,point.y-RANK_SIZE.y/2);bounds.maxY=Math.max(bounds.maxY,point.y+RANK_SIZE.y/2);bounds.minZ=Math.min(bounds.minZ,point.z-RANK_SIZE.z/2);bounds.maxZ=Math.max(bounds.maxZ,point.z+RANK_SIZE.z/2);});
        const padding=1.5+level*.35,path=prefix.map(pathAxis=>({axis:pathAxis,value:members[0].coordinate[pathAxis]}));frames.push({axis,value:members[0].coordinate[axis],level,path,ranks:members.map(item=>item.rank),center:{x:(bounds.minX+bounds.maxX)/2,y:(bounds.minY+bounds.maxY)/2,z:(bounds.minZ+bounds.maxZ)/2},size:{x:bounds.maxX-bounds.minX+padding,y:bounds.maxY-bounds.minY+padding,z:bounds.maxZ-bounds.minZ+padding}});
      });
    });
    const bounds={minX:Infinity,minY:Infinity,minZ:Infinity,maxX:-Infinity,maxY:-Infinity,maxZ:-Infinity};positions.forEach(point=>{bounds.minX=Math.min(bounds.minX,point.x-RANK_SIZE.x/2);bounds.maxX=Math.max(bounds.maxX,point.x+RANK_SIZE.x/2);bounds.minY=Math.min(bounds.minY,point.y-RANK_SIZE.y/2);bounds.maxY=Math.max(bounds.maxY,point.y+RANK_SIZE.y/2);bounds.minZ=Math.min(bounds.minZ,point.z-RANK_SIZE.z/2);bounds.maxZ=Math.max(bounds.maxZ,point.z+RANK_SIZE.z/2);});
    return{positions,frames,width:bounds.maxX-bounds.minX,height:bounds.maxY-bounds.minY,depth:bounds.maxZ-bounds.minZ};
  }

  function numericStyle(element,key){const value=Number.parseFloat(element.style[key]);return Number.isFinite(value)?value:0;}

  function payloadPoint(template,x,y,depth){return{x:depth,y:(template.centerY-y)*SOURCE_SCALE,z:(x-template.centerX)*SOURCE_SCALE};}

  function sampleSvgPath(pathData,curveSteps=5){
    const tokens=String(pathData||'').match(/[A-Za-z]|-?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?/gi)||[],points=[];
    let index=0,command='',cursor={x:0,y:0};
    const number=()=>Number(tokens[index++]);
    while(index<tokens.length){
      if(/[A-Za-z]/.test(tokens[index]))command=tokens[index++].toUpperCase();
      if(command==='M'||command==='L'){
        if(index+1>=tokens.length)break;
        cursor={x:number(),y:number()};points.push({...cursor});if(command==='M')command='L';
      }else if(command==='C'){
        if(index+5>=tokens.length)break;
        const p0={...cursor},p1={x:number(),y:number()},p2={x:number(),y:number()},p3={x:number(),y:number()};
        for(let step=1;step<=curveSteps;step+=1){const t=step/curveSteps,u=1-t;points.push({x:u*u*u*p0.x+3*u*u*t*p1.x+3*u*t*t*p2.x+t*t*t*p3.x,y:u*u*u*p0.y+3*u*u*t*p1.y+3*u*t*t*p2.y+t*t*t*p3.y});}
        cursor=p3;
      }else break;
    }
    return points;
  }

  function parseTemplate(element,{key,role,layer=null}){
    const nodeElements=[...element.querySelectorAll('[data-node]')];
    const nodes=nodeElements.map((node,sourceIndex)=>({
      id:node.dataset.node,label:node.textContent.trim()||node.getAttribute('aria-label')?.split('·')[0].trim()||node.dataset.node,
      op:node.dataset.op||'linear',x:numericStyle(node,'left'),y:numericStyle(node,'top'),w:numericStyle(node,'width'),h:numericStyle(node,'height'),
      sourceIndex,description:node.getAttribute('aria-label')||'',role,layer
    }));
    const clusters=[...element.querySelectorAll('.pto-model-deck__cluster')].map((cluster,sourceIndex)=>({
      id:`cluster-${sourceIndex}`,label:cluster.textContent.trim(),x:numericStyle(cluster,'left'),y:numericStyle(cluster,'top'),w:numericStyle(cluster,'width'),h:numericStyle(cluster,'height'),sourceIndex,role,layer
    }));
    const paths=[...element.querySelectorAll('path[data-source][data-target]')];
    const edges=paths.map((path,sourceIndex)=>({
      id:`edge-${sourceIndex}`,source:path.dataset.source,target:path.dataset.target,kind:path.dataset.kind||path.dataset.edge||'activation',
      points:sampleSvgPath(path.getAttribute('d')),sourceIndex,role,layer
    }));
    const bounds=[...nodes,...clusters].reduce((acc,item)=>({
      minX:Math.min(acc.minX,item.x),minY:Math.min(acc.minY,item.y),maxX:Math.max(acc.maxX,item.x+item.w),maxY:Math.max(acc.maxY,item.y+item.h)
    }),{minX:Infinity,minY:Infinity,maxX:-Infinity,maxY:-Infinity});
    if(!Number.isFinite(bounds.minX))Object.assign(bounds,{minX:0,minY:0,maxX:720,maxY:1180});
    return{key,role,layer,nodes,clusters,edges,bounds,centerX:(bounds.minX+bounds.maxX)/2,centerY:(bounds.minY+bounds.maxY)/2,width:bounds.maxX-bounds.minX,height:bounds.maxY-bounds.minY};
  }

  function buildModelSceneSpec(base,modelInput={}){
    if(!base?.renderLayerScene||!base?.renderStaticScene)throw new TypeError('Canonical model architecture renderer is required');
    const model=base.getConfig({config:modelInput}),layers=[];
    for(let layer=0;layer<model.layerCount;layer+=1){
      const element=base.renderLayerScene(layer,{config:model,context:{interactionMode:'scene-spec'}});
      layers.push(parseTemplate(element,{key:`layer:${layer}`,role:'layer',layer}));
    }
    const input=parseTemplate(base.renderStaticScene('input',{config:model,context:{interactionMode:'scene-spec'}}),{key:'static:input',role:'input'});
    const output=parseTemplate(base.renderStaticScene('output',{config:model,context:{interactionMode:'scene-spec'}}),{key:'static:output',role:'output'});
    const templates=[...layers,input,output],counts=templates.reduce((sum,template)=>({nodes:sum.nodes+template.nodes.length,clusters:sum.clusters+template.clusters.length,edges:sum.edges+template.edges.length}),{nodes:0,clusters:0,edges:0});
    return{id:model.id,label:model.label,model,sourceLevel:'official',layers,static:{input,output},templates,counts,createdAt:new Date().toISOString()};
  }

  function mount(rootInput,options={}){
    const root=qs(rootInput),base=global.PtoModelArchitecture3dDeck,THREE=options.THREE||global.THREE;
    if(!root||!base)return null;
    if(!THREE?.WebGLRenderer)throw new Error('Three.js runtime is required before mounting model-parallel-rank-deck');
    let model=base.getConfig({preset:typeof options.model==='string'?options.model:(options.preset||'openpangu-flash'),config:typeof options.model==='object'?options.model:options.config});
    let topology=normalizeTopology(options.topology||DEFAULT_TOPOLOGY),partitionRules=options.partitionRules||{},plan=partitionModel(model,topology,partitionRules);
    const spec=buildModelSceneSpec(base,model),params=new URLSearchParams(global.location?.search||'');
    const requestedScene=options.initialState?.sceneMode||params.get('scene')||'exploded',requestedTopology=options.initialState?.groupingAxes||options.initialState?.topologyMode||params.get('topology')||'standard',requestedGroupingAxes=normalizeGroupingAxes(requestedTopology);
    const requestedRank=options.initialState?.selectedRank??(params.has('rank')?Number(params.get('rank')):null),requestedLayer=options.initialState?.selectedLayer??(params.has('layer')?Number(params.get('layer')):null);
    const state={
      sceneMode:SCENE_MODES.has(requestedScene)?requestedScene:'packed',groupingAxes:requestedGroupingAxes,topologyMode:topologyModeForAxes(requestedGroupingAxes),
      theme:THEMES.has(options.initialState?.theme)?options.initialState.theme:(params.get('theme')==='light'?'light':document.documentElement.dataset.theme==='light'?'light':'dark'),
      selectedRank:Number.isInteger(requestedRank)&&plan.manifestByRank.has(requestedRank)?requestedRank:null,selectedLayer:Number.isInteger(requestedLayer)?requestedLayer:null,
      selectedNode:null,selectedGroup:null,hoveredRank:null,camera:{...ISOMETRIC_POSE,zoom:1,fitHalfHeight:72,target:{x:0,y:0,z:0}},ready:false
    };
    if(state.sceneMode==='rank'&&state.selectedRank===null)state.sceneMode='packed';
    let destroyed=false,dirty=true,renderRaf=0,layoutTweenRaf=0,pointer=null,hoverRaf=0;

    root.classList.add('pto-model-rank-deck','pto-model-rank-three');root.dataset.sharedPattern='model-parallel-rank-deck';root.dataset.sceneMode=state.sceneMode;root.dataset.topologyMode=state.topologyMode;root.dataset.groupingAxes=state.groupingAxes.join(' ');root.dataset.theme=state.theme;
    const d=topology.dimensions;
    root.innerHTML=`
      <header class="pto-model-rank-deck__header" data-stage-ui>
        <div class="pto-model-rank-deck__identity"><b>${esc(model.label)}</b><span>Three.js / ${plan.topology.worldSize} ranks / 选择切分查看 Layers</span></div>
        <div class="pto-model-rank-deck__grouping-panel" role="group" aria-label="当前范围下继续分组">
          <button type="button" class="pto-model-rank-deck__scope" data-grouping-reset aria-label="恢复全部 ${plan.topology.worldSize} 个 Rank">
            <span>当前范围</span><b data-grouping-scope>全部 · ${plan.topology.worldSize} Ranks</b>
          </button>
          <i aria-hidden="true"></i>
          <span class="pto-model-rank-deck__grouping-prompt">继续按</span>
          <div class="pto-model-rank-deck__grouping-actions">
            <button type="button" data-topology-mode="pp" data-tip="按 Pipeline Stage 分组；再次点击取消这一层。"><b>PP</b><small>×${d.pp}</small></button>
            <button type="button" data-topology-mode="tp" data-tip="在当前分组内继续按 Tensor Shard 分组；再次点击取消。"><b>TP</b><small>×${d.tp}</small></button>
            <button type="button" data-topology-mode="ep" data-tip="在当前分组内继续按 Expert Shard 分组；再次点击取消。"><b>EP</b><small>×${d.ep}</small></button>
            <button type="button" data-topology-mode="replica" data-tip="在当前分组内继续按 Expert-Data Replica 分组；再次点击取消。"><b>EDP</b><small>×${d.edp}</small></button>
            <button type="button" data-dimension-static="cp" data-tip="当前 CP=1，没有产生 Rank 拆分。" disabled><b>CP</b><small>×${d.cp}</small></button>
          </div>
        </div>
        <div class="pto-model-rank-deck__toolbar" aria-label="Rank scene controls">
          <div class="pto-model-rank-deck__control"><button type="button" data-rank-fit>适配</button><span data-rank-zoom>100%</span></div>
          <div class="pto-model-rank-deck__control"><button type="button" data-rank-theme>浅色</button></div>
        </div>
      </header>
      <div class="pto-model-rank-deck__viewport" tabindex="0" aria-label="128 Rank Three.js topology viewport">
        <div class="pto-model-rank-deck__canvas"></div>
        <div class="pto-model-rank-deck__group-badge" aria-live="polite"></div>
        <div class="pto-model-rank-deck__tooltip" role="tooltip"></div>
        <div class="pto-model-rank-deck__loading" role="status"><b>正在编译完整 Layer 到 GPU 批次</b><span>ModelSceneSpec → RankManifest</span><i><em style="width:72%"></em></i></div>
        <div class="pto-model-rank-deck__a11y" aria-live="polite"></div>
      </div>
      <aside class="pto-model-rank-deck__inspector" data-stage-ui aria-live="polite"></aside>`;
    const viewport=root.querySelector('.pto-model-rank-deck__viewport'),canvasHost=root.querySelector('.pto-model-rank-deck__canvas'),inspector=root.querySelector('.pto-model-rank-deck__inspector'),tooltip=root.querySelector('.pto-model-rank-deck__tooltip'),groupBadge=root.querySelector('.pto-model-rank-deck__group-badge'),zoomReadout=root.querySelector('[data-rank-zoom]'),scopeReadout=root.querySelector('[data-grouping-scope]'),loading=root.querySelector('.pto-model-rank-deck__loading'),a11y=root.querySelector('.pto-model-rank-deck__a11y');
    base.applySemanticPalette(root,state.theme);

    const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,powerPreference:'high-performance'});
    renderer.setPixelRatio(Math.min(global.devicePixelRatio||1,1.5));
    if('outputColorSpace'in renderer&&THREE.SRGBColorSpace)renderer.outputColorSpace=THREE.SRGBColorSpace;
    canvasHost.appendChild(renderer.domElement);
    const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(16,1,.1,2400),raycaster=new THREE.Raycaster(),pointerNdc=new THREE.Vector2(),dummy=new THREE.Object3D(),tempColor=new THREE.Color(),tempTint=new THREE.Color();
    scene.add(new THREE.AmbientLight(0xffffff,1.12));const keyLight=new THREE.DirectionalLight(0xffffff,1.6);keyLight.position.set(50,80,90);scene.add(keyLight);const fillLight=new THREE.DirectionalLight(0x7dd3fc,.5);fillLight.position.set(-40,10,-60);scene.add(fillLight);
    const nodeBatches=new Map(),edgeBatches=new Map(),rankPositions=new Map(),shellDisplayPositions=new Map(),sceneIndex={nodes:new Map(),edges:new Map(),layers:new Map(),ranks:new Map()};
    const layerMetas=[],clusterMetas=[],nodeMetasByRank=new Map(),edgeSemanticCounts={};

    function cssColor(name,fallback){return getComputedStyle(root).getPropertyValue(`--pto-model-deck-${name}`).trim()||fallback;}
    function palette(){const values={};Object.entries(OP_COLORS).forEach(([key,fallback])=>{values[key]=cssColor(key,fallback);});return values;}
    let colors=palette();
    const backgroundColor=()=>getComputedStyle(root).getPropertyValue('--background').trim()||(state.theme==='light'?'#f8fafc':'#070b14');

    function templatePlacement(manifest,template,localDepth,role='layer'){
      const placementId=role==='layer'?`rank:${manifest.rank}/layer:${template.layer}`:`rank:${manifest.rank}/static:${role}`,solid=role==='layer'&&template.layer===manifest.layerSegments[0].start;
      const card={id:placementId,rank:manifest.rank,layer:template.layer,role,solid,local:{x:localDepth-.08,y:0,z:0},scale:{x:.035,y:Math.min(RANK_SIZE.y-.7,template.height*SOURCE_SCALE+.38),z:Math.min(RANK_SIZE.z-.7,template.width*SOURCE_SCALE+.38)}};
      layerMetas.push(card);sceneIndex.layers.set(placementId,card);
      template.clusters.forEach(cluster=>{
        const p=payloadPoint(template,cluster.x+cluster.w/2,cluster.y+cluster.h/2,localDepth-.035);
        clusterMetas.push({id:`${placementId}/cluster:${cluster.id}`,rank:manifest.rank,layer:template.layer,role,solid,label:cluster.label,local:p,scale:{x:.045,y:cluster.h*SOURCE_SCALE,z:cluster.w*SOURCE_SCALE}});
      });
      template.nodes.forEach(node=>{
        const p=payloadPoint(template,node.x+node.w/2,node.y+node.h/2,localDepth+NODE_DEPTH/2),semanticId=`${placementId}/node:${node.id}`;
        const meta={semanticId,rank:manifest.rank,layer:template.layer,role,solid,nodeId:node.id,label:node.label,op:node.op,description:node.description,sourceIndex:node.sourceIndex,ownership:ownershipForNode(node.id,manifest),local:p,scale:{x:NODE_DEPTH,y:Math.max(.1,node.h*SOURCE_SCALE),z:Math.max(.12,node.w*SOURCE_SCALE)}},batchKey=`${node.op}:${solid?'solid':'ghost'}`;
        if(!nodeBatches.has(batchKey))nodeBatches.set(batchKey,{key:batchKey,op:node.op,tone:solid?'solid':'ghost',metas:[],mesh:null});nodeBatches.get(batchKey).metas.push(meta);sceneIndex.nodes.set(semanticId,meta);
        if(!nodeMetasByRank.has(manifest.rank))nodeMetasByRank.set(manifest.rank,[]);nodeMetasByRank.get(manifest.rank).push(meta);
      });
      template.edges.forEach(edge=>{
        const semanticId=`${placementId}/edge:${edge.source}->${edge.target}:${edge.sourceIndex}`,points=edge.points.map(point=>payloadPoint(template,point.x,point.y,localDepth+NODE_DEPTH+.012));
        const meta={semanticId,rank:manifest.rank,layer:template.layer,role,solid,kind:edge.kind,source:edge.source,target:edge.target,points},batchKey=`${edge.kind}:${solid?'solid':'ghost'}`;
        if(!edgeBatches.has(batchKey))edgeBatches.set(batchKey,{key:batchKey,kind:edge.kind,tone:solid?'solid':'ghost',metas:[],segments:[],line:null});edgeBatches.get(batchKey).metas.push(meta);sceneIndex.edges.set(semanticId,meta);edgeSemanticCounts[edge.kind]=(edgeSemanticCounts[edge.kind]||0)+1;
      });
    }

    plan.manifests.forEach(manifest=>{
      sceneIndex.ranks.set(manifest.rank,manifest);const [segment]=manifest.layerSegments,count=segment.end-segment.start+1;
      for(let layer=segment.start;layer<=segment.end;layer+=1){const localIndex=layer-segment.start,localDepth=(localIndex-(count-1)/2)*LAYER_GAP;templatePlacement(manifest,spec.layers[layer],localDepth,'layer');}
      if(manifest.staticRoles.includes('input'))templatePlacement(manifest,spec.static.input,-(count/2+.58)*LAYER_GAP,'input');
      if(manifest.staticRoles.includes('output'))templatePlacement(manifest,spec.static.output,(count/2+.58)*LAYER_GAP,'output');
    });
    const expected=plan.manifests.reduce((totals,manifest)=>{
      const [segment]=manifest.layerSegments;
      for(let layer=segment.start;layer<=segment.end;layer+=1){totals.nodes+=spec.layers[layer].nodes.length;totals.clusters+=spec.layers[layer].clusters.length;totals.edges+=spec.layers[layer].edges.length;}
      if(manifest.staticRoles.includes('input')){totals.nodes+=spec.static.input.nodes.length;totals.clusters+=spec.static.input.clusters.length;totals.edges+=spec.static.input.edges.length;}
      if(manifest.staticRoles.includes('output')){totals.nodes+=spec.static.output.nodes.length;totals.clusters+=spec.static.output.clusters.length;totals.edges+=spec.static.output.edges.length;}
      return totals;
    },{nodes:0,clusters:0,edges:0});
    const integrity={expected,actual:{nodes:sceneIndex.nodes.size,clusters:clusterMetas.length,edges:sceneIndex.edges.size,layers:sceneIndex.layers.size},passed:false};
    integrity.passed=expected.nodes===integrity.actual.nodes&&expected.clusters===integrity.actual.clusters&&expected.edges===integrity.actual.edges&&integrity.actual.layers===plan.stats.layerInstances+plan.stats.staticInstances;
    if(!integrity.passed)throw new Error(`ThreeSceneCompiler integrity mismatch: ${JSON.stringify(integrity)}`);

    function materialColor(op){return colors[op]||OP_COLORS[op]||'#64748b';}
    const rankGeometry=new THREE.BoxGeometry(RANK_SIZE.x,RANK_SIZE.y,RANK_SIZE.z),rankMaterial=new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:.11,depthWrite:false,vertexColors:true,toneMapped:false});
    const rankMesh=new THREE.InstancedMesh(rankGeometry,rankMaterial,plan.manifests.length);rankMesh.frustumCulled=false;scene.add(rankMesh);
    const nodeGeometry=new THREE.BoxGeometry(1,1,1);
    nodeBatches.forEach(batch=>{const material=new THREE.MeshBasicMaterial({color:batch.tone==='solid'?materialColor(batch.op):GHOST_COLOR,transparent:true,opacity:batch.tone==='solid'?.94:.065,depthWrite:batch.tone==='solid',toneMapped:false});batch.mesh=new THREE.InstancedMesh(nodeGeometry,material,batch.metas.length);batch.mesh.frustumCulled=false;batch.mesh.userData.batch=batch;scene.add(batch.mesh);});
    const cardMaterial=new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:0,depthWrite:false,vertexColors:true,side:THREE.DoubleSide,toneMapped:false}),cardMesh=new THREE.InstancedMesh(nodeGeometry,cardMaterial,layerMetas.length);cardMesh.frustumCulled=false;cardMesh.visible=false;scene.add(cardMesh);
    const clusterMaterial=new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:0,depthWrite:false,vertexColors:true,side:THREE.DoubleSide,toneMapped:false}),clusterMesh=new THREE.InstancedMesh(nodeGeometry,clusterMaterial,clusterMetas.length);clusterMesh.frustumCulled=false;clusterMesh.visible=false;scene.add(clusterMesh);
    const outlineGeometry=new THREE.BufferGeometry(),outlineMaterial=new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.78,depthWrite:false}),rankOutlines=new THREE.LineSegments(outlineGeometry,outlineMaterial);rankOutlines.frustumCulled=false;scene.add(rankOutlines);
    const selectionGeometry=new THREE.EdgesGeometry(new THREE.BoxGeometry(1,1,1)),selectionMaterial=new THREE.LineBasicMaterial({color:0xffffff,transparent:true,opacity:.95,depthTest:false}),selectionBox=new THREE.LineSegments(selectionGeometry,selectionMaterial);selectionBox.visible=false;selectionBox.renderOrder=12;scene.add(selectionBox);
    const commGeometry=new THREE.BufferGeometry(),commMaterial=new THREE.LineBasicMaterial({color:0x38bdf8,transparent:true,opacity:.95,depthTest:false}),commLines=new THREE.LineSegments(commGeometry,commMaterial);commLines.visible=false;commLines.renderOrder=10;scene.add(commLines);
    const parallelFrameGroup=new THREE.Group();parallelFrameGroup.renderOrder=3;scene.add(parallelFrameGroup);
    const operatorLabelGroup=new THREE.Group(),operatorLabelTextures=new Map(),operatorLabelGeometry=new THREE.PlaneGeometry(1,1);operatorLabelGroup.renderOrder=20;scene.add(operatorLabelGroup);

    edgeBatches.forEach(batch=>{
      batch.metas.forEach(meta=>{for(let i=0;i<meta.points.length-1;i+=1)batch.segments.push({meta,a:meta.points[i],b:meta.points[i+1]});});
      const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(batch.segments.length*6),3));
      const material=new THREE.LineBasicMaterial({color:batch.tone==='solid'?(EDGE_COLORS[batch.kind]||'#94a3b8'):GHOST_COLOR,transparent:true,opacity:batch.tone==='solid'?(batch.kind==='parameter'?.3:.46):.028,depthWrite:false});
      batch.line=new THREE.LineSegments(geometry,material);batch.line.frustumCulled=false;scene.add(batch.line);
    });

    let payloadVisibleRanks=null,currentGroupingLayout=null;
    function labelLines(text,context,maxWidth){
      const words=String(text).trim().split(/\s+/);if(words.length<2||context.measureText(text).width<=maxWidth)return[String(text)];
      let best=[String(text)],bestWidth=Infinity;
      for(let split=1;split<words.length;split+=1){const lines=[words.slice(0,split).join(' '),words.slice(split).join(' ')],width=Math.max(...lines.map(line=>context.measureText(line).width));if(width<bestWidth){best=lines;bestWidth=width;}}
      return best;
    }
    function operatorLabelTexture(label,nodeWidth){
      const fontSize=42,lineHeight=48,worldPerPixel=.0045,padding=6,maxWidth=clamp(Math.floor(nodeWidth/worldPerPixel)-padding*2,84,480),key=`${label}:${maxWidth}`;
      if(operatorLabelTextures.has(key))return operatorLabelTextures.get(key);
      const canvas=document.createElement('canvas');let context=canvas.getContext('2d');context.font=`600 ${fontSize}px Inter, system-ui, sans-serif`;const lines=labelLines(label,context,maxWidth),textWidth=Math.ceil(Math.max(...lines.map(line=>context.measureText(line).width)));
      canvas.width=Math.max(1,textWidth+padding*2);canvas.height=lines.length*lineHeight+padding*2;context=canvas.getContext('2d');context.font=`600 ${fontSize}px Inter, system-ui, sans-serif`;context.textAlign='center';context.textBaseline='middle';context.fillStyle='#111111';
      lines.forEach((line,index)=>context.fillText(line,canvas.width/2,padding+lineHeight*(index+.5)));
      const texture=new THREE.CanvasTexture(canvas);if('colorSpace'in texture&&THREE.SRGBColorSpace)texture.colorSpace=THREE.SRGBColorSpace;texture.minFilter=THREE.LinearFilter;texture.magFilter=THREE.LinearFilter;const asset={texture,width:canvas.width*worldPerPixel,height:canvas.height*worldPerPixel};operatorLabelTextures.set(key,asset);return asset;
    }
    function clearOperatorLabels(){operatorLabelGroup.children.slice().forEach(sprite=>{operatorLabelGroup.remove(sprite);sprite.material.dispose();});}
    function payloadNodeLabel(meta,manifest){
      const ownership=meta.ownership;if(ownership?.axis==='tp')return`${meta.label} · TP${ownership.index}/${ownership.count}`;
      if(ownership?.axis==='ep'){if(meta.nodeId==='expert_pool')return`${meta.label} · E${manifest.expertOwnership.start}–E${manifest.expertOwnership.end}`;return`${meta.label} · EP${ownership.index}/${ownership.count}`;}
      return meta.label;
    }
    function addOperatorLabel(label,x,y,z,width,semanticId){const asset=operatorLabelTexture(label,width),material=new THREE.MeshBasicMaterial({map:asset.texture,transparent:true,depthTest:false,depthWrite:false,side:THREE.FrontSide,toneMapped:false}),labelMesh=new THREE.Mesh(operatorLabelGeometry,material);labelMesh.position.set(x,y,z);labelMesh.rotation.y=Math.PI/2;labelMesh.scale.set(asset.width,asset.height,1);labelMesh.renderOrder=20;labelMesh.userData.semanticId=semanticId;operatorLabelGroup.add(labelMesh);}
    function updateOperatorLabels(){
      clearOperatorLabels();if(state.sceneMode!=='rank'||state.selectedRank===null)return;
      const rank=rankPositions.get(state.selectedRank),manifest=plan.manifestByRank.get(state.selectedRank),frontLayer=manifest?.layerSegments[0]?.start;if(!rank||frontLayer===undefined)return;
      const [segment]=manifest.layerSegments,e=manifest.expertOwnership,c=manifest.coordinate,frontNodes=(nodeMetasByRank.get(state.selectedRank)||[]).filter(meta=>meta.layer===frontLayer),frontX=Math.max(...frontNodes.map(meta=>meta.local.x+meta.scale.x*.52));
      frontNodes.forEach(meta=>addOperatorLabel(payloadNodeLabel(meta,manifest),rank.x+meta.local.x+meta.scale.x*.52,rank.y+meta.local.y,rank.z+meta.local.z,meta.scale.z*.9,meta.semanticId));
      addOperatorLabel(`R${manifest.rank} · PP${c.pp} L${segment.start}–L${segment.end} · TP${c.tp}/${topology.dimensions.tp} · EP${c.ep}/${topology.dimensions.ep} E${e.start}–E${e.end} · EDP${c.edp}/${topology.dimensions.edp}`,rank.x+frontX+.012,rank.y+RANK_SIZE.y/2-.34,rank.z,RANK_SIZE.z-1,`rank:${manifest.rank}/payload`);
    }
    function compose(meta,index,mesh){const rank=rankPositions.get(meta.rank),hidden=(state.sceneMode==='rank'&&state.selectedRank!==null&&meta.rank!==state.selectedRank)||(payloadVisibleRanks&&!payloadVisibleRanks.has(meta.rank));dummy.position.set(rank.x+meta.local.x,rank.y+meta.local.y,rank.z+meta.local.z);dummy.rotation.set(0,0,0);dummy.scale.set(hidden?0:meta.scale.x,hidden?0:meta.scale.y,hidden?0:meta.scale.z);dummy.updateMatrix();mesh.setMatrixAt(index,dummy.matrix);}
    function boxEdges(center,size,target,offset){
      const hx=size.x/2,hy=size.y/2,hz=size.z/2,corners=[[-hx,-hy,-hz],[hx,-hy,-hz],[hx,hy,-hz],[-hx,hy,-hz],[-hx,-hy,hz],[hx,-hy,hz],[hx,hy,hz],[-hx,hy,hz]],pairs=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
      let cursor=offset;pairs.forEach(pair=>pair.forEach(id=>{target[cursor++]=center.x+corners[id][0];target[cursor++]=center.y+corners[id][1];target[cursor++]=center.z+corners[id][2];}));return cursor;
    }
    function selectedGroup(){
      if(state.selectedGroup&&plan.groupById.has(state.selectedGroup))return plan.groupById.get(state.selectedGroup);
      const axis=state.groupingAxes[state.groupingAxes.length-1];if(!AXES.includes(axis)||state.selectedRank===null)return null;
      return(plan.groupByAxisRank[axis].get(state.selectedRank)||[])[0]||null;
    }
    function rankColor(manifest){
      const group=selectedGroup(),member=!group||group.ranks.includes(manifest.rank);let value='#64748b';
      const axis=state.groupingAxes[state.groupingAxes.length-1];if(axis==='pp')value=PP_COLORS[manifest.coordinate.pp%PP_COLORS.length];else if(axis==='tp')value=TP_COLORS[manifest.coordinate.tp%TP_COLORS.length];else if(axis==='ep')value=EP_COLORS[manifest.coordinate.ep%EP_COLORS.length];else if(axis==='edp')value=EDP_COLORS[manifest.coordinate.edp%EDP_COLORS.length];
      tempColor.set(value);if(!member)tempColor.lerp(new THREE.Color(backgroundColor()),.985);if(manifest.rank===state.selectedRank)tempColor.lerp(new THREE.Color('#ffffff'),.28);return tempColor.clone();
    }
    function visualFactor(meta){const group=selectedGroup();if(group&&!group.ranks.includes(meta.rank))return .08;if(state.sceneMode==='rank'&&state.selectedRank!==null&&meta.rank!==state.selectedRank)return .035;if(state.selectedRank===meta.rank&&state.selectedLayer!==null&&meta.layer!==null&&meta.layer!==state.selectedLayer)return .26;return 1;}
    function fadedColor(value,factor){const color=new THREE.Color(value),background=new THREE.Color(backgroundColor());return color.lerp(background,1-factor);}

    function updateLayout(){
      const mode=state.sceneMode==='packed'?'packed':'exploded',group=selectedGroup(),payloadEnabled=state.groupingAxes.length>0||state.sceneMode==='rank';currentGroupingLayout=state.groupingAxes.length?buildHierarchicalRankLayout(plan.manifests,state.groupingAxes,mode,topology):null;payloadVisibleRanks=payloadEnabled?(group?new Set(group.ranks):null):new Set();root.dataset.payloadVisible=String(payloadEnabled);plan.manifests.forEach((manifest,index)=>{const p=currentGroupingLayout?.positions.get(manifest.rank)||layoutPosition(manifest.coordinate,topology,mode,'standard'),hidden=state.sceneMode==='rank'&&state.selectedRank!==null&&manifest.rank!==state.selectedRank;rankPositions.set(manifest.rank,p);shellDisplayPositions.set(manifest.rank,{...p});dummy.position.set(p.x,p.y,p.z);dummy.rotation.set(0,0,0);dummy.scale.setScalar(hidden?.0001:1);dummy.updateMatrix();rankMesh.setMatrixAt(index,dummy.matrix);rankMesh.setColorAt(index,rankColor(manifest));});rankMesh.instanceMatrix.needsUpdate=true;if(rankMesh.instanceColor)rankMesh.instanceColor.needsUpdate=true;
      layerMetas.forEach((meta,index)=>compose(meta,index,cardMesh));cardMesh.instanceMatrix.needsUpdate=true;
      clusterMetas.forEach((meta,index)=>compose(meta,index,clusterMesh));clusterMesh.instanceMatrix.needsUpdate=true;
      nodeBatches.forEach(batch=>{batch.metas.forEach((meta,index)=>compose(meta,index,batch.mesh));batch.mesh.instanceMatrix.needsUpdate=true;});
      edgeBatches.forEach(batch=>{const array=batch.line.geometry.attributes.position.array;let cursor=0;batch.segments.forEach(segment=>{const rank=rankPositions.get(segment.meta.rank),hidden=(state.sceneMode==='rank'&&state.selectedRank!==null&&segment.meta.rank!==state.selectedRank)||(payloadVisibleRanks&&!payloadVisibleRanks.has(segment.meta.rank));if(hidden){array[cursor++]=0;array[cursor++]=0;array[cursor++]=0;array[cursor++]=0;array[cursor++]=0;array[cursor++]=0;}else{array[cursor++]=rank.x+segment.a.x;array[cursor++]=rank.y+segment.a.y;array[cursor++]=rank.z+segment.a.z;array[cursor++]=rank.x+segment.b.x;array[cursor++]=rank.y+segment.b.y;array[cursor++]=rank.z+segment.b.z;}});batch.line.geometry.attributes.position.needsUpdate=true;batch.line.geometry.computeBoundingSphere();});
      const visibleManifests=state.sceneMode==='rank'&&state.selectedRank!==null?[plan.manifestByRank.get(state.selectedRank)]:plan.manifests,positions=new Float32Array(visibleManifests.length*72),lineColors=new Float32Array(visibleManifests.length*72);let cursor=0,colorCursor=0;
      visibleManifests.forEach(manifest=>{cursor=boxEdges(rankPositions.get(manifest.rank),RANK_SIZE,positions,cursor);const color=rankColor(manifest);for(let i=0;i<24;i+=1){lineColors[colorCursor++]=color.r;lineColors[colorCursor++]=color.g;lineColors[colorCursor++]=color.b;}});
      outlineGeometry.setAttribute('position',new THREE.BufferAttribute(positions,3));outlineGeometry.setAttribute('color',new THREE.BufferAttribute(lineColors,3));outlineGeometry.computeBoundingSphere();updateCommunication();updateParallelFrames();updateSelection();requestRender();
    }

    function updateColors(){
      plan.manifests.forEach((manifest,index)=>rankMesh.setColorAt(index,rankColor(manifest)));if(rankMesh.instanceColor)rankMesh.instanceColor.needsUpdate=true;
      layerMetas.forEach((meta,index)=>{const manifest=plan.manifestByRank.get(meta.rank),stageColor=PP_COLORS[manifest.coordinate.pp%PP_COLORS.length],factor=visualFactor(meta),value=meta.solid?stageColor:GHOST_COLOR;cardMesh.setColorAt(index,fadedColor(value,meta.solid?Math.max(.72,factor):.11));});if(cardMesh.instanceColor)cardMesh.instanceColor.needsUpdate=true;
      clusterMetas.forEach((meta,index)=>{const factor=visualFactor(meta),value=meta.solid?(/MoE|FFN|专家/i.test(meta.label)?colors.moe:colors.attention):GHOST_COLOR;clusterMesh.setColorAt(index,fadedColor(value,meta.solid?Math.max(.68,factor):.1));});if(clusterMesh.instanceColor)clusterMesh.instanceColor.needsUpdate=true;
      rankMaterial.opacity=state.theme==='light'?.018:.04;
      nodeBatches.forEach(batch=>{batch.mesh.material.color.set('#ffffff');batch.metas.forEach((meta,index)=>{tempColor.set(batch.tone==='solid'?materialColor(batch.op):GHOST_COLOR);if(batch.tone==='solid'&&meta.ownership?.axis==='tp'){tempTint.set(TP_COLORS[meta.ownership.index%TP_COLORS.length]);tempColor.lerp(tempTint,.3);}else if(batch.tone==='solid'&&meta.ownership?.axis==='ep'){tempTint.set(EP_COLORS[meta.ownership.index%EP_COLORS.length]);tempColor.lerp(tempTint,.38);}batch.mesh.setColorAt(index,tempColor);});if(batch.mesh.instanceColor)batch.mesh.instanceColor.needsUpdate=true;batch.mesh.material.opacity=batch.tone==='solid'?(state.sceneMode==='rank'?.96:.9):(state.sceneMode==='rank'?.4:(state.theme==='light'?.028:.04));});
      edgeBatches.forEach(batch=>{batch.line.material.opacity=batch.tone==='solid'?(state.sceneMode==='rank'?.55:.25):(state.sceneMode==='rank'?.16:.014);});
      const lineColors=outlineGeometry.attributes.color?.array;if(lineColors){let cursor=0;const visibleManifests=state.sceneMode==='rank'&&state.selectedRank!==null?[plan.manifestByRank.get(state.selectedRank)]:plan.manifests;visibleManifests.forEach(manifest=>{const color=rankColor(manifest);for(let i=0;i<24;i+=1){lineColors[cursor++]=color.r;lineColors[cursor++]=color.g;lineColors[cursor++]=color.b;}});outlineGeometry.attributes.color.needsUpdate=true;}
      updateSelection();requestRender();
    }

    function updateCommunication(){
      const group=selectedGroup();if(!group){commLines.visible=false;groupBadge.textContent='';return;}
      const members=group.ranks.map(rank=>plan.manifestByRank.get(rank)).sort((a,b)=>a.coordinate[group.axis]-b.coordinate[group.axis]),positions=[];
      for(let index=0;index<members.length-1;index+=1){const a=rankPositions.get(members[index].rank),b=rankPositions.get(members[index+1].rank);positions.push(a.x,a.y,a.z,b.x,b.y,b.z);}
      commGeometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));commGeometry.computeBoundingSphere();commMaterial.color.set(TOPOLOGY_COLORS[group.axis]||'#38bdf8');commLines.visible=true;groupBadge.textContent=`${group.axis.toUpperCase()} · ${group.ranks.length} ranks · ${group.sourceLevel}`;
    }
    function clearParallelFrames(){
      [...parallelFrameGroup.children].forEach(child=>{parallelFrameGroup.remove(child);child.traverse(object=>{object.geometry?.dispose?.();if(Array.isArray(object.material))object.material.forEach(material=>material.dispose?.());else object.material?.dispose?.();});});
    }
    function updateParallelFrames(){
      clearParallelFrames();parallelFrameGroup.visible=Boolean(currentGroupingLayout?.frames.length)&&state.sceneMode!=='rank';if(!parallelFrameGroup.visible)return;
      const colorFor=(axis,value)=>axis==='pp'?PP_COLORS[value%PP_COLORS.length]:axis==='tp'?TP_COLORS[value%TP_COLORS.length]:axis==='ep'?EP_COLORS[value%EP_COLORS.length]:EDP_COLORS[value%EDP_COLORS.length];
      const frames=currentGroupingLayout.frames,fillGeometry=new THREE.BoxGeometry(1,1,1),fillMaterial=new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:.009,depthWrite:false,vertexColors:true,side:THREE.BackSide,toneMapped:false}),fills=new THREE.InstancedMesh(fillGeometry,fillMaterial,frames.length),edgePositions=new Float32Array(frames.length*72),edgeColors=new Float32Array(frames.length*72);let positionCursor=0,colorCursor=0;
      frames.forEach((frame,index)=>{
        const color=new THREE.Color(colorFor(frame.axis,frame.value));color.lerp(new THREE.Color(backgroundColor()),Math.min(.56,frame.level*.18));dummy.position.set(frame.center.x,frame.center.y,frame.center.z);dummy.rotation.set(0,0,0);dummy.scale.set(frame.size.x,frame.size.y,frame.size.z);dummy.updateMatrix();fills.setMatrixAt(index,dummy.matrix);fills.setColorAt(index,color);positionCursor=boxEdges(frame.center,frame.size,edgePositions,positionCursor);for(let vertex=0;vertex<24;vertex+=1){edgeColors[colorCursor++]=color.r;edgeColors[colorCursor++]=color.g;edgeColors[colorCursor++]=color.b;}
      });
      fills.frustumCulled=false;fills.instanceMatrix.needsUpdate=true;if(fills.instanceColor)fills.instanceColor.needsUpdate=true;fills.userData.parallelFrames=frames;parallelFrameGroup.add(fills);
      const edgeGeometry=new THREE.BufferGeometry();edgeGeometry.setAttribute('position',new THREE.BufferAttribute(edgePositions,3));edgeGeometry.setAttribute('color',new THREE.BufferAttribute(edgeColors,3));edgeGeometry.computeBoundingSphere();const edges=new THREE.LineSegments(edgeGeometry,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.84,depthTest:false,depthWrite:false}));edges.frustumCulled=false;edges.renderOrder=8;edges.userData.parallelFrames=frames;parallelFrameGroup.add(edges);
    }
    function writeShellPositions(positionMap){
      plan.manifests.forEach((manifest,index)=>{const p=positionMap.get(manifest.rank);shellDisplayPositions.set(manifest.rank,{...p});dummy.position.set(p.x,p.y,p.z);dummy.rotation.set(0,0,0);dummy.scale.setScalar(1);dummy.updateMatrix();rankMesh.setMatrixAt(index,dummy.matrix);rankMesh.setColorAt(index,rankColor(manifest));});rankMesh.instanceMatrix.needsUpdate=true;if(rankMesh.instanceColor)rankMesh.instanceColor.needsUpdate=true;
      const positions=new Float32Array(plan.manifests.length*72),lineColors=new Float32Array(plan.manifests.length*72);let cursor=0,colorCursor=0;
      plan.manifests.forEach(manifest=>{cursor=boxEdges(positionMap.get(manifest.rank),RANK_SIZE,positions,cursor);const color=rankColor(manifest);for(let index=0;index<24;index+=1){lineColors[colorCursor++]=color.r;lineColors[colorCursor++]=color.g;lineColors[colorCursor++]=color.b;}});
      outlineGeometry.setAttribute('position',new THREE.BufferAttribute(positions,3));outlineGeometry.setAttribute('color',new THREE.BufferAttribute(lineColors,3));outlineGeometry.computeBoundingSphere();requestRender();
    }
    function setPayloadTransition(hidden){
      if(hidden){nodeBatches.forEach(batch=>{batch.mesh.material.opacity=0;});edgeBatches.forEach(batch=>{batch.line.material.opacity=0;});cardMesh.material.opacity=0;clusterMesh.material.opacity=0;parallelFrameGroup.visible=false;return;}
      updateColors();updateParallelFrames();
    }
    function animateTopologyLayout(fromPositions){
      cancelAnimationFrame(layoutTweenRaf);const targets=new Map([...rankPositions].map(([rank,p])=>[rank,{...p}]));if(!fromPositions.size){writeShellPositions(targets);return;}
      setPayloadTransition(true);writeShellPositions(fromPositions);const started=performance.now(),duration=560,interpolated=new Map();
      const tick=now=>{const raw=clamp((now-started)/duration,0,1),t=1-Math.pow(1-raw,3);targets.forEach((target,rank)=>{const source=fromPositions.get(rank)||target;interpolated.set(rank,{x:source.x+(target.x-source.x)*t,y:source.y+(target.y-source.y)*t,z:source.z+(target.z-source.z)*t});});writeShellPositions(interpolated);if(raw<1)layoutTweenRaf=requestAnimationFrame(tick);else{layoutTweenRaf=0;writeShellPositions(targets);setPayloadTransition(false);}};
      layoutTweenRaf=requestAnimationFrame(tick);
    }
    function updateSelection(){
      let meta=null;if(state.selectedNode)meta=sceneIndex.nodes.get(state.selectedNode.semanticId);if(!meta){selectionBox.visible=false;return;}
      const rank=rankPositions.get(meta.rank);selectionBox.position.set(rank.x+meta.local.x,rank.y+meta.local.y,rank.z+meta.local.z);selectionBox.scale.set(meta.scale.x*1.18,meta.scale.y*1.28,meta.scale.z*2.4);selectionBox.visible=true;
    }

    function setCamera(){
      const width=Math.max(1,viewport.clientWidth),height=Math.max(1,viewport.clientHeight),aspect=width/height,pitch=-state.camera.rx*Math.PI/180,yaw=state.camera.ry*Math.PI/180,target=state.camera.target;
      const halfFov=THREE.MathUtils.degToRad(camera.fov*.5),distance=Math.max(24,state.camera.fitHalfHeight/Math.tan(halfFov)),cosPitch=Math.cos(pitch),sinPitch=Math.sin(pitch),sinYaw=Math.sin(yaw),cosYaw=Math.cos(yaw);
      camera.position.set(target.x+sinYaw*cosPitch*distance,target.y+sinPitch*distance,target.z+cosYaw*cosPitch*distance);
      // Use the orbit tangent as camera-up so crossing either pole stays continuous.
      camera.up.set(-sinYaw*sinPitch,cosPitch,-cosYaw*sinPitch).normalize();
      camera.lookAt(target.x,target.y,target.z);camera.aspect=aspect;camera.zoom=state.camera.zoom;camera.near=Math.max(.05,distance/5000);camera.far=distance+Math.max(1200,state.camera.fitHalfHeight*30);camera.updateProjectionMatrix();zoomReadout.textContent=`${Math.round(state.camera.zoom*100)}%`;requestRender();
    }
    function fit(){
      if(state.sceneMode==='rank'&&state.selectedRank!==null){const p=rankPositions.get(state.selectedRank);state.camera.target={...p};state.camera.fitHalfHeight=9.2;}
      else{const box=new THREE.Box3();plan.manifests.forEach(manifest=>{const p=rankPositions.get(manifest.rank);box.expandByPoint(new THREE.Vector3(p.x-RANK_SIZE.x/2,p.y-RANK_SIZE.y/2,p.z-RANK_SIZE.z/2));box.expandByPoint(new THREE.Vector3(p.x+RANK_SIZE.x/2,p.y+RANK_SIZE.y/2,p.z+RANK_SIZE.z/2));});const center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());state.camera.target={x:center.x,y:center.y,z:center.z};state.camera.fitHalfHeight=Math.max(24,size.length()*.52);}
      state.camera.zoom=1;setCamera();return api;
    }
    function resize(){const width=Math.max(1,viewport.clientWidth),height=Math.max(1,viewport.clientHeight);renderer.setSize(width,height,false);setCamera();}
    function requestRender(){dirty=true;if(!renderRaf)renderRaf=global.requestAnimationFrame(frame);}
    function frame(){renderRaf=0;if(destroyed)return;if(dirty){renderer.setClearColor(backgroundColor(),1);renderer.render(scene,camera);dirty=false;}}

    function screenPointer(event){const rect=renderer.domElement.getBoundingClientRect();pointerNdc.set(((event.clientX-rect.left)/rect.width)*2-1,-((event.clientY-rect.top)/rect.height)*2+1);raycaster.setFromCamera(pointerNdc,camera);}
    function pickRank(event){screenPointer(event);const hit=raycaster.intersectObject(rankMesh,false)[0];return hit&&Number.isInteger(hit.instanceId)?plan.manifests[hit.instanceId]:null;}
    function pickNode(event){
      if(state.sceneMode!=='rank'||state.selectedRank===null)return null;screenPointer(event);const candidates=[];nodeBatches.forEach(batch=>raycaster.intersectObject(batch.mesh,false).forEach(hit=>{const meta=batch.metas[hit.instanceId];if(meta?.rank===state.selectedRank)candidates.push({distance:hit.distance,meta});}));candidates.sort((a,b)=>a.distance-b.distance);return candidates[0]?.meta||null;
    }

    function renderInspector(){
      const stats={rankVolumes:plan.stats.rankVolumes,layerInstances:plan.stats.layerInstances,staticInstances:plan.stats.staticInstances,nodeInstances:sceneIndex.nodes.size,edgeInstances:sceneIndex.edges.size,clusterInstances:clusterMetas.length,drawBatches:2+nodeBatches.size+edgeBatches.size+4+parallelFrameGroup.children.length};
      if(state.selectedRank===null){const d=topology.dimensions,stages=Array.from({length:d.pp},(_,stage)=>plan.manifests.find(item=>item.coordinate.pp===stage)?.layerSegments[0]).filter(Boolean),expertsPerShard=model.routedExperts/d.ep,ranksPerStage=d.tp*d.cp*d.ep*d.edp;inspector.innerHTML=`<div class="pto-model-rank-deck__inspector-head"><span>模型训练并行策略</span><b>${stats.rankVolumes} RANKS</b></div>
        <div class="pto-model-rank-deck__stat-grid"><span>Pipeline<b>PP${d.pp}</b></span><span>Tensor<b>TP${d.tp}</b></span><span>Expert<b>EP${d.ep}</b></span><span>Expert Data<b>EDP${d.edp}</b></span><span>Context<b>CP${d.cp}</b></span><span>World Size<b>${stats.rankVolumes}</b></span></div>
        <section><h3>128 Rank 如何组成 <small>demo</small></h3><p><b>${d.tp} TP × ${d.pp} PP × ${d.cp} CP × ${d.ep} EP × ${d.edp} EDP = ${stats.rankVolumes}</b>。每个 Pipeline Stage 包含 ${ranksPerStage} 个 Rank；每个 Rank 对应一组唯一的 PP、TP、EP、EDP 坐标。</p></section>
        <section><h3>Pipeline Parallel · PP${d.pp}</h3><p>${stages.map(segment=>`PP${segment.stage}：L${segment.start}–L${segment.end}`).join('；')}。输入载荷只进入 PP0，最终 Norm、LM Head、Logits 与 MTP 位于 PP${d.pp-1}。</p></section>
        <section><h3>Tensor / Expert 切分</h3><p>TP${d.tp} 将 Attention、Linear、Dense FFN 等张量算子切成 ${d.tp} 份；EP${d.ep} 将 ${model.routedExperts} 个 Routed Experts 分成 ${d.ep} 组，每个 EP shard 持有 ${expertsPerShard} 个专家，并通过 Dispatch / Combine 完成跨 Rank 路由。</p></section>
        <section><h3>Replica / Context</h3><p>EDP${d.edp} 为相同的 PP × TP × EP 分片保留 ${d.edp} 份数据并行副本；CP${d.cp} 表示当前演示不再切分上下文序列。该 128 Rank 配置是产品演示映射，不宣称为官方训练部署参数。</p></section>
        <section><h3>场景完整性 <small>${integrity.passed?'verified':'error'}</small></h3><p>${stats.layerInstances} 个 Layer placement、${stats.nodeInstances.toLocaleString()} 个算子实例和 ${stats.edgeInstances.toLocaleString()} 条边均保留在 GPU SceneIndex 中；默认 Rank 单行仅显示空容器，选择任一切分后显示完整模型结构。</p></section>
        <div class="pto-model-rank-deck__source"><i data-source="official"></i>模型源信息 <i data-source="derived"></i>并行推导 <i data-source="demo"></i>128 Rank 演示映射</div>`;return;
      }
      const manifest=plan.manifestByRank.get(state.selectedRank),c=manifest.coordinate,[segment]=manifest.layerSegments,e=manifest.expertOwnership,layers=Array.from({length:segment.end-segment.start+1},(_,index)=>segment.start+index),group=selectedGroup(),template=state.selectedLayer!==null?spec.layers[state.selectedLayer]:null;
      inspector.innerHTML=`<div class="pto-model-rank-deck__inspector-head"><span>Rank ${manifest.rank}</span>${state.sceneMode==='rank'?'<button type="button" data-enter-selected>返回全局</button>':''}</div>
        <div class="pto-model-rank-deck__coordinate"><span>PP<b>${c.pp}</b></span><span>TP<b>${c.tp}/${topology.dimensions.tp}</b></span><span>EP<b>${c.ep}/${topology.dimensions.ep}</b></span><span>EDP<b>${c.edp}/${topology.dimensions.edp}</b></span></div>
        <section><h3>模型载荷 <small>完整 GPU instances</small></h3><p>L${segment.start}–L${segment.end} · ${layers.length} 个 Decoder Layer${manifest.staticRoles.includes('input')?' · Input':''}${manifest.staticRoles.includes('output')?' · Output/MTP':''}</p><div class="pto-model-rank-deck__layer-list">${layers.map(layer=>`<button type="button" data-inspector-layer="${layer}" class="${layer===state.selectedLayer?'is-active':''}">L${layer}</button>`).join('')}</div></section>
        ${template?`<section><h3>L${template.layer} 原始节点 <small>${template.nodes.length} / ${template.edges.length} edges</small></h3><div class="pto-model-rank-deck__node-list">${template.nodes.map(node=>`<button type="button" data-inspector-node="${esc(node.id)}" class="${state.selectedNode?.nodeId===node.id?'is-active':''}"><span>${esc(node.label)}</span><small>${esc(node.id)}</small></button>`).join('')}</div></section>`:''}
        <section><h3>Expert Shard <small>demo</small></h3><p>E${e.start}–E${e.end} · ${e.ids.length} / ${model.routedExperts} routed experts</p></section>
        ${group?`<section><h3>${group.axis.toUpperCase()} Group <small>${group.sourceLevel}</small></h3><p>${group.ranks.length} ranks · ${group.ranks.map(rank=>`R${rank}`).join(' → ')}</p></section>`:''}
        ${state.selectedNode?`<section class="pto-model-rank-deck__node-detail"><h3>Selected Operator</h3><b>${esc(state.selectedNode.label)}</b><code>${esc(state.selectedNode.semanticId)}</code><p>${state.selectedNode.layer===null?`Static ${esc(state.selectedNode.role)}`:`L${state.selectedNode.layer}`} · ${esc(state.selectedNode.ownership.kind)}${state.selectedNode.ownership.axis?` · ${state.selectedNode.ownership.axis.toUpperCase()} ${state.selectedNode.ownership.index}/${state.selectedNode.ownership.count}`:''}</p></section>`:''}
        <section><h3>Payload Signature <small>derived</small></h3><code>${esc(manifest.payloadSignature)}</code></section><div class="pto-model-rank-deck__source"><i data-source="official"></i>模型源信息 <i data-source="derived"></i>分组推导 <i data-source="demo"></i>演示映射</div>`;
    }

    function updateButtons(){
      root.querySelectorAll('[data-scene-mode]').forEach(button=>button.classList.toggle('is-active',button.dataset.sceneMode===state.sceneMode));
      root.querySelectorAll('[data-topology-mode]').forEach(button=>{const axis=MODE_TO_AXIS[button.dataset.topologyMode];button.classList.toggle('is-active',state.groupingAxes.includes(axis));button.setAttribute('aria-pressed',String(state.groupingAxes.includes(axis)));button.disabled=state.sceneMode==='rank';});
      const rankButton=root.querySelector('[data-scene-mode="rank"]');if(rankButton)rankButton.disabled=state.selectedRank===null;
      const scopeReset=root.querySelector('[data-grouping-reset]'),rankFocused=state.sceneMode==='rank'&&state.selectedRank!==null;
      scopeReadout.textContent=rankFocused?`Rank ${state.selectedRank} · 1 Rank`:`全部 · ${plan.topology.worldSize} Ranks`;
      if(scopeReset){scopeReset.classList.toggle('is-resettable',state.groupingAxes.length>0||rankFocused);scopeReset.setAttribute('aria-label',rankFocused||state.groupingAxes.length?`清除分组并恢复全部 ${plan.topology.worldSize} 个 Rank`:`当前范围：全部 ${plan.topology.worldSize} 个 Rank`);}
      root.querySelector('[data-rank-theme]').textContent=state.theme==='light'?'深色':'浅色';
    }
    function updateUrl(){if(options.syncUrl===false||!global.history?.replaceState)return;const url=new URL(global.location.href);url.searchParams.set('scene',state.sceneMode);url.searchParams.set('topology',state.groupingAxes.length?state.groupingAxes.join(','):'standard');url.searchParams.set('theme',state.theme);if(state.selectedRank===null)url.searchParams.delete('rank');else url.searchParams.set('rank',state.selectedRank);if(state.selectedLayer===null)url.searchParams.delete('layer');else url.searchParams.set('layer',state.selectedLayer);global.history.replaceState(null,'',url);}
    function sync({layout=false,refit=false}={}){state.topologyMode=topologyModeForAxes(state.groupingAxes);root.dataset.sceneMode=state.sceneMode;root.dataset.topologyMode=state.topologyMode;root.dataset.groupingAxes=state.groupingAxes.join(' ');root.dataset.theme=state.theme;document.documentElement.dataset.theme=state.theme;updateButtons();if(layout)updateLayout();else updateCommunication();updateColors();updateOperatorLabels();renderInspector();updateUrl();if(refit)fit();else setCamera();options.onStateChange?.(copy(state),api);}
    function selectRank(rank){const next=Number(rank),manifest=plan.manifestByRank.get(next);if(!manifest)return null;state.selectedRank=next;const [segment]=manifest.layerSegments;if(state.selectedLayer===null||state.selectedLayer<segment.start||state.selectedLayer>segment.end)state.selectedLayer=segment.start;state.selectedNode=null;state.selectedGroup=null;a11y.textContent=`Selected Rank ${next}, Layer ${segment.start} to ${segment.end}`;sync({layout:state.sceneMode==='rank'||state.groupingAxes.length>0,refit:state.sceneMode==='rank'});options.onRankSelect?.(manifest,api);return manifest;}
    function clearRankSelection(){if(state.sceneMode==='rank'||state.selectedRank===null&&state.selectedGroup===null)return false;state.selectedRank=null;state.selectedLayer=null;state.selectedNode=null;state.selectedGroup=null;state.hoveredRank=null;tooltip.classList.remove('is-visible');a11y.textContent='Rank selection cleared';sync({layout:true});return true;}
    function enterRank(rank=state.selectedRank){if(selectRank(rank)===null)return api;state.sceneMode='rank';sync({layout:true,refit:true});return api;}
    function leaveRank(){if(state.sceneMode!=='rank')return api;state.sceneMode='exploded';state.selectedNode=null;sync({layout:true,refit:true});return api;}
    function setSceneMode(mode){if(!SCENE_MODES.has(mode))return api;if(mode==='rank')return enterRank();state.sceneMode=mode;state.selectedNode=null;sync({layout:true,refit:true});return api;}
    function setGroupingAxes(axesInput=[]){const axes=normalizeGroupingAxes(axesInput),fromPositions=new Map([...shellDisplayPositions].map(([rank,p])=>[rank,{...p}]));state.groupingAxes=axes;state.selectedRank=null;state.selectedLayer=null;state.selectedNode=null;state.selectedGroup=null;if(state.sceneMode==='rank')state.sceneMode='exploded';sync({layout:true,refit:true});animateTopologyLayout(fromPositions);return api;}
    function setTopologyMode(mode){if(!TOPOLOGY_MODES.has(mode))return api;if(mode==='standard')return setGroupingAxes([]);return setGroupingAxes(toggleGroupingAxis(state.groupingAxes,mode));}
    function resetGrouping(){
      if(state.groupingAxes.length===0&&state.sceneMode!=='rank'&&state.selectedRank===null)return api;state.sceneMode='exploded';return setGroupingAxes([]);
    }
    function setTheme(theme){state.theme=THEMES.has(theme)?theme:'dark';base.applySemanticPalette(root,state.theme);colors=palette();sync();options.onThemeChange?.(state.theme,api);return api;}
    function selectLayer(layer,rank=state.selectedRank){const next=Number(layer);if(rank!==state.selectedRank)selectRank(rank);const manifest=plan.manifestByRank.get(state.selectedRank),segment=manifest?.layerSegments[0];if(!segment||!Number.isInteger(next)||next<segment.start||next>segment.end)return null;state.selectedLayer=next;state.selectedNode=null;sync();options.onLayerSelect?.(next,manifest,api);return next;}
    function selectNode(nodeId,layer=state.selectedLayer,rank=state.selectedRank){const semanticId=`rank:${rank}/layer:${layer}/node:${nodeId}`,meta=sceneIndex.nodes.get(semanticId);if(!meta)return null;state.selectedNode=meta;state.selectedLayer=meta.layer;updateSelection();updateOperatorLabels();renderInspector();requestRender();options.onNodeSelect?.(meta,api);return meta;}
    function selectExpert(expertId){const next=Number(expertId);if(!Number.isInteger(next)||next<0||next>=model.routedExperts)return null;const baseManifest=plan.manifestByRank.get(state.selectedRank??0),ep=Math.floor(next/(model.routedExperts/topology.dimensions.ep)),target=plan.manifests.find(item=>item.coordinate.pp===baseManifest.coordinate.pp&&item.coordinate.tp===baseManifest.coordinate.tp&&item.coordinate.cp===baseManifest.coordinate.cp&&item.coordinate.edp===baseManifest.coordinate.edp&&item.coordinate.ep===ep);if(!target)return null;enterRank(target.rank);return target;}
    function selectGroup(groupId){const group=plan.groupById.get(groupId);if(!group)return null;state.selectedGroup=group.id;if(!state.groupingAxes.includes(group.axis))state.groupingAxes=[...state.groupingAxes,group.axis];if(state.selectedRank===null||!group.ranks.includes(state.selectedRank))state.selectedRank=group.ranks[0];sync({layout:true});return group;}
    function setPose(pose={}){if(Number.isFinite(Number(pose.rx)))state.camera.rx=Number(pose.rx);if(Number.isFinite(Number(pose.ry)))state.camera.ry=Number(pose.ry);if(Number.isFinite(Number(pose.zoom)))state.camera.zoom=clamp(Number(pose.zoom),.2,5);setCamera();return api;}

    function rootClick(event){
      const action=event.target.closest('[data-scene-mode],[data-topology-mode],[data-grouping-reset],[data-rank-fit],[data-rank-theme],[data-enter-selected],[data-inspector-layer],[data-inspector-node]');if(!action)return;
      if(action.dataset.sceneMode)setSceneMode(action.dataset.sceneMode);else if(action.dataset.topologyMode)setTopologyMode(action.dataset.topologyMode);else if(action.hasAttribute('data-grouping-reset'))resetGrouping();else if(action.hasAttribute('data-rank-fit'))fit();else if(action.hasAttribute('data-rank-theme'))setTheme(state.theme==='light'?'dark':'light');else if(action.hasAttribute('data-enter-selected'))state.sceneMode==='rank'?leaveRank():enterRank();else if(action.dataset.inspectorLayer)selectLayer(Number(action.dataset.inspectorLayer));else if(action.dataset.inspectorNode)selectNode(action.dataset.inspectorNode);
    }
    function pointerDown(event){if(event.button!==0)return;pointer={id:event.pointerId,x:event.clientX,y:event.clientY,rx:state.camera.rx,ry:state.camera.ry,moved:false};renderer.domElement.setPointerCapture?.(event.pointerId);viewport.classList.add('is-grabbing');}
    function pointerMove(event){
      if(pointer&&event.pointerId===pointer.id){const dx=event.clientX-pointer.x,dy=event.clientY-pointer.y;pointer.moved=pointer.moved||Math.hypot(dx,dy)>3;state.camera.ry=pointer.ry-dx*.2;state.camera.rx=pointer.rx-dy*.18;setCamera();return;}
      if(hoverRaf)return;hoverRaf=requestAnimationFrame(()=>{hoverRaf=0;const manifest=pickRank(event);state.hoveredRank=manifest?.rank??null;if(!manifest){tooltip.classList.remove('is-visible');return;}const [segment]=manifest.layerSegments,e=manifest.expertOwnership,c=manifest.coordinate;tooltip.innerHTML=`<b>Rank ${manifest.rank}</b><span>PP${c.pp} · TP${c.tp} · EP${c.ep} · EDP${c.edp}</span><small>L${segment.start}–L${segment.end} · E${e.start}–E${e.end}</small>`;const rect=viewport.getBoundingClientRect();tooltip.style.left=`${event.clientX-rect.left+14}px`;tooltip.style.top=`${event.clientY-rect.top+14}px`;tooltip.classList.add('is-visible');});
    }
    function pointerUp(event){if(!pointer||event.pointerId!==pointer.id)return;const moved=pointer.moved;pointer=null;viewport.classList.remove('is-grabbing');try{renderer.domElement.releasePointerCapture?.(event.pointerId);}catch(_){}if(moved)return;const node=pickNode(event);if(node){state.selectedNode=node;state.selectedLayer=node.layer;updateSelection();updateOperatorLabels();renderInspector();requestRender();return;}const manifest=pickRank(event);if(manifest){selectRank(manifest.rank);return;}clearRankSelection();}
    function doubleClick(event){const manifest=pickRank(event);if(manifest)enterRank(manifest.rank);}
    function wheel(event){event.preventDefault();state.camera.zoom=clamp(state.camera.zoom*Math.exp(-event.deltaY*.0012),.2,5);setCamera();}
    function keydown(event){if(event.key!=='Escape')return;if(state.selectedNode){state.selectedNode=null;sync();}else if(state.sceneMode==='rank')leaveRank();else if(selectedGroup()){state.selectedGroup=null;state.selectedRank=null;state.selectedLayer=null;sync({layout:true});}else if(state.selectedRank!==null){state.selectedRank=null;state.selectedLayer=null;sync({layout:true});}}

    function disposeObject(object){object.geometry?.dispose?.();if(Array.isArray(object.material))object.material.forEach(material=>material.dispose?.());else object.material?.dispose?.();}
    function destroy(){destroyed=true;cancelAnimationFrame(renderRaf);cancelAnimationFrame(layoutTweenRaf);cancelAnimationFrame(hoverRaf);resizeObserver?.disconnect();root.removeEventListener('click',rootClick);viewport.removeEventListener('keydown',keydown);renderer.domElement.removeEventListener('pointerdown',pointerDown);renderer.domElement.removeEventListener('pointermove',pointerMove);renderer.domElement.removeEventListener('pointerup',pointerUp);renderer.domElement.removeEventListener('pointercancel',pointerUp);renderer.domElement.removeEventListener('dblclick',doubleClick);renderer.domElement.removeEventListener('wheel',wheel);clearOperatorLabels();operatorLabelTextures.forEach(asset=>asset.texture.dispose());operatorLabelGeometry.dispose();scene.traverse(disposeObject);renderer.dispose();renderer.domElement.remove();}
    const ready=Promise.resolve();
    const api={root,state,spec,get plan(){return plan;},sceneIndex,integrity,renderer,scene,camera,ready,setSceneMode,setTopologyMode,setGroupingAxes,resetGrouping,setTheme,setPose,fit,selectRank,enterRank,leaveRank,selectLayer,selectNode,selectExpert,selectGroup,getRankManifest(rank){return plan.manifestByRank.get(Number(rank))||null;},getState(){return copy(state);},getSceneStats(){return{rankVolumes:plan.stats.rankVolumes,layerInstances:plan.stats.layerInstances,staticInstances:plan.stats.staticInstances,nodeInstances:sceneIndex.nodes.size,edgeInstances:sceneIndex.edges.size,clusterInstances:clusterMetas.length,nodeBatches:nodeBatches.size,edgeBatches:edgeBatches.size,drawBatches:2+nodeBatches.size+edgeBatches.size+4+parallelFrameGroup.children.length,integrity:copy(integrity),renderer:{calls:renderer.info.render.calls,triangles:renderer.info.render.triangles}};},refresh(){resize();updateColors();return api;},destroy};

    root.addEventListener('click',rootClick);viewport.addEventListener('keydown',keydown);renderer.domElement.addEventListener('pointerdown',pointerDown);renderer.domElement.addEventListener('pointermove',pointerMove);renderer.domElement.addEventListener('pointerup',pointerUp);renderer.domElement.addEventListener('pointercancel',pointerUp);renderer.domElement.addEventListener('dblclick',doubleClick);renderer.domElement.addEventListener('wheel',wheel,{passive:false});
    const resizeObserver=global.ResizeObserver?new ResizeObserver(resize):null;resizeObserver?.observe(viewport);
    updateLayout();updateColors();updateOperatorLabels();renderInspector();updateButtons();resize();fit();state.ready=true;root.dataset.ready='true';loading.hidden=true;requestRender();options.onReady?.(api,api.getSceneStats());return api;
  }

  global.PtoModelParallelRankDeck={
    DEFAULT_TOPOLOGY,ISOMETRIC_POSE,normalizeTopology,enumerateRanks,partitionModel,layoutPosition,
    normalizeGroupingAxes,toggleGroupingAxis,buildHierarchicalRankLayout,payloadPoint,sampleSvgPath,parseTemplate,buildModelSceneSpec,render:mount,mount
  };
})(window);
