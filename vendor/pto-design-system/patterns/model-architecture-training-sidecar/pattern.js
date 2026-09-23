(function attachPtoModelArchitectureTrainingSidecar(global){
  'use strict';

  const NS='http://www.w3.org/2000/svg';
  const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
  const esc=(value)=>String(value??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const qs=(value,root=document)=>typeof value==='string'?root.querySelector(value):value;
  const seeded=(layer,salt=0)=>{
    const raw=Math.sin((layer+1)*12.9898+(salt+1)*78.233)*43758.5453;
    return raw-Math.floor(raw);
  };

  function stageForLayer(layer,stageRanges=[]){
    const stage=stageRanges.findIndex(([lo,hi])=>layer>=lo&&layer<=hi);
    return stage<0?null:stage;
  }

  function previewSnapshot(layer,context={}){
    const stage=context.stage??stageForLayer(layer,context.stageRanges);
    const dense=layer<(context.firstMoeLayer??0);
    const mean=(seeded(layer,1)-.5)*.018;
    const std=.91+seeded(layer,2)*.16;
    const amax=6.8+seeded(layer,3)*2.9;
    const gradNorm=.42+seeded(layer,4)*.72;
    const weightNorm=21+seeded(layer,5)*11;
    const updateRatio=1.2e-4+seeded(layer,6)*2.2e-4;
    const latency=(dense?1.18:1.86)+seeded(layer,7)*(dense?.28:.54);
    const optimizerStep=context.step??18420,microbatch=context.microbatch??3,microbatchCount=context.microbatchCount??8;
    const lastLayer=context.layerCount==null?null:Math.max(0,context.layerCount-1),rank=context.rank??null;
    return{
      step:optimizerStep,microbatch,microbatchCount,rank,stage,layer,tensorName:'layer_output_hidden_state',
      module:{kind:dense?'Attention + Dense FFN':'Attention + MoE',dense},
      tensors:{hidden:{mean,std,min:-amax*(.86+seeded(layer,8)*.12),max:amax,amax,norm:std*(87+seeded(layer,9)*16)},activationGradient:{norm:gradNorm,amax:gradNorm*(4.4+seeded(layer,10)*2.2)}},
      parameter:{
        weightNorm,updateRatio,optimizerStep,weightVersionBefore:optimizerStep,weightVersionAfter:optimizerStep+1,
        gradientContribution:{microbatch,norm:gradNorm*(.36+seeded(layer,11)*.22)},
        accumulatedGradient:{microbatches:microbatchCount,total:microbatchCount,norm:gradNorm*(.78+seeded(layer,12)*.45)}
      },
      optimizer:{accumulated:microbatchCount,total:microbatchCount,lossScale:65536,clipThreshold:1,overflow:false,state:'m / v sharded'},
      metric:{std,norm:std,amax,latency,loss:layer===lastLayer?2.84:null}
    };
  }

  function tip(node,text,detail=null){
    if(!node||!text)return node;
    const fallback={title:String(text).split('·')[0].trim(),category:'Annotation',definition:String(text),status:'info',statusLabel:'语义说明'};
    const payload=detail||fallback;node.dataset.tip=String(text);node.dataset.detail=JSON.stringify(payload);node.dataset.detailKey=String(payload.key||text);node.setAttribute('aria-label',`${String(text)} · 点击查看详情`);return node;
  }

  function label(container,className,text,x,y,tooltip=text,detail=null){
    const node=document.createElement('div');
    node.className=className;node.textContent=text;node.style.left=`${x}px`;node.style.top=`${y}px`;container.appendChild(node);return tip(node,tooltip,detail);
  }

  function path(points,className,attrs={}){
    const node=document.createElementNS(NS,'path');node.setAttribute('class',className);node.setAttribute('d',points);Object.entries(attrs).forEach(([key,value])=>node.setAttribute(key,value));return node;
  }

  function circle(x,y,r,className){
    const node=document.createElementNS(NS,'circle');node.setAttribute('class',className);node.setAttribute('cx',x.toFixed(1));node.setAttribute('cy',y.toFixed(1));node.setAttribute('r',String(r));return node;
  }

  function mount(rootInput,options={}){
    const root=qs(rootInput);if(!root||!global.PtoModelArchitecture3dDeck)return null;
    const metricViewOptions=[
      ['numerics','数值'],
      ['gradient','梯度'],
      ['performance','性能'],
      ['moe','MoE'],
      ['training','训练']
    ];
    const metricViewValues=new Set(metricViewOptions.map(([value])=>value));
    const requestedMetricViews=Array.isArray(options.metricViews)?options.metricViews:(options.metricView&&options.metricView!=='structure'?[options.metricView]:[]);
    let controller=null,raf=0,focusFitRaf=0,focusTransitionTimer=0,destroyed=false,selectedLayer=null,pendingLayer=null,selectedDetail=null,selectedDetailKey=null,lossExpanded=false,fitZoom=1,dragging=false,hoveredLayer=null,measurementCache=null,structureDirty=true,focusRestorePose=null;
    let activeMetricViews=new Set(requestedMetricViews.filter(value=>metricViewValues.has(value)));
    const context={step:options.step??18420,microbatch:options.microbatch??3,microbatchCount:options.microbatchCount??8};
    const topology=options.topology||{};
    const snapshots=new Map(Object.entries(options.snapshots||{}).map(([layer,data])=>[Number(layer),data]));
    const snapshotFor=(layer)=>snapshots.get(Number(layer))||previewSnapshot(Number(layer),{...context,stageRanges:controller?.config?.stageRanges,layerCount:controller?.config?.layerCount,firstMoeLayer:controller?.config?.firstMoeLayer,rank:topology.rank??null});
    const metricValue=(snapshot,key)=>key==='std'?(snapshot.metric?.std??snapshot.metric?.norm??snapshot.tensors?.hidden?.std):snapshot.metric?.[key];
    const presetConfig=typeof options.preset==='string'?global.PtoModelArchitecture3dDeck.PRESETS?.[options.preset]:options.preset;
    const sourceConfig=options.config||presetConfig||{},sourceDepthGap=Number(sourceConfig.depthGap)||46,sideDepthGap=Number(options.sideDepthGap)||Math.round(sourceDepthGap*1.26);
    const baseOptions={...options,config:{...sourceConfig,depthGap:sideDepthGap},showSideLabels:false,initialView:options.initialView||'right',onViewChange(view,api){options.onViewChange?.(view,api);queueSidecarFit();},onThemeChange(theme,api){options.onThemeChange?.(theme,api);schedule();},onParallelModeChange(mode,api){options.onParallelModeChange?.(mode,api);schedule();},onZoomChange(zoom,api){options.onZoomChange?.(zoom,api);syncOverlayTransform();}};
    controller=global.PtoModelArchitecture3dDeck.render(root,baseOptions);if(!controller)return null;
    root.classList.add('pto-model-training-sidecar');root.dataset.trainingSidecar='true';
    const viewport=root.querySelector('.pto-model-deck__viewport');
    const overlayScene=document.createElement('div');overlayScene.className='pto-training-sidecar__scene';overlayScene.setAttribute('aria-label','Training semantics canvas scene');
    const focusScene=document.createElement('div');focusScene.className='pto-training-sidecar__focus-scene';focusScene.setAttribute('aria-label','Expanded Layer slice scene');
    const svg=document.createElementNS(NS,'svg');svg.classList.add('pto-training-sidecar__svg');svg.setAttribute('aria-label','Training semantics overlay');
    const labels=document.createElement('div');labels.className='pto-training-sidecar__labels';
    const layerHits=document.createElementNS(NS,'svg');layerHits.classList.add('pto-training-sidecar__layer-hits');layerHits.setAttribute('aria-label','Decoder Layer column hit areas');
    const focus=document.createElement('aside');focus.className='pto-training-sidecar__focus';focus.setAttribute('aria-live','polite');
    const dataControl=document.createElement('div');dataControl.className='pto-model-deck__control pto-training-sidecar__data-control';dataControl.dataset.stageUi='';
    dataControl.innerHTML=`<button class="pto-model-deck__button pto-training-sidecar__data-toggle" type="button" data-metric-menu-toggle aria-haspopup="true" aria-expanded="false">数据<span data-metric-count></span><i aria-hidden="true">⌄</i></button>
      <div class="pto-training-sidecar__data-menu" role="menu" aria-label="数据标注多选">
        <div class="pto-training-sidecar__data-actions">
          <button type="button" data-metric-action="all">全选</button>
          <button type="button" data-metric-action="none">全部隐藏</button>
        </div>
        <div class="pto-training-sidecar__data-options">${metricViewOptions.map(([value,labelText])=>`<button type="button" role="menuitemcheckbox" aria-checked="false" data-metric-view="${value}"><span aria-hidden="true">✓</span>${labelText}</button>`).join('')}</div>
      </div>`;
    const toolbar=root.querySelector('.pto-model-deck__toolbar');
    if(toolbar)toolbar.insertBefore(dataControl,toolbar.children[2]||null);
    const tooltip=document.createElement('div');tooltip.className='pto-training-sidecar__tooltip';tooltip.setAttribute('role','tooltip');
    const detailPanel=document.createElement('aside');detailPanel.className='pto-training-sidecar__inspector';detailPanel.setAttribute('aria-live','polite');
    overlayScene.append(svg,labels,layerHits);focusScene.append(focus);viewport.append(overlayScene,focusScene,tooltip,detailPanel);

    function metricSummary(summaries){
      return metricViewOptions.flatMap(([value])=>activeMetricViews.has(value)?summaries[value]||[]:[]);
    }

    function syncMetricViews(){
      const values=metricViewOptions.map(([value])=>value).filter(value=>activeMetricViews.has(value));
      root.dataset.metricViews=values.join(' ')||'none';
      dataControl.querySelectorAll('[data-metric-view]').forEach(button=>{
        const active=activeMetricViews.has(button.dataset.metricView);
        button.classList.toggle('is-selected',active);button.setAttribute('aria-checked',String(active));
      });
      const toggle=dataControl.querySelector('[data-metric-menu-toggle]'),count=dataControl.querySelector('[data-metric-count]');
      toggle?.classList.toggle('is-active',values.length>0);
      if(count)count.textContent=values.length?String(values.length):'';
      toggle?.setAttribute('aria-label',values.length?`数据标注：已显示 ${values.length} 类`:'数据标注：全部隐藏');
    }

    const metricElementSelector=[
      '.pto-training-sidecar__metric-band','.pto-training-sidecar__metric-midline','.pto-training-sidecar__metric-line','.pto-training-sidecar__metric-point',
      '.pto-training-sidecar__metric-label','.pto-training-sidecar__sample-label','.pto-training-sidecar__flow-texture','.pto-training-sidecar__hidden',
      '.pto-training-sidecar__residual-flow-arrow','.pto-training-sidecar__residual-flow-hit','.pto-training-sidecar__residual-flow-hit-dot',
      '.pto-training-sidecar__hidden-dot','.pto-training-sidecar__backward','.pto-training-sidecar__backward-dot','.pto-training-sidecar__parameter-stem',
      '.pto-training-sidecar__parameter-dot','.pto-training-sidecar__step-gradient-dot','.pto-training-sidecar__optimizer','.pto-training-sidecar__communication',
      '.pto-training-sidecar__comm-label','.pto-training-sidecar__value-label:not(.is-structure)','.pto-training-sidecar__loss-composition',
      '.pto-training-sidecar__focus-node-metrics','.pto-training-sidecar__focus-graph-aux','.pto-training-sidecar__focus-backward'
    ].join(',');
    const metricGroupSelectors={
      numerics:'.is-metric-std,.is-metric-amax,.pto-training-sidecar__sample-label,.pto-training-sidecar__residual-flow-arrow,.pto-training-sidecar__residual-flow-hit,.pto-training-sidecar__residual-flow-hit-dot,.pto-training-sidecar__focus-node-metrics',
      gradient:'.pto-training-sidecar__flow-texture.is-backward,.pto-training-sidecar__backward,.pto-training-sidecar__backward-dot,.pto-training-sidecar__parameter-stem,.pto-training-sidecar__parameter-dot,.pto-training-sidecar__step-gradient-dot,.pto-training-sidecar__optimizer,.pto-training-sidecar__value-label.is-backward,.pto-training-sidecar__value-label.is-optimizer,.pto-training-sidecar__focus-node-metrics,.pto-training-sidecar__focus-backward',
      performance:'.is-metric-latency,.pto-training-sidecar__sample-label,.pto-training-sidecar__communication,.pto-training-sidecar__comm-label,.pto-training-sidecar__focus-node-metrics',
      moe:'.pto-training-sidecar__focus-node-metrics,.pto-training-sidecar__focus-graph-aux',
      training:'.pto-training-sidecar__flow-texture.is-backward,.pto-training-sidecar__backward,.pto-training-sidecar__backward-dot,.pto-training-sidecar__parameter-stem,.pto-training-sidecar__parameter-dot,.pto-training-sidecar__step-gradient-dot,.pto-training-sidecar__optimizer,.pto-training-sidecar__value-label.is-backward,.pto-training-sidecar__value-label.is-optimizer,.pto-training-sidecar__value-label.is-checkpoint,.pto-training-sidecar__loss-composition,.pto-training-sidecar__focus-node-metrics,.pto-training-sidecar__focus-graph-aux,.pto-training-sidecar__focus-backward'
    };
    function applyMetricVisibility(){
      root.querySelectorAll(metricElementSelector).forEach(node=>node.classList.add('is-metric-hidden'));
      activeMetricViews.forEach(value=>root.querySelectorAll(metricGroupSelectors[value]||'').forEach(node=>node.classList.remove('is-metric-hidden')));
    }

    function syncOverlayTransform(){
      if(!controller||!overlayScene)return;
      [overlayScene,focusScene].forEach(scene=>{
        scene.style.left=`calc(50% + ${controller.state.panX}px)`;
        scene.style.top=`calc(50% + ${controller.state.panY}px)`;
        scene.style.transform=`scale(${controller.state.zoom})`;
      });
    }

    function geometry(){
      const width=viewport.clientWidth,height=viewport.clientHeight,base=viewport.getBoundingClientRect();
      const cards=Array.from(root.querySelectorAll('.pto-model-deck__layer[data-layer]'));
      if(!width||!height||!cards.length)return null;
      const zoom=Math.max(.001,controller.state.zoom),originX=width/2+controller.state.panX,originY=height/2+controller.state.panY;
      const worldX=(screenX)=>(screenX-originX)/zoom,worldY=(screenY)=>(screenY-originY)/zoom;
      const screenX=(worldValue)=>originX+worldValue*zoom,screenY=(worldValue)=>originY+worldValue*zoom;
      const operatorGeometry=(node)=>{
        const rect=node.getBoundingClientRect();if(rect.height<=.5)return null;
        const computed=getComputedStyle(node),color=computed.getPropertyValue('--node-color').trim()||'var(--foreground-muted)';
        return{node,op:node.dataset.op||'linear',color,top:worldY(rect.top-base.top),bottom:worldY(rect.bottom-base.top)};
      };
      const points=cards.map(card=>{
        const rect=card.getBoundingClientRect(),operators=Array.from(card.querySelectorAll('.pto-model-deck__node,.pto-model-deck__experts')).map(operatorGeometry).filter(Boolean);
        const cardTop=worldY(rect.top-base.top),cardBottom=worldY(rect.bottom-base.top),operatorTop=operators.length?Math.min(...operators.map(item=>item.top)):cardTop,operatorBottom=operators.length?Math.max(...operators.map(item=>item.bottom)):cardBottom;
        return{layer:Number(card.dataset.layer),stage:Number(card.dataset.stage),x:worldX(rect.left+rect.width/2-base.left),top:operatorTop,bottom:operatorBottom,cardTop,cardBottom,operatorTop,operatorBottom,operators,card,rect};
      }).sort((a,b)=>a.layer-b.layer);
      const modelTop=Math.min(...points.map(point=>point.operatorTop)),modelBottom=Math.max(...points.map(point=>point.operatorBottom));
      const centerOf=(selector)=>{const node=root.querySelector(selector),rect=node?.getBoundingClientRect();return rect?{x:worldX(rect.left+rect.width/2-base.left),y:worldY(rect.top+rect.height/2-base.top)}:null;};
      const input=centerOf('.pto-model-deck__static--input [data-node="embedding"]')||{x:points[0].x-42,y:modelTop};
      const output=centerOf('.pto-model-deck__static--output [data-node="final_norm"]')||{x:points[points.length-1].x+42,y:modelTop};
      const residualConnector=root.querySelector('.pto-model-deck__side-residual-connector[data-state-rail="0"]'),residualMatch=residualConnector?.getAttribute('d')?.match(/([-\d.]+),([-\d.]+)$/);
      const residualInputStart=residualMatch?{x:worldX(Number(residualMatch[1])),y:worldY(Number(residualMatch[2]))}:{x:input.x,y:modelTop-4/Math.max(.001,fitZoom)};
      const residualRails=Array.from(root.querySelectorAll('.pto-model-deck__side-residual-spine[data-state-rail]')).map(node=>{
        const match=node.getAttribute('d')?.match(/^M([-\d.]+),([-\d.]+)L([-\d.]+),([-\d.]+)$/);if(!match)return null;
        return{rail:Number(node.dataset.stateRail),x0:worldX(Number(match[1])),y:worldY(Number(match[2])),x1:worldX(Number(match[3]))};
      }).filter(Boolean).sort((a,b)=>a.rail-b.rail);
      const gap=points.length>1?Math.max(4/fitZoom,Math.abs(points[1].x-points[0].x)):12/fitZoom,scale=1/Math.max(.001,fitZoom);
      const axisY=modelTop-(activeMetricViews.size?250:150)*scale;
      const stageY=axisY-32*scale;
      const metricY=axisY+38*scale;
      const inputSummaryY=modelTop-78*scale;
      const hiddenY=modelTop-60*scale;
      const embeddingY=modelTop-34*scale;
      const backwardY=modelBottom+28*scale;
      const parameterY=backwardY+34*scale;
      const optimizerY=parameterY+38*scale;
      return{width,height,base,points,modelTop,modelBottom,input,output,residualInputStart,residualRails,gap,scale,axisY,stageY,metricY,inputSummaryY,hiddenY,embeddingY,backwardY,parameterY,optimizerY,worldX,worldY,screenX,screenY};
    }

    function layerColumnWidth(g){
      return 10*g.scale;
    }

    function operatorSideWidth(g){
      return 2*g.scale;
    }

    function renderLayerBackgrounds(g){
      const width=layerColumnWidth(g),padding=2*g.scale;
      g.points.forEach(point=>{
        const background=document.createElementNS(NS,'rect');
        background.setAttribute('class',`pto-training-sidecar__layer-column${point.layer===selectedLayer?' is-flipped':''}`);
        background.dataset.layer=String(point.layer);
        background.setAttribute('x',(point.x-width/2).toFixed(1));
        background.setAttribute('y',(point.operatorTop-padding).toFixed(1));
        background.setAttribute('width',width.toFixed(1));
        background.setAttribute('height',Math.max(4*g.scale,point.operatorBottom-point.operatorTop+padding*2).toFixed(1));
        background.setAttribute('rx',(2*g.scale).toFixed(1));
        svg.appendChild(background);
      });
    }

    function renderOperatorSides(g){
      const width=operatorSideWidth(g),halfWidth=width/2,seen=new Set();
      g.points.forEach(point=>{
        point.operators.forEach(operator=>{
          const key=`${point.layer}|${operator.op}|${operator.top.toFixed(1)}|${operator.bottom.toFixed(1)}`;if(seen.has(key))return;seen.add(key);
          const bar=document.createElementNS(NS,'rect');
          bar.setAttribute('class',`pto-training-sidecar__operator-side${point.layer===selectedLayer?' is-flipped':''}`);
          bar.dataset.layer=String(point.layer);bar.dataset.op=operator.op;
          bar.setAttribute('x',(point.x-halfWidth).toFixed(1));
          bar.setAttribute('y',operator.top.toFixed(1));
          bar.setAttribute('width',width.toFixed(1));
          bar.setAttribute('height',Math.max(g.scale,operator.bottom-operator.top).toFixed(1));
          bar.setAttribute('rx',Math.min(g.scale,width/2).toFixed(1));
          bar.style.setProperty('--operator-side-color',operator.color);
          svg.appendChild(bar);
        });
      });
    }

    function syncHoveredLayer(){
      svg.querySelectorAll('.pto-training-sidecar__layer-column.is-hovered,.pto-training-sidecar__operator-side.is-hovered').forEach(node=>node.classList.remove('is-hovered'));
      if(!Number.isFinite(hoveredLayer))return;
      svg.querySelectorAll(`.pto-training-sidecar__layer-column[data-layer="${hoveredLayer}"],.pto-training-sidecar__operator-side[data-layer="${hoveredLayer}"]`).forEach(node=>node.classList.add('is-hovered'));
    }

    function setHoveredLayer(layer){
      const next=layer===null||layer===undefined||!Number.isFinite(Number(layer))?null:Number(layer);if(next===hoveredLayer)return;
      hoveredLayer=next;syncHoveredLayer();
    }

    function flowTexture(x0,x1,y,direction,scale,anchors=[]){
      const group=document.createElementNS(NS,'g'),height=16*scale,rx=4*scale;
      group.setAttribute('class',`pto-training-sidecar__flow-texture is-${direction}`);
      const base=document.createElementNS(NS,'rect');base.setAttribute('class','pto-training-sidecar__flow-texture-base');base.setAttribute('x',x0.toFixed(1));base.setAttribute('y',(y-height/2).toFixed(1));base.setAttribute('width',Math.max(1,x1-x0).toFixed(1));base.setAttribute('height',height.toFixed(1));base.setAttribute('rx',rx.toFixed(1));
      group.append(base);
      const points=[...new Set(anchors.map(Number).filter(Number.isFinite))].sort((a,b)=>a-b),halfHeight=5*scale;
      for(let index=0;index<points.length-1;index+=1){
        const left=points[index],right=points[index+1],gap=right-left,x=(left+right)/2,halfWidth=Math.min(4*scale,gap*.26);
        const d=direction==='forward'?`M${(x-halfWidth).toFixed(1)} ${(y-halfHeight).toFixed(1)}L${(x+halfWidth).toFixed(1)} ${y.toFixed(1)}L${(x-halfWidth).toFixed(1)} ${(y+halfHeight).toFixed(1)}`:`M${(x+halfWidth).toFixed(1)} ${(y-halfHeight).toFixed(1)}L${(x-halfWidth).toFixed(1)} ${y.toFixed(1)}L${(x+halfWidth).toFixed(1)} ${(y+halfHeight).toFixed(1)}`;
        group.append(path(d,'pto-training-sidecar__flow-texture-arrow'));
      }
      return group;
    }

    function metricDetail(key,name,value,min,max,layer){
      const definitions={
        std:'Layer 输出 hidden state 的标准差 σ，用于观察激活分布的离散程度与层间突变；它不是 L2/RMS Norm。',
        amax:'张量绝对值的最大值 max(|x|)，常用于观察动态范围、量化溢出风险和异常尖峰。',
        latency:'该 Layer 一次前向执行的示意耗时，用于定位计算或通信热点；实际值必须绑定 step、microbatch、rank 和设备事件。'
      };
      const units=key==='latency'?' ms':'',span=max-min,near=value<min+span*.1||value>max-span*.1;
      const status=value<min||value>max?'abnormal':near?'warning':'normal';
      const statusLabel=status==='abnormal'?'异常：超出参考范围':status==='warning'?'关注：接近参考边界':'正常：位于参考范围';
      const snapshot=snapshotFor(layer);
      return{title:`${name} · L${layer}`,category:'Metric',definition:definitions[key],values:[['当前值',`${Number(value).toFixed(key==='latency'?2:3)}${units}`],['参考范围',`${min}–${max}${units}`]],status,statusLabel,context:`step ${snapshot.step} · MB ${snapshot.microbatch}/${snapshot.microbatchCount} · PP${snapshot.stage??'?'} · layer ${layer} · tensor ${snapshot.tensorName}`};
    }

    function rankGroupForStage(stage){
      const groups=topology.rankGroups;if(groups==null||stage==null)return null;
      const group=Array.isArray(groups)?groups[stage]:(groups[stage]??groups[`pp${stage}`]??groups[`PP${stage}`]);
      if(group==null)return null;
      return Array.isArray(group)?group.join(', '):String(group);
    }

    function parallelSummary(){
      if(topology.parallelGroups)return String(topology.parallelGroups);
      return (controller.config.annotations||[]).filter(item=>item.kind!=='pp').map(item=>item.label.split(' · ')[0].replace(/\s+/g,' ')).join(' · ');
    }

    function executionContext(snapshot,extra=''){
      const group=rankGroupForStage(snapshot.stage),rank=snapshot.rank!=null?`rank ${snapshot.rank}`:group?`ranks ${group}`:parallelSummary()||'rank map unavailable';
      return `step ${snapshot.step} · MB ${snapshot.microbatch}/${snapshot.microbatchCount} · PP${snapshot.stage??'?'} · ${rank}${extra?` · ${extra}`:''}`;
    }

    function attentionDescriptor(layer){
      const dsa=(controller.config.dsaLayers||[]).includes(layer);
      return dsa
        ?{kind:'DSA',label:`DSA · Sparse Global Attention · index top-k ${controller.config.indexTopK??'configured'}`,kernel:'FlashAttention kernel'}
        :{kind:'SWA',label:`SWA · Sliding Window ${controller.config.slidingWindow??'configured'}`,kernel:'FlashAttention kernel'};
    }

    function communicationDetail(lo,direction,key){
      const forward=direction==='forward',sourceLayer=forward?lo-1:lo,snapshot=snapshotFor(sourceLayer),sourceStage=forward?snapshot.stage:snapshot.stage,destinationStage=forward?sourceStage+1:sourceStage-1,latency=.18+seeded(lo,forward?17:18)*.14;
      if(forward){
        const hidden=snapshot.tensors.hidden,status=hidden.amax>9.7?'abnormal':hidden.amax>9.35?'warning':'normal',sourceRanks=rankGroupForStage(sourceStage),destinationRanks=rankGroupForStage(destinationStage);
        const payload=32,bandwidth=payload/1024/(latency/1000);
        return{key,title:`PP forward boundary · L${lo-1} → L${lo}`,category:'PP Communication',definition:'相邻 Pipeline Stage 之间发送/接收前向 activation hidden state。普通 Layer 边界是本地张量依赖，只有该 PP 边界发生 P2P 通信。',values:[['Boundary',`L${lo-1} → L${lo}`],['Source',`PP${sourceStage}${sourceRanks?` · ranks ${sourceRanks}`:''}`],['Destination',`PP${destinationStage}${destinationRanks?` · ranks ${destinationRanks}`:''}`],['Tensor',`H${lo} ×4`],['Payload',`${payload.toFixed(1)} MB`],['Bandwidth',`${bandwidth.toFixed(1)} GB/s`],['P2P latency',`${latency.toFixed(2)} ms`]],status,statusLabel:status==='abnormal'?'异常：边界 activation Amax 超出参考范围':status==='warning'?'关注：边界 activation Amax 接近参考边界':'正常：边界 activation 统计位于参考范围',context:`${executionContext(snapshot)} · PP${sourceStage} → PP${destinationStage} · Send/Recv forward activation`};
      }
      const gradient=snapshot.tensors.activationGradient,status=gradient.norm>1.2?'warning':'normal',sourceRanks=rankGroupForStage(sourceStage),destinationRanks=rankGroupForStage(destinationStage);
      const payload=32,bandwidth=payload/1024/(latency/1000);
      return{key,title:`PP backward boundary · L${lo} → L${lo-1}`,category:'PP Communication',definition:'相邻 Pipeline Stage 之间发送/接收反向 activation gradient ∂L/∂H；跨边界传输的不是参数梯度 ∂L/∂W。',values:[['Boundary',`L${lo} → L${lo-1}`],['Source',`PP${sourceStage}${sourceRanks?` · ranks ${sourceRanks}`:''}`],['Destination',`PP${destinationStage}${destinationRanks?` · ranks ${destinationRanks}`:''}`],['Tensor','∂L/∂H ×4'],['Payload',`${payload.toFixed(1)} MB`],['Bandwidth',`${bandwidth.toFixed(1)} GB/s`],['P2P latency',`${latency.toFixed(2)} ms`]],status,statusLabel:status==='warning'?'关注：边界激活梯度偏高':'正常：边界激活梯度位于参考范围',context:`${executionContext(snapshot)} · PP${sourceStage} → PP${destinationStage} · Send/Recv backward activation gradient`};
    }

    function internalOperatorMetric(layer,label,op,index,nodeId){
      const snapshot=snapshotFor(layer),sample=(salt,min,max)=>min+seeded(layer,index*7+salt)*(max-min);
      const mean=sample(31,-.008,.008),std=sample(32,.72,1.08),amax=sample(33,5.4,9.65),grad=sample(34,.28,1.08);
      const latencyBase=/expert/i.test(`${label} ${op}`)?.56:/attention/i.test(`${label} ${op}`)?.24:/comm|gather|scatter|dispatch|combine/i.test(`${label} ${op}`)?.13:/norm|add|merge/i.test(`${label} ${op}`)?.03:.09;
      const latency=latencyBase+sample(35,.01,latencyBase*.62+.04),values=[],summaries={numerics:[],gradient:[],performance:[],moe:[],training:[]},descriptor=`${label} ${op}`;
      let definition='该算子行展示与其对象类型匹配的诊断指标。',schema='Tensor',status=amax>9.35?'warning':'normal';
      if(/Router/i.test(descriptor)){
        schema='Router';
        const pmax=sample(36,.48,.72),imbalance=sample(37,1.02,1.18),entropy=sample(38,1.56,2.08);
        summaries.moe.push(`Pmax ${pmax.toFixed(2)}`,`H ${entropy.toFixed(2)}`,`imb ${imbalance.toFixed(2)}×`);summaries.performance.push(`${latency.toFixed(2)}ms`);summaries.training.push(`imb ${imbalance.toFixed(2)}×`);values.push(['Router Pmax',pmax.toFixed(3)],['Routing entropy',entropy.toFixed(2)],['Expert imbalance',`${imbalance.toFixed(2)}×`],['Latency',`${latency.toFixed(2)} ms`]);definition='Router 根据 token hidden state 产生 expert 概率与 Top-k routing map；诊断关注概率集中度、熵与 expert 负载失衡。';status=imbalance>1.15?'warning':status;
      }else if(/Dispatch|Combine|AllGather|Reduce-Scatter|comm/i.test(descriptor)){
        schema='Communication';
        const tokens=Math.round(sample(39,3584,4608)),bytes=sample(40,24,38);
        const bandwidth=bytes/1024/(latency/1000);summaries.performance.push(`${bytes.toFixed(0)}MB`,`${bandwidth.toFixed(0)}GB/s`,`${latency.toFixed(2)}ms`);if(/Dispatch|Combine/i.test(descriptor))summaries.moe.push(`${tokens} tok`,`${bytes.toFixed(0)}MB`);values.push(['Token count',tokens],['Payload',`${bytes.toFixed(1)} MB`],['Bandwidth',`${bandwidth.toFixed(1)} GB/s`],['Latency',`${latency.toFixed(2)} ms`]);definition=/Dispatch|Combine/i.test(label)?'Layer 内部的 EP token dispatch/combine 通信；通信对象只展示 payload、带宽和延迟。':'该算子执行 Layer 内部的集合通信；通信节点没有独立 parameter gradient。';
      }else if(!/mHC/i.test(descriptor)&&/Attention Core|DSA|SWA|FlashAttention/i.test(descriptor)){
        schema='Attention tensor';
        const logit=sample(41,11.8,15.6),pmax=sample(42,.46,.76),rowError=sample(43,.00004,.00032);
        summaries.numerics.push(`logit ±${logit.toFixed(1)}`,`Pmax ${pmax.toFixed(2)}`);summaries.performance.push(`${latency.toFixed(2)}ms`);values.push(['Attention mode',attentionDescriptor(layer).label],['Kernel',attentionDescriptor(layer).kernel],['Logits range',`−${logit.toFixed(1)}…${logit.toFixed(1)}`],['Probability max',pmax.toFixed(3)],['Softmax row-sum error',rowError.toExponential(1)],['Latency',`${latency.toFixed(2)} ms`]);definition='Attention 语义由 Layer 配置决定；FlashAttention 只作为底层 kernel 单独记录。logits 与 Softmax probability 保持为不同对象。';
      }else if(/Expert/i.test(descriptor)){
        schema='Expert';
        const tokens=Math.round(sample(44,36,76)),load=sample(45,.72,.96);
        summaries.moe.push(`${tokens} tok`,`${(load*100).toFixed(0)}% cap`);summaries.numerics.push(`Amax ${amax.toFixed(1)}`);summaries.performance.push(`${latency.toFixed(2)}ms`);values.push(['Assigned tokens',tokens],['Capacity utilization',`${(load*100).toFixed(0)}%`],['Output Amax',amax.toFixed(2)],['Latency',`${latency.toFixed(2)} ms`]);definition='Expert 对分配到本地的 token 执行 FFN；诊断关注 token 负载、容量利用率、输出范围和耗时。';
      }else if(/mHC|Residual Add|Merge|Routed \+ Shared Add/i.test(descriptor)){
        schema='Residual / mHC';
        const inputNorm=sample(46,82,108),outputNorm=inputNorm*sample(47,.94,1.09),delta=Math.abs(outputNorm-inputNorm)/inputNorm,mixing=sample(48,.06,.22);
        summaries.numerics.push(`in ${inputNorm.toFixed(0)}`,`Δ ${(delta*100).toFixed(1)}%`,`mix ${mixing.toFixed(2)}`);values.push(['Input L2',inputNorm.toFixed(1)],['Output L2',outputNorm.toFixed(1)],['Increment ratio',`${(delta*100).toFixed(2)}%`],['Stream mixing spread',mixing.toFixed(3)],['Streams',controller.config.mhcStreams??4]);definition=/mHC/i.test(descriptor)?'mHC Mix/Merge 在四路状态之间执行训练配置定义的融合；这里不将未知融合矩阵假画为标准 Add。':'该分支汇合节点关注输入/输出范数与增量比例。';
      }else if(/Norm/i.test(descriptor)){
        schema='Norm';
        const inputRms=sample(49,.84,1.17),outputRms=sample(50,.97,1.03),scaleDrift=sample(51,-.018,.018);
        summaries.numerics.push(`in ${inputRms.toFixed(2)}`,`out ${outputRms.toFixed(2)}`,`γΔ ${scaleDrift.toFixed(3)}`);summaries.performance.push(`${latency.toFixed(2)}ms`);values.push(['Input RMS',inputRms.toFixed(3)],['Output RMS',outputRms.toFixed(3)],['Scale drift',scaleDrift.toFixed(4)],['Latency',`${latency.toFixed(2)} ms`]);definition='Norm 诊断关注归一化前后 RMS 与可学习 scale 的漂移，不套用通用 activation 指标集合。';
      }else if(/Linear|Projection|Weight/i.test(descriptor)){
        schema='Parameter operator';
        const accumulated=snapshot.parameter.accumulatedGradient;
        summaries.gradient.push(`W ${snapshot.parameter.weightNorm.toFixed(1)}`,`∇ ${accumulated.norm.toFixed(2)}`,`Δ/W ${snapshot.parameter.updateRatio.toExponential(1)}`);summaries.training.push(`W v${snapshot.parameter.weightVersionBefore}→${snapshot.parameter.weightVersionAfter}`);summaries.performance.push(`${latency.toFixed(2)}ms`);values.push(['Weight L2',snapshot.parameter.weightNorm.toFixed(1)],['Step gradient L2',accumulated.norm.toFixed(3)],['Update / Weight',snapshot.parameter.updateRatio.toExponential(2)],['Weight version',`${snapshot.parameter.weightVersionBefore} → ${snapshot.parameter.weightVersionAfter}`],['Latency',`${latency.toFixed(2)} ms`]);definition='带参数算子展示 weight、step-accumulated gradient 与 update ratio；参数版本在 optimizer step 前后显式区分。';
      }else{
        summaries.numerics.push(`σ ${std.toFixed(2)}`,`Amax ${amax.toFixed(1)}`);summaries.gradient.push(`∇ ${grad.toFixed(2)}`);summaries.performance.push(`${latency.toFixed(2)}ms`);values.push(['Mean',mean.toFixed(4)],['Std σ',std.toFixed(3)],['Amax',amax.toFixed(2)],['Activation gradient L2',grad.toFixed(3)],['Latency',`${latency.toFixed(2)} ms`]);
      }
      const summary=metricSummary(summaries);
      return{summary,detail:{key:`layer-${layer}-operator-${nodeId}`,title:`${label} · L${layer}`,category:`${schema} metric`,definition,values,status,statusLabel:status==='warning'?'关注：至少一项接近参考边界':'正常：指标位于参考范围',context:executionContext(snapshot,`L${layer} · ${nodeId}`)}};
    }

    function positionDetailPanel(){
      if(!selectedDetail||!detailPanel.classList.contains('is-open'))return;
      const target=selectedDetailKey
        ?focus.querySelector(`[data-detail-key="${CSS.escape(selectedDetailKey)}"]`)||root.querySelector(`[data-detail-key="${CSS.escape(selectedDetailKey)}"]`)
        :null;
      const viewportRect=viewport.getBoundingClientRect(),panelWidth=detailPanel.offsetWidth||330,panelHeight=detailPanel.offsetHeight||414,gap=12,padding=12;
      if(!target){
        detailPanel.style.left='auto';detailPanel.style.right='16px';detailPanel.style.top='108px';detailPanel.style.bottom='auto';return;
      }
      const targetRect=target.getBoundingClientRect(),targetLeft=targetRect.left-viewportRect.left,targetRight=targetRect.right-viewportRect.left,targetCenterY=targetRect.top-viewportRect.top+targetRect.height/2;
      const rightCandidate=targetRight+gap,leftCandidate=targetLeft-panelWidth-gap;
      const left=rightCandidate+panelWidth<=viewportRect.width-padding
        ?rightCandidate
        :leftCandidate>=padding
          ?leftCandidate
          :clamp(targetLeft+(targetRect.width-panelWidth)/2,padding,Math.max(padding,viewportRect.width-panelWidth-padding));
      const top=clamp(targetCenterY-panelHeight/2,padding,Math.max(padding,viewportRect.height-panelHeight-padding));
      detailPanel.style.left=`${left}px`;detailPanel.style.right='auto';detailPanel.style.top=`${top}px`;detailPanel.style.bottom='auto';
    }

    function openDetail(detail,target=null){
      selectedDetail=detail;selectedDetailKey=detail?(target?.dataset.detailKey||detail.key||null):null;if(!detail){detailPanel.classList.remove('is-open');detailPanel.replaceChildren();return api;}
      if(!selectedDetailKey){detailPanel.style.left='auto';detailPanel.style.right='16px';detailPanel.style.top='108px';}
      const values=(detail.values||[]).map(([key,value])=>`<div><span>${esc(key)}</span><b>${esc(value)}</b></div>`).join('');
      detailPanel.innerHTML=`<header><div><span>${esc(detail.category||'Annotation')}</span><h3>${esc(detail.title||'Detail')}</h3></div><button type="button" aria-label="关闭详情">×</button></header><p>${esc(detail.definition||'')}</p>${values?`<section class="pto-training-sidecar__inspector-values">${values}</section>`:''}<section class="pto-training-sidecar__assessment is-${esc(detail.status||'info')}"><b>${esc(detail.statusLabel||'语义说明')}</b>${detail.reference?`<span>${esc(detail.reference)}</span>`:''}</section>${detail.context?`<footer>${esc(detail.context)}</footer>`:''}`;
      detailPanel.classList.add('is-open');tooltip.classList.remove('is-visible');requestAnimationFrame(positionDetailPanel);return api;
    }

    detailPanel.addEventListener('click',event=>{if(event.target.closest('button'))openDetail(null);});

    function operatorRows(g){
      const rows=(controller.config.sideRows||[]).map(row=>{
        const nodes=g.points.flatMap(point=>row.ids.map(id=>point.card.querySelector(`[data-node="${CSS.escape(id)}"]`)).filter(Boolean));
        const rects=nodes.map(node=>node.getBoundingClientRect()).filter(rect=>rect.height>0);if(!rects.length)return null;
        const left=g.worldX(Math.min(...rects.map(rect=>rect.left))-g.base.left)-4*g.scale,right=g.worldX(Math.max(...rects.map(rect=>rect.right))-g.base.left)+4*g.scale;
        const top=g.worldY(Math.min(...rects.map(rect=>rect.top))-g.base.top)-3*g.scale,bottom=g.worldY(Math.max(...rects.map(rect=>rect.bottom))-g.base.top)+3*g.scale;
        const pseudo=getComputedStyle(nodes[0],'::after'),computed=getComputedStyle(nodes[0]),color=pseudo.backgroundColor&&pseudo.backgroundColor!=='rgba(0, 0, 0, 0)'?pseudo.backgroundColor:computed.getPropertyValue('--node-color').trim()||'var(--foreground-muted)';
        return{label:row.label,ids:row.ids,nodes,left,right,top,bottom,y:(top+bottom)/2,color};
      }).filter(Boolean).sort((a,b)=>a.y-b.y||a.left-b.left);
      const merged=[],mergeDistance=12*g.scale;
      rows.forEach(item=>{
        const match=merged.find(candidate=>{
          const overlap=Math.max(0,Math.min(candidate.right,item.right)-Math.max(candidate.left,item.left));
          return Math.abs(candidate.y-item.y)<mergeDistance&&overlap>Math.min(candidate.right-candidate.left,item.right-item.left)*.55;
        });
        if(!match){merged.push({...item});return;}
        match.label=`${match.label} / ${item.label}`;match.ids=[...new Set([...match.ids,...item.ids])];match.nodes.push(...item.nodes);match.left=Math.min(match.left,item.left);match.right=Math.max(match.right,item.right);match.top=Math.min(match.top,item.top);match.bottom=Math.max(match.bottom,item.bottom);match.y=(match.top+match.bottom)/2;
      });
      return merged;
    }

    function renderOperatorBands(g,measuredRows=operatorRows(g)){
      measuredRows.forEach(item=>{
        const detail={title:item.label,category:'Operator row',definition:'该框对应侧视投影中同一计算高度上的算子集合；横向范围表示这些算子在哪些 Decoder Layers 中存在。',values:[['Operator IDs',item.ids.join(', ')],['投影范围',`${Math.round(item.left)}–${Math.round(item.right)} px`],['数据来源','模型结构']],status:'info',statusLabel:'结构标注：不进行数值异常判定'};
        const band=document.createElementNS(NS,'rect');band.setAttribute('class','pto-training-sidecar__operator-band');band.setAttribute('x',item.left.toFixed(1));band.setAttribute('y',item.top.toFixed(1));band.setAttribute('width',Math.max(8,item.right-item.left).toFixed(1));band.setAttribute('height',Math.max(8,item.bottom-item.top).toFixed(1));band.setAttribute('rx','4');band.style.setProperty('--operator-color',item.color);tip(band,`${item.label} · operator row`,detail);svg.appendChild(band);
        const text=label(labels,'pto-training-sidecar__operator-label is-layer-row',item.label,(item.left+item.right)/2,item.y,`${item.label} · operator row`,detail);
        text.style.setProperty('--operator-color',item.color);
      });
    }

    function measureStaticOperatorBands(g){
      const specs=[
        ['input',['token_ids','positions','attention_context','embedding_weight'],'Input IDs · Attn Context · Embedding W'],
        ['input',['embedding'],'Parallel Embedding'],
        ['output',['final_norm'],'Final RMSNorm'],['output',['lm_head_weight','lm_head'],'LM Head Weight / LM Head'],
        ['output',['logits_allgather'],'Logits All-Gather'],['output',['logits'],'Logits'],
        ['output',['mtp_input_norms'],'MTP Input Norms'],['output',['mtp_eh_proj'],'EH Projection'],
        ['output',['mtp_decoder_layer'],'MTP Decoder ×3'],['output',['mtp_head_weight','mtp_shared_head'],'MTP Head Weight / MTP Shared Head'],
        ['output',['mtp_logits'],'MTP Logits']
      ];
      return specs.map(([kind,ids,text])=>{
        const scope=root.querySelector(`.pto-model-deck__static--${kind}`);if(!scope)return;
        const rects=ids.map(id=>scope.querySelector(`[data-node="${CSS.escape(id)}"]`)).filter(Boolean).map(node=>node.getBoundingClientRect()).filter(rect=>rect.height>0);if(!rects.length)return;
        let left=g.worldX(Math.min(...rects.map(rect=>rect.left))-g.base.left)-4*g.scale,right=g.worldX(Math.max(...rects.map(rect=>rect.right))-g.base.left)+4*g.scale;
        let top=g.worldY(Math.min(...rects.map(rect=>rect.top))-g.base.top)-3*g.scale,bottom=g.worldY(Math.max(...rects.map(rect=>rect.bottom))-g.base.top)+3*g.scale,y=(top+bottom)/2;
        const operatorX=kind==='input'?g.input.x:g.output.x,labelGap=(kind==='input'?6:10)*g.scale;
        if(kind==='input'){
          const embedding=ids.includes('embedding'),laneY=embedding?g.embeddingY:g.inputSummaryY,laneHeight=Math.max(16*g.scale,bottom-top),laneWidth=(embedding?184:330)*g.scale;
          y=laneY;top=y-laneHeight/2;bottom=y+laneHeight/2;right=operatorX-labelGap;left=right-laneWidth;
        }else{
          const laneWidth=clamp(text.length*6.2,92,260)*g.scale;
          left=operatorX+labelGap;right=left+laneWidth;
        }
        return{kind,ids,text,left,right,top,bottom,y,operatorX};
      }).filter(Boolean);
    }

    function renderStaticOperatorBands(g,measured=measureStaticOperatorBands(g)){
      const sideWidth=operatorSideWidth(g),halfSide=sideWidth/2,columnWidth=layerColumnWidth(g),padding=2*g.scale;
      ['input','output'].forEach(kind=>{
        const items=measured.filter(item=>item.kind===kind);if(!items.length)return;
        const operatorX=items[0].operatorX,top=Math.min(...items.map(item=>item.top))-padding,bottom=Math.max(...items.map(item=>item.bottom))+padding,column=document.createElementNS(NS,'rect');
        column.setAttribute('class',`pto-training-sidecar__layer-column is-static is-${kind}`);
        column.setAttribute('x',(operatorX-columnWidth/2).toFixed(1));column.setAttribute('y',top.toFixed(1));column.setAttribute('width',columnWidth.toFixed(1));column.setAttribute('height',Math.max(4*g.scale,bottom-top).toFixed(1));column.setAttribute('rx',(2*g.scale).toFixed(1));svg.appendChild(column);
      });
      const inputSummary=measured.find(item=>item.kind==='input'&&!item.ids.includes('embedding')),embedding=measured.find(item=>item.kind==='input'&&item.ids.includes('embedding'));
      if(inputSummary&&embedding){
        const x=inputSummary.operatorX,startY=inputSummary.bottom,endY=embedding.top;
        svg.appendChild(path(`M${x.toFixed(1)} ${startY.toFixed(1)}L${x.toFixed(1)} ${endY.toFixed(1)}`,'pto-training-sidecar__static-connector'));
        const residual=g.residualInputStart;
        if(residual&&residual.y>embedding.bottom){
          const bendY=(embedding.bottom+residual.y)/2;
          svg.appendChild(path(`M${x.toFixed(1)} ${embedding.bottom.toFixed(1)}C${x.toFixed(1)} ${bendY.toFixed(1)} ${residual.x.toFixed(1)} ${bendY.toFixed(1)} ${residual.x.toFixed(1)} ${residual.y.toFixed(1)}`,'pto-training-sidecar__static-connector is-residual-handoff'));
        }
      }
      measured.forEach(({kind,ids,text,left,right,top,bottom,y,operatorX})=>{
        const detail={title:text,category:kind==='input'?'Model input':'Model output',definition:kind==='input'?'模型入口算子：将离散输入和位置/上下文信息转换为进入 Decoder stack 的初始 hidden state。':'模型末端算子：将最终 hidden state 归一化并投影到词表 logits，或生成 MTP 辅助输出。',values:[['Node IDs',ids.join(', ')],['数据来源','模型结构']],status:'info',statusLabel:'结构标注：不进行数值异常判定'};
        const leaderStart=kind==='input'?right:operatorX+halfSide,leaderEnd=kind==='input'?operatorX-halfSide:left;
        svg.appendChild(path(`M${leaderStart.toFixed(1)} ${y.toFixed(1)}L${leaderEnd.toFixed(1)} ${y.toFixed(1)}`,'pto-training-sidecar__static-connector is-leader'));
        const side=document.createElementNS(NS,'rect');side.setAttribute('class',`pto-training-sidecar__operator-side is-static is-${kind}`);side.setAttribute('x',(operatorX-halfSide).toFixed(1));side.setAttribute('y',top.toFixed(1));side.setAttribute('width',sideWidth.toFixed(1));side.setAttribute('height',Math.max(g.scale,bottom-top).toFixed(1));side.setAttribute('rx',Math.min(g.scale,halfSide).toFixed(1));svg.appendChild(side);
        const band=document.createElementNS(NS,'rect');band.setAttribute('class',`pto-training-sidecar__operator-band is-${kind}`);band.setAttribute('x',left.toFixed(1));band.setAttribute('y',top.toFixed(1));band.setAttribute('width',Math.max(8,right-left).toFixed(1));band.setAttribute('height',Math.max(8,bottom-top).toFixed(1));band.setAttribute('rx','4');band.style.setProperty('--operator-color',kind==='input'?'var(--pto-model-deck-embedding)':'var(--pto-model-deck-head)');tip(band,`${text} · ${kind==='input'?'model input':'model output'}`,detail);svg.appendChild(band);
        const node=label(labels,`pto-training-sidecar__operator-label is-${kind}`,text,kind==='input'?right:(left+right)/2,y,`${text} · ${kind==='input'?'model input':'model output'}`,detail);node.style.setProperty('--operator-color',kind==='input'?'var(--pto-model-deck-embedding)':'var(--pto-model-deck-head)');
      });
    }

    function measureBaseAnnotationTips(g){
      const specs=[
        ['.pto-model-deck__annotation.is-side',node=>`${node.textContent.trim()} · pipeline-parallel layer partition`],
        ['.pto-model-deck__side-ffn-label',node=>`${node.textContent.trim()} · decoder FFN range`],
        ['.pto-model-deck__side-residual-label',()=>`mHC residual state ×4 · four parallel residual rails connected from model input to output`]
      ];
      const measured=[];specs.forEach(([selector,describe])=>root.querySelectorAll(selector).forEach(source=>{
        const rect=source.getBoundingClientRect();if(rect.width<2||rect.height<2)return;
        measured.push({left:g.worldX(rect.left-g.base.left),top:g.worldY(rect.top-g.base.top),width:rect.width/controller.state.zoom,height:rect.height/controller.state.zoom,text:describe(source)});
      }));
      return measured;
    }

    function renderBaseAnnotationTips(g,measured=measureBaseAnnotationTips(g)){
      measured.forEach(item=>{
        const hit=document.createElement('div');hit.className='pto-training-sidecar__base-tip-hit';hit.style.left=`${item.left}px`;hit.style.top=`${item.top}px`;hit.style.width=`${item.width}px`;hit.style.height=`${item.height}px`;tip(hit,item.text);labels.appendChild(hit);
      });
    }

    function measureFocus(g){
      if(!Number.isFinite(selectedLayer))return null;
      const point=g.points.find(item=>item.layer===selectedLayer);if(!point)return;
      const sourceGraph=point.card.querySelector('.pto-model-deck__graph');if(!sourceGraph)return null;
      const graphWidth=activeMetricViews.size?1000:720,graphHeight=1124,worldPerScreen=1/Math.max(.001,controller.state.zoom);
      const panelWidth=activeMetricViews.size?1000:600,panelTop=point.cardTop,panelHeight=graphHeight;
      const graphScale=1,graphOffsetY=0;
      const left=point.x-panelWidth/2;
      return{point,sourceGraph,panelWidth,panelTop,panelHeight,left,graphWidth,graphHeight,graphScale,graphOffsetY,worldPerScreen};
    }

    function renderFocus(g,layout=measureFocus(g)){
      root.classList.toggle('has-expanded-layer',Boolean(layout));
      if(!layout){focus.classList.remove('is-open','is-flipping');focus.removeAttribute('data-layer');focus.replaceChildren();return;}
      const {panelWidth,panelTop,panelHeight,left,graphWidth,graphScale,graphOffsetY,worldPerScreen}=layout,snapshot=snapshotFor(selectedLayer);
      const hidden=snapshot.tensors.hidden,activationGradient=snapshot.tensors.activationGradient,streams=controller.config.mhcStreams??4;
      const hiddenStatus=hidden.amax>9.35?'warning':'normal',hiddenDetail={key:`layer-${selectedLayer}-hidden-input`,title:`mHC state · H${selectedLayer} ×${streams}`,category:'Layer tensor',definition:'当前 Layer 的输入是四路 mHC state。Attention 和 FFN 均通过训练配置定义的 Mix/Merge 变换四路状态；没有可靠融合矩阵时不画成标准 Add。',values:[['Streams',streams],['Mean',hidden.mean.toFixed(4)],['Std σ',hidden.std.toFixed(3)],['Amax',hidden.amax.toFixed(2)],['L2 proxy',hidden.norm.toFixed(1)],['Activation gradient L2',activationGradient.norm.toFixed(3)]],status:hiddenStatus,statusLabel:hiddenStatus==='warning'?'关注：Amax 接近参考边界':'正常：指标位于参考范围',context:executionContext(snapshot,`L${selectedLayer}`)};
      const outputDetail={...hiddenDetail,key:`layer-${selectedLayer}-hidden-output`,title:`Output mHC state · H${selectedLayer+1} ×${streams}`,definition:'FFN Mix/Merge 后的四路 mHC state，也是下一 Layer 输入；若位于 PP 边界则通过 activation Send/Recv 传输。'};
      const attention=attentionDescriptor(selectedLayer);
      const layerChanged=focus.dataset.layer!==String(selectedLayer);
      focus.style.setProperty('--pto-training-focus-scale',worldPerScreen.toFixed(4));focus.style.setProperty('--pto-training-focus-graph-scale',graphScale.toFixed(4));focus.style.left=`${left}px`;focus.style.top=`${panelTop}px`;focus.style.width=`${panelWidth}px`;focus.style.height=`${panelHeight}px`;focus.dataset.layer=String(selectedLayer);focus.classList.add('is-open');
      focus.innerHTML=`<button class="pto-training-sidecar__focus-close" type="button" aria-label="收起 Layer">×</button><div class="pto-training-sidecar__focus-graph-stage" style="top:${graphOffsetY}px;width:${graphWidth}px"></div>`;
      const graph=layout.sourceGraph.cloneNode(true);graph.classList.add('pto-training-sidecar__focus-graph');
      graph.style.width=`${graphWidth}px`;
      graph.querySelectorAll('.is-selected').forEach(node=>node.classList.remove('is-selected'));
      const semanticById=new Map();(controller.config.sideRows||[]).forEach((row,index)=>row.ids.forEach(id=>semanticById.set(id,{label:row.ids.includes('attention_core')?attention.label:row.label,index})));
      const graphNodes=Array.from(graph.querySelectorAll('[data-node]'));
      const dataMode=activeMetricViews.size>0,presentations=new Map();
      graph.classList.toggle('has-focus-data',dataMode);
      if(dataMode)graph.querySelectorAll('.pto-model-deck__cluster').forEach(cluster=>{cluster.style.left='24px';cluster.style.width=`${graphWidth-48}px`;});
      graphNodes.forEach((node,nodeIndex)=>{
        const nodeId=node.dataset.node,op=node.dataset.op||'',semantic=semanticById.get(nodeId),originalLabel=node.textContent.trim()||semantic?.label||node.getAttribute('aria-label')?.split('·')[0]||nodeId;
        const stateInput=nodeId==='mhc_state_in',stateOutput=nodeId==='mhc_state_out';
        let displayLabel=originalLabel,summary,detail;
        if(stateInput||stateOutput){
          displayLabel=`H${stateOutput?selectedLayer+1:selectedLayer} · mHC state ×${streams}`;
          const factor=stateOutput?1.012:1,amaxFactor=stateOutput?1.018:1;
          const summaries={
            numerics:[`σ ${(hidden.std*factor).toFixed(2)}`,`Amax ${(hidden.amax*amaxFactor).toFixed(1)}`],
            gradient:[`∇ ${activationGradient.norm.toFixed(2)}`],
            performance:[],moe:[],training:[`H ×${streams}`]
          };
          summary=metricSummary(summaries);detail=stateOutput?outputDetail:hiddenDetail;
        }else{
          const metric=internalOperatorMetric(selectedLayer,semantic?.label||originalLabel,op,semantic?.index??nodeIndex,nodeId);summary=metric.summary;detail=metric.detail;
        }
        presentations.set(nodeId,{displayLabel,summary,detail});
      });
      const graphEdges=graph.querySelector('.pto-model-deck__edges');graphEdges?.setAttribute('viewBox',`0 0 ${graphWidth} 1124`);if(graphEdges)graphEdges.style.width=`${graphWidth}px`;
      const boxes=new Map(),layoutByNode=new Map(),layoutGroups=new Map();
      graphNodes.forEach(node=>{
        const nodeId=node.dataset.node,add=node.dataset.op==='add',state=node.dataset.op==='mhc-state';
        const attentionBranch=/^(q_|kv_|query_tensor|key_tensor)/.test(nodeId),moeOuter=/^(gate|a2a_dispatch|expert_pool|shared_expert)$/.test(nodeId);
        const group=state?'state':attentionBranch?'attention-branch':moeOuter?'moe-outer':'center';
        const presentation=presentations.get(nodeId),summaryItems=presentation?.summary?.slice(0,3)||[],showData=dataMode&&summaryItems.length>0&&!add;
        const labelScreenWidth=clamp(String(presentation?.displayLabel||'').length*6+18,82,160);
        const metricCharacters=summaryItems.reduce((sum,value)=>sum+String(value).length,0);
        const metricScreenWidth=showData?clamp(metricCharacters*5.5+Math.max(0,summaryItems.length-1)*6+8,54,176):0;
        const maxWidth=state?340:attentionBranch?420:nodeId==='a2a_dispatch'?340:moeOuter?300:380;
        layoutByNode.set(nodeId,{add,state,attentionBranch,moeOuter,group,showData,labelScreenWidth,maxWidth});
        if(!add){
          const aggregate=layoutGroups.get(group)||{labelScreenWidth:0,metricScreenWidth:0,showData:false,maxWidth:0};
          aggregate.labelScreenWidth=Math.max(aggregate.labelScreenWidth,labelScreenWidth);
          aggregate.metricScreenWidth=Math.max(aggregate.metricScreenWidth,metricScreenWidth);
          aggregate.showData ||= showData;
          aggregate.maxWidth=Math.max(aggregate.maxWidth,maxWidth);
          layoutGroups.set(group,aggregate);
        }
      });
      layoutGroups.forEach(group=>{
        const desiredWidth=(group.labelScreenWidth+group.metricScreenWidth+(group.showData?4:0))*worldPerScreen;
        const minWidth=(group.showData?140:82)*worldPerScreen;
        group.width=dataMode?clamp(desiredWidth,minWidth,group.maxWidth):168;
        group.labelWidth=group.labelScreenWidth*worldPerScreen;
      });
      graphNodes.forEach(node=>{
        const x=parseFloat(node.style.left)||0,y=parseFloat(node.style.top)||0,w=parseFloat(node.style.width)||196,h=parseFloat(node.style.height)||30,cx=x+w/2,cy=y+h/2;
        const nodeId=node.dataset.node,{add,state,group,showData}=layoutByNode.get(nodeId),addSize=Math.min(w,h),groupLayout=layoutGroups.get(group);
        const targetWidth=add?addSize:groupLayout?.width??168,targetHeight=add?addSize:showData?Math.max(h,38):h;
        const normalizedCenter=(dataMode?{
          q_a_proj:250,q_causal_conv:250,q_residual_add:250,q_a_norm:250,q_b_proj:250,query_tensor:250,
          kv_a_proj:750,kv_causal_conv:750,kv_residual_add:750,kv_a_norm:750,kv_b_proj:750,key_tensor:750,
          gate:170,a2a_dispatch:170,expert_pool:500,shared_expert:830,a2a_combine:500,moe_branch_add:500
        }:{
          q_a_proj:240,q_causal_conv:240,q_residual_add:240,q_a_norm:240,q_b_proj:240,query_tensor:240,
          kv_a_proj:480,kv_causal_conv:480,kv_residual_add:480,kv_a_norm:480,kv_b_proj:480,key_tensor:480,
          gate:180,a2a_dispatch:180,expert_pool:360,shared_expert:540,a2a_combine:360,moe_branch_add:360
        })[nodeId];
        const compactCenterX=state?(dataMode?790:540):(normalizedCenter??(dataMode?cx*graphWidth/720:cx)),compactCenterY=cy;
        const box={x:compactCenterX-targetWidth/2,y:compactCenterY-targetHeight/2,w:targetWidth,h:targetHeight};boxes.set(nodeId,box);
        node.style.left=`${box.x}px`;node.style.top=`${box.y}px`;node.style.width=`${box.w}px`;node.style.height=`${box.h}px`;
        if(showData)node.style.setProperty('--pto-training-focus-label-width',`${groupLayout.labelWidth}px`);
        else node.style.removeProperty('--pto-training-focus-label-width');
      });
      const pointFor=(box,side)=>side==='right'?{x:box.x+box.w,y:box.y+box.h/2}:side==='left'?{x:box.x,y:box.y+box.h/2}:side==='top'?{x:box.x+box.w/2,y:box.y}:{x:box.x+box.w/2,y:box.y+box.h};
      Array.from(graphEdges?.querySelectorAll('path')||[]).forEach(edge=>{
        const sourceId=edge.dataset.source,targetId=edge.dataset.target,source=boxes.get(sourceId),target=boxes.get(targetId);if(!source||!target)return;
        const kind=edge.dataset.kind||'activation',mhcRail=kind==='state-spine'||(sourceId==='mhc_state_in'&&targetId==='mhc_attention_post')||(sourceId==='mhc_attention_post'&&targetId==='ffn_residual_add');
        let p0=pointFor(source,'bottom'),p1=pointFor(target,'top'),d;
        if(sourceId==='a2a_dispatch'&&targetId==='expert_pool'){
          p0=pointFor(source,'right');p1=pointFor(target,'left');const mid=(p0.x+p1.x)/2;d=`M${p0.x} ${p0.y}C${mid} ${p0.y},${mid} ${p1.y},${p1.x} ${p1.y}`;
        }else if(sourceId==='shared_expert'&&targetId==='moe_branch_add'){
          const routeX=dataMode?graphWidth-20:624;p0=pointFor(source,'right');p1=pointFor(target,'right');d=`M${p0.x} ${p0.y}L${routeX} ${p0.y}L${routeX} ${p1.y}L${p1.x} ${p1.y}`;
        }else if(kind==='residual'&&(sourceId==='mhc_state_in'||sourceId==='mhc_attention_post')){
          const routeX=dataMode?graphWidth-30:620;p0=pointFor(source,'right');p1=pointFor(target,'right');d=`M${p0.x} ${p0.y}L${routeX} ${p0.y}L${routeX} ${p1.y}L${p1.x} ${p1.y}`;
        }else if(Math.abs(p0.x-p1.x)<8){d=`M${p0.x} ${p0.y}L${p1.x} ${p1.y}`;
        }else{const mid=(p0.y+p1.y)/2;d=`M${p0.x} ${p0.y}C${p0.x} ${mid},${p1.x} ${mid},${p1.x} ${p1.y}`;}
        edge.setAttribute('d',d);
        if(mhcRail){
          edge.classList.add('is-focus-mhc-rail');
          [-4.5,-1.5,1.5,4.5].forEach((offset,index)=>{
            const rail=index?edge.cloneNode(true):edge;rail.setAttribute('transform',`translate(${offset} ${offset})`);rail.dataset.rail=String(index+1);if(index)edge.after(rail);
          });
        }
      });
      graphNodes.forEach(node=>{
        const nodeId=node.dataset.node,{displayLabel,summary,detail}=presentations.get(nodeId);
        node.replaceChildren();
        const labelNode=document.createElement('span');labelNode.className='pto-training-sidecar__focus-node-label';labelNode.textContent=displayLabel;node.appendChild(labelNode);
        const showData=summary.length>0&&node.dataset.op!=='add';
        node.classList.toggle('has-focus-data',showData);
        if(showData){const metricNode=document.createElement('span');metricNode.className='pto-training-sidecar__focus-node-metrics';summary.slice(0,3).forEach(value=>{const item=document.createElement('i');item.textContent=value;metricNode.appendChild(item);});node.appendChild(metricNode);}
        tip(node,summary.length?`${displayLabel} · ${summary.join(' · ')}`:displayLabel,detail);
      });
      if(!snapshot.module.dense&&(activeMetricViews.has('moe')||activeMetricViews.has('training'))){const aux=document.createElement('div');aux.className='pto-training-sidecar__focus-graph-aux';aux.textContent='Router Aux · balance 0.014 · z-loss 0.003 · λ from training job';graph.appendChild(aux);}
      focus.querySelector('.pto-training-sidecar__focus-graph-stage')?.appendChild(graph);
      focus.querySelector('.pto-training-sidecar__focus-close')?.addEventListener('click',()=>selectLayer(null));
      if(layerChanged){
        focus.classList.remove('is-flipping');
        void focus.offsetWidth;
        focus.classList.add('is-flipping');
      }
    }

    function renderLossComposition(g){
      if(Number.isFinite(selectedLayer))return;
      const lastLayer=controller.config.layerCount-1,lastStage=stageForLayer(lastLayer,controller.config.stageRanges),snapshot=snapshotFor(lastLayer),loss=snapshot.metric.loss??2.84,mtpHeads=controller.config.mtpHeads??3;
      const job=options.trainingJob||{},moeWeight=job.lossWeights?.moeAux??'λ configured by training job',mtpWeight=job.lossWeights?.mtp??'λ configured by training job';
      const mtpParticipation=job.mtpLossParticipates===false?'not in current training objective':job.mtpLossParticipates===true?'included by training job':'participation from training job config';
      const worldLeft=g.worldX(0),worldRight=g.worldX(g.width),worldTop=g.worldY(0),worldBottom=g.worldY(g.height);
      const anchorX=clamp(g.output.x-12*g.scale,worldLeft+350*g.scale,worldRight-8*g.scale),anchorY=clamp(g.optimizerY+112*g.scale,worldTop+110*g.scale,worldBottom-90*g.scale);
      const node=label(labels,'pto-training-sidecar__loss-composition','',anchorX,anchorY,'Loss composition · Main CE + Router Aux + MTP ×3 → Total Training Loss');
      node.removeAttribute('data-detail');node.removeAttribute('data-detail-key');node.removeAttribute('data-tip');node.removeAttribute('aria-label');
      node.classList.toggle('is-expanded',lossExpanded);node.classList.toggle('is-collapsed',!lossExpanded);
      if(!lossExpanded){
        node.innerHTML=`<button type="button" class="pto-training-sidecar__loss-summary" data-loss-toggle aria-expanded="false"><span>Training objective</span><b>CE + Aux + MTP ×${mtpHeads} → Total</b><i>展开</i></button>`;
        node.querySelector('[data-loss-toggle]')?.addEventListener('click',event=>{event.stopPropagation();lossExpanded=true;schedule(false);});return;
      }
      node.innerHTML=`<div class="pto-training-sidecar__loss-header"><strong>Training objective · output tail</strong><button type="button" data-loss-toggle aria-label="收起 Training objective" aria-expanded="true">−</button></div><div class="pto-training-sidecar__loss-input">LM Head logits + shifted target token IDs / labels</div><div class="pto-training-sidecar__loss-terms"><button type="button" data-loss="main">Main CE <b>${loss.toFixed(2)}</b></button><span>+</span><button type="button" data-loss="moe">Router Aux <b>${esc(moeWeight)}</b></button><span>+</span><button type="button" data-loss="mtp">MTP ×${mtpHeads} <b>${esc(mtpWeight)}</b></button></div><button type="button" class="pto-training-sidecar__loss-total" data-loss="total">Total Training Loss</button><small>PP${lastStage??'?'} · beside LM Head / logits · MTP: ${esc(mtpParticipation)}</small>`;
      node.querySelector('[data-loss-toggle]')?.addEventListener('click',event=>{event.stopPropagation();lossExpanded=false;schedule(false);});
      tip(node.querySelector('[data-loss="main"]'),'Main CE Loss · LM Head logits + shifted target labels',{title:'Main CE Loss',category:'Training objective',definition:'Final Norm 与 LM Head 产生 logits；主损失由 logits 与 shifted target token IDs / labels 计算 CrossEntropy。',values:[['当前值',loss.toFixed(2)],['输入','LM Head logits + shifted target labels']],status:'normal',statusLabel:'正常：位于参考范围',context:`step ${context.step} · MB ${context.microbatch}/${context.microbatchCount} · PP${lastStage??'?'} · LM Head logits · target labels`});
      tip(node.querySelector('[data-loss="moe"]'),'MoE Router Aux Loss · coefficient from training job',{title:'MoE Router Aux Loss',category:'Training objective',definition:'Router balance / z-loss 等辅助项是否进入总目标以及具体系数必须由真实训练任务配置决定。',values:[['Coefficient',moeWeight],['Router diagnostics','balance 0.014 · z-loss 0.003']],status:'info',statusLabel:'需要训练任务配置确认'});
      tip(node.querySelector('[data-loss="mtp"]'),`MTP Loss ×${mtpHeads} · coefficient and participation from training job`,{title:`MTP Loss ×${mtpHeads}`,category:'Training objective',definition:'公开模型结构确认有 3 个 MTP head，但当前预览不假定具体训练 Loss 权重或参与状态。',values:[['Heads',mtpHeads],['Coefficient',mtpWeight],['Participation',mtpParticipation]],status:'info',statusLabel:'需要训练任务配置确认'});
      tip(node.querySelector('[data-loss="total"]'),'Total Training Loss · configured composition',{title:'Total Training Loss',category:'Training objective',definition:'总训练目标由 Main CE、Router 辅助项和启用的 MTP losses 按训练任务配置组合。未提供配置时只展示符号关系，不填造系数。',status:'info',statusLabel:'Composition depends on training job config'});
    }

    function optimizerFlow(){
      const method=String(options.trainingJob?.gradientSync??topology.gradientSync??'').toLowerCase();
      if(method.includes('reduce'))return{method:'Reduce-Scatter',label:'Accumulation → DP Gradient Sync: Reduce-Scatter → Unscale / Clip → Optimizer → Param All-Gather → W(t+1)',allGather:true};
      if(method.includes('all-reduce')||method.includes('allreduce'))return{method:'All-Reduce',label:'Accumulation → DP Gradient Sync: All-Reduce → Unscale / Clip → Optimizer → W(t+1)',allGather:false};
      return{method:'configured by training job',label:'Accumulation → DP Gradient Sync → Unscale / Clip → Optimizer Step → W(t+1)',allGather:null};
    }

    function render(){
      raf=0;if(destroyed)return;syncOverlayTransform();
      const state=controller.state,cacheKey=[viewport.clientWidth,viewport.clientHeight,state.view,state.rx,state.ry,state.expandedLayer,state.expansionDepth,selectedLayer,fitZoom].join('|');
      let g,measuredRows,staticBands,baseTips,focusLayout;
      if(!structureDirty&&measurementCache?.key===cacheKey){({g,measuredRows,staticBands,baseTips,focusLayout}=measurementCache);}else{
        g=geometry();if(!g)return;
        measuredRows=operatorRows(g);staticBands=measureStaticOperatorBands(g);baseTips=measureBaseAnnotationTips(g);focusLayout=measureFocus(g);
        measurementCache={key:cacheKey,g,measuredRows,staticBands,baseTips,focusLayout};structureDirty=false;
      }
      root.style.setProperty('--pto-training-sidecar-scale',g.scale.toFixed(4));
      svg.removeAttribute('viewBox');layerHits.removeAttribute('viewBox');svg.replaceChildren();labels.replaceChildren();layerHits.replaceChildren();
      const first=g.points[0],last=g.points[g.points.length-1],x0=Math.min(g.input.x,first.x),x1=Math.max(g.output.x,last.x);
      const stageRanges=controller.config.stageRanges||[],lastLayer=controller.config.layerCount-1,lastState=controller.config.layerCount;
      const lowerTrainingVisible=activeMetricViews.has('gradient')||activeMetricViews.has('training'),structureBottom=lowerTrainingVisible?g.optimizerY+10*g.scale:g.modelBottom+2*g.scale;
      renderLayerBackgrounds(g);
      stageRanges.slice(1).forEach(([lo])=>{const before=g.points.find(point=>point.layer===lo-1),after=g.points.find(point=>point.layer===lo);if(!before||!after)return;const x=(before.x+after.x)/2,guide=path(`M${x.toFixed(1)} ${(g.stageY-18*g.scale).toFixed(1)}L${x.toFixed(1)} ${structureBottom.toFixed(1)}`,'pto-training-sidecar__stage-guide');tip(guide,`PP boundary · L${lo-1}/L${lo} · spans visible training lanes`);svg.append(guide);});
      g.points.forEach(point=>svg.append(path(`M${point.x.toFixed(1)} ${(g.axisY+5*g.scale).toFixed(1)}L${point.x.toFixed(1)} ${structureBottom.toFixed(1)}`,'pto-training-sidecar__layer-guide')));
      renderOperatorSides(g);
      label(labels,'pto-training-sidecar__value-label is-structure','MODEL DEPTH / TOPOLOGY',first.x-78*g.scale,g.axisY-5*g.scale,'Model depth / topology axis · layer order, not runtime time');

      const metricRows=[
        {key:'std',name:'Std σ',color:'var(--training-metric-std)',min:.82,max:1.08,unit:'',precision:3},
        {key:'amax',name:'Amax',color:'var(--training-metric-amax)',min:6.8,max:9.7,unit:'',precision:2},
        {key:'latency',name:'Latency',color:'var(--training-metric-latency)',min:1.1,max:2.5,unit:' ms',precision:2}
      ];
      metricRows.forEach((metric,row)=>{
        const {key,name,color,min,max,unit,precision}=metric,chartHeight=18*g.scale,rowStep=24*g.scale,top=g.metricY+row*rowStep,span=max-min,domainMin=min-span*.18,domainMax=max+span*.18;
        const yFor=(value)=>top+chartHeight*(1-clamp((value-domainMin)/(domainMax-domainMin),0,1));
        const band=document.createElementNS(NS,'rect'),bandTop=yFor(max),bandBottom=yFor(min);
        band.setAttribute('class',`pto-training-sidecar__metric-band is-metric-${key}`);band.setAttribute('x',first.x.toFixed(1));band.setAttribute('y',bandTop.toFixed(1));band.setAttribute('width',Math.max(1,last.x-first.x).toFixed(1));band.setAttribute('height',Math.max(1,bandBottom-bandTop).toFixed(1));band.style.setProperty('--metric-color',color);svg.appendChild(band);
        const midline=path(`M${first.x.toFixed(1)} ${yFor((min+max)/2).toFixed(1)}L${last.x.toFixed(1)} ${yFor((min+max)/2).toFixed(1)}`,`pto-training-sidecar__metric-midline is-metric-${key}`);midline.style.setProperty('--metric-color',color);svg.appendChild(midline);
        const series=g.points.map(point=>({point,value:Number(metricValue(snapshotFor(point.layer),key))}));
        const line=path(series.map(({point,value},index)=>`${index?'L':'M'}${point.x.toFixed(1)} ${yFor(value).toFixed(1)}`).join(' '),`pto-training-sidecar__metric-line is-metric-${key}`);line.style.setProperty('--metric-color',color);tip(line,`${name} · L0–L${lastLayer} trend`,{title:`${name} · Layer trend`,category:'Metric line',definition:metricDetail(key,name,(min+max)/2,min,max,0).definition,values:[['参考范围',`${min}–${max}${unit}`],['横轴',`Decoder Layer L0–L${lastLayer}`]],status:'info',statusLabel:'折线显示逐 Layer 趋势；淡色带为参考范围'});svg.appendChild(line);
        series.forEach(({point,value})=>{const dot=circle(point.x,yFor(value),2.1*g.scale,`pto-training-sidecar__metric-point is-metric-${key}`);dot.style.setProperty('--metric-color',color);tip(dot,`L${point.layer} · ${name} ${value.toFixed(precision)}${unit}`,metricDetail(key,name,value,min,max,point.layer));svg.appendChild(dot);});
        label(labels,`pto-training-sidecar__metric-label is-metric-${key}`,`${name} · ${min}–${max}${unit}`,x0-20*g.scale,top+chartHeight/2,`${name} line chart · reference ${min}–${max}${unit}`,{title:name,category:'Metric line',definition:metricDetail(key,name,(min+max)/2,min,max,0).definition,values:[['参考范围',`${min}–${max}${unit}`],['横轴',`L0–L${lastLayer}`]],status:'info',statusLabel:'点击折线节点查看对应 Layer 数值'});
      });

      renderOperatorBands(g,measuredRows);
      renderStaticOperatorBands(g,staticBands);
      renderBaseAnnotationTips(g,baseTips);

      const layerAnchors=g.points.map(point=>point.x);
      const forwardY=g.residualRails.length?g.residualRails.reduce((sum,rail)=>sum+rail.y,0)/g.residualRails.length:g.hiddenY;
      const forwardDetail={title:'Forward mHC state',category:'Tensor flow',definition:'四条 residual rails 本身就是 Layer 间的 forward mHC state，不再另画一条重复的单路 forward lane。',values:[['路径',`H₀ ×4 → H₁ ×4 → … → H${lastState} ×4`],['上下文',`step ${context.step} · MB ${context.microbatch}/${context.microbatchCount}`]],status:'info',statusLabel:'数据依赖：非运行时间轴'};
      g.residualRails.forEach(rail=>{
        const hit=path(`M${rail.x0.toFixed(1)} ${rail.y.toFixed(1)}L${rail.x1.toFixed(1)} ${rail.y.toFixed(1)}`,'pto-training-sidecar__residual-flow-hit');tip(hit,`Forward mHC state · rail ${rail.rail+1}/4`,forwardDetail);svg.append(hit);
      });
      if(g.residualRails.length)for(let index=0;index<layerAnchors.length-1;index+=1){
        const x=(layerAnchors[index]+layerAnchors[index+1])/2,halfWidth=Math.min(4*g.scale,(layerAnchors[index+1]-layerAnchors[index])*.26),halfHeight=5*g.scale;
        svg.append(path(`M${(x-halfWidth).toFixed(1)} ${(forwardY-halfHeight).toFixed(1)}L${(x+halfWidth).toFixed(1)} ${forwardY.toFixed(1)}L${(x-halfWidth).toFixed(1)} ${(forwardY+halfHeight).toFixed(1)}`,'pto-training-sidecar__residual-flow-arrow'));
      }
      if(g.residualRails.length)g.points.forEach(point=>{
        const snapshot=snapshotFor(point.layer),hidden=snapshot.tensors.hidden,status=hidden.amax>9.7?'abnormal':hidden.amax>9.35?'warning':'normal',hit=circle(point.x,forwardY,5*g.scale,'pto-training-sidecar__residual-flow-hit-dot');
        tip(hit,`L${point.layer} · mHC state ×4 · σ ${hidden.std.toFixed(3)} · amax ${hidden.amax.toFixed(2)}`,{key:`hidden-state-L${point.layer}`,title:`mHC state · L${point.layer}`,category:'Tensor',definition:'该命中点覆盖当前 Layer 的四条 residual rails；它们共同表示对应 Decoder Layer 输出的四路 mHC state。',values:[['Streams',controller.config.mhcStreams??4],['Mean',hidden.mean.toFixed(4)],['Std σ',hidden.std.toFixed(3)],['Amax',hidden.amax.toFixed(2)],['L2 proxy',hidden.norm.toFixed(1)]],status,statusLabel:status==='abnormal'?'异常：Amax 超出参考范围':status==='warning'?'关注：Amax 接近参考边界':'正常：mHC-state 统计位于参考范围',context:executionContext(snapshot,`L${point.layer} · ${snapshot.tensorName}`)});svg.append(hit);
      });

      svg.append(flowTexture(x0,x1,g.backwardY,'backward',g.scale,layerAnchors));
      const backwardPath=path(`M${x1.toFixed(1)} ${g.backwardY.toFixed(1)}L${x0.toFixed(1)} ${g.backwardY.toFixed(1)}`,'pto-training-sidecar__backward');tip(backwardPath,'Backward activation gradient · ∂L/∂Hₗ ×4',{title:'Activation gradient',category:'Tensor flow',definition:'跨 Layer 向后传递的是四路 mHC state 的激活梯度 ∂L/∂H，而不是参数梯度 ∂L/∂W。',values:[['方向',`L${lastLayer} → L0`],['符号','∂L/∂Hₗ ×4']],status:'info',statusLabel:'反向数据依赖'});svg.append(backwardPath);
      g.points.forEach(point=>{
        const snapshot=snapshotFor(point.layer),activationGradient=snapshot.tensors.activationGradient,contribution=snapshot.parameter.gradientContribution,accumulated=snapshot.parameter.accumulatedGradient;
        const backwardDot=circle(point.x,g.backwardY,1.75,'pto-training-sidecar__backward-dot');tip(backwardDot,`L${point.layer} · activation gradient · norm ${activationGradient.norm.toFixed(3)}`,{key:`activation-gradient-L${point.layer}`,title:`Activation gradient · L${point.layer}`,category:'Tensor',definition:'该点表示沿四路 mHC state 主干反向传递的 activation gradient；它不是 parameter gradient。',values:[['Gradient norm',activationGradient.norm.toFixed(3)],['Gradient amax',activationGradient.amax.toFixed(3)]],status:activationGradient.norm>1.2?'warning':'normal',statusLabel:activationGradient.norm>1.2?'关注：激活梯度偏高':'正常：位于参考范围',context:executionContext(snapshot,`L${point.layer}`)});svg.append(backwardDot);
        svg.append(path(`M${point.x.toFixed(1)} ${(g.backwardY+3*g.scale).toFixed(1)}L${point.x.toFixed(1)} ${g.parameterY.toFixed(1)}`,'pto-training-sidecar__parameter-stem'));
        const contributionDot=circle(point.x,g.parameterY,2.2,'pto-training-sidecar__parameter-dot');tip(contributionDot,`L${point.layer} · Δ∇Wₗ,MB${contribution.microbatch} ${contribution.norm.toFixed(3)}`,{title:`Microbatch gradient contribution · L${point.layer}`,category:'Parameter gradient',definition:'当前选定 microbatch 对该 Layer parameter gradient 的增量贡献；它不是 optimizer 消费的完整 step gradient。',values:[['Object',`Δ∇Wₗ,MB${contribution.microbatch}`],['Contribution L2',contribution.norm.toFixed(3)],['Microbatch',`${contribution.microbatch}/${snapshot.microbatchCount}`]],status:'normal',statusLabel:'Current microbatch contribution',context:executionContext(snapshot,`L${point.layer}`)});svg.append(contributionDot);
        const accumulatedDot=circle(point.x,g.optimizerY,2.35,'pto-training-sidecar__step-gradient-dot');tip(accumulatedDot,`L${point.layer} · ∇Wₗ,step accumulated ${accumulated.microbatches}/${accumulated.total}`,{title:`Step-accumulated gradient · L${point.layer}`,category:'Parameter gradient',definition:'完成所有 microbatch 累积后的 step gradient；DP Gradient Sync 与 Optimizer 消费这个对象。',values:[['Object','∇Wₗ,step'],['Accumulation',`${accumulated.microbatches}/${accumulated.total}`],['Gradient L2',accumulated.norm.toFixed(3)]],status:accumulated.microbatches===accumulated.total?'normal':'warning',statusLabel:accumulated.microbatches===accumulated.total?'Ready for DP Gradient Sync':'Accumulation incomplete',context:`optimizer_step ${snapshot.parameter.optimizerStep} · L${point.layer}`});svg.append(accumulatedDot);
      });
      label(labels,'pto-training-sidecar__value-label is-backward','activation gradient ∂L/∂Hₗ ×4',x1-138*g.scale,g.backwardY-5*g.scale,'Backward activation gradient · not a parameter gradient');

      const optimizer=optimizerFlow(),parameter=snapshotFor(0).parameter,optimizerPath=path(`M${x0.toFixed(1)} ${g.optimizerY.toFixed(1)}L${x1.toFixed(1)} ${g.optimizerY.toFixed(1)}`,'pto-training-sidecar__optimizer');tip(optimizerPath,'Optimizer step · step-accumulated and synchronized parameter gradients',{title:'Optimizer step',category:'Parameter update',definition:'完成 microbatch 梯度累积与 DP Gradient Sync 后，执行 unscale / overflow check / clip 和 Optimizer；distributed optimizer 配置可继续执行 Param All-Gather。',values:[['Accumulation',`${context.microbatchCount}/${context.microbatchCount}`],['DP Gradient Sync',optimizer.method],['Param All-Gather',optimizer.allGather==null?'from training job config':optimizer.allGather?'yes':'not required by this path'],['Optimizer step',parameter.optimizerStep],['Weight version before',parameter.weightVersionBefore],['Weight version after',parameter.weightVersionAfter]],status:'normal',statusLabel:'正常：step 已具备更新条件',context:`optimizer_step ${parameter.optimizerStep}`});svg.append(optimizerPath);
      label(labels,'pto-training-sidecar__value-label is-optimizer',optimizer.label,(x0+x1)/2,g.optimizerY-5*g.scale,'One logical optimizer step after accumulation and configured DP Gradient Sync');

      stageRanges.slice(1).forEach(([lo])=>{const before=g.points.find(point=>point.layer===lo-1),after=g.points.find(point=>point.layer===lo);if(!before||!after)return;const x=(before.x+after.x)/2,forwardText=`PP boundary L${lo-1}/L${lo} · Send/Recv forward mHC state H ×4`,backwardText=`PP boundary L${lo-1}/L${lo} · Send/Recv backward activation gradient ∂L/∂H ×4`,forwardLine=path(`M${x.toFixed(1)} ${(forwardY-7*g.scale).toFixed(1)}L${x.toFixed(1)} ${(forwardY+7*g.scale).toFixed(1)}`,'pto-training-sidecar__communication is-forward'),backwardLine=path(`M${x.toFixed(1)} ${(g.backwardY-7*g.scale).toFixed(1)}L${x.toFixed(1)} ${(g.backwardY+7*g.scale).toFixed(1)}`,'pto-training-sidecar__communication is-backward');tip(forwardLine,forwardText,communicationDetail(lo,'forward',`pp-forward-boundary-${lo}-line`));tip(backwardLine,backwardText,communicationDetail(lo,'backward',`pp-backward-boundary-${lo}-line`));svg.append(forwardLine,backwardLine);label(labels,'pto-training-sidecar__comm-label is-forward','H×4 S/R',x,forwardY-14*g.scale,forwardText,communicationDetail(lo,'forward',`pp-forward-boundary-${lo}-label`));label(labels,'pto-training-sidecar__comm-label is-backward','∂H×4 S/R',x,g.backwardY+15*g.scale,backwardText,communicationDetail(lo,'backward',`pp-backward-boundary-${lo}-label`));});

      if(state.parallelMode!=='none')stageRanges.forEach(([lo,hi],stage)=>{
        const a=g.points.find(point=>point.layer===lo),b=g.points.find(point=>point.layer===hi);if(!a||!b)return;
        const ranks=rankGroupForStage(stage),parallel=parallelSummary(),deployment=ranks?`ranks ${ranks}`:parallel||'';
        const node=label(labels,'pto-training-sidecar__stage-label','',(a.x+b.x)/2,g.stageY,`Pipeline stage ${stage} · decoder layers ${lo}–${hi}${deployment?` · ${deployment}`:''}`);
        node.innerHTML=`<b>PP Stage ${stage} · L${lo}–L${hi}</b>${deployment?`<span>${esc(deployment)}</span>`:''}`;
      });

      const samples=new Set([0,...stageRanges.map(([,hi])=>hi)].filter(layer=>layer>=0&&layer<=lastLayer));
      g.points.forEach(point=>{
        label(labels,'pto-training-sidecar__layer-number',`L${point.layer}`,point.x,g.axisY-7*g.scale,`Decoder Layer ${point.layer}`);
        const hitWidth=layerColumnWidth(g),padding=2*g.scale,hit=document.createElementNS(NS,'rect');hit.setAttribute('class','pto-training-sidecar__layer-hit');hit.setAttribute('data-deck-no-drag','true');hit.dataset.layer=String(point.layer);hit.dataset.tip=`Layer L${point.layer} · 点击展开内部算子与指标`;hit.setAttribute('tabindex','0');hit.setAttribute('role','button');hit.setAttribute('aria-label',`展开 Layer ${point.layer}`);hit.setAttribute('x',(point.x-hitWidth/2).toFixed(1));hit.setAttribute('y',(point.operatorTop-padding).toFixed(1));hit.setAttribute('width',hitWidth.toFixed(1));hit.setAttribute('height',Math.max(4*g.scale,point.operatorBottom-point.operatorTop+padding*2).toFixed(1));hit.setAttribute('rx',(2*g.scale).toFixed(1));hit.addEventListener('click',activateLayerHit);hit.addEventListener('keydown',activateLayerHit);layerHits.appendChild(hit);
        if(samples.has(point.layer)&&(activeMetricViews.has('numerics')||activeMetricViews.has('performance'))){
          const snapshot=snapshotFor(point.layer),std=Number(metricValue(snapshot,'std')),status=snapshot.metric.amax>9.35||snapshot.metric.latency>2.35?'warning':'normal',sampleValues=[];
          if(activeMetricViews.has('numerics'))sampleValues.push(`σ${std.toFixed(2)}`,`Amax ${snapshot.metric.amax.toFixed(1)}`);
          if(activeMetricViews.has('performance'))sampleValues.push(`${snapshot.metric.latency.toFixed(2)}ms`);
          const sampleText=`L${point.layer} · ${sampleValues.join(' · ')}`;
          label(labels,'pto-training-sidecar__sample-label',sampleText,point.x,g.metricY+76*g.scale,`L${point.layer} sample · Std σ ${std.toFixed(3)} · Amax ${snapshot.metric.amax.toFixed(2)} · Latency ${snapshot.metric.latency.toFixed(2)} ms`,{title:`Layer metric summary · L${point.layer}`,category:'Metric',definition:'选定 Layer 的 hidden-state 标准差、绝对最大值与执行耗时摘要。点击折线节点可查看各指标定义及独立阈值判断。',values:[['Std σ',std.toFixed(3)],['Amax',snapshot.metric.amax.toFixed(2)],['Latency',`${snapshot.metric.latency.toFixed(2)} ms`]],status,statusLabel:status==='warning'?'关注：至少一项接近参考边界':'正常：摘要指标均位于参考范围',context:executionContext(snapshot)});
        }
      });

      const activeLabels=metricViewOptions.filter(([value])=>activeMetricViews.has(value)).map(([,labelText])=>labelText);
      if(activeLabels.length){
        const node=label(labels,'pto-training-sidecar__semantic-label','',g.worldX(14),g.metricY+28*g.scale,`数据标注 · ${activeLabels.join(' / ')}`);
        node.innerHTML=`数据标注<span>${esc(activeLabels.join(' / '))}</span>`;
      }
      renderLossComposition(g);
      label(labels,'pto-training-sidecar__value-label is-checkpoint','Checkpoint · every 1000 steps',g.output.x,g.optimizerY+25*g.scale,'Checkpoint save cadence · independent of optimizer-step cadence');
      renderFocus(g,focusLayout);
      applyMetricVisibility();
      syncHoveredLayer();
      positionDetailPanel();
    }

    function schedule(structural=true){if(structural)structureDirty=true;if(!raf)raf=requestAnimationFrame(render);}
    function poseSnapshot(){
      const {zoom,panX,panY,rx,ry}=controller.state;
      return{zoom,panX,panY,rx,ry};
    }
    function expandedLayerPose(layout){
      const width=Math.max(1,viewport.clientWidth),height=Math.max(1,viewport.clientHeight);
      const topInset=Math.max(56,Number(options.layerFitTopInset)||76),bottomInset=Math.max(24,Number(options.layerFitBottomInset)||40),horizontalInset=Math.max(24,Number(options.layerFitHorizontalInset)||44);
      const usableHeight=Math.max(320,height-topInset-bottomInset),usableWidth=Math.max(320,width-horizontalInset*2);
      const heightZoom=usableHeight/Math.max(1,layout.panelHeight),widthZoom=usableWidth/Math.max(1,layout.panelWidth);
      const zoom=clamp(Math.min(heightZoom,widthZoom),.12,1.35);
      const centerY=layout.panelTop+layout.panelHeight/2,targetY=topInset+usableHeight/2;
      return{zoom,panX:-layout.point.x*zoom,panY:targetY-height/2-centerY*zoom};
    }
    function transitionCameraTo(pose,{expandedLayer,expansionDepth,onComplete}={}){
      cancelAnimationFrame(focusFitRaf);
      focusFitRaf=0;
      root.classList.add('is-layer-camera-transitioning');
      viewport.offsetWidth;
      if(expandedLayer!==undefined)controller.setLayerExpansion?.(expandedLayer,expansionDepth||0);
      controller.setPose(pose);syncOverlayTransform();
      clearTimeout(focusTransitionTimer);
      focusTransitionTimer=window.setTimeout(()=>{
        onComplete?.();
        schedule();
        requestAnimationFrame(()=>root.classList.remove('is-layer-camera-transitioning'));
      },540);
    }
    function fitExpandedLayer(){
      focusFitRaf=0;
      if(selectedLayer===null||controller.state.view!=='right'||root.classList.contains('is-layer-camera-transitioning'))return api;
      const g=geometry(),layout=g&&measureFocus(g);if(!layout)return api;
      transitionCameraTo(expandedLayerPose(layout));
      return api;
    }
    function queueExpandedLayerFit(){
      cancelAnimationFrame(focusFitRaf);
      focusFitRaf=requestAnimationFrame(fitExpandedLayer);
      return api;
    }
    function fitSidecar(){
      if(!controller||!viewport)return api;
      if(root.classList.contains('is-layer-camera-transitioning'))return api;
      if(selectedLayer!==null&&controller.state.view==='right')return queueExpandedLayerFit();
      if(controller.state.view!=='right'){controller.fit();schedule();return api;}
      const width=Math.max(1,viewport.clientWidth),height=Math.max(1,viewport.clientHeight);
      const topReserve=Number(options.topLaneReserve)||260,bottomReserve=Number(options.bottomLaneReserve)||130;
      const usableHeight=Math.max(420,height-topReserve-bottomReserve),zoom=clamp(Math.min(width/2100,usableHeight/1900),.12,.86);
      const compactStructureShift=activeMetricViews.size===0?100:0;
      fitZoom=zoom;controller.setPose({zoom,panY:(topReserve-bottomReserve)/2-compactStructureShift});schedule();return api;
    }
    function queueSidecarFit(){requestAnimationFrame(fitSidecar);}
    function closeSelectedLayer(afterLayer=null){
      if(selectedDetail)openDetail(null);
      const restorePose=focusRestorePose||poseSnapshot();
      focusRestorePose=null;pendingLayer=afterLayer;
      transitionCameraTo(restorePose,{expandedLayer:null,expansionDepth:0,onComplete(){
        selectedLayer=null;
        options.onLayerSelect?.(null,api);
        const next=pendingLayer;pendingLayer=null;
        if(next!==null)window.setTimeout(()=>selectLayer(next),40);
      }});
      return api;
    }
    function selectLayer(layer){
      const nextLayer=layer===null||layer===undefined||!Number.isFinite(Number(layer))?null:clamp(Number(layer),0,controller.config.layerCount-1);
      if(nextLayer===selectedLayer)return nextLayer===null?api:closeSelectedLayer();
      if(selectedLayer!==null)return closeSelectedLayer(nextLayer);
      if(nextLayer===null)return api;
      hideTooltip();
      focusRestorePose=poseSnapshot();selectedLayer=nextLayer;lossExpanded=false;
      const expansionDepth=Number(options.layerExpansionDepth)||640;
      if(selectedDetail)openDetail(null);
      const g=geometry(),layout=g&&measureFocus(g);
      if(layout){
        renderFocus(g,layout);applyMetricVisibility();
        transitionCameraTo(expandedLayerPose(layout),{expandedLayer:selectedLayer,expansionDepth});
      }else{
        controller.setLayerExpansion?.(selectedLayer,expansionDepth);schedule();queueExpandedLayerFit();
      }
      options.onLayerSelect?.({layer:selectedLayer,snapshot:snapshotFor(selectedLayer)},api);return api;
    }
    function setLayerSnapshot(layer,data){const key=Number(layer);if(Number.isFinite(key)){if(data)snapshots.set(key,data);else snapshots.delete(key);schedule(false);}return api;}
    function setMetricViews(values){
      const next=new Set((Array.isArray(values)?values:[]).filter(value=>metricViewValues.has(value)));
      const unchanged=next.size===activeMetricViews.size&&[...next].every(value=>activeMetricViews.has(value));if(unchanged)return api;
      activeMetricViews=next;syncMetricViews();schedule();
      if(selectedLayer!==null)requestAnimationFrame(queueExpandedLayerFit);
      else requestAnimationFrame(fitSidecar);
      const legacyValue=activeMetricViews.size===0?'structure':activeMetricViews.size===1?[...activeMetricViews][0]:'mixed';
      options.onMetricViewsChange?.([...activeMetricViews],api);options.onMetricViewChange?.(legacyValue,api);return api;
    }
    function setMetricView(value){return setMetricViews(value==='structure'?[]:[value]);}
    function toggleMetricView(value){if(!metricViewValues.has(value))return api;const next=new Set(activeMetricViews);if(next.has(value))next.delete(value);else next.add(value);return setMetricViews([...next]);}
    function eventLayer(event){const target=event.target.closest?.('.pto-training-sidecar__layer-hit');return target?Number(target.dataset.layer):null;}
    function activateLayerHit(event){
      if(event.type==='keydown'&&event.key!=='Enter'&&event.key!==' ')return;
      const layer=Number(event.currentTarget.dataset.layer);
      if(!Number.isFinite(layer))return;
      event.preventDefault();event.stopPropagation();
      selectLayer(layer===selectedLayer?null:layer);
    }
    function layerHoverIn(event){const layer=eventLayer(event);if(layer!==null&&Number.isFinite(layer))setHoveredLayer(layer);}
    function layerHoverOut(event){
      const from=eventLayer(event),to=event.relatedTarget?.closest?.('.pto-training-sidecar__layer-hit'),toLayer=to?Number(to.dataset.layer):null;
      if(from!==null&&from===toLayer)return;setHoveredLayer(toLayer);
    }
    function layerFocusIn(event){const layer=eventLayer(event);if(layer!==null&&Number.isFinite(layer))setHoveredLayer(layer);}
    function layerFocusOut(event){const target=event.relatedTarget?.closest?.('.pto-training-sidecar__layer-hit');setHoveredLayer(target?Number(target.dataset.layer):null);}
    function detailClick(event){
      const target=event.target.closest?.('[data-detail]');
      if(target&&!target.closest('.pto-training-sidecar__inspector')&&!target.classList.contains('pto-training-sidecar__layer-hit')){
        try{openDetail(JSON.parse(target.dataset.detail),target);}catch(error){openDetail({title:'Annotation',category:'Detail',definition:target.dataset.tip||'Sidecar annotation',status:'info',statusLabel:'语义说明'},target);}
        return;
      }
      if(selectedLayer!==null&&!event.target.closest?.('.pto-training-sidecar__focus,.pto-training-sidecar__layer-hit,[data-stage-ui]'))selectLayer(null);
    }
    function protectDetailPointer(event){if(event.target.closest?.('[data-detail],.pto-training-sidecar__inspector'))event.stopPropagation();}
    function keydown(event){
      if(event.key==='Escape'&&selectedDetail){event.preventDefault();openDetail(null);return;}
      if(event.key==='Escape'&&selectedLayer!==null){event.preventDefault();selectLayer(null);return;}
      if((event.key==='ArrowLeft'||event.key==='ArrowRight')&&selectedLayer!==null){event.preventDefault();selectLayer(selectedLayer+(event.key==='ArrowLeft'?-1:1));}
    }
    function interaction(){syncOverlayTransform();}
    function tooltipTarget(event){return event.target.closest?.('[data-tip]');}
    function showTooltip(event){const target=tooltipTarget(event);if(!target)return;tooltip.textContent=target.dataset.tip;tooltip.classList.add('is-visible');moveTooltip(event);}
    function moveTooltip(event){if(!tooltip.classList.contains('is-visible'))return;const rect=viewport.getBoundingClientRect(),x=clamp(event.clientX-rect.left+12,8,rect.width-tooltip.offsetWidth-8),y=clamp(event.clientY-rect.top+12,8,rect.height-tooltip.offsetHeight-8);tooltip.style.left=`${x}px`;tooltip.style.top=`${y}px`;}
    function hideTooltip(event){if(event?.relatedTarget?.closest?.('[data-tip]'))return;tooltip.classList.remove('is-visible');}
    function pointerDown(event){if(event.button!==0||event.target.closest('[data-stage-ui],button,.pto-training-sidecar__layer-hit'))return;dragging=true;root.classList.add('is-sidecar-dragging');tooltip.classList.remove('is-visible');}
    function pointerMove(event){if(dragging)syncOverlayTransform();else moveTooltip(event);}
    function pointerUp(){if(!dragging)return;dragging=false;root.classList.remove('is-sidecar-dragging');syncOverlayTransform();}
    function closeDataMenu(){dataControl.classList.remove('is-open');dataControl.querySelector('[data-metric-menu-toggle]')?.setAttribute('aria-expanded','false');}
    function dataControlClick(event){
      const toggle=event.target.closest('[data-metric-menu-toggle]'),action=event.target.closest('[data-metric-action]'),option=event.target.closest('[data-metric-view]');
      if(toggle){const open=!dataControl.classList.contains('is-open');dataControl.classList.toggle('is-open',open);toggle.setAttribute('aria-expanded',String(open));return;}
      if(action){setMetricViews(action.dataset.metricAction==='all'?metricViewOptions.map(([value])=>value):[]);return;}
      if(option)toggleMetricView(option.dataset.metricView);
    }
    function dataOutsidePointer(event){if(!dataControl.contains(event.target))closeDataMenu();}
    function dataMenuKeydown(event){if(event.key==='Escape'&&dataControl.classList.contains('is-open')){event.preventDefault();closeDataMenu();dataControl.querySelector('[data-metric-menu-toggle]')?.focus();}}
    layerHits.addEventListener('pointerover',layerHoverIn);layerHits.addEventListener('pointerout',layerHoverOut);layerHits.addEventListener('focusin',layerFocusIn);layerHits.addEventListener('focusout',layerFocusOut);viewport.addEventListener('click',detailClick);viewport.addEventListener('pointerover',showTooltip);viewport.addEventListener('pointerdown',protectDetailPointer,true);viewport.addEventListener('pointerdown',pointerDown);viewport.addEventListener('pointermove',pointerMove);viewport.addEventListener('pointerup',pointerUp);viewport.addEventListener('pointercancel',pointerUp);viewport.addEventListener('pointerout',hideTooltip);viewport.addEventListener('wheel',interaction,{passive:true});viewport.addEventListener('keydown',keydown);
    dataControl.addEventListener('click',dataControlClick);dataControl.addEventListener('keydown',dataMenuKeydown);document.addEventListener('pointerdown',dataOutsidePointer);syncMetricViews();syncOverlayTransform();
    const observer=global.ResizeObserver?new ResizeObserver(()=>{fitSidecar();schedule();}):null;observer?.observe(viewport);
    const fitButton=root.querySelector('[data-deck-fit]'),fitClick=()=>queueSidecarFit();fitButton?.addEventListener('click',fitClick);
    const baseDestroy=controller.destroy.bind(controller);
    const api={
      root,base:controller,get selectedLayer(){return selectedLayer;},get selectedDetail(){return selectedDetail;},get metricView(){return activeMetricViews.size===0?'structure':activeMetricViews.size===1?[...activeMetricViews][0]:'mixed';},get metricViews(){return [...activeMetricViews];},selectLayer,collapseLayer(){return selectLayer(null);},openDetail,closeDetail(){return openDetail(null);},setLayerSnapshot,setMetricView,setMetricViews,toggleMetricView,
      setView(view){controller.setView(view);queueSidecarFit();return api;},setParallelMode(mode){controller.setParallelMode(mode);schedule();return api;},setTheme(theme){controller.setTheme(theme);schedule();return api;},setZoom(value){controller.setZoom(value);syncOverlayTransform();return api;},setPose(pose){controller.setPose(pose);syncOverlayTransform();return api;},setFrontLayer(layer){controller.setFrontLayer(layer);schedule();return api;},selectNode(nodeId,layer){return controller.selectNode(nodeId,layer);},fit(){return fitSidecar();},refresh(){controller.refresh();schedule();return api;},
      destroy(){destroyed=true;cancelAnimationFrame(raf);cancelAnimationFrame(focusFitRaf);clearTimeout(focusTransitionTimer);observer?.disconnect();fitButton?.removeEventListener('click',fitClick);dataControl.removeEventListener('click',dataControlClick);dataControl.removeEventListener('keydown',dataMenuKeydown);document.removeEventListener('pointerdown',dataOutsidePointer);layerHits.removeEventListener('pointerover',layerHoverIn);layerHits.removeEventListener('pointerout',layerHoverOut);layerHits.removeEventListener('focusin',layerFocusIn);layerHits.removeEventListener('focusout',layerFocusOut);viewport.removeEventListener('click',detailClick);viewport.removeEventListener('pointerover',showTooltip);viewport.removeEventListener('pointerdown',protectDetailPointer,true);viewport.removeEventListener('pointerdown',pointerDown);viewport.removeEventListener('pointermove',pointerMove);viewport.removeEventListener('pointerup',pointerUp);viewport.removeEventListener('pointercancel',pointerUp);viewport.removeEventListener('pointerout',hideTooltip);viewport.removeEventListener('wheel',interaction);viewport.removeEventListener('keydown',keydown);baseDestroy();}
    };
    requestAnimationFrame(()=>{fitSidecar();if(options.initialLayer!==null&&options.initialLayer!==undefined&&Number.isFinite(Number(options.initialLayer)))selectLayer(Number(options.initialLayer));});
    return api;
  }

  global.PtoModelArchitectureTrainingSidecar={render:mount,mount};
})(window);
