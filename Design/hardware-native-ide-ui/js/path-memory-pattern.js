/* Adapter for shared memory-architecture, AIC and AIV renderers.
 * Compact source-bound preset. Grid cells express storage texture, not byte counts.
 * Quantitative occupancy is exclusively represented by the adjacent UB address bar.
 */
(function () {
  const buffer = (key, capacity, label = key+' buffers') => ({kind:'buffer',key,label:key,capacity,frame:{width:200,height:72},surfaceRenderer:()=>PathGridGlyphs.memory(label,capacity,200,72,true),
    grid:{rows:2,cols:8,cellSize:10,gap:2}});
  let resizeObserver;
  function drawRelations(mount,canvas,onLayout) {
    const svg=mount.closest('svg');if(!svg||!mount.isConnected)return;
    svg.querySelector('[data-path-relations]')?.remove();
    const ns='http://www.w3.org/2000/svg', layer=document.createElementNS(ns,'g');layer.dataset.pathRelations='true';
    const point=(selector,side)=>{
      const e=canvas.querySelector(selector);if(!e)return null;
      const r=e.getBoundingClientRect(),p=svg.createSVGPoint();
      p.x=side==='right'?r.right:side==='left'?r.left:r.left+r.width/2;
      p.y=side==='top'?r.top:side==='bottom'?r.bottom:r.top+r.height/2;
      return p.matrixTransform(svg.getScreenCTM().inverse());
    };
    const route=(d,label,x,y,mapping=false,region,detail=false)=>{
      const group=document.createElementNS(ns,'g');group.classList.add('sv-region');if(region)group.dataset.region=region;if(detail)group.classList.add('path-detail-route');
      const path=document.createElementNS(ns,'path');path.setAttribute('d',d);path.setAttribute('class',mapping?'path-map-link':'sv-arrow path-data-link');group.appendChild(path);
      const text=document.createElementNS(ns,'text');text.setAttribute('x',x);text.setAttribute('y',y);text.setAttribute('class','path-relation-label');text.textContent=label;group.appendChild(text);layer.appendChild(group);
    };
    const l0c=point('[data-aic-node="buffer:L0C"]','bottom'),ub=point('[data-aiv-node="buffer:UB"]','top'),left=point('[data-aic-node="buffer:L0A"]','bottom'),ur=point('[data-aiv-node="buffer:UB"]','top');
    const ubRight=point('[data-aiv-node="buffer:UB"]','right');
    if(l0c&&ur)route(`M${l0c.x} ${l0c.y} V${ur.y-10} H${ur.x+35} V${ur.y}`, 'C2V · QK → softmax',l0c.x+5,l0c.y+18,false,'ch-c2v');
    if(left&&ur)route(`M${ur.x} ${ur.y} H${left.x} V${left.y}`, 'V2C · P → PV',left.x+7,(ur.y+left.y)/2,false,'ch-v2c');
    const ubBottom=point('[data-aiv-node="buffer:UB"]','bottom');
    if(ubBottom)route(`M${ubBottom.x} ${ubBottom.y} V486 H418 V462`, '同一 UB · 地址展开',337,491,true,'ub');
    const l0b=point('[data-aic-node="buffer:L0B"]','left');
    if(l0b) {
      route(`M99 74 V144 H18 V${l0b.y} H${l0b.x}`, '读 K / V',32,109,false,'gm-kcache');
      route(`M257 74 V140 H14 V${l0b.y+4} H${l0b.x}`, '',0,0,false,'gm-vcache');
    }
    route('M425 74 V86 H393 V96','',0,0,true,'gm-table');
    if(l0c)route(`M${l0c.x-7} ${l0c.y} H18 V80 H607 V74`, 'oi_tmp 写回',26,240,false,'gm-out',true);
    const ubLeft=point('[data-aiv-node="buffer:UB"]','left');
    if(ubLeft)route(`M${ubLeft.x} ${ubLeft.y} H18 V80 H607 V74`, 'mi / li 写回',26,386,false,'gm-out',true);
    svg.appendChild(layer);
    if (onLayout) onLayout();
  }
  window.PathMemoryPattern = { render(mount, k, onLayout) {
    PtoAicCorePattern.presets.pathAic = {
      id:'pathAic',title:'AIC · CUBE',frame:{width:540},routes:[
      {from:'buffer:L0A',to:'cube:Cube',color:'neutral',style:'straight',fromSide:'top',toSide:'bottom',toBias:0.5},
      {from:'buffer:L0B',to:'cube:Cube',color:'neutral',style:'straight',fromSide:'bottom',toSide:'top',toBias:0.5},
      {from:'cube:Cube',to:'buffer:L0C',color:'neutral',style:'straight',fromSide:'bottom',toSide:'top',fromBias:0.8}],
      layout:{kind:'group',className:'pto-aic-core__layout',children:[
        buffer('L0B',Math.round(k.kDB/1024)+' / 64 KB · K / V','L0B · K / V'),
        {kind:'cube',label:'Cube',frame:{width:110,height:110}},
        {kind:'group',className:'pto-aic-core__top-row',children:[buffer('L0A','[16,128] · 4 KB','L0A · q_padded'),buffer('L0C',Math.round(k.acc/1024)+' KB · oi_tmp 8 KB','L0C · raw_scores')]}
      ]}
    };
    const aiv = structuredClone(PtoAivCorePattern.presets.ascend910b);
    aiv.id='pathAiv';aiv.title='AIV · VECTOR';aiv.frame={width:540};aiv.routes=aiv.routes.filter(r=>/vector/.test(r.from+r.to)).map((r,i)=>({...r,color:'neutral',fromBias:i===0?35/90:75/110,toBias:i===0?35/110:55/90}));
    aiv.layout.children[0].children=[{kind:'buffer',key:'UB',label:'UB',capacity:'192 KB · 可用 184 KB',frame:{width:250,height:90},surfaceRenderer:()=>PathGridGlyphs.memory('UB · softmax','192 KB · 可用 184 KB',250,90,true),grid:{rows:3,cols:12,cellSize:10,gap:2}}];
    aiv.layout.children[1].frame={width:110,height:110};
    delete aiv.layout.children[1].surfaceRenderer;
    PtoAivCorePattern.presets.pathAiv=aiv;
    resizeObserver?.disconnect();
    const canvas=document.createElement('div');canvas.style.width='max-content';canvas.style.transformOrigin='top left';mount.appendChild(canvas);
    PtoMemoryArchitecturePattern.renderArchitecture(canvas,{id:'path-a2a3',rails:[],cores:[
      {id:'path-aic',kind:'aic',title:'AIC · CUBE',presetKey:'pathAic'},
      {id:'path-aiv',kind:'aiv',title:'AIV · VECTOR',presetKey:'pathAiv'}],routes:[]});
    const aicStage=canvas.querySelector('.pto-aic-core');
    if(aicStage){
      aicStage.classList.add('sv-region');aicStage.dataset.region='aic';
    }
    const aivStage=canvas.querySelector('.pto-aiv-core');
    if(aivStage){aivStage.classList.add('sv-region');aivStage.dataset.region='aiv';}
    const regions={'buffer:L0A':'l0a','buffer:L0B':'l0b','buffer:L0C':'l0c','cube:Cube':'cube','buffer:UB':'ub','vector:Vector':'aiv'};
    canvas.querySelectorAll('[data-aic-node],[data-aiv-node]').forEach(el=>{
      const id=regions[el.dataset.aicNode||el.dataset.aivNode];if(id){el.classList.add('sv-region');el.dataset.region=id;}
    });
    canvas.querySelectorAll('[data-aic-route-from],[data-aiv-route-from]').forEach(path=>{
      const from=path.dataset.aicRouteFrom||path.dataset.aivRouteFrom;
      const to=path.dataset.aicRouteTo||path.dataset.aivRouteTo;
      path.classList.add('sv-region','path-internal-route');
      path.dataset.region=regions[from]||regions[to]||'aiv';
    });
    const fit=()=>{
      if(!mount.isConnected)return;
      canvas.style.transform='none';
      const stack=canvas.querySelector('.pto-mem950__stack'), cr=canvas.getBoundingClientRect(),sr=stack.getBoundingClientRect();
      const scale=Math.min(mount.clientWidth/stack.offsetWidth,mount.clientHeight/stack.offsetHeight);
      const screenScale=mount.getBoundingClientRect().width/mount.clientWidth;
      canvas.style.transform='translate('+(-(sr.left-cr.left)/screenScale*scale)+'px,'+(-(sr.top-cr.top)/screenScale*scale)+'px) scale('+scale+')';
      requestAnimationFrame(()=>drawRelations(mount,canvas,onLayout));
    };
    requestAnimationFrame(fit);
    resizeObserver=new ResizeObserver(fit);resizeObserver.observe(mount);
  }};
})();
