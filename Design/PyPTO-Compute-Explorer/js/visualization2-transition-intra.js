/* Block-local semantic morph. All geometry is measured from existing rendered objects. */
(function(global){
  'use strict';
  const NS='http://www.w3.org/2000/svg';
  const stageNode={qk:'qk',softmax:'softmax',pv:'pv',accumulate:'context-acc'};
  const nodeStage=Object.fromEntries(Object.entries(stageNode).map(([stage,node])=>[node,stage]));
  const panes={assignment:'#viz2-anatomy',pipeline:'#viz2-pipeline-intra',graph:'#viz2-graph-intra'};
  const clamp=n=>Math.max(0,Math.min(1,n));
  const ease=n=>n<.5?4*n*n*n:1-Math.pow(-2*n+2,3)/2;
  const rect=element=>{const r=element.getBoundingClientRect();return {x:r.left,y:r.top,width:r.width,height:r.height};};
  const mix=(a,b,t)=>a+(b-a)*t;
  const visible=(item,clip)=>item&&item.box.width>0&&item.box.height>0&&item.box.x<clip.x+clip.width&&item.box.x+item.box.width>clip.x&&item.box.y<clip.y+clip.height&&item.box.y+item.box.height>clip.y;
  const svg=(name,attributes,parent)=>{const element=document.createElementNS(NS,name);for(const [key,value] of Object.entries(attributes))element.setAttribute(key,value);parent.append(element);return element;};
  function stackOf(focus,task){
    if(Number.isInteger(focus?.stackIndex))return focus.stackIndex;
    if(['stack','stage','event'].includes(focus?.kind))return focus.index;
    if(focus?.kind==='page')return task?.pages?.[focus.index]?.stack;
    return null;
  }
  function mapFocus(from,to,focus,task){
    if(!focus)return null;
    const stack=stackOf(focus,task),valid=Number.isInteger(stack)&&stack>=0&&stack<task.stacks.length;
    if(to==='assignment'){
      if(focus.kind==='q'||focus.node==='q')return {kind:'q',index:0};
      if(focus.kind==='output'||['output','output-norm'].includes(focus.node))return {kind:'output',index:0};
      if(focus.kind==='page'&&task.pages[focus.index])return {kind:'page',index:focus.index};
      if(focus.kind==='slot'&&task.slots[focus.index])return {kind:'slot',index:focus.index};
      return valid?{kind:'stack',index:stack}:null;
    }
    if(to==='pipeline'){
      if(focus.kind==='output'||['output','output-norm'].includes(focus.node))return {kind:'output',index:0};
      if(focus.kind==='slot'&&task.slots[focus.index])return focus;
      if(focus.kind==='event'&&valid)return focus;
      const stage=focus.kind==='stage'?focus.stage:nodeStage[focus.node?.split('/')[0]];
      if(stage&&valid)return {kind:'stage',index:stack,stage};
      return valid?{kind:'stack',index:stack}:null;
    }
    if(to==='graph'){
      if(focus.kind==='output')return {kind:'output',index:0,stackIndex:valid?stack:null};
      if(focus.kind==='q')return {kind:'graph',node:'q',index:valid?stack:0,stackIndex:valid?stack:null};
      if(focus.kind==='event'&&valid)return {...focus,stackIndex:stack};
      if(focus.kind==='stage'&&stageNode[focus.stage]&&valid)return {...focus,stackIndex:stack};
      if(focus.kind==='page'&&task.pages[focus.index])return {...focus,stackIndex:stack};
      return valid?{kind:'stack',index:stack,stackIndex:stack}:null;
    }
    return null;
  }
  function create({root,getTask}){
    let job=null,serial=0;
    const media=global.matchMedia('(prefers-reduced-motion: reduce)');
    const pane=tab=>root.querySelector(panes[tab]);
    const origin=()=>{const r=root.getBoundingClientRect();return {x:r.left+root.clientLeft,y:r.top+root.clientTop,width:root.clientWidth,height:root.clientHeight};};
    function snapshot(tab,task,precise=false){
      const host=pane(tab),viewport=host?.querySelector(tab==='graph'?'.viz2-graph-viewport':'.viz2-anatomy-viewport');
      if(!host||!viewport)return null;
      const objects=new Map(),add=(id,element,paint=element)=>{
        if(!element)return;
        const box=rect(precise?paint:element),style=getComputedStyle(paint),vector=paint instanceof SVGElement,item={id,element,box,
          fill:vector?(style.fill&&style.fill!=='none'?style.fill:'transparent'):style.backgroundColor,
          stroke:vector?(style.stroke&&style.stroke!=='none'?style.stroke:'transparent'):style.borderColor,
          radius:(parseFloat(style.rx)||parseFloat(style.borderRadius)||5)*(vector?(paint.getScreenCTM()?.a||1):(element.offsetWidth?box.width/element.offsetWidth:1))};
        if(precise){item.radius=Math.min(item.radius,box.width/2,box.height/2);
        const scale=vector?Math.abs(paint.getScreenCTM()?.a||1):(paint.offsetWidth?box.width/paint.offsetWidth:1);
        item.strokeWidth=(parseFloat(vector?style.strokeWidth:style.borderTopWidth)||0)*scale;
        // CSS borders paint inside the box; SVG strokes are centered.
        if(!vector){const inset=item.strokeWidth/2;item.box={x:box.x+inset,y:box.y+inset,width:box.width-2*inset,height:box.height-2*inset};item.radius=Math.max(0,item.radius-inset);}
        }
        objects.set(id,item);
      };
      if(tab==='assignment'){
        const current=host.querySelector(`[data-anatomy-task="${task}"]`);
        for(const element of [...host.querySelectorAll('.viz2-anatomy-query [data-anatomy-kind]'),...(current?.querySelectorAll('[data-anatomy-kind]')||[])]){
          const {anatomyKind:kind,anatomyIndex:index}=element.dataset;
          add(`${kind}:${index}`,element,element.querySelector('rect')||element);
        }
      }else if(tab==='pipeline'){
        const current=host.querySelector(`[data-intra-task="${task}"]`);
        if(!current)return null;
        for(const element of current.querySelectorAll('[data-intra-kind="stage"]'))add(`stage:${element.dataset.intraIndex}:${element.dataset.intraStage}`,element,element.querySelector('rect')||element);
        const output=current.querySelector('[data-intra-kind="output"]'),cards=output?.querySelectorAll('.viz2-intra-output');
        if(cards?.[0])add('normalize',cards[0]);if(cards?.[1])add('output',cards[1]);
      }else{
        for(const element of host.querySelectorAll('[data-pypa-node]'))add(`node:${element.dataset.pypaNode}`,element,element);
        add('container:0',host.querySelector('[data-pypa-region="task-compute"]'));
        for(const element of host.querySelectorAll('[data-stack-container-layer]'))add(`container:${element.dataset.stackContainerLayer}`,element);
        add('qk-layer:0',host.querySelector('[data-pypa-node="qk"]'));
        for(const element of host.querySelectorAll('[data-pypa-execution-stack="qk"] [data-pypa-execution-instance]'))add(`qk-layer:${Number(element.dataset.pypaExecutionInstance)-1}`,element);
        const operations=['qk','softmax','pv','context-acc'].map(id=>objects.get(`node:${id}`)).filter(Boolean);
        if(operations.length===4){
          const left=Math.min(...operations.map(item=>item.box.x)),top=Math.min(...operations.map(item=>item.box.y));
          const right=Math.max(...operations.map(item=>item.box.x+item.box.width)),bottom=Math.max(...operations.map(item=>item.box.y+item.box.height));
          objects.set('region:stack-loop',{id:'region:stack-loop',element:operations[0].element,elements:operations.map(item=>item.element),box:{x:left,y:top,width:right-left,height:bottom-top},fill:'transparent',stroke:operations[0].stroke,radius:8});
        }
      }
      return {tab,task,host,clip:rect(viewport),objects};
    }
    function pairs(source,target,focus,task){
      const result=[],add=(a,b,kind,label,priority=0)=>{
        const s=source.objects.get(a),t=target.objects.get(b);
        if(visible(s,source.clip)&&visible(t,target.clip))result.push({source:s,target:t,kind,label,priority});
      };
      const direction=`${source.tab}>${target.tab}`,s=stackOf(focus,task),selected=Number.isInteger(s)?s:0;
      if(direction==='assignment>pipeline'||direction==='pipeline>assignment'){
        task.stacks.forEach(stack=>Object.keys(stageNode).forEach((stage,index)=>{
          const a=`stack:${stack.index}`,b=`stage:${stack.index}:${stage}`;
          if(source.tab==='assignment')add(a,b,'stack-stage',`S${stack.index} · ${stage}`,index*38+stack.index*8);
          else add(b,a,'stage-stack',`S${stack.index}`,index*38+stack.index*8);
        }));
        add(source.tab==='assignment'?'output:0':'output',source.tab==='assignment'?'output':'output:0','output','Output',120);
      }else if(direction==='pipeline>graph'||direction==='graph>pipeline'){
        Object.entries(stageNode).forEach(([stage,node],index)=>{
          const a=`stage:${selected}:${stage}`,b=`node:${node}`;
          if(source.tab==='pipeline')add(a,b,'stage-node',stage.toUpperCase(),index*45);
          else add(b,a,'node-stage',`S${selected} · ${stage}`,index*45);
        });
        if(source.tab==='pipeline'){add('normalize','node:output-norm','operation','Normalize',170);add('output','node:output','output','Output',210);}
        else {add('node:output-norm','normalize','operation','Normalize',170);add('node:output','output','output','Output',210);}
      }else if(direction==='assignment>graph'||direction==='graph>assignment'){
        const forward=source.tab==='assignment',assignment=forward?source:target,graph=forward?target:source;
        const groups=[['stack',task.stacks,'container:'],['page',task.pages,'qk-layer:']];
        for(const [kind,entities,prefix] of groups){
          const entries=entities.map((entity,index)=>({a:assignment.objects.get(`${kind}:${entity.index??index}`),g:graph.objects.get(`${prefix}${index}`),index}));
          // Require a complete one-layer-per-instance mapping; Structure falls back safely.
          if(entries.some(({a,g})=>!a||!g))return [];
          const center=entries.reduce((v,{a})=>({x:v.x+(a.box.x+a.box.width/2)/entries.length,y:v.y+(a.box.y+a.box.height/2)/entries.length}),{x:0,y:0});
          const base=entries[0].a.box;
          const graphBase=entries[0].g.box;
          for(const {a,g,index} of entries){
            const gather={x:center.x-base.width/2+(g.box.x-graphBase.x),y:center.y-base.height/2+(g.box.y-graphBase.y),width:base.width,height:base.height};
            result.push({source:forward?a:g,target:forward?g:a,kind:'gather',family:kind,index,gather,forward,label:'',priority:0});
          }
        }
        // Paint the backmost instances first, Pages above the large Stack outlines.
        result.sort((a,b)=>(a.family==='stack'?0:1)-(b.family==='stack'?0:1)||b.index-a.index);
      }
      return result;
    }
    function cancel(){
      if(!job)return;
      cancelAnimationFrame(job.frame);
      job.animations?.forEach(animation=>animation.cancel());
      job.masked?.forEach(({element,previous})=>{if(previous)element.style.visibility=previous;else element.style.removeProperty('visibility');});
      job.destination?.style.removeProperty('opacity');
      if(job.pendingDestination)job.pendingDestination.style.opacity=job.pendingOpacity;
      job.restores?.forEach(restore=>restore());
      job.hold?.remove();job.overlay?.remove();
      root.classList.remove('viz2-morph-preparing');delete root.dataset.morphDestination;
      job=null;
    }
    function finish(current){
      if(job!==current)return;
      current.phase='handoff';
      current.animations.forEach(animation=>animation.finish());
      current.masked.forEach(({element,previous})=>{element.style.visibility=previous;});
      current.masked=[];
      if(current.pendingDestination)current.pendingDestination.style.opacity=current.pendingOpacity;
      // Keep the exact final outlines over a painted destination for one full frame.
      current.frame=requestAnimationFrame(()=>{if(job===current)current.frame=requestAnimationFrame(()=>{if(job===current)cancel();});});
    }
    function startGather(current,target,mapping){
      const o=origin(),overlay=svg('svg',{class:'viz2-transition-overlay',width:o.width,height:o.height,viewBox:`0 0 ${o.width} ${o.height}`,'aria-hidden':'true','data-direction':`${current.from}-${current.to}`},root);
      Object.assign(overlay.style,{overflow:'visible',visibility:'visible',opacity:'1'});
      current.overlay=overlay;current.destination=target.host;current.restores=[];
      const mask=element=>{
        if(current.masked.some(entry=>entry.element===element))return;
        current.masked.push({element,previous:element.style.visibility});element.style.visibility='hidden';
      };
      const items=mapping.map(pair=>{
        const box=svg('rect',{'data-intra-morph':`${pair.source.id}>${pair.target.id}`,fill:'none',opacity:1},overlay);
        const graph=pair.forward?pair.target:pair.source;
        const outline=item=>({...item,stroke:['none','transparent','rgba(0, 0, 0, 0)'].includes(item.stroke)?graph.stroke:item.stroke,strokeWidth:item.strokeWidth||1.5});
        return {...pair,source:outline(pair.source),target:outline(pair.target),box};
      });
      function draw(elapsed){
        const time=mapping[0].forward?Math.min(elapsed,900):Math.max(0,900-elapsed);
        for(const item of items){
          const a=item.forward?item.source:item.target,b=item.forward?item.target:item.source;
          const gathering=time<=400,t=ease(clamp(gathering?time/400:(time-400)/500));
          const left=gathering?a.box:item.gather,right=gathering?item.gather:b.box;
          const shape=ease(clamp((time-400)/500));
          for(const key of ['x','y','width','height'])item.box.setAttribute(key,String(mix(left[key],right[key],t)-(key==='x'?o.x:key==='y'?o.y:0)));
          item.box.setAttribute('rx',String(mix(a.radius,b.radius,shape)));
          item.box.setAttribute('stroke-width',String(mix(a.strokeWidth,b.strokeWidth,shape)));
          item.box.style.stroke=`color-mix(in srgb, ${a.stroke} ${100*(1-shape)}%, ${b.stroke})`;
        }
        overlay.dataset.progress=String(Math.min(1,elapsed/1100));
      }
      // Draw at source coordinates BEFORE hiding either real or cloned outlines.
      draw(0);
      for(const item of items){
        const clone=current.cloneMap.get(item.source.element);if(clone)clone.style.visibility='hidden';
        mask(item.target.element);
        if(item.target.id==='qk-layer:0'){const label=target.host.querySelector('[data-viz2-graph-label="qk"]');if(label)mask(label);}
        if(item.source.id==='qk-layer:0'){const label=current.source.host.querySelector('[data-viz2-graph-label="qk"]');const clone=label&&current.cloneMap.get(label);if(clone)clone.style.visibility='hidden';}
      }
      // Only non-proxy content can fade. The SVG is a sibling of both hosts.
      current.animations=[current.hold.animate([{opacity:1},{opacity:0}],{duration:400,fill:'both'}),target.host.animate([{opacity:0},{opacity:0,offset:.55},{opacity:1,offset:.85},{opacity:1}],{duration:1100,fill:'both'})];
      root.classList.remove('viz2-morph-preparing');delete root.dataset.morphDestination;
      const wasInert=target.host.inert;target.host.inert=true;current.restores.push(()=>{target.host.inert=wasInert;});
      current.phase='animating';const started=performance.now();
      function frame(now){
        if(job!==current)return;
        const elapsed=Math.min(1100,now-started);draw(elapsed);
        if(elapsed<1100)current.frame=requestAnimationFrame(frame);else finish(current);
      }
      current.frame=requestAnimationFrame(frame);
    }
    function prepare(state,to){
      cancel();
      const from=state.activeTab,task=state.selection?.task;
      if(!state.open||from===to||media.matches||state.selection?.kind!=='task'||state.views[from]?.subview!=='block'||!Number.isInteger(task))return null;
      const gathered=[from,to].includes('assignment')&&[from,to].includes('graph');
      const source=snapshot(from,task,gathered);
      if(!source)return null;
      const bounds=rect(source.host),o=origin(),hold=source.host.cloneNode(true);
      hold.removeAttribute('id');hold.setAttribute('aria-hidden','true');hold.inert=true;
      Object.assign(hold.style,{position:'absolute',left:`${bounds.x-o.x}px`,top:`${bounds.y-o.y}px`,width:`${bounds.width}px`,height:`${bounds.height}px`,zIndex:11,pointerEvents:'none',margin:0});
      const originals=source.host.querySelectorAll('canvas');hold.querySelectorAll('canvas').forEach((canvas,index)=>canvas.getContext('2d').drawImage(originals[index],0,0));
      root.append(hold);
      const descendants=hold.querySelectorAll('*');source.host.querySelectorAll('*').forEach((element,index)=>{if(element.scrollTop||element.scrollLeft){descendants[index].scrollTop=element.scrollTop;descendants[index].scrollLeft=element.scrollLeft;}});
      const cloneMap=new WeakMap(),bind=(a,b)=>{cloneMap.set(a,b);for(let i=0;i<a.children.length;i++)bind(a.children[i],b.children[i]);};bind(source.host,hold);
      const current={id:++serial,from,to,task,source,gathered,focus:state.selection.subobject,hold,cloneMap,frame:0,phase:'preparing',masked:[],animations:[]};
      if(gathered){current.pendingDestination=pane(to);current.pendingOpacity=current.pendingDestination.style.opacity;current.pendingDestination.style.opacity='0';}
      job=current;root.classList.add('viz2-morph-preparing');root.dataset.morphDestination=to;
      return current;
    }
    function play(current){
      if(!current||current!==job)return;
      const gatherPair=[current.from,current.to].includes('assignment')&&[current.from,current.to].includes('graph');
      let last='',stable=0,attempts=0;
      function ready(){
        if(job!==current)return;
        const target=snapshot(current.to,current.task,current.gathered);
        const signature=target?JSON.stringify([...target.objects].map(([id,item])=>[id,item.box])):'';
        stable=signature&&signature===last?stable+1:0;last=signature;
        if(!gatherPair||stable>=2){start(current);return;}
        if(++attempts>=30){cancel();return;}
        current.frame=requestAnimationFrame(ready);
      }
      current.frame=requestAnimationFrame(ready);
    }
    function start(current){
      if(current!==job)return;
      const target=snapshot(current.to,current.task,current.gathered);
      if(!target||target.task!==current.task){cancel();return;}
      // The source and destination renderers share Task-local Stack/Page IDs.
      const task=getTask(current.task);
      if(!task){cancel();return;}
      const mapping=pairs(current.source,target,current.focus,task);
      if(!mapping.length){cancel();return;}
      if(mapping[0]?.kind==='gather'){startGather(current,target,mapping);return;}
      const o=origin(),overlay=svg('svg',{class:'viz2-transition-overlay',width:o.width,height:o.height,'aria-hidden':'true','data-direction':`${current.from}-${current.to}`},root);
      overlay.style.overflow='visible';
      current.overlay=overlay;current.destination=target.host;
      // A cloned source holds unmatched content; matched elements are owned by the overlay.
      for(const pair of mapping)for(const element of pair.source.elements||[pair.source.element])current.cloneMap.get(element)?.style.setProperty('visibility','hidden');
      for(const pair of mapping)for(const element of pair.target.elements||[pair.target.element]){if(current.masked.some(item=>item.element===element))continue;current.masked.push({element,previous:element.style.visibility});element.style.setProperty('visibility','hidden');}
      root.classList.remove('viz2-morph-preparing');delete root.dataset.morphDestination;
      const destination=target.host;
      current.animations=[current.hold.animate([{opacity:1},{opacity:1,offset:.17},{opacity:0}],{duration:780,fill:'both',easing:'ease-in-out'}),destination.animate([{opacity:0},{opacity:0,offset:.35},{opacity:1}],{duration:900,fill:'both',easing:'ease-in-out'})];
      const items=mapping.map(pair=>{
        const group=svg('g',{'data-intra-morph':`${pair.source.id}>${pair.target.id}`},overlay);
        const box=svg('rect',{rx:pair.source.radius,fill:pair.source.fill||'transparent',stroke:pair.source.stroke||'none','stroke-width':1.5},group);
        const label=svg('text',{'text-anchor':'middle','dominant-baseline':'central','font-size':11,'font-family':'var(--font-mono)','fill':'var(--foreground)'},group);
        label.textContent=pair.label;
        current.animations.push(box.animate([{fill:pair.source.fill||'transparent',stroke:pair.source.stroke||'none'},{fill:pair.target.fill||'transparent',stroke:pair.target.stroke||'none'}],{duration:650,fill:'both',easing:'ease-in-out'}));
        return {...pair,box,label};
      });
      current.phase='animating';const started=performance.now(),duration=900;
      function frame(now){
        if(job!==current)return;
        const elapsed=now-started;
        for(const item of items){
          const p=ease(clamp((elapsed-150-item.priority*.5)/610)),a=item.source.box,b=item.target.box;
          let geometry={x:mix(a.x,b.x,p),y:mix(a.y,b.y,p),width:mix(a.width,b.width,p),height:mix(a.height,b.height,p)};

          const {width,height}=geometry,x=geometry.x-o.x,y=geometry.y-o.y;
          for(const [key,value] of Object.entries({x,y,width,height,rx:mix(item.source.radius,item.target.radius,p)}))item.box.setAttribute(key,String((key==='width'||key==='height'?Math.max(.1,value):value)));
          item.label.setAttribute('x',String(x+width/2));item.label.setAttribute('y',String(y+height/2));
          item.label.style.opacity=String(Math.min(1,p*4)*(1-Math.max(0,(p-.78)/.22)));
          item.box.style.opacity=String(1-Math.max(0,(p-.88)/.12));
        }
        if(elapsed<duration)current.frame=requestAnimationFrame(frame);else cancel();
      }
      frame(started);
    }
    global.addEventListener('resize',cancel);media.addEventListener('change',cancel);
    root.addEventListener('scroll',event=>{if(job?.phase==='animating'&&!job.hold?.contains(event.target))cancel();},true);
    return {prepare,play,cancel,get active(){return !!job;},get animating(){return job?.phase==='animating';}};
  }
  global.PtoVisualization2IntraMorph=Object.freeze({create,mapFocus,stackOf});
})(window);
