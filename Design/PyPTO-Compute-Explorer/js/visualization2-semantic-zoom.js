/* Screen-space semantic zoom: one Task retains identity while Q-head stripes
 * become Q rows and the Batch capacity strip unfolds into Pages nested in Stack containers. */
(function(global){
  'use strict';
  const NS='http://www.w3.org/2000/svg',mix=(a,b,t)=>a+(b-a)*t;
  const interpolate=(a,b,t)=>Object.fromEntries(['x','y','width','height'].map(key=>[key,mix(a[key],b[key],t)]));
  function create({root,getMatrix,getAnatomy}){
    let job=null;const media=global.matchMedia('(prefers-reduced-motion: reduce)');
    const origin=()=>{const r=root.getBoundingClientRect();return {x:r.left+root.clientLeft,y:r.top+root.clientTop,width:root.clientWidth,height:root.clientHeight};};
    function cancel(){
      if(!job)return;const old=job;job=null;cancelAnimationFrame(old.frame);old.animations?.forEach(animation=>animation.cancel());
      old.overlay?.remove();old.hold?.remove();root.classList.remove('viz2-semantic-mask');
      old.masked?.forEach(([element,visibility])=>{element.style.visibility=visibility;});
    }
    function prepare(direction,id){
      cancel();if(media.matches)return null;
      const source=direction==='in'?getMatrix(id):getAnatomy();if(!source)return null;
      const pane=root.querySelector(direction==='in'?'#viz2-attention':'#viz2-anatomy'),r=pane.getBoundingClientRect(),o=origin();
      const hold=pane.cloneNode(true);hold.removeAttribute('id');hold.inert=true;hold.setAttribute('aria-hidden','true');
      Object.assign(hold.style,{position:'absolute',left:`${r.left-o.x}px`,top:`${r.top-o.y}px`,width:`${r.width}px`,height:`${r.height}px`,margin:0,zIndex:12,pointerEvents:'none'});
      const descendants=hold.querySelectorAll('*');root.append(hold);pane.querySelectorAll('*').forEach((element,index)=>{if(element.scrollTop||element.scrollLeft){descendants[index].scrollTop=element.scrollTop;descendants[index].scrollLeft=element.scrollLeft;}});
      job={direction,id,source,hold,frame:0,phase:'preparing',animations:[],masked:[]};return job;
    }
    function play(current){if(!current||job!==current)return;current.phase='queued';current.frame=requestAnimationFrame(()=>{current.frame=requestAnimationFrame(()=>start(current));});}
    function start(current){
      if(job!==current)return;
      const entering=current.direction==='in',target=entering?getAnatomy():getMatrix(current.id);
      if(!target){cancel();return;}
      const matrix=entering?current.source:target,anatomy=entering?target:current.source,o=origin();
      const style=getComputedStyle(root),color=name=>style.getPropertyValue(name).trim();
      const make=(tag,attrs={},parent)=>{const element=document.createElementNS(NS,tag);for(const [key,value]of Object.entries(attrs))element.setAttribute(key,value);parent?.append(element);return element;};
      const overlay=make('svg',{class:'viz2-semantic-overlay',width:o.width,height:o.height,'aria-hidden':'true','data-semantic-task':current.id,'data-direction':current.direction});
      current.overlay=overlay;root.append(overlay);
      const frame=make('rect',{rx:4,fill:color('--surface-3'),stroke:color('--foreground'),'stroke-width':1.5},overlay);
      const prefix=make('text',{'text-anchor':'middle','dominant-baseline':'central',fill:color('--foreground'),'font-family':color('--font-mono')},overlay);prefix.textContent='Task ';
      const number=make('text',{'text-anchor':'middle','dominant-baseline':'central',fill:color('--foreground'),'font-family':color('--font-mono')},overlay);number.textContent=current.id;
      const qRows=matrix.qRows.map((source,index)=>({source,target:anatomy.qRows[index],node:make('rect',{fill:color('--primary'),rx:1},overlay)}));
      const cells=new Map(anatomy.cells.map(cell=>[cell.key,cell.rect]));
      const parts=[];
      matrix.segments.forEach((segment,stack)=>{
        for(let page=0;page<anatomy.model.pagesPerStack;page++){
          const count=anatomy.model.pagesPerStack,index=stack*count+page;
          const source={x:segment.x+segment.width*page/count,y:segment.y,width:segment.width/count,height:segment.height};
          const pageTokens=anatomy.model.pageTokens;
          const fraction=Math.max(0,Math.min(1,(anatomy.model.seqLen-index*pageTokens)/pageTokens));
          const targetCell=cells.get(`${stack}:${page}`);
          const add=(destination,disappears=false)=>parts.push({source,target:destination,fraction,disappears,
            track:make('rect',{fill:color('--surface-4'),rx:1},overlay),fill:make('rect',{fill:color('--primary'),rx:1},overlay)});
          if(targetCell)add(targetCell);else add({...anatomy.frame,x:anatomy.frame.x+anatomy.frame.width/2,y:anatomy.frame.y+anatomy.frame.height/2,width:0,height:0},true);
        }
      });
      if(entering){
        root.classList.add('viz2-semantic-mask');
        current.hold.querySelector(`[data-pa-task="${current.id}"]`)?.setAttribute('visibility','hidden');
        current.hold.querySelector(`[data-pa-column="${matrix.model.batch}"] .kf-pa-map__sequence`)?.setAttribute('visibility','hidden');
      }else{
        current.hold.querySelectorAll('[data-anatomy-q-row],[data-anatomy-cell],[data-anatomy-page-fill],[data-anatomy-task-frame],.viz2-anatomy-task-title').forEach(element=>{element.style.visibility='hidden';});
        const cell=root.querySelector(`#viz2-attention [data-pa-task="${current.id}"]`),strip=root.querySelector(`#viz2-attention [data-pa-column="${matrix.model.batch}"] .kf-pa-map__sequence`);
        [cell,strip].forEach(element=>{current.masked.push([element,element.style.visibility]);element.style.visibility='hidden';});
      }
      const destination=root.querySelector(entering?'#viz2-anatomy':'#viz2-attention');
      current.animations.push(current.hold.animate([{opacity:1},{opacity:0}],{duration:650,fill:'both',easing:'ease-in-out'}));
      current.animations.push(destination.animate([{opacity:0},{opacity:0,offset:.25},{opacity:1}],{duration:720,fill:'both',easing:'ease-in-out'}));
      const local=r=>({...r,x:r.x-o.x,y:r.y-o.y});
      const set=(element,r)=>{for(const key of ['x','y','width','height'])element.setAttribute(key,r[key]);};
      current.phase='animating';const startTime=performance.now(),duration=760;
      function tick(now){
        if(job!==current)return;const raw=Math.min(1,(now-startTime)/duration),eased=raw<.5?4*raw*raw*raw:1-Math.pow(-2*raw+2,3)/2,p=entering?eased:1-eased;
        set(frame,local(interpolate(matrix.frame,anatomy.frame,p)));frame.style.opacity=String(mix(1,.35,p));frame.setAttribute('stroke-opacity',1-p);
        const numberWidth=anatomy.title.width*String(current.id).length/(`Task ${current.id}`).length;
        const targetNumber={x:anatomy.title.x+anatomy.title.width-numberWidth/2,y:anatomy.title.y+anatomy.title.height/2};
        const x=mix(matrix.title.x+matrix.title.width/2,targetNumber.x,p)-o.x,y=mix(matrix.title.y+matrix.title.height/2,targetNumber.y,p)-o.y;
        number.setAttribute('x',x);number.setAttribute('y',y);number.setAttribute('font-size',mix(matrix.title.height*.85,anatomy.title.height*.85,p));
        prefix.setAttribute('x',anatomy.title.x+(anatomy.title.width-numberWidth)/2-o.x);prefix.setAttribute('y',anatomy.title.y+anatomy.title.height/2-o.y);prefix.setAttribute('font-size',anatomy.title.height*.85);prefix.style.opacity=String(p);
        qRows.forEach(part=>set(part.node,local(interpolate(part.source,part.target,p))));
        parts.forEach(part=>{const r=local(interpolate(part.source,part.target,p));set(part.track,r);set(part.fill,{...r,width:r.width*part.fraction});part.track.style.opacity=part.fill.style.opacity=String(part.disappears?1-p:1);});
        if(raw<1)current.frame=requestAnimationFrame(tick);else cancel();
      }
      tick(startTime);
    }
    global.addEventListener('resize',cancel);media.addEventListener('change',cancel);
    return {prepare,play,cancel,get animating(){return job?.phase==='queued'||job?.phase==='animating';}};
  }
  global.PtoVisualization2SemanticZoom=Object.freeze({create});
})(window);
