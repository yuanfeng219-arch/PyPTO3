/* Screen-space Task identity transition. Cameras belong to their view controllers. */
(function(global){
  'use strict';
  const NS='http://www.w3.org/2000/svg';
  function create({root,getOuter,getInner}){
    let job=null;const media=matchMedia('(prefers-reduced-motion: reduce)');
    function cancel(){if(!job)return;cancelAnimationFrame(job.frame);job.animations.forEach(a=>a.cancel());job.overlay?.remove();job.hold.remove();getOuter()?.setSemanticTaskMask(null);getInner()?.setTransitionMasked(false);job=null;}
    const outer=id=>getOuter()?.getTransitionSnapshot()?.tasks.find(t=>t.id===id);
    function prepare(direction,id){
      cancel();if(media.matches)return null;
      const source=direction==='in'?outer(id):getInner()?.getTransitionSnapshot();if(!source)return null;
      const pane=root.querySelector(direction==='in'?'#viz2-pipeline-inter':'#viz2-pipeline-intra');
      const r=pane.getBoundingClientRect(),o=root.getBoundingClientRect(),hold=pane.cloneNode(true);hold.removeAttribute('id');hold.inert=true;hold.setAttribute('aria-hidden','true');
      Object.assign(hold.style,{position:'absolute',left:`${r.left-o.left-root.clientLeft}px`,top:`${r.top-o.top-root.clientTop}px`,width:`${r.width}px`,height:`${r.height}px`,margin:0,zIndex:12,pointerEvents:'none'});
      const originals=pane.querySelectorAll('canvas');hold.querySelectorAll('canvas').forEach((canvas,i)=>canvas.getContext('2d').drawImage(originals[i],0,0));root.append(hold);
      const descendants=hold.querySelectorAll('*');pane.querySelectorAll('*').forEach((element,index)=>{descendants[index].scrollTop=element.scrollTop;descendants[index].scrollLeft=element.scrollLeft;});
      job={direction,id,source,hold,animations:[],frame:0,phase:'preparing'};return job;
    }
    function play(current){if(!current||current!==job)return;current.frame=requestAnimationFrame(()=>{current.frame=requestAnimationFrame(()=>start(current));});}
    function start(current){
      if(current!==job)return;const entering=current.direction==='in',target=entering?getInner()?.getTransitionSnapshot():outer(current.id);if(!target){cancel();return;}
      const a=entering?current.source:target,b=entering?target:current.source,o=root.getBoundingClientRect(),style=getComputedStyle(root),color=n=>style.getPropertyValue(n).trim();
      const make=(tag,attrs,parent)=>{const el=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));parent?.append(el);return el;};
      const overlay=make('svg',{class:'viz2-semantic-overlay',width:root.clientWidth,height:root.clientHeight,'data-pipeline-semantic-task':current.id,'data-direction':current.direction,'aria-hidden':'true'},root);current.overlay=overlay;
      const frame=make('rect',{fill:color('--surface-3'),rx:4,stroke:color('--foreground'),'stroke-width':1},overlay);
      const output=make('rect',{fill:color('--surface-3'),stroke:color('--accent'),'stroke-width':1.5,rx:3},overlay);
      const outputFill=make('rect',{fill:color('--accent'),rx:3},overlay);
      const label=make('text',{fill:color('--foreground'),'font-family':color('--font-mono'),'text-anchor':'middle','dominant-baseline':'central'},overlay);label.textContent=`Task ${current.id}`;
      const segments=b.stacks.map(s=>({source:{...a.rect,x:a.rect.x+a.rect.width*s.index/b.stacks.length,width:a.rect.width/b.stacks.length},target:s.rect,fraction:b.model.stacks[s.index].tokens/b.model.stackTokens,
        track:make('rect',{fill:color('--surface-4'),rx:3},overlay),fill:make('rect',{fill:color('--primary'),rx:3},overlay)}));
      overlay.append(label);
      if(entering){
        getInner().setTransitionMasked(true);
        const canvas=current.hold.querySelector('canvas'),r=canvas.getBoundingClientRect();
        const scale=canvas.width/r.width,context=canvas.getContext('2d');
        context.clearRect((a.rect.x-r.left)*scale,(a.rect.y-r.top)*scale,(a.rect.width+a.output.width)*scale,a.rect.height*scale);
      }else{
        getOuter().setSemanticTaskMask(current.id);
        current.hold.querySelector(`[data-intra-task="${current.id}"]`)?.querySelectorAll('[data-intra-frame],[data-intra-kind="output"],[data-intra-operation$=":qk"],.viz2-intra-heading').forEach(el=>el.style.visibility='hidden');
      }
      current.phase='animating';
      const destination=root.querySelector(entering?'#viz2-pipeline-intra':'#viz2-pipeline-inter');
      current.animations.push(current.hold.animate([{opacity:1},{opacity:0}],{duration:420,easing:'ease-in-out',fill:'both'}));
      current.animations.push(destination.animate([{opacity:0},{opacity:0,offset:.15},{opacity:1}],{duration:720,easing:'ease-in-out',fill:'both'}));
      const mix=(x,y,t)=>x+(y-x)*t,lerp=(x,y,t)=>Object.fromEntries(['x','y','width','height'].map(k=>[k,mix(x[k],y[k],t)]));
      const set=(el,r)=>Object.entries({...r,x:r.x-o.left-root.clientLeft,y:r.y-o.top-root.clientTop}).forEach(([k,v])=>el.setAttribute(k,v));
      const start=performance.now();function tick(now){if(job!==current)return;const raw=Math.min(1,(now-start)/780),e=raw<.5?4*raw**3:1-(-2*raw+2)**3/2,p=entering?e:1-e;
        const bounds=lerp(a.rect,b.frame,p);set(frame,bounds);frame.style.opacity=String(1-p*.7);set(output,lerp(a.output,b.output,p));set(outputFill,lerp(a.output,b.output,p));outputFill.style.opacity=String(1-p);frame.setAttribute('stroke-opacity',1-p);
        label.setAttribute('text-anchor',p>.95?'start':'middle');
        label.setAttribute('x',mix(a.rect.x+a.rect.width/2,b.title.x,p)-o.left-root.clientLeft);label.setAttribute('y',mix(a.rect.y+a.rect.height/2,b.title.y+b.title.height/2,p)-o.top-root.clientTop);label.setAttribute('font-size',mix(12,b.title.height*.7,p));
        segments.forEach(s=>{const r=lerp(s.source,s.target,p);set(s.track,r);set(s.fill,{...r,width:r.width*s.fraction});s.fill.style.opacity=String(1-p);});
        if(raw<1)current.frame=requestAnimationFrame(tick);else cancel();
      }tick(start);
    }
    global.addEventListener('resize',cancel);media.addEventListener('change',cancel);
    return {prepare,play,cancel,get animating(){return job?.phase==='animating';}};
  }
  global.PtoVisualization2PipelineZoom=Object.freeze({create});
})(window);
