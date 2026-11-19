/* Identity-based transition coordinator. View renderers own camera/layout; this layer
 * only borrows screen-space snapshots, and never writes selection or camera state. */
(function(global){
  'use strict';
  const NS='http://www.w3.org/2000/svg';
  const rect=r=>({x:r.left,y:r.top,width:r.width,height:r.height});
  const mix=(a,b,t)=>a+(b-a)*t;
  const lerp=(a,b,t)=>Object.fromEntries(['x','y','width','height'].map(k=>[k,mix(a[k],b[k],t)]));
  const ease=t=>t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;
  function pairs(assignment,pipeline){
    const blocks=new Map(assignment.blocks.map(b=>[b.id,b]));
    const tasks=new Map(assignment.tasks.map(t=>[t.id,t]));
    return {blocks:pipeline.blocks.filter(b=>blocks.has(b.id)).map(b=>({source:blocks.get(b.id),target:b})),
      tasks:pipeline.tasks.filter(t=>tasks.has(t.id)).map(t=>({source:tasks.get(t.id),target:t}))};
  }
  function focusBlock(snapshot,selected){
    const tasksByBlock=new Map();
    snapshot.tasks.forEach(task=>{
      const block=task.task?.block??task.block;
      tasksByBlock.set(block,(tasksByBlock.get(block)||0)+1);
    });
    const visible=block=>block.rect.y>=snapshot.clip.y&&
      block.rect.y+block.rect.height<=snapshot.clip.y+snapshot.clip.height;
    const choice=snapshot.blocks.find(block=>block.id===selected&&tasksByBlock.has(block.id)&&visible(block));
    return choice?.id??snapshot.blocks.find(block=>tasksByBlock.has(block.id)&&visible(block))?.id??null;
  }
  function create({root,getPipeline,getGraph,getIntraTask}){
    let job=null,serial=0,lastFocus=null;
    const intra=global.PtoVisualization2IntraMorph.create({root,getTask:getIntraTask});
    const media=global.matchMedia('(prefers-reduced-motion: reduce)');
    const originRect=()=>{
      const bounds=root.getBoundingClientRect();
      return {left:bounds.left+root.clientLeft,top:bounds.top+root.clientTop,width:root.clientWidth,height:root.clientHeight};
    };
    const el=(tag,attrs={},parent)=>{const n=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));parent?.append(n);return n;};
    function assignmentSnapshot(){
      const host=root.querySelector('#viz2-attention'), viewport=host.querySelector('.kf-pa-map__viewport');
      const blockCount=host.querySelectorAll('[data-pa-block]').length;
      return {clip:rect(viewport.getBoundingClientRect()),
        blocks:[...host.querySelectorAll('[data-pa-block]')].map(button=>{
          const frame=button.querySelector('.viz2-am-block-frame'),label=button.querySelector('.viz2-am-block-label');
          const icon=document.createElementNS(NS,'svg');icon.setAttribute('viewBox','0 0 1 1');
          return {id:Number(button.dataset.paBlock),rect:rect(frame.getBoundingClientRect()),icon,
            iconRect:rect(frame.getBoundingClientRect()),labelRect:rect(label.getBoundingClientRect())};
        }),
        tasks:[...host.querySelectorAll('[data-pa-task]')].map(task=>{
          const frame=task.querySelector('.kf-pa-map__cell'),r=rect(frame.getBoundingClientRect());
          const sequence=task.querySelector('.viz2-am-seq-encoding');
          const tracks=[...sequence.querySelectorAll('.ai-stack-track')];
          const fills=[...sequence.querySelectorAll('.ai-stack-fill')];
          const label=task.querySelector('.kf-pa-map__task-id') || frame;
          const sourceScale=r.width/(Number(frame.getAttribute('width'))||r.width);
          return {id:Number(task.dataset.paTask),block:Number(task.dataset.paTask)%blockCount,rect:r,
            cell:{fill:getComputedStyle(frame).fill,stroke:getComputedStyle(frame).stroke,strokeWidth:getComputedStyle(frame).strokeWidth},
            label:{rect:rect(label.getBoundingClientRect()),fontSize:(parseFloat(getComputedStyle(label).fontSize)||12)*sourceScale,
              fill:getComputedStyle(label).fill,stroke:getComputedStyle(label).stroke},
            segments:tracks.map((track,i)=>({track:rect(track.getBoundingClientRect()),fill:rect(fills[i].getBoundingClientRect()),
              trackColor:getComputedStyle(track).fill,fillColor:getComputedStyle(fills[i]).fill,
              fillOpacity:parseFloat(getComputedStyle(fills[i]).fillOpacity)||1}))};
        })};
    }
    function mask(view,value){
      if(view==='pipeline')getPipeline()?.setTransitionMasked(value);
      else if(view==='graph')root.classList.toggle('viz2-morph-graph',value);
      else root.classList.toggle('viz2-morph-assignment',value);
    }
    function cancel(){
      intra.cancel();
      if(!job)return;
      cancelAnimationFrame(job.frame);job.animations?.forEach(animation=>animation.cancel());job.overlay?.remove();job.hold?.remove();
      job.hiddenTasks?.forEach(([node,visibility])=>{node.style.visibility=visibility;});
      getPipeline()?.setTransitionFocusBlock(null);
      root.querySelector(`#viz2-${job.to}`)?.style.removeProperty('opacity');
      mask(job.to,false);root.classList.remove('viz2-morph-preparing');delete root.dataset.morphDestination;job=null;
    }
    function prepare(state,to){
      cancel();
      if(state.views[state.activeTab]?.subview==='block')return intra.prepare(state,to);
      if(!state.open||state.activeTab===to||media.matches)return null;
      const from=state.activeTab;
      const assignmentPipeline=(from==='assignment'&&to==='pipeline')||(from==='pipeline'&&to==='assignment');
      const pipelineGraph=(from==='pipeline'&&to==='graph')||(from==='graph'&&to==='pipeline');
      const assignmentGraph=(from==='assignment'&&to==='graph')||(from==='graph'&&to==='assignment');
      if(!assignmentPipeline&&!pipelineGraph&&!assignmentGraph)return null;
      if(assignmentGraph&&state.views.assignment.phase!=='attention')return null;
      if(assignmentPipeline&&state.views.assignment.phase!=='attention')return null;
      const focus=assignmentPipeline&&from==='pipeline'?(state.selection?.block??lastFocus??0):null;
      if(focus!==null)getPipeline()?.ensureTransitionBlockVisible(focus);
      const source=from==='assignment'?assignmentSnapshot():from==='graph'?getGraph()?.getBlockTransitionSnapshot():getPipeline()?.getTransitionSnapshot();
      if(!source)return null;
      job={id:++serial,from,to,source,selection:state.selection,focus,frame:0,phase:'preparing'};
      // Keep a frozen source visible while the destination measures itself. Canvas
      // pixels must be copied explicitly: cloneNode alone leaves them empty.
      if(assignmentGraph||(pipelineGraph&&from==='graph')){
        root.classList.add('viz2-morph-preparing');root.dataset.morphDestination=to;
        return job;
      }
      const pane=root.querySelector(`#viz2-${from}`), bounds=pane.getBoundingClientRect(), origin=originRect();
      const hold=pane.cloneNode(true);hold.removeAttribute('id');hold.setAttribute('aria-hidden','true');hold.inert=true;
      Object.assign(hold.style,{position:'absolute',left:`${bounds.left-origin.left}px`,top:`${bounds.top-origin.top}px`,width:`${bounds.width}px`,height:`${bounds.height}px`,zIndex:11,pointerEvents:'none',margin:0});
      const originals=pane.querySelectorAll('canvas');hold.querySelectorAll('canvas').forEach((canvas,index)=>canvas.getContext('2d').drawImage(originals[index],0,0));
      job.hold=hold;root.append(hold);
      const descendants=hold.querySelectorAll('*');pane.querySelectorAll('*').forEach((node,index)=>{
        if(node.scrollTop||node.scrollLeft){descendants[index].scrollTop=node.scrollTop;descendants[index].scrollLeft=node.scrollLeft;}
      });
      // Hide the destination before its very first paint. Visibility retains layout.
      root.classList.add('viz2-morph-preparing');root.dataset.morphDestination=to;
      return job;
    }
    function play(current){
      if(current?.phase&&current.source?.objects){intra.play(current);return;}
      if(!current||job!==current)return;
      // Two frames settle first-entry camera and Inspector-dependent width, including ResizeObserver.
      current.frame=requestAnimationFrame(()=>{current.frame=requestAnimationFrame(()=>start(current));});
    }
    // Paged Attention owns the same execution Block IDs as Assignment.
    // Never fall back to Preparation: its lane set can be a strict subset.
    function startAssignmentGraph(current){
      const forward=current.to==='graph';
      const target=forward?getGraph()?.getBlockTransitionSnapshot():assignmentSnapshot();
      if(!target){cancel();return;}
      const graph=forward?target:current.source, assignment=forward?current.source:target;
      const normalize=r=>({x:r.x??r.left,y:r.y??r.top,width:r.width,height:r.height});
      const visible=(block,clip)=>{
        const r=normalize(block.rect),c=normalize(clip);
        return r.width>0&&r.height>0&&r.x<c.x+c.width&&r.x+r.width>c.x&&r.y<c.y+c.height&&r.y+r.height>c.y;
      };
      const anchors=new Map(graph.blocks.filter(b=>b.stage==='paged-attention').map(b=>[b.id,b]));
      const origin=originRect(),style=getComputedStyle(root),color=n=>style.getPropertyValue(n).trim();
      const overlay=el('svg',{class:'viz2-transition-overlay',width:origin.width,height:origin.height,
        'aria-hidden':'true','data-direction':forward?'assignment-graph':'graph-assignment'},root);
      current.overlay=overlay;
      const items=assignment.blocks.filter(b=>anchors.has(b.id)&&visible(b,assignment.clip)&&visible(anchors.get(b.id),graph.clip)).map(b=>{
        const anchor=anchors.get(b.id),selected=b.id===current.selection?.block;
        const group=el('g',{'data-morph-block':b.id},overlay);
        const box=el('rect',{fill:color(selected?'--state-selected':'--surface-3'),stroke:color(selected?'--primary':'--border-subtle'),'stroke-width':selected?2:1},group);
        const label=el('text',{'text-anchor':'middle','dominant-baseline':'central',fill:color('--foreground'),'font-family':color('--font-mono'),'font-size':12},group);
        window.PtoPhotoNarrative.text(label,String(b.id).padStart(2,'0'));
        return {a:normalize(forward?b.rect:anchor.rect),b:normalize(forward?anchor.rect:b.rect),box,label};
      });
      const destination=root.querySelector('#viz2-'+current.to);
      mask(current.to,true);
      root.classList.remove('viz2-morph-preparing');
      current.phase='animating';
      const started=performance.now(),duration=760;
      function frame(now){
        if(job!==current)return;
        const t=Math.min(1,(now-started)/duration),p=ease(t);
        destination.style.opacity=String(Math.max(0,(t-.35)/.65));
        items.forEach(item=>{
          const r=lerp(item.a,item.b,p);
          for(const k of ['x','y','width','height'])item.box.setAttribute(k,k==='x'?r.x-origin.left:k==='y'?r.y-origin.top:r[k]);
          item.box.setAttribute('rx',mix(forward?4:3,forward?3:4,p));
          item.label.setAttribute('x',r.x-origin.left+r.width/2);
          item.label.setAttribute('y',r.y-origin.top+r.height/2);
          item.box.style.opacity=item.label.style.opacity=String(1-Math.max(0,(t-.85)/.15));
        });
        if(t<1)current.frame=requestAnimationFrame(frame);else cancel();
      }
      frame(started);
    }
    function startPipelineGraph(current){
      const forward=current.to==='graph';
      const target=forward?getGraph()?.getBlockTransitionSnapshot():getPipeline()?.getTransitionSnapshot();
      if(!target?.blocks.length){cancel();return;}
      const origin=originRect(),style=getComputedStyle(root),color=name=>style.getPropertyValue(name).trim();
      const pipeline=forward?current.source:target,graph=forward?target:current.source;
      const normalize=r=>({x:r.x??r.left,y:r.y??r.top,width:r.width,height:r.height});
      const sources=new Map(pipeline.blocks.map(block=>[block.id,block]));
      const visible=block=>block.rect.y+block.rect.height>pipeline.clip.y&&block.rect.y<pipeline.clip.y+pipeline.clip.height;
      const mapping=graph.blocks.filter(block=>sources.has(block.id)&&visible(sources.get(block.id))).map(block=>{
        const lane=sources.get(block.id);
        return {source:{...(forward?lane:block),rect:normalize(forward?lane.rect:block.rect)},
          target:{...(forward?block:lane),rect:normalize(forward?block.rect:lane.rect)},stage:block.stage};
      });
      const overlay=el('svg',{class:'viz2-transition-overlay',width:origin.width,height:origin.height,'aria-hidden':'true','data-direction':forward?'pipeline-graph':'graph-pipeline'},root);
      current.overlay=overlay;
      const items=mapping.map(pair=>{
        const selected=pair.source.id===current.selection?.block;
        const group=el('g',{'data-morph-block':pair.source.id,'data-morph-stage':pair.stage||''},overlay);
        const box=el('rect',{rx:4,fill:selected?color('--state-selected'):color('--surface-3'),stroke:selected?color('--primary'):color('--border-subtle')},group);
        const label=el('text',{'text-anchor':'middle','dominant-baseline':'central',fill:selected?color('--foreground'):color('--foreground-secondary'),'font-family':color('--font-mono'),'font-size':12},group);
        window.PtoPhotoNarrative.text(label,String(pair.source.id).padStart(2,'0'));return {...pair,box,label};
      });
      mask(current.to,true);root.classList.remove('viz2-morph-preparing');
      const destination=root.querySelector('#viz2-'+current.to);
      current.animations=[destination.animate([{opacity:0},{opacity:1}],{duration:700,fill:'both',easing:'ease-in-out'})];
      if(current.hold)current.animations.push(current.hold.animate([{opacity:1},{opacity:0}],{duration:420,fill:'both',easing:'ease-in-out'}));
      current.phase='animating';const started=performance.now(),duration=760;
      function frame(now){
        if(job!==current)return;
        const progress=Math.min(1,(now-started)/duration),p=ease(progress);
        items.forEach(item=>{
          const r=lerp(item.source.rect,item.target.rect,p);
          for(const key of ['x','y','width','height'])item.box.setAttribute(key,key==='x'?r.x-origin.left:key==='y'?r.y-origin.top:r[key]);
          item.label.setAttribute('x',r.x-origin.left+r.width/2);
          item.label.setAttribute('y',r.y-origin.top+r.height/2);
          item.box.style.opacity=item.label.style.opacity=String(1-Math.max(0,(progress-.82)/.18));
        });
        if(progress<1)current.frame=requestAnimationFrame(frame);else cancel();
      }
      frame(started);
    }
    function start(current){
      if(job!==current)return;
      if((current.from==='assignment'&&current.to==='graph')||(current.from==='graph'&&current.to==='assignment')){startAssignmentGraph(current);return;}
      if((current.from==='pipeline'&&current.to==='graph')||(current.from==='graph'&&current.to==='pipeline')){startPipelineGraph(current);return;}
      const forward=current.to==='pipeline';
      let focus=forward?focusBlock(current.source,current.selection?.block):current.focus;
      if(forward&&focus!==null)getPipeline()?.ensureTransitionBlockVisible(focus);
      const target=forward?getPipeline()?.getTransitionSnapshot():assignmentSnapshot();
      const destination=root.querySelector(`#viz2-${current.to}`);
      if(!target||!destination){cancel();return;}
      const assignment=forward?current.source:target,pipeline=forward?target:current.source;
      if(!forward&&focus===null)focus=focusBlock(assignment,current.selection?.block);
      const assigned=new Map(assignment.tasks.map(task=>[task.id,task]));
      const pipelineTasks=pipeline.tasks.filter(task=>task.task.block===focus);
      const assignmentTasks=assignment.tasks.filter(task=>task.block===focus);
      const valid=focus!==null&&pipelineTasks.length>0&&pipelineTasks.length===assignmentTasks.length&&
        pipelineTasks.every(task=>assigned.has(task.id)&&assigned.get(task.id).segments.length&&task.rect.width>0&&task.task.segments.length)&&
        assignment.blocks.some(block=>block.id===focus)&&pipeline.blocks.some(block=>block.id===focus)&&
        assignment.clip.width>0&&pipeline.clip.width>0;
      if(valid)lastFocus=focus;
      const origin=originRect(),style=getComputedStyle(root),color=name=>style.getPropertyValue(name).trim();
      const overlay=el('svg',{class:'viz2-transition-overlay',width:origin.width,height:origin.height,
        'aria-hidden':'true','data-direction':forward?'forward':'reverse','data-focus-block':focus??''},root);
      current.overlay=overlay;
      const local=r=>({x:r.x-origin.left,y:r.y-origin.top,width:r.width,height:r.height});
      const setRect=(node,r)=>{for(const key of ['x','y','width','height'])node.setAttribute(key,r[key]);};
      const shapes=valid?pipelineTasks.map(targetTask=>{
        const source=assigned.get(targetTask.id),target=targetTask;
        const group=el('g',{'data-morph-task':target.id,'data-morph-block':focus},overlay);
        const base=el('rect',{rx:3,fill:source.cell.fill,stroke:source.cell.stroke,
          'stroke-width':source.cell.strokeWidth},group);
        const count=Math.max(source.segments.length,target.task.segments.length);
        const segments=Array.from({length:count},(_,index)=>{
          const sourceSegment=source.segments[index];
          const fromTrack=sourceSegment?local(sourceSegment.track):local({x:source.rect.x+source.rect.width,y:source.rect.y,width:0,height:source.rect.height});
          const fromFill=sourceSegment?local(sourceSegment.fill):{...fromTrack,width:0};
          const unit=target.rect.width/target.task.segments.length,gap=Math.min(2,unit*.1);
          const stack=target.task.segments[index];
          const toTrack=local({x:target.rect.x+Math.min(index,target.task.segments.length)*unit,y:target.rect.y,
            width:stack?unit-gap:0,height:target.rect.height});
          const toFill={...toTrack,width:toTrack.width*(stack?.fraction||0)};
          const sourceTrack=el('rect',{rx:1,fill:sourceSegment?.trackColor||color('--surface-2')},group);
          const targetTrack=el('rect',{rx:1,fill:color('--surface-4')},group);
          const sourceFill=el('rect',{rx:1,fill:sourceSegment?.fillColor||color('--ai-data-fill')},group);
          const targetFill=el('rect',{rx:1,fill:color('--ai-data-fill')},group);
          const pages=Array.from({length:Math.max(0,(stack?.pages.length||0)-1)},()=>el('line',{
            stroke:color('--surface-2'),'stroke-width':.7},group));
          return {fromTrack,fromFill,toTrack,toFill,sourceTrack,targetTrack,sourceFill,targetFill,
            sourceFillOpacity:sourceSegment?.fillOpacity||1,pages,stack};
        });
        const backing=el('rect',{fill:color('--surface-2')},group);
        const outline=el('rect',{fill:'none',stroke:target.id===current.selection?.task?color('--foreground'):color('--primary'),
          'stroke-width':target.id===current.selection?.task?2:1},group);
        const label=el('text',{'text-anchor':'middle','dominant-baseline':'central',fill:source.label.fill,
          stroke:source.label.stroke,'paint-order':'stroke fill','stroke-width':2,
          'stroke-linejoin':'round','font-family':target.label.fontFamily,'font-weight':600},group);
        window.PtoPhotoNarrative.text(label,String(target.id));
        return {source,target,group,base,segments,backing,outline,label};
      }):[];
      // The frozen source omits only the focus Tasks. The overlay owns those pixels
      // from the first frame until the destination renderer takes over.
      if(valid){
        if(forward){
          const ids=new Set(shapes.map(shape=>shape.target.id));
          current.hold.querySelectorAll('[data-pa-task]').forEach(node=>{
            if(ids.has(Number(node.dataset.paTask)))node.style.visibility='hidden';
          });
          getPipeline()?.setTransitionFocusBlock(focus);
        }else{
          const canvas=current.hold.querySelector('canvas');
          const original=root.querySelector('#viz2-pipeline canvas');
          if(canvas&&original){
            const bounds=original.getBoundingClientRect(),sx=canvas.width/bounds.width,sy=canvas.height/bounds.height;
            const ctx=canvas.getContext('2d');
            shapes.forEach(({target})=>ctx.clearRect((target.rect.x-bounds.left)*sx,(target.rect.y-bounds.top)*sy,
              target.rect.width*sx,target.rect.height*sy));
          }
          current.hiddenTasks=shapes.map(({source})=>{
            const node=root.querySelector(`#viz2-attention [data-pa-task="${source.id}"]`),previous=node.style.visibility;
            node.style.visibility='hidden';return [node,previous];
          });
        }
      }
      destination.style.opacity='0';current.hold.style.opacity='1';
      root.classList.remove('viz2-morph-preparing');current.phase='animating';
      const started=performance.now(),duration=valid?850:350;
      const fromClip=forward?assignment.clip:pipeline.clip,toClip=forward?pipeline.clip:assignment.clip;
      function frame(now){
        if(job!==current)return;
        const t=Math.max(0,Math.min(1,(now-started)/duration));
        const p=ease(t),sourceAlpha=valid?1-Math.min(1,t/.35):1-t;
        const targetAlpha=valid?Math.max(0,Math.min(1,(t-.55)/.45)):t;
        current.hold.style.opacity=String(sourceAlpha);destination.style.opacity=String(targetAlpha);
        if(valid){
          const q=forward?p:1-p;
          shapes.forEach(({source,target,group,base,segments,backing,outline,label})=>{
            group.style.opacity='1';
            const r=local(lerp(source.rect,target.rect,q));
            setRect(base,r);base.style.opacity=String(1-q);
            segments.forEach(part=>{
              const track=lerp(part.fromTrack,part.toTrack,q),fill=lerp(part.fromFill,part.toFill,q);
              setRect(part.sourceTrack,track);setRect(part.targetTrack,track);
              setRect(part.sourceFill,fill);setRect(part.targetFill,fill);
              part.sourceTrack.style.opacity=String(1-q);part.targetTrack.style.opacity=String(q);
              part.sourceFill.style.opacity=String((1-q)*part.sourceFillOpacity);
              part.targetFill.style.opacity=String(q);
              part.pages.forEach((line,index)=>{
                const x=track.x+track.width*(index+1)/(part.pages.length+1);
                line.setAttribute('x1',x);line.setAttribute('x2',x);
                line.setAttribute('y1',track.y);line.setAttribute('y2',track.y+track.height);
                line.style.opacity=String(part.stack&&target.rect.width/target.task.segments.length>=28?q:0);
              });
            });
            const sourceLabel=local(source.label.rect),lx=mix(sourceLabel.x+sourceLabel.width/2,r.x+r.width/2,q);
            const ly=mix(sourceLabel.y+sourceLabel.height/2,r.y+r.height/2,q);
            label.setAttribute('x',lx);label.setAttribute('y',ly);
            label.setAttribute('font-size',mix(source.label.fontSize,target.label.size,q));
            label.setAttribute('font-weight',Math.round(mix(600,400,q)));
            label.style.opacity=String(q);
            label.style.strokeOpacity=String(1-q);
            const bw=target.label.width+4,bh=Math.min(target.rect.height,13);
            setRect(backing,{x:r.x+(r.width-bw)/2,y:r.y+(r.height-bh)/2,width:bw,height:bh});
            backing.style.opacity=String(q);
            setRect(outline,{x:r.x+1,y:r.y+1,width:Math.max(0,r.width-2),height:Math.max(0,r.height-2)});
            outline.style.opacity=String(q*(target.id===current.selection?.task||target.task.block===current.selection?.block?1:0));
          });
          const clip=lerp(fromClip,toClip,p);
          overlay.style.clipPath=`inset(${Math.max(0,clip.y-origin.top)}px ${Math.max(0,origin.left+origin.width-clip.x-clip.width)}px ${Math.max(0,origin.top+origin.height-clip.y-clip.height)}px ${Math.max(0,clip.x-origin.left)}px)`;
        }
        if(t<1)current.frame=requestAnimationFrame(frame);else cancel();
      }
      frame(started);
    }
    global.addEventListener('resize',cancel);media.addEventListener('change',cancel);
    root.addEventListener('scroll',event=>{if(job?.phase==='animating'&&!job.hold?.contains(event.target))cancel();},true);
    return {prepare,play,cancel,get active(){return !!job||intra.active;},get animating(){return job?.phase==='animating'||intra.animating;}};
  }
  global.PtoVisualization2Transition=Object.freeze({create,pairs,focusBlock});
})(window);
