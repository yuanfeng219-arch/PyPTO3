/* Viewport culling / hit-testing derived from pto/swimlane. Task appearance and
 * tooltip are direct PTO Pattern calls. No task bar geometry or palette fork. */
window.TraceRenderer=class {
 constructor(canvas,callbacks){
  this.canvas=canvas;this.ctx=canvas.getContext('2d');this.callbacks=callbacks;this.model=null;this.lanes=[];this.view=[0,1];this.scroll=0;this.row=27;this.header=30;this.labelWidth=130;this.hits=[];this.markers=[];this.activeMarkerKey=null;this.diagnosticMode=false;this.brush=false;this.color='semantic';this.selected=null;this.related=new Set();this.pending=0;
  this.palette=PtoSwimlaneTaskPattern.createTaskColormap();
  this.tipTarget=canvas.id==='traceCanvas'?document.getElementById('taskHit'):document.createElement('div');
  if(canvas.id!=='traceCanvas'){this.tipTarget.className='task-hit';this.tipTarget.hidden=true;canvas.parentElement.append(this.tipTarget);}
  this.tooltip=PtoSwimlaneTaskPattern.initHoverTooltip({root:canvas.parentElement,targets:[this.tipTarget],getTask:target=>target.__ptoSwimlaneTask,bounds:canvas.parentElement,durationUnit:'µs'});
  this.resize=new ResizeObserver(()=>this.draw());this.resize.observe(canvas.parentElement);
  canvas.addEventListener('wheel',e=>{e.preventDefault();this.hideTip();if(e.ctrlKey||e.metaKey)this.zoom(Math.exp(-e.deltaY*.008),(e.offsetX-this.labelWidth)/this.plotWidth());else if(e.shiftKey||Math.abs(e.deltaX)>Math.abs(e.deltaY))this.pan((e.deltaX||e.deltaY)/this.plotWidth()*(this.view[1]-this.view[0]));else{this.scroll+=e.deltaY;this.draw();}},{passive:false});
  canvas.addEventListener('pointerdown',e=>{if(!this.model)return;this.hideTip();canvas.focus();canvas.setPointerCapture(e.pointerId);this.drag={x:e.offsetX,y:e.offsetY,view:this.view.slice(),scroll:this.scroll,end:e.offsetX,endY:e.offsetY};});
  canvas.addEventListener('pointermove',e=>{if(this.drag){this.drag.end=e.offsetX;this.drag.endY=e.offsetY;if(!this.brush){const delta=(this.drag.x-e.offsetX)/this.plotWidth()*(this.drag.view[1]-this.drag.view[0]);this.view=this.clamp([this.drag.view[0]+delta,this.drag.view[1]+delta]);this.scroll=this.drag.scroll+this.drag.y-e.offsetY;}this.draw();}else this.hover(e);});
  canvas.addEventListener('pointerup',e=>{
   const d=this.drag;this.drag=null;if(!d)return;
   if(this.brush&&Math.abs(e.offsetX-d.x)>5){const range=[this.time(d.x),this.time(e.offsetX)].sort((a,b)=>a-b);const y1=Math.min(d.y,e.offsetY),y2=Math.max(d.y,e.offsetY);const from=Math.max(0,Math.floor((y1-this.header+this.scroll)/this.row)),to=Math.max(from,Math.floor((y2-this.header+this.scroll)/this.row));this.callbacks.onRange(range,this.lanes.slice(from,to+1).map(l=>l.id));}
   else if(Math.hypot(e.offsetX-d.x,e.offsetY-d.y)<5){const hit=this.hit(e.offsetX,e.offsetY);if(hit)this.callbacks.onTask(hit.task);else{const m=this.markerHits?.filter(m=>e.offsetX>=m.x&&e.offsetX<=m.x+m.w&&e.offsetY>=this.header).sort((a,b)=>(a.preset?1:0)-(b.preset?1:0)||a.w-b.w)[0];if(m)this.callbacks.onMarker(m.key);}}
   this.draw();
  });
  canvas.addEventListener('pointercancel',()=>{this.drag=null;this.draw();});canvas.addEventListener('pointerleave',()=>this.hideTip());
  canvas.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-','Enter'].includes(e.key)){e.preventDefault();if(e.key==='ArrowLeft'||e.key==='ArrowRight')this.pan((e.key==='ArrowLeft'?-1:1)*(this.view[1]-this.view[0])*.12);else if(e.key==='ArrowUp'||e.key==='ArrowDown'){this.scroll+=(e.key==='ArrowUp'?-1:1)*this.row;this.draw();}else if(e.key==='Enter'){const t=this.hits[0]?.task;if(t)this.callbacks.onTask(t);}else this.zoom(e.key==='-'?.7:1.4);}});
 }
 plotWidth(){return Math.max(1,this.canvas.clientWidth-this.labelWidth-12);}
 time(x){return Math.max(0,Math.min(this.model.duration,this.view[0]+(Math.max(this.labelWidth,Math.min(this.canvas.clientWidth-12,x))-this.labelWidth)/this.plotWidth()*(this.view[1]-this.view[0])));}
 x(t){return this.labelWidth+(t-this.view[0])/(this.view[1]-this.view[0])*this.plotWidth();}
 usableHeight(){const h=this.canvas.clientHeight,drawer=document.getElementById('investigationDrawer'),overlay=document.getElementById('investigationOverlay');return this.canvas.id==='traceCanvas'&&overlay&&!overlay.hidden?Math.max(this.header+this.row,drawer.getBoundingClientRect().top-this.canvas.getBoundingClientRect().top):h;}
 markerRects(marker){const w=this.canvas.clientWidth,limit=this.usableHeight(),left=Math.max(this.labelWidth,this.x(marker.range[0])),right=Math.min(w-12,this.x(marker.range[1]));if(right<=left)return [];const ids=new Set(marker.lanes||[]),indices=this.lanes.map((lane,index)=>ids.has(lane.id)?index:-1).filter(index=>index>=0);if(!indices.length)return [];const runs=[];for(const index of indices){const last=runs.at(-1);if(last&&index===last[1]+1)last[1]=index;else runs.push([index,index]);}return runs.map(([from,to])=>{const top=Math.max(this.header,this.header+from*this.row-this.scroll),bottom=Math.min(limit,this.header+(to+1)*this.row-this.scroll);return {x:left,y:top,w:right-left,h:bottom-top};}).filter(rect=>rect.h>0);}
 clamp([a,b]){const width=Math.min(this.model?.duration||1,Math.max(1,b-a));a=Math.max(0,Math.min((this.model?.duration||1)-width,a));return [a,a+width];}
 setModel(model){this.model=model;this.view=[0,model.duration];this.scroll=0;this.selected=null;this.related.clear();this.draw();}
 setLanes(lanes){this.lanes=lanes;this.draw();}
 pan(delta){this.view=this.clamp(this.view.map(v=>v+delta));this.draw();}
 zoom(factor,anchor=.5){if(!this.model)return;anchor=Math.max(0,Math.min(1,anchor));const old=this.view[1]-this.view[0],width=old/factor,t=this.view[0]+old*anchor;this.view=this.clamp([t-width*anchor,t+width*(1-anchor)]);this.hideTip();this.draw();}
 fit(){if(this.model){this.view=[0,this.model.duration];this.draw();}}
 focus(range,laneId){const margin=Math.max(20,(range[1]-range[0])*.18);this.view=this.clamp([range[0]-margin,range[1]+margin]);if(laneId){const i=this.lanes.findIndex(l=>l.id===laneId);if(i>=0)this.scroll=Math.max(0,i*this.row-this.row*2);}this.draw();}
 hit(x,y){return this.hits.findLast(h=>x>=h.x&&x<=h.x+h.w&&y>=h.y&&y<=h.y+h.h);}
 hideTip(){this.tipTarget.dispatchEvent(new PointerEvent('pointerleave'));this.tipTarget.hidden=true;this.hoverId=null;}
 hover(e){const h=this.hit(e.offsetX,e.offsetY);if(!h){this.hideTip();return;}if(this.hoverId===h.task.id)return;this.hideTip();this.hoverId=h.task.id;const target=this.tipTarget;target.hidden=false;target.__ptoSwimlaneTask={...h.task,laneId:h.task.laneName,start:h.task.start,end:h.task.end};Object.assign(target.style,{left:h.x+'px',top:h.y+'px',width:h.w+'px',height:h.h+'px'});target.dispatchEvent(new PointerEvent('pointerenter',{clientX:e.clientX,clientY:e.clientY}));}
 snapshot(){return {time_range:this.view.slice(),scroll:this.scroll,lane_ids:this.lanes.map(l=>l.id),color:this.color};}
 draw(){if(this.pending)return;this.pending=requestAnimationFrame(()=>{this.pending=0;this.paint();});}
 paint(){
  const c=this.canvas,ctx=this.ctx,w=c.clientWidth,h=c.clientHeight;if(!w||!h)return;const dpr=window.devicePixelRatio||1;c.width=Math.round(w*dpr);c.height=Math.round(h*dpr);ctx.scale(dpr,dpr);if(!this.model)return;
  this.labelWidth=w<450?102:142;const css=getComputedStyle(document.documentElement),color=n=>css.getPropertyValue(n).trim();const fg=color('--foreground-secondary'),grid=color('--border-subtle'),strong=color('--foreground');
  ctx.clearRect(0,0,w,h);ctx.font='12px '+css.getPropertyValue('--font-mono');ctx.textBaseline='middle';this.hits=[];this.markerHits=[];
  const usable=this.usableHeight();
  this.scroll=Math.max(0,Math.min(this.scroll,Math.max(0,this.lanes.length*this.row-(usable-this.header))));
  ctx.fillStyle=fg;ctx.fillText('Lane / 占用率',12,17);
  const count=Math.max(2,Math.floor(this.plotWidth()/110));
  for(let i=0;i<=count;i++){const x=this.labelWidth+this.plotWidth()*i/count,t=this.view[0]+(this.view[1]-this.view[0])*i/count;ctx.strokeStyle=grid;ctx.beginPath();ctx.moveTo(x,29);ctx.lineTo(x,h);ctx.stroke();ctx.fillStyle=fg;ctx.textAlign=i===count?'right':'left';ctx.fillText(t.toFixed(0)+' µs',x,17);}ctx.textAlign='left';
  ctx.save();ctx.beginPath();ctx.rect(this.labelWidth,28,this.plotWidth(),h-28);ctx.clip();
  if(this.caseRange||this.selectionRange){const [a,b]=this.selectionRange||this.caseRange;ctx.strokeStyle=color('--warning');ctx.setLineDash([4,4]);for(const x of [this.x(a),this.x(b)]){ctx.beginPath();ctx.moveTo(x,this.header);ctx.lineTo(x,h);ctx.stroke();}ctx.setLineDash([]);}ctx.restore();
  const first=Math.floor(this.scroll/this.row),last=Math.min(this.lanes.length,first+Math.ceil((h-this.header)/this.row)+1);
  ctx.save();ctx.beginPath();ctx.rect(0,this.header,w,h-this.header);ctx.clip();
  for(let i=first;i<last;i++){
   const lane=this.lanes[i],y=this.header+i*this.row-this.scroll;
   ctx.strokeStyle=grid;ctx.beginPath();ctx.moveTo(0,y+this.row);ctx.lineTo(w,y+this.row);ctx.stroke();ctx.fillStyle=fg;ctx.fillText(lane.name.length>16?lane.name.slice(0,15)+'…':lane.name,12,y+this.row/2);if(w>=450){ctx.textAlign='right';ctx.fillText(lane.worker?lane.utilization.toFixed(0)+'%':'—',this.labelWidth-10,y+this.row/2);ctx.textAlign='left';}
   ctx.save();ctx.beginPath();ctx.rect(this.labelWidth,y,this.plotWidth(),this.row);ctx.clip();
   for(const t of lane.tasks){if(t.start>this.view[1])break;if(t.end<this.view[0])continue;const x=this.x(t.start),width=Math.max(.8,this.x(t.end)-x);PtoSwimlaneTaskPattern.drawTaskBar(ctx,{task:t,x,y:y+4,width,height:this.row-8,baseColor:this.diagnosticMode?'#b8b8b8':this.palette.colorForTask(t,this.color),isSelected:t.id===this.selected,isRelated:this.related.has(t.id),fontFamily:css.getPropertyValue('--font-sans')});this.hits.push({x:Math.max(this.labelWidth,x),y:y+4,w:Math.min(w-12,x+width)-Math.max(this.labelWidth,x),h:this.row-8,task:t});}ctx.restore();
  }ctx.restore();
  if(this.diagnosticMode){ctx.save();ctx.beginPath();ctx.rect(this.labelWidth,this.header,this.plotWidth(),Math.max(0,usable-this.header));ctx.clip();for(const marker of this.markers){const active=marker.key===this.activeMarkerKey,muted=Boolean(this.activeMarkerKey&&!active);for(const rect of this.markerRects(marker)){this.markerHits.push({...rect,key:marker.key,preset:marker.preset,active,muted});ctx.fillStyle=muted?'rgba(255, 242, 0, .055)':active?'rgba(255, 242, 0, .30)':'rgba(255, 242, 0, .22)';ctx.fillRect(rect.x,rect.y,rect.w,rect.h);ctx.strokeStyle=muted?'rgba(240, 217, 0, .30)':'#f0d900';ctx.lineWidth=active?2.5:muted?1:2;ctx.setLineDash(marker.kind==='candidate'?[5,4]:[]);ctx.strokeRect(rect.x,rect.y,rect.w,rect.h);ctx.setLineDash([]);}}ctx.restore();}
  if(this.drag&&this.brush){const x=Math.max(this.labelWidth,Math.min(this.drag.x,this.drag.end)),y=Math.max(this.header,Math.min(this.drag.y,this.drag.endY));ctx.fillStyle=color('--state-selected');ctx.fillRect(x,y,Math.abs(this.drag.end-this.drag.x),Math.max(24,Math.abs(this.drag.endY-this.drag.y)));ctx.strokeStyle=color('--primary');ctx.strokeRect(x,y,Math.abs(this.drag.end-this.drag.x),Math.max(24,Math.abs(this.drag.endY-this.drag.y)));}
  if(this.lanes.length*this.row>h-this.header){const vh=h-this.header,total=this.lanes.length*this.row;ctx.fillStyle=color('--border-strong');ctx.fillRect(w-5,this.header+this.scroll/total*vh,3,Math.max(20,vh/total*vh));}
  if(!this.lanes.length){ctx.fillStyle=strong;ctx.fillText('没有匹配泳道，请调整过滤条件。',this.labelWidth+12,100);}
  this.callbacks.onView?.(this.snapshot());
 }
 destroy(){cancelAnimationFrame(this.pending);this.resize.disconnect();this.tooltip?.destroy();if(this.canvas.id!=='traceCanvas')this.tipTarget.remove();}
};
