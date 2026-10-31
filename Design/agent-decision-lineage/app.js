const data=window.LINEAGE_DATA,view=window.LINEAGE_VIEW,$=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';
let selected=null,fullTrace=false;
const semanticEdges=[...window.EXPLORE_DESIGN_RELATIONS.map(r=>({source:r.sourceId,target:r.targetId,type:r.relationType,evidenceRefs:r.evidenceRefs})),...data.edges.filter(e=>!['workflow','stage_membership'].includes(e.type)),...view.branches.filter(b=>['x-rows','x-reuse'].includes(b.id)).map(b=>({source:b.parent,target:b.id,type:'not_explored',evidenceRefs:[{sourceId:'PERFORMANCE_REPORT.md',location:'§7 Candidate Coverage Ledger（优化方向关系，非执行因果）'}]}))];
const relationNames={open_opportunity:'未执行的开放机会',includes:'组成部分',observed_in:'观测线索',not_explored:'未探索方向',not_explored:'未探索方向',informs:'依据 → 设计',constrains:'约束',rules_out:'设计未触发',implemented_by:'produces',validated_by:'validated by',measured_by:'测量结果',motivates:'improved by',replaced_by:'失败尝试被替代',supports:'支持停止'};
function el(tag,attrs={},text){const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text)n.textContent=text;return n}
function button(text,fn){const b=document.createElement('button');b.className='btn btn-ghost';b.textContent=text;b.onclick=fn;return b}
function sourceLink(ref){const a=document.createElement('a');a.href=data.sources.find(s=>s.id===ref.sourceId).path;a.target='_blank';a.rel='noopener';a.textContent=ref.sourceId+' · '+ref.location;return a}
function neighborhood(){const set=new Set(selected?[selected]:[]);if(!selected)return set;for(const direction of ['up','down']){const queue=[selected],visited=new Set(queue);while(queue.length){const id=queue.shift();for(const e of semanticEdges){const next=direction==='up'&&e.target===id?e.source:direction==='down'&&e.source===id?e.target:null;if(next&&!visited.has(next)){visited.add(next);set.add(next);if(fullTrace)queue.push(next)}}}}return set}
function reset(){selected=null;fullTrace=false;render()}
let analysisActive=false,analysisOrigin=null,lineageScroll=0,mainScroll=0,analysisContextId=null,analysisScroll=0;
function choose(id){if(!data.nodes.some(n=>n.id===id))return;
 NodeTooltip.hide();selected=id;fullTrace=false;render();AnalysisWorkspace.open(id);window.LineageDock.inspect(id);
}
function showAnalysis(id){analysisContextId=id;window.DecisionInspector?.clear();$('detail').replaceChildren();$('analysis-title').textContent=data.nodes.find(n=>n.id===id)?.title||id;
 if(window.SchedulingAnalysis?.open(id))return;
 if(window.UBBufferAnalysis?.open(id))return;
 if(window.SingleTileAnalysis?.open(id))return;
 if(window.ReviewInspector?.open(id))return;
 if(window.ExperimentInspector?.open(id))return;
 if(window.DecisionInspector?.open(id))return;
 const node=data.nodes.find(n=>n.id===id);const p=document.createElement('p');p.textContent=node.reason;$('detail').append(p);
}
window.AnalysisWorkspace={open(id){analysisActive=true;
 $('analysis').hidden=false;if(analysisContextId!==id||!$('detail').children.length){showAnalysis(id);analysisScroll=0}$('analysis').scrollTop=analysisScroll;
 $('analysis-title').focus({preventScroll:true});
},back(){analysisScroll=$('analysis').scrollTop;analysisActive=false;$('analysis').hidden=true;
 window.LineageDock.expand();$('graph').focus({preventScroll:true});
}};
$('analysis-back').onclick=()=>AnalysisWorkspace.back();
document.addEventListener('keydown',event=>{if(event.key!=='Escape'||event.defaultPrevented)return;NodeTooltip.hide();if(analysisActive)AnalysisWorkspace.back()});
function activate(node,fn){node.setAttribute('tabindex','0');node.setAttribute('role','button');node.addEventListener('click',e=>{e.stopPropagation();fn()});node.addEventListener('keydown',e=>{if(['Enter',' '].includes(e.key)){e.preventDefault();e.stopPropagation();fn()}})}
function render(){window.ExploreDesignHover?.clear();const g=$('graph');g.replaceChildren();const related=neighborhood(),positions=new Map(),groups=new Map(view.groups.map(g=>[g.id,g]));
let box;
for(const group of view.groups)group.items.forEach(([id],i)=>positions.set(id,FlowGeometry.itemBox(group,i)));for(const b of view.branches)positions.set(b.id,{x:b.x,y:b.y,w:b.w,h:b.h});for(const p of window.ExperimentRiver.positions(view))positions.set(p.id,p);
positions.set('h-mask',positions.get('x-mask'));
const defs=el('defs'),marker=el('marker',{id:'arrow',viewBox:'0 0 8 8',refX:7,refY:4,markerWidth:5,markerHeight:5,orient:'auto'});marker.append(el('path',{d:'M0 0 L8 4 L0 8',fill:'var(--foreground-muted)'}));defs.append(marker);g.append(defs);
const stageSegments=[[24,176,'01 SPEC'],[218,396,'02 EXPLORE'],[438,643,'03 DESIGN'],[687,1021,'04 DEVELOP'],[1057,1665,'05 OPTIMIZE'],[1770,2085,'06 OPEN OPPORTUNITIES']];for(const [x,end,label]of stageSegments){const ruler=el('g',{class:'stage-ruler','data-region':/OPTIMIZE|OPPORTUNITIES/.test(label)?'optimize':'design','aria-label':label+(label.startsWith('06')?' · 运行后总结':' 已完成')});ruler.append(el('line',{x1:x,y1:63,x2:end,y2:63}),el('text',{x,y:49},label+(label.startsWith('06')?'':' ✓')));g.append(ruler)}
g.append(el('text',{x:1770,y:88,class:'zone-label','data-region':'optimize'},'运行后总结 · 分支不代表执行轨迹'));
g.append(el('text',{x:24,y:108,class:'zone-label','data-region':'design'},'DESIGN DECISION LINEAGE'),el('text',{x:875,y:88,class:'zone-label','data-region':'optimize'},'EXPERIMENT LINEAGE'));
for(const [aId,bId,type,file,loc]of view.links){const a=groups.get(aId),b=groups.get(bId),from=FlowGeometry.anchor(view,a,'out'),to=FlowGeometry.anchor(view,b,'in'),t=b.stage===5?view.river.flow.mainHalfWidth:view.river.flow.otherHalfWidth;const path=el('path',{d:FlowGeometry.band(from,to,t),'data-region':b.stage===5?'optimize':'design','data-related':[...a.items,...b.items].map(([id])=>id).join(' '),class:'lineage-ribbon '+(b.stage===5?'tone-3':'tone-1')});path.append(el('title',{},`${type} · ${file} · ${loc}（非数量编码）`));const on=selected&&[...a.items,...b.items].some(([id])=>related.has(id));if(on)path.classList.add('ribbon-active');if(selected&&!on)path.style.opacity='.08';g.append(path);}
window.ExperimentRiver.render({svg:g,view,selected,related,choose,layer:'connections'});
for(const group of view.groups){const container=el('g',{'data-group':group.id,'data-region':group.stage===5?'optimize':'design'});container.append(el('rect',{x:group.x,y:group.y,width:group.w,height:group.h,rx:17,class:'group-plate'}),el('text',{x:group.x+12,y:group.y+24,class:'group-label'},group.label),el('text',{x:group.x+12,y:group.y+43,class:'group-sub'},group.sub));if(group.id==='optimization'){const port=FlowGeometry.internalPorts(group).tailHandling,tail=el('g',{class:'node tail-handling','data-object':'tailHandling','aria-label':'Tail Handling · 尾寄存器实现选择'});tail.append(el('text',{x:group.x+12,y:port.y,class:'group-sub'},'Tail Handling · 实现选择'),el('circle',{cx:port.x,cy:port.y,r:4,class:'tail-port'}));activate(tail,()=>choose('x-tail'));container.append(tail);activate(container.querySelector('.group-label'),()=>choose('x-mask'));}for(const [index,[id,title,subtitle]]of group.items.entries()){const p=FlowGeometry.itemBox(group,index),item=el('g',{'data-node-id':id,class:'node '+(selected===id?'selected':''),'aria-label':title+' '+subtitle});item.append(el('rect',{x:p.x-4,y:p.y-16,width:p.w+8,height:40,rx:6,class:'node-hit'}),el('circle',{cx:p.x+3,cy:p.y-3,r:3,class:'candidate-dot'}),el('text',{x:p.x+13,y:p.y,class:'item-title'},title),el('text',{x:p.x+13,y:p.y+16,class:'item-sub'},subtitle));if(selected&&!related.has(id))item.style.opacity='.36';activate(item,()=>choose(id));container.append(item)}g.append(container);for(const text of container.querySelectorAll('text'))FlowGeometry.fitText(text,group.w-(text.classList.contains('item-title')||text.classList.contains('item-sub')?34:24))}
for(const b of view.branches){if(selected!==b.id)continue;const p=positions.get(b.parent),startX=p.x+p.w/2,startY=p.y+22;const path=el('path',{d:`M${startX} ${startY} C${startX} ${startY+60},${b.x+b.w/2} ${b.y-60},${b.x+b.w/2} ${b.y}`,class:'relationship '+(b.id==='x-if'?'':'open'),'marker-end':'url(#arrow)'});path.append(el('title',{},b.relation+' · '+(b.id==='d-online'?'DESIGN.md 归约轴容量结论':b.id==='d-stop'?'PERFORMANCE_REPORT.md §6、§11':'PERFORMANCE_REPORT.md §7')));g.append(path);const item=el('g',{'data-node-id':b.id,class:'node '+(b.id==='x-if'?'branch-failed':'branch-open')+(selected===b.id?' selected':''),'aria-label':b.title+' '+b.subtitle});item.append(el('rect',{x:b.x,y:b.y,width:b.w,height:b.h,rx:12}),el('text',{x:b.x+12,y:b.y+24,class:'item-title'},b.title),el('text',{x:b.x+12,y:b.y+44,class:'item-sub'},b.subtitle),el('text',{x:b.x+b.w/2,y:b.y-12,'text-anchor':'middle',class:'edge-label'},b.relation));activate(item,()=>choose(b.id));if(selected&&!related.has(b.id))item.style.opacity='.3';g.append(item)}
window.ExperimentRiver.render({svg:g,view,selected,related,choose,layer:'nodes'});
// Always fit the complete run in one coordinate space; selection never filters stages.
const boundElements=[...g.children].filter(n=>n.tagName!=='defs');
box=FlowGeometry.bounds(boundElements,view.river.flow.padding);
// Keep one SVG unit per CSS pixel at every viewport width; overflow scrolls.
g.style.width=box[2]+'px';g.style.height=box[3]+'px';g.style.aspectRatio=String(box[2]/box[3]);g.setAttribute('viewBox',box.join(' '));
const clip=el('clipPath',{id:'view-clip'});clip.append(el('rect',{x:box[0],y:box[1],width:box[2],height:box[3]}));defs.append(clip);const scene=el('g',{'clip-path':'url(#view-clip)'});for(const child of [...g.children])if(child!==defs)scene.append(child);g.append(scene);window.ExploreDesignHover?.attach(g,view);
NodeTooltip.attach(g);requestAnimationFrame(updateStageNavigation);
}
$('graph').addEventListener('click',e=>{if(!e.target.closest('.node,.stage-ruler,.opportunity-category'))reset()});$('context-toggle').onclick=()=>{$('context').hidden=!$('context').hidden;$('context-toggle').setAttribute('aria-expanded',String(!$('context').hidden));render()};for(const s of data.sources){const a=document.createElement('a');a.href=s.path;a.target='_blank';a.textContent=s.file;$('artifacts').append(a)}window.PtoIdeFrame?.initAll();reset();

window.addEventListener('decision-topic-select',event=>choose(event.detail));

window.addEventListener('experiment-select',event=>choose(event.detail));

window.addEventListener('review-select',event=>choose(event.detail));


window.addEventListener('resize',()=>render());

// Read rendered ruler geometry: chapter 06 remains a post-run summary.
const graphScroller=$('graph').parentElement;
let stageNavigationFrame=0;
function stageVisibility(){
 const viewport=graphScroller.getBoundingClientRect();
 const stages=[...$('graph').querySelectorAll('.stage-ruler')].map(node=>({node,rect:node.querySelector('text').getBoundingClientRect()}));
 return {viewport,stages,right:stages.filter(s=>s.rect.right>viewport.left+graphScroller.clientWidth-2),left:stages.filter(s=>s.rect.left<viewport.left-2)};
}
function updateStageNavigation(){
 stageNavigationFrame=0;
 const next=$('stage-scroll-next'),back=$('stage-scroll-back');
 const {viewport,stages,right,left}=stageVisibility();
 const overflow=graphScroller.scrollWidth>graphScroller.clientWidth+2;
 next.hidden=!overflow||!right.length||graphScroller.scrollLeft+graphScroller.clientWidth>=graphScroller.scrollWidth-2;
 back.hidden=!overflow||!left.length||graphScroller.scrollLeft<=2;
 next.textContent='还有 '+right.length+' 个阶段 →';
 next.setAttribute('aria-label','向右查看 '+(right[0]?.node.getAttribute('aria-label')||'后续阶段'));
 if(stages.length){
 // Occupy the existing gap above the ruler titles, never the node body.
 const frame=graphScroller.parentElement.getBoundingClientRect();
 $('stage-scroll-next').parentElement.style.top=Math.max(0,stages[0].rect.top-frame.top-24)+'px';
 }
}
function scheduleStageNavigation(){if(!stageNavigationFrame)stageNavigationFrame=requestAnimationFrame(updateStageNavigation)}
function navigateStage(direction){
 const {viewport,right,left}=stageVisibility();
 const target=direction>0?right[0]:left[left.length-1];if(!target)return;
 const offset=target.rect.left-viewport.left;
 const desired=graphScroller.scrollLeft+offset-24;
 graphScroller.scrollTo({left:Math.max(0,Math.min(graphScroller.scrollWidth-graphScroller.clientWidth,desired)),behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});
}
$('stage-scroll-next').onclick=()=>navigateStage(1);
$('stage-scroll-back').onclick=()=>navigateStage(-1);
graphScroller.addEventListener('scroll',scheduleStageNavigation,{passive:true});
new ResizeObserver(scheduleStageNavigation).observe(graphScroller);
scheduleStageNavigation();

window.selectLineageNode=choose;
