/* Semantic compression, not a time scale. The app owns selectedNodeId. */
(()=>{
 const view=window.LINEAGE_VIEW,data=window.LINEAGE_DATA,NS='http://www.w3.org/2000/svg';
 const pane=document.querySelector('main.pto-ide-frame__pane'),full=document.querySelector('.workspace'),scroller=document.querySelector('.graph-host'),analysis=document.getElementById('analysis');
 const dock=document.createElement('nav');dock.className='lineage-dock';dock.hidden=true;dock.setAttribute('aria-label','Decision Lineage 导航');full.before(dock);
 const labels=['SPEC','EXPLORE','DESIGN','DEVELOP','OPTIMIZE','OPEN OPPORTUNITIES'];
 const stages=labels.map((label,i)=>({stageId:i+1,label,nodes:[]}));
 const title=id=>data.nodes.find(n=>n.id===id)?.title||id;
 const add=(stage,id,group,status,position,label)=>{const n={stageId:stage,nodeId:id,groupId:group,relationType:'stage_membership',status,compactPosition:position,title:label||title(id)};stages[stage-1].nodes.push(n);return n};
 for(const g of view.groups){g.items.forEach(([id,label],i)=>{
  let point;if(g.stage===1)point=[75,29];if(g.stage===2)point=[55+i*40,16];if(g.stage===3)point=[30+i*35,16];if(g.stage===4)point=[g.id==='kernel'?48:138,29];
  if(g.stage===5)point=({baseline:[20,29],optimization:[99,29],final:[187,29]})[g.id];if(id==='x-tail')point=[126,48];
  add(g.stage,id,g.id,'adopted',point,label);
 })}
 // Branch ownership is resolved through the existing parent, never a global append.
 for(const b of view.branches){const owner=stages.flatMap(s=>s.nodes).find(n=>n.nodeId===b.parent);if(owner)add(owner.stageId,b.id,owner.groupId,'not-adopted',[owner.compactPosition[0],53],b.title)}
 add(5,view.river.finding.id,'baseline','finding',[53,10],view.river.finding.label);
 add(5,view.river.failed.id,'optimization','failed',[158,53],view.river.failed.label);
 add(5,'d-stop','final','stopped',[224,29]);
 for(const [i,c]of view.river.postRunOpportunities.categories.entries())for(const [j,id]of c.candidateIds.entries()){const n=add(6,id,c.categoryId,'not_attempted',[166+j*32,i===0?18:46],view.river.opportunities.find(n=>n.id===id)?.label);n.relationType='not_explored'}
 const svgEl=(tag,attrs={},text)=>{const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text)n.textContent=text;return n};
 const header=document.createElement('header'),context=document.createElement('div'),expandButton=document.createElement('button');context.className='dock-context';expandButton.className='btn btn-ghost';expandButton.textContent='展开图谱 ↗';expandButton.onclick=()=>returnToOverview('explicit');header.append(context,expandButton);dock.append(header);
 const strip=document.createElement('div');strip.className='dock-strip';dock.append(strip);
 let mode='overview',savedX=0,version=0,picker=null,upward=0,lastIntent=0,touchY=null,topReachedAt=0,wasAtTop=true;
 const reduced=()=>matchMedia('(prefers-reduced-motion: reduce)').matches;
 const nextFrame=()=>new Promise(requestAnimationFrame);
 function setMode(next){mode=next;pane.dataset.lineageMode=next;upward=0}
 function closePicker(focus=false){if(!picker)return;const trigger=dock.querySelector('[aria-expanded=true]');picker.remove();picker=null;trigger?.setAttribute('aria-expanded','false');if(focus)trigger?.focus()}
 function openPicker(stage,button){const same=button.getAttribute('aria-expanded')==='true';closePicker();if(same)return;button.setAttribute('aria-expanded','true');picker=document.createElement('div');picker.className='dock-picker';picker.setAttribute('role','dialog');picker.setAttribute('aria-label',stage.label+' 节点');
  const groups=stage.stageId===6?view.river.postRunOpportunities.categories.map(c=>({label:c.categoryName+' · 未尝试',nodes:stage.nodes.filter(n=>n.groupId===c.categoryId)})):stage.stageId===5?[{label:'成功路径 / 观测',nodes:stage.nodes.filter(n=>n.status!=='failed')},{label:'Tail Handling · 失败尝试',nodes:stage.nodes.filter(n=>n.status==='failed')}]:[{label:'本次运行的'+(stage.stageId===3?'设计集合':'节点'),nodes:stage.nodes.filter(n=>n.status!=='not-adopted')}];
  for(const group of groups){const label=document.createElement('small');label.textContent=group.label;picker.append(label);for(const n of group.nodes){const b=document.createElement('button');b.className='btn btn-ghost';b.textContent=n.title;b.dataset.status=n.status;b.setAttribute('aria-pressed',String(window.LineageSelection.nodeId===n.nodeId));b.onclick=()=>window.selectLineageNode(n.nodeId);picker.append(b)}}
  dock.append(picker);picker.style.left=Math.max(0,Math.min(button.getBoundingClientRect().left-dock.getBoundingClientRect().left,dock.clientWidth-picker.offsetWidth-8))+'px';picker.querySelector('[aria-pressed=true],button')?.focus();
 }
 const glyphs=new Map();
 for(const stage of stages){const b=document.createElement('button');b.type='button';b.className='dock-stage';b.dataset.stage=stage.stageId;b.setAttribute('aria-expanded','false');b.setAttribute('aria-haspopup','dialog');const count=stage.nodes.filter(n=>n.status!=='not-adopted').length;b.title=`${stage.label} · ${count} 个节点`;b.setAttribute('aria-label',`0${stage.stageId} ${stage.label} · ${count} 个节点`);const label=document.createElement('span');label.textContent=`0${stage.stageId} ${stage.label}`;b.append(label);
  const w=stage.stageId===5?244:200,svg=svgEl('svg',{viewBox:`0 0 ${w} 64`,'aria-hidden':'true'});b.append(svg);
  const node=id=>stage.nodes.find(n=>n.nodeId===id),point=id=>node(id)?.compactPosition;
  const line=(a,z,cls='')=>{if(a&&z)svg.append(svgEl('path',{d:`M${a[0]} ${a[1]} L${z[0]} ${z[1]}`,class:cls}))};
  if(stage.stageId===2||stage.stageId===3){line([0,29],[w,29],'handoff');for(const n of stage.nodes){if(n.status==='not-adopted'){line([n.compactPosition[0],29],n.compactPosition,'untried');continue}line([n.compactPosition[0],29],n.compactPosition,'collection')}}
  else if(stage.stageId===5){line([0,29],[244,29],'success');line(point(view.river.finding.id),point('x-base'),'finding');for(const e of data.edges.filter(e=>e.type==='includes'))line(point(e.source),point(e.target),'success');const replacement=data.edges.find(e=>e.source===view.river.failed.id&&e.type==='replaced_by');if(replacement)line(point(replacement.target),point(replacement.source),'failed');svg.append(svgEl('text',{x:142,y:62,class:'failure-label'},'Runtime if'));svg.append(svgEl('text',{x:40,y:8},'Finding'));}
  else if(stage.stageId===6){for(const [i,c]of view.river.postRunOpportunities.categories.entries()){const y=i===0?18:46;line([0,29],[18,y],'untried');line([18,y],[w,y],'untried');svg.append(svgEl('text',{x:25,y:y-5},c.categoryName))}}
  else{line([0,29],[w,29],'handoff')}
  for(const n of stage.nodes){const [x,y]=n.compactPosition,g=svgEl('g',{'data-compact-node':n.nodeId,class:'compact-node '+n.status});g.append(svgEl(n.status==='failed'?'rect':'circle',n.status==='failed'?{x:x-3,y:y-3,width:6,height:6}:{cx:x,cy:y,r:3}));g.append(svgEl('title',{},n.title));svg.append(g);glyphs.set(n.nodeId,g)}
  b.onclick=()=>openPicker(stage,b);strip.append(b);
 }
 const cursor=svgEl('svg',{class:'dock-selection-marker',width:12,height:18,'aria-hidden':'true'});cursor.append(svgEl('path',{d:'M6 0 V14 M3 11 L6 14 L9 11'}));dock.append(cursor);
 function update(id){closePicker();const stage=stages.find(s=>s.nodes.some(n=>n.nodeId===id));context.replaceChildren();for(const [cls,text]of [['task','Softmax'],['stage',stage?`0${stage.stageId} ${stage.label}`:'Lineage'],['current',title(id)]]){const span=document.createElement('span');span.className=cls;span.textContent=text;context.append(span)}
  for(const [nodeId,g]of glyphs)g.classList.toggle('selected',nodeId===id);for(const b of strip.children)b.classList.toggle('is-current',Number(b.dataset.stage)===stage?.stageId);
  const sources=new Set(window.EXPLORE_DESIGN_RELATIONS.filter(r=>r.targetId===id).map(r=>r.sourceId));for(const [nodeId,g]of glyphs)g.classList.toggle('related',sources.has(nodeId));
  requestAnimationFrame(()=>{const b=strip.querySelector('.is-current');if(b){if(b.offsetLeft<strip.scrollLeft||b.offsetLeft+b.offsetWidth>strip.scrollLeft+strip.clientWidth)strip.scrollLeft=b.offsetLeft-strip.clientWidth/2+b.offsetWidth/2}positionCursor()});
 }
 function positionCursor(){const n=glyphs.get(window.LineageSelection?.nodeId);if(!n)return;const r=n.getBoundingClientRect(),d=dock.getBoundingClientRect();cursor.style.left=(r.left+r.width/2-d.left-6)+'px';cursor.style.visibility=r.left>=d.left&&r.right<=d.right?'visible':'hidden'}
 strip.addEventListener('scroll',positionCursor,{passive:true});new ResizeObserver(positionCursor).observe(dock);
 async function animateHeight(from,to){full.style.height=to+'px';if(reduced())return;const a=full.animate([{height:from+'px',opacity:from?1:0},{height:to+'px',opacity:to?1:0}],{duration:300,easing:'cubic-bezier(.2,.7,.2,1)'});await a.finished.catch(()=>{})}
 async function enterInspect(id){const ticket=++version,first=mode==='overview';if(first)savedX=scroller.scrollLeft;setMode('entering-inspect');closePicker();const height=full.getBoundingClientRect().height;full.inert=true;update(id);dock.hidden=false;pane.classList.add('lineage-transition');
  if(first&&!reduced())dock.animate([{opacity:0},{opacity:1}],{duration:300});await animateHeight(height,0);if(ticket!==version)return;await nextFrame();
  // One geometry-based, immediate scroll after the height transition. No smooth-scroll tail
  // can be mistaken for user intent; the layout transition provides continuous motion.
  const target=pane.scrollTop+analysis.getBoundingClientRect().top-pane.getBoundingClientRect().top-dock.offsetHeight;pane.scrollTo({top:Math.max(0,target),behavior:'instant'});await nextFrame();if(ticket!==version)return;pane.classList.remove('lineage-transition');setMode('inspect');document.getElementById('analysis-title').focus({preventScroll:true});
 }
 async function returnToOverview(reason='explicit'){if(mode==='overview'||mode==='returning-overview')return;const ticket=++version;setMode('returning-overview');closePicker();pane.classList.add('lineage-transition');window.AnalysisWorkspace.closeContent();full.inert=false;full.style.height='auto';const height=full.getBoundingClientRect().height;full.style.height='0px';pane.scrollTo({top:0,behavior:'instant'});scroller.scrollLeft=savedX;
  if(!reduced())dock.animate([{opacity:1},{opacity:0}],{duration:280,fill:'forwards'});await animateHeight(0,height);if(ticket!==version)return;dock.hidden=true;dock.getAnimations().forEach(a=>a.cancel());full.style.height='auto';scroller.scrollLeft=savedX;pane.scrollTo({top:0,behavior:'instant'});pane.classList.remove('lineage-transition');setMode('overview');document.getElementById('graph').focus({preventScroll:true});
 }
 function localScroll(target){for(let el=target instanceof Element?target:null;el&&el!==pane;el=el.parentElement){if(el.matches('.dock-picker,input,textarea,select,[contenteditable=true]'))return true;const css=getComputedStyle(el);if(/auto|scroll/.test(css.overflowY)&&el.scrollHeight>el.clientHeight+1)return true}return false}
 function atTop(){return analysis.getBoundingClientRect().top<=pane.getBoundingClientRect().top+dock.offsetHeight+8&&pane.scrollTop<=8}
 pane.addEventListener('scroll',()=>{const top=atTop();if(top&&!wasAtTop){topReachedAt=performance.now();upward=0}wasAtTop=top},{passive:true});
 function intent(delta,target){if(mode!=='inspect'||localScroll(target)){upward=0;return}const now=performance.now();if(delta>=0||!wasAtTop||!atTop()){upward=0;return}if(now-topReachedAt<140){upward=0;return}if(now-lastIntent>450)upward=0;lastIntent=now;upward+=Math.min(90,-delta);if(upward>=78)returnToOverview('scroll-up')}
 pane.addEventListener('wheel',e=>{if(Math.abs(e.deltaY)>Math.abs(e.deltaX))intent(e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?pane.clientHeight:1),e.target)},{passive:true});
 pane.addEventListener('touchstart',e=>{touchY=e.touches.length===1?e.touches[0].clientY:null;upward=0},{passive:true});pane.addEventListener('touchmove',e=>{if(touchY===null||e.touches.length!==1)return;const y=e.touches[0].clientY;intent(touchY-y,e.target);touchY=y},{passive:true});pane.addEventListener('touchend',()=>{touchY=null},{passive:true});
 pane.addEventListener('keydown',e=>{if(e.defaultPrevented||localScroll(e.target)||e.target.closest('button')&&[' ','Enter'].includes(e.key))return;if(e.key==='ArrowUp')intent(-40,e.target);else if(e.key==='PageUp'||e.key==='Home'||e.key===' '&&e.shiftKey)intent(-80,e.target);else if(['ArrowDown','PageDown','End',' '].includes(e.key))upward=0});
 document.addEventListener('pointerdown',e=>{if(!dock.contains(e.target))closePicker()});document.addEventListener('keydown',e=>{if(!picker)return;if(e.key==='Escape'){e.preventDefault();e.stopImmediatePropagation();closePicker(true)}else if(['ArrowDown','ArrowUp','Home','End'].includes(e.key)){const buttons=[...picker.querySelectorAll('button')],i=buttons.indexOf(document.activeElement);e.preventDefault();buttons[e.key==='Home'?0:e.key==='End'?buttons.length-1:(i+(e.key==='ArrowDown'?1:-1)+buttons.length)%buttons.length].focus()}},true);
 window.LineageDock={enterInspect,returnToOverview,get mode(){return mode},projection:stages};
})();
