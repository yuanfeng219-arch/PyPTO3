(() => {
 const base=new URL('../lab-tour-demo-v4/',document.baseURI);const observed=new WeakSet();let queued=false;
 const cssURL=new URL('result-liquid-controls.css?v=panels12',base).href;
 function mark(el,type,preset){if(!el)return;el.dataset.resultLiquid=type;if(type==='notes'||type==='metrics'){if(el.hasAttribute('data-liquid-glass'))window.LiquidGlassControls?.destroy(el);el.removeAttribute('data-liquid-glass');el.removeAttribute('data-liquid-optics');return;}el.setAttribute('data-liquid-glass',preset)}
 function scan(root){
  if(root!==document&&!root.querySelector('[data-result-liquid-styles]')){
   for(const href of [new URL('vendor/liquid-glass-controls/liquid-glass.css',base).href,new URL('liquid-final.css',base).href,cssURL]){const l=document.createElement('link');l.rel='stylesheet';l.href=href;l.dataset.resultLiquidStyles='';root.append(l);}
  }
  root.querySelectorAll('[data-qk-inline-pass]').forEach(button=>{
   if(button.querySelector('foreignObject'))return;const rect=button.querySelector('rect');if(!rect)return;
   const fo=document.createElementNS('http://www.w3.org/2000/svg','foreignObject');for(const key of ['x','y','width','height'])fo.setAttribute(key,rect.getAttribute(key));fo.style.pointerEvents='none';fo.style.overflow='visible';
   const face=document.createElement('div');face.setAttribute('data-liquid-glass','control');face.style.cssText='position:relative;width:100%;height:100%;border-radius:999px;pointer-events:none';fo.append(face);rect.before(fo);rect.setAttribute('fill','transparent');rect.setAttribute('stroke','none');
  });
  if(root!==document&&!root.__passSync){root.__passSync=true;root.addEventListener('click',event=>{const button=event.target.closest?.('.pass-tick');if(!button)return;const match=(button.getAttribute('aria-label')||'').match(/Pass (\d+)/);queueMicrotask(()=>{const matched=!!match&&window.INLINE_PASS_TRANSFORM?.seek(Number(match[1]));document.querySelector('.qk-inline-pipeline')?.classList.toggle('compile-pass-no-counterpart',!matched);});});}
  root.querySelectorAll('.linked-pass-rail .phase-passes').forEach(el=>mark(el,'passnav','control'));
  root.querySelectorAll('.react-flow__controls,.zoom-controls').forEach(el=>mark(el,'tools','control'));
  root.querySelectorAll('header button').forEach(el=>{if(['适配','×'].includes(el.textContent.trim()))mark(el,'button','icon');if(el.textContent.trim()==='适配')el.dataset.resultHideFit='true'});
  mark(root.querySelector('#lead-finding'),'metrics','panel');
  root.querySelectorAll('summary,header,button,[role=button],h2,h3').forEach(title=>{if(!/体验设计亮点|DESIGN NOTES/.test(title.textContent||''))return;const panel=title.closest('details,aside,section,[role=dialog]')||title.parentElement;mark(panel,'notes','panel');title.dataset.resultNotesHeader='';});
  root.querySelectorAll('details').forEach(el=>{const text=el.querySelector(':scope > summary')?.textContent||'';if(/体验设计亮点|DESIGN NOTES/.test(text))mark(el,'notes','panel')});
  for(const el of root.querySelectorAll('*'))if(el.shadowRoot)scan(el.shadowRoot);
  if(!observed.has(root)){observed.add(root);new MutationObserver(records=>{if(records.some(r=>[...r.addedNodes].some(n=>n.nodeType===1&&!n.matches('.liquid-b-shine,svg,link'))))schedule()}).observe(root,{childList:true,subtree:true})}
 }
 function schedule(){if(queued)return;queued=true;requestAnimationFrame(()=>{queued=false;scan(document);window.LiquidGlassControls?.refresh()})}
 schedule();
 // The shared compile drawer attaches its shadow root lazily, after its host is inserted.
 const drawer=window.CompileDrawer;if(drawer?.open){const open=drawer.open;drawer.open=function(...args){const result=open.apply(this,args);schedule();setTimeout(schedule,100);return result}}
})();