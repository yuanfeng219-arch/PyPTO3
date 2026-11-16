
/* Shared material only: retain host interactions, geometry and semantic states. */
(()=>{
 const groups='.canvas-tools,.lane-toolbar,.pass-rail .rail-surface,.hp-controls,.zoom-controls';
 const panels='.menu-panel,.pane-info-body,.inspector,.overlay-panel';
 const icons='.demo-home,.hp-topbtn:not(.run-button),.pane-info>summary';
 function sync(){
  for(const el of document.querySelectorAll(groups))if(!el.closest('[data-liquid-glass]')&&!el.hasAttribute('data-liquid-glass'))el.setAttribute('data-liquid-glass','control');
  for(const el of document.querySelectorAll(panels))if(!el.closest('[data-liquid-glass]'))el.setAttribute('data-liquid-glass','panel');
  for(const el of document.querySelectorAll(icons))if(!el.closest('[data-liquid-glass]'))el.setAttribute('data-liquid-glass','icon');
 }
 sync();
 let pending=false;
 new MutationObserver(records=>{
  if(pending||!records.some(r=>[...r.addedNodes].some(n=>n.nodeType===1&&!n.matches('svg,.liquid-b-shine')&&(n.matches(groups+','+panels+','+icons)||n.querySelector(groups+','+panels+','+icons)))))return;
  pending=true;requestAnimationFrame(()=>{pending=false;sync();});
 }).observe(document.body,{childList:true,subtree:true});
})();
