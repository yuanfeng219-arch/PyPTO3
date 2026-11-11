/* Fit the shared compile viewer through its existing action; local to Lab Tour. */
(()=>{
 const drawer=window.CompileDrawer;if(!drawer?.open)return;
 const open=drawer.open;let cancelPending;
 drawer.open=function(...args){
  cancelPending?.();const result=open.apply(this,args);
  const host=document.querySelector('.compile-linked-host'),root=host?.shadowRoot;
  if(!root)return result;
  if(!root.querySelector('[data-lab-fit-style]')){
   const style=document.createElement('style');style.dataset.labFitStyle='';
   style.textContent=':host(.lab-fit-pending) .graph{opacity:0}';root.append(style);
  }
  host.classList.add('lab-fit-pending');let timer=0,done=false,readyAt=0;const started=performance.now();
  const finish=()=>{
   if(done)return;done=true;clearTimeout(timer);host.classList.remove('lab-fit-pending');
   root.removeEventListener('pointerdown',finish);root.removeEventListener('wheel',finish);
   if(cancelPending===finish)cancelPending=undefined;
  };
  cancelPending=finish;
  root.addEventListener('pointerdown',finish,{once:true});root.addEventListener('wheel',finish,{once:true,passive:true});
  const fit=()=>{
   if(done)return;if(host.hidden||!host.isConnected){finish();return;}
   const graph=root.querySelector('.graph');
   const button=[...root.querySelectorAll('.linked-panel > header button')].find(node=>node.textContent.trim()==='适配');
   const nodes=[...root.querySelectorAll('.react-flow__node')];
   const ready=graph?.clientWidth&&graph.clientHeight&&nodes.length&&nodes.every(node=>node.offsetWidth&&node.offsetHeight&&node.style.visibility!=='hidden')&&button;
   if(ready){
    if(!readyAt)readyAt=performance.now();
    if(performance.now()-readyAt>=160){button.click();requestAnimationFrame(()=>requestAnimationFrame(finish));return;}
   }else readyAt=0;
   if(performance.now()-started<3000){timer=setTimeout(fit,80);}else finish();
  };
  // Run after the viewer's initial 120ms selection zoom, using the final split dimensions.
  timer=setTimeout(fit,260);return result;
 };
})();
