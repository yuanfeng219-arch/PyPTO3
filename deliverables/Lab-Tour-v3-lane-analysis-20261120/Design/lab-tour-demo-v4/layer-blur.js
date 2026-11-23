(() => {
 const scene=document.querySelector('.layer-scene');if(!scene)return;
 const faces=[...scene.querySelectorAll('article.glass-layer')];
 function paint(){
  scene.querySelectorAll('.blur-sample').forEach(n=>n.remove());
  for(const back of scene.querySelectorAll(':scope > .glass-back')){
   const style=getComputedStyle(back),w=back.offsetWidth,h=back.offsetHeight;
   const m=new DOMMatrix(style.transform),inv=m.inverse();
   const tx=back.offsetLeft+w/2-m.a*w/2-m.c*h/2;
   const ty=back.offsetTop+h/2-m.b*w/2-m.d*h/2;
   const sample=document.createElement('div');sample.className='blur-sample';
   sample.style.cssText=`width:${scene.clientWidth}px;height:${scene.clientHeight}px;transform:matrix(${inv.a},${inv.b},${inv.c},${inv.d},${-inv.a*tx-inv.c*ty},${-inv.b*tx-inv.d*ty})`;
   const content=document.createElement('div');content.className='blur-sample-content';
   const depth=parseInt(style.zIndex);
   for(const face of faces){if(parseInt(getComputedStyle(face).zIndex)>=depth)continue;const clone=face.cloneNode(true);clone.setAttribute('aria-hidden','true');content.append(clone);}
   const axis=scene.querySelector(':scope > .mapping-axis');// Mapping axis is rendered once in front of the glass.
   sample.append(content);back.append(sample);
  }
 }
 requestAnimationFrame(paint);let frame;window.addEventListener('resize',()=>{cancelAnimationFrame(frame);frame=requestAnimationFrame(paint)});
})();