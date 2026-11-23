/* Liquid Glass Controls v2.0.0 — classic script, no build step. */
(() => {
"use strict";
if (window.LiquidGlassControls) return;
/*
MIT License

Copyright (c) 2026 Riz Roze

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
/**
 * Liquid Glass — Real optical refraction for the web.
 *
 * Uses Canvas 2D to generate displacement maps fed into SVG feDisplacementMap
 * filters. Three separate passes (one per RGB channel) with slightly different
 * scale values create chromatic aberration at the edges.
 *
 * Chromium only. Safari/Firefox fall back to regular backdrop-filter: blur().
 *
 * @see https://github.com/rizzytoday/liquid-glass
 * @license MIT
 */
function resolveConfig(el, opts) {
  const rect = el.getBoundingClientRect();
  const ab = opts.aberration ?? [0, 10, 20];
  return {
    width: opts.width ?? Math.round(rect.width),
    height: opts.height ?? Math.round(rect.height),
    radius: opts.borderRadius ?? 50,
    scale: opts.scale ?? -180,
    border: opts.border ?? 0.07,
    lightness: opts.lightness ?? 50,
    alpha: opts.alpha ?? 0.93,
    blur: opts.blur ?? 11,
    r: ab[0],
    g: ab[1],
    b: ab[2],
    frost: opts.frost ?? 0,
    saturation: opts.saturation ?? 1,
    displace: opts.displaceBlur ?? 0
  };
}
const isChromium = typeof navigator !== "undefined" && /Chrome\//.test(navigator.userAgent);
const _mapCache = /* @__PURE__ */ new Map();
function buildDisplacementMap(c) {
  const key = `${c.width}:${c.height}:${c.radius}:${c.scale}:${c.border}:${c.blur}:${c.lightness}:${c.alpha}`;
  const cached = _mapCache.get(key);
  if (cached) return cached;
  const maxDisplace = Math.max(Math.abs(c.scale) * 0.5, 20);
  const padX = Math.ceil(maxDisplace);
  const padY = Math.ceil(maxDisplace);
  const totalW = c.width + padX * 2;
  const totalH = c.height + padY * 2;
  const canvas = document.createElement("canvas");
  canvas.width = totalW;
  canvas.height = totalH;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgb(128, 128, 128)";
  ctx.fillRect(0, 0, totalW, totalH);
  const ox = padX;
  const oy = padY;
  ctx.save();
  ctx.beginPath();
  ctx.roundRect(ox, oy, c.width, c.height, c.radius);
  ctx.clip();
  ctx.fillStyle = "#000000";
  ctx.fillRect(ox, oy, c.width, c.height);
  const redGrad = ctx.createLinearGradient(ox + c.width, oy, ox, oy);
  redGrad.addColorStop(0, "#000000");
  redGrad.addColorStop(1, "#ff0000");
  ctx.fillStyle = redGrad;
  ctx.fillRect(ox, oy, c.width, c.height);
  ctx.globalCompositeOperation = "difference";
  const blueGrad = ctx.createLinearGradient(ox, oy, ox, oy + c.height);
  blueGrad.addColorStop(0, "#000000");
  blueGrad.addColorStop(1, "#0000ff");
  ctx.fillStyle = blueGrad;
  ctx.fillRect(ox, oy, c.width, c.height);
  ctx.globalCompositeOperation = "source-over";
  const borderPx = Math.min(c.width, c.height) * (c.border * 0.5);
  ctx.filter = `blur(${c.blur}px)`;
  ctx.fillStyle = `hsla(0, 0%, ${c.lightness}%, ${c.alpha})`;
  ctx.beginPath();
  ctx.roundRect(
    ox + borderPx,
    oy + borderPx,
    c.width - borderPx * 2,
    c.height - borderPx * 2,
    c.radius
  );
  ctx.fill();
  ctx.restore();
  const uri = canvas.toDataURL();
  _mapCache.set(key, uri);
  return uri;
}
let _instanceCount = 0;
function createFilterSVG(id) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  svg.style.cssText = "position:absolute;width:0;height:0;pointer-events:none;";
  svg.innerHTML = `
    <defs>
      <filter id="${id}" color-interpolation-filters="sRGB" x="-38%" y="-188%" width="176%" height="476%">
        <feImage result="map" preserveAspectRatio="none" />
        <feDisplacementMap in="SourceGraphic" in2="map" xChannelSelector="R" yChannelSelector="B" result="dispRed" data-channel="red" />
        <feColorMatrix in="dispRed" type="matrix" values="1 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0" result="red" />
        <feDisplacementMap in="SourceGraphic" in2="map" xChannelSelector="R" yChannelSelector="B" result="dispGreen" data-channel="green" />
        <feColorMatrix in="dispGreen" type="matrix" values="0 0 0 0 0  0 1 0 0 0  0 0 0 0 0  0 0 0 1 0" result="green" />
        <feDisplacementMap in="SourceGraphic" in2="map" xChannelSelector="R" yChannelSelector="B" result="dispBlue" data-channel="blue" />
        <feColorMatrix in="dispBlue" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 1 0 0  0 0 0 1 0" result="blue" />
        <feBlend in="red" in2="green" mode="screen" result="rg" />
        <feBlend in="rg" in2="blue" mode="screen" result="output" />
        <feGaussianBlur in="output" stdDeviation="0" />
      </filter>
    </defs>
  `;
  const filter = svg.querySelector("filter");
  const feImage = svg.querySelector("feImage");
  const red = svg.querySelector('[data-channel="red"]');
  const green = svg.querySelector('[data-channel="green"]');
  const blue = svg.querySelector('[data-channel="blue"]');
  const blurEl = svg.querySelector("feGaussianBlur");
  return { svg, feImage, red, green, blue, blur: blurEl, filter };
}
function applyConfig(c, refs) {
  const uri = buildDisplacementMap(c);
  const maxD = Math.max(Math.abs(c.scale) * 0.5, 20);
  const pctX = Math.ceil(maxD / c.width * 100);
  const pctY = Math.ceil(maxD / c.height * 100);
  refs.filter.setAttribute("x", `-${pctX}%`);
  refs.filter.setAttribute("y", `-${pctY}%`);
  refs.filter.setAttribute("width", `${100 + pctX * 2}%`);
  refs.filter.setAttribute("height", `${100 + pctY * 2}%`);
  refs.feImage.setAttributeNS("http://www.w3.org/1999/xlink", "href", uri);
  refs.feImage.setAttribute("href", uri);
  refs.red.setAttribute("scale", String(c.scale + c.r));
  refs.green.setAttribute("scale", String(c.scale + c.g));
  refs.blue.setAttribute("scale", String(c.scale + c.b));
  refs.blur.setAttribute("stdDeviation", String(c.displace));
}
function createLegacyGlass(element, options = {}) {
  const fallback = options.fallbackFilter ?? "blur(12px)";
  if (!isChromium) {
    const prev = element.style.backdropFilter;
    const prevWebkit = element.style.getPropertyValue("-webkit-backdrop-filter");
    element.style.backdropFilter = fallback;
    element.style.setProperty("-webkit-backdrop-filter", fallback);
    const dummySvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    return {
      isActive: false,
      filterElement: dummySvg,
      update() {
      },
      destroy() {
        element.style.backdropFilter = prev;
        element.style.setProperty("-webkit-backdrop-filter", prevWebkit);
      }
    };
  }
  const id = options.filterId ?? `liquid-glass-${++_instanceCount}`;
  const refs = createFilterSVG(id);
  document.body.appendChild(refs.svg);
  let currentOpts = { ...options };
  let config = resolveConfig(element, currentOpts);
  applyConfig(config, refs);
  const applyStyles = (c) => {
    element.style.backdropFilter = `url(#${id}) saturate(${c.saturation})`;
    element.style.setProperty(
      "-webkit-backdrop-filter",
      `url(#${id}) saturate(${c.saturation})`
    );
    if (c.frost > 0) {
      element.style.background = `hsl(0 0% 0% / ${c.frost})`;
    }
  };
  applyStyles(config);
  let resizeRaf = 0;
  const ro = new ResizeObserver(() => {
    cancelAnimationFrame(resizeRaf);
    resizeRaf = requestAnimationFrame(() => {
      if (currentOpts.width == null || currentOpts.height == null) {
        config = resolveConfig(element, currentOpts);
        if (!config.width || !config.height) return;
        applyConfig(config, refs);
        applyStyles(config);
      }
    });
  });
  ro.observe(element);
  return {
    isActive: true,
    filterElement: refs.svg,
    update(newOpts) {
      currentOpts = { ...currentOpts, ...newOpts };
      config = resolveConfig(element, currentOpts);
      applyConfig(config, refs);
      applyStyles(config);
    },
    destroy() {
      ro.disconnect();
      cancelAnimationFrame(resizeRaf);
      refs.svg.remove();
      element.style.backdropFilter = "";
      element.style.setProperty("-webkit-backdrop-filter", "");
      element.style.background = "";
    }
  };
}


// Screen-space slab approximation. Separate transmitted face and compressed
// sidewall rays; both sample the live DOM backdrop. No WebGL or fixed tint.
function renderB(el,v,id,r){
 const w=Math.round(el.clientWidth),h=Math.round(el.clientHeight),ss=3;
 const W=w*ss,H=h*ss,scale=Math.max(128,h*1.5),c=document.createElement('canvas');c.width=W;c.height=H;
 const ctx=c.getContext('2d'),face=ctx.createImageData(W,H),rim=ctx.createImageData(W,H),mask=ctx.createImageData(W,H),light=ctx.createImageData(W,H);
 const bezel=Math.min(h*.44,v.bezel*h/74),rimWidth=v.rimWidth*Math.sqrt(h/74),depth=v.depth*h/74;
 const faceGain=.23+1.15*Math.min(1,Math.pow(74/h,4));
 // Focus changes both the radial highlight width and its angular concentration.
 // Keep the approved default (28) as the width reference.
 const focus=Math.max(0,Math.min(100,v.power));
 const spread=Math.exp((28-focus)/45);
 const angularPower=.45+8*Math.pow(focus/100,1.5);
 // Do not encode a small face displacement against the much larger rim range:
 // the previous shared 285px range quantized face offsets to >1 CSS pixel.
 const faceScale=Math.max(8,Math.ceil(depth*(v.strength/60)*faceGain*2.5));
 for(let py=0;py<H;py++)for(let px=0;px<W;px++){
  const x=(px+.5)/ss,y=(py+.5)/ss,i=(py*W+px)*4,d=-sdf(x,y,w,h,r),e=.15;
  let gx=sdf(x+e,y,w,h,r)-sdf(x-e,y,w,h,r),gy=sdf(x,y+e,w,h,r)-sdf(x,y-e,w,h,r),n=Math.hypot(gx,gy)||1;gx/=n;gy/=n;
  let dx=0,dy=0,nx=0,ny=0,nz=1;
  if(d>0&&d<bezel){
   const t=d/bezel,q=1-t,slope=Math.pow(q,3)/Math.pow(Math.max(.0001,1-q**4),.75);
   nz=1/Math.sqrt(1+slope*slope);nx=gx*slope*nz;ny=gy*slope*nz;
   const eta=1/v.ior,f=eta*nz-Math.sqrt(1-eta*eta*(1-nz*nz)),tz=-eta+f*nz;
   const height=depth*(.28+.72*Math.pow(1-q**4,.25));
   const travel=height/Math.max(.1,-tz)*(v.strength/60)*faceGain;
   // feDisplacementMap is an inverse lookup (output -> source), not a forward
   // motion vector. Sample outward to contract the transmitted image at bevels.
   const entry=Math.min(1,d/2),fade=entry*entry*(3-2*entry);
   dx=-f*nx*travel*fade;dy=-f*ny*travel*fade;
  }
  function encode(img,a,b,range=scale){img.data.set([Math.round(127.5+a/range*255),Math.round(127.5+b/range*255),128,255],i)}
  encode(face,dx,dy,faceScale);
  const t=Math.max(0,Math.min(1,d/Math.max(1,rimWidth)));
  // The rim transports an interior strip to the narrow thickness band.
  const reach=Math.max(0,h*.45-d*.8);
  encode(rim,-gx*reach,-gy*reach);
  const aa=Math.min(1,Math.max(0,d*ss+.5));
  // Reserve the outer ~0.8 CSS px for the air/glass highlight. The coloured
  // thickness starts underneath it, rather than painting over the silhouette.
  const colourD=d-.85,colourGate=Math.max(0,Math.min(1,colourD*2));
  const band=(Math.exp(-Math.pow((colourD-rimWidth*.43)/(rimWidth*.36),2))+.16*Math.exp(-Math.pow((colourD-rimWidth*1.15)/(rimWidth*.30),2)))*aa*colourGate;
  mask.data.set([255,255,255,Math.round(255*Math.min(1,band)*v.rim/100)],i);
  const facing=-gx*.56-gy*.83;
  // A nearby upper-left area light also varies with position. Normal-only
  // lighting makes every pixel along a straight top edge equally bright.
  // Broad softbox rather than a very close point light: keep a readable top
  // highlight across long toolbars, with a gentle 1 -> 0.33 horizontal falloff.
  const lightFalloff=Math.exp(-1.1*Math.pow(x/w,.9)-.35*y/h);
  const edge=Math.exp(-Math.pow((d-.48)/(.44*Math.sqrt(spread)),2));
  const bevel=Math.exp(-Math.pow((d-rimWidth*.88)/(rimWidth*.65*spread),2));
  const soft=Math.exp(-Math.pow((d-rimWidth*1.4)/(rimWidth*1.6*spread),2));
  const directional=Math.pow(Math.max(0,facing),angularPower);
  const returnLight=Math.pow(Math.max(0,gx*.62+gy*.78),angularPower);
  const innerReflection=Math.exp(-Math.pow((d-rimWidth*1.7)/(rimWidth*1.2*spread),2));
  // No constant rim term: the highlight must disappear away from the light,
  // otherwise the entire silhouette becomes a white UI border in dark mode.
  const outerWhite=edge*1.2*Math.pow(Math.max(0,facing),Math.max(2,angularPower))*v.shine*lightFalloff;
  const primaryReflection=(bevel*.24*directional+soft*.07*directional)*lightFalloff;
  // Lower-right environment reflection has its own footprint; it must not be
  // attenuated a second time by the upper-left key light's distance field.
  const returnFalloff=Math.exp(-Math.pow((x/w-.91)/.30,2));
  const returnCore=Math.exp(-Math.pow((d-rimWidth*.90)/(rimWidth*.32*spread),2));
  const returnHalo=Math.exp(-Math.pow((d-rimWidth*1.25)/(rimWidth*.50*spread),2));
  const secondaryReflection=(returnCore*.45+returnHalo*.12)*returnLight*returnFalloff;
  const innerWhite=(primaryReflection+secondaryReflection)*v.shine*(h<100?.65:1);
  const white=outerWhite+innerWhite;
  const innerShade=Math.exp(-Math.pow((d-h*.085)/(h*.045),2))*.025*Math.max(0,facing);
  const dark=Math.exp(-Math.pow((d-rimWidth*.8)/.75,2))*(.012+.04*Math.max(0,-facing))+innerShade;
  const a=Math.max(white,dark)*aa;
  const tone=white>=dark?255:35;
  light.data.set([tone,tone,tone,Math.round(Math.min(1,a)*255)],i);
 }
 function url(img){ctx.putImageData(img,0,0);return c.toDataURL()}
 const fURL=url(face),rURL=url(rim),mURL=url(mask),lURL=url(light);
 el.querySelector('.liquid-b-shine').style.backgroundImage=`url(${lURL})`;
 el.querySelector('.liquid-b-shine').style.backgroundSize='100% 100%';
 let svg=document.getElementById(id+'-svg');if(!svg){svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.id=id+'-svg';svg.style.cssText='position:absolute;width:0;height:0;pointer-events:none';document.body.append(svg)}
 const img=(u,result)=>`<feImage href="${u}" x="0" y="0" width="${w}" height="${h}" preserveAspectRatio="none" result="${result}"/>`;
 svg.innerHTML=`<filter id="${id}" color-interpolation-filters="sRGB" x="-30%" y="-100%" width="160%" height="300%">${img(fURL,'rawFaceMap')}<feGaussianBlur in="rawFaceMap" stdDeviation=".35" result="map"/><feDisplacementMap in="SourceGraphic" in2="map" scale="${v.strength===0?0:faceScale}" xChannelSelector="R" yChannelSelector="G"/><feGaussianBlur stdDeviation="${0.22}" result="face"/><feMorphology in="SourceGraphic" operator="erode" radius="2" result="rimChroma"/><feGaussianBlur in="rimChroma" stdDeviation="1" result="rimBackdrop"/>${img(rURL,'rimMap')}<feDisplacementMap in="rimBackdrop" in2="rimMap" scale="${scale}" xChannelSelector="R" yChannelSelector="G" result="sideSample"/><feColorMatrix in="sideSample" type="saturate" values="1.3" result="rimColor"/>${img(mURL,'mask')}<feComposite in="rimColor" in2="mask" operator="in" result="side"/><feComposite in="side" in2="face" operator="over"/></filter>`;
 const filter=/Chrome/.test(navigator.userAgent)?`url(#${id}) blur(${v.blur}px)`:`blur(${v.blur}px)`;el.style.backdropFilter=filter;el.style.webkitBackdropFilter=filter;
}


function sdf(x,y,w,h,r){let qx=Math.abs(x-w/2)-(w/2-r),qy=Math.abs(y-h/2)-(h/2-r);return Math.hypot(Math.max(qx,0),Math.max(qy,0))+Math.min(Math.max(qx,qy),0)-r}
let slabId=0;
function createLiquidGlass(element,options){
 if(element.dataset.liquidGlass==='panel')return {destroy(){}};
 const id='liquid-slab-'+(++slabId),shine=document.createElement('span');
 shine.className='liquid-b-shine';shine.setAttribute('aria-hidden','true');element.append(shine);
 let raf=0,lastKey='';
 const render=()=>{cancelAnimationFrame(raf);raf=requestAnimationFrame(()=>{
 const key=element.clientWidth+':'+element.clientHeight+':'+JSON.stringify(activeSettings());
 if(key===lastKey)return;lastKey=key;renderB(element,activeSettings(),id,Math.min(options.borderRadius,element.clientWidth/2,element.clientHeight/2));
 });};
 const resize=new ResizeObserver(render);resize.observe(element);render();
 return {destroy(){resize.disconnect();cancelAnimationFrame(raf);shine.remove();document.getElementById(id+'-svg')?.remove();}};
}

const presets=Object.freeze({
 control:Object.freeze({scale:-110,aberration:[0,4,8],alpha:1,blur:8,displaceBlur:4,saturation:1,frost:0,fallbackFilter:'blur(12px)'}),
 panel:Object.freeze({scale:-12,aberration:[0,0,0],alpha:1,blur:8,displaceBlur:18,saturation:1,frost:0,fallbackFilter:'blur(18px)'}),
 icon:Object.freeze({scale:-36,aberration:[0,1,2],alpha:1,blur:6,displaceBlur:4,saturation:1,frost:0,fallbackFilter:'blur(12px)'})
});
const selector='[data-liquid-glass]:not(.ub-warning)';
const instances=new Map();
const defaults={strength:100,bezel:36,shine:2.2,power:28,depth:37,blur:1,rim:95,rimWidth:2.75,ior:1.5};
const limits={strength:[0,100],bezel:[2,36],shine:[0,6],power:[0,100],depth:[0,48],blur:[0,20],rim:[0,100],rimWidth:[1,6],ior:[1,1.8]};
let tuning={...defaults};
let darkShine=.8;
try{const saved=JSON.parse(localStorage.getItem('liquid-glass-controls-dark-shine'));if(Number.isFinite(saved))darkShine=Math.max(0,Math.min(6,saved));}catch{}
function activeSettings(){return {...tuning,...(document.documentElement.dataset.theme!=='light'?{shine:darkShine}:{})};}
function normalize(values){for(const key of Object.keys(defaults))if(Number.isFinite(values?.[key]))tuning[key]=Math.max(limits[key][0],Math.min(limits[key][1],values[key]));}
try{normalize(JSON.parse(localStorage.getItem('liquid-glass-controls-v2')));}catch{}
const properties=['backdrop-filter','-webkit-backdrop-filter','background'];
let observer=null,frame=0,started=false;
function restore(element,record){
 record.glass.destroy();
 for(const [name,value,priority] of record.styles){
  if(value)element.style.setProperty(name,value,priority);else element.style.removeProperty(name);
 }
}
function remove(element){const record=instances.get(element);if(record){instances.delete(element);restore(element,record);}}
function refresh(){
 if(!started)return;
 const candidates=new Set(document.querySelectorAll(selector));
 for(const element of instances.keys())if(!candidates.has(element))remove(element);
 for(const element of candidates){
  const preset=element.getAttribute('data-liquid-glass');
  const rect=element.getBoundingClientRect(),css=getComputedStyle(element);
  const visible=rect.width>0&&rect.height>0&&css.visibility!=='hidden';
  const radius=Math.min(parseFloat(css.borderTopLeftRadius)||0,rect.width/2,rect.height/2);
  const record=instances.get(element);
  if(record&&(!visible||record.preset!==preset||record.radius!==radius))remove(element);
  if(!visible||!presets[preset]||instances.has(element))continue;
  const styles=properties.map(name=>[name,element.style.getPropertyValue(name),element.style.getPropertyPriority(name)]);
  const glass=createLiquidGlass(element,{...presets[preset],borderRadius:radius});
  instances.set(element,{glass,styles,preset,radius});
 }
}
function schedule(){if(!started||frame)return;frame=requestAnimationFrame(()=>{frame=0;refresh();});}
function start(){
 if(started)return;started=true;
 observer=new MutationObserver(records=>{if(records.some(r=>r.attributeName==='data-theme')){for(const element of [...instances.keys()])remove(element);window.dispatchEvent(new Event('glass-theme-change'));}schedule();});
 observer.observe(document.documentElement,{subtree:true,childList:true,attributes:true,attributeFilter:['data-theme','data-liquid-glass','hidden','class','style']});
 window.addEventListener('resize',schedule);
 refresh();
}
function disconnect(){
 started=false;if(observer)observer.disconnect();observer=null;
 cancelAnimationFrame(frame);frame=0;window.removeEventListener('resize',schedule);
 for(const element of instances.keys())remove(element);
}
window.LiquidGlassControls=Object.freeze({
 version:'2.0.0',
 init(element,preset='control'){
  if(!presets[preset])throw new Error('Unknown liquid glass preset: '+preset);
  element.setAttribute('data-liquid-glass',preset);start();refresh();return element;
 },
 destroy(element){element.removeAttribute('data-liquid-glass');remove(element);},
 getSettings(){return activeSettings();},
 configure(values){const next={...values};if(document.documentElement.dataset.theme!=='light'){if(Number.isFinite(next.shine))darkShine=Math.max(0,Math.min(6,next.shine));delete next.shine;}normalize(next);try{localStorage.setItem('liquid-glass-controls-dark-shine',JSON.stringify(darkShine));localStorage.setItem("liquid-glass-controls-v2",JSON.stringify(tuning));}catch{}for(const element of [...instances.keys()])remove(element);schedule();},
 refresh,start,disconnect
});
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',start,{once:true});else start();
window.addEventListener('pagehide',disconnect);
window.addEventListener('pageshow',start);
})();
