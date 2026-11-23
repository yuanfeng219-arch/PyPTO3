/* Q LoRA for the archived TP=2 FLASH build. Logical tiles come from
 * qkv_proj_rope.py:32–73, 244–552; rank ownership from decode_csa.py:273–319. */
(function (global) {
  'use strict';

  const file='qkv_proj_rope.py', caller='decode_csa.py';
  const D=4096, L=1024, H=64, head=512, rope=64, nope=head-rope, Q=H*head;
  const QR_M=16, QR_N=128, QR_K=256, SPLITS=2, SPLIT_K=D/SPLITS;
  const Q_M=64, Q_TAIL_M=16, Q_N=512, Q_K=128, VECTOR_M=8, QUANT_N=256, HEAD_GROUP=4;
  const CAPACITY=256, TAIL_CASE=248, TP=2;
  const definitions={
    qr_proj_matmul:['qr','qr',277,299],
    qr_proj_seed:['qr','seed',270,275],
    qr_rms_norm_quant:['qr','quant',301,368],
    qproj_matmul:['q','qmat',370,416],
    qproj_dequant_rms_nope_rope:['q','post',418,552],
    q_rope_prepare_0:['rope','qrope',120,241],
    q_rope_prepare:['rope','kvrope',120,241],
  };
  const labels={
    qr_proj_matmul:'QR Split-K Matmul', qr_proj_seed:'QR Zero Seed', qr_rms_norm_quant:'QR RMSNorm + Quant',
    qproj_matmul:'Q INT8 Matmul', qproj_dequant_rms_nope_rope:'Dequant + RMSNorm + RoPE',
    q_rope_prepare_0:'Q RoPE Prepare', q_rope_prepare:'KV RoPE Prepare',
  };
  const sourceStage=global.PTO_CSA_RUNTIME_DATA.stages.find(stage=>stage.id==='input/q');
  const kernels=new Map(Object.entries(definitions).map(([name,[branch,type,first,last]])=>[
    name,{...sourceStage.kernels.find(k=>k.name===name),title:labels[name],branch,type,file,first,last},
  ]));
  const stage={...sourceStage,kernels:[...kernels.values()]};
  const measured=global.PTO_CSA_Q_PROJECTION_DATA;
  const stageEdges=stage.internalEdges.filter(e=>kernels.has(e.from)&&kernels.has(e.to));
  stageEdges.push({from:'qr_rms_norm_quant',to:'qproj_dequant_rms_nope_rope',source:'source',label:'QR scale pad · manual_dep'});

  const ticks=count=>Array.from({length:count},(_,i)=>i);
  const range=(start,size)=>`[${start}, ${start+size})`;
  const ceil=(value,step)=>Math.ceil(value/step);
  const isMatmul=info=>['qr','qmat'].includes(info.type);
  const groupSize=info=>info.type==='qr'?SPLITS:8;
  const blockCount=(info,T)=>({
    qr:L/QR_N*SPLITS, seed:1, quant:ceil(T,VECTOR_M),
    qmat:Q/Q_N, post:H/HEAD_GROUP, qrope:ceil(T,VECTOR_M), kvrope:ceil(T*TP,VECTOR_M),
  })[info.type];

  function mTiles(info,T,block) {
    const m=(start,size,extent,path)=>({start,size,valid:Math.min(size,extent-start),path});
    switch(info.type) {
      case 'qmat': {
        const full=Math.floor(T/Q_M)*Q_M;
        return [...ticks(full/Q_M).map(i=>m(i*Q_M,Q_M,T,'Full')),
          ...ticks(ceil(T-full,Q_TAIL_M)).map(i=>m(full+i*Q_TAIL_M,Q_TAIL_M,T,'Tail'))];
      }
      case 'qr':
      case 'seed':
        return ticks(ceil(T,QR_M)).map(i=>m(i*QR_M,QR_M,T,'M Tile'));
      case 'post':
        return ticks(ceil(T,VECTOR_M)).map(i=>m(i*VECTOR_M,VECTOR_M,T,'Token Tile'));
      case 'kvrope':
        return [m(block*VECTOR_M,VECTOR_M,T*TP,'全组 KV')];
      default:
        return [m(block*VECTOR_M,VECTOR_M,T,'Rank-local')];
    }
  }

  function blockShape(info,block,{tokens: T=CAPACITY,tileIndex=0}={}) {
    const tiles=mTiles(info,T,block), m=tiles[tileIndex]||tiles[0];
    const mText=`M ${range(m.start,m.valid)}${m.valid!==m.size?` · ${m.size} 行物理 Tile`:''}`;
    const mLabel=tiles.length>1?`${tiles.length} 个 M Tiles · 当前 ${mText}`:mText;
    if(info.type==='qr'){
      const group=Math.floor(block/SPLITS), split=block%SPLITS;
      return {...m,tiles,group,split,n:QR_N,col:group*QR_N,k:split*SPLIT_K,kLength:SPLIT_K,kTile:QR_K,
        text:`N Group ${group} · Split ${split} · ${mLabel} · N ${range(group*QR_N,QR_N)} · K ${range(split*SPLIT_K,SPLIT_K)}`};
    }
    if(info.type==='qmat')return {...m,tiles,group:block,n:Q_N,col:block*Q_N,k:0,kLength:L,kTile:Q_K,
      text:`N Group ${block} · ${mLabel} · N ${range(block*Q_N,Q_N)} · ${m.path}`};
    if(info.type==='post')return {...m,tiles,text:`Head Group ${block} · Heads ${range(block*HEAD_GROUP,HEAD_GROUP)} · ${mLabel}`};
    return {...m,tiles,text:`${info.type==='kvrope'?'全组 KV':'Rank-local'} · ${mLabel}`};
  }

  const objectId=(rank,info,block,op,view={})=>{
    const parts=['q-projection',info.branch,info.name,`rank${rank}`];
    if(block!=null)parts.push(`T${view.tokens??CAPACITY}`,`block${block}`);
    if(op!=null){
      if(block!=null)parts.push(`m${view.tileIndex??0}`);
      parts.push(op);
    }
    return parts.join('/');
  };

  function builder(){
    const nodes=[],edges=[],byId=new Map();
    function add(op,title,lane,lines,inputs=[],options={}){
      const id=options.id??(options.step==null?op:`${op}:${options.step}`);
      const rank=Math.max(0,...inputs.map(item=>byId.get(typeof item==='string'?item:item.id).rank+1));
      const node={op,id,title,lane,lines,rank,role:'operation',step:null,file,...options};
      nodes.push(node);byId.set(id,node);
      for(const item of inputs)edges.push(typeof item==='string'?{from:item,to:id,type:'data'}:{from:item.id,to:id,type:item.type??'data'});
      return id;
    }
    function input(op,title,lines,detail,options={}){
      return add(op,title,'Input',lines,[],{role:'activation',detail,...options});
    }
    function output(op,title,lines,inputs,detail,options={}){
      return add(op,title,'Write',lines,inputs,{role:'aggregate',detail,...options});
    }
    return {add,input,output,nodes,edges};
  }
  const carry=id=>({id,type:'state'});

  function matmul(info,expanded,view){
    const {add,input,output,nodes,edges}=builder();
    const shape=blockShape(info,view.block,view), qr=info.type==='qr', tail=!qr&&shape.path==='Tail';
    const [loadLines,mulLines,storeLines]=qr?[[284,294],[295,298],[299,299]]:
      tail?[[395,408],[409,412],[413,416]]:[[380,387],[388,391],[392,392]];
    const span=shape.kLength/shape.kTile, count=expanded?span:1;
    const nRange=range(shape.col,shape.n), mRange=range(shape.start,shape.valid);
    const splits=qr?ticks(SPLITS):[0];
    const contributions=[];
    if(qr)contributions.push(input('zero','QR Zero Seed',[270,275],`FP32 · M ${mRange} × N ${nRange}`,
      {layoutColumn:0, layoutRow:0, kernelLink:'qr_proj_seed'}));
    for(const split of splits){
      const splitBlock=qr?shape.group*SPLITS+split:view.block, kBase=qr?split*SPLIT_K:0;
      const prefix=qr?`Split ${split} · `:'';
      let accumulated;
      for(const i of ticks(count)){
        const kRange=expanded?range(kBase+i*shape.kTile,shape.kTile):range(kBase,shape.kLength);
        const opts={step:expanded?i:null,split,block:splitBlock,layoutColumn:i+1,detail:`K ${kRange}`};
        const tag=qr?`s${split}`:'q';
        const data=(op,title,detail,row)=>add(op,title,prefix+'Load',loadLines,[],
          {...opts,id:`${op}:${tag}:${i}`,role:'activation',detail,layoutRow:split*3+row});
        const x=data('load-x',qr?'Load X BF16':'Load QR INT8',`M ${mRange} × K ${kRange}`,1);
        const w=data('load-w',qr?'Load Wq_a BF16':'Load Wq_b INT8',`K ${kRange} × N ${nRange}`,2);
        accumulated=add('matmul',expanded?(i?'Matmul Acc':'Matmul'):'Matmul → Acc',prefix+'Cube',mulLines,
          i?[x,w,carry(accumulated)]:[x,w],{...opts,id:`matmul:${tag}:${i}`,
            detail:expanded?`K${i} · ${qr?'FP32':'INT32'} Acc`:`×${span} K Tiles · ${qr?'FP32':'INT32'} Acc`,
            nameRange:kRange,layoutRow:split*3+3});
      }
      const partial=output(qr?'partial':'i32',qr?`Split ${split} Partial`:'Q INT32 Tile',storeLines,[accumulated],
        `${qr?'FP32':'INT32'} · ${shape.size}×${shape.n}`,{id:qr?`partial:${split}`:'i32',split,block:splitBlock,layoutColumn:count+1,layoutRow:split*3+3,lane:prefix+'Write'});
      if(qr)contributions.push(add('atomic','Atomic Add',prefix+'Write',storeLines,[partial],{
        id:`atomic:${split}`,split,block:splitBlock,layoutColumn:count+2,layoutRow:split*3+3,detail:`共享 QR · N ${nRange}`,
      }));
    }
    if(qr)output('qr','Shared QR FP32',[270,299],contributions.map(carry),
      `M ${mRange} × N ${nRange} · 两个 Partial`,{layoutColumn:count+3,layoutRow:4});
    return {nodes,edges,pipelineLanes:true};
  }

  function operationModel(info,expanded,view={block:0,tileIndex:0,tokens:CAPACITY}){
    if(isMatmul(info))return matmul(info,expanded,view);
    const {add,input,output,nodes,edges}=builder();
    const shape=blockShape(info,view.block,view), m=`M ${range(shape.start,shape.valid)}`;
    switch(info.type){
      case 'seed': {
        const region=input('seed-region','QR FP32 Scratch',[263,270],`${ceil(view.tokens,QR_M)*QR_M} × ${L} · GM`);
        for(const n of ticks(expanded?L/QR_N:1)){
          const opts={step:expanded?n:null};
          const offset=expanded?`N ${range(n*QR_N,QR_N)}`:`×${L/QR_N} N Tiles`;
          const init=add('zero','Fill FP32 Zero','Vector',[271,275],[region],{...opts,detail:`${m} · ${offset}`});
          output('qr-seeded','Seeded QR Tile',[274,275],[init],offset,{...opts,kernelLink:'qr_proj_matmul'});
        }
        break;
      }
      case 'quant': {
        const fp32=input('qr','QR FP32',[315,319],`${m} × ${L}`);
        const gamma=input('gamma','Gamma cq',[320,325],`BF16 · ${L}`);
        let reduce;
        const nCount=L/QUANT_N;
        for(const i of ticks(expanded?nCount:1)){
          reduce=add('reduce','Σ x² / max |xγ|','Vector',[315,325],reduce?[fp32,gamma,carry(reduce)]:[fp32,gamma],{
            step:expanded?i:null,detail:expanded?`N ${range(i*QUANT_N,QUANT_N)}`:`Pass 1 · ×${nCount} N Tiles`,
          });
        }
        const inverse=add('rms','Inverse RMS','Vector',[326,330],[reduce],{detail:'FP32 · rsqrt(Σ x² / 1024 + ε)'});
        const quantScale=add('quant-scale','Quant Scale','Vector',[328,333],[reduce,inverse],{detail:'INT8_SCALE_MAX / amax'});
        output('qr-scale','QR Dequant Scale',[334,346],[quantScale],`${m} × 1 · FP32 · 1 / Quant Scale`,{kernelLink:'qproj_dequant_rms_nope_rope'});
        for(const i of ticks(expanded?nCount:1)){
          const opts={step:expanded?i:null};
          const n=expanded?`N ${range(i*QUANT_N,QUANT_N)}`:`Pass 2 · ×${nCount} N Tiles`;
          const quant=add('quant','Normalize + Quantize','Vector',[348,356],[fp32,gamma,inverse,quantScale],{...opts,detail:n});
          output('qr-int8','QR INT8 Tile',[357,368],[quant],`${m} · ${n}`,{...opts,kernelLink:'qproj_matmul'});
        }
        break;
      }
      case 'post': {
        const tail=shape.valid<VECTOR_M;
        const acc=input('q-i32','Q INT32 Accumulator',tail?[513,521]:[432,436],`${m} · Head Group ${view.block}`);
        const qr=input('qr-scale','QR Dequant Scale',tail?[464,469]:[427,431],`${m} × 1 · FP32`,{kernelLink:'qr_rms_norm_quant'});
        const weight=input('weight-scale','Wq_b Scale',tail?[522,528]:[435,436],'FP32 · per output column');
        const params=input('rope','Q-local RoPE Params',tail?[470,502]:[429,431],'Cos / signed Sin / Swap',{kernelLink:'q_rope_prepare_0'});
        for(const i of ticks(expanded?HEAD_GROUP:1)){
          const h0=view.block*HEAD_GROUP+i, opts={step:expanded?i:null};
          const detail=expanded?`Head ${h0} · N ${range(h0*head,head)}`:`×${HEAD_GROUP} Heads · ${m}`;
          const dequant=add('dequant','Dequantize','Vector',tail?[529,531]:[437,439],[acc,qr,weight],{...opts,detail});
          const rms=add('head-rms','Per-head RMSNorm','Vector',tail?[533,537]:[440,446],[dequant],{...opts,detail:'FP32 · reduce 512 columns'});
          const plain=add('nope','NoPE · Cast BF16','Vector',tail?[539,542]:[448,450],[dequant,rms],{...opts,detail:`Head cols ${range(0,nope)}`});
          const rotate=add('rotate','RoPE · Swap + Rotate','Vector',tail?[544,552]:[452,459],[dequant,rms,params],{...opts,detail:`Head cols ${range(nope,rope)}`});
          output('q','Q BF16',tail?[539,552]:[448,459],[plain,rotate],detail,opts);
        }
        break;
      }
      case 'qrope':
      case 'kvrope': {
        const owner=info.type==='qrope'?'Rank-local Q':'全组 KV';
        const src=input('cos-sin','Cos / Sin Tables',[120,130],`${owner} · BF16 · ${m} × ${rope}`);
        const indices=add('indices','Build Dup / Swap Index','Vector',[143,156],[],{detail:'j//2 · j^1 · 2(j%2)−1'});
        const gather=add('interleave','Gather Cos / Fold Sin Sign','Vector',shape.valid<VECTOR_M?[168,221]:[157,164],[src,indices],{detail:'FP32'});
        output('params',`${owner} RoPE Params`,shape.valid<VECTOR_M?[222,240]:[165,167],[gather,indices],'Cos / signed Sin FP32 · Swap INT32',
          info.type==='qrope'?{kernelLink:'qproj_dequant_rms_nope_rope'}:{});
        break;
      }
    }
    return {nodes,edges};
  }

  function overview(T){
    const {add,input,output,nodes,edges}=builder();
    const tensor=(id,title,detail,lines,layoutColumn,layoutRow,inputs=[],sourceFile=file)=>
      add(id,title,'Input',lines,inputs,{detail,file:sourceFile,role:'activation',layoutColumn,layoutRow});
    const kernel=(name,layoutColumn,layoutRow,inputs)=>{
      const k=kernels.get(name);
      return add(name,k.title,'Kernel',[k.first,k.last],inputs,{
        role:'operation',kernelLink:name,unit:k.coreType,detail:`${name} · ${blockCount(k,T)} Blocks`,layoutColumn,layoutRow,
      });
    };
    const prenorm=add('prenorm','HC Pre-Norm','Context',[249,275],[],{
      role:'operation',file:caller,detail:'本地输入混合与 RMSNorm',layoutColumn:0,layoutRow:2,
    });
    const x=tensor('x','x_normed_t',`Rank-local · BF16 [${T}, ${D}]`,[273,275],1,2,[prenorm],caller);
    const all=add('gather','CP Token AllGather','Context',[277,289],[x],{file:caller,detail:'跨 Rank · KV 输入',layoutColumn:2,layoutRow:4});
    tensor('x-full','x_normed_full',`组内 KV · BF16 [${T*TP}, ${D}]`,[277,289],3,4,[all],caller);
    const a=tensor('wqa','Wq_a',`BF16 [${D}, ${L}]`,[245,249],1,1);
    const seed=kernel('qr_proj_seed',1,3,[]);
    const mm=kernel('qr_proj_matmul',2,2,[x,a,seed]);
    const f32=tensor('qr-fp32','QR FP32',`FP32 [${ceil(T,QR_M)*QR_M}, ${L}] · Atomic Add`,[270,299],3,2,[mm]);
    const gamma=tensor('gamma','Gamma cq',`BF16 [${L}]`,[249,249],3,1);
    const norm=kernel('qr_rms_norm_quant',4,2,[f32,gamma]);
    const q8=tensor('qr8','QR INT8',`INT8 [${ceil(T,Q_TAIL_M)*Q_TAIL_M}, ${L}] · padded`,[301,305],5,2,[norm]);
    const scale=tensor('qr-scale','QR Dequant Scale',`FP32 [${T}, 1] · + padded store`,[332,346],5,3,[norm]);
    const b=tensor('wqb','Wq_b',`INT8 [${L}, ${Q}]`,[246,248],5,1);
    const qmat=kernel('qproj_matmul',6,2,[q8,b]);
    const q32=tensor('q32','Q INT32 Accumulator',`INT32 [${ceil(T,Q_TAIL_M)*Q_TAIL_M}, ${Q}]`,[370,392],7,2,[qmat]);
    const wscale=tensor('wscale','Wq_b Scale',`FP32 [${Q}]`,[247,248],7,1);
    const local=tensor('rope-local','Q-local Cos / Sin',`BF16 [${T}, ${rope}] × 2`,[304,307],4,4,[],caller);
    const qrope=kernel('q_rope_prepare_0',5,4,[local]);
    const qparams=tensor('qparams','Q-local RoPE Params',`FP32 cos/sin · INT32 swap · [${T}, ${rope}]`,[304,311],6,4,[qrope],caller);
    const post=kernel('qproj_dequant_rms_nope_rope',8,2,[q32,scale,wscale,qparams]);
    output('q','Q BF16',[253,253],[post],`BF16 [${T}, ${H}, ${head}]`,{layoutColumn:9,layoutRow:2});
    const full=tensor('rope-full','KV Group Cos / Sin',`BF16 [${T*TP}, ${rope}] × 2`,[299,302],4,5,[],caller);
    const kvrope=kernel('q_rope_prepare',5,5,[full]);
    tensor('kvparams','KV Group RoPE Params',`→ KV Projection · [${T*TP}, ${rope}]`,[299,318],6,5,[kvrope],caller);
    return {nodes,edges};
  }

  global.PtoCsaQProjectionModel=Object.freeze({
    title:'Q Projection',stage,measured,kernels,stageEdges,rows:QR_M,kTile:QR_K,kCount:SPLIT_K/QR_K,
    branches:{qr:'QR Projection',q:'Q Projection',rope:'RoPE Prepare'},defaultKernel:'qr_proj_matmul',
    cases:[{tokens:CAPACITY,label:'256 · 实测形状'},{tokens:TAIL_CASE,label:'248 · 静态 Tail 示例'}],
    ticks,range,isMatmul,groupSize,blockCount,blockShape,objectId,overview,operationModel,
  });
})(window);
