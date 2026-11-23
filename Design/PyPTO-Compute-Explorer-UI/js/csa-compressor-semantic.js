/* Compressor data adapter. No DOM, layout, selection or camera ownership here.
 * Both scales use the CSA Semantic Graph schema and the shared graph renderer. */
(function (global) {
  'use strict';
  const base=global.PTO_CSA_SEMANTIC_GRAPH, logical=global.PtoCsaCompressorModel;
  const cache=new Map();
  const ports=(inputs=['default'],outputs=['default'])=>({inputs:Object.fromEntries(inputs.map(id=>[id,{}])),outputs:Object.fromEntries(outputs.map(id=>[id,{}]))});
  const mapping=(node,port='default',relation='view')=>({node,port,relation});
  const source=(file,start,end)=>({file,ranges:[{start,end}],symbols:[]});
  const estWidth=(label,expandable)=>Math.max(194,Math.ceil([...String(label)].reduce((w,ch)=>w+(/[\u2e80-\u9fff\uff00-\uffef]/.test(ch)?15.5:8.8),0))+(expandable?96:66));
  function node(id,type,name,description,src,payload){
    const w=['operation','composite'].includes(type)?estWidth(name,type==='composite'):undefined;
    return {id,type,name,description,source:src,ports:ports(),...(w?{w}:{}),[type]:payload,
      provenance:[{kind:'resolved',origin:'Compressor 源码与现有计算模型',fields:[],note:'静态计算关系；循环实例不表示实测核内时间。'}]};
  }
  function tensor(id,name,src,shape,dtype,role='intermediate',storage){
    return node(id,'data',name,name,src,{role,granularity:'tensor',entity:id,
      ...(shape?{shape:{allocated:shape}}:{}),...(dtype?{dtype}:{}),...(storage?{storage:{space:storage}}:{})});
  }
  const edge=(s,t,label,type='data',sourcePort='default',targetPort='default')=>({id:`${s}>${t}:${label}:${type}`,s,t,label,type,sourcePort,targetPort});
  function kernelGraph(info,instances){
    const graph=logical.operationModel(info,instances), nodes=[], edges=[];
    for(const item of graph.nodes){
      const src=source(info.file,...item.lines),description=[item.detail,item.condition].filter(Boolean).join(' · ')||item.title;
      let n;
      if(item.role==='operation'){
        n=node(item.id,'operation',item.title+(item.step==null?'':` · ${info.type==='projection'?'K':'Iteration'} ${item.step}`),description,src,{semantics:description,category:'compute',api:[]});
      }else{
        // Only source-backed attributes are emitted. Generic input descriptions
        // in the logical model are not parsed into tensor dimensions.
        n=tensor(item.id,item.title,src,null,null,item.role==='activation'?'input':'output');
        n.description=description;
      }
      if(info.type==='projection'){
        const width=info.branch==='main'?64:32;
        const attrs={x:[[16,512],'BF16'],wkv:[[width,512],'BF16'],wgate:[[width,512],'BF16'],kvOut:[[16,width],'FP32'],scoreOut:[[16,width],'FP32']}[item.op];
        if(attrs)Object.assign(n.data,{shape:{allocated:attrs[0]},dtype:attrs[1]});
        // Expose the actual FP32 accumulator between matmul_acc iterations.
        if(['kv','score'].includes(item.op)){
          const acc=tensor(item.id+'-acc',item.op==='kv'?'kv_acc':'score_acc',src,[16,width],'FP32','intermediate','L0C');
          acc.compressor={kernel:info.name,branch:info.branch,operation:item.op,...(item.step!=null?{iteration:item.step}:{}),coreType:info.coreType};
          nodes.push(acc);edges.push(edge(item.id,acc.id,'FP32 accumulator'));
        }
      }
      n.compressor={kernel:info.name,branch:info.branch,operation:item.op,...(item.step!=null?{iteration:item.step}:{}),coreType:info.coreType,...(info.type==='projection'&&item.step!=null?{kRange:logical.range(item.step*logical.kTile,logical.kTile)}:{})};
      nodes.push(n);
    }
    for(const link of graph.edges){
      const from=graph.nodes.find(n=>n.id===link.from);
      const accumulator=info.type==='projection'&&['kv','score'].includes(from.op);
      const carrying=accumulator&&graph.nodes.find(n=>n.id===link.to)?.op===from.op;
      const type=carrying?'state':from.lane==='Control'?'control':link.type;
      edges.push(edge(accumulator?from.id+'-acc':link.from,link.to,carrying?'accumulate':from.title,type));
    }
    const inputs=graph.nodes.filter(n=>n.role==='activation'),outputs=graph.nodes.filter(n=>n.role==='aggregate');
    const ins=Object.fromEntries(inputs.map(n=>[n.id,mapping(n.id)])),outs=Object.fromEntries(outputs.map(n=>[n.id,mapping(n.id)]));
    ins.default=mapping(inputs[0]?.id||nodes[0].id);
    ins.ready=mapping(graph.nodes.find(n=>n.role==='operation')?.id||nodes[0].id,'default','control');
    outs.default=mapping(outputs.at(-1)?.id||nodes.at(-1).id);
    outs.tid=mapping(outputs.at(-1)?.id||nodes.at(-1).id,'default','control');
    const n=node(info.name,'composite',info.name,`${info.coreType} · ${info.blockNum} logical Blocks · ${info.file}:${info.first}–${info.last}`,source(info.file,info.first,info.last),{
      semantics:'源码计算与条件路径；展开查看 Operation / Tensor。',category:'module',api:[],
      subgraph:{schemaVersion:2,id:info.name+'-'+(instances?'instances':'structure'),nodes,edges,groups:[]},portMappings:{inputs:ins,outputs:outs}});
    n.compressor={kernel:info.name,branch:info.branch,coreType:info.coreType};
    n.ports=ports(Object.keys(ins),Object.keys(outs));
    return n;
  }
  function branchGraph(original,branch,instances){
    const n=structuredClone(original),main=branch==='main';
    const infos=[...logical.kernels.values()].filter(k=>k.branch===branch),file=infos[0].file;
    const [proj,pool,commit,norm,had,write]=infos.map(k=>k.name);
    const nodes=infos.map(k=>kernelGraph(k,instances)),edges=[];
    const add=(...args)=>edges.push(edge(...args));
    const src=(a,b)=>source(file,a,b),head=main?512:128;
    nodes.push(tensor('x','x',src(main?76:75,main?76:75),['T_DYN',4096],'BF16','input','GM'),
      tensor('projected-kv',main?'cmp4_kv_proj_pad':'kv_proj_pad',src(100,102),['BS_PAD',head*2],'FP32','intermediate','GM'),
      tensor('projected-score',main?'cmp4_score_proj_pad':'score_proj_pad',src(100,102),['BS_PAD',head*2],'FP32','intermediate','GM'),
      tensor('pooled','pooled_kv',src(main?142:145,main?142:145),['BS_PAD',head],'FP32','intermediate','GM'));
    const history=node('history','state','History Ring','读取上一 Decode Step 的状态；本次 Pool 完成后才提交。',src(main?78:77,main?78:77),{initial:'上一 Decode Step 的持久状态',update:'本次 Pool 后由 State Commit 写入',dtype:'FP32',granularity:'request'});
    nodes.push(history);
    add('x',proj,'BF16 input','data','default','x');
    for(const [id,port] of [['projected-kv','kvOut'],['projected-score','scoreOut']]){
      add(proj,id,id,'data',port);add(id,pool,id,'data','default','projection');add(id,commit,id,'data','default','projection');
    }
    add('history',pool,'previous Decode Step','state','default','ring');
    add(pool,'pooled','pooled KV','data','pooled');add('pooled',norm,'pooled KV','data','default','pooled');
    if(!main){
      nodes.push(tensor('normalized','normed_kv',src(243,243),['BS_PAD',128],'BF16','intermediate','GM'),tensor('final','kv_final',src(296,296),['BS_PAD',128],'FP32','intermediate','GM'));
      add(norm,'normalized','normalized KV','data','normalized');add('normalized',had,'Hadamard input','data','default','kv');
      if(instances){
        add(had,'final','N [0, 64)','data','write:0');
        add(had,'final','N [64, 128)','data','write:1');
      }else{
        add(had,'final','FP32 Hadamard output');
      }
      add('final',write,'quantize / cache write','data','default','kv');
    }
    // Runtime explicit Task dependencies remain distinct from tensor dataflow.
    for(const dep of logical.stageEdges.filter(e=>e.sources.includes('explicit')&&infos.some(k=>k.name===e.from)&&infos.some(k=>k.name===e.to)))add(dep.from,dep.to,'Task completion','control','tid','ready');
    const maps=n.composite.portMappings;
    maps.inputs.full=mapping('x');maps.inputs.state=mapping('history');maps.inputs.slot=mapping(main?norm:write,'slots');
    maps.outputs.cache=mapping(main?norm:write,main?'cache':'cache');
    maps.outputs.tid=mapping(main?norm:write,'tid','control');maps.outputs.state=mapping(commit,'ring');
    if(!main)maps.outputs.scale=mapping(write,'scaleOut');
    n.composite.subgraph={schemaVersion:2,id:branch+'-compressor',nodes,edges,groups:[]};
    return n;
  }
  function getModel({focused=false,instances=false}={}){
    const key=focused+':'+instances;if(cache.has(key))return cache.get(key);
    const model=structuredClone(base),stage=model.nodes.find(n=>n.id==='cache');
    stage.composite.subgraph.nodes=stage.composite.subgraph.nodes.map(n=>n.id==='main'?branchGraph(n,'main',instances):n.id==='idxcomp'?branchGraph(n,'indexer',instances):n);
    if(focused){
      const keep=new Set(['main','idxcomp','main-state','idx-state','compressed','indexcache','indexscale']);
      stage.name='Compressor';stage.description='Main + Indexer Compressor · 源码计算、持久状态与 Cache';
      stage.ports=ports([],[]);stage.composite.portMappings={inputs:{},outputs:{}};
      stage.composite.subgraph.nodes=stage.composite.subgraph.nodes.filter(n=>keep.has(n.id));
      stage.composite.subgraph.edges=stage.composite.subgraph.edges.filter(e=>keep.has(e.s)&&keep.has(e.t));
      model.nodes=[stage];model.edges=[];model.id='csa-compressor';
    }
    function evidence(nodes){for(const n of nodes){if(n.provenance?.[0]?.origin==='Compressor 源码与现有计算模型'){const payload=n[n.type];n.provenance[0].fields=['data','state'].includes(n.type)?Object.keys(payload).map(k=>n.type+'.'+k):[];}if(n.composite)evidence(n.composite.subgraph.nodes);}}
    evidence(model.nodes);global.PTO_SEMANTIC_COMPUTATION_GRAPH.validate(model);cache.set(key,model);return model;
  }
  global.PtoCsaCompressorSemantic=Object.freeze({getModel});
})(window);
