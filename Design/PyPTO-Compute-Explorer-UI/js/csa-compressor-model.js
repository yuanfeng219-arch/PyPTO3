/* Logical Compressor objects for the archived a2a3 build. Conditional bodies are
 * source paths; their instances do not assert that the enclosing branch ran. */
(function (global) {
  'use strict';

  const stage = global.PTO_CSA_RUNTIME_DATA.stages.find(item => item.id === 'cache/main');
  const measured = global.PTO_CSA_COMPRESSOR_DATA;
  const main = 'decode_compressor_ratio4.py', indexer = 'decode_indexer_compressor.py';
  const rows = 16, kTile = 512, kCount = 4096 / kTile;
  const definitions = {
    kv_score_proj: [main, 110, 136, 512, 'projection'],
    scatter_softmax_pool: [main, 145, 215, 512, 'pool'],
    compress_state_commit: [main, 221, 233, 512, 'commit'],
    rmsnorm_rope_cache_write: [main, 237, 287, 512, 'norm'],
    kv_score_proj_0: [indexer, 112, 139, 128, 'projection'],
    scatter_softmax_pool_0: [indexer, 148, 221, 128, 'pool'],
    compress_state_commit_0: [indexer, 227, 241, 128, 'commit'],
    rmsnorm_rope: [indexer, 245, 294, 128, 'norm'],
    kv_hadamard: [indexer, 297, 304, 128, 'hadamard'],
    kv_and_cache_write: [indexer, 306, 333, 128, 'write'],
  };

  const kernels = new Map(stage.kernels.map(entry => {
    const [file, first, last, head, type] = definitions[entry.name];
    const branch = file === main ? 'main' : 'indexer';
    return [entry.name, {...entry, file, first, last, head, type, branch}];
  }));
  const range = (start, size) => `[${start}, ${start + size})`;
  const ticks = length => Array.from({length}, (_, i) => i);
  const requestScoped = kernel => ['pool','commit'].includes(kernel.type);
  const objectId = (rank, kernel, block, op) =>
    ['compressor', kernel.branch, kernel.name, `rank${rank}`, block == null ? null : `block${block}`, op].filter(value => value != null).join('/');

  function blockShape(kernel, block) {
    if (requestScoped(kernel)) return {kind:'Request', row:block, group:block, text:`Request ${block}`};
    if (kernel.type !== 'projection') return {kind:'M Tile', row:block * rows, group:block, text:`M ${range(block * rows, rows)}`};
    const n = kernel.branch === 'main' ? 64 : 32;
    const nTiles = 2 * kernel.head / n;
    const row = Math.floor(block / nTiles) * rows, col = block % nTiles * n;
    return {kind:'M / N Tile', row, col, n, nTiles, group:row / rows, text:`M ${range(row, rows)} · N ${range(col,n)}`};
  }

  const stageEdges = new Map();
  for (const edge of stage.internalEdges) {
    const key = `${edge.from}>${edge.to}`;
    if (!stageEdges.has(key)) stageEdges.set(key, {from:edge.from, to:edge.to, sources:[]});
    stageEdges.get(key).sources.push(edge.source);
  }

  function operationModel(kernel, expanded) {
    const nodes = [], edges = [];
    const mainBranch = kernel.branch === 'main';
    const source = (mainRange, indexRange) => mainBranch ? mainRange : indexRange;
    const add = (op,title,lane,lines,inputs=[],options={}) => {
      const id = options.step == null ? op : `${op}:${options.step}`;
      const rank = Math.max(0, ...inputs.map(from => (nodes.find(n => n.id === from)?.rank ?? -1) + 1));
      nodes.push({op,id,title,lane,lines,rank,step:null,role:'operation', ...options});
      for (const from of inputs) edges.push({from,to:id,type:nodes.find(n=>n.id===from)?.op===op?'state':'data'});
      return id;
    };
    const input = (id,title,lines,detail) => add(id,title,'Input',lines,[],{role:'activation',detail});
    const output = (id,title,lines,inputs,detail,lane='Write') => add(id,title,lane,lines,inputs,{role:'aggregate',detail});
    const when = (condition,detail='') => ({condition,detail});

    switch (kernel.type) {
      case 'projection': {
        const shape=blockShape(kernel,0);
        const x=input('x','X',source([120,121],[122,123]),'BF16 · M × 4096');
        const wkv=input('wkv','Wkv',source([122,126],[124,129]),'BF16 · N × 4096');
        const gate=input('wgate','Wgate',source([122,127],[124,130]),'BF16 · N × 4096');
        const out=source([[135,135],[136,136]],[[138,138],[139,139]]);
        let kv,score;
        for (const i of ticks(expanded?kCount:1)) {
          const detail=expanded?`K${i} · ${range(i*kTile,kTile)}`:`K Tile × ${kCount} · source pipeline(…, stage=2)`;
          const opts={step:expanded?i:null, detail};
          const load=add('load','Load X / Weights','MTE2',source([120,127],[122,130]),[x,wkv,gate],opts);
          const extract=add('extract','Tile Extract','MTE1',source([128,133],[131,136]),[load],
            {...opts,detail:`${detail} · 2 × K256 / 投影 · PTO IR`});
          kv=add('kv',i?'KV Matmul Acc':'KV Matmul','M',source(i?[132,132]:[128,132],i?[135,135]:[131,135]),
            kv?[extract,kv]:[extract],opts);
          score=add('score',i?'Score Matmul Acc':'Score Matmul','M',source(i?[133,133]:[128,133],i?[136,136]:[131,136]),
            score?[extract,score]:[extract],opts);
        }
        output('kvOut','KV Tile',out[0],[kv],`FP32 · 16 × ${shape.n}`,'FIX');
        output('scoreOut','Score Tile',out[1],[score],`FP32 · 16 × ${shape.n}`,'FIX');
        break;
      }
      case 'pool': {
        const positions=input('positions','Position IDs',source([146,150],[149,153]),'每请求 · s_dim 个 Token');
        const projection=input('projection','KV / Score + APE',source([155,170],[158,173]),'本步投影');
        const ring=input('ring','History Ring',source([178,193],[181,196]),'上一 Decode Step 的状态');
        const boundary=add('boundary','定位压缩边界','Control',source([149,154],[152,157]),[positions],{detail:'逐 Token · (token_pos + 1) % 4'});
        output('zero','初始化 Pooled KV',source([151,151],[154,154]),[positions],'每个 Token 先写 0');
        const guard='仅压缩边界 · 每个 Head Tile';
        const seed=add('seed','初始化 m / l / o','Vector',source([155,170],[158,173]),[boundary,projection],when(guard,`${kernel.head/64} 个 Head Tiles`));
        let partial=seed;
        for(const i of ticks(expanded?7:1)) {
          const opts={...when(guard,expanded?`Window Slot ${i}`:'STATE_LEN − 1 = 7 个历史位置'),step:expanded?i:null};
          const lookup=add('window','历史 / 当前步取值','Gather',source([171,208],[174,214]),[ring,projection,boundary],opts);
          partial=add('carry','Online Softmax 更新','Vector',source([209,214],[215,220]),[lookup,partial],opts);
        }
        add('normalize','输出 o / l','Vector',source([215,215],[221,221]),[partial],when(guard,'覆盖本 Token 的 0 初始化'));
        output('pooled','Pooled KV',source([151,215],[154,221]),[nodes.at(-1).id,'zero'],'边界写池化结果，其余保留 0');
        break;
      }
      case 'commit': {
        const projected=input('projection','KV / Score',source([230,233],[236,241]),'本步投影');
        const slot=input('slots','State Slot Mapping',source([223,228],[229,234]),'每请求 · s_dim 个 Token');
        const check=add('guard','检查 State Slot','Control',source([223,227],[229,233]),[slot],{detail:'state_row ≥ 0'});
        const ape=add('ape','Score 加位置偏置','Vector',source([228,233],[234,241]),[check,projected],when('有效 State Slot','按 position % 4 读取 APE'));
        output('ring','更新 State Ring',source([230,233],[236,241]),[projected,ape],'有效 State Slot · KV 与 Score 分区');
        break;
      }
      case 'norm': {
        const pooled=input('pooled','Pooled KV',source([246,247],[255,256]),`16 × ${kernel.head}`);
        const weights=input('weights','Gamma / Cos / Sin',source([243,244],[252,253]),'Norm 与 RoPE 参数');
        let sum;
        for(const i of ticks(expanded?kernel.head/64:1)) {
          const opts={step:expanded?i:null,detail:expanded?`Head ${range(i*64,64)}`:`×${kernel.head/64} Head Tiles`};
          sum=add('sum','平方和累加','Vector',source([245,250],[254,259]),sum?[pooled,sum]:[pooled],opts);
        }
        const rms=add('inv','Inverse RMS','Vector',source([252,253],[261,262]),[sum],{detail:'rsqrt(mean(x²) + ε)'});
        const noRopeCount=kernel.head/64-1;
        const nope=ticks(expanded?noRopeCount:1).map(i=>add('nope','NoPE 归一化','Vector',source([254,258],[263,271]),[pooled,rms,weights],
          {step:expanded?i:null,detail:expanded?`Head ${range(i*64,64)}`:`×${noRopeCount} Head Tiles`}));
        const rope=add('rope','RoPE 旋转','Vector',source([260,277],[273,294]),[pooled,rms,weights],{detail:'末尾 64 列 · interleaved swap'});
        const normalized=output('normalized','归一化 KV',source([254,277],[263,294]),[...nope,rope],mainBranch?'FP32':'BF16');
        if(mainBranch) {
          const slot=input('slots','Cache Slot Mapping',[279,283],'逐有效 Token 行');
          output('cache','Main KV Cache',[279,287],[normalized,slot],'有效 Slot · BF16');
        }
        break;
      }
      case 'hadamard': {
        const kv=input('kv','Normalized KV',[298,300],'BF16 · 16 × 128');
        const weights=input('weights','Hadamard Weights',[301,302],'BF16 · 128 × 128');
        for(const i of ticks(expanded?2:1)) {
          const options={step:expanded?i:null,detail:expanded?`N ${range(i*64,64)}`:'×2 N Tiles'};
          const matmul=add('mix','Hadamard Matmul','M',[301,303],[kv,weights],options);
          add('write','KV 写回','FIX',[304,304],[matmul],{...options,role:'aggregate'});
        }
        break;
      }
      case 'write': {
        const kv=input('kv','Indexer KV',[310,314],'FP32 · 16 × 128');
        const slots=input('slots','Indexer Slot Mapping',[325,329],'逐有效 Token 行');
        const rounded=add('round','BF16 Round-trip','Vector',[312,314],[kv]);
        const scale=add('scale','Per-row Scale','Vector',[315,320],[rounded],{detail:'max(abs(x)) → Scale'});
        const quant=add('quant','INT8 量化','Vector',[321,324],[rounded,scale]);
        output('cache','Indexer INT8 Cache',[325,331],[quant,slots],'有效 Slot');
        output('scaleOut','Dequant Scale',[325,333],[scale,slots],'有效 Slot · 每行一个 Scale');
        break;
      }
    }
    return {nodes,edges};
  }

  global.PtoCsaCompressorModel=Object.freeze({stage,measured,kernels,rows,kTile,kCount,ticks,range,blockShape,objectId,requestScoped,operationModel,stageEdges:[...stageEdges.values()]});
})(window);
