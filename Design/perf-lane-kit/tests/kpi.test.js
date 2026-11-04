import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {calculateKpis} from '../src/kpi.js';

test('KPI 合并重叠单块区间并统计完整窗口的低并行与空闲',()=>{
  const rank={windowUs:100,cores:[{id:0,ct:'AIC'},{id:1,ct:'AIV'},{id:2,ct:'AIV'}],launches:[
    {ct:'AIC',computeUs:30,setupUs:5,blocks:{items:[{startUs:10,endUs:50},{startUs:20,endUs:40},{startUs:20,endUs:40}]}},
    {ct:'AIV',computeUs:20,setupUs:5,blocks:{items:[]}}
  ],criticalPath:[{onCore:true,blockNum:1,startUs:10,endUs:40},{onCore:true,blockNum:1,startUs:30,endUs:50},{onCore:true,blockNum:2,startUs:50,endUs:90}]};
  const k=calculateKpis(rank);
  assert.equal(k.pathUs,80);assert.equal(k.singleUs,40);assert.equal(k.singleShare,.5);
  assert.equal(k.occupancy,50/300);assert.equal(k.byType.AIC,.3);assert.equal(k.byType.AIV,.1);
  assert.equal(k.lowUs,80);assert.equal(k.idleUs,60);assert.equal(k.setupShare,.2);
});

test('缺少测量或有效分母时不生成虚假 KPI',()=>{
  const rank={windowUs:10,cores:[{id:0,ct:'AIC'}],launches:[{ct:'AIC',blocks:{items:[]}}],criticalPath:[]};
  const k=calculateKpis(rank);assert.equal(k.singleShare,null);assert.equal(k.occupancy,null);assert.equal(k.setupShare,null);assert.equal(k.lowShare,1);
  const unknown=calculateKpis({...rank,criticalPath:[{onCore:true,startUs:0,endUs:5}]});assert.equal(unknown.singleShare,null);
});

test('内置数据的 KPI 与原 Demo 数据口径一致',async()=>{
  const data=JSON.parse(await readFile(new URL('../data/example.json',import.meta.url),'utf8'));
  for(const rank of data.ranks){const k=calculateKpis(rank), expected=rank.kpi;
    assert.ok(Math.abs(k.singleUs-expected.singleBlockOnPathUs)<.2);
    assert.ok(Math.abs(k.occupancy-expected.occupancy)<.0001);
    assert.ok(Math.abs(k.setupShare-expected.setupShare)<.0001);
  }
});
