const assert = require('node:assert/strict');
const path = require('node:path');
require(path.join(__dirname, '../js/semantic-computation-presentation.js'));
require(path.join(__dirname, '../js/semantic-computation-layout.js'));
const layout = globalThis.PTO_COMPUTATION_LAYOUT;
// Adjacent domain regions must reserve their frames even in compact flow mode.
const model={id:'regions',nodes:[{id:'a',type:'operation'},{id:'b',type:'operation'}],edges:[{s:'a',t:'b',type:'data'}],groups:[{id:'first',name:'First',members:['a']},{id:'second',name:'Second',members:['b']}]};
const graph=layout.layout(model,new Set(),{flowLayout:true,compactGlyphs:true,rootLayerGap:64});
assert.ok(graph.groups[1].y>=graph.groups[0].y+graph.groups[0].h+32,'region frames overlap');
// A lateral state dependency must leave and enter the top, not cross a peer node.
const state={id:'state',nodes:[{id:'a',type:'state'},{id:'peer',type:'state'},{id:'b',type:'state'}],edges:[],groups:[]};
const placed=layout.layout(state,new Set(),{flowLayout:true,compactGlyphs:true});
const byId=new Map(placed.nodes.map(n=>[n.id,n]));
const a=byId.get('a'),b=byId.get('b');
placed.edges=[{s:'a',t:'b',type:'state'}];
const routes=layout.routeEdges(placed.nodes,placed.edges,placed.width,{flowLayout:true});
assert.equal(routes[0].points[0].y,a.y);
assert.ok(routes[0].points[1].y<a.y,'state channel must sit above the row');
assert.equal(routes[0].points.at(-1).x,routes[0].points.at(-2).x,'arrow must enter vertically');
assert.ok(!routes[0].path.includes('NaN'));
console.log('PASS: compact region separation and state-edge routing');
