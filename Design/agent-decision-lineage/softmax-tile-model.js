/* Shared logical tile identity; independent of UI and per-core slot rotation. */
window.SoftmaxTileModel=(()=>{
 const presets=[[7,200],[32,4096]];let state={selectedPreset:0,selectedTileId:'tile-0',selectedBlockId:0,coreCapacityScenario:'enough'};
 function tiles(index=state.selectedPreset){const [b,n]=presets[index];return Array.from({length:Math.ceil(b/3)},(_,i)=>({id:'tile-'+i,rowRange:[i*3,Math.min(b-1,i*3+2)],allocatedShape:[3,8192],validShape:[Math.min(3,b-i*3),n]}))}
 function assignment(index=state.selectedPreset,scenario=state.coreCapacityScenario){const list=tiles(index),capacity=scenario==='limited'?(index===0?2:4):list.length,blockDim=Math.min(capacity,list.length),blocks=Array.from({length:blockDim},(_,id)=>({id,tileIds:[]})),tileOwner={};
 for(let blockId=0;blockId<blockDim;blockId++)for(let i=blockId;i<list.length;i+=blockDim){const localIteration=blocks[blockId].tileIds.length;blocks[blockId].tileIds.push(list[i].id);tileOwner[list[i].id]={blockId,localIteration,slotPair:['in'+localIteration%2,'out'+localIteration%2]}}
 return {blockDim,capacity,blocks,tileOwner};}
 function setState(next){const selectedPreset=next.selectedPreset??state.selectedPreset;if(!presets[selectedPreset])return;const list=tiles(selectedPreset),requested=next.selectedTileId??state.selectedTileId;const selectedTileId=list.some(t=>t.id===requested)?requested:list[0].id,coreCapacityScenario=['enough','limited'].includes(next.coreCapacityScenario)?next.coreCapacityScenario:state.coreCapacityScenario;const mapping=assignment(selectedPreset,coreCapacityScenario);state={selectedPreset,selectedTileId,coreCapacityScenario,selectedBlockId:Math.min(next.selectedBlockId??mapping.tileOwner[selectedTileId].blockId,mapping.blockDim-1)};}
 return {presets,getState:()=>({...state}),setState,tiles,assignment};
})();
