/* Shared logical tile identity; independent of UI and per-core slot rotation. */
window.SoftmaxTileModel=(()=>{
 const presets=[[7,200],[32,4096]];let state={selectedPreset:0,selectedTileId:'tile-0'};
 function tiles(index=state.selectedPreset){const [b,n]=presets[index];return Array.from({length:Math.ceil(b/3)},(_,i)=>({id:'tile-'+i,rowRange:[i*3,Math.min(b-1,i*3+2)],allocatedShape:[3,8192],validShape:[Math.min(3,b-i*3),n]}))}
 function setState(next){const selectedPreset=next.selectedPreset??state.selectedPreset;if(!presets[selectedPreset])return;const list=tiles(selectedPreset),requested=next.selectedTileId??state.selectedTileId;state={selectedPreset,selectedTileId:list.some(t=>t.id===requested)?requested:list[0].id};}
 return {presets,getState:()=>({...state}),setState,tiles};
})();
