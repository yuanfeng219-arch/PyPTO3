/* Selected M/N/K slices in operand and result tensors. The matrix steps use
 * source tile counts, while the task markers represent SPMD Block IDs. */
(function(global){
  'use strict';
  const model=global.PtoCsaQProjectionModel;
  const {ticks,range}=model;
  const diagramWidth=1000, diagramHeight=315;
  const matrixWidth=262, matrixHeight=166, y=71;

  function mark(label,attr,value,active,body){
    return `<g role="button" tabindex="0" aria-label="${label}" aria-pressed="${active}" ${attr}="${value}">${body}<title>${label}</title></g>`;
  }
  const rect=(x,y,w,h,cls)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" class="${cls}" rx="2"/>`;
  const text=(x,y,value,cls='')=>`<text x="${x}" y="${y}" class="${cls}">${value}</text>`;

  function render(info,view,selectedK){
    if(!model.isMatmul(info))return '';
    const shape=model.blockShape(info,view.block,view);
    const qr=info.type==='qr';
    const splitCount=qr?model.groupSize(info):1;
    const kTotal=shape.kLength*splitCount;
    const nCount=model.blockCount(info,view.tokens)/splitCount;
    const nTotal=nCount*shape.n;
    const x=32, w=365, out=697;
    const kCount=shape.kLength/shape.kTile;
    const selectedStart=shape.k+selectedK*shape.kTile;
    const pixelN=matrixWidth/nCount;
    const globalKCount=kTotal/shape.kTile;
    const kPosition=(i,start)=>start+i*matrixWidth/globalKCount;
    const xKWidth=matrixWidth/globalKCount;
    const wKHeight=matrixHeight/globalKCount;
    const nSelected=shape.col/shape.n;
    const paddedRows=shape.tiles.at(-1).start+shape.tiles.at(-1).size;
    const pixelsPerRow=matrixHeight/paddedRows;
    const mY=tile=>y+tile.start*pixelsPerRow;
    const mHeight=tile=>tile.size*pixelsPerRow;
    const validHeight=shape.valid*pixelsPerRow;
    const arrow=(from,to)=>`<path class="csa-qproj-matrix-edge" d="M${from} 154H${to-10}m-6 -4 6 4 -6 4"/>`;
    const selectedN = rect(w+nSelected*pixelN,y,pixelN,matrixHeight,'matrix-range-n');
    const selectedKRow = rect(w,y+selectedStart/shape.kTile*wKHeight,matrixWidth,wKHeight,'matrix-range-k');
    const intersection=rect(w+nSelected*pixelN,y+selectedStart/shape.kTile*wKHeight,pixelN,wKHeight,'matrix-intersection');
    const outputCell=rect(out+nSelected*pixelN,mY(shape),pixelN,validHeight,'matrix-intersection');

    return `<div class="csa-qproj-tiling">
      <svg class="csa-qproj-matrices" width="${diagramWidth}" height="${diagramHeight}" viewBox="0 0 ${diagramWidth} ${diagramHeight}" role="group" aria-label="M N K 张量切分">
        ${text(x,22,qr?'X · x_normed_t':'QR INT8 · qr_i8_matmul','matrix-title')}
        ${text(w,22,qr?'Wq_a':'Wq_b','matrix-title')}
        ${text(out,22,qr?'Shared QR FP32':'Q INT32 Accumulator','matrix-title')}
        ${text(x,45,`${qr?'BF16':'INT8'} · ${qr?view.tokens:paddedRows} × ${kTotal}`)}
        ${text(w,45,`${qr?'BF16':'INT8'} · ${kTotal} × ${nTotal}`)}
        ${text(out,45,`${qr?'FP32':'INT32'} · ${paddedRows} × ${nTotal}`)}
        ${rect(x,y,matrixWidth,matrixHeight,'matrix-base')}
        ${rect(w,y,matrixWidth,matrixHeight,'matrix-base')}
        ${rect(out,y,matrixWidth,matrixHeight,'matrix-base')}
        ${rect(x+shape.k/kTotal*matrixWidth,y,shape.kLength/kTotal*matrixWidth,matrixHeight,'matrix-k-domain')}
        ${shape.tiles.map((m,i)=>mark(`${m.path} M ${range(m.start,m.valid)}`,'data-kvsp-tile',i,i===view.tileIndex,
          rect(x,mY(m),matrixWidth,mHeight(m),'matrix-tile matrix-row'))).join('')}
        ${rect(kPosition(selectedStart/shape.kTile,x),mY(shape),xKWidth,validHeight,'matrix-intersection')}
        ${selectedN}${selectedKRow}${intersection}
        ${ticks(nCount).map(i=>mark(`N Group ${i} · ${range(i*shape.n,shape.n)}`,'data-kvsp-logical',qr?i*splitCount+shape.split:i,i===nSelected,
          rect(w+i*pixelN,y,pixelN,matrixHeight,'matrix-tile matrix-n'))).join('')}
        ${shape.tiles.map(m=>rect(out,mY(m),matrixWidth,mHeight(m),'matrix-tile matrix-row')).join('')}
        ${outputCell}
        ${arrow(x+matrixWidth+12,w)}${arrow(w+matrixWidth+12,out)}
        ${ticks(kCount).map(i=>mark(`K Iteration ${i} · ${range(shape.k+i*shape.kTile,shape.kTile)}`,'data-kvsp-k',i,i===selectedK,
          rect(32+i*34,270,29,25,'matrix-k-btn')+text(38+i*34,287,'K'+i,'matrix-k-label'))).join('')}
        ${text(x,254,'K Tile 选择')}${text(w,254,`N ${range(shape.col,shape.n)} · K ${range(selectedStart,shape.kTile)}`)}
        ${text(out,254,qr?`Split ${shape.split} · Atomic Add → 同一 N Tile`:`${shape.path} · ${shape.valid}/${shape.size} 行`)}
        ${text(w,285,`Block ${view.block} · M ${range(shape.start,shape.valid)}`)}
      </svg></div>`;
  }

  global.PtoCsaQProjectionTiles=Object.freeze({render});
})(window);
