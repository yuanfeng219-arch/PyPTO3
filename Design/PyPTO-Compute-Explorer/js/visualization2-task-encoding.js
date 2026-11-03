/* Token capacity geometry shared by the Canvas renderer and the transition overlay.
 * These are logical Stack/Page units, never elapsed execution time. */
(function (global) {
  'use strict';
  function segments(seqLen, stackTokens, pageTokens) {
    return Array.from({length: Math.ceil(seqLen / stackTokens)}, (_, index) => {
      const tokens = Math.max(0, Math.min(stackTokens, seqLen - index * stackTokens));
      return {index, tokens, fraction: tokens / stackTokens,
        pages: Array.from({length: stackTokens / pageTokens}, (_, page) =>
          Math.max(0, Math.min(pageTokens, tokens - page * pageTokens))) };
    });
  }
  function draw(ctx, {task, x, y, width, height, colors, selected, related, hovered, fontFamily}) {
    const unit = width / task.segments.length, gap = Math.min(2, unit * .1);
    ctx.save();
    ctx.beginPath(); ctx.rect(x,y,width,height); ctx.clip();
    task.segments.forEach(segment => {
      const left = x + segment.index * unit, w = unit - gap;
      ctx.fillStyle = colors.track; ctx.fillRect(left,y,w,height);
      ctx.fillStyle = colors.fill; ctx.fillRect(left,y,w * segment.fraction,height);
      if (unit >= 28) {
        ctx.strokeStyle = colors.page; ctx.lineWidth = .7;
        for (let page=1; page<segment.pages.length; page++) {
          const px=left+w*page/segment.pages.length;
          ctx.beginPath();ctx.moveTo(px,y);ctx.lineTo(px,y+height);ctx.stroke();
        }
      }
    });
    if (selected || related || hovered) {
      ctx.strokeStyle = selected ? colors.text : colors.fill;
      ctx.lineWidth = selected ? 2 : 1; ctx.strokeRect(x+1,y+1,Math.max(0,width-2),Math.max(0,height-2));
    }
    // A neutral label backing preserves contrast over both filled and empty capacity.
    const size=Math.min(11, Math.max(7, (width-3)/String(task.id).length/ .62));
    ctx.font=`${size}px ${fontFamily}`;ctx.textAlign='center';ctx.textBaseline='middle';
    const tw=ctx.measureText(String(task.id)).width;
    ctx.fillStyle=colors.label;ctx.fillRect(x+(width-tw)/2-2,y+Math.max(0,(height-13)/2),tw+4,Math.min(height,13));
    ctx.fillStyle=colors.text;ctx.fillText(String(task.id),x+width/2,y+height/2);
    ctx.restore();
  }
  global.PtoVisualization2TaskEncoding=Object.freeze({segments,draw});
})(window);
