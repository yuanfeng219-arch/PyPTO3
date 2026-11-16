/* 将工作台的正交折线转成圆角路径；短线段自动缩小圆角。 */
(function () {
  'use strict';
  window.roundWorkbenchConnector = function (path, radius) {
    const tokens = path.match(/[A-Za-z]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:e[-+]?\d+)?/gi) || [];
    const points = [];
    let x = 0, y = 0, i = 0;
    while (i < tokens.length) {
      const command = tokens[i++];
      if (command === 'M' || command === 'L') { x = Number(tokens[i++]); y = Number(tokens[i++]); }
      else if (command === 'H') x = Number(tokens[i++]);
      else if (command === 'V') y = Number(tokens[i++]);
      else return path;
      if (!Number.isFinite(x) || !Number.isFinite(y)) return path;
      const prev = points[points.length - 1];
      if (!prev || prev.x !== x || prev.y !== y) points.push({ x: x, y: y });
    }
    if (points.length < 3) return path;
    const parts = ['M ' + points[0].x + ' ' + points[0].y];
    for (let n = 1; n < points.length - 1; n++) {
      const a = points[n - 1], b = points[n], c = points[n + 1];
      const ab = Math.hypot(a.x - b.x, a.y - b.y), bc = Math.hypot(c.x - b.x, c.y - b.y);
      if ((a.x - b.x) * (c.y - b.y) === (a.y - b.y) * (c.x - b.x)) { parts.push('L ' + b.x + ' ' + b.y); continue; }
      const r = Math.min(radius || 8, ab / 2, bc / 2);
      parts.push('L ' + (b.x + (a.x - b.x) * r / ab) + ' ' + (b.y + (a.y - b.y) * r / ab));
      parts.push('Q ' + b.x + ' ' + b.y + ' ' + (b.x + (c.x - b.x) * r / bc) + ' ' + (b.y + (c.y - b.y) * r / bc));
    }
    const last = points[points.length - 1];
    parts.push('L ' + last.x + ' ' + last.y);
    return parts.join(' ');
  };
})();
