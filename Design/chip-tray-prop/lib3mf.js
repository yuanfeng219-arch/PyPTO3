'use strict';
// 最小 3MF 写出器：ZIP 容器 + 3D/3dmodel.model + Metadata/model_settings.config
// 多个零件用 <components> 组装成一个对象，坐标保持原样，所以打开就是拼好的。
// model_settings.config 给每个零件指定耗材号（Bambu Studio / OrcaSlicer 读这个文件）。
// 即使切片软件忽略了该配置，对象本身仍是组装好的，只需手动选耗材。

const zlib = require('zlib');

/* ---------- ZIP ---------- */

let CRC_TABLE = null;
function crc32(buf) {
  if (!CRC_TABLE) {
    CRC_TABLE = new Int32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      CRC_TABLE[n] = c;
    }
  }
  let c = -1;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}

function zip(entries) {
  const locals = [], centrals = [];
  let offset = 0;
  for (const { name, data } of entries) {
    const nameBuf = Buffer.from(name, 'utf8');
    const comp = zlib.deflateRawSync(data, { level: 9 });
    const crc = crc32(data);
    const lh = Buffer.alloc(30);
    lh.writeUInt32LE(0x04034b50, 0);
    lh.writeUInt16LE(20, 4); lh.writeUInt16LE(0, 6); lh.writeUInt16LE(8, 8);
    lh.writeUInt16LE(0, 10); lh.writeUInt16LE(0x2821, 12);   // 固定时间戳，输出可复现
    lh.writeUInt32LE(crc, 14);
    lh.writeUInt32LE(comp.length, 18); lh.writeUInt32LE(data.length, 22);
    lh.writeUInt16LE(nameBuf.length, 26); lh.writeUInt16LE(0, 28);
    locals.push(lh, nameBuf, comp);

    const ch = Buffer.alloc(46);
    ch.writeUInt32LE(0x02014b50, 0);
    ch.writeUInt16LE(20, 4); ch.writeUInt16LE(20, 6); ch.writeUInt16LE(0, 8);
    ch.writeUInt16LE(8, 10); ch.writeUInt16LE(0, 12); ch.writeUInt16LE(0x2821, 14);
    ch.writeUInt32LE(crc, 16);
    ch.writeUInt32LE(comp.length, 20); ch.writeUInt32LE(data.length, 24);
    ch.writeUInt16LE(nameBuf.length, 28);
    ch.writeUInt32LE(0, 38);
    ch.writeUInt32LE(offset, 42);
    centrals.push(ch, nameBuf);
    offset += lh.length + nameBuf.length + comp.length;
  }
  const cd = Buffer.concat(centrals);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(entries.length, 8); eocd.writeUInt16LE(entries.length, 10);
  eocd.writeUInt32LE(cd.length, 12); eocd.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, cd, eocd]);
}

/* ---------- 网格 ---------- */

function meshXml(tris) {
  const index = new Map();
  const verts = [];
  const key = (p) => `${p[0].toFixed(4)},${p[1].toFixed(4)},${p[2].toFixed(4)}`;
  const idOf = (p) => {
    const k = key(p);
    let i = index.get(k);
    if (i === undefined) { i = verts.length; index.set(k, i); verts.push(p); }
    return i;
  };
  const faces = tris.map((t) => t.v.map(idOf));
  const v = verts.map((p) => `    <vertex x="${+p[0].toFixed(4)}" y="${+p[1].toFixed(4)}" z="${+p[2].toFixed(4)}"/>`).join('\n');
  const f = faces.map((a) => `    <triangle v1="${a[0]}" v2="${a[1]}" v3="${a[2]}"/>`).join('\n');
  return `   <mesh>\n   <vertices>\n${v}\n   </vertices>\n   <triangles>\n${f}\n   </triangles>\n   </mesh>`;
}

const esc = (s) => String(s).replace(/[<>&"']/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&apos;' }[c]));
const IDENTITY = '1 0 0 0 1 0 0 0 1 0 0 0';

/* ---------- 入口 ---------- */

// parts: [{ name, tris, extruder }]，所有零件共用同一套坐标
function build3mf(objectName, parts) {
  const objs = [];
  const comps = [];
  const cfg = [];
  parts.forEach((p, i) => {
    const id = i + 1;
    objs.push(`  <object id="${id}" type="model">\n${meshXml(p.tris)}\n  </object>`);
    comps.push(`    <component objectid="${id}" transform="${IDENTITY}"/>`);
    cfg.push(
      `    <part id="${id}" subtype="normal_part">\n` +
      `      <metadata key="name" value="${esc(p.name)}"/>\n` +
      `      <metadata key="extruder" value="${p.extruder}"/>\n` +
      `      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n` +
      `    </part>`
    );
  });
  const assemblyId = parts.length + 1;

  const model =
    `<?xml version="1.0" encoding="UTF-8"?>\n` +
    `<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:BambuStudio="http://schemas.bambulab.com/package/2021">\n` +
    ` <metadata name="Application">PyPTO chip-tray-prop generator</metadata>\n` +
    ` <metadata name="Title">${esc(objectName)}</metadata>\n` +
    ` <resources>\n${objs.join('\n')}\n` +
    `  <object id="${assemblyId}" type="model">\n   <components>\n${comps.join('\n')}\n   </components>\n  </object>\n` +
    ` </resources>\n` +
    ` <build>\n  <item objectid="${assemblyId}" transform="${IDENTITY}" printable="1"/>\n </build>\n` +
    `</model>\n`;

  const settings =
    `<?xml version="1.0" encoding="UTF-8"?>\n<config>\n` +
    `  <object id="${assemblyId}">\n` +
    `    <metadata key="name" value="${esc(objectName)}"/>\n` +
    `    <metadata key="extruder" value="${parts[0].extruder}"/>\n` +
    `${cfg.join('\n')}\n  </object>\n</config>\n`;

  const contentTypes =
    `<?xml version="1.0" encoding="UTF-8"?>\n` +
    `<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n` +
    ` <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n` +
    ` <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n` +
    ` <Default Extension="config" ContentType="application/xml"/>\n</Types>\n`;

  const rels =
    `<?xml version="1.0" encoding="UTF-8"?>\n` +
    `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n` +
    ` <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n</Relationships>\n`;

  return zip([
    { name: '[Content_Types].xml', data: Buffer.from(contentTypes, 'utf8') },
    { name: '_rels/.rels', data: Buffer.from(rels, 'utf8') },
    { name: '3D/3dmodel.model', data: Buffer.from(model, 'utf8') },
    { name: 'Metadata/model_settings.config', data: Buffer.from(settings, 'utf8') },
  ]);
}

module.exports = { build3mf };
