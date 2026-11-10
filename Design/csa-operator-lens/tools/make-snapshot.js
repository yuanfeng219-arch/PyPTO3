/* 重新生成 js/source-snapshot.js：在仓库根目录运行
 *   node Design/csa-operator-lens/tools/make-snapshot.js Data/DeepseekV4/deepseek_v4_flash_dspark Design/csa-operator-lens/js/source-snapshot.js
 * 原文件更新后需同时核对 lens-data.js / csa-graph-model.js 里写死的行号。 */
const fs=require('fs'),crypto=require('crypto'),path=require('path');
const [src,out]=process.argv.slice(2);
const files=['decode_csa.py','decode_sparse_attn_csa.py','hc_pre.py','rmsnorm.py','qkv_proj_rope.py','decode_cp_token_allgather.py','decode_compressor_ratio4.py','decode_indexer_compressor.py','decode_indexer.py','decode_o_proj.py','hc_post.py'];
const o={base:'Data/DeepseekV4/deepseek_v4_flash_dspark/',files:{}};
for(const f of files){const buf=fs.readFileSync(path.join(src,f));const text=buf.toString('utf8').replace(/\r\n/g,'\n');
o.files[f]={path:o.base+f,sha256:crypto.createHash('sha256').update(buf).digest('hex'),lines:text.split('\n').length,text};}
fs.writeFileSync(out,'/* 源码本地快照：由 Data/DeepseekV4/deepseek_v4_flash_dspark/ 原文件逐字生成，SHA-256 对应原文件字节。原文件更新后需重新生成并核对行号。 */\nwindow.LENS_SOURCE = '+JSON.stringify(o)+';\n');
console.log(files.map(f=>f+' '+o.files[f].lines+' '+o.files[f].sha256.slice(0,12)).join('\n'));
