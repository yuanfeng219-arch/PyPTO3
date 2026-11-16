// 逐帧渲染 index.html 并合成 MP4。
// 依赖：playwright-core、ffmpeg-static（通过 NODE_PATH 指向安装目录），以及本机 Chrome。
// 用法：
//   node record.cjs                       -> 输出 cuda-agent-paged-attention.mp4
//   node record.cjs --stills 10,40,80     -> 只导出指定秒数的 PNG 截图（用于检查）
//   node record.cjs --fps 30 --out x.mp4
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const { chromium } = require('playwright-core');
const ffmpegPath = require('ffmpeg-static');

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf('--' + name); return i >= 0 ? args[i + 1] : def; };
const FPS = Number(opt('fps', 30));
const OUT = path.resolve(opt('out', path.join(__dirname, 'cuda-agent-paged-attention.mp4')));
const STILLS = opt('stills', null);
const STILL_DIR = path.resolve(opt('still-dir', path.join(__dirname, 'stills')));
const CHROME = opt('chrome', 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe');

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME, args: ['--force-color-profile=srgb', '--hide-scrollbars'] });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  const url = 'file:///' + path.join(__dirname, 'index.html').replace(/\\/g, '/') + '?capture=1';
  await page.goto(url);
  await page.waitForFunction(() => window.__ready === true);
  await page.evaluate(() => document.fonts.ready);
  const duration = await page.evaluate(() => window.__duration);
  console.log(`duration ${duration.toFixed(2)}s`);

  if (STILLS) {
    fs.mkdirSync(STILL_DIR, { recursive: true });
    for (const s of STILLS.split(',').map(Number)) {
      await page.evaluate((t) => window.__render(t), s);
      const file = path.join(STILL_DIR, `t${String(s).padStart(6, '0')}.png`);
      await page.screenshot({ path: file });
      console.log('still', file);
    }
    await browser.close();
    return;
  }

  const ff = spawn(ffmpegPath, ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', '-',
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT], { stdio: ['pipe', 'inherit', 'inherit'] });
  const total = Math.ceil(duration * FPS);
  const t0 = Date.now();
  for (let i = 0; i <= total; i++) {
    await page.evaluate((t) => window.__render(t), i / FPS);
    const buf = await page.screenshot({ type: 'jpeg', quality: 92 });
    if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once('drain', r));
    if (i % 150 === 0) console.log(`frame ${i}/${total}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  ff.stdin.end();
  await new Promise((r) => ff.on('close', r));
  await browser.close();
  console.log('wrote', OUT);
})().catch((e) => { console.error(e); process.exit(1); });
