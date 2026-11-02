import { App } from '@modelcontextprotocol/ext-apps';

const host = document.getElementById('kit');
const status = document.getElementById('connection-status');
const kernels = [window.HNKitKernels.fa_fused, window.HNKitKernels.matmul_64];
let instance;
const app = new App({ name: 'Hardware-Native Kit', version: '1.0.0' });
app.ontoolresult = (result) => {
  const data = result.structuredContent;
  if (result.isError || !data || (!data.manifest && !kernels.some((item) => item.id === data.kernel))) {
    status.textContent = '无法加载算子：' + (result.content?.find((item) => item.type === 'text')?.text || '工具未返回有效结果');
    return;
  }
  const kernel = data.manifest || kernels.find((item) => item.id === data.kernel);
  if (data.manifest) {
    const i = kernels.findIndex((item) => item.id === data.manifest.id);
    if (i < 0) kernels.push(data.manifest); else kernels[i] = data.manifest;
  }
  if (!instance || data.manifest) instance = window.HNKit.mount(host, { kernels, kernel });
  else instance.setKernel(kernel);
  // 同步切换按钮，保证工具选择的 kernel 与界面选中态一致。
  host.querySelectorAll('[data-pick]').forEach((button) => {
    button.setAttribute('aria-pressed', String(kernels[Number(button.dataset.pick)] === kernel));
  });
  instance.setConfig(data.knobs || {});
  status.hidden = true;
};
app.connect().catch((error) => { status.textContent = '无法连接 Agent：' + error.message; });
