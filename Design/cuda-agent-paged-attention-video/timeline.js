// 视频分镜：开发者准备环境 → 定义任务 → CUDA Agent 的 ReAct 迭代 → 审阅交付。
(function () {
  const E = ENGINE;
  const { B, seq, par, wait, set, move, click, caption, ff, clockTo, prompt, typeCmd, out, userMsg, say, thinking,
    toolStart, toolEnd, openFile, fileStatus, writeFile, setFile, settle, editFile, C, esc } = E;
  const O = (lines) => lines.map(esc);
  const typeD = (s, cps = 24) => Math.max(0.4, s.length / cps);
  const glide = (x, y, d = 0.8) => { const fx = B.cur.x, fy = B.cur.y; par(d, (s, p) => { const e = E.ease(p); s.cursor.x = fx + (x - fx) * e; s.cursor.y = fy + (y - fy) * e; }); B.cur = { x, y }; };

  const HOST = SRC.HOST, WD = SRC.WD;
  const LP = (path) => C('tgb', `dev@${HOST}`) + ':' + C('tb', path) + '$ ';
  const TP = LP('~/CUDA-Agent/agent_workdir');
  const STEP = (n, name) => `${String(n).padStart(2, '0')} / 12 · ${name}`;

  // ------------------------------------------------------------------ 片头
  set((s) => { s.cursor.x = 1300; s.cursor.y = 640; });
  B.cur = { x: 1300, y: 640 };
  wait(4.6);
  seq(0.9, (s, p) => { s.intro = 1 - p; });

  // ------------------------------------------------------------------ 01 准备环境
  caption(STEP(1, '准备环境'), '开发者打开终端，SSH 登录装有 H100 的 GPU 开发机');
  move(984, 1056, 0.9); click();
  seq(0.45, (s, p) => { s.wt.vis = p; s.taskbarActive = 'terminal'; });
  glide(1500, 760, 0.8);
  prompt('wt', 'PS C:\\Users\\dev> ');
  wait(0.5);
  typeCmd('wt', `ssh dev@${HOST}`, 0.9);
  wait(0.4);
  out('wt', O([`Welcome to Ubuntu 22.04.4 LTS (GNU/Linux 5.15.0-119-generic x86_64)`, '', `Last login: Fri Nov 13 21:07:42 2026 from 10.12.3.45`]), 0.5);
  set((s) => { s.wt.title = `dev@${HOST}: ~`; });
  prompt('wt', LP('~'));
  wait(0.5);
  typeCmd('wt', 'nvidia-smi', 0.5);
  out('wt', O([
    'Sat Nov 14 14:32:08 2026',
    '+-----------------------------------------------------------------------------------------+',
    '| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |',
    '|-----------------------------------------+------------------------+----------------------+',
    '| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |',
    '| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |',
    '|=========================================+========================+======================|',
    '|   0  NVIDIA H100 80GB HBM3          On  |   00000000:18:00.0 Off |                    0 |',
    '| N/A   31C    P0             71W /  700W |       1MiB /  81559MiB |      0%      Default |',
    '+-----------------------------------------+------------------------+----------------------+',
  ]), 0.6);
  prompt('wt', LP('~'));
  wait(1.6);
  typeCmd('wt', 'nvcc --version | tail -1 && python3 -c "import torch; print(torch.__version__)"', 2.2);
  out('wt', O(['Build cuda_12.4.r12.4/compiler.34097967_0', '2.5.1+cu124']), 0.4);
  prompt('wt', LP('~'));
  wait(1.0);

  // ------------------------------------------------------------------ 02 获取 CUDA-Agent
  caption(STEP(2, '获取 CUDA-Agent'), '克隆仓库：agent_workdir 是 Agent 的沙箱工作目录');
  typeCmd('wt', 'git clone https://github.com/BytedTsinghua-SIA/CUDA-Agent.git', 2.2);
  out('wt', O([
    "Cloning into 'CUDA-Agent'...",
    'remote: Enumerating objects: 96, done.',
    'remote: Counting objects: 100% (96/96), done.',
    'remote: Compressing objects: 100% (71/71), done.',
    'remote: Total 96 (delta 24), reused 80 (delta 14), pack-reused 0 (from 0)',
    'Receiving objects: 100% (96/96), 1.21 MiB | 4.87 MiB/s, done.',
    'Resolving deltas: 100% (24/24), done.',
  ]), 1.4);
  prompt('wt', LP('~'));
  wait(0.5);
  typeCmd('wt', 'cd CUDA-Agent/agent_workdir && ls', 1.2);
  out('wt', ['binding.cpp  binding_registry.h  ' + C('tb', 'kernels') + '  model.py  model_new.py  SKILL.md  ' + C('tb', 'utils')]);
  prompt('wt', TP);
  wait(1.0);
  caption(STEP(2, '获取 CUDA-Agent'), 'utils/ 下的编译、校验、性能脚本是固定的验收工具，Agent 不能修改');
  typeCmd('wt', 'cat utils/compile.sh', 0.8);
  out('wt', O(SRC.compileSh.split('\n')), 0.5);
  prompt('wt', TP);
  wait(2.6);

  // ------------------------------------------------------------------ 03 打开工作区 & 定义任务
  caption(STEP(3, '定义算子任务'), '用 VS Code Remote-SSH 打开 agent_workdir');
  move(1032, 1056, 0.9); click();
  set((s) => { s.vs.connecting = true; s.taskbarActive = 'vscode'; s.term.caret = true; });
  seq(0.55, (s, p) => { s.vs.vis = p; });
  glide(700, 560, 0.8);
  const initial = ['kernels/axpby.cu', 'kernels/axpby_binding.cpp', 'utils/__init__.py', 'utils/compile.py', 'utils/compile.sh',
    'utils/profiling.py', 'utils/verification.py', 'binding.cpp', 'binding_registry.h', 'model.py', 'model_new.py', 'SKILL.md'];
  wait(1.4);
  set((s) => {
    s.vs.connecting = false;
    initial.forEach((p) => (s.paths[p] = ''));
    s.fs['model.py'] = SRC.modelOld.split('\n').map((t) => ({ t, k: '' }));
    s.fs['model_new.py'] = SRC.modelNewOld.split('\n').map((t) => ({ t, k: '' }));
  });
  prompt('term', TP);
  wait(1.0);
  // 点击 model.py
  const rowsInit = (() => { const paths = {}; initial.forEach((p) => (paths[p] = '')); return E.treeRows(paths, { kernels: true }); })();
  const rowY = (rows, path) => 92 + rows.findIndex((r) => r.path === path) * 22 + 11;
  move(130, rowY(rowsInit, 'model.py'), 0.8); click();
  openFile('model.py');
  wait(1.4);
  caption(STEP(3, '定义算子任务'), 'model.py 是任务输入：替换成 PagedAttention decode 的 PyTorch 参考实现');
  move(760, 300, 0.6); click();
  wait(0.4);
  set((s) => { s.editor.selAll = true; });
  wait(0.7);
  set((s) => {
    s.editor.selAll = false;
    s.fs['model.py'] = SRC.model.split('\n').map((t) => ({ t, k: '' }));
    s.paths['model.py'] = 'M';
    s.editor.caret = { path: 'model.py', line: SRC.model.split('\n').length - 1, col: 37 };
    s.editor.scroll['model.py'] = 38;
  });
  wait(1.0);
  seq(2.2, (s, p) => { s.editor.scroll['model.py'] = Math.round(38 - 38 * E.ease(p)); s.editor.caret = null; });
  wait(1.2);
  caption(STEP(3, '定义算子任务'), 'vLLM 风格分页 KV Cache：fp16，32 个 Q 头 / 8 个 KV 头（GQA），head_dim 128，上下文 1K–4K');
  seq(1.6, (s, p) => { s.editor.scroll['model.py'] = Math.round(16 * E.ease(p)); });
  wait(2.6);

  // ------------------------------------------------------------------ 04 下达任务
  caption(STEP(4, '下达任务'), '在 CUDA Agent 面板描述需求，并引用 model.py 与 SKILL.md');
  move(1640, 930, 0.9); click();
  set((s) => { s.agent.focus = true; });
  const ask = '@model.py 已替换为 PagedAttention decode（fp16，GQA 32/8，head_dim=128，block_size=16）。请按 @SKILL.md 在 kernels/ 中实现 CUDA 算子并接入 model_new.py：先通过 utils.verification，再用 utils.profiling 做到比 torch.compile 至少快 5%。';
  seq(5.0, (s, p) => { s.agent.input = ask.slice(0, Math.round(ask.length * p)); });
  wait(0.5);
  move(1885, 979, 0.6); click();
  set((s) => { s.agent.input = ''; s.agent.focus = false; s.agent.running = true; s.term.name = 'bash · CUDA Agent'; });
  userMsg('u1', ask.replace('@model.py', '`model.py`').replace('@SKILL.md', '`SKILL.md`'));
  thinking();
  glide(1180, 520, 1.0);
  wait(1.2);

  // ------------------------------------------------------------------ 05 理解约束
  caption(STEP(5, '理解约束'), 'Agent 先读规则与参考实现：只能在 kernels/ 写原生 CUDA，utils/ 与 model.py 不可改');
  say('a1', "I'll start with the workspace rules and the reference model.", 0.9, true);
  const read = (id, file, meta, d = 0.6) => { toolStart(id, 'read', file); clockTo(B.clock + 6, d); wait(d); toolEnd(id, 'ok', meta); };
  read('r1', 'SKILL.md', 'done');
  set((s) => { s.treeSel = 'SKILL.md'; });
  read('r2', 'model.py', `${SRC.model.split('\n').length} lines`);
  const bash = (id, cmd, outLines, o = {}) => {
    toolStart(id, 'bash', cmd);
    typeCmd('term', cmd, o.typeD || 0.5);
    if (o.ff) ff(o.ff);
    clockTo(B.clock + (o.sim || 3), o.runD || 0.6);
    if (o.during) o.during();
    out('term', outLines, o.runD || 0.6);
    ff(null);
    toolEnd(id, o.status || 'ok', o.meta || 'exit 0', o.card);
    prompt('term', TP);
  };
  bash('b1', 'ls kernels utils', ['kernels:', 'axpby_binding.cpp  axpby.cu', '', 'utils:', '__init__.py  compile.py  compile.sh  profiling.py  verification.py'],
    { card: ['kernels: axpby_binding.cpp  axpby.cu', 'utils:   __init__.py  compile.py  compile.sh  profiling.py  verification.py'] });
  thinking(); clockTo(60, 1.2); wait(1.2);
  say('a2', [
    '**Task**: decode-phase PagedAttention — q `[32, 32, 128]` fp16, paged KV cache `[8192, 8, 16, 128]`, GQA group = 4, context 1024–4096.',
    '**Rules (SKILL.md)**: raw CUDA in `kernels/` only; `model_new.py` just calls the extension (no torch ops); pass `utils.verification` (atol = rtol = 1e-2) and be ≥ 5% faster than torch.compile.',
    '**Plan v1**: one CTA per (head, seq) — warp-per-token q·k, block softmax in shared memory, thread-per-channel P·V. Correct first, then fast.',
  ].join('\n'), 5.2);
  wait(1.8);

  // ------------------------------------------------------------------ 06 实现 v1
  caption(STEP(6, '实现 v1'), '第 1 版：先保证正确 —— 每个 (head, seq) 一个线程块');
  say('a3', 'The axpby files are only the template example — removing them so just our op gets built.', 1.2);
  bash('b2', 'rm kernels/axpby.cu kernels/axpby_binding.cpp', [], {
    sim: 5, during: () => set((s) => { s.paths['kernels/axpby.cu'] = 'D'; s.paths['kernels/axpby_binding.cpp'] = 'D'; }) });
  const nK1 = SRC.kernelV1.split('\n').length, nB1 = SRC.bindingV1.split('\n').length;
  toolStart('w1', 'write', 'kernels/paged_attention.cu');
  clockTo(135, 8.0);
  writeFile('kernels/paged_attention.cu', SRC.kernelV1, 8.0);
  toolEnd('w1', 'ok', `+${nK1} lines`);
  settle('kernels/paged_attention.cu');
  toolStart('w2', 'write', 'kernels/paged_attention_binding.cpp');
  clockTo(170, 4.2);
  writeFile('kernels/paged_attention_binding.cpp', SRC.bindingV1, 4.2);
  toolEnd('w2', 'ok', `+${nB1} lines`);
  settle('kernels/paged_attention_binding.cpp');
  toolStart('w3', 'edit', 'model_new.py');
  clockTo(180, 2.0);
  writeFile('model_new.py', SRC.modelNew, 2.0, 'M');
  toolEnd('w3', 'ok', '+6 −4', null, '<span class="r">-  return cuda_extension.axpby_forward(a, b, self.alpha, 0)</span>\n<span class="a">+  return cuda_extension.paged_attention_forward(</span>\n<span class="a">+      query, key_cache, value_cache, block_tables, context_lens, self.scale, 0)</span>');
  settle('model_new.py');
  wait(0.6);

  // ------------------------------------------------------------------ 07 编译与修复
  caption(STEP(7, '编译'), 'Agent 调用仓库的 compile.sh（torch cpp_extension + nvcc，目标 sm_90）');
  const CU = `${WD}/kernels/paged_attention.cu`;
  const l1 = SRC.lineOf(SRC.kernelV1, '__shfl_xor(v, offset)'), l2 = SRC.lineOf(SRC.kernelV1, '__shfl_xor(v, offset)', l1);
  const compileCmd = 'TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh';
  const compiling = 'Compiling 3 files: binding.cpp, kernels/paged_attention.cu, kernels/paged_attention_binding.cpp';
  bash('c1', compileCmd, [
    compiling,
    C('tr', 'Compilation failed.'),
    esc(`Error building extension 'cuda_extension': [1/4] /usr/local/cuda/bin/nvcc --generate-dependencies-with-compile -DTORCH_EXTENSION_NAME=cuda_extension -gencode=arch=compute_90,code=sm_90 -O3 --use_fast_math -std=c++17 -c ${CU} -o paged_attention.cuda.o`),
    C('tr', 'FAILED: paged_attention.cuda.o'),
    esc(`${CU}(${l1}): `) + C('tr', 'error') + esc(': identifier "__shfl_xor" is undefined'),
    esc('          v += __shfl_xor(v, offset);'),
    esc('               ^'),
    '',
    esc(`${CU}(${l2}): `) + C('tr', 'error') + esc(': identifier "__shfl_xor" is undefined'),
    esc('          v = fmaxf(v, __shfl_xor(v, offset));'),
    esc('                       ^'),
    '',
    esc(`2 errors detected in the compilation of "${CU}".`),
    'ninja: build stopped: subcommand failed.',
    '[TIME] Compilation took 18.63s',
  ], { typeD: 0.7, runD: 2.0, sim: 19, ff: '⏩ 快进 · 编译 18.6 s', status: 'err', meta: 'exit 1 · 18.6s',
    during: () => set((s) => { s.paths['build/'] = 'ign'; }),
    card: [C('tr', esc(`paged_attention.cu(${l1}): error: identifier "__shfl_xor" is undefined`)), esc(`paged_attention.cu(${l2}): error: identifier "__shfl_xor" is undefined`), '2 errors detected · [TIME] Compilation took 18.63s'] });
  caption(STEP(7, '编译失败 → 修复'), 'nvcc 报错：sm_70 起已移除无 _sync 的 warp shuffle，Agent 按报错改用 __shfl_xor_sync');
  thinking(); clockTo(212, 1.4); wait(1.4);
  say('a4', 'nvcc: `__shfl_xor` is undefined — the non-`_sync` warp shuffles were removed for sm_70+. Switching both reductions to `__shfl_xor_sync(0xffffffff, …)`.', 2.4);
  toolStart('e1', 'edit', 'kernels/paged_attention.cu');
  clockTo(224, 2.4);
  editFile('kernels/paged_attention.cu', ['        v += __shfl_xor(v, offset);'], ['        v += __shfl_xor_sync(0xffffffff, v, offset);'], 1.2);
  editFile('kernels/paged_attention.cu', ['        v = fmaxf(v, __shfl_xor(v, offset));'], ['        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));'], 1.2);
  toolEnd('e1', 'ok', '+2 −2', null, `<span class="r">-        v += __shfl_xor(v, offset);</span>\n<span class="a">+        v += __shfl_xor_sync(0xffffffff, v, offset);</span>\n<span class="r">-        v = fmaxf(v, __shfl_xor(v, offset));</span>\n<span class="a">+        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));</span>`);
  wait(0.8);
  settle('kernels/paged_attention.cu');
  bash('c2', compileCmd, [compiling, C('tg', 'Compile success: cuda_extension.so'), '[TIME] Compilation took 52.41s'],
    { typeD: 0.7, runD: 1.8, sim: 52, ff: '⏩ 快进 · 编译 52.4 s', meta: 'exit 0 · 52.4s',
      during: () => set((s) => { s.paths['cuda_extension.so'] = 'ign'; }),
      card: [C('tg', 'Compile success: cuda_extension.so'), '[TIME] Compilation took 52.41s'] });
  wait(0.6);

  // ------------------------------------------------------------------ 08 正确性
  caption(STEP(8, '正确性校验'), 'utils.verification：5 组随机输入与参考实现逐元素比对（atol = rtol = 1e-2）');
  const VPY = `${WD}/utils/verification.py`;
  bash('v1', 'sudo python3 -m utils.verification', O([
    'Traceback (most recent call last):',
    '  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main',
    '    return _run_code(code, main_globals, None,',
    '  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code',
    '    exec(code, run_globals)',
    `  File "${VPY}", line 101, in <module>`,
    '    main()',
    `  File "${VPY}", line 93, in main`,
    '    check_equal(cuda_output, torch_output)',
    `  File "${VPY}", line 47, in check_equal`,
    '    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)',
    '  File "/usr/local/lib/python3.10/dist-packages/torch/testing/_comparison.py", line 1530, in assert_close',
    '    raise error_metas[0].to_error(msg)',
  ]).concat([
    C('tr', 'AssertionError: Tensor-likes are not close!'), '',
    'Mismatched elements: 126976 / 131072 (96.9%)',
    'Greatest absolute difference: 0.2861 at index (17, 23, 64) (up to 0.01 allowed)',
    'Greatest relative difference: 3846.1 at index (5, 11, 90) (up to 0.01 allowed)',
  ]), { typeD: 0.6, runD: 1.4, sim: 14, status: 'err', meta: 'exit 1 · 13.9s',
    card: [C('tr', 'AssertionError: Tensor-likes are not close!'), 'Mismatched elements: 126976 / 131072 (96.9%)', 'Greatest absolute difference: 0.2861 at index (17, 23, 64)'] });
  caption(STEP(8, '校验失败 → 定位'), '31/32 个头不匹配、只有 head 0 正确 → Agent 推断是 GQA 的 kv_head 映射写错');
  thinking(); clockTo(306, 1.6); wait(1.6);
  say('a5', '126976 / 131072 mismatched = **31 of 32 heads wrong**; only head 0 matches. So the K/V head index is off: with GQA, query head `h` must read KV head `h / (num_heads / num_kv_heads)`, not `h`.', 3.4);
  toolStart('e2', 'edit', 'kernels/paged_attention.cu');
  clockTo(330, 1.6);
  editFile('kernels/paged_attention.cu', ['    const int kv_head = head;'], ['    const int kv_head = head / (num_heads / num_kv_heads);'], 1.6);
  toolEnd('e2', 'ok', '+1 −1', null, '<span class="r">-    const int kv_head = head;</span>\n<span class="a">+    const int kv_head = head / (num_heads / num_kv_heads);</span>');
  wait(0.8);
  settle('kernels/paged_attention.cu');
  const PASS = ['[PASS] check 1/5', '[PASS] check 2/5', '[PASS] check 3/5', '[PASS] check 4/5', '[PASS] check 5/5', '[PASS] verify success'].map((l) => C('tg', l));
  bash('v2', `${compileCmd} && sudo python3 -m utils.verification`, [compiling, C('tg', 'Compile success: cuda_extension.so'), '[TIME] Compilation took 51.87s'].concat(PASS),
    { typeD: 0.9, runD: 2.4, sim: 66, ff: '⏩ 快进 · 编译 + 校验 66 s', meta: 'exit 0 · 66.0s', card: [PASS[4], PASS[5]] });
  caption(STEP(8, '正确性通过'), '5 组随机输入全部通过 —— 正确性闭环完成，进入性能阶段');
  say('a6', 'Correct on all 5 random inputs. Measuring against eager and torch.compile.', 1.2);
  wait(0.6);

  // ------------------------------------------------------------------ 09 性能分析
  caption(STEP(9, '性能分析'), 'utils.profiling：用 torch.profiler 统计 Eager / torch.compile / CUDA 扩展的 GPU 时间');
  const prof = (b, c, x) => `Torch Baseline: ${b}us, Torch Compile: ${c}us, CUDA Extension: ${x}us`;
  bash('p1', 'sudo python3 -m utils.profiling', [prof('4131.552', '683.904', '603.872')],
    { typeD: 0.6, runD: 1.8, sim: 38, ff: '⏩ 快进 · warm-up ×5 + 10 iters', meta: 'exit 0 · 38.2s', card: [prof('4131.552', '683.904', C('ty2', '603.872'))] });
  say('a7', 'CUDA **603.9 µs** vs torch.compile 683.9 µs → **1.13×**. Minimum met (≤ 649.7 µs), goal (≤ 547.1 µs) not yet. Let me look inside the kernel with Nsight Compute.', 3.0);
  caption(STEP(9, '性能分析'), '达到最低要求（≤0.95×）但未达目标（≤0.8×）—— Agent 调用 Nsight Compute 找瓶颈');
  const NCU = 'sudo ncu --section SpeedOfLight -k regex:paged_attention -c 1 python3 -m utils.profiling --single-run cuda_extension --iters 1';
  bash('n1', NCU, [
    C('tm', '==PROF==') + ' Connected to process 51842 (/usr/bin/python3.10)',
    C('tm', '==PROF==') + ' Profiling "paged_attention_v1_kernel" - 0: 0%....50%....100% - 9 passes',
    '[DONE] single-run completed',
    C('tm', '==PROF==') + ' Disconnected from process 51842',
    '[51842] python3.10@127.0.0.1',
    esc('  void paged_attention_v1_kernel<128, 16, 128>(__half *, const __half *, const __half *, const __half *, const int *, const int *, float, int, int) (32, 32, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 9.0'),
    '    Section: GPU Speed Of Light Throughput',
    '    ----------------------- ----------- ------------',
    '    Metric Name             Metric Unit Metric Value',
    '    ----------------------- ----------- ------------',
    '    DRAM Frequency                  Ghz         2.62',
    '    SM Frequency                    Ghz         1.75',
    '    Elapsed Cycles                cycle    1,034,872',
    '    Memory Throughput                 %        24.63',
    C('ty2', '    DRAM Throughput                   %        17.36'),
    '    Duration                         us       591.23',
    '    L2 Cache Throughput               %        24.63',
    C('ty2', '    Compute (SM) Throughput           %        19.84'),
    '    ----------------------- ----------- ------------',
    '',
    C('tc', '    OPT') + '   This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak',
    '          performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak',
    '          typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.',
  ], { typeD: 1.0, runD: 2.6, sim: 95, ff: '⏩ 快进 · Nsight Compute 9 passes', meta: 'exit 0 · 95s',
    card: ['DRAM Throughput  17.36 %', 'Compute (SM)     19.84 %', 'Duration         591.23 us'] });
  caption(STEP(9, '定位瓶颈'), 'DRAM 吞吐仅 17%：同组 4 个 Q 头重复读取同一份 K/V，且只有 1024 个线程块');
  thinking(); clockTo(560, 1.4); wait(1.4);
  say('a8', [
    'DRAM throughput is only **17%**; 591 µs is ~6× the bandwidth floor (~335 MB of K/V → ~100 µs at 3.35 TB/s). Why:',
    '1. the 4 query heads of a GQA group each re-read the same K/V rows;',
    '2. only 1024 CTAs, each with a long serial P·V loop per thread;',
    '3. 2-byte scalar loads.',
    '**Plan v2**: one CTA per (kv_head, seq, 512-token partition) serving all 4 heads, 16-byte vectorized loads, plus a tiny reduce kernel to merge partitions (split-K / flash-decoding).',
  ].join('\n'), 6.4);
  clockTo(604, 0.1);
  wait(1.6);

  // ------------------------------------------------------------------ 10 优化 v2
  caption(STEP(10, '优化 v2'), 'Split-K 分区 + GQA 共享 K/V + 128-bit 向量化访存 + 分区归约 kernel');
  bash('b3', 'cp kernels/paged_attention.cu kernels/paged_attention.cu.bak', [], {
    sim: 3, during: () => set((s) => { s.paths['kernels/paged_attention.cu.bak'] = 'U'; }) });
  const nK2 = SRC.kernelV2.split('\n').length, nB2 = SRC.bindingV2.split('\n').length;
  toolStart('w4', 'write', 'kernels/paged_attention.cu');
  clockTo(757, 9.5);
  writeFile('kernels/paged_attention.cu', SRC.kernelV2, 9.5);
  toolEnd('w4', 'ok', `+${nK2} −${nK1}`);
  settle('kernels/paged_attention.cu');
  toolStart('w5', 'write', 'kernels/paged_attention_binding.cpp');
  clockTo(797, 3.6);
  writeFile('kernels/paged_attention_binding.cpp', SRC.bindingV2, 3.6);
  toolEnd('w5', 'ok', `+${nB2} −${nB1}`);
  settle('kernels/paged_attention_binding.cpp');
  bash('v3', `${compileCmd} && sudo python3 -m utils.verification`, [compiling, C('tg', 'Compile success: cuda_extension.so'), '[TIME] Compilation took 55.02s'].concat(PASS),
    { typeD: 0.9, runD: 2.2, sim: 70, ff: '⏩ 快进 · 编译 + 校验 70 s', meta: 'exit 0 · 70.1s', card: [PASS[4], PASS[5]] });
  bash('p2', 'sudo python3 -m utils.profiling', [prof('4124.380', '684.517', '191.902')],
    { typeD: 0.6, runD: 1.6, sim: 36, ff: '⏩ 快进 · warm-up ×5 + 10 iters', meta: 'exit 0 · 36.4s', card: [prof('4124.380', '684.517', C('tg', '191.902'))] });
  caption(STEP(10, '优化 v2'), 'v2：191.9 µs，比 torch.compile 快 3.57×，有效带宽约 1.75 TB/s');
  say('a9', 'v2: **191.9 µs** → **3.57×** vs torch.compile, 21.5× vs eager. Effective bandwidth ≈ 1.75 TB/s (~52% of HBM3 peak). Goal met. Cleaning up intermediate files as SKILL.md requires, then re-running the full check.', 3.6);
  wait(0.8);

  // ------------------------------------------------------------------ 11 清理复测
  caption(STEP(11, '清理与复测'), '按 SKILL.md 删除中间版本，只保留最终 .cu 与 _binding.cpp，再完整复测一次');
  bash('b4', 'rm -f kernels/*_v[0-9].cu kernels/*_old.cu kernels/*_test.cu kernels/*.bak', [], {
    typeD: 0.9, sim: 4, during: () => set((s) => { delete s.paths['kernels/paged_attention.cu.bak']; }) });
  bash('f1', `${compileCmd} && sudo python3 -m utils.verification && sudo python3 -m utils.profiling`,
    [compiling, C('tg', 'Compile success: cuda_extension.so'), '[TIME] Compilation took 54.66s'].concat(PASS, [prof('4127.906', '684.215', '191.337')]),
    { typeD: 1.1, runD: 2.6, sim: 110, ff: '⏩ 快进 · 编译 + 校验 + 性能 110 s', meta: 'exit 0 · 110s',
      card: [PASS[5], prof('4127.906', '684.215', C('tg', '191.337'))] });
  thinking(); clockTo(1026, 1.0); wait(1.0);
  say('a10', [
    '**Done.** `ModelNew` now runs a custom split-K PagedAttention kernel.',
    '| | latency | vs compile |',
    '| --- | --- | --- |',
    '| torch eager | 4127.9 µs | 0.17× |',
    '| torch.compile | 684.2 µs | 1.00× |',
    '| **CUDA extension** | **191.3 µs** | **3.58×** |',
    'Changed: `kernels/paged_attention.cu`, `kernels/paged_attention_binding.cpp`, `model_new.py` (axpby example removed). `model.py`, `binding.cpp` and `utils/` untouched.',
  ].join('\n'), 3.6);
  set((s) => { s.agent.running = false; });
  wait(2.4);

  // ------------------------------------------------------------------ 12 审阅交付
  caption(STEP(12, '审阅交付'), '开发者在源代码管理中逐个审阅 Agent 的改动');
  move(24, 155, 1.0); click();
  set((s) => { s.vs.view = 'scm'; });
  wait(0.8);
  move(150, 164 + 2 * 22 + 11, 0.8); click();
  set((s) => {
    s.treeSel = 'kernels/paged_attention.cu'; s.editor.active = 'kernels/paged_attention.cu';
    s.editor.center['kernels/paged_attention.cu'] = null; s.editor.scroll['kernels/paged_attention.cu'] = 0;
  });
  glide(980, 420, 0.8);
  wait(1.2);
  seq(5.0, (s, p) => { s.editor.scroll['kernels/paged_attention.cu'] = Math.round(78 * E.ease(p)); });
  wait(1.6);
  caption(null);

  // ------------------------------------------------------------------ 片尾
  set((s) => {
    s.outroData = {
      title: 'PagedAttention decode · CUDA Agent 交付结果',
      lead: 'H100 80GB HBM3 · fp16 · 32 seqs × 32 heads（GQA 32/8）· head_dim 128 · block_size 16 · context 1K–4K',
      kpis: [
        { l: '正确性（5 组随机输入）', v: '5/5', u: 'PASS' },
        { l: '对比 torch.compile', v: '3.58', u: '×' },
        { l: '对比 PyTorch Eager', v: '21.6', u: '×' },
        { l: 'Agent 工具调用 / 模拟耗时', v: '22', u: '次 · 约 17 min' },
      ],
      bars: [
        { l: 'PyTorch Eager', v: 4127.906, c: '#5b6474' },
        { l: 'torch.compile', v: 684.215, c: '#8a94a6' },
        { l: 'CUDA v1（naive）', v: 603.872, c: '#d9a441' },
        { l: 'CUDA v2（split-K）', v: 191.337, c: '#76b900' },
      ],
      path: [['v1 实现'], ['编译失败 __shfl_xor', 'bad'], ['改 _sync'], ['校验失败 31/32 头', 'bad'], ['修 GQA 映射'], ['5/5 PASS', 'good'], ['1.13×'], ['ncu：DRAM 17%'], ['v2 split-K + 向量化'], ['3.58×', 'good'], ['清理复测', 'good']],
      src: '工作流来源：github.com/BytedTsinghua-SIA/CUDA-Agent（agent_workdir/SKILL.md、utils/compile.sh、verification.py、profiling.py）· 本视频为还原演示，算子代码为示意实现，性能数据为示意值',
    };
  });
  seq(0.9, (s, p) => { s.outro = p; s.cursor.vis = 1 - p; });
  wait(9.0);
})();
