// OP paths reuse diagram-legend-library; performance features use the same 24px stroke grammar.
window.TaskSemanticIcons = (() => {
  const features = {
    recommended:'M4 5h16M4 10h12M4 15h8M15 17l3 3 4-6',
    none:'M5 19L19 5',
    tail:'M4 5h9M4 12h11M4 19h16M17 16l3 3-3 3',
    parallel:'M4 5h16M4 12h6M4 19h6M14 12h6M14 19h6',
    imbalance:'M4 5h10M4 12h16M4 19h6',
    underfill:'M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h2M20 14v2M20 20h-2M14 20v-2',
    fragmented:'M3 5h4M11 5h3M18 5h3M5 12h3M12 12h4M3 19h3M10 19h3M17 19h4',
    setup:'M4 5v14M4 12h16M9 8v8M16 8l4 4-4 4',
    bubble:'M3 8h5v8H3zM16 8h5v8h-5zM10 12h1M13 12h1'
  };
  const op = window.PASS_LEGEND_ICONS;
  function taskPath(name='') {
    const rules = [[/^(load|store)$/i,'KV Read / Write'],[/^(mul|multiply)$/i,'Gate / Multiply'],[/rope/i,'RoPE'],[/matmul|gemm/i,'MatMul'],[/hadamard/i,'Gate / Multiply'],[/rms|norm/i,'RMSNorm'],[/topk|top_k/i,'Top-K'],[/softmax/i,'Softmax'],[/a2a|allgather|all_to_all|gather/i,'All-to-All'],[/split/i,'Split'],[/concat|merge|pack/i,'Concat'],[/reduce|pool/i,'Reduce / Pool'],[/transpose/i,'Transpose'],[/reshape/i,'Reshape'],[/silu/i,'SiLU'],[/mix|hc_/i,'HC Mix'],[/qk_|pv_|attention/i,'Attention'],[/proj|linear/i,'Linear Projection'],[/kv_|cache/i,'KV Read / Write']];
    return op[rules.find(([re])=>re.test(name))?.[1]] || 'M5 5h14v14H5zM9 9h6v6H9z';
  }
  const featurePath = mode => features[mode] || features.recommended;
  const badge = (path, background) => `<span class="semantic-icon" style="${background ? `background:${background};color:#111` : ''}" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="${path}"/></svg></span>`;
  return {taskPath,featurePath,task:(name,background)=>badge(taskPath(name),background),feature:(mode,background)=>badge(featurePath(mode),background)};
})();
