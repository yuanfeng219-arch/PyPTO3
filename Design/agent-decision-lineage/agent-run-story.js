/* Reconstructed presentation, never an original transcript. Contract snapshot: evidence/SPEC.md machine-contract. */
window.AGENT_RUN_STORY = {
 contract: {"schema_version": 1, "op_name": "softmax", "formula": "y[b,i] = exp(x[b,i] - m[b]) / s[b]; m[b] = max_j(x[b,j]); s[b] = sum_j(exp(x[b,j] - m[b]))", "supported_dtypes": ["float16"], "inputs": [{"name": "x", "shape": ["B", "N"], "dtype": "float16", "value_range": [-65504, 65504]}], "outputs": [{"name": "y", "shape": ["B", "N"], "dtype": "float16", "value_range": [0, 1]}], "default_params": {}, "tolerance": {"atol": 0.001, "rtol": 0.001}, "dynamic_axes_ranges": {"B": [1, 1024], "N": [1, 8192]}, "shape_constraints": ["N >= 1 (reduction axis must be non-empty so each row has a well-defined max and sum)", "B >= 1 (batch dimension must be non-empty)"], "p0_cases": [{"name": "p0_batch1_n128", "params": {}, "input_shapes": {"x": [1, 128]}, "output_shapes": {"y": [1, 128]}}, {"name": "p0_batch4_n2048", "params": {}, "input_shapes": {"x": [4, 2048]}, "output_shapes": {"y": [4, 2048]}}, {"name": "p0_batch32_n4096", "params": {}, "input_shapes": {"x": [32, 4096]}, "output_shapes": {"y": [32, 4096]}}], "perf_target": null},
 provenance:'基于历史产物重构的对话故事线 · 非原始逐条对话',
 userMessage:'帮我开发一个 FP16 Softmax 算子，输入是二维 Tensor [B,N]，沿最后一个维度进行归一化，要求数值稳定。\n\nB 支持 1–1024，N 支持 1–8192。\n\n重点验证三个输入：[1,128]、[4,2048]、[32,4096]。\n\n希望能够完成正确性验证和性能优化。',
 stages:[
 {stage:1,name:'SPEC',summary:'建立需求合同，确认功能、接口、动态 Shape 和验收条件。',detail:'输入输出 FP16 [B,N]；固定 axis=-1，使用 max-subtract 保证数值稳定。'},
 {stage:2,name:'EXPLORE',summary:'检索并分析 VF API、相关知识模式和硬件约束。',detail:'知识选择、VF API 适配和硬件约束共同构成设计依据。'},
 {stage:3,name:'DESIGN',summary:'采用单 Vector / VF 方案，确定精度、Tile、缓冲与分核设计。',detail:'混合精度、Single Tile、UB 双缓冲和行块分核是本次采用的设计集合。'},
 {stage:4,name:'DEVELOP',summary:'生成 PyPTO Kernel，并通过正确性验证。',detail:'三趟 max / exp+sum / div；行尾与列尾用例纳入验证。'},
 {stage:5,name:'OPTIMIZE',summary:'采用 Mask Lifting，完成最终性能比较。',detail:'复用循环外 Full Mask；Runtime if 尾寄存器尝试失败，由 Empty-tail-loop 替代。'}
 ],
 files:['SPEC.md','DESIGN.md','test_softmax.py','PERFORMANCE_REPORT.md'],
 get workflows(){return window.LINEAGE_DATA.nodes.filter(n=>n.type==='Workflow').sort((a,b)=>a.stage-b.stage)},
 get performance(){return window.EXPERIMENT_DATA.cases},
 get correctness(){return window.AUDIT_EVIDENCE.correctness},
 get opportunities(){return window.LINEAGE_VIEW.river.postRunOpportunities}
};
