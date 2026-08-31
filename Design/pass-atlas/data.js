window.ATLAS = {
 "passes": [
  {
   "order": 1,
   "name": "InlineFunctions",
   "snake": "inline_functions",
   "phase": "front",
   "role": "mech",
   "layer": null,
   "brief": "把 Inline 函数体原地展开到每个调用点",
   "detail": "遍历程序，找出所有 FunctionType::Inline 的函数，把函数体拼接进每一个调用点，然后把这些函数从 Program 里删掉。支持 Inline 调 Inline 的嵌套展开（迭代到不动点），并对 Inline→Inline 的调用图做环检测，有环直接抛 ValueError。展开时对内联进来的局部变量做 alpha 重命名，避免多个调用点之间撞名。",
   "watch": "它是流水线第一个 Pass，理由写在 pass_manager.py 的注释里：**这样下游任何 Pass 都不必再处理 Inline 函数或指向它的 Call**。这是典型的「用位置换复杂度」——把一类情况在最前面消灭掉，后面 46 个 Pass 就少一个分支。",
   "file": "src/ir/transforms/inline_functions_pass.cpp",
   "lines": 784,
   "factory": "Pass InlineFunctions() {\n  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {\n    // Collect inline functions\n    std::unordered_map<std::string, FunctionPtr> inline_fns;\n    for (const auto& [gvar, fn] : program->functions_) {\n      if (fn->func_type_ == FunctionType::Inline) {\n        INTERNAL_CHECK_SPAN(inline_fns.count(fn->name_) == 0, fn->span_)\n            << \"Duplicate FunctionType::Inline function name '\" << fn->name_ << \"' in program\";\n        inline_fns[fn->name_] = fn;\n      }\n    }\n\n    // Fast path: nothing to do\n    if (inline_fns.empty()) return program;\n\n    // Cycle detection\n    DetectInlineCycles(inline_fns);\n\n    // Iterate to fixpoint. Each iteration mutates every function (incl. Inline\n    // ones, so that Inline-calls-Inline expands too). The loop terminates after\n    // at most (inline_fns.size() + 1) iterations because each iteration either\n    // makes progress or hits the fixpoint.\n    std::unordered_map<std::string, FunctionPtr> current;\n    for (const auto& [gvar, fn] : program->functions_) {\n      current[fn->name_] = fn;\n    }\n\n    const size_t max_iters = inline_fns.size() + 1;\n    for (size_t iter = 0; iter < max_iters; ++iter) {\n      bool any_changed = false;\n\n      // Refresh inline_fns view to point at the *latest* bodies — important\n      // because a previous iteration may have inlined Inline-calls-Inline.\n      std::unordered_map<std::string, FunctionPtr> latest_inline;\n      for (const auto& [name, fn] : inline_fns) {\n        latest_inline[name] = current[name];\n      }\n\n      for (auto& [name, fn] : current) {\n        InlineCallsMutator mutator(latest_inline);\n        auto new_body = mutator.VisitStmt(fn->body_);\n        if (mutator.Changed()) {\n          auto updated = MutableCopy(fn);\n          updated->body_ = new_body;\n          fn = updated;\n          any_changed = true;\n        }\n      }\n\n      if (!any_changed) break;\n\n      INTERNAL_CHECK(iter + 1 < max_iters) << \"InlineFunctions did not reach a fixpoint within \" << max_iters\n                                           << \" iterations; this indicates a bug or an undetected cycle\";\n    }\n\n    // Drop inline functions and rebuild the program\n    std::vector<FunctionPtr> kept_functions;\n    for (const auto& [gvar, fn] : program->functions_) {\n      auto it = current.find(fn->name_);\n      INTERNAL_CHECK(it != current.end()) << \"Internal error: function '\" << fn->name_ << \"' missing\";\n      const auto& latest = it->second;\n      if (latest->func_type_ == FunctionType::Inline) continue;\n      kept_functions.push_back(latest);\n    }\n\n    return std::make_shared<Program>(kept_functions, program->name_, program->span_);\n  };\n\n  return CreateProgramPass(pass_func, \"InlineFunctions\", kInlineFunctionsProperties);\n}",
   "factoryRef": "src/ir/transforms/inline_functions_pass.cpp:710",
   "required": [],
   "produced": [
    "InlineFunctionsEliminated"
   ],
   "invalidated": [],
   "origin": {},
   "downstream": {
    "InlineFunctionsEliminated": []
   },
   "snippets": []
  },
  {
   "order": 2,
   "name": "UnrollLoops",
   "snake": "unroll_loops",
   "phase": "front",
   "role": "decide",
   "layer": "tiling",
   "brief": "编译期展开 ForKind::Unroll 循环",
   "detail": "对每个标了 Unroll 的 ForStmt，从 start/stop/step 取编译期常量算出 trip count，然后把循环体复制 trip_count 份。不允许 Unroll 循环带 iter_args（有跨迭代累加就不能简单复制）。trip count 超过 kMaxUnrollIterations 会抛错，并提示改用 pl.range()。",
   "watch": "它是切分层的第一个决策点：**展开与否直接改变后面所有 Pass 的输入规模**。展开后 tile 数量翻倍，MemoryReuse 的生命周期区间、InsertSync 的同步边数量全都跟着变。pypto#1242 记录过 pl.unroll 导致 K cache NaN 而 pl.range 不会。",
   "file": "src/ir/transforms/unroll_loops_pass.cpp",
   "lines": 198,
   "factory": "Pass UnrollLoops() { return CreateFunctionPass(TransformUnrollLoops, \"UnrollLoops\", kUnrollLoopsProperties); }",
   "factoryRef": "src/ir/transforms/unroll_loops_pass.cpp:194",
   "required": [],
   "produced": [
    "UnrollResolved"
   ],
   "invalidated": [],
   "origin": {},
   "downstream": {
    "UnrollResolved": []
   },
   "snippets": [
    {
     "file": "src/ir/transforms/unroll_loops_pass.cpp",
     "from": 83,
     "to": 108,
     "label": "trip count 与上限",
     "code": "StmtPtr UnrollForStmt(const ForStmtPtr& op) {\n  // Validate: no iter_args for unroll loops\n  INTERNAL_CHECK_SPAN(op->iter_args_.empty(), op->span_)\n      << \"Unroll loops cannot have iter_args (init_values)\";\n\n  // Extract compile-time constants for start/stop/step\n  int64_t start = GetConstIntValue(op->start_, \"start\");\n  int64_t stop = GetConstIntValue(op->stop_, \"stop\");\n  int64_t step = GetConstIntValue(op->step_, \"step\");\n  if (step == 0) {\n    throw pypto::ValueError(\"Unroll loop step cannot be zero\");\n  }\n\n  // Compute trip count and enforce max unroll limit\n  int64_t trip_count = 0;\n  if (step > 0 && start < stop) {\n    trip_count = (stop - start + step - 1) / step;\n  } else if (step < 0 && start > stop) {\n    trip_count = (start - stop + (-step) - 1) / (-step);\n  }\n  if (trip_count > kMaxUnrollIterations) {\n    throw pypto::ValueError(\"Unroll loop trip count \" + std::to_string(trip_count) +\n                            \" exceeds maximum allowed (\" + std::to_string(kMaxUnrollIterations) +\n                            \"). Reduce the loop range or use pl.range() instead\");\n  }\n"
    }
   ]
  },
  {
   "order": 3,
   "name": "CtrlFlowTransform",
   "snake": "ctrl_flow_transform",
   "phase": "front",
   "role": "mech",
   "layer": null,
   "brief": "把 break / continue 改写成结构化控制流",
   "detail": "把非结构化跳转（BreakStmt / ContinueStmt）改写成等价的 if-else + while 结构，产出 StructuredCtrlFlow 属性。",
   "watch": "放在 SSA 之前是必须的——SSA 的 phi 节点建立依赖支配关系，非结构化跳转会让支配关系失效。",
   "file": "src/ir/transforms/ctrl_flow_transform_pass.cpp",
   "lines": 842,
   "factory": "Pass CtrlFlowTransform() {\n  return CreateFunctionPass(TransformCtrlFlow, \"CtrlFlowTransform\", kCtrlFlowTransformProperties);\n}",
   "factoryRef": "src/ir/transforms/ctrl_flow_transform_pass.cpp:836",
   "required": [],
   "produced": [
    "StructuredCtrlFlow"
   ],
   "invalidated": [],
   "origin": {},
   "downstream": {
    "StructuredCtrlFlow": []
   },
   "snippets": []
  },
  {
   "order": 4,
   "name": "ConvertToSSA",
   "snake": "convert_to_ssa",
   "phase": "front",
   "role": "mech",
   "layer": null,
   "brief": "转成 SSA 形式：重命名、phi、iter_args",
   "detail": "把非 SSA 的 IR 转成静态单赋值形式：变量重命名保证每个名字只被赋值一次，控制流汇合处插 phi 节点，循环的跨迭代值变成 iter_args。产出 SSAForm。",
   "watch": "**SSAForm 是全流水线被消费最多的属性——21 个下游 Pass 依赖它**。而它会在 InitMemRef 处被 invalidate：一旦变量绑上物理 MemRef，「一个名字一次赋值」就不再成立了。所以内存相关的 Pass 全部排在 SSA 失效之后。",
   "file": "src/ir/transforms/convert_to_ssa_pass.cpp",
   "lines": 1230,
   "factory": "Pass ConvertToSSA() {\n  return CreateFunctionPass(TransformConvertToSSA, \"ConvertToSSA\", kConvertToSSAProperties);\n}",
   "factoryRef": "src/ir/transforms/convert_to_ssa_pass.cpp:1224",
   "required": [],
   "produced": [
    "SSAForm"
   ],
   "invalidated": [
    "NormalizedStmtStructure"
   ],
   "origin": {},
   "downstream": {
    "SSAForm": [
     "FlattenCallExpr",
     "OutlineHierarchyScopes",
     "OutlineIncoreScopes",
     "OutlineClusterScopes",
     "ConvertTensorToTileOps",
     "FlattenTileNdTo2D",
     "AutoTileMatmulL0",
     "CanonicalizeTileSlice",
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SplitVectorKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ]
   },
   "snippets": []
  },
  {
   "order": 5,
   "name": "Simplify",
   "snake": "simplify",
   "phase": "front",
   "role": "mech",
   "layer": null,
   "brief": "折叠算术、shape 表达式与标量常量",
   "detail": "用代数重写规则和区间分析折叠三类东西：算术表达式、类型注解里内嵌的 shape 表达式、标量常量绑定。比如把 `CHUNK_K: Scalar[INDEX] = 512` 的值传播到所有下游使用点。",
   "watch": "**它在流水线里跑两次**：第一次紧跟 ConvertToSSA（注释写明「跑在 SSA 之后，利用单定义性质」），第二次在分布式 Pass 之后收尾。pypto#1461 记录过它曾把承载语义的 output alias 赋值当成可折叠的 cast 而删掉，导致返回张量读回全零。",
   "file": "src/ir/transforms/simplify_pass.cpp",
   "lines": 846,
   "factory": "Pass Simplify() { return CreateProgramPass(TransformSimplifyProgram, \"Simplify\", kSimplifyProperties); }",
   "factoryRef": "src/ir/transforms/simplify_pass.cpp:841",
   "required": [],
   "produced": [],
   "invalidated": [],
   "origin": {},
   "downstream": {},
   "snippets": []
  },
  {
   "order": 6,
   "name": "NormalizeStmtStructure",
   "snake": "normalize_stmt_structure",
   "phase": "front",
   "role": "mech",
   "layer": null,
   "brief": "规范语句结构，消除冗余嵌套块",
   "detail": "把语句树规范化：去掉单子节点的 SeqStmts、拍平嵌套的 SeqStmts。产出 NormalizedStmtStructure。",
   "watch": "这个属性被 12 个 Pass 要求。多数变换 Pass 会在返回前自我规范化并重新产出它，所以流水线里只显式调用一次。",
   "file": "src/ir/transforms/normalize_stmt_structure_pass.cpp",
   "lines": 133,
   "factory": "Pass NormalizeStmtStructure() {\n  return CreateFunctionPass(ir::NormalizeStmtStructure, \"NormalizeStmtStructure\",\n                            kNormalizeStmtStructureProperties);\n}",
   "factoryRef": "src/ir/transforms/normalize_stmt_structure_pass.cpp:125",
   "required": [],
   "produced": [
    "NormalizedStmtStructure"
   ],
   "invalidated": [],
   "origin": {},
   "downstream": {
    "NormalizedStmtStructure": [
     "FlattenCallExpr",
     "ConvertTensorToTileOps",
     "FlattenTileNdTo2D",
     "AutoTileMatmulL0",
     "CanonicalizeTileSlice",
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "ExpandManualPhaseFence"
    ]
   },
   "snippets": []
  },
  {
   "order": 7,
   "name": "FlattenCallExpr",
   "snake": "flatten_call_expr",
   "phase": "front",
   "role": "mech",
   "layer": null,
   "brief": "把嵌套调用摊平成三地址形式",
   "detail": "把 `f(g(x))` 这样的嵌套调用表达式摊平成 `t = g(x); f(t)`，产出 NoNestedCalls。",
   "watch": "pypto#176 记录过 InsertSync 无法分析 FlattenCallExpr 造出来的 OpStmts 块内部的依赖——摊平本身没错，但它造出的新结构下游未必都认。",
   "file": "src/ir/transforms/flatten_call_expr_pass.cpp",
   "lines": 575,
   "factory": "Pass FlattenCallExpr() {\n  return CreateFunctionPass(TransformFlattenCallExpr, \"FlattenCallExpr\", kFlattenCallExprProperties);\n}",
   "factoryRef": "src/ir/transforms/flatten_call_expr_pass.cpp:569",
   "required": [
    "SSAForm",
    "NormalizedStmtStructure"
   ],
   "produced": [
    "SSAForm",
    "NoNestedCalls",
    "NormalizedStmtStructure"
   ],
   "invalidated": [],
   "origin": {
    "SSAForm": "ConvertToSSA",
    "NormalizedStmtStructure": "NormalizeStmtStructure"
   },
   "downstream": {
    "SSAForm": [
     "OutlineHierarchyScopes",
     "OutlineIncoreScopes",
     "OutlineClusterScopes",
     "ConvertTensorToTileOps",
     "FlattenTileNdTo2D",
     "AutoTileMatmulL0",
     "CanonicalizeTileSlice",
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SplitVectorKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "NoNestedCalls": [
     "ExpandManualPhaseFence"
    ],
    "NormalizedStmtStructure": [
     "ConvertTensorToTileOps",
     "FlattenTileNdTo2D",
     "AutoTileMatmulL0",
     "CanonicalizeTileSlice",
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "ExpandManualPhaseFence"
    ]
   },
   "snippets": []
  },
  {
   "order": 8,
   "name": "OutlineHierarchyScopes",
   "snake": "outline_hierarchy_scopes",
   "phase": "scope",
   "role": "mech",
   "layer": null,
   "brief": "把 Hierarchy scope 外提成带 level/role 的函数",
   "detail": "把 Hierarchy scope 外提成独立函数，并在函数上带 level / role 元数据。同时产出 OrchestrationReferencesResolved：保证 Orchestration 函数里每个非 builtin 的 Call 都指向 Program 里真实存在的 Function。",
   "watch": "三个 Outline Pass 的顺序是 Hierarchy → InCore → Cluster，从外层往内层剥。",
   "file": "src/ir/transforms/outline_hierarchy_scopes_pass.cpp",
   "lines": 145,
   "factory": "Pass OutlineHierarchyScopes() {\n  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {\n    std::vector<FunctionPtr> new_functions;\n    std::vector<FunctionPtr> all_outlined_functions;\n\n    // Program-wide set of outlined function names, seeded with the existing\n    // function names, shared across each function's ScopeOutliner so duplicate\n    // `name_hint` values across functions auto-disambiguate instead of colliding\n    // at Program construction (#1711).\n    auto reserved_func_names = std::make_shared<std::unordered_set<std::string>>();\n    for (const auto& [gvar, func] : program->functions_) {\n      reserved_func_names->insert(func->name_);\n    }\n\n    for (const auto& [gvar, func] : program->functions_) {\n      // Only process Opaque functions (hierarchy scopes appear in user-written programs)\n      if (func->func_type_ != FunctionType::Opaque) {\n        new_functions.push_back(func);\n        continue;\n      }\n\n      // Build symbol table for this function\n      outline_utils::VarCollector type_collector;\n      for (const auto& var : func->params_) {\n        type_collector.var_types[var.get()] = var->GetType();\n        type_collector.var_objects[var.get()] = var;\n        type_collector.known_names.insert(var->name_hint_);\n      }\n      type_collector.VisitStmt(func->body_);\n\n      // Outline Hierarchy scopes in this function\n      outline_utils::ScopeOutliner outliner(func->name_, type_collector.var_types, type_collector.var_objects,\n                                            type_collector.known_names, ScopeKind::Hierarchy,\n                                            FunctionType::Opaque, \"_hierarchy_\", /*program=*/nullptr,\n                                            reserved_func_names);\n      auto new_body = outliner.VisitStmt(func->body_);\n\n      // Preserve parent function type (don't promote — hierarchy is orthogonal to FunctionType)\n      auto new_func = MutableCopy(func);\n      new_func->body_ = new_body;\n      new_functions.push_back(new_func);\n\n      const auto& outlined = outliner.GetOutlinedFunctions();\n      all_outlined_functions.insert(all_outlined_functions.end(), outlined.begin(), outlined.end());\n    }\n\n    // Add all outlined functions before the originals\n    all_outlined_functions.insert(all_outlined_functions.end(), new_functions.begin(), new_functions.end());\n\n    // Create new program with all functions\n    return std::make_shared<Program>(all_outlined_functions, program->name_, program->span_);\n  };\n\n  return CreateProgramPass(pass_func, \"OutlineHierarchyScopes\", kOutlineHierarchyScopesProperties);\n}",
   "factoryRef": "src/ir/transforms/outline_hierarchy_scopes_pass.cpp:54",
   "required": [
    "SSAForm"
   ],
   "produced": [
    "SSAForm",
    "HierarchyOutlined",
    "OrchestrationReferencesResolved"
   ],
   "invalidated": [],
   "origin": {
    "SSAForm": "ConvertToSSA"
   },
   "downstream": {
    "SSAForm": [
     "OutlineIncoreScopes",
     "OutlineClusterScopes",
     "ConvertTensorToTileOps",
     "FlattenTileNdTo2D",
     "AutoTileMatmulL0",
     "CanonicalizeTileSlice",
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SplitVectorKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "HierarchyOutlined": [],
    "OrchestrationReferencesResolved": []
   },
   "snippets": []
  },
  {
   "order": 9,
   "name": "OutlineIncoreScopes",
   "snake": "outline_incore_scopes",
   "phase": "scope",
   "role": "mech",
   "layer": null,
   "brief": "把 InCore scope 外提成独立函数",
   "detail": "把 InCore scope 外提成独立函数，产出 SplitIncoreOrch（编排与核内计算分离）。同时**打开 AivSplitValid 的验证窗口**：它保留 InCore 函数里的 SplitAivScopeStmt 区域，让结构验证器从这里一直能跑到 LowerAutoVectorSplit 把节点擦掉为止。",
   "watch": "SplitIncoreOrch 是整条流水线后半段的地基属性——**从这里往后 20 多个 Pass 都要求它**。",
   "file": "src/ir/transforms/outline_incore_scopes_pass.cpp",
   "lines": 334,
   "factory": "Pass OutlineIncoreScopes() {\n  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {\n    std::vector<FunctionPtr> new_functions;\n    std::vector<FunctionPtr> all_outlined_functions;\n\n    // Program-wide set of outlined function names, seeded with the existing\n    // function names. Shared across each function's ScopeOutliner so that two\n    // functions outlining InCore scopes with the same `name_hint` (e.g. a\n    // shared `@pl.jit.inline` helper reused across child kernels) get\n    // suffix-disambiguated instead of colliding at Program construction (#1711).\n    auto reserved_func_names = std::make_shared<std::unordered_set<std::string>>();\n    for (const auto& [gvar, func] : program->functions_) {\n      reserved_func_names->insert(func->name_);\n    }\n\n    for (const auto& [gvar, func] : program->functions_) {\n      // Process Opaque and Orchestration functions; other function types\n      // (InCore/Group/Spmd) are already outlined or not expected to carry\n      // InCore scopes.\n      if (func->func_type_ != FunctionType::Opaque && func->func_type_ != FunctionType::Orchestration) {\n        new_functions.push_back(func);\n        continue;\n      }\n\n      // An Opaque body that carries an InCore scope is about to be promoted to\n      // Orchestration (see below). Fold its param dyn-dim reads first, so the\n      // outliner sees the same body the parser hands an already-Orchestration\n      // function — one runtime extent, one IR name, on both paths.\n      //\n      // The probe is not speculative: ScopeOutliner::VisitScopeKind outlines\n      // every InCoreScopeStmt it reaches unconditionally, so \"body has an InCore\n      // scope\" and the promotion condition below (`!outlined.empty()`) are the\n      // same predicate. An Opaque function that stays Opaque is never folded —\n      // it may be a callee (OutlineHierarchyScopes mints Opaque callees), whose\n      // symbol placeholder is not the caller's and may be reached with a\n      // statically-shaped actual.\n      StmtPtr source_body = func->body_;\n      if (func->func_type_ == FunctionType::Opaque) {\n        HasInCoreScope probe;\n        probe.VisitStmt(source_body);\n        if (probe.found_) source_body = FoldParamDimReads(func->params_, source_body);\n      }\n\n      // Build symbol table for this function\n      outline_utils::VarCollector type_collector;\n      for (const auto& var : func->params_) {\n        type_collector.var_types[var.get()] = var->GetType();\n        type_collector.var_objects[var.get()] = var;\n        type_collector.known_names.insert(var->name_hint_);\n      }\n      type_collector.VisitStmt(source_body);\n\n      // Outline InCore scopes in this function\n      outline_utils::ScopeOutliner outliner(\n          func->name_, type_collector.var_types, type_collector.var_objects, type_collector.known_names,\n          ScopeKind::InCore, FunctionType::InCore, \"_incore_\", /*program=*/nullptr, reserved_func_names);\n      auto new_body = outliner.VisitStmt(source_body);\n\n      // Create new function with transformed body.\n      // If any InCore scopes were outlined, promote Opaque -> Orchestration.\n      const auto& outlined = outliner.GetOutlinedFunctions();\n      FunctionType new_func_type = outlined.empty() ? func->func_type_ : FunctionType::Orchestration;\n      auto new_func = MutableCopy(func);\n      new_func->body_ = new_body;\n      new_func->func_type_ = new_func_type;\n      if (new_func_type == FunctionType::Orchestration) {\n        new_func->level_ = FunctionTypeToLevel(new_func_type);\n        new_func->role_ = Role::Orchestrator;\n      }\n      new_functions.push_back(new_func);",
   "factoryRef": "src/ir/transforms/outline_incore_scopes_pass.cpp:185",
   "required": [
    "SSAForm"
   ],
   "produced": [
    "SSAForm",
    "SplitIncoreOrch",
    "AivSplitValid"
   ],
   "invalidated": [],
   "origin": {
    "SSAForm": "ConvertToSSA"
   },
   "downstream": {
    "SSAForm": [
     "OutlineClusterScopes",
     "ConvertTensorToTileOps",
     "FlattenTileNdTo2D",
     "AutoTileMatmulL0",
     "CanonicalizeTileSlice",
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SplitVectorKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "SplitIncoreOrch": [
     "ConvertTensorToTileOps",
     "OptimizeOrchTensors",
     "AutoTileMatmulL0",
     "CanonicalizeTileSlice",
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "StampTfreeSplit",
     "NormalizeReturnOrder",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape",
     "DeriveCallDirections",
     "AutoDeriveTaskDependencies",
     "MaterializeRuntimeScopes",
     "InsertCommFence"
    ],
    "AivSplitValid": []
   },
   "snippets": []
  },
  {
   "order": 10,
   "name": "OutlineClusterScopes",
   "snake": "outline_cluster_scopes",
   "phase": "scope",
   "role": "mech",
   "layer": null,
   "brief": "把 Cluster / Spmd scope 外提成 Group / Spmd 函数",
   "detail": "把 Cluster scope 外提成 Group 函数，把独立的 Spmd scope 外提成 Spmd 函数，产出 ClusterOutlined。",
   "watch": "",
   "file": "src/ir/transforms/outline_cluster_scopes_pass.cpp",
   "lines": 388,
   "factory": "Pass OutlineClusterScopes() {\n  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {\n    std::vector<FunctionPtr> new_functions;\n    std::vector<FunctionPtr> all_outlined_functions;\n\n    // Program-wide set of outlined function names, seeded with the existing\n    // function names and shared across every ScopeOutliner (both the Cluster and\n    // Spmd passes, across all functions) so duplicate `name_hint` values produced\n    // from reused helpers auto-disambiguate instead of colliding at Program\n    // construction (#1711).\n    auto reserved_func_names = std::make_shared<std::unordered_set<std::string>>();\n    for (const auto& [gvar, func] : program->functions_) {\n      reserved_func_names->insert(func->name_);\n    }\n\n    for (const auto& [gvar, func] : program->functions_) {\n      // Only process Opaque and Orchestration functions (Group functions are already outlined)\n      if (func->func_type_ != FunctionType::Opaque && func->func_type_ != FunctionType::Orchestration) {\n        new_functions.push_back(func);\n        continue;\n      }\n\n      // First pass: outline Cluster scopes\n      outline_utils::VarCollector type_collector;\n      for (const auto& var : func->params_) {\n        type_collector.var_types[var.get()] = var->GetType();\n        type_collector.var_objects[var.get()] = var;\n        type_collector.known_names.insert(var->name_hint_);\n      }\n      type_collector.VisitStmt(func->body_);\n\n      outline_utils::ScopeOutliner cluster_outliner(\n          func->name_, type_collector.var_types, type_collector.var_objects, type_collector.known_names,\n          ScopeKind::Cluster, FunctionType::Group, \"_cluster_\", program, reserved_func_names);\n      auto body_after_cluster = cluster_outliner.VisitStmt(func->body_);\n\n      // Unwrap a ``pl.spmd`` nested inside each freshly outlined Group and move\n      // its launch spec onto the dispatch the outliner just synthesised in THIS\n      // body. Done here rather than in a trailing program-wide sweep because\n      // that is the only point where the dispatch is still reachable.\n      auto cluster_outlined = cluster_outliner.GetOutlinedFunctions();\n      std::unordered_map<std::string, SpmdLaunchSpec> group_launch_specs;\n      for (auto& outlined : cluster_outlined) {\n        if (!outlined || outlined->func_type_ != FunctionType::Group) continue;\n        SpmdLaunchSpec spec;\n        outlined = UnwrapNestedSpmd(outlined, &spec);\n        if (!spec.core_num) continue;\n        spec.group = outlined;\n        // Snapshot what the Group binds once, so the per-dispatch scope check\n        // below is a hash lookup rather than a re-walk of the body.\n        for (const auto& param : outlined->params_) {\n          if (param) spec.callee_bound.insert(param.get());\n        }\n        outline_utils::VarDefUseCollector group_defs;\n        if (outlined->body_) group_defs.VisitStmt(outlined->body_);\n        spec.callee_bound.insert(group_defs.var_defs.begin(), group_defs.var_defs.end());\n        group_launch_specs.emplace(outlined->name_, std::move(spec));\n      }\n      if (!group_launch_specs.empty()) {\n        LaunchSpecStamper stamper(group_launch_specs);\n        body_after_cluster = stamper.VisitStmt(body_after_cluster);\n      }\n\n      all_outlined_functions.insert(all_outlined_functions.end(), cluster_outlined.begin(),\n                                    cluster_outlined.end());\n\n      // Second pass: outline standalone Spmd scopes (those not inside a Cluster)\n      outline_utils::VarCollector refreshed_collector;\n      for (const auto& var : func->params_) {\n        refreshed_collector.var_types[var.get()] = var->GetType();",
   "factoryRef": "src/ir/transforms/outline_cluster_scopes_pass.cpp:239",
   "required": [
    "SSAForm"
   ],
   "produced": [
    "SSAForm",
    "ClusterOutlined"
   ],
   "invalidated": [],
   "origin": {
    "SSAForm": "ConvertToSSA"
   },
   "downstream": {
    "SSAForm": [
     "ConvertTensorToTileOps",
     "FlattenTileNdTo2D",
     "AutoTileMatmulL0",
     "CanonicalizeTileSlice",
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SplitVectorKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "ClusterOutlined": []
   },
   "snippets": []
  },
  {
   "order": 11,
   "name": "ConvertTensorToTileOps",
   "snake": "convert_tensor_to_tile_ops",
   "phase": "tile",
   "role": "mech",
   "layer": null,
   "brief": "InCore 里 tensor 算子转 tile 算子",
   "detail": "在 InCore 函数里把 tensor 算子转成 tile 算子，同时更新 orchestration 侧的调用点。产出 IncoreTileOps。",
   "watch": "它**重开** AivSplitValid 窗口（先 invalidate 再 produce）。原因写在属性声明里：OutlineIncoreScopes 建立该属性时，AIV 切分边界还是 tensor.aiv_shard / tensor.aic_gather，而 TensorType 不带内存空间，验证器的边界内存契约检查只能跳过；这个 Pass 把它们改写成 tile 形式并附上声明的边界内存，正好是那项检查要看的东西，所以强制再验一次。",
   "file": "src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp",
   "lines": 2284,
   "factory": "Pass ConvertTensorToTileOps() {\n  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {\n    // Phase 1: Transform InCore functions\n    std::unordered_map<std::string, size_t> incore_added_outputs;\n    std::unordered_map<std::string, FunctionPtr> transformed_incore_funcs;\n    std::vector<FunctionPtr> functions_phase1;\n\n    for (const auto& [gvar, func] : program->functions_) {\n      if (func->func_type_ == FunctionType::InCore) {\n        auto result = TransformIncoreFunction(func);\n        incore_added_outputs[func->name_] = result.num_added_outputs;\n        transformed_incore_funcs[func->name_] = result.func;\n        functions_phase1.push_back(result.func);\n      } else {\n        functions_phase1.push_back(func);\n      }\n    }\n\n    // Phase 2a: Propagate added output params through Spmd/Group wrappers so\n    // they remain transparent 1:1 forwarders of their params to the inner\n    // call (an invariant relied on by orchestration codegen).\n    std::unordered_map<std::string, size_t> wrapper_added_outputs;\n    std::unordered_map<std::string, FunctionPtr> transformed_wrapper_funcs;\n    std::vector<FunctionPtr> functions_phase2a;\n    functions_phase2a.reserve(functions_phase1.size());\n    for (const auto& func : functions_phase1) {\n      if (IsWrapperType(func->func_type_)) {\n        auto result = PropagateOutputsThroughWrapper(func, incore_added_outputs, transformed_incore_funcs);\n        functions_phase2a.push_back(result.func);\n        if (result.num_added_outputs > 0) {\n          wrapper_added_outputs[func->name_] = result.num_added_outputs;\n          transformed_wrapper_funcs[func->name_] = result.func;\n        }\n      } else {\n        functions_phase2a.push_back(func);\n      }\n    }\n\n    // Phase 2b: Update call sites in orchestration/opaque functions. The\n    // callee map covers both transformed InCore functions and wrappers that\n    // absorbed their output params.\n    std::unordered_map<std::string, size_t> all_added_outputs = incore_added_outputs;\n    all_added_outputs.insert(wrapper_added_outputs.begin(), wrapper_added_outputs.end());\n    std::unordered_map<std::string, FunctionPtr> all_transformed_funcs = transformed_incore_funcs;\n    all_transformed_funcs.insert(transformed_wrapper_funcs.begin(), transformed_wrapper_funcs.end());\n\n    std::vector<FunctionPtr> functions_phase2b;\n    functions_phase2b.reserve(functions_phase2a.size());\n    for (const auto& func : functions_phase2a) {\n      // Skip InCore (rewritten in Phase 1) and every Spmd/Group (rewritten in\n      // Phase 2a when forwarding a transformed InCore; otherwise nothing to\n      // forward because ForwardedCallFinder rejects callees that gained zero\n      // Out params). The postcondition check in PropagateOutputsThroughWrapper\n      // turns any finder/mutator mismatch into a hard INTERNAL_CHECK rather\n      // than a silent mis-rewrite.\n      if (func->func_type_ == FunctionType::InCore || func->func_type_ == FunctionType::Spmd ||\n          func->func_type_ == FunctionType::Group) {\n        functions_phase2b.push_back(func);\n      } else {\n        functions_phase2b.push_back(UpdateCallSites(func, all_added_outputs, all_transformed_funcs));\n      }\n    }\n\n    // Phase 3: Propagate Function::param_directions_ along the call chain.\n    //\n    // When the user writes inline `pl.at(...)` blocks, OutlineHierarchyScopes\n    // extracts them into a host_orch → chip_orch → incore chain. The outlined\n    // chip_orch has no direction info on its own parameters yet. Phase 1 has\n    // already marked the InCore's tile-written params as Out/InOut; if\n    // chip_orch(a, b, f) forwards its own `f` to that InCore, chip_orch's",
   "factoryRef": "src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp:2054",
   "required": [
    "SSAForm",
    "SplitIncoreOrch",
    "NormalizedStmtStructure"
   ],
   "produced": [
    "SSAForm",
    "IncoreTileOps",
    "NormalizedStmtStructure",
    "AivSplitValid"
   ],
   "invalidated": [
    "AivSplitValid"
   ],
   "origin": {
    "SSAForm": "ConvertToSSA",
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "NormalizedStmtStructure": "NormalizeStmtStructure"
   },
   "downstream": {
    "SSAForm": [
     "FlattenTileNdTo2D",
     "AutoTileMatmulL0",
     "CanonicalizeTileSlice",
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SplitVectorKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "IncoreTileOps": [
     "OptimizeOrchTensors",
     "FlattenTileNdTo2D",
     "AutoTileMatmulL0",
     "CanonicalizeTileSlice",
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "NormalizeReturnOrder",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "NormalizedStmtStructure": [
     "FlattenTileNdTo2D",
     "AutoTileMatmulL0",
     "CanonicalizeTileSlice",
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "ExpandManualPhaseFence"
    ],
    "AivSplitValid": []
   },
   "snippets": []
  },
  {
   "order": 12,
   "name": "OptimizeOrchTensors",
   "snake": "optimize_orch_tensors",
   "phase": "tile",
   "role": "mech",
   "layer": null,
   "brief": "消除编排层冗余张量分配",
   "detail": "优化 orchestration 与 InCore 之间的 tensor buffer 使用：消除冗余分配、改善数据流。",
   "watch": "pypto#1444 记录过它的 out-window 外置丢掉了父 SSA 重绑，导致依赖边缺失、indexer topk 竞争。",
   "file": "src/ir/transforms/optimize_orch_tensors_pass.cpp",
   "lines": 1631,
   "factory": "Pass OptimizeOrchTensors() {\n  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {\n    // Collect InCore function names\n    std::unordered_set<std::string> incore_names;\n    for (const auto& [gvar, func] : program->functions_) {\n      if (func->func_type_ == FunctionType::InCore) {\n        incore_names.insert(func->name_);\n      }\n    }\n\n    // Pattern 1: Iter-arg reuse (may remove Out params)\n    auto p1 = IterArgReuseOptimizer().Run(program, incore_names);\n\n    // Pattern 2: Assemble parent strides (sees Pattern 1 results)\n    auto p2 = AssembleParentStridesOptimizer().Run(p1, incore_names);\n\n    // Pattern 3: Assemble-loop rewrite (sees Pattern 2 results)\n    auto p3 = AssembleLoopRewriter().Run(p2, incore_names);\n\n    // Pattern 4: Slice input strides (propagate parent strides to In params)\n    auto p4 = SliceInputStridesOptimizer().Run(p3, incore_names);\n\n    // Optional Pattern 5 module: default off unless a kernel explicitly opts in.\n    if (!window_externalization::HasWindowizeEnabledFunction(p4)) return p4;\n    return window_externalization::ApplyWindowExternalization(p4);\n  };\n\n  return CreateProgramPass(pass_func, \"OptimizeOrchTensors\", kOptimizeOrchTensorsProperties);\n}",
   "factoryRef": "src/ir/transforms/optimize_orch_tensors_pass.cpp:1599",
   "required": [
    "SplitIncoreOrch",
    "IncoreTileOps"
   ],
   "produced": [
    "SplitIncoreOrch",
    "IncoreTileOps"
   ],
   "invalidated": [],
   "origin": {
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "IncoreTileOps": "ConvertTensorToTileOps"
   },
   "downstream": {
    "SplitIncoreOrch": [
     "AutoTileMatmulL0",
     "CanonicalizeTileSlice",
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "StampTfreeSplit",
     "NormalizeReturnOrder",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape",
     "DeriveCallDirections",
     "AutoDeriveTaskDependencies",
     "MaterializeRuntimeScopes",
     "InsertCommFence"
    ],
    "IncoreTileOps": [
     "FlattenTileNdTo2D",
     "AutoTileMatmulL0",
     "CanonicalizeTileSlice",
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "NormalizeReturnOrder",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ]
   },
   "snippets": []
  },
  {
   "order": 13,
   "name": "LowerCompositeOps",
   "snake": "lower_composite_ops",
   "phase": "tile",
   "role": "mech",
   "layer": null,
   "brief": "复合算子拆成原语组合",
   "detail": "把复合 tile / 分布式算子拆成原语组合。今天覆盖 tile.sin / tile.cos（Cody-Waite 范围规约 + 9 次 Horner 多项式）和显式 signal 的 InCore pld.tensor.allreduce；host 级 allreduce 跳过，留给后面的 LowerHostTensorCollectives。新增复合算子只需往文件内的分派表加一条规则。",
   "watch": "属性声明是空的，注释解释得很清楚：**它在既有算子词汇内改写，既不建立也不破坏任何 IRProperty**。这是判断一个 Pass 是不是「结构性」的好标准。",
   "file": "src/ir/transforms/lower_composite_ops_pass.cpp",
   "lines": 2398,
   "factory": "Pass LowerCompositeOps() {\n  return CreateFunctionPass(TransformLowerCompositeOps, \"LowerCompositeOps\", kLowerCompositeOpsProperties);\n}",
   "factoryRef": "src/ir/transforms/lower_composite_ops_pass.cpp:2391",
   "required": [],
   "produced": [],
   "invalidated": [],
   "origin": {},
   "downstream": {},
   "snippets": []
  },
  {
   "order": 14,
   "name": "FlattenTileNdTo2D",
   "snake": "flatten_tile_nd_to_2d",
   "phase": "tile",
   "role": "mech",
   "layer": null,
   "brief": "3D+ tile 操作摊平成 2D",
   "detail": "把 InCore 里 3D 及以上的 tile 操作摊平成 2D：除最后一维外全部合并。产出 TileOps2D。",
   "watch": "TileOps2D 是硬件契约的体现——cube/vector 单元只认 2D tile。后面 10 多个 Pass 都要求它。",
   "file": "src/ir/transforms/flatten_tile_nd_to_2d/",
   "lines": 2844,
   "factory": "// 工厂在 src/ir/transforms/flatten_tile_nd_to_2d/ 子目录内拆分实现",
   "factoryRef": "src/ir/transforms/flatten_tile_nd_to_2d/",
   "required": [
    "SSAForm",
    "IncoreTileOps",
    "NormalizedStmtStructure"
   ],
   "produced": [
    "SSAForm",
    "TileOps2D",
    "NormalizedStmtStructure"
   ],
   "invalidated": [],
   "origin": {
    "SSAForm": "ConvertToSSA",
    "IncoreTileOps": "ConvertTensorToTileOps",
    "NormalizedStmtStructure": "NormalizeStmtStructure"
   },
   "downstream": {
    "SSAForm": [
     "AutoTileMatmulL0",
     "CanonicalizeTileSlice",
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SplitVectorKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "TileOps2D": [
     "AutoTileMatmulL0",
     "CanonicalizeTileSlice",
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "NormalizedStmtStructure": [
     "AutoTileMatmulL0",
     "CanonicalizeTileSlice",
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "ExpandManualPhaseFence"
    ]
   },
   "snippets": []
  },
  {
   "order": 15,
   "name": "LegalizeTileCast",
   "snake": "legalize_tile_cast",
   "phase": "tile",
   "role": "mech",
   "layer": null,
   "brief": "把 ISA 不支持的 cast 对展开成最短原生链",
   "detail": "当前 pto.tcvt profile 一条指令做不到的 (src, dst) cast 对，展开成最短的原生 cast 链。比如 A5 上 INT32→FP16 展开成 INT32→FP32→FP16。",
   "watch": "**位置有讲究**：pass_manager.py 的注释写明它必须排在 AutoTileMatmulL0 之前，因为后者可能把已经原生的 f32→bf16/f16 折进 FIXPIPE，折完就看不见了。「最短链」由 ISA 能力表唯一确定，所以仍算机械改写。",
   "file": "src/ir/transforms/legalize_tile_cast_pass.cpp",
   "lines": 333,
   "factory": "Pass LegalizeTileCast() {\n  return CreateFunctionPass(TransformLegalizeTileCast, \"LegalizeTileCast\", kLegalizeTileCastProperties);\n}",
   "factoryRef": "src/ir/transforms/legalize_tile_cast_pass.cpp:326",
   "required": [],
   "produced": [],
   "invalidated": [],
   "origin": {},
   "downstream": {},
   "snippets": []
  },
  {
   "order": 16,
   "name": "AutoTileMatmulL0",
   "snake": "auto_tile_matmul_l0",
   "phase": "tile",
   "role": "decide",
   "layer": "tiling",
   "brief": "为 matmul 选 L0 tile 形状 (m,n,k) 并改写成 K 循环",
   "detail": "对每个静态 2D 的 tile.matmul / matmul_acc / matmul_bias，从当前 BackendHandler 的 L0 容量出发，经 utils::ChooseL0Tile 选一组 (m, n, k)，把调用改写成 K 循环。右操作数 B 必须 Mat 常驻；左操作数 A 可以是 Mat（QK 模式）也可以是 Vec（融合注意力的 score·V / PV 模式，softmax 输出以 Vec 形态跨越 cube↔vector 边界）。K 循环被标成 ForKind::Pipeline 且 pipeline_stages=2，交给下游 LowerPipelineLoops 生成 2 级 ping-pong。",
   "watch": "**这是流水线里唯一带完整代价模型的决策 Pass**：ChooseL0Tile 输入包含 L0A/L0B/L0C 容量、各级带宽 bw_a/bw_b/bw_drain、drain 的固定与逐行周期、MAD 头部周期、fractal 对齐——一个 roofline 求解器。选不出来时不硬来，而是发 perf hint 并 `left untouched`。在树已有 8 个理由码 PH-AT-003/005/006/007/008/009/010/011，**全部是「我为什么没帮你切」**。",
   "file": "src/ir/transforms/auto_tile_matmul_l0_pass.cpp",
   "lines": 3162,
   "factory": "Pass AutoTileMatmulL0() {\n  auto run = [](const ProgramPtr& program) -> ProgramPtr {\n    if (!program) return program;\n    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;\n    bool any_change = false;\n    std::vector<Diagnostic> hints;\n    for (const auto& [gvar, func] : program->functions_) {\n      auto new_func = TransformFunction(func, hints);\n      if (new_func != func) any_change = true;\n      new_functions.emplace(gvar, new_func);\n    }\n    if (!hints.empty()) EmitDiagnostics(hints, kPassName);\n    if (!any_change) return program;\n    auto new_program = MutableCopy(program);\n    new_program->functions_ = std::move(new_functions);\n    return new_program;\n  };\n  return CreateProgramPass(run, kPassName, kAutoTileMatmulL0Properties);\n}",
   "factoryRef": "src/ir/transforms/auto_tile_matmul_l0_pass.cpp:3139",
   "required": [
    "SSAForm",
    "SplitIncoreOrch",
    "IncoreTileOps",
    "TileOps2D",
    "NormalizedStmtStructure"
   ],
   "produced": [
    "SSAForm",
    "SplitIncoreOrch",
    "IncoreTileOps",
    "TileOps2D",
    "NormalizedStmtStructure"
   ],
   "invalidated": [],
   "origin": {
    "SSAForm": "ConvertToSSA",
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "IncoreTileOps": "ConvertTensorToTileOps",
    "TileOps2D": "FlattenTileNdTo2D",
    "NormalizedStmtStructure": "NormalizeStmtStructure"
   },
   "downstream": {
    "SSAForm": [
     "CanonicalizeTileSlice",
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SplitVectorKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "SplitIncoreOrch": [
     "CanonicalizeTileSlice",
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "StampTfreeSplit",
     "NormalizeReturnOrder",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape",
     "DeriveCallDirections",
     "AutoDeriveTaskDependencies",
     "MaterializeRuntimeScopes",
     "InsertCommFence"
    ],
    "IncoreTileOps": [
     "CanonicalizeTileSlice",
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "NormalizeReturnOrder",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "TileOps2D": [
     "CanonicalizeTileSlice",
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "NormalizedStmtStructure": [
     "CanonicalizeTileSlice",
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "ExpandManualPhaseFence"
    ]
   },
   "snippets": [
    {
     "file": "src/ir/transforms/auto_tile_matmul_l0_pass.cpp",
     "from": 927,
     "to": 949,
     "label": "把 backend 容量与 roofline 代价模型灌进 ChooseL0Tile",
     "code": "cfg.M = static_cast<int>(M);\ncfg.N = static_cast<int>(N);\ncfg.K = static_cast<int>(K);\ncfg.l0a_bytes = handler->GetL0aCapacityBytes();\ncfg.l0b_bytes = handler->GetL0bCapacityBytes();\ncfg.l0c_bytes = handler->GetL0cCapacityBytes();\nconst auto cost_model = handler->GetL0CostModel();\ncfg.bw_a = cost_model.bw_l0a;\ncfg.bw_b = cost_model.bw_l0b;\ncfg.bw_drain = cost_model.bw_drain;\ncfg.drain_fixed_cycles = cost_model.drain_fixed_cycles;\ncfg.drain_row_cycles = cost_model.drain_row_cycles;\ncfg.drain_penalty_cycles = cost_model.drain_penalty_cycles;\ncfg.drain_c0_bytes = cost_model.drain_c0_bytes;\ncfg.mad_head = cost_model.mad_head_cycles;\ncfg.mad_k_fractal_bytes = cost_model.mad_k_fractal_bytes;\ncfg.mad_fp32_passes = cost_model.mad_fp32_passes;\ncfg.bytes_a = bytes_a;\ncfg.bytes_b = bytes_b;\ncfg.bytes_c = bytes_c;\ncfg.align_m = handler->GetL0FractalAlignment();\ncfg.align_n = handler->GetL0FractalAlignment();\ncfg.align_k = handler->GetL0FractalAlignment();"
    },
    {
     "file": "src/ir/transforms/auto_tile_matmul_l0_pass.cpp",
     "from": 1080,
     "to": 1089,
     "label": "放弃切分时发 perf hint，而不是硬来",
     "code": "if (K % cfg.align_k != 0) {\n  hints.emplace_back(DiagnosticSeverity::PerfHint, kPassName, 0, \"PH-AT-007\",\n                     op_name + \": K=\" + std::to_string(K) + \" is not a multiple of the cube fractal \" +\n                         std::to_string(cfg.align_k) +\n                         \" — non-16-aligned K is unsupported; left untouched.\",\n                     assign->span_);\n  return std::nullopt;\n}\n\nutils::L0TileResult res;"
    }
   ]
  },
  {
   "order": 17,
   "name": "CanonicalizeTileSlice",
   "snake": "canonicalize_tile_slice",
   "phase": "tile",
   "role": "mech",
   "layer": null,
   "brief": "tile.slice 下降成规范的 tile.extract",
   "detail": "把 tile.slice 下降成规范的 tile.extract 形式，让所有搬运统一走 pto.textract。Mat 常驻的 slice 折进 matmul / tile.extract 的消费者；Vec 上那些惰性物化会破坏源数据的 slice 则强制物化。",
   "watch": "pypto#2010 记录过 pl.tile.slice 在多行 tile 上静默返回错数据——它把一个稠密 MemRef 编码在自己仍存活的源上。",
   "file": "src/ir/transforms/canonicalize_tile_slice_pass.cpp",
   "lines": 534,
   "factory": "Pass CanonicalizeTileSlice() {\n  auto pass_func = [](const FunctionPtr& func) -> FunctionPtr {\n    if (!func || !func->body_) return func;\n    if (!IsInCoreType(func->func_type_)) return func;\n\n    // Phase 1 — index every canonical tile.slice.\n    SliceCollector collector;\n    collector.VisitStmt(func->body_);\n    if (collector.slices.empty()) return func;\n\n    // Phase 2 — fold each slice into its tile.extract / matmul / col_expand_mul\n    // consumers.\n    CanonicalizeMutator mutator(collector.slices);\n    auto new_body = mutator.VisitStmt(func->body_);\n\n    // Phase 3 — drop the slice defs that no longer have any use.  A chained\n    // slice (a slice of a slice) only becomes dead once the slice that consumes\n    // it is dropped, so iterate to a fixpoint — bounded by the slice count,\n    // since every non-terminating iteration drops at least one statement.  A\n    // slice still used at the end had a consumer this pass does not\n    // canonicalize; it is left intact (no regression versus the pre-pass IR).\n    for (size_t round = 0; round <= collector.slices.size(); ++round) {\n      VarUseCollector uses;\n      uses.VisitStmt(new_body);\n      std::unordered_set<const Var*> dead;\n      for (const auto& [slice_var, info] : collector.slices) {\n        if (uses.used.find(slice_var) == uses.used.end()) dead.insert(slice_var);\n      }\n      if (dead.empty()) break;\n      DropDeadSliceMutator dropper(dead);\n      auto dropped = dropper.VisitStmt(new_body);\n      if (dropped.get() == new_body.get()) break;  // nothing left to remove\n      new_body = dropped;\n    }\n\n    if (new_body.get() == func->body_.get()) return func;\n    auto new_func = MutableCopy(func);\n    new_func->body_ = new_body;\n    return new_func;\n  };\n  return CreateFunctionPass(pass_func, kPassName, kCanonicalizeTileSliceProperties);\n}",
   "factoryRef": "src/ir/transforms/canonicalize_tile_slice_pass.cpp:489",
   "required": [
    "SSAForm",
    "SplitIncoreOrch",
    "IncoreTileOps",
    "TileOps2D",
    "NormalizedStmtStructure"
   ],
   "produced": [
    "SSAForm",
    "SplitIncoreOrch",
    "IncoreTileOps",
    "TileOps2D",
    "NormalizedStmtStructure"
   ],
   "invalidated": [],
   "origin": {
    "SSAForm": "ConvertToSSA",
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "IncoreTileOps": "ConvertTensorToTileOps",
    "TileOps2D": "FlattenTileNdTo2D",
    "NormalizedStmtStructure": "NormalizeStmtStructure"
   },
   "downstream": {
    "SSAForm": [
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SplitVectorKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "SplitIncoreOrch": [
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "StampTfreeSplit",
     "NormalizeReturnOrder",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape",
     "DeriveCallDirections",
     "AutoDeriveTaskDependencies",
     "MaterializeRuntimeScopes",
     "InsertCommFence"
    ],
    "IncoreTileOps": [
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "NormalizeReturnOrder",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "TileOps2D": [
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "NormalizedStmtStructure": [
     "InferTileMemorySpace",
     "InsertMxScaleAddr",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "ExpandManualPhaseFence"
    ]
   },
   "snippets": []
  },
  {
   "order": 18,
   "name": "InferTileMemorySpace",
   "snake": "infer_tile_memory_space",
   "phase": "tile",
   "role": "decide",
   "layer": "tiling",
   "brief": "推断每个 tile 的片上内存空间，必要时插 tile.move",
   "detail": "分三个阶段：Phase 1 为 InCore 里每个 TileType 变量推断片上 MemorySpace（Vec / Mat / Left / Right / Acc / Bias 等）；Phase 2 收集生产者与消费者约束不匹配处需要插入的 tile.move；Phase 3 落实——写入 memory_space_、插 tile.move、替换实参。另外让可证明循环不变的 Mat 操作数跨顺序迭代常驻。产出 TileMemoryInferred 与 AccToGmStoreValid。",
   "watch": "**空间归属是有代价的选择**：推错就多一次搬运，或者根本走不通。代码里能看到大量 `InheritFromInput(call).value_or(MemorySpace::Vec)` —— 推不出来就默认 Vec。AccToGmStoreValid 也在这里才可验证：同一份 DSL 程序，结果走 Vec 合法、走 Acc 就要求 backend 的 fix-pipe 能把 dtype 收窄。",
   "file": "src/ir/transforms/infer_tile_memory_space_pass.cpp",
   "lines": 992,
   "factory": "Pass InferTileMemorySpace() {\n  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {\n    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;\n    for (const auto& [gvar, func] : program->functions_) {\n      if (func->func_type_ == FunctionType::InCore) {\n        new_functions[gvar] = TransformInferTileMemorySpace(func);\n      } else {\n        new_functions[gvar] = func;\n      }\n    }\n    auto inferred = std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);\n    return loop_invariant_mat_residency::Apply(inferred);\n  };\n  return CreateProgramPass(pass_func, \"InferTileMemorySpace\", kInferTileMemorySpaceProperties);\n}",
   "factoryRef": "src/ir/transforms/infer_tile_memory_space_pass.cpp:879",
   "required": [
    "SSAForm",
    "IncoreTileOps",
    "SplitIncoreOrch",
    "NormalizedStmtStructure"
   ],
   "produced": [
    "SSAForm",
    "TileMemoryInferred",
    "NormalizedStmtStructure",
    "AivSplitValid",
    "AccToGmStoreValid"
   ],
   "invalidated": [
    "AivSplitValid"
   ],
   "origin": {
    "SSAForm": "ConvertToSSA",
    "IncoreTileOps": "ConvertTensorToTileOps",
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "NormalizedStmtStructure": "NormalizeStmtStructure"
   },
   "downstream": {
    "SSAForm": [
     "InsertMxScaleAddr",
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SplitVectorKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "TileMemoryInferred": [
     "InsertMxScaleAddr",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "NormalizedStmtStructure": [
     "InsertMxScaleAddr",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "ExpandManualPhaseFence"
    ],
    "AivSplitValid": [
     "LowerAutoVectorSplit"
    ],
    "AccToGmStoreValid": []
   },
   "snippets": [
    {
     "file": "src/ir/transforms/infer_tile_memory_space_pass.cpp",
     "from": 320,
     "to": 333,
     "label": "推不出来就默认 Vec",
     "code": "  if (entry.HasRetargetableMemoryKwarg()) {\n    auto demand_it = demands_.find(out_var);\n    if (demand_it != demands_.end()) {\n      MemorySpace demand = demand_it->second;\n      // Retargetable DDR-facing producers (tile.load) can only directly\n      // produce {Vec, Mat}; specialized demands (Left/Right/Acc/Bias) from\n      // downstream compute ops (matmul etc.) must be reached via a\n      // tile.move inserted by Phase 2 MoveCollector. Clamping here keeps\n      // the producer's output hardware-valid and preserves the move chain.\n      if (demand == MemorySpace::Vec || demand == MemorySpace::Mat) return demand;\n    }\n  }\n  return InheritFromInput(call).value_or(MemorySpace::Vec);\n}"
    }
   ]
  },
  {
   "order": 19,
   "name": "InsertMxScaleAddr",
   "snake": "insert_mx_scale_addr",
   "phase": "tile",
   "role": "mech",
   "layer": null,
   "brief": "为 MX matmul 消费者插 scale 地址绑定",
   "detail": "在所有操作数的内存空间都已确定之后，为 MX matmul 的消费者插入编译器生成的 tile.tget_scale_addr 绑定。",
   "watch": "必须排在 InferTileMemorySpace 之后——它要求 Left/LeftScale 和 Right/RightScale 的空间都是具体的。",
   "file": "src/ir/transforms/insert_mx_scale_addr_pass.cpp",
   "lines": 221,
   "factory": "Pass InsertMxScaleAddr() {\n  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {\n    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;\n    for (const auto& [gvar, func] : program->functions_) {\n      if (func->func_type_ == FunctionType::InCore) {\n        new_functions[gvar] = TransformInsertMxScaleAddr(func);\n      } else {\n        new_functions[gvar] = func;\n      }\n    }\n    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);\n  };\n  return CreateProgramPass(pass_func, \"InsertMxScaleAddr\", kInsertMxScaleAddrProperties);\n}",
   "factoryRef": "src/ir/transforms/insert_mx_scale_addr_pass.cpp:203",
   "required": [
    "SSAForm",
    "IncoreTileOps",
    "SplitIncoreOrch",
    "NormalizedStmtStructure",
    "TileMemoryInferred"
   ],
   "produced": [
    "SSAForm",
    "IncoreTileOps",
    "SplitIncoreOrch",
    "NormalizedStmtStructure",
    "TileMemoryInferred"
   ],
   "invalidated": [],
   "origin": {
    "SSAForm": "ConvertToSSA",
    "IncoreTileOps": "ConvertTensorToTileOps",
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "NormalizedStmtStructure": "NormalizeStmtStructure",
    "TileMemoryInferred": "InferTileMemorySpace"
   },
   "downstream": {
    "SSAForm": [
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SplitVectorKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "IncoreTileOps": [
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "NormalizeReturnOrder",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "SplitIncoreOrch": [
     "ResolveBackendOpLayouts",
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "StampTfreeSplit",
     "NormalizeReturnOrder",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape",
     "DeriveCallDirections",
     "AutoDeriveTaskDependencies",
     "MaterializeRuntimeScopes",
     "InsertCommFence"
    ],
    "NormalizedStmtStructure": [
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "ExpandManualPhaseFence"
    ],
    "TileMemoryInferred": [
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ]
   },
   "snippets": []
  },
  {
   "order": 20,
   "name": "ResolveBackendOpLayouts",
   "snake": "resolve_backend_op_layouts",
   "phase": "tile",
   "role": "decide",
   "layer": "tiling",
   "brief": "修复 elementwise 算子的 backend 要求布局",
   "detail": "两条路径：[N,1] 列主向量重塑成 [1,N] 行主视图（零成本，改视图）；其余非行主 tile 走 tile.move(..., blayout=row_major) 强制转换（有成本，多一次搬运）。Pass 返回前会自我规范化语句结构。",
   "watch": "**两条路代价不同，所以是决策点**。仓里 layout 类 Issue 长期高发：#137 blayout 错误发射、#345 ptoas 把 Left tile 从 col_major 转成 row_major、#762 matmul rhs 用错块布局。",
   "file": "src/ir/transforms/resolve_backend_op_layouts_pass.cpp",
   "lines": 317,
   "factory": "Pass ResolveBackendOpLayouts() {\n  return CreateFunctionPass(RewriteFunction, \"ResolveBackendOpLayouts\", kResolveBackendOpLayoutsProperties);\n}",
   "factoryRef": "src/ir/transforms/resolve_backend_op_layouts_pass.cpp:310",
   "required": [
    "SSAForm",
    "IncoreTileOps",
    "SplitIncoreOrch",
    "TileOps2D"
   ],
   "produced": [
    "SSAForm",
    "IncoreTileOps",
    "SplitIncoreOrch",
    "TileOps2D",
    "NormalizedStmtStructure"
   ],
   "invalidated": [],
   "origin": {
    "SSAForm": "ConvertToSSA",
    "IncoreTileOps": "ConvertTensorToTileOps",
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "TileOps2D": "FlattenTileNdTo2D"
   },
   "downstream": {
    "SSAForm": [
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SplitVectorKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "IncoreTileOps": [
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "NormalizeReturnOrder",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "SplitIncoreOrch": [
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "StampTfreeSplit",
     "NormalizeReturnOrder",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape",
     "DeriveCallDirections",
     "AutoDeriveTaskDependencies",
     "MaterializeRuntimeScopes",
     "InsertCommFence"
    ],
    "TileOps2D": [
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "NormalizedStmtStructure": [
     "LowerAutoVectorSplit",
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "ExpandManualPhaseFence"
    ]
   },
   "snippets": []
  },
  {
   "order": 21,
   "name": "LowerAutoVectorSplit",
   "snake": "lower_auto_vector_split",
   "phase": "split",
   "role": "decide",
   "layer": "tiling",
   "brief": "AUTO pl.split 转成显式 split_aiv 形式",
   "detail": "把 AUTO 的 pl.split 混合 InCore 函数转成显式的 split_aiv 形式（aiv_shard / aic_gather + 半宽向量子区域），使 ExpandMixedKernel 能统一地把它们折成带 split 标记的 tpush/tpop。RFC #1300 的产物，Default 策略下无条件运行。",
   "watch": "它**关闭** AivSplitValid 窗口：消费并擦除 SplitAivScopeStmt 区域，所以入口要求该属性、出口 invalidate 它。跑完之后每个 split 函数到达 SplitVectorKernel 时都已经带好 split_aiv 标记，SplitVectorKernel 就只剩盖属性的活。",
   "file": "src/ir/transforms/lower_auto_vector_split_pass.cpp",
   "lines": 1057,
   "factory": "Pass LowerAutoVectorSplit() {\n  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {\n    std::vector<FunctionPtr> new_functions;\n    bool changed = false;\n    new_functions.reserve(program->functions_.size());\n\n    for (const auto& [gvar, func] : program->functions_) {\n      auto mode = func->GetSplitMode();\n      const bool is_incore = (func->func_type_ == FunctionType::InCore);\n      // EXPLICIT region path: an InCore function whose body still carries one or\n      // more SplitAivScopeStmt regions (preserved through OutlineIncoreScopes).\n      // Each region carries its own mode, so this is checked before the AUTO path\n      // and handles the multi-mode case the single func-level mode cannot.\n      if (is_incore && BodyContainsSplitAivScope(func->body_)) {\n        new_functions.push_back(LowerExplicitRegionFunction(func));\n        changed = true;\n        continue;\n      }\n      // AUTO whole-function path (unchanged): lower genuinely mixed\n      // (cube<->vector) functions. Pure-vector pl.split functions have no boundary\n      // to converge; ExpandMixedKernel strips their split, so marking them\n      // split_aiv here would desync.\n      if (is_incore && mode.has_value() && mode.value() != SplitMode::None &&\n          !IsAlreadyExplicitSplitAiv(func) && IsMixedCubeVector(func)) {\n        new_functions.push_back(LowerFunction(func, mode.value()));\n        changed = true;\n      } else {\n        new_functions.push_back(func);\n      }\n    }\n\n    if (!changed) return program;\n    return std::make_shared<Program>(new_functions, program->name_, program->span_);\n  };\n\n  return CreateProgramPass(pass_func, \"LowerAutoVectorSplit\", kLowerAutoVectorSplitProperties);\n}",
   "factoryRef": "src/ir/transforms/lower_auto_vector_split_pass.cpp:1017",
   "required": [
    "SSAForm",
    "IncoreTileOps",
    "SplitIncoreOrch",
    "TileOps2D",
    "TileMemoryInferred",
    "NormalizedStmtStructure",
    "AivSplitValid"
   ],
   "produced": [
    "SSAForm",
    "IncoreTileOps",
    "SplitIncoreOrch",
    "TileOps2D",
    "TileMemoryInferred",
    "NormalizedStmtStructure"
   ],
   "invalidated": [
    "AivSplitValid"
   ],
   "origin": {
    "SSAForm": "ConvertToSSA",
    "IncoreTileOps": "ConvertTensorToTileOps",
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "TileOps2D": "FlattenTileNdTo2D",
    "TileMemoryInferred": "InferTileMemorySpace",
    "NormalizedStmtStructure": "NormalizeStmtStructure",
    "AivSplitValid": "OutlineIncoreScopes"
   },
   "downstream": {
    "SSAForm": [
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SplitVectorKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "IncoreTileOps": [
     "ExpandMixedKernel",
     "NormalizeReturnOrder",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "SplitIncoreOrch": [
     "ExpandMixedKernel",
     "StampTfreeSplit",
     "NormalizeReturnOrder",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape",
     "DeriveCallDirections",
     "AutoDeriveTaskDependencies",
     "MaterializeRuntimeScopes",
     "InsertCommFence"
    ],
    "TileOps2D": [
     "ExpandMixedKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "TileMemoryInferred": [
     "ExpandMixedKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "NormalizedStmtStructure": [
     "ExpandMixedKernel",
     "InjectGMPipeBuffer",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "ExpandManualPhaseFence"
    ]
   },
   "snippets": []
  },
  {
   "order": 22,
   "name": "ExpandMixedKernel",
   "snake": "expand_mixed_kernel",
   "phase": "split",
   "role": "decide",
   "layer": "tiling",
   "brief": "混合 InCore 拆成 AIC + AIV 两个 kernel",
   "detail": "把混合 InCore 函数拆成独立的 AIC（Cube）+ AIV（Vector）kernel，外面包一层 Group 函数；非混合的 InCore 函数直接把 FunctionType 改成 AIC 或 AIV。产出 MixedKernelExpanded 与 HardSyncallOccupancyValid。",
   "watch": "HardSyncallOccupancyValid 不是它做的变换带来的，而是因为**它解析出了每个 kernel 的 FunctionType**——硬 syncall 占用率验证器依赖这个前提，所以验证器就在这个 Pass 之后触发一次。pypto#1935 记录过 hard syncall 在部分核占用下死锁（AICore 507018）且没有编译期检查。#1083 记录过它把纯 AIC kernel 误拆，因为 tile.create 被误判成 VECTOR。",
   "file": "src/ir/transforms/expand_mixed_kernel_pass.cpp",
   "lines": 1933,
   "factory": "Pass ExpandMixedKernel() {\n  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {\n    // Phase 1: Pre-scan — find InCore functions that have existing callers.\n    std::unordered_set<std::string> incore_names;\n    for (const auto& [gvar, func] : program->functions_) {\n      if (func->func_type_ == FunctionType::InCore) {\n        incore_names.insert(func->name_);\n      }\n    }\n\n    // Map InCore name -> callers that can be rewritten in place.\n    std::unordered_set<std::string> incore_with_group_caller;\n    // Map InCore name -> callers that still need the original function name to remain callable.\n    std::unordered_set<std::string> incore_with_preserved_name_caller;\n    for (const auto& [gvar, func] : program->functions_) {\n      for (const auto& name : incore_names) {\n        if (!FunctionCallsFunction(func, name)) {\n          continue;\n        }\n        if (func->func_type_ == FunctionType::Group) {\n          incore_with_group_caller.insert(name);\n        } else {\n          incore_with_preserved_name_caller.insert(name);\n        }\n      }\n    }\n\n    // Phase 2: Expand InCore functions, collect rewrite info\n    struct RewriteInfo {\n      std::string aic_name;\n      std::string aiv_name;\n    };\n    std::unordered_map<std::string, RewriteInfo> rewrite_map;\n    std::vector<FunctionPtr> new_functions;\n\n    for (const auto& [gvar, func] : program->functions_) {\n      if (func->func_type_ != FunctionType::InCore) {\n        new_functions.push_back(func);\n        continue;\n      }\n\n      // Check if function is mixed (recursive analysis detects ops inside loops/conditionals)\n      auto stmts = FlattenBody(func->body_);\n      auto tpop_defs = CollectTpopDefs(stmts);\n      std::unordered_map<const Stmt*, CoreAffinity> stmt_map;\n      std::unordered_map<const Var*, CoreAffinity> var_affinity;\n      auto combined = AnalyzeStmtsAffinity(stmts, stmt_map, var_affinity, tpop_defs);\n\n      // A function is mixed if combined affinity says so. Leaf boundary moves\n      // (tile.move across the C/V divide) classify as MIXED via ClassifyCallAffinity,\n      // so the roll-up captures them without a separate enum value.\n      bool is_mixed = (combined == CoreAffinity::MIXED);\n\n      if (!is_mixed) {\n        // Not mixed — convert InCore to the corresponding AIC or AIV type\n        FunctionType new_type = (combined == CoreAffinity::CUBE) ? FunctionType::AIC : FunctionType::AIV;\n        // Clear split mode — pure AIC/AIV functions don't need vector splitting\n        auto attrs = func->attrs_;\n        attrs.erase(\n            std::remove_if(attrs.begin(), attrs.end(), [](const auto& kv) { return kv.first == \"split\"; }),\n            attrs.end());\n        auto converted = MutableCopy(func);\n        converted->func_type_ = new_type;\n        converted->level_ = FunctionTypeToLevel(new_type);\n        converted->role_ = Role::SubWorker;\n        converted->attrs_ = attrs;\n        new_functions.push_back(converted);\n        continue;\n      }\n      // Expand mixed kernel.",
   "factoryRef": "src/ir/transforms/expand_mixed_kernel_pass.cpp:1820",
   "required": [
    "SSAForm",
    "IncoreTileOps",
    "SplitIncoreOrch",
    "TileOps2D",
    "TileMemoryInferred",
    "NormalizedStmtStructure"
   ],
   "produced": [
    "SSAForm",
    "MixedKernelExpanded",
    "NormalizedStmtStructure",
    "HardSyncallOccupancyValid"
   ],
   "invalidated": [],
   "origin": {
    "SSAForm": "ConvertToSSA",
    "IncoreTileOps": "ConvertTensorToTileOps",
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "TileOps2D": "FlattenTileNdTo2D",
    "TileMemoryInferred": "InferTileMemorySpace",
    "NormalizedStmtStructure": "NormalizeStmtStructure"
   },
   "downstream": {
    "SSAForm": [
     "InjectGMPipeBuffer",
     "SplitVectorKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "MixedKernelExpanded": [
     "InjectGMPipeBuffer",
     "SplitVectorKernel"
    ],
    "NormalizedStmtStructure": [
     "InjectGMPipeBuffer",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "ExpandManualPhaseFence"
    ],
    "HardSyncallOccupancyValid": []
   },
   "snippets": []
  },
  {
   "order": 23,
   "name": "InjectGMPipeBuffer",
   "snake": "inject_gm_pipe_buffer",
   "phase": "split",
   "role": "mech",
   "layer": null,
   "brief": "为经 GM 中转的跨核 pipe 注入 workspace 参数",
   "detail": "为需要把 slot 数据经 GM 中转的 backend（当前 Ascend910B）注入 __gm_pipe_buffer workspace 参数。按 backend 能力表决定注不注。",
   "watch": "从 ExpandMixedKernel 里抽出来的独立 Pass，紧跟其后运行。",
   "file": "src/ir/transforms/inject_gm_pipe_buffer_pass.cpp",
   "lines": 532,
   "factory": "Pass InjectGMPipeBuffer() {\n  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {\n    if (!backend::BackendConfig::IsConfigured() ||\n        !PassContext::Current()->GetBackendHandler()->RequiresGMPipeBuffer()) {\n      return program;\n    }\n    std::vector<FunctionPtr> functions;\n    functions.reserve(program->functions_.size());\n    for (const auto& [gvar, func] : program->functions_) {\n      functions.push_back(func);\n    }\n    InjectGMSlotBufferInPlace(functions);\n    return std::make_shared<Program>(functions, program->name_, program->span_);\n  };\n\n  return CreateProgramPass(pass_func, \"InjectGMPipeBuffer\", kInjectGMPipeBufferProperties);\n}",
   "factoryRef": "src/ir/transforms/inject_gm_pipe_buffer_pass.cpp:511",
   "required": [
    "SSAForm",
    "MixedKernelExpanded",
    "NormalizedStmtStructure"
   ],
   "produced": [
    "SSAForm",
    "MixedKernelExpanded",
    "NormalizedStmtStructure"
   ],
   "invalidated": [],
   "origin": {
    "SSAForm": "ConvertToSSA",
    "MixedKernelExpanded": "ExpandMixedKernel",
    "NormalizedStmtStructure": "NormalizeStmtStructure"
   },
   "downstream": {
    "SSAForm": [
     "SplitVectorKernel",
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "MixedKernelExpanded": [
     "SplitVectorKernel"
    ],
    "NormalizedStmtStructure": [
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "ExpandManualPhaseFence"
    ]
   },
   "snippets": []
  },
  {
   "order": 24,
   "name": "SplitVectorKernel",
   "snake": "split_vector_kernel",
   "phase": "split",
   "role": "decide",
   "layer": "tiling",
   "brief": "盖 split 属性，处理 no-split 双 AIV 路径",
   "detail": "经过分阶段收敛重构之后，它只剩两件窄活：盖 split 属性、处理 no-split 的双 AIV 路径。产出 VectorKernelSplit。",
   "watch": "**RFC #1820 要求整体替换掉它的推断**，诊断原文是「a syntactic, per-statement shape-halver with no model of the index space」——它是一个按算子白名单的语法半分器：load/store/full/create/reshape 会改实参，tile.slice 和其他算子只改结果类型，于是造出 tstore(subview<16>, partition<8>) 这种不匹配。失败模式对 online-softmax 是**结构性**的，不是偶发。",
   "file": "src/ir/transforms/split_vector_kernel_pass.cpp",
   "lines": 701,
   "factory": "Pass SplitVectorKernel() {\n  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {\n    std::vector<FunctionPtr> new_functions;\n    bool changed = false;\n\n    for (const auto& [gvar, func] : program->functions_) {\n      // External kernels are signature-only declarations. Their hand-written\n      // source owns sub-lane partitioning, so preserve launch attrs but never\n      // synthesize a DSL body (which would violate the external-source contract).\n      if (func->HasAttr(kExternalSourceAttr)) {\n        if (func->func_type_ == FunctionType::AIV) {\n          auto explicit_mode = func->GetSplitMode();\n          bool split_aiv = func->HasAttr(\"split_aiv\") && func->GetAttr<bool>(\"split_aiv\", false);\n          if (explicit_mode.has_value() && explicit_mode.value() != SplitMode::None) {\n            auto external_func = MutableCopy(func);\n            external_func->attrs_ = WithSplitAttrs(func, explicit_mode.value(), /*is_aiv=*/true);\n            new_functions.push_back(external_func);\n            changed = true;\n            continue;\n          } else if (split_aiv && !func->GetAttr<bool>(kDualAivDispatchAttr, false)) {\n            auto external_func = MutableCopy(func);\n            auto attrs = external_func->attrs_;\n            attrs.erase(std::remove_if(attrs.begin(), attrs.end(),\n                                       [](const auto& kv) { return kv.first == kDualAivDispatchAttr; }),\n                        attrs.end());\n            attrs.emplace_back(kDualAivDispatchAttr, true);\n            external_func->attrs_ = std::move(attrs);\n            new_functions.push_back(external_func);\n            changed = true;\n            continue;\n          }\n        }\n        new_functions.push_back(func);\n        continue;\n      }\n\n      // split_aiv kernels arrive here already in the explicit form: either\n      // hand-written, or produced by LowerAutoVectorSplit from an AUTO pl.split\n      // mixed InCore function. Their tile.aiv_shard / tile.aic_gather have been\n      // folded into split-stamped tpush/tpop pairs (via ExpandMixedKernel's\n      // boundary machinery) and they carry already-halved compute tiles. This is\n      // the SOLE split path through SplitVectorKernel: just stamp split +\n      // dual_aiv_dispatch and pass the body through unchanged. The former per-op\n      // halving driver was deleted — after LowerAutoVectorSplit runs, every split\n      // function reaches here split_aiv-marked, so re-halving here would\n      // double-halve the (already-half) body.\n      if ((func->func_type_ == FunctionType::AIV || func->func_type_ == FunctionType::AIC) &&\n          func->HasAttr(\"split_aiv\") && func->GetAttr<bool>(\"split_aiv\", false)) {\n        auto explicit_mode = func->GetSplitMode();\n        auto new_func = MutableCopy(func);\n        if (explicit_mode.has_value() && explicit_mode.value() != SplitMode::None) {\n          // Single-mode split_aiv: a function-level \"split\" attr survives (a\n          // hand-written kernel, an AUTO function converged by LowerAutoVectorSplit,\n          // or a single-mode explicit region). Stamp split + dual_aiv_dispatch.\n          new_func->attrs_ =\n              WithSplitAttrs(func, explicit_mode.value(), func->func_type_ == FunctionType::AIV);\n        } else {\n          // Multi-mode explicit split_aiv: the per-region modes were lowered and\n          // erased by LowerAutoVectorSplit (pass 20); no single function-level mode\n          // survives. The authoritative per-op \"split\" ints already sit on the\n          // tpop/tpush pairs, so only the mode-agnostic dual_aiv_dispatch bool needs\n          // stamping here (all RequiresDualAivDispatch consults).\n          auto attrs = func->attrs_;\n          attrs.erase(std::remove_if(attrs.begin(), attrs.end(),\n                                     [](const auto& kv) { return kv.first == kDualAivDispatchAttr; }),\n                      attrs.end());\n          if (func->func_type_ == FunctionType::AIV) {\n            attrs.emplace_back(kDualAivDispatchAttr, true);\n          }\n          new_func->attrs_ = std::move(attrs);",
   "factoryRef": "src/ir/transforms/split_vector_kernel_pass.cpp:590",
   "required": [
    "SSAForm",
    "MixedKernelExpanded"
   ],
   "produced": [
    "SSAForm",
    "VectorKernelSplit",
    "NormalizedStmtStructure"
   ],
   "invalidated": [],
   "origin": {
    "SSAForm": "ConvertToSSA",
    "MixedKernelExpanded": "ExpandMixedKernel"
   },
   "downstream": {
    "SSAForm": [
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "VectorKernelSplit": [],
    "NormalizedStmtStructure": [
     "SkewCrossCorePipeline",
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "ExpandManualPhaseFence"
    ]
   },
   "snippets": []
  },
  {
   "order": 25,
   "name": "StampTfreeSplit",
   "snake": "stamp_tfree_split",
   "phase": "split",
   "role": "mech",
   "layer": null,
   "brief": "把 tpop 的 split / pipe id 复制到配对的 tfree",
   "detail": "把每个跨核 tpop 的 split 和 pipe id 复制到与之配对的 tfree 算子上，这样 codegen 直接从算子上读，不需要维护一张 tpop 查找表。",
   "watch": "**位置卡得很死**：必须在 SplitVectorKernel 敲定 tpop 的 split 之后、SkewCrossCorePipeline 克隆 tpop/tfree 对之前——否则克隆体不带 split。",
   "file": "src/ir/transforms/stamp_tfree_split_pass.cpp",
   "lines": 125,
   "factory": "Pass StampTfreeSplit() {\n  auto pass_func = [](const FunctionPtr& func) -> FunctionPtr {\n    if (!func || !func->body_) return func;\n    StampTfreeSplitMutator mutator;\n    auto new_body = mutator.VisitStmt(func->body_);\n    if (new_body.get() == func->body_.get()) return func;\n    return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,\n                                      func->return_types_, new_body, func->span_, func->func_type_,\n                                      func->level_, func->role_, func->attrs_);\n  };\n  return CreateFunctionPass(pass_func, \"StampTfreeSplit\", kStampTfreeSplitProperties);\n}",
   "factoryRef": "src/ir/transforms/stamp_tfree_split_pass.cpp:110",
   "required": [
    "SplitIncoreOrch"
   ],
   "produced": [],
   "invalidated": [],
   "origin": {
    "SplitIncoreOrch": "OutlineIncoreScopes"
   },
   "downstream": {},
   "snippets": []
  },
  {
   "order": 26,
   "name": "NormalizeReturnOrder",
   "snake": "normalize_return_order",
   "phase": "split",
   "role": "mech",
   "layer": null,
   "brief": "把 InCore 返回元组重排成规范顺序",
   "detail": "重排每个 InCore 函数的返回元组，使 return[i] 与参数顺序对应。产出 ReturnParamsExplicit：InCore/Group/Spmd 的 tensor 返回按指针身份引用函数参数，于是「返回值 → 参数」的映射变成一次查表。",
   "watch": "ReturnParamsExplicit 是 pypto#1702 的产物——之前 codegen 用 out_indices[0] 兜底，把多输出 spmd scope 的返回值别名到了错的输出张量，numel 不匹配的 reshape 直接 507018。",
   "file": "src/ir/transforms/normalize_return_order_pass.cpp",
   "lines": 461,
   "factory": "Pass NormalizeReturnOrder() {\n  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {\n    // Step A: Analyze InCore functions and compute permutations.\n    std::unordered_map<std::string, std::vector<int>> permutations;\n    std::vector<FunctionPtr> functions;\n    bool modified = false;\n\n    for (const auto& [gvar, func] : program->functions_) {\n      const bool is_wrapper = IsWrapperType(func->func_type_);\n      if (IsInCoreType(func->func_type_) || is_wrapper) {\n        // Step A0: make every tensor return an explicit param reference.\n        FunctionPtr current = func;\n        if (auto canonical = CanonicalizeReturnValues(current, program)) {\n          current = canonical;\n          modified = true;\n        }\n        std::vector<int> perm;\n        if (IsInCoreType(current->func_type_)) perm = ComputeReturnPermutation(current);\n        if (!perm.empty()) {\n          auto new_func = ReorderReturns(current, perm);\n          permutations[current->name_] = std::move(perm);\n          functions.push_back(new_func);\n          modified = true;\n        } else {\n          functions.push_back(current);\n        }\n      } else {\n        functions.push_back(func);\n      }\n    }\n\n    if (!modified) return program;\n\n    // Step B: Update TupleGetItemExpr indices in non-InCore functions.\n    std::vector<FunctionPtr> final_functions;\n    for (const auto& func : functions) {\n      if (!IsInCoreType(func->func_type_)) {\n        TupleIndexPermutationMutator mutator(permutations);\n        auto new_body = mutator.VisitStmt(func->body_);\n        if (new_body.get() != func->body_.get()) {\n          final_functions.push_back(std::make_shared<Function>(\n              func->name_, func->params_, func->param_directions_, func->return_types_, new_body, func->span_,\n              func->func_type_, func->level_, func->role_, func->attrs_));\n        } else {\n          final_functions.push_back(func);\n        }\n      } else {\n        final_functions.push_back(func);\n      }\n    }\n\n    return std::make_shared<Program>(final_functions, program->name_, program->span_);\n  };\n\n  return CreateProgramPass(pass_func, \"NormalizeReturnOrder\", kNormalizeReturnOrderProperties);\n}",
   "factoryRef": "src/ir/transforms/normalize_return_order_pass.cpp:402",
   "required": [
    "SplitIncoreOrch",
    "IncoreTileOps"
   ],
   "produced": [
    "ReturnParamsExplicit"
   ],
   "invalidated": [],
   "origin": {
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "IncoreTileOps": "ConvertTensorToTileOps"
   },
   "downstream": {
    "ReturnParamsExplicit": []
   },
   "snippets": []
  },
  {
   "order": 27,
   "name": "SkewCrossCorePipeline",
   "snake": "skew_cross_core_pipeline",
   "phase": "split",
   "role": "decide",
   "layer": "tiling",
   "brief": "跨核流水错峰，让 cube 与 vector 重叠",
   "detail": "把混合 cube/vector 的跨核 pl.pipeline 循环改写成 prologue / steady / epilogue 的错峰结构，让两个核重叠执行；做不到就降级成 Sequential。取代了过去用 unroll + IO 聚类处理跨核循环的老做法。",
   "watch": "**错峰多少、哪些语句进哪一级，都是调度选择**。pypto#2130 指出它生成不了 skewed 跨核调度时，用户只能回落到手工 sync_set / sync_wait。",
   "file": "src/ir/transforms/skew_cross_core_pipeline_pass.cpp",
   "lines": 843,
   "factory": "Pass SkewCrossCorePipeline() {\n  return CreateFunctionPass(TransformSkewCrossCorePipeline, \"SkewCrossCorePipeline\",\n                            kSkewCrossCorePipelineProperties);\n}",
   "factoryRef": "src/ir/transforms/skew_cross_core_pipeline_pass.cpp:836",
   "required": [
    "SSAForm",
    "SplitIncoreOrch",
    "IncoreTileOps",
    "TileOps2D",
    "TileMemoryInferred",
    "NormalizedStmtStructure"
   ],
   "produced": [
    "SSAForm",
    "SplitIncoreOrch",
    "IncoreTileOps",
    "TileOps2D",
    "TileMemoryInferred",
    "NormalizedStmtStructure"
   ],
   "invalidated": [],
   "origin": {
    "SSAForm": "ConvertToSSA",
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "IncoreTileOps": "ConvertTensorToTileOps",
    "TileOps2D": "FlattenTileNdTo2D",
    "TileMemoryInferred": "InferTileMemorySpace",
    "NormalizedStmtStructure": "NormalizeStmtStructure"
   },
   "downstream": {
    "SSAForm": [
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "SplitIncoreOrch": [
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape",
     "DeriveCallDirections",
     "AutoDeriveTaskDependencies",
     "MaterializeRuntimeScopes",
     "InsertCommFence"
    ],
    "IncoreTileOps": [
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "TileOps2D": [
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "TileMemoryInferred": [
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "NormalizedStmtStructure": [
     "LowerPipelineToSlots",
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "ExpandManualPhaseFence"
    ]
   },
   "snippets": []
  },
  {
   "order": 28,
   "name": "LowerPipelineToSlots",
   "snake": "lower_pipeline_to_slots",
   "phase": "split",
   "role": "decide",
   "layer": "tiling",
   "brief": "流水体在一块分配的 N 个 slot 间轮转（PTOAS planner 路径）",
   "detail": "把 pl.pipeline(N, stage=F) 的循环体在**一块分配的 F 个 slot** 之间轮转，而不是把循环体复制 F 份。自身按 memory_planner=PTOAS 自门控。",
   "watch": "**这是同一个问题的两种策略并存的典型例子**。pass_manager.py 的注释解释得很直白：它能多缓冲的就多缓冲并把循环降级，它拒绝的循环仍留在 ForKind.Pipeline 交给下一个 Pass 复制——所以两个 Pass 都跑，而不是互相替代。",
   "file": "src/ir/transforms/lower_pipeline_to_slots_pass.cpp",
   "lines": 572,
   "factory": "Pass LowerPipelineToSlots() {\n  return CreateFunctionPass(TransformLowerPipelineToSlots, \"LowerPipelineToSlots\",\n                            kLowerPipelineToSlotsProperties);\n}",
   "factoryRef": "src/ir/transforms/lower_pipeline_to_slots_pass.cpp:565",
   "required": [
    "SSAForm",
    "SplitIncoreOrch",
    "IncoreTileOps",
    "TileOps2D",
    "TileMemoryInferred",
    "NormalizedStmtStructure"
   ],
   "produced": [
    "SSAForm",
    "SplitIncoreOrch",
    "IncoreTileOps",
    "TileOps2D",
    "TileMemoryInferred",
    "NormalizedStmtStructure"
   ],
   "invalidated": [],
   "origin": {
    "SSAForm": "ConvertToSSA",
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "IncoreTileOps": "ConvertTensorToTileOps",
    "TileOps2D": "FlattenTileNdTo2D",
    "TileMemoryInferred": "InferTileMemorySpace",
    "NormalizedStmtStructure": "NormalizeStmtStructure"
   },
   "downstream": {
    "SSAForm": [
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "SplitIncoreOrch": [
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape",
     "DeriveCallDirections",
     "AutoDeriveTaskDependencies",
     "MaterializeRuntimeScopes",
     "InsertCommFence"
    ],
    "IncoreTileOps": [
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "TileOps2D": [
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "TileMemoryInferred": [
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "NormalizedStmtStructure": [
     "LowerPipelineLoops",
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "ExpandManualPhaseFence"
    ]
   },
   "snippets": []
  },
  {
   "order": 29,
   "name": "LowerPipelineLoops",
   "snake": "lower_pipeline_loops",
   "phase": "split",
   "role": "decide",
   "layer": "tiling",
   "brief": "流水体复制 F 份实现 ping-pong",
   "detail": "在 tile 层下降 pl.pipeline(N, stage=F)：每次外层迭代把循环体复制 F 份以支持 ping-pong 缓冲，外层循环保持顺序执行。",
   "watch": "它复制出来的 F 份体，正是 MemoryReuse 后面要决定「合不合并」的对象。**实际达成的深度可能低于用户请求**——容量门不够时会降级，这正是在树的 PH-MR-001 报告的事。",
   "file": "src/ir/transforms/lower_pipeline_loops_pass.cpp",
   "lines": 671,
   "factory": "Pass LowerPipelineLoops() {\n  return CreateFunctionPass(TransformLowerPipelineLoops, \"LowerPipelineLoops\", kLowerPipelineLoopsProperties);\n}",
   "factoryRef": "src/ir/transforms/lower_pipeline_loops_pass.cpp:665",
   "required": [
    "SSAForm",
    "SplitIncoreOrch",
    "IncoreTileOps",
    "TileOps2D",
    "TileMemoryInferred",
    "NormalizedStmtStructure"
   ],
   "produced": [
    "SSAForm",
    "SplitIncoreOrch",
    "IncoreTileOps",
    "TileOps2D",
    "TileMemoryInferred",
    "NormalizedStmtStructure"
   ],
   "invalidated": [],
   "origin": {
    "SSAForm": "ConvertToSSA",
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "IncoreTileOps": "ConvertTensorToTileOps",
    "TileOps2D": "FlattenTileNdTo2D",
    "TileMemoryInferred": "InferTileMemorySpace",
    "NormalizedStmtStructure": "NormalizeStmtStructure"
   },
   "downstream": {
    "SSAForm": [
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "SplitIncoreOrch": [
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape",
     "DeriveCallDirections",
     "AutoDeriveTaskDependencies",
     "MaterializeRuntimeScopes",
     "InsertCommFence"
    ],
    "IncoreTileOps": [
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "TileOps2D": [
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "TileMemoryInferred": [
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "NormalizedStmtStructure": [
     "CanonicalizeIOOrder",
     "MaterializeTensorStrides",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "ExpandManualPhaseFence"
    ]
   },
   "snippets": []
  },
  {
   "order": 30,
   "name": "CanonicalizeIOOrder",
   "snake": "canonicalize_io_order",
   "phase": "split",
   "role": "decide",
   "layer": "tiling",
   "brief": "流水体内按 scalar→load→compute→store 阶梯重排语句",
   "detail": "作用域限定在 ForKind::Pipeline 循环体内的 SeqStmts，按同核硬件单元的阶梯（scalar → load → compute → store）重排语句，受 SSA 依赖图约束。产出 PipelineResolved（不再有 ForKind::Pipeline 存活）。",
   "watch": "**重排就是调度**。把复制体的 load 聚到前面直接决定流水气泡大小——这一步的效果最终体现在泳道图上，但用户看不到是谁排的。",
   "file": "src/ir/transforms/canonicalize_io_order_pass.cpp",
   "lines": 535,
   "factory": "Pass CanonicalizeIOOrder() {\n  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {\n    INTERNAL_CHECK(program) << \"CanonicalizeIOOrder cannot run on null program\";\n\n    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;\n    bool any_change = false;\n    for (const auto& [gvar, func] : program->functions_) {\n      // Validate the InOut-use discipline once per function: variable scopes\n      // don't cross function boundaries, so a single walk over the function\n      // body catches every violation that could affect any nested SeqStmts.\n      // Under strict verification such violations are rejected earlier, but\n      // with VerificationLevel.NONE a non-conforming function can reach us,\n      // and we must not reorder potentially-unsound dataflow.\n      if (!stmt_dep::CollectInOutUseDisciplineDiagnostics(func->body_, program).empty()) {\n        new_functions.emplace(gvar, func);\n        continue;\n      }\n      CanonicalizeIOOrderMutator mutator;\n      auto new_body = mutator.VisitStmt(func->body_);\n      if (new_body.get() == func->body_.get()) {\n        new_functions.emplace(gvar, func);\n      } else {\n        auto new_func = MutableCopy(func);\n        new_func->body_ = new_body;\n        new_functions.emplace(gvar, new_func);\n        any_change = true;\n      }\n    }\n    if (!any_change) return program;\n\n    auto new_program = MutableCopy(program);\n    new_program->functions_ = std::move(new_functions);\n    return new_program;\n  };\n\n  return CreateProgramPass(pass_func, \"CanonicalizeIOOrder\", kCanonicalizeIOOrderProperties);\n}",
   "factoryRef": "src/ir/transforms/canonicalize_io_order_pass.cpp:495",
   "required": [
    "SSAForm",
    "SplitIncoreOrch",
    "IncoreTileOps",
    "TileOps2D",
    "TileMemoryInferred",
    "NormalizedStmtStructure"
   ],
   "produced": [
    "SSAForm",
    "SplitIncoreOrch",
    "IncoreTileOps",
    "TileOps2D",
    "TileMemoryInferred",
    "NormalizedStmtStructure",
    "PipelineResolved"
   ],
   "invalidated": [],
   "origin": {
    "SSAForm": "ConvertToSSA",
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "IncoreTileOps": "ConvertTensorToTileOps",
    "TileOps2D": "FlattenTileNdTo2D",
    "TileMemoryInferred": "InferTileMemorySpace",
    "NormalizedStmtStructure": "NormalizeStmtStructure"
   },
   "downstream": {
    "SSAForm": [
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "SplitIncoreOrch": [
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape",
     "DeriveCallDirections",
     "AutoDeriveTaskDependencies",
     "MaterializeRuntimeScopes",
     "InsertCommFence"
    ],
    "IncoreTileOps": [
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "TileOps2D": [
     "MaterializeTensorStrides",
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "TileMemoryInferred": [
     "MaterializeTensorStrides",
     "InitMemRef"
    ],
    "NormalizedStmtStructure": [
     "MaterializeTensorStrides",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "ExpandManualPhaseFence"
    ],
    "PipelineResolved": []
   },
   "snippets": []
  },
  {
   "order": 31,
   "name": "MaterializeTensorStrides",
   "snake": "materialize_tensor_strides",
   "phase": "mem",
   "role": "mech",
   "layer": null,
   "brief": "为每个 TensorView 填上紧致规范 stride",
   "detail": "为程序里每一个 view.has_value() 但 stride 为空的 TensorType / DistributedTensorType 填上该布局的紧致规范 stride（RFC #1300 §2.4）。产出 TensorViewCanonical。",
   "watch": "pypto#739 记录过 [M,1] 张量缺 pl.DN 静默生成错误 stride；#2087 记录过动态 pitch 张量的列切片经 orch→incore 后 stride 被稠密化，静默读错数据。",
   "file": "src/ir/transforms/materialize_tensor_strides_pass.cpp",
   "lines": 384,
   "factory": "Pass MaterializeTensorStrides() {\n  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {\n    bool modified = false;\n    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;\n    for (const auto& [gvar, func] : program->functions_) {\n      auto new_func = TransformFunction(func);\n      if (new_func.get() != func.get()) modified = true;\n      new_functions[gvar] = std::move(new_func);\n    }\n    if (!modified) return program;\n    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);\n  };\n  return CreateProgramPass(pass_func, \"MaterializeTensorStrides\", kMaterializeTensorStridesProperties);\n}",
   "factoryRef": "src/ir/transforms/materialize_tensor_strides_pass.cpp:366",
   "required": [
    "SSAForm",
    "SplitIncoreOrch",
    "IncoreTileOps",
    "TileOps2D",
    "TileMemoryInferred",
    "NormalizedStmtStructure"
   ],
   "produced": [
    "SSAForm",
    "SplitIncoreOrch",
    "IncoreTileOps",
    "TileOps2D",
    "TileMemoryInferred",
    "NormalizedStmtStructure",
    "TensorViewCanonical"
   ],
   "invalidated": [],
   "origin": {
    "SSAForm": "ConvertToSSA",
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "IncoreTileOps": "ConvertTensorToTileOps",
    "TileOps2D": "FlattenTileNdTo2D",
    "TileMemoryInferred": "InferTileMemorySpace",
    "NormalizedStmtStructure": "NormalizeStmtStructure"
   },
   "downstream": {
    "SSAForm": [
     "InitMemRef"
    ],
    "SplitIncoreOrch": [
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape",
     "DeriveCallDirections",
     "AutoDeriveTaskDependencies",
     "MaterializeRuntimeScopes",
     "InsertCommFence"
    ],
    "IncoreTileOps": [
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "TileOps2D": [
     "InitMemRef",
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "TileMemoryInferred": [
     "InitMemRef"
    ],
    "NormalizedStmtStructure": [
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "ExpandManualPhaseFence"
    ],
    "TensorViewCanonical": []
   },
   "snippets": []
  },
  {
   "order": 32,
   "name": "InitMemRef",
   "snake": "init_mem_ref",
   "phase": "mem",
   "role": "decide",
   "layer": "memory",
   "brief": "为所有变量建立 MemRef，创建未分配地址的 alloc",
   "detail": "为函数里所有变量初始化 MemRef 并创建 alloc 操作（地址还未分配）。默认内存空间设为 UB，tile.load / tile.store 的操作数设为 DDR。产出 HasMemRefs。",
   "watch": "**它 invalidate SSAForm** —— 这是整条流水线最重要的分界点之一。一旦变量绑上物理 MemRef，「一个名字只被赋值一次」就不再成立，后面所有内存相关的 Pass 都活在非 SSA 的世界里。谁跟谁共用一个 MemRef 的初始判断也在这里定型，而且 alloc 全部被提到函数体头部（这个事实后来被 MemoryReuse 的 largest-first 打包直接利用）。",
   "file": "src/ir/transforms/init_memref.cpp",
   "lines": 962,
   "factory": "Pass InitMemRef() { return CreateFunctionPass(TransformInitMemRef, \"InitMemRef\", kInitMemRefProperties); }",
   "factoryRef": "src/ir/transforms/init_memref.cpp:897",
   "required": [
    "SSAForm",
    "SplitIncoreOrch",
    "IncoreTileOps",
    "TileOps2D",
    "TileMemoryInferred"
   ],
   "produced": [
    "HasMemRefs",
    "NormalizedStmtStructure"
   ],
   "invalidated": [
    "SSAForm"
   ],
   "origin": {
    "SSAForm": "ConvertToSSA",
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "IncoreTileOps": "ConvertTensorToTileOps",
    "TileOps2D": "FlattenTileNdTo2D",
    "TileMemoryInferred": "InferTileMemorySpace"
   },
   "downstream": {
    "HasMemRefs": [
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "AllocateMemoryAddr",
     "FoldNoOpReshape"
    ],
    "NormalizedStmtStructure": [
     "MaterializeSemanticAliases",
     "MemoryReuse",
     "ExpandManualPhaseFence"
    ]
   },
   "snippets": []
  },
  {
   "order": 33,
   "name": "MaterializeSemanticAliases",
   "snake": "materialize_semantic_aliases",
   "phase": "mem",
   "role": "decide",
   "layer": "memory",
   "brief": "把语义要求必须同一块分配的 buffer 强制合并",
   "detail": "沿 yield / producer 链把每个 loop-carried iter_arg / initValue 的 MemRef 传播下去，让累加器的生产者直接写进被携带的 buffer。这是**语义要求**的别名（循环累加器必须活在同一块 buffer 里），从 MemoryReuse 里拆出来，好让它在不跑机会性生命周期复用时也能运行（DSA-RP 或 ptoas 接管复用时）。",
   "watch": "拆分本身说明了一件事：**「必须合并」和「可以合并」是两种性质完全不同的决策**，混在一个 Pass 里会让人分不清哪些复用是省内存、哪些是语义正确性要求。",
   "file": "src/ir/transforms/memory_reuse_pass.cpp",
   "lines": 3325,
   "factory": "Pass MaterializeSemanticAliases() {\n  return CreateFunctionPass(TransformMaterializeSemanticAliases, \"MaterializeSemanticAliases\",\n                            kMaterializeSemanticAliasesProperties);\n}",
   "factoryRef": "src/ir/transforms/memory_reuse_pass.cpp:3318",
   "required": [
    "SplitIncoreOrch",
    "IncoreTileOps",
    "HasMemRefs",
    "TileOps2D",
    "NormalizedStmtStructure"
   ],
   "produced": [
    "NormalizedStmtStructure"
   ],
   "invalidated": [],
   "origin": {
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "IncoreTileOps": "ConvertTensorToTileOps",
    "HasMemRefs": "InitMemRef",
    "TileOps2D": "FlattenTileNdTo2D",
    "NormalizedStmtStructure": "NormalizeStmtStructure"
   },
   "downstream": {
    "NormalizedStmtStructure": [
     "MemoryReuse",
     "ExpandManualPhaseFence"
    ]
   },
   "snippets": []
  },
  {
   "order": 34,
   "name": "MemoryReuse",
   "snake": "memory_reuse",
   "phase": "mem",
   "role": "decide",
   "layer": "memory",
   "brief": "按生命周期把互不重叠的 buffer 合并，删掉多余 alloc",
   "detail": "核心是 can_share 判据（见右侧代码）：生命周期重叠、hazard、forbid-alias、pipeline 冲突、Vec ND/NZ 不兼容——五道门任一命中即拒绝。通过的按内存空间分组，组内按**尺寸从大到小**排序做 first-fit 打包：每个后来的区间加入第一个「与其所有成员都能共享」的 buffer。buffer 大小由它第一个（最大的）成员决定，所以后面接纳更小的成员零成本。",
   "watch": "**这是整条流水线最锋利的决策点**。禁令来自 ForbidAliasCollector 的三个来源：not_inplace_safe()（输出不得与任何输入共享）、forbid_output_alias(i)（不得与指定操作数共享）、加宽 cast（写游标跑赢读游标）。理由今天只走 LOG_DEBUG，**唯一对外发声的是流水深度降级 PH-MR-001**。pypto#1475（open）：它把两个独立 matmul 的 K buffer 合并，制造出一条只因复用而存在的 MTE1→MTE2 WAR 边，cube 上两个 matmul 从并行变串行。pypto#2007：tile.concat 漏标 not_inplace_safe，dst 被别名到 src，设备上静默返回错数据。",
   "file": "src/ir/transforms/memory_reuse_pass.cpp",
   "lines": 3325,
   "factory": "Pass MemoryReuse() { return CreateFunctionPass(TransformMemoryReuse, \"MemoryReuse\", kMemoryReuseProperties); }",
   "factoryRef": "src/ir/transforms/memory_reuse_pass.cpp:3322",
   "required": [
    "SplitIncoreOrch",
    "IncoreTileOps",
    "HasMemRefs",
    "TileOps2D",
    "NormalizedStmtStructure"
   ],
   "produced": [
    "NormalizedStmtStructure"
   ],
   "invalidated": [],
   "origin": {
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "IncoreTileOps": "ConvertTensorToTileOps",
    "HasMemRefs": "InitMemRef",
    "TileOps2D": "FlattenTileNdTo2D",
    "NormalizedStmtStructure": "NormalizeStmtStructure"
   },
   "downstream": {
    "NormalizedStmtStructure": [
     "ExpandManualPhaseFence"
    ]
   },
   "snippets": [
    {
     "file": "src/ir/transforms/memory_reuse_pass.cpp",
     "from": 2173,
     "to": 2184,
     "label": "can_share：五道拒绝门",
     "code": "auto can_share = [&](const LifetimeInterval& cand, const LifetimeInterval& member) {\n  // Group-interval overlap is a fast reject; when it fires, fall back to the\n  // precise per-var check so mutually-exclusive / same-value phi-family tiles\n  // may still share while a genuine conflict (incl. one hidden behind a\n  // branch-local alias of an outside-live buffer) is still caught.\n  if (LifetimesOverlap(cand, member) && overlap_blocks_sharing(cand, member)) return false;\n  if (hazard_blocks(cand, member) || hazard_blocks(member, cand)) return false;\n  if (forbid_blocks(cand, member) || forbid_blocks(member, cand)) return false;\n  if (pipeline_blocks(cand, member)) return false;  // symmetric — one call suffices\n  if (!AreVecNdNzCompatible(cand.variable, member.variable)) return false;\n  return true;\n};"
    },
    {
     "file": "src/ir/transforms/memory_reuse_pass.cpp",
     "from": 2363,
     "to": 2372,
     "label": "PH-MR-001 的设计意图（注释原文）",
     "code": "// Loud diagnostic (perf hint): a pipeline group whose achieved depth F_g fell below the requested D_g\n// means the capacity gate could not honor the programmer's `pl.pipeline(stage=D)` — stages k and k+F_g\n// share a buffer and re-serialize (the false WAR the pipeline meant to avoid). Surface it (correct, just\n// slower) with the concrete fix, rather than silently degrading. Spaces that hit the legacy fallback\n// already emitted a Warning above, so skip them here to avoid a double signal.\nif (out_hints != nullptr) {\n  for (const auto& [key, achieved] : group_depth) {\n    if (capacity_known_spaces.count(key.first) == 0) continue;  // unknown capacity ⇒ not gated\n    if (force_legacy_spaces.count(key.first) != 0) continue;    // already warned at the space level\n    auto rit = group_requested_depth.find(key);"
    },
    {
     "file": "src/ir/transforms/memory_reuse_pass.cpp",
     "from": 2398,
     "to": 2416,
     "label": "PH-MR-001：把决定、代价与两条出口一次说清",
     "code": "  msg << \"software pipelining requested depth \" << requested << \" for pipeline group \" << key.second\n      << \" in \" << MemorySpaceToString(key.first) << \", but only \" << achieved << \" of \" << requested\n      << \" buffers fit (\" << aligned_slot << \" B per stage, \" << free_cap << \" B free\";\n  if (reserved != 0) msg << \" after \" << reserved << \" B reserved\";\n  msg << \") — stages \" << achieved << \" apart share storage and serialize. \";\n  if (slot_bound) {\n    msg << \"This operand alone needs \" << need << \" B for depth \" << requested\n        << \"; shrink the per-stage tile to <= \"\n        << (requested > 0 ? free_cap / static_cast<uint64_t>(requested) : free_cap)\n        << \" B, or reduce the pipeline depth (e.g. `pl.pipeline(stage=)`) to \" << achieved << \".\";\n  } else {\n    msg << \"The operand would fit depth \" << requested\n        << \" on its own, but co-resident buffers / other pipeline groups over-subscribe the space; \"\n        << \"relieve the co-residents (smaller or fewer co-live tiles) or reduce the pipeline depth to \"\n        << achieved << \".\";\n  }\n  out_hints->emplace_back(DiagnosticSeverity::PerfHint, \"MemoryReuse\", 0, \"PH-MR-001\", msg.str(),\n                          func ? func->span_ : Span::unknown());\n}"
    }
   ]
  },
  {
   "order": 35,
   "name": "AllocateMemoryAddr",
   "snake": "allocate_memory_addr",
   "phase": "mem",
   "role": "decide",
   "layer": "memory",
   "brief": "给已有 alloc 分配真实地址",
   "detail": "为已存在 alloc 的 MemRef 分配真实内存地址，就地更新 MemRef 地址和 alloc 语句的实参。产出 AllocatedMemoryAddr（所有 MemRef 都有合法且在 buffer 上限内的地址）。",
   "watch": "**packing 顺序与对齐策略直接决定 high-water**。DSA-RP 模式下这里还要在容量与复用惩罚下联合求解——注释写明 MemoryReuse 在 DSA_RP 下会被跳过，因为「先合并会在 DSA-RP 评估之前就抹掉备选方案」。已有理由码 PH-DSA-001。#1908 请求 liveness-aware 的偏移打包。",
   "file": "src/ir/transforms/allocate_memory_addr_pass.cpp",
   "lines": 772,
   "factory": "Pass AllocateMemoryAddr() {\n  return CreateFunctionPass(TransformAllocateMemoryAddr, \"AllocateMemoryAddr\", kAllocateMemoryAddrProperties);\n}",
   "factoryRef": "src/ir/transforms/allocate_memory_addr_pass.cpp:639",
   "required": [
    "SplitIncoreOrch",
    "IncoreTileOps",
    "HasMemRefs",
    "TileOps2D"
   ],
   "produced": [
    "AllocatedMemoryAddr"
   ],
   "invalidated": [],
   "origin": {
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "IncoreTileOps": "ConvertTensorToTileOps",
    "HasMemRefs": "InitMemRef",
    "TileOps2D": "FlattenTileNdTo2D"
   },
   "downstream": {
    "AllocatedMemoryAddr": []
   },
   "snippets": []
  },
  {
   "order": 36,
   "name": "FoldNoOpReshape",
   "snake": "fold_no_op_reshape",
   "phase": "mem",
   "role": "mech",
   "layer": null,
   "brief": "折叠既不改形状也不改分配的 reshape",
   "detail": "折叠那些既不改变物理形状、也不改变分配的 tile.reshape 调用。",
   "watch": "",
   "file": "src/ir/transforms/fold_no_op_reshape_pass.cpp",
   "lines": 96,
   "factory": "Pass FoldNoOpReshape() {\n  auto pass_func = [](const FunctionPtr& func) -> FunctionPtr {\n    if (!func || !func->body_) return func;\n    if (!IsInCoreType(func->func_type_)) return func;\n    FoldNoOpReshapeMutator mutator;\n    auto new_body = mutator.VisitStmt(func->body_);\n    if (new_body.get() == func->body_.get()) return func;\n    return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,\n                                      func->return_types_, new_body, func->span_, func->func_type_,\n                                      func->level_, func->role_, func->attrs_);\n  };\n  return CreateFunctionPass(pass_func, \"FoldNoOpReshape\", kFoldNoOpReshapeProperties);\n}",
   "factoryRef": "src/ir/transforms/fold_no_op_reshape_pass.cpp:80",
   "required": [
    "SplitIncoreOrch",
    "IncoreTileOps",
    "HasMemRefs",
    "TileOps2D"
   ],
   "produced": [],
   "invalidated": [],
   "origin": {
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "IncoreTileOps": "ConvertTensorToTileOps",
    "HasMemRefs": "InitMemRef",
    "TileOps2D": "FlattenTileNdTo2D"
   },
   "downstream": {},
   "snippets": []
  },
  {
   "order": 37,
   "name": "FuseCreateAssembleToSlice",
   "snake": "fuse_create_assemble_to_slice",
   "phase": "mem",
   "role": "mech",
   "layer": null,
   "brief": "create + assemble 融成一个 slice 视图",
   "detail": "把 tensor.create + tensor.assemble 配对融成一个 tensor.slice 视图，消掉中间 buffer。",
   "watch": "pypto#1006 记录过它在形状 padding 之后没更新变量类型。",
   "file": "src/ir/transforms/fuse_create_assemble_to_slice_pass.cpp",
   "lines": 429,
   "factory": "Pass FuseCreateAssembleToSlice() {\n  return CreateProgramPass(TransformFuseCreateAssembleToSlice, \"FuseCreateAssembleToSlice\",\n                           kFuseCreateAssembleToSliceProperties);\n}",
   "factoryRef": "src/ir/transforms/fuse_create_assemble_to_slice_pass.cpp:421",
   "required": [],
   "produced": [],
   "invalidated": [],
   "origin": {},
   "downstream": {},
   "snippets": []
  },
  {
   "order": 38,
   "name": "DeriveCallDirections",
   "snake": "derive_call_directions",
   "phase": "dep",
   "role": "decide",
   "layer": "deps",
   "brief": "推导每个实参的 In / InOut / Output 方向",
   "detail": "两个阶段。Phase 1：把每个 Group/Spmd wrapper 的有效 ParamDirection 落到签名上，然后遍历函数体，为每个跨函数 Call 的每个实参从被调方 ParamDirection 和 buffer 血缘推导出 ArgDirection。Phase 2 做 manual-scope 下降。产出 CallDirectionsResolved。",
   "watch": "**判定规则在代码里有名字**（见右侧）：R-seq（有顺序祖先就强制 InOut，保跨迭代 WAW 链正确）、R-prior（本 scope 内已有前序写者 → InOut）、R-enclosing（外层参数被用户声明 InOut 就尊重声明）。都不命中才给 OutputExisting。**方向判定是 WAR 丢失的源头之一**：读者是纯 Input 就不产生新版本，runtime 也就看不到需要排序的写者。",
   "file": "src/ir/transforms/derive_call_directions_pass.cpp",
   "lines": 619,
   "factory": "Pass DeriveCallDirections() {\n  auto pass_func = [](const ProgramPtr& input_program) -> ProgramPtr {\n    if (!input_program) return input_program;\n\n    // Phase 0: recover Group/Spmd wrapper signatures. Every later phase — and\n    // every downstream consumer — reads ``callee->param_directions_`` directly.\n    ProgramPtr program = MaterializeWrapperDirections(input_program);\n\n    // We need a non-const handle to rewrite functions with new bodies.\n    auto new_functions = program->functions_;\n\n    for (auto& [gvar, func] : new_functions) {\n      if (!func || !func->body_) continue;\n\n      // kFirstOutput: direction derivation must keep *some* root for an\n      // ambiguous single-return call, else a later write to the returned var\n      // skips the R-prior / enclosing-param InOut promotion and silently drops\n      // the WAW/InOut dependency. Preserves the naive pre-dedup behavior.\n      BufferRootCollector br_collector(program, AmbiguousRootPolicy::kFirstOutput);\n      br_collector.Initialize(func->params_);\n      br_collector.VisitStmt(func->body_);\n\n      // Build a Var* → ParamDirection map for the enclosing function's params,\n      // so call sites can honor an explicit ``pl.InOut`` declaration when the\n      // arg traces back to such a param via the buffer-root map.\n      std::unordered_map<const Var*, ParamDirection> enclosing_param_dir_by_root;\n      enclosing_param_dir_by_root.reserve(func->params_.size());\n      for (size_t i = 0; i < func->params_.size() && i < func->param_directions_.size(); ++i) {\n        enclosing_param_dir_by_root.emplace(func->params_[i].get(), func->param_directions_[i]);\n      }\n\n      PriorWriterCollector pw_collector(program, br_collector.buffer_roots);\n      pw_collector.Run(func->body_);\n\n      CallDirectionMutator mutator(program, br_collector.buffer_roots, br_collector.ambiguous_buffer_vars,\n                                   pw_collector.first_writer_roots, enclosing_param_dir_by_root);\n      auto new_body = mutator.VisitStmt(func->body_);\n\n      if (new_body.get() == func->body_.get()) continue;\n      func = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,\n                                        func->return_types_, new_body, func->span_, func->func_type_,\n                                        func->level_, func->role_, func->attrs_);\n    }\n\n    if (new_functions == program->functions_) return program;\n    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);\n  };\n\n  return CreateProgramPass(pass_func, \"DeriveCallDirections\", kDeriveCallDirectionsProperties);\n}",
   "factoryRef": "src/ir/transforms/derive_call_directions_pass.cpp:566",
   "required": [
    "SplitIncoreOrch"
   ],
   "produced": [
    "CallDirectionsResolved"
   ],
   "invalidated": [],
   "origin": {
    "SplitIncoreOrch": "OutlineIncoreScopes"
   },
   "downstream": {
    "CallDirectionsResolved": [
     "AutoDeriveTaskDependencies",
     "ExpandManualPhaseFence",
     "MaterializeRuntimeScopes",
     "ClassifyIterArgCarry"
    ]
   },
   "snippets": [
    {
     "file": "src/ir/transforms/derive_call_directions_pass.cpp",
     "from": 482,
     "to": 523,
     "label": "方向判定的三条具名规则：R-seq / R-prior / R-enclosing",
     "code": "  // ParamDirection::Out — apply the promotion rules uniformly to both\n  // locally-allocated roots and roots that trace back to an enclosing\n  // function parameter.\n  const Var* root = ResolveAnyRoot(arg, buffer_roots_);\n\n  // R-seq: any sequential ancestor forces InOut to keep cross-iteration\n  // WAW chains correct. Applied unconditionally — see the class comment\n  // for why the prior \"disjoint variable-offset store\" exception was\n  // removed.\n  if (sequential_depth_ > 0) {\n    dirs.push_back(ArgDirection::InOut);\n    continue;\n  }\n  // A control-flow result may refer to different buffers on different\n  // paths (zero-trip loop, while, or differing if branches). A single\n  // canonical root cannot prove first-writer status, so retain the\n  // dependency conservatively.\n  if (auto var = AsVarLike(arg); var && ambiguous_buffer_vars_.count(var.get()) != 0) {\n    dirs.push_back(ArgDirection::InOut);\n    continue;\n  }\n  // R-prior: a prior writer-unit in this scope already wrote to this root → InOut.\n  if (root) {\n    bool is_first_writer = first_writer_set != nullptr && first_writer_set->count(root) > 0;\n    if (!is_first_writer) {\n      dirs.push_back(ArgDirection::InOut);\n      continue;\n    }\n  }\n  // R-enclosing: if the root is an enclosing function param that the user\n  // declared InOut, honor that declaration — the function effectively reads\n  // the prior-call value and writes a new one back into the same buffer.\n  if (root) {\n    auto it = enclosing_param_dir_by_root_.find(root);\n    if (it != enclosing_param_dir_by_root_.end() && it->second == ParamDirection::InOut) {\n      dirs.push_back(ArgDirection::InOut);\n      continue;\n    }\n  }\n  // Default: first writer, no sequential ancestor, no InOut declaration → OutputExisting.\n  dirs.push_back(ArgDirection::OutputExisting);\n}"
    }
   ]
  },
  {
   "order": 39,
   "name": "AutoDeriveTaskDependencies",
   "snake": "auto_derive_task_dependencies",
   "phase": "dep",
   "role": "decide",
   "layer": "deps",
   "brief": "推导任务间依赖边；证明不了就回退",
   "detail": "在 AUTO scope 内推导保守的任务间依赖边。先建一张保守的 tensor 存储位置图（直接别名、loop carry、tuple 元素、tensor.assemble、跨函数输出都继承同一个 storage root），再逐 scope 按源码顺序遍历、维护前序访问记录。编译器边写进 attrs['compiler_manual_dep_edges']，用户写的 deps 留在 attrs['manual_dep_edges'] —— **两者故意分开存**，属性声明里的原话是 so IR dumps preserve provenance。",
   "watch": "代码里有一条 needs_fallback 路径：SummarizeAccesses 判定证明不了覆盖，就 MarkCurrentScopeFallback() 并原样返回。**这条回退就是 pypto#1744 的降级**——`arr[i]` 形式的 TaskId 依赖 Pass 静态匹配不上，判定「未被用户边覆盖」，于是剥掉 scope 降级成 AUTO，runtime 重新接管并保守全串行。用户写 manual_scope 就是为了避免串行。另外 **WAR 反依赖至今不推**（pypto#2058 open）：RAW/WAW 都推得出，唯独 loop-carried buffer 的 reader(N)→writer(N+1) 边没有，导致静默数据竞争。",
   "file": "src/ir/transforms/auto_derive_task_dependencies_pass.cpp",
   "lines": 2304,
   "factory": "Pass AutoDeriveTaskDependencies(bool analyze_auto_scopes) {\n  auto pass_func = [analyze_auto_scopes](const ProgramPtr& program) -> ProgramPtr {\n    if (!program) return program;\n\n    auto new_functions = program->functions_;\n    bool changed = false;\n\n    for (auto& [gvar, func] : new_functions) {\n      (void)gvar;\n      if (!func || !func->body_) continue;\n\n      StorageRootAnalysis storage(program);\n      storage.Initialize(func->params_);\n      storage.VisitStmt(func->body_);\n\n      SubmitTaskIdCollector task_ids;\n      task_ids.VisitStmt(func->body_);\n\n      const bool analyze_whole_body_as_auto_scope = analyze_auto_scopes &&\n                                                    func->func_type_ == FunctionType::Orchestration &&\n                                                    func->GetAttr<bool>(\"auto_scope\", true);\n      AutoDepMutator mutator(program, &storage, &task_ids.task_id_by_expr(), &task_ids.task_id_by_var_id(),\n                             &task_ids.task_ids_by_var_id(), &task_ids.task_id_dynamic_slots_by_var_id(),\n                             &task_ids.task_id_array_extent_by_var_id(), &task_ids.task_ids_by_array_var_id(),\n                             &task_ids.complete_task_id_array_var_ids(), analyze_auto_scopes,\n                             analyze_whole_body_as_auto_scope);\n      auto new_body = mutator.AnalyzeBody(func->body_);\n      const bool whole_body_manual_candidate = mutator.whole_body_manual_candidate();\n      const bool had_whole_body_manual_candidate =\n          func->GetAttr<bool>(kAttrCompilerAutoManualLayerCandidate, false);\n      if (new_body.get() == func->body_.get() &&\n          whole_body_manual_candidate == had_whole_body_manual_candidate) {\n        continue;\n      }\n\n      changed = true;\n      auto new_attrs = StripAttr(func->attrs_, kAttrCompilerAutoManualLayerCandidate);\n      if (whole_body_manual_candidate) {\n        new_attrs = WithBoolAttr(std::move(new_attrs), kAttrCompilerAutoManualLayerCandidate, true);\n      }\n      func = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,\n                                        func->return_types_, new_body, func->span_, func->func_type_,\n                                        func->level_, func->role_, std::move(new_attrs));\n    }\n\n    if (!changed) return program;\n    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);\n  };\n\n  return CreateProgramPass(pass_func, \"AutoDeriveTaskDependencies\", kAutoDeriveTaskDependenciesProperties);\n}",
   "factoryRef": "src/ir/transforms/auto_derive_task_dependencies_pass.cpp:2250",
   "required": [
    "SplitIncoreOrch",
    "CallDirectionsResolved"
   ],
   "produced": [
    "CallDirectionsResolved"
   ],
   "invalidated": [],
   "origin": {
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "CallDirectionsResolved": "DeriveCallDirections"
   },
   "downstream": {
    "CallDirectionsResolved": [
     "ExpandManualPhaseFence",
     "MaterializeRuntimeScopes",
     "ClassifyIterArgCarry"
    ]
   },
   "snippets": [
    {
     "file": "src/ir/transforms/auto_derive_task_dependencies_pass.cpp",
     "from": 1635,
     "to": 1642,
     "label": "证明不了就回退 —— pypto#1744 的降级入口",
     "code": "auto user_edges = CanonicalizeTaskIds(raw_user_edges);\nauto summary = SummarizeAccesses(call, user_edges, &needs_fallback);\nif (needs_fallback) {\n  MarkCurrentScopeFallback();\n  return call;\n}\nauto& accesses = summary.accesses;\nif (accesses.empty()) return call;"
    }
   ]
  },
  {
   "order": 40,
   "name": "ExpandManualPhaseFence",
   "snake": "expand_manual_phase_fence",
   "phase": "dep",
   "role": "decide",
   "layer": "deps",
   "brief": "把划算的整数组 TaskId 依赖压成相位栅栏",
   "detail": "在 manual scope 里把划算的整数组 TaskId 依赖压缩成相位栅栏，减少依赖边数量。",
   "watch": "「划算」是启发式判断，所以是决策点。simpler#412 记录过 fan-in ≥16 会导致静默依赖截断——依赖边太多本身就是个真实问题。",
   "file": "src/ir/transforms/expand_manual_phase_fence_pass.cpp",
   "lines": 505,
   "factory": "Pass ExpandManualPhaseFence() {\n  return CreateProgramPass(TransformExpandManualPhaseFence, \"ExpandManualPhaseFence\",\n                           kExpandManualPhaseFenceProperties);\n}",
   "factoryRef": "src/ir/transforms/expand_manual_phase_fence_pass.cpp:498",
   "required": [
    "NoNestedCalls",
    "NormalizedStmtStructure",
    "CallDirectionsResolved"
   ],
   "produced": [
    "NoNestedCalls",
    "NormalizedStmtStructure",
    "CallDirectionsResolved"
   ],
   "invalidated": [],
   "origin": {
    "NoNestedCalls": "FlattenCallExpr",
    "NormalizedStmtStructure": "NormalizeStmtStructure",
    "CallDirectionsResolved": "DeriveCallDirections"
   },
   "downstream": {
    "NoNestedCalls": [],
    "NormalizedStmtStructure": [],
    "CallDirectionsResolved": [
     "MaterializeRuntimeScopes",
     "ClassifyIterArgCarry"
    ]
   },
   "snippets": []
  },
  {
   "order": 41,
   "name": "SynthesizeAllReduceSignals",
   "snake": "synthesize_allreduce_signals",
   "phase": "dist",
   "role": "mech",
   "layer": null,
   "brief": "把省略 signal 的 host allreduce 规范成显式形式",
   "detail": "把 host 级 allreduce 调用里省略了 signal 的形式，规范成显式的 internal allreduce(data, signal, op=...) IR。",
   "watch": "",
   "file": "src/ir/transforms/synthesize_allreduce_signals_pass.cpp",
   "lines": 309,
   "factory": "Pass SynthesizeAllReduceSignals() {\n  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {\n    NameCollector name_collector;\n    name_collector.VisitProgram(program);\n    int64_t next_signal_id = 0;\n\n    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;\n    bool modified = false;\n    for (const auto& [gvar, func] : program->functions_) {\n      if (!IsHostOrch(func)) {\n        new_functions[gvar] = func;\n        continue;\n      }\n\n      AllReduceSignalSynthesizer synthesizer(&name_collector.names, &next_signal_id);\n      auto new_body = synthesizer.VisitStmt(func->body_);\n      if (!synthesizer.modified()) {\n        new_functions[gvar] = func;\n        continue;\n      }\n\n      auto new_func = MutableCopy(func);\n      new_func->body_ = new_body;\n      new_functions[gvar] = new_func;\n      modified = true;\n    }\n\n    if (!modified) return program;\n    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);\n  };\n\n  return CreateProgramPass(pass_func, \"SynthesizeAllReduceSignals\", kSynthesizeAllReduceSignalsProperties);\n}",
   "factoryRef": "src/ir/transforms/synthesize_allreduce_signals_pass.cpp:273",
   "required": [],
   "produced": [],
   "invalidated": [],
   "origin": {},
   "downstream": {},
   "snippets": []
  },
  {
   "order": 42,
   "name": "MaterializeCommDomainScopes",
   "snake": "materialize_comm_domain_scopes",
   "phase": "dist",
   "role": "mech",
   "layer": null,
   "brief": "装配 WindowBuffer 与通信域 scope",
   "detail": "遍历每个 host_orch，追踪 pld.tensor.alloc_window_buffer → pld.tensor.window → dispatch(device=r) / allreduce 链，在每个 DistributedTensorType 视图上物化 WindowBuffer 反向引用，并把 host_orch 函数体包进嵌套的 CommDomainScopeStmt。产出 CommDomainScopesMaterialized。",
   "watch": "属性声明的注释解释了为什么排这么晚：**host_orch 从不做 tile 下降，所以 alloc/window/dispatch/allreduce 链一直可发现**——放在最后反而最安全。",
   "file": "src/ir/transforms/materialize_comm_domain_scopes_pass.cpp",
   "lines": 755,
   "factory": "Pass MaterializeCommDomainScopes() {\n  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {\n    // Index chip-level Orchestration functions by name so the dispatch\n    // analyzer can recognise host → chip Calls.\n    std::map<std::string, FunctionPtr> chip_orchs;\n    for (const auto& [gv, func] : program->functions_) {\n      if (IsChipOrch(func)) chip_orchs[func->name_] = func;\n    }\n\n    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;\n    bool modified = false;\n\n    for (const auto& [gvar, func] : program->functions_) {\n      if (!IsHostOrch(func)) {\n        new_functions[gvar] = func;\n        continue;\n      }\n      auto new_func = ProcessHostOrch(func, chip_orchs);\n      new_functions[gvar] = new_func;\n      if (new_func.get() != func.get()) modified = true;\n    }\n\n    if (!modified) return program;\n    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);\n  };\n\n  return CreateProgramPass(pass_func, \"MaterializeCommDomainScopes\", kMaterializeCommDomainScopesProperties);\n}",
   "factoryRef": "src/ir/transforms/materialize_comm_domain_scopes_pass.cpp:655",
   "required": [],
   "produced": [
    "CommDomainScopesMaterialized"
   ],
   "invalidated": [],
   "origin": {},
   "downstream": {
    "CommDomainScopesMaterialized": [
     "LowerHostTensorCollectives",
     "MaterializeDistTensorCtx"
    ]
   },
   "snippets": []
  },
  {
   "order": 43,
   "name": "LowerHostTensorCollectives",
   "snake": "lower_host_tensor_collectives",
   "phase": "dist",
   "role": "mech",
   "layer": null,
   "brief": "host 级集合通信改写成 chip dispatch",
   "detail": "把 host 编排器对 tensor 集合通信的调用，改写成内部 builtin 的 chip dispatch。",
   "watch": "",
   "file": "src/ir/transforms/lower_host_tensor_collectives_pass.cpp",
   "lines": 714,
   "factory": "Pass LowerHostTensorCollectives() {\n  return CreateProgramPass(TransformProgram, \"LowerHostTensorCollectives\",\n                           kLowerHostTensorCollectivesProperties);\n}",
   "factoryRef": "src/ir/transforms/lower_host_tensor_collectives_pass.cpp:706",
   "required": [
    "CommDomainScopesMaterialized"
   ],
   "produced": [
    "CommDomainScopesMaterialized"
   ],
   "invalidated": [],
   "origin": {
    "CommDomainScopesMaterialized": "MaterializeCommDomainScopes"
   },
   "downstream": {
    "CommDomainScopesMaterialized": [
     "MaterializeDistTensorCtx"
    ]
   },
   "snippets": []
  },
  {
   "order": 44,
   "name": "MaterializeDistTensorCtx",
   "snake": "materialize_dist_tensor_ctx",
   "phase": "dist",
   "role": "mech",
   "layer": null,
   "brief": "为每个 DistributedTensor 物化 CommCtx 参数",
   "detail": "为每个 DistributedTensor 物化一个显式的 CommCtxType 参数与实参。",
   "watch": "pypto#1913 记录过 spmd 任务提交漏了 DistributedTensor 的 CommContext 实参，跨 rank 的 notify/wait/put 直接死锁（AICore 507015）。",
   "file": "src/ir/transforms/materialize_dist_tensor_ctx_pass.cpp",
   "lines": 418,
   "factory": "Pass MaterializeDistTensorCtx() {\n  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr { return TransformProgram(program); };\n  return CreateProgramPass(pass_func, \"MaterializeDistTensorCtx\", kMaterializeDistTensorCtxProperties);\n}",
   "factoryRef": "src/ir/transforms/materialize_dist_tensor_ctx_pass.cpp:411",
   "required": [
    "CommDomainScopesMaterialized"
   ],
   "produced": [
    "CommDomainScopesMaterialized"
   ],
   "invalidated": [],
   "origin": {
    "CommDomainScopesMaterialized": "MaterializeCommDomainScopes"
   },
   "downstream": {
    "CommDomainScopesMaterialized": []
   },
   "snippets": []
  },
  {
   "order": 45,
   "name": "Simplify",
   "snake": "simplify",
   "phase": "dist",
   "role": "mech",
   "layer": null,
   "brief": "折叠算术、shape 表达式与标量常量（第二次调用）",
   "detail": "用代数重写规则和区间分析折叠三类东西：算术表达式、类型注解里内嵌的 shape 表达式、标量常量绑定。比如把 `CHUNK_K: Scalar[INDEX] = 512` 的值传播到所有下游使用点。",
   "watch": "**它在流水线里跑两次**：第一次紧跟 ConvertToSSA（注释写明「跑在 SSA 之后，利用单定义性质」），第二次在分布式 Pass 之后收尾。pypto#1461 记录过它曾把承载语义的 output alias 赋值当成可折叠的 cast 而删掉，导致返回张量读回全零。 这里是第二次调用：分布式 Pass 把 host 集合通信改写成 chip dispatch 之后会留下一批可折叠的常量表达式，需要再收一遍。",
   "file": "src/ir/transforms/simplify_pass.cpp",
   "lines": 846,
   "factory": "Pass Simplify() { return CreateProgramPass(TransformSimplifyProgram, \"Simplify\", kSimplifyProperties); }",
   "factoryRef": "src/ir/transforms/simplify_pass.cpp:841",
   "required": [],
   "produced": [],
   "invalidated": [],
   "origin": {},
   "downstream": {},
   "snippets": []
  },
  {
   "order": 46,
   "name": "MaterializeRuntimeScopes",
   "snake": "materialize_runtime_scopes",
   "phase": "tail",
   "role": "decide",
   "layer": "deps",
   "brief": "插入显式 AUTO RuntimeScopeStmt",
   "detail": "往 Orchestration 函数里插入显式的 AUTO RuntimeScopeStmt 节点（函数体 + for/if 体），使 codegen 能 1:1 发出 PTO2_SCOPE，不再需要隐式包装。产出 RuntimeScopesMaterialized。",
   "watch": "**它消费 AutoDeriveTaskDependencies 留下的标记**，决定这一段最终发 AUTO 还是编译器自有的 MANUAL。位置注释写明：跑在最后一次 Simplify 与所有重写变换之后，这样没有任何变换需要考虑这些插进来的 scope 包装。",
   "file": "src/ir/transforms/materialize_runtime_scopes_pass.cpp",
   "lines": 286,
   "factory": "Pass MaterializeRuntimeScopes() {\n  auto pass_func = [](const FunctionPtr& func) -> FunctionPtr {\n    if (!func || !func->body_) return func;\n    // Only Orchestration functions are wrapped in PTO2_SCOPE blocks by codegen;\n    // InCore/AIC/AIV/Group/Spmd bodies are never scope-wrapped.\n    if (func->func_type_ != FunctionType::Orchestration) return func;\n\n    // ``@pl.function(auto_scope=False)`` opts out of automatic AUTO-scope\n    // insertion: the user places every scope by hand (``with pl.scope()`` /\n    // ``pl.manual_scope()``), which the parser already materialised into the\n    // IR. The simpler runtime's implicit top-level scope covers correctness, so\n    // emitting zero compiler scopes is valid. Leave such functions untouched.\n    if (!func->GetAttr<bool>(\"auto_scope\", true)) return func;\n\n    const bool whole_layer_manual = func->GetAttr<bool>(kAttrCompilerAutoManualLayerCandidate, false);\n    StmtPtr inner = func->body_;\n    if (!whole_layer_manual) {\n      InsertAutoScopeMutator mutator;\n      inner = mutator.VisitStmt(func->body_);\n    }\n\n    // Always wrap the whole function body in an AUTO scope, matching the\n    // always-on outermost ``PTO2_SCOPE()`` codegen emitted at function entry.\n    StmtPtr new_body = whole_layer_manual ? WrapCompilerAutoManualLayer(inner)\n                                          : (IsAutoScope(inner) ? inner : WrapAuto(inner));\n    new_body = StripCompilerAutoManualCallCandidates(new_body);\n\n    // Mark the function ``auto_scope=False`` now that scopes are materialized.\n    // This makes the pass idempotent (a second run early-returns) and lets the\n    // output round-trip: the inserted ``with pl.scope()`` blocks parse back only\n    // under ``auto_scope=False`` (the parser rejects hand-placed AUTO scopes in\n    // the default auto_scope=True mode, where the compiler owns placement).\n    std::vector<std::pair<std::string, std::any>> new_attrs;\n    new_attrs.reserve(func->attrs_.size() + 1);\n    for (const auto& kv : func->attrs_) {\n      if (kv.first != \"auto_scope\" && kv.first != kAttrCompilerAutoManualLayerCandidate) {\n        new_attrs.push_back(kv);\n      }\n    }\n    new_attrs.emplace_back(\"auto_scope\", std::any(false));\n\n    return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,\n                                      func->return_types_, new_body, func->span_, func->func_type_,\n                                      func->level_, func->role_, std::move(new_attrs));\n  };\n  return CreateFunctionPass(pass_func, \"MaterializeRuntimeScopes\", kMaterializeRuntimeScopesProperties);\n}",
   "factoryRef": "src/ir/transforms/materialize_runtime_scopes_pass.cpp:236",
   "required": [
    "SplitIncoreOrch",
    "CallDirectionsResolved"
   ],
   "produced": [
    "RuntimeScopesMaterialized"
   ],
   "invalidated": [],
   "origin": {
    "SplitIncoreOrch": "OutlineIncoreScopes",
    "CallDirectionsResolved": "DeriveCallDirections"
   },
   "downstream": {
    "RuntimeScopesMaterialized": [
     "ClassifyIterArgCarry"
    ]
   },
   "snippets": []
  },
  {
   "order": 47,
   "name": "ClassifyIterArgCarry",
   "snake": "classify_iter_arg_carry",
   "phase": "tail",
   "role": "mech",
   "layer": null,
   "brief": "把编排层 iter_arg 分类成别名或重绑 carry",
   "detail": "把 Orchestration 里每个 ForStmt 的 iter_arg 分类成平凡别名或需要物化的重绑 carry，并给 manual-scope 的 TaskId 数组 carry 定尺寸，把方案盖在 ForStmt.attrs 上。产出 IterArgCarryClassified。",
   "watch": "**放在 MaterializeRuntimeScopes 之后是刻意的**：注释说这样「被分类的 IR 恰好就是编排 codegen 要下降的那份 IR」——避免分类结果和最终形态对不上。",
   "file": "src/ir/transforms/classify_iter_arg_carry_pass.cpp",
   "lines": 437,
   "factory": "Pass ClassifyIterArgCarry() {\n  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {\n    auto new_functions = program->functions_;\n    for (auto& [gvar, func] : new_functions) {\n      if (!func || !func->body_) continue;\n      // Only Orchestration functions carry loop-carried runtime state that the\n      // orchestration codegen lowers into carry variables / TaskId arrays.\n      if (func->func_type_ != FunctionType::Orchestration) continue;\n\n      IterArgCarryStamper stamper(program);\n      auto new_body = stamper.VisitStmt(func->body_);\n      if (new_body.get() == func->body_.get()) continue;\n      func = std::make_shared<Function>(func->name_, func->params_, func->param_directions_,\n                                        func->return_types_, new_body, func->span_, func->func_type_,\n                                        func->level_, func->role_, func->attrs_);\n    }\n    if (new_functions == program->functions_) return program;\n    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);\n  };\n\n  return CreateProgramPass(pass_func, \"ClassifyIterArgCarry\", kClassifyIterArgCarryProperties);\n}",
   "factoryRef": "src/ir/transforms/classify_iter_arg_carry_pass.cpp:412",
   "required": [
    "CallDirectionsResolved",
    "RuntimeScopesMaterialized"
   ],
   "produced": [
    "IterArgCarryClassified",
    "RuntimeScopesMaterialized"
   ],
   "invalidated": [],
   "origin": {
    "CallDirectionsResolved": "DeriveCallDirections",
    "RuntimeScopesMaterialized": "MaterializeRuntimeScopes"
   },
   "downstream": {
    "IterArgCarryClassified": [],
    "RuntimeScopesMaterialized": []
   },
   "snippets": []
  },
  {
   "order": 48,
   "name": "InsertCommFence",
   "snake": "insert_comm_fence",
   "phase": "tail",
   "role": "decide",
   "layer": "sync",
   "brief": "在发布写与 notify 之间插 cacheinvalid + fence",
   "detail": "实现 data-before-signal 的内存一致性契约。三种情况：本地发布写之后插区域 system.cacheinvalid + GM system.fence；不透明写（Submit 或未注册调用）之后插全 GM cacheinvalid + fence；wait 之后插全 GM cacheinvalid（notify 本身不需要，远端写由它自己的 codegen 处理）。",
   "watch": "**排在倒数第二是刻意的**：注释写明「跑在所有语句重排 Pass 之后，这样插入的算子在 codegen 之前一直紧挨着它的 notify」。pypto#1561 记录过 PTOAS 没在 pto.comm.tnotify 前插 pipe_barrier，signal 抢在远端 MTE3 TSTORE 之前。",
   "file": "src/ir/transforms/insert_comm_fence_pass.cpp",
   "lines": 363,
   "factory": "Pass InsertCommFence() {\n  auto pass_func = [](const FunctionPtr& func) -> FunctionPtr {\n    if (!func || !func->body_) return func;\n    // The data-before-signal contract is an InCore-only concern: the publishing\n    // writes, waits, and the system.cacheinvalid / system.fence markers are\n    // InCore GM builtins. Orchestration / HOST functions only dispatch tasks —\n    // their cross-function calls are not GM publishing writes, and inserting an\n    // InCore builtin there is rejected by orchestration codegen.\n    if (!IsInCoreType(func->func_type_)) return func;\n    InsertCommMarkers mutator;\n    auto new_body = mutator.MarkTopLevel(func->body_);\n    if (new_body.get() == func->body_.get()) return func;\n    return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,\n                                      func->return_types_, new_body, func->span_, func->func_type_,\n                                      func->level_, func->role_, func->attrs_);\n  };\n  return CreateFunctionPass(pass_func, \"InsertCommFence\", kInsertCommFenceProperties);\n}",
   "factoryRef": "src/ir/transforms/insert_comm_fence_pass.cpp:342",
   "required": [
    "SplitIncoreOrch"
   ],
   "produced": [],
   "invalidated": [],
   "origin": {
    "SplitIncoreOrch": "OutlineIncoreScopes"
   },
   "downstream": {},
   "snippets": []
  },
  {
   "order": 49,
   "name": "MaterializeValidShapeSymbols",
   "snake": "materialize_valid_shape_symbols",
   "phase": "tail",
   "role": "mech",
   "layer": null,
   "brief": "把 kernel 绑不上的 valid_shape 符号提成参数",
   "detail": "把每个设备 kernel 绑不上的 valid_shape 符号（既不是物理张量维度、也不是标量参数）变成前置的 Scalar[INDEX] 参数，在每个调用点喂进调用方的真实有效范围。",
   "watch": "**排在最后是刻意的**：注释说它只扩展签名和调用实参列表，而到这里两者都已定型，所以没有后续 Pass 需要为这个追加的参数买单。PTOAS#544 记录过 valid_row/valid_col 运行期为负时 ptoas 静默产出挂死的 kernel。",
   "file": "src/ir/transforms/materialize_valid_shape_symbols_pass.cpp",
   "lines": 352,
   "factory": "Pass MaterializeValidShapeSymbols() {\n  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr { return TransformProgram(program); };\n  return CreateProgramPass(pass_func, \"MaterializeValidShapeSymbols\",\n                           kMaterializeValidShapeSymbolsProperties);\n}",
   "factoryRef": "src/ir/transforms/materialize_valid_shape_symbols_pass.cpp:344",
   "required": [],
   "produced": [],
   "invalidated": [],
   "origin": {},
   "downstream": {},
   "snippets": []
  }
 ],
 "phases": [
  {
   "id": "front",
   "name": "前端与规范化",
   "desc": "把 DSL 变成规范、可分析的 IR"
  },
  {
   "id": "scope",
   "name": "作用域外提",
   "desc": "把 scope 变成独立函数，编排与核内计算分家"
  },
  {
   "id": "tile",
   "name": "Tensor→Tile 与合法化",
   "desc": "进入片上世界：tile 算子、切分、内存空间、布局"
  },
  {
   "id": "split",
   "name": "核拆分与流水调度",
   "desc": "拆 AIC/AIV、跨核错峰、流水多缓冲"
  },
  {
   "id": "mem",
   "name": "内存布局与复用",
   "desc": "建 MemRef、强制别名、机会性复用、分地址"
  },
  {
   "id": "dep",
   "name": "任务方向与依赖",
   "desc": "实参方向、任务依赖边、相位栅栏"
  },
  {
   "id": "dist",
   "name": "分布式与通信域",
   "desc": "集合通信、通信域、CommCtx，以及最后一次 Simplify"
  },
  {
   "id": "tail",
   "name": "运行时 scope 与收尾",
   "desc": "RuntimeScope、carry 分类、fence、valid_shape"
  }
 ],
 "layers": {
  "tiling": "自动切分与调度",
  "memory": "自动内存复用",
  "deps": "自动依赖分析",
  "sync": "自动插入同步"
 },
 "props": {
  "InlineFunctionsEliminated": {
   "produced_by": [
    "InlineFunctions"
   ],
   "required_by": [],
   "invalidated_by": []
  },
  "UnrollResolved": {
   "produced_by": [
    "UnrollLoops"
   ],
   "required_by": [],
   "invalidated_by": []
  },
  "StructuredCtrlFlow": {
   "produced_by": [
    "CtrlFlowTransform"
   ],
   "required_by": [],
   "invalidated_by": []
  },
  "SSAForm": {
   "produced_by": [
    "ConvertToSSA",
    "FlattenCallExpr",
    "OutlineHierarchyScopes",
    "OutlineIncoreScopes",
    "OutlineClusterScopes",
    "ConvertTensorToTileOps",
    "FlattenTileNdTo2D",
    "AutoTileMatmulL0",
    "CanonicalizeTileSlice",
    "InferTileMemorySpace",
    "InsertMxScaleAddr",
    "ResolveBackendOpLayouts",
    "LowerAutoVectorSplit",
    "ExpandMixedKernel",
    "InjectGMPipeBuffer",
    "SplitVectorKernel",
    "SkewCrossCorePipeline",
    "LowerPipelineToSlots",
    "LowerPipelineLoops",
    "CanonicalizeIOOrder",
    "MaterializeTensorStrides"
   ],
   "required_by": [
    "FlattenCallExpr",
    "OutlineHierarchyScopes",
    "OutlineIncoreScopes",
    "OutlineClusterScopes",
    "ConvertTensorToTileOps",
    "FlattenTileNdTo2D",
    "AutoTileMatmulL0",
    "CanonicalizeTileSlice",
    "InferTileMemorySpace",
    "InsertMxScaleAddr",
    "ResolveBackendOpLayouts",
    "LowerAutoVectorSplit",
    "ExpandMixedKernel",
    "InjectGMPipeBuffer",
    "SplitVectorKernel",
    "SkewCrossCorePipeline",
    "LowerPipelineToSlots",
    "LowerPipelineLoops",
    "CanonicalizeIOOrder",
    "MaterializeTensorStrides",
    "InitMemRef"
   ],
   "invalidated_by": [
    "InitMemRef"
   ]
  },
  "NormalizedStmtStructure": {
   "produced_by": [
    "NormalizeStmtStructure",
    "FlattenCallExpr",
    "ConvertTensorToTileOps",
    "FlattenTileNdTo2D",
    "AutoTileMatmulL0",
    "CanonicalizeTileSlice",
    "InferTileMemorySpace",
    "InsertMxScaleAddr",
    "ResolveBackendOpLayouts",
    "LowerAutoVectorSplit",
    "ExpandMixedKernel",
    "InjectGMPipeBuffer",
    "SplitVectorKernel",
    "SkewCrossCorePipeline",
    "LowerPipelineToSlots",
    "LowerPipelineLoops",
    "CanonicalizeIOOrder",
    "MaterializeTensorStrides",
    "InitMemRef",
    "MaterializeSemanticAliases",
    "MemoryReuse",
    "ExpandManualPhaseFence"
   ],
   "required_by": [
    "FlattenCallExpr",
    "ConvertTensorToTileOps",
    "FlattenTileNdTo2D",
    "AutoTileMatmulL0",
    "CanonicalizeTileSlice",
    "InferTileMemorySpace",
    "InsertMxScaleAddr",
    "LowerAutoVectorSplit",
    "ExpandMixedKernel",
    "InjectGMPipeBuffer",
    "SkewCrossCorePipeline",
    "LowerPipelineToSlots",
    "LowerPipelineLoops",
    "CanonicalizeIOOrder",
    "MaterializeTensorStrides",
    "MaterializeSemanticAliases",
    "MemoryReuse",
    "ExpandManualPhaseFence"
   ],
   "invalidated_by": [
    "ConvertToSSA"
   ]
  },
  "NoNestedCalls": {
   "produced_by": [
    "FlattenCallExpr",
    "ExpandManualPhaseFence"
   ],
   "required_by": [
    "ExpandManualPhaseFence"
   ],
   "invalidated_by": []
  },
  "HierarchyOutlined": {
   "produced_by": [
    "OutlineHierarchyScopes"
   ],
   "required_by": [],
   "invalidated_by": []
  },
  "OrchestrationReferencesResolved": {
   "produced_by": [
    "OutlineHierarchyScopes"
   ],
   "required_by": [],
   "invalidated_by": []
  },
  "SplitIncoreOrch": {
   "produced_by": [
    "OutlineIncoreScopes",
    "OptimizeOrchTensors",
    "AutoTileMatmulL0",
    "CanonicalizeTileSlice",
    "InsertMxScaleAddr",
    "ResolveBackendOpLayouts",
    "LowerAutoVectorSplit",
    "SkewCrossCorePipeline",
    "LowerPipelineToSlots",
    "LowerPipelineLoops",
    "CanonicalizeIOOrder",
    "MaterializeTensorStrides"
   ],
   "required_by": [
    "ConvertTensorToTileOps",
    "OptimizeOrchTensors",
    "AutoTileMatmulL0",
    "CanonicalizeTileSlice",
    "InferTileMemorySpace",
    "InsertMxScaleAddr",
    "ResolveBackendOpLayouts",
    "LowerAutoVectorSplit",
    "ExpandMixedKernel",
    "StampTfreeSplit",
    "NormalizeReturnOrder",
    "SkewCrossCorePipeline",
    "LowerPipelineToSlots",
    "LowerPipelineLoops",
    "CanonicalizeIOOrder",
    "MaterializeTensorStrides",
    "InitMemRef",
    "MaterializeSemanticAliases",
    "MemoryReuse",
    "AllocateMemoryAddr",
    "FoldNoOpReshape",
    "DeriveCallDirections",
    "AutoDeriveTaskDependencies",
    "MaterializeRuntimeScopes",
    "InsertCommFence"
   ],
   "invalidated_by": []
  },
  "AivSplitValid": {
   "produced_by": [
    "OutlineIncoreScopes",
    "ConvertTensorToTileOps",
    "InferTileMemorySpace"
   ],
   "required_by": [
    "LowerAutoVectorSplit"
   ],
   "invalidated_by": [
    "ConvertTensorToTileOps",
    "InferTileMemorySpace",
    "LowerAutoVectorSplit"
   ]
  },
  "ClusterOutlined": {
   "produced_by": [
    "OutlineClusterScopes"
   ],
   "required_by": [],
   "invalidated_by": []
  },
  "IncoreTileOps": {
   "produced_by": [
    "ConvertTensorToTileOps",
    "OptimizeOrchTensors",
    "AutoTileMatmulL0",
    "CanonicalizeTileSlice",
    "InsertMxScaleAddr",
    "ResolveBackendOpLayouts",
    "LowerAutoVectorSplit",
    "SkewCrossCorePipeline",
    "LowerPipelineToSlots",
    "LowerPipelineLoops",
    "CanonicalizeIOOrder",
    "MaterializeTensorStrides"
   ],
   "required_by": [
    "OptimizeOrchTensors",
    "FlattenTileNdTo2D",
    "AutoTileMatmulL0",
    "CanonicalizeTileSlice",
    "InferTileMemorySpace",
    "InsertMxScaleAddr",
    "ResolveBackendOpLayouts",
    "LowerAutoVectorSplit",
    "ExpandMixedKernel",
    "NormalizeReturnOrder",
    "SkewCrossCorePipeline",
    "LowerPipelineToSlots",
    "LowerPipelineLoops",
    "CanonicalizeIOOrder",
    "MaterializeTensorStrides",
    "InitMemRef",
    "MaterializeSemanticAliases",
    "MemoryReuse",
    "AllocateMemoryAddr",
    "FoldNoOpReshape"
   ],
   "invalidated_by": []
  },
  "TileOps2D": {
   "produced_by": [
    "FlattenTileNdTo2D",
    "AutoTileMatmulL0",
    "CanonicalizeTileSlice",
    "ResolveBackendOpLayouts",
    "LowerAutoVectorSplit",
    "SkewCrossCorePipeline",
    "LowerPipelineToSlots",
    "LowerPipelineLoops",
    "CanonicalizeIOOrder",
    "MaterializeTensorStrides"
   ],
   "required_by": [
    "AutoTileMatmulL0",
    "CanonicalizeTileSlice",
    "ResolveBackendOpLayouts",
    "LowerAutoVectorSplit",
    "ExpandMixedKernel",
    "SkewCrossCorePipeline",
    "LowerPipelineToSlots",
    "LowerPipelineLoops",
    "CanonicalizeIOOrder",
    "MaterializeTensorStrides",
    "InitMemRef",
    "MaterializeSemanticAliases",
    "MemoryReuse",
    "AllocateMemoryAddr",
    "FoldNoOpReshape"
   ],
   "invalidated_by": []
  },
  "TileMemoryInferred": {
   "produced_by": [
    "InferTileMemorySpace",
    "InsertMxScaleAddr",
    "LowerAutoVectorSplit",
    "SkewCrossCorePipeline",
    "LowerPipelineToSlots",
    "LowerPipelineLoops",
    "CanonicalizeIOOrder",
    "MaterializeTensorStrides"
   ],
   "required_by": [
    "InsertMxScaleAddr",
    "LowerAutoVectorSplit",
    "ExpandMixedKernel",
    "SkewCrossCorePipeline",
    "LowerPipelineToSlots",
    "LowerPipelineLoops",
    "CanonicalizeIOOrder",
    "MaterializeTensorStrides",
    "InitMemRef"
   ],
   "invalidated_by": []
  },
  "AccToGmStoreValid": {
   "produced_by": [
    "InferTileMemorySpace"
   ],
   "required_by": [],
   "invalidated_by": []
  },
  "MixedKernelExpanded": {
   "produced_by": [
    "ExpandMixedKernel",
    "InjectGMPipeBuffer"
   ],
   "required_by": [
    "InjectGMPipeBuffer",
    "SplitVectorKernel"
   ],
   "invalidated_by": []
  },
  "HardSyncallOccupancyValid": {
   "produced_by": [
    "ExpandMixedKernel"
   ],
   "required_by": [],
   "invalidated_by": []
  },
  "VectorKernelSplit": {
   "produced_by": [
    "SplitVectorKernel"
   ],
   "required_by": [],
   "invalidated_by": []
  },
  "ReturnParamsExplicit": {
   "produced_by": [
    "NormalizeReturnOrder"
   ],
   "required_by": [],
   "invalidated_by": []
  },
  "PipelineResolved": {
   "produced_by": [
    "CanonicalizeIOOrder"
   ],
   "required_by": [],
   "invalidated_by": []
  },
  "TensorViewCanonical": {
   "produced_by": [
    "MaterializeTensorStrides"
   ],
   "required_by": [],
   "invalidated_by": []
  },
  "HasMemRefs": {
   "produced_by": [
    "InitMemRef"
   ],
   "required_by": [
    "MaterializeSemanticAliases",
    "MemoryReuse",
    "AllocateMemoryAddr",
    "FoldNoOpReshape"
   ],
   "invalidated_by": []
  },
  "AllocatedMemoryAddr": {
   "produced_by": [
    "AllocateMemoryAddr"
   ],
   "required_by": [],
   "invalidated_by": []
  },
  "CallDirectionsResolved": {
   "produced_by": [
    "DeriveCallDirections",
    "AutoDeriveTaskDependencies",
    "ExpandManualPhaseFence"
   ],
   "required_by": [
    "AutoDeriveTaskDependencies",
    "ExpandManualPhaseFence",
    "MaterializeRuntimeScopes",
    "ClassifyIterArgCarry"
   ],
   "invalidated_by": []
  },
  "CommDomainScopesMaterialized": {
   "produced_by": [
    "MaterializeCommDomainScopes",
    "LowerHostTensorCollectives",
    "MaterializeDistTensorCtx"
   ],
   "required_by": [
    "LowerHostTensorCollectives",
    "MaterializeDistTensorCtx"
   ],
   "invalidated_by": []
  },
  "RuntimeScopesMaterialized": {
   "produced_by": [
    "MaterializeRuntimeScopes",
    "ClassifyIterArgCarry"
   ],
   "required_by": [
    "ClassifyIterArgCarry"
   ],
   "invalidated_by": []
  },
  "IterArgCarryClassified": {
   "produced_by": [
    "ClassifyIterArgCarry"
   ],
   "required_by": [],
   "invalidated_by": []
  }
 },
 "meta": {
  "total": 49,
  "unique": 48,
  "decide": 20,
  "src": "repo/hw-native-sys/pypto",
  "snapshot": "2026-08-11"
 }
};
