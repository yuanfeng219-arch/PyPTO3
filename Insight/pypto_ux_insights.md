# PyPTO Issues UX Insights

Source: `pypto_issues.json`, `pypto_issues.md`, `pypto_issues.csv`

Scope: 561 real issues from `hw-native-sys/pypto` after filtering pull requests.

- Open: 45
- Closed: 516
- Top labels: bug 229, enhancement 160, code health 39, rfc 23, documentation 14
- Top title types: Bug 183, Feature 142, Pass Bug 77, Code Health 32, RFC 27
- Median close time for closed issues: about 1.7 days

## Executive Summary

PyPTO's issue set reads less like a general beginner product backlog and more like a high-intensity developer experience backlog for compiler/runtime/kernel authors. The project appears responsive, with many issues closed quickly, but the remaining pain points cluster around advanced developer workflows: expressing hardware-aware intent in the DSL, trusting compiler transformations, diagnosing silent correctness failures, understanding task dependencies, and optimizing distributed/model kernels without falling into opaque runtime or codegen traps.

The strongest UX opportunity is not a prettier surface. It is a safer and more explainable development loop: write PyPTO code, understand what it means, compile it with actionable diagnostics, run it with traceability, and tune it with confidence.

## Developer Journey

### 1. Setup, Run, And Validate

Representative issue: #63, "How to run the tests such as tests/ut/ir/core/".

Pain points:
- README/test commands can drift from the actual environment and package layout.
- Import/setup errors such as `ModuleNotFoundError` block users before they reach product value.
- The test matrix spans unit, system, runtime, hardware, Docker, and CI contexts, which creates uncertainty about which command validates what.

User expectations:
- A new contributor can run a minimal smoke test within minutes.
- The project tells the user whether their Python path, submodules, runtime pins, and hardware assumptions are valid.
- Test commands are layered by intent: quick local check, compiler-only check, device check, distributed check, full CI-equivalent check.

Design opportunities:
- Add `pypto doctor` or `scripts/doctor.ps1` to validate environment, imports, submodules, CANN/runtime pins, and device availability.
- Create a "first successful test" path in docs with exact shell commands and expected output.
- Add a test command map: "I changed DSL parser", "I changed codegen", "I changed distributed runtime", "I changed model kernel".

### 2. Model Intent In The DSL

Representative issues: #1647, #1968, #1368, #2059, #1189.

Pain points:
- The DSL sometimes exposes implementation details too early. Example: window buffer allocation requires manual byte-size math even though shape and dtype are restated later.
- Equivalent programming models can require invasive code reshaping. Example: switching between `pl.parallel + pl.at` and `pl.spmd` can mean re-indenting the whole body.
- Manual task dependency wiring is correct but verbose and error-prone for non-trivial pipelines.
- Important runtime capabilities can exist before the frontend exposes them, leaving users unable to express real model needs.
- Distributed programming concepts such as HOST/CHIP/CORE_GROUP, windows, signals, ranks, predicates, and collectives need a stable mental model.

User expectations:
- PyPTO should let users write intent at the level they are thinking: tensor shape, dtype, task dependency, dispatch condition, communication pattern.
- Advanced escape hatches should exist, but the common path should be hard to misuse.
- Switching execution strategy should be a small local edit when semantics are unchanged.

Design opportunities:
- Move byte-oriented APIs toward shape/dtype-oriented overloads, while preserving low-level forms as explicit escape hatches.
- Provide "semantic DSL recipes" for common patterns: SPMD block, manual scope with dependencies, cross-rank collective, conditional expert dispatch, window allocation.
- Add opt-in automation for dependency derivation, with an explain mode that prints the inferred edges.
- Introduce a DSL compatibility guide that explains when `pl.parallel`, `pl.at`, `pl.spmd`, `manual_scope`, and orchestration-level collectives should be used.

### 3. Compile With Trust

Representative issues: #1525, #1305, #2005, #2006, #2047, #2058.

Pain points:
- Several high-severity issues are silent correctness failures: all-zero outputs, cache-line corruption, missing writes, lost dependencies, scalar aliasing ambiguity.
- Some verifier hints are noisy, duplicated, or hard to map back to source code.
- Users can write code that compiles cleanly but later fails on device or produces wrong results.
- IR-level issues often require expert forensic work across many pass dumps.

User expectations:
- Unsafe or ambiguous patterns should fail at compile time when possible.
- Warnings should be source-mapped, deduplicated, and actionable.
- When the compiler transforms code, users should be able to inspect why a dependency, alias, buffer, or event was created.

Design opportunities:
- Build a diagnostic hierarchy: correctness errors, likely correctness hazards, performance hints, informational notes.
- Add source spans and "why this happened" context to verifier output and pass diagnostics.
- Add compile-time guards for known foot-guns: mixed tensor/scalar stores to overlapping GM cache lines, lost WAR dependencies, invalid split semantics, detached `pl.Out` reassignment, scalar aliasing ambiguity.
- Provide a one-command repro bundle that includes source, generated IR snapshots, pass order, runtime pins, and device metadata.

### 4. Run, Debug, And Reproduce

Representative issues: #1789, #1869, #1840.

Pain points:
- Hardware/runtime failures can surface as opaque device errors or deadlocks, e.g. AICPU 507018.
- Flaky failures and daily CI failures accumulate discussion but are hard to convert into user-understandable root causes.
- Runtime traces, codegen events, and source-level operations are not always connected in a way a developer can navigate.

User expectations:
- A device failure should point to a probable compiler/runtime/source cause, not just a low-level code.
- Flaky test reports should preserve enough context to reproduce.
- Developers should be able to correlate source operations, IR ops, event IDs, fences, runtime tasks, and trace spans.

Design opportunities:
- Create a runtime error explainer for common device/AICPU error codes, with likely PyPTO causes and next commands.
- Add trace correlation IDs from source region to IR op to runtime task/event.
- Standardize CI failure issue templates with environment, commit pins, trace artifacts, failing command, and suspected subsystem.

### 5. Optimize Performance

Representative issues: #1475, #2040, #1958, #1980.

Pain points:
- Correct compiler optimizations can still harm pipeline overlap or hardware utilization.
- Users need to understand why native PyPTO kernels lag library kernels such as CANN FAI.
- Memory planning, buffer reuse, valid shape preservation, and layout decisions are difficult to reason about from source alone.

User expectations:
- Performance regressions should be explainable in terms of pipeline overlap, memory traffic, fences, buffer reuse, and layout.
- The compiler should expose enough information to tell whether an optimization helped or hurt.
- Users want practical knobs, not just low-level IR spelunking.

Design opportunities:
- Add a performance report mode that summarizes buffer reuse, liveness, pipeline overlap, GM round trips, and emitted fences.
- Make performance hints source-aware and ranked by expected impact.
- Provide before/after IR and timeline diff tooling for suspected optimization regressions.
- Add model-level benchmarking templates that compare PyPTO kernels to baseline library kernels with consistent metrics.

### 6. Build Distributed And MoE Workloads

Representative issues: #1189, #1906, #2027, #2029, #2059.

Pain points:
- Distributed work requires composing frontend DSL, compiler lowering, runtime scheduling, HCCL/window concepts, and model-level constraints.
- MoE workloads need conditional dispatch and skip-empty-expert behavior without stalling orchestration.
- Communication context/provenance bugs are hard for users to detect manually.

User expectations:
- Distributed tensors and communication contexts should carry enough provenance to prevent invalid merges and aliases.
- Users should be able to express collectives and conditional dispatch directly in the DSL.
- The system should preserve static graph benefits while supporting runtime-dependent scheduling choices.

Design opportunities:
- Provide a distributed programming guide with canonical examples for TP, EP, allreduce, allgather, window buffers, signals, and dispatch predicates.
- Add visual explanations of HOST/CHIP/CORE_GROUP execution and task dependency flow.
- Add validators for DistributedTensor context provenance and cross-rank communication invariants.

## Cross-Cutting UX Themes

### Silent Failure Is The Highest-Risk UX Problem

Multiple issues describe clean compilation followed by wrong output, zeros, corruption, dropped work, stale reads, or deadlock. For a compiler DSL targeting specialized hardware, the worst user experience is not a compile error. It is trust erosion: the user no longer knows whether source, compiler, runtime, hardware, or test oracle is wrong.

Priority opportunity:
- Turn known silent failure classes into early diagnostics.
- When early diagnostics are impossible, emit runtime assertions or trace markers.
- Provide source-to-runtime provenance for failures.

### Users Need Progressive Disclosure, Not A Fully Hidden System

The users filing these issues are advanced. They do not need the system to hide IR, memory, or scheduling details entirely. They need those details to appear at the right level, with clear source mapping and a path from symptom to cause.

Priority opportunity:
- Create three levels of explanation: beginner command, compiler explanation, hardware/runtime detail.
- Let users drill down from source line to IR pass to runtime event.

### API Ergonomics Matter Most Where Code Is Rewritten Often

Issues around `pl.spmd`, `pl.parallel`, window allocation, manual deps, and split semantics show that small syntax/API decisions have large workflow costs during kernel development.

Priority opportunity:
- Optimize for edit-locality: changing dispatch or memory strategy should not require rewriting the body.
- Reduce duplicated declarations of shape/dtype/dependency information.

## Prioritized Design Opportunities

1. Diagnostic upgrade for silent correctness hazards.
   - Impact: very high
   - Evidence: #1525, #2005, #2006, #2058, #1789
   - Outcome: fewer wrong-output debugging sessions; higher trust in compiler.

2. Developer doctor and test workflow map.
   - Impact: high
   - Evidence: #63, daily CI issues, broad test complexity
   - Outcome: faster onboarding and fewer environment-related stalls.

3. Source-to-IR-to-runtime traceability.
   - Impact: high
   - Evidence: pass bugs, event deadlocks, runtime failures, noisy perf hints
   - Outcome: quicker root cause analysis across compiler and hardware boundaries.

4. Safer high-level DSL affordances.
   - Impact: high
   - Evidence: #1968, #1647, #1368, #2059
   - Outcome: users express intent with fewer low-level foot-guns.

5. Distributed programming guide and validators.
   - Impact: medium-high
   - Evidence: #1189, #2027, #2029, #1906, #2059
   - Outcome: clearer mental model for TP/EP/MoE workflows.

6. Performance explainability toolkit.
   - Impact: medium-high
   - Evidence: #1475, #2040, #1958, #1980
   - Outcome: developers can understand and tune performance regressions without forensic IR archaeology.

## Suggested Product Artifacts

- `pypto doctor`: environment, imports, pins, submodules, device, and smoke-test validation.
- `pypto explain <file_or_dump>`: summarize diagnostics, source spans, IR pass changes, inferred dependencies, and runtime mapping.
- `pypto trace report`: source-to-task/event/fence timeline.
- "Known hazards" compiler verifier pack: overlapping stores, lost dependencies, invalid split semantics, alias ambiguity, detached outputs.
- Developer journey docs:
  - Quickstart and first test
  - DSL mental model
  - Dependency and scheduling model
  - Distributed programming model
  - Debugging wrong output
  - Performance tuning workflow

