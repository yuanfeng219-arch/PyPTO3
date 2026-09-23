import rawIdentityIndex from '@/data/lightningIndexerIdentityIndex.json'
import {
  ControlFlowReviewFixture,
  GeneratedControlPath,
  IdentityIndex,
  ReviewSelection,
  ReviewStageId,
} from '@/types/controlFlowReview'

function formatNumber(value: number) {
  return new Intl.NumberFormat('en-US').format(value)
}

function sourceSelection(index: IdentityIndex): ReviewSelection {
  const source = index.sourceRegions[0]
  return {
    id: source.id,
    stage: 'source',
    kind: 'Semantic control region',
    title: source.userName,
    subtitle: `Python L${source.startLine}–${source.endLine}`,
    confidence: source.effectiveUnrollFactors.confidence,
    identity: [
      ['源码范围', `L${source.startLine}–${source.endLine}`],
      ['动态边界', `${source.dynamicBound.expression} · L${source.dynamicBound.sourceLine}`],
      ['循环范围', `${source.range.start} → ${source.range.end} · step ${source.range.step}`],
      ['有效档位', source.effectiveUnrollFactors.values.join(', ')],
    ],
    role: '沿输入第 0 维组织 Query、Key 与 Weight 计算，并声明可被前端物化的循环展开策略。',
    change: '本阶段只表达源码控制意图；尚不能把六条生成路径归因给某个可见 Compiler Pass。',
    impact:
      't = x_in.shape[0] 读取输入第 0 维的 token 数；range(0, t) 以它作为 loop 上界，因此输入长度决定迭代次数以及哪些生成路径会执行。',
    nextStep: '运行控制流审查，并在 Transformation 中核对路径首次出现的证据边界。',
    rawEvidence: [
      ['sourceRegionId', source.id],
      ['sourceSnapshot', index.reviewRun.sourceSnapshot.path],
      ['snapshotHash', index.reviewRun.sourceSnapshot.contentHash ?? 'missing'],
    ],
  }
}

function pathSelection(path: GeneratedControlPath): ReviewSelection {
  return {
    id: path.id,
    stage: 'codegen',
    kind: 'Generated control path',
    title: `Unroll ×${path.factor}`,
    subtitle: path.runtime.observed
      ? `${path.runtime.rootInvocationCount} runtime invocations`
      : '本次运行未命中',
    confidence: path.mappingConfidence,
    identity: [
      ['路径条件', path.condition],
      ['Root magic', String(path.rootMagic)],
      ['最终操作数', formatNumber(path.finalRootOperationCount)],
      ['运行 Task', formatNumber(path.runtime.taskEventCount)],
    ],
    role: `每次推进 ${path.factor} 个 iteration 的已生成 orchestration 路径。`,
    change: '该 specialized function 在 per-function pass pipeline 开始前已经存在。',
    impact: path.runtime.observed
      ? `输入长度：16 token；本次场景执行 ${path.runtime.rootInvocationCount} 次，共关联 ${path.runtime.taskEventCount} 个 task event。`
      : '该路径存在于生成结构中，但“输入长度：16 token”的本次运行没有执行。',
    nextStep: path.runtime.observed
      ? '查看对应 Runtime task 与调用 identity。'
      : '用边界 shape 场景验证该路径的命中条件。',
    rawEvidence: [
      ['outerFunctionHash', path.outerFunctionHash],
      ['pathHash', path.pathHash],
      ['rootHash', path.rootHash],
      ['pathMagic', String(path.pathMagic)],
    ],
  }
}

export function adaptIdentityIndexFixture(input: unknown): ControlFlowReviewFixture {
  const index = input as IdentityIndex
  if (!index.reviewRun?.id || !index.sourceRegions?.length) {
    throw new Error('Identity Index fixture is missing its review run or source region')
  }
  if (!index.generatedPaths?.length || !index.findings?.length) {
    throw new Error('Identity Index fixture is missing paths or diagnostic findings')
  }

  const source = sourceSelection(index)
  const materialization: ReviewSelection = {
    id: 'transform:frontend-materialization',
    stage: 'transform',
    kind: 'Evidence boundary',
    title: 'Frontend Control Expansion',
    subtitle: '6 specialized functions first observed',
    confidence: index.compilerEvidence.firstObservedSpecializedFunctions.confidence,
    identity: [
      ['首次观测', index.compilerEvidence.firstObservedSpecializedFunctions.boundary],
      ['生成路径', `${index.generatedPaths.length} variants`],
      ['可见 Pass 归因', 'unknown'],
      ['LoopUnroll Diff', 'Before = After'],
    ],
    role: '标记源码循环意图已经物化为 specialized functions、但尚无可见 Pass 能精确归因的边界。',
    change: '从一个 Source loop region 变为 6 条带 factor、hash 与 root identity 的生成路径。',
    impact: '不能把路径膨胀错误归因给 Pass_00_LoopUnroll；诊断必须保留 unknown attribution。',
    nextStep: '补采 frontend materialization dump，或验证构建版本后再定位真正的形成步骤。',
    rawEvidence: [
      ['artifactDirectory', index.compilerEvidence.firstObservedSpecializedFunctions.artifactDirectory],
      ['beforeSha256', index.compilerEvidence.loopUnrollPass.beforeSha256],
      ['afterSha256', index.compilerEvidence.loopUnrollPass.afterSha256],
    ],
  }
  const runtimeEvent = index.sampleRuntimeEvents[0]
  const runtime: ReviewSelection = {
    id: runtimeEvent.id,
    stage: 'runtime',
    kind: 'Runtime task event',
    title: `${runtimeEvent.semanticLabel} · Task ${runtimeEvent.taskId}`,
    subtitle: `${runtimeEvent.duration.toFixed(2)} trace units · lane ${runtimeEvent.tid}`,
    confidence: 'exact',
    identity: [
      ['语义标签', runtimeEvent.semanticLabel],
      ['Generated root', runtimeEvent.rootHash],
      ['Compiler CALL', `${runtimeEvent.rootHash}:${runtimeEvent.callOpMagic}`],
      ['Leaf function', runtimeEvent.leafHash],
    ],
    role: '一条真实 runtime task event，通过 rootHash 与 callOpMagic 回链到 compiler CALL 和语义源码。',
    change: '该事件不是预测结果；它来自 merged_swimlane.json 的实际观测。',
    impact: '精确链路止于 compiled source L310；跳转本地副本 L313 仍是 derived，因为缺少 source snapshot hash。',
    nextStep: '记录 sourceSnapshotHash，并用 rootHash + callOpMagic 作为路径内操作身份。',
    rawEvidence: [
      ['eventId', runtimeEvent.id],
      ['timestamp', String(runtimeEvent.timestamp)],
      ['pid / tid', `${runtimeEvent.pid} / ${runtimeEvent.tid}`],
      ['compiled → local', 'L310 → L313 · derived'],
    ],
  }

  const selections: Record<string, ReviewSelection> = {
    [source.id]: source,
    [materialization.id]: materialization,
    [runtime.id]: runtime,
  }
  index.generatedPaths.forEach((path) => {
    selections[path.id] = pathSelection(path)
  })

  const defaultSelectionByStage: Record<ReviewStageId, string> = {
    source: source.id,
    transform: materialization.id,
    codegen: index.generatedPaths.find((path) => path.factor === 16)?.id ?? index.generatedPaths[0].id,
    runtime: runtime.id,
  }

  return {
    identityIndex: index,
    reviewRun: {
      id: index.reviewRun.id,
      status: 'fixture',
      generatedAt: index.generatedAt,
      sourceSnapshot: index.reviewRun.sourceSnapshot,
      sourceRegion: index.sourceRegions[0],
      compileEvidence: index.compilerEvidence,
      generatedPaths: index.generatedPaths,
      runtimeScenario: index.reviewRun.runtimeScenario,
      runtimeSummary: index.runtimeSummary,
      findings: index.findings,
      selections,
    },
    defaultSelectionByStage,
  }
}

export const lightningIndexerReviewFixture = adaptIdentityIndexFixture(
  rawIdentityIndex,
)
