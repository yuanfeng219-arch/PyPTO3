import React, { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import Loading from '@/components/modules/loading/Loading'
import loopUnrollCfg from '@/data/loopUnrollCfg.json'
import operatorSource from '@/data/lightning_indexer_prolog_quant.py'
import { lightningIndexerReviewFixture } from '@/data/adapters/controlFlowReviewFixture'
import {
  DEMO_UNROLL_FACTORS,
  DemoReviewConfig,
  DemoScenario,
  DemoTruthStatus,
  resolveDemoScenario,
  sameFactors,
} from '@/data/controlFlowDemo'
import { ReviewSelection, ReviewStageId } from '@/types/controlFlowReview'
import { GraphJSON, GraphState, ReviewNodePresentation } from '@/types/graph'
import styles from './controlFlowExplorer.module.scss'

const LayoutFlowFactory = React.lazy(
  () => import('@/components/modules/llvmcfg/LayoutFlowFactory'),
)

type LeftMode = 'code' | 'review'
type TransformScope = 'materialization' | 'loop-unroll'
type WorkflowState =
  | 'initial'
  | 'reviewing'
  | 'baselineReady'
  | 'draft'
  | 'diffing'
  | 'diffReady'
  | 'applied'

interface StageDefinition {
  id: ReviewStageId
  step: string
  label: string
  title: string
  question: string
}

const FIXTURE = lightningIndexerReviewFixture
const RUN = FIXTURE.reviewRun
const SOURCE = RUN.sourceRegion
const LOOP_NAME = SOURCE.userName
const SOURCE_PATH = RUN.sourceSnapshot.path
const EVIDENCE_SHAPE = RUN.runtimeScenario.inputT.value

const STAGES: StageDefinition[] = [
  {
    id: 'source',
    step: '01',
    label: 'SOURCE FLOW',
    title: '源码意图',
    question: '源码声明了什么约束？',
  },
  {
    id: 'transform',
    step: '02',
    label: 'TRANSFORMATION FLOW',
    title: '编译变换',
    question: '结构从哪里开始改变？',
  },
  {
    id: 'codegen',
    step: '03',
    label: 'GENERATED FLOW',
    title: '生成结构',
    question: '最终生成了哪些路径？',
  },
  {
    id: 'runtime',
    step: '04',
    label: 'EXECUTION FLOW',
    title: '真实执行',
    question: '设备实际执行了什么？',
  },
]

const SOURCE_LINES = operatorSource.replace(/\r\n/g, '\n').split('\n')
const CONTROL_LINES = new Set([229, 251, 252, 253, 254, 261])

function Icon({
  name,
}: {
  name:
    | 'flow'
    | 'code'
    | 'review'
    | 'play'
    | 'panel'
    | 'sun'
    | 'moon'
    | 'arrow'
    | 'link'
}) {
  const paths: Record<string, React.ReactNode> = {
    flow: (
      <>
        <circle cx="5" cy="5" r="2" />
        <circle cx="19" cy="12" r="2" />
        <circle cx="5" cy="19" r="2" />
        <path d="M7 5h4a3 3 0 0 1 3 3v1M7 19h4a3 3 0 0 0 3-3v-1M14 12h3" />
      </>
    ),
    code: (
      <>
        <path d="m8 9-3 3 3 3M16 9l3 3-3 3M14 5l-4 14" />
      </>
    ),
    review: (
      <>
        <path d="M4 5h16v14H4zM8 9h8M8 13h5" />
        <path d="m15 16 2 2 4-5" />
      </>
    ),
    play: <path d="m9 7 8 5-8 5z" />,
    panel: (
      <>
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <path d="M15 3v18" />
      </>
    ),
    sun: (
      <>
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2" />
      </>
    ),
    moon: <path d="M20.5 14.2A8 8 0 0 1 9.8 3.5 8.5 8.5 0 1 0 20.5 14.2Z" />,
    arrow: <path d="m9 18 6-6-6-6" />,
    link: (
      <>
        <path d="M10 13a5 5 0 0 0 7.5.5l2-2a5 5 0 0 0-7-7l-1 1" />
        <path d="M14 11a5 5 0 0 0-7.5-.5l-2 2a5 5 0 0 0 7 7l1-1" />
      </>
    ),
  }
  return (
    <svg className={styles.icon} viewBox="0 0 24 24" aria-hidden="true">
      {paths[name]}
    </svg>
  )
}

function makeGraph(
  name: string,
  nodes: Array<{ id: number; label: string }>,
  links: Array<[number, number, string?]>,
): GraphJSON {
  return {
    name,
    directed: true,
    strict: false,
    label: name,
    _subgraph_cnt: 0,
    objects: nodes.map((node) => ({
      _gvid: node.id,
      name: `Node${node.id}`,
      label: `{${node.label}:\\l}`,
      shape: 'record',
    })),
    edges: links.map(([tail, head, label], index) => ({
      _gvid: 1000 + index,
      tail,
      head,
      label,
    })),
  }
}

const OBSERVED_FACTORS = RUN.generatedPaths
  .filter((path) => path.runtime.observed)
  .map((path) => path.factor)

function formatInputLength(value: number) {
  return `输入长度：${value} token`
}

interface CoverageGraph {
  graph: GraphJSON
  presentations: Record<number, ReviewNodePresentation>
}

function coverageGraph(
  side: 'baseline' | 'request',
  config: DemoReviewConfig,
  scenario: DemoScenario,
  otherFactors: number[],
  baselineKind: 'observed' | 'updated' = 'observed',
): CoverageGraph {
  const factors = scenario.eligibleFactors
  const comparedFactors = new Set(otherFactors)
  const nodes = [{ id: 0, label: `${side}-scenario` }]
  const links: Array<[number, number, string?]> = []
  const presentations: Record<number, ReviewNodePresentation> = {}
  const requestChanged =
    config.inputT !== EVIDENCE_SHAPE ||
    !sameFactors(config.unrollFactors, DEMO_UNROLL_FACTORS)

  presentations[0] = {
    selectionId: 'demo:scenario',
    eyebrow:
      side === 'baseline'
        ? baselineKind === 'updated'
          ? 'Updated baseline'
          : 'Observed baseline'
        : `${scenario.status} request`,
    title:
      side === 'baseline'
        ? `${
            baselineKind === 'updated' ? 'Updated review' : 'Runtime evidence'
          } · ${formatInputLength(config.inputT)}`
        : `What-if review · ${formatInputLength(config.inputT)}`,
    description:
      side === 'baseline'
        ? baselineKind === 'updated'
          ? '应用源码候选并重新审查后形成的演示基线。'
          : '真实 runtime 证据中命中的 factor-specific control paths。'
        : scenario.status === 'unavailable'
        ? `当前 policy 留下 ${scenario.remainder} 个 iteration 无法覆盖。`
        : '根据 dynamic bound 与 unroll policy 推演可能命中的路径；不是编译结果。',
    confidence:
      side === 'baseline'
        ? baselineKind === 'updated'
          ? 'demo baseline'
          : 'derived evidence'
        : scenario.status,
    tone:
      side === 'baseline'
        ? 'success'
        : scenario.status === 'unavailable'
        ? 'warning'
        : 'info',
    diffStatus: requestChanged ? 'changed' : 'same',
    metrics: [
      { label: 'eligible paths', value: String(factors.length) },
      {
        label: side === 'baseline' ? 'task evidence' : 'uncovered tail',
        value:
          side === 'baseline'
            ? String(RUN.runtimeSummary.controlLoopTaskEventCount)
            : String(scenario.remainder),
      },
    ],
    tags:
      side === 'baseline'
        ? baselineKind === 'updated'
          ? ['已更新', 'Demo review']
          : ['Observed', 'merged_swimlane.json']
        : [scenario.status, 'What-if · not compiled'],
  }

  factors.forEach((factor, index) => {
    const id = index + 1
    const path = RUN.generatedPaths.find(
      (candidate) => candidate.factor === factor,
    )
    const witness = scenario.coverage.find((item) => item.factor === factor)
    const diffStatus = comparedFactors.has(factor)
      ? 'same'
      : side === 'baseline'
      ? 'removed'
      : 'added'
    nodes.push({ id, label: `${side}-factor-${factor}` })
    links.push([0, id, `×${factor}`])
    presentations[id] = {
      selectionId: `demo:path:${factor}`,
      eyebrow:
        side === 'baseline'
          ? 'Observed execution path'
          : witness?.witnessCount
          ? `Coverage witness · ${witness.witnessCount}×`
          : 'Eligible fallback path',
      title: `Unroll ×${factor}`,
      description:
        side === 'baseline'
          ? `${path?.runtime.rootInvocationCount ?? 0} invocations · ${
              path?.runtime.taskEventCount ?? 0
            } task events in the captured run.`
          : `当剩余长度不少于 ${factor} 时该路径可被选择；这里只表达约束推演。`,
      confidence: side === 'baseline' ? 'observed' : scenario.status,
      tone:
        diffStatus === 'removed'
          ? 'muted'
          : diffStatus === 'added'
          ? 'success'
          : 'neutral',
      diffStatus,
      metrics: [
        {
          label: side === 'baseline' ? 'invocations' : 'witness count',
          value: String(
            side === 'baseline'
              ? path?.runtime.rootInvocationCount ?? 0
              : witness?.witnessCount ?? 0,
          ),
        },
        {
          label: 'factor',
          value: String(factor),
        },
      ],
      tags: [
        diffStatus,
        side === 'baseline' ? 'runtime evidence' : 'projected eligibility',
      ],
    }
  })

  return {
    graph: makeGraph(
      `coverage-${side}-${config.inputT}-${config.unrollFactors.join('-')}`,
      nodes,
      links,
    ),
    presentations,
  }
}

function pathGraph(mode: 'materialization' | 'codegen' | 'runtime') {
  const nodes = [
    { id: 0, label: mode === 'runtime' ? 'runtime-scenario' : 'source-loop' },
  ]
  const links: Array<[number, number, string?]> = []
  const presentations: Record<number, ReviewNodePresentation> = {}
  const materializationId = FIXTURE.defaultSelectionByStage.transform
  presentations[0] =
    mode === 'runtime'
      ? {
          selectionId: FIXTURE.defaultSelectionByStage.runtime,
          eyebrow: 'Observed scenario',
          title: `Runtime · ${formatInputLength(EVIDENCE_SHAPE)}`,
          description: `${RUN.runtimeSummary.controlLoopTaskEventCount} control-loop task events in merged_swimlane.json.`,
          confidence: RUN.runtimeScenario.inputT.confidence,
          tone: 'info',
          metrics: [
            {
              label: 'task events',
              value: String(RUN.runtimeSummary.controlLoopTaskEventCount),
            },
            {
              label: 'observed paths',
              value: String(RUN.runtimeSummary.observedFactors.length),
            },
          ],
        }
      : {
          selectionId: materializationId,
          eyebrow:
            mode === 'materialization' ? 'Source identity' : 'Generated entry',
          title: LOOP_NAME,
          description:
            mode === 'materialization'
              ? 'One semantic region is first observed as six specialized functions at the frontend boundary.'
              : 'Generated orchestration entry fans out to six factor-specific roots.',
          confidence: mode === 'materialization' ? 'derived mapping' : 'exact',
          tone: 'success',
          tags: [`Python L${SOURCE.startLine}–${SOURCE.endLine}`],
        }

  RUN.generatedPaths.forEach((path, index) => {
    const id = index + 1
    nodes.push({ id, label: path.id })
    links.push([
      0,
      id,
      mode === 'runtime'
        ? `${path.runtime.rootInvocationCount}×`
        : `×${path.factor}`,
    ])
    const observed = path.runtime.observed
    presentations[id] = {
      selectionId: path.id,
      eyebrow:
        mode === 'materialization'
          ? `First observed · factor ${path.factor}`
          : mode === 'runtime'
          ? observed
            ? 'Executed path'
            : 'Not observed'
          : `Generated root · magic ${path.rootMagic}`,
      title: `Unroll ×${path.factor}`,
      description:
        mode === 'materialization'
          ? `${path.finalRootOperationCount} operations before the visible per-function pipeline.`
          : mode === 'runtime'
          ? observed
            ? `${path.runtime.rootInvocationCount} invocations · ${path.runtime.taskEventCount} task events.`
            : 'Present in generated control; 输入长度：16 token 时未执行。'
          : path.condition,
      confidence: path.mappingConfidence,
      tone: observed ? 'success' : mode === 'runtime' ? 'muted' : 'neutral',
      metrics:
        mode === 'codegen'
          ? [
              { label: 'root magic', value: String(path.rootMagic) },
              {
                label: 'operations',
                value: String(path.finalRootOperationCount),
              },
            ]
          : mode === 'runtime'
          ? [
              {
                label: 'invocations',
                value: String(path.runtime.rootInvocationCount),
              },
              {
                label: 'task events',
                value: String(path.runtime.taskEventCount),
              },
            ]
          : undefined,
      tags: [`path ${path.pathMagic}`, `root ${path.rootHash.slice(0, 8)}…`],
    }
  })

  if (mode === 'runtime') {
    const event = FIXTURE.identityIndex.sampleRuntimeEvents[0]
    const taskId = nodes.length
    nodes.push({ id: taskId, label: 'sample-runtime-event' })
    links.push([2, taskId, 'rootHash + callOpMagic'])
    presentations[taskId] = {
      selectionId: event.id,
      eyebrow: 'Exact reverse-link sample',
      title: `${event.semanticLabel} · Task ${event.taskId}`,
      description:
        'Runtime event → generated root → compiler CALL → semantic source.',
      confidence: 'exact to compiled source',
      tone: 'warning',
      metrics: [
        { label: 'call magic', value: String(event.callOpMagic) },
        { label: 'duration', value: event.duration.toFixed(2) },
      ],
      tags: ['compiled L310', 'local L313 · derived'],
    }
  }

  return { graph: makeGraph(mode, nodes, links), presentations }
}

function ReadOnlyGraph({
  graph,
  presentations,
  paneLabel,
  selectionId,
  onSelect,
}: {
  graph: GraphJSON
  presentations: Record<number, ReviewNodePresentation>
  paneLabel: string
  selectionId: string
  onSelect: (id: string) => void
}) {
  return (
    <Suspense fallback={<Loading />}>
      <LayoutFlowFactory
        key={graph.name}
        llvmJson={graph}
        llvmJson_compare={graph}
        title="dev"
        paneLabel={paneLabel}
        variant="reviewLarge"
        selectedBlockId={selectionId}
        onSelectBlock={(_, id) => onSelect(id)}
        nodePresentations={presentations}
        readOnly
        showMinimap={false}
      />
    </Suspense>
  )
}

function SourceCodePanel({
  workflow,
  appliedFactors,
  onUndo,
}: {
  workflow: WorkflowState
  appliedFactors: number[]
  onUndo: () => void
}) {
  const [query, setQuery] = useState('')
  const normalizedQuery = query.trim().toLowerCase()
  const matches = normalizedQuery
    ? SOURCE_LINES.filter((line) =>
        line.toLowerCase().includes(normalizedQuery),
      ).length
    : 0
  const applied = workflow === 'applied'
  const hasLocalOverride = !sameFactors(appliedFactors, DEMO_UNROLL_FACTORS)
  const focusControl = () => {
    document
      .getElementById('source-line-251')
      ?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  }

  return (
    <section className={`${styles.codePanel} ${styles.codePanelFull}`}>
      <header>
        <div>
          <strong>lightning_indexer_prolog_quant.py</strong>
          <span title={SOURCE_PATH}>完整源码 · 364 行 · 演示工作区</span>
        </div>
        <div className={styles.codeTools}>
          <button type="button" onClick={focusControl}>
            定位控制区域
          </button>
          {applied ? (
            <button type="button" onClick={onUndo}>
              撤销应用
            </button>
          ) : null}
        </div>
        <label className={styles.codeSearch}>
          <span>搜索源码</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="函数、变量或行内容"
          />
          <em>{normalizedQuery ? `${matches} 处` : '⌘F'}</em>
        </label>
      </header>
      {applied ? (
        <div className={styles.outdatedBanner}>
          <span>源码已变化 · 当前基线已过期</span>
          <small>Demo 已应用到内存代码草稿，未写入磁盘。</small>
        </div>
      ) : null}
      <div className={styles.codeViewport}>
        {SOURCE_LINES.map((line, index) => {
          const lineNumber = index + 1
          const matched =
            normalizedQuery && line.toLowerCase().includes(normalizedQuery)
          if (applied && lineNumber === 251) {
            return (
              <React.Fragment key={lineNumber}>
                <div
                  id="source-line-251"
                  className={`${styles.codeLine} ${styles.diffRemoved}`}
                >
                  <span>{lineNumber}</span>
                  <i>−</i>
                  <code>{line}</code>
                </div>
                <div className={`${styles.codeLine} ${styles.diffAdded}`}>
                  <span>{lineNumber}</span>
                  <i>+</i>
                  <code>{`    unroll_list = [${appliedFactors.join(
                    ', ',
                  )}]`}</code>
                </div>
              </React.Fragment>
            )
          }
          if (hasLocalOverride && lineNumber === 251) {
            return (
              <div
                id="source-line-251"
                key={lineNumber}
                className={`${styles.codeLine} ${styles.focusLine}`}
              >
                <span>{lineNumber}</span>
                <i>◆</i>
                <code>{`    unroll_list = [${appliedFactors.join(
                  ', ',
                )}]`}</code>
              </div>
            )
          }
          return (
            <div
              id={`source-line-${lineNumber}`}
              key={lineNumber}
              className={`${styles.codeLine} ${
                CONTROL_LINES.has(lineNumber) ? styles.focusLine : ''
              } ${matched ? styles.searchMatch : ''}`}
            >
              <span>{lineNumber}</span>
              <i>{CONTROL_LINES.has(lineNumber) ? '◆' : ''}</i>
              <code>{line || ' '}</code>
            </div>
          )
        })}
      </div>
    </section>
  )
}

function ReviewSidebar({
  stage,
  leftMode,
  onLeftMode,
  draftShape,
  acceptedShape,
  draftFactors,
  acceptedFactors,
  onShape,
  onFactors,
  onRun,
  onRunDiff,
  onApply,
  onUndo,
  runMessage,
  scenario,
  workflow,
  appliedFactors,
  transformScope,
  onTransformScope,
  selectionId,
  onSelect,
}: {
  stage: ReviewStageId
  leftMode: LeftMode
  onLeftMode: (mode: LeftMode) => void
  draftShape: number
  acceptedShape: number
  draftFactors: number[]
  acceptedFactors: number[]
  onShape: (shape: number) => void
  onFactors: (factors: number[]) => void
  onRun: () => void
  onRunDiff: () => void
  onApply: () => void
  onUndo: () => void
  runMessage: string
  scenario: DemoScenario
  workflow: WorkflowState
  appliedFactors: number[]
  transformScope: TransformScope
  onTransformScope: (scope: TransformScope) => void
  selectionId: string
  onSelect: (id: string) => void
}) {
  const stageInfo = STAGES.find((item) => item.id === stage) ?? STAGES[0]
  const dirty =
    draftShape !== acceptedShape || !sameFactors(draftFactors, acceptedFactors)
  const scenarioChanged = draftShape !== acceptedShape
  const sourceChanged = !sameFactors(draftFactors, acceptedFactors)
  const reviewAvailable = workflow !== 'initial' && workflow !== 'reviewing'
  return (
    <aside
      className={`pto-ide-frame__pane pto-ide-frame__explorer ${styles.sidebar}`}
      data-ide-pane="explorer"
      id="llvmcfg-explorer-pane"
    >
      <header className="pto-ide-frame__pane-header">
        <h2 className="pto-ide-frame__pane-title">{stageInfo.title}</h2>
        <span className="pto-ide-frame__pane-meta">
          {stageInfo.label} · {stageInfo.step}
        </span>
      </header>
      <div className={`pto-ide-frame__pane-body ${styles.sidebarBody}`}>
        {stage === 'source' ? (
          <>
            <div
              className={styles.segmented}
              role="tablist"
              aria-label="Source review mode"
            >
              <button
                type="button"
                className={leftMode === 'code' ? styles.active : ''}
                onClick={() => onLeftMode('code')}
              >
                <Icon name="code" />
                代码
              </button>
              <button
                type="button"
                className={leftMode === 'review' ? styles.active : ''}
                onClick={() => onLeftMode('review')}
                disabled={!reviewAvailable}
                title={reviewAvailable ? '打开控制面板' : '请先运行控制流审查'}
              >
                <Icon name="review" />
                约束审查
              </button>
            </div>
            {leftMode === 'code' ? (
              <SourceCodePanel
                workflow={workflow}
                appliedFactors={appliedFactors}
                onUndo={onUndo}
              />
            ) : (
              <section className={styles.reviewControls}>
                <div className={styles.panelIntro}>
                  <span>控制流审查已完成</span>
                  <strong>调整场景与源码策略</strong>
                  <p>修改只进入候选草稿；运行 Diff 前不会改变中间画布。</p>
                </div>
                <div className={styles.controlGroup}>
                  <div className={styles.controlHeading}>
                    <label htmlFor="shape-t">审查场景 · 输入长度</label>
                    <em className={styles.scenarioBadge}>不修改源码</em>
                  </div>
                  <div className={styles.shapeInput}>
                    <code>输入长度：</code>
                    <input
                      id="shape-t"
                      type="number"
                      min="1"
                      max="512"
                      value={draftShape}
                      onChange={(event) =>
                        onShape(Math.max(1, Number(event.target.value) || 1))
                      }
                    />
                    <code>token</code>
                  </div>
                  <div className={styles.scenarioButtons}>
                    {[1, 16, 31, 32, 33, 64].map((value) => (
                      <button
                        type="button"
                        key={value}
                        className={draftShape === value ? styles.active : ''}
                        onClick={() => onShape(value)}
                      >
                        {value}
                      </button>
                    ))}
                  </div>
                  <p>
                    t = x_in.shape[0] 读取输入第 0 维的 token 数；range(0, t)
                    处理索引 0 至 t−1，因此 t 成为 loop 上界。只有
                    {formatInputLength(EVIDENCE_SHAPE)} 有真实运行证据。
                  </p>
                </div>
                <div className={styles.controlGroup}>
                  <div className={styles.controlHeading}>
                    <label>源码策略 · unroll_list</label>
                    <em className={styles.sourceBadge}>可应用</em>
                  </div>
                  <div className={styles.factorButtons}>
                    {DEMO_UNROLL_FACTORS.map((factor) => {
                      const enabled = draftFactors.includes(factor)
                      return (
                        <button
                          type="button"
                          key={factor}
                          className={enabled ? styles.active : ''}
                          aria-pressed={enabled}
                          onClick={() =>
                            onFactors(
                              enabled
                                ? draftFactors.filter(
                                    (candidate) => candidate !== factor,
                                  )
                                : [...draftFactors, factor].sort(
                                    (left, right) => right - left,
                                  ),
                            )
                          }
                        >
                          ×{factor}
                        </button>
                      )
                    })}
                  </div>
                  <p>
                    修改将作为当前算子的局部策略覆盖；Demo Apply
                    只更新内存代码草稿。
                  </p>
                </div>
                <div className={styles.readonlyGroup}>
                  <span>循环意图 · 只读</span>
                  <dl>
                    <div>
                      <dt>范围</dt>
                      <dd>0 → t</dd>
                    </div>
                    <div>
                      <dt>步长</dt>
                      <dd>1</dd>
                    </div>
                    <div>
                      <dt>源码</dt>
                      <dd>Python 252–253</dd>
                    </div>
                  </dl>
                </div>
                <div className={styles.constraintList}>
                  <span>
                    <b>动态边界</b>
                    <code>{SOURCE.dynamicBound.expression}</code>
                  </span>
                  <span>
                    <b>循环范围</b>
                    <code>0 → t · step 1</code>
                  </span>
                  <span>
                    <b>Tail 覆盖</b>
                    <code>
                      {scenario.remainder === 0
                        ? '完整覆盖'
                        : `余数 ${scenario.remainder} 未覆盖`}
                    </code>
                  </span>
                  <span>
                    <b>候选事实</b>
                    <em className={styles[`truth_${scenario.status}`]}>
                      {scenario.status}
                    </em>
                  </span>
                  <span>
                    <b>valid_shape</b>
                    <code>已检测 · 14 处</code>
                  </span>
                </div>
              </section>
            )}
            <section
              className={`${styles.runCard} ${
                dirty ? styles.runCardDirty : ''
              }`}
            >
              <span>
                {workflow === 'initial'
                  ? '第一步'
                  : workflow === 'reviewing'
                  ? '正在建立基线'
                  : workflow === 'diffReady'
                  ? 'Diff 已就绪'
                  : workflow === 'applied'
                  ? '源码已应用'
                  : dirty
                  ? '候选草稿'
                  : '基线已就绪'}
              </span>
              <strong>
                {workflow === 'initial'
                  ? '从当前源码开始审查'
                  : workflow === 'reviewing'
                  ? '解析源码与控制意图…'
                  : workflow === 'diffing'
                  ? '正在生成 Before / After…'
                  : workflow === 'applied'
                  ? '当前基线已经过期'
                  : dirty
                  ? `${scenarioChanged ? '场景' : ''}${
                      scenarioChanged && sourceChanged ? ' + ' : ''
                    }${sourceChanged ? '源码策略' : ''}已修改`
                  : `${formatInputLength(acceptedShape)} · ${scenario.status}`}
              </strong>
              <p>
                {runMessage ||
                  (workflow === 'initial'
                    ? '识别动态边界、循环范围和展开策略，建立 Source Flow 基线。'
                    : dirty
                    ? '草稿尚未影响画布；运行 Diff 后查看结构变化。'
                    : '当前代码与 Source Flow 基线一致。')}
              </p>
              {workflow === 'initial' || workflow === 'reviewing' ? (
                <button
                  type="button"
                  onClick={onRun}
                  disabled={workflow === 'reviewing'}
                >
                  <Icon name="play" />
                  {workflow === 'reviewing'
                    ? '正在运行审查…'
                    : '运行控制流审查'}
                </button>
              ) : null}
              {workflow === 'draft' || workflow === 'diffing' ? (
                <button
                  type="button"
                  onClick={onRunDiff}
                  disabled={workflow === 'diffing'}
                >
                  <Icon name="play" />
                  {workflow === 'diffing' ? '正在运行 Diff…' : '运行 Diff'}
                </button>
              ) : null}
              {workflow === 'diffReady' ? (
                <>
                  <button
                    type="button"
                    onClick={onApply}
                    disabled={!sourceChanged}
                    title={
                      sourceChanged
                        ? '将源码候选应用到 Demo 代码草稿'
                        : 't 是审查场景，不会写入源码'
                    }
                  >
                    <Icon name="code" />
                    应用到代码
                  </button>
                  {!sourceChanged ? (
                    <small>本次只修改了场景，无需应用到源码。</small>
                  ) : null}
                </>
              ) : null}
              {workflow === 'applied' ? (
                <button type="button" onClick={onRun}>
                  <Icon name="play" />
                  重新运行审查
                </button>
              ) : null}
            </section>
          </>
        ) : null}

        {stage === 'transform' ? (
          <section className={styles.stageMenu}>
            <span className={styles.sectionLabel}>EVIDENCE BOUNDARIES</span>
            <button
              type="button"
              className={
                transformScope === 'materialization' ? styles.active : ''
              }
              onClick={() => onTransformScope('materialization')}
            >
              <i>00</i>
              <span>
                <strong>Frontend Materialization</strong>
                <small>6 paths first observed · attribution unknown</small>
              </span>
              <em>+6</em>
            </button>
            <button
              type="button"
              className={transformScope === 'loop-unroll' ? styles.active : ''}
              onClick={() => onTransformScope('loop-unroll')}
            >
              <i>01</i>
              <span>
                <strong>Pass_00_LoopUnroll</strong>
                <small>Before / After byte-identical</small>
              </span>
              <em>0</em>
            </button>
            <div className={styles.sidebarFinding}>
              <b>关键纠偏</b>不能把六条 specialized path 归因给当前可见的
              LoopUnroll Pass。
            </div>
          </section>
        ) : null}

        {stage === 'codegen' ? (
          <section className={styles.stageMenu}>
            <span className={styles.sectionLabel}>GENERATED PATHS</span>
            {RUN.generatedPaths.map((path) => (
              <button
                type="button"
                key={path.id}
                className={selectionId === path.id ? styles.active : ''}
                onClick={() => onSelect(path.id)}
              >
                <i>×{path.factor}</i>
                <span>
                  <strong>Root magic {path.rootMagic}</strong>
                  <small>{path.finalRootOperationCount} operations</small>
                </span>
                <em>{path.mappingConfidence}</em>
              </button>
            ))}
          </section>
        ) : null}

        {stage === 'runtime' ? (
          <section className={styles.stageMenu}>
            <span className={styles.sectionLabel}>
              OBSERVED EXECUTION · t={EVIDENCE_SHAPE}
            </span>
            {RUN.generatedPaths.map((path) => (
              <button
                type="button"
                key={path.id}
                className={selectionId === path.id ? styles.active : ''}
                onClick={() => onSelect(path.id)}
              >
                <i>×{path.factor}</i>
                <span>
                  <strong>
                    {path.runtime.observed
                      ? `${path.runtime.rootInvocationCount} invocations`
                      : 'not observed'}
                  </strong>
                  <small>{path.runtime.taskEventCount} task events</small>
                </span>
                <em>{path.runtime.observed ? 'seen' : '—'}</em>
              </button>
            ))}
            <button
              type="button"
              className={
                selectionId === FIXTURE.defaultSelectionByStage.runtime
                  ? styles.active
                  : ''
              }
              onClick={() => onSelect(FIXTURE.defaultSelectionByStage.runtime)}
            >
              <i>↥</i>
              <span>
                <strong>Task 1048691 · Key-Linear</strong>
                <small>exact reverse-link sample</small>
              </span>
              <em>exact</em>
            </button>
          </section>
        ) : null}
      </div>
    </aside>
  )
}

function CanvasHeader({
  stage,
  workflow,
  draftShape,
  acceptedShape,
  draftFactors,
  acceptedFactors,
  transformScope,
  scenario,
}: {
  stage: ReviewStageId
  workflow: WorkflowState
  draftShape: number
  acceptedShape: number
  draftFactors: number[]
  acceptedFactors: number[]
  transformScope: TransformScope
  scenario: DemoScenario
}) {
  const dirty =
    draftShape !== acceptedShape || !sameFactors(draftFactors, acceptedFactors)
  const sourceCopy =
    workflow === 'initial'
      ? ['SOURCE REVIEW', '从源码建立控制流基线', '等待审查']
      : workflow === 'reviewing'
      ? ['BASELINE REVIEW', '正在分析当前算子', '审查中']
      : workflow === 'diffReady' || workflow === 'applied'
      ? [
          'CONTROL PATH DIFF',
          '基线与候选控制流对比',
          `${scenario.status} · candidate`,
        ]
      : ['SOURCE FLOW', '当前算子的控制流基线', 'baseline ready']
  const copy = {
    source: sourceCopy,
    transform:
      transformScope === 'materialization'
        ? [
            'FRONTEND MATERIALIZATION',
            '六条路径首次出现，但形成步骤未知',
            'exact observation',
          ]
        : [
            'PASS BEFORE / AFTER',
            '当前 LoopUnroll dump 没有结构变化',
            '0 delta · exact',
          ],
    codegen: [
      'GENERATED ORCHESTRATION',
      '六条 factor-specific root 路径',
      'exact mapping',
    ],
    runtime: [
      'OBSERVED EXECUTION',
      `${formatInputLength(EVIDENCE_SHAPE)} · 实际命中五条生成路径`,
      'derived scenario',
    ],
  }[stage]
  return (
    <header className={`pto-ide-frame__pane-header ${styles.canvasHeader}`}>
      <div>
        <span>{copy[0]}</span>
        <h1>{copy[1]}</h1>
        <p>{STAGES.find((item) => item.id === stage)?.question}</p>
      </div>
      <div className={styles.canvasStatus}>
        {stage === 'source' && dirty ? (
          <em>
            Preview t={draftShape} · reviewed t={acceptedShape}
          </em>
        ) : null}
        <strong>{copy[2]}</strong>
      </div>
    </header>
  )
}

function FlowCanvas({
  stage,
  workflow,
  selectionId,
  onSelect,
  transformScope,
  draftShape,
  acceptedShape,
  draftFactors,
  acceptedFactors,
  scenario,
}: {
  stage: ReviewStageId
  workflow: WorkflowState
  selectionId: string
  onSelect: (id: string) => void
  transformScope: TransformScope
  draftShape: number
  acceptedShape: number
  draftFactors: number[]
  acceptedFactors: number[]
  scenario: DemoScenario
}) {
  const config = useMemo(
    () => ({ inputT: draftShape, unrollFactors: draftFactors }),
    [draftFactors, draftShape],
  )
  const baselineConfig = useMemo(
    () => ({ inputT: acceptedShape, unrollFactors: acceptedFactors }),
    [acceptedFactors, acceptedShape],
  )
  const baselineScenario = useMemo(
    () => resolveDemoScenario(baselineConfig, EVIDENCE_SHAPE),
    [baselineConfig],
  )
  const sourceBaseline = useMemo(
    () =>
      coverageGraph(
        'baseline',
        baselineConfig,
        baselineScenario,
        scenario.eligibleFactors,
        acceptedShape === EVIDENCE_SHAPE &&
          sameFactors(acceptedFactors, DEMO_UNROLL_FACTORS)
          ? 'observed'
          : 'updated',
      ),
    [
      acceptedFactors,
      acceptedShape,
      baselineConfig,
      baselineScenario,
      scenario,
    ],
  )
  const sourceRequest = useMemo(
    () =>
      coverageGraph(
        'request',
        config,
        scenario,
        baselineScenario.eligibleFactors,
      ),
    [baselineScenario.eligibleFactors, config, scenario],
  )
  const materialization = useMemo(() => pathGraph('materialization'), [])
  const codegen = useMemo(() => pathGraph('codegen'), [])
  const runtime = useMemo(() => pathGraph('runtime'), [])
  const cfg = loopUnrollCfg as GraphState
  return (
    <section
      className={`pto-ide-frame__pane ${styles.centerPane}`}
      data-ide-pane="editor-preview"
    >
      <CanvasHeader
        stage={stage}
        workflow={workflow}
        draftShape={draftShape}
        acceptedShape={acceptedShape}
        draftFactors={draftFactors}
        acceptedFactors={acceptedFactors}
        transformScope={transformScope}
        scenario={scenario}
      />
      <div className="pto-ide-frame__pane-body">
        <div className={styles.canvasShell}>
          <div className={styles.graphArea}>
            {stage === 'source' && workflow === 'initial' ? (
              <div className={styles.emptyCanvas}>
                <span>01</span>
                <strong>尚未建立控制流基线</strong>
                <p>
                  请先在左侧查看完整源码，然后运行控制流审查。Graph
                  只会在审查完成后出现。
                </p>
              </div>
            ) : null}
            {stage === 'source' && workflow === 'reviewing' ? (
              <div className={styles.reviewProgress}>
                <span className={styles.progressMark} />
                <strong>正在建立 Baseline Review</strong>
                <ol>
                  <li className={styles.complete}>索引完整源码</li>
                  <li className={styles.active}>识别动态边界与循环意图</li>
                  <li>整理控制路径与证据</li>
                </ol>
              </div>
            ) : null}
            {stage === 'source' &&
            workflow !== 'initial' &&
            workflow !== 'reviewing' &&
            workflow !== 'diffReady' &&
            workflow !== 'applied' ? (
              <ReadOnlyGraph
                {...sourceBaseline}
                paneLabel={`BASELINE · ${formatInputLength(acceptedShape)}`}
                selectionId={selectionId}
                onSelect={onSelect}
              />
            ) : null}
            {stage === 'source' &&
            (workflow === 'diffReady' || workflow === 'applied') ? (
              <div className={styles.cfgDiff}>
                <section>
                  <ReadOnlyGraph
                    {...sourceBaseline}
                    paneLabel={`修改前 · ${formatInputLength(acceptedShape)}`}
                    selectionId={selectionId}
                    onSelect={onSelect}
                  />
                </section>
                <div className={styles.diffDivider}>
                  <span>{scenario.status === 'observed' ? '=' : 'Δ'}</span>
                  <strong>
                    +
                    {
                      scenario.eligibleFactors.filter(
                        (factor) =>
                          !baselineScenario.eligibleFactors.includes(factor),
                      ).length
                    }
                    {' / −'}
                    {
                      baselineScenario.eligibleFactors.filter(
                        (factor) => !scenario.eligibleFactors.includes(factor),
                      ).length
                    }
                  </strong>
                  <em>{scenario.status}</em>
                </div>
                <section>
                  <ReadOnlyGraph
                    {...sourceRequest}
                    paneLabel={`修改后 · ${scenario.status} · ${formatInputLength(
                      draftShape,
                    )}`}
                    selectionId={selectionId}
                    onSelect={onSelect}
                  />
                </section>
              </div>
            ) : null}
            {stage === 'transform' && transformScope === 'materialization' ? (
              <ReadOnlyGraph
                {...materialization}
                paneLabel="BOUNDARY · First observed specialized functions"
                selectionId={selectionId}
                onSelect={onSelect}
              />
            ) : null}
            {stage === 'transform' && transformScope === 'loop-unroll' ? (
              <div className={styles.cfgDiff}>
                <section>
                  <Suspense fallback={<Loading />}>
                    <LayoutFlowFactory
                      llvmJson={cfg.before_json}
                      llvmJson_compare={cfg.after_json}
                      llvmOutput={cfg.before_output}
                      title="dev"
                      paneLabel="BEFORE · Pass_00_LoopUnroll"
                      variant="simpleSmall"
                      selectedBlockId={selectionId}
                      onSelectBlock={(_, id) => onSelect(id)}
                      readOnly
                      showMinimap={false}
                    />
                  </Suspense>
                </section>
                <div className={styles.diffDivider}>
                  <span>=</span>
                  <strong>0</strong>
                  <em>byte-identical</em>
                </div>
                <section>
                  <Suspense fallback={<Loading />}>
                    <LayoutFlowFactory
                      llvmJson={cfg.after_json}
                      llvmJson_compare={cfg.before_json}
                      llvmOutput={cfg.after_output}
                      title="host"
                      paneLabel="AFTER · Pass_00_LoopUnroll"
                      variant="simpleSmall"
                      selectedBlockId={selectionId}
                      onSelectBlock={(_, id) => onSelect(id)}
                      readOnly
                      showMinimap={false}
                    />
                  </Suspense>
                </section>
              </div>
            ) : null}
            {stage === 'codegen' ? (
              <ReadOnlyGraph
                {...codegen}
                paneLabel="GENERATED · controlFlow.cpp roots"
                selectionId={selectionId}
                onSelect={onSelect}
              />
            ) : null}
            {stage === 'runtime' ? (
              <ReadOnlyGraph
                {...runtime}
                paneLabel="RUNTIME · merged_swimlane task identity"
                selectionId={selectionId}
                onSelect={onSelect}
              />
            ) : null}
          </div>
          {stage === 'source' &&
          (workflow === 'diffReady' || workflow === 'applied') ? (
            <div
              className={`${styles.sourceDiff} ${
                scenario.status !== 'observed' ? styles.sourceDiffDraft : ''
              }`}
            >
              <span>
                <b>BASELINE</b>
                <code>{formatInputLength(acceptedShape)}</code>
                <small>Baseline · 已审查</small>
              </span>
              <i>→</i>
              <span>
                <b>REQUEST</b>
                <code>{formatInputLength(draftShape)}</code>
                <small>{scenario.status} · What-if review</small>
              </span>
              <strong>
                {scenario.status === 'observed'
                  ? '0 DIFF · evidence match'
                  : scenario.status === 'projected'
                  ? 'PREVIEW · not compiled'
                  : `GAP · ${scenario.remainder} uncovered`}
              </strong>
            </div>
          ) : null}
          {stage === 'transform' ? (
            <div className={styles.evidenceNote}>
              <span>!</span>
              <p>
                <strong>Attribution remains unknown.</strong> The six variants
                exist before the per-function pipeline; the visible LoopUnroll
                snapshot cannot prove where they were created.
              </p>
            </div>
          ) : null}
        </div>
      </div>
    </section>
  )
}

function DiagnosticSection({
  title,
  children,
}: {
  title: string
  children: React.ReactNode
}) {
  return (
    <section className={styles.diagnosticSection}>
      <h3>{title}</h3>
      {children}
    </section>
  )
}

function Inspector({ selection }: { selection: ReviewSelection }) {
  return (
    <aside
      className={`pto-ide-frame__pane ${styles.inspector}`}
      data-ide-pane="inspector"
      id="llvmcfg-inspector-pane"
    >
      <header className="pto-ide-frame__pane-header">
        <h2 className="pto-ide-frame__pane-title">证据与诊断</h2>
        <span className="pto-ide-frame__pane-meta">
          DIAGNOSTIC INSPECTOR · {selection.confidence}
        </span>
      </header>
      <div className={`pto-ide-frame__pane-body ${styles.inspectorBody}`}>
        <section className={styles.selectionCard}>
          <span>{selection.kind}</span>
          <h2>{selection.title}</h2>
          <p>{selection.subtitle}</p>
        </section>
        <DiagnosticSection title="身份">
          <dl>
            {selection.identity.map(([term, value]) => (
              <div key={term}>
                <dt>{term}</dt>
                <dd>{value}</dd>
              </div>
            ))}
          </dl>
        </DiagnosticSection>
        <DiagnosticSection title="作用">
          <p>{selection.role}</p>
        </DiagnosticSection>
        <DiagnosticSection title="变化">
          <p>{selection.change}</p>
        </DiagnosticSection>
        <DiagnosticSection title="影响">
          <div className={styles.impact}>{selection.impact}</div>
        </DiagnosticSection>
        <DiagnosticSection title="下一步">
          <div className={styles.nextStep}>{selection.nextStep}</div>
        </DiagnosticSection>
        <details className={styles.rawEvidence}>
          <summary>原始证据与技术身份</summary>
          <dl>
            {selection.rawEvidence.map(([term, value]) => (
              <div key={term}>
                <dt>{term}</dt>
                <dd>{value}</dd>
              </div>
            ))}
          </dl>
        </details>
      </div>
    </aside>
  )
}

function demoSelection(
  selectionId: string,
  config: DemoReviewConfig,
  scenario: DemoScenario,
): ReviewSelection | null {
  if (!selectionId.startsWith('demo:')) return null
  const added = scenario.eligibleFactors.filter(
    (factor) => !OBSERVED_FACTORS.includes(factor),
  )
  const removed = OBSERVED_FACTORS.filter(
    (factor) => !scenario.eligibleFactors.includes(factor),
  )
  const confidence =
    scenario.status === 'observed'
      ? 'derived'
      : scenario.status === 'projected'
      ? 'heuristic'
      : 'unknown'

  if (selectionId === 'demo:scenario') {
    const statusCopy: Record<DemoTruthStatus, string> = {
      observed: '与现有“输入长度：16 token”artifact 和 runtime 证据一致。',
      projected: '这是约束驱动的 What-if 结果，尚未经过真实编译或运行。',
      unavailable: `当前 policy 留下 ${scenario.remainder} 个 iteration 无法覆盖。`,
    }
    return {
      id: selectionId,
      stage: 'source',
      kind: `${scenario.status} review scenario`,
      title: `Control coverage · ${formatInputLength(config.inputT)}`,
      subtitle: statusCopy[scenario.status],
      confidence,
      identity: [
        ['输入长度', formatInputLength(config.inputT)],
        ['展开档位', config.unrollFactors.join(', ') || 'none'],
        ['事实状态', scenario.status],
        ['基线', `Observed · ${formatInputLength(EVIDENCE_SHAPE)}`],
      ],
      role: '比较真实基线与当前约束请求下可能具备资格的 factor-specific control paths。',
      change:
        scenario.status === 'observed'
          ? '请求与真实基线一致，节点和路径均无变化。'
          : `相对基线新增 ${added.length} 条、移除 ${removed.length} 条 eligible path。`,
      impact:
        scenario.status === 'unavailable'
          ? '当前配置无法形成完整 coverage，Demo 不会补造 tail path，也不会显示为编译成功。'
          : scenario.status === 'projected'
          ? '该结果适合评审控制意图，但不能证明 compiler 最终会生成或 runtime 一定会命中这些路径。'
          : '可以继续沿 Transformation、Generated 和 Runtime 查看真实证据链。',
      nextStep:
        scenario.status === 'unavailable'
          ? '恢复能够覆盖余数的较小 factor（通常是 ×1），再确认审查。'
          : scenario.status === 'projected'
          ? '评审新增/移除路径是否符合意图；需要工程验证时再接入真实 manifest。'
          : '切换到 Transformation，确认路径首次出现的证据边界。',
      rawEvidence: [
        ['projection rule', 'factor <= t + greedy coverage witness'],
        ['added factors', added.join(', ') || 'none'],
        ['removed factors', removed.join(', ') || 'none'],
        [
          'artifact manifest',
          scenario.status === 'observed' ? RUN.id : 'not produced',
        ],
      ],
    }
  }

  const factor = Number(selectionId.replace('demo:path:', ''))
  if (!Number.isFinite(factor)) return null
  const baselineHas = OBSERVED_FACTORS.includes(factor)
  const requestHas = scenario.eligibleFactors.includes(factor)
  const diffStatus =
    baselineHas && requestHas ? 'unchanged' : requestHas ? 'added' : 'removed'
  const witness = scenario.coverage.find((item) => item.factor === factor)
  const path = RUN.generatedPaths.find(
    (candidate) => candidate.factor === factor,
  )
  return {
    id: selectionId,
    stage: 'source',
    kind: `${diffStatus} control path`,
    title: `Unroll ×${factor}`,
    subtitle:
      diffStatus === 'added'
        ? 'Request 侧新增的 eligible path'
        : diffStatus === 'removed'
        ? '只存在于 Observed baseline'
        : 'Baseline 与 Request 均保留',
    confidence:
      baselineHas && scenario.status === 'observed' ? 'derived' : confidence,
    identity: [
      ['Factor', `×${factor}`],
      ['Diff 状态', diffStatus],
      ['Baseline', baselineHas ? 'observed' : 'absent'],
      ['Request', requestHas ? scenario.status : 'ineligible'],
    ],
    role: `当剩余循环长度允许时，每次覆盖 ${factor} 个 iteration 的控制路径。`,
    change:
      diffStatus === 'added'
        ? `${formatInputLength(config.inputT)}，使 ×${factor} 首次具备资格。`
        : diffStatus === 'removed'
        ? `当前 shape 或 policy 使 ×${factor} 不再具备资格。`
        : '该 factor 在真实基线与当前请求中都具备资格。',
    impact:
      scenario.status === 'observed'
        ? `捕获运行中记录了 ${
            path?.runtime.rootInvocationCount ?? 0
          } 次 root invocation。`
        : `Coverage witness 使用 ${
            witness?.witnessCount ?? 0
          } 次；它只解释 policy 可覆盖性，不预测真实调用次数。`,
    nextStep:
      scenario.status === 'observed'
        ? '进入 Generated 或 Runtime 查看 root identity 与 task 证据。'
        : '确认该路径变化符合算子意图；需要事实结论时再运行真实编译。',
    rawEvidence: [
      ['projection', 'eligible when factor <= dynamic bound'],
      ['coverage witness', String(witness?.witnessCount ?? 0)],
      ['generated root', path?.rootHash ?? 'not available'],
      [
        'runtime evidence',
        baselineHas ? 'captured · 输入长度：16 token' : 'none',
      ],
    ],
  }
}

function workflowSelection(
  workflow: WorkflowState,
  acceptedShape: number,
  acceptedFactors: number[],
  draftShape: number,
  draftFactors: number[],
  scenario: DemoScenario,
): ReviewSelection | null {
  const scenarioChanged = draftShape !== acceptedShape
  const sourceChanged = !sameFactors(draftFactors, acceptedFactors)
  if (workflow === 'initial') {
    return {
      id: 'workflow:initial',
      stage: 'source',
      kind: '等待操作',
      title: '尚未运行控制流审查',
      subtitle: '当前只完成了轻量源码索引，画布不会预先展示 Sample Graph。',
      confidence: 'unknown',
      identity: [
        ['文件', 'lightning_indexer_prolog_quant.py'],
        ['源码规模', '364 行'],
        ['工作区', 'Demo · 内存模式'],
      ],
      role: '从用户自己的完整源码开始建立可信的控制流基线。',
      change: '尚未形成任何 Graph 或候选修改。',
      impact:
        '运行审查后，系统会识别动态边界、循环范围、展开策略和 tail coverage。',
      nextStep: '在左侧查看源码，然后点击“运行控制流审查”。',
      rawEvidence: [['source snapshot', SOURCE_PATH]],
    }
  }
  if (workflow === 'reviewing' || workflow === 'diffing') {
    return {
      id: `workflow:${workflow}`,
      stage: 'source',
      kind: workflow === 'reviewing' ? 'Baseline Review' : 'Candidate Diff',
      title:
        workflow === 'reviewing' ? '正在建立控制流基线' : '正在生成控制流 Diff',
      subtitle: 'Demo 正在演示解析、编译和证据整理流水线。',
      confidence: 'derived',
      identity: [
        [
          '当前步骤',
          workflow === 'reviewing' ? 'Source indexing' : 'Candidate analysis',
        ],
      ],
      role: '将源码控制意图转换为可比较的语义 Flow。',
      change: '处理完成前保留上一份可信画布。',
      impact: '不会修改真实源码或工程文件。',
      nextStep: '等待当前操作完成。',
      rawEvidence: [['mode', 'front-end demo simulation']],
    }
  }
  if (
    workflow === 'draft' ||
    workflow === 'diffReady' ||
    workflow === 'applied'
  ) {
    const stateCopy =
      workflow === 'draft'
        ? '候选修改尚未运行 Diff。'
        : workflow === 'diffReady'
        ? 'Before / After Diff 已生成。'
        : '源码候选已应用到内存代码草稿，原基线已过期。'
    return {
      id: `workflow:${workflow}`,
      stage: 'source',
      kind: workflow === 'applied' ? '已应用候选' : '候选变化原因',
      title: `${scenarioChanged ? '场景变化' : ''}${
        scenarioChanged && sourceChanged ? ' + ' : ''
      }${sourceChanged ? '源码策略变化' : ''}`,
      subtitle: stateCopy,
      confidence: scenario.status === 'observed' ? 'derived' : 'heuristic',
      identity: [
        [
          '审查场景',
          scenarioChanged
            ? `${formatInputLength(acceptedShape)} → ${formatInputLength(
                draftShape,
              )}`
            : '未修改',
        ],
        [
          '源码策略',
          sourceChanged
            ? `[${acceptedFactors.join(', ')}] → [${draftFactors.join(', ')}]`
            : '未修改',
        ],
        [
          'Tail 覆盖',
          scenario.remainder === 0 ? '完整' : `缺口 ${scenario.remainder}`,
        ],
      ],
      role: '分开解释临时场景输入与可以应用到源码的策略候选。',
      change: `${scenarioChanged ? '输入长度只改变本次分析场景；' : ''}${
        sourceChanged ? 'unroll_list 改变可生成的展开路径。' : ''
      }`,
      impact:
        scenario.status === 'unavailable'
          ? `当前策略留下 ${scenario.remainder} 个 iteration 无法覆盖。`
          : `候选包含 ${scenario.eligibleFactors.length} 条 eligible path。`,
      nextStep:
        workflow === 'draft'
          ? '运行 Diff，查看两个变化来源共同产生的结构结果。'
          : workflow === 'diffReady' && sourceChanged
          ? '确认 Diff 后应用源码策略；输入长度场景不会被写入源码。'
          : workflow === 'applied'
          ? '查看左侧红绿源码 Diff，然后重新运行审查。'
          : '本次只有场景变化，无需应用源码。',
      rawEvidence: [
        [
          'scenario patch',
          scenarioChanged ? formatInputLength(draftShape) : 'none',
        ],
        ['source patch', sourceChanged ? 'unroll_list local override' : 'none'],
        ['demo write mode', 'in-memory only'],
      ],
    }
  }
  return null
}

function LLVMcfg() {
  const frameRef = useRef<HTMLElement>(null)
  const actionTimerRef = useRef<number | null>(null)
  const [stage, setStage] = useState<ReviewStageId>('source')
  const [workflow, setWorkflow] = useState<WorkflowState>('initial')
  const [theme, setTheme] = useState<'light' | 'dark'>('light')
  const [leftMode, setLeftMode] = useState<LeftMode>('code')
  const [transformScope, setTransformScope] =
    useState<TransformScope>('materialization')
  const [draftShape, setDraftShape] = useState(EVIDENCE_SHAPE)
  const [acceptedShape, setAcceptedShape] = useState(EVIDENCE_SHAPE)
  const [draftFactors, setDraftFactors] =
    useState<number[]>(DEMO_UNROLL_FACTORS)
  const [acceptedFactors, setAcceptedFactors] =
    useState<number[]>(DEMO_UNROLL_FACTORS)
  const [appliedFactors, setAppliedFactors] =
    useState<number[]>(DEMO_UNROLL_FACTORS)
  const [reviewedStatus, setReviewedStatus] =
    useState<DemoTruthStatus>('observed')
  const [runMessage, setRunMessage] = useState('')
  const [selectionId, setSelectionId] = useState(
    FIXTURE.defaultSelectionByStage.source,
  )
  const scenario = useMemo(
    () =>
      resolveDemoScenario(
        { inputT: draftShape, unrollFactors: draftFactors },
        EVIDENCE_SHAPE,
      ),
    [draftFactors, draftShape],
  )

  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])
  useEffect(() => {
    const frame = frameRef.current
    const ideFrame = (
      window as Window & {
        PtoIdeFrame?: {
          init: (root: HTMLElement) => { destroy?: () => void } | undefined
        }
      }
    ).PtoIdeFrame
    if (!frame || !ideFrame?.init) return
    const controller = ideFrame.init(frame)
    return () => controller?.destroy?.()
  }, [])
  useEffect(
    () => () => {
      if (actionTimerRef.current !== null) {
        window.clearTimeout(actionTimerRef.current)
      }
    },
    [],
  )

  const changeStage = (next: ReviewStageId) => {
    if (
      (workflow === 'initial' || workflow === 'reviewing') &&
      next !== 'source'
    ) {
      return
    }
    setStage(next)
    setSelectionId(FIXTURE.defaultSelectionByStage[next])
  }
  const runReview = () => {
    if (actionTimerRef.current !== null)
      window.clearTimeout(actionTimerRef.current)
    setStage('source')
    setLeftMode('code')
    setWorkflow('reviewing')
    setRunMessage('正在索引完整源码并整理控制流证据。')
    setSelectionId('workflow:reviewing')
    actionTimerRef.current = window.setTimeout(() => {
      setAcceptedShape(draftShape)
      setAcceptedFactors(draftFactors)
      setAppliedFactors(draftFactors)
      setReviewedStatus(scenario.status)
      setLeftMode('review')
      setWorkflow('baselineReady')
      setSelectionId('demo:scenario')
      setRunMessage('控制流基线已建立。调整场景或源码策略后运行 Diff。')
      actionTimerRef.current = null
    }, 900)
  }
  const runDiff = () => {
    if (actionTimerRef.current !== null)
      window.clearTimeout(actionTimerRef.current)
    setWorkflow('diffing')
    setRunMessage('正在分析场景与源码候选，不会修改当前基线。')
    setSelectionId('workflow:diffing')
    actionTimerRef.current = window.setTimeout(() => {
      setWorkflow('diffReady')
      setRunMessage('Diff 已生成。场景变化与源码变化已分别标注。')
      setSelectionId('workflow:diffReady')
      actionTimerRef.current = null
    }, 700)
  }
  const applyToCode = () => {
    if (sameFactors(draftFactors, acceptedFactors)) return
    setAppliedFactors(draftFactors)
    setWorkflow('applied')
    setLeftMode('code')
    setRunMessage('源码候选已应用到 Demo 内存草稿；当前基线已过期。')
    setSelectionId('workflow:applied')
  }
  const undoApply = () => {
    setAppliedFactors(acceptedFactors)
    setWorkflow('diffReady')
    setRunMessage('已撤销代码应用，候选 Diff 仍保留。')
    setSelectionId('workflow:diffReady')
  }
  const baseSelection =
    RUN.selections[selectionId] ??
    RUN.selections[FIXTURE.defaultSelectionByStage[stage]]
  const selected = useMemo<ReviewSelection>(() => {
    return (
      (stage === 'source' &&
      (selectionId.startsWith('workflow:') ||
        workflow === 'initial' ||
        workflow === 'reviewing')
        ? workflowSelection(
            workflow,
            acceptedShape,
            acceptedFactors,
            draftShape,
            draftFactors,
            scenario,
          )
        : null) ??
      demoSelection(
        selectionId,
        { inputT: draftShape, unrollFactors: draftFactors },
        scenario,
      ) ??
      baseSelection
    )
  }, [
    acceptedFactors,
    acceptedShape,
    baseSelection,
    draftFactors,
    draftShape,
    scenario,
    selectionId,
    stage,
    workflow,
  ])
  const active = STAGES.find((item) => item.id === stage) ?? STAGES[0]

  return (
    <main
      ref={frameRef}
      className={`pto-ide-frame ${styles.page}`}
      data-ide-frame
      data-host="standalone"
    >
      <header className="pto-ide-frame__topbar">
        <div className={`pto-ide-frame__topbar-left ${styles.product}`}>
          <span className={styles.productMark}>
            <Icon name="flow" />
          </span>
          <div>
            <strong>PyPTO Control Flow Explorer</strong>
            <span>Intent → compiler → runtime evidence</span>
          </div>
        </div>
        <nav
          className={`pto-ide-frame__topbar-center ${styles.stageNav}`}
          aria-label="Review stages"
        >
          {STAGES.map((item, index) => (
            <React.Fragment key={item.id}>
              <button
                type="button"
                className={stage === item.id ? styles.active : ''}
                onClick={() => changeStage(item.id)}
                disabled={
                  (workflow === 'initial' || workflow === 'reviewing') &&
                  item.id !== 'source'
                }
              >
                <span>{item.step}</span>
                <strong>{item.title}</strong>
              </button>
              {index < STAGES.length - 1 ? (
                <i>
                  <Icon name="arrow" />
                </i>
              ) : null}
            </React.Fragment>
          ))}
        </nav>
        <div className={`pto-ide-frame__topbar-right ${styles.windowActions}`}>
          <button
            type="button"
            className="pto-ide-frame__window-action"
            onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
            aria-label="切换主题"
            title="切换主题"
          >
            <Icon name={theme === 'light' ? 'sun' : 'moon'} />
          </button>
          <button
            type="button"
            className="pto-ide-frame__window-action"
            data-ide-toggle="inspector"
            aria-controls="llvmcfg-inspector-pane"
            aria-label="切换 Inspector"
            title="切换 Inspector"
          >
            <Icon name="panel" />
          </button>
        </div>
      </header>
      <div className="pto-ide-frame__body">
        <nav
          className="pto-ide-frame__activity-rail pto-ide-frame__standalone-chrome"
          data-ide-slot="activity-rail"
          aria-label="Activity rail"
        >
          <button
            className="btn btn-icon btn-ghost pto-ide-frame__rail-button is-selected"
            type="button"
            data-ide-toggle="explorer"
            aria-label="Toggle explorer"
            aria-controls="llvmcfg-explorer-pane"
            aria-expanded="true"
            aria-pressed="true"
            title="Explorer"
          >
            <svg
              className="pto-ide-frame__rail-icon"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9l-.81-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z" />
            </svg>
          </button>
          <button
            className="btn btn-icon btn-ghost pto-ide-frame__rail-button"
            type="button"
            aria-label="Search"
            title="Search"
          >
            <svg
              className="pto-ide-frame__rail-icon"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.3-4.3" />
            </svg>
          </button>
          <button
            className="btn btn-icon btn-ghost pto-ide-frame__rail-button"
            type="button"
            aria-label="Source control"
            title="Source control"
          >
            <svg
              className="pto-ide-frame__rail-icon"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <circle cx="18" cy="18" r="3" />
              <circle cx="6" cy="6" r="3" />
              <path d="M13 6h3a2 2 0 0 1 2 2v7" />
              <path d="M6 9v12" />
            </svg>
          </button>
          <button
            className="btn btn-icon btn-ghost pto-ide-frame__rail-button"
            type="button"
            aria-label="Terminal"
            title="Terminal"
          >
            <svg
              className="pto-ide-frame__rail-icon"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <rect x="3" y="5" width="18" height="14" rx="2" />
              <path d="m7 9 3 3-3 3" />
              <path d="M12 15h5" />
            </svg>
          </button>
        </nav>
        <div className="pto-ide-frame__workarea">
          <div
            className="pto-ide-frame__split"
            data-ide-split="standalone-main"
            data-storage-key="llvmcfg-main-split-v2"
            data-split-direction="horizontal"
            data-sizes="24,52,24"
            data-pixel-sizes="320"
            data-min-size="280,560,300"
          >
            <ReviewSidebar
              stage={stage}
              leftMode={leftMode}
              onLeftMode={setLeftMode}
              draftShape={draftShape}
              acceptedShape={acceptedShape}
              draftFactors={draftFactors}
              acceptedFactors={acceptedFactors}
              onShape={(value) => {
                setDraftShape(value)
                const changed =
                  value !== acceptedShape ||
                  !sameFactors(draftFactors, acceptedFactors)
                setWorkflow(changed ? 'draft' : 'baselineReady')
                setRunMessage(
                  changed
                    ? '场景草稿已记录；画布仍保留当前基线。'
                    : '草稿已恢复为当前基线。',
                )
                setSelectionId(changed ? 'workflow:draft' : 'demo:scenario')
              }}
              onFactors={(factors) => {
                setDraftFactors(factors)
                const changed =
                  draftShape !== acceptedShape ||
                  !sameFactors(factors, acceptedFactors)
                setWorkflow(changed ? 'draft' : 'baselineReady')
                setRunMessage(
                  changed
                    ? '源码策略草稿已记录；运行 Diff 前不会改变画布。'
                    : '草稿已恢复为当前基线。',
                )
                setSelectionId(changed ? 'workflow:draft' : 'demo:scenario')
              }}
              onRun={runReview}
              onRunDiff={runDiff}
              onApply={applyToCode}
              onUndo={undoApply}
              runMessage={runMessage}
              scenario={scenario}
              workflow={workflow}
              appliedFactors={appliedFactors}
              transformScope={transformScope}
              onTransformScope={(scope) => {
                setTransformScope(scope)
                setSelectionId(FIXTURE.defaultSelectionByStage.transform)
              }}
              selectionId={selectionId}
              onSelect={setSelectionId}
            />
            <FlowCanvas
              stage={stage}
              workflow={workflow}
              selectionId={selectionId}
              onSelect={setSelectionId}
              transformScope={transformScope}
              draftShape={draftShape}
              acceptedShape={acceptedShape}
              draftFactors={draftFactors}
              acceptedFactors={acceptedFactors}
              scenario={scenario}
            />
            <Inspector selection={selected} />
          </div>
        </div>
      </div>
      <footer className={`pto-ide-frame__status-strip ${styles.statusbar}`}>
        <span>
          <b>●</b>
          {active.label}
        </span>
        <span>
          {workflow === 'initial'
            ? '等待首次审查'
            : workflow === 'applied'
            ? '源码已变化 · 基线过期'
            : `${reviewedStatus} · baseline ready`}
        </span>
        <span>
          <Icon name="link" />
          {LOOP_NAME}
        </span>
        <span>
          Python {SOURCE.startLine}–{SOURCE.endLine}
        </span>
        <span>Demo 工作区 · 不写入磁盘</span>
      </footer>
    </main>
  )
}

export default LLVMcfg
