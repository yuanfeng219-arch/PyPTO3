export type ReviewStageId = 'source' | 'transform' | 'codegen' | 'runtime'

export type EvidenceConfidence = 'exact' | 'derived' | 'heuristic' | 'unknown'

export type EvidenceCompleteness = 'complete' | 'partial' | 'missing'

export type FindingSeverity = 'info' | 'warning' | 'error'

export interface EvidenceValue<T> {
  value: T
  confidence: EvidenceConfidence
  derivation?: string
}

export interface SourceControlRegion {
  id: string
  kind: 'loop_unroll' | 'loop' | 'branch' | 'dynamic_path'
  file: string
  startLine: number
  endLine: number
  userName: string
  indexName: string
  range: {
    start: string
    end: string
    step: string
  }
  dynamicBound: {
    expression: string
    sourceLine: number
  }
  requestedUnrollConfig: {
    expression: string
    sourceLine: number
    concreteValueInSource: number[] | null
  }
  effectiveUnrollFactors: {
    values: number[]
    confidence: EvidenceConfidence
    evidence: string
  }
}

export interface GeneratedControlPath {
  id: string
  factor: number
  order: number
  outerFunctionHash: string
  pathMagic: number
  pathHash: string
  rootMagic: number
  rootHash: string
  condition: string
  finalRootOperationCount: number
  mappingConfidence: EvidenceConfidence
  runtime: {
    observed: boolean
    rootInvocationCount: number
    taskEventsPerInvocation: number | null
    taskEventCount: number
    durationTotalTraceUnits: number
  }
}

export interface IdentityHop {
  relation: string
  target: string
  confidence: EvidenceConfidence
  evidence: string
}

export interface RuntimeEvidenceEvent {
  id: string
  taskId: number
  name: string
  pid: number
  tid: number
  timestamp: number
  duration: number
  semanticLabel: string
  rootHash: string
  callOpMagic: number
  leafHash: string
  identityChain: IdentityHop[]
}

export interface ReviewFinding {
  id: string
  severity: FindingSeverity
  confidence: EvidenceConfidence
  summary: string
  productMeaning: string
}

export interface IdentityIndex {
  schemaVersion: string
  status: string
  generatedAt: string
  reviewRun: {
    id: string
    artifactRoot: string
    entryFunction: {
      name: string
      functionHash: string
      funcmagic: number
    }
    sourceSnapshot: {
      path: string
      contentHash: string | null
      identityStatus: string
      note: string
    }
    runtimeScenario: {
      id: string
      inputT: EvidenceValue<number>
    }
  }
  sourceRegions: SourceControlRegion[]
  compilerEvidence: {
    loopUnrollPass: {
      passId: string
      beforeArtifact: string
      afterArtifact: string
      beforeSha256: string
      afterSha256: string
      structuralDeltaObserved: boolean
      dumpFunctionCount: number
      dumpFunction: string
      attribution: {
        status: string
        confidence: EvidenceConfidence
        note: string
      }
    }
    firstObservedSpecializedFunctions: {
      boundary: string
      artifactDirectory: string
      confidence: EvidenceConfidence
      functions: Array<{
        factor: number
        pathMagic: number
        pathHash: string
        operationCount: number
        reportedSource: string
      }>
    }
  }
  generatedPaths: GeneratedControlPath[]
  runtimeSummary: {
    artifact: string
    traceEventCount: number
    taskEventCount: number
    reshapeFakeTaskEventCount: number
    controlLoopTaskEventCount: number
    controlLoopDurationTotalTraceUnits: number
    observedFactors: number[]
    unobservedFactors: number[]
    rootInvocationCounts: Record<string, number>
    generatedIntervalsOverlap: boolean
    frameworkRevisionVerified: boolean
  }
  sampleRuntimeEvents: RuntimeEvidenceEvent[]
  semanticSourceDrift: Array<{
    label: string
    compiledLine: number
    localArtifactLine: number
  }>
  findings: ReviewFinding[]
  evidenceCompleteness: Record<string, string>
}

export interface ReviewSelection {
  id: string
  stage: ReviewStageId
  kind: string
  title: string
  subtitle: string
  confidence: EvidenceConfidence
  identity: Array<[string, string]>
  role: string
  change: string
  impact: string
  nextStep: string
  rawEvidence: Array<[string, string]>
}

export interface ReviewRun {
  id: string
  status: 'fixture' | 'draft' | 'running' | 'complete' | 'failed'
  generatedAt: string
  sourceSnapshot: IdentityIndex['reviewRun']['sourceSnapshot']
  sourceRegion: SourceControlRegion
  compileEvidence: IdentityIndex['compilerEvidence']
  generatedPaths: GeneratedControlPath[]
  runtimeScenario: IdentityIndex['reviewRun']['runtimeScenario']
  runtimeSummary: IdentityIndex['runtimeSummary']
  findings: ReviewFinding[]
  selections: Record<string, ReviewSelection>
}

export interface ControlFlowReviewFixture {
  identityIndex: IdentityIndex
  reviewRun: ReviewRun
  defaultSelectionByStage: Record<ReviewStageId, string>
}

export interface ReviewRunCapture {
  schemaVersion: string
  id: string
  status: 'complete' | 'draft'
  capturedAt: string
  requestedScenario: {
    inputT: number
    evidenceStatus: 'bound' | 'request-only'
  }
  baselineScenario: {
    inputT: number
    confidence: EvidenceConfidence
    runId: string
  }
  sourceSnapshot: {
    path: string
    exists: boolean
    sha256: string | null
    expectedSha256: string | null
    verified: boolean
    sizeBytes: number | null
    identityStatus: 'captured' | 'missing'
  }
  frontendMaterialization: {
    boundary: string
    artifactDirectory: string
    dumpCount: number
    functions: Array<{
      artifact: string
      sha256: string
      funcMagic: number
      functionHash: string
      functionName: string
      operationCount: number
    }>
    expectedFunctionHashes: string[]
    missingFunctionHashes: string[]
    verified: boolean
    attribution: 'unknown'
  }
  loopUnrollEvidence: {
    passId: string
    beforeSha256: string | null
    afterSha256: string | null
    byteIdentical: boolean
    verified: boolean
  }
  evidenceCompleteness: Record<string, string>
  message: string
}

export type ReviewRunProductionStatus =
  | 'queued'
  | 'running'
  | 'complete'
  | 'blocked'
  | 'failed'

export interface ReviewRunProduction {
  schemaVersion: string
  id: string
  status: ReviewRunProductionStatus
  producer: 'auto' | 'artifact-import' | 'pypto-jit'
  requestedScenario: {
    inputT: number
  }
  submittedAt: string
  updatedAt: string
  message: string
  reasonCode?:
    | 'controlled-runner-not-configured'
    | 'artifact-scenario-mismatch'
    | 'producer-failed'
  result?: ReviewRunCapture
  manifest?: {
    schemaVersion: string
    path: string
    sha256: string
    artifactDigest: string
    artifactFileCount: number
    compilerRevisionStatus: 'unverified'
    frameworkRevisionStatus: 'unverified'
  }
}
