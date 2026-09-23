export type DemoTruthStatus = 'observed' | 'projected' | 'unavailable'

export interface DemoReviewConfig {
  inputT: number
  unrollFactors: number[]
}

export interface DemoPathCoverage {
  factor: number
  witnessCount: number
}

export interface DemoScenario {
  status: DemoTruthStatus
  eligibleFactors: number[]
  coverage: DemoPathCoverage[]
  remainder: number
}

export const DEMO_UNROLL_FACTORS = [32, 16, 8, 4, 2, 1]

export function sameFactors(left: number[], right: number[]) {
  return (
    left.length === right.length &&
    left.every((factor, index) => factor === right[index])
  )
}

export function resolveDemoScenario(
  config: DemoReviewConfig,
  observedInputT: number,
): DemoScenario {
  const factors = [...new Set(config.unrollFactors)]
    .filter((factor) => factor > 0)
    .sort((left, right) => right - left)
  let remainder = config.inputT
  const coverage = factors.map((factor) => {
    const witnessCount = Math.floor(remainder / factor)
    remainder %= factor
    return { factor, witnessCount }
  })
  const observed =
    config.inputT === observedInputT &&
    sameFactors(factors, DEMO_UNROLL_FACTORS)

  return {
    status: observed ? 'observed' : remainder === 0 ? 'projected' : 'unavailable',
    eligibleFactors: factors.filter((factor) => factor <= config.inputT),
    coverage,
    remainder,
  }
}
