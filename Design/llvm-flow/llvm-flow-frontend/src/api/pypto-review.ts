import { httpClient } from '@/api/http-common'
import {
  ReviewRunCapture,
  ReviewRunProduction,
} from '@/types/controlFlowReview'

export async function capturePyPTOReviewRun(inputT: number) {
  const response = await httpClient.post<ReviewRunCapture>(
    '/pypto/review-runs',
    { input_t: inputT },
  )
  return response.data
}

export async function startPyPTOReviewRun(inputT: number) {
  const response = await httpClient.post<ReviewRunProduction>(
    '/pypto/review-runs/produce',
    { input_t: inputT, producer: 'auto' },
  )
  return response.data
}

export async function getPyPTOReviewRun(runId: string) {
  const response = await httpClient.get<ReviewRunProduction>(
    `/pypto/review-runs/${runId}`,
  )
  return response.data
}

const TERMINAL_STATES = new Set(['complete', 'blocked', 'failed'])

export async function producePyPTOReviewRun(inputT: number) {
  let state = await startPyPTOReviewRun(inputT)
  for (let attempt = 0; attempt < 80; attempt += 1) {
    if (TERMINAL_STATES.has(state.status)) return state
    await new Promise((resolve) => window.setTimeout(resolve, 250))
    state = await getPyPTOReviewRun(state.id)
  }
  return state
}
