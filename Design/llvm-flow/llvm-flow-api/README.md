# LLVM Flow API (FastAPI)

## PyPTO control-flow Review Run

`POST /pypto/review-runs` performs a read-only capture of the canonical local
PyPTO evidence set. The request body is `{ "input_t": 16 }`. A run is
`complete` only when the source snapshot, LoopUnroll hashes, and frontend
materialization function identities validate and the requested shape matches
the artifact-backed scenario. Other shapes remain `draft`; this endpoint does
not invoke the compiler.

The producer lifecycle uses three additional endpoints:

- `GET /pypto/review-runs/capabilities` reports whether controlled producers
  are available.
- `POST /pypto/review-runs/produce` accepts
  `{ "input_t": 16, "producer": "auto" }` and returns `202 queued`.
- `GET /pypto/review-runs/{id}` returns `queued`, `running`, `complete`,
  `blocked`, or `failed` state.

For the canonical `t=16` scenario, `auto` selects `artifact-import`, validates
the nine-file minimum control-flow evidence set, and writes a read-only,
immutable manifest under `media/pypto-control-flow-review-runs/{id}`. For new
shapes, `auto` selects `pypto-jit`; it currently returns `blocked` because a
controlled Linux/Ascend runner is not configured. This is intentional: an
existing artifact import is never represented as a new compiler invocation.

## 소개

LLVM-FLOW의 API

## 실행 방법

1. Run

```
docker compose up
```

## API 문서

- Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)
- Redoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## 주요 엔드포인트

- POST `/upload/C` : C 파일 업로드 및 분석
- POST `/upload/CPP` : C++ 파일 업로드 및 분석
- POST `/upload/LL` : LL 파일 업로드 및 분석
- GET `/optimization-history`: 파일을 업로드 해서 pass를 적용했던 히스토리
- POST `/show` : 선택한 데이터의 그래프 다시 보기
