# flash_attention_mha — PANKO optimization

## Baseline
- preopt latency: 4188.06 us | P_ref: 800.0 us | J0: 19.1019
- device: 7

## Trajectory
- init: code_version=aedb8d47 anchoring=True seeded_symptoms=[] open=51 held=5 predicates={'has_cube_op': True, 'has_broadcast': False, 'has_chunked_scan': False, 'has_inkernel_layout_op': False, 'has_nested_pypto_loop': True, 'materializes_large_intermediate': True, 'module_count': 1, 'ub_occupancy': 0.6667, 'l1_occupancy': 0.5, 'determined': True} semantic_fold_calls=67
- refine[u1] cand 5f9a0961: s=0 p=0.0 J=0.0 -> reverted (n=1)
- block[u1]: 8 trials (0 feasible), best=None stopped_early=False device_trials=2 charged=2 env_faults=0 failures=2 replayed=0 free=static:6 ceiling=None live_tiles=5 | reason: F-1 Q_TILE 128->256 + cube M 128->256 moved view extents [128,1]->[256,1]; retile cube/vec pair on new structure (bounded retry)
- refine[u1] cand 7b9541a1: s=0 p=0.0 J=0.0 -> reverted (n=2)
- refine[u1] cand 7b9541a1: DUPLICATE rejected (already measured; no budget charged, n unchanged at 2) | duplicates so far: 1
- refine[u1] cand a2026090: s=1 p=3785.2 J=21.1349 -> kept (n=0)
- block[u1]: 12 trials (0 feasible), best=None stopped_early=False device_trials=2 charged=2 env_faults=0 failures=2 replayed=0 free=static:10 ceiling=None live_tiles=5 | reason: F-1 Q_TILE 256->384 moved view extents [256,1]->[384,1]; retile cube/vec pair on new structure
- refine[u1] cand f245e112: s=1 p=4091.3 J=19.5537 -> reverted (n=1)
- refine[u1] cand 6cc25fc2: s=0 p=0.0 J=0.0 -> reverted (n=2)
- refine[u1] cand 72e3bbe3: s=0 p=0.0 J=0.0 -> reverted (n=3)
- refine[u1] cand 5b354d6c: s=1 p=3696.14 J=21.6442 -> kept (n=0)
- refine[u1] cand 46e4048b: s=0 p=0.0 J=0.0 -> reverted (n=1)
- refine[u1] cand 0d695c8f: s=1 p=3541.72 J=22.5879 -> kept (n=0)
- refine[u1] cand faa28a10: s=0 p=0.0 J=0.0 -> reverted (n=1)
- refine[u1] cand 03a24f9a: s=1 p=3573.38 J=22.3878 -> reverted (n=2)
- refine[u1] cand 15ece419: s=1 p=3489.94 J=22.923 -> kept (n=0)
- refine[u1] cand 43fc8643: s=0 p=0.0 J=0.0 -> reverted (n=1)
- refine[u1] cand 6cbf875e: s=1 p=3563.4 J=22.4505 -> reverted (n=2)
- refine[u1] cand 43fc8643: DUPLICATE rejected (already measured; no budget charged, n unchanged at 2) | duplicates so far: 2
- refine[u1] cand 43fc8643: DUPLICATE rejected (already measured; no budget charged, n unchanged at 2) | duplicates so far: 3
- refine[u1] cand 66280007: s=0 p=0.0 J=0.0 -> reverted (n=3)
- refine[u1] cand 1672d169: s=1 p=3794.64 J=21.0824 -> reverted (n=4)
- refine[u1] cand 43fc8643: DUPLICATE rejected (already measured; no budget charged, n unchanged at 4) | duplicates so far: 4
- refine[u1] cand 2bdbfa77: s=1 p=3528.74 J=22.671 -> reverted (n=5)
- refine[u1] cand 1d5e3e1e: s=1 p=3540.62 J=22.5949 -> reverted (n=6)
- refine[u1] cand a7c702f9: s=0 p=0.0 J=0.0 -> reverted (n=7)
- close[u1] -> x57: best J=22.923 p=3489.94 (global best_J=22.923) | stagnation 0 (limit off) | symptoms -
- evolve: +0 insert, 2 update, 0 prune (0 permanent) | reasons: F-1 (u1) closed 1.20x: Q_TILE 128->256 + cube M tile 128->256, C2 N tile [160,320]->[128,128] aligned to head_dim=128. Larger task granularity wins; both attached retune blocks (Q_TILE 256 and 384 structures) found 0 feasible tile configs, so the tile space is near-exhausted for this structure.; Cube L0A budget 64KB pins kL0<=128 at mL0=256: candidate C2 K-tile [128,256]->[160,320] overflowed (256*160*2=81920B>65536B). The K (reduction, K_TILE=320) tile axis is exhausted; M is UB-bound at Q_TILE=256 and N equals head_dim=128, so no further cube-tile gain remains.; Post-F-1 symptom: AIC util 92.14% vs AIV 82.51%, pred_stall 12.08%. The serial online-softmax recurrence (cube C1 -> vec softmax -> cube C2 -> vec update) now dominates and the vector pipe has slack. Raise V on F-23 (interleave outer-loop iterations into concurrent chains, the documented flash_attention_mha lever) and S-15 (pipeline/double-buffer so MTE/cube/vec overlap) which target exactly this.
- refine[u2] cand 58ba4c5d: s=0 p=0.0 J=0.0 -> reverted (n=1)
- refine[u2] cand 58ba4c5d: DUPLICATE rejected (already measured; no budget charged, n unchanged at 1) | duplicates so far: 5
- refine[u2] cand 5129a325: s=0 p=0.0 J=0.0 -> reverted (n=2)
- refine[u2] cand f85fbd7e: s=0 p=0.0 J=0.0 -> reverted (n=3)
- refine[u2] cand 5129a325: DUPLICATE rejected (already measured; no budget charged, n unchanged at 3) | duplicates so far: 6
