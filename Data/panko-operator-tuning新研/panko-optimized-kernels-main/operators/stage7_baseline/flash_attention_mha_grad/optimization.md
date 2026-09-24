- normalize: ok=False tunable 5 -> 5 | tunable tile sites did not increase (5 -> 5); the lever can see no more than it could before
# flash_attention_mha_grad — PANKO optimization

## Baseline
- preopt latency: 4188.9 us | P_ref: 800.0 us | J0: 19.0981
- device: 4

## Trajectory
- init: code_version=aedb8d4 anchoring=True seeded_symptoms=['occupancy', 'l1_occupancy', 'bubble'] open=53 held=3 predicates={'has_cube_op': True, 'has_broadcast': False, 'has_chunked_scan': False, 'has_inkernel_layout_op': True, 'has_nested_pypto_loop': True, 'materializes_large_intermediate': True, 'module_count': None, 'ub_occupancy': 0.3333, 'l1_occupancy': 0.3333, 'determined': True} semantic_fold_calls=67
- refine[u2] cand 99b5b8f3: s=0 p=0.0 J=0.0 -> reverted (n=1)
- refine[u2] cand 03b44de2: s=0 p=0.0 J=0.0 -> reverted (n=2)
- refine[u2] cand db2ab945: s=0 p=0.0 J=0.0 -> reverted (n=3)
- block[u2]: 28 trials (0 feasible), best=None stopped_early=False device_trials=1 charged=1 env_faults=4 failures=1 replayed=0 free=static:23 ceiling=None live_tiles=4 | ABORTED: 3 consecutive environment faults (device) | reason: S2_TILE 512->768 moved KV tile extents [512,128]->[768,128]; retune cube/vec tiles for the new matmul space
- device probe after env-fault abort: usable=True -> run continues
- block[u2]: 179 trials (2 feasible), best=4188.9 stopped_early=False device_trials=4 charged=4 env_faults=7 failures=2 replayed=0 free=static:168 ceiling=None live_tiles=4 | ABORTED: 3 consecutive environment faults (device) | reason: S1_TILE 512->1024 moved Q-side view extents; retune vec/cube tile for new structure
- device probe after env-fault abort: usable=True -> run continues
- block[u2]: 145 trials (2 feasible), best=4188.9 stopped_early=False device_trials=1 charged=1 env_faults=3 failures=1 replayed=2 free=static:139 ceiling=None live_tiles=4 | ABORTED: 3 consecutive environment faults (device) | reason: resume retune: S2_TILE 512->768 moved KV tile extents to [768,128]; prior block aborted on device faults
- device probe after env-fault abort: usable=True -> run continues
- device probe after env-fault abort: usable=True -> run continues
- refine[u2] cand a3b418e5: s=0 p=0.0 J=0.0 -> reverted (n=3)
- block[u2]: 147 trials (2 feasible), best=4292.5 stopped_early=False device_trials=1 charged=0 env_faults=4 failures=1 replayed=2 free=static:140 ceiling=None live_tiles=4 | ABORTED: 3 consecutive environment faults (device) | reason: S2_TILE 512->768 moved KV tile extents [512,128]->[768,128]; retune cube/vec tiles for new matmul space
- refine[u2] cand e13ef14c: s=1 p=4292.5 J=18.6372 -> reverted (n=4)
- device probe after env-fault abort: usable=True -> run continues
- refine[u2] cand 6b30e12a: s=0 p=0.0 J=0.0 -> reverted (n=5)
- refine[u2] cand 72154673: s=0 p=0.0 J=0.0 -> reverted (n=6)
- refine[u2] cand eaef9fca: s=0 p=0.0 J=0.0 -> reverted (n=7)
- close[u2] -> (no candidate beat the best) | stagnation 1 (limit off) | retired against the current structure (revives if it changes)
- evolve: +0 insert, 1 update, 0 prune (0 permanent) | reasons: F-2 (u2) granularity axis exhausted: S1_TILE/S2_TILE 512->768/1024 broke precision (s=0); unroll_list=[64,...] OOMd at compile (192 GiB workspace request); one feasible larger-tile config measured 4292.5us vs 4188.9us incumbent -- per-loop compute granularity is not this kernel bottleneck; Profile: aic_util 89.55% (cube-bound), aiv_util 53.1% (vector half-idle), pred_stall 36.09% (serial cube->vec->cube dependency) -- overlap/fusion levers (S-15/S-16, F-21, I-8/I-12) are the untried high-value axes; u3 (F-3, reduce loop count via larger tile) is the same tile-increase axis u2 just disproved at 768/1024 -> lower V to 0.6, keep alive
- device probe after env-fault abort: usable=True -> run continues
- refine[u1] cand a07766a1: s=1 p=4219.22 J=18.9609 -> reverted (n=1)
- refine[u1] cand 62e479cb: s=1 p=4216.5 J=18.9731 -> reverted (n=2)
- refine[u1] cand a5e1b645: s=1 p=4157.54 J=19.2421 -> kept (n=0)
- refine[u1] cand 69313edf: s=1 p=4437.14 J=18.0296 -> reverted (n=1)
- refine[u1] cand c989d46b: s=1 p=4175.84 J=19.1578 -> reverted (n=2)
- refine[u1] cand d03b46d0: s=1 p=4144.5 J=19.3027 -> kept (n=0)
- refine[u1] cand 30ee9dff: s=1 p=4343.32 J=18.4191 -> reverted (n=1)
- refine[u1] cand 05335aa0: s=1 p=4332.18 J=18.4665 -> reverted (n=2)
- refine[u1] cand 09017593: s=1 p=4338.82 J=18.4382 -> reverted (n=3)
