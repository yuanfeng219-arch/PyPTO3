# Golden NPU Performance Reference

- **Device ID**: 1
- **Iterations**: 1
- **Warmup**: 3
- **Repeats**: 3
- **Seed**: 42

## Performance Summary

| case | Input Shape | dtype | Median E2E (us) |
|------|-------------|-------|-----------------|
| p0_batch1_n128 | [1,128] | fp16 | 11.865000 |
| p0_batch4_n2048 | [4,2048] | fp16 | 18.550000 |
| p0_batch32_n4096 | [32,4096] | fp16 | 26.562000 |

## Raw Repeats

- `p0_batch1_n128`: [12.144000, 11.865000, 11.838000] us
- `p0_batch4_n2048`: [18.703000, 18.459000, 18.550000] us
- `p0_batch32_n4096`: [26.562000, 27.248000, 25.848000] us

All positive NPU kernel durations emitted by the Golden callable are included.
Input preparation and warmup are outside each formal profiler region.
