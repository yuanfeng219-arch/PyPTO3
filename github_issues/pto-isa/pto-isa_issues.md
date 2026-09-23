# Issues for hw-native-sys/pto-isa

Downloaded: 2026-08-17T19:39:43.5522024+08:00
Total issues: 76

## #6 Stride Symbol Ambiguity Causes Compilation Failure on A5 Platform (CANN 9.0.0)

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/6
- Created: 2026-03-09T11:45:50Z
- Updated: 2026-03-09T12:46:39Z
- Closed: 2026-03-09T12:46:39Z

### Body

## Stride Symbol Ambiguity Causes Compilation Failure on A5 Platform (CANN 9.0.0)

### Description

Compiling PTO kernels on the A5 platform with the CANN 9.0.0 bisheng compiler fails due to a `Stride` name conflict between PTO and a newly introduced CANN built-in type.

### Error
```bash
error: reference to 'Stride' is ambiguous
using DynStridDim5 = Stride<1, 1, 1, kTCols_, 1>;
^
/usr/local/Ascend/cann-9.0.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_vector_intrinsics.h:114:12: note: candidate found by name lookup is 'Stride'
enum class Stride {
^
/path/to/pto-isa/include/pto/common/pto_tile.hpp:138:8: note: candidate found by name lookup is 'pto::Stride'
struct Stride {
^
```


### Root Cause

CANN 9.0.0 introduced `enum class Stride` in global scope via `__clang_cce_vector_intrinsics.h`. This conflicts with `pto::Stride` (`pto/common/pto_tile.hpp:138`) during unqualified name lookup, even when `using namespace pto;` is in effect.

### Impact

| | Affected |
|---|---|
| **Platform** | A5 (DAV-3510) with bisheng ccec compiler |
| **Not affected** | A2A3 (older ccec, no conflicting built-in) |
| **Scope** | All PTO kernels using `Stride<...>` unqualified |

### Current Workaround

Manually qualify every `Stride` usage with `pto::`:

```cpp
// Before
using DynStridDim5 = Stride<1, 1, 1, kTCols_, 1>;

// After
using DynStridDim5 = pto::Stride<1, 1, 1, kTCols_, 1>;
```
This is not a sustainable solution — it requires touching all existing kernel code and must be repeated for every future CANN compiler symbol addition.

### Environment

| Field | Value |
|---|---|
| CANN version | 9.0.0 |
| Compiler | bisheng ccec (clang 15.0.5) |
| Compiler path | `/usr/local/Ascend/cann-9.0.0/tools/bisheng_compiler/bin/ccec` |
| Conflicting header | `__clang_cce_vector_intrinsics.h:114` |
| Platform | A5 (DAV-3510) |

### Steps to Reproduce

1. Target the A5 platform with `--cce-aicore-arch=dav-c310-vec`
2. Compile a PTO kernel that uses `using namespace pto;` and references `Stride<...>` without explicit namespace qualification
3. Observe: `error: reference to 'Stride' is ambiguous`

> **Note:** Using `pto::Stride<...>` explicitly avoids the error. The conflict only surfaces with unqualified lookup via `using namespace pto;`.


---

## #7 TASSIGN No-Op Under CPU Simulation Breaks Buffer Sharing, Causing `paged_attention` Accuracy Failure

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/7
- Created: 2026-03-11T01:46:55Z
- Updated: 2026-03-17T00:57:22Z
- Closed: 2026-03-17T00:57:22Z

### Body

# TASSIGN No-Op Under CPU Simulation Breaks Buffer Sharing, Causing `paged_attention` Accuracy Failure

## Problem

`TASSIGN` is a no-op under CPU simulation (`-p a2a3sim`), so buffer sharing
across tiles does not actually occur. When multiple tiles are assigned to the
same UB address via `TASSIGN`, each tile retains its own independent `data_`
array in simulation instead of pointing to shared storage. Consequently, writes
performed through one tile (e.g., `TFILLPAD_INPLACE` padding via `sijPadTile`)
are invisible to reads through another tile (e.g., subsequent arithmetic via
`sijTile`), even though both tiles were assigned to UB offset `0x0`.

The root cause is the CPU-side data model: under `__CPU_SIM`, `Tile::data_` is
a C array (`DType[Rows * Cols]`) that cannot be reassigned, so `TASSIGN`
degrades to a no-op (see `pto-isa/include/pto/cpu/TAssign.hpp`). On hardware,
`data_` is a pointer redirected by `assignData()` to the specified UB address.

## Impact

Any kernel that relies on `TASSIGN` buffer sharing — assigning multiple tiles
to the same UB address so that writes through one tile are visible through
another — will produce incorrect results under CPU simulation while working
correctly on hardware.

The immediate impact is on the `paged_attention` example
(`examples/tensormap_and_ringbuffer/paged_attention`), where the softmax
preparation kernel uses this pattern for partial-block masking. In this kernel,
three tiles share UB address `0x0`:

```cpp
TASSIGN(sijTile,    0x0);   // data tile: loaded from GM, consumed by arithmetic
TASSIGN(sijDynTile, 0x0);   // dynamic-cols boundary marker
TASSIGN(sijPadTile, 0x0);   // padded tile: TFILLPAD_INPLACE fills [valid_len, N) with -inf
```

On hardware, `TFILLPAD_INPLACE(sijPadTile, sijDynTile)` pads the shared buffer
in-place, and subsequent operations on `sijTile` see the padded values. Under
simulation, the padding is written to `sijPadTile.data_` (a separate buffer),
so `sijTile` retains the original unpadded data. This causes:

- `TROWMAX` computes an incorrect row maximum (includes garbage values instead
  of `-inf`)
- `TEXP` produces non-zero weights for invalid positions (should be
  `exp(-inf) = 0`)
- Attention weights are incorrectly distributed, corrupting the final output

The error propagates through the entire online softmax accumulation pipeline,
affecting 190 out of 256 output elements.

## Steps to Reproduce

**Environment**: Linux with `g++-15`, simulation platform (`a2a3sim`)

**Reproduction PR**: https://github.com/ChaoWao/simpler/pull/247

Run the `paged_attention` example under simulation:

```bash
python examples/scripts/run_example.py \
    -k examples/tensormap_and_ringbuffer/paged_attention/kernels \
    -g examples/tensormap_and_ringbuffer/paged_attention/golden.py \
    -p a2a3sim
```

The same command with `-p a2a3` (on-device) passes.

## Error Output

```
[INFO] Comparing out: shape=torch.Size([256]), dtype=torch.float32
[ERROR] TEST FAILED: Output 'out' does not match golden.
Mismatched elements: 190/256
rtol=0.01, atol=0.01
```

## Expected Behavior

Simulation (`-p a2a3sim`) should produce results matching the golden reference
within the specified tolerance (`rtol=0.01, atol=0.01`), consistent with the
on-device (`-p a2a3`) result.

## Root Cause Analysis

The failure originates in `aiv_softmax_prepare.cpp` (the softmax preparation
kernel) and is caused by the `TASSIGN` simulation limitation described above.
Below is the detailed data-flow analysis:

### On Hardware (`-p a2a3`) — Correct Behavior

1. `TASSIGN(sijTile, 0x0)`, `TASSIGN(sijDynTile, 0x0)`,
   `TASSIGN(sijPadTile, 0x0)` — all three tiles point to the same physical UB
   buffer at offset `0x0`.
2. `TLOAD(sijTile, sijGlobal)` — loads `sij` data (including garbage in
   `[valid_len, N)` columns for partial blocks) into UB `0x0`.
3. `TFILLPAD_INPLACE(sijPadTile, sijDynTile)` — pads columns `[valid_len, N)`
   with `-inf` **in the shared UB buffer**.
4. `TMULS(sijTile, sijTile, scale)` — operates on the padded data (garbage
   columns are now `-inf`).
5. `TROWMAX` → `TEXP` → `TROWSUM` — softmax correctly produces zero weight for
   invalid positions (`exp(-inf) = 0`).

### Under Simulation (`-p a2a3sim`) — Incorrect Behavior

1. `TASSIGN` is a **no-op** — each tile retains its own independent `data_`
   array.
2. `TLOAD(sijTile, sijGlobal)` — loads data into `sijTile.data_`.
3. `TFILLPAD_INPLACE(sijPadTile, sijDynTile)` — writes `-inf` into
   `sijPadTile.data_` (a **separate buffer**, not `sijTile.data_`).
4. `TMULS(sijTile, sijTile, scale)` — operates on the **unpadded**
   `sijTile.data_`; garbage values in `[valid_len, N)` are not masked.
5. `TROWMAX` picks up garbage → `TEXP(garbage) ≠ 0` → `TROWSUM` is wrong →
   attention weights are corrupted → output diverges from golden.


---

## #9 A2/A3 TSCATTER behavior appears flattened while docs describe row-index scatter

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/9
- Created: 2026-03-12T15:10:49Z
- Updated: 2026-03-23T11:50:17Z
- Closed: 2026-03-23T11:50:17Z

### Body

## Summary
During PTO-DSL bring-up for `moe_distribute_combine`, the current A2/A3 `TSCATTER` behavior appears to scatter against flattened destination indices, while `docs/isa/TSCATTER.md` currently describes row-index semantics (`dst[idx[i,j], j] = src[i,j]`).

## Repro
Minimal local smoke on 910B / CANN 9.0.0-beta.1:
- destination tile shape: `2 x 16` fp16
- source tile shape: `1 x 16` fp16
- index tile shape: `1 x 16` int16
- all index values set to `1`

Expected from the current doc text:
- entire destination row `1` should receive the source row

Observed:
- only flattened destination position `1` is updated (last-writer-wins behavior), not the full destination row.

## Why this matters
PTODSL kernel design depends on the semantic contract here. We initially modeled `moe_distribute_combine` with row-local scatter indices and got incorrect results. Switching to chunk-local flattened indices made the local PTO smoke correctness-green.

## Request
Please clarify one of these:
1. The implementation is correct and the docs need to describe flattened destination indexing.
2. The docs are correct and the A2/A3 implementation needs to be fixed.

Either way, the semantic contract should be unambiguous because PTOAS and PTODSL frontends need to generate the right index shapes.


---

## #13 A2/A3 lacks native vec quantized-store path for rope-cache kernels

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/13
- Created: 2026-03-13T08:18:10Z
- Updated: 2026-03-23T11:50:41Z
- Closed: 2026-03-23T11:50:19Z

### Body

## Summary
A2/A3 currently has no native vec-tile quantized store path exposed through PTO-ISA for the rope cache style contract (`vec f16 -> GM i8`).

## Evidence
- `TSTORE_IMPL(GlobalData&, TileData&)` for `TileType::Vec` in `include/pto/npu/a2a3/TStore.hpp` hard-checks same source/destination element size and uses the plain vec `TStore` path.
- The quantized overloads in the same file are ACC-only:
  - `TSTORE_IMPL(dst, src, uint64_t preQuantScalar)`
  - `TSTORE_IMPL(dst, src, FpTileData &fp)`
- The current static assertion hit from the rope-cache rewrite is:
  - `Source dtype must be same with dst dtype!`

## Impact
PTODSL rope cache kernels cannot use a fused store-side quantization op for vec tiles. The legal implementation is currently:
1. `TCVT(f16 -> i8)` in vec
2. `TSTORE(i8 -> i8 GM)`

## Request
Either:
- confirm that A2/A3 hardware does not support a vec quantized-store contract and keep `tcvt + tstore` as the intended implementation, or
- add a native vec quantized-store backend contract with matching PTO-ISA intrinsic coverage and legality rules.

## Affected kernels
- `rope_quant_kvcache`
- `dequant_rope_quant_kvcache`
- `qkv_rms_norm_rope_cache`


---

## #14 cpu_sim: TRESHAPE copies data instead of aliasing, causing stale reads after source mutation

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/14
- Created: 2026-03-20T09:09:26Z
- Updated: 2026-03-23T11:50:19Z
- Closed: 2026-03-23T11:50:19Z

### Body

## Summary

`TRESHAPE_IMPL` in `include/pto/cpu/TReshape.hpp` performs a **byte-by-byte copy** from source to destination tile. On real hardware, reshape is a zero-cost reinterpretation of the same physical memory (alias). The sim copy semantics diverge: when the source tile is later mutated (e.g. by `TROWSUM`), the reshaped destination tile still holds **stale** pre-mutation data.

## Root Cause

```cpp
// TReshape.hpp:51-56
const std::byte *src_bytes = reinterpret_cast<const std::byte *>(src.data());
std::byte *dst_bytes = reinterpret_cast<std::byte *>(dst.data());
for (size_t i = 0; i < N; ++i) {
    dst_bytes[i] = src_bytes[i];   // ← copy, not alias
}
```

After the TASSIGN aliasing work in 978846f7, `TASSIGN` correctly points `data_` to shared NPU memory via `NPUMemoryModel::GetPointer`. But `TRESHAPE` copies into `dst`'s own buffer (either `internalStorage_` or a separately-assigned region), breaking the alias chain.

## Reproduction

LayerNorm kernel — two `TRESHAPE` calls on the same source tile, separated by a mutation:

```cpp
TROWSUM(v32, v28, v31);      // 1st write to v32: sum(x)
TRESHAPE(v47, v32);           // v47 = copy of v32 → OK for now
TMULS(v33, v47, v13);         // mean = sum(x)/N  ✓

// ... compute centred² ...

TROWSUM(v32, v34, v35);      // 2nd write to v32: sum(centred²)
                               // v47 still holds stale sum(x) — NOT updated
TMULS(v33, v47, v13);         // WRONG: reads sum(x)/N instead of sum(centred²)/N
```

Same pattern with `v48`/`v33` after `TRSQRT`.

### Test Results

| Platform | Without workaround TRESHAPE | With redundant TRESHAPE after each mutation |
|---|---|---|
| **a2a3** (hardware) | PASS ✓ | PASS ✓ |
| **a2a3sim** (cpu_sim) | **FAIL** (123772/131072 mismatch) | PASS ✓ |

Tested on latest `main` (66228909).

## Expected Behavior

`TRESHAPE(dst, src)` should make `dst` a **view/alias** of `src`'s underlying memory, consistent with hardware semantics. Subsequent writes to `src` should be visible through `dst`.

## Suggested Fix

In `__CPU_SIM` mode, `TRESHAPE_IMPL` should redirect `dst.data()` to point at `src.data()` (reinterpret-cast to the destination type) instead of copying, similar to how `TASSIGN_IMPL` works:

```cpp
#ifdef __CPU_SIM
    // Alias: dst points to src's memory (zero-cost reshape, matching hardware)
    dst.data() = reinterpret_cast<typename TileDataOut::DType *>(
        const_cast<std::byte *>(reinterpret_cast<const std::byte *>(src.data())));
#else
    // Hardware: no-op
#endif
```

The existing copy path can be retained as a fallback for non-sim builds.

---

## #15 Support TPUSH/TPOP instructions in CPU simulation (a5sim)

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/15
- Created: 2026-03-20T09:54:44Z
- Updated: 2026-05-28T03:11:26Z
- Closed: 2026-03-23T11:50:18Z

### Body

### Description

Currently, when running simulations with a5sim, only `TLOAD` and `TSTORE` instructions can be used to ensure correctness. The `TPUSH` and `TPOP` instructions are not yet implemented in the CPU simulation model, which limits the ability to test and verify code that relies on these operations.

### Expected Behavior

The CPU simulation (a5im) should fully support `TPUSH` and `TPOP` instructions, allowing them to be used interchangeably with `TLOAD`/`TSTORE` for buffer-ring-based memory operations.

### Impact

Without this support, developers working with the PTO ISA cannot properly simulate code that uses buffer-ring-based push/pop semantics. This creates a gap between the intended ISA behavior and the simulation environment, potentially leading to verification issues or workarounds that don't reflect actual hardware usage.

### Additional Context

- The missing instructions affect the ability to run certain test cases or benchmarks that rely on `TPUSH`/`TPOP`.

- Implementing these would bring the simulator closer to full ISA compliance and improve the development experience.

---

## #24 CPU simulation (a5sim) missing TPipe/TPUSH/TPOP/TFREE support — API mismatch with hardware backend

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/24
- Created: 2026-03-27T03:11:54Z
- Updated: 2026-05-28T03:10:47Z
- Closed: 2026-03-27T08:29:32Z

### Body

## Summary

A working BGEMM example using the TPUSH/TPOP pipe mechanism runs correctly on A5 hardware, but **cannot be simulated on a5sim** (CPU-based simulation on an a2a3 host) because the CPU simulation backend in pto-isa exposes a completely different API from the hardware backend.

**Working example**: The [`a5` branch of lwDavid/simpler](https://github.com/lwDavid/simpler/tree/a5/examples/a5/tensormap_and_ringbuffer/bgemm) contains a BGEMM kernel that uses `TPipe`, `TPUSH`, `TPOP`, `TFREE`, and `get_subblockid()`. It compiles and runs correctly on A5 hardware via CCEC, but fails to compile under `__CPU_SIM`.

## Root Cause: API mismatch between backends

The include chain in `pto_instr_impl.hpp` dispatches by three mutually exclusive guards:

```cpp
#ifdef PTO_NPU_ARCH_A5    →  pto/npu/a5/TPush.hpp     // TPipe, VEC_FIFO support
#ifdef PTO_NPU_ARCH_A2A3  →  pto/npu/a2a3/TPush.hpp   // TPipe, GM_FIFO only
#ifdef __CPU_SIM           →  pto/cpu/TPush.hpp         // TFIFOSync, mutex-based
```

When compiling for a5sim, the compiler defines `__CPU_SIM` but **not** `PTO_NPU_ARCH_A5` (that macro is set by the CCEC hardware compiler only). So the CPU path is taken.

### Hardware path (`PTO_NPU_ARCH_A5`)

`pto/npu/a5/TPush.hpp` provides:

```cpp
TPipe<FlagID, FIFOType, Depth, Period, TileDataProd, TileDataCons>
```

Kernel usage:
```cpp
using PipeT = TPipe<PP_FLAG_ID, FIFOType::VEC_FIFO, PP_FIFO_DEPTH, PP_FIFO_PERIOD,
                    AccTileT, VecFifoTileT>;
TPUSH(tile, pipe);
TPOP(tile, pipe);
TFREE(pipe);
get_subblockid();
```

### CPU simulation path (`__CPU_SIM`)

`pto/cpu/TPush.hpp` provides a **completely different** type:

```cpp
TFIFOSync<FlagID, DataFIFO, ProducerOp, ConsumerOp>
```

- Different name (`TFIFOSync` vs `TPipe`)
- Different template parameters
- Different semantics
- No `TPipe` alias
- No `TPUSH`/`TPOP`/`TFREE` macros that work with it
- No `get_subblockid()` stub

## Suggested Fix

A unified abstraction layer that maps the same API (`TPipe`, `TPUSH`, `TPOP`, `TFREE`) to both hardware intrinsics and CPU simulation primitives. Specifically:

1. **Add a `TPipe` template alias** in `pto/cpu/TPush.hpp` that wraps `TFIFOSync` with matching template parameters
2. **Add `TPUSH`/`TPOP`/`TFREE` implementations** for the CPU backend that work with `VEC_FIFO` type (currently only `GM_FIFO` has a CPU `TPUSH_IMPL`)
3. **Add a `get_subblockid()` stub** returning `0` for single-threaded simulation

This would allow A5 kernels using the pipe mechanism to compile and run under CPU simulation without any source-level `#ifdef` workarounds.

## Current Workaround

In userland code, we use `#ifdef __CPU_SIM` to fall back to GM-based separate tasks — functionally correct, but without the pipe optimization, and requiring duplicated code paths.

---

## #26 a5sim: TPUSH/TPOP simulation produces incorrect results for BGEMM (precision errors)

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/26
- Created: 2026-03-30T02:06:58Z
- Updated: 2026-05-28T03:10:15Z
- Closed: 2026-03-30T04:15:46Z

### Body

## Summary

After PR #25 resolved the API mismatch for `TPipe`/`TPUSH`/`TPOP`/`TFREE` in CPU simulation (closing #24), a5sim can now **compile and run** kernels that use these instructions. However, the BGEMM example produces **incorrect computation results** (precision errors) under a5sim, while the same kernel computes correctly on real A5 hardware.

## Reproduction

```bash
git clone git@github.com:hw-native-sys/simpler.git
cd simpler
git checkout main
python examples/scripts/run_example.py \
    -k tests/st/a5/tensormap_and_ringbuffer/bgemm/kernels \
    -g tests/st/a5/tensormap_and_ringbuffer/bgemm/golden.py \
    -p a5sim
```

The test will complete but report precision/correctness errors when comparing the a5sim output against the golden reference.

## Expected Behavior

a5sim results should match the golden reference (and match real hardware results), within acceptable tolerance.

## Actual Behavior

a5sim produces results with precision errors. The same BGEMM kernel runs correctly on actual A5 hardware with no precision issues.

## Context

- Issue #24 reported that TPUSH/TPOP were unsupported in a5sim (API mismatch)
- PR #25 added CPU simulation support for the A5-style pipe API
- The API now works (compilation and execution succeed), but the underlying simulation logic appears to produce numerically incorrect results for the pipe-based data movement in BGEMM

---

## #27 Bug: TCOLARGMIN_IMPL / TCOLARGMAX_IMPL duplicate definitions cause ODR violation (a2a3)

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/27
- Created: 2026-03-30T02:36:56Z
- Updated: 2026-03-31T03:31:14Z
- Closed: 2026-03-31T03:31:14Z

### Body

## Description

`pto_instr_impl.hpp` includes both the standalone headers (`TColArgMin.hpp`, `TColArgMax.hpp`) and the unified header (`TColReduceIdx.hpp`) for the a2a3 target. Both sets define `TCOLARGMIN_IMPL` and `TCOLARGMAX_IMPL` with **different implementations**, causing a C++ One Definition Rule (ODR) violation.

When any kernel `#include`s `pto/pto-inst.hpp`, the compiler sees two definitions of each function template and emits a hard error:

```
error: redefinition of 'TCOLARGMIN_IMPL'
error: redefinition of 'TCOLARGMAX_IMPL'
```

## Reproduction

Include `pto/pto-inst.hpp` (which pulls in `pto/common/pto_instr_impl.hpp`) in any a2a3 kernel and compile with ccec.

## Root Cause

In `include/pto/common/pto_instr_impl.hpp`, the a2a3 section includes:

- **Line 57**: `#include "pto/npu/a2a3/TColArgMax.hpp"` — defines `TCOLARGMAX_IMPL` (line 184)
- **Line 96**: `#include "pto/npu/a2a3/TColArgMin.hpp"` — defines `TCOLARGMIN_IMPL` (line 184)
- **Line 97**: `#include "pto/npu/a2a3/TColReduceIdx.hpp"` — **re-defines** both `TCOLARGMIN_IMPL` (line 240) and `TCOLARGMAX_IMPL` (line 245)

The two implementations are also semantically different:
- `TColArgMin.hpp` / `TColArgMax.hpp`: Perform validity checks and dispatch to dtype-specific routines (`TColArgMin16`, `TColArgMax32`, etc.)
- `TColReduceIdx.hpp`: Dispatches through a unified `TCOLARG_DISPATCH` template with an `IsArgMax` bool parameter — no validity checks


## Environment

- Commit: `64cedf5` (sync: merge cann/master into main, 2026-03-29)
- Target: a2a3
- Compiler: ccec (BiSheng)

---

## #30 CPU sim TMatmul rejects PTOAS-emitted Left tile layout

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/30
- Created: 2026-03-30T11:48:36Z
- Updated: 2026-04-01T03:20:51Z
- Closed: 2026-04-01T03:20:51Z

### Body

## Summary

CPU simulation matmul validation is stricter than the PTOAS-emitted Left tile representation used by current matmul kernels. The CPU TMatmul checker treats !TileLeft::isRowMajor as a required condition even though the generated Left tile remains valid for CPU offset computation.

## Reproduction

1. Export PTO_ISA_ROOT to a pto-isa checkout containing the current CPU sim implementation.
2. In the pypto workspace, run:
   pytest -sv tests/st/runtime/test_matmul.py --save-kernels --dump-passes --platform a2a3sim --forked -k matmulacc_pto_64x64x64
3. Inspect the PTOAS-emitted kernel and note the explicit Left tile form:
   Tile<TileType::Left, ..., BLayout::RowMajor, ..., SLayout::RowMajor, ...>
4. Compare that with the CPU-side TMatmul validation in include/pto/cpu/TMatmul.hpp.

## Expected Behavior

CPU simulation should validate Left tiles based on the constraints that actually affect matmul semantics and offset calculation, so PTOAS-emitted Left tiles that are valid for CPU addressing should be accepted.

## Actual Behavior

The CPU-side CheckMadValid() requires !TileLeft::isRowMajor in addition to TileType::Left and SLayout::RowMajor. That rejects the PTOAS-emitted explicit Left tile form even though CPU GetTileElementOffset() can address it correctly.

## Proposed Fix

Relax the Left tile validation in include/pto/cpu/TMatmul.hpp by removing the !TileLeft::isRowMajor restriction while keeping the remaining role, fractal, dtype, and shape checks intact.

## Environment

- Commit: 7ba5a0ed04b9d35635ac537209c7fe4fc3533d46
- Host platform: Linux (x86_64)
- Related failing system test: pypto tests/st/runtime/test_matmul.py::TestMatmulOperations::test_matmulacc_pto_64x64x64 on a2a3sim

## Notes

The local fix keeps the existing CPU sim address-binding behavior in TAssign.hpp unchanged and only relaxes the overly strict Left tile validation in TMatmul.hpp.

---

## #34 sim: TMATMUL static assertion fails for FP32 matmul on a2a3sim (non-conforming matrix fractal)

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/34
- Created: 2026-04-02T03:16:02Z
- Updated: 2026-04-02T04:02:45Z
- Closed: 2026-04-02T04:02:45Z

### Body

## Summary

Running `examples/beginner/matmul.py` with `-p a2a3sim` fails to compile the kernel with a static assertion in the CPU simulator:

```
/data/pto-isa/include/pto/cpu/TMatmul.hpp:56:59: error: static assertion failed: Non-conforming matrix fractal
    ((TileLeft::Loc == TileType::Left) && (!TileLeft::isRowMajor) && (TileLeft::SFractal == SLayout::RowMajor)) &&
```

## Root Cause

`CheckMadValid` in `include/pto/cpu/TMatmul.hpp:56` requires `!TileLeft::isRowMajor` (i.e. L0A must be stored in column-major block layout). However, pypto's codegen for FP32 matmul emits a `TileLeft` with `BLayout::RowMajor`, which causes `isRowMajor = true` and trips the assertion:

```
note: '!(bool)pto::Tile<pto::TileType::Left, float, 64, 256, pto::BLayout::RowMajor, 64, 256,
      pto::SLayout::RowMajor, 512, pto::PadValue::Null>::isRowMajor' evaluates to false
```

The same code compiles and runs correctly on real a2a3 hardware, so either:
- the sim's layout constraint is stricter than the hardware ISA, or
- `BLayout::RowMajor` on the Left tile should be accepted by the sim.

## Reproduction

```bash
# In pypto-lib
python examples/beginner/matmul.py -p a2a3sim
```

## Environment

- Platform: a2a3sim (CPU simulator)
- Dtype: FP32
- Tile shape: Left[64×256] RowMajor, Right[256×64] ColMajor, Acc[64×64]
- File: `include/pto/cpu/TMatmul.hpp`, line 56

---

## #35 sim: Add BF16 (bfloat16) support to CPU simulator

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/35
- Created: 2026-04-02T03:16:07Z
- Updated: 2026-04-02T12:50:24Z
- Closed: 2026-04-02T12:50:24Z

### Body

## Summary

The CPU simulator (`include/pto/cpu/`) does not support `bfloat16` (BF16). This blocks simulation of any model that uses BF16 tensors, such as Qwen3-32B and DeepSeek V3.

## Root Cause

`CheckMadValid` in `include/pto/cpu/TMatmul.hpp` only whitelists three dtype combinations:

```cpp
static_assert(
    (std::is_same_v<AType, int8_t> && std::is_same_v<BType, int8_t> && std::is_same_v<CType, int32_t>) || // s8
    (std::is_same_v<AType, half>   && std::is_same_v<BType, half>   && std::is_same_v<CType, float>)   || // f16→f32
    (std::is_same_v<AType, float>  && std::is_same_v<BType, float>  && std::is_same_v<CType, float>)      // f32→f32
    , "Not supported data type");
```

`bfloat16` is absent. A search of `include/pto/cpu/` confirms only `TLoad.hpp` mentions BF16; none of the compute kernels (matmul, elementwise, reductions) handle it.

## Expected Behavior

BF16 matmul and elementwise operations should be simulatable on the CPU sim, at minimum as `bf16→fp32` accumulation (matching hardware semantics).

## Impact

- Cannot simulate any BF16-only model kernel on a2a3sim/a5sim
- Blocks developer iteration for production LLM workloads (Qwen3, DeepSeek, etc.)

---

## #37 [Bug] BF16 run_cpu.py with macOS GNU toolchain still links against incompatible GTest ABI

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/37
- Created: 2026-04-02T06:51:36Z
- Updated: 2026-04-02T07:17:27Z
- Closed: 2026-04-02T07:17:27Z

### Body

## Summary

On macOS arm64, `tests/run_cpu.py --enable-bf16 --cxx g++-15 --cc gcc-15` still fails to link CPU ST binaries against GTest because the resulting `gtest` archive resolves to `std::__1` symbols while the test objects use GNU `std::__cxx11` symbols.

## Reproducer

```bash
cd tests
python3 run_cpu.py --testcase tcvt --gtest_filter 'TCVTTest.case10:TCVTTest.case11' \
  --build-dir "$PWD/cpu/st/build-bf16" --enable-bf16 --cxx g++-15 --cc gcc-15 --clean
```

Observed failure on April 2, 2026:

- linker errors against `testing::internal::*`
- mixed `std::__1` and `std::__cxx11` symbol families in the same link

## Notes

PR #36 now prefers installed GTest by default and only falls back to `FetchContent` for an Apple+GNU compatibility path, but that is still insufficient on this host: even the fetched `googletest` build produced `std::__1` symbols and did not link against the BF16 GCC test objects.

## Impact

- BF16 `run_cpu.py` validation remains broken on macOS GNU toolchains
- local BF16 verification on Apple Silicon still requires a workaround or a different toolchain path

## Expected Fix Direction

Investigate how the fetched `googletest` build is picking up `libc++` on macOS GNU builds, and ensure the CPU ST build uses a GTest binary compiled with the same C++ runtime/ABI as the selected BF16 compiler.


---

## #40 docs: track repository-wide Markdown lint backlog

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/40
- Created: 2026-04-02T07:51:03Z
- Updated: 2026-04-14T02:33:12Z
- Closed: 2026-04-14T02:33:12Z

### Body

## Background

After landing unified CI in PR #39 on April 2, 2026, the repository now checks Markdown changes through `pre-commit`, but the existing documentation corpus still has a large historical backlog.

Current baseline from a full run of:

```bash
pre-commit run markdownlint-cli2 --all-files
```

Observed backlog:
- about 6534 Markdown lint errors
- about 388 Markdown files with errors
- most errors are concentrated under `docs/`

Top rules by count:
- `MD007`: 2090
- `MD060`: 1446
- `MD031`: 1246
- `MD012`: 574
- `MD032`: 377
- `MD022`: 228
- `MD040`: 158
- `MD036`: 136

Top hotspots by file count:
- `kernels/manual/a5/engram_simt/README.md`: 334
- `docs/HL_ptoisa_newfeature20260306_TPUSH_TPOP.md`: 131
- `docs/menu_apis.md`: 120
- `docs/coding/version-compatibility_zh.md`: 119
- `kernels/manual/common/flash_atten/README.md`: 110
- `tests/npu/a2a3/src/st/testcase/tfa/TFA_kernel.md`: 110
- `docs/assembly/scalar-arith-ops.md`: 105
- `docs/assembly/scalar-arith-ops_zh.md`: 100

Top hotspots by top-level area:
- `docs/`: 5235
- `kernels/`: 727
- `tests/`: 333
- `demos/`: 73
- `include/`: 44

## Why this issue exists

We intentionally scoped the new CI to changed files so PR validation is immediately usable. This issue tracks the remaining repository-wide Markdown debt so it can be burned down in planned batches instead of blocking unrelated work.

## Proposed cleanup plan

1. Batch 1: `docs/assembly/`, `docs/coding/`, and other highest-volume files under `docs/`
2. Batch 2: `kernels/**` Markdown docs and related READMEs
3. Batch 3: `tests/**`, `demos/**`, and root-level Markdown files
4. Batch 4: tighten config only if needed after backlog is materially reduced

## Suggested acceptance target

- reduce the full-repo `markdownlint-cli2` backlog to zero, or to a consciously documented residual allowlist
- keep CI on changed files throughout the cleanup so new debt does not accumulate
- avoid mixing semantic doc rewrites with mechanical lint cleanup unless necessary

## Notes

This should likely be handled as multiple PRs, each scoped to one directory family or one hotspot cluster, to keep reviews manageable.


---

## #50 [BUG] Regression: a5sim BGEMM segmentation fault after CPU_SIM memory manager refactor in `d940c05b`

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/50
- Created: 2026-04-03T06:39:20Z
- Updated: 2026-04-10T01:51:51Z
- Closed: 2026-04-09T12:56:26Z

### Body


## Summary

We identified a regression in `pto-isa` that causes the BGEMM test in `simpler-a5` to fail on `a5sim` with a segmentation fault.

A `git bisect` on `examples/scripts/_deps/pto-isa` shows:

- First bad commit: `d940c05b` (`Memory manager and tests refactoring`)
- Author: `fzhar <zharinov.fedor@huawei.com>`
- Date: `2026-03-31`
- Merge request: `cann/pto-isa!615`

The last known good commit is:

- `882c4db95570dfeaf04e0ee2c0ab32477ed372fc`

This makes `d940c05b` the first confirmed commit that introduces the regression.

## Impact

- Consumer project: `simpler-a5`
- Affected platform: `a5sim`
- Affected workload: BGEMM using TPUSH/TPOP
- User-visible symptom: simulation terminates with `Segmentation fault (core dumped)`

## Reproduction

In `simpler-a5`, set the `pto-isa` dependency under `examples/scripts/_deps/pto-isa` to commit `d940c05b`, then run:

```bash
python examples/scripts/run_example.py \
  -k tests/st/a5/tensormap_and_ringbuffer/bgemm/kernels/ \
  -g tests/st/a5/tensormap_and_ringbuffer/bgemm/golden.py \
  -p a5sim \
  -c 482131249bd2dfc54f8ccf9949ddbd9ad69f6280 \
  --build
```

With `pto-isa` pinned to `882c4db9`, the same test passes.

## Actual Behavior

The orchestration stage completes and reports successful task submission, but the process crashes when the scheduled AICore work begins to execute in CPU_SIM.

Representative log pattern:

```text
[INFO] aicpu_orchestration_entry: [bgemm_orch] Submitted tasks for 2 batches, 4x4 output tiles, 4 K steps each
[ALWAYS] run: PTO2 total submitted tasks = 128, already executed 0 tasks
[INFO] run: Thread 3: Orchestrator completed (orch_idx=0)
[INFO] run: Thread 3: Completed
[INFO] aicpu_execute: aicpu_execute: Kernel execution completed successfully
Segmentation fault (core dumped)
```

From the consumer perspective, this appears as an `a5sim` crash during BGEMM execution rather than a numerical mismatch.

## Expected Behavior

The BGEMM test should complete successfully on `a5sim`, as it does when `pto-isa` is pinned to `882c4db9`.

## Regression Scope

The failing commit is part of MR `cann/pto-isa!615`, which refactors the NPU memory model and CPU_SIM tile handling, and also adds `TASSIGN` support for CPU_SIM tests.

The main files touched by that change include:

- `include/pto/common/pto_tile.hpp`
- `include/pto/cpu/NPUMemoryModel.hpp`
- `include/pto/cpu/TAssign.hpp`

Based on the bisection result and observed runtime behavior, the regression appears to be associated with the CPU_SIM memory-model / tile-management refactor introduced in this change set.

## Root Cause Analysis

The affected BGEMM path uses TPUSH/TPOP and depends on correct CPU_SIM tile allocation and default tile backing storage semantics.

Before `d940c05b`, CPU_SIM `Tile` objects had internal fallback storage, so a tile declared without an explicit `TASSIGN` still had valid backing memory.

After `d940c05b`, this behavior changed in `include/pto/common/pto_tile.hpp`:

- the CPU_SIM internal fallback storage path was removed
- CPU_SIM tile lazy allocation became conditional on `__PTO_AUTO__`
- when `__CPU_SIM` is defined but `__PTO_AUTO__` is not defined, a tile may keep an uninitialized backing pointer unless it is explicitly `TASSIGN`'d

In our integration, simulation kernels are compiled with `__CPU_SIM`, but not with `__PTO_AUTO__`.

This becomes observable in the BGEMM kernel because it declares a TPUSH/TPOP consumer tile without explicitly assigning storage:

```cpp
VecFifoTileT vecFifoTile;
```

That tile is later used as the destination of split `TPOP`. In the CPU_SIM split path, `TPOP` eventually writes into `dst.data()[...]`. With the old behavior this was safe because `vecFifoTile` had internal backing storage. After `d940c05b`, `dst.data()` can be invalid in the non-`__PTO_AUTO__` CPU_SIM configuration, which leads to the observed segmentation fault.

In other words, this looks like a regression in CPU_SIM `Tile` default-storage semantics rather than a problem in the BGEMM algorithm itself.

## Temporary Consumer Workaround

As a local consumer-side workaround, explicitly assigning storage to the BGEMM FIFO consumer tile appears to avoid the crash, for example by adding `TASSIGN` for `vecFifoTile` before `TPOP`.

However, we do not believe that should be considered the root fix in `pto-isa`, because:

- the previous CPU_SIM behavior allowed such tiles to work without explicit `TASSIGN`
- other kernels may rely on the same implicit-storage behavior
- the regression was introduced by the CPU_SIM memory / tile refactor, not by a change in the BGEMM kernel itself


---

## #51 A2A3 TRSQRT on FP32 vector path shows large accuracy loss compared with sqrt+div

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/51
- Created: 2026-04-08T03:26:20Z
- Updated: 2026-04-14T02:30:19Z
- Closed: 2026-04-14T02:30:19Z

### Body

## Summary

On A2A3, the vector `rsqrt` path appears to have significantly worse accuracy than the equivalent `sqrt + div` path.

A minimal rsqrt-only kernel reproduces the issue. The same setup with `sqrt + div` passes accuracy checks, which suggests the problem is specifically in the lowering/runtime behavior of `TRSQRT`.

## Reproduction

Environment:

- repo used for reproduction: `high-cloud/pto_workspace`
- target platform: `a2a3`
- device: `15`
- conda env: `pypto-lib`

### 1. rsqrt-only test

Script:

- `modules/pypto-lib/examples/models/qwen3/qwen3_32b_rsqrt_only_accuracy.py`

Command:

```bash
source /data/miniconda3/etc/profile.d/conda.sh
conda activate pypto-lib
cd /data/yangyaodong/code/pto_workspace
source scripts/env.sh
cd modules/pypto-lib
python examples/models/qwen3/qwen3_32b_rsqrt_only_accuracy.py -p a2a3 -d 15
```

Result directory:

- `build_output/RsqrtOnlyAccuracyProgram_20260407_202402`

Observed failure:

```text
AssertionError: Output 'inv_rms_out' does not match golden.
Mismatched elements: 16/16
```

Mismatch dump:

- `build_output/RsqrtOnlyAccuracyProgram_20260407_202402/mismatch_dump/inv_rms_out.pt`

Measured from dump:

- mismatch: `16 / 16`
- max abs diff: `0.01098489761352539`
- mean abs diff: `0.004985183477401733`
- max rel diff: `0.0021205416414886713`

Example values:

- actual:
  `[4.59375, 4.0625, 5.125, 3.6796875, 3.7890625, 4.09375, 4.703125, 4.953125, 4.78125, 5.28125, 4.203125, 4.890625, 4.921875, 4.9375, 3.7265625, 5.578125]`
- expected:
  `[4.5922627449035645, 4.066621780395508, 5.127498149871826, 3.674419641494751, 3.7861320972442627, 4.099695682525635, 4.693172931671143, 4.962160587310791, 4.772738933563232, 5.292234897613525, 4.197691440582275, 4.889339923858643, 4.922595024108887, 4.942502021789551, 3.7241029739379883, 5.5822529792785645]`

## Lowering Path

The generated code lowers `pl.rsqrt(...)` directly to `TRSQRT`.

Frontend / pass dump:

- `build_output/RsqrtOnlyAccuracyProgram_20260407_202402/passes_dump/00_frontend.py`
- `build_output/RsqrtOnlyAccuracyProgram_20260407_202402/passes_dump/23_after_AllocateMemoryAddr.py`

PTO IR:

- `build_output/RsqrtOnlyAccuracyProgram_20260407_202402/ptoas/compute_rsqrt_incore_0.pto`

Relevant line:

```mlir
pto.trsqrt ins(%variance_row__tile ...) outs(%inv_rms_row__tile ...)
```

Generated kernel:

- `build_output/RsqrtOnlyAccuracyProgram_20260407_202402/kernels/aiv/compute_rsqrt_incore_0.cpp`

Relevant line:

```cpp
TRSQRT(v14, v13);
```

## Control Experiment

A control kernel that replaces:

```python
inv_rms = pl.rsqrt(variance)
```

with:

```python
rms = pl.sqrt(variance)
one = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=1.0)
inv_rms = pl.div(one, rms)
```

passes accuracy checks under the same setup.

Script:

- `modules/pypto-lib/examples/models/qwen3/qwen3_32b_sqrt_div_accuracy.py`

This suggests the accuracy issue is specific to the `TRSQRT` path, not the surrounding FP32 load/store or tensor reshape path.

## Impact

This affects RMSNorm-related kernels, including Qwen3 scope1 experiments. In larger kernels, the rsqrt error propagates and amplifies downstream projection mismatch.

## Expected

`pl.rsqrt` on FP32 vector input should have accuracy comparable to `1.0 / sqrt(x)` for this use case, or the backend/docs should clearly state that `TRSQRT` is an approximate instruction with materially lower precision.


---

## #52 TPARTADD/MAX/MIN/MUL 的文档定义不准确，请更正

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/52
- Created: 2026-04-08T06:54:49Z
- Updated: 2026-04-14T02:29:09Z
- Closed: 2026-04-14T02:29:09Z

### Body

https://github.com/hw-native-sys/pto-isa/blob/main/docs/isa/TPARTADD_zh.md

<img width="1050" height="1080" alt="Image" src="https://github.com/user-attachments/assets/a41027a2-e633-4b07-8b56-4fe3e8289ebf" />

**如上图所示：**
1. **指令示意图**：src0.shape=(3,3), src1.shape=(2,4), dst.shape=(3,3), 与**②指令示意图左下角的伪码**对不上。另据文档约束，这种src0，src1互相不掩盖的情况，是不支持的，请更正；
2. **指令示意图左下角的伪码**：最后一个else中的：implementation-defined，据了解，当前未实现；请确认：如有实现，请刷新**④数学语义**，否则请删除；
3. **简介**：中文晦涩难懂，请更新。
4. **数学语义**：这里跟**②指令示意图左下角的伪码**不一致，请确认。

另：TPARTMAX/MIN/MUL有相同问题。

---

## #58 [A5][TMOV] Potential UB unaligned access on col_major 16x1 vec->vec path

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/58
- Created: 2026-04-08T11:19:22Z
- Updated: 2026-04-14T01:30:51Z
- Closed: 2026-04-14T01:30:51Z

### Body

## Background

During A5 board validation for PTOAS Sync cases, we observed intermittent but reproducible runtime failures:

- `error code = 340`
- `The instruction access UB address is not aligned`

After excluding sync-order effects (including forcing `PIPE_ALL` barriers), the strongest signal points to A5 `TMOV` Vec->Vec behavior in a specific layout/shape combination.

## Suspected risky scenario

- Target: A5
- Op path: `TMOV` Vec->Vec
- Tile shape/layout: `rows=16, cols=1, blayout=col_major`
- DType: `f32`

In this case, `RowStride=1` and address progression in `TMovVecToVec` can reach `base + 4B` at `i=1`, which may violate vector load alignment requirements (32B), then trigger error 340.

## Relevant code snippets

### A5 TMOV Vec->Vec address formula

```cpp
for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
    sreg = (uint32_t)validCol;
    for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
        preg = CreatePredicate<T>(sreg);
        vlds(vreg0, src, i * SrcTileData::RowStride + j * nRepeatElem, NORM);
        vsts(vreg0, dst, i * DstTileData::RowStride + j * nRepeatElem, distValue, preg);
    }
}
```

Source: `include/pto/npu/a5/TMov.hpp`

### RowStride definition for col-major

```cpp
static constexpr int RowStride = BFractal_ == BLayout::RowMajor ? Cols : 1;
```

Source: `include/pto/common/pto_tile.hpp`

## Repro notes

We prepared an observable PTO case in PTOAS (GM->UB `tload` -> `tmov` -> UB->GM `tstore`) to ensure TMOV path is actually executed:

- [PTOAS PR #440](https://github.com/hw-native-sys/PTOAS/pull/440)
- Related PTOAS tracking issues: [#421](https://github.com/hw-native-sys/PTOAS/issues/421), [#441](https://github.com/hw-native-sys/PTOAS/issues/441)

## What we need help confirming in PTO-ISA

1. Whether A5 `TMOV_V2V` should enforce stronger alignment preconditions for `col_major + small RowStride`.
2. Whether this case should use a guarded fallback/specialized path instead of direct `vlds/vsts` stepping.
3. Whether we should add a dedicated ISA regression case for `16x1 col_major f32` to prevent future regressions.

Thanks a lot. We can provide additional logs / CA-sim artifacts if needed.


---

## #62 demo: baseline auto_mode add test needs op_extension side-effect import

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/62
- Created: 2026-04-09T07:24:19Z
- Updated: 2026-04-09T07:27:09Z
- Closed: 2026-04-09T07:26:20Z

### Body

## Summary

The baseline auto-mode add demo test depends on `import op_extension` for side-effect operator registration. Without that import, `torch.ops.npu.my_add(...)` is not registered and the documented demo flow fails before exercising the custom operator.

## Expected

Running `demos/auto_mode/baseline/add/test/test.py` should register `libop_extension.so` and execute the example successfully in a configured environment.

## Proposed fix

Restore the intentional side-effect import and keep it marked so cleanup tools do not remove it again.

## Validation

- `python3 -m py_compile demos/auto_mode/baseline/add/test/test.py`
- `ruff check demos/auto_mode/baseline/add/test/test.py`

---

## #63 demo: torch-jit add entrypoint requires torch_npu initialization import

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/63
- Created: 2026-04-09T07:24:40Z
- Updated: 2026-04-09T07:27:11Z
- Closed: 2026-04-09T07:26:25Z

### Body

## Summary

The torch-jit add demo entrypoint uses `device="npu"` tensors and `torch.npu.synchronize()`, but that runtime surface is registered by `import torch_npu`. Without the import, the demo fails before kernel execution.

## Expected

Running `demos/auto_mode/torch_jit/add/add_compile_and_run.py` in a configured environment should initialize the NPU runtime and execute the demo successfully.

## Proposed fix

Restore the explicit `import torch_npu` side-effect import and mark it as intentional so cleanup tools do not remove it again.

## Validation

- `python3 -m py_compile demos/auto_mode/torch_jit/add/add_compile_and_run.py`
- `ruff check demos/auto_mode/torch_jit/add/add_compile_and_run.py`

---

## #64 tests: all_cpu_tests should not reuse a shared gen_data.py filename

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/64
- Created: 2026-04-09T07:24:55Z
- Updated: 2026-04-09T07:27:13Z
- Closed: 2026-04-09T07:26:31Z

### Body

## Summary

`tests/script/all_cpu_tests.py` copies testcase generators into the build directory before execution. Reusing a single destination filename (`gen_data.py`) creates an unnecessary collision point and makes the runner fragile to future parallelism or generator assumptions.

## Expected

Each testcase generator should execute from an isolated temporary script path while preserving the existing build-directory output layout expected by the CPU ST binaries.

## Proposed fix

Copy each generator to a unique filename derived from its testcase directory before running it.

## Validation

- `python3 -m py_compile tests/script/all_cpu_tests.py`
- `ruff check tests/script/all_cpu_tests.py`
- `python3 tests/script/all_cpu_tests.py -g Ninja -b build/cpu_pr_all_cpu_tests`

---

## #66 [BUG] pypto fails to generate alloc_tile for tpop_from_aic/tpop_from_aiv return tiles, causing missing TASSIGN and runtime Segfault

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/66
- Created: 2026-04-10T03:57:22Z
- Updated: 2026-04-14T10:10:30Z
- Closed: 2026-04-14T10:10:30Z

### Body

### Summary

During Qwen3Scope3 model compilation, we identified that pypto does not generate `pto.alloc_tile` for the return tiles of `pto.tpop_from_aic` / `pto.tpop_from_aiv` in the .pto IR. As a result, ptoas compiles TPOP without a corresponding `TASSIGN`, and the tile has no valid backing physical memory address at runtime, leading to a Segmentation fault.

This is confirmed to be a **pypto memory allocation pass issue**, not a ptoas compilation issue. ptoas behaves correctly — it faithfully compiles `pto.alloc_tile` into `TASSIGN`, and without `alloc_tile`, no `TASSIGN` is generated.

### Impact

- **Affected model:** Qwen3Scope3
- **Affected component:** pypto memory allocation passes
- **User-visible symptom:** Segmentation fault (core dumped) at runtime

### Reproduction

File: `pypto-lib/examples/models/qwen3/qwen3_32b_decode_scope3.py`

### Actual Behavior

**.pto IR (scope3_incore_1.pto:76-78):**

```mlir
%t__tile_Vec = pto.tpop_from_aic {split = 1} -> !pto.tile_buf<loc=vec, ...>
%0 = pto.alloc_tile addr = %c36864 : !pto.tile_buf<loc=vec, ...>
pto.tadd ins(%o_acc__tile, %t__tile_Vec : ...) outs(%0 : ...)
```

`%t__tile_Vec` is returned directly by `pto.tpop_from_aic` with **no corresponding `pto.alloc_tile`**.

**Generated C++ code (scope3_incore_1.cpp:140-147):**

```cpp
Tile<...> v30;
wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
TPOP<...>(v25, v30);  // v30 has no TASSIGN — segfault!
```

By contrast, all other tile operations (TLOAD, TEXPANDS, TADD, etc.) have a complete `alloc_tile` -> `TASSIGN` chain:

```mlir
%o_acc__tile = pto.alloc_tile addr = %c32768 : !pto.tile_buf<...>
pto.texpands ins(%cst : f32) outs(%o_acc__tile : ...)
```

```cpp
Tile<...> v28;
TASSIGN(v28, v20);  // v20 = 32768, from pto.alloc_tile
TEXPANDS(v28, v12);
```

### Expected Behavior

pypto should generate `pto.alloc_tile addr = X` for the return tiles of `pto.tpop_from_aic` / `pto.tpop_from_aiv`, so that ptoas can correctly emit `TASSIGN` and the tile is bound to a valid physical memory address at runtime.

| Step | Normal tile operations | TPOP (current behavior) |
|------|----------------------|------------------------|
| pypto generates .pto | `pto.alloc_tile addr = X` + operation | Only `pto.tpop_from_aic`, **missing alloc_tile** |
| ptoas compiles to C++ | `TASSIGN(tile, X)` + operation | Only `TPOP(pipe, tile)`, **no TASSIGN** |
| Runtime | Executes normally | **Segfault** (tile has no backing memory) |

### Root Cause Analysis

pypto's memory allocation pipeline (`InitMemRef` -> `MemoryReuse` -> `AllocateMemoryAddr`) generates `pto.alloc_tile addr = X` for most tile operations (TLOAD, TEXPANDS, TADD, etc.), but **misses the return tiles of `pto.tpop_from_aic` / `pto.tpop_from_aiv`**.

Specifically, `MakeTpopFromAicCodegenPTO` (`src/backend/common/pto_ops_common.cpp:1015-1037`) generates `pto.tpop_from_aic`, but its return value is not picked up by the subsequent memory allocation passes, resulting in the missing `pto.alloc_tile` in the final .pto IR.

ptoas is not at fault — it has no responsibility to auto-insert `TASSIGN` for tiles that lack `alloc_tile` in the IR.

---

## #67 A2A3 TPipe template is incompatible with ptoas 0.24 old-style LocalSlotNum emission

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/67
- Created: 2026-04-12T03:41:59Z
- Updated: 2026-04-15T02:27:24Z
- Closed: 2026-04-15T02:27:24Z

### Body

## Summary

`include/pto/npu/a2a3/TPush.hpp` on `main` still declares `TPipe` with the 5th template parameter as `bool IsNoSplit`, which is incompatible with the current `ptoas v0.24` release output when it emits the old-style form:

```cpp
TPipe<FlagID, DirType, SlotSize, SlotNum, LocalSlotNum>
```

This causes valid generated code to fail to compile when the 5th template argument is a value like `8`.

## Current A2A3 signature

From `include/pto/npu/a2a3/TPush.hpp` on `main`:

```cpp
template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum, bool IsNoSplit = false,
          uint32_t LocalSlotNum = 2, bool EN_UNIT_FLAG = false>
struct TPipe
```

## Failing generated form

`ptoas v0.24` may emit instantiations like:

```cpp
TPipe<0, Direction::DIR_C2V, 4096, 8, 8>
```

With the current signature, the `5th` argument is parsed as `bool IsNoSplit`, so compilation fails with a narrowing / invalid non-type template argument error.

## Expected behavior

A2A3 `TPipe` should remain compatible with both forms:

- old style: `TPipe<..., SlotNum, LocalSlotNum>`
- new style: `TPipe<..., SlotNum, IsNoSplit, LocalSlotNum>`

## Suggested fix

Adopt the same compatibility approach used in local downstream workarounds:

- make the 5th template parameter a compatibility slot that can represent either `IsNoSplit` (`0/1`) or `LocalSlotNum` (`>1`)
- derive `IsNoSplit` and the effective local-slot count from that value
- keep behavior unchanged for current call sites while allowing current `ptoas v0.24` generated code to compile

## Notes

`include/pto/cpu/TPush.hpp` already uses a `LocalSlotNum`-style 5th parameter, so the A2A3 variant is the outlier here.

## Impact

This blocks users compiling code generated by the latest `ptoas v0.24` release against current `pto-isa` A2A3 headers, even though the generated `TPipe<..., 8, 8>` form is otherwise semantically valid.

---

## #79 [Bug] A2A3 pto.textract rejects row extraction from loc=vec after LEFT_RIGHT split

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/79
- Created: 2026-04-14T09:13:05Z
- Updated: 2026-05-08T01:48:37Z
- Closed: 2026-05-08T01:48:37Z

### Body

### Diagnosis

**ptoas / pto-isa** — the generated `.pto` reaches ptoas, but ptoas rejects `pto.textract` because the source tile is `loc=vec`. The failure happens during PTO assembly, after PyPTO has generated PTO IR for an A2/A3 target.

### Description

A `LEFT_RIGHT` split pattern that sends a matmul result from AIC to AIV, casts it to BF16, then extracts rows from the cast tile fails during ptoas compilation on A2/A3:

```text
ptoas compilation failed:
error: 'pto.textract' op expects A2/A3 textract src to use loc=mat
```

The problematic shape is:

1. `chunked_loop_optimizer(split=pl.SplitMode.LEFT_RIGHT)`
2. `matmul` + `matmul_acc` produces an FP32 tile on the AIC side
3. the result is transferred to the AIV side as a `loc=vec` tile
4. the AIV side casts the full tile to BF16
5. a row is sliced from that BF16 tile, lowering to `pto.textract` from `loc=vec`
6. ptoas rejects it because A2/A3 `pto.textract` expects a `loc=mat` source

### Minimal Reproducer

```python
import pypto.language as pl

BATCH = 16
HIDDEN = 256
OUT_COLS = 256
K_CHUNK = 128
OUT_CHUNK = 128
HIDDEN_BLOCKS = HIDDEN // K_CHUNK
OUT_BLOCKS = OUT_COLS // OUT_CHUNK


def build_program():
    @pl.program
    class ReproTextractVecSource:
        @pl.function(type=pl.FunctionType.Opaque)
        def repro(
            self,
            x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            w: pl.Tensor[[HIDDEN, OUT_COLS], pl.BF16],
            out: pl.Out[pl.Tensor[[BATCH, OUT_COLS], pl.BF16]],
        ) -> pl.Tensor[[BATCH, OUT_COLS], pl.BF16]:
            with pl.at(
                level=pl.Level.CORE_GROUP,
                optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.LEFT_RIGHT),
            ):
                for ob in pl.parallel(OUT_BLOCKS, chunk=2):
                    col0 = ob * OUT_CHUNK
                    tile_a = pl.slice(x, [BATCH, K_CHUNK], [0, 0])
                    tile_w = pl.slice(w, [K_CHUNK, OUT_CHUNK], [0, col0])
                    acc = pl.matmul(tile_a, tile_w, out_dtype=pl.FP32)
                    for kb in pl.range(1, HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        tile_a_i = pl.slice(x, [BATCH, K_CHUNK], [0, k0])
                        tile_w_i = pl.slice(w, [K_CHUNK, OUT_CHUNK], [k0, col0])
                        acc = pl.matmul_acc(acc, tile_a_i, tile_w_i)

                    acc_bf16 = pl.cast(acc, target_type=pl.BF16)
                    for bi in pl.range(BATCH):
                        row = pl.slice(acc_bf16, [1, OUT_CHUNK], [bi, 0])
                        out = pl.assemble(out, row, [bi, col0])
            return out

    return ReproTextractVecSource


def build_tensor_specs():
    import torch
    from pypto.runtime import TensorSpec

    return [
        TensorSpec("x", [BATCH, HIDDEN], torch.bfloat16, init_value=lambda: torch.randn(BATCH, HIDDEN)),
        TensorSpec("w", [HIDDEN, OUT_COLS], torch.bfloat16, init_value=lambda: torch.randn(HIDDEN, OUT_COLS)),
        TensorSpec("out", [BATCH, OUT_COLS], torch.bfloat16, is_output=True),
    ]


def golden(tensors, params):
    tensors["out"][:] = (tensors["x"].float() @ tensors["w"].float()).bfloat16()


def compile_and_run(platform: str = "a2a3", device_id: int = 0):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    backend = BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B
    return run(
        program=build_program(),
        tensor_specs=build_tensor_specs(),
        golden=golden,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=3e-3,
            atol=3e-3,
            strategy=OptimizationStrategy.Default,
            dump_passes=True,
            backend_type=backend,
            skip_golden=True,
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()

    result = compile_and_run(platform=args.platform, device_id=args.device)
    if not result.passed:
        if result.error:
            print(f"Result: {result.error}")
        raise SystemExit(1)
    print("PASSED")
```

Run:

```bash
python repro_textract_vec_source.py -p a2a3 -d <device_id>
```

### Observed Error

```text
Failed to compile group 'repro_incore_0' [repro_incore_0_aic, repro_incore_0_aiv]:
ptoas compilation failed: loc(".../ptoas/repro_incore_0.pto":93:9): error:
'pto.textract' op expects A2/A3 textract src to use loc=mat
Error: Failed to parse MLIR.
```

### Lowered PTO Around the Failure

```mlir
%acc__rv_v2_Vec = pto.tpop_from_aic {split = 2}
  -> !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, ...>

%acc_bf16__tile = pto.alloc_tile
  : !pto.tile_buf<loc=vec, dtype=bf16, rows=16, cols=64, ...>

pto.tcvt ins(%acc__rv_v2_Vec) outs(%acc_bf16__tile)

pto.textract ins(%acc_bf16__tile, %bi__idx_v0, %c0_index
  : !pto.tile_buf<loc=vec, dtype=bf16, rows=16, cols=64, ...>, index, index)
  outs(%row__tile : !pto.tile_buf<loc=vec, dtype=bf16, rows=1, cols=64, ...>)
```

ptoas rejects the `pto.textract` because the source is `loc=vec`.

### Expected Behavior

One of the following should happen:

1. `pto.textract` should support this A2/A3 row-extraction pattern from `loc=vec`, if that is semantically valid; or
2. the lowering/codegen should insert the required move/conversion so that `pto.textract` receives a `loc=mat` source; or
3. the compiler should reject this earlier with a clearer diagnostic explaining the required source location.

### Actual Behavior

The compiler emits PTO IR containing:

```mlir
pto.textract ins(%acc_bf16__tile ... !pto.tile_buf<loc=vec, ...>)
```

and ptoas fails with:

```text
'pto.textract' op expects A2/A3 textract src to use loc=mat
```

### Environment

| Component | Version |
|---|---|
| pypto | `2e498406` (branch: `main`) |
| simpler | `c83754f` (branch: `stable`) |
| ptoas | `ptoas 0.24` |
| CANN | `[CANN:8.5.0.alpha001]` |

### Host Platform

Linux aarch64

### Related Issues

Related but different: hw-native-sys/pto-isa#66


---

## #82 TPUSH_IMPL: missing pipe_barrier(PIPE_MTE3) between Acc→GM TSTORE and record() causes data race

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/82
- Created: 2026-04-15T07:06:16Z
- Updated: 2026-04-15T08:32:14Z
- Closed: 2026-04-15T08:32:14Z

### Body

## Summary

In the C2V (Cube-to-Vector) path where an `AccTile` is pushed to a GM FIFO, `TPUSH_IMPL` calls `TSTORE_IMPL` (which issues a DMA on **PIPE_MTE3**) followed immediately by `record()` (which signals the consumer on **PIPE_FIX**). Because PIPE_MTE3 and PIPE_FIX are independent pipelines with no ordering guarantee, the cross-core signal can reach the consumer **before** the GM write completes, causing the consumer to read stale or partially-written data.

## Affected Code

### A5 — `TPipe::TPUSH_IMPL` (`include/pto/npu/a5/TPush.hpp` ~L602-619)

```cpp
template <typename Pipe, typename TileProd, TileSplitAxis Split>
PTO_INTERNAL void TPUSH_IMPL(Pipe &pipe, TileProd &tile)
{
    // 1. allocate
    ...
    // 2. push — TSTORE Acc→GM goes through PIPE_MTE3
    pipe.prod.template push<TileProd, Split>(pipe.fifo, tile);
    pipe.prod.tileIndex++;

    // 3. record — signals consumer on PIPE_FIX  ← no barrier before this!
    if (isRecord) {
        pipe.prod.template record<Split>();
    }
}
```

`record()` for C2V uses `set_intra_block(PIPE_FIX, FlagID)`, which is on a **different pipeline** than the preceding TSTORE (PIPE_MTE3).

### A5 — `TMPipe::TPUSH_IMPL` (`include/pto/npu/a5/TPush.hpp` ~L1183-1201)

Same pattern — `push()` then `record()` with no MTE3 barrier in between.

### A2A3 — `TPipe::TPUSH_IMPL` (`include/pto/npu/a2a3/TPush.hpp` ~L413-431)

```cpp
pipe.prod.template push<TileProd, Split>(pipe.fifo, tile);  // TSTORE → PIPE_MTE3
pipe.prod.tileIndex++;
pipe.prod.record();  // ffts_cross_core_sync(PIPE_FIX, ...) ← different pipeline
```

### A2A3 — `TMPipe::TPUSH_IMPL` (`include/pto/npu/a2a3/TPush.hpp` ~L789-805)

Same issue.

## Root Cause

- `TSTORE_IMPL` from AccTile to GM dispatches work on **PIPE_MTE3**.
- `record()` in C2V mode signals on **PIPE_FIX** (A5: `set_intra_block`; A2A3: `ffts_cross_core_sync`).
- These two pipelines have **no implicit ordering** — the signal can be issued and delivered to the consumer before MTE3 finishes the GM write.

## Why V2C Is Unaffected

In V2C mode (Vec→GM), `record()` signals on **PIPE_MTE3** — the same pipeline as `TSTORE`. Same-pipeline instructions are naturally ordered, so no barrier is needed.

## Suggested Fix

Insert `pipe_barrier(PIPE_MTE3)` between `push()` and `record()` when the producer tile is an AccTile on a GM FIFO path. For example:

```cpp
// 2. push
pipe.prod.template push<TileProd, Split>(pipe.fifo, tile);
pipe.prod.tileIndex++;

// 2.5 ensure GM write completes before signaling consumer
if constexpr (TileProd::Loc == TileType::Acc && /* GM FIFO path */) {
    pipe_barrier(PIPE_MTE3);
}

// 3. record
if (isRecord) {
    pipe.prod.template record<Split>();
}
```

This needs to be applied in four locations:
1. `include/pto/npu/a5/TPush.hpp` — `TPipe::TPUSH_IMPL` (~L602)
2. `include/pto/npu/a5/TPush.hpp` — `TMPipe::TPUSH_IMPL` (~L1183)
3. `include/pto/npu/a2a3/TPush.hpp` — `TPipe::TPUSH_IMPL` (~L413)
4. `include/pto/npu/a2a3/TPush.hpp` — `TMPipe::TPUSH_IMPL` (~L789)

## Impact

- **Severity:** High — data race on GM memory; consumer may read incomplete data.
- **Trigger condition:** C2V GM FIFO path with AccTile producer, under timing where MTE3 DMA is slower than PIPE_FIX signal dispatch.
- **Platforms affected:** A2A3 and A5 NPU backends.

---

## #83 Clarify/support true single-Vec TILE_NO_SPLIT semantics on A2A3 mixed C/V kernels

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/83
- Created: 2026-04-15T10:31:25Z
- Updated: 2026-05-28T03:09:49Z
- Closed: 2026-04-29T08:15:17Z

### Body

  ### Summary

  We are trying to use `TILE_NO_SPLIT` for mixed C/V kernels on A2A3 with the same semantics as A5: a logical 1C1V case where only one Vec sub-core actively participates in the `tpush/tpop` path.

  After reading the current implementation, A2A3 does not seem to support that model in the same way as A5.

  On A5, `TILE_NO_SPLIT` appears to be a true single-Vec-subcore path:
  - the sync path is split-aware
  - the second Vec-side sync is suppressed for `TILE_NO_SPLIT`
  - the no-split tests gate execution to `AIV0` only

  On A2A3, however:
  - the producer/consumer sync path still uses `wait_flag_dev(...)` / `ffts_cross_core_sync(... CV_CORES_SYNC ...)` without a `TILE_NO_SPLIT`-specific sync branch
  - the `tpushpop_vc_nosplit` test explicitly states that both `AIV0` and `AIV1` participate in the CV handshake, and that hardware collects both signals before unblocking Cube
  - the `tpushpop_cv_nosplit` test comments describe a single-`AIV0` flow, but the Vec-side code is not gated by `get_subblockid() == 0`, so both Vec sub-cores still appear to execute the path

  From the upstream integration perspective, this is also why PyPTO currently treats `SplitMode.NONE` as unsupported for mixed kernels.

  ### Why this matters

  For a logical 1C1V no-split case on A2A3, the current behavior appears to still require 1C2V-style participation:
  - one Vec sub-core does the real work
  - the other Vec sub-core may still need to participate in the sync protocol, even if it does no useful computation

  Otherwise, the Cube-side wait condition may not be satisfied/released correctly, which looks like a deadlock/hang risk.

  This is different from A5, where `TILE_NO_SPLIT` seems to provide true single-Vec-subcore semantics.

  ### Questions

  1. Is this understanding correct?
  2. On A2A3, is `TILE_NO_SPLIT` intentionally only a "no data split" mode, while the CV sync protocol still fundamentally requires both Vec sub-cores to participate?
  3. If yes, is this a hardware / low-level FFTS constraint that should be documented explicitly?
  4. If not, could A2A3 support an A5-like `TILE_NO_SPLIT` path where only one Vec sub-core participates and the sync condition is still correctly satisfied/released?

  ### Relevant code references

  - A5 split-aware no-split sync:
    - `include/pto/npu/a5/TPush.hpp`
  - A5 no-split tests with single-Vec-subcore behavior:
    - `tests/npu/a5/src/st/testcase/tpushpop_vc_nosplit/tpushpop_vc_nosplit_kernel.cpp`
    - `tests/npu/a5/src/st/testcase/tpushpop_cv_nosplit/tpushpop_cv_nosplit_kernel.cpp`
  - A2A3 sync path using `CV_CORES_SYNC`:
    - `include/pto/npu/a2a3/TPush.hpp`
  - A2A3 no-split tests showing both-Vec participation / ambiguity:
    - `tests/npu/a2a3/src/st/testcase/tpushpop_vc_nosplit/tpushpop_vc_nosplit_kernel.cpp`
    - `tests/npu/a2a3/src/st/testcase/tpushpop_cv_nosplit/tpushpop_cv_nosplit_kernel.cpp`
  - Upstream impact in PyPTO:
    - `python/pypto/language/dsl_api.py`
    - `src/ir/transforms/expand_mixed_kernel_pass.cpp`

---

## #88 [Bug] A5: expand_clone(d2) has numerical mismatch between simulator and hardware for [B, M, 1]

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/88
- Created: 2026-04-17T07:48:39Z
- Updated: 2026-04-17T09:03:18Z
- Closed: 2026-04-17T09:03:18Z

### Body

### Summary

On A5, `tensor.expand_clone` with `broadcast_dim=2` can produce inconsistent numeric results between simulator and hardware execution for the same input pattern.

The problematic pattern is an input tensor with:

- shape: `[B, M, 1]`
- degenerate stride: `[M, 1, 1]`

### Impact

- Same kernel/pattern yields different output values between sim and real device on A5.
- Upstream cannot rely on simulator results to validate this case.
- PyPTO currently has to skip this A5 system-test scenario as a temporary workaround.

### Reproducer (from PyPTO)

PyPTO test:

- `tests/st/runtime/test_broadcast.py`
- case: `test_tensor_expand_clone`
- parameters: `backend == a5`, `broadcast_dim == 2`

Pattern details:

- Input tensor shape: `[B, M, 1]`
- Input stride: `[M, 1, 1]`
- Expand target: `[B, M, K]`
- Broadcast on last dim (`dim=2`)

### Observed Behavior

For the same test and input pattern on A5:

- simulator result and hardware result are numerically inconsistent
- mismatch is in output values (not a compile-time failure)

### Expected Behavior

Simulator and hardware should produce numerically consistent outputs (within normal tolerance) for the same A5 kernel and inputs.

### Environment

- Upstream project: PyPTO
- Upstream commit: `43dcdd4e9c3c66ce260a2c55757b7d010178a561`
- Host platform: `Linux x86_64`
- NPU kind: A5 / Ascend950

### Notes

Please help confirm whether A5 handling of `[B, M, 1]` with degenerate stride in this broadcast path has a simulator-vs-hardware semantic gap.

---

## #90 Reclassify TSETFMATRIX / TSET_IMG2COL_* / TGET_SCALE_ADDR as micro instructions and remove them from tile docs

- State: open
- URL: https://github.com/hw-native-sys/pto-isa/issues/90
- Created: 2026-04-18T06:16:11Z
- Updated: 2026-05-08T01:36:44Z
- Closed: 

### Body

## Problem

The current PTO ISA documentation still presents the following `T*` APIs as part of the tile instruction surface:

- `TSETFMATRIX`
- `TSET_IMG2COL_RPT`
- `TSET_IMG2COL_PADDING`
- `TGET_SCALE_ADDR`

That categorization is misleading at the API/semantic level.

These operations do not primarily describe tile payload transformation. They program or derive control/state used by later execution:

- `TSETFMATRIX` programs FMATRIX-related configuration state.
- `TSET_IMG2COL_RPT` programs IMG2COL repeat metadata.
- `TSET_IMG2COL_PADDING` programs IMG2COL padding metadata.
- `TGET_SCALE_ADDR` derives/binds an address relationship rather than performing tile arithmetic.

As a result, the tile-family docs overstate what the tile surface actually is, and PTO-AS / ISA taxonomy remains blurry.

## Requested change

Reclassify these four operations into the **micro-instruction** surface and temporarily remove them from the tile instruction documentation.

This should be treated as a **semantic/API taxonomy change**, not just a wording tweak.

## Scope

### Reclassify
Move the semantic classification of:

- `TSETFMATRIX`
- `TSET_IMG2COL_RPT`
- `TSET_IMG2COL_PADDING`
- `TGET_SCALE_ADDR`

from tile-instruction docs into the micro-instruction docs.

### Temporarily remove from tile docs
Until the micro-instruction placement is complete, remove these ops from tile-instruction taxonomies and tile-family summaries so the tile docs remain semantically clean.

## Affected documentation areas

At minimum, review and update:

- `docs/isa/instruction-surfaces/tile-instructions*.md`
- `docs/isa/instruction-families/tile-families*.md`
- `docs/isa/instruction-surfaces/README*.md`
- `docs/assembly/PTO-AS*.md`
- `docs/mkdocs/src/manual/07-instructions*.md`
- `docs/isa/manifest.yaml`
- generated index / matrix outputs derived from the manifest
- the micro-instruction landing / group pages under `docs/isa/scalar/ops/micro-instruction/`

## Acceptance criteria

- The tile instruction docs no longer classify these four operations as tile instructions.
- The micro-instruction docs explicitly include and explain these four operations.
- PTO-AS docs make the semantic family clear and do not imply that a `T*` name automatically belongs to the tile surface.
- `docs/isa/manifest.yaml` and generated family/index pages reflect the new classification.
- English and Chinese docs remain aligned.
- `git diff --check` passes.
- Manifest-derived doc generation/check scripts pass.

## Notes

This issue is intentionally limited to these four operations. A broader semantic cleanup of the `T*` namespace may still be needed later, but that should be tracked separately.

---

## #94 [Bug] 0dcce451 breaks A5 cross-core TPUSH compilation with missing TInsertMode::NZ_PLUS_1

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/94
- Created: 2026-04-21T07:02:38Z
- Updated: 2026-05-28T03:09:35Z
- Closed: 2026-04-22T02:20:11Z

### Body

### Component

A5 backend (`include/pto/npu/a5/*`), cross-core `TPUSH` / V2C path

### Description

A5 cross-core kernels stop compiling starting from commit `0dcce451ea95f7fd42e090c559dfaacc3b04494f`.

The failure happens because `include/pto/npu/a5/TInsert.hpp` no longer defines `TInsertMode::NZ_PLUS_1`, but `include/pto/npu/a5/TPush.hpp` still references `TInsertMode::NZ_PLUS_1` in `pushVec2MatFiFo()` for the split-M vector-to-matrix FIFO path.

### Steps to Reproduce

1. Checkout `pto-isa` at `0dcce451ea95f7fd42e090c559dfaacc3b04494f`.
2. Use this ISA revision with PyPTO on A5 hardware.
3. Run:

```bash
pytest tests/st/runtime/test_cross_core.py -k test_tpush_tpop_v2c_updown -v --forked --device 0 --platform a5
```

I also reproduced this with the full cross-core suite:

```bash
pytest tests/st/runtime/test_cross_core.py -v --forked --device 0 --platform a5
```

### Expected Behavior

The A5 cross-core test should compile and run successfully.

This works with last known good commit `54c7c6dcfab5c21d523de6939bf8d7b305c5a804`.

### Actual Behavior

Compilation fails in the A5 ISA headers with:

```text
error: no member named 'NZ_PLUS_1' in 'pto::TInsertMode'
```

The failing reference is in `include/pto/npu/a5/TPush.hpp` inside `pushVec2MatFiFo()`, where the split-M path still instantiates `TINSERT_IMPL<TInsertMode::NZ_PLUS_1>(...)`.

### Git Commit ID

First bad commit: `0dcce451ea95f7fd42e090c559dfaacc3b04494f`

Last good commit: `54c7c6dcfab5c21d523de6939bf8d7b305c5a804`

### NPU Kind

Ascend 950 (A5)

### Host Platform

Linux (aarch64)

### Additional Context

The regression range is:

- Good: `54c7c6dcfab5c21d523de6939bf8d7b305c5a804`
- Bad: `0dcce451ea95f7fd42e090c559dfaacc3b04494f` (`add Tinsert vec-to-vec, ub2l1 unaligned, fp4/hif8 dtype`)

What changed across that boundary:

- In `include/pto/npu/a5/TInsert.hpp`, `TInsertMode` was reduced to only `SPLIT2` and `SPLIT4`.
- In `include/pto/npu/a5/TPush.hpp`, the A5 vector-to-matrix FIFO path still uses `TInsertMode::NZ_PLUS_1`.

I also checked the generated A5 kernel for the failing case. It goes through a V2C path using `TPipe<..., Direction::DIR_V2C, ...>` and emits `TPUSH(...)` on the AIV side, which reaches this stale `NZ_PLUS_1` reference during compilation.


---

## #96 [Bug] SIMT kernels (MSCATTER/MGATHER) hang due to missing cce::async_invoke context in direct function pointer dispatch

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/96
- Created: 2026-04-22T07:50:37Z
- Updated: 2026-05-13T07:16:21Z
- Closed: 2026-05-13T07:16:21Z

### Body

### Platform

a5 (Ascend 950 hardware)

### Runtime Variant

tensormap_and_ringbuffer

### Description

Kernels that internally use `cce::async_invoke` to launch SIMT sub-threads (e.g., pto-isa's `MSCATTER` and `MGATHER`) hang indefinitely when dispatched via the simpler runtime's direct function pointer call. The same kernels execute correctly when launched via the standard CANN `<<<1, nullptr, stream>>>` mechanism (confirmed by running pto-isa's own ST tests on the same device).

The root cause is that `aicore_executor.cpp` dispatches kernels as plain C function calls:

```cpp
// src/a5/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp:34-42
UnifiedKernelFunc kernel = (UnifiedKernelFunc)payload->function_bin_addr;
kernel(reinterpret_cast<__gm__ int64_t *>(payload->args));
```

This is sufficient for regular SPMD kernels (TLOAD, TSTORE, TADD, etc.), but SIMT kernels require additional context that the standard kernel launch infrastructure (`rtKernelLaunchWithHandleV2`) sets up:
- Thread ID context (`__cce_simt_get_TID_X/Y`)
- Warp/lane configuration (e.g., 32 warps x 32 lanes = 1024 threads)
- Vector pipe scheduling state for `cce::async_invoke`

Without this context, `cce::async_invoke` inside MSCATTER has no thread dispatch target and the kernel hangs.


### Steps to Reproduce

```markdown
A minimal AIV kernel that triggers the hang — any kernel using `cce::async_invoke` via pto-isa MSCATTER:


#include <pto/pto-inst.hpp>
using namespace pto;

// This kernel hangs when dispatched via direct function pointer call
static __aicore__ void mscatter_kernel(__gm__ float* src_ptr, __gm__ int32_t* idx_ptr, __gm__ float* out_ptr) {
  #if defined(__DAV_VEC__)
  constexpr int kRows = 8, kCols = 32, kOutSize = 256;

  using TileData_src = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor, -1, -1>;
  using TileData_idx = Tile<TileType::Vec, int32_t, kRows, kCols, BLayout::RowMajor, -1, -1>;
  using GlobalData_out = GlobalTensor<float, Shape<1,1,1,1,kOutSize>, Stride<1,1,1,kOutSize,1>, Layout::ND>;

  TileData_src srcTile(kRows, kCols);
  TileData_idx idxTile(kRows, kCols);
  TASSIGN(idxTile, 0x0);
  TASSIGN(srcTile, kRows * kCols * sizeof(int32_t));

  GlobalTensor<float, Shape<1,1,1,kRows,kCols>, Stride<1,1,1,kCols,1>> srcGlobal(src_ptr);
  GlobalTensor<int32_t, Shape<1,1,1,kRows,kCols>, Stride<1,1,1,kCols,1>> idxGlobal(idx_ptr);
  GlobalData_out outGlobal(out_ptr);

  TLOAD(srcTile, srcGlobal);
  TLOAD(idxTile, idxGlobal);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

  // This call hangs — cce::async_invoke inside MSCATTER has no SIMT context
  MSCATTER(outGlobal, srcTile, idxTile);

  pipe_barrier(PIPE_ALL);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  #endif
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    // standard tensor unpacking ...
    mscatter_kernel(src, idx, out);
}


**Control test:** The same kernel code built and launched via pto-isa's native test framework (`<<<1, nullptr, stream>>>` + `aclrtSynchronizeStream`) passes:

[  PASSED  ] MSCATTERTest.case_float_8x32_512 (6349 ms)


Additionally, commenting out just the `MSCATTER(...)` call while keeping all TLOAD and sync instructions makes the kernel complete successfully via simpler, confirming the hang is specifically inside MSCATTER's `cce::async_invoke`.
```

### Expected Behavior

MSCATTER kernel completes and scatter-stores tile data to global memory, returning results within seconds (as in pto-isa's test: 6349 ms).

### Actual Behavior

Kernel hangs indefinitely. The runtime logs show initialization completes but never reaches task completion:
```
[INFO] ensure_binaries_loaded: [device_runner.cpp:271] DeviceRunner: binaries loaded
[INFO] init_aicore_register_addresses: [host_regs.cpp:150] Successfully initialized register addresses
[taskqueue] Task timed out (300s), automatically killed
```


### Git Commit ID

 a9f3ea951bf9f39f9c960cf4af40db2e559fc90d

### CANN Version

CANN 9.0.0

### Driver Version

7.0.t9.0.B709

### Host Platform

Linux (aarch64)

### Additional Context

### Affected pto-isa instructions

All instructions using `cce::async_invoke` for SIMT execution are affected:
- **MSCATTER** — scatter-store via `simt_mscatter_elem_kernel` (32x32 = 1024 threads)
- **MGATHER** — gather-load (same SIMT pattern)
- Any future SIMT-based instructions

### MSCATTER SIMT implementation (pto-isa)

```cpp
// pto-isa: include/pto/npu/a5/MScatter.hpp
__tf__ AICORE void MScatterElemImpl(...) {
    __ubuf__ const T *srcPtr = (__ubuf__ const T *)__cce_get_tile_ptr(src);
    __ubuf__ const TIdx *idxPtr = (__ubuf__ const TIdx *)__cce_get_tile_ptr(indices);
    // This launches 1024 SIMT threads — requires kernel launch context
    cce::async_invoke<simt_mscatter_elem_kernel<...>>(
        cce::dim3{/*WARP_SIZE=*/32, /*NUM_WARPS=*/32}, tablePtr, srcPtr, idxPtr);
}
```

### Possible fix directions

1. **Detect SIMT kernels at compile time** and use the full kernel launch pipeline (`rtKernelLaunchWithHandleV2`) instead of direct function pointer dispatch
2. **Add SIMT context initialization** before the function pointer call in `aicore_executor.cpp` (if the hardware supports runtime SIMT context setup without full kernel launch)
3. **Mark kernels in `PTO2DispatchPayload`** with a flag indicating whether they require SIMT context, and branch the dispatch path accordingly


---

## #97 Add valid_row/valid_col negative checking in setvalidshape's debug version

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/97
- Created: 2026-04-23T01:59:53Z
- Updated: 2026-04-27T01:35:09Z
- Closed: 2026-04-27T01:35:09Z

### Body

When an input .pto program has a tile whose dynamic valid_row / valid_col can evaluate to a negative value at runtime (for example, minsi(A - k, C) where k > A), ptoas happily generates a kernel that issues TLOAD/TMOV/TMATMUL with that negative valid dim. On the hardware the malformed DMA lands the pipe event scoreboard in an unrecoverable state and the kernel hangs forever — no error, no fault, no log.

The invariant valid_row >= 0 and valid_col >= 0 is part of the implicit contract of pto.alloc_tile / pto.tload etc., but it is nowhere enforced — neither at IR verification time, nor by a runtime guard in the generated kernel. A user who accidentally emits an overshooting driver loop (e.g. for n in range(0, 8192 + 383, 384) which produces n = 8448 > 8192) has no visible signal that anything is wrong; the only symptom is "kernel 16 never returns."

Related: https://github.com/hw-native-sys/PTOAS/issues/322 (verifier should reject shape mismatches on TStore), https://github.com/hw-native-sys/PTOAS/issues/533 (missing sync when loop zero-iterates). These all fall under the broader theme of "ptoas accepts malformed tile sizes and emits code that breaks at runtime."

Reproduction (minimal)
Submit to ptoas any kernel where the allocated tile's valid dim can be negative — the pattern in real code looks like this (trimmed from a qwen3 decode kernel):

%c8192_index  = arith.constant 8192 : index
%c384_index   = arith.constant 384  : index

// arg3 is the external loop's `n` — caller passes n = 8448 on the last iter
func.func @k(..., %arg3: index) {
  %diff  = arith.subi %c8192_index, %arg3 : index         // 8192 - 8448 = -256
  %vcol  = arith.minsi %diff, %c384_index : index         // min(-256, 384) = -256
  %tile  = pto.alloc_tile ... valid_col = %vcol
          : !pto.tile_buf<loc=mat, dtype=bf16, rows=64, cols=384, v_row=64, v_col=?, ...>
  %part  = pto.partition_view ...
  pto.tload ins(%part) outs(%tile)
  // ...tmov, tmatmul on the same tile
}
Compile:

ptoas input.pto -o out.cpp
A full reproducible case is the Qwen3Decode kernel 16 generated at pypto commit 103aaa94 — dump in build_output/Qwen3Decode_20260422_122654/ptoas/qwen3_decode_incore_16.pto, emitted kernel in the same directory's .cpp. Caller at orchestration/qwen3_decode.cpp:375 passes n = 8448 on the 23rd iteration.

Expected behavior
Either:

Compile-time — ptoas does a value-range check on valid_row/valid_col operands and errors out if it can statically prove the value can be negative (e.g. minsi(sub(const_A, x), const_C) with no lower bound on x). At minimum, warn.
Runtime — generated kernel emits a guard such as
if ((int32_t)valid_col < 0 || (int32_t)valid_row < 0) {
    // either: early return (treat as no-op)
    // or:     trap / write error code + return
}
placed before any TLOAD/TMOV/TMATMUL that uses the tile. Silent hang is the worst possible failure mode.
Option 2 is strictly more actionable for users since it surfaces the bug at the first bad iteration rather than requiring the user to diff against a known-good version. Option 1 is a "nice to have" that catches a subset of cases earlier.

---

## #99 Perf: TPUSH/TPOP ~2x slower than equivalent FFTS + GM workspace on spmd_paged_attention

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/99
- Created: 2026-04-24T08:40:58Z
- Updated: 2026-05-08T08:14:57Z
- Closed: 2026-05-08T08:14:57Z

### Body

## Summary

On the `spmd_paged_attention` end-to-end benchmark, replacing `TPUSH`/`TPOP` pipe synchronization with semantically-equivalent `ffts_cross_core_sync` + manual GM workspace ping-pong yields ~**1.9x–2.0x** speedup with no algorithmic change. The two implementations produce identical numerical results and share all tile types, L1/L0/UB layouts, matmul/softmax/online-update logic, and software pipeline schedule — only the AIC↔AIV sync primitive differs.

This suggests the current `TPUSH`/`TPOP` implementation in the PTO ISA carries substantial overhead (vs. the underlying FFTS + DMA it ultimately lowers to) that is worth investigating.

## Benchmark numbers

Measured via `sh tools/benchmark_rounds.sh` on the `simpler` repo (tensormap_and_ringbuffer runtime, onboard platform):

| Example | Case | Elapsed (us) | Speedup |
|---|---|---:|---:|
| `spmd_paged_attention-tpush` (TPUSH/TPOP) | Case1 | 2892.6 | 1.00x |
| `spmd_paged_attention-ffts` (FFTS)  | Case1 | 1495.0 | **1.93x** |
| `spmd_paged_attention-tpush` (TPUSH/TPOP) | Case2 | 1458.1 | 1.00x |
| `spmd_paged_attention-ffts` (FFTS)  | Case2 |  760.4 | **1.92x** |

## Reproducer

- PR: https://github.com/hw-native-sys/simpler/pull/671

Two test directories in the `simpler` repo (simpler PR #671):

- `tests/st/a2a3/tensormap_and_ringbuffer/spmd_paged_attention-tpush/`        — TPUSH/TPOP version (baseline)
- `tests/st/a2a3/tensormap_and_ringbuffer/spmd_paged_attention-ffts/`   — FFTS + GM workspace version

```bash
sh tools/benchmark_rounds.sh   # runs both cases and prints the table above
```

## What is different

Only the AIC↔AIV synchronization primitive:

| | TPUSH/TPOP version (baseline) | FFTS version (fast) |
|---|---|---|
| AIC produces sij | `TPUSH<C2V>(sij_pipe, cTile)` + `prod.record()` | `copy_matrix_cc_to_gm(s_ws_dst, ...)` + `FftsCrossCoreSync<PIPE_FIX, 2>(QK_READY)` |
| AIV consumes sij | `TPOP<C2V>(sij_pipe, sijTile)` | `WaitFlagDev(QK_READY)` + `copy_gm_to_ubuf(ub, s_ws_src, ...)` |
| AIV produces pij | `TPUSH<V2C>(pij_pipe, pijBf16Tile)` | `copy_ubuf_to_gm(p_ws_dst, ...)` + `FftsCrossCoreSync<PIPE_MTE3, 2>(SF_READY)` |
| AIC consumes pij | `TPOP<V2C>(pij_pipe, pijMatTile)` | `TLOAD(pijMatTile, pijGlobal)` from `p_ws` + `WaitFlagDev(SF_READY)` |
| AIC produces oi | `TPUSH<C2V>(oi_pipe, cTile_PV)` | `copy_matrix_cc_to_gm(o_ws_dst, ...)` + `FftsCrossCoreSync<PIPE_FIX, 2>(UP_READY)` |
| AIV consumes oi | `TPOP<C2V>(oi_pipe, oiNewTile)` | `WaitFlagDev(UP_READY)` + `copy_gm_to_ubuf(oi_ub, o_ws_src, ...)` |

Both versions ultimately move the same bytes through the same GM region and synchronize via FFTS flags under the hood — but the TPUSH/TPOP path is ~2x slower.


## Expected outcome

If the TPUSH/TPOP lowering can be tuned to match the manual FFTS pattern, the pipe abstraction becomes usable for performance-critical kernels. Otherwise, users writing high-perf attention kernels will have to drop to raw FFTS + DMA, which defeats the purpose of the FIFO abstraction.

---

## #101 [Bug] cpu/TPush.hpp sim does not correctly model a5 cross-core pipe semantics

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/101
- Created: 2026-04-25T01:57:11Z
- Updated: 2026-05-15T09:04:57Z
- Closed: 2026-05-15T09:04:57Z

### Body

### Description

Running pypto's cross-core test suite on `--platform a5sim` (sim runtime `runtime/src/a5/platform/sim`) produces precision mismatch on 4 of 9 tests **only when codegen uses `BackendType.Ascend950` (the architecturally-correct mapping)**:

- `test_tpop_c2v_nosplit[a5sim]`
- `test_tpop_bidirect_updown[a5sim]`
- `test_tpop_bidirect_leftright[a5sim]`
- `test_tpop_bidirect_nosplit[a5sim]`

The error pattern is precision mismatch (`Mismatched elements: N/N` on the output tensor, values off by orders of magnitude — not numerical drift).

### Key Observation: 910B-style codegen on the same a5 sim runtime passes

If the same test file maps `a5sim → BackendType.Ascend910B` (i.e. emits a2a3-style TPUSH/TPOP code) and runs it on the **unchanged** a5 sim runtime, **all 9 tests pass**. Only the generated kernel changes; `runtime/src/a5/platform/sim` is identical in both runs.

This isolates the bug to the SIM side of pto-isa: **the cpu sim implementation handles a2a3-flavored TPUSH/TPOP correctly, but does not correctly model the a5 cross-core pipe code that the a5 codegen path emits.**

### Observed Codegen Difference (a5 vs a2a3)

For `cross_core_c2v_nosplit`, the AIC kernel emitted under `--pto-arch a5` differs from `a2a3` in three places:

\`\`\`cpp
// a2a3 (passes on a5 sim)
auto v14 = TPipe<0, Direction::DIR_C2V, 8192, 8, 8, true>(v4 /* GM buffer */, v13, v13);
Tile<TileType::Left, float, 32, 64, BLayout::RowMajor, 32, 64, SLayout::RowMajor, 512, ...> v24;

// a5 (fails on a5 sim)
__gm__ void *v6 = nullptr;
auto v14 = TPipe<0, Direction::DIR_C2V, 8192, 8, 2, true>(v6, v13, v13);
//                                          ^^^ SlotNum=2 (not 8)
Tile<TileType::Left, float, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, ...> v24;
//                                              ^^^ NZ (not ND)
\`\`\`

The a5 codegen produces:

1. `nullptr` pipe pointer (a5 hardware uses on-chip cross-core pipe, not a GM buffer).
2. `SlotNum=2` instead of `SlotNum=8`.
3. `Left` tile in NZ layout (col_major blayout, row_major slayout) instead of ND (row_major / row_major).

These are correct for real a5 NPU hardware, but the SIM path uses `pto-isa/include/pto/cpu/TPush.hpp`:

\`\`\`cpp
template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum,
          uint32_t LocalSlotNum = 2, bool EN_UNIT_FLAG = false>
struct TPipe { ... };
\`\`\`

It is a single `dlsym`-based shared-storage simulation with no `--pto-arch a5` differentiation, no fractal-layout pipe semantics, and no special handling for `nullptr` pipe pointer. The npu-side counterparts (`pto/npu/a5/TPush.hpp` vs `pto/npu/a2a3/TPush.hpp`) differ substantially (different direction constants, slot semantics, NZ insertion via `TINSERT_IMPL<TInsertMode::NZ>`), but `cpu/TPush.hpp` only models the a2a3 GM-buffer flavor.

### Reproduction

\`\`\`bash
# In pypto checkout (commit e89dc93)
PYTHONPATH=python:\$PYTHONPATH \\
PTO_ISA_ROOT=\$(pwd)/build_output/_deps/pto-isa \\
python3 -m pytest --forked --platform a5sim tests/st/runtime/test_cross_core.py
\`\`\`

Expected: 9 passed (matches the 910B-style baseline on the same a5 sim runtime).
Actual: 5 passed, 4 failed.

### Suggested Direction

Either:

1. Add a5-aware code paths in `pto-isa/include/pto/cpu/TPush.hpp` (and `TPop.hpp`) that model the on-chip cross-core pipe — recognise `SlotNum != 8`, `nullptr` pipe pointer, and NZ-layout tile transfers — so SIM behaviour matches hardware semantics. **OR**
2. Provide separate `pto/cpu/a5/TPush.hpp` and `pto/cpu/a2a3/TPush.hpp` parallel to the npu-side split, dispatched by `--pto-arch`.

Option 2 mirrors the existing npu-side directory structure and would be the cleaner long-term fix.

### Environment

| Component | Version |
|---|---|
| pypto | `e89dc93` |
| pto-isa | `3c1f56a` |
| Host | Linux aarch64 |

---

## #103 Perf: make TPipe reverse dependency respect FIFO depth

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/103
- Created: 2026-04-28T03:54:40Z
- Updated: 2026-05-08T01:54:08Z
- Closed: 2026-05-08T01:54:08Z

### Body

## Summary

The current A2A3 `TPipe` reverse-dependency implementation appears to behave like a per-tile rendezvous rather than a depth-aware FIFO back-pressure mechanism.

For a GM FIFO with `FIFO_DEPTH = 2`, a producer should be able to have up to two outstanding tiles before it must wait for the consumer to free a slot. In the current implementation, `TPUSH` calls `prod.allocate()` before every push, and `allocate()` waits on `FlagID + 1` unconditionally when reverse dependency is enabled. The matching `FlagID + 1` is emitted by `cons.free()` after `TPOP`.

This means a producer may wait for the consumer to free an older tile even when the next FIFO slot is still unused.

Parent issue: #99
Related simpler fix: https://github.com/hw-native-sys/simpler/pull/676

## Concrete example: C2V `sij_pipe`

In `spmd_paged_attention`, `sij_pipe` is a C2V GM FIFO:

```cpp
using SijPipeT = TPipe<SIJ_FLAG_ID, Direction::DIR_C2V, SIJ_SLOT_SIZE, 2>;
```

With two slots, the desired storage-level behavior is:

```text
sij[0] -> slot 0
sij[1] -> slot 1
sij[2] -> slot 0  // only this write needs slot 0 to be free
```

However, with reverse dependency enabled, the control flow is effectively:

```text
AIC TPUSH(sij[i]):
  allocate()
    wait FlagID + 1
  write GM FIFO slot
  record data-ready FlagID

AIV TPOP(sij[j]):
  wait data-ready FlagID
  read GM FIFO slot
  free FlagID + 1
```

So after the initial free token is consumed by `TPUSH(sij[0])`, `TPUSH(sij[1])` waits for AIV to `TPOP/free(sij[0])`, even though slot 1 is still available.

## Why this matters

This behavior prevents kernels from using the intended two-slot lead window. In `spmd_paged_attention`, the desired software pipeline is:

```text
AIC: QK[i] -> TPUSH(sij[i]) -> TPOP(pij[i-1]) -> PV[i-1] -> TPUSH(oi[i-1])
AIV: TPOP(sij[i-1]) -> SF[i-1] -> TPUSH(pij[i-1]) -> TPOP(oi[i-2]) -> UP[i-2]
```

The pipeline relies on a 2-deep FIFO to keep AIC and AIV offset by one iteration, so that:

- `QK[i]` overlaps with `SF[i-1]`;
- `PV[i-1]` overlaps with `UP[i-2]`.

When reverse dependency is enabled, the per-push `allocate()` wait serializes the producer with the consumer's previous `free()` signal. This reduces the effective lead window from two outstanding tiles to approximately one tile and weakens the intended AIC/AIV overlap.

In the `simpler` benchmark, this was the dominant reason the TPUSH/TPOP version of `spmd_paged_attention` ran around 2900 us, while the same kernel with reverse dependency disabled ran around 1500 us.

## Expected behavior

Reverse dependency should provide depth-aware back-pressure:

- The first `SLOT_NUM` pushes should be allowed without waiting for consumer frees, assuming the slots have not wrapped.
- The producer should wait only when the next `tileIndex % SLOT_NUM` slot may still be occupied.
- For `SLOT_NUM = 2`, `push[0]` and `push[1]` should be allowed to fill slot 0 and slot 1; `push[2]` should wait for slot 0 to be freed.

Equivalently, reverse dependency should behave like a counting semaphore or slot-specific availability protocol, not as a mandatory per-push rendezvous.

## Actual behavior

With reverse dependency enabled:

- `TPUSH` performs `prod.allocate()` before every push.
- For C2V on Cube, `allocate()` waits on `FlagID + 1`.
- `FlagID + 1` is only emitted by the Vector consumer's `free()` after `TPOP`.
- The wait does not appear to distinguish between initial empty slots and wrapped/reused slots.

As a result, `FIFO_DEPTH = 2` does not fully expose two outstanding slots to the producer when reverse dependency is enabled.

## Suggested investigation

Please consider one of the following implementation directions:

1. Make `allocate()/free()` depth-aware, so the producer only waits when it would overwrite a slot that may still be occupied.
2. Initialize reverse-dependency state with `SLOT_NUM` available credits rather than a single initial free signal.
3. Provide an explicit fast path/API for statically scheduled pipelines that can prove `producer_lead <= SLOT_NUM`, while documenting that reverse dependency can be disabled safely only under that proof.

The third option is effectively what `simpler` PR #676 does locally, but a depth-aware default would make `TPush/TPop` safer and more performant for common double-buffered pipelines.


---

## #106 [Feature] Add TColGather / TColScatter — row-axis selection by predicate mask

- State: open
- URL: https://github.com/hw-native-sys/pto-isa/issues/106
- Created: 2026-04-30T07:50:05Z
- Updated: 2026-05-08T01:36:34Z
- Closed: 

### Body

### Summary

Add two new tile ops that select / scatter **whole rows** by a 1-D predicate
mask, keeping the inner (column) dimension intact. Naming follows the existing
`TCol*` family (which already operates along the row-stacking axis):

- `TColGather(src, mask) → dst` — keep rows where `mask[i] == 1`.
- `TColScatter(src, mask) → dst` — write src rows back into the rows of dst
  selected by `mask`.

Concrete shape example for `TColGather`:

```
src  : [8, 256]
mask : [1, 1, 0, 0, 1, 1, 1, 0]    # 1-D, length = src.rows
dst  : [5, 256]                    # rows where mask==1, preserving order; inner 256 untouched
```

`TColScatter` is the dual:

```
src  : [5, 256]
mask : [1, 1, 0, 0, 1, 1, 1, 0]    # length = dst.rows, popcount = src.rows
dst  : [8, 256]                    # src rows are placed into mask==1 positions; mask==0 rows untouched (or zero-filled)
```

The mask is a **runtime tile**, naturally produced by `TCmp` / `TCmpS` (e.g.
`mask = (router_id == expert_id)`), so the same op handles both compile-time
constant and data-dependent selection.

### Motivation / Use Case

The existing `TGather` mask form (cce `vreducev2`, include/pto/npu/a2a3/TGather.hpp:98-115)
only supports 7 fixed strided patterns (P0101 / P1010 / P0001..P1000 / P1111,
include/pto/common/type.hpp:149-159) and operates **along the inner column
axis**: `[R, C] → [R, C/2]` or `[R, C/4]`. There is no row-axis equivalent.

`TColReduce` family (`TColMax / TColMin / TColSum / TColProd`,
include/pto/npu/a2a3/TColReduceOps.hpp) operates along the row-stacking axis
but always **reduces** to `[1, C]` — it cannot output `[x, C]` for arbitrary
x, and it combines rows via sum/max/min/prod rather than selecting them.

Use cases this gap blocks:

1. **MoE / token routing.** DeepSeek-V4 / Qwen MoE flows want to take
   `[num_tokens, hidden]` and produce `[num_selected_tokens, hidden]` based
   on a per-token boolean predicate (this expert / top-k membership). On the
   write-back side, the selected expert outputs need to be scattered back to
   the original token positions — exactly `TColScatter`.

2. **Variable-length sequence packing / unpacking.** Compact a `[max_seq,
   hidden]` tile by a per-token validity mask, then run dense compute on the
   packed `[valid_count, hidden]`.

3. **Predicate-driven row filtering.** Anything of the form
   ```
   mask = TCmpS(src_summary, threshold, GT)
   dst  = src[mask, :]
   ```
   without falling back to the 3-arg `TGather` index form (which needs an
   INT32 index tile of shape `[x, C]` plus a `tmp` UB scratch and emits a
   `vmuls + vgather` per row — see include/pto/npu/a2a3/TGather.hpp index
   form for the cost profile).

### Proposed API / Behavior

```cpp
// pseudo signatures, mask is 1-D over the row axis of src/dst respectively
template <typename DstTile, typename SrcTile, typename MaskTile>
__tf__ AICORE void TColGather(DstTile  __out__ dst,
                              SrcTile  __in__  src,
                              MaskTile __in__  mask);   // mask : [src.rows]

template <typename DstTile, typename SrcTile, typename MaskTile>
__tf__ AICORE void TColScatter(DstTile  __out__ dst,
                               SrcTile  __in__  src,
                               MaskTile __in__  mask);  // mask : [dst.rows]
```

Constraints:

- `mask` is a 1-D predicate tile (one bit / one element per row). Format
  ideally matches what `TCmp` / `TCmpS` produces (packed 1-bit-per-element)
  so the chain `TCmpS → TColGather` requires no format conversion.
- Inner column dim is preserved unchanged. `dst.cols == src.cols`.
- For `TColGather`: `dst.rows == popcount(mask)` (or upper-bound + tail
  filled with a sentinel; an explicit `validRow` out-parameter is fine).
- For `TColScatter`: `src.rows == popcount(mask)`, `dst.rows == mask.length`.

Frontend exposure (suggested):

```python
# pypto-lib
mask = pl.cmps(scores, threshold, mode=pl.CmpMode.GT)  # [num_tokens]
selected = pl.colgather(tokens, mask)                  # [num_selected, hidden]
# ...compute on selected...
pl.colscatter(out, selected, mask, output_tensor=out_tensor)
```

### Alternatives Considered

- **Index form `TGather(dst, src, indices, tmp)`.** Works, but: (1) caller
  must materialize an INT32 index tile of shape `[selected_rows, hidden]`
  (full inner dim repeated, since the index form indexes elements not rows),
  (2) needs a `tmp` UB scratch on the B16 path, (3) emits `vmuls + vgather`
  per row. Heavy compared to a single row-axis selection op driven by a
  `[src.rows]` mask.
- **`pl.load` with computed offsets.** O(selected_rows) GM↔UB transfers,
  defeats the point of having data already in UB.
- **`TColReduce` family.** Wrong semantics — reduces to `[1, C]` and combines
  rows via sum/max/min, can't preserve per-row data.
- **`TGather` mask form (current).** Wrong axis — operates within each row
  along the column axis, and only 7 fixed strided patterns; cannot select
  rows.

### Additional Context

- pypto frontend gather entry points (for reference of where this would
  surface): pypto python/pypto/language/op/tile_ops.py:1818-1878 (mask form),
  pypto python/pypto/language/op/tile_ops.py:1881-1900 (mscatter).
- pypto codegen mapping: pypto src/backend/common/pto_ops_common.cpp:1847-1849.
- Existing `TGatherOp` IR in PTOAS: include/PTO/IR/PTOOps.td:2256-2280 — a
  `TColGatherOp` / `TColScatterOp` would naturally live next to it.
- The HW-side bring-up question: is there a CCE intrinsic that can drive a
  row-stride packed copy under a 1-D predicate (e.g. `vreducev2` over the
  outer loop, or a `pto_copy_ubuf_to_ubuf` driven by a popcount-prefix-sum)?
  If a single-instruction implementation is not available, even a templated
  software composition (popcount-prefix-sum + per-row `pto_copy_ubuf_to_ubuf`)
  exposed as `TColGather` would already be a meaningful API improvement —
  the value is in having a stable op that the compiler can pattern-match on,
  rather than requiring every user to hand-roll the index materialization.

---

## #107 [Feature] TExpandS on A2/A3: explicit BF16 support and test coverage for the Vec (UB) path

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/107
- Created: 2026-04-30T08:24:41Z
- Updated: 2026-05-13T01:37:41Z
- Closed: 2026-05-13T01:37:41Z

### Body

### Summary

On A2/A3, `TExpandS` (broadcast scalar into a tile, the kernel behind frontend
`pl.full` / `pto.texpands`) supports BF16 only on the **Mat (L1)** path.
The **Vec (UB)** path is reachable but has no explicit dtype guarantee and no
test coverage for `bfloat16_t`. Please:

1. Add `bfloat16_t` to the dtype whitelist (`static_assert`) in the A2/A3
   `TEXPANDS_IMPL`.
2. Add A2/A3 NPU testcases that exercise `TExpandS` Vec with BF16
   (`tests/npu/a2a3/src/st/testcase/texpands/`).
3. (Optional but ideal) confirm cce `vector_dup` on A2/A3 accepts a
   `bfloat16_t` scalar / pointer with the existing call shape, or document
   any required reinterpret.

### Motivation / Use Case

Frontend lowering of `pl.full(shape, pl.BF16, value)` ultimately wants to call
`TExpandS` on a UB tile. Today on A2/A3:

- `tile.full` → `pto.texpands` → `TExpandS<TileData>(...)` Vec path.
- BF16 is **not** in the white-listed dtype set in `TEXPANDS_IMPL` for A2/A3
  (the impl carries no dtype `static_assert` at all for A2/A3 — see comparison
  below — so it's neither denied nor blessed).
- No A2/A3 BF16 testcase exists for the Vec path, so we cannot rely on it.

This blocks BF16 use cases that need a UB-resident initialized tile, e.g.:

- Initializing accumulators / running max / running sum to a sentinel value
  (`-inf`, `+inf`, `0`) before a flash-attention-style loop on BF16 inputs.
- Filling a UB tile with a routing mask sentinel in MoE / DeepSeek-V4 decode
  flows where the rest of the pipeline is BF16.
- Writing a BF16 zero / constant into UB before a partial-update pattern that
  then runs `TAdd` / `TMax` against streamed BF16 data.

A5 already handles BF16 cleanly on both Vec and Mat:

- `include/pto/npu/a5/TExpandS.hpp:144-149` — `TEXPANDS_IMPL` `static_assert`
  explicitly lists `bfloat16_t`.
- `include/pto/npu/a5/TExpandS.hpp:67-68` — Mat path uses
  `create_cbuf_matrix_bf16`.
- A5 NPU tests under `tests/npu/a5/src/st/testcase/texpands/` cover BF16.

A2/A3 today:

- `include/pto/npu/a2a3/TExpandS.hpp:87-92` — Mat path has the BF16 branch
  (`create_cbuf_matrix_bf16`), and `tests/npu/a2a3/src/st/testcase/texpands_mat/`
  exercises it.
- `include/pto/npu/a2a3/TExpandS.hpp:140-152` — Vec path's `TEXPANDS_IMPL`
  has no dtype `static_assert`, so `bfloat16_t` template-instantiates and
  falls into `B82B16Trait<bfloat16_t>::TransType == bfloat16_t` (utils.hpp:90-105,
  pass-through; the trait only converts B8 → B16) → `vector_dup(dstPtr,
  scalar, ...)`.
- `tests/npu/a2a3/src/st/testcase/texpands/` (the Vec testcase) has **zero**
  matches for `bfloat16` / `bf16`.

So BF16 is **silently reachable but unverified** on A2/A3 Vec. We want it
explicitly supported and tested, in line with A5.

Note that A2/A3 already supports BF16 in many other Vec / unary / binary ops
(`TInsert`, `TImg2col`, `TMatmul`, `TExtract`, `TStore`, `TScatter`,
`TAnd`/`TOr`, `TUnaryOp`, `TRowExpand` — all carry BF16 paths via
`B82B16Trait`), so adding BF16 to `TExpandS` is consistent with the rest of
the A2/A3 surface, not a one-off ask.

### Proposed Change

**(1) `include/pto/npu/a2a3/TExpandS.hpp`** — add BF16 to the impl-level
`static_assert` (mirroring A5):

```cpp
template <typename TileData>
PTO_INTERNAL void TEXPANDS_IMPL(TileData &dst, typename TileData::DType scalar)
{
    using T = typename TileData::DType;
    static_assert(
        std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
        std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
        std::is_same<T, int8_t>::value  || std::is_same<T, uint8_t>::value ||
        std::is_same<T, half>::value    || std::is_same<T, float>::value ||
        std::is_same<T, bfloat16_t>::value,                       // <-- add
        "TEXPANDS: Invalid data type");
    // ... rest unchanged
}
```

**(2) `tests/npu/a2a3/src/st/testcase/texpands/`** — add BF16 cases analogous
to the existing FP16 / FP32 ones (and to the BF16 cases already present in
`texpands_mat/` and in A5 `texpands/`).

**(3) Confirm or fix the cce intrinsic call.** If `vector_dup` on A2/A3
rejects `bfloat16_t` directly and needs a `reinterpret_cast` to `int16_t` /
`uint16_t` (analogous to how A5 paths handle it), patch the Vec path to do
that conversion when `T == bfloat16_t`. Otherwise the change is purely a
white-list + tests addition.

### Additional Context

- Frontend entry point: `pypto python/pypto/language/op/tile_ops.py:438`
  (`pl.full(shape, dtype, value)`).
- pypto codegen mapping: `pypto src/backend/common/pto_ops_common.cpp:1981-1988`
  (`tile.full` → `pto.texpands`).
- A2/A3 Vec impl: `include/pto/npu/a2a3/TExpandS.hpp:43-74` (uses
  `B82B16Trait` and `vector_dup`).
- A2/A3 Mat impl (already BF16): `include/pto/npu/a2a3/TExpandS.hpp:76-117`
  (uses `create_cbuf_matrix_bf16` for `bfloat16_t`).
- A5 reference: `include/pto/npu/a5/TExpandS.hpp` — full Vec + Mat BF16,
  including `static_assert` whitelist.
- BF16 is a first-class element type for our DSv4 / Qwen MoE work in
  pypto-lib; missing BF16 on `pl.full` forces awkward FP16/FP32 fill +
  cast workarounds.

---

## #110 Fix TPush CPU-SIM A5 support issue

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/110
- Created: 2026-05-04T13:55:03Z
- Updated: 2026-05-15T08:46:30Z
- Closed: 2026-05-15T08:46:30Z

### Body

This pull request modifies the TPush implementation to correctly handle DIR_BOTH pipes by incorporating TileProd::Loc checks when determining the transfer direction. Feedback includes a potential bug where TileType::Mat transfers might be incorrectly recorded due to an incomplete check in the underlying IsC2VProducerTile function, along with a suggestion to improve code formatting and comment placement for better readability.

---

## #111 TDIV的ST高精度标识highPrecision没用到

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/111
- Created: 2026-05-06T07:28:57Z
- Updated: 2026-05-13T01:27:45Z
- Closed: 2026-05-13T01:27:45Z

### Body

https://github.com/hw-native-sys/pto-isa/blob/0a1ce522c1c6b5fa61d941b5ec439b8671a134eb/tests/npu/a5/src/st/testcase/tdiv/tdiv_kernel.cpp#L60


LaunchTDivHalf 有相同问题

---

## #113 [Bug] CPU sim TADD_IMPL rejects subview tiles: template deduction fails when src physical shape differs from dst

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/113
- Created: 2026-05-06T09:43:33Z
- Updated: 2026-05-07T02:34:51Z
- Closed: 2026-05-07T02:34:51Z

### Body

## Summary

`TADD_IMPL` in `include/pto/cpu/TAdd.hpp` uses a single template parameter `tile_shape` for all three operands, requiring `dst`, `src0`, and `src1` to be **identical C++ types**. When `src` is a subview of a wider tile (e.g., a 32×32 window inside a 32×64 physical buffer), it retains the parent's `Cols=64` and therefore has a different C++ type from an independently-allocated 32×32 `dst` tile (`Cols=32`). Template deduction fails at compile time.

The same program compiles and runs correctly on real hardware (`a2a3`/`ccec`) because the NPU `TADD_IMPL` uses **three independent template parameters** (`TileDataDst`, `TileDataSrc0`, `TileDataSrc1`).

---

## Affected File

`include/pto/cpu/TAdd.hpp` — lines 62–68

```cpp
// Current (broken for subview operands)
template <typename tile_shape>
PTO_INTERNAL void TADD_IMPL(tile_shape &dst, tile_shape &src0, tile_shape &src1)
{
    unsigned row = dst.GetValidRow();
    unsigned col = dst.GetValidCol();
    TAdd_Impl<tile_shape>(dst.data(), src0.data(), src1.data(), row, col);
}
```

---

## Steps to Reproduce

**Compiler requirement:** GCC ≥ 14 or Clang ≥ 15 (needed for `_Float16` and `<format>`).

```cpp
// tadd_subview_repro.cpp
#include <pto/common/cpu_stub.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/cpu/TAdd.hpp>

int main()
{
    // Parent tile: physical 32×64, RowStride = 64
    pto::Tile<pto::TileType::Vec, float, 32, 64,
              pto::BLayout::RowMajor, -1, -1,
              pto::SLayout::NoneBox, 512,
              pto::PadValue::Null, pto::CompactMode::Null> parent;

    // Subview: same physical buffer (Cols=64 → RowStride=64),
    // but valid window is 32×32
    pto::Tile<pto::TileType::Vec, float, 32, 64,
              pto::BLayout::RowMajor, 32, 32,
              pto::SLayout::NoneBox, 512,
              pto::PadValue::Null, pto::CompactMode::Null> left_view, right_view;

    // Output: independent 32×32 allocation (Cols=32 → RowStride=32)
    pto::Tile<pto::TileType::Vec, float, 32, 32,
              pto::BLayout::RowMajor, -1, -1,
              pto::SLayout::NoneBox, 512,
              pto::PadValue::Null, pto::CompactMode::Null> dst;

    // Equivalent to TADD(dst, left_view, right_view)
    pto::TADD_IMPL(dst, left_view, right_view);  // ← compile error
}
```

```bash
g++-15 -std=c++20 -D__CPU_SIM -include cstdint \
    -I<pto-isa-root>/include tadd_subview_repro.cpp
```

---

## Expected Behavior

Compilation succeeds. `TADD` on a subview operand should work in CPU sim just as it does on hardware.

## Actual Behavior

```
error: no matching function for call to 'TADD_IMPL(
  Tile<Vec, float, 32, 32, RowMajor, -1, -1, ...>&,
  Tile<Vec, float, 32, 64, RowMajor, 32, 32, ...>&,
  Tile<Vec, float, 32, 64, RowMajor, 32, 32, ...>&)'
note: deduced conflicting types for parameter 'tile_shape'
  ('Tile<...,Cols=32,...,ValidRow=-1,ValidCol=-1,...>'
   and 'Tile<...,Cols=64,...,ValidRow=32,ValidCol=32,...>')
```

---

## Root Cause

`include/pto/cpu/TAdd.hpp` (line 62) uses one template parameter:

```cpp
template <typename tile_shape>   // ← single param forces dst==src0==src1 type
TADD_IMPL(tile_shape &dst, tile_shape &src0, tile_shape &src1)
```

The NPU implementation (`include/pto/npu/a2a3/TAdd.hpp`) uses three independent parameters:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
TADD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
// enforces only valid_shape equality at runtime via TAddCheck
```

A subview retains the parent's `Cols` (and thus `RowStride`), making it a distinct C++ type from an independently-allocated tile with the same valid shape. The CPU sim's single-parameter signature cannot accommodate this.

---

## Proposed Fix

Align `include/pto/cpu/TAdd.hpp` with the NPU signature:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TADD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    // Mirror TAddCheck: enforce valid shape consistency at runtime
    assert(src0.GetValidRow() == dst.GetValidRow() &&
           src0.GetValidCol() == dst.GetValidCol());
    assert(src1.GetValidRow() == dst.GetValidRow() &&
           src1.GetValidCol() == dst.GetValidCol());

    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    // Use per-tile RowStride via TAdd_Impl_3 (new helper, or template each pointer type)
    TAdd_Impl_3<TileDataDst, TileDataSrc0, TileDataSrc1>(
        dst.data(), src0.data(), src1.data(), validRow, validCol);
}
```

The same pattern likely applies to other elementwise binary ops in `include/pto/cpu/` (`TSub`, `TMul`, `TDiv`, etc.) that share the same single-template-parameter design.

---

## Impact

| Scenario | a2a3sim | a2a3 hardware |
|----------|---------|---------------|
| `pto.alloc_tile` → `pto.subview` → `TADD` | ❌ compile error | ✅ correct |
| Any binary elementwise op on subview in Vec | ❌ compile error | ✅ correct |
| Unary ops, matmul, non-subview operands | ✅ unaffected | ✅ correct |

---

## Environment

- **pto-isa commit:** `0a1ce522c1c6b5fa61d941b5ec439b8671a134eb`
- **Host:** Linux aarch64
- **NPU:** N/A (CPU sim — not hardware-specific)
- **Compiler tested:** g++-15 (GCC 15.1), `-std=c++20`

---

## #118 [Bug] flash_atten-v2 (PR #117) emits TPipe<...,SlotNum=8,LocalSlotNum=8,...> for gm_slot_tensor pipe init, diverging from manual FA effective LocalSlotNum=2 and causing long-sequence timeout

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/118
- Created: 2026-05-09T06:41:57Z
- Updated: 2026-05-09T07:52:26Z
- Closed: 2026-05-09T07:52:26Z

### Body

### Component

PTO Dialect / ODS (`include/PTO/IR`) and `lib/PTO/Transforms/PTOLowerFrontendPipeOpsPass.cpp`.

### Description

The PTO-DSL FlashAttention v2 example in [PR #117 `kernels/python/flash_atten-v2/`](https://github.com/hw-native-sys/pto-isa/pull/117) is structurally aligned with the manual reference [`kernels/manual/common/flash_atten/fa_performance_kernel.cpp`](https://github.com/hw-native-sys/pto-isa/blob/main/kernels/manual/common/flash_atten/fa_performance_kernel.cpp): `TILE_S1 = 256`, `CUBE_S1 = 128`, `kTileFactor = 2`, address-based slot model on all three pipes (`pto.aic_initialize_pipe(gm_slot_tensor=...)` / `talloc_to_aiv` / `tpush_to_aiv` / `tpop_from_aic` / `tfree_from_aic`).

Single-call correctness (`atol = rtol = 1e-3` against fp32 reference, fresh process per length, on A3):

| S1   | NUM_TILES | status         | max_err  |
|-----:|----------:|----------------|----------|
| 1024 |         4 | **PASSED**     | 4.43e-05 |
| 2048 |         8 | **PASSED**     | 2.72e-05 |
| 4096 |        16 | aicore timeout |          |
| 8192 |        32 | aicore timeout |          |

The manual C++ reference at the same `case_float_H_128_S0_128_S1_8192` shape runs to completion. The DSL-generated kernel differs from the manual reference in the `TPipe` template parameters used for the three cross-core FIFO pipes:

| Source | QK pipe | P pipe | PV pipe |
|---|---|---|---|
| Manual `fa_performance_kernel.cpp:790,795,799` | `TPipe<..., SlotNum=8, LocalSlotNum=2, IsNoSplit=false, EN_UNIT_FLAG=true>` | `TPipe<..., SlotNum=8>`; effective `LocalSlotNum=2` via C++ default | `TPipe<..., SlotNum=8, LocalSlotNum=2, IsNoSplit=false, EN_UNIT_FLAG=true>` |
| DSL after ptoas lowering | `TPipe<..., SlotNum=8, LocalSlotNum=8, IsNoSplit=false>` | `TPipe<..., SlotNum=8, LocalSlotNum=8, IsNoSplit=false>` | `TPipe<..., SlotNum=8, LocalSlotNum=8, IsNoSplit=false>` |

Per [`include/pto/npu/a2a3/TPush.hpp:28`](https://github.com/hw-native-sys/pto-isa/blob/main/include/pto/npu/a2a3/TPush.hpp#L28), the C++ default for `LocalSlotNum` is **2**. Therefore the manual P pipe, although it does not spell out the fifth template argument, also has effective `LocalSlotNum=2`.

Note: QK/PV also differ in `EN_UNIT_FLAG` (`true` in the manual reference, default `false` in DSL-generated code). The experiment below specifically isolates `LocalSlotNum` by only rewriting `8 -> 2`; however, this issue should not claim that `LocalSlotNum` is the only template-level difference.

DSL gets `LocalSlotNum=8` because three places in ptoas conspire to drop the manual/default `LocalSlotNum=2` behavior:

#### Defect A — verifier rejects `local_slot_num` on globaltensor pipe init

[`lib/PTO/IR/PTO.cpp` (HEAD `eeeb1f4`, lines 10680–10682)](https://github.com/hw-native-sys/PTOAS/blob/eeeb1f4/lib/PTO/IR/PTO.cpp#L10680-L10682):

```cpp
if (op.getLocalSlotNumAttr())
  return op.emitOpError(
      "globaltensor pipe init does not use 'local_slot_num'");
```

The DSL has no legal way to override `LocalSlotNum` on the address-based / `gm_slot_tensor` form added in PTOAS PR #606. PR #569 (legacy `local_slot_num` support) only covers `gm_slot_buffer`.

#### Defect B — lowering hard-codes empty `localSlotNumAttr` for the globaltensor branch

[`lib/PTO/Transforms/PTOLowerFrontendPipeOpsPass.cpp` (HEAD `eeeb1f4`, lines 123–134)](https://github.com/hw-native-sys/PTOAS/blob/eeeb1f4/lib/PTO/Transforms/PTOLowerFrontendPipeOpsPass.cpp#L123-L134):

```cpp
if (initOp.getGmSlotTensor()) {
  ...
  auto pipe = rewriter.create<InitializeL2G2LPipeOp>(
      loc, pipeTy, dirAttr, slotSizeAttr, slotNumAttr,
      IntegerAttr{},     // ← localSlotNumAttr
      IntegerAttr{},     // ← flagBaseAttr
      noSplitAttr, initOp.getGmSlotTensor(), Value{}, Value{});
  ...
}
```

Even if Defect A were lifted, this branch would still drop the user attribute. The non-globaltensor branch (lines 152–156) at least passes `getLocalSlotNumAttr()` through.

#### Defect C — EmitC fallback is `getSlotNum()` (= 8), not the C++ template default 2

[`lib/PTO/Transforms/PTOToEmitC.cpp` (HEAD `eeeb1f4`, lines 628–630)](https://github.com/hw-native-sys/PTOAS/blob/eeeb1f4/lib/PTO/Transforms/PTOToEmitC.cpp#L628-L630):

```cpp
int32_t localSlotNum = initOp.getLocalSlotNumAttr()
                           ? initOp.getLocalSlotNumAttr().getInt()
                           : initOp.getSlotNum();   // ← =8 in this kernel
```

When the attr is absent, EmitC writes `LocalSlotNum=SlotNum` explicitly. This does not match the C++ API default (`LocalSlotNum=2`) that the manual reference relies on.

### What the source code proves, and the likely timeout mechanism

The source-level mismatch is clear:

1. A2/A3 `TPipe` defaults `LocalSlotNum` to `2`.
2. The manual FA reference uses effective `LocalSlotNum=2` on QK/P/PV.
3. The `gm_slot_tensor` frontend form cannot legally carry `local_slot_num`.
4. The globaltensor lowering branch drops/omits `localSlotNumAttr`.
5. EmitC falls back to `getSlotNum()` when the attr is absent, so `SlotNum=8` becomes `LocalSlotNum=8`.

What can be directly seen in `include/pto/npu/a2a3/TPush.hpp` is that `LocalSlotNum` is used through `RingFIFO<SlotSize, SlotNum, LocalSlotNum>` and affects the local consumer-buffer address rotation:

```cpp
fifo.C2V_CONSUMER_BUF +
    (tileIndex % RingFiFo::LOCAL_SLOT_NUM) * ConsM * ConsN * sizeof(T);
```

and similarly for `V2C_CONSUMER_BUF`.

Therefore the most conservative source-backed statement is:

- Manual FA rotates consumer local buffers with period 2.
- DSL-generated FA rotates consumer local buffers with period 8.
- Both use the same GM ring depth (`SlotNum=8`).
- For long sequences, GM slots are reused after tile index 8, so the generated kernel exercises ring reuse with a different local-buffer rotation policy than the manual reference.

This is a plausible cause of the observed AICore timeout: with `LocalSlotNum=8`, the consumer-side local buffer lifetime and the FIFO free/ready synchronization no longer match the manual kernel's intended two-slot ping-pong schedule. When the 8-slot GM ring is reused, stale data, premature reuse, or unmatched producer/consumer progress can lead to a wait that never observes the expected signal.

The original explanation involving “8 × 3 = 24 event identities exceeding an 8-event pool” is a possible hypothesis, but it is not directly proven by `TPush.hpp`: the visible `TPipe` implementation uses fixed `FlagID` / `FlagID+1`-style FFTS messages, while `LocalSlotNum` is directly visible in local buffer address rotation. To prove the event-ID explanation, we would need to compare the IR/C++ emitted by `--enable-insert-sync` and show that event lifetimes or assigned event IDs become conflicting only in the `LocalSlotNum=8` version.

Observed behavior is still consistent with the `LocalSlotNum` mismatch:

- `NUM_TILES <= 8`: the GM ring has not been reused beyond its 8 slots, so the mismatch is less likely to surface.
- `NUM_TILES >= 16`: the 8-slot GM ring has been reused multiple times, and the generated `LocalSlotNum=8` local-buffer rotation diverges substantially from the manual two-slot ping-pong pattern.

### Reproduction

Using PR #117 commit `35b35de4` (kernels/python/flash_atten-v2/) on A3 with `ptoas --pto-arch=a3 --enable-insert-sync` and bisheng built kernel:

```bash
cd kernels/python/flash_atten-v2
bash run_fa.sh --tiles 4 --lengths 1024  # PASSED, max_err 4.43e-05
bash run_fa.sh --tiles 8 --lengths 2048  # PASSED, max_err 2.72e-05
bash run_fa.sh --tiles 32 --lengths 8192 # builds, runtime aicore timeout
```

Inspecting the emitted `build_artifacts/fa_32.cpp`:

```cpp
auto v40 = TPipe<0, Direction::DIR_C2V, 131072, 8, 8, false>(v39, v18, v18);
auto v43 = TPipe<2, Direction::DIR_C2V, 65536, 8, 8, false>(v42, v18, v18);
auto v46 = TPipe<4, Direction::DIR_V2C, 65536, 8, 8, false>(v45, v18, v18);
```

All three `LocalSlotNum=8`. Manually rewriting only the generated `LocalSlotNum` from `8` to `2` and rebuilding allows S1=8192 to run to completion in this setup. This strongly implicates the `LocalSlotNum` mismatch, although QK/PV still differ from the manual reference in `EN_UNIT_FLAG`, so a full semantic parity fix should treat `local_slot_num` as the primary bug and track `EN_UNIT_FLAG` separately if needed.

### Related issues / PRs

- PTOAS PR #606 ("fix global tensor half-slot split pipes", merged 2026-04-29) — introduced the `gm_slot_tensor` form; the verifier rejection in Defect A landed in this PR.
- PTOAS PR #569 ("feat: support `local_slot_num` on legacy pipe init", merged 2026-04-25) — added the attribute on `gm_slot_buffer` form only.
- pto-isa #629 ("FA lit regression test with S1_TILE=512 crashes at runtime") — same family, OPEN.
- pto-isa #621 ("Expose FIFO consumer sync period (`cons_sync_period`)") — `kFaCvFifoConsSyncPeriod=4` is the *other* manual knob currently missing on globaltensor pipe init; would be natural to expose alongside (1) above.
- pto-isa #622 ("`QK_PRELOAD=4` deadlock") — closed without code fix; the same `LocalSlotNum` chain likely contributed.

### Additional context

- ptoas binary in use: `/usr/local/bin/ptoas-bin/bin/ptoas` (mtime 2026-04-30 15:01, includes PTOAS PR #606's `pto.talloc_to_aiv`/`pto.talloc_to_aic`).
- mlir_combined Python bindings rebuilt locally from `hw-native-sys/PTOAS@c3a2395` to expose the talloc op classes.
- v2 kernel reproduces the manual's row_slice loop (`Vec_S0=32` × `kTileFactor=2`) so VEC UB stays under 192 KiB at `S1_TILE=256`; that part is independent of this issue.


---

## #119 [Bug] TROWSUM hangs on a2a3 hardware for INT32 tiles (FP32 path works with same shape/layout)

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/119
- Created: 2026-05-10T15:39:39Z
- Updated: 2026-05-28T09:37:02Z
- Closed: 2026-05-28T09:37:02Z

### Body

### Platform

a2a3 (Ascend 910 hardware)

### Runtime Variant

tensormap_and_ringbuffer

### Description

`TROWSUM` declared with `bfloat16_t`/`float` tile element types runs correctly on a2a3 hardware, but the **identical kernel structure** with `int32_t` tiles hangs indefinitely. The hang persists with the latest pto-isa main (`687af1a6`).

The pto-isa header `include/pto/npu/a2a3/TRowSum.hpp` (lines 94–133) declares INT32 support — the dtype dispatch falls into a dedicated `vadd`-based integer path. So this looks like the integer code path is reaching the device but not draining a pipeline event correctly, rather than being a missing template instantiation.

The same hang reproduces with both INT16 and INT32; we have not had a chance to test the simpler `int8_t` / unsigned variants.

### Steps to Reproduce

A direct, side-by-side comparison: same compaction trick (TLOAD a wide row × pad grid → TROWSUM along the pad axis → TSTORE column), only changing the element type.

```cpp
// FP32 — works:
//   TLOAD wide_tile (R x W_PAD, RowMajor, float) from [L, R, W_PAD] FP32 GM
//   TROWSUM(sum_tile, wide_tile, tmp_tile)   // sum_tile is R x 1 ColMajor float
//   TSTORE sum_tile to [L, R] FP32 GM (Layout::DN)
//
// INT32 — hangs:  swap `float` → `int32_t` everywhere, all other code identical.

using WWideShape  = pto::Shape<1, 1, 1, R, W_PAD>;
using WWideStride = pto::Stride<R * W_PAD, R * W_PAD, R * W_PAD, W_PAD, 1>;
using WWideG      = pto::GlobalTensor<float /* or int32_t */, WWideShape, WWideStride>;
using WWideTile   = pto::Tile<pto::TileType::Vec, float /* or int32_t */, R, W_PAD,
                              pto::BLayout::RowMajor, R, W_PAD>;
using WSumShape   = pto::Shape<1, 1, 1, R, 1>;
using WSumStride  = pto::Stride<1, 1, 1, 1, 1>;
using WSumG       = pto::GlobalTensor<float /* or int32_t */, WSumShape, WSumStride,
                                       pto::Layout::DN>;
using WSumTile    = pto::Tile<pto::TileType::Vec, float /* or int32_t */, R, 1,
                              pto::BLayout::ColMajor, R, 1>;

WWideTile wide_tile;
WSumTile  sum_tile;
WWideTile tmp_tile;     // same shape as src, per pto-isa convention
TASSIGN(wide_tile, 0x10000);
TASSIGN(sum_tile,  0x20000);
TASSIGN(tmp_tile,  0x21000);

// The pipeline scaffolding is identical for both dtypes:
TLOAD(wide_tile, win_g);
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
pipe_barrier(PIPE_V);
TROWSUM(sum_tile, wide_tile, tmp_tile);     // ← INT32 hangs here
pipe_barrier(PIPE_V);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
TSTORE(out_g, sum_tile);
```

Constants in our reproducer: `R = 32`, `W_PAD = IDX_PAD = 8`. Total wide tile size = `32 × 8 × sizeof(T) = 1 KB` for both FP32 and INT32.

The full kernel that exhibits the hang is checked in at:

`examples/workers/l3/ep_dispatch_distributed/kernels/aiv/ep_dispatch_kernel.cpp` in the simpler repo (function `kernel_entry`, the Phase 4 stage-out section). Today it uses FP32 TROWSUM for the `recv_w` channel and a scalar GM copy fallback for the `recv_idx` channel; switching the idx fallback to INT32 TROWSUM with otherwise identical code is what triggers the hang.

### Expected Behavior

INT32 TROWSUM compacts the wide row tile and `TSTORE`s the per-row sum to the destination GlobalTensor, identical in shape semantics to the FP32 path. Total runtime should be on the order of microseconds for a 32×8 reduction.

### Actual Behavior

Kernel hangs indefinitely. Task scheduler logs show bootstrap completing on both ranks, the dispatch task finishing, then the kernel sitting in the stage-out task until the watchdog kills it:

```
Resource phase: 1 case(s), pool=[12, 14], max_parallel=2
[scheduler] START standalone test_ep_dispatch_distributed (rt=tensormap_and_ringbuffer, dev=2) pid=... devices=[12, 14]
[taskqueue] task timed out (60s), automatically killed
```

Replacing only the INT32 TROWSUM with a scalar copy loop (or replacing the dtype with `float`) is sufficient to make the kernel return immediately and the test pass.

### Git Commit ID

`687af1a6bdd9ddd6a47a56cea773896d9d494e0f` (latest main as of report time)

### CANN Version

CANN 8.5.0

### Driver Version

25.3.rc1 (ascendhal 7.35.23)

### Host Platform

Linux aarch64 (5.10.0 kernel)

### Additional Context

- Same shape (`R × W_PAD = 32 × 8`), same layout (RowMajor src, ColMajor `R × 1` dst with `Layout::DN` GlobalTensor), same UB tile addresses, same pipe barriers — only the element type differs between the working FP32 path and the hanging INT32 path.
- TRowSum.hpp lines 94–133 contain a dedicated INT32 implementation that uses `vector_dup` + `vadd` + `pipe_barrier(PIPE_V)` + `pipe_barrier(PIPE_ALL)` followed by a final scalar reduction across 8 lanes. The internal `pipe_barrier(PIPE_ALL)` mid-implementation is unusual relative to the FP32 vcadd path and may be implicated.
- Workaround in our codebase: scalar GM copy of column 0 (`out[i] = wide[i * PAD]`). For our usage volume (≤ a few hundred INT32 stores in the final stage-out) the perf cost is negligible, but a working tile-level INT32 TROWSUM would be valuable for higher-volume reductions in production EP combine paths.

---

## #121 Add reverse instruction for mask-pattern TGATHER: mask-pattern TSCATTER

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/121
- Created: 2026-05-12T02:51:28Z
- Updated: 2026-05-25T01:13:50Z
- Closed: 2026-05-25T01:13:50Z

### Body

## Background

`TGATHER` currently has a mask-pattern overload (`include/pto/common/pto_instr.hpp:977`):

```cpp
template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(DstTileData &dst, SrcTileData &src, WaitEvents &...events);
```

Semantics (see the mask-pattern `TGather` in `include/pto/cpu/TGather.hpp`): iterate over `src`'s valid region; for each row, filter columns by `pto::MaskPattern` (e.g. `P0101` = take the 1st of every 2 elements); selected elements are **compacted** into `dst`'s contiguous storage (`didx++`), requiring `dst.GetValidCol() == DstTileData::Cols`. Effectively a fixed-pattern stream compaction.

`MaskPattern` is used **only** by `TGATHER` today (a repo-wide grep for `maskPattern` / `#pto.mask_pattern` only hits the TGATHER intrinsic, `type.hpp`, the per-backend `TGather` impls, the costmodel, and the tgather docs).

## Request

Add a symmetric reverse instruction: scatter a compacted `src` back into the selected column positions of a wider `dst` according to a `MaskPattern` (deposit / expand-by-pattern), i.e. the inverse of mask-pattern `TGATHER`. Proposed shape:

```cpp
template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(DstTileData &dst, SrcTileData &src, WaitEvents &...events);
```

Reference semantics (CPU-SIM):

```text
sidx = 0
for r in [0, src.ValidRow):
  for c in [0, dst.ValidCol):
    if MaskSelect(maskPattern, c):
      dst[r, c] = src.flat[sidx]; sidx++
    # behavior for non-selected dst positions (zero / keep / caller's responsibility) must be specified
```

## Current workaround

Use the index-based `TSCATTER(dst, src, idx)` (`include/pto/common/pto_instr.hpp:1680`) and manually build an index tile that expands the `MaskPattern` into flattened destination offsets; or `TSEL` + a mask tile. Both require extra data prep and are asymmetric with the forward op.

## Open questions

- Semantics for non-selected positions: zero / leave untouched / caller-managed (affects whether a preceding `TDUP` is needed).
- Whether dtype / element-width constraints should mirror mask-pattern `TGATHER` (A2/A3 limited to 2/4B; A5 includes 1B and fp8).
- Lowering feasibility on the three backends (A2/A3, A5, CPU-SIM) — is there a hardware pattern-deposit primitive, or does it lower to an index scatter?
- Whether the GM-side `MGATHER` / `MSCATTER` need a symmetric form too; scope this to the tile side for now.


---

## #122 CPU backend TPUT_IMPL appears to copy in the wrong direction

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/122
- Created: 2026-05-12T07:32:09Z
- Updated: 2026-05-14T06:42:52Z
- Closed: 2026-05-14T06:42:52Z

### Body

# CPU backend `TPUT_IMPL` appears to copy in the wrong direction

## Summary

The CPU simulation implementation of `pto::comm::TPUT` appears to reverse the source and destination operands.

The public `TPUT(dst, src, tile)` API and the a2a3 backend implement a remote write:

```text
srcGlobalData -> stagingTileData -> dstGlobalData
```

However, the CPU backend currently calls `Copy_Data(src, dst)`, while `Copy_Data(dstTensor, srcTensor)` assigns `dst = src`. This makes CPU `TPUT(dst, src, ...)` behave like `src = dst`.

## Relevant code

In `include/pto/comm/pto_comm_inst.hpp`, `TPUT` forwards arguments as `(dstGlobalData, srcGlobalData, stagingTileData)`:

```cpp
::pto::comm::TPUT_IMPL<GlobalDstData, GlobalSrcData, TileData, atomicType>(
    dstGlobalData, srcGlobalData, stagingTileData);
```

In `include/pto/cpu/comm/TGet.hpp`, `Copy_Data` has destination-first semantics:

```cpp
template <typename GlobalDstData, typename GlobalSrcData, AtomicType atomicType = AtomicType::AtomicNone>
PTO_INTERNAL void Copy_Data(GlobalDstData &dstTensor, GlobalSrcData &srcTensor)
{
    typename GlobalDstData::DType *dst = dstTensor.data();
    typename GlobalSrcData::DType *src = srcTensor.data();
    ...
    dst[index] = src[index];
}
```

But in `include/pto/cpu/comm/TPut.hpp`, CPU `TPUT_IMPL` calls it with reversed operands:

```cpp
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, AtomicType atomicType>
PTO_INTERNAL void TPUT_IMPL(GlobalDstData &dst, GlobalSrcData &src, TileData &src1)
{
    Copy_Data(src, dst);
}

template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TPUT_IMPL(GlobalDstData &dst, GlobalSrcData &src, TileData &ping, TileData &pong)
{
    Copy_Data(src, dst);
}

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_IMPL(GlobalDstData &dst, GlobalSrcData &src, const AsyncSession &session)
{
    Copy_Data(src, dst);
    return AsyncEvent(0, engine);
}
```

The a2a3 backend in `include/pto/comm/a2a3/TPut.hpp` uses the expected direction:

```cpp
TLOAD(stagingTileData, srcGlobalData);
...
TSTORE_IMPL<TileData, GlobalDstData, atomicType>(dstGlobalData, stagingTileData);
```

## Observed behavior

In a CPU-simulated distributed allreduce example, each rank calls:

```cpp
pto::comm::TPUT(remote_slot, partial_local, staging_tile);
```

Expected behavior:

```text
remote_slot = partial_local
```

Observed behavior with the CPU backend:

```text
partial_local appears to be overwritten/read from remote_slot instead
```

As a result, the simulated allreduce only sees the local contribution and fails the golden check. Replacing CPU-sim `TPUT` with an explicit elementwise copy:

```cpp
for (int i = 0; i < n; ++i) {
    remote_slot_ptr[i] = partial_local_ptr[i];
}
```

makes the same test pass with zero diff. The onboard/a2a3 path using the normal `TPUT` implementation already passes.

## Expected fix

The CPU backend likely should call `Copy_Data(dst, src)` instead:

```cpp
Copy_Data(dst, src);
```

This should be applied consistently to:

- `TPUT_IMPL(dst, src, tile)`
- `TPUT_IMPL(dst, src, ping, pong)`
- `TPUT_ASYNC_IMPL(dst, src, session)`

Please confirm whether this is indeed a CPU backend bug or whether CPU `TPUT` intentionally has different operand semantics from the public `TPUT(dst, src, ...)` API.


---

## #127 TPipe destructor leaks FFTS free signals on a2a3/a5 (regression in 687af1a6)

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/127
- Created: 2026-05-18T01:35:10Z
- Updated: 2026-05-22T11:00:30Z
- Closed: 2026-05-22T11:00:30Z

### Body

## Summary

Commit **687af1a6 "Optimize reverse dependencies with sync periods"** introduces a signal-count imbalance in `TPipe<>` on both **a2a3** (`include/pto/npu/a2a3/TPush.hpp`) and **a5** (`include/pto/npu/a5/TPush.hpp`). For any pipeline with `SlotNum > SyncPeriod` (e.g. the common `SlotNum=8, SyncPeriod=4` case), each kernel invocation leaks `SlotNum / SyncPeriod` FFTS `free` signals into the cross-core flag register. On NPU hardware (A2/A3) this manifests as **CANN runtime error 507018 (`ACL_ERROR_RT_AICPU_EXCEPTION`)** after one or more invocations.

The bug is still present on `main` at `d779cd01` — no commits between `687af1a6..d779cd01` modify the relevant headers.

## Reproduction

Downstream PyPTO test (NPU A2/A3, real hardware):

```bash
pytest tests/st/runtime/test_qwen3_decode_scope3_mixed.py \
       --platform a2a3 --device <N> --save-kernels \
       --pto-isa-commit=d779cd0
```

Result on `d779cd01`:

```
RuntimeError: run_prepared failed with code 507018
[ERROR] aclrtSynchronizeStreamWithTimeout (AICPU) failed: 507018
```

Reverting to the prior commit `fcc6f420` makes the test pass. Re-applying the patch in §Proposed Fix on top of `d779cd01` also makes the test pass.

The failing kernels (`scope3_incore_{1,4,5}`) use:

```cpp
TPipe<0, Direction::DIR_C2V, /*SlotSize=*/4096, /*SlotNum=*/8,
      /*LocalSlotNum=*/8, /*IsNoSplit=*/false>
// => SyncPeriod = SlotNum / 2 = 4
```

## Root Cause (numerical)

Consider a producer/consumer loop of `T` tiles with `SlotNum = 8, SyncPeriod = 4`:

| Stage | Code site | Signals (free pulses to `FlagID+1`) |
| --- | --- | --- |
| Constructor (consumer) | `TPipe::TPipe` runs `cons.free()` × `SyncPeriod` | **+4** |
| In-loop consumer | `TPOP_IMPL` calls `cons.free()` iff `shouldNotifyFree(i)` = `((i+1) % SP) == 0` | **+T / 4** |
| In-loop producer | `TPUSH_IMPL` calls `prod.allocate()` iff `shouldWaitFree(i)` = `(i ≥ SlotNum) && (i % SP) == 0` | **−(T − 8) / 4** |
| Destructor (producer) | `TPipe::~TPipe` runs `prod.allocate()` × `SyncPeriod` | **−4** |

Net residual on `FlagID+1` per kernel invocation:

```
(SyncPeriod + T/SP) - ((T - SlotNum)/SP + SyncPeriod)
= SlotNum / SyncPeriod
= 8 / 4
= 2  (leaked free signals)
```

The asymmetry comes from `shouldWaitFree` skipping the first `SlotNum` tiles (the "startup protection") while `shouldNotifyFree` does **not** skip the matching tail; the destructor doesn't compensate either.

Relevant code (a2a3, `include/pto/npu/a2a3/TPush.hpp`):

```cpp
// L53-72
PTO_INTERNAL static bool shouldWaitFree(uint32_t tileIndex) {
    if constexpr (SlotNum == 1) return true;
    else {
        if (tileIndex < SlotNum) return false;   // <-- skips first SlotNum
        return (tileIndex % SyncPeriod) == 0;
    }
}
PTO_INTERNAL static bool shouldNotifyFree(uint32_t tileIndex) {
    if constexpr (SlotNum == 1) return true;
    else return ((tileIndex + 1) % SyncPeriod) == 0;   // <-- no matching skip
}

// L444-450
PTO_INTERNAL ~TPipe() {
    for (uint32_t i = 0; i < SyncPeriod; ++i) {
        prod.allocate();    // <-- drains SyncPeriod, but needs SyncPeriod + SlotNum/SyncPeriod
    }
}
```

The a5 backend has the identical pattern at `include/pto/npu/a5/TPush.hpp:611-621`.

## Proposed Fix

Increase the destructor drain count to balance the constructor + in-loop pulse counts:

```cpp
PTO_INTERNAL ~TPipe()
{
    constexpr uint32_t kSkippedBatches = (SlotNum > 1) ? (SlotNum / SyncPeriod) : 0;
    constexpr uint32_t kDestructorWaits = SyncPeriod + kSkippedBatches;
    for (uint32_t i = 0; i < kDestructorWaits; ++i) {
        prod.allocate();
    }
}
```

Verified on real NPU hardware: with this patch applied on top of `d779cd01`, the failing PyPTO test passes (`1 passed in 9.77s`).

The same change should be mirrored in `include/pto/npu/a5/TPush.hpp:~TPipe()` (same algebra, same template pattern using the split-axis `allocate<>` overload).

## Why the CV regression test didn't catch this

The new test in `tests/npu/a2a3/src/st/testcase/tpushpop_cv/tpushpop_cv_kernel.cpp` exercises only `FIFO_DEPTH = 1` (i.e. `SlotNum == 1`), which short-circuits both `shouldWaitFree` and `shouldNotifyFree` to `true` and trivially preserves balance. The leak only appears for `SlotNum ≥ 4`.

Adding a `SlotNum=8, SyncPeriod=4` variant to that test (and asserting absence of residual FFTS counts after a few invocations) would catch this class of bug.

## Environment

- CANN: 9.0.0
- Platform: A2/A3 NPU, single device
- pto-isa: `d779cd01` (also reproduces on `687af1a6`)
- Downstream: PyPTO `tests/st/runtime/test_qwen3_decode_scope3_mixed.py`


---

## #128 [Bug] CPU SIM ST CI build failure: ambiguous std::exp(half) overload in ElementOp.h

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/128
- Created: 2026-05-18T15:43:07Z
- Updated: 2026-05-21T11:59:07Z
- Closed: 2026-05-21T11:59:07Z

### Body

### What happened?

The `CPU SIM full ST` job fails when building `tests/cpu/st`. The first failing target is:

```
FAILED: testcase/hashfind/CMakeFiles/hashfind.dir/hashfind_kernel.cpp.o
```

Root cause is at [include/pto/cpu/ElementOp.h:277](https://github.com/hw-native-sys/pto-isa/blob/main/include/pto/cpu/ElementOp.h#L277):

```cpp
#if defined(__GNUC__) && !defined(__clang__)
template <>
struct ElementOpCal<half, ElementOp::OP_EXPDIF> {
    static void apply(half &dst, const half &src0, const half &src1)
    {
        dst = std::exp(src0 - src1);   // <-- error
    }
};
#endif
```

GCC 13 reports an ambiguous overload:

```
error: call of overloaded 'exp(half)' is ambiguous
note: candidate: 'double exp(double)'
note: candidate: 'constexpr float std::exp(float)'
note: candidate: 'constexpr long double std::exp(long double)'
```

`half` (`_Float16`) implicitly converts to `float`, `double`, and `long double`, so all three `std::exp` overloads match and GCC 13 cannot pick one. This cascades into compilation failures across multiple testcases (`hashfind`, `mgather`, etc.).

**Affected CI runs (same root cause):**
- PR #117 / run [26042893738](https://github.com/hw-native-sys/pto-isa/actions/runs/26042893738/job/76559448734)
- PR #116 / run [26026454466](https://github.com/hw-native-sys/pto-isa/actions/runs/26026454466/job/76501130922)

### Expected behavior

`CPU SIM full ST` should build successfully under GCC 13, with the `OP_EXPDIF` half-precision specialization compiling without overload ambiguity.

### How to reproduce

1. Check out `main` (commit `9562e76b`).
2. On a GCC 13 / Ubuntu 24.04 environment, run:
   ```bash
   bash tests/run_cpu_tests.sh --generator Ninja --build-folder build/cpu_st_ci
   ```
3. Compilation of `testcase/hashfind/hashfind_kernel.cpp.o` fails with `error: call of overloaded 'exp(half)' is ambiguous`.

### Environment & version info

- Repo commit: `9562e76bff3adcfdb5ba3f726f5e54c940f3a24a` (main)
- Compiler: GCC 13.3.0
- OS: Ubuntu 24.04.4 LTS (GitHub-hosted runner `ubuntu-24.04`)
- C++ standard: `-std=c++20`
- Affected job: `CPU SIM full ST` → `Run CPU SIM ST suite`

### Other

**Suggested fix:** make the argument type explicit so overload resolution is unambiguous, e.g.:

```cpp
dst = static_cast<half>(std::exp(static_cast<float>(src0 - src1)));
```

Alternatively, provide a `half`-specific `exp` implementation.

---

## #133 CPU TRSQRT tmp overload is misnamed

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/133
- Created: 2026-05-20T03:27:02Z
- Updated: 2026-05-21T02:01:14Z
- Closed: 2026-05-21T02:01:14Z

### Body

## Problem

The public TRSQRT wrapper has an overload for `TRSQRT(dst, src, tmp, ...)` and calls `TRSQRT_IMPL<PrecisionType>(dst, src, tmp)`.

In `include/pto/cpu/TRSqrt.hpp`, the CPU implementation defines the three-argument overload as `TSQRT_IMPL` instead of `TRSQRT_IMPL`. As a result, CPU simulation builds fail when generated code uses the three-argument TRSQRT form.

## Failure Mode

Compilation reports that `TRSQRT_IMPL<...>(dst, src, tmp)` has no matching function, with only the two-argument candidate available.

## Expected Behavior

The CPU TRSQRT implementation should expose the same three-argument `TRSQRT_IMPL(dst, src, tmp)` overload that the common wrapper and NPU implementations expect.

## Scope

This is limited to the CPU simulation shim. The a2a3 and a5 NPU headers already define three-argument `TRSQRT_IMPL` overloads.

---

## #139 [Bug] CPU SIM full ST build failure: tscatter_kernel.cpp calls 2-arg TSCATTER<MaskPattern>(dst, src) but only 3-arg overload exists

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/139
- Created: 2026-05-22T03:41:08Z
- Updated: 2026-05-22T04:26:55Z
- Closed: 2026-05-22T04:26:55Z

### Body

### What happened?

The `CPU SIM full ST` job fails when building `tests/cpu/st`. The failing target is:

```
FAILED: testcase/tscatter/CMakeFiles/tscatter.dir/tscatter_kernel.cpp.o
```

[tests/cpu/st/testcase/tscatter/tscatter_kernel.cpp:83](https://github.com/hw-native-sys/pto-isa/blob/main/tests/cpu/st/testcase/tscatter/tscatter_kernel.cpp#L83) calls a 2-arg form of `TSCATTER` parameterized only on a `pto::MaskPattern`:

```cpp
TSCATTER<maskPattern>(dstTile, srcTile);
```

However the only declaration visible at [include/pto/common/pto_instr.hpp:1754](https://github.com/hw-native-sys/pto-isa/blob/main/include/pto/common/pto_instr.hpp#L1754) is the 3-arg form:

```cpp
template <typename TileDataD, typename TileDataS, typename TileDataI, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(TileDataD &dst, TileDataS &src, TileDataI &indexes, WaitEvents &...events);
```

so GCC errors out with:

```
tests/cpu/st/testcase/tscatter/tscatter_kernel.cpp:83:26:
  error: no matching function for call to 'TSCATTER<pto::MaskPattern::P0101>(DstTileData&, SrcTileData&)'
note: candidate expects at least 3 arguments, 2 provided
```

The same error repeats for every mask pattern × shape combination: `P0101`, `P1010`, `P0001`, `P0010`, `P0100`, `P1000`, `P1111`.

This breakage is on `main` and is not introduced by any single PR. It surfaces in every PR run that exercises `CPU SIM full ST` (e.g. PR #123 / run [26218721403](https://github.com/hw-native-sys/pto-isa/actions/runs/26218721403/job/77153773315)). Recent main-branch `CI` workflow runs (2026-05-20 ~ 2026-05-21) consistently show `CI: failure`.

### Expected behavior

`CPU SIM full ST` should build successfully on `main`. Either:

- `tscatter_kernel.cpp` should be updated to use the existing 3-arg `TSCATTER(dst, src, indexes, ...)` API (providing an `indexes` tile); **or**
- `pto_instr.hpp` should expose a 2-arg `TSCATTER<MaskPattern>(dst, src, ...)` overload mirroring the existing mask-pattern `TGATHER<MaskPattern>` form at `include/pto/common/pto_instr.hpp:977`.

The design intent (mask-pattern variant) is tracked by #121 — once that lands, the test call site becomes valid. Until then, either side of the contract needs to align so `main` stops red-lining.

### How to reproduce

1. Check out `main` (commit `9dc91fee`).
2. On a GCC 13 / Ubuntu 24.04 environment, run:
   ```bash
   bash tests/run_cpu_tests.sh --generator Ninja --build-folder build/cpu_st_ci
   ```
3. Compilation of `testcase/tscatter/tscatter_kernel.cpp.o` fails with `error: no matching function for call to 'TSCATTER<pto::MaskPattern::P0101>(DstTileData&, SrcTileData&)'`.

### Environment & version info

- Repo commit: `9dc91fee409c53476811f4acbc865e40025020a8` (main, observed 2026-05-21)
- Compiler: GCC 13 (Ubuntu 24.04 runner)
- OS: Ubuntu 24.04.4 LTS (GitHub-hosted runner `ubuntu-24.04`)
- C++ standard: `-std=c++20`
- Affected job: `CPU SIM full ST` → `Run CPU SIM ST suite`

### Other

Related: #121 (design + feature request for mask-pattern `TSCATTER`).

**Suggested directions** (pick one — defer to the ISA design notes in `docs/isa/comm/TSCATTER.md` / `TSCATTER_zh.md` first):

1. Add a 2-arg `TSCATTER<MaskPattern>(dst, src, ...)` overload to `include/pto/common/pto_instr.hpp` (likely as part of resolving #121). Use a non-type `MaskPattern` template parameter with SFINAE so it coexists with the 3-arg index-based form.
2. Update `tests/cpu/st/testcase/tscatter/tscatter_kernel.cpp` to call the existing 3-arg `TSCATTER(dst, src, indexes, ...)` API and supply an `indexes` tile.

---

## #143 [Spec] Make Rv=0 / Cv=0 a normative no-op for store-class instructions, and document odd-axis padding rules

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/143
- Created: 2026-05-27T05:59:54Z
- Updated: 2026-06-01T02:06:40Z
- Closed: 2026-06-01T02:06:40Z

### Body

### Summary

`docs/isa/programming-model/tiles-and-valid-regions.md` already says that the iteration domain of any instruction is `0 <= i < dst.Rv, 0 <= j < dst.Cv`, which implies that `Rv = 0` (or `Cv = 0`) makes the domain empty and the instruction semantically a no-op.

However, this is currently a derived consequence rather than a normative statement, and several practical questions are not addressed:

1. For **store-class instructions** (`TSTORE`, `TMOV` writing to a different memory space, `TASSEMBLE`-like ops, `TPUSH_TO_AIC`, `TPUSH_TO_AIV`), is the "no element written" guarantee binding on the hardware, or only on the math model? In other words: when `dst.Rv = 0`, is the hardware required not to issue any DRAM/SRAM write transaction, or could it speculatively write garbage that happens to be ignored by the read-back logic?

2. For **the V2C/C2V cross-pipe handoff** (`TPUSH_TO_AIC`, `TPOP_FROM_AIV`, etc.), what is the semantics when the pushed tile has `Rv = 0`? Is the slot "consumed" by the AIC side (pop must succeed) or is the push silently dropped?

3. For **odd-axis tile shapes**, is there a documented physical alignment requirement on `rows` / `cols` beyond what fractal / `SLayout` already imposes? Specifically: can a `Vec` tile have `rows = 16, valid_row = 5` legally, and does any ISA op behave differently based on the gap?

### Motivation / Use Case

PyPTO and PTOAS are coordinating to support odd-axis tile shapes (see hw-native-sys/pypto#1031 and hw-native-sys/PTOAS#708). The intended lowering pattern relies on the "physical even, valid odd" tile shape plus a `valid_row = 0` per-lane no-op contract under UP_DOWN split.

Concrete kernel pattern (Qwen3-14B decode, trying to fuse the online-softmax tail into the mixed cube+vec `fa_fused` root):

- dst tile: `[16, 128]` physical, lower lane gets `valid_row = 0` after UP_DOWN split
- chain: `subview [0:5]` → `tcvt fp32→bf16` → `tassemble` (stores one row of 640 cols to GM `attn_out[b, head_offset]`)

We need to guarantee that the lower lane's `tassemble` does **not** write garbage to GM (which would overwrite the upper lane's correct output). If the hardware honors `dst.Rv = 0` as "no STU/MTE issued", PTOAS can simply lower the op as-is and rely on hardware. If not, PTOAS must emit an explicit predicate or skip codegen entirely.

Without a normative spec answer here, PTOAS implementers have to choose conservatively (always skip codegen on Rv=0), which is fine but needs to be documented as the contract.

### Proposed Spec Additions

Add to `docs/isa/programming-model/tiles-and-valid-regions.md` a new section, e.g. "Empty Valid Regions":

> **Empty Domain (Rv = 0 or Cv = 0)**: When the destination tile's valid region is empty (`Rv = 0` or `Cv = 0`), the iteration domain is empty and no `dst[i, j]` is defined. For **destructive** ops (those that produce a new value in `dst`), the instruction is a normative no-op: no architectural state outside `dst`'s storage is modified.
>
> For **store-class** instructions (`TSTORE`, cross-space `TMOV`, `TASSEMBLE`-like ops, `TPUSH_TO_AIC`, `TPUSH_TO_AIV`), the empty domain additionally implies:
>
>   - No memory transaction is issued to the destination address space.
>   - For pipe handoffs (`TPUSH_*` / `TPOP_*`), the slot is **not** consumed; the matching `TPOP_*` must not be issued by the consumer side for this iteration.
>
> Hardware implementations MUST honor this contract. Compilers MAY additionally elide such instructions at codegen time as an optimization.

Add to each store-class instruction page an explicit "Empty Domain" sub-section reinforcing the above.

Add to `docs/isa/programming-model/tiles-and-valid-regions.md` (or a new "Padding Rules" page):

> **Physical vs. Valid Dimensions**: A tile's physical `rows` and `cols` may legally exceed `valid_row` / `valid_col`. The physical dimensions must respect any per-`SLayout` / `Fractal` alignment requirements (see Layout Reference), but the valid dimensions are not constrained beyond `0 <= valid <= physical`. In particular, odd valid dimensions are legal when the physical dimension is rounded up to the next aligned size.

### Acceptance Criteria

- The "Empty Domain" section is added with the wording above (or equivalent normative content).
- At least `TSTORE`, `TMOV`, `TASSEMBLE`, `TPUSH_TO_AIC`, `TPUSH_TO_AIV`, `TPOP_FROM_AIV`, `TPOP_FROM_AIC` instruction pages link to or quote the new section.
- The "Physical vs. Valid Dimensions" rule is documented.

### Related

- hw-native-sys/pypto#1031 (PyPTO frontend `SplitVectorKernel`)
- hw-native-sys/PTOAS#708 (PTOAS verifier + codegen)

---

## #146 TSCATTER index form returns wrong results for 2-byte dtypes (fp16/bf16/int16): values land in the wrong row

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/146
- Created: 2026-05-28T08:53:12Z
- Updated: 2026-06-01T02:56:23Z
- Closed: 2026-06-01T02:56:23Z

### Body

## Summary

Index-form `TSCATTER(dst, src, idx)` produces **incorrect results for all 2-byte element types** — `fp16`, `bf16`, `int16` (paired with `int16` indices) — on the **A2/A3** backend. The 4-byte paths (`fp32` and `int32` with `int32` indices) are correct, and the mask-form scatter is correct.

The scattered **values are intact** (the expected magnitudes all appear in the output), but they are written to the **wrong destination row**: in every mismatch the *column* index is preserved while the *row* offset is miscomputed. Since the dtype `static_assert`s in `TScatter.hpp` accept these types, this is a **silent wrong-result bug** — no compile error, no runtime crash.

## Affected version

pto-isa commit `0f171f7327472e79c8def3b5f5b46b692c48b117` (`0f171f7`).

Context: we bumped our pin from `2c607938` to `0f171f7` to pick up the mask-form 2-arg `TSCATTER<maskPattern>(dst, src)` overload. The 2-byte index-form cases were passing before the bump; they regressed after moving to `0f171f7`.

## Op / location

- `include/pto/npu/a2a3/TScatter.hpp` — index-form `TSCATTER` (`TScatterImpl` / `TSCATTER_IMPL` path; **not** the mask form).
- The type guard at `TSCATTER_IMPL` accepts `(sizeof(TD) == 2 && sizeof(TI) == 2)`, i.e. fp16/bf16/int16 dst with int16 idx are explicitly allowed — so the wrong output is returned silently.

## Symptom

Per-row column scatter `out[b, index[b, k]] = val[b, k]`. Shapes: dst `[16, 32]`, src/idx `[16, 16]`. Indices passed to TSCATTER are flat per-element offsets into dst (`b * dst_cols + col`).

| dst / src | idx | result |
| --------- | --- | ------ |
| fp32 | int32 | PASS |
| int32 | int32 | PASS |
| **fp16** | **int16** | **FAIL** |
| **bf16** | **int16** | **FAIL** |
| **int16** | **int16** | **FAIL** |
| mask-form (fp32) | — | PASS |

In every mismatch the **column is preserved and only the row is wrong** (flat index = `row * 32 + col`):

- **fp16** (33/512 mismatched): `val[14,0] = 225` should write to dst flat **462** (row 14, col 14). Instead flat 462 keeps the base sentinel (`-29`) and `225` appears at flat **494** (row 15, col 14) — column 14 preserved, **row 14 → 15**.
- **bf16 / int16** (49/512 mismatched): `val[14,0] = 225` again expected at flat **462** (row 14, col 14), but appears at flat **14** (row 0, col 14) — column 14 preserved, **row 14 → 0**.

The set of wrong rows differs between fp16 and bf16 even though the inputs and index tensors are identical, so this is **not** a simple fixed shift — it points to a defect in the destination row-offset / stride computation of the 2-byte index-form path.

## Expected

Index-form `TSCATTER` for 2-byte dst (fp16/bf16/int16) with int16 indices should match the 4-byte behaviour: `dst[flat_idx[i,j]] = src[i,j]` for all valid `(i,j)`, leaving rows untouched by any index at their original (DPS-preserved) values.

## Repro

Reproduced through the PyPTO system test `tests/st/runtime/ops/test_scatter.py::TestScatterIndexForm` on platform `a2a3`, with pto-isa pinned at `0f171f7`. `fp32`/`int32` pass; `fp16`/`bf16`/`int16` fail with golden mismatches as above.

<details>
<summary>Failing output excerpt (fp16)</summary>

```
test_scatter_fp16[a2a3] FAILED
AssertionError: Output 'output' does not match golden.
  Mismatched elements: 33/512
  rtol=1e-05, atol=1e-05
  First mismatches:
    [462] actual=-29.0, expected=225.0   # row14,col14: stayed base, value 225 missing
    [463] actual=-30.0, expected=226.0
    ...
    [494] actual=225.0, expected=-30.0   # row15,col14: got 225 (belongs to row14)
    [495] actual=226.0, expected=241.0
```

bf16 / int16 show the same shape with the misplaced row being 0 instead of 15.
</details>

---

## #147 [Bug] Heap-ring deadlock (PTO2_ERROR_HEAP_RING_DEADLOCK) introduced by 687af1a6 in Qwen3-14B fused decode

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/147
- Created: 2026-05-28T11:51:55Z
- Updated: 2026-06-02T01:04:58Z
- Closed: 2026-06-02T01:04:58Z

### Body

## Summary

A pto-isa change at commit **`687af1a6`** ("Optimize reverse dependencies with sync periods") causes the PTO2 runtime to **deadlock the heap ring** (`PTO2_ERROR_HEAP_RING_DEADLOCK`) when executing the Qwen3-14B **optimized fused decode** on a2a3. The immediate parent commit **`fcc6f420`** runs the exact same workload correctly. Verified by an A/B test where **only `PTO_ISA_ROOT` was changed** between the two runs.

## Bisect (single-commit boundary)

| pto-isa commit | result |
|---|---|
| `fcc6f420` "Tune A3 TROWPROD pipeline synchronization" (parent) | ✅ decode runs, generates output |
| `687af1a6` "Optimize reverse dependencies with sync periods" | ❌ decode **deadlocks** |

All other components (pypto, simpler, pypto-lib, ptoas, ring sizes, device, prompt) were identical across the two runs; only `PTO_ISA_ROOT` differed.

## Symptom

Prefill completes normally; the **per-token decode step** (40-layer fused Qwen3-14B) deadlocks — the AICPU orchestrator spins waiting for heap-ring space that never frees:

```
[ERROR] run: [device_runner.cpp] Stream sync timeout: stream=AICPU timeout_ms=2000 ...   # host-side, stock 2s timeout
[ERROR] validate_runtime_impl: PTO2 runtime failed: orch_error_code=2 sched_error_code=0 runtime_status=-2
RuntimeError: run_prepared failed with code 507018
```

`orch_error_code=2` = `PTO2_ERROR_HEAP_RING_DEADLOCK` (see `runtime/.../common/pto_runtime_status.h`). With the stock 2000 ms AICPU stream-sync timeout it first surfaces as a host-side `Stream sync timeout` (`507046`); raising the timeout lets the device run longer and then report the deadlock directly (`507017`/`507018`).

## Reproduction

- pypto-lib @ `3834f3d` (Qwen3-14B), pypto @ `b0d1d49a`, simpler @ `a94d5140`, ptoas `0.41`, a2a3 NPU, CANN 9.0.0
- Workload: the optimized non-L3 `models/qwen3/14b/decode_layer.py` (mix-fuse cube+vec, scope-3 flat `pl.spmd`) driven via pypto-serving:

```bash
PTO_ISA_ROOT=<pto-isa checkout> \
PTO2_RING_HEAP=2147483648 PTO2_RING_TASK_WINDOW=262144 PTO2_RING_DEP_POOL=262144 \
python examples/model/qwen3_14b/npu_generate.py \
  --model-dir <Qwen3-14B> --prompt 'Huawei is' \
  --platform a2a3 --max-seq-len 128 --max-new-tokens 16
```

- `PTO_ISA_ROOT` at **`687af1a6`** → deadlock (`507018` / heap-ring).
- `PTO_ISA_ROOT` at **`fcc6f420`** → runs, generates `token_ids [264, 8453, 2813, 13, 576, 2813, 374, 7407, 304, 279, 3639, 4180, 13, 576, 2813, 374]`.

(Single-layer decode runs fine on `687af1a6`; the deadlock only appears at full 40-layer depth.)

## Notes

- The deadlock is **not** avoidable via runtime tuning: enlarging the rings OOMs the device (the 14B weights nearly fill the card), smaller rings still deadlock, and `PTO2_ORCH_TO_SCHED` / `PTO2_READY_QUEUE_SHARDS` do not help.
- `687af1a6` changes reverse-dependency / sync-period handling, which plausibly alters the emitted task dependency/sync structure so the heap ring can no longer be reclaimed for this graph — a good place to start.


---

## #149 [Bug] A2A3/A5 pushVec2GMFiFo TILE_NO_SPLIT path lacks inactive-lane guard → AIV1 clobbers AIV0 store (silent all-zero in mixed C/V NONE scopes)

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/149
- Created: 2026-05-29T01:08:04Z
- Updated: 2026-05-29T05:12:54Z
- Closed: 2026-05-29T05:12:54Z

### Body

### Summary

On A2A3 (and A5), the device-side vec→cube producer intrinsic `pushVec2GMFiFo` has **no inactive-lane guard for the `TILE_NO_SPLIT` path**. Under `SplitMode.NONE` in a mixed Cube/Vec scope, hardware runs the Vec kernel on **both** sub-cores (AIV0 + AIV1). For `TILE_NO_SPLIT` the device stores the full static tile at `subAIVOffset = 0` for *both* lanes, so AIV1 (the inactive replay lane, whose tile carries runtime `valid_shape=[0,0]`) writes a full tile of **zeros** into the same v2c slot that AIV0 wrote its real result to — clobbering it. The cube then pops zeros → **silent all-zero output** with a clean compile and no assert.

The CPU sim does **not** surface this because it has an inactive-lane guard that the device path is missing (see "Sim vs device divergence" below). This is the device-side root cause confirmed for `hw-native-sys/pypto#1525` (a board owner reproduced it on a real a2a3 board and handed it off to pto-isa).

### Affected code

Producer (the clobbering store), `TILE_NO_SPLIT` branch sets `subAIVOffset = 0` for both lanes and unconditionally `TSTORE_IMPL`s, with no `get_subblockid()` guard:

- A2A3: `include/pto/npu/a2a3/TPush.hpp` — `pushVec2GMFiFo`, `TILE_NO_SPLIT` branch around **line 208-210** ("single writer, no offset needed").
- A5: `include/pto/npu/a5/TPush.hpp` — `pushVec2GMFiFo`, `TILE_NO_SPLIT` branch around **line 301-302** (same "single writer, no offset needed" comment). A5's split-aware sync suppresses the *second Vec sync*, but the no-split **store** itself is likewise un-guarded — please confirm whether A5 is actually affected or saved by its sync gating.

Consumer side has the symmetric no-guard `TILE_NO_SPLIT` branch (`subAIVOffset = 0`, "single reader") at `a2a3/TPush.hpp:380` and `a5/TPush.hpp:540` — read-side is harmless but worth keeping in mind for any fix.

Confirmed live at pto-isa `0f171f73` (`grep -r IsInactiveNoSplitVecLane include/pto/npu/` → no match). The fix commit `491d7f23` referenced on pypto#1525 does **not** exist in this repo.

### Mechanism (per the a2a3 board investigation on pypto#1525)

1. pypto's `SplitVectorKernel` correctly dual-dispatches the AIV under NONE: AIV0 does the real work; AIV1 is an empty `valid_shape=[0,0]` replay.
2. AIV1's `tpush_to_aic` replay is **load-bearing** — it participates in the FFTS `CV_CORES_SYNC` barrier (fixed cube+AIV0+AIV1). Dropping it in the pass → cube hangs `rtStreamSynchronize (AICPU) failed: 507018`.
3. The device `pushVec2GMFiFo` `TILE_NO_SPLIT` branch ignores runtime `valid_shape` and stores the full static tile at `subAIVOffset = 0` for both lanes → AIV1 overwrites AIV0 with zeros.

### Why the obvious fixes do not work (already tried)

- **Skip only the store on the inactive lane, keep `record()`** → also hangs `507018`: v2c `record()` syncs on `PIPE_MTE3` (the store pipe, `TPush.hpp:165`), so an inactive lane that records without storing deadlocks the cube.
- **Drop AIV1's push in the pypto pass** → hangs `507018` (FFTS barrier needs AIV1's notify).
- The NONE-mode v2c slot is sized for **one** tile, so the inactive lane cannot just retarget to a non-conflicting scratch offset.

So this needs an FFTS-aware device fix, not a one-line guard.

### Suggested direction

Mirror the CPU-sim semantics on the device in an FFTS-safe way, e.g. size the NONE-mode v2c slot to hold **two** tiles and route the inactive lane's store to the second region (cube reads the first), so the `PIPE_MTE3`/`CV_CORES_SYNC` sync still sees a real store from AIV1 but there is no clobber. This is the conceptual target; the exact shape needs device iteration. (See also the closed design-clarification #83, where the agreed model was "AIV1 strips the store / side-effecting ops" — this issue is that the device code does not actually do so, plus the constraint that a naive strip deadlocks.)

### Reproducer

~40-line pypto program (single `CORE_GROUP`, `SplitMode.NONE`, cube → vec cast → cube):

```python
import pypto.language as pl
M = K = N = 64
@pl.program
class Repro:
    @pl.function(type=pl.FunctionType.Opaque)
    def fused(self, a: pl.Tensor[[M, K], pl.BF16], w: pl.Tensor[[K, N], pl.BF16],
              w2: pl.Tensor[[K, N], pl.BF16], y: pl.Out[pl.Tensor[[M, N], pl.FP32]]) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.NONE)]):
            t1 = pl.matmul(a, w, out_dtype=pl.FP32)
            t1b = pl.cast(t1, target_type=pl.BF16, mode="rint")
            y = pl.matmul(t1b, w2, out_dtype=pl.FP32, b_trans=True)
        return y
```

**Observed on a2a3 board:** compiles clean, runs, output all-zero (4095/4096 mismatch, `actual=0.0`). **CPU sim:** does not reproduce (guard present).

### Workaround (consumers)

Use `pl.split(pl.SplitMode.UP_DOWN)` + `pl.assemble` instead of `NONE` + direct return — the chained cube→vec→cube fusion runs correctly on a2a3 that way.

### Sim vs device divergence (the smoking gun)

The CPU sim has the inactive-NO_SPLIT-vec-lane guard; the NPU device tree has none:

- Guard exists ONLY in sim: `IsInactiveNoSplitVecLane` at `include/pto/cpu/TPush.hpp:175`, applied as an early-return in `TPUSH_IMPL` at `include/pto/cpu/TPush.hpp:797`; consumer variant `IsInactiveNoSplitVecConsumerLane` at `cpu/TPush.hpp:194`.
- `grep -rn "IsInactive" include/pto/npu/` → **no matches**. Neither a2a3 nor a5 device paths gate the inactive lane.

### Related

- `hw-native-sys/pypto#1525` (origin; board root-cause + handoff to pto-isa)
- pto-isa #83 (closed design-clarification: "A2A3 NONE requires both Vec sub-cores; AIV1 should strip side-effecting store" — this issue is the bug that it does not)
- pto-isa #82 (`TPUSH_IMPL` PIPE_MTE3 barrier vs `record()` — same sync pipe involved)
- pypto symptom family: #1507, #1523, #1564

### Environment

| Component | Version |
|---|---|
| pto-isa | `0f171f73` (bug also present at `0a1ce52` reported on #1525) |
| pypto | `b0d1d49a` |
| pypto-lib | `53a2efa` |
| pypto runtime (submodule) | `324df3d6` |
| ptoas | 0.41 |
| CANN | 9.0.0 |
| Host | Linux (aarch64) |


---

## #157 [Bug] A2A3 copy_ubuf_to_gm_align_b8 silently drops writes when dst is peer-rank-mapped GM — b16/b32 paths unaffected

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/157
- Created: 2026-06-02T11:26:57Z
- Updated: 2026-06-09T09:28:32Z
- Closed: 2026-06-09T09:28:32Z

### Body

## Summary

On A2/A3, `TSTORE` of a `Tile<TileType::Vec, int8_t, ...>` to a `GlobalTensor<int8_t, ...>` whose underlying pointer is a **peer-rank-mapped GM region** (a remote window slot in a multi-rank topology) silently writes nothing — subsequent `TLOAD` from the same address on the destination rank returns the prior (zero-initialised) buffer contents. The same `TSTORE` with `bfloat16_t` / `half` / `float` / `int32_t` to the **same** peer GM region works correctly. The same `TSTORE` with `int8_t` to a **local** GM region also works correctly.

The bug is the **intersection** of `{ sizeof(DType) == 1 }` and `{ dst points into a peer-mapped GM window }`. Neither condition alone reproduces.

The only code path that diverges between the working and failing cases is the dtype dispatch in `TStoreUb2gmInstr` / `TLoadInstrGm2ub`, which selects `copy_ubuf_to_gm_align_b8` instead of `copy_ubuf_to_gm_align_b16` / `_b32`. The template wrapper itself is structurally symmetric (same `nBurst`, `lenBurst`, `gmGap`, `ubGap`, `ubPad`); only the intrinsic name differs.

Silent wrong-result — clean compile, no assert, the TSTORE returns and downstream MTE3→MTE2 sync flags fire normally; just no bytes ever land at the dst address from the perspective of the destination rank's later TLOAD.

## Affected version

pto-isa commit `6909e5c406227e09720d56a8ff0878ef57c06f9f` (`6909e5c4`).

## Op / location

`include/pto/npu/a2a3/TStore.hpp:17-28` — `TStoreUb2gmInstr` dispatch:

```cpp
template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreUb2gmInstr(typename GlobalData::DType *dst, __ubuf__ typename TileData::DType *src,
                                   uint16_t nBurst, uint32_t lenBurst, uint32_t gmGap, uint32_t ubGap)
{
    if constexpr (sizeof(typename TileData::DType) == 1) {
        copy_ubuf_to_gm_align_b8(dst, src, 0, nBurst, lenBurst, 0, 0, ubGap, gmGap);   // ← failing path
    } else if constexpr (sizeof(typename TileData::DType) == 2) {
        copy_ubuf_to_gm_align_b16(dst, src, 0, nBurst, lenBurst, 0, 0, ubGap, gmGap);
    } else if constexpr (sizeof(typename TileData::DType) == 4 || sizeof(typename TileData::DType) == 8) {
        copy_ubuf_to_gm_align_b32(dst, src, 0, nBurst, lenBurst, 0, 0, ubGap, gmGap);
    }
}
```

Symmetric loader at `include/pto/npu/a2a3/TLoad.hpp:15-28` — `TLoadInstrGm2ub` selects `copy_gm_to_ubuf_align_b8` for `sizeof(DType) == 1`.

For the failing case the burst args are emitted symmetrically with the working b16/b32 calls — a `Tile<TileType::Vec, int8_t, 1, 64, ..., PadValue::Null>` + `GlobalTensor<int8_t, Shape<1,1,1,1,64>, Stride<64,64,64,64,1>, Layout::ND>` lowers (via `TStoreUb2gmNd2nd`) to:

```cpp
copy_ubuf_to_gm_align_b8(dst, src, /*sid=*/0, /*nBurst=*/1, /*lenBurst=*/64, 0, 0, /*ubGap=*/0, /*gmGap=*/0);
```

`lenBurst=64` is `2 * BLOCK_BYTE_SIZE` (32-byte aligned). The BF16 baseline emits the same shape with `lenBurst=128` and `copy_ubuf_to_gm_align_b16`.

## Symptom — what we observed

Identical 2-rank dispatch program, parameterised on the x-channel dtype only (protocol / `TNOTIFY` / `TWAIT` / barrier signal cells / address arithmetic / surrounding b32 channels held fixed):

| TSTORE call | DType (sizeof) | dst GM region | Result |
| - | - | - | - |
| `copy_ubuf_to_gm_align_b16` | bfloat16_t (2) | peer-rank-mapped window | PASS |
| `copy_ubuf_to_gm_align_b8`  | int8_t (1), `lenBurst=64`  | peer-rank-mapped window | **FAIL** — dst read-back is all zero (max abs diff = 91, full INT8 range) |
| `copy_ubuf_to_gm_align_b8`  | int8_t (1), `lenBurst=128` | peer-rank-mapped window | **FAIL** — same all-zero |
| `copy_ubuf_to_gm_align_b16` | half (2) (int8 TCVT→ fp16 → wire → fp16 TCVT→ int8) | peer-rank-mapped window | PASS |
| `copy_ubuf_to_gm_align_b8`  | int8_t (1) | **local** rank GM | PASS (`torch.equal == True`) |

The `lenBurst=64` vs `lenBurst=128` comparison rules out row-byte vs HW visibility-granule alignment (BF16 `lenBurst=128` works; INT8 `lenBurst=128` still fails — only the intrinsic selection differs). The fp16-wire variant — INT8 input cast through TCVT to FP16 on the UB side before the cross-rank `TSTORE`, then TCVT back to INT8 after the destination's `TLOAD` — passes byte-identically against the same golden, confirming that the b16 cross-rank path itself is correct and that swapping out the b8 intrinsic is sufficient to make the same data flow land. The local-GM baseline confirms `copy_ubuf_to_gm_align_b8` is correct when dst is on the same rank; the failure surfaces only when dst is a peer-mapped GM address.

Adjacent b32 / b16 transfers in the **same kernel run** (a parallel FP32 channel via `copy_ubuf_to_gm_align_b32` and an INT32 channel) all match expected, isolating the failure to the b8 path specifically.

Same downstream `MTE3→MTE2` sync via `wait_flag`/`set_flag` with identical `EVENT_ID` pattern on both BF16 and INT8 — the failure is not a sync timing issue (verified by structural diff of the emitted device code).

## Expected

`copy_ubuf_to_gm_align_b8(dst, ...)` with dst pointing into a peer-rank-mapped GM window should deliver the source bytes byte-identically, matching the behaviour of `copy_ubuf_to_gm_align_b16` / `_b32` on the same address.

## Repro

Reproduced through PyPTO system tests at `e3879f26`. The tests build small 2-rank programs whose generated AIV kernels emit the `TSTORE` calls described above; the difference between passing/failing variants is the dtype of the cross-rank window and the consequent `_b8` / `_b16` intrinsic selection in `TStoreUb2gmInstr`.

- `tests/st/distributed/test_l3_ep_dispatch_combine.py` — BF16 baseline (b16 cross-rank TSTORE) — **passes**.
- `tests/st/distributed/test_l3_ep_dispatch_combine_int8.py` — same program with x channel re-typed INT8 (b8 cross-rank TSTORE) — **fails**, dst all-zero.
- `tests/st/distributed/test_l3_ep_dispatch_combine_int8_via_fp16.py` — same program, x channel TCVT-cast to FP16 on the UB side before TSTORE and TCVT-cast back to INT8 after TLOAD (b16 cross-rank TSTORE on `half`) — **passes**.
- `tests/st/distributed/test_l3_ep_dispatch_combine_int8_local_dbg.py` — INT8 baseline plus an extra `TSTORE` of the same UB tile to a **local** GM buffer right before the cross-rank `TSTORE` (b8 local TSTORE) — local read-back is `torch.equal == True` in the same run where the cross-rank read-back is all-zero.

Generated AIV kernel for the failing path (excerpt from the int8 build):

```cpp
Tile<TileType::Vec, int8_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v114
    = Tile<TileType::Vec, int8_t, 1, 64, ...>(/*Rows=*/1, /*Cols=*/64);
TASSIGN(v114, /*zero-init*/ 0);
GlobalTensor<int8_t, Shape<1,1,1,1,64>, Stride<64,64,64,64,1>, Layout::ND> v118(
    /*local src ptr*/ v2 + slot_local * 64, ...);
wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
TLOAD(v114, v118);                                       // local int8 TLOAD — OK
set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

int32_t v119 = CommRemoteOffset_i8(/*comm_table*/ v18, /*peer=*/ v107);
__gm__ int8_t* v120 = v11 + v119;                        // base + (peer_base − my_base) → peer-mapped GM
GlobalTensor<int8_t, Shape<1,1,1,1,64>, Stride<64,64,64,64,1>, Layout::ND> v123(
    v120 + slot_peer * 64, ...);
wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
TSTORE(v123, v114);                                      // ← b8 cross-rank — silently delivers zeros
set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
```

The BF16 baseline build is byte-identical in structure, differs only in `int8_t → bfloat16_t`, the offset helper picks `÷2` instead of `÷1`, and the burst goes through `copy_ubuf_to_gm_align_b16`.

## What we ruled out

| Hypothesis | Verified ruled out |
| - | - |
| Sync flag insertion (wait/set EVENT_ID, PIPE_MTE2/MTE3) | Generated code is byte-identical between BF16 (passes) and INT8 (fails) for the cross-rank channel. |
| Burst arg asymmetry vs b16 path | `nBurst`, `lenBurst`, `gmGap`, `ubGap`, `ubPad` all symmetric; only the intrinsic-name selection at TStore.hpp:21-22 / TLoad.hpp:19-20 differs. |
| `lenBurst` vs HW visibility-granule | Doubling `lenBurst` from 64 to 128 bytes (D=64 → D=128) still fails; BF16 with `lenBurst=128` succeeds. |
| b8 intrinsic correctness on local GM | Local `copy_ubuf_to_gm_align_b8` baseline matches `torch.equal == True`. |
| Cross-rank correctness in general | b16 (bfloat16_t baseline, fp16-cast variant) / b32 cross-rank transfers in the same kernel all pass. |
| `CommRemoteOffset_<dtype>` arithmetic | Returns the same byte address for both dtypes after pointer arithmetic (`int8* + delta_bytes` ≡ `bf16* + delta_bytes/2`). |

## Environment

| Component | Version |
| - | - |
| pto-isa | `6909e5c4` |
| pypto (repro harness) | `e3879f26` |
| pypto runtime (submodule) | `324df3d6` |
| CANN | 9.0.0 |
| Host | Linux (aarch64) |
| Platform | a2a3 |


---

## #161 [Bug] RHS mask producer tail lanes fail unless dummy cmp materializes the path

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/161
- Created: 2026-06-05T07:54:21Z
- Updated: 2026-06-09T03:10:52Z
- Closed: 2026-06-09T03:10:52Z

### Body

### Summary

PyPTO can generate a producer kernel that builds an RHS / identity-like mask through row expansion, compare/select, BF16 cast, and GM store. After the PyPTO-side reshape lowering fix, the producer is still intermittently writing stale/zero tail lanes in the original race repros.

Adding a dummy `cmp`-style materialization/control changes the behavior: the same race2/race3/raceB families pass 10/10 in fresh L2 runs. This looks like a PTO ISA / codegen-runtime interaction around Vec lane materialization, `TSEL` scratch/aliasing, barriers, or final MTE/BF16 store ordering, not a missing PyPTO dependency edge.

### PyPTO version under test

- PyPTO PR: https://github.com/hw-native-sys/pypto/pull/1690
- Branch: `sunkaixuan2018:codex/fix-issue-1503-reshape-pr`
- Tested commit after rebase: `391614e73e6ef8d75e5effb5180e0945f5952a88`
- Scope of that PR: fixes the confirmed PyPTO lowering bug where static `[1, K] -> [K, 1]` reshape must be materialized instead of treated as a view-only shape change.
- Focused PyPTO test passed on `myserver`: `python3 -m pytest tests/ut/codegen/test_pto_codegen_ops.py -k 'row_vector or k64_row' -v` -> 3 passed, 46 deselected.

### Problem pattern

The producer builds an RHS mask similar to:

```python
idx = pl.tensor.ci(0, [1, K], dtype=pl.INT32)
idx_col = pl.tensor.reshape(idx, [K, 1])
row = pl.tensor.cast(idx_col, target_type=pl.FP32)
row2d = pl.row_expand_mul(ones, row)
cmp = pl.tensor.cmp(row2d, cols, cmp_type=0)
w_build[:, :] = pl.tensor.cast(cmp, target_type=pl.BF16)
```

PyPTO now emits a real transpose/materialization for `[1, K] -> [K, 1]`. However, the original RHS race repros still intermittently produce zero/stale tail diagonal points. Extra dummy compare/materialization code makes the bad points disappear in the sampled runs.

### Reproduction scripts

Original repros on `myserver`:

```text
/data/sunkaixuan/codex_sh/_tmp_rhs_race2.py
/data/sunkaixuan/codex_sh/_tmp_rhs_race3.py
/data/sunkaixuan/codex_sh/_tmp_rhs_raceB.py
```

Dummy-cmp control repros:

```text
/data/sunkaixuan/codex_sh/issue1503_rhs_race2_dummy_cmp.py
/data/sunkaixuan/codex_sh/issue1503_rhs_race3_dummy_cmp.py
/data/sunkaixuan/codex_sh/issue1503_rhs_raceB_dummy_cmp.py
```

10x loop command:

```bash
ssh myserver "bash /data/sunkaixuan/codex_sh/issue1503_loop_rhs_dummy_cmp_basic_perf_10x_l2.sh"
```

Important: every L2 sample must be a fresh `task-submit` invocation. Do not run several cases in one Python process.

### 10x result

Summary/log artifacts:

```text
/data/sunkaixuan/skx_log_output/issue1503_rhs_dummy_cmp_basic_perf_10x_20260605_151549/summary.log
/data/sunkaixuan/skx_log_output/issue1503_rhs_dummy_cmp_basic_perf_10x_20260605_151549/results.tsv
```

| Case | Samples | Tensor PASS | Tensor FAIL | Runtime/device failure | First known mismatch positions |
| --- | ---: | ---: | ---: | ---: | --- |
| race2 original | 10 | 1 | 9 | 0 | `[60, 61, 62, 63, ...]`; 2035-2042/32768 mismatches in failing runs |
| race3 original | 10 | 3 | 7 | 0 | `[3900, 3965, 4030, 4095, ...]`; 128/131072 mismatches |
| raceB original | 10 | 3 | 7 | 0 | `[3900, 3965, 4030, 4095, ...]`; 128/131072 mismatches |
| race2 dummy-cmp | 10 | 10 | 0 | 0 | None |
| race3 dummy-cmp | 10 | 10 | 0 | 0 | None |
| raceB dummy-cmp | 10 | 10 | 0 | 0 | None |

Representative artifacts:

```text
race2 original FAIL: /data/sunkaixuan/pypto-lib/models/deepseek/v4/build_output/_jit_rhs_race2_test_20260605_151551
race2 original PASS: /data/sunkaixuan/pypto-lib/models/deepseek/v4/build_output/_jit_rhs_race2_test_20260605_151720
race3 original FAIL: /data/sunkaixuan/pypto-lib/models/deepseek/v4/build_output/_jit_rhs_race3_test_20260605_151826
race3 original PASS: /data/sunkaixuan/pypto-lib/models/deepseek/v4/build_output/_jit_rhs_race3_test_20260605_151852
raceB original FAIL: /data/sunkaixuan/pypto-lib/models/deepseek/v4/build_output/_jit_rhs_raceB_test_20260605_152038
raceB original PASS: /data/sunkaixuan/pypto-lib/models/deepseek/v4/build_output/_jit_rhs_raceB_test_20260605_152052
race2 dummy-cmp PASS: /data/sunkaixuan/pypto-lib/models/deepseek/v4/build_output/_jit_rhs_race2_test_20260605_152303
race3 dummy-cmp PASS: /data/sunkaixuan/pypto-lib/models/deepseek/v4/build_output/_jit_rhs_race3_test_20260605_152547
raceB dummy-cmp PASS: /data/sunkaixuan/pypto-lib/models/deepseek/v4/build_output/_jit_rhs_raceB_test_20260605_152802
```

### Current observations

- The PyPTO dependency graph contains the expected producer -> consumer TensorMap edge.
- L2 swimlane evidence showed the consumer dispatching after the producer finishes.
- A plain peek/read of `w_build` can still observe the bad mask data, so this is not only a consumer scheduling problem.
- Removing matmul did not remove the stale tail points.
- Simple wait/fence/double-store variants did not fix the original bad tail points.
- Intermediate dumps or dummy compare/materialization can mask the issue.

### Suggested pto-isa investigation directions

Please check the generated producer path around:

```text
TROWEXPANDMUL / TCMP -> TSEL -> TMOV -> TCVT -> BF16 TSTORE
```

Questions to investigate:

- Does `TROWEXPANDMUL` guarantee all valid/tail lanes are materialized before a following `TCMP` / `TSEL` chain consumes them?
- Does `TSEL` require a full-sized scratch tile for this full-lane materialization pattern?
- Are there aliasing restrictions between `TSEL` output and inputs such as `cmp_one`, `cmp_zero`, `cmp_mask`, or `cmp_tmp`?
- Are extra barriers required between `TSEL`, `TMOV`, `TCVT`, and BF16 `TSTORE` for tail rows?
- Why does an unrelated dummy compare / materialization path make race2/race3/raceB pass in 10/10 fresh L2 runs, while the original path still fails frequently?

### Expected behavior

The original and dummy-cmp variants should both produce the same correct RHS mask. Adding an unrelated dummy compare/materialization should not be required to make tail diagonal lanes visible to later readers.

### Actual behavior

The original variants still fail frequently after the PyPTO reshape materialization fix, while the dummy-cmp variants passed 10/10 in the same fresh L2 sampling setup.

---

## #164 Non-templated MSCATTER default coalesce semantics diverge between CPU sim (Elem) and NPU onboard (Row)

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/164
- Created: 2026-06-08T06:34:47Z
- Updated: 2026-06-12T02:17:31Z
- Closed: 2026-06-12T02:17:31Z

### Body

## Summary

The **non-templated** `MSCATTER` overload is available on both the CPU
simulator and NPU targets, but its **default coalesce semantics differ by
backend**: the CPU sim models `Coalesce::Elem`, while NPU hardware defaults to
`Coalesce::Row`. As a result, a single non-templated `MSCATTER(dst, src, idx)`
kernel source produces **different results on sim vs. onboard**, so one golden
cannot validate both — defeating the purpose of offering a portable
non-templated surface.

## Details

The dispatch layer exposes a non-templated `MSCATTER` unconditionally, but all
templated overloads are gated behind `#ifdef PTO_NPU_ARCH_A5`:

- `include/pto/common/pto_instr.hpp:1936` — non-templated `MSCATTER` (no `#ifdef`)
- `include/pto/common/pto_instr.hpp:1943-1978` — all templated `MSCATTER<Coalesce, ...>` overloads inside `#ifdef PTO_NPU_ARCH_A5`

Backend behavior of the non-templated form:

- **CPU sim** (`include/pto/cpu/MGatherScatter.hpp`, `MSCATTER_IMPL`) — no
  `Coalesce` parameter at all; always walks `validRow × validCol` and writes
  `dst[idx[i,j]] = src[i,j]`, i.e. **`Coalesce::Elem`** semantics.
- **A5 onboard** (`include/pto/npu/a5/MScatter.hpp`, `MSCATTER_IMPL` default
  `Coalesce Mode = Coalesce::Row`) — non-templated call deduces the default
  `Coalesce::Row`, dispatching to `MScatterRowImpl` (per-row scatter, index
  treated as a logical row index).

The docs confirm the asymmetry:

- `docs/isa/tile/ops/memory-and-data-movement/mscatter.md:11` — `Coalesce::Row` is the default.
- `mscatter.md:22` / `:127` / `:242` — the CPU simulator "always" does `Coalesce::Elem` and "does not have a separate Row coalesce path".

## Impact

A kernel that wants element-scatter semantics and must build for both the CPU
simulator and A5 onboard cannot use a single non-templated `MSCATTER` call:

- On sim, non-templated = Elem (correct).
- On A5 onboard, non-templated = Row (wrong for an element-scatter test), and
  for an `[R, C]`-shaped index tile it additionally fails `MScatterCheck`,
  which requires `idx.ValidRow == 1` in Row mode — so it may not even compile.

Today the only way to get consistent element-scatter on both is a per-target
split:

```cpp
#ifdef __CPU_SIM
    MSCATTER(out, src, idx);                                        // cpu: Elem (only form)
#else
    MSCATTER<Coalesce::Elem, ScatterAtomicOp::None, ScatterOOB::Skip>(out, src, idx);  // a5: explicit Elem
#endif
```

Notably, the repo's own ST suite sidesteps this by keeping **separate kernel
files per backend** (`tests/cpu/st/testcase/mscatter/mscatter_kernel.cpp` uses
non-templated; `tests/npu/a5/src/st/testcase/mscatter/mscatter_kernel.cpp` uses
only explicit `MSCATTER<Coalesce::Row|Elem, ...>`), so it never shares one
kernel source across sim and onboard.

## Suggested fix (any one)

1. **Make the non-templated default consistent** — have the CPU sim model the
   same default (`Coalesce::Row`) as hardware, or make hardware's non-templated
   default `Coalesce::Elem`, so one non-templated source behaves identically on
   both.
2. **Remove the non-templated overload on hardware** (or `static_assert` it
   out), forcing the mode to be explicit at every call site. This makes the
   sim/onboard divergence a compile error instead of a silent golden mismatch.
3. **Document the trap loudly** at the non-templated `MSCATTER` declaration if
   the asymmetry is intentional, so callers know it is not portable.

Option 1 best matches the apparent intent of offering a portable non-templated
surface.

## Environment

- Observed against `hw-native-sys/pto-isa` (current `main`).
- Surfaced while making an a5 SIMT element-scatter ST run under both `a5sim`
  (CPU sim, `__CPU_SIM`) and `a5` onboard in a downstream consumer.


---

## #168 TCONCAT (a2a3): asymmetric-width concat corrupts dst — src1 copy reuses src0's blockLen

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/168
- Created: 2026-06-15T12:34:36Z
- Updated: 2026-06-30T01:36:35Z
- Closed: 2026-06-30T01:36:35Z

### Body

## Summary

On a2a3, `TCONCAT` corrupts the destination tile whenever the two sources have **different valid column counts** (`validCol0 != validCol1`) **and** `validCol0` is block-aligned. The aligned fast path copies `src1` using the **block length derived from `validCol0`**, so it writes `validCol0` columns of `src1` (instead of `validCol1`) starting at column `validCol0`, overrunning each destination row into the following rows.

The symmetric case (`validCol0 == validCol1`, e.g. the shipped `[32,16]+[32,16]` example and the `tconcat` ST tests) is unaffected, which is why this has gone unnoticed.

## Affected code

`include/pto/npu/a2a3/TConcat.hpp` — `TConcatImpl(...)`, the `isAligned` fast path (commit `b9122ec5`):

```cpp
unsigned blockLen = (validCol0 * sizeof(TD) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;  // from validCol0
... // src0 copy uses blockLen  -- correct

bool isAligned = (validCol0 % elementsPerBlock) == 0;
if (isAligned) {
    unsigned src1Gap = (TileDataS1::Cols * sizeof(TD) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE - blockLen; // (A)
    for (int i = 0; i < validRow; i++) {
        pto_copy_ubuf_to_ubuf(dstPtr + i * dstRowStride + validCol0, src1Ptr + i * src1RowStride,
                              1, blockLen,                 // (B) <-- uses src0's blockLen for src1
                              src1Gap, dstGap);
    }
}
```

- **(B)** copies `blockLen` blocks of `src1`, where `blockLen` was computed from `validCol0`. It should copy a block count derived from `validCol1`.
- **(A)** `src1Gap` also underflows: for `validCol1 < validCol0`, `ceil(S1::Cols/block) - blockLen` is negative and wraps to a huge `unsigned`.

The unaligned `else` branch (element-wise copy of exactly `validCol1` columns) is correct, and the CPU reference (`include/pto/cpu/TConcat.hpp`) is correct (copies `cols0` from src0, `cols1` from src1). Only the a2a3 aligned fast path is wrong.

## Concrete example

bf16 (`sizeof=2`), `BLOCK_BYTE_SIZE=32` ⇒ `elementsPerBlock=16`.
`src0` valid `[16, 448]`, `src1` valid `[16, 64]`, `dst` `[16, 512]` (row stride 512):

- `blockLen = ceil(448*2/32) = 28` blocks (= 448 elements) — correct for src0.
- `isAligned = (448 % 16 == 0) = true` ⇒ fast path taken.
- src1 copy writes **`blockLen` = 28 blocks = 448 elements** of `src1` at `dst[:, 448:896]`, overrunning the 512-wide rows. Expected: 64 elements at `dst[:, 448:512]`.

## Minimal repro

Any bf16 `tconcat` with `validCol0` a multiple of `elementsPerBlock` and `validCol0 != validCol1`, e.g. `src0[16,32] + src1[16,16] -> dst[16,48]`: `dst[:, 32:48]` receives 32 elements of src1 (16 valid + 16 stale) instead of 16. The existing `tests/.../testcase/tconcat` cases only use equal widths; adding an asymmetric-width case reproduces it.

## How it was found

Surfaced building a `[16,448] (no-RoPE) + [16,64] (RoPE)` head tile in a PyPTO sparse-attention kernel (one `pl.concat` for a 512-wide head). Device validation showed **99.18%** of outputs wrong (`max abs diff ≈ 0.055`); replacing the single `concat` with two separate column-range stores of the same data is **bit-exact**, confirming the corruption is inside `TCONCAT`, not the producer.

## Proposed fix

Give `src1` its own block length:

```cpp
    if (isAligned) {
        unsigned blockLen1 = (validCol1 * sizeof(TD) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
        unsigned src1Gap = (TileDataS1::Cols * sizeof(TD) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE - blockLen1;
        for (int i = 0; i < validRow; i++) {
            pto_copy_ubuf_to_ubuf(dstPtr + i * dstRowStride + validCol0, src1Ptr + i * src1RowStride,
                                  1, blockLen1, src1Gap, dstGap);
        }
    }
```

Worth also adding an asymmetric-width (`validCol0 != validCol1`) case to the `tconcat` ST tests, and checking the a5/other backends for the same pattern (a5's `TConcat.hpp` uses a different path and does not appear affected).

## Environment

- pto-isa commit `b9122ec5`
- Backend: a2a3 (`dav-c220`), bf16
- Toolchain: CANN cann-8.5.1


---

## #170 Vector UB usable size appears to be ~184KB (not 192KB) on Ascend910B1 / cann-9.0.0 — silent corruption when tile allocations exceed ~184KB

- State: open
- URL: https://github.com/hw-native-sys/pto-isa/issues/170
- Created: 2026-06-17T07:00:59Z
- Updated: 2026-06-17T09:14:44Z
- Closed: 

### Body

## Summary

On **Ascend910B1 / CANN `cann-9.0.0`** (target `a2a3`), a pure-Vector kernel whose `pto.alloc_tile` allocations reach a **188.8 KB** Vec high-water — well within the **192 KB** Vec UB — produces **NaN** in its output on device. The *same* kernel reduced to a **183.3 KB** high-water is correct. The corruption switches on **between 183.3 KB (clean) and 188.8 KB (NaN)** — i.e. right around **184 KB**. Nothing errors at compile in either case; the 188.8 KB case **silently corrupts** at runtime.

| case | Vec footprint | # buffers | device output |
| ---- | ------------- | --------- | ------------- |
| HD512 | **188.8 KB** / 192 KB | 81 | **NaN** (112 NaNs) |
| HD496 | **183.3 KB** / 192 KB | 81 | correct (0 NaN) |

Both are the *same* kernel with an identical op sequence; only the tile column size differs (→ footprint).

## CANN version (important)

The device reproduction runs on **cann-9.0.0** (active `ASCEND_HOME_PATH` / `LD_LIBRARY_PATH` / `PATH` all point at `.../Ascend/cann-9.0.0`, Version=9.0.0). We were told this Vec-UB reservation "was removed" — but **188.8 KB (< 192 KB) still corrupts on 9.0.0**, so it appears to still be in effect. (`cann-8.5.1` was used *only* for the in-core op-simulator, which needs a TL-capable `bisheng` for the camodel build; it does not run the device NaN.)

## Hypothesis

The real usable Vec UB is **~184 KB, not 192 KB** — i.e. ~8 KB at the top is reserved (we see `ReserveBufferOp` / `PTOInferValidatePipeInitPass` in PTOAS). A tile placed in that top region reads/writes garbage → NaN. The upstream (PyPTO) allocator's 192 KB budget does not subtract this, so it emits a `.pto` reaching 188.8 KB without any error.

## Evidence the NaN is memory corruption (not numerics, not a logic bug)

- **Localized**: every NaN is in the highest-address accumulator tiles (the `m_oi_nope` accumulator, NOPE half of the output); lower-address tiles are clean.
- **Footprint-causal**: drop the high-water below ~184 KB (smaller tiles, or set the inner pipeline to `stage=1`) → 0 NaN; same kernel otherwise.
- **Not a numerical artifact**: with inputs that keep the divisor `n_denom = li + exp(...)` strictly positive, the 188.8 KB case **still** NaNs → corrupted buffer data, not a division.
- **Data-dependent cells**: NaN cells vary with input seed (88 vs 112) — consistent with a high-address buffer overlapping a reserved/garbage region.

## Reproduction

Artifacts — both `.pto`, the per-buffer memory reports, and the repro scripts:
**https://gist.github.com/Hzfengsy/8f655587af225f73665a1e9a3112441a**

For PTOAS directly: compile `HD512_WRONG_merge_rope_pack.pto` and `HD496_CORRECT_merge_rope_pack.pto` for `a2a3` / `Ascend910B1` / cann-9.0.0, run, and check the output for NaN. HD512's top `alloc_tile` reaches addr ~188.6 KB (→ NaN); HD496 stays ≤ ~183.1 KB (→ clean). `diff` the two `.pto` — same structure, only tile sizes / `addr` values differ.

<details><summary>PyPTO-side standalone device harness — merge_h.py (HD env var sets the hidden dim / footprint)</summary>

```python
# Copyright (c) PyPTO Contributors.
# Standalone device harness: lifts the dsv4 `merge_rope_pack` scope out of
# decode_sparse_attn.py and runs it as its own kernel on device (-d 0) to check
# whether o_packed (its OWN output) contains NaN/Inf. Isolates whether the
# coloring corruption is intrinsic to the kernel.
import sys

sys.path.insert(0, "/home/syfeng/pypto-pipeline-stage-scope/python")
import pypto  # noqa: E402

assert "pipeline-stage-scope" in pypto.__file__, pypto.__file__
assert "pipeline-stage-scope" in pypto.pypto_core.__file__, pypto.pypto_core.__file__

sys.path.insert(0, "/home/syfeng/pypto/dsv4_kernels")
import pypto.language as pl  # noqa: E402
import decode_sparse_attn as M  # noqa: E402
import os

# Reference constants through M.* so they stay in sync with the kernel.
T = M.T
H = M.H
HEAD_DIM = int(os.environ.get('HD', M.HEAD_DIM))
NOPE_DIM = HEAD_DIM - M.ROPE_DIM
ROPE_DIM = M.ROPE_DIM
HALF_ROPE = M.HALF_ROPE
ROPE_H_TILE = M.ROPE_H_TILE
ROPE_SUBTILES = M.ROPE_SUBTILES
H_TILE = M.H_TILE
SPARSE_BLOCKS = M.SPARSE_BLOCKS
HEADS_PER_GROUP = M.HEADS_PER_GROUP
O_GROUPS = M.O_GROUPS
D = M.HEADS_PER_GROUP * HEAD_DIM
FUSE_ALIGNED_SINGLE_STORE = M.FUSE_ALIGNED_SINGLE_STORE

# T*(H//H_TILE)*SPARSE_BLOCKS*H_TILE = 128*4*5*16 = 40960
SPARSE_ROWS = T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE
assert SPARSE_ROWS == 40960, SPARSE_ROWS


@pl.jit
def main(
    sparse_blk_mi: pl.Tensor[[40960, 1], pl.FP32],
    sparse_blk_li: pl.Tensor[[40960, 1], pl.FP32],
    sparse_blk_oi: pl.Tensor[[40960, HEAD_DIM], pl.FP32],
    attn_sink: pl.Tensor[[64], pl.FP32],
    freqs_cos: pl.Tensor[[128, 64], pl.BF16],
    freqs_sin: pl.Tensor[[128, 64], pl.BF16],
    o_packed: pl.Out[pl.Tensor[[1024, D], pl.BF16]],
):
    # ---- BEGIN verbatim lift of decode_sparse_attn.py lines 346..449 ----
    for m_t in pl.spmd(T, name_hint="merge_rope_pack"):
        m_token_base = m_t * (H // ROPE_H_TILE) * SPARSE_BLOCKS * ROPE_H_TILE
        # Inverse-RoPE interleave pattern (j^1 swap, j>>1 dup, [+1,-1,...] sign)
        # over the FULL ROPE_DIM at once (the standalone rope chunked into 32-col
        # tloads; here the rope tail is a single contiguous [ROPE_H_TILE, ROPE_DIM]
        # tile, so we rotate it in one pass -- no chunk-slicing). Column-only ->
        # build once per token-task, on ROPE_H_TILE rows.
        sp_ones = pl.full([ROPE_H_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        sp_col = pl.col_expand_mul(
            sp_ones, pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32)
        )
        sp_dup_f = pl.cast(
            pl.cast(pl.mul(sp_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32
        )
        sp_dup_idx = pl.cast(sp_dup_f, target_type=pl.INT32)  # j>>1
        sp_lane = pl.sub(sp_col, pl.mul(sp_dup_f, 2.0))  # j%2
        sp_swap_idx = pl.cast(pl.sub(pl.add(sp_col, 1.0), pl.mul(sp_lane, 2.0)), target_type=pl.INT32)  # j^1
        sp_sign = pl.neg(pl.sub(pl.mul(sp_lane, 2.0), 1.0))  # [+1,-1,...] (conjugate)

        for m_h_idx in pl.pipeline(H // ROPE_H_TILE, stage=2):
            m_h0 = m_h_idx * ROPE_H_TILE
            # sparse_blk_* is stored in H_TILE-row blocks (qk_pv layout). Map this
            # ROPE_H_TILE-row sub-tile to its stored head-tile m_ht and the half
            # m_half within that block; the base offsets into the block by
            # m_half * ROPE_H_TILE, and the sparse-block stride below stays H_TILE
            # (the storage stride), not ROPE_H_TILE.
            m_ht = m_h_idx // ROPE_SUBTILES
            m_half = m_h_idx % ROPE_SUBTILES
            m_blk_base = m_token_base + m_ht * SPARSE_BLOCKS * H_TILE + m_half * ROPE_H_TILE
            m_mi = sparse_blk_mi[m_blk_base : m_blk_base + ROPE_H_TILE, 0:1]
            m_li = sparse_blk_li[m_blk_base : m_blk_base + ROPE_H_TILE, 0:1]
            # Split the attention-output accumulator into NOPE + ROPE column
            # ranges, each loaded as its own contiguous tile (tensor slice ->
            # tload). PyPTO Vec ops need full contiguous operands, so the rope
            # tail must be a standalone tile, not a column-subview of a wide one.
            m_oi_nope = sparse_blk_oi[m_blk_base : m_blk_base + ROPE_H_TILE, 0:NOPE_DIM]
            m_oi_rope = sparse_blk_oi[m_blk_base : m_blk_base + ROPE_H_TILE, NOPE_DIM:HEAD_DIM]

            for m_sb in pl.pipeline(1, SPARSE_BLOCKS, stage=2):
                m_row = m_blk_base + m_sb * H_TILE
                m_cur_mi = sparse_blk_mi[m_row : m_row + ROPE_H_TILE, 0:1]
                m_cur_li = sparse_blk_li[m_row : m_row + ROPE_H_TILE, 0:1]
                m_cur_oi_nope = sparse_blk_oi[m_row : m_row + ROPE_H_TILE, 0:NOPE_DIM]
                m_cur_oi_rope = sparse_blk_oi[m_row : m_row + ROPE_H_TILE, NOPE_DIM:HEAD_DIM]
                m_mi_new = pl.maximum(m_mi, m_cur_mi)
                m_alpha = pl.exp(pl.sub(m_mi, m_mi_new))
                m_beta = pl.exp(pl.sub(m_cur_mi, m_mi_new))
                m_li = pl.add(pl.mul(m_alpha, m_li), pl.mul(m_beta, m_cur_li))
                m_oi_nope = pl.add(
                    pl.row_expand_mul(m_oi_nope, m_alpha), pl.row_expand_mul(m_cur_oi_nope, m_beta)
                )
                m_oi_rope = pl.add(
                    pl.row_expand_mul(m_oi_rope, m_alpha), pl.row_expand_mul(m_cur_oi_rope, m_beta)
                )
                m_mi = m_mi_new

            n_sink_bias = pl.reshape(attn_sink[m_h0 : m_h0 + ROPE_H_TILE], [ROPE_H_TILE, 1])
            n_sink_tile = pl.add(pl.sub(m_mi, m_mi), n_sink_bias)
            n_denom = pl.add(m_li, pl.exp(pl.sub(n_sink_tile, m_mi)))
            # NOPE half: normalize + BF16 (unchanged from the old NOPE store path).
            n_nope = pl.cast(pl.row_expand_div(m_oi_nope, n_denom), target_type=pl.BF16)
            # ROPE half: normalize, then BF16-round the rope INPUT (matches the old
            # attn_rope_stage BF16 round-trip that golden also does), inverse-RoPE
            # the contiguous [ROPE_H_TILE, ROPE_DIM] tile, BF16-round the output.
            #   out[j] = x[j]*cos_il[j] + x[j^1]*sign[j]*sin_il[j]
            n_rope_in = pl.cast(
                pl.cast(pl.row_expand_div(m_oi_rope, n_denom), target_type=pl.BF16), target_type=pl.FP32
            )
            r_cos = pl.cast(freqs_cos[m_t : m_t + 1, 0:HALF_ROPE], target_type=pl.FP32)
            r_sin = pl.cast(freqs_sin[m_t : m_t + 1, 0:HALF_ROPE], target_type=pl.FP32)
            r_cos_h = pl.col_expand_mul(pl.full([ROPE_H_TILE, HALF_ROPE], dtype=pl.FP32, value=1.0), r_cos)
            r_sin_h = pl.col_expand_mul(pl.full([ROPE_H_TILE, HALF_ROPE], dtype=pl.FP32, value=1.0), r_sin)
            r_cos_il = pl.gather(r_cos_h, dim=-1, index=sp_dup_idx)
            r_sin_il = pl.gather(r_sin_h, dim=-1, index=sp_dup_idx)
            r_swapped = pl.gather(n_rope_in, dim=-1, index=sp_swap_idx)
            r_rot = pl.add(pl.mul(n_rope_in, r_cos_il), pl.mul(pl.mul(r_swapped, sp_sign), r_sin_il))
            n_rope = pl.cast(r_rot, target_type=pl.BF16, mode="rint")

            # Write each head to o_packed. Two store modes (compile-time switch,
            # FUSE_ALIGNED_SINGLE_STORE) -- both bit-identical in intent, they only
            # differ in store granularity / alignment:
            #   - single aligned: join [16,448]+[16,64] -> [16,512] via pl.concat
            #     and emit one [1,HEAD_DIM]=1024B 512B-aligned store per head.
            #     Faster (VECTOR-bound) but blocked by pto-isa TCONCAT bug #168.
            #   - two-store: emit NOPE [1,448] and ROPE [1,64] separately (same as
            #     the original NOPE + rope_pack stores). Correct today.
            if FUSE_ALIGNED_SINGLE_STORE:
                n_full = pl.concat(n_nope, n_rope)
                for n_hi in pl.range(ROPE_H_TILE):
                    n_gh = m_h0 + n_hi
                    n_g = n_gh // HEADS_PER_GROUP
                    n_hh = n_gh - n_g * HEADS_PER_GROUP
                    n_pack_row = n_g * T + m_t
                    n_col = n_hh * HEAD_DIM
                    o_packed[n_pack_row : n_pack_row + 1, n_col : n_col + HEAD_DIM] = n_full[n_hi : n_hi + 1, 0:HEAD_DIM]
            else:
                for n_hi in pl.range(ROPE_H_TILE):
                    n_gh = m_h0 + n_hi
                    n_g = n_gh // HEADS_PER_GROUP
                    n_hh = n_gh - n_g * HEADS_PER_GROUP
                    n_pack_row = n_g * T + m_t
                    n_col = n_hh * HEAD_DIM
                    o_packed[n_pack_row : n_pack_row + 1, n_col : n_col + NOPE_DIM] = n_nope[n_hi : n_hi + 1, 0:NOPE_DIM]
                    o_packed[n_pack_row : n_pack_row + 1, n_col + NOPE_DIM : n_col + HEAD_DIM] = n_rope[n_hi : n_hi + 1, 0:ROPE_DIM]
    # ---- END verbatim lift ----
    return o_packed


def _build_specs():
    """7-entry TensorSpec list: 6 finite-random inputs + o_packed output."""
    import torch
    from golden import TensorSpec

    def seeded_uniform(shape, seed, dtype):
        def _init():
            gen = torch.Generator()
            gen.manual_seed(seed)
            # SMALL finite uniform[-0.5, 0.5] so online-softmax exp() cannot
            # overflow from data -- any NaN/Inf is therefore CORRUPTION, not math.
            return (torch.rand(*shape, generator=gen) - 0.5).to(dtype)

        return _init

    return [
        TensorSpec("sparse_blk_mi", [40960, 1], torch.float32, init_value=seeded_uniform((40960, 1), 11, torch.float32)),
        TensorSpec("sparse_blk_li", [40960, 1], torch.float32, init_value=seeded_uniform((40960, 1), 12, torch.float32)),
        TensorSpec("sparse_blk_oi", [40960, HEAD_DIM], torch.float32, init_value=seeded_uniform((40960, HEAD_DIM), 13, torch.float32)),
        # attn_sink = zeros (matches golden).
        TensorSpec("attn_sink", [64], torch.float32, init_value=0.0),
        TensorSpec("freqs_cos", [128, 64], torch.bfloat16, init_value=seeded_uniform((128, 64), 14, torch.bfloat16)),
        TensorSpec("freqs_sin", [128, 64], torch.bfloat16, init_value=seeded_uniform((128, 64), 15, torch.bfloat16)),
        TensorSpec("o_packed", [1024, D], torch.bfloat16, is_output=True),
    ]


def _golden_zero(values):
    """Trivial golden: zero-fill o_packed. _validate still prints the device
    output's illegal-values (NaN/Inf) line regardless of golden match."""
    import torch

    values["o_packed"] = torch.zeros((1024, D), dtype=torch.bfloat16)


def main_run(device: int = 0, platform: str = "a2a3") -> bool:
    from golden import ratio_allclose, run_jit

    specs = _build_specs()
    runtime_cfg = dict(platform=platform, device_id=device, enable_l2_swimlane=False)
    # Loose max_error_ratio so the ratio check does not abort before the NaN
    # report; the NaN/Inf hard check inside the comparator always runs first and
    # prints "illegal values in actual: NaN=.. Inf=.." for the DEVICE output.
    compare_fn = {"o_packed": ratio_allclose(atol=1e9, rtol=1e9, max_error_ratio=1.0)}
    result = run_jit(
        fn=main,
        specs=specs,
        golden_fn=_golden_zero,
        save_data=False,
        rtol=1e9,
        atol=1e9,
        compare_fn=compare_fn,
        runtime_cfg=runtime_cfg,
    )
    print(f"[merge_rope_standalone] passed={result.passed} error={result.error!r}")
    return result.passed


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--device", type=int, default=0)
    ap.add_argument("-p", "--platform", type=str, default="a2a3")
    ap.add_argument("--compile-only", action="store_true")
    args = ap.parse_args()

    if args.compile_only:
        from golden import run_jit

        res = run_jit(fn=main, specs=_build_specs(), compile_only=True,
                      runtime_cfg=dict(platform=args.platform, device_id=args.device))
        print(f"[merge_rope_standalone] compile-only passed={res.passed} error={res.error!r}")
    else:
        main_run(device=args.device, platform=args.platform)
```
</details>

## Questions for PTO-ISA

1. Is there a Vec UB region PTOAS reserves (pipe-init/validate, or runtime scratch) that a kernel's `alloc_tile` allocations must **not** cross? What is the **real usable Vec UB** for `a2a3` / `Ascend910B1` on cann-9.0.0?
2. We were told the reservation was removed — but it still corrupts at 188.8 KB on 9.0.0. Is it still in effect, or is this a separate near-full-footprint bug?
3. If it's a fixed reservation, should the upstream allocator subtract it from the 192 KB budget so an over-tall layout **errors at compile** instead of corrupting silently? Can you confirm the exact reserved size?


---

## #172 flash_atten: ~10-12% large-S perf regression from TPipe::SyncPeriod change in TPush.hpp

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/172
- Created: 2026-06-17T12:28:33Z
- Updated: 2026-06-25T01:18:25Z
- Closed: 2026-06-25T01:18:25Z

### Body

## Summary

`kernels/python/flash_atten` (the PTO-DSL four-stage Cube/Vec + GM software-FIFO pipeline) shows a **~10–12% performance regression** on large sequences (S ≥ 32768) versus `torch_npu.npu_fused_infer_attention_score` (speedup drops from ~1.00× to ~0.89–0.90×).

A clean line-by-line isolation traces it to **a single line of code** — the value of `TPipe::SyncPeriod` in `include/pto/npu/a2a3/TPush.hpp`:

```cpp
// flash_atten instantiates TPipe<0, DIR_C2V, SlotSize=131072, SlotNum=8, ...>
old (faster):   static constexpr uint32_t SyncPeriod = (SlotNum <= 2) ? SlotNum : SlotNum / 2;  // = 4
current (slow): static constexpr uint32_t SyncPeriod = SlotNum;                                 // = 8
```

This was introduced by commit `014920a8` (*"Clean pending free signals during TPUSH"*, csjlchen, 2026-05-22), which — while reworking the TPUSH drain / pending-free-signal cleanup — also bumped `SyncPeriod` from `SlotNum/2` to `SlotNum`. That commit's own message notes `Not-tested: Full NPU hardware validation pending`.

`SyncPeriod = SlotNum` (= the full FIFO depth) makes the Cube↔Vec cross-core synchronization **coarse-grained and bursty**, which breaks steady-state pipeline overlap and slows the DSL kernel by ~10–12% at large S.

> **Related PR: #169** (`flash_atten: refine README, optimize perf (split-KV)`). The ~0.86–0.93× band for `case5`–`case8` in that PR's verification table is exactly this regression. It is **independent of** PR #169's split-KV optimization and its `pto_instr.hpp` build fix — this is a **header-level** performance regression (`ptoas` version and timing mode are *not* the real cause).

---

## Environment

- Platform: Ascend A3 (24 cube cores)
- `ptoas 0.45`, `bisheng`/CANN `9.0.0`, `torch_npu 2.9.0`
- Kernel: `kernels/python/flash_atten`; reference: `torch_npu.npu_fused_infer_attention_score`
- Timing: `--timing sync` (more stable than event timing at large cases; avoids the occasional event-timing glitch)

---

## Experiment: full A/B performance comparison

**Clean isolation:** same card, same `ptoas 0.45`, same MLIR / generated C++; only the bisheng `-I` header root (`PTO_LIB_PATH`) is switched. The two header trees are **byte-identical except the single `SyncPeriod` line at `TPush.hpp:37`**. Full suite `case1`–`case8`, `--timing sync`, 2026-06-17.

- **A** = current main / PR #169 header: `SyncPeriod = SlotNum` (= 8)
- **B** = proposed: `SyncPeriod = (SlotNum <= 2) ? SlotNum : SlotNum / 2` (= 4)

| case | S0=S1 | tiles | A `fa µs` | B `fa µs` | **kernel speedup (A→B)** | A speedup | B speedup | A TFLOP/s | B TFLOP/s | err_kernel (A=B) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| case1 | 1024 | 4 | 41.64 | 41.49 | +0.4% | 3.13× | 3.13× | 13.04 | 13.09 | 4.50e-05 |
| case2 | 2048 | 8 | 59.01 | 52.74 | **+10.6%** | 2.62× | 2.87× | 36.82 | 41.19 | 3.99e-05 |
| case3 | 4096 | 16 | 115.43 | 115.52 | −0.1% | 1.70× | 1.74× | 75.29 | 75.23 | 2.47e-05 |
| case4 | 8192 | 32 | 289.18 | 277.74 | +4.0% | 1.20× | 1.26× | 120.21 | 125.16 | 1.78e-05 |
| case5 | 16384 | 64 | 992.03 | 943.79 | **+4.9%** | 0.92× | 0.96× | 140.17 | 147.33 | 1.27e-05 |
| case6 | 32768 | 64 | 3520.91 | 3154.31 | **+10.4%** | 0.89× | **1.00×** | 157.97 | 176.33 | 1.76e-05 |
| case7 | 65536 | 128 | 13532.27 | 12028.19 | **+11.1%** | 0.89× | **1.01×** | 164.41 | 184.96 | 1.36e-05 |
| case8 | 131072 | 256 | 53116.38 | 48033.67 | **+9.6%** | 0.90× | **1.00×** | 167.54 | 185.27 | 9.07e-06 |

**Reading the table:**
- `kernel speedup (A→B) = (A_fa − B_fa) / A_fa`, measuring the speedup of the DSL kernel itself. `torch_npu`'s `fused_us` does not use our headers, so it is consistent across the two runs (±noise) and serves as a clean baseline.
- **Large S (`case6`–`case8`) recovers from 0.89–0.90× to parity with `torch_npu` (~1.00–1.01×, ~185 TFLOP/s)**; `case5` 0.92→0.96×; small S is neutral-to-slightly-up (`case2` also +10.6%, `case1`/`case3` within noise).
- `err_kernel` is **byte-identical** between A and B (host FP32 / `npu_fused` reference, ≤ 4.5e-05) → correctness is unchanged; the kernel is simply faster.

---

## Mechanism: why `SyncPeriod = SlotNum/2` is faster

The producer protocol in `TPush.hpp` is fully parameterized by `SyncPeriod` (`shouldWaitFree` / `shouldNotifyFree` / the destructor drain all use `% SyncPeriod`). flash_atten's GM-FIFO depth is `SlotNum = 8`:

- **`SyncPeriod = 8` (= full FIFO depth):** the Cube side fills **all 8 slots** before it checks/waits, then must hard-wait for the Vec side to drain a whole period (8 tiles) at once. Synchronization is **coarse and bursty** — Cube sprints a stretch, then stalls for Vec to catch up, and the Cube↔Vec pipeline overlap is broken.
- **`SyncPeriod = 4` (= SlotNum/2):** the sync frequency doubles, Cube and Vec stay **tightly coupled**, and in steady state the two advance in a smoother interleave with better overlap.

The regression **grows with S** (`case1` ≈ noise → `case6`–`case8` +10–11%) because at large S the steady-state per-tile loop dominates, and `SyncPeriod` is precisely the "metronome" for Cube↔Vec overlap inside that loop; coarsening it directly drags down steady-state throughput.

---

## Recommendation & caveats

Changing `SyncPeriod` back to `(SlotNum <= 2) ? SlotNum : SlotNum / 2` recovers ~10–12% at large S (verified correct for C2V / SlotNum=8).

⚠️ **Do not blindly revert globally.** Commit `014920a8` also introduced the destructor drain, the `FlagID+3 < 16` hardware assertion, and Ctrl-type support — the `SyncPeriod` bump **may have been made for the pending-free correctness of other configurations (DIR_BOTH / V2C / other SlotNum)**. Before reverting globally, verify push/pop synchronization correctness on those configurations/kernels. Possible safe paths:

1. Restore `SlotNum/2` **only on the C2V single-producer / single-consumer path**;
2. Expose `SyncPeriod` as a `TPipe` template parameter (defaulting conservatively to `SlotNum`), letting kernels opt into `SlotNum/2`;
3. Keep `SlotNum/2` as the default but add the regression test for the pending-free correctness scenario `014920a8` intended to fix.

---

## Reproduction

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/bin/ptoas-bin:$PATH      # ptoas 0.45
export PTOAS_ROOT=/usr/local/bin/ptoas-bin; unset PTOAS
cd kernels/python/flash_atten

# A: current header (SyncPeriod = SlotNum)
FA_SUMMARY_TSV=/tmp/A.tsv PTO_LIB_PATH=<repo-root> python3 run.py --timing sync

# B: copy include/ elsewhere, change only TPush.hpp:37 SyncPeriod to (SlotNum<=2)?SlotNum:SlotNum/2
FA_SUMMARY_TSV=/tmp/B.tsv PTO_LIB_PATH=<patched-include-root> python3 run.py --timing sync
```

> Each NPU command is submitted via `task-submit --device auto`; A and B run back-to-back on the **same card** to eliminate card-to-card variation.


---

## #173 [a2a3] FP32 TTRANS miscomputes / hangs (507018) when source validRow < 16 (non-multiple of Y_ELEM_OTHER=16)

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/173
- Created: 2026-06-18T07:59:33Z
- Updated: 2026-06-23T03:08:44Z
- Closed: 2026-06-23T03:08:44Z

### Body

## Summary

On a2a3, a FP32 `TTRANS` (tile transpose) whose **source `validRow` is not a multiple of `Y_ELEM_OTHER` (16)** — in particular `validRow < 16` — produces an **incorrect transpose**. It assembles cleanly (no ptoas/compile error); at runtime it either returns wrong values or hangs the AICore with error **507018**.

A `[8, 8]` FP32 transpose is the minimal trigger. A `[16, 8] -> [8, 16]` transpose (validRow = 16) is correct.

## Repro (pypto / a2a3)

Minimal kernel — load `[8,8]` FP32, transpose, store; golden is `x.T`:

```python
@pl.jit
def transpose_88(x: pl.Tensor[[8, 8], pl.FP32], y: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        xt = pl.load(x, [0, 0], [8, 8], target_memory=pl.MemorySpace.Vec)
        yt = pl.transpose(xt, axis1=0, axis2=1)
        pl.store(yt, [0, 0], y)
    return y
# golden: y = x.T ; compare ratio/allclose
```

Generated kernel: `TTRANS(dst, src, tmp)` with all three tiles `Tile<Vec, float, 8, 8, RowMajor, validRow=8, validCol=8>`.

## Observed (a2a3 device)

- `[8, 8]` FP32 transpose: **FAIL** — wrong output values (~28/64 elements mismatch in one run) or **507018** AICore error (deterministic on a fresh card in other runs). Undefined-behaviour-like; a failed run also poisons the card.
- `[16, 8] -> [8, 16]` FP32 transpose (validRow = 16): **PASS**.

## Root cause

`include/pto/npu/a2a3/TTrans.hpp`, `TTransOperation` (around line 224). For FP32, `yTileSizeElem = Y_ELEM_OTHER = 16` (line 27). With `validRow = 8`:

```cpp
int numSubTileY = validRow / yTileSizeElem;   // 8 / 16 = 0  -> full sub-tile path skipped
...
int remainY = validRow % yTileSizeElem;       // 8           -> only the Y-tail path runs
if (remainY > 0) {
    TransYTailTiles<...>(tmpPtr, srcPtr, tmpStride, numSubTileX, numSubTileY /*=0*/, remainY, srcStride);
}
```

So the whole transpose is handled by `TransYTailTiles` with `numSubTileY == 0`, and that tail path does not correctly transpose the 8 rows. (Same shape applies to the b8 path with `Y_ELEM_B8 = 32` for `validRow < 32`.)

## Expected

A FP32 (and b8) transpose with `validRow` smaller than / not a multiple of the row tile should still produce a correct transpose — or, if unsupported, fail loudly at assembly rather than silently miscomputing or hanging the AICore.

## Environment

- Target: a2a3 (Ascend910B)
- pto-isa: `e25732f0` (PTO-ISA/pto-isa, as consumed by the pypto runtime submodule)
- Surfaced from pypto issue hw-native-sys/pypto#1790 (the kernel split path produced `[8,8]` FP32 transposes).

---

## #178 [Bug] a2a3 MGatherRowImpl missing PtoSetWaitFlag<PIPE_MTE2,PIPE_S> -> index DMA race (507018 / wrong gather)

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/178
- Created: 2026-06-24T01:56:39Z
- Updated: 2026-06-24T03:09:39Z
- Closed: 2026-06-24T03:09:39Z

### Body

## Summary

On a2a3, `tile.mgather` in **row** mode produces wrong gathered rows or hangs
the AICore (507018), nondeterministically across runs. The same case flips
between a 507018 hang and wrong output values depending on timing, which points
to a data race rather than a deterministic compute error. **Elem** mode works
correctly.

## Root cause

In `include/pto/npu/a2a3/MGather.hpp`, `MGatherRowImpl` reads the index tile
from UB on the scalar pipe (`idxPtr[r]`) **without first waiting for the MTE2
DMA** (`pto.tload`) that fills that index tile.

Its prologue contains only:

```cpp
PtoSetWaitFlag<PIPE_V, PIPE_S>();
PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
```

and is **missing** `PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();`.

The sibling `MGatherElemImpl` has that MTE2->S wait, which is why elem mode
works. Without it, the scalar index reads race the index DMA, yielding
stale/garbage indices. The bad indices then cause either:

- wrong gathered rows (wrong output values), or
- out-of-bounds row addresses (AICore 507018 hang).

## One-line fix

Add the missing wait to the `MGatherRowImpl` prologue, right after the existing
MTE3->S wait, mirroring `MGatherElemImpl`:

```cpp
PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
```

## Evidence

Observed via PyPTO `tile.mgather` system tests on Ascend a2a3, each run
isolated on a freshly force-reset device:

- row FP32, leading idx `[0..15]`: AICore 507018 hang
- row FP32, reversed idx `[63..48]`: AICore 507018 hang
- row FP32, random idx: wrong output values, 480/512 elements mismatched
- row FP16, random idx: wrong output values, 480/512 elements mismatched
- row FP32 large (mem `[128,64]`, gather 32 rows): wrong output values,
  1088/2048 mismatched
- contrast: elem mode FP32 PASSES; one row INT32 run PASSED (race won that time)
- the SAME case flips between 507018 and wrong-values across runs (full-suite
  vs isolated) -> confirms a data race, not a deterministic compute error

## Environment

- Arch: a2a3
- ptoas: v0.46
- pto-isa: b2d297e1
- Flags: `--enable-insert-sync --pto-level=level3 --pto-arch a3`


---

## #179 a2a3 TFMOD/TFMODS (FP32) produce all-zero output on real a2a3

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/179
- Created: 2026-06-24T03:14:07Z
- Updated: 2026-06-27T01:34:26Z
- Closed: 2026-06-27T01:33:56Z

### Body

## Summary

On **real a2a3** hardware, `pto.tfmod` / `pto.tfmods` (FP32) produce an **all-zero** output tile. The host-side codegen and the ptoas lowering are both verified correct (see analysis below), so the defect appears to be in the **a2a3 device-side `TFMOD`/`TFMODS` implementation** (or the pto-isa revision the on-board runtime is built against).

`tile.maximum` (`TMAX`), built with a byte-identical kernel scaffold, runs correctly on the same a2a3 device — only the op differs.

## Symptom (device golden mismatch)

A 16×16 FP32 `fmod(lhs, rhs)` kernel returns all zeros:

```
Output 'out' does not match golden.
Mismatched elements: 212/256   (the 44 "matches" are exactly the elements whose golden value is 0)
rtol=1e-05, atol=1e-05
First mismatches:
    [2] actual=0.0, expected=-0.5
    [3] actual=0.0, expected=-3.0
    [4] actual=0.0, expected=-2.0
    ...
```

`tile.fmods` (tile×scalar) shows the same all-zero behavior across scalars {-2.5, 2.5, 3.0}.

## Environment / versions

| Component | Version |
| --- | --- |
| Arch | a2a3 (real NPU, CI on-board) |
| CANN | `cann-9.0.0` (`ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0`) |
| ptoas | **v0.45** (sha256 `d5e4380df7edd4d3eeb23502d57f923e4c912345eccceb5a9a0ec1aac3b5e7d4`); also reproduced the same lowering with **v0.46** |
| pto-isa | `origin/main` @ `5a40ed77`; `include/pto/npu/a2a3/TFmod.hpp` last changed in `47db50aa` (2026-04-18). On-board runtime pto-isa pin = whatever `SIMPLER_PTO_ISA_COMMIT` / scene-test `--pto-isa-commit` resolved to (please confirm the deployed revision) |
| pypto | branch `feat-add-ptoas-math-ops` @ `9d0eb9cd` (base `d6d699a8`) |

## Failing `.pto` (a2a3, tile-tile fmod)

```mlir
module attributes {pto.target_arch = "a2a3"} {
  func.func @kernel(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<f32>, %arg2: !pto.ptr<f32>) attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
  ...
  %a__ssa_v0 = pto.alloc_tile addr = %c0_i64 valid_row = %c16_index valid_col = %c16_index : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, ...>
  pto.tload ins(%lhs__ssa_v0_pview ...) outs(%a__ssa_v0 ...)
  %b__ssa_v0 = pto.alloc_tile addr = %c1024_i64 valid_row = %c16_index valid_col = %c16_index : ...
  pto.tload ins(%rhs__ssa_v0_pview ...) outs(%b__ssa_v0 ...)
  %c__ssa_v0 = pto.alloc_tile addr = %c0_i64 valid_row = %c16_index valid_col = %c16_index : ...
  pto.tfmod ins(%a__ssa_v0, %b__ssa_v0 : ...f32..., ...f32...) outs(%c__ssa_v0 : ...f32...)
  pto.tstore ins(%c__ssa_v0 ...) outs(%out__ssa_v0_pview ...)
  return
  }
}
```

Note `alloc_tile ... valid_row=16 valid_col=16` — the dst/src tiles carry valid_row = valid_col = 16, so `dst.GetValidRow()/GetValidCol()` should be 16 (not 0).

## Generated device `.cpp` (ptoas v0.45, `--pto-level=level3 --enable-insert-sync`)

```cpp
// v8 = lhs, v13 = rhs, v18 = dst, all Tile<Vec,float,16,16,RowMajor,...>(v5, v5)
TLOAD(v8, v12);
TLOAD(v13, v17);
... Tile<...> v18 = Tile<...>(v5, v5);
TASSIGN(v18, v19);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
TFMOD(v18, v8, v13);          // <-- produces all zeros on a2a3
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TSTORE(v22, v18);
```

`TFMOD(dst, src0, src1)` is lowered correctly (correct operand order, no tmp needed — unlike `TREM`/`TREMS` which take a tmp). Swapping `TFMOD` → `TMAX` in this exact scaffold passes on the same device.

## Why we believe this is pto-isa, not pypto / ptoas

1. **pypto codegen is correct**: the pypto-emitted `.pto` for `tile.fmod` is byte-identical to the working `tile.maximum` `.pto` except the op name (same tile alloc, valid_shape=16, layout, load/store, sync).
2. **ptoas lowering is correct**: both v0.45 and v0.46 lower `pto.tfmod` → `TFMOD(v18, v8, v13)`. The a5 TileLang template `share/ptoas/TileOps/tfmod_template.py` (`target="a5"`) is byte-identical between v0.45 and v0.46, and a2a3 does not use it — a2a3 resolves `TFMOD` from `pto/pto-inst.hpp` (pto-isa) at ccec compile time.
3. **a2a3 `TFmod.hpp` on origin/main looks correct** (`vdiv → vconv_f322f32z(trunc) → vmul → vsub`), so the deployed/pinned pto-isa revision used by the on-board runtime is the likely suspect.

## Questions

1. Is FP32 `TFMOD`/`TFMODS` on **a2a3** expected to be supported and correct at this point? (There has been recent churn: `e6d79157` add A3 TFMOD, `2a3db556` remove A3 fp16, `f7f22f30` remove A2A3 int16/int32.)
2. What pto-isa commit should the a2a3 on-board runtime be pinned to for a correct a2a3 TFMOD? Is `47db50aa`/`origin/main` known-good on a2a3, or is the all-zero a real defect in the a2a3 `TFMOD` device path?

## Repro

Minimal: assemble the `.pto` above with `ptoas <file> -o k.cpp --enable-insert-sync --pto-level=level3`, build the kernel against the on-board a2a3 pto-isa, run with `lhs` in [-6,6], nonzero `rhs` in [1.5,5.5]; expected `torch.fmod(lhs, rhs)`, actual all zeros.


---

## #182 [Bug] GM-FIFO auto-split uses CCE `get_subblockid()`, which is stale (0/0) inside PTO2-dispatched user kernels

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/182
- Created: 2026-06-27T01:31:05Z
- Updated: 2026-07-14T01:32:23Z
- Closed: 2026-07-14T01:32:23Z

### Body

### Component

Backend (ISA library / per-AIV auto-split address computation for the GM ring-buffer)

---

### Description

When the ISA library performs `TILE_UP_DOWN` / row-column auto-split for a `TPipe` GM ring-buffer, it uses the **CCE hardware builtin `get_subblockid()`** to compute each AIV lane's in-slot offset `subAIVOffset`:

- `include/pto/npu/a2a3/TPush.hpp:208` (Producer, V2C push)
  `subAIVOffset = get_subblockid() * gmValidR * gmValidC * sizeof(T);`
- `include/pto/npu/a2a3/TPush.hpp:435` (Consumer, C2V pop)
  `subAIVOffset = get_subblockid() * ConsM * ConsN * sizeof(T);`
- Final address: `addr = GM_SLOT_BUFFER + entryBase + subAIVOffset + entryOffset`
  (`TPush.hpp:216` / `TPush.hpp:440`)

The same pattern appears in many auto-split paths across both a2a3 and a5 (`TPush.hpp` / `TPop.hpp` / `TAlloc.hpp`), so it is a systemic assumption of the ISA library.

**The problem**: `get_subblockid()` reads the AICore **hardware sub-block register**, which is only meaningful at a genuine hardware kernel-launch entry. The `tensormap_and_ringbuffer` (PTO2) runtime does **not** launch user kernels per task. It launches **one persistent MIX kernel** per cluster (1 AIC + 2 AIV) and then dispatches user kernels onto those already-running cores through a software executor that **calls them via a raw function pointer** (`aicore_executor.cpp: kernel(args)`). The ISA auto-split runs *inside that dispatched user kernel*, and there `get_subblockid()` no longer reflects the lane — it returns **0 for both AIV0 and AIV1**. As a result `subAIVOffset` is constantly 0 for both lanes, and both map to the **same GM slot offset**:

- Producer side: AIV1's half is never written (overwritten by / colliding with AIV0);
- Consumer side: AIV1 reads a region it never wrote (uninitialized GM); the value is amplified by a downstream `TEXP` and overflows to `nan`, corrupting the output.

> **Important correction (measured on hardware, see Evidence):** `get_subblockid()` is **not** "never programmed". At the platform persistent-kernel entry (`src/.../platform/onboard/aicore/kernel.cpp:93`) it is **correctly 0/1** and the runtime relies on it there. It only goes stale to **0** by the time a user kernel is reached through the executor's function-pointer call. So the bug is specifically: **the ISA auto-split reads this builtin from a context (the dispatched user kernel) where it is no longer valid.**

Crucially, the runtime's software lane id is **itself a snapshot of `get_subblockid()` taken while it was still valid** — not an independent source. At the persistent-kernel entry each core computes `block_idx = get_block_idx()*get_subblockdim() + get_subblockid() + get_block_num()`, encoding the entry-time sub-block id into its handshake slot (the two AIVs of a cluster land on adjacent slots that differ only by the `+ get_subblockid()` term). The AICPU scheduler then discovers the AIV cores in slot order, pairs them, and writes `GlobalContext.sub_block_id` = 0/1 — which therefore **equals the entry-time `get_subblockid()`**. Kernels read this snapshot via `get_sub_block_id(args)` and it stays correct in the user kernel. The ISA library instead re-reads `get_subblockid()` *live*, after it has gone stale.

> So the two values are the **same `get_subblockid()`, read at two different times** — not two independent lane sources. That difference in read time is exactly why the runtime and the ISA library disagree:
>
> | When / where `get_subblockid()` is read | AIV0 | AIV1 | Value |
> | --- | :--: | :--: | --- |
> | Runtime: at the **persistent-kernel entry**, captured into `GlobalContext.sub_block_id`, then read in the user kernel via `get_sub_block_id(args)` | 0 | **1** ✅ | correct — register still valid at entry |
> | ISA library: **live, inside the dispatched user kernel** | 0 | **0** ❌ | stale — register already clobbered |

#### Minimal reproducing semantics (flash-attention decode kernel)

The cube produces a 16x128 QK^T score block (Acc f32) -> a `TPipe<DIR_BOTH>` with `TILE_UP_DOWN` splits it into two 8x128 halves, AIV0 handling rows 0-7 and AIV1 handling rows 8-15. Each lane runs online softmax and pushes back its 8x128 P (bf16) for the PV matmul. With `subAIVOffset` constantly 0, AIV1's half is invalid.

---

### Steps to Reproduce

1. On a2a3 hardware, run a MIX kernel that uses `TPipe<DIR_BOTH>` + `TILE_UP_DOWN` auto-split (cube -> 2xAIV up/down split, e.g. flash-attention decode; see the qwen3_14b_decode example under `tensormap_and_ringbuffer` in the `simpler` repo), dispatched by the PTO2 runtime per cluster (1 AIC + 2 AIV).
2. Let the ISA library's internal `subAIVOffset` (derived from `get_subblockid()`) drive the per-lane GM offset, with **no** kernel-side per-lane compensation.
3. Run the golden comparison.

---

### Expected Behavior

The ISA library's `TILE_UP_DOWN` / row-column auto-split should place AIV0 and AIV1 in their respective GM slot halves under the PTO2 runtime, without the kernel author manually patching the offset; the golden comparison should PASS.

---

### Actual Behavior

In the dispatched user kernel `get_subblockid()` returns 0 for both AIV0 and AIV1 -> `subAIVOffset` is constantly 0 -> both lanes alias the same GM slot offset; AIV1's half is unwritten / reads uninitialized GM, which overflows to `nan` through `TEXP` and corrupts the output.

Measured on a2a3 hardware (qwen3_14b_decode):

| Configuration | Result | Symptom |
| --- | --- | --- |
| Per-lane offset supplied from the runtime software lane | ✅ **PASS** | -- |
| ISA-internal `subAIVOffset` (`get_subblockid()`) only | ❌ **FAIL** | `AssertionError: Golden mismatch on 'k_cache': max_diff=nan, rtol=0.05, atol=0.1` |

---

### Evidence (on-hardware instrumentation)

The AICore was instrumented to stash `get_subblockid()` / `get_block_idx()` (raw CCE builtins) into the handshake; the AICPU scheduler logged them per core. Plus three PASS/FAIL probes in the user kernel. Findings:

- `get_block_idx()` is **per-MIX-block** (identical for the AIC and both AIVs of a cluster): values `0,0,0 / 1,1,1 / ... / 23,23,23`.
- At the **persistent-kernel entry**, `get_subblockid()` is **0/1** (proven indirectly but unambiguously: the two AIVs of a cluster register at *adjacent, distinct* handshake slots, e.g. 24 and 25; since `block_idx = get_block_idx()*2 + get_subblockid() + N` and `get_block_idx()*2 + N` is even, only the `+ get_subblockid()` term can produce the odd slot — so one AIV had `get_subblockid()=1` at entry).
- In the **user kernel** (and when re-read later in the executor), `get_subblockid()` is **0 for both lanes** (3 PASS/FAIL probes: `lane=get_subblockid()` FAILs with `nan`; `lane=sub_block_id - get_subblockid()` PASSes ⇒ the builtin contributes 0; instrumented read in `aicore_execute` confirms 0).

Conclusion: the sub-block register is valid at launch entry and **clobbered/stale** by the time the user kernel runs.

#### Root cause summary

`get_subblockid()` (CCE hardware register) is only valid at the genuine kernel-launch entry. PTO2 dispatches user kernels onto persistent cores via a software function-pointer call, where that register is stale (0/0). The runtime had already captured the **entry-time** `get_subblockid()` into software (`GlobalContext.sub_block_id`, read via `get_sub_block_id(args)`), so it still has the correct 0/1. The ISA auto-split, however, executes in the dispatched context and re-reads the **same builtin live**, getting the stale 0/0 — so AIV0 and AIV1 land on the same (0) offset. Same builtin, different read time; the ISA library reads it at the wrong time and never uses the runtime's already-captured snapshot.

---

### Suggested fix direction (implemented + validated)

The ISA library must not read `get_subblockid()` from inside the dispatched user kernel. It should use the runtime software lane id instead. **It cannot read `GlobalContext`/`args` directly** — the ISA library is header-only with no access to `args`, and simpler's AICore loader forbids any hidden per-core global (a `[[block_local]]` global emits a `.rela.text` relocation the loader rejects). So the lane must be **threaded in from the kernel through the `TPipe` object** (stack member, folds into a single `.text` via `always_inline` — no relocation).

Implemented and validated on a2a3 (qwen3_14b_decode now PASSes with no kernel-side offset math):

- `TPipe` gains `setSubBlockId(int32_t lane)`; `Producer`/`Consumer` store it and expose `laneId()` returning the stored lane when set, else falling back to `get_subblockid()` (preserves native CANN/AscendC dispatch, where the builtin is valid).
- All auto-split sites use `laneId()` instead of `get_subblockid()`: a2a3 `TPush.hpp`/`TPop.hpp`/`TAlloc.hpp` (and the symmetric a5 files).
- The kernel passes the runtime lane once: `pipe.setSubBlockId(get_sub_block_id(args))` (codegen should emit this for MIX `TILE_UP_DOWN` pipes).

Note: this **replaces** the broken builtin rather than adding to it, so there is no additive "2x offset" hazard. (An earlier kernel-side workaround that *added* `sub_block_id*bytes` via `setEntryOffset` did carry that hazard; the threaded approach removes it.)

---

### Git Commit ID

`e722679b6eb286de84ce0d668bd19e09eafee929` (the same logic is also present in the earlier `8e436661`)

---

### NPU Kind

Ascend 910B

---

### Host Platform

Linux (aarch64)

---

### Additional Context

- Same fault family as issue #900 / PR #899 ("CCE builtin `get_subblockid()` reads 0 for both AIV0/AIV1 in PTO2 user kernels, partial output is 0/wrong"). That case was a kernel directly misusing the builtin; this one is the **ISA library internally** misusing the same builtin in its auto-split.
- Related interface: `get_sub_block_id(args)` reads `GlobalContext.sub_block_id` (0/1) — the scheduler's software capture of the **entry-time** `get_subblockid()`; it is the reliable lane identity inside PTO2 user kernels precisely because it was snapshotted before the register went stale.
- Blast radius: every MIX kernel that relies on ISA-library GM-FIFO auto-split (`TILE_UP_DOWN` and row/column split) hits this under the PTO2 runtime.


---

## #183 [Bug] fa_fused DIR_BOTH+TILE_UP_DOWN cross-core tile-pipe stalls under CPU sim (a2a3sim) — passes on real a2a3

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/183
- Created: 2026-06-29T07:01:04Z
- Updated: 2026-07-08T01:34:24Z
- Closed: 2026-07-08T01:34:24Z

### Body

> **History.** This issue originally reported a CPU `Consumer::setentryOffset`
> casing typo — that was **already fixed** in `e066bbd7` *"[CPU-SIM] Fix
> Consumer::setEntryOffset casing to match Producer and NPU header"*. After
> pinning to a fixed rev (`32064ca0`) compilation succeeds; this issue now
> tracks the **real remaining a2a3sim blocker** that surfaces once it compiles.

## Summary

The `qwen3_14b_decode` fused-attention kernel **`fa_fused`** — a persistent
grid-stride kernel driving a cross-core `TPipe<0, Direction::DIR_BOTH, 8192, 4,
4, false>` GM-FIFO with `TPUSH`/`TPOP` and `TileSplitAxis::TILE_UP_DOWN` between
the AIC producer and the two AIV-lane consumers — **never makes forward progress
under the CPU functional simulator (`a2a3sim`)**. The cluster cores enter the
kernel and never signal completion, so the scheduler stalls and the graph times
out (`rc=-100`). The **same case passes deterministically on real a2a3 silicon**
(output + both layers' KV-cache match the torch reference).

## Repro

```bash
# onboard — PASSES
python examples/a2a3/tensormap_and_ringbuffer/qwen3_14b_decode/test_qwen3_14b_decode.py -p a2a3   # ✅
# CPU sim — STALLS / times out
python examples/a2a3/tensormap_and_ringbuffer/qwen3_14b_decode/test_qwen3_14b_decode.py -p a2a3sim # ❌
```

Case `StressBatch16Seq3500` (BATCH=16, seq_len=3500, 2 fused decode layers).
Reproduced **3×**, including against an isolated worktree pinned at `32064ca0`
(rules out the header-flip / stale-cache confound).

## Diagnostic (a2a3sim scheduler stall snapshot)

```
PTO2 total submitted tasks = 713, already executed 23
[STALL] reason=scheduler_timeout idle_iterations=281813
SUMMARY completed=30/713 last_progress_iteration=28 scan_ready=0 scan_waiting=682 scan_running=1

TASK task_id=...610 state=RUNNING fanin 5/5 kernels=[aic:16 aiv0:17 aiv1:17]
CLUSTER cluster_id=0 aic=core0 (busy kernel=16 task=...610 cond_reg_state=ack)
                    aiv0=core24(busy kernel=17 …)  aiv1=core25(busy kernel=17 …)
# every cluster assigned ...610 has all of {aic, aiv0, aiv1} "busy" inside
# kernel 16/17 and never completes; 682 downstream tasks WAIT missing_deps=1 on it.
```

Forward progress freezes after **30/713** tasks. The single `RUNNING` task is the
fused-attention task; it never returns on **any** cluster. After the scheduler
timeout the cores never acknowledge exit (`Emergency shutdown: N cores did not
acknowledge exit`) and the runtime returns `rc=-100`.

`kernel=16/17` map to `fa_fused` in this case's `CALLABLE`:

| func_id | name | source | core |
| ------- | ---- | ------ | ---- |
| 16 | `fa_fused_aic` | `kernels/aic/fa_fused_aic.cpp` | aic |
| 17 | `fa_fused_aiv` | `kernels/aiv/fa_fused_aiv.cpp` | aiv (2 lanes) |

## Ruled out

- **`subblock_dim` / dual-lane gating.** The CPU `TPush.hpp` dual-lane C2V
  protocol that depends on `get_subblockdim() >= 2`
  (`IsDualLaneC2VActive`) is only taken for `TILE_NO_SPLIT`. `fa_fused` uses
  `TILE_UP_DOWN`, whose per-lane offset comes from
  `GetSplitRowOffset = get_subblockid() * Rows/2` — i.e. it uses
  **`get_subblockid()`** (which the sim host injects correctly as 0/1 via
  `pto_sim_get_subblock_id`), **not** `get_subblockdim()`. Forcing
  `get_subblockdim()=2` in sim does not change the stall. So this is *not* the
  same mechanism as #182.

## Open / where help is needed

Root cause is not yet pinned. The cores spin **inside** the kernel (a
spin-wait in the CPU `TPUSH`/`TPOP` `TILE_UP_DOWN` / `DIR_BOTH` handshake that is
never satisfied), not in the scheduler. gdb can't unwind the simulated-core
stacks to localize the exact wait.

The hang lives in pto-isa's **CPU** cross-core tile-pipe implementation
(`pto/cpu/TPush.hpp` / `TPop.hpp`) as driven by the sim host's per-core identity
(`pto_sim_get_subblock_id`) and per-cluster pipe shared state
(`pto_sim_get_pipe_shared_state`). It is therefore a pto-isa-CPU-primitive vs.
sim-host-integration **boundary** question — "passes on board" localizes the
defect to the sim-only path but does not by itself prove the ISA side.

Questions for maintainers:
- Is the `DIR_BOTH` + `TILE_UP_DOWN` cross-core tile-pipe expected to work under
  the CPU functional sim with a **persistent grid-stride** kernel and **PTO2
  multi-cluster dispatch** (1 AIC + 2 AIV per cluster, many clusters)?
- What does the CPU `TPUSH`/`TPOP` `TILE_UP_DOWN` producer/consumer rendezvous
  require from the sim host beyond `subblock_id` + per-cluster pipe shared state?

Full scheduler stall log and a stripped-down repro available on request.


---

## #184 fix(cpu-sim): ping-pong TPUT_IMPL missing AtomicType template parameter

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/184
- Created: 2026-06-29T14:38:16Z
- Updated: 2026-07-02T02:45:06Z
- Closed: 2026-07-02T02:45:06Z

### Body

## Problem

The CPU simulator ping-pong (double-buffered) overload of `TPUT_IMPL` in `include/pto/cpu/comm/TPut.hpp` is missing the `AtomicType atomicType` template parameter, but the public wrapper in `include/pto/comm/pto_comm_inst.hpp` always passes it:

```cpp
// pto_comm_inst.hpp:67 — passes 4 template args
::pto::comm::TPUT_IMPL<GlobalDstData, GlobalSrcData, TileData, atomicType>(
    dstGlobalData, srcGlobalData, pingTile, pongTile);
```

```cpp
// TPut.hpp:25 — only accepted 3 template args (broken)
template <typename GDD, typename GSD, typename TD>
PTO_INTERNAL void TPUT_IMPL(GDD &dst, GSD &src, TD &ping, TD &pong) { ... }
```

This causes a template argument mismatch when compiling any kernel that uses `TPUT` with `pipeline=True` (double-buffered) on the CPU simulator (`a5sim` / `a2a3sim`).

## Reproduction

```bash
pytest tests/st/distributed/test_l3_put.py::TestL3Put::test_ring_shuffle_pipeline \
  -v --forked --platform=a5sim --device=0,1,2,3
```

Fails with:
```
error: wrong number of template arguments (4, should be 3)
```

## Why CI passes

CI runs distributed tests on real NPU hardware (not sim), so the NPU `TPut.hpp` implementations (which handle `AtomicType` correctly) are used instead of the CPU sim path.

## Fix

Add `AtomicType atomicType` as a template parameter to the ping-pong `TPUT_IMPL` and forward it to `Copy_Data`:

- `include/pto/cpu/comm/TPut.hpp:25`
- `include/pto/cpu/TGet.hpp:71` (duplicate definition in same namespace)

---

## #188 [Bug] TPUSH destructor drain off-by-one on even push counts causes AICore 507018 (014920a8 regression)

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/188
- Created: 2026-07-02T04:04:20Z
- Updated: 2026-07-03T09:04:01Z
- Closed: 2026-07-03T09:04:01Z

### Body

### Summary

Commit `014920a8` "Clean pending free signals during TPUSH" changed the `TPipe` destructor drain logic and removed the constructor pre-free. For a `SlotNum=2` C2V/V2C pipe with an **even number of TPUSH** operations, the destructor's computed `drainCount` is off-by-one (yields 1, should be 0), so the destructor's `prod.allocate()` (`wait_flag_dev`) waits for a free flag that never arrives -> AICore running-stalled -> `aclrtSynchronizeStreamWithTimeout` 507018. Odd push counts yield `drainCount=0` and are unaffected.

Stably reproduced on simpler's `spmd_paged_attention` case (MIX AIC+AIV cooperative TPUSH/TPOP pipeline, `FIFO_DEPTH=2`), and confirmed via bisect + revert verification.

### Root-cause commit

- First bad: `014920a894f0ac47bcc453f889f1fd1a3080f573` "Clean pending free signals during TPUSH" (csjlchen, 2026-05-22)
- The commit self-notes: `Tested: Pending final sync verification.` / `Not-tested: Full NPU hardware validation pending final sync verification.` — i.e. it was merged without completing NPU hardware validation.
- The follow-up `1d873362` "Update SyncPeroid and drain counts." attempted a fix (reverted `SyncPeriod` to `(SlotNum<=2)?SlotNum:SlotNum/2` and tweaked the `drainCount` formula) but did **not** fix the even-push off-by-one; versions carrying `1d873362` (e.g. simpler's pinned pto-isa `e722679b`) still 507018 on even cases.

### Changed sites (`include/pto/npu/a2a3/TPush.hpp`, a5 changed the same way)

| Site | pre-014920a8 | 014920a8 |
| --- | --- | --- |
| Constructor | `for(i<SyncPeriod) cons.free()` pre-free | `{}` removed |
| Destructor | `for(i<SyncPeriod) prod.allocate()` fixed count | `drainCount = numPopFree - numPushWait` computed |
| `shouldWaitFree` (SlotNum==1) | `return true` | `return tileIndex > 0` |

For `SlotNum=2`, `SyncPeriod=2`; mid-stream free send/wait is self-balancing, so there is **no pending free signal to clean**. 014920a8 both removed the constructor pre-free (-2 sends) and made the destructor drain computed: to keep the flag balance the destructor should wait 0; but the formula yields 1 for even `prod.tileIndex` -> one extra wait for a non-existent free flag -> hang. Odd yields 0 -> fine.

### Reproduction

Using simpler's `spmd_paged_attention`; `n_blocks = context_len / block_size` is the per-pipe TPUSH count. `SlotNum=2` (`FIFO_DEPTH=2`), `DIR_C2V`, `TILE_UP_DOWN`.

Parity matrix (a3 onboard, simpler main + simpler's pinned pto-isa `e722679b` which contains 014920a8, `--skip-golden`):

| case | context_len | n_blocks | parity | result | time |
| --- | --- | --- | --- | --- | --- |
| Even2  | 256  | 2  | even | **FAIL 507018** | 65.97s |
| Odd3   | 384  | 3  | odd  | **PASS** | 16.55s |
| Even32 | 4096 | 32 | even | **FAIL 507018** | 74.71s |
| Odd31  | 3968 | 31 | odd  | **PASS** | 14.73s |
| Even64 | 8192 | 64 | even | **FAIL 507018** | 74.62s |
| Odd63  | 8064 | 63 | odd  | **PASS** | 15.05s |

Error signature:
```
RuntimeError: run failed with code 507018
aclrtSynchronizeStreamWithTimeout (AICPU) failed: 507018
PTO2 scheduler timeout sub_class=S1:running-stalled completed=0/1 running=1 orch_done=1 stuck_task_id=4294967296
```

Repro cases are submitted to simpler: https://github.com/yanghaoran29/simpler/tree/spmd_pa_error (`tests/st/a2a3/tensormap_and_ringbuffer/spmd_paged_attention/test_spmd_paged_attention.py`, case names `Even2/Odd3/Even32/Odd31/Even64/Odd63`, manual). To run:
```bash
pytest .../test_spmd_paged_attention.py --platform a2a3 --device <dev> \
  --case TestPagedAttentionUnrollTpushPop::Even2 --manual include --skip-golden -v
```

### Bisect (simpler fixed at ecfb1663, only pto-isa varies, test = spmd_paged_attention on a3)

- good: pto-isa `ddafa8da` (before 014920a8) -> PASS
- bad: pto-isa `016396b5` (contains 014920a8) -> FAIL 507018
- isolation: simpler@ecfb1663 + pto-isa 016396b5 -> FAIL (i.e. pto-isa-version-dependent, not simpler-code-dependent)
- `git bisect` ddafa8da..016396b5 (125 commits, 6 steps) -> first bad = `014920a8`

### Revert verification (reverting 014920a8's a2a3 TPush.hpp changes on simpler's pinned pto-isa `e722679b`)

Restoring constructor pre-free + fixed destructor `for(i<SyncPeriod) prod.allocate()` + `shouldWaitFree return true`:
- Even2 / Even64 -> **PASS**

I.e. reverting 014920a8's TPUSH sync changes makes the even cases pass again, confirming the root cause is in those three sites.

### Expected behavior

A `SlotNum=2` pipe should complete for any push count (even or odd) without 507018.

### Actual behavior

Even push counts -> the destructor waits for a free flag that never arrives -> 507018 running-stalled.

### Environment

- pto-isa commit: first bad `014920a894f0ac47bcc453f889f1fd1a3080f573`; verified-bad version: simpler's pinned pto-isa `e722679b`; good `ddafa8da`
- NPU: Ascend A2/A3 (reproduced a3 onboard)
- Host: Linux aarch64
- simpler repro branch: https://github.com/yanghaoran29/simpler/tree/spmd_pa_error

### Possible fixes

- Reverting 014920a8's changes (restore constructor `for(i<SyncPeriod) cons.free()` pre-free + fixed destructor `for(i<SyncPeriod) prod.allocate()` + `shouldWaitFree return true`). Verified: even cases pass after the revert. (Whether this re-introduces whatever 014920a8 was trying to fix — the "split-pipe sync state leak across producer handoffs" — should be checked against the scenario 014920a8 referenced.)
- Track pending free-flag credits explicitly (consumer `free()` increments, producer `allocate()` decrements); the destructor drains only the actual remaining credits. This is config-independent and correct by construction, and realizes 014920a8's intent of cleaning pending free signals.
- If keeping 014920a8's no-pre-free style: destructor drain = 0 (with no constructor pre-free, mid-stream send/wait is self-balancing and there is nothing to drain); needs separate validation for `SlotNum=1` / `DIR_BOTH` / `TILE_LEFT_RIGHT`.
- Fallback: make the destructor `prod.allocate()` non-blocking / only-wait-if-credit, so a miscount cannot deadlock.

Any fix should be validated against the parity matrix above plus paged_attention Case1, qwen3, and a5 bgemm on a3/a5 boards.

### Related

- pto-isa `1d873362` "Update SyncPeroid and drain counts." (did not fix the even case)
- simpler repro branch `yanghaoran29/simpler@spmd_pa_error` (commit `ddb5fb64` adds cases, `8a389654` removes the module skip)
- 014920a8 references GitCode `00d3220d02d0c4eb018baa5c924dd5a36ac43318`

---

## #195 DIR_BOTH split-tile pipe C2V/V2C wrong-slot fix (#193) not ported to the a2a3 NPU device path → cross-core corruption on real hardware

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/195
- Created: 2026-07-09T08:51:38Z
- Updated: 2026-08-03T13:24:15Z
- Closed: 2026-07-12T09:41:00Z

### Body

## Summary

[PR #193](https://github.com/hw-native-sys/pto-isa/pull/193) ("fix(backend): Separate C2V/V2C cursors in DIR_BOTH split tile pipe", *Fixes #183*) fixed the DIR_BOTH split-tile-pipe wrong-slot bug — but **only in the CPU-SIM path** `include/pto/cpu/TPush.hpp` (its testing was on `a2a3sim`). The **real-hardware NPU device path** `include/pto/npu/a2a3/TPush.hpp` never received the equivalent fix and still has the bug on device, causing non-deterministic cross-core corruption on a2a3 hardware.

Downstream tracking issue (end-to-end DSL repro): [hw-native-sys/pypto#1981](https://github.com/hw-native-sys/pypto/issues/1981).

## Symptom (real a2a3 hardware)

A kernel that uses a DIR_BOTH C↔V split-tile pipe (`pl.split_aiv`, `TILE_UP_DOWN` — i.e. `aiv_shard`/`aic_gather`) inside a repeated MIX `pl.spmd` + `system.syncall(core_type="mix")` loop (e.g. qwen3-14b fused decode over 40 layers) produces **non-deterministic numerical corruption** under load, clustered in the block every core cross-reads. A single dispatch is deterministic-correct; it only manifests across repeated dispatches under load, and is timing/card-dependent (some cards lose the race far more than others).

## Root cause on the device path (`include/pto/npu/a2a3/TPush.hpp`)

For `DIR_BOTH`:

- C2V (`pushAcc2GMFiFo` / `popVecTileFromGMFiFo`) and V2C (`pushVec2GMFiFo` / `popMatTileFromGMFiFo`) **both index the same GM ring** as `GM_SLOT_BUFFER + (tileIndex % SLOT_NUM) * SLOT_SIZE`. `entryOffset` is never set (no caller of `setEntryOffset`) → always 0. So **C2V and V2C share the same physical GM slots.**
- But the two directions use **separate free-flags** (C2V frees on `FlagIDPlusOne`, V2C on `FlagIDPlusThree`) and separate `tileIndex` cursors. So when a producer reuses a ring slot, it only waits for **its own** direction's consumer to free that slot — **not the other direction's consumer** that last touched the same physical slot.
- Under load, a later C2V push can overwrite a slot before the V2C consumer has read the previous V2C payload (or vice versa) → **cross-direction wrong-slot / stale read** → the corruption.

This is exactly the class of bug #193 fixed in CPU-SIM (separate C2V/V2C consumer cursors + compile-time split-lane id). The device path additionally still selects the split-lane GM offset via **runtime `get_subblockid()`** (lines ~211/214/438/441/548) instead of the compile-time split lane — the other half of what #193 flagged.

## Why it is NOT a memory-visibility issue

We tried four separate memory fences on the device path (`dsb(DSB_DDR)` in `SYNCALL_IMPL`, `dsb` in `TPush::record()`, `dcci` before the pop `TLOAD`, and `pipe_barrier(PIPE_ALL)`); none of them fixed it. It is a **wrong-slot / overwrite** bug, not a stale-data-visibility bug, so fences cannot help.

## Reproduction & evidence

- A ~130-line minimal kernel (see pypto#1981) reproduces it: one MIX `pl.spmd(24)` per "layer" × 40, two hard `syncall(core_type="mix")` barriers, a cube matmul, and a `split_aiv` UP_DOWN C↔V exchange.
- A **no-split control** (identical two barriers + cube + cross-core block-0 read, but *without* the C↔V pipe) is deterministic-correct 9/9 on the same losing card, while the split_aiv version fails → confirms the bug is the DIR_BOTH pipe, **not** the barrier and **not** a bad card.
- Same card, same window, back-to-back, with isolation verified (a deliberately-wrong-slot device build fails 3/3 while the base build passes):
  - **Unfixed device path: 35/50 FAIL (70%)**
  - **With C2V and V2C separated onto disjoint physical GM slots: 50/50 PASS**

## Suggested fix

Port #193 to the NPU device path `include/pto/npu/a2a3/TPush.hpp`:

1. Separate the C2V and V2C consumer cursors / GM slot regions so the two directions never share a physical slot (or make each direction's producer also gate on the *other* direction's free when a slot is shared).
2. Resolve the split-lane GM offset from the compile-time split lane (as in `cpu_pipe::GetSplitLaneId<Split>()`), not runtime `get_subblockid()`.
3. Mirror the CPU ST coverage added in #193 (`tests/cpu/st/testcase/tpushpop_vc`) with a **device** ST test for interleaved DIR_BOTH `TILE_UP_DOWN` traffic.

## Environment

a2a3; ptoas 0.48; pto-isa pinned `83d01313` (managed clone actually ran `ea34d7a5`). The device pipe code is unchanged between those two revisions w.r.t. this bug — only #193's CPU-SIM file differs — so the device path lacks the fix in both.


---

## #197 [Bug] soft SYNCALL busy-spins up to 1e6 iterations (multi-second latency) when blocks arrive in waves

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/197
- Created: 2026-07-14T03:46:10Z
- Updated: 2026-07-17T01:35:52Z
- Closed: 2026-07-17T01:35:52Z

### Body

## Background

While experimenting with folding a MoE dispatch/combine `wait` handshake into a
neighbouring `pl.spmd` launch using `pl.system.syncall(mode="soft", core_type="aiv_only", ...)`
at **partial occupancy** (`pl.spmd(N_LOCAL=32)`, *without* `sync_start=True`) in
`pypto-lib`'s `models/deepseek/v4/moe.py`, single SPMD tasks took **~1.25 s** on
device (observed makespan 1253 ms for one `combine`/`dispatch` SPMD task, vs
sub-ms expected). The kernel still produced correct results — it was pure latency,
not a wrong answer — which pointed at a spin/timeout inside the soft barrier
rather than a data bug.

Reproduction environment:

| Component | Version |
|---|---|
| pypto-lib | `a895306` (branch: `feat/moe-syncall-fold-poc`) |
| pypto | `ccbf2870` (branch: `main`) |
| simpler | `438d5cb1` (branch: `detached`) |
| ptoas | `0.48` |
| pto-isa | `83d01313` |
| CANN | not detected |

Diagnosis: **pto-isa** — the soft (GM-polling) `SYNCALL` barrier busy-spins for
up to `SYNCALL_SOFT_MAX_POLL_ITERATIONS = 1000000` iterations when participating
blocks arrive late (e.g. a partial-occupancy SPMD launch without `sync_start=True`,
where the runtime dispatches blocks in waves). Early blocks spin at the barrier
waiting for blocks that have not yet been dispatched, producing multi-ms to
multi-second task latency.

**Still present on latest `origin/main`.** Verified against `ecb6c303` (48
commits ahead of the pinned `83d01313`): the spin loop, the
`SYNCALL_SOFT_MAX_POLL_ITERATIONS = 1000000` cap, and the `PTO_CPU_ASSERT` +
`break` soft-fail are all unchanged. The two intervening commits that touch
`SyncAll.hpp` (`bdeb4e7f`, `756787a9`) add new-arch / tinsert coverage and do not
alter the soft-barrier logic.

### Details

The soft barrier is implemented in `include/pto/npu/a2a3/SyncAll.hpp` and
`include/pto/npu/a5/SyncAll.hpp` (both platforms). `SYNCALL_SOFT_AIV_BARRIER`
(and the AIC / mix counterparts) does:

```cpp
int32_t pollCnt = 0;
while (true) {
    // read all totalBlks slots from GM, count how many have arrived (>= curVal)
    if (readyCnt >= totalBlks) break;              // all blocks arrived
    ++pollCnt;
    if (pollCnt >= SYNCALL_SOFT_MAX_POLL_ITERATIONS) {   // = 1000000
        PTO_CPU_ASSERT(false, "SYNCALL soft barrier timeout - possible deadlock");
        break;                                     // give up, proceed anyway
    }
}
```

Constants (`include/pto/common/type.hpp`):

```cpp
constexpr int32_t SYNCALL_SOFT_SLOT_INT32 = 8;
constexpr int32_t SYNCALL_SOFT_BACKOFF_THRESHOLD = 16;
constexpr int32_t SYNCALL_SOFT_MAX_POLL_ITERATIONS = 1000000;
```

Each poll iteration re-reads the whole GM workspace + a `pipe_barrier(PIPE_ALL)`,
so a full 1,000,000-iteration spin costs on the order of hundreds of ms to
seconds — matching the observed ~1.25 s.

**Trigger.** Soft syncall is the recommended path for **partial occupancy** (the
`HardSyncallOccupancy` verifier on the pypto side explicitly suggests
`mode="soft"` as the partial-occupancy alternative to hard). But if the enclosing
SPMD launch is not `sync_start=True`, the runtime may dispatch blocks **in waves**:
the first wave reaches the barrier and spins waiting for later blocks, which are
themselves waiting for the spinning cores to free up — a near-deadlock resolved
only by the 1e6-iteration timeout. This is the **same** wave-dispatch hazard that
hard syncall documents and guards against with `sync_start=True`, but for soft it
is neither documented nor guarded.

### Concerns / requested changes

1. **Documentation gap.** `docs/en/dev/ir/05-operators.md` (and the pto-isa
   `SYNCALL` docs) state the soft form "works at partial occupancy" but do **not**
   warn that it still requires `sync_start=True` (co-resident blocks) to avoid the
   wave-dispatch spin. Hard syncall documents this; soft should too.

2. **No guard.** Hard syncall has the `HardSyncallOccupancy` verifier (pypto
   issue #1935) that fails at compile time on a missing `sync_start`. Soft has no
   equivalent, so a partial-occupancy soft launch without `sync_start` silently
   degrades to a multi-second spin at runtime.

3. **Soft-fail masks the problem.** On timeout the barrier only does
   `PTO_CPU_ASSERT(false, ...)` then `break`s and proceeds; in non-debug builds
   the assert may be a no-op, so the failure surfaces only as extreme task latency
   (not an error). Consider a louder / non-maskable timeout signal, and/or a
   smaller default poll cap with an explicit runtime error.

Reproduction path (for reference, though the defect is analysable directly from
`SyncAll.hpp`): a `pl.spmd(32)` launch with `pl.system.syncall(mode="soft",
core_type="aiv_only", gm_workspace=ws, used_cores=32)` and **no** `sync_start=True`
exhibits the multi-second spin; adding `sync_start=True` (co-resident blocks) is
expected to remove it.

---

## #198 [BUG][TSTORE] ColMajor Vec[8,1] is incorrectly stored contiguously into an ND strided column

- State: open
- URL: https://github.com/hw-native-sys/pto-isa/issues/198
- Created: 2026-07-20T14:10:19Z
- Updated: 2026-08-10T02:36:15Z
- Closed: 

### Body

## Summary

On real a2a3 hardware, TSTORE ignores the destination row stride when storing an FP32 ColMajor Vec tile of shape [8,1] into a non-contiguous column view of an ND row-major tensor. Instead, it writes the eight elements as one contiguous 32-byte burst.

a2a3sim honors the destination stride and produces the correct result. The same kernel compiles, launches, and completes on real a2a3 hardware, but its output values are incorrect. This is a silent wrong-result issue.

The reproducer depends only on PyTorch, PyPTO, the PyPTO runtime, and PTO-ISA. It contains no higher-level framework logic.

## Minimal trigger

```text
Source tile:
  TileType::Vec
  dtype = FP32
  shape = [8, 1]
  layout = BLayout::ColMajor

Destination tensor:
  shape = [8, 2]
  layout = ND / row-major

Destination view:
  ws[:, 1]
  shape = [8, 1]
  element strides = [2, 1]

Operation:
  TSTORE(ws[:, 1], source_tile)
```

Adjacent elements in the destination column are two FP32 elements apart, not one element apart.

## Self-contained minimal reproducer

Save the following code as tstore_colmajor_nd_min.py:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import pypto.language as pl
from pypto.runtime import RunConfig, run


M = 8
N = 1024
NCOLS = 2
TARGET_COL = 1


@pl.program
class TStoreRepro:
    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        x: pl.Tensor[[M, N], pl.FP32],
        ws: pl.Out[pl.Tensor[[M, NCOLS], pl.FP32]],
    ) -> pl.Tensor[[M, NCOLS], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="tstore_colmajor_nd_min"):
            partial = pl.row_sum(x)                 # ColMajor Vec [8, 1]
            ws = pl.assemble(ws, partial, [0, TARGET_COL])  # ND ws[:, 1]
        return ws


def make_input():
    row_values = (
        100
        + torch.arange(M, dtype=torch.float32).reshape(M, 1) * 10
    )
    x = row_values.repeat(1, N).contiguous()
    expected_ws = torch.zeros(M, NCOLS, dtype=torch.float32)
    expected_ws[:, TARGET_COL] = row_values[:, 0] * N
    return x, expected_ws


def run_once(platform: str, output_root: str) -> int:
    output_dir = Path(output_root) / platform
    output_dir.mkdir(parents=True, exist_ok=True)

    x, expected_ws = make_input()
    ws = torch.zeros(M, NCOLS, dtype=torch.float32)

    run(
        TStoreRepro,
        x,
        ws,
        config=RunConfig(
            platform=platform,
            save_kernels=True,
            save_kernels_dir=str(output_dir),
            enable_dump_tensor=2,
            enable_l2_swimlane=platform == "a2a3",
            compile_profiling=True,
        ),
    )

    contiguous = torch.zeros_like(expected_ws)
    contiguous.reshape(-1)[TARGET_COL:TARGET_COL + M] = expected_ws[:, TARGET_COL]

    ws_ok = torch.allclose(ws, expected_ws, rtol=0, atol=0)
    contiguous_ok = torch.allclose(ws, contiguous, rtol=0, atol=0)

    print(f"platform={platform}")
    print(f"ws_matches_expected={ws_ok}")
    print(f"ws_matches_contiguous_write_hypothesis={contiguous_ok}")
    print("ws_after:")
    print(ws)
    print("expected_ws:")
    print(expected_ws)
    print("contiguous_write_hypothesis:")
    print(contiguous)

    if platform == "a2a3sim":
        expected_behavior = ws_ok
        verdict = "PASS" if expected_behavior else "UNEXPECTED_FAIL"
    else:
        expected_behavior = (not ws_ok) and contiguous_ok
        verdict = "GAP_REPRODUCED" if expected_behavior else "UNEXPECTED_BEHAVIOR"

    print("VERDICT:", verdict)
    return 0 if expected_behavior else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("platform", choices=["a2a3sim", "a2a3"])
    parser.add_argument(
        "--output-root",
        default="./outputs",
    )
    args = parser.parse_args()
    raise SystemExit(run_once(args.platform, args.output_root))
```

## How to run

Run the following commands in an environment configured with PyPTO, the PyPTO runtime, PTOAS, and the corresponding PTO-ISA version:

```bash
python tstore_colmajor_nd_min.py a2a3sim
python tstore_colmajor_nd_min.py a2a3
```

Notes:

- a2a3sim does not require real hardware.
- a2a3 requires access to a real device.
- output-root controls where generated kernels and diagnostic artifacts are saved.

## Results

### a2a3sim

```text
ws_matches_expected=True
ws_matches_contiguous_write_hypothesis=False
VERDICT: PASS
```

The expected workspace is:

```text
[[0, 102400],
 [0, 112640],
 [0, 122880],
 [0, 133120],
 [0, 143360],
 [0, 153600],
 [0, 163840],
 [0, 174080]]
```

### Real a2a3 hardware

The issue reproduces on real a2a3 hardware as follows:

```text
ws_matches_expected=False
ws_matches_contiguous_write_hypothesis=True
VERDICT: GAP_REPRODUCED
```

A typical real-hardware result is:

```text
[[0, 102400],
 [112640, 122880],
 [133120, 143360],
 [153600, 163840],
 [174080, 0],
 [0, 0],
 [0, 0],
 [0, 0]]
```

The observed results on both platforms are:

```text
a2a3sim: VERDICT: PASS
a2a3:    VERDICT: GAP_REPRODUCED
```

Explanation:

- Logically, partial[0] through partial[7] must be stored into destination column 1.
- Real hardware instead begins at flat[1] and writes eight consecutive elements.
- As a result, partial[1] is stored in ws[1,0], partial[2] is stored in ws[1,1], and the remaining values continue across row boundaries.

## Address semantics

The destination ws is a row-major tensor of shape [8,2]. For fixed column 1:

```text
index(ws[r,1]) = r * 2 + 1
```

Therefore, the correct flat destination indices are:

```text
1, 3, 5, 7, 9, 11, 13, 15
```

Real a2a3 hardware instead writes to:

```text
1, 2, 3, 4, 5, 6, 7, 8
```

This is equivalent to:

```text
ws.reshape(-1)[1:9] = partial[:]
```

## Generated code shape

The reproducer generates the following relevant layouts:

```text
source tile:
  Tile<Vec, FP32, rows=8, cols=1, BLayout::ColMajor>

target view:
  GlobalTensor ND
  logical shape = [8, 1]
  target strides = [2, 1]

operation:
  TSTORE(target_view, source_tile)
```

In the 5D GlobalTensor representation, the destination is equivalent to:

```text
shape   = [1, 1, 1, 8, 1]
strides = [16, 16, 16, 2, 1]
layout  = ND
```

The PyPTO/PTOAS IR and generated C++ are identical between a2a3 and a2a3sim. The behavioral difference occurs in the platform-specific PTO-ISA TSTORE implementation.

## PTO-ISA root-cause analysis

Relevant files:

```text
include/pto/common/arch/memory/tstore_common.hpp
include/pto/npu/a2a3/TStore.hpp
include/pto/cpu/TStore.hpp
```

The common static check permits a source tile with a single row or a single column:

```cpp
(TileData::Rows == 1) || (TileData::Cols == 1)
```

Therefore, the ColMajor [8,1] source and ND destination combination is not rejected at compile time. The a2a3 TStore implementation then selects:

```cpp
TStoreUb2gmDn2dn(...)
```

because the source tile is ColMajor.

That function uses:

```cpp
uint16_t nBurst = gShape4;
uint32_t lenBurst = validRow * sizeof(T);
uint32_t gmGap = (gStride4 - gShape3) * sizeof(T);
```

For this reproducer:

```text
gShape3 = 8
gShape4 = 1
gStride3 = 2
gStride4 = 1
validRow = 8
validCol = 1
nBurst = 1
lenBurst = 32B
```

The implementation consequently emits one contiguous 32-byte burst and does not use the destination gStride3=2 to address the eight elements individually.

The CPU simulator TSTORE path computes destination addresses using the target strides, so a2a3sim produces the correct result.

This does not imply that TStoreUb2gmDn2dn is incorrect for a genuine compact DN destination. The problem is that the single-column exception permits an ND strided destination to reach an implementation path that assumes a contiguous DN column.

## Suggested fixes

Any of the following approaches would address the correctness issue:

1. Correctly support storing a ColMajor Vec[M,1] into an ND strided destination:

   ~~~text
   dst[r * gStride3 + c * gStride4] = src[r, c]
   ~~~

2. Legalize the layout/stride combination into a supported sequence before entering TStoreUb2gmDn2dn.
3. Reject unsupported layout/stride combinations at compile time or during verification instead of silently producing incorrect output.

## Acceptance criteria

- a2a3sim and real a2a3 produce identical results for the reproducer above.
- A non-contiguous column store honors the destination strides.
- Unsupported patterns fail explicitly instead of completing with silently corrupted output.
- Add the minimal reproducer above as a regression test.
- Add a genuine compact DN destination regression test to ensure the existing Dn2dn path remains correct.
- Consider coverage for supported element types other than FP32.

## Versions

The issue was reproduced with:

```text
PyPTO       8ebddcb87a95c091c02f7f53b0a877c1e3f3d444
PyPTO runtime
            c94aa9f359a6c1825c9ba71ecf25576f1fa045b1
PTOAS       v0.48 / a45233eef74a9e2b79494fb9cfd2f9419a5d6eaf
PTO-ISA     83d01313d9bfc247c4b7c8bcf969d1019f0d106f
```



---

## #200 [BUG][TMov/a5] Acc ValidShape strip C2V uses validRow (not Rows) for NZ srcStride -> silent wrong results from 2nd strip

- State: open
- URL: https://github.com/hw-native-sys/pto-isa/issues/200
- Created: 2026-07-21T09:23:01Z
- Updated: 2026-07-21T09:25:28Z
- Closed: 

### Body

## Summary

On Ascend950 (A5), draining a **parent Acc** tile Cube→Vector (C2V) through a `ValidShape` **strip window** (`ValidRow = H < Rows`) produces **silent wrong results**. The first strip (`row=0`) is usually correct; from the second strip on (`row ≥ H`) the values are systematically wrong.

Root cause: in [`include/pto/npu/a5/TMov.hpp`](https://github.com/hw-native-sys/pto-isa/blob/main/include/pto/npu/a5/TMov.hpp), `TMovCcToUb` (and `TMovCcToCb`) compute the Acc **NZ `srcStride` from `validRow`** instead of from the tile's allocated `Rows`. The NZ `n1` pitch must follow the allocated Acc `M` (`Rows`); `ValidShape` should only bound *how many* rows/cols are moved, not the stride. When `validRow < Rows` the stride is computed too small, so every strip after the first reads the wrong region of the Acc.

```text
buggy:   srcStride = align(validRow)          // Valid(16) -> stride = 16
correct: srcStride = align(SrcTileData::Rows)  // parent Acc M=128 -> stride = 128
```

This is **not** a missing NZ→ND conversion on the consumer side. On A5, `TPUSH`'s internal `TMovCcToUb` already does NZ→ND and the `TPOP` side receives an ND Vec. The defect is purely the NZ `srcStride` on the AIC drain side.

Cross-reference: [`include/pto/npu/a2a3/TMov.hpp`](https://github.com/hw-native-sys/pto-isa/blob/main/include/pto/npu/a2a3/TMov.hpp) Acc→Mat already uses `SrcTileData::Rows` for `srcStride`, so the same `ValidShape` window does not expose the bug there; A5's direct L0C→UB path surfaces it.

## Environment

- Device: Ascend950 (A5), real board.
- Path: PyPTO runtime (simpler) tensormap_and_ringbuffer runtime → PTO-ISA.
- Shapes (UB-safe, still `validRow < Rows`): `M=32, N=128, H=16, K=32` → 2 strips. Seed `torch.manual_seed(0)`.
- Threshold: `rtol=1e-4, atol=1e-4`.

## Minimal trigger

```text
Parent Acc tile:
  TileType::Acc, dtype = FP32
  Rows      = M   (e.g. 32)
  ValidRow  = H   (e.g. 16)      <-- validRow < Rows
  BLayout   = ColMajor (NZ in L0C)

Drain loop (AIC):
  for row in range(0, M, H):
      TASSIGN(acc_window, addr = row * 64)     # ColMajor boxed row stride
      TPUSH<DIR_C2V, ...>(pipe, acc_window)    # internal TMovCcToUb(validRow=H, validCol=N)

Consumer (AIV):
  for row in range(0, M, H):
      TPOP<DIR_C2V, ...>(pipe, vec_strip)      # H x N ND vec, expected = Acc rows [row, row+H)
```

`TMovCcToUb` computes `srcStride = align(validRow) = 16`, but the parent Acc's NZ `n1` pitch is `align(Rows) = M`. Strip 0 (`addr=0`) can coincidentally look correct; strip 1+ (`addr = row*64`) reads with the wrong pitch → wrong data.

## Self-contained minimal reproducer

**Download (ready-to-unzip package):** [`ISSUE_A5_ACC_VALIDSHAPE_C2V_SRCSTRIDE.zip`](https://github.com/yanghaoran29/pto-isa-repro-a5-acc-c2v-validshape/releases/download/repro-v1/ISSUE_A5_ACC_VALIDSHAPE_C2V_SRCSTRIDE.zip) &nbsp;·&nbsp; [browse the files](https://github.com/yanghaoran29/pto-isa-repro-a5-acc-c2v-validshape/tree/main/acc_c2v_strip_validshape_min)

The package drains the **same Acc two ways** and requires the results to match — no external golden needed:

| Output | Path |
|--------|------|
| `C_full` | one full Acc `TPUSH`, `Valid ≡ Rows` (always correct) |
| `C_strip` | Acc + `ValidShape(H)` windows at `addr = row*64` (buggy pattern) |

Host assertion: `C_strip ≈ C_full`.

```text
acc_c2v_strip_validshape_min/
  README.md
  run_acc_c2v_strip_vs_full_board.sh                      # task-submit wrapper (edit SIMPLER)
  examples/a5/tensormap_and_ringbuffer/acc_c2v_strip_validshape/
    README.md                                             # test notes + board numbers
    test_acc_c2v_strip_vs_full.py                         # self-compare (C_strip vs C_full)
    kernels/mix/kernel_acc_c2v_compare.cpp                # matmul + full + strip TPUSH
    kernels/orchestration/acc_c2v_compare_orch.cpp        # [A, B, C_full, C_strip]
```

Run (in an A5-configured simpler tree; the package contains only the example, not simpler/pto-isa itself):

```bash
cp -a acc_c2v_strip_validshape_min/examples/a5/tensormap_and_ringbuffer/acc_c2v_strip_validshape \
  "$SIMPLER/examples/a5/tensormap_and_ringbuffer/"
cd "$SIMPLER"
export PYTHONPATH="$SIMPLER/python:$SIMPLER:${PYTHONPATH:-}"
export PTO_ISA_ROOT="${PTO_ISA_ROOT:-$SIMPLER/build/pto-isa}"
python examples/a5/tensormap_and_ringbuffer/acc_c2v_strip_validshape/test_acc_c2v_strip_vs_full.py \
  -p a5 -d <device>
```

<details>
<summary>AIC/AIV kernel — <code>kernels/mix/kernel_acc_c2v_compare.cpp</code> (key parts)</summary>

```cpp
constexpr int M = 32, N = 128, H = 16, K = 32;
constexpr uint64_t kAccRowByteStride = 64;

using AccWinFullT  = Tile<TileType::Acc, float, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 1024, ...>;
using AccWinStripT = Tile<TileType::Acc, float, M, N, BLayout::ColMajor, H, N, SLayout::RowMajor, 1024, ...>; // ValidRow=H
using VecFullT     = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N, SLayout::NoneBox, 512, ...>;
using VecStripT    = Tile<TileType::Vec, float, H, N, BLayout::RowMajor, H, N, SLayout::NoneBox, 512, ...>;

// AIC: acc = A@B once, then
// (1) full Acc C2V — Valid ≡ Rows, always correct
{
    AccWinFullT full;  TASSIGN(full, 0x0);
    TPUSH<PipeFullT, AccWinFullT, TileSplitAxis::TILE_NO_SPLIT>(pipeFull, full);
}
// (2) strip Acc C2V — Valid(H) windows @ addr=row*64 (the buggy pattern)
for (int row = 0; row < M; row += H) {
    AccWinStripT strip;
    TASSIGN(strip, static_cast<uint64_t>(row) * kAccRowByteStride);
    TPUSH<PipeStripT, AccWinStripT, TileSplitAxis::TILE_NO_SPLIT>(pipeStrip, strip);
}

// AIV: TPOP full -> store C_full; loop TPOP strips -> store C_strip[row*N ..]
```
</details>

<details>
<summary>Python self-compare — <code>test_acc_c2v_strip_vs_full.py</code> (assertion)</summary>

```python
M, N, K = 32, 128, 32
H = 16
# ...
full  = test_args.C_full.reshape(M, N)
strip = test_args.C_strip.reshape(M, N)
if not torch.allclose(strip, full, rtol=1e-4, atol=1e-4):
    diff = (strip - full).abs().max().item()
    s0 = (strip[:H]      - full[:H]).abs().max().item()
    s1 = (strip[H:2*H]   - full[H:2*H]).abs().max().item()
    raise AssertionError(
        f"C_strip vs C_full mismatch: max_diff={diff}, "
        f"strip0_max={s0}, strip1_max={s1}")
```
</details>

## Board results (2026-07-21, device dump, seed=0)

### Before fix (`srcStride = align(validRow)`)

strip0 (rows 0…15) matches full; strip1 (rows 16…) diverges systematically:

```text
# strip0 — equal
row=0    C_full[0,0]=10.4928    C_strip[0,0]=10.4928    diff=0
row=15   C_full[15,0]=-2.02309  C_strip[15,0]=-2.02309  diff=0
# strip1 — wrong
row=16   C_full[16,71]=-12.8381 C_strip[16,71]=13.6886  |diff|=26.5268
row=17   C_full[17,44]=-11.6004 C_strip[17,44]=11.7646  |diff|=23.365
# summary: strip0_max=0.0   strip1_max=31.9782   -> FAILED
```

### After fix (`srcStride = align(Rows)`)

```text
row=0    C_full[0,0]=10.4928    C_strip[0,0]=10.4928    diff=0
row=15   C_full[15,0]=-2.02309  C_strip[15,0]=-2.02309  diff=0
row=16   C_full[16,0]=2.29007   C_strip[16,0]=2.29007   diff=0
row=17   C_full[17,0]=5.26025   C_strip[17,0]=5.26025   diff=0
# summary: strip0_max=0   strip1_max=0   -> PASSED
```

| Case | Before | After |
|------|--------|-------|
| strip C2V: parent Acc + `ValidShape(H,N)` | **FAIL** | **PASS** |
| full C2V: one `TPUSH`, `Valid ≡ Rows` | PASS | PASS (no regression) |

## Proposed fix

In [`include/pto/npu/a5/TMov.hpp`](https://github.com/hw-native-sys/pto-isa/blob/main/include/pto/npu/a5/TMov.hpp) (`TMovCcToUb`, `TMovCcToCb`), derive the Acc NZ `srcStride` from the allocated `Rows`, matching a2a3 Acc→Mat:

```cpp
// Acc NZ n1 pitch follows the allocated Acc M (Rows), not ValidShape.
constexpr auto srcStride =
    (SrcTileData::Rows + BLOCK_LEN - 1) / BLOCK_LEN * BLOCK_LEN;
// validRow / validCol still bound the transfer window size.
```

## Risk / impact

- When `validRow == Rows` (full drain), new and old behavior are identical → no regression on the common path.
- Only changes `validRow < Rows` Acc→UB / Acc→L1 `ValidShape` windows, aligning them with the NZ layout.
- `Rows` is a static positive integer here; a future dynamic `Rows == -1` would need a runtime stride and should be revisited separately.


---

## #204 [Bug] a2a3→common extraction (9e6c6d5b) leaves duplicate TStoreAccFp + undefined dstC0 in TTrans.hpp → kernels fail to compile

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/204
- Created: 2026-07-25T02:25:37Z
- Updated: 2026-08-11T01:57:44Z
- Closed: 2026-08-11T01:57:44Z

### Body

## Summary

Two pre-existing **compile-time** breaks on `origin/main` (`439faf48`) prevent any kernel whose translation unit pulls in both `pto/common/pto_instr_impl.hpp` (→ `common/arch/memory/tstore_common.hpp`) and `pto/npu/a2a3/TPush.hpp` (→ `npu/a2a3/TStore.hpp` + `TTrans.hpp`) from building. Both were introduced by commit `9e6c6d5b` ("Extract the duplicate code from a2a3 and kirinx90."), which moved code into the new common layer but left the a2a3 originals in place. I confirmed both breaks are present on `origin/main` itself (not just feature branches) via a fresh `git fetch origin` — there is no newer upstream commit that fixes them.

## Root cause & evidence

### 1. `TStoreAccFp` double definition

- `include/pto/common/arch/memory/tstore_common.hpp:416` defines `TStoreAccFp` (the extraction target).
- `include/pto/npu/a2a3/TStore.hpp:168` defines `TStoreAccFp` **again** — and `TStore.hpp` already `#include`s `tstore_common.hpp` at line 138.
- The two bodies are **semantically identical**; the only diff is formatting (line wrapping + `*` placement, e.g. `DType *p` vs `DType* p`). The a2a3 copy is a pure leftover.

```
TStore.hpp:170:20: error: redefinition of 'TStoreAccFp'
tstore_common.hpp:419:20: note: previous definition is here
TStore.hpp:169:42: error: template parameter redefines default argument
```

### 2. `TTrans.hpp` references undefined `dstC0`

Two functions had their last parameter renamed `dstC0 → validC0`, but the function bodies were not updated and still reference `dstC0`:

- `ConvNCHW2NC1HWC0Unalign` (`include/pto/npu/a2a3/TTrans.hpp:406`, param `validC0` at :407) — body uses undefined `dstC0` 5× (e.g. `:410 unsigned dstStride = dstC0;`).
- `ConvGNCHW2GNC1HWC0Unalign` (`include/pto/npu/a2a3/TTrans.hpp:532`, param `validC0` at :534) — body uses undefined `dstC0` 6×.

Other functions such as `ConvNCDHWPlane2NCHWUnalign` (:680) and `ConvNCDHWPlane2NCHW` (:706) legitimately take `dstC0` as a parameter — **only** the two `*Unalign` functions above are broken.

```
TTrans.hpp:410:26: error: use of undeclared identifier 'dstC0'
```

## Reproduce

```bash
bash kernels/manual/a2a3/distributed_ffn_grid/run_treduce_reducesum.sh --build-only -r npu
# also reproduces with run_tbroadcast_allgather.sh
```

Both fail at `[ 25%] Building CXX object .../*_compute_kernel.cpp.o` with the errors above (≈20 errors before `-ferror-limit` halts).

## Recommended fix

1. **TStore** — delete the now-redundant `TStoreAccFp` template + body from `include/pto/npu/a2a3/TStore.hpp` (lines ~168 to its closing brace). It is superseded by the identical definition in `tstore_common.hpp`, which `TStore.hpp` already includes. A diff of the two bodies shows only formatting differences, so this is a zero-behavior-change removal that completes the intent of `9e6c6d5b`.

2. **TTrans** — in `ConvNCHW2NC1HWC0Unalign` and `ConvGNCHW2GNC1HWC0Unalign`, replace the body's `dstC0` references with `validC0` (the actual parameter name). 11 sites total (5 + 6).

## Environment

- Commit: `origin/main` = `439faf48d7ab5c36a7bf72bc99d29c213a315550`
- Trigger commit: `9e6c6d5b` "Extract the duplicate code from a2a3 and kirinx90."
- Host: Linux aarch64, CANN 9.0.0
- Build: `-r npu -v Ascend910B1` (compile-time break; runtime not reached)


---

## #211 CPU stub / a5sim 缺少与 NPU 对齐的 MX TMATMUL_MX 通路（GetScaleAddr + MX_A_ZZ/MX_B_NN TLOAD）

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/211
- Created: 2026-07-28T11:06:59Z
- Updated: 2026-08-17T01:47:42Z
- Closed: 2026-08-17T01:47:35Z

### Body

## 背景

在 simpler 中增加了基于 pto-isa 的 MXFP8/MXFP4 ST（直接调用 `TMATMUL_MX`，host-prequant E8M0，scale GM 布局为 `MX_A_ZZ` / `MX_B_NN`）。真机 A5 路径可跑通；**a5sim（`__CPU_SIM`）无法用同一套内核**，目前只能 `platforms: ["a5"]`。

## 问题

### 1. CPU stub 没有 `GetScaleAddr`

NPU 样例（`tests/npu/a5/.../tmatmul_mx`）在非 `__PTO_AUTO__` 路径使用：

```cpp
uint64_t scaleAAddr = GetScaleAddr(aTile.data());
TASSIGN(aScaleTile, scaleAAddr);
```

`GetScaleAddr` 定义在 `include/pto/npu/a5/utils.hpp`。
a5sim 编译走 CPU stub，该接口不可见，无法用与真机相同的 scale tile 绑定方式。

CPU 侧现有替代是 `TGET_SCALE_ADDR`，或 `tests/cpu/.../tmatmul_mx` 里把 scale tile `TASSIGN` 到线性 buffer —— **与 NPU 写法不一致**。

### 2. CPU `TLOAD` 不支持 MX scale layout

CPU stub `TLoad.hpp` 有静态断言，仅支持 `ND` / `DN` / `NZ`：

> Only ND, DN and NZ Global Tensors are currently supported

因此对 scale 使用 `Layout::MX_A_ZZ` / `Layout::MX_B_NN` 的 `TLOAD` 在 a5sim 上**直接编译失败**。

这与 NPU 侧 `tmatmul_mx`（ZZ/NN + 真机装载语义）以及 CPU 侧 `tmatmul_mx`（scale 用 `Layout::ND`）的分裂是一致的：

| 环境 | scale GM layout | scale 地址绑定 |
|------|-----------------|----------------|
| NPU A5 | `MX_A_ZZ` / `MX_B_NN` | `GetScaleAddr` |
| CPU stub | `ND`（等） | `TGET_SCALE_ADDR` / 线性 `TASSIGN` |

## 影响

- 下游（如 simpler）无法用**同一套** `TMATMUL_MX` + ZZ/NN 内核同时覆盖 a5sim 与真机 A5。
- MX 相关 ST 只能上板，CI sim 无法回归这条原生路径。
- 在 simpler 层用 `#ifdef` 双路径统一成本高，且容易和 pto-isa 已有 NPU/CPU 分叉重复维护。

## 期望

优先任选其一（或组合）：

1. **CPU stub 补齐与 NPU 对齐的能力**
   - 提供与 `GetScaleAddr` 等价的 CPU 接口（或保证 `TGET_SCALE_ADDR` 语义可替代且文档写清）；
   - `TLOAD` 支持 `MX_A_ZZ` / `MX_B_NN`（或明确文档：CPU 仅支持 ND，并给出推荐写法）。

2. **若短期不统一**：在文档 / README / `tmatmul_mx` 处明确写清
   - NPU vs CPU 两套样例的差异；
   - a5sim / `__CPU_SIM` 不支持 ZZ/NN MX scale TLOAD 的限制；
   - 下游应如何选择路径。

## 复现要点

- 参考：`tests/npu/a5/src/st/testcase/tmatmul_mx/tmatmul_mx_kernel.cpp`（`GetScaleAddr` + `MX_A_ZZ`/`MX_B_NN`）
- 对比：`tests/cpu/st/testcase/tmatmul_mx/tmatmul_mx_kernel.cpp`（scale `Layout::ND` + `TGET_SCALE_ADDR`）
- 在 `__CPU_SIM` 下编译含 `GlobalTensor<..., Layout::MX_A_ZZ>` 的 `TLOAD` 会触发上述 static_assert。

## 相关

- simpler PR：https://github.com/hw-native-sys/simpler/pull/1473
- 验证：真机 A5 上 MXFP8/MXFP4 `TMATMUL_MX` ST 已通过；a5sim 因上述限制未覆盖。

---

## #214 INT8 matmul yields wrong INT32 results with full-M accumulator in prefill_indexer (M-tiling fixes it; not reproducible standalone)

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/214
- Created: 2026-07-29T10:36:17Z
- Updated: 2026-08-17T07:32:06Z
- Closed: 2026-08-17T07:30:49Z

### Body

## Background

Hit in `pypto-lib` while bringing up DeepSeek-V4-**Pro** on A5. The case
`models/deepseek/v4-pro/prefill_indexer.py` failed its `score` check on **every**
run, in a way that looks like a precision problem but is not: ~all values are
wrong, yet per-row norms are preserved.

I traced it to a single INT8 → INT32 tile matmul that **silently returns wrong
products**. Tiling the M dimension makes it correct with no other change. All
tiles involved fit a5's on-chip limits, so this is not a capacity overflow.

I could **not** separate `pto-isa` from `ptoas` with the tooling I have — filing
here because the fault is in the tile-matmul implementation path; please reroute
if it belongs to the assembler.

### Environment

| Component | Version |
|---|---|
| pto-isa | `83d01313d9bfc247c4b7c8bcf969d1019f0d106f` (the pin in `runtime/pto_isa.pin`) |
| ptoas | v0.48 |
| pypto | `a8ef572f` |
| simpler | `8e00319e` |
| pypto-lib | `4c346f0` |
| driver | 25.7.rc1.6 |
| CANN | 9.1.0-beta.3 |
| host | x86_64, Ascend 950 (A5) |

**Pin consistency:** pto-isa and ptoas are exactly on their pins. `simpler` is
deliberately ahead of the commit `pypto` pins (`9922afdb`), and carries one local
patch unrelated to this bug (an AICPU topology probe workaround). The failure
reproduces with `simpler` at pypto's pin as well, so it is not implicated.

## What goes wrong

The kernel accumulates a Q projection over K:

```python
qr_acc = pl.create_tensor([T, Q_OUT_TILE], dtype=pl.INT32)      # T=128, Q_OUT_TILE=256
for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):            # 1536/128 = 12 iterations
    q0 = kb * Q_TILE
    qr_tile = qr[:, q0 : q0 + Q_TILE]                           # [128, 128] INT8
    wq_tile = wq_b[q0 : q0 + Q_TILE, o0 : o0 + Q_OUT_TILE]      # [128, 256] INT8
    if q0 == 0:
        qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32)
    else:
        qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile)
```

The emitted PTO (`prefill_idx_qr_proj.pto`) allocates exactly what you would
expect, and all of it is within a5's documented limits:

| tile | shape | bytes | a5 limit |
|---|---|---|---|
| `loc=left` (L0A) | 128 x 128 i8 | 16 KB | 64 KB |
| `loc=right` (L0B) | 128 x 256 i8 | 32 KB | 64 KB |
| `loc=acc` (L0C) | 128 x 256 i32 | 128 KB | **256 KB** |

(L0C = 256 KB on a5 per `pypto/include/pypto/backend/950/backend_950_handler.h`
and `pto-isa/include/pto/common/buffer_limits.hpp`.)

It compiles clean — 1 `pto.tmatmul` + 2 `pto.tmatmul.acc` — and produces **wrong
INT32 results**.

## Evidence

I exported the kernel's intermediates and diffed each against a host
recomputation of the same arithmetic (`qr.int32() @ wq_b.int32()`, then dequant).

| kernel intermediate | agreement with host reference | correlation |
|---|---|---|
| `weights` projection (a BF16 matmul, same scope) | **100.00 %** | **1.00000** |
| **`qr_proj`** (this INT8 matmul) | **6.85 %** | **0.086** |
| q quant scale (derived from `qr_proj`) | 2.61 % | 0.081 |
| q int8 (derived from `qr_proj`) | 0.69 % exact | 0.059 |

`qr_proj` is the first stage that diverges, and its **NOPE half** — the 64
columns that the later RoPE never touches — is wrong on its own, so nothing
downstream can be responsible.

The corruption is norm-preserving, which is why it masquerades as precision
noise rather than a crash:

```
|qr_proj_device| per-row mean 28.1307     |reference| 27.7017
```

Downstream, this made `score` wrong at **2010 of 2016** visible positions
(6.13 % of the tensor, tolerance 0.5 %), with sign flips and diffs up to 4.0
against tolerances of ~0.002.

## Ruled out

- **Capacity** — every tile fits (table above).
- **Everything downstream of the matmul** — RoPE convention (interleaved /
  half-split / none), the RoPE cos/sin token mapping (7 variants), Hadamard
  orientation (the matrix is symmetric, so normal and transposed are identical),
  INT8 quantisation granularity, relu placement, and the scale application order.
  None of ~20 offline variants reproduced the device output; a host
  reimplementation of the golden recipe matches golden to 99.60 %.
- **Component versions** — reproduces with `simpler` at pypto's pin `9922afdb`,
  and with `pypto` back at `d64380cb`.
- **Preset/shape specific to V4-Pro** — reproduces under the Flash preset too.

## Reproduction

The single change that fixes it is tiling M — accumulate 16 rows at a time
instead of all 128, everything else identical:

```python
for r0 in pl.range(0, T, 16):
    qr_acc = pl.create_tensor([16, Q_OUT_TILE], dtype=pl.INT32)
    for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
        q0 = kb * Q_TILE
        qr_tile = qr[r0 : r0 + 16, q0 : q0 + Q_TILE]
        wq_tile = wq_b[q0 : q0 + Q_TILE, o0 : o0 + Q_OUT_TILE]
        qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32) if q0 == 0 \
                 else pl.matmul_acc(qr_acc, qr_tile, wq_tile)
```

With that, `qr_proj` matches the reference at **100.00 % / correlation 1.00000**
and the case passes 4/4. Workaround merged as hw-native-sys/pypto-lib#856.

`models/deepseek/v4-pro/decode_indexer.py` performs the same projection and has
always passed — its decode `T` is 16, so it never builds the large-M form.

## Ask

A matmul that cannot be lowered correctly should fail loudly rather than return
wrong data. Note the contrast: when I tried an intermediate row tile of 64, the
toolchain **did** reject it at compile time on a Vec buffer limit
(`591104 bytes exceeds platform limit 245760`). So some buffer constraints are
enforced while this one silently miscomputes.


---

## #226 [Bug] A2A3 DIR_BOTH TPipe: the C2V and V2C rings alias the same GM slots (supersedes #195)

- State: closed
- URL: https://github.com/hw-native-sys/pto-isa/issues/226
- Created: 2026-08-03T13:16:48Z
- Updated: 2026-08-12T07:33:56Z
- Closed: 2026-08-12T07:33:56Z

### Body

> **Supersedes #195** (and its downstream hw-native-sys/pypto#1981), which reported this same defect on 2026-07-09 and was closed three days later as *"not a problem with pto-isa"*. No fix landed, and `main` (`e8f558d5`) still has the aliasing. This report adds what #195 did not have: the ISA spec that mandates the layout, a minimal repro with no `split_aiv`/`syncall`/multi-core contention, the missing frontend half that makes a device-only fix look wrong, and a hardware-validated fix at the stock ring depth. Frontend half: hw-native-sys/pypto#2269.

## Summary

On A2A3 a bidirectional `TPipe` places both of its rings at the same GM addresses, so cube→vector and vector→cube traffic silently overwrite each other's tiles once more than one tile is in flight.

## Root cause

Every GM entry in `include/pto/npu/a2a3/TPush.hpp` is addressed as

```cpp
size_t entryBase = (tileIndex % RingFiFo::SLOT_NUM) * RingFiFo::SLOT_SIZE;
... (__gm__ T*)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + entryOffset)
```

— producer at `:155`, `:187`, `:252`; consumer at `:398`, `:420` (line numbers on `main` `e8f558d5`). The only term that could separate the two directions is `entryOffset`, and for `DIR_BOTH` nothing ever sets it: the `TPipe` constructor (`:462`) only primes credits, and `grep setEntryOffset` over generated kernels comes back empty. Both directions therefore index slot `k % SLOT_NUM` of the **same** buffer. The cube's matmul results and the vector's operands overwrite each other.

## This is a conformance gap, not a design question

The design doc in this repo specifies both halves of the layout. `docs/HL_ptoisa_newfeature20260306_TPUSH_TPOP.md`:

```text
:169   gm_slot_buf = gm_alloc(2 * SLOT_NUM * SLOT_SIZE)    // bidirectional
:222   buf_offset = (DIR_MASK & DIR_C2V) ? SLOT_NUM * SLOT_SIZE : 0
:223   v2c_ring_buf = GM_SLOT_BUFFER + buf_offset
```

and `:288` draws it:

```text
GM_SLOT_BUFFER (total size = 2 * SLOT_NUM * SLOT_SIZE for bidirectional):

    ┌─────────────────────────────┬─────────────────────────────┐
    │  C2V ring buffer            │  V2C ring buffer            │
    │  slot[0] .. slot[SLOT_NUM-1]│  slot[0] .. slot[SLOT_NUM-1]│
    │  offset: 0                  │  offset: SLOT_NUM*SLOT_SIZE │
    └─────────────────────────────┴─────────────────────────────┘
```

`docs/zh/reference/pto-isa/01-tpush_tpop.md` says the same. The implementation applies neither the offset nor the doubled size.

## Symptom

Silent wrong results — no fault, no hang. Hit in production in a sequence-parallel GLA/ZeCO operator built on pypto: 12 of 21 dispatches wrong, error O(1..8) on values of order 1. Same binary and same inputs give different results run to run; `a2a3sim` is clean at every size; bad chunks repeat with period `SLOT_NUM` (the ring wrapping).

## Reproduction

A single `DIR_BOTH` GM pipe in one InCore loop, on one device — no `split_aiv`, no `syncall`, no multi-layer loop, no contention. The emitted pipe is `TPipe<0, DIR_BOTH, 1024, 4, 4, true>` (`IsNoSplit=true`, so #195's second complaint about the split-lane `get_subblockid()` offset does not apply here).

```bash
python3 repro.py <device_id> a2a3 2,4,5,6,7,8,16 3     # torch + pypto only, no other deps
```

On stock, 12/21 dispatches are corrupted; the first bad chunk index is always ≡ 2 (mod 4).

<details>
<summary>repro.py</summary>

```python
"""Minimal repro: a pypto InCore loop's carried tile is corrupted on a2a3 hardware.

One InCore kernel, ONE device, no distribution, no communication. A chunk-recurrent loop
carries a [16,16] tile `s_run` and does four matmuls per iteration (so four cube round-trips
through the emitted cube<->vector TPipe). The state written out each iteration is compared
against a torch mirror of the same arithmetic.

Expected: every chunk within fp32 tolerance (~1e-6), for every trip count N.
Actual on a2a3 HW: exact for the first chunks, then one iteration's carry is corrupted by
O(1..8) and the error decays with the gate; corruption re-injects periodically. The first
corrupted index is always == 2 (mod 4), and 4 is the slot depth of the generated
`TPipe<0, DIR_BOTH, 1024, 4, 4, true>`.

    N= 2  CLEAN
    N= 4  CLEAN
    N= 5  run0 CLEAN, run1/run2 bad=[2,3,4]     <- same binary, same input, different result
    N= 6  CLEAN
    N= 7  bad=[6]
    N= 8  bad=[6,7]
    N=16  bad=[2..15]  (fresh corruption at 2, 6, 10, 14)

The generated device code for N=4 (clean) and N=8 (corrupted) is byte-identical apart from
the loop trip-count constant, and a2a3sim is clean at every N — so this is a hardware timing
race in the emitted synchronisation, not a value/codegen difference.

Usage:  python3 repro.py [device] [platform] [N_csv] [repeats]
"""
from __future__ import annotations

import sys

import torch

import pypto.language as pl

C = 16          # chunk size
DK = DV = 16    # key / value dims
TOL = 1e-2      # errors are O(1); a correct run is ~1e-6


def build(N: int):
    """InCore chunk recurrence: s <- s * g_full + kbar^T @ v, storing s after every chunk."""
    L = N * C
    NDK = N * DK

    @pl.program
    class LoopCarryProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def chunk_scan(
            self,
            A: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            ones_cc: pl.Tensor[[C, C], pl.FP32],
            ones_cdv: pl.Tensor[[C, DV], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            Sall: pl.Out[pl.Tensor[[NDK, DV], pl.FP32]],
        ) -> pl.Tensor[[NDK, DV], pl.FP32]:
            tril_t = pl.load(tril, [0, 0], [C, C])
            ones_cc_t = pl.load(ones_cc, [0, 0], [C, C])
            ones_cdv_t = pl.load(ones_cdv, [0, 0], [C, DV])
            s_init = pl.load(zero, [0, 0], [DK, DV])
            out = Sall
            for n, (s_run,) in pl.range(0, N, init_values=(s_init,)):
                off = n * C
                k = pl.load(Kmat, [off, 0], [C, DK])
                v = pl.load(Vmat, [off, 0], [C, DV])
                a = pl.load(A, [off, 0], [C, DK])
                la = pl.log(a)
                b = pl.exp(pl.matmul(tril_t, la, out_dtype=pl.FP32))
                g_row_full = pl.exp(pl.matmul(ones_cc_t, la, out_dtype=pl.FP32))
                g_full = pl.exp(pl.matmul(pl.transpose(la, 0, 1), ones_cdv_t, out_dtype=pl.FP32))
                kb = pl.div(k, b)
                kbar = pl.mul(kb, g_row_full)
                kv = pl.matmul(pl.transpose(kbar, 0, 1), v, out_dtype=pl.FP32)
                s_scaled = pl.mul(s_run, g_full)
                s_new = pl.add(s_scaled, kv)
                out = pl.store(s_new, [n * DK, 0], out)
                s_run = pl.yield_(s_new)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def entry(
            self,
            A: pl.Tensor[[L, DK], pl.FP32],
            Kmat: pl.Tensor[[L, DK], pl.FP32],
            Vmat: pl.Tensor[[L, DV], pl.FP32],
            tril: pl.Tensor[[C, C], pl.FP32],
            ones_cc: pl.Tensor[[C, C], pl.FP32],
            ones_cdv: pl.Tensor[[C, DV], pl.FP32],
            zero: pl.Tensor[[DK, DV], pl.FP32],
            Sall: pl.Out[pl.Tensor[[NDK, DV], pl.FP32]],
        ) -> pl.Tensor[[NDK, DV], pl.FP32]:
            return self.chunk_scan(A, Kmat, Vmat, tril, ones_cc, ones_cdv, zero, Sall)

    return LoopCarryProgram


def reference(A, K, V, N, tril, ones_cc, ones_cdv):
    """Torch mirror of the kernel, op for op."""
    s = torch.zeros(DK, DV)
    ref = []
    for n in range(N):
        lo, hi = n * C, (n + 1) * C
        la = torch.log(A[lo:hi])
        b = torch.exp(tril @ la)
        g_row_full = torch.exp(ones_cc @ la)
        g_full = torch.exp(la.t() @ ones_cdv)
        kb = K[lo:hi] / b
        kbar = kb * g_row_full
        kv = kbar.t() @ V[lo:hi]
        s = s * g_full + kv
        ref.append(s.clone())
    return torch.stack(ref)


def run(N: int, device: int, platform: str, repeats: int):
    from pypto import ir
    from pypto.runtime.runner import RunConfig

    L = N * C
    torch.manual_seed(42)
    K = torch.randn(L, DK)
    V = torch.randn(L, DV)
    A = 0.9 + 0.1 * torch.sigmoid(torch.randn(L, DK))   # decay gates in (0.9, 1.0)

    tril = torch.tril(torch.ones(C, C))
    ones_cc = torch.ones(C, C)
    ones_cdv = torch.ones(C, DV)
    zero = torch.zeros(DK, DV)
    ref = reference(A, K, V, N, tril, ones_cc, ones_cdv)

    compiled = ir.compile(build(N), platform=platform)
    cfg = RunConfig(platform=platform, device_id=device)
    for i in range(repeats):
        Sall = torch.zeros(N * DK, DV)
        compiled(A, K, V, tril, ones_cc, ones_cdv, zero, Sall, config=cfg)
        errs = [(Sall[n * DK:(n + 1) * DK] - ref[n]).abs().max().item() for n in range(N)]
        bad = [n for n, e in enumerate(errs) if e > TOL]
        print(f"N={N:>3} run{i}: {'CLEAN' if not bad else 'bad=' + str(bad)}  "
              f"max_err={max(errs):.3e}", flush=True)
        print("        per-chunk: " + " ".join(f"{e:.1e}" for e in errs), flush=True)


if __name__ == "__main__":
    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    platform = sys.argv[2] if len(sys.argv) > 2 else "a2a3"
    Ns = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [2, 4, 5, 6, 7, 8, 16]
    repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    print(f"platform={platform} device={device} C={C} dk=dv={DK}")
    for N in Ns:
        run(N, device, platform, repeats)
```

</details>

## Why existing tests do not catch it

- **CPU-SIM is a different shape.** #193 (*"Separate C2V/V2C cursors in DIR_BOTH split tile pipe"*, fixing #183) fixed this class of bug in `include/pto/cpu/TPush.hpp` only. There a `DIR_BOTH` pipe is a **single** ring whose slots are direction-tagged at runtime (`transfer_dirs[]`) and handed out by one mutex-protected cursor, so aliasing is structurally impossible. Every CPU ST test passes.
- **The device ST case is a single round trip.** `tests/npu/a2a3/src/st/testcase/tpushpop_dir_both` has the vector push one V2C tile, the cube pop it, the cube push one C2V tile, the vector pop it. Both directions use slot 0 and do alias, but the cube cannot push its result before it has consumed its input, so the data dependency serialises the overwrite behind the read. It passes. Exposing this needs two tiles in flight in opposite directions, which needs a loop.

That test also allocates `2 * M * N * sizeof(T)` = exactly one ring, so it needs its buffer doubled along with the fix.

## Suggested fix

Set the V2C ring base in the `TPipe` constructor for `is_both` — the consumer under `__DAV_CUBE__`, the producer under `__DAV_VEC__` — both compile-time constants, so the generated address arithmetic is unchanged in cost. PR to follow.

**Ordering matters.** The frontend must allocate `2 * SLOT_NUM * SLOT_SIZE` for a bidirectional pipe, as the doc has always specified. pypto sizes it as one ring; that is hw-native-sys/pypto#2269 and **must ship first**, or this fix writes the V2C ring past the end of the allocation. That is plausibly why a device-only fix looked wrong in #195: correct on its own terms, but presenting as a new corruption elsewhere.

## Validation

Atlas A2/A3, fp32, at the **stock** ring depth of 4, no DSL overrides, both halves applied:

| Check | Before | After |
|---|---|---|
| `repro.py`, C=16, N ∈ {2,4,5,6,7,8,16} × 3 dispatches | 12/21 corrupted | **21/21 clean** |
| Same kernel at 4 KiB slots, N ∈ {4,8,16,32} | corrupt at every N and every ring depth (4/8/16/32) | **clean** |
| GLA/ZeCO fused forward, C ∈ {16,32} × P ∈ {1,2,4} × N ∈ {2,4,8,16} | 5/12 fail at C=16, 6/12 at C=32 | **24/24 pass**, max abs err ≤ 3.1e-5 |
| Downstream regression suites | 4 / 3 / 3 pass | **4 / 3 / 3 pass** |

## A related question

`include/pto/npu/a5/TPush.hpp` looks like it has the same pattern for `DIR_BOTH_GM`: `:290`, `:319`, `:399`, `:655` all address `GM_SLOT_BUFFER + entryBase + entryOffset` and the a5 `TPipe` constructor (`:704`) is empty. a5's plain `DIR_BOTH` is safe because each direction lives in its own consumer SRAM, but `DIR_BOTH_GM` puts both in GM. I have no A5 hardware, so I have not touched it — should that be part of the same fix or a separate issue?

## Environment

Atlas A2/A3, fp32; pto-isa `83d01313` (our runtime's pin) with the defect verified unchanged on `main` `e8f558d5`; pypto `f621eca4`; PTOAS 0.54; runtime `9922afdb`.


---

## #238 [Feature][A5] 增加 TMXFP4TOFP8（MXFP4 E2M1 到 MXFP8 E4M3）复合指令

- State: open
- URL: https://github.com/hw-native-sys/pto-isa/issues/238
- Created: 2026-08-10T06:26:08Z
- Updated: 2026-08-12T08:11:44Z
- Closed: 

### Body


## Issue 类型

Feature request / ISA capability gap


## 背景

DeepSeek V4 Pro 在 A5 上需要执行 MXFP8 激活与 MXFP4 权重的混合精度矩阵乘。A5 现有 `TMATMUL_MX`/`mad_mx` 仅接受同一数据宽度族内的组合，即 FP4 × FP4 或 FP8 × FP8，不原生接受 MXFP8 × MXFP4。

因此，现阶段要复用已有的 FP8 × FP8 矩阵乘能力，需要：

1. 在 AIV/Vector 核把 packed MXFP4 E2M1 weight payload 按位展开成 MXFP8 E4M3 payload；
2. 同步补偿 MX block scale；
3. 在 AIC/Cube 核调用现有的 FP8 × FP8 `TMATMUL_MX`/`mad_mx` 路径。

这里需要的不是普通浮点 `cast`，而是一个改变物理存储宽度、具有固定 scale 关系的位级表示变换。


## 当前问题

PTO-ISA 当前缺少能够在 Tile 层完整表达该转换的操作：

1. `TCVT` 未定义 `float4_e2m1x2_t -> float8_e4m3_t` 的直接语义。已有路径主要是 FP4 → BF16，不能表达 AscendC 使用的无舍入、bit-exact 位展开。
2. 输入每字节打包两个 FP4，输出每字节存放一个 FP8。两者逻辑元素数相同，但物理存储量从 `N/2` 字节变为 `N` 字节，不能建模为普通等宽逐元素 cast。
3. FP4 payload 在内存中以一个字节打包两个 FP4 元素，而 A5 上大量后续计算需要使用 FP8 payload，因此 FP4 到 FP8 的位展开会成为 FP4 场景中非常常用的基础数据准备能力；如果上层业务场景重复封装底层向量位运算指令，会造成冗余开发工作、抬高维护成本并引入更多正确性隐患。
4. 当前可选回退路径为 FP4 → BF16 → FP8，或 FP4 → BF16 → FP32 → FP8；这些路径增加中间 Tile、转换步骤和舍入/量化语义。

因此，PyPTO 等上层目前无法用一个稳定且与 AscendC 对齐的 PTO-ISA 操作表达这条路径。


## 期望能力

本需求提供两种可评审的接口方案。两种方案统一使用 `TMXFP4TOFP8` 指令名，并通过操作数数量区分重载；payload 位展开算法完全相同，区别是 MX block scale 由调用方通过独立操作调整，还是融合进同一条复合指令。该命名直接限定 MXFP4 到 MXFP8 的转换方向；“位展开”用于描述 payload 算法，不再作为公共指令名。

### 方案 A：仅转换 payload

增加二操作数 AIV/Vector 复合指令：

```cpp
TMXFP4TOFP8(dstFp8, srcFp4);
```

第一版约束如下：

- `srcFp4`：`TileType::Vec`、RowMajor、`float4_e2m1x2_t`；
- `dstFp8`：`TileType::Vec`、RowMajor、`float8_e4m3_t`；
- 源和目标静态逻辑 shape、有效 shape 相同；
- 源静态行 stride 包含偶数个 FP4 元素，保证每行从字节边界开始；
- 有效列数允许为奇数，尾字节未使用的高 nibble 不产生输出；
- payload 源/目标是否允许重叠与现有 cast/`TCVT` 的别名规则保持一致；
- 操作属于 `PIPE_V`，支持与 `TLOAD`、`TSTORE_VEC` 建立 event 依赖。

方案 A 不能只单独交付 payload-only `TMXFP4TOFP8`。同批能力还必须增加独立的 E8M0 指数调整操作，用于将配套 scale 的 E8M0 指数码调整 `+6`。具体指令命名和签名由 PTO-ISA 评审细化。

### 方案 B：融合 payload 与 scale 转换

增加四操作数重载：

```cpp
TMXFP4TOFP8(dstFp8, dstScale8, srcFp4, srcScale4);
```

单次语义操作同时完成：

```text
dstFp8       = bit_unpack_fp4_e2m1_to_fp8_e4m3(srcFp4)
dstScale8[b] = srcScale4[b] * 64
             = srcScale4[b] * 2^6
```

方案 B 的接口要求为：

- payload 的类型、shape、stride、nibble 顺序和尾块语义与方案 A 相同；
- `srcScale4`/`dstScale8` 均为 E8M0 scale Tile，并与对应 payload 使用一致的 MX block 映射；
- scale 的具体 shape、layout 和多行 block 映射交由 PTO-ISA 在正式接口和文档中细化；
- 允许 `dstScale8` 与 `srcScale4` 指向同一块内存；不允许部分重叠或与 payload Tile 重叠；
- payload 和 scale 均在 `PIPE_V` 内完成，一个完成 event 同时保护两个输出；
- 接口元数据必须显式声明两个输出，CPU trace、依赖分析、后端及 PTOAS 都必须识别 `dstFp8` 和 `dstScale8` 两个写集。

“一条指令完成”指调用方提交一条 PTO-ISA 复合指令并获得一个统一完成 event，不表示 A5 硬件只执行一条 Vector 微指令。A5 lowering 仍由现有 payload 位操作序列和 scale 指数调整序列组成。

不建议将该能力重载为普通 `TCVT`。`TMXFP4TOFP8` 是方向明确的 MX 格式专用转换：方案 A 是 payload-only 重载，必须配合独立 E8M0 操作才能闭环 MX block 转换；方案 B 则直接完成完整的 payload+scale 转换。

## 精确位语义

约定每个源字节中：

- 低 nibble 是逻辑元素 `2*i`；
- 高 nibble 是逻辑元素 `2*i+1`。

对单个四位 E2M1 payload `x`：

```text
FP4 E2M1:  s ee m
FP8 E4M3:  s 00ee m00

fp8_bits = ((x & 0x8) << 4) | ((x & 0x7) << 2)
```

全部 16 种编码的映射如下：

| FP4 | FP8 | FP4 | FP8 |
|---:|---:|---:|---:|
| `0x0` | `0x00` | `0x8` | `0x80` |
| `0x1` | `0x04` | `0x9` | `0x84` |
| `0x2` | `0x08` | `0xA` | `0x88` |
| `0x3` | `0x0C` | `0xB` | `0x8C` |
| `0x4` | `0x10` | `0xC` | `0x90` |
| `0x5` | `0x14` | `0xD` | `0x94` |
| `0x6` | `0x18` | `0xE` | `0x98` |
| `0x7` | `0x1C` | `0xF` | `0x9C` |

该映射没有浮点计算、舍入或重新量化，并保留正负零的 bit pattern。


## Quant/scale 语义

位展开只处理 bit pattern，不应先对 FP4 做 BF16/FP16/FP32 反量化。payload 满足：

```text
value(fp8_payload) = value(fp4_payload) / 64
```

因此等价的 MXFP8 block 必须使用：

```text
scale8 = scale4 * 64
W = fp4_payload * scale4
  = fp8_payload * scale8
```

本需求对 E8M0 编码规定以下边界语义：

- `0x00..0xF8 -> src + 6`，该范围内与 `scale4 × 64` 精确等价；
- `0xFF -> 0xFF`，NaN 传播；
- `0xF9..0xFE` 执行 `+6` 时超出 E8M0 可表示范围，具体溢出处理由 PTO-ISA 细化，不属于本需求承诺的精确等价范围。

方案 A 由同批交付的独立 E8M0 指数调整操作完成上述转换；方案 B 在四操作数 `TMXFP4TOFP8` 内完成。两种方案对 `0x00..0xF8` 的有效 scale 都必须保证最终 MXFP8 block 与原 MXFP4 block 数值精确等价。

不建议用 FP4 → BF16 → FP8 替代该指令：这条路径属于数值反量化/再量化，需要 BF16 中间 Tile 和 FP8 舍入规则，不能自然复现 AscendC 的 bit-exact payload 映射，内存与指令开销也更高。

## AscendC 对标实现

本仓已有可直接对标的 A8W4 实现：

```text
ops-transformer/gmm/grouped_matmul/op_kernel/arch35/
  weight_quant_basic_block/basic_block_vf_mx.h
```

核心函数为 `AntiQuantMxA8W4NzNkVf`，使用的常量为：

```cpp
E2M1_SHIFT_RIGHT_SIZE = 0x2;
SHIFT_LEFT_SIZE       = 0x4;
E2M1_AND_MASK         = 0x9C;
```

核心 Register API 序列为：

```cpp
MicroAPI::LoadAlign<uint8_t, MicroAPI::LoadDist::DIST_US_B8>(wLoad, src, srcAddr);

MicroAPI::ShiftRight(wShr0, wLoad, wShrReg, preg);
MicroAPI::ShiftLeft(wShl, wLoad, wShlReg, preg);
MicroAPI::ShiftRight(wShr1, wShl, wShrReg, preg);
MicroAPI::Select(wSel, wShr1, wShr0, pregVsel);
MicroAPI::And(wAnd, wSel, wAndReg, preg);

MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_NORM_B8>(dst, wAnd, dstAddr, preg);
```

同类代码还可见：

```text
ops-transformer/gmm/grouped_matmul_finalize_routing/op_kernel/arch35/
  weight_quant_basic_block/gmm_fr_mx_a8w4_vf.h

ops-transformer/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/arch35/
  weight_quant_basic_block/gmmsq_quant_mx_vf.h
```

### `DIST_US_B8` 的具体作用

`LoadAlign<uint8_t, DIST_US_B8>` 对应带 `US_B8` distribution 字段的 `vlds` 硬件指令。它一次读取 128 字节，并把每个输入字节复制到相邻的两个 B8 lane：

```text
src: b0 b1 b2 b3 ... b127
dst: b0 b0 b1 b1 ... b127 b127
```

复制以后，两个 lane 分别通过移位得到低 nibble 路径和高 nibble 路径，再由 `vsel` 交替选出正确结果。因此：

- `LoadAlign<uint8_t, DIST_US_B8>` 本身是一条硬件 `vlds`；
- 完整的 FP4 → FP8 转换不是单条硬件转换指令；
- 每 256 个逻辑元素的稳态核心序列约为 7 条 Vector 指令。

| AscendC Register API | A5 Vector intrinsic/硬件动作 |
|---|---|
| `LoadAlign<uint8_t, DIST_US_B8>` | `vlds(..., US_B8)`，128 B → 256 B8 lane |
| `ShiftRight` | `vshr` |
| `ShiftLeft` | `vshl` |
| `Select` | `vsel` |
| `And` | `vand` |
| `StoreAlign<uint8_t, DIST_NORM_B8>` | `vsts(..., NORM_B8)` |

循环外还需要三次 `vdup`，分别生成常量 `2`、`4` 和 `0x9C`。整个过程均在 AIV/Vector 核完成；Cube 核只消费展开后的 FP8 数据执行 `mad_mx`。

### AscendC MXFP8 × MXFP4 端到端数据流示例

AscendC `grouped_matmul` 的 Host 检查器明确识别 `FLOAT8_E4M3FN` 激活和 `FLOAT4_E2M1` 权重组合，并要求相关 scale 使用 E8M0：

```cpp
bool GroupedMatmulWeightQuantChecker::IsMxA8W4NZ(const ge::DataType xDtype,
                                                  const ge::DataType weightDtype) const
{
    return xDtype == ge::DT_FLOAT8_E4M3FN &&
           (weightDtype == ge::DT_FLOAT4_E2M1 || weightDtype == ge::DT_FLOAT);
}
```

AIV 调用 `AntiQuantMxA8W4NzNkVf` 将 packed 权重写到类型为 `xType = fp8_e4m3fn_t` 的目标，再发布 AIV→AIC 完成信号；AIC 等待后消费展开权重：

```cpp
// AIV
AntiQuantMxA8W4NzNkVf<xType, wType, biasType, false, false>(mxA8W4NzParams);
SetAivToAic();

// AIC
WaitAivToAic();
cubeCompute_.LaunchMatmul(weightL1_[(cvLoopIdx_ & 1) * weightL1DbOffset_],
                          kbGmOffset, kbL1RealSize, cvLoopIdx_, offsetParam);
```

`LaunchMatmul` 的 MX 分支把 FP8 激活、展开后的 FP8 权重及两路 E8M0 scale 传给 `mmObj_.Iterate`，最终调用 `AscendC::Mmad`。源码在该抽象层没有直接书写 `mad_mx`；结合模板类型和调用参数，可以确认现有 A8W4 路径是在 AIV 把 MXFP4 权重准备成 MXFP8 表示，再由 AIC 执行同宽 MX 矩阵乘，底层具体指令名仍以正式反汇编为准。

主要源码路径：

```text
ops-transformer/gmm/grouped_matmul/op_host/
  grouped_matmul_infershape_weight_quant_checker.cpp

ops-transformer/gmm/grouped_matmul/op_kernel/arch35/
  weight_quant_basic_block/basic_block_vf_mx.h
  weight_quant_basic_block/weight_quant_vec_compute.h
  weight_quant_basic_block/weight_quant_basic_block.h
  weight_quant_basic_block/weight_quant_cube_compute.h
  weight_quant_basic_block/basic_api/weight_quant_basic_api_v1.h
```


## 建议修改方法


### 1. 公共 ISA 接口

方案 A 在 `include/pto/common/pto_instr.hpp` 增加二操作数形式，并在同批能力中增加独立 E8M0 指数调整操作：

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMXFP4TOFP8(DstTileData &dst, SrcTileData &src, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMXFP4TOFP8, dst, src);
    return {};
}
```

方案 B 增加四操作数重载。该重载是“双输出、双输入”操作，必须使用双输出角色声明：

```cpp
TMXFP4TOFP8(dstFp8, dstScale8, srcFp4, srcScale4, events...);

// 分发层使用以下形式或语义等价的双输出声明：
MAP_INSTR_IMPL_OUTS(TMXFP4TOFP8, 2, dstFp8, dstScale8, srcFp4, srcScale4);
```

如果公共宏的实际参数形式不同，可以采用等价声明，但不能退化为普通单输出 `MAP_INSTR_IMPL`。两个写集必须贯穿 CPU trace、依赖分析、CPU/A5 实现和 PTOAS 表示。

在 `include/pto/common/pto_instr_impl.hpp` 注册 CPU 和 A5 实现。


### 2. Event/pipeline 元数据

在 `include/pto/common/event.hpp`：

```cpp
enum class Op {
    // ...
    TMXFP4TOFP8,
};

PTO_DEFINE_OP_PIPE(Op::TMXFP4TOFP8, PIPE_V);
```

方案 B 的一个完成 event 必须同时覆盖 payload 与 scale 两个输出，消费者不得只等待其中一路。这里的 `PIPE_V` event 只描述 AIV 内部依赖；AIV 权重转换到 AIC `TMATMUL_MX` 的跨核依赖由算子 orchestration 的完成信号建立。

### 3. A5 lowering

新增 `include/pto/npu/a5/TMxFp4ToFp8.hpp`。每轮处理最多 256 个逻辑元素：

```cpp
vlds(loadReg, srcPtr, srcOffset, US_B8);
vshr(highReg, loadReg, shiftRightReg, activeB8, MODE_ZEROING);
vshl(shiftedLeftReg, loadReg, shiftLeftReg, activeB8, MODE_ZEROING);
vshr(lowReg, shiftedLeftReg, shiftRightReg, activeB8, MODE_ZEROING);
vsel(selectedReg, lowReg, highReg, selectB16);
vand(outputReg, selectedReg, andMaskReg, activeB8, MODE_ZEROING);
vsts(outputReg, dstPtr, dstOffset, NORM_B8, activeB8);
```

地址计算应按物理字节区分：

```text
srcRowStrideBytes = srcLogicalRowStride / 2
dstRowStrideBytes = dstLogicalRowStride
srcOffset         = row * srcRowStrideBytes + repeat * 128
dstOffset         = row * dstRowStrideBytes + repeat * 256
```

尾块使用 B8 predicate 屏蔽无效输出。

方案 B 在同一 lowering 中增加 E8M0 scale 的读取、指数调整和写回，并与 payload 路径统一排程。scale 中间结果不得物化为额外 UB/GM Tile。对 `0x00..0xF8` 执行 `+6`，对 `0xFF` 保持 `0xFF`，`0xF9..0xFE` 的溢出行为按 PTO-ISA 最终定义实现。


### 4. CPU reference

新增 `include/pto/cpu/TMxFp4ToFp8.hpp`，按逻辑元素读取相应 nibble，并使用同一位公式生成 golden/reference 结果，用于验证 packed addressing、奇数尾部和多行 stride。


### 5. 文档和 PTOAS

新增 `TMXFP4TOFP8` ISA 文档，至少明确：

- 数据类型与 Tile 约束；
- low/high nibble 顺序；
- 逻辑 shape 与物理字节数；
- 尾块行为；
- payload `/64` 和 scale `×64` 关系；
- E8M0 编码边界；
- payload 与 cast/`TCVT` 一致的别名规则；
- 方案 B 的双输出、统一 event 和 scale 别名语义；
- A5 lowering 和 scratch 内存需求。

scale 的具体 shape、layout 和多行 block 映射由 PTO-ISA 文档细化。

PTOAS 同时支持二操作数和四操作数形式。四操作数形式应使用双结果表示，例如：

```text
%dst = pto.tmxfp4tofp8 %src : !pto.tile<...> -> !pto.tile<...>
%dst_payload, %dst_scale = pto.tmxfp4tofp8 %src_payload, %src_scale : ...
```


### 6. 测试

CPU 和 A5 ST 至少覆盖：

- 全部 16 种 FP4 编码；
- 正零和负零；
- 128 字节输入展开为 256 个有序 FP8 元素；
- 跨 256-lane repeat；
- 多行及行 stride；
- 奇数有效列尾部；
- `TLOAD -> TMXFP4TOFP8 -> TSTORE_VEC` event 依赖；
- 方案 A 的独立 E8M0 指数调整以及方案 B 的融合 scale 路径；
- E8M0 `0x00..0xF8 -> src + 6` 和 `0xFF -> 0xFF`；
- 方案 B 的两个输出写集、原地/非原地 scale 和统一完成 event；
- MXFP8 激活 × MXFP4 权重端到端样例：方案 A/B 分别在 AIV 生成 MXFP8 权重及 scale，通过跨核完成信号交给 AIC，再调用现有 FP8 × FP8 `TMATMUL_MX`。

## 内存开销

设 Tile 有 `R` 行、每行逻辑 stride 为 `C` 个元素：

| 项目 | 空间 |
|---|---:|
| 输入 FP4 payload | `R * C / 2` 字节 |
| 输出 FP8 payload | `R * C` 字节 |
| 表示展开导致的容量增加 | `R * C / 2` 字节 |
| 额外 scratch UB/GM | `0` 字节 |

中间结果只保存在 Vector 寄存器中。上表的输出扩容是结果本身，不属于辅助内存。

方案 B 的 scale 路径也不得产生额外 UB/GM 临时 Tile。若 `dstScale8` 与 `srcScale4` 原地别名，scale workspace 为 `0`；若二者独立，只增加必要的目标 scale 容量。scale shape/layout 及每个 payload block 对应的 scale 数量由 PTO-ISA 细化。

本需求只确认减少中间转换和访存步骤的定性收益，不将具体时延、吞吐提升或相对回退路径的性能对比作为验收项。




## 验收标准

1. 全部 16 种 E2M1 编码得到本文规定的 E4M3 字节。
2. 每个 packed byte 的低 nibble 先输出、高 nibble 后输出，正负零 bit pattern 保留。
3. 支持多行、跨 256-lane repeat、静态行 stride 和奇数有效列尾部。
4. A5 编译/反汇编中出现预期的 `vlds`、3 条 shift、`vsel`、`vand`、`vsts` 序列，不生成标量逐元素循环。
5. 不经过 BF16、FP16 或 FP32，不产生浮点舍入。
6. 除输入、输出及可选的独立目标 scale 外，不分配额外 UB/GM workspace；正式实现记录 Vector 寄存器分配并检查 spill。
7. `TMXFP4TOFP8` 标记为 `PIPE_V`，AIV 内 event 同步正确。
8. CPU reference、A5 simulator 与真实板卡输出一致。
9. 方案 A 的独立 E8M0 操作和方案 B 的融合 scale 路径均满足 `0x00..0xF8 -> src + 6`、`0xFF -> 0xFF`；对 `0x00..0xF8` 校验最终 MXFP8 block 与原 MXFP4 block 数值精确等价。
10. 方案 B 使用 `MAP_INSTR_IMPL_OUTS(TMXFP4TOFP8, 2, ...)` 或等价双输出声明，CPU trace、依赖分析、后端和 PTOAS 均正确识别 payload/scale 两个写集，一个完成 event 同时保护两个输出。
11. 方案 A/B 均通过 MXFP8 激活 × MXFP4 权重的 AIV→AIC→`TMATMUL_MX` 端到端数值测试。
12. 方案 B 不为 scale 算术分配额外 UB/GM 临时 Tile；原地 scale workspace 为 `0`，非原地只增加必要的目标 scale 容量。

## 可行性验证



为确认该能力缺失属于 PTO-ISA 表达层问题，而不是 A5 硬件能力缺失，本地实现了一版完整的 `TMXFP4TOFP8` 原型。该原型在一次回退检查后已经按原补丁恢复，并重新完成 CPU、simulator 和真实 A5 板卡验证。本节记录修改方法和实际验证结果，便于后续正式实现复现。

当前原型和以下验证结果只覆盖方案 A 的 payload-only `TMXFP4TOFP8`，尚未证明独立 E8M0 指数调整操作或方案 B 已实现。方案 B 的融合 scale、双输出元数据、统一 event 和 AIV→AIC 端到端样例仍属于本次需求新增的设计与验收范围。


### 原型修改范围

原型沿用现有 PTO-ISA Tile 指令的分层结构，修改范围如下：

| 层次 | 原型修改 | 目的 |
|---|---|---|
| 公共 API | 在 `include/pto/common/pto_instr.hpp` 增加 `TMXFP4TOFP8(dst, src, events...)` | 向上层提供一个 packed FP4 到 unpacked FP8 的稳定 Tile 语义 |
| 指令分发 | 在 `include/pto/common/pto_instr_impl.hpp` 注册 CPU 与 A5 实现 | 使同一接口可用于 CPU golden 和 A5 kernel |
| Pipeline/event | 在 `include/pto/common/event.hpp` 增加 `Op::TMXFP4TOFP8`，映射到 `PIPE_V` | 允许与 `TLOAD`、`TSTORE_VEC` 建立现有 event 依赖 |
| CPU reference | 新增 `include/pto/cpu/TMxFp4ToFp8.hpp` | 按 nibble 读取源数据并生成 bit-exact FP8 golden |
| A5 lowering | 新增 `include/pto/npu/a5/TMxFp4ToFp8.hpp` | 将复合指令展开为现有 A5 Vector intrinsic 序列 |
| CPU ST | 增加 `tests/cpu/st/testcase/tmxfp4tofp8/` 并注册到 CPU ST CMake | 验证接口约束、packed addressing 和完整编码映射 |
| A5 ST | 增加 `tests/npu/a5/src/st/testcase/tmxfp4tofp8/` 并注册到 A5 ST CMake/运行脚本 | 验证 CCE 编译以及 `dav_3510` simulator 执行结果 |
| ISA 文档 | 增加中英文 `TMXFP4TOFP8` 文档并更新 Tile ISA 索引 | 固化类型、shape、nibble 顺序和 scale 责任 |

该原型没有修改芯片指令定义，也没有增加新的硬件 intrinsic；A5 实现只组合已经存在的 `vlds`、`vshr`、`vshl`、`vsel`、`vand` 和 `vsts`。

### A5 lowering 的实现方式

A5 实现按行处理，每个完整 repeat 消费 128 字节打包 FP4，即 256 个逻辑 FP4 元素，并输出 256 字节 FP8：

```text
128 B packed FP4
    |
    | vlds(..., DIST_US_B8)
    v
256 个 B8 lane：b0,b0,b1,b1,...,b127,b127
    |
    | 3 × shift + vsel + vand
    v
256 个 FP8 payload
    |
    | vsts(..., DIST_NORM_B8)
    v
256 B unpacked FP8
```

具体计算过程为：

1. `vlds(..., DIST_US_B8)` 将每个 packed byte 复制到相邻两个 B8 lane；
2. 一条 `vshr` 形成高 nibble 路径；
3. `vshl` 加一条 `vshr` 形成低 nibble 路径；
4. `vsel` 根据交替 predicate，使偶数 lane 选择低 nibble、奇数 lane 选择高 nibble；
5. `vand 0x9C` 清除不属于目标 E4M3 编码的位；
6. `vsts(..., DIST_NORM_B8)` 将 256 个 FP8 byte 连续写回目标 Tile。

循环外通过 `vdup` 生成 `2`、`4` 和 `0x9C` 三个常量寄存器。中间结果始终保存在 Vector 寄存器中，没有申请额外 UB 或 GM scratch。对不足 256 个元素的尾块，使用 B8 predicate 限制有效 lane；源地址以 packed byte 为单位推进，目标地址以 FP8 byte 为单位推进。

### CPU reference 的实现方式

CPU 实现没有调用浮点 cast，而是对每个逻辑元素执行如下步骤：

```cpp
uint8_t packed = src[row * srcStrideBytes + col / 2];
uint8_t fp4 = (col % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
uint8_t fp8 = ((fp4 & 0x8) << 4) | ((fp4 & 0x7) << 2);
dst[row * dstStrideBytes + col] = fp8;
```

该 reference 同时验证了三个容易出错的语义：

- 每个源 byte 的低 nibble 对应偶数逻辑元素，高 nibble 对应奇数逻辑元素；
- 源行 stride 按两个 FP4 共用一个 byte 计算，目标行 stride 按一个 FP8 一个 byte 计算；
- 有效列数为奇数时，最后一个 packed byte 的未使用 nibble 不写入目标。

### 测试数据设计

CPU ST 和 A5 ST 使用 `2 × 257` 的有效 shape。每行按 `0x0` 到 `0xF` 循环填充 FP4 code，因此一个用例同时覆盖：

- 全部 16 种 E2M1 bit pattern，包括正零 `0x0` 和负零 `0x8`；
- packed byte 内 low/high nibble 的输出顺序；
- 第一个 256-lane 完整 repeat；
- 第 257 个元素形成的跨 repeat 尾块；
- 奇数有效列导致的半字节尾部；
- 第二行起始地址和行 stride 计算。

测试的期望值由 CPU 位公式生成，A5 simulator 输出按逻辑元素与期望 FP8 byte 比较。该测试比较的是 payload bit pattern，而不是将结果转成更高精度浮点后做容差比较，因此可以发现 nibble 顺序、符号位、尾块 predicate 或地址步长错误。

### 实际验证结果

| 验证项 | 结果 | 说明 |
|---|---|---|
| CPU 全量构建 | 通过 | 新增公共接口、CPU 实现和测试注册未破坏已有 CPU 构建 |
| CPU 全量测试 | 通过 | `TMXFP4TOFP8` CPU reference 及原有 CPU 用例均通过 |
| A5 CCE 编译 | 通过 | `TMXFP4TOFP8` 模板实例化、`PIPE_V` event 元数据以及全部 Vector intrinsic 均可被 A5 工具链接受 |
| `dav_3510` simulator | 通过 | `2 × 257` 用例输出与 CPU bit-exact golden 一致 |
| Ascend950PR 真实板卡 | 通过 | `task-submit` 分配设备 0，NPU 模式重新编译后在板执行同一 bit-exact 用例，1/1 case 通过 |
| 16 种 FP4 编码 | 通过 | 映射依次为 `00,04,08,0C,10,14,18,1C,80,84,88,8C,90,94,98,9C` |
| 256-lane 边界 | 通过 | 一个完整 repeat 加一个 tail 元素的地址和 predicate 正确 |
| 奇数列尾部 | 通过 | 第 257 个元素正确输出，未使用 nibble 不产生额外输出 |
| 多行访问 | 通过 | 两行输出均与 golden 一致，packed 源 stride 与 FP8 目标 stride 正确 |
| 辅助内存 | 0 字节 | 除源 FP4 Tile 和目标 FP8 Tile 外，中间值仅占用 Vector 寄存器 |

另对全部 16 种编码进行了独立的 host 侧数值关系校验：位展开结果均满足

```text
value(fp4_payload) = value(fp8_payload) * 64
```

因此调用方使用 `scale8 = scale4 * 64` 后，可恢复与原 MXFP4 block 相同的实数值。该检查也确认 `0x0 -> 0x00`、`0x8 -> 0x80`，正负零的符号 bit pattern 均得到保留。

### 真实板卡验证

上述方案已通过真实板卡验证。原型结果说明：A5 已具备完成该位展开所需的全部底层 Vector 指令，PTO-ISA 只需增加复合 Tile 指令、CPU reference、A5 lowering、文档和测试，不需要新增芯片指令，也不需要额外 workspace。

本次原型已经验证功能正确性、CCE 可编译性、simulator 可执行性和 Ascend950PR 真实板卡可执行性。正式合入前仍建议补充两项验收证据：

1. 对生成的 A5 二进制做反汇编，确认稳态循环确实为预期的 `vlds`、3 条 shift、`vsel`、`vand`、`vsts`，且没有退化为标量逐元素循环；
2. 增加 `TLOAD -> TMXFP4TOFP8 -> TSTORE_VEC` 的独立 event 并发/同步用例，以及接入 `TMATMUL_MX` 后应用 `scale ×64` 的端到端矩阵乘数值测试。

## 验证环境

```text
Initial simulator: CANN 9.1.0-beta.3 / A5 dav_3510
Revalidation: CANN 9.2.0 / A5 dav_3510 simulator
Hardware: Ascend950PR device 0 / RUN_MODE=npu / compile SoC Ascend910_9599
```


---

## #242 [Bug] TPREFETCH_ASYNC is 12.8x slower than an equivalent TLOAD and scales negatively with SDMA channels

- State: open
- URL: https://github.com/hw-native-sys/pto-isa/issues/242
- Created: 2026-08-12T01:01:22Z
- Updated: 2026-08-12T01:09:14Z
- Closed: 

### Body

## Background

In pypto-lib, `models/deepseek_v4_flash_mtp/decode_hca.py` on a2a3 has a **performance** problem: `TPREFETCH_ASYNC` used to warm a 67.1 MB weight into L2 occupies the issuing AICore for 0.4–4.2 ms, one to two orders of magnitude longer than an ordinary `TLOAD` of the same bytes, and it gets **worse as SDMA channels are added**.

The behaviour is not specific to that model — a standalone repro is inlined below.

Reproduce with:

    python models/deepseek_v4_flash_mtp/prefetch_contention_repro.py -p a2a3 -d 0 \
        --mode {tload|prefetch} --channels {1|8} --enable-l2-swimlane 1

then read `kernel-duration-us` of the `warm` task from
`build_output/_jit_warm_*/dfx_outputs/merged_swimlane_*.json`.

The repro case is **not committed** — save the file below at
`models/deepseek_v4_flash_mtp/prefetch_contention_repro.py` to run it.

Reproduction environment:

| Component | Version |
|---|---|
| pypto-lib | `61989ebd` (branch: `eplb`) |
| pypto | `d315f62d` (branch: `main`) |
| simpler | `3165cc89` (branch: detached) |
| ptoas | 0.54 |
| pto-isa | `83d01313` |
| CANN | 9.0.0 |

All three pins verified consistent (simpler gitlink, `runtime/pto_isa.pin`, `toolchain/versions.env`).

Diagnosis: **pto-isa** — the cost is inside `TPREFETCH_ASYNC` / `__sdma_cmo_prefetch`; the AICore's own `kernel-duration-us` covers it, and the only variable between the two measured modes is which tile op the warm scope uses.

## Measurements

Standalone repro, a2a3 silicon, 67.1 MB warmed while an independent scope streams 33.6 MB through `pto.tload` (the concurrent traffic is required — see Notes):

| warm implementation | blocks | kern median | kern max | effective BW |
|---|---:|---:|---:|---:|
| `pto.tload` (control) | 8 | **83.6 us** | 90.5 us | **803 GB/s** |
| `pto.tprefetch_async` | 1 | 432.1 us | 432.1 us | 155 GB/s |
| `pto.tprefetch_async` | 8 | **1068.6 us** | 2983.5 us | **63 GB/s** |

Two problems:

1. **Absolute cost** — at 8 blocks, `TPREFETCH_ASYNC` is **12.8x** slower than `TLOAD` moving the same bytes, even though the CMO only warms L2 and lands nothing in UB.
2. **Negative scaling** — going from 1 to 8 channels makes it **2.5x slower** (155 -> 63 GB/s). This contradicts the interface, which sets `channelGroupIdx = get_block_idx()` and allows up to `kSdmaMaxChannel = 48`, i.e. it is designed to be sharded across channels.

The same trend reproduces inside a real kernel (`decode_hca.py`, DeepSeek-V4 Flash HCA decode attention, `--start-pos 8192`), where the warm scope is the only difference:

| channels | kern median | effective BW |
|---:|---:|---:|
| 1 | 454.1 us | 148 GB/s |
| 8 | 1000.4 us | 67 GB/s |
| 32 | 4158.4 us | 16 GB/s |

At 32 channels the per-channel rate collapses to 0.5 GB/s.

## Notes

- **Concurrent load is required to expose this.** With the warm scope alone on an idle chip, the CMO submission returns quickly (~15 us for 8 shards) and the cost is invisible; the descriptors land later, untimed. The repro therefore includes an independent `busy_stream` scope.
- `TPREFETCH_ASYNC` correctly does **not** contend with the vector MTE path: in the real kernel the neighbouring `hc_pre` scope (24 AIVs) is unaffected (21.7 us vs 20.1 us baseline), whereas a `TLOAD`-based warm pushes it to 56.4 us. The problem is the CMO's own throughput and its channel scaling, not interference.
- No `wait` is used in any variant, so this measures submission + whatever the core blocks on inside `__sdma_cmo_prefetch`, not completion.
- Generated `.cpp` / `.pto` for all three variants are attached below.

<details>
<summary>Reproduction case — <code>prefetch_contention_repro.py</code> (save as <code>models/deepseek_v4_flash_mtp/prefetch_contention_repro.py</code>)</summary>

```python
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim  # TPREFETCH_ASYNC has no simulator provider
"""Standalone repro: TPREFETCH_ASYNC cost under concurrent memory traffic.

Warms a 67.1 MB GM region into L2 two ways while an independent scope streams a
second region through ordinary tile loads, so the device is busy the way a real
kernel is:

  --mode prefetch   warm via pto.tprefetch_async (SDMA CMO, opcode 6)
  --mode tload      warm via pto.tload (control: same bytes, same block count)

``--channels N`` sets the warm scope's SPMD width; for the prefetch mode that is
also the SDMA channel count, since channelGroupIdx = get_block_idx().

Read the warm task's cost from the swimlane record:

    python prefetch_contention_repro.py -p a2a3 -d 0 --mode prefetch --channels 8 \
        --enable-l2-swimlane 1

then take ``kernel-duration-us`` of the ``warm`` task out of
``build_output/_jit_*/dfx_outputs/merged_swimlane_*.json``.

In isolation (no concurrent scope) the CMO submission returns quickly and the
cost is hidden; the concurrent scope is what exposes it.
"""

import argparse

import pypto.language as pl

# model config (one DeepSeek-V4 Flash layer's o-projection A matrix)
WARM_ROWS = 8192
WARM_COLS = 4096                              # 8192 x 4096 BF16 = 67.1 MB
WARM_NELEM = WARM_ROWS * WARM_COLS

# concurrent traffic: an independent region streamed while the warm runs
BUSY_ROWS = 4096
BUSY_COLS = 4096                              # 33.6 MB BF16
BUSY_BLOCKS = 16                              # 4096/16 = 256 rows, divisible by BUSY_ROW_TILE
BUSY_ROW_TILE = 8

# tiling
ROW_TILE = 8                                  # 8 x 4096 BF16 = 64 KB per tile
MARK = 128

CHANNELS = 8                                  # overwritten from --channels at import
MODE = "prefetch"                             # overwritten from --mode at import


def _parse_argv():
    """Read --channels / --mode before the kernels are traced."""
    import sys

    ch, mode = 8, "prefetch"
    for i, a in enumerate(sys.argv):
        if a == "--channels" and i + 1 < len(sys.argv):
            ch = int(sys.argv[i + 1])
        if a == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
    return ch, mode


CHANNELS, MODE = _parse_argv()
SHARD = WARM_NELEM // CHANNELS
WARM_ROWS_PER_BLOCK = WARM_ROWS // CHANNELS

assert WARM_NELEM % CHANNELS == 0
assert WARM_ROWS % CHANNELS == 0 and WARM_ROWS_PER_BLOCK % ROW_TILE == 0
assert CHANNELS <= 48, "channelGroupIdx must stay under kSdmaMaxChannel = 48"
assert (BUSY_ROWS // BUSY_BLOCKS) % BUSY_ROW_TILE == 0

SHARD8 = WARM_NELEM // 8      # per-channel byte extent must be a compile-time constant


@pl.jit
def warm_prefetch_1(
    w: pl.Tensor[[WARM_ROWS, WARM_COLS], pl.BF16],
    busy: pl.Tensor[[BUSY_ROWS, BUSY_COLS], pl.BF16],
    mark: pl.Out[pl.Tensor[[BUSY_BLOCKS, MARK], pl.BF16]],
):
    # Independent streaming scope: keeps the device busy so the warm scope is
    # measured under load rather than on an idle chip.
    for busy_blk in pl.spmd(BUSY_BLOCKS, name_hint="busy_stream"):
        b0 = busy_blk * (BUSY_ROWS // BUSY_BLOCKS)
        for kb in pl.pipeline((BUSY_ROWS // BUSY_BLOCKS) // BUSY_ROW_TILE, stage=2):
            rb = b0 + kb * BUSY_ROW_TILE
            tb = busy[rb : rb + BUSY_ROW_TILE, 0:BUSY_COLS]
        mark[busy_blk : busy_blk + 1, 0:MARK] = busy[b0 : b0 + 1, 0:MARK]
    # One SDMA channel carries the whole region.
    w_flat = pl.reshape(w, [WARM_NELEM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="warm"):
        ctx = pl.prefetch.make_context()
        pl.prefetch.async_prefetch(w_flat, ctx)
    return mark


@pl.jit
def warm_prefetch_8(
    w: pl.Tensor[[WARM_ROWS, WARM_COLS], pl.BF16],
    busy: pl.Tensor[[BUSY_ROWS, BUSY_COLS], pl.BF16],
    mark: pl.Out[pl.Tensor[[BUSY_BLOCKS, MARK], pl.BF16]],
):
    # Independent streaming scope: keeps the device busy so the warm scope is
    # measured under load rather than on an idle chip.
    for busy_blk in pl.spmd(BUSY_BLOCKS, name_hint="busy_stream"):
        b0 = busy_blk * (BUSY_ROWS // BUSY_BLOCKS)
        for kb in pl.pipeline((BUSY_ROWS // BUSY_BLOCKS) // BUSY_ROW_TILE, stage=2):
            rb = b0 + kb * BUSY_ROW_TILE
            tb = busy[rb : rb + BUSY_ROW_TILE, 0:BUSY_COLS]
        mark[busy_blk : busy_blk + 1, 0:MARK] = busy[b0 : b0 + 1, 0:MARK]
    # Eight SDMA channels, one shard each (channelGroupIdx = get_block_idx()).
    # async_prefetch needs a whole flat-1D GM tensor, so each branch names its
    # own compile-time slice.
    w_flat = pl.reshape(w, [WARM_NELEM])
    s0 = pl.slice(w_flat, [SHARD8], [0 * SHARD8])
    s1 = pl.slice(w_flat, [SHARD8], [1 * SHARD8])
    s2 = pl.slice(w_flat, [SHARD8], [2 * SHARD8])
    s3 = pl.slice(w_flat, [SHARD8], [3 * SHARD8])
    s4 = pl.slice(w_flat, [SHARD8], [4 * SHARD8])
    s5 = pl.slice(w_flat, [SHARD8], [5 * SHARD8])
    s6 = pl.slice(w_flat, [SHARD8], [6 * SHARD8])
    s7 = pl.slice(w_flat, [SHARD8], [7 * SHARD8])
    with pl.spmd(8, name_hint="warm"):
        blk = pl.tile.get_block_idx()
        ctx = pl.prefetch.make_context()
        if blk == 0:
            pl.prefetch.async_prefetch(s0, ctx)
        elif blk == 1:
            pl.prefetch.async_prefetch(s1, ctx)
        elif blk == 2:
            pl.prefetch.async_prefetch(s2, ctx)
        elif blk == 3:
            pl.prefetch.async_prefetch(s3, ctx)
        elif blk == 4:
            pl.prefetch.async_prefetch(s4, ctx)
        elif blk == 5:
            pl.prefetch.async_prefetch(s5, ctx)
        elif blk == 6:
            pl.prefetch.async_prefetch(s6, ctx)
        else:
            pl.prefetch.async_prefetch(s7, ctx)
    return mark


@pl.jit
def warm_tload(
    w: pl.Tensor[[WARM_ROWS, WARM_COLS], pl.BF16],
    busy: pl.Tensor[[BUSY_ROWS, BUSY_COLS], pl.BF16],
    mark: pl.Out[pl.Tensor[[BUSY_BLOCKS, MARK], pl.BF16]],
):
    # Independent streaming scope: keeps the device busy so the warm scope is
    # measured under load rather than on an idle chip.
    for busy_blk in pl.spmd(BUSY_BLOCKS, name_hint="busy_stream"):
        b0 = busy_blk * (BUSY_ROWS // BUSY_BLOCKS)
        for kb in pl.pipeline((BUSY_ROWS // BUSY_BLOCKS) // BUSY_ROW_TILE, stage=2):
            rb = b0 + kb * BUSY_ROW_TILE
            tb = busy[rb : rb + BUSY_ROW_TILE, 0:BUSY_COLS]
        mark[busy_blk : busy_blk + 1, 0:MARK] = busy[b0 : b0 + 1, 0:MARK]
    # Control: same bytes, same block count, via ordinary tile loads. The loaded
    # tile is never consumed -- an unused tile.load still lowers to pto.tload,
    # so no arithmetic is involved.
    with pl.spmd(CHANNELS, name_hint="warm"):
        blk = pl.tile.get_block_idx()
        a0 = blk * WARM_ROWS_PER_BLOCK
        for ka in pl.pipeline(WARM_ROWS_PER_BLOCK // ROW_TILE, stage=2):
            ra = a0 + ka * ROW_TILE
            ta = w[ra : ra + ROW_TILE, 0:WARM_COLS]
    return mark


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    return [
        TensorSpec("w", [WARM_ROWS, WARM_COLS], torch.bfloat16,
                   init_value=lambda: torch.randn(WARM_ROWS, WARM_COLS).to(torch.bfloat16)),
        TensorSpec("busy", [BUSY_ROWS, BUSY_COLS], torch.bfloat16,
                   init_value=lambda: torch.randn(BUSY_ROWS, BUSY_COLS).to(torch.bfloat16)),
        TensorSpec("mark", [BUSY_BLOCKS, MARK], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    from golden import run_jit

    parser = argparse.ArgumentParser(description="TPREFETCH_ASYNC cost under concurrent traffic")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["prefetch", "tload"], default="prefetch")
    parser.add_argument("--channels", type=int, default=8, choices=[1, 8])
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    print(f"--- warm {WARM_NELEM * 2 / 1e6:.1f} MB via {MODE} over {CHANNELS} blocks, "
          f"concurrent {BUSY_ROWS * BUSY_COLS * 2 / 1e6:.1f} MB tload stream ---", flush=True)
    result = run_jit(
        fn=(warm_tload if args.mode == "tload"
            else (warm_prefetch_1 if args.channels == 1 else warm_prefetch_8)),
        specs=build_tensor_specs(),
        golden_fn=None,
        runtime_cfg=dict(platform=args.platform, device_id=args.device,
                         enable_l2_swimlane=args.enable_l2_swimlane),
        compile_only=args.compile_only,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
```

</details>

## Attached artifacts

`pto_isa_242_repro.tar.gz` (680 KB) carries the repro plus the complete build output of all three
variants, so the generated instruction sequences can be inspected without rebuilding:

```
prefetch_contention_repro.py                        the repro above, ready to drop into
                                                    models/deepseek_v4_flash_mtp/
build_output/_jit_warm_tload_20260811_175645/       control: warm via pto.tload,          8 blocks
build_output/_jit_warm_prefetch_1_20260811_175707/  warm via pto.tprefetch_async, 1 SDMA channel
build_output/_jit_warm_prefetch_8_20260811_175731/  warm via pto.tprefetch_async, 8 SDMA channels
```

Inside each build directory:

| Path | Contents |
|---|---|
| `ptoas/warm.pto` | **PTO bytecode of the warm scope — the instruction sequence in question** |
| `ptoas/busy_stream.pto` | PTO bytecode of the concurrent streaming scope (identical across all three) |
| `ptoas/*.cpp` | ptoas-generated sources |
| `kernels/aiv/*.cpp`, `*.o` | per-core kernel sources / objects |
| `orchestration/*.cpp`, `*.so` | task orchestration |
| `dfx_outputs/merged_swimlane_*.json` | L2 swimlane; the `warm` task's `kernel-duration-us` is the number quoted above |
| `dfx_outputs/l2_swimlane_records.json` | raw swimlane records |
| `report/perf_hints.log` | absolute paths replaced with `<repo>` / `<home>` placeholders |

`README.txt` at the archive root repeats this layout and the measured numbers.


---

## #245 [Review] PR #1442/#1453/#1463 follow-ups: API closure and conflicting instruction lifecycle

- State: open
- URL: https://github.com/hw-native-sys/pto-isa/issues/245
- Created: 2026-08-13T01:53:41Z
- Updated: 2026-08-14T03:36:08Z
- Closed: 

### Body

## 背景

对以下 GitCode PR 做了联合 review：

- [PR #1442：为 TSTORE、TEXTRACT、TINSERT、TMOV 新增同名 fp 重载](https://gitcode.com/cann/pto-isa/pull/1442)
- [PR #1453：移除已标记清理的旧指令接口](https://gitcode.com/cann/pto-isa/pull/1453)
- [PR #1463：重命名两条融合乘加指令](https://gitcode.com/cann/pto-isa/pull/1463)

总体建议：这三个 PR 作为同一批次目前应当 **Request Changes**。

## Blocking issues

### 1. PR #1453 与 PR #1463 的目标直接冲突

PR #1453 删除 `TFUSEDMULADD`、`TMULADDDST` 的公共接口、CPU/NPU 实现、ST 和文档；PR #1463 又将同一组指令重命名为：

- `TFUSEDMULADD` → `TMADD`
- `TMULADDDST` → `TMULA`

对两个分支进行合并分析后，在公共头文件、CPU/NPU 实现、ST 注册、manifest 和文档中均存在 delete/rename 或内容冲突。不同合入顺序也可能导致已删除接口被恢复。

建议先明确最终状态：

- 如果保留这两条指令的语义并采用新名称：从 #1453 的清理范围中排除这两条指令，然后将 #1463 rebase 到调整后的 #1453；
- 如果目标是彻底移除：则不应再合入 #1463。

合并解决后应重新执行旧名称全树扫描，以及 CPU/A2A3/A5 的 target、CMake、gtest 和 runner 闭环检查。

### 2. PR #1442 遗漏 Cost Model 公共接口

`include/pto/pto-inst.hpp` 在 `__COSTMODEL` 下选择 `include/pto/costmodel/pto_instr.hpp`，但新的同名 fp 重载只增加到了 `include/pto/common/pto_instr.hpp`。

Cost Model facade 目前仍然只有：

- `TSTORE_FP`
- `TEXTRACT_FP`
- `TINSERT_FP`
- `TMOV_FP`

因此使用新接口的同一份源码在 CPU/NPU 模式可见，但在 `__COSTMODEL` 下不可用。

建议在 Cost Model facade 中同步增加同名重载和 forwarding contract，并保持 `TMOV` scaling tile、`STPhase` 等约束一致；如果 Cost Model 明确不支持，则需要显式记录并用编译约束阻止误用。

### 3. PR #1442 缺少新重载测试

该 PR 的 effective diff 没有修改测试文件；现有 ST 主要仍调用旧 `*_FP` 接口，尚未证明以下内容：

- 四个新同名重载的模板匹配和 overload resolution；
- `TMOV` 的 `STPhase` 形式；
- scaling tile 与普通临时 tile 重载不存在歧义；
- `reluMode`、wait event 的转发；
- 旧 `*_FP` forwarding alias 仍兼容。

建议增加 CPU、A2A3、A5 和 Cost Model 的编译/运行覆盖，并保留代表性的旧接口兼容测试。

## Documentation and hygiene issues

### PR #1463

- `docs/isa/TMADD.md` 和 `docs/isa/TMULA.md` 声称输入 valid shape 没有显式检查，但 A5 的 `TMAddCheck` / `TMulaCheck` 使用了 `PTO_ASSERT`，CPU/A2A3 也有对应约束；中文页面的描述是正确的。
- `include/README.md` 的英文支持表把 `TMULA`、`TMADD` 链接到了 `_zh.md` 页面，应改为英文页面。
- `git diff --check` 实际不通过，涉及 `docs/figures/isa/TMULA.svg` 以及 `TMADD*.md`、`TMULA*.md` 的 CRLF/行尾空白，应统一为 LF。

### PR #1442

- `docs/isa/TEXTRACT_FP.md` 的 AS Level 1/2 示例遗漏 `%fp` 操作数及其 tile 类型。该问题可能不是本次新引入，但本次既然同步修改相关文档，建议一起修正。
- 如果 `*_FP` 仅作为 C++ source-compatible alias 保留，需要确认 manifest 是否仍应把它们作为独立 ISA 指令展示。

### Breaking-change migration notes

仓库手册要求 breaking architectural/AS changes 包含明确的版本与迁移说明。建议至少记录：

```text
TFUSEDMULADD -> TMADD
TMULADDDST  -> TMULA
```

并为 #1453 的其他删除项说明替代接口，或者明确标记“无替代、直接移除”及兼容窗口。

## Per-PR recommendation

- **#1442: Request Changes** — Cost Model API 和测试闭环不完整。
- **#1453: 单独看静态删除闭环较完整**；但作为本批变更的一部分，需要先解决与 #1463 的指令生命周期冲突并补充迁移说明。
- **#1463: Request Changes** — 需要解决与 #1453 的合并冲突，并修正文档和行尾格式。

## Validation performed

- 检查三个 PR 的 effective first-parent diff；
- #1453 删除目标的全树旧名称扫描：无残留；
- #1463 被重命名的两个旧名称扫描：无残留；
- ISA manifest 可解析；
- `tests/run_st.sh` shell 语法检查通过；
- #1442、#1453 的 `git diff --check` 通过；#1463 不通过；
- 对 #1453/#1463 做了合并冲突分析。

历史树 CPU 编译在本地受 macOS/工具链基线限制，NPU ST 环境不可用，因此本次 review 不将完整 CPU/NPU 动态测试标记为通过。


---

## #248 [Bug] A2A3: a codegen change in 83d01313..0cefc9a5 stalls the PTO2 scheduler on mixed cube+vector kernels (S1:running-stalled)

- State: open
- URL: https://github.com/hw-native-sys/pto-isa/issues/248
- Created: 2026-08-14T02:38:48Z
- Updated: 2026-08-17T01:11:54Z
- Closed: 

### Body

## Summary

Somewhere in `83d01313..0cefc9a5`, a2a3 cube/matmul device codegen changed in a way that makes a mixed cube+vector InCore kernel stall the PTO2 scheduler after a few dozen task submissions. The scheduler reports a task that started and never completed:

```
507018 ACL_ERROR_RT_AICPU_EXCEPTION
PTO2 runtime failed: orch_error_code=0 sched_error_code=100 runtime_status=-100
PTO2 scheduler timeout sub_class=S1:running-stalled (detail=1)
    completed=96/160 running=1 ready=0 waiting=63 orch_done=1 stuck_core=1
```

This blocks the pypto runtime bump hw-native-sys/pypto#2345 (`system-tests` + `pypto-lib-model`). **It is still present on `main` (`661b266b`, 2026-08-13)** — bumping forward is not a workaround.

## Attribution

Bisected on device against the pypto ST case `test_qwen3_decode_scope3_mixed`, holding the simpler runtime fixed and varying only the pto-isa pin:

| simpler | pto-isa | result |
| ------- | ------- | ------ |
| `e4ab544a` | `0cefc9a5` (its own pin) | stall |
| `e4ab544a` | `83d01313` (forced back) | **pass** |
| `7a1b9b11` (the pin-bump commit itself) | `83d01313` (forced back) | **pass** |
| `aa1d7c7d` (the wire-ABI change the PR is named after) | `83d01313` | pass |

So the regression follows the **pto-isa pin**, not the simpler source that was bumped alongside it.

Narrowing inside pto-isa: `83d01313` is good, `e8f558d5` (merge #222) is bad, leaving **`(83d01313, e8f558d5]`, 278 commits**. It could not be narrowed further from the pypto side, because the middle of that range is unbuildable in two independent ways: simpler's host build breaks until `3a11aeed` ("keep SDMA host headers CPU-safe"), and below merge #222 the a2a3 headers themselves do not compile (`redefinition of 'TStoreAccFp'`, `undeclared identifier 'dstC0'`). Grafting around the latter would mean porting later ISA fixes, which changes device semantics and voids the experiment.

## What actually differs (artifact comparison)

Compiling the *same* generated kernels against both pins isolates this tightly:

- Every generated source is **byte-identical** — the `.pto`, ptoas' kernel `.cpp`, and the orchestration `.cpp`. Nothing in pypto or ptoas participates.
- Compiled device `.text` differs for **only the two kernels that use the cube**:

| kernel | uses cube | `.text` (good → bad) |
| ------ | --------- | -------------------- |
| rmsnorm, postnorm_block, final_resid_block, zero_down | no | identical |
| **oproj_block** | yes | 996 → 1016 B (aiv), 548 → 560 B (aic) |
| **mlp_block** | yes | 2204 → 2324 B (aiv), 1612 → 1668 B (aic) |

The bad build emits *more* code. Word-level deltas across the four differing objects:

- two encodings swapped 1:1, 15x each: `18709082` → `20909482`, `18708482` → `1c808682`
- ~10 inserted `0010f041`, plus a family of `8008d?02` insertions

(We cannot decode these — CANN's `llvm-objdump` prints `<not available>` for this ISA — but you can.)

## Suspect commits

Filtering the 278 by "touches `include/pto/npu/a2a3/` or `include/pto/common/`" (98), then by the operators that appear only in the two changed kernels and in none of the four unchanged ones — `TMATMUL / TPUSH / TPOP / TPipe / TFREE / TMOV / SetValidShape` — leaves the cross-core FIFO and event paths. `TLOAD` / `TSTORE` / `TCVT` are exonerated: rmsnorm uses them and is byte-identical (caveat: a matmul kernel may instantiate different L0/L1 fractal templates of those same headers). All of these predate the confirmed-bad `e8f558d5`:

- `92eb7f88` Add TPUSH/TPOP interface with subBlockId parameter — moves `get_subblockid()` from callee to call site, changing where the GM FIFO offset is evaluated, and adds the cross-core commit & signal block
- `8d74fb38` update TALLOC/TPUSH/TPOP/TFREE to support push or pop GlobalTensor (122 lines in `pto_instr.hpp`)
- `92492c9d` Update tpush/tpop to transfer data with valid shape
- `fcf53057` refactor: extract Event CRTP base class, replace opPipeList with OpPipeEntry (240 lines in `event.hpp`) — best fit for a systematic 15x 1:1 encoding swap
- lower confidence: `ddb55d58`, `7c52ec53`, `505dbfa6`, `b115184a`, `0ac04ff3`

## Reproduction

Minimal pypto ST case — one Orchestration driver looping 80 output blocks over a single mixed InCore kernel (40 bf16 `[16,128]@[128,64]` matmuls accumulated into one fp32 accumulator, then a cast bias add). 160 tasks, versus 1043 in the model test it was reduced from.

```
pytest test_min_repro.py -q --device <id> --platform a2a3

pto-isa 83d01313  ->  1 passed in 5.35s
pto-isa 0cefc9a5  ->  FAIL, 64s: S1:running-stalled completed=96/160 running=1 ready=0 waiting=63 stuck_core=1
pto-isa 661b266b  ->  FAIL, 63s: S1:running-stalled completed=80/160 running=1 ready=0 waiting=79 stuck_core=2
                       (main @ 2026-08-13 — same reproducer, still stalls)
```

```python
ROWS, K, K_CHUNK, COLS = 16, 5120, 128, 64
K_BLOCKS, OUT_BLOCKS = K // K_CHUNK, K // COLS   # 40, 80

@pl.program
class MatmulChainProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def mm_chain(self, a, w, bias, o0, out):
        acc = pl.full([ROWS, COLS], dtype=pl.FP32, value=0.0)
        for kb in pl.range(K_BLOCKS):
            k0 = kb * K_CHUNK
            acc = pl.add(acc, pl.matmul(
                pl.slice(a, [ROWS, K_CHUNK], [0, k0]),
                pl.slice(w, [K_CHUNK, COLS], [k0, o0]), out_dtype=pl.FP32))
        acc = pl.add(acc, pl.cast(pl.slice(bias, [ROWS, COLS], [0, o0]), target_type=pl.FP32))
        return pl.assemble(out, acc, [0, o0])

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, a, w, bias, out):
        for ob in pl.range(OUT_BLOCKS):
            o0 = ob * COLS
            out = self.mm_chain(a, w, bias, o0, out)
        return out
```

Two properties the reduction must keep, both found by cutting too far:

- **The driver loop.** A single invocation (2 tasks) does not stall. The model case dies near `completed=490/1043` and this one near `96/160` — both about half way, so the trigger needs repeated task submission, not one mixed kernel.
- **Exact-integer inputs** (`torch.randint(-2, 3).float()`). With `randn`, golden and device differ by ~2e-5 from bf16 accumulation order, which masks stall-vs-pass.

Environment: a2a3 hardware, ptoas 0.54 (0.55+ require CPython 3.11, unavailable on our host), pypto at hw-native-sys/pypto#2345.

## Generated CCE

`ptoas/mm_chain.cpp` — the ccec input for the reproducer above, containing both
core halves (`mm_chain_aic` / `mm_chain_aiv`). **This file is byte-identical
under both pins** (as are the `.pto`, the per-core `kernels/{aic,aiv}/*.cpp`,
and the orchestration `main.cpp`) — the same CCE source compiles to different
device code, which is what isolates the change to the ISA headers.

The cross-core traffic it emits:

```cpp
// cube half
auto v16 = TPipe<0, Direction::DIR_C2V, 4096, 8, 8, true>(v5, v15, v15);
TMATMUL(v34, v30, v32);
TPUSH<TPipe<0, Direction::DIR_C2V, 4096, 8, 8, true>,
      Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1,
           SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>,
      TileSplitAxis::TILE_NO_SPLIT>(v16, v34);

// vector half
auto v21 = TPipe<0, Direction::DIR_C2V, 4096, 8, 8, true>(v5, v20, v20);
TPOP<..., Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1,
                SLayout::NoneBox, 512, ...>, TileSplitAxis::TILE_NO_SPLIT>(v21, v25);
TFREE<TPipe<0, Direction::DIR_C2V, 4096, 8, 8, true>,
      TileSplitAxis::TILE_NO_SPLIT>(v21);
```

i.e. a single `DIR_C2V` pipe, `SLOT_SIZE=4096`, `SLOT_NUM=8`, `IsNoSplit=true`,
pushing a `[16,64]` fp32 Acc tile per output block and popping it as a Vec tile.

To compile it exactly as pypto does:

```bash
ccec -c -O3 -x cce --cce-aicore-only --cce-aicore-arch=dav-c220-vec \
     -I<pto-isa>/include -I<pto-isa>/include/pto ... -o mm_chain_aiv.o mm_chain.cpp
# and --cce-aicore-arch=dav-c220-cube for the aic half
```

<details>
<summary>Full generated CCE (ptoas/mm_chain.cpp, 256 lines)</summary>

```cpp
#include "pto/pto-inst.hpp"
using namespace pto;

enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

template <typename Ptr>
static AICORE inline void PTOAS__DCCI_SINGLE_CACHE_LINE(Ptr ptr) {
  dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
}

AICORE void mm_chain_aic(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, __gm__ float* v4, __gm__ float* v5, int64_t v6) {
  const int64_t v7 = 64;
  const int64_t v8 = 128;
  const int64_t v9 = 40;
  const int64_t v10 = 1;
  const int64_t v11 = 5120;
  const int64_t v12 = 16;
  const int64_t v13 = 4096;
  const int64_t v14 = 0;
  const int32_t v15 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  auto v16 = TPipe<0, Direction::DIR_C2V, 4096, 8, 8, true>(v5, v15, v15);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  for (int64_t i17 = v14; i17 < v9; i17 += v10) {
    // pto: %0
    int64_t v18 = (int64_t) ((uint64_t) i17 * (uint64_t) v8);
    // pto: %a_chunk__tile
    Tile<TileType::Mat, bfloat16_t, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v19 = Tile<TileType::Mat, bfloat16_t, 16, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v12, v8);
    // pto: %a_chunk__tile
    uint64_t v20 = (uint64_t) v14;
    TASSIGN(v19, v20);
    // pto: %1
    int64_t v21 = v18 < v14 ? v14 : v18;
    // pto: %a__ssa_v0_pview
    pto::Shape<1, 1, 1, 16, 128> v22 = pto::Shape<1, 1, 1, 16, 128>();
    // pto: %a__ssa_v0_pview
    pto::Stride<81920, 81920, 81920, 5120, 1> v23 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    // pto: %a__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v24 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v1 + ((v14 + v14 * v11) + v21 * v10), v22, v23);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v19, v24);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    // pto: %w_chunk__tile
    Tile<TileType::Mat, bfloat16_t, 128, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v25 = Tile<TileType::Mat, bfloat16_t, 128, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v8, v7);
    // pto: %w_chunk__tile
    uint64_t v26 = (uint64_t) v13;
    TASSIGN(v25, v26);
    // pto: %w__ssa_v0_pview
    pto::Shape<1, 1, 1, 128, 64> v27 = pto::Shape<1, 1, 1, 128, 64>();
    // pto: %w__ssa_v0_pview
    pto::Stride<655360, 655360, 655360, 5120, 1> v28 = pto::Stride<655360, 655360, 655360, 5120, 1>();
    // pto: %w__ssa_v0_pview, %3
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 64>, pto::Stride<655360, 655360, 655360, 5120, 1>, pto::Layout::ND> v29 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 64>, pto::Stride<655360, 655360, 655360, 5120, 1>, pto::Layout::ND>(v2 + ((v14 + v21 * v11) + (v6 < v14 ? v14 : v6) * v10), v27, v28);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    TLOAD(v25, v29);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    // pto: %a_chunk__tile_Left
    Tile<TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v30 = Tile<TileType::Left, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v12, v8);
    // pto: %a_chunk__tile_Left
    uint64_t v31 = (uint64_t) v14;
    TASSIGN(v30, v31);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    TMOV(v30, v19);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    // pto: %w_chunk__tile_Right
    Tile<TileType::Right, bfloat16_t, 128, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v32 = Tile<TileType::Right, bfloat16_t, 128, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v8, v7);
    // pto: %w_chunk__tile_Right
    uint64_t v33 = (uint64_t) v14;
    TASSIGN(v32, v33);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TMOV(v32, v25);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    // pto: %t__tile
    Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v34 = Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v12, v7);
    // pto: %t__tile
    uint64_t v35 = (uint64_t) v14;
    TASSIGN(v34, v35);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    TMATMUL(v34, v30, v32);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TPUSH<TPipe<0, Direction::DIR_C2V, 4096, 8, 8, true>, Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v16, v34);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

AICORE void mm_chain_aiv(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, __gm__ float* v4, __gm__ float* v5, int64_t v6, int32_t v7) {
  SaturationMode v8 = SaturationMode::OFF;
  RoundMode v9 = RoundMode::CAST_ROUND;
  const int64_t v10 = 40;
  const float v11 = 0.0f;
  const int64_t v12 = 64;
  const int64_t v13 = 0;
  const int64_t v14 = 1;
  const int64_t v15 = 5120;
  const int64_t v16 = 16;
  const int64_t v17 = 38912;
  const int64_t v18 = 36864;
  const int64_t v19 = 32768;
  const int32_t v20 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  auto v21 = TPipe<0, Direction::DIR_C2V, 4096, 8, 8, true>(v5, v20, v20);
  // pto: %subblock_idx, %8
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  if ((int64_t) v7 == v13) {
    // pto: %acc__tile
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v22 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v12);
    // pto: %acc__tile
    uint64_t v23 = (uint64_t) v19;
    TASSIGN(v22, v23);
    TEXPANDS(v22, v11);
    for (int64_t i24 = v13; i24 < v10; i24 += v14) {
      // pto: %t__tile_Vec
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v25 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v12);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TPOP<TPipe<0, Direction::DIR_C2V, 4096, 8, 8, true>, Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v21, v25);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %0
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v26 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v12);
      // pto: %0
      uint64_t v27 = (uint64_t) v19;
      TASSIGN(v26, v27);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      pipe_barrier(PIPE_V);
      TADD(v26, v22, v25);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TFREE<TPipe<0, Direction::DIR_C2V, 4096, 8, 8, true>, TileSplitAxis::TILE_NO_SPLIT>(v21);
    }
    // pto: %t__tile
    Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v28 = Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v12);
    // pto: %t__tile
    uint64_t v29 = (uint64_t) v18;
    TASSIGN(v28, v29);
    // pto: %9
    int64_t v30 = v6 < v13 ? v13 : v6;
    // pto: %bias__ssa_v0_pview
    pto::Shape<1, 1, 1, 16, 64> v31 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %bias__ssa_v0_pview
    pto::Stride<81920, 81920, 81920, 5120, 1> v32 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    // pto: %bias__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v33 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v3 + ((v13 + v13 * v15) + v30 * v14), v31, v32);
    TLOAD(v28, v33);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %1
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v34 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v12);
    // pto: %1
    uint64_t v35 = (uint64_t) v17;
    TASSIGN(v34, v35);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TCVT(v34, v28, v9, v8);
    // pto: %2
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v36 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v12);
    // pto: %2
    uint64_t v37 = (uint64_t) v19;
    TASSIGN(v36, v37);
    pipe_barrier(PIPE_V);
    TADD(v36, v22, v34);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %out__ssa_v0_pview
    pto::Shape<1, 1, 1, 16, 64> v38 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %out__ssa_v0_pview
    pto::Stride<81920, 81920, 81920, 5120, 1> v39 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    // pto: %out__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v40 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v4 + ((v13 + v13 * v15) + v30 * v14), v38, v39);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v40, v36);
  } else {
    // pto: %3
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v41 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v13);
    // pto: %3
    uint64_t v42 = (uint64_t) v19;
    TASSIGN(v41, v42);
    TEXPANDS(v41, v11);
    for (int64_t i43 = v13; i43 < v10; i43 += v14) {
      // pto: %12
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v44 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v12);
      v44.SetValidShape(v13, v13);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      TPOP<TPipe<0, Direction::DIR_C2V, 4096, 8, 8, true>, Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v21, v44);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %4
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v45 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v13);
      // pto: %4
      uint64_t v46 = (uint64_t) v19;
      TASSIGN(v45, v46);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      pipe_barrier(PIPE_V);
      TADD(v45, v41, v44);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      TFREE<TPipe<0, Direction::DIR_C2V, 4096, 8, 8, true>, TileSplitAxis::TILE_NO_SPLIT>(v21);
    }
    // pto: %5
    Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v47 = Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v13);
    // pto: %5
    uint64_t v48 = (uint64_t) v18;
    TASSIGN(v47, v48);
    // pto: %6
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v49 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v13);
    // pto: %6
    uint64_t v50 = (uint64_t) v17;
    TASSIGN(v49, v50);
    TCVT(v49, v47, v9, v8);
    // pto: %7
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v51 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v13);
    // pto: %7
    uint64_t v52 = (uint64_t) v19;
    TASSIGN(v51, v52);
    pipe_barrier(PIPE_V);
    TADD(v51, v41, v49);
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}
```

</details>


---

