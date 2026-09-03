#include "pto/pto-inst.hpp"
using namespace pto;

template <typename Tensor>
static AICORE inline auto PTOAS__GLOBAL_TENSOR_DATA(Tensor &tensor)
    -> decltype(tensor.data()) {
  return tensor.data();
}


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

AICORE void tp_o_rs_reduce(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ int64_t* v3, int64_t v4, int32_t v5, int32_t v6) {
  RoundMode v7 = RoundMode::CAST_RINT;
  SaturationMode v8 = SaturationMode::OFF;
  RoundMode v9 = RoundMode::CAST_NONE;
  const int64_t v10 = 48;
  const int64_t v11 = 256;
  const int64_t v12 = 1;
  const int64_t v13 = 4096;
  const int64_t v14 = 16384;
  const int64_t v15 = 0;
  const int64_t v16 = 24576;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %worker_inline2455__ssa_v0
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  for (int64_t i17 = (int64_t) v5; i17 < v11; i17 += v10) {
    // pto: %own_partial_inline2446__ssa_v0
    Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v18 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v13);
    // pto: %own_partial_inline2446__ssa_v0
    uint64_t v19 = (uint64_t) v16;
    TASSIGN(v18, v19);
    // pto: %2
    int64_t v20 = i17 < v15 ? v15 : i17;
    // pto: %o_window__ssa_v0_pview
    pto::Shape<1, 1, 1, 1, 4096> v21 = pto::Shape<1, 1, 1, 1, 4096>();
    // pto: %o_window__ssa_v0_pview
    pto::Stride<4096, 4096, 4096, 4096, 1> v22 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %o_window__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v23 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v1 + (v15 + v20 * v13), v21, v22);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v18, v23);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %reduce_acc_inline2445__ssa_v0
    Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v24 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v13);
    // pto: %reduce_acc_inline2445__ssa_v0
    uint64_t v25 = (uint64_t) v15;
    TASSIGN(v24, v25);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TCVT(v24, v18, v9, v8);
    // pto: %4
    int64_t v26 = (int64_t) ((uint64_t) i17 + (uint64_t) v11);
    // pto: %source_partial_inline2443__ssa_v0
    Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v27 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v13);
    // pto: %source_partial_inline2443__ssa_v0
    uint64_t v28 = (uint64_t) v14;
    TASSIGN(v27, v28);
    // pto: %7
    pto::Shape<1, 1, 1, 1, 4096> v29 = pto::Shape<1, 1, 1, 1, 4096>();
    // pto: %7
    pto::Stride<4096, 4096, 4096, 4096, 1> v30 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %5, %7
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v31 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v1 + (v15 + (v26 < v15 ? v15 : v26) * v13), v29, v30);
    TLOAD(v27, v31);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %source_fp32_inline2506__ssa_v0
    Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v32 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v13);
    // pto: %source_fp32_inline2506__ssa_v0
    uint64_t v33 = (uint64_t) v16;
    TASSIGN(v32, v33);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    pipe_barrier(PIPE_V);
    TCVT(v32, v27, v9, v8);
    // pto: %reduce_acc_inline2445__ssa_v3
    Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v34 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v13);
    // pto: %reduce_acc_inline2445__ssa_v3
    uint64_t v35 = (uint64_t) v15;
    TASSIGN(v34, v35);
    pipe_barrier(PIPE_V);
    TADD(v34, v24, v32);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    // pto: %reduced_inline2553__ssa_v0
    Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v36 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v13);
    // pto: %reduced_inline2553__ssa_v0
    uint64_t v37 = (uint64_t) v15;
    TASSIGN(v36, v37);
    pipe_barrier(PIPE_V);
    TCVT(v36, v34, v7, v8);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %attn_out_inline1284__ssa_v0_pview
    pto::Shape<1, 1, 1, 1, 4096> v38 = pto::Shape<1, 1, 1, 1, 4096>();
    // pto: %attn_out_inline1284__ssa_v0_pview
    pto::Stride<4096, 4096, 4096, 4096, 1> v39 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %attn_out_inline1284__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v40 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v2 + (v15 + v20 * v13), v38, v39);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v40, v36);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}