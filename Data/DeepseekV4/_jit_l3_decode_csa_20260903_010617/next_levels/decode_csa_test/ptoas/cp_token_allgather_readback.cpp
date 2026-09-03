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

AICORE void cp_token_allgather_readback(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ int32_t* v3, int64_t v4, int64_t v5, int32_t v6, int32_t v7, __gm__ int64_t* v8, __gm__ int64_t* v9, int64_t v10, int32_t v11, int32_t v12) {
  pto::comm::NotifyOp v13 = pto::comm::NotifyOp::AtomicAdd;
  const int32_t v14 = 1;
  const int64_t v15 = 4;
  const int64_t v16 = 256;
  const int64_t v17 = 16;
  const int64_t v18 = 2;
  const int64_t v19 = 1;
  const int64_t v20 = 4096;
  const int64_t v21 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %worker_v1_inline1638__ssa_v0
  int64_t v22 = (int64_t) v11;
  // pto: %0
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  for (int64_t i23 = (int64_t) ((uint64_t) v22 * (uint64_t) v17); i23 < v4; i23 += v16) {
    // pto: %window_tile_inline1650__tile
    Tile<TileType::Vec, bfloat16_t, 16, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v24 = Tile<TileType::Vec, bfloat16_t, 16, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v20);
    // pto: %window_tile_inline1650__tile
    uint64_t v25 = (uint64_t) v21;
    TASSIGN(v24, v25);
    // pto: %1
    int64_t v26 = i23 < v21 ? v21 : i23;
    // pto: %gather_window__ssa_v0_pview
    pto::Shape<1, 1, 1, 16, 4096> v27 = pto::Shape<1, 1, 1, 16, 4096>();
    // pto: %gather_window__ssa_v0_pview
    pto::Stride<65536, 65536, 65536, 4096, 1> v28 = pto::Stride<65536, 65536, 65536, 4096, 1>();
    // pto: %gather_window__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 4096>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND> v29 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 4096>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(v2 + (v21 + v26 * v20), v27, v28);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v24, v29);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    // pto: %x_normed_full_inline1240__iter_v1_pview
    pto::Shape<1, 1, 1, 16, 4096> v30 = pto::Shape<1, 1, 1, 16, 4096>();
    // pto: %x_normed_full_inline1240__iter_v1_pview
    pto::Stride<65536, 65536, 65536, 4096, 1> v31 = pto::Stride<65536, 65536, 65536, 4096, 1>();
    // pto: %x_normed_full_inline1240__iter_v1_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 4096>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND> v32 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 4096>, pto::Stride<65536, 65536, 65536, 4096, 1>, pto::Layout::ND>(v1 + (v21 + v26 * v20), v30, v31);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v32, v24);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  // pto: %3
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  for (int64_t i33 = (int64_t) ((uint64_t) v4 + (uint64_t) v22); i33 < v5; i33 += v17) {
    // pto: %window_row_inline1631__tile
    Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v34 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v20);
    // pto: %window_row_inline1631__tile
    uint64_t v35 = (uint64_t) v21;
    TASSIGN(v34, v35);
    // pto: %4
    int64_t v36 = i33 < v21 ? v21 : i33;
    // pto: %5
    pto::Shape<1, 1, 1, 1, 4096> v37 = pto::Shape<1, 1, 1, 1, 4096>();
    // pto: %5
    pto::Stride<4096, 4096, 4096, 4096, 1> v38 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %5
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v39 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v2 + (v21 + v36 * v20), v37, v38);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    TLOAD(v34, v39);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    // pto: %x_normed_full_inline1240__iter_v4_pview
    pto::Shape<1, 1, 1, 1, 4096> v40 = pto::Shape<1, 1, 1, 1, 4096>();
    // pto: %x_normed_full_inline1240__iter_v4_pview
    pto::Stride<4096, 4096, 4096, 4096, 1> v41 = pto::Stride<4096, 4096, 4096, 4096, 1>();
    // pto: %x_normed_full_inline1240__iter_v4_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v42 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v1 + (v21 + v36 * v20), v40, v41);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    TSTORE(v42, v34);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  }
  for (int64_t i43 = v21; i43 < v18; i43 += v19) {
    // pto: %8
    int64_t v44 = (int64_t) v6;
    // pto: %9
    if (i43 != v44) {
      // pto: %12
      int64_t v45 = (v9)[v18];
      // pto: %13, %14, %15, %16
      int64_t v46 = (v9)[(int64_t) ((uint64_t) ((int64_t) ((int32_t) v45)) + (uint64_t) v15)];
      // pto: %10, %11, %17, %18
      int64_t v47 = (v9)[(int64_t) ((uint64_t) ((int64_t) ((uint64_t) ((int64_t) v7) + (uint64_t) i43)) + (uint64_t) v15)];
      // pto: %gather_signal__ssa_v0_peer_pview
      pto::Shape<1, 1, 1, 1, 1> v48 = pto::Shape<1, 1, 1, 1, 1>();
      // pto: %gather_signal__ssa_v0_peer_pview
      pto::Stride<1, 1, 1, 1, 2> v49 = pto::Stride<1, 1, 1, 1, 2>();
      // pto: %19, %20, %22, %24, %gather_signal__ssa_v0_peer_pview
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN> v50 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN>((v3 + (int64_t) ((uint64_t) v47 - (uint64_t) v46) / v15) + (v21 + (v44 < v21 ? v21 : v44)), v48, v49);
      pto::comm::TNOTIFY(v50, v14, v13);
    }
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}