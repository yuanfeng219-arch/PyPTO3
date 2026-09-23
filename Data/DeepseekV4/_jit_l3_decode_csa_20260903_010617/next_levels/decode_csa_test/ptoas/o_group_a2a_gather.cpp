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

AICORE void o_group_a2a_gather(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ int64_t* v3, int32_t v4, int32_t v5) {
  const int64_t v6 = 48;
  const int64_t v7 = 512;
  const int64_t v8 = 4;
  const int64_t v9 = 1;
  const int64_t v10 = 4096;
  const int64_t v11 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %worker_inline2433__ssa_v0
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t i12 = v11; i12 < v8; i12 += v9) {
    for (int64_t j13 = (int64_t) v4; j13 < v7; j13 += v6) {
      // pto: %0, %1
      int64_t v14 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) i12 * (uint64_t) v7)) + (uint64_t) j13);
      // pto: %t__tile
      Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v15 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v9, v10);
      // pto: %t__tile
      uint64_t v16 = (uint64_t) v11;
      TASSIGN(v15, v16);
      // pto: %2
      int64_t v17 = v14 < v11 ? v11 : v14;
      // pto: %attention_window__ssa_v0_pview
      pto::Shape<1, 1, 1, 1, 4096> v18 = pto::Shape<1, 1, 1, 1, 4096>();
      // pto: %attention_window__ssa_v0_pview
      pto::Stride<4096, 4096, 4096, 4096, 1> v19 = pto::Stride<4096, 4096, 4096, 4096, 1>();
      // pto: %attention_window__ssa_v0_pview
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v20 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v2 + (v11 + v17 * v10), v18, v19);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      TLOAD(v15, v20);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      // pto: %attention_local_flat_inline1292__iter_v3_pview
      pto::Shape<1, 1, 1, 1, 4096> v21 = pto::Shape<1, 1, 1, 1, 4096>();
      // pto: %attention_local_flat_inline1292__iter_v3_pview
      pto::Stride<4096, 4096, 4096, 4096, 1> v22 = pto::Stride<4096, 4096, 4096, 4096, 1>();
      // pto: %attention_local_flat_inline1292__iter_v3_pview
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v23 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v1 + (v11 + v17 * v10), v21, v22);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v23, v15);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}