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

AICORE void csa_cache_writeback(__gm__ bfloat16_t* v1, __gm__ int64_t* v2, __gm__ bfloat16_t* v3, int64_t v4, int64_t v5, int64_t v6, int32_t v7, int32_t v8) {
  const int64_t v9 = 8;
  const int64_t v10 = 1;
  const int64_t v11 = 512;
  const int64_t v12 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t i13 = v12; i13 < v9; i13 += v10) {
    // pto: %wb_blk_inline1315__ssa_v0, %0, %1
    int64_t v14 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) ((int64_t) v7) * (uint64_t) v9)) + (uint64_t) i13);
    // pto: %write_row_i64_inline1319__tile
    int64_t v15 = (v2)[v14];
    // pto: %3
    if (v15 >= v12) {
      // pto: %t__tile
      Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v16 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v11);
      // pto: %t__tile
      uint64_t v17 = (uint64_t) v12;
      TASSIGN(v16, v17);
      // pto: %kv_full_inline1265__ssa_v0_pview
      pto::Shape<1, 1, 1, 1, 512> v18 = pto::Shape<1, 1, 1, 1, 512>();
      // pto: %kv_full_inline1265__ssa_v0_pview
      pto::Stride<512, 512, 512, 512, 1> v19 = pto::Stride<512, 512, 512, 512, 1>();
      // pto: %5, %kv_full_inline1265__ssa_v0_pview
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v20 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v3 + (v12 + (v14 < v12 ? v12 : v14) * v11), v18, v19);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      TLOAD(v16, v20);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      // pto: %kv_cache_flat_inline1312__iter_v1_pview
      pto::Shape<1, 1, 1, 1, 512> v21 = pto::Shape<1, 1, 1, 1, 512>();
      // pto: %kv_cache_flat_inline1312__iter_v1_pview
      pto::Stride<512, 512, 512, 512, 1> v22 = pto::Stride<512, 512, 512, 512, 1>();
      // pto: %6, %kv_cache_flat_inline1312__iter_v1_pview
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v23 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v1 + (v12 + (v15 < v12 ? v12 : v15) * v11), v21, v22);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v23, v16);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}