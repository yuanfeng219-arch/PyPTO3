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

AICORE void cp_token_allgather_retire(__gm__ bfloat16_t* v1, __gm__ int32_t* v2, int32_t v3, int32_t v4, __gm__ int64_t* v5, int64_t v6) {
  pto::comm::NotifyOp v7 = pto::comm::NotifyOp::AtomicAdd;
  const int64_t v8 = 4;
  const int64_t v9 = 0;
  const int64_t v10 = 2;
  const int64_t v11 = 1;
  const int32_t v12 = -32;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %completion_anchor_inline1626__tile
  bfloat16_t v13 = (v1)[v9];
  for (int64_t i14 = v9; i14 < v10; i14 += v11) {
    // pto: %2, %3
    if (i14 != (int64_t) v4) {
      // pto: %4
      int64_t v15 = (v5)[v10];
      // pto: %5, %6, %7, %8
      int64_t v16 = (v5)[(int64_t) ((uint64_t) ((int64_t) ((int32_t) v15)) + (uint64_t) v8)];
      // pto: %1, %self_rank_inline1625__ssa_v0_idx, %9, %10
      int64_t v17 = (v5)[(int64_t) ((uint64_t) ((int64_t) ((int32_t) ((uint32_t) v3 + (uint32_t) v4))) + (uint64_t) v8)];
      // pto: %gather_signal__ssa_v0_peer_pview
      pto::Shape<1, 1, 1, 1, 1> v18 = pto::Shape<1, 1, 1, 1, 1>();
      // pto: %gather_signal__ssa_v0_peer_pview
      pto::Stride<1, 1, 1, 1, 2> v19 = pto::Stride<1, 1, 1, 1, 2>();
      // pto: %11, %12, %14, %16, %gather_signal__ssa_v0_peer_pview
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN> v20 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN>((v2 + (int64_t) ((uint64_t) v17 - (uint64_t) v16) / v8) + (v9 + (i14 < v9 ? v9 : i14)), v18, v19);
      pto::comm::TNOTIFY(v20, v12, v7);
    }
  }
  (v1)[v9] = v13;
  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  #endif // __DAV_VEC__

  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}