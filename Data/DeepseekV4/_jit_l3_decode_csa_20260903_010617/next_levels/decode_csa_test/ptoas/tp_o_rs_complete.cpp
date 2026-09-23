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

AICORE void tp_o_rs_complete(__gm__ bfloat16_t* v1, __gm__ int32_t* v2, int32_t v3, int32_t v4, __gm__ int64_t* v5, int64_t v6) {
  pto::comm::WaitCmp v7 = pto::comm::WaitCmp::GE;
  pto::comm::NotifyOp v8 = pto::comm::NotifyOp::AtomicAdd;
  const int32_t v9 = 1;
  const int64_t v10 = 4;
  const int64_t v11 = 0;
  const int64_t v12 = 2;
  const int64_t v13 = 1;
  const int32_t v14 = 25;
  const int32_t v15 = -25;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %completion_anchor_inline2442__tile
  bfloat16_t v16 = (v1)[v11];
  for (int64_t i17 = v11; i17 < v12; i17 += v13) {
    // pto: %0
    int64_t v18 = (int64_t) v3;
    // pto: %1
    if (i17 != v18) {
      // pto: %4
      int64_t v19 = (v5)[v12];
      // pto: %5, %6, %7, %8
      int64_t v20 = (v5)[(int64_t) ((uint64_t) ((int64_t) ((int32_t) v19)) + (uint64_t) v10)];
      // pto: %2, %3, %9, %10
      int64_t v21 = (v5)[(int64_t) ((uint64_t) ((int64_t) ((uint64_t) ((int64_t) v4) + (uint64_t) i17)) + (uint64_t) v10)];
      // pto: %o_signal__ssa_v0_peer_pview
      pto::Shape<1, 1, 1, 1, 1> v22 = pto::Shape<1, 1, 1, 1, 1>();
      // pto: %o_signal__ssa_v0_peer_pview
      pto::Stride<1, 1, 1, 1, 2> v23 = pto::Stride<1, 1, 1, 1, 2>();
      // pto: %11, %12, %14, %16, %o_signal__ssa_v0_peer_pview
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN> v24 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN>((v2 + (int64_t) ((uint64_t) v21 - (uint64_t) v20) / v10) + (v11 + (v18 < v11 ? v11 : v18)), v22, v23);
      pto::comm::TNOTIFY(v24, v9, v8);
    }
  }
  for (int64_t i25 = v11; i25 < v12; i25 += v13) {
    // pto: %18, %19
    if (i25 != (int64_t) v3) {
      // pto: %o_signal__ssa_v0_local_pview
      pto::Shape<1, 1, 1, 1, 1> v26 = pto::Shape<1, 1, 1, 1, 1>();
      // pto: %o_signal__ssa_v0_local_pview
      pto::Stride<1, 1, 1, 1, 2> v27 = pto::Stride<1, 1, 1, 1, 2>();
      // pto: %20, %o_signal__ssa_v0_local_pview
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN> v28 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN>(v2 + (v11 + (i25 < v11 ? v11 : i25)), v26, v27);
      pto::comm::TWAIT(v28, v14, v7);
    }
  }
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  for (int64_t i29 = v11; i29 < v12; i29 += v13) {
    // pto: %23, %24
    if (i29 != (int64_t) v3) {
      // pto: %25
      int64_t v30 = (v5)[v12];
      // pto: %26, %27, %28, %29
      int64_t v31 = (v5)[(int64_t) ((uint64_t) ((int64_t) ((int32_t) v30)) + (uint64_t) v10)];
      // pto: %22, %self_rank_inline2440__ssa_v0_idx, %30, %31
      int64_t v32 = (v5)[(int64_t) ((uint64_t) ((int64_t) ((int32_t) ((uint32_t) v4 + (uint32_t) v3))) + (uint64_t) v10)];
      // pto: %38
      pto::Shape<1, 1, 1, 1, 1> v33 = pto::Shape<1, 1, 1, 1, 1>();
      // pto: %38
      pto::Stride<1, 1, 1, 1, 2> v34 = pto::Stride<1, 1, 1, 1, 2>();
      // pto: %32, %33, %35, %37, %38
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN> v35 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN>((v2 + (int64_t) ((uint64_t) v32 - (uint64_t) v31) / v10) + (v11 + (i29 < v11 ? v11 : i29)), v33, v34);
      pto::comm::TNOTIFY(v35, v15, v8);
    }
  }
  (v1)[v11] = v16;
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