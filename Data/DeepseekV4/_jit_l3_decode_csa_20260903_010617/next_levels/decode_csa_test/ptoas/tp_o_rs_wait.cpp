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

AICORE void tp_o_rs_wait(__gm__ int32_t* v1, int32_t v2, __gm__ int64_t* v3) {
  pto::comm::WaitCmp v4 = pto::comm::WaitCmp::GE;
  const int64_t v5 = 0;
  const int64_t v6 = 1;
  const int64_t v7 = 2;
  const int32_t v8 = 24;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  for (int64_t i9 = v5; i9 < v7; i9 += v6) {
    // pto: %1, %2
    if (i9 != (int64_t) v2) {
      // pto: %o_signal__ssa_v0_local_pview
      pto::Shape<1, 1, 1, 1, 1> v10 = pto::Shape<1, 1, 1, 1, 1>();
      // pto: %o_signal__ssa_v0_local_pview
      pto::Stride<1, 1, 1, 1, 2> v11 = pto::Stride<1, 1, 1, 1, 2>();
      // pto: %3, %o_signal__ssa_v0_local_pview
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN> v12 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN>(v1 + (v5 + (i9 < v5 ? v5 : i9)), v10, v11);
      pto::comm::TWAIT(v12, v8, v4);
    }
  }
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}