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

AICORE void tp_o_b_publish(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ int32_t* v3, int32_t v4, int32_t v5, __gm__ int64_t* v6, __gm__ int64_t* v7, int32_t v8, int32_t v9) {
  pto::comm::NotifyOp v10 = pto::comm::NotifyOp::AtomicAdd;
  const int32_t v11 = 1;
  const int64_t v12 = 4;
  const int64_t v13 = 256;
  const int64_t v14 = 8;
  const int64_t v15 = 32;
  const int64_t v16 = 24;
  const int64_t v17 = 64;
  const int64_t v18 = 2;
  const int64_t v19 = 1;
  const int64_t v20 = 4096;
  const int64_t v21 = 512;
  const int64_t v22 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %publish_all_inline2525__rv_v2_view
  int64_t v23 = v21 * v20;
  // pto: %publish_all_inline2525__rv_v2_view
  int64_t v24 = v19 * v23;
  // pto: %publish_all_inline2525__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v25 = pto::Shape<1, 1, 1, -1, -1>(v19, v19, v19, v21, v20);
  // pto: %publish_all_inline2525__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v26 = pto::Stride<-1, -1, -1, -1, -1>(v19 * v24, v24, v23, v20, v19);
  // pto: %publish_all_inline2525__rv_v2_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v27 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v25, v26);
  // pto: %pub_worker_inline2453__ssa_v0
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
  for (int64_t i28 = (int64_t) v8; i28 < v17; i28 += v16) {
    // pto: %0
    int64_t v29 = i28 / v15;
    // pto: %2, %1, %3
    int64_t v30 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) i28 - (uint64_t) ((int64_t) ((uint64_t) v29 * (uint64_t) v15)))) * (uint64_t) v14);
    // pto: %4
    int64_t v31 = (int64_t) ((uint64_t) v13 - (uint64_t) v30);
    // pto: %5
    int64_t v32 = v31 < v14 ? v31 : v14;
    // pto: %6, %7
    int64_t v33 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) v29 * (uint64_t) v13)) + (uint64_t) v30);
    // pto: %8, %9, %10
    int64_t v34 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) ((int64_t) v4) * (uint64_t) v13)) + (uint64_t) v30);
    // pto: %tput_stage
    Tile<TileType::Vec, bfloat16_t, 8, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v35 = Tile<TileType::Vec, bfloat16_t, 8, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v20);
    // pto: %tput_stage
    uint64_t v36 = (uint64_t) v22;
    TASSIGN(v35, v36);
    // pto: %15
    int64_t v37 = (v6)[v18];
    // pto: %16, %17, %18, %19
    int64_t v38 = (v6)[(int64_t) ((uint64_t) ((int64_t) ((int32_t) v37)) + (uint64_t) v12)];
    // pto: %13, %14, %20, %21
    int64_t v39 = (v6)[(int64_t) ((uint64_t) ((int64_t) ((uint64_t) ((int64_t) v5) + (uint64_t) v29)) + (uint64_t) v12)];
    // pto: %26
    int64_t v40 = v21 * v20;
    // pto: %26
    int64_t v41 = v19 * v40;
    // pto: %26
    pto::Shape<1, 1, 1, -1, -1> v42 = pto::Shape<1, 1, 1, -1, -1>(v19, v19, v19, v21, v20);
    // pto: %26
    pto::Stride<-1, -1, -1, -1, -1> v43 = pto::Stride<-1, -1, -1, -1, -1>(v19 * v41, v41, v40, v20, v19);
    // pto: %22, %23, %25, %26
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v44 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1 + (int64_t) ((uint64_t) v39 - (uint64_t) v38) / v18, v42, v43);
    // pto: %o_window__ssa_v0_peer_pview
    __gm__ bfloat16_t* v45 = PTOAS__GLOBAL_TENSOR_DATA(v44);
    // pto: %o_window__ssa_v0_peer_pview
    int64_t v46 = v32 * v20;
    // pto: %o_window__ssa_v0_peer_pview
    int64_t v47 = v19 * v46;
    // pto: %o_window__ssa_v0_peer_pview
    pto::Shape<1, 1, 1, -1, 4096> v48 = pto::Shape<1, 1, 1, -1, 4096>(v19, v19, v19, v32, v20);
    // pto: %o_window__ssa_v0_peer_pview
    pto::Stride<-1, -1, -1, -1, -1> v49 = pto::Stride<-1, -1, -1, -1, -1>(v19 * v47, v47, v46, v20, v19);
    // pto: %11, %o_window__ssa_v0_peer_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 4096>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v50 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 4096>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v45 + ((v22 + (v34 < v22 ? v22 : v34) * v20) + v22 * v19), v48, v49);
    // pto: %publish_all_inline2525__rv_v2_local_pview
    __gm__ bfloat16_t* v51 = PTOAS__GLOBAL_TENSOR_DATA(v27);
    // pto: %publish_all_inline2525__rv_v2_local_pview
    int64_t v52 = v32 * v20;
    // pto: %publish_all_inline2525__rv_v2_local_pview
    int64_t v53 = v19 * v52;
    // pto: %publish_all_inline2525__rv_v2_local_pview
    pto::Shape<1, 1, 1, -1, 4096> v54 = pto::Shape<1, 1, 1, -1, 4096>(v19, v19, v19, v32, v20);
    // pto: %publish_all_inline2525__rv_v2_local_pview
    pto::Stride<-1, -1, -1, -1, -1> v55 = pto::Stride<-1, -1, -1, -1, -1>(v19 * v53, v53, v52, v20, v19);
    // pto: %12, %publish_all_inline2525__rv_v2_local_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 4096>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v56 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 4096>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v51 + ((v22 + (v33 < v22 ? v22 : v33) * v20) + v22 * v19), v54, v55);
    pipe_barrier(PIPE_ALL);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    pipe_barrier(PIPE_MTE2);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    pto::comm::TPUT(v50, v56, v35);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
    __gm__ bfloat16_t* v57 = PTOAS__GLOBAL_TENSOR_DATA(v50);
    PTOAS__DCCI_SINGLE_CACHE_LINE(v57);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
  }
  for (int64_t i58 = v22; i58 < v18; i58 += v19) {
    // pto: %27
    int64_t v59 = (int64_t) v4;
    // pto: %28
    if (i58 != v59) {
      // pto: %31
      int64_t v60 = (v7)[v18];
      // pto: %32, %33, %34, %35
      int64_t v61 = (v7)[(int64_t) ((uint64_t) ((int64_t) ((int32_t) v60)) + (uint64_t) v12)];
      // pto: %29, %30, %36, %37
      int64_t v62 = (v7)[(int64_t) ((uint64_t) ((int64_t) ((uint64_t) ((int64_t) v5) + (uint64_t) i58)) + (uint64_t) v12)];
      // pto: %o_signal__ssa_v0_peer_pview
      pto::Shape<1, 1, 1, 1, 1> v63 = pto::Shape<1, 1, 1, 1, 1>();
      // pto: %o_signal__ssa_v0_peer_pview
      pto::Stride<1, 1, 1, 1, 2> v64 = pto::Stride<1, 1, 1, 1, 2>();
      // pto: %38, %39, %41, %43, %o_signal__ssa_v0_peer_pview
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN> v65 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN>((v3 + (int64_t) ((uint64_t) v62 - (uint64_t) v61) / v12) + (v22 + (v59 < v22 ? v22 : v59)), v63, v64);
      pto::comm::TNOTIFY(v65, v11, v10);
    }
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}