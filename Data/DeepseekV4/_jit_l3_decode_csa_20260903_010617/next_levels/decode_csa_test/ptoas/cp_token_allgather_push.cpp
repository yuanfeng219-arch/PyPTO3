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

AICORE void cp_token_allgather_push(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ int32_t* v3, int64_t v4, int32_t v5, int32_t v6, int32_t v7, int32_t v8, __gm__ int64_t* v9, __gm__ int64_t* v10, int64_t v11, int32_t v12, int32_t v13) {
  pto::comm::NotifyOp v14 = pto::comm::NotifyOp::AtomicAdd;
  const int32_t v15 = 1;
  const int64_t v16 = 16;
  const int64_t v17 = 4;
  const int64_t v18 = 128;
  const int64_t v19 = 8;
  const int64_t v20 = 2;
  const int64_t v21 = 1;
  const int64_t v22 = 4096;
  const int64_t v23 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %worker_inline1647__ssa_v0
  int64_t v24 = (int64_t) v12;
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
  for (int64_t i25 = v23; i25 < v20; i25 += v21) {
    // pto: %1
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    for (int64_t j26 = (int64_t) ((uint64_t) v24 * (uint64_t) v19); j26 < v4; j26 += v18) {
      // pto: %tput_stage
      Tile<TileType::Vec, bfloat16_t, 8, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v27 = Tile<TileType::Vec, bfloat16_t, 8, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v22);
      // pto: %tput_stage
      uint64_t v28 = (uint64_t) v23;
      TASSIGN(v27, v28);
      // pto: %2, %3
      int64_t v29 = (int64_t) ((uint64_t) ((int64_t) v6) + (uint64_t) j26);
      // pto: %8
      int64_t v30 = (v9)[v20];
      // pto: %9, %10, %11, %12
      int64_t v31 = (v9)[(int64_t) ((uint64_t) ((int64_t) ((int32_t) v30)) + (uint64_t) v17)];
      // pto: %6, %7, %13, %14
      int64_t v32 = (v9)[(int64_t) ((uint64_t) ((int64_t) ((uint64_t) ((int64_t) v5) + (uint64_t) i25)) + (uint64_t) v17)];
      // pto: %gather_window__ssa_v0_peer_pview
      pto::Shape<1, 1, 1, 8, 4096> v33 = pto::Shape<1, 1, 1, 8, 4096>();
      // pto: %gather_window__ssa_v0_peer_pview
      pto::Stride<32768, 32768, 32768, 4096, 1> v34 = pto::Stride<32768, 32768, 32768, 4096, 1>();
      // pto: %15, %16, %18, %4, %gather_window__ssa_v0_peer_pview
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 4096>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND> v35 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 4096>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>((v1 + (int64_t) ((uint64_t) v32 - (uint64_t) v31) / v20) + (v23 + (v29 < v23 ? v23 : v29) * v22), v33, v34);
      // pto: %x_normed_t_inline1243__phi_v4_local_pview
      pto::Shape<1, 1, 1, 8, 4096> v36 = pto::Shape<1, 1, 1, 8, 4096>();
      // pto: %x_normed_t_inline1243__phi_v4_local_pview
      pto::Stride<32768, 32768, 32768, 4096, 1> v37 = pto::Stride<32768, 32768, 32768, 4096, 1>();
      // pto: %5, %x_normed_t_inline1243__phi_v4_local_pview
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 4096>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND> v38 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 4096>, pto::Stride<32768, 32768, 32768, 4096, 1>, pto::Layout::ND>(v2 + (v23 + (j26 < v23 ? v23 : j26) * v22), v36, v37);
      pipe_barrier(PIPE_ALL);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      pipe_barrier(PIPE_MTE2);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      pto::comm::TPUT(v35, v38, v27);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      pipe_barrier(PIPE_ALL);
      __gm__ bfloat16_t* v39 = PTOAS__GLOBAL_TENSOR_DATA(v35);
      PTOAS__DCCI_SINGLE_CACHE_LINE(v39);
      pipe_barrier(PIPE_ALL);
      dsb(DSB_DDR);
    }
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    // pto: %20
    // pto: %local_t_inline1640__ssa_v0_idx
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    for (int64_t j40 = (int64_t) ((uint64_t) v4 + (uint64_t) v24); j40 < ((int64_t) v7); j40 += v16) {
      // pto: %0
      Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v41 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v21, v22);
      // pto: %0
      uint64_t v42 = (uint64_t) v23;
      TASSIGN(v41, v42);
      // pto: %21, %22
      int64_t v43 = (int64_t) ((uint64_t) ((int64_t) v6) + (uint64_t) j40);
      // pto: %27
      int64_t v44 = (v9)[v20];
      // pto: %28, %29, %30, %31
      int64_t v45 = (v9)[(int64_t) ((uint64_t) ((int64_t) ((int32_t) v44)) + (uint64_t) v17)];
      // pto: %25, %26, %32, %33
      int64_t v46 = (v9)[(int64_t) ((uint64_t) ((int64_t) ((uint64_t) ((int64_t) v5) + (uint64_t) i25)) + (uint64_t) v17)];
      // pto: %39
      pto::Shape<1, 1, 1, 1, 4096> v47 = pto::Shape<1, 1, 1, 1, 4096>();
      // pto: %39
      pto::Stride<4096, 4096, 4096, 4096, 1> v48 = pto::Stride<4096, 4096, 4096, 4096, 1>();
      // pto: %34, %35, %37, %23, %39
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v49 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>((v1 + (int64_t) ((uint64_t) v46 - (uint64_t) v45) / v20) + (v23 + (v43 < v23 ? v23 : v43) * v22), v47, v48);
      // pto: %40
      pto::Shape<1, 1, 1, 1, 4096> v50 = pto::Shape<1, 1, 1, 1, 4096>();
      // pto: %40
      pto::Stride<4096, 4096, 4096, 4096, 1> v51 = pto::Stride<4096, 4096, 4096, 4096, 1>();
      // pto: %24, %40
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v52 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v2 + (v23 + (j40 < v23 ? v23 : j40) * v22), v50, v51);
      pipe_barrier(PIPE_ALL);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      pipe_barrier(PIPE_MTE2);
      pipe_barrier(PIPE_MTE3);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
      pto::comm::TPUT(v49, v52, v41);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      pipe_barrier(PIPE_ALL);
      __gm__ bfloat16_t* v53 = PTOAS__GLOBAL_TENSOR_DATA(v49);
      PTOAS__DCCI_SINGLE_CACHE_LINE(v53);
      pipe_barrier(PIPE_ALL);
      dsb(DSB_DDR);
    }
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  }
  for (int64_t i54 = v23; i54 < v20; i54 += v21) {
    // pto: %41
    int64_t v55 = (int64_t) v8;
    // pto: %42
    if (i54 != v55) {
      // pto: %45
      int64_t v56 = (v10)[v20];
      // pto: %46, %47, %48, %49
      int64_t v57 = (v10)[(int64_t) ((uint64_t) ((int64_t) ((int32_t) v56)) + (uint64_t) v17)];
      // pto: %43, %44, %50, %51
      int64_t v58 = (v10)[(int64_t) ((uint64_t) ((int64_t) ((uint64_t) ((int64_t) v5) + (uint64_t) i54)) + (uint64_t) v17)];
      // pto: %gather_signal__ssa_v0_peer_pview
      pto::Shape<1, 1, 1, 1, 1> v59 = pto::Shape<1, 1, 1, 1, 1>();
      // pto: %gather_signal__ssa_v0_peer_pview
      pto::Stride<1, 1, 1, 1, 2> v60 = pto::Stride<1, 1, 1, 1, 2>();
      // pto: %52, %53, %55, %57, %gather_signal__ssa_v0_peer_pview
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN> v61 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 2>, pto::Layout::DN>((v3 + (int64_t) ((uint64_t) v58 - (uint64_t) v57) / v17) + (v23 + (v55 < v23 ? v23 : v55)), v59, v60);
      pto::comm::TNOTIFY(v61, v15, v14);
    }
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}