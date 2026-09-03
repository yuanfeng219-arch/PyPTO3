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

AICORE void tp_o_b_dequant(__gm__ bfloat16_t* v1, __gm__ int32_t* v2, __gm__ float* v3, __gm__ float* v4, int64_t v5, int32_t v6, int32_t v7) {
  RoundMode v8 = RoundMode::CAST_RINT;
  SaturationMode v9 = SaturationMode::OFF;
  RoundMode v10 = RoundMode::CAST_NONE;
  const int64_t v11 = 2;
  const float v12 = 0.0f;
  const int64_t v13 = 16;
  const int64_t v14 = 8;
  const int64_t v15 = 12;
  const int64_t v16 = 128;
  const int64_t v17 = 4;
  const int64_t v18 = 16384;
  const int64_t v19 = 256;
  const int64_t v20 = 1;
  const int64_t v21 = 4096;
  const int64_t v22 = 512;
  const int64_t v23 = 98368;
  const int64_t v24 = 65600;
  const int64_t v25 = 65536;
  const int64_t v26 = 32768;
  const int64_t v27 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %publish_all_inline2525__iter_v1_view
  int64_t v28 = v22 * v21;
  // pto: %publish_all_inline2525__iter_v1_view
  int64_t v29 = v20 * v28;
  // pto: %publish_all_inline2525__iter_v1_view
  pto::Shape<1, 1, 1, -1, -1> v30 = pto::Shape<1, 1, 1, -1, -1>(v20, v20, v20, v22, v21);
  // pto: %publish_all_inline2525__iter_v1_view
  pto::Stride<-1, -1, -1, -1, -1> v31 = pto::Stride<-1, -1, -1, -1, -1>(v20 * v29, v29, v28, v21, v20);
  // pto: %publish_all_inline2525__iter_v1_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v32 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v30, v31);
  // pto: %own_b_i32_inline2528__rv_v2_view
  int64_t v33 = v19 * v18;
  // pto: %own_b_i32_inline2528__rv_v2_view
  int64_t v34 = v20 * v33;
  // pto: %own_b_i32_inline2528__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v35 = pto::Shape<1, 1, 1, -1, -1>(v20, v20, v20, v19, v18);
  // pto: %own_b_i32_inline2528__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v36 = pto::Stride<-1, -1, -1, -1, -1>(v20 * v34, v34, v33, v18, v20);
  // pto: %own_b_i32_inline2528__rv_v2_view
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v37 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v35, v36);
  // pto: %own_scale_inline2494__rv_v2_view
  int64_t v38 = v17 * v19;
  // pto: %own_scale_inline2494__rv_v2_view
  int64_t v39 = v20 * v38;
  // pto: %own_scale_inline2494__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v40 = pto::Shape<1, 1, 1, -1, -1>(v20, v20, v20, v17, v19);
  // pto: %own_scale_inline2494__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v41 = pto::Stride<-1, -1, -1, -1, -1>(v20 * v39, v39, v38, v19, v20);
  // pto: %own_scale_inline2494__rv_v2_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v42 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v40, v41);
  // pto: %dq_worker_inline2469__ssa_v0
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  for (int64_t i43 = (int64_t) v6; i43 < v16; i43 += v15) {
    // pto: %10
    int64_t v44 = i43 / v14;
    // pto: %13
    int64_t v45 = (int64_t) ((uint64_t) v44 * (uint64_t) v13);
    // pto: %12, %11, %14
    int64_t v46 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) i43 - (uint64_t) ((int64_t) ((uint64_t) v44 * (uint64_t) v14)))) * (uint64_t) v22);
    // pto: %15
    int64_t v47 = (int64_t) ((uint64_t) v19 - (uint64_t) v45);
    // pto: %16
    int64_t v48 = v47 < v13 ? v47 : v13;
    // pto: %dq_acc_inline2464__tile
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v49 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v22);
    // pto: %dq_acc_inline2464__tile
    uint64_t v50 = (uint64_t) v27;
    TASSIGN(v49, v50);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TEXPANDS(v49, v12);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    for (int64_t j51 = v27; j51 < v17; j51 += v11) {
      // pto: %17, %18
      int64_t v52 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) j51 * (uint64_t) v21)) + (uint64_t) v46);
      // pto: %21
      int64_t v53 = (int64_t) ((uint64_t) v52 + (uint64_t) v21);
      // pto: %dq_i32_inline2461__tile
      Tile<TileType::Vec, int32_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v54 = Tile<TileType::Vec, int32_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v22);
      // pto: %dq_i32_inline2461__tile
      uint64_t v55 = (uint64_t) v26;
      TASSIGN(v54, v55);
      // pto: %22
      int64_t v56 = v45 < v27 ? v27 : v45;
      // pto: %own_b_i32_inline2528__rv_v2_pview
      __gm__ int32_t* v57 = PTOAS__GLOBAL_TENSOR_DATA(v37);
      // pto: %own_b_i32_inline2528__rv_v2_pview
      int64_t v58 = v48 * v18;
      // pto: %own_b_i32_inline2528__rv_v2_pview
      int64_t v59 = v20 * v58;
      // pto: %own_b_i32_inline2528__rv_v2_pview
      pto::Shape<1, 1, 1, -1, 512> v60 = pto::Shape<1, 1, 1, -1, 512>(v20, v20, v20, v48, v22);
      // pto: %own_b_i32_inline2528__rv_v2_pview
      pto::Stride<-1, -1, -1, -1, -1> v61 = pto::Stride<-1, -1, -1, -1, -1>(v20 * v59, v59, v58, v18, v20);
      // pto: %own_b_i32_inline2528__rv_v2_pview, %23
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v62 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v57 + ((v27 + v56 * v18) + (v52 < v27 ? v27 : v52) * v20), v60, v61);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      TLOAD(v54, v62);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %dq_srow_inline2518__tile
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v63 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v20, v48);
      // pto: %dq_srow_inline2518__tile
      uint64_t v64 = (uint64_t) v25;
      TASSIGN(v63, v64);
      // pto: %own_scale_inline2494__rv_v2_pview
      __gm__ float* v65 = PTOAS__GLOBAL_TENSOR_DATA(v42);
      // pto: %own_scale_inline2494__rv_v2_pview
      int64_t v66 = v20 * v19;
      // pto: %own_scale_inline2494__rv_v2_pview
      int64_t v67 = v20 * v66;
      // pto: %own_scale_inline2494__rv_v2_pview
      pto::Shape<1, 1, 1, 1, -1> v68 = pto::Shape<1, 1, 1, 1, -1>(v20, v20, v20, v20, v48);
      // pto: %own_scale_inline2494__rv_v2_pview
      pto::Stride<-1, -1, -1, -1, -1> v69 = pto::Stride<-1, -1, -1, -1, -1>(v20 * v67, v67, v66, v19, v20);
      // pto: %24, %own_scale_inline2494__rv_v2_pview
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v70 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v65 + ((v27 + (j51 < v27 ? v27 : j51) * v19) + v56 * v20), v68, v69);
      TLOAD(v63, v70);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %0
      Tile<TileType::Vec, int32_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v71 = Tile<TileType::Vec, int32_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v22);
      // pto: %0
      uint64_t v72 = (uint64_t) v24;
      TASSIGN(v71, v72);
      // pto: %28
      __gm__ int32_t* v73 = PTOAS__GLOBAL_TENSOR_DATA(v37);
      // pto: %28
      int64_t v74 = v48 * v18;
      // pto: %28
      int64_t v75 = v20 * v74;
      // pto: %28
      pto::Shape<1, 1, 1, -1, 512> v76 = pto::Shape<1, 1, 1, -1, 512>(v20, v20, v20, v48, v22);
      // pto: %28
      pto::Stride<-1, -1, -1, -1, -1> v77 = pto::Stride<-1, -1, -1, -1, -1>(v20 * v75, v75, v74, v18, v20);
      // pto: %28, %27
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v78 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v73 + ((v27 + v56 * v18) + (v53 < v27 ? v27 : v53) * v20), v76, v77);
      TLOAD(v71, v78);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %1
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v79 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v20, v48);
      // pto: %1
      uint64_t v80 = (uint64_t) v23;
      TASSIGN(v79, v80);
      // pto: %29
      int64_t v81 = (int64_t) ((uint64_t) j51 + (uint64_t) v20);
      // pto: %32
      __gm__ float* v82 = PTOAS__GLOBAL_TENSOR_DATA(v42);
      // pto: %32
      int64_t v83 = v20 * v19;
      // pto: %32
      int64_t v84 = v20 * v83;
      // pto: %32
      pto::Shape<1, 1, 1, 1, -1> v85 = pto::Shape<1, 1, 1, 1, -1>(v20, v20, v20, v20, v48);
      // pto: %32
      pto::Stride<-1, -1, -1, -1, -1> v86 = pto::Stride<-1, -1, -1, -1, -1>(v20 * v84, v84, v83, v19, v20);
      // pto: %30, %32
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v87 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v82 + ((v27 + (v81 < v27 ? v27 : v81) * v19) + v56 * v20), v85, v86);
      TLOAD(v79, v87);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      // pto: %dq_fp32_inline2459__tile
      Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v88 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v22);
      // pto: %dq_fp32_inline2459__tile
      uint64_t v89 = (uint64_t) v26;
      TASSIGN(v88, v89);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(v88, v54, v10, v9);
      // pto: %dq_scol_inline2462__tile
      Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v90 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v20);
      // pto: %dq_scol_inline2462__tile
      uint64_t v91 = (uint64_t) v25;
      TASSIGN(v90, v91);
      // pto: %t__tile
      Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v92 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v22);
      // pto: %t__tile
      uint64_t v93 = (uint64_t) v26;
      TASSIGN(v92, v93);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TROWEXPANDMUL(v92, v88, v90);
      // pto: %2
      Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v94 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v22);
      // pto: %2
      uint64_t v95 = (uint64_t) v26;
      TASSIGN(v94, v95);
      pipe_barrier(PIPE_V);
      TADD(v94, v49, v92);
      // pto: %3
      Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v22);
      // pto: %3
      uint64_t v97 = (uint64_t) v24;
      TASSIGN(v96, v97);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      TCVT(v96, v71, v10, v9);
      // pto: %4
      Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v98 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v20);
      // pto: %4
      uint64_t v99 = (uint64_t) v23;
      TASSIGN(v98, v99);
      // pto: %5
      Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v100 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v22);
      // pto: %5
      uint64_t v101 = (uint64_t) v24;
      TASSIGN(v100, v101);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      TROWEXPANDMUL(v100, v96, v98);
      // pto: %6
      Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v22);
      // pto: %6
      uint64_t v103 = (uint64_t) v27;
      TASSIGN(v102, v103);
      pipe_barrier(PIPE_V);
      TADD(v102, v94, v100);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    }
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    // pto: %7
    Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v104 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v20, v22);
    // pto: %7
    uint64_t v105 = (uint64_t) v26;
    TASSIGN(v104, v105);
    // pto: %33
    int64_t v106 = v46 < v27 ? v27 : v46;
    // pto: %wo_b_scale__ssa_v0_pview
    pto::Shape<1, 1, 1, 1, 512> v107 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %wo_b_scale__ssa_v0_pview
    pto::Stride<512, 512, 512, 512, 1> v108 = pto::Stride<512, 512, 512, 512, 1>();
    // pto: %wo_b_scale__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v109 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v4 + (v27 + v106), v107, v108);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    TLOAD(v104, v109);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    // pto: %dq_wscale_inline2458__tile
    Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v110 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v20, v22);
    // pto: %dq_wscale_inline2458__tile
    uint64_t v111 = (uint64_t) v26;
    TASSIGN(v110, v111);
    // pto: %8
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v112 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v22);
    // pto: %8
    uint64_t v113 = (uint64_t) v27;
    TASSIGN(v112, v113);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v112, v49, v110);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    // pto: %dq_bf16_inline2457__tile
    Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v114 = Tile<TileType::Vec, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v22);
    // pto: %dq_bf16_inline2457__tile
    uint64_t v115 = (uint64_t) v27;
    TASSIGN(v114, v115);
    pipe_barrier(PIPE_V);
    TCVT(v114, v112, v8, v9);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %34, %35
    int64_t v116 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) v5 * (uint64_t) v19)) + (uint64_t) v45);
    v114.SetValidShape(v48, v22);
    // pto: %publish_all_inline2525__iter_v3_pview
    __gm__ bfloat16_t* v117 = PTOAS__GLOBAL_TENSOR_DATA(v32);
    // pto: %publish_all_inline2525__iter_v3_pview
    int64_t v118 = v48 * v21;
    // pto: %publish_all_inline2525__iter_v3_pview
    int64_t v119 = v20 * v118;
    // pto: %publish_all_inline2525__iter_v3_pview
    pto::Shape<1, 1, 1, -1, 512> v120 = pto::Shape<1, 1, 1, -1, 512>(v20, v20, v20, v48, v22);
    // pto: %publish_all_inline2525__iter_v3_pview
    pto::Stride<-1, -1, -1, -1, -1> v121 = pto::Stride<-1, -1, -1, -1, -1>(v20 * v119, v119, v118, v21, v20);
    // pto: %36, %publish_all_inline2525__iter_v3_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v122 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v117 + ((v27 + (v116 < v27 ? v27 : v116) * v21) + v106 * v20), v120, v121);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v122, v114);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}