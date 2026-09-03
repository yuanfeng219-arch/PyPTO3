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

AICORE void qproj_dequant_rms_nope_rope(__gm__ bfloat16_t* v1, __gm__ float* v2, __gm__ float* v3, __gm__ float* v4, __gm__ int32_t* v5, __gm__ int32_t* v6, __gm__ float* v7, int64_t v8, int64_t v9, int64_t v10, int64_t v11, int64_t v12, int64_t v13, int32_t v14, int32_t v15) {
  RoundMode v16 = RoundMode::CAST_TRUNC;
  RoundMode v17 = RoundMode::CAST_ROUND;
  RoundMode v18 = RoundMode::CAST_RINT;
  SaturationMode v19 = SaturationMode::OFF;
  RoundMode v20 = RoundMode::CAST_NONE;
  const float v21 = 64.0f;
  const float v22 = 2.0f;
  const float v23 = 0.5f;
  const int32_t v24 = 0;
  const float v25 = 1.0f;
  const int64_t v26 = 448;
  const float v27 = 9.99999997E-7f;
  const float v28 = 0.001953125f;
  const int64_t v29 = 512;
  const int64_t v30 = 2;
  const int64_t v31 = 8;
  const int64_t v32 = 4;
  const int64_t v33 = 64;
  const int64_t v34 = 1;
  const int64_t v35 = 32768;
  const int64_t v36 = 101120;
  const int64_t v37 = 20224;
  const int64_t v38 = 100864;
  const int64_t v39 = 84480;
  const int64_t v40 = 67840;
  const int64_t v41 = 3840;
  const int64_t v42 = 67584;
  const int64_t v43 = 51200;
  const int64_t v44 = 68096;
  const int64_t v45 = 18432;
  const int64_t v46 = 34816;
  const int64_t v47 = 2048;
  const int64_t v48 = 0;
  const int64_t v49 = 103456;
  const int64_t v50 = 101408;
  const int64_t v51 = 101376;
  const int64_t v52 = 256;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %q_flat_inline1856__ssa_v0_view
  int64_t v53 = v11 * v35;
  // pto: %q_flat_inline1856__ssa_v0_view
  int64_t v54 = v34 * v53;
  // pto: %q_flat_inline1856__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v55 = pto::Shape<1, 1, 1, -1, -1>(v34, v34, v34, v11, v35);
  // pto: %q_flat_inline1856__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v56 = pto::Stride<-1, -1, -1, -1, -1>(v34 * v54, v54, v53, v35, v34);
  // pto: %q_flat_inline1856__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v57 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v55, v56);
  // pto: %qr_scale_pad_store_inline1814__ssa_v1_view
  int64_t v58 = v12 * v34;
  // pto: %qr_scale_pad_store_inline1814__ssa_v1_view
  int64_t v59 = v34 * v58;
  // pto: %qr_scale_pad_store_inline1814__ssa_v1_view
  pto::Shape<1, 1, 1, -1, -1> v60 = pto::Shape<1, 1, 1, -1, -1>(v34, v34, v34, v12, v34);
  // pto: %qr_scale_pad_store_inline1814__ssa_v1_view
  pto::Stride<-1, -1, -1, -1, -1> v61 = pto::Stride<-1, -1, -1, -1, -1>(v34 * v59, v59, v58, v34, v12);
  // pto: %qr_scale_pad_store_inline1814__ssa_v1_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v62 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v60, v61);
  // pto: %q_cos_il_inline1311__ssa_v0_view
  int64_t v63 = v13 * v33;
  // pto: %q_cos_il_inline1311__ssa_v0_view
  int64_t v64 = v34 * v63;
  // pto: %q_cos_il_inline1311__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v65 = pto::Shape<1, 1, 1, -1, -1>(v34, v34, v34, v13, v33);
  // pto: %q_cos_il_inline1311__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v66 = pto::Stride<-1, -1, -1, -1, -1>(v34 * v64, v64, v63, v33, v34);
  // pto: %q_cos_il_inline1311__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v67 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v65, v66);
  // pto: %q_sin_signed_inline1295__ssa_v0_view
  int64_t v68 = v13 * v33;
  // pto: %q_sin_signed_inline1295__ssa_v0_view
  int64_t v69 = v34 * v68;
  // pto: %q_sin_signed_inline1295__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v70 = pto::Shape<1, 1, 1, -1, -1>(v34, v34, v34, v13, v33);
  // pto: %q_sin_signed_inline1295__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v71 = pto::Stride<-1, -1, -1, -1, -1>(v34 * v69, v69, v68, v33, v34);
  // pto: %q_sin_signed_inline1295__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v72 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v70, v71);
  // pto: %hg_idx_inline1857__ssa_v0, %32
  int64_t v73 = (int64_t) ((uint64_t) ((int64_t) v14) * (uint64_t) v32);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  for (int64_t i74 = v48; i74 < v9; i74 += v31) {
    // pto: %33
    int64_t v75 = (int64_t) ((uint64_t) v10 + (uint64_t) i74);
    // pto: %34, %35
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    if ((int64_t) ((uint64_t) i74 + (uint64_t) v31) <= v9) {
      // pto: %qr_scale_dq_t_inline1860__tile
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v76 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v34);
      // pto: %qr_scale_dq_t_inline1860__tile
      uint64_t v77 = (uint64_t) v51;
      TASSIGN(v76, v77);
      // pto: %36
      int64_t v78 = i74 < v48 ? v48 : i74;
      // pto: %qr_scale_pad_store_inline1814__ssa_v1_pview
      __gm__ float* v79 = PTOAS__GLOBAL_TENSOR_DATA(v62);
      // pto: %qr_scale_pad_store_inline1814__ssa_v1_pview
      int64_t v80 = v31 * v34;
      // pto: %qr_scale_pad_store_inline1814__ssa_v1_pview
      int64_t v81 = v34 * v80;
      // pto: %qr_scale_pad_store_inline1814__ssa_v1_pview
      pto::Shape<1, 1, 1, 8, 1> v82 = pto::Shape<1, 1, 1, 8, 1>(v34, v34, v34, v31, v34);
      // pto: %qr_scale_pad_store_inline1814__ssa_v1_pview
      pto::Stride<-1, -1, -1, -1, -1> v83 = pto::Stride<-1, -1, -1, -1, -1>(v34 * v81, v81, v80, v34, v12);
      // pto: %qr_scale_pad_store_inline1814__ssa_v1_pview
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v84 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v79 + ((v48 + v78 * v34) + v48 * v12), v82, v83);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v76, v84);
      // pto: %q_cos_il_inline1849__tile
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v85 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
      // pto: %q_cos_il_inline1849__tile
      uint64_t v86 = (uint64_t) v50;
      TASSIGN(v85, v86);
      // pto: %37
      int64_t v87 = v75 < v48 ? v48 : v75;
      // pto: %q_cos_il_inline1311__ssa_v0_pview
      pto::Shape<1, 1, 1, 8, 64> v88 = pto::Shape<1, 1, 1, 8, 64>();
      // pto: %q_cos_il_inline1311__ssa_v0_pview
      pto::Stride<512, 512, 512, 64, 1> v89 = pto::Stride<512, 512, 512, 64, 1>();
      // pto: %q_cos_il_inline1311__ssa_v0_pview
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND> v90 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(v3 + (v48 + v87 * v33), v88, v89);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      TLOAD(v85, v90);
      // pto: %q_sin_signed_inline1862__tile
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v91 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
      // pto: %q_sin_signed_inline1862__tile
      uint64_t v92 = (uint64_t) v49;
      TASSIGN(v91, v92);
      // pto: %q_sin_signed_inline1295__ssa_v0_pview
      pto::Shape<1, 1, 1, 8, 64> v93 = pto::Shape<1, 1, 1, 8, 64>();
      // pto: %q_sin_signed_inline1295__ssa_v0_pview
      pto::Stride<512, 512, 512, 64, 1> v94 = pto::Stride<512, 512, 512, 64, 1>();
      // pto: %q_sin_signed_inline1295__ssa_v0_pview
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND> v95 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(v4 + (v48 + v87 * v33), v93, v94);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      TLOAD(v91, v95);
      // pto: %q_swap_idx_inline1788__tile
      Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
      // pto: %q_swap_idx_inline1788__tile
      uint64_t v97 = (uint64_t) v48;
      TASSIGN(v96, v97);
      // pto: %q_swap_idx_inline1313__ssa_v0_pview
      pto::Shape<1, 1, 1, 8, 64> v98 = pto::Shape<1, 1, 1, 8, 64>();
      // pto: %q_swap_idx_inline1313__ssa_v0_pview
      pto::Stride<512, 512, 512, 64, 1> v99 = pto::Stride<512, 512, 512, 64, 1>();
      // pto: %q_swap_idx_inline1313__ssa_v0_pview
      GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND> v100 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(v5 + (v48 + v87 * v33), v98, v99);
      TLOAD(v96, v100);
      for (int64_t j101 = v48; j101 < v32; j101 += v30) {
        // pto: %40, %41
        int64_t v102 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) v73 + (uint64_t) j101)) * (uint64_t) v29);
        // pto: %43, %42, %44
        int64_t v103 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) v73 + (uint64_t) ((int64_t) ((uint64_t) j101 + (uint64_t) v34)))) * (uint64_t) v29);
        // pto: %q_head_acc_inline1865__tile
        Tile<TileType::Vec, int32_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v104 = Tile<TileType::Vec, int32_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
        // pto: %q_head_acc_inline1865__tile
        uint64_t v105 = (uint64_t) v47;
        TASSIGN(v104, v105);
        // pto: %46
        int64_t v106 = v102 < v48 ? v48 : v102;
        // pto: %q_proj_i32_inline1835__rv_v5_pview
        pto::Shape<1, 1, 1, 8, 512> v107 = pto::Shape<1, 1, 1, 8, 512>();
        // pto: %q_proj_i32_inline1835__rv_v5_pview
        pto::Stride<262144, 262144, 262144, 32768, 1> v108 = pto::Stride<262144, 262144, 262144, 32768, 1>();
        // pto: %q_proj_i32_inline1835__rv_v5_pview
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND> v109 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND>(v6 + ((v48 + v78 * v35) + v106), v107, v108);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        TLOAD(v104, v109);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %t__tile
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v110 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v29);
        // pto: %t__tile
        uint64_t v111 = (uint64_t) v46;
        TASSIGN(v110, v111);
        // pto: %wq_b_scale__ssa_v0_pview
        pto::Shape<1, 1, 1, 1, 512> v112 = pto::Shape<1, 1, 1, 1, 512>();
        // pto: %wq_b_scale__ssa_v0_pview
        pto::Stride<512, 512, 512, 512, 1> v113 = pto::Stride<512, 512, 512, 512, 1>();
        // pto: %wq_b_scale__ssa_v0_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v114 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v7 + (v48 + v106), v112, v113);
        TLOAD(v110, v114);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %0
        Tile<TileType::Vec, int32_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v115 = Tile<TileType::Vec, int32_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
        // pto: %0
        uint64_t v116 = (uint64_t) v45;
        TASSIGN(v115, v116);
        // pto: %49
        int64_t v117 = v103 < v48 ? v48 : v103;
        // pto: %50
        pto::Shape<1, 1, 1, 8, 512> v118 = pto::Shape<1, 1, 1, 8, 512>();
        // pto: %50
        pto::Stride<262144, 262144, 262144, 32768, 1> v119 = pto::Stride<262144, 262144, 262144, 32768, 1>();
        // pto: %50
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND> v120 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND>(v6 + ((v48 + v78 * v35) + v117), v118, v119);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        TLOAD(v115, v120);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %1
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v121 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v29);
        // pto: %1
        uint64_t v122 = (uint64_t) v44;
        TASSIGN(v121, v122);
        // pto: %52
        pto::Shape<1, 1, 1, 1, 512> v123 = pto::Shape<1, 1, 1, 1, 512>();
        // pto: %52
        pto::Stride<512, 512, 512, 512, 1> v124 = pto::Stride<512, 512, 512, 512, 1>();
        // pto: %52
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v125 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v7 + (v48 + v117), v123, v124);
        TLOAD(v121, v125);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %q_head_scale_inline1843__tile
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v126 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v29);
        // pto: %q_head_scale_inline1843__tile
        uint64_t v127 = (uint64_t) v46;
        TASSIGN(v126, v127);
        // pto: %q_head_acc_fp32_inline1866__tile
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v128 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
        // pto: %q_head_acc_fp32_inline1866__tile
        uint64_t v129 = (uint64_t) v47;
        TASSIGN(v128, v129);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TCVT(v128, v104, v20, v19);
        // pto: %q_head_row_scaled_inline1867__tile
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v130 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
        // pto: %q_head_row_scaled_inline1867__tile
        uint64_t v131 = (uint64_t) v47;
        TASSIGN(v130, v131);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v130, v128, v76);
        // pto: %q_head_dq_inline1825__tile
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v132 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
        // pto: %q_head_dq_inline1825__tile
        uint64_t v133 = (uint64_t) v47;
        TASSIGN(v132, v133);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TCOLEXPANDMUL(v132, v130, v126);
        // pto: %q_head_sq_inline1868__tile
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v134 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
        // pto: %q_head_sq_inline1868__tile
        uint64_t v135 = (uint64_t) v46;
        TASSIGN(v134, v135);
        pipe_barrier(PIPE_V);
        TMUL(v134, v132, v132);
        // pto: %tmp_tile
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v136 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
        // pto: %tmp_tile
        uint64_t v137 = (uint64_t) v43;
        TASSIGN(v136, v137);
        // pto: %q_head_sq_row_inline1887__tile
        Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v138 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v34);
        // pto: %q_head_sq_row_inline1887__tile
        uint64_t v139 = (uint64_t) v42;
        TASSIGN(v138, v139);
        pipe_barrier(PIPE_V);
        TROWSUM(v138, v134, v136);
        // pto: %q_head_sq_sum_inline1870__tile
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v140 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v31);
        // pto: %q_head_sq_sum_inline1870__tile
        uint64_t v141 = (uint64_t) v42;
        TASSIGN(v140, v141);
        // pto: %q_head_sq_mean_inline1846__tile
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v142 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v31);
        // pto: %q_head_sq_mean_inline1846__tile
        uint64_t v143 = (uint64_t) v46;
        TASSIGN(v142, v143);
        pipe_barrier(PIPE_V);
        TMULS(v142, v140, v28);
        // pto: %q_head_var_inline1864__tile
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v144 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v31);
        // pto: %q_head_var_inline1864__tile
        uint64_t v145 = (uint64_t) v46;
        TASSIGN(v144, v145);
        pipe_barrier(PIPE_V);
        TADDS(v144, v142, v27);
        // pto: %rsqrt_tmp
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v146 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v31);
        // pto: %rsqrt_tmp
        uint64_t v147 = (uint64_t) v43;
        TASSIGN(v146, v147);
        // pto: %q_head_inv_rms_inline1871__tile
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v148 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v31);
        // pto: %q_head_inv_rms_inline1871__tile
        uint64_t v149 = (uint64_t) v42;
        TASSIGN(v148, v149);
        pipe_barrier(PIPE_V);
        TRSQRT(v148, v144, v146);
        // pto: %q_head_inv_rms_t_inline1771__tile
        Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v150 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v34);
        // pto: %q_head_inv_rms_t_inline1771__tile
        uint64_t v151 = (uint64_t) v42;
        TASSIGN(v150, v151);
        // pto: %2
        Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v152 = Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v26);
        // pto: %2
        uint64_t v153 = (uint64_t) v47;
        TASSIGN(v152, v153);
        // pto: %slice_view
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, 8, 448, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v154;
        // pto: %slice_view
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, 8, 448, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v155 = v154;
        // pto: %slice_view
        uint64_t v156 = (uint64_t) v47;
        TASSIGN(v155, v156);
        // pto: %q_nope_normed_inline1872__tile
        Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v157 = Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v26);
        // pto: %q_nope_normed_inline1872__tile
        uint64_t v158 = (uint64_t) v46;
        TASSIGN(v157, v158);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v157, v155, v150);
        // pto: %q_nope_bf16_inline1873__tile
        Tile<TileType::Vec, bfloat16_t, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v159 = Tile<TileType::Vec, bfloat16_t, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v26);
        // pto: %q_nope_bf16_inline1873__tile
        uint64_t v160 = (uint64_t) v46;
        TASSIGN(v159, v160);
        pipe_barrier(PIPE_V);
        TCVT(v159, v157, v18, v19);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %q_rope_chunk_raw_inline1837__tile
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v161 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %q_rope_chunk_raw_inline1837__tile
        uint64_t v162 = (uint64_t) v41;
        TASSIGN(v161, v162);
        // pto: %53
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v163;
        // pto: %53
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v164 = v163;
        // pto: %53
        uint64_t v165 = (uint64_t) v41;
        TASSIGN(v164, v165);
        // pto: %q_rope_chunk_inline1811__tile
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v166 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %q_rope_chunk_inline1811__tile
        uint64_t v167 = (uint64_t) v47;
        TASSIGN(v166, v167);
        TROWEXPANDMUL(v166, v164, v150);
        // pto: %gather_acc_init
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v168 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %gather_acc_init
        uint64_t v169 = (uint64_t) v43;
        TASSIGN(v168, v169);
        for (int64_t k170 = v48; k170 < v31; k170 += v34) {
          // pto: %gather_inp_row
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v171 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
          // pto: %gather_inp_row
          uint64_t v172 = (uint64_t) v47;
          TASSIGN(v171, v172);
          // pto: %54
          int64_t v173 = (int64_t) ((uint64_t) k170 * (uint64_t) v52);
          // pto: %54
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v174;
          // pto: %54
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v175 = v174;
          // pto: %54
          uint64_t v176 = (uint64_t) ((int64_t) ((uint64_t) v173 + (uint64_t) v47));
          TASSIGN(v175, v176);
          // pto: %gather_idx_row
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v177 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
          // pto: %gather_idx_row
          uint64_t v178 = (uint64_t) v48;
          TASSIGN(v177, v178);
          // pto: %55
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v179;
          // pto: %55
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v180 = v179;
          // pto: %55
          uint64_t v181 = (uint64_t) v173;
          TASSIGN(v180, v181);
          // pto: %gather_row_tmp
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v182 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
          // pto: %gather_row_tmp
          uint64_t v183 = (uint64_t) v42;
          TASSIGN(v182, v183);
          // pto: %gather_row
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v184 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
          // pto: %gather_row
          uint64_t v185 = (uint64_t) v40;
          TASSIGN(v184, v185);
          pipe_barrier(PIPE_V);
          TGATHER(v184, v175, v180, v182);
          // pto: %assemble_view
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v186;
          // pto: %assemble_view
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v187 = v186;
          // pto: %assemble_view
          uint64_t v188 = (uint64_t) ((int64_t) ((uint64_t) v173 + (uint64_t) v43));
          TASSIGN(v187, v188);
          pipe_barrier(PIPE_V);
          TMOV(v187, v184);
        }
        // pto: %q_rope_swapped_inline1778__tile
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v189 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %q_rope_swapped_inline1778__tile
        uint64_t v190 = (uint64_t) v43;
        TASSIGN(v189, v190);
        // pto: %q_rope_base_inline1838__tile
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v191 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %q_rope_base_inline1838__tile
        uint64_t v192 = (uint64_t) v47;
        TASSIGN(v191, v192);
        pipe_barrier(PIPE_V);
        TMUL(v191, v166, v85);
        // pto: %q_rope_delta_inline1874__tile
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v193 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %q_rope_delta_inline1874__tile
        uint64_t v194 = (uint64_t) v43;
        TASSIGN(v193, v194);
        TMUL(v193, v189, v91);
        // pto: %q_rope_rot_inline1875__tile
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v195 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %q_rope_rot_inline1875__tile
        uint64_t v196 = (uint64_t) v47;
        TASSIGN(v195, v196);
        pipe_barrier(PIPE_V);
        TADD(v195, v191, v193);
        // pto: %q_rope_bf16_inline1833__tile
        Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v197 = Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %q_rope_bf16_inline1833__tile
        uint64_t v198 = (uint64_t) v47;
        TASSIGN(v197, v198);
        pipe_barrier(PIPE_V);
        TCVT(v197, v195, v18, v19);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        // pto: %q_flat_inline1856__iter_v3_pview
        pto::Shape<1, 1, 1, 8, 448> v199 = pto::Shape<1, 1, 1, 8, 448>();
        // pto: %q_flat_inline1856__iter_v3_pview
        pto::Stride<262144, 262144, 262144, 32768, 1> v200 = pto::Stride<262144, 262144, 262144, 32768, 1>();
        // pto: %q_flat_inline1856__iter_v3_pview
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 448>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND> v201 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 448>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND>(v1 + ((v48 + v87 * v35) + v106), v199, v200);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v201, v159);
        // pto: %60
        int64_t v202 = (int64_t) ((uint64_t) v102 + (uint64_t) v26);
        // pto: %q_flat_inline1856__tile_pview
        pto::Shape<1, 1, 1, 8, 64> v203 = pto::Shape<1, 1, 1, 8, 64>();
        // pto: %q_flat_inline1856__tile_pview
        pto::Stride<262144, 262144, 262144, 32768, 1> v204 = pto::Stride<262144, 262144, 262144, 32768, 1>();
        // pto: %q_flat_inline1856__tile_pview, %61
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND> v205 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND>(v1 + ((v48 + v87 * v35) + (v202 < v48 ? v48 : v202)), v203, v204);
        pipe_barrier(PIPE_MTE3);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TSTORE(v205, v197);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        // pto: %3
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v206 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v29);
        // pto: %3
        uint64_t v207 = (uint64_t) v44;
        TASSIGN(v206, v207);
        // pto: %4
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v208 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
        // pto: %4
        uint64_t v209 = (uint64_t) v45;
        TASSIGN(v208, v209);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        TCVT(v208, v115, v20, v19);
        // pto: %5
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v210 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
        // pto: %5
        uint64_t v211 = (uint64_t) v45;
        TASSIGN(v210, v211);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v210, v208, v76);
        // pto: %6
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v212 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
        // pto: %6
        uint64_t v213 = (uint64_t) v45;
        TASSIGN(v212, v213);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        TCOLEXPANDMUL(v212, v210, v206);
        // pto: %7
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v214 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
        // pto: %7
        uint64_t v215 = (uint64_t) v44;
        TASSIGN(v214, v215);
        pipe_barrier(PIPE_V);
        TMUL(v214, v212, v212);
        // pto: %8
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v216 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
        // pto: %8
        uint64_t v217 = (uint64_t) v39;
        TASSIGN(v216, v217);
        // pto: %9
        Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v218 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v34);
        // pto: %9
        uint64_t v219 = (uint64_t) v38;
        TASSIGN(v218, v219);
        pipe_barrier(PIPE_V);
        TROWSUM(v218, v214, v216);
        // pto: %10
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v220 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v31);
        // pto: %10
        uint64_t v221 = (uint64_t) v38;
        TASSIGN(v220, v221);
        // pto: %11
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v222 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v31);
        // pto: %11
        uint64_t v223 = (uint64_t) v44;
        TASSIGN(v222, v223);
        pipe_barrier(PIPE_V);
        TMULS(v222, v220, v28);
        // pto: %12
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v224 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v31);
        // pto: %12
        uint64_t v225 = (uint64_t) v44;
        TASSIGN(v224, v225);
        pipe_barrier(PIPE_V);
        TADDS(v224, v222, v27);
        // pto: %13
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v226 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v31);
        // pto: %13
        uint64_t v227 = (uint64_t) v39;
        TASSIGN(v226, v227);
        // pto: %14
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v228 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v31);
        // pto: %14
        uint64_t v229 = (uint64_t) v38;
        TASSIGN(v228, v229);
        pipe_barrier(PIPE_V);
        TRSQRT(v228, v224, v226);
        // pto: %15
        Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v230 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v34);
        // pto: %15
        uint64_t v231 = (uint64_t) v38;
        TASSIGN(v230, v231);
        // pto: %16
        Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v232 = Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v26);
        // pto: %16
        uint64_t v233 = (uint64_t) v45;
        TASSIGN(v232, v233);
        // pto: %62
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, 8, 448, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v234;
        // pto: %62
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, 8, 448, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v235 = v234;
        // pto: %62
        uint64_t v236 = (uint64_t) v45;
        TASSIGN(v235, v236);
        // pto: %17
        Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v237 = Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v26);
        // pto: %17
        uint64_t v238 = (uint64_t) v44;
        TASSIGN(v237, v238);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v237, v235, v230);
        // pto: %18
        Tile<TileType::Vec, bfloat16_t, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v239 = Tile<TileType::Vec, bfloat16_t, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v26);
        // pto: %18
        uint64_t v240 = (uint64_t) v44;
        TASSIGN(v239, v240);
        pipe_barrier(PIPE_V);
        TCVT(v239, v237, v18, v19);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        // pto: %19
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v241 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %19
        uint64_t v242 = (uint64_t) v37;
        TASSIGN(v241, v242);
        // pto: %63
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v243;
        // pto: %63
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v244 = v243;
        // pto: %63
        uint64_t v245 = (uint64_t) v37;
        TASSIGN(v244, v245);
        // pto: %20
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v246 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %20
        uint64_t v247 = (uint64_t) v45;
        TASSIGN(v246, v247);
        TROWEXPANDMUL(v246, v244, v230);
        // pto: %21
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v248 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %21
        uint64_t v249 = (uint64_t) v39;
        TASSIGN(v248, v249);
        for (int64_t k250 = v48; k250 < v31; k250 += v34) {
          // pto: %22
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v251 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
          // pto: %22
          uint64_t v252 = (uint64_t) v45;
          TASSIGN(v251, v252);
          // pto: %65
          int64_t v253 = (int64_t) ((uint64_t) k250 * (uint64_t) v52);
          // pto: %65
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v254;
          // pto: %65
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v255 = v254;
          // pto: %65
          uint64_t v256 = (uint64_t) ((int64_t) ((uint64_t) v253 + (uint64_t) v45));
          TASSIGN(v255, v256);
          // pto: %23
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v257 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
          // pto: %23
          uint64_t v258 = (uint64_t) v48;
          TASSIGN(v257, v258);
          // pto: %66
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v259;
          // pto: %66
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v260 = v259;
          // pto: %66
          uint64_t v261 = (uint64_t) v253;
          TASSIGN(v260, v261);
          // pto: %24
          Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v262 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
          // pto: %24
          uint64_t v263 = (uint64_t) v38;
          TASSIGN(v262, v263);
          // pto: %25
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v264 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
          // pto: %25
          uint64_t v265 = (uint64_t) v36;
          TASSIGN(v264, v265);
          pipe_barrier(PIPE_V);
          TGATHER(v264, v255, v260, v262);
          // pto: %67
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v266;
          // pto: %67
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v267 = v266;
          // pto: %67
          uint64_t v268 = (uint64_t) ((int64_t) ((uint64_t) v253 + (uint64_t) v39));
          TASSIGN(v267, v268);
          pipe_barrier(PIPE_V);
          TMOV(v267, v264);
        }
        // pto: %27
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v269 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %27
        uint64_t v270 = (uint64_t) v39;
        TASSIGN(v269, v270);
        // pto: %28
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v271 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %28
        uint64_t v272 = (uint64_t) v45;
        TASSIGN(v271, v272);
        pipe_barrier(PIPE_V);
        TMUL(v271, v246, v85);
        // pto: %29
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v273 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %29
        uint64_t v274 = (uint64_t) v39;
        TASSIGN(v273, v274);
        TMUL(v273, v269, v91);
        // pto: %30
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v275 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %30
        uint64_t v276 = (uint64_t) v45;
        TASSIGN(v275, v276);
        pipe_barrier(PIPE_V);
        TADD(v275, v271, v273);
        // pto: %31
        Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v277 = Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %31
        uint64_t v278 = (uint64_t) v45;
        TASSIGN(v277, v278);
        pipe_barrier(PIPE_V);
        TCVT(v277, v275, v18, v19);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        // pto: %71
        pto::Shape<1, 1, 1, 8, 448> v279 = pto::Shape<1, 1, 1, 8, 448>();
        // pto: %71
        pto::Stride<262144, 262144, 262144, 32768, 1> v280 = pto::Stride<262144, 262144, 262144, 32768, 1>();
        // pto: %71
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 448>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND> v281 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 448>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND>(v1 + ((v48 + v87 * v35) + v117), v279, v280);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v281, v239);
        // pto: %74
        int64_t v282 = (int64_t) ((uint64_t) v103 + (uint64_t) v26);
        // pto: %76
        pto::Shape<1, 1, 1, 8, 64> v283 = pto::Shape<1, 1, 1, 8, 64>();
        // pto: %76
        pto::Stride<262144, 262144, 262144, 32768, 1> v284 = pto::Stride<262144, 262144, 262144, 32768, 1>();
        // pto: %76, %75
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND> v285 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND>(v1 + ((v48 + v87 * v35) + (v282 < v48 ? v48 : v282)), v283, v284);
        pipe_barrier(PIPE_MTE3);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        TSTORE(v285, v277);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      }
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    } else {
      // pto: %77
      int64_t v286 = (int64_t) ((uint64_t) v9 - (uint64_t) i74);
      // pto: %qr_scale_dq_tail_inline1876__ssa_v0
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v287 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v34);
      // pto: %qr_scale_dq_tail_inline1876__ssa_v0
      uint64_t v288 = (uint64_t) v49;
      TASSIGN(v287, v288);
      // pto: %78
      int64_t v289 = i74 < v48 ? v48 : i74;
      // pto: %79
      __gm__ float* v290 = PTOAS__GLOBAL_TENSOR_DATA(v62);
      // pto: %79
      int64_t v291 = v31 * v34;
      // pto: %79
      int64_t v292 = v34 * v291;
      // pto: %79
      pto::Shape<1, 1, 1, 8, 1> v293 = pto::Shape<1, 1, 1, 8, 1>(v34, v34, v34, v31, v34);
      // pto: %79
      pto::Stride<-1, -1, -1, -1, -1> v294 = pto::Stride<-1, -1, -1, -1, -1>(v34 * v292, v292, v291, v34, v12);
      // pto: %79
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v295 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v290 + ((v48 + v289 * v34) + v48 * v12), v293, v294);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      TLOAD(v287, v295);
      // pto: %q_cos_il_tail_inline1880__ssa_v0
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v296 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v286, v33);
      // pto: %q_cos_il_tail_inline1880__ssa_v0
      uint64_t v297 = (uint64_t) v43;
      TASSIGN(v296, v297);
      // pto: %80
      int64_t v298 = v75 < v48 ? v48 : v75;
      // pto: %81
      __gm__ float* v299 = PTOAS__GLOBAL_TENSOR_DATA(v67);
      // pto: %81
      int64_t v300 = v286 * v33;
      // pto: %81
      int64_t v301 = v34 * v300;
      // pto: %81
      pto::Shape<1, 1, 1, -1, 64> v302 = pto::Shape<1, 1, 1, -1, 64>(v34, v34, v34, v286, v33);
      // pto: %81
      pto::Stride<-1, -1, -1, -1, -1> v303 = pto::Stride<-1, -1, -1, -1, -1>(v34 * v301, v301, v300, v33, v34);
      // pto: %81
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v304 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v299 + ((v48 + v298 * v33) + v48 * v34), v302, v303);
      pipe_barrier(PIPE_ALL);
      TLOAD(v296, v304);
      // pto: %q_sin_signed_tail_inline1882__ssa_v0
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v305 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v286, v33);
      // pto: %q_sin_signed_tail_inline1882__ssa_v0
      uint64_t v306 = (uint64_t) v44;
      TASSIGN(v305, v306);
      // pto: %83
      __gm__ float* v307 = PTOAS__GLOBAL_TENSOR_DATA(v72);
      // pto: %83
      int64_t v308 = v286 * v33;
      // pto: %83
      int64_t v309 = v34 * v308;
      // pto: %83
      pto::Shape<1, 1, 1, -1, 64> v310 = pto::Shape<1, 1, 1, -1, 64>(v34, v34, v34, v286, v33);
      // pto: %83
      pto::Stride<-1, -1, -1, -1, -1> v311 = pto::Stride<-1, -1, -1, -1, -1>(v34 * v309, v309, v308, v33, v34);
      // pto: %83
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v312 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v307 + ((v48 + v298 * v33) + v48 * v34), v310, v311);
      pipe_barrier(PIPE_ALL);
      TLOAD(v305, v312);
      // pto: %t__tmp_v140
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v313 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
      // pto: %t__tmp_v140
      uint64_t v314 = (uint64_t) v47;
      TASSIGN(v313, v314);
      pipe_barrier(PIPE_V);
      TEXPANDS(v313, v25);
      // pto: %t__tmp_v141
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v315 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
      // pto: %t__tmp_v141
      uint64_t v316 = (uint64_t) v45;
      TASSIGN(v315, v316);
      wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      TCI<Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v315, v24);
      set_flag(PIPE_S, PIPE_V, EVENT_ID0);
      // pto: %t__tmp_v142
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v317 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v33);
      // pto: %t__tmp_v142
      uint64_t v318 = (uint64_t) v45;
      TASSIGN(v317, v318);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
      TCVT(v317, v315, v17, v19);
      // pto: %q_col_inline1883__ssa_v0
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v319 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
      // pto: %q_col_inline1883__ssa_v0
      uint64_t v320 = (uint64_t) v47;
      TASSIGN(v319, v320);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v319, v313, v317);
      // pto: %t__tmp_v143
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v321 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
      // pto: %t__tmp_v143
      uint64_t v322 = (uint64_t) v45;
      TASSIGN(v321, v322);
      pipe_barrier(PIPE_V);
      TMULS(v321, v319, v23);
      // pto: %t__tmp_v144
      Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v323 = Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
      // pto: %t__tmp_v144
      uint64_t v324 = (uint64_t) v45;
      TASSIGN(v323, v324);
      pipe_barrier(PIPE_V);
      TCVT(v323, v321, v16, v19);
      // pto: %q_dup_f_inline1884__ssa_v0
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v325 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
      // pto: %q_dup_f_inline1884__ssa_v0
      uint64_t v326 = (uint64_t) v45;
      TASSIGN(v325, v326);
      pipe_barrier(PIPE_V);
      TCVT(v325, v323, v17, v19);
      // pto: %t__tmp_v145
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v327 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
      // pto: %t__tmp_v145
      uint64_t v328 = (uint64_t) v45;
      TASSIGN(v327, v328);
      pipe_barrier(PIPE_V);
      TMULS(v327, v325, v22);
      // pto: %q_lane_inline1869__ssa_v0
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v329 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
      // pto: %q_lane_inline1869__ssa_v0
      uint64_t v330 = (uint64_t) v45;
      TASSIGN(v329, v330);
      pipe_barrier(PIPE_V);
      TSUB(v329, v319, v327);
      // pto: %t__tmp_v146
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v331 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
      // pto: %t__tmp_v146
      uint64_t v332 = (uint64_t) v47;
      TASSIGN(v331, v332);
      pipe_barrier(PIPE_V);
      TADDS(v331, v319, v25);
      // pto: %t__tmp_v147
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v333 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
      // pto: %t__tmp_v147
      uint64_t v334 = (uint64_t) v45;
      TASSIGN(v333, v334);
      TMULS(v333, v329, v22);
      // pto: %q_swap_f_inline1803__ssa_v0
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v335 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
      // pto: %q_swap_f_inline1803__ssa_v0
      uint64_t v336 = (uint64_t) v47;
      TASSIGN(v335, v336);
      pipe_barrier(PIPE_V);
      TSUB(v335, v331, v333);
      set_flag(PIPE_V, PIPE_S, EVENT_ID1);
      // pto: %t__tmp_v148
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v337 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v31);
      // pto: %t__tmp_v148
      uint64_t v338 = (uint64_t) v45;
      TASSIGN(v337, v338);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
      TCI<Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v337, v24);
      set_flag(PIPE_S, PIPE_V, EVENT_ID1);
      // pto: %t__tmp_v149
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v339 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v31);
      // pto: %t__tmp_v149
      uint64_t v340 = (uint64_t) v45;
      TASSIGN(v339, v340);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
      TCVT(v339, v337, v17, v19);
      // pto: %q_row_seed_inline1759__ssa_v0
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v341 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v31);
      // pto: %q_row_seed_inline1759__ssa_v0
      uint64_t v342 = (uint64_t) v46;
      TASSIGN(v341, v342);
      pipe_barrier(PIPE_V);
      TMULS(v341, v339, v21);
      // pto: %t__tmp_v150
      Tile<TileType::Vec, float, 64, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v343 = Tile<TileType::Vec, float, 64, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
      // pto: %t__tmp_v150
      uint64_t v344 = (uint64_t) v45;
      TASSIGN(v343, v344);
      pipe_barrier(PIPE_V);
      TEXPANDS(v343, v25);
      // pto: %q_row_grid_inline1820__ssa_v0
      Tile<TileType::Vec, float, 64, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v345 = Tile<TileType::Vec, float, 64, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
      // pto: %q_row_grid_inline1820__ssa_v0
      uint64_t v346 = (uint64_t) v45;
      TASSIGN(v345, v346);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v345, v343, v341);
      // pto: %transpose_tmp
      Tile<TileType::Vec, float, 64, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v347 = Tile<TileType::Vec, float, 64, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
      // pto: %transpose_tmp
      uint64_t v348 = (uint64_t) v46;
      TASSIGN(v347, v348);
      // pto: %q_row_offset_inline1885__ssa_v0
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v349 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
      // pto: %q_row_offset_inline1885__ssa_v0
      uint64_t v350 = (uint64_t) v39;
      TASSIGN(v349, v350);
      pipe_barrier(PIPE_V);
      TTRANS(v349, v345, v347);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      // pto: %t__tmp_v151
      Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v351 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
      // pto: %t__tmp_v151
      uint64_t v352 = (uint64_t) v47;
      TASSIGN(v351, v352);
      pipe_barrier(PIPE_V);
      TADD(v351, v335, v349);
      // pto: %q_swap_idx_tail_inline1877__ssa_v0
      Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v353 = Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
      // pto: %q_swap_idx_tail_inline1877__ssa_v0
      uint64_t v354 = (uint64_t) v39;
      TASSIGN(v353, v354);
      pipe_barrier(PIPE_V);
      TCVT(v353, v351, v17, v19);
      // pto: %q_head_reduce_tmp_inline1760__ssa_v0
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v355 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
      // pto: %q_head_reduce_tmp_inline1760__ssa_v0
      uint64_t v356 = (uint64_t) v47;
      TASSIGN(v355, v356);
      // pto: %q_gather_tmp_inline1888__ssa_v0
      Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v357 = Tile<TileType::Vec, int32_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
      // pto: %q_gather_tmp_inline1888__ssa_v0
      uint64_t v358 = (uint64_t) v50;
      TASSIGN(v357, v358);
      pipe_barrier(PIPE_ALL);
      for (int64_t j359 = v48; j359 < v32; j359 += v34) {
        // pto: %84, %85
        int64_t v360 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) v73 + (uint64_t) j359)) * (uint64_t) v29);
        // pto: %q_head_acc_tail_inline1839__ssa_v0
        Tile<TileType::Vec, int32_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v361 = Tile<TileType::Vec, int32_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
        // pto: %q_head_acc_tail_inline1839__ssa_v0
        uint64_t v362 = (uint64_t) v45;
        TASSIGN(v361, v362);
        // pto: %87
        int64_t v363 = v360 < v48 ? v48 : v360;
        // pto: %88
        pto::Shape<1, 1, 1, 8, 512> v364 = pto::Shape<1, 1, 1, 8, 512>();
        // pto: %88
        pto::Stride<262144, 262144, 262144, 32768, 1> v365 = pto::Stride<262144, 262144, 262144, 32768, 1>();
        // pto: %88
        GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND> v366 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<262144, 262144, 262144, 32768, 1>, pto::Layout::ND>(v6 + ((v48 + v289 * v35) + v363), v364, v365);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
        TLOAD(v361, v366);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        // pto: %q_head_scale_input_tail_inline1840__ssa_v0
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v367 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v29);
        // pto: %q_head_scale_input_tail_inline1840__ssa_v0
        uint64_t v368 = (uint64_t) v46;
        TASSIGN(v367, v368);
        // pto: %90
        pto::Shape<1, 1, 1, 1, 512> v369 = pto::Shape<1, 1, 1, 1, 512>();
        // pto: %90
        pto::Stride<512, 512, 512, 512, 1> v370 = pto::Stride<512, 512, 512, 512, 1>();
        // pto: %90
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v371 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v7 + (v48 + v363), v369, v370);
        TLOAD(v367, v371);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        // pto: %q_head_scale_tail_inline1816__ssa_v0
        Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v372 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v29);
        // pto: %q_head_scale_tail_inline1816__ssa_v0
        uint64_t v373 = (uint64_t) v46;
        TASSIGN(v372, v373);
        // pto: %q_head_acc_fp32_tail_inline1755__ssa_v0
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v374 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
        // pto: %q_head_acc_fp32_tail_inline1755__ssa_v0
        uint64_t v375 = (uint64_t) v45;
        TASSIGN(v374, v375);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        TCVT(v374, v361, v20, v19);
        // pto: %q_head_row_scaled_tail_inline1754__ssa_v0
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v376 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
        // pto: %q_head_row_scaled_tail_inline1754__ssa_v0
        uint64_t v377 = (uint64_t) v45;
        TASSIGN(v376, v377);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v376, v374, v287);
        // pto: %q_head_dq_tail_inline1806__ssa_v0
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v378 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
        // pto: %q_head_dq_tail_inline1806__ssa_v0
        uint64_t v379 = (uint64_t) v45;
        TASSIGN(v378, v379);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        TCOLEXPANDMUL(v378, v376, v372);
        // pto: %q_head_sq_tail_inline1753__ssa_v0
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v380 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
        // pto: %q_head_sq_tail_inline1753__ssa_v0
        uint64_t v381 = (uint64_t) v46;
        TASSIGN(v380, v381);
        pipe_barrier(PIPE_V);
        TMUL(v380, v378, v378);
        // pto: %q_head_sq_sum_tail_inline1752__ssa_v0
        Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v382 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v34);
        // pto: %q_head_sq_sum_tail_inline1752__ssa_v0
        uint64_t v383 = (uint64_t) v48;
        TASSIGN(v382, v383);
        pipe_barrier(PIPE_V);
        TROWSUM(v382, v380, v355);
        // pto: %t__rm_a0_tmp_v0
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v384 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v31);
        // pto: %t__rm_a0_tmp_v0
        uint64_t v385 = (uint64_t) v48;
        TASSIGN(v384, v385);
        // pto: %t__row_major_tmp_v1
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v386 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v31);
        // pto: %t__row_major_tmp_v1
        uint64_t v387 = (uint64_t) v46;
        TASSIGN(v386, v387);
        pipe_barrier(PIPE_V);
        TMULS(v386, v384, v28);
        // pto: %t__tmp_v152
        Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v388 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v34);
        // pto: %t__tmp_v152
        uint64_t v389 = (uint64_t) v46;
        TASSIGN(v388, v389);
        // pto: %t__rm_a0_tmp_v2
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v390 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v31);
        // pto: %t__rm_a0_tmp_v2
        uint64_t v391 = (uint64_t) v46;
        TASSIGN(v390, v391);
        // pto: %t__row_major_tmp_v3
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v392 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v31);
        // pto: %t__row_major_tmp_v3
        uint64_t v393 = (uint64_t) v46;
        TASSIGN(v392, v393);
        pipe_barrier(PIPE_V);
        TADDS(v392, v390, v27);
        // pto: %t__tmp_v153
        Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v394 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v34);
        // pto: %t__tmp_v153
        uint64_t v395 = (uint64_t) v46;
        TASSIGN(v394, v395);
        // pto: %t__rm_a0_tmp_v4
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v396 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v31);
        // pto: %t__rm_a0_tmp_v4
        uint64_t v397 = (uint64_t) v46;
        TASSIGN(v396, v397);
        // pto: %t__row_major_tmp_v5
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v398 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v31);
        // pto: %t__row_major_tmp_v5
        uint64_t v399 = (uint64_t) v46;
        TASSIGN(v398, v399);
        pipe_barrier(PIPE_V);
        TSQRT(v398, v396);
        // pto: %t__tmp_v154
        Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v400 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v34);
        // pto: %t__tmp_v154
        uint64_t v401 = (uint64_t) v46;
        TASSIGN(v400, v401);
        // pto: %q_head_inv_rms_tail_inline1783__rm_a0_tmp_v6
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v402 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v31);
        // pto: %q_head_inv_rms_tail_inline1783__rm_a0_tmp_v6
        uint64_t v403 = (uint64_t) v46;
        TASSIGN(v402, v403);
        // pto: %q_head_inv_rms_tail_inline1783__row_major_tmp_v7
        Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v404 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v34, v31);
        // pto: %q_head_inv_rms_tail_inline1783__row_major_tmp_v7
        uint64_t v405 = (uint64_t) v48;
        TASSIGN(v404, v405);
        pipe_barrier(PIPE_V);
        TRECIP(v404, v402);
        // pto: %q_head_inv_rms_tail_inline1783__ssa_v0
        Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v406 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v34);
        // pto: %q_head_inv_rms_tail_inline1783__ssa_v0
        uint64_t v407 = (uint64_t) v48;
        TASSIGN(v406, v407);
        // pto: %t__tmp_v155
        Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v408 = Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v26);
        // pto: %t__tmp_v155
        uint64_t v409 = (uint64_t) v45;
        TASSIGN(v408, v409);
        // pto: %91
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, 8, 448, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v410;
        // pto: %91
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, 8, 448, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v411 = v410;
        // pto: %91
        uint64_t v412 = (uint64_t) v45;
        TASSIGN(v411, v412);
        // pto: %q_nope_normed_tail_inline1751__ssa_v0
        Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v413 = Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v26);
        // pto: %q_nope_normed_tail_inline1751__ssa_v0
        uint64_t v414 = (uint64_t) v46;
        TASSIGN(v413, v414);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v413, v411, v406);
        // pto: %q_nope_bf16_tail_inline1750__ssa_v0
        Tile<TileType::Vec, bfloat16_t, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v415 = Tile<TileType::Vec, bfloat16_t, 8, 448, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v26);
        // pto: %q_nope_bf16_tail_inline1750__ssa_v0
        uint64_t v416 = (uint64_t) v46;
        TASSIGN(v415, v416);
        pipe_barrier(PIPE_V);
        TCVT(v415, v413, v18, v19);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
        v415.SetValidShape(v286, v26);
        // pto: %q_flat_inline1856__iter_v1_pview
        __gm__ bfloat16_t* v417 = PTOAS__GLOBAL_TENSOR_DATA(v57);
        // pto: %q_flat_inline1856__iter_v1_pview
        int64_t v418 = v286 * v35;
        // pto: %q_flat_inline1856__iter_v1_pview
        int64_t v419 = v34 * v418;
        // pto: %q_flat_inline1856__iter_v1_pview
        pto::Shape<1, 1, 1, -1, 448> v420 = pto::Shape<1, 1, 1, -1, 448>(v34, v34, v34, v286, v26);
        // pto: %q_flat_inline1856__iter_v1_pview
        pto::Stride<-1, -1, -1, -1, -1> v421 = pto::Stride<-1, -1, -1, -1, -1>(v34 * v419, v419, v418, v35, v34);
        // pto: %q_flat_inline1856__iter_v1_pview
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 448>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v422 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 448>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v417 + ((v48 + v298 * v35) + v363 * v34), v420, v421);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
        TSTORE(v422, v415);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        // pto: %q_rope_chunk_raw_tail_inline1748__ssa_v0
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v423 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %q_rope_chunk_raw_tail_inline1748__ssa_v0
        uint64_t v424 = (uint64_t) v37;
        TASSIGN(v423, v424);
        // pto: %94
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v425;
        // pto: %94
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v426 = v425;
        // pto: %94
        uint64_t v427 = (uint64_t) v37;
        TASSIGN(v426, v427);
        // pto: %q_rope_chunk_tail_inline1812__ssa_v0
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v428 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %q_rope_chunk_tail_inline1812__ssa_v0
        uint64_t v429 = (uint64_t) v45;
        TASSIGN(v428, v429);
        TROWEXPANDMUL(v428, v426, v406);
        // pto: %q_rope_swapped_tail_inline1747__ssa_v0
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v430 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %q_rope_swapped_tail_inline1747__ssa_v0
        uint64_t v431 = (uint64_t) v46;
        TASSIGN(v430, v431);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        TGATHER(v430, v428, v353, v357);
        // pto: %q_rope_base_tail_inline1746__ssa_v0
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v432 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %q_rope_base_tail_inline1746__ssa_v0
        uint64_t v433 = (uint64_t) v45;
        TASSIGN(v432, v433);
        pipe_barrier(PIPE_V);
        TMUL(v432, v428, v296);
        // pto: %q_rope_delta_tail_inline1745__ssa_v0
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v434 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %q_rope_delta_tail_inline1745__ssa_v0
        uint64_t v435 = (uint64_t) v46;
        TASSIGN(v434, v435);
        TMUL(v434, v430, v305);
        // pto: %q_rope_rot_tail_inline1854__ssa_v0
        Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v436 = Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %q_rope_rot_tail_inline1854__ssa_v0
        uint64_t v437 = (uint64_t) v45;
        TASSIGN(v436, v437);
        pipe_barrier(PIPE_V);
        TADD(v436, v432, v434);
        // pto: %q_rope_bf16_tail_inline1744__ssa_v0
        Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v438 = Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
        // pto: %q_rope_bf16_tail_inline1744__ssa_v0
        uint64_t v439 = (uint64_t) v45;
        TASSIGN(v438, v439);
        pipe_barrier(PIPE_V);
        TCVT(v438, v436, v18, v19);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
        v438.SetValidShape(v286, v33);
        // pto: %96
        int64_t v440 = (int64_t) ((uint64_t) v360 + (uint64_t) v26);
        // pto: %98
        __gm__ bfloat16_t* v441 = PTOAS__GLOBAL_TENSOR_DATA(v57);
        // pto: %98
        int64_t v442 = v286 * v35;
        // pto: %98
        int64_t v443 = v34 * v442;
        // pto: %98
        pto::Shape<1, 1, 1, -1, 64> v444 = pto::Shape<1, 1, 1, -1, 64>(v34, v34, v34, v286, v33);
        // pto: %98
        pto::Stride<-1, -1, -1, -1, -1> v445 = pto::Stride<-1, -1, -1, -1, -1>(v34 * v443, v443, v442, v35, v34);
        // pto: %98, %97
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v446 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v441 + ((v48 + v298 * v35) + (v440 < v48 ? v48 : v440) * v34), v444, v445);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
        TSTORE(v446, v438);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      }
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      set_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
    }
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}