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

AICORE void scatter_softmax_pool(__gm__ int32_t* v1, __gm__ float* v2, __gm__ float* v3, __gm__ float* v4, __gm__ float* v5, __gm__ int32_t* v6, __gm__ float* v7, int64_t v8, int64_t v9, int64_t v10, int64_t v11, int32_t v12, int32_t v13) {
  const int64_t v14 = 2048;
  const int64_t v15 = 2;
  const float v16 = -3.40282347E+38f;
  const int64_t v17 = 7;
  const int64_t v18 = 64;
  const int64_t v19 = 8;
  const float v20 = 0.0f;
  const int64_t v21 = 4;
  const int64_t v22 = 1;
  const int64_t v23 = 1280;
  const int64_t v24 = 1024;
  const int64_t v25 = 768;
  const int64_t v26 = 512;
  const int64_t v27 = 256;
  const int64_t v28 = 0;
  const int64_t v29 = 3584;
  const int64_t v30 = 1536;
  const int64_t v31 = -7;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %score_inline2003__phi_v3
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v32 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
  // pto: %score_inline2003__phi_v3
  uint64_t v33 = (uint64_t) v24;
  TASSIGN(v32, v33);
  // pto: %value_inline2050__phi_v3
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v34 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
  // pto: %value_inline2050__phi_v3
  uint64_t v35 = (uint64_t) v25;
  TASSIGN(v34, v35);
  // pto: %score_inline2003__phi_v2
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v36 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
  // pto: %score_inline2003__phi_v2
  uint64_t v37 = (uint64_t) v24;
  TASSIGN(v36, v37);
  // pto: %value_inline2050__phi_v2
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v38 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
  // pto: %value_inline2050__phi_v2
  uint64_t v39 = (uint64_t) v25;
  TASSIGN(v38, v39);
  // pto: %score_inline2003__phi_v6
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v40 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
  // pto: %score_inline2003__phi_v6
  uint64_t v41 = (uint64_t) v26;
  TASSIGN(v40, v41);
  // pto: %value_inline2050__phi_v6
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v42 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
  // pto: %value_inline2050__phi_v6
  uint64_t v43 = (uint64_t) v27;
  TASSIGN(v42, v43);
  // pto: %score_inline2003__phi_v5
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v44 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
  // pto: %score_inline2003__phi_v5
  uint64_t v45 = (uint64_t) v26;
  TASSIGN(v44, v45);
  // pto: %value_inline2050__phi_v5
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v46 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
  // pto: %value_inline2050__phi_v5
  uint64_t v47 = (uint64_t) v27;
  TASSIGN(v46, v47);
  // pto: %c_idx_inline2049__ssa_v0
  int64_t v48 = (int64_t) v12;
  // pto: %21
  int64_t v49 = (int64_t) ((uint64_t) v48 * (uint64_t) v8);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  // pto: %first_pos_b_inline2028__tile
  int32_t v50 = (v1)[v49];
  for (int64_t i51 = v28; i51 < v8; i51 += v22) {
    // pto: %23
    int64_t v52 = (int64_t) ((uint64_t) v49 + (uint64_t) i51);
    // pto: %token_pos_inline2041__tile
    int32_t v53 = (v1)[v52];
    // pto: %t__tile
    Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v54 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v26);
    // pto: %t__tile
    uint64_t v55 = (uint64_t) v30;
    TASSIGN(v54, v55);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TEXPANDS(v54, v20);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %24
    int64_t v56 = v52 < v28 ? v28 : v52;
    // pto: %pooled_kv_inline2008__iter_v1_pview
    pto::Shape<1, 1, 1, 1, 512> v57 = pto::Shape<1, 1, 1, 1, 512>();
    // pto: %pooled_kv_inline2008__iter_v1_pview
    pto::Stride<512, 512, 512, 512, 1> v58 = pto::Stride<512, 512, 512, 512, 1>();
    // pto: %pooled_kv_inline2008__iter_v1_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v59 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v2 + (v28 + v56 * v26), v57, v58);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v59, v54);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    // pto: %25
    int64_t v60 = (int64_t) v53;
    // pto: %26, %27, %28
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    if ((int64_t) ((uint64_t) v60 + (uint64_t) v22) % v21 == v28) {
      for (int64_t j61 = v28; j61 < v26; j61 += v18) {
        // pto: %33
        int64_t v62 = v60 % v21;
        // pto: %0
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v63 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
        // pto: %0
        uint64_t v64 = (uint64_t) v30;
        TASSIGN(v63, v64);
        // pto: %36
        int64_t v65 = (int64_t) ((uint64_t) j61 + (uint64_t) v26);
        // pto: %37
        int64_t v66 = v65 < v28 ? v28 : v65;
        // pto: %cmp4_score_proj_pad_inline2019__ssa_v1_pview
        pto::Shape<1, 1, 1, 1, 64> v67 = pto::Shape<1, 1, 1, 1, 64>();
        // pto: %cmp4_score_proj_pad_inline2019__ssa_v1_pview
        pto::Stride<1024, 1024, 1024, 1024, 1> v68 = pto::Stride<1024, 1024, 1024, 1024, 1>();
        // pto: %cmp4_score_proj_pad_inline2019__ssa_v1_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v69 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v3 + ((v28 + v56 * v24) + v66), v67, v68);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        TLOAD(v63, v69);
        // pto: %1
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v70 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
        // pto: %1
        uint64_t v71 = (uint64_t) v29;
        TASSIGN(v70, v71);
        // pto: %cmp_ape__ssa_v0_pview
        pto::Shape<1, 1, 1, 1, 64> v72 = pto::Shape<1, 1, 1, 1, 64>();
        // pto: %cmp_ape__ssa_v0_pview
        pto::Stride<1024, 1024, 1024, 1024, 1> v73 = pto::Stride<1024, 1024, 1024, 1024, 1>();
        // pto: %38, %cmp_ape__ssa_v0_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v74 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v4 + ((v28 + (v62 < v28 ? v28 : v62) * v24) + v66), v72, v73);
        TLOAD(v70, v74);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %mi_inline2045__tile
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v75 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
        // pto: %mi_inline2045__tile
        uint64_t v76 = (uint64_t) v30;
        TASSIGN(v75, v76);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TADD(v75, v63, v70);
        // pto: %2
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v77 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
        // pto: %2
        uint64_t v78 = (uint64_t) v29;
        TASSIGN(v77, v78);
        pipe_barrier(PIPE_V);
        TSUB(v77, v75, v75);
        // pto: %li_inline2029__tile
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v79 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
        // pto: %li_inline2029__tile
        uint64_t v80 = (uint64_t) v29;
        TASSIGN(v79, v80);
        pipe_barrier(PIPE_V);
        TEXP(v79, v77);
        // pto: %oi_inline2046__tile
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v81 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
        // pto: %oi_inline2046__tile
        uint64_t v82 = (uint64_t) v28;
        TASSIGN(v81, v82);
        // pto: %cmp4_kv_proj_pad_inline2031__ssa_v1_pview
        pto::Shape<1, 1, 1, 1, 64> v83 = pto::Shape<1, 1, 1, 1, 64>();
        // pto: %cmp4_kv_proj_pad_inline2031__ssa_v1_pview
        pto::Stride<1024, 1024, 1024, 1024, 1> v84 = pto::Stride<1024, 1024, 1024, 1024, 1>();
        // pto: %cmp4_kv_proj_pad_inline2031__ssa_v1_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v85 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v5 + ((v28 + v56 * v24) + v66), v83, v84);
        TLOAD(v81, v85);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        for (int64_t k86 = v28; k86 < v17; k86 += v22) {
          // pto: %44
          int64_t v87 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) v60 + (uint64_t) v31)) + (uint64_t) k86);
          // pto: %value_inline2050__tile
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v88 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
          // pto: %value_inline2050__tile
          uint64_t v89 = (uint64_t) v27;
          TASSIGN(v88, v89);
          pipe_barrier(PIPE_V);
          TEXPANDS(v88, v20);
          // pto: %score_inline2003__tile
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v90 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
          // pto: %score_inline2003__tile
          uint64_t v91 = (uint64_t) v26;
          TASSIGN(v90, v91);
          TEXPANDS(v90, v16);
          // pto: %45, %state_half_inline2052__phi_v2
          int64_t v92 = k86 >= v21 ? v26 : v28;
          // pto: %47
          int64_t v93 = (int64_t) v50;
          // pto: %46, %48, %49
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          if (v87 >= v28 & v87 < v93) {
            // pto: %50
            int64_t v94 = v87 % v19;
            // pto: %flat_offset_mul, %flat_offset, %51, %state_blk_id_i32_inline1996__tile
            int32_t v95 = (v6)[(int64_t) ((uint64_t) ((int64_t) ((uint64_t) v48 * (uint64_t) v21)) + (uint64_t) (v94 / v15))];
            // pto: %52
            int64_t v96 = (int64_t) v95;
            // pto: %53
            if (v96 >= v28) {
              // pto: %55, %57, %56
              int64_t v97 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) v96 * (uint64_t) v15)) + (uint64_t) (v94 % v15));
              // pto: %3
              Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v98 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
              // pto: %3
              uint64_t v99 = (uint64_t) v25;
              TASSIGN(v98, v99);
              // pto: %58
              int64_t v100 = v97 < v28 ? v28 : v97;
              // pto: %59
              int64_t v101 = (int64_t) ((uint64_t) v92 + (uint64_t) j61);
              // pto: %compress_state_flat_inline2023__ssa_v0_pview
              pto::Shape<1, 1, 1, 1, 64> v102 = pto::Shape<1, 1, 1, 1, 64>();
              // pto: %compress_state_flat_inline2023__ssa_v0_pview
              pto::Stride<2048, 2048, 2048, 2048, 1> v103 = pto::Stride<2048, 2048, 2048, 2048, 1>();
              // pto: %compress_state_flat_inline2023__ssa_v0_pview, %60
              GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND> v104 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>(v7 + ((v28 + v100 * v14) + (v101 < v28 ? v28 : v101)), v102, v103);
              TLOAD(v98, v104);
              // pto: %4
              Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v105 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
              // pto: %4
              uint64_t v106 = (uint64_t) v24;
              TASSIGN(v105, v106);
              // pto: %63
              int64_t v107 = (int64_t) ((uint64_t) v101 + (uint64_t) v24);
              // pto: %65
              pto::Shape<1, 1, 1, 1, 64> v108 = pto::Shape<1, 1, 1, 1, 64>();
              // pto: %65
              pto::Stride<2048, 2048, 2048, 2048, 1> v109 = pto::Stride<2048, 2048, 2048, 2048, 1>();
              // pto: %65, %64
              GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND> v110 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<2048, 2048, 2048, 2048, 1>, pto::Layout::ND>(v7 + ((v28 + v100 * v14) + (v107 < v28 ? v28 : v107)), v108, v109);
              TLOAD(v105, v110);
            } else {
              // pto: %score_inline2003__tile_mv
              Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v111 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
              // pto: %score_inline2003__tile_mv
              uint64_t v112 = (uint64_t) v24;
              TASSIGN(v111, v112);
              pipe_barrier(PIPE_V);
              TMOV(v111, v90);
              // pto: %value_inline2050__tile_mv
              Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v113 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
              // pto: %value_inline2050__tile_mv
              uint64_t v114 = (uint64_t) v25;
              TASSIGN(v113, v114);
              TMOV(v113, v88);
            }
          } else {
            // pto: %5
            Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v115 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
            // pto: %5
            uint64_t v116 = (uint64_t) v24;
            TASSIGN(v115, v116);
            pipe_barrier(PIPE_V);
            TMOV(v115, v90);
            // pto: %6
            Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v117 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
            // pto: %6
            uint64_t v118 = (uint64_t) v25;
            TASSIGN(v117, v118);
            TMOV(v117, v88);
          }
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
          // pto: %67
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
          if (v93 <= v87) {
            // pto: %69
            if (v87 <= v60) {
              // pto: %71, %73
              int64_t v119 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) v49 + (uint64_t) v87)) - (uint64_t) v93);
              // pto: %74
              int64_t v120 = v87 % v21;
              // pto: %7
              Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v121 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
              // pto: %7
              uint64_t v122 = (uint64_t) v27;
              TASSIGN(v121, v122);
              // pto: %76
              int64_t v123 = v119 < v28 ? v28 : v119;
              // pto: %77
              int64_t v124 = (int64_t) ((uint64_t) v92 + (uint64_t) j61);
              // pto: %78
              int64_t v125 = v124 < v28 ? v28 : v124;
              // pto: %79
              pto::Shape<1, 1, 1, 1, 64> v126 = pto::Shape<1, 1, 1, 1, 64>();
              // pto: %79
              pto::Stride<1024, 1024, 1024, 1024, 1> v127 = pto::Stride<1024, 1024, 1024, 1024, 1>();
              // pto: %79
              GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v128 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v5 + ((v28 + v123 * v24) + v125), v126, v127);
              TLOAD(v121, v128);
              // pto: %8
              Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v129 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
              // pto: %8
              uint64_t v130 = (uint64_t) v26;
              TASSIGN(v129, v130);
              // pto: %83
              pto::Shape<1, 1, 1, 1, 64> v131 = pto::Shape<1, 1, 1, 1, 64>();
              // pto: %83
              pto::Stride<1024, 1024, 1024, 1024, 1> v132 = pto::Stride<1024, 1024, 1024, 1024, 1>();
              // pto: %83
              GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v133 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v3 + ((v28 + v123 * v24) + v125), v131, v132);
              TLOAD(v129, v133);
              // pto: %9
              Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v134 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
              // pto: %9
              uint64_t v135 = (uint64_t) v23;
              TASSIGN(v134, v135);
              // pto: %87
              pto::Shape<1, 1, 1, 1, 64> v136 = pto::Shape<1, 1, 1, 1, 64>();
              // pto: %87
              pto::Stride<1024, 1024, 1024, 1024, 1> v137 = pto::Stride<1024, 1024, 1024, 1024, 1>();
              // pto: %84, %87
              GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v138 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v4 + ((v28 + (v120 < v28 ? v28 : v120) * v24) + v125), v136, v137);
              TLOAD(v134, v138);
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
              // pto: %10
              Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v139 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
              // pto: %10
              uint64_t v140 = (uint64_t) v26;
              TASSIGN(v139, v140);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
              TADD(v139, v129, v134);
            } else {
              // pto: %score_inline2003__phi_v3_mv
              Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v141 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
              // pto: %score_inline2003__phi_v3_mv
              uint64_t v142 = (uint64_t) v26;
              TASSIGN(v141, v142);
              pipe_barrier(PIPE_V);
              TMOV(v141, v32);
              // pto: %value_inline2050__phi_v3_mv
              Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v143 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
              // pto: %value_inline2050__phi_v3_mv
              uint64_t v144 = (uint64_t) v27;
              TASSIGN(v143, v144);
              TMOV(v143, v34);
            }
          } else {
            // pto: %11
            Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v145 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
            // pto: %11
            uint64_t v146 = (uint64_t) v26;
            TASSIGN(v145, v146);
            pipe_barrier(PIPE_V);
            TMOV(v145, v32);
            // pto: %12
            Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v147 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
            // pto: %12
            uint64_t v148 = (uint64_t) v27;
            TASSIGN(v147, v148);
            TMOV(v147, v34);
          }
          // pto: %mi_next_inline2059__tile
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v149 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
          // pto: %mi_next_inline2059__tile
          uint64_t v150 = (uint64_t) v25;
          TASSIGN(v149, v150);
          pipe_barrier(PIPE_V);
          TMAX(v149, v75, v40);
          // pto: %13
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v151 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
          // pto: %13
          uint64_t v152 = (uint64_t) v24;
          TASSIGN(v151, v152);
          pipe_barrier(PIPE_V);
          TSUB(v151, v75, v149);
          // pto: %alpha_inline2027__tile
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v153 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
          // pto: %alpha_inline2027__tile
          uint64_t v154 = (uint64_t) v24;
          TASSIGN(v153, v154);
          pipe_barrier(PIPE_V);
          TEXP(v153, v151);
          // pto: %14
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v155 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
          // pto: %14
          uint64_t v156 = (uint64_t) v26;
          TASSIGN(v155, v156);
          TSUB(v155, v40, v149);
          // pto: %beta_inline1999__tile
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v157 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
          // pto: %beta_inline1999__tile
          uint64_t v158 = (uint64_t) v26;
          TASSIGN(v157, v158);
          pipe_barrier(PIPE_V);
          TEXP(v157, v155);
          // pto: %15
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v159 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
          // pto: %15
          uint64_t v160 = (uint64_t) v23;
          TASSIGN(v159, v160);
          TMUL(v159, v153, v79);
          // pto: %16
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v161 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
          // pto: %16
          uint64_t v162 = (uint64_t) v29;
          TASSIGN(v161, v162);
          pipe_barrier(PIPE_V);
          TADD(v161, v159, v157);
          // pto: %17
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v163 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
          // pto: %17
          uint64_t v164 = (uint64_t) v24;
          TASSIGN(v163, v164);
          TMUL(v163, v81, v153);
          // pto: %18
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v165 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
          // pto: %18
          uint64_t v166 = (uint64_t) v27;
          TASSIGN(v165, v166);
          TMUL(v165, v42, v157);
          // pto: %19
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v167 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
          // pto: %19
          uint64_t v168 = (uint64_t) v28;
          TASSIGN(v167, v168);
          pipe_barrier(PIPE_V);
          TADD(v167, v163, v165);
          // pto: %mi_inline2045__ssa_v3
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v169 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
          // pto: %mi_inline2045__ssa_v3
          uint64_t v170 = (uint64_t) v25;
          TASSIGN(v169, v170);
          // pto: %mi_inline2045__ssa_v3_mv
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v171 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
          // pto: %mi_inline2045__ssa_v3_mv
          uint64_t v172 = (uint64_t) v30;
          TASSIGN(v171, v172);
          TMOV(v171, v169);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        }
        // pto: %20
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v173 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v18);
        // pto: %20
        uint64_t v174 = (uint64_t) v30;
        TASSIGN(v173, v174);
        pipe_barrier(PIPE_V);
        TDIV(v173, v81, v79);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        // pto: %pooled_kv_inline2008__iter_v4_pview
        pto::Shape<1, 1, 1, 1, 64> v175 = pto::Shape<1, 1, 1, 1, 64>();
        // pto: %pooled_kv_inline2008__iter_v4_pview
        pto::Stride<512, 512, 512, 512, 1> v176 = pto::Stride<512, 512, 512, 512, 1>();
        // pto: %pooled_kv_inline2008__iter_v4_pview, %90
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v177 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v2 + ((v28 + v56 * v26) + (j61 < v28 ? v28 : j61)), v175, v176);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TSTORE(v177, v173);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      }
    }
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}