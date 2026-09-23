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

AICORE void tp_o_a(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, int64_t v4, int64_t v5, int32_t v6, int32_t v7) {
  const int64_t v8 = 512;
  const int64_t v9 = 3840;
  const int64_t v10 = 128;
  const int64_t v11 = 8;
  const int64_t v12 = 256;
  const int64_t v13 = 1;
  const int64_t v14 = 4096;
  const int64_t v15 = 2048;
  const int64_t v16 = 65536;
  const int64_t v17 = 32768;
  const int64_t v18 = 0;
  const int64_t v19 = 196608;
  const int64_t v20 = 131072;
  using T = float;

  #if defined(__DAV_CUBE__)
  // pto: %attn_2d_inline2548__ssa_v0_view
  int64_t v21 = v15 * v14;
  // pto: %attn_2d_inline2548__ssa_v0_view
  int64_t v22 = v13 * v21;
  // pto: %attn_2d_inline2548__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v23 = pto::Shape<1, 1, 1, -1, -1>(v13, v13, v13, v15, v14);
  // pto: %attn_2d_inline2548__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v24 = pto::Stride<-1, -1, -1, -1, -1>(v13 * v22, v22, v21, v14, v13);
  // pto: %attn_2d_inline2548__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v25 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v23, v24);
  // pto: %own_a_fp32_inline2500__iter_v1_view
  int64_t v26 = v12 * v14;
  // pto: %own_a_fp32_inline2500__iter_v1_view
  int64_t v27 = v13 * v26;
  // pto: %own_a_fp32_inline2500__iter_v1_view
  pto::Shape<1, 1, 1, -1, -1> v28 = pto::Shape<1, 1, 1, -1, -1>(v13, v13, v13, v12, v14);
  // pto: %own_a_fp32_inline2500__iter_v1_view
  pto::Stride<-1, -1, -1, -1, -1> v29 = pto::Stride<-1, -1, -1, -1, -1>(v13 * v27, v27, v26, v14, v13);
  // pto: %own_a_fp32_inline2500__iter_v1_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v30 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v28, v29);
  // pto: %pa_unit_inline2493__ssa_v0
  int64_t v31 = (int64_t) v6;
  // pto: %27
  int64_t v32 = v31 / v11;
  // pto: %30
  int64_t v33 = (int64_t) ((uint64_t) v32 * (uint64_t) v10);
  // pto: %32
  int64_t v34 = (int64_t) ((uint64_t) v12 - (uint64_t) v33);
  // pto: %33
  int64_t v35 = v34 < v10 ? v34 : v10;
  // pto: %34
  int64_t v36 = (int64_t) ((uint64_t) v4 + (uint64_t) v33);
  // pto: %35, %29, %28, %31
  int64_t v37 = (int64_t) ((uint64_t) v5 + (uint64_t) ((int64_t) ((uint64_t) ((int64_t) ((uint64_t) v31 - (uint64_t) ((int64_t) ((uint64_t) v32 * (uint64_t) v11)))) * (uint64_t) v10)));
  // pto: %pa_x0_inline2535__tile
  Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v38 = Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v35, v12);
  // pto: %pa_x0_inline2535__tile
  uint64_t v39 = (uint64_t) v20;
  TASSIGN(v38, v39);
  // pto: %36
  int64_t v40 = v36 < v18 ? v18 : v36;
  // pto: %attn_2d_inline2548__ssa_v0_pview
  __gm__ bfloat16_t* v41 = PTOAS__GLOBAL_TENSOR_DATA(v25);
  // pto: %attn_2d_inline2548__ssa_v0_pview
  int64_t v42 = v35 * v14;
  // pto: %attn_2d_inline2548__ssa_v0_pview
  int64_t v43 = v13 * v42;
  // pto: %attn_2d_inline2548__ssa_v0_pview
  pto::Shape<1, 1, 1, -1, 256> v44 = pto::Shape<1, 1, 1, -1, 256>(v13, v13, v13, v35, v12);
  // pto: %attn_2d_inline2548__ssa_v0_pview
  pto::Stride<-1, -1, -1, -1, -1> v45 = pto::Stride<-1, -1, -1, -1, -1>(v13 * v43, v43, v42, v14, v13);
  // pto: %attn_2d_inline2548__ssa_v0_pview
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v46 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v41 + ((v18 + v40 * v14) + v18 * v13), v44, v45);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  TLOAD(v38, v46);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  // pto: %pa_w0_inline2523__tile
  Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v47 = Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v10, v12);
  // pto: %pa_w0_inline2523__tile
  uint64_t v48 = (uint64_t) v19;
  TASSIGN(v47, v48);
  // pto: %37
  int64_t v49 = v37 < v18 ? v18 : v37;
  // pto: %wo_a_flat_inline2521__ssa_v0_pview
  pto::Shape<1, 1, 1, 128, 256> v50 = pto::Shape<1, 1, 1, 128, 256>();
  // pto: %wo_a_flat_inline2521__ssa_v0_pview
  pto::Stride<524288, 524288, 524288, 4096, 1> v51 = pto::Stride<524288, 524288, 524288, 4096, 1>();
  // pto: %wo_a_flat_inline2521__ssa_v0_pview
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>, pto::Layout::ND> v52 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>, pto::Layout::ND>(v2 + (v18 + v49 * v14), v50, v51);
  TLOAD(v47, v52);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  // pto: %pa_w0_inline2523__tile_t
  Tile<TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v53 = Tile<TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v12, v10);
  // pto: %pa_w0_inline2523__tile_t
  uint64_t v54 = (uint64_t) v19;
  TASSIGN(v53, v54);
  // pto: %pa_acc_inline2537__tile_l0_init_storage
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal> v55 = Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal>(v10, v10);
  // pto: %pa_acc_inline2537__tile_l0_init_storage
  uint64_t v56 = (uint64_t) v18;
  TASSIGN(v55, v56);
  v55.SetValidShape(v35, v10);
  // pto: %pa_acc_inline2537__tile_l0_a
  Tile<TileType::Left, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v57 = Tile<TileType::Left, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v35, v10);
  // pto: %pa_acc_inline2537__tile_l0_a
  uint64_t v58 = (uint64_t) v18;
  TASSIGN(v57, v58);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TEXTRACT(v57, v38, v18, v18);
  // pto: %pa_acc_inline2537__tile_l0_b
  Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v59 = Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v10, v10);
  // pto: %pa_acc_inline2537__tile_l0_b
  uint64_t v60 = (uint64_t) v18;
  TASSIGN(v59, v60);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  TEXTRACT(v59, v53, v18, v18);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  // pto: %0
  Tile<TileType::Left, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v61 = Tile<TileType::Left, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v35, v10);
  // pto: %0
  uint64_t v62 = (uint64_t) v17;
  TASSIGN(v61, v62);
  TEXTRACT(v61, v38, v18, v10);
  // pto: %1
  Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v63 = Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v10, v10);
  // pto: %1
  uint64_t v64 = (uint64_t) v17;
  TASSIGN(v63, v64);
  TEXTRACT(v63, v53, v10, v18);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v55, v57, v59);
  pipe_barrier(PIPE_M);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
  TMATMUL_ACC(v55, v55, v61, v63);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (int64_t i65 = v12; i65 < v9; i65 += v8) {
    // pto: %pa_xk_inline2539__tile
    Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v66 = Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v35, v12);
    // pto: %pa_xk_inline2539__tile
    uint64_t v67 = (uint64_t) v20;
    TASSIGN(v66, v67);
    // pto: %39
    int64_t v68 = i65 < v18 ? v18 : i65;
    // pto: %40
    __gm__ bfloat16_t* v69 = PTOAS__GLOBAL_TENSOR_DATA(v25);
    // pto: %40
    int64_t v70 = v35 * v14;
    // pto: %40
    int64_t v71 = v13 * v70;
    // pto: %40
    pto::Shape<1, 1, 1, -1, 256> v72 = pto::Shape<1, 1, 1, -1, 256>(v13, v13, v13, v35, v12);
    // pto: %40
    pto::Stride<-1, -1, -1, -1, -1> v73 = pto::Stride<-1, -1, -1, -1, -1>(v13 * v71, v71, v70, v14, v13);
    // pto: %40
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v74 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v69 + ((v18 + v40 * v14) + v68 * v13), v72, v73);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    TLOAD(v66, v74);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
    // pto: %pa_wk_inline2540__tile
    Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v75 = Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %pa_wk_inline2540__tile
    uint64_t v76 = (uint64_t) v19;
    TASSIGN(v75, v76);
    // pto: %43
    pto::Shape<1, 1, 1, 128, 256> v77 = pto::Shape<1, 1, 1, 128, 256>();
    // pto: %43
    pto::Stride<524288, 524288, 524288, 4096, 1> v78 = pto::Stride<524288, 524288, 524288, 4096, 1>();
    // pto: %43
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>, pto::Layout::ND> v79 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>, pto::Layout::ND>(v2 + ((v18 + v49 * v14) + v68), v77, v78);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    TLOAD(v75, v79);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
    // pto: %3
    Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v80 = Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v35, v12);
    // pto: %3
    uint64_t v81 = (uint64_t) v18;
    TASSIGN(v80, v81);
    // pto: %45
    int64_t v82 = (int64_t) ((uint64_t) i65 + (uint64_t) v12);
    // pto: %46
    int64_t v83 = v82 < v18 ? v18 : v82;
    // pto: %47
    __gm__ bfloat16_t* v84 = PTOAS__GLOBAL_TENSOR_DATA(v25);
    // pto: %47
    int64_t v85 = v35 * v14;
    // pto: %47
    int64_t v86 = v13 * v85;
    // pto: %47
    pto::Shape<1, 1, 1, -1, 256> v87 = pto::Shape<1, 1, 1, -1, 256>(v13, v13, v13, v35, v12);
    // pto: %47
    pto::Stride<-1, -1, -1, -1, -1> v88 = pto::Stride<-1, -1, -1, -1, -1>(v13 * v86, v86, v85, v14, v13);
    // pto: %47
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v89 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v84 + ((v18 + v40 * v14) + v83 * v13), v87, v88);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    TLOAD(v80, v89);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
    // pto: %4
    Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v90 = Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v10, v12);
    // pto: %4
    uint64_t v91 = (uint64_t) v16;
    TASSIGN(v90, v91);
    // pto: %51
    pto::Shape<1, 1, 1, 128, 256> v92 = pto::Shape<1, 1, 1, 128, 256>();
    // pto: %51
    pto::Stride<524288, 524288, 524288, 4096, 1> v93 = pto::Stride<524288, 524288, 524288, 4096, 1>();
    // pto: %51
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>, pto::Layout::ND> v94 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>, pto::Layout::ND>(v2 + ((v18 + v49 * v14) + v83), v92, v93);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
    TLOAD(v90, v94);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID5);
    // pto: %pa_wk_inline2540__tile_t
    Tile<TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v95 = Tile<TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v12, v10);
    // pto: %pa_wk_inline2540__tile_t
    uint64_t v96 = (uint64_t) v19;
    TASSIGN(v95, v96);
    // pto: %5
    Tile<TileType::Left, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v97 = Tile<TileType::Left, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v35, v10);
    // pto: %5
    uint64_t v98 = (uint64_t) v18;
    TASSIGN(v97, v98);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    TEXTRACT(v97, v66, v18, v18);
    // pto: %6
    Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v99 = Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v10, v10);
    // pto: %6
    uint64_t v100 = (uint64_t) v18;
    TASSIGN(v99, v100);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
    TEXTRACT(v99, v95, v18, v18);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
    // pto: %7
    Tile<TileType::Left, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v101 = Tile<TileType::Left, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v35, v10);
    // pto: %7
    uint64_t v102 = (uint64_t) v17;
    TASSIGN(v101, v102);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    TEXTRACT(v101, v66, v18, v10);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    // pto: %8
    Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v103 = Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v10, v10);
    // pto: %8
    uint64_t v104 = (uint64_t) v17;
    TASSIGN(v103, v104);
    TEXTRACT(v103, v95, v10, v18);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
    pipe_barrier(PIPE_M);
    TMATMUL_ACC(v55, v55, v97, v99);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    pipe_barrier(PIPE_M);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    TMATMUL_ACC(v55, v55, v101, v103);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    // pto: %11
    Tile<TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v105 = Tile<TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v12, v10);
    // pto: %11
    uint64_t v106 = (uint64_t) v16;
    TASSIGN(v105, v106);
    // pto: %12
    Tile<TileType::Left, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v107 = Tile<TileType::Left, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v35, v10);
    // pto: %12
    uint64_t v108 = (uint64_t) v18;
    TASSIGN(v107, v108);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
    TEXTRACT(v107, v80, v18, v18);
    // pto: %13
    Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v109 = Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v10, v10);
    // pto: %13
    uint64_t v110 = (uint64_t) v18;
    TASSIGN(v109, v110);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID5);
    TEXTRACT(v109, v105, v18, v18);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
    // pto: %14
    Tile<TileType::Left, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v111 = Tile<TileType::Left, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v35, v10);
    // pto: %14
    uint64_t v112 = (uint64_t) v17;
    TASSIGN(v111, v112);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    TEXTRACT(v111, v80, v18, v10);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    // pto: %15
    Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v113 = Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v10, v10);
    // pto: %15
    uint64_t v114 = (uint64_t) v17;
    TASSIGN(v113, v114);
    TEXTRACT(v113, v105, v10, v18);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
    pipe_barrier(PIPE_M);
    TMATMUL_ACC(v55, v55, v107, v109);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    pipe_barrier(PIPE_M);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
    TMATMUL_ACC(v55, v55, v111, v113);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  }
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
  // pto: %18
  Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v115 = Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v35, v12);
  // pto: %18
  uint64_t v116 = (uint64_t) v20;
  TASSIGN(v115, v116);
  // pto: %53
  __gm__ bfloat16_t* v117 = PTOAS__GLOBAL_TENSOR_DATA(v25);
  // pto: %53
  int64_t v118 = v35 * v14;
  // pto: %53
  int64_t v119 = v13 * v118;
  // pto: %53
  pto::Shape<1, 1, 1, -1, 256> v120 = pto::Shape<1, 1, 1, -1, 256>(v13, v13, v13, v35, v12);
  // pto: %53
  pto::Stride<-1, -1, -1, -1, -1> v121 = pto::Stride<-1, -1, -1, -1, -1>(v13 * v119, v119, v118, v14, v13);
  // pto: %53
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v122 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v117 + ((v18 + v40 * v14) + v9 * v13), v120, v121);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
  TLOAD(v115, v122);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID6);
  // pto: %19
  Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v123 = Tile<TileType::Mat, bfloat16_t, 128, 256, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v10, v12);
  // pto: %19
  uint64_t v124 = (uint64_t) v19;
  TASSIGN(v123, v124);
  // pto: %55
  pto::Shape<1, 1, 1, 128, 256> v125 = pto::Shape<1, 1, 1, 128, 256>();
  // pto: %55
  pto::Stride<524288, 524288, 524288, 4096, 1> v126 = pto::Stride<524288, 524288, 524288, 4096, 1>();
  // pto: %55
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>, pto::Layout::ND> v127 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<524288, 524288, 524288, 4096, 1>, pto::Layout::ND>(v2 + (v9 + v49 * v14), v125, v126);
  TLOAD(v123, v127);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
  // pto: %20
  Tile<TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v128 = Tile<TileType::Mat, bfloat16_t, 256, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v12, v10);
  // pto: %20
  uint64_t v129 = (uint64_t) v19;
  TASSIGN(v128, v129);
  // pto: %21
  Tile<TileType::Left, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v130 = Tile<TileType::Left, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v35, v10);
  // pto: %21
  uint64_t v131 = (uint64_t) v18;
  TASSIGN(v130, v131);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID6);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
  TEXTRACT(v130, v115, v18, v18);
  // pto: %22
  Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v132 = Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v10, v10);
  // pto: %22
  uint64_t v133 = (uint64_t) v18;
  TASSIGN(v132, v133);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
  TEXTRACT(v132, v128, v18, v18);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
  // pto: %23
  Tile<TileType::Left, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v134 = Tile<TileType::Left, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v35, v10);
  // pto: %23
  uint64_t v135 = (uint64_t) v17;
  TASSIGN(v134, v135);
  TEXTRACT(v134, v115, v18, v10);
  // pto: %24
  Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v136 = Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v10, v10);
  // pto: %24
  uint64_t v137 = (uint64_t) v17;
  TASSIGN(v136, v137);
  TEXTRACT(v136, v128, v10, v18);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
  pipe_barrier(PIPE_M);
  TMATMUL_ACC(v55, v55, v130, v132);
  pipe_barrier(PIPE_M);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
  TMATMUL_ACC(v55, v55, v134, v136);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  // pto: %pa_acc_inline2537__rv_v2
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal> v138 = Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal>(v35, v10);
  // pto: %pa_acc_inline2537__rv_v2
  uint64_t v139 = (uint64_t) v18;
  TASSIGN(v138, v139);
  v138.SetValidShape(v35, v10);
  // pto: %own_a_fp32_inline2500__iter_v1_pview
  __gm__ float* v140 = PTOAS__GLOBAL_TENSOR_DATA(v30);
  // pto: %own_a_fp32_inline2500__iter_v1_pview
  int64_t v141 = v35 * v14;
  // pto: %own_a_fp32_inline2500__iter_v1_pview
  int64_t v142 = v13 * v141;
  // pto: %own_a_fp32_inline2500__iter_v1_pview
  pto::Shape<1, 1, 1, -1, 128> v143 = pto::Shape<1, 1, 1, -1, 128>(v13, v13, v13, v35, v10);
  // pto: %own_a_fp32_inline2500__iter_v1_pview
  pto::Stride<-1, -1, -1, -1, -1> v144 = pto::Stride<-1, -1, -1, -1, -1>(v13 * v142, v142, v141, v14, v13);
  // pto: %56, %own_a_fp32_inline2500__iter_v1_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, 128>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v145 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 128>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v140 + ((v18 + (v33 < v18 ? v18 : v33) * v14) + v49 * v13), v143, v144);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v145, v138);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}