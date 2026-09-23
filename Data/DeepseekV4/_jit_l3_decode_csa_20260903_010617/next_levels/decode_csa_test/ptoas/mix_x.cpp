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

AICORE void mix_x(__gm__ float* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, __gm__ float* v4, int64_t v5, int64_t v6, int64_t v7, int64_t v8, int32_t v9, int32_t v10) {
  SaturationMode v11 = SaturationMode::OFF;
  RoundMode v12 = RoundMode::CAST_RINT;
  const int64_t v13 = 12288;
  const int64_t v14 = 256;
  const int64_t v15 = 16;
  const int64_t v16 = 2;
  const int64_t v17 = 4096;
  const int64_t v18 = 1;
  const int64_t v19 = 8;
  const int64_t v20 = 32768;
  const int64_t v21 = 24576;
  const int64_t v22 = 16384;
  const int64_t v23 = 8192;
  const int64_t v24 = 0;
  const int64_t v25 = 57600;
  const int64_t v26 = 41056;
  const int64_t v27 = 41024;
  const int64_t v28 = 40992;
  const int64_t v29 = 40960;
  const int64_t v30 = 49408;
  const int64_t v31 = 41216;
  const int64_t v32 = 4352;
  const int64_t v33 = 8448;
  const int64_t v34 = 12544;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %x_mixed_inline1253__ssa_v0_view
  int64_t v35 = v7 * v17;
  // pto: %x_mixed_inline1253__ssa_v0_view
  int64_t v36 = v18 * v35;
  // pto: %x_mixed_inline1253__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v37 = pto::Shape<1, 1, 1, -1, -1>(v18, v18, v18, v7, v17);
  // pto: %x_mixed_inline1253__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v38 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v36, v36, v35, v17, v18);
  // pto: %x_mixed_inline1253__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v39 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v37, v38);
  // pto: %x_mixed_tail_store_inline1462__ssa_v0_view
  int64_t v40 = v19 * v17;
  // pto: %x_mixed_tail_store_inline1462__ssa_v0_view
  int64_t v41 = v18 * v40;
  // pto: %x_mixed_tail_store_inline1462__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v42 = pto::Shape<1, 1, 1, -1, -1>(v18, v18, v18, v19, v17);
  // pto: %x_mixed_tail_store_inline1462__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v43 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v41, v41, v40, v17, v18);
  // pto: %x_mixed_tail_store_inline1462__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v44 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v42, v43);
  // pto: %x_flat_inline1497__ssa_v0_view
  int64_t v45 = v8 * v22;
  // pto: %x_flat_inline1497__ssa_v0_view
  int64_t v46 = v18 * v45;
  // pto: %x_flat_inline1497__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v47 = pto::Shape<1, 1, 1, -1, -1>(v18, v18, v18, v8, v22);
  // pto: %x_flat_inline1497__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v48 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v46, v46, v45, v22, v18);
  // pto: %x_flat_inline1497__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v49 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v47, v48);
  // pto: %blk_inline1491__ssa_v0, %19
  int64_t v50 = (int64_t) ((uint64_t) ((int64_t) v9) * (uint64_t) v19);
  // pto: %20
  int64_t v51 = (int64_t) ((uint64_t) v8 - (uint64_t) v50);
  // pto: %21
  int64_t v52 = v51 < v19 ? v51 : v19;
  // pto: %t__tile
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v53 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v19);
  // pto: %t__tile
  uint64_t v54 = (uint64_t) v31;
  TASSIGN(v53, v54);
  // pto: %22
  int64_t v55 = v50 < v24 ? v24 : v50;
  // pto: %pre_val_store_inline1529__ssa_v1_pview
  pto::Shape<1, 1, 1, 8, 8> v56 = pto::Shape<1, 1, 1, 8, 8>();
  // pto: %pre_val_store_inline1529__ssa_v1_pview
  pto::Stride<64, 64, 64, 8, 1> v57 = pto::Stride<64, 64, 64, 8, 1>();
  // pto: %pre_val_store_inline1529__ssa_v1_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 8, 8>, pto::Stride<64, 64, 64, 8, 1>, pto::Layout::ND> v58 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 8>, pto::Stride<64, 64, 64, 8, 1>, pto::Layout::ND>(v1 + (v24 + v55 * v19), v56, v57);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  TLOAD(v53, v58);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  // pto: %transpose_tmp
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v59 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v19);
  // pto: %transpose_tmp
  uint64_t v60 = (uint64_t) v30;
  TASSIGN(v59, v60);
  // pto: %pre_tile_t_inline1443__tile
  Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v61 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v19);
  // pto: %pre_tile_t_inline1443__tile
  uint64_t v62 = (uint64_t) v29;
  TASSIGN(v61, v62);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TTRANS(v61, v53, v59);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  // pto: %0
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v63 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v19);
  // pto: %0
  uint64_t v64 = (uint64_t) v29;
  TASSIGN(v63, v64);
  // pto: %pre0_inline1526__tile
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v65 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v18);
  // pto: %pre0_inline1526__tile
  uint64_t v66 = (uint64_t) v29;
  TASSIGN(v65, v66);
  // pto: %1
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v67 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v19);
  // pto: %1
  uint64_t v68 = (uint64_t) v28;
  TASSIGN(v67, v68);
  // pto: %pre1_inline1442__tile
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v69 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v18);
  // pto: %pre1_inline1442__tile
  uint64_t v70 = (uint64_t) v28;
  TASSIGN(v69, v70);
  // pto: %2
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v71 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v19);
  // pto: %2
  uint64_t v72 = (uint64_t) v27;
  TASSIGN(v71, v72);
  // pto: %pre2_inline1441__tile
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v73 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v18);
  // pto: %pre2_inline1441__tile
  uint64_t v74 = (uint64_t) v27;
  TASSIGN(v73, v74);
  // pto: %3
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v75 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v19);
  // pto: %3
  uint64_t v76 = (uint64_t) v26;
  TASSIGN(v75, v76);
  // pto: %pre3_inline1440__tile
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v77 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v19, v18);
  // pto: %pre3_inline1440__tile
  uint64_t v78 = (uint64_t) v26;
  TASSIGN(v77, v78);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  for (int64_t i79 = v24; i79 < v15; i79 += v16) {
    // pto: %26
    int64_t v80 = (int64_t) ((uint64_t) i79 * (uint64_t) v14);
    // pto: %28
    int64_t v81 = (int64_t) ((uint64_t) v80 + (uint64_t) v14);
    // pto: %x0_inline1439__tile
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v82 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %x0_inline1439__tile
    uint64_t v83 = (uint64_t) v31;
    TASSIGN(v82, v83);
    // pto: %30
    int64_t v84 = v80 < v24 ? v24 : v80;
    // pto: %x_flat_inline1497__ssa_v0_pview
    __gm__ float* v85 = PTOAS__GLOBAL_TENSOR_DATA(v49);
    // pto: %x_flat_inline1497__ssa_v0_pview
    int64_t v86 = v52 * v22;
    // pto: %x_flat_inline1497__ssa_v0_pview
    int64_t v87 = v18 * v86;
    // pto: %x_flat_inline1497__ssa_v0_pview
    pto::Shape<1, 1, 1, -1, 256> v88 = pto::Shape<1, 1, 1, -1, 256>(v18, v18, v18, v52, v14);
    // pto: %x_flat_inline1497__ssa_v0_pview
    pto::Stride<-1, -1, -1, -1, -1> v89 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v87, v87, v86, v22, v18);
    // pto: %x_flat_inline1497__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v90 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v85 + ((v24 + v55 * v22) + v84 * v18), v88, v89);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    TLOAD(v82, v90);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %x1_inline1446__tile
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v91 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %x1_inline1446__tile
    uint64_t v92 = (uint64_t) v30;
    TASSIGN(v91, v92);
    // pto: %32
    int64_t v93 = (int64_t) ((uint64_t) v80 + (uint64_t) v17);
    // pto: %34
    __gm__ float* v94 = PTOAS__GLOBAL_TENSOR_DATA(v49);
    // pto: %34
    int64_t v95 = v52 * v22;
    // pto: %34
    int64_t v96 = v18 * v95;
    // pto: %34
    pto::Shape<1, 1, 1, -1, 256> v97 = pto::Shape<1, 1, 1, -1, 256>(v18, v18, v18, v52, v14);
    // pto: %34
    pto::Stride<-1, -1, -1, -1, -1> v98 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v96, v96, v95, v22, v18);
    // pto: %34, %33
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v99 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v94 + ((v24 + v55 * v22) + (v93 < v24 ? v24 : v93) * v18), v97, v98);
    TLOAD(v91, v99);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    // pto: %x2_inline1542__tile
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v100 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %x2_inline1542__tile
    uint64_t v101 = (uint64_t) v25;
    TASSIGN(v100, v101);
    // pto: %36
    int64_t v102 = (int64_t) ((uint64_t) v80 + (uint64_t) v23);
    // pto: %38
    __gm__ float* v103 = PTOAS__GLOBAL_TENSOR_DATA(v49);
    // pto: %38
    int64_t v104 = v52 * v22;
    // pto: %38
    int64_t v105 = v18 * v104;
    // pto: %38
    pto::Shape<1, 1, 1, -1, 256> v106 = pto::Shape<1, 1, 1, -1, 256>(v18, v18, v18, v52, v14);
    // pto: %38
    pto::Stride<-1, -1, -1, -1, -1> v107 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v105, v105, v104, v22, v18);
    // pto: %38, %37
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v108 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v103 + ((v24 + v55 * v22) + (v102 < v24 ? v24 : v102) * v18), v106, v107);
    TLOAD(v100, v108);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    // pto: %x3_inline1438__tile
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v109 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %x3_inline1438__tile
    uint64_t v110 = (uint64_t) v24;
    TASSIGN(v109, v110);
    // pto: %40
    int64_t v111 = (int64_t) ((uint64_t) v80 + (uint64_t) v13);
    // pto: %42
    __gm__ float* v112 = PTOAS__GLOBAL_TENSOR_DATA(v49);
    // pto: %42
    int64_t v113 = v52 * v22;
    // pto: %42
    int64_t v114 = v18 * v113;
    // pto: %42
    pto::Shape<1, 1, 1, -1, 256> v115 = pto::Shape<1, 1, 1, -1, 256>(v18, v18, v18, v52, v14);
    // pto: %42
    pto::Stride<-1, -1, -1, -1, -1> v116 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v114, v114, v113, v22, v18);
    // pto: %42, %41
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v117 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v112 + ((v24 + v55 * v22) + (v111 < v24 ? v24 : v111) * v18), v115, v116);
    TLOAD(v109, v117);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    // pto: %4
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v118 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %4
    uint64_t v119 = (uint64_t) v23;
    TASSIGN(v118, v119);
    // pto: %44
    int64_t v120 = v81 < v24 ? v24 : v81;
    // pto: %45
    __gm__ float* v121 = PTOAS__GLOBAL_TENSOR_DATA(v49);
    // pto: %45
    int64_t v122 = v52 * v22;
    // pto: %45
    int64_t v123 = v18 * v122;
    // pto: %45
    pto::Shape<1, 1, 1, -1, 256> v124 = pto::Shape<1, 1, 1, -1, 256>(v18, v18, v18, v52, v14);
    // pto: %45
    pto::Stride<-1, -1, -1, -1, -1> v125 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v123, v123, v122, v22, v18);
    // pto: %45
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v126 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v121 + ((v24 + v55 * v22) + v120 * v18), v124, v125);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    TLOAD(v118, v126);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    // pto: %5
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v127 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %5
    uint64_t v128 = (uint64_t) v22;
    TASSIGN(v127, v128);
    int64_t v129 = (int64_t) ((uint64_t) v80 + (uint64_t) v32);
    // pto: %49
    __gm__ float* v130 = PTOAS__GLOBAL_TENSOR_DATA(v49);
    // pto: %49
    int64_t v131 = v52 * v22;
    // pto: %49
    int64_t v132 = v18 * v131;
    // pto: %49
    pto::Shape<1, 1, 1, -1, 256> v133 = pto::Shape<1, 1, 1, -1, 256>(v18, v18, v18, v52, v14);
    // pto: %49
    pto::Stride<-1, -1, -1, -1, -1> v134 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v132, v132, v131, v22, v18);
    // pto: %49, %48
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v135 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v130 + ((v24 + v55 * v22) + (v129 < v24 ? v24 : v129) * v18), v133, v134);
    TLOAD(v127, v135);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
    // pto: %6
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v136 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %6
    uint64_t v137 = (uint64_t) v21;
    TASSIGN(v136, v137);
    int64_t v138 = (int64_t) ((uint64_t) v80 + (uint64_t) v33);
    // pto: %53
    __gm__ float* v139 = PTOAS__GLOBAL_TENSOR_DATA(v49);
    // pto: %53
    int64_t v140 = v52 * v22;
    // pto: %53
    int64_t v141 = v18 * v140;
    // pto: %53
    pto::Shape<1, 1, 1, -1, 256> v142 = pto::Shape<1, 1, 1, -1, 256>(v18, v18, v18, v52, v14);
    // pto: %53
    pto::Stride<-1, -1, -1, -1, -1> v143 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v141, v141, v140, v22, v18);
    // pto: %53, %52
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v144 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v139 + ((v24 + v55 * v22) + (v138 < v24 ? v24 : v138) * v18), v142, v143);
    TLOAD(v136, v144);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    // pto: %7
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v145 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %7
    uint64_t v146 = (uint64_t) v20;
    TASSIGN(v145, v146);
    int64_t v147 = (int64_t) ((uint64_t) v80 + (uint64_t) v34);
    // pto: %57
    __gm__ float* v148 = PTOAS__GLOBAL_TENSOR_DATA(v49);
    // pto: %57
    int64_t v149 = v52 * v22;
    // pto: %57
    int64_t v150 = v18 * v149;
    // pto: %57
    pto::Shape<1, 1, 1, -1, 256> v151 = pto::Shape<1, 1, 1, -1, 256>(v18, v18, v18, v52, v14);
    // pto: %57
    pto::Stride<-1, -1, -1, -1, -1> v152 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v150, v150, v149, v22, v18);
    // pto: %57, %56
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v153 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v148 + ((v24 + v55 * v22) + (v147 < v24 ? v24 : v147) * v18), v151, v152);
    TLOAD(v145, v153);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %y0_inline1437__tile
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v154 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %y0_inline1437__tile
    uint64_t v155 = (uint64_t) v31;
    TASSIGN(v154, v155);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TROWEXPANDMUL(v154, v82, v65);
    // pto: %y1_inline1550__tile
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v156 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %y1_inline1550__tile
    uint64_t v157 = (uint64_t) v30;
    TASSIGN(v156, v157);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    TROWEXPANDMUL(v156, v91, v69);
    // pto: %y2_inline1436__tile
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v158 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %y2_inline1436__tile
    uint64_t v159 = (uint64_t) v25;
    TASSIGN(v158, v159);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    TROWEXPANDMUL(v158, v100, v73);
    // pto: %y3_inline1435__tile
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v160 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %y3_inline1435__tile
    uint64_t v161 = (uint64_t) v24;
    TASSIGN(v160, v161);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    TROWEXPANDMUL(v160, v109, v77);
    // pto: %8
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v162 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %8
    uint64_t v163 = (uint64_t) v31;
    TASSIGN(v162, v163);
    pipe_barrier(PIPE_V);
    TADD(v162, v154, v156);
    // pto: %9
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v164 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %9
    uint64_t v165 = (uint64_t) v30;
    TASSIGN(v164, v165);
    pipe_barrier(PIPE_V);
    TADD(v164, v158, v160);
    // pto: %y_tile_inline1434__tile
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v166 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %y_tile_inline1434__tile
    uint64_t v167 = (uint64_t) v31;
    TASSIGN(v166, v167);
    pipe_barrier(PIPE_V);
    TADD(v166, v162, v164);
    // pto: %y_bf16_inline1433__tile
    Tile<TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v168 = Tile<TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %y_bf16_inline1433__tile
    uint64_t v169 = (uint64_t) v31;
    TASSIGN(v168, v169);
    pipe_barrier(PIPE_V);
    TCVT(v168, v166, v12, v11);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %58
    bool v170 = v52 == v19;
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    if (v170) {
      // pto: %x_mixed_inline1253__iter_v1_pview
      __gm__ bfloat16_t* v171 = PTOAS__GLOBAL_TENSOR_DATA(v39);
      // pto: %x_mixed_inline1253__iter_v1_pview
      int64_t v172 = v52 * v17;
      // pto: %x_mixed_inline1253__iter_v1_pview
      int64_t v173 = v18 * v172;
      // pto: %x_mixed_inline1253__iter_v1_pview
      pto::Shape<1, 1, 1, -1, 256> v174 = pto::Shape<1, 1, 1, -1, 256>(v18, v18, v18, v52, v14);
      // pto: %x_mixed_inline1253__iter_v1_pview
      pto::Stride<-1, -1, -1, -1, -1> v175 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v173, v173, v172, v17, v18);
      // pto: %x_mixed_inline1253__iter_v1_pview
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v176 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v171 + ((v24 + v55 * v17) + v84 * v18), v174, v175);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v176, v168);
    } else {
      // pto: %x_mixed_tail_store_inline1462__iter_v1_pview
      __gm__ bfloat16_t* v177 = PTOAS__GLOBAL_TENSOR_DATA(v44);
      // pto: %x_mixed_tail_store_inline1462__iter_v1_pview
      int64_t v178 = v52 * v17;
      // pto: %x_mixed_tail_store_inline1462__iter_v1_pview
      int64_t v179 = v18 * v178;
      // pto: %x_mixed_tail_store_inline1462__iter_v1_pview
      pto::Shape<1, 1, 1, -1, 256> v180 = pto::Shape<1, 1, 1, -1, 256>(v18, v18, v18, v52, v14);
      // pto: %x_mixed_tail_store_inline1462__iter_v1_pview
      pto::Stride<-1, -1, -1, -1, -1> v181 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v179, v179, v178, v17, v18);
      // pto: %x_mixed_tail_store_inline1462__iter_v1_pview
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v182 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v177 + ((v24 + v24 * v17) + v84 * v18), v180, v181);
      TSTORE(v182, v168);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      // pto: %y_out_inline1521__ssa_v0
      Tile<TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v183 = Tile<TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
      // pto: %y_out_inline1521__ssa_v0
      uint64_t v184 = (uint64_t) v31;
      TASSIGN(v183, v184);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      TLOAD(v183, v182);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      // pto: %65
      __gm__ bfloat16_t* v185 = PTOAS__GLOBAL_TENSOR_DATA(v39);
      // pto: %65
      int64_t v186 = v52 * v17;
      // pto: %65
      int64_t v187 = v18 * v186;
      // pto: %65
      pto::Shape<1, 1, 1, -1, 256> v188 = pto::Shape<1, 1, 1, -1, 256>(v18, v18, v18, v52, v14);
      // pto: %65
      pto::Stride<-1, -1, -1, -1, -1> v189 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v187, v187, v186, v17, v18);
      // pto: %65
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v190 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v185 + ((v24 + v55 * v17) + v84 * v18), v188, v189);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v190, v183);
    }
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    // pto: %10
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v191 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %10
    uint64_t v192 = (uint64_t) v23;
    TASSIGN(v191, v192);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    TROWEXPANDMUL(v191, v118, v65);
    // pto: %11
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v193 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %11
    uint64_t v194 = (uint64_t) v22;
    TASSIGN(v193, v194);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
    TROWEXPANDMUL(v193, v127, v69);
    // pto: %12
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v195 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %12
    uint64_t v196 = (uint64_t) v21;
    TASSIGN(v195, v196);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    TROWEXPANDMUL(v195, v136, v73);
    // pto: %13
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v197 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %13
    uint64_t v198 = (uint64_t) v20;
    TASSIGN(v197, v198);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWEXPANDMUL(v197, v145, v77);
    // pto: %14
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v199 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %14
    uint64_t v200 = (uint64_t) v23;
    TASSIGN(v199, v200);
    pipe_barrier(PIPE_V);
    TADD(v199, v191, v193);
    // pto: %15
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v201 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %15
    uint64_t v202 = (uint64_t) v22;
    TASSIGN(v201, v202);
    pipe_barrier(PIPE_V);
    TADD(v201, v195, v197);
    // pto: %16
    Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v203 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %16
    uint64_t v204 = (uint64_t) v23;
    TASSIGN(v203, v204);
    pipe_barrier(PIPE_V);
    TADD(v203, v199, v201);
    // pto: %17
    Tile<TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v205 = Tile<TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
    // pto: %17
    uint64_t v206 = (uint64_t) v23;
    TASSIGN(v205, v206);
    pipe_barrier(PIPE_V);
    TCVT(v205, v203, v12, v11);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    if (v170) {
      // pto: %x_mixed_inline1253__phi_v4_pview
      __gm__ bfloat16_t* v207 = PTOAS__GLOBAL_TENSOR_DATA(v39);
      // pto: %x_mixed_inline1253__phi_v4_pview
      int64_t v208 = v52 * v17;
      // pto: %x_mixed_inline1253__phi_v4_pview
      int64_t v209 = v18 * v208;
      // pto: %x_mixed_inline1253__phi_v4_pview
      pto::Shape<1, 1, 1, -1, 256> v210 = pto::Shape<1, 1, 1, -1, 256>(v18, v18, v18, v52, v14);
      // pto: %x_mixed_inline1253__phi_v4_pview
      pto::Stride<-1, -1, -1, -1, -1> v211 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v209, v209, v208, v17, v18);
      // pto: %x_mixed_inline1253__phi_v4_pview
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v212 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v207 + ((v24 + v55 * v17) + v120 * v18), v210, v211);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v212, v205);
    } else {
      // pto: %x_mixed_tail_store_inline1462__phi_v4_pview
      __gm__ bfloat16_t* v213 = PTOAS__GLOBAL_TENSOR_DATA(v44);
      // pto: %x_mixed_tail_store_inline1462__phi_v4_pview
      int64_t v214 = v52 * v17;
      // pto: %x_mixed_tail_store_inline1462__phi_v4_pview
      int64_t v215 = v18 * v214;
      // pto: %x_mixed_tail_store_inline1462__phi_v4_pview
      pto::Shape<1, 1, 1, -1, 256> v216 = pto::Shape<1, 1, 1, -1, 256>(v18, v18, v18, v52, v14);
      // pto: %x_mixed_tail_store_inline1462__phi_v4_pview
      pto::Stride<-1, -1, -1, -1, -1> v217 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v215, v215, v214, v17, v18);
      // pto: %x_mixed_tail_store_inline1462__phi_v4_pview
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v218 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v213 + ((v24 + v24 * v17) + v120 * v18), v216, v217);
      TSTORE(v218, v205);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      // pto: %18
      Tile<TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v219 = Tile<TileType::Vec, bfloat16_t, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v14);
      // pto: %18
      uint64_t v220 = (uint64_t) v23;
      TASSIGN(v219, v220);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      TLOAD(v219, v218);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      // pto: %76
      __gm__ bfloat16_t* v221 = PTOAS__GLOBAL_TENSOR_DATA(v39);
      // pto: %76
      int64_t v222 = v52 * v17;
      // pto: %76
      int64_t v223 = v18 * v222;
      // pto: %76
      pto::Shape<1, 1, 1, -1, 256> v224 = pto::Shape<1, 1, 1, -1, 256>(v18, v18, v18, v52, v14);
      // pto: %76
      pto::Stride<-1, -1, -1, -1, -1> v225 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v223, v223, v222, v17, v18);
      // pto: %76
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v226 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 256>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v221 + ((v24 + v55 * v17) + v120 * v18), v224, v225);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
      TSTORE(v226, v219);
    }
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}