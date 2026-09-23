#include "pto/pto-inst.hpp"
using namespace pto;

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

AICORE void rms_recip(__gm__ float* v1, __gm__ float* v2) {
  const float v3 = 9.99999997E-7f;
  const float v4 = 1.95312503E-4f;
  const int64_t v5 = 768;
  const int64_t v6 = 512;
  const int64_t v7 = 256;
  const int64_t v8 = 4;
  const int64_t v9 = 20;
  const float v10 = 0.0f;
  const int64_t v11 = 1;
  const int64_t v12 = 5120;
  const int64_t v13 = 16;
  const int64_t v14 = 81984;
  const int64_t v15 = 65600;
  const int64_t v16 = 49216;
  const int64_t v17 = 32832;
  const int64_t v18 = 16448;
  const int64_t v19 = 64;
  const int64_t v20 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v21 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
  uint64_t v22 = (uint64_t) v20;
  TASSIGN(v21, v22);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  TEXPANDS(v21, v10);
  for (size_t v23 = (size_t) v20; v23 < ((size_t) v9); v23 += (size_t) v8) {
    int64_t v24 = (int64_t) ((uint64_t) ((int64_t) v23) * (uint64_t) v7);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v25 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v7);
    uint64_t v26 = (uint64_t) v19;
    TASSIGN(v25, v26);
    pto::Shape<1, 1, 1, 16, 256> v27 = pto::Shape<1, 1, 1, 16, 256>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v28 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v29 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v1 + (v20 + v20 * v12 + v24 * v11), v27, v28);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v25, v29);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v30 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v7);
    uint64_t v31 = (uint64_t) v18;
    TASSIGN(v30, v31);
    pto::Shape<1, 1, 1, 16, 256> v32 = pto::Shape<1, 1, 1, 16, 256>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v33 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v34 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v1 + (v20 + v20 * v12 + (int64_t) ((uint64_t) v24 + (uint64_t) v7) * v11), v32, v33);
    TLOAD(v30, v34);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v35 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v7);
    uint64_t v36 = (uint64_t) v17;
    TASSIGN(v35, v36);
    pto::Shape<1, 1, 1, 16, 256> v37 = pto::Shape<1, 1, 1, 16, 256>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v38 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v39 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v1 + (v20 + v20 * v12 + (int64_t) ((uint64_t) v24 + (uint64_t) v6) * v11), v37, v38);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    TLOAD(v35, v39);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v40 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v7);
    uint64_t v41 = (uint64_t) v16;
    TASSIGN(v40, v41);
    pto::Shape<1, 1, 1, 16, 256> v42 = pto::Shape<1, 1, 1, 16, 256>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v43 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v44 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v1 + (v20 + v20 * v12 + (int64_t) ((uint64_t) v24 + (uint64_t) v5) * v11), v42, v43);
    TLOAD(v40, v44);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v45 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v7);
    uint64_t v46 = (uint64_t) v19;
    TASSIGN(v45, v46);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMUL(v45, v25, v25);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v47 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v7);
    uint64_t v48 = (uint64_t) v15;
    TASSIGN(v47, v48);
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v49 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v11);
    uint64_t v50 = (uint64_t) v14;
    TASSIGN(v49, v50);
    pipe_barrier(PIPE_V);
    TROWSUM(v49, v45, v47);
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v51 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
    uint64_t v52 = (uint64_t) v14;
    TASSIGN(v51, v52);
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v53 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
    uint64_t v54 = (uint64_t) v15;
    TASSIGN(v53, v54);
    pipe_barrier(PIPE_V);
    TADD(v53, v21, v51);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v55 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v7);
    uint64_t v56 = (uint64_t) v19;
    TASSIGN(v55, v56);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TMUL(v55, v30, v30);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v57 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v7);
    uint64_t v58 = (uint64_t) v18;
    TASSIGN(v57, v58);
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v59 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v11);
    uint64_t v60 = (uint64_t) v14;
    TASSIGN(v59, v60);
    pipe_barrier(PIPE_V);
    TROWSUM(v59, v55, v57);
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v61 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
    uint64_t v62 = (uint64_t) v14;
    TASSIGN(v61, v62);
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v63 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
    uint64_t v64 = (uint64_t) v15;
    TASSIGN(v63, v64);
    pipe_barrier(PIPE_V);
    TADD(v63, v53, v61);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v65 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v7);
    uint64_t v66 = (uint64_t) v19;
    TASSIGN(v65, v66);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    TMUL(v65, v35, v35);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v67 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v7);
    uint64_t v68 = (uint64_t) v18;
    TASSIGN(v67, v68);
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v69 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v11);
    uint64_t v70 = (uint64_t) v17;
    TASSIGN(v69, v70);
    pipe_barrier(PIPE_V);
    TROWSUM(v69, v65, v67);
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v71 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
    uint64_t v72 = (uint64_t) v17;
    TASSIGN(v71, v72);
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v73 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
    uint64_t v74 = (uint64_t) v17;
    TASSIGN(v73, v74);
    pipe_barrier(PIPE_V);
    TADD(v73, v63, v71);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v75 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v7);
    uint64_t v76 = (uint64_t) v19;
    TASSIGN(v75, v76);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    TMUL(v75, v40, v40);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v77 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v7);
    uint64_t v78 = (uint64_t) v18;
    TASSIGN(v77, v78);
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v79 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v11);
    uint64_t v80 = (uint64_t) v16;
    TASSIGN(v79, v80);
    pipe_barrier(PIPE_V);
    TROWSUM(v79, v75, v77);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v81 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
    uint64_t v82 = (uint64_t) v16;
    TASSIGN(v81, v82);
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v83 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
    uint64_t v84 = (uint64_t) v20;
    TASSIGN(v83, v84);
    pipe_barrier(PIPE_V);
    TADD(v83, v73, v81);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  }
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v85 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
  uint64_t v86 = (uint64_t) v19;
  TASSIGN(v85, v86);
  pipe_barrier(PIPE_V);
  TMULS(v85, v21, v4);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v87 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
  uint64_t v88 = (uint64_t) v19;
  TASSIGN(v87, v88);
  pipe_barrier(PIPE_V);
  TADDS(v87, v85, v3);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v89 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
  uint64_t v90 = (uint64_t) v19;
  TASSIGN(v89, v90);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v91 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
  uint64_t v92 = (uint64_t) v19;
  TASSIGN(v91, v92);
  pipe_barrier(PIPE_V);
  TSQRT(v91, v89);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v93 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
  uint64_t v94 = (uint64_t) v19;
  TASSIGN(v93, v94);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v95 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v13);
  uint64_t v96 = (uint64_t) v18;
  TASSIGN(v95, v96);
  pipe_barrier(PIPE_V);
  TRECIP(v95, v93);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v97 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v11);
  uint64_t v98 = (uint64_t) v18;
  TASSIGN(v97, v98);
  pto::Shape<1, 1, 1, 16, 1> v99 = pto::Shape<1, 1, 1, 16, 1>();
  pto::Stride<16, 16, 16, 1, 16> v100 = pto::Stride<16, 16, 16, 1, 16>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 16>, pto::Layout::DN> v101 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<16, 16, 16, 1, 16>, pto::Layout::DN>(v2 + (v20 + v20 * v11 + v20 * v13), v99, v100);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v101, v97);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}