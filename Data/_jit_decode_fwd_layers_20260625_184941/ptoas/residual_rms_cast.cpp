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

AICORE void residual_rms_cast(__gm__ bfloat16_t* v1, __gm__ float* v2, __gm__ float* v3, __gm__ float* v4, __gm__ float* v5) {
  SaturationMode v6 = SaturationMode::OFF;
  RoundMode v7 = RoundMode::CAST_ROUND;
  const int64_t v8 = 256;
  const int64_t v9 = 896;
  const int64_t v10 = 2;
  const int64_t v11 = 4;
  const int64_t v12 = 1;
  const int64_t v13 = 5120;
  const int64_t v14 = 16;
  const int64_t v15 = 0;
  const int64_t v16 = 58368;
  const int64_t v17 = 41984;
  const int64_t v18 = 1024;
  const int64_t v19 = 25600;
  const int64_t v20 = 9216;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  int64_t v21;
  v21 = (int64_t) ((size_t) v9);
  for (int64_t v22 = (int64_t) ((size_t) v15); v22 < ((int64_t) ((size_t) v11)); v22 += (int64_t) ((size_t) v10)) {
    int64_t v23 = (int64_t) ((uint64_t) v22 * (uint64_t) v8);
    int64_t v24 = (int64_t) ((uint64_t) v23 + (uint64_t) v8);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v25 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v8);
    uint64_t v26 = (uint64_t) v20;
    TASSIGN(v25, v26);
    pto::Shape<1, 1, 1, 16, 256> v27 = pto::Shape<1, 1, 1, 16, 256>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v28 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v29 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v3 + (v15 + v15 * v13 + v23 * v12), v27, v28);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v25, v29);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v30 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v8);
    uint64_t v31 = (uint64_t) v19;
    TASSIGN(v30, v31);
    pto::Shape<1, 1, 1, 16, 256> v32 = pto::Shape<1, 1, 1, 16, 256>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v33 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v34 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v4 + (v15 + v15 * v13 + v23 * v12), v32, v33);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    TLOAD(v30, v34);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v35 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v8);
    uint64_t v36 = (uint64_t) v18;
    TASSIGN(v35, v36);
    pto::Shape<1, 1, 1, 1, 256> v37 = pto::Shape<1, 1, 1, 1, 256>();
    pto::Stride<5120, 5120, 5120, 5120, 1> v38 = pto::Stride<5120, 5120, 5120, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND> v39 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND>(v5 + (v15 + v15 * v13 + v23 * v12), v37, v38);
    TLOAD(v35, v39);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v40 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v8);
    uint64_t v41 = (uint64_t) v17;
    TASSIGN(v40, v41);
    pto::Shape<1, 1, 1, 16, 256> v42 = pto::Shape<1, 1, 1, 16, 256>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v43 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v44 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v3 + (v15 + v15 * v13 + v24 * v12), v42, v43);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    TLOAD(v40, v44);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v45 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v8);
    uint64_t v46 = (uint64_t) v16;
    TASSIGN(v45, v46);
    pto::Shape<1, 1, 1, 16, 256> v47 = pto::Shape<1, 1, 1, 16, 256>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v48 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v49 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v4 + (v15 + v15 * v13 + v24 * v12), v47, v48);
    TLOAD(v45, v49);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v50 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v8);
    uint64_t v51 = (uint64_t) v15;
    TASSIGN(v50, v51);
    pto::Shape<1, 1, 1, 1, 256> v52 = pto::Shape<1, 1, 1, 1, 256>();
    pto::Stride<5120, 5120, 5120, 5120, 1> v53 = pto::Stride<5120, 5120, 5120, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND> v54 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND>(v5 + (v15 + v15 * v13 + v24 * v12), v52, v53);
    TLOAD(v50, v54);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v55 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v8);
    uint64_t v56 = (uint64_t) v20;
    TASSIGN(v55, v56);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(v55, v25, v30);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v57 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v8);
    uint64_t v58 = (uint64_t) v19;
    TASSIGN(v57, v58);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TCOLEXPANDMUL(v57, v55, v35);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    Tile<TileType::Vec, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v59 = Tile<TileType::Vec, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v8);
    uint64_t v60 = (uint64_t) v18;
    TASSIGN(v59, v60);
    pipe_barrier(PIPE_V);
    TCVT(v59, v57, v7, v6);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v61 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v8);
    uint64_t v62 = (uint64_t) v19;
    TASSIGN(v61, v62);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    TADD(v61, v40, v45);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v63 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v8);
    uint64_t v64 = (uint64_t) v17;
    TASSIGN(v63, v64);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    TCOLEXPANDMUL(v63, v61, v50);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    Tile<TileType::Vec, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v65 = Tile<TileType::Vec, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v14, v8);
    uint64_t v66 = (uint64_t) v17;
    TASSIGN(v65, v66);
    pipe_barrier(PIPE_V);
    TCVT(v65, v63, v7, v6);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    pto::Shape<1, 1, 1, 16, 256> v67 = pto::Shape<1, 1, 1, 16, 256>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v68 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v69 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v2 + (v15 + v15 * v13 + v23 * v12), v67, v68);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v69, v55);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 256> v70 = pto::Shape<1, 1, 1, 16, 256>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v71 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v72 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v1 + (v15 + v15 * v13 + v23 * v12), v70, v71);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v72, v59);
    pto::Shape<1, 1, 1, 16, 256> v73 = pto::Shape<1, 1, 1, 16, 256>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v74 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v75 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v2 + (v15 + v15 * v13 + v24 * v12), v73, v74);
    pipe_barrier(PIPE_MTE3);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    TSTORE(v75, v61);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    pto::Shape<1, 1, 1, 16, 256> v76 = pto::Shape<1, 1, 1, 16, 256>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v77 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v78 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v1 + (v15 + v15 * v13 + v24 * v12), v76, v77);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    TSTORE(v78, v65);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    v21 = (int64_t) ((size_t) v24);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}