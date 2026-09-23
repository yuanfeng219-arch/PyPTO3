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

AICORE void x_gamma0(__gm__ bfloat16_t* v1, __gm__ float* v2, __gm__ float* v3, int64_t v4) {
  SaturationMode v5 = SaturationMode::OFF;
  RoundMode v6 = RoundMode::CAST_ROUND;
  const int64_t v7 = 256;
  const int64_t v8 = 2;
  const int64_t v9 = 4;
  const int64_t v10 = 1;
  const int64_t v11 = 5120;
  const int64_t v12 = 16;
  const int64_t v13 = 32768;
  const int64_t v14 = 16384;
  const int64_t v15 = 33792;
  const int64_t v16 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (size_t v17 = (size_t) v16; v17 < ((size_t) v9); v17 += (size_t) v8) {
    int64_t v18 = (int64_t) ((uint64_t) ((int64_t) v17) * (uint64_t) v7);
    int64_t v19 = (int64_t) ((uint64_t) v4 + (uint64_t) v18);
    int64_t v20 = (int64_t) ((uint64_t) v4 + (uint64_t) ((int64_t) (uint64_t) v18 + (uint64_t) v7));
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v21 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v7);
    uint64_t v22 = (uint64_t) v16;
    TASSIGN(v21, v22);
    pto::Shape<1, 1, 1, 16, 256> v23 = pto::Shape<1, 1, 1, 16, 256>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v24 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v25 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v2 + (v16 + v16 * v11 + v19 * v10), v23, v24);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v21, v25);
    Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v26 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v7);
    uint64_t v27 = (uint64_t) v15;
    TASSIGN(v26, v27);
    pto::Shape<1, 1, 1, 1, 256> v28 = pto::Shape<1, 1, 1, 1, 256>();
    pto::Stride<5120, 5120, 5120, 5120, 1> v29 = pto::Stride<5120, 5120, 5120, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND> v30 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND>(v3 + (v16 + v16 * v11 + v19 * v10), v28, v29);
    TLOAD(v26, v30);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v31 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v7);
    uint64_t v32 = (uint64_t) v14;
    TASSIGN(v31, v32);
    pto::Shape<1, 1, 1, 16, 256> v33 = pto::Shape<1, 1, 1, 16, 256>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v34 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v35 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v2 + (v16 + v16 * v11 + v20 * v10), v33, v34);
    TLOAD(v31, v35);
    Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v36 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v10, v7);
    uint64_t v37 = (uint64_t) v13;
    TASSIGN(v36, v37);
    pto::Shape<1, 1, 1, 1, 256> v38 = pto::Shape<1, 1, 1, 1, 256>();
    pto::Stride<5120, 5120, 5120, 5120, 1> v39 = pto::Stride<5120, 5120, 5120, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND> v40 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND>(v3 + (v16 + v16 * v11 + v20 * v10), v38, v39);
    TLOAD(v36, v40);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v41 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v7);
    uint64_t v42 = (uint64_t) v16;
    TASSIGN(v41, v42);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCOLEXPANDMUL(v41, v21, v26);
    Tile<TileType::Vec, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v43 = Tile<TileType::Vec, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v7);
    uint64_t v44 = (uint64_t) v15;
    TASSIGN(v43, v44);
    pipe_barrier(PIPE_V);
    TCVT(v43, v41, v6, v5);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v45 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v7);
    uint64_t v46 = (uint64_t) v16;
    TASSIGN(v45, v46);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TCOLEXPANDMUL(v45, v31, v36);
    Tile<TileType::Vec, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v47 = Tile<TileType::Vec, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v7);
    uint64_t v48 = (uint64_t) v16;
    TASSIGN(v47, v48);
    pipe_barrier(PIPE_V);
    TCVT(v47, v45, v6, v5);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    pto::Shape<1, 1, 1, 16, 256> v49 = pto::Shape<1, 1, 1, 16, 256>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v50 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v51 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v1 + (v16 + v16 * v11 + v19 * v10), v49, v50);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v51, v43);
    pto::Shape<1, 1, 1, 16, 256> v52 = pto::Shape<1, 1, 1, 16, 256>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v53 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v54 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v1 + (v16 + v16 * v11 + v20 * v10), v52, v53);
    pipe_barrier(PIPE_MTE3);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(v54, v47);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}