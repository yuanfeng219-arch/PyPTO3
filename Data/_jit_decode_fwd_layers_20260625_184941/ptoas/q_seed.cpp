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

AICORE void q_seed(__gm__ float* v1) {
  const float v2 = 0.0f;
  const int64_t v3 = 512;
  const int64_t v4 = 2;
  const int64_t v5 = 10;
  const int64_t v6 = 1;
  const int64_t v7 = 5120;
  const int64_t v8 = 16;
  const int64_t v9 = 32768;
  const int64_t v10 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  for (size_t v11 = (size_t) v10; v11 < ((size_t) v5); v11 += (size_t) v4) {
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v12 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v8, v3);
    uint64_t v13 = (uint64_t) v10;
    TASSIGN(v12, v13);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TEXPANDS(v12, v2);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v14 = Tile<TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v8, v3);
    uint64_t v15 = (uint64_t) v9;
    TASSIGN(v14, v15);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    TEXPANDS(v14, v2);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    int64_t v16 = (int64_t) ((uint64_t) ((int64_t) v11) * (uint64_t) v3);
    pto::Shape<1, 1, 1, 16, 512> v17 = pto::Shape<1, 1, 1, 16, 512>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v18 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v19 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v1 + (v10 + v10 * v7 + v16 * v6), v17, v18);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v19, v12);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 512> v20 = pto::Shape<1, 1, 1, 16, 512>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v21 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v22 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v1 + (v10 + v10 * v7 + (int64_t) ((uint64_t) v16 + (uint64_t) v3) * v6), v20, v21);
    pipe_barrier(PIPE_MTE3);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(v22, v14);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  }
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}