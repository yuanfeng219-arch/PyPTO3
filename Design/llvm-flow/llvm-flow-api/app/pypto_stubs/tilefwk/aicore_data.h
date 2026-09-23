#pragma once

#include <cstdint>

namespace npu {
namespace tile_fwk {

using CallRootEntryType = void *(*)(void *, std::uint64_t);

struct DevShape {
  std::int32_t dimSize;
  std::int64_t dim[8];
};

struct DevTensorData {
  std::uint64_t address;
  DevShape shape;
};

struct DevStartArgsBase {
  DevTensorData *devTensorList;
};

}  // namespace tile_fwk
}  // namespace npu

#define ARG_IN_0 0
#define ARG_IN_1 1
#define ARG_IN_2 2
#define ARG_IN_9 9
#define ARG_IN_10 10
#define ARG_IN_13 13
#define ARG_IN_14 14
#define ARG_IN_15 15
