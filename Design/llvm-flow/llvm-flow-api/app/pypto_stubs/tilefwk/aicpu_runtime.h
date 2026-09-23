#pragma once

#include "tilefwk/aicore_data.h"

#define RUNTIME_FUNCKEY_FINISH static_cast<std::uint64_t>(-1)

#define RUNTIME_GetInputShapeDim(inputIndex, dimIndex) \
  (startArgs->devTensorList[(inputIndex)].shape.dim[(dimIndex)])

#define RUNTIME_GetSymbol(index) (symbolTable[(index)])

#define RUNTIME_SetExpr(destination, index, expression) \
  do { (destination)[(index)] = (expression); } while (false)

#define RUNTIME_RootAlloc(functionKey) callRootList[0](ctx, (functionKey))

// The real runtime uses this return value to stop stitching early. Keeping the
// branch here is important: it preserves the generated source's control flow.
#define RUNTIME_RootStitch(functionKey)                                  \
  do {                                                                   \
    if (callRootList[1](ctx, (functionKey)) ==                           \
        reinterpret_cast<void *>(static_cast<std::uintptr_t>(1))) {      \
      return 0;                                                          \
    }                                                                    \
  } while (false)
