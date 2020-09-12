#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace arcise::dialects {
using namespace mlir;
#define GET_OP_CLASSES
#include "dialects/arrow/ArrowOps.h.inc"
} // namespace arcise::dialiects