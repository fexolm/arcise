#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace arcise::dialects::arrow {
using namespace mlir;
#define GET_OP_CLASSES
#include "dialects/arrow/tablegen/ArrowOps.h.inc"
} // namespace arcise::dialects::arrow