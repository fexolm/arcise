#pragma once

#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#define GET_OP_CLASSES
#include "arcise/dialects/relalg/tablegen/RelalgOps.h.inc"
#undef GET_OP_CLASSES