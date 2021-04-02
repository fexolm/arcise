#include "arcise/dialects/relalg/dialect/RelalgOps.h"
#include "arcise/dialects/relalg/dialect/RelalgDialect.h"
#include "arcise/dialects/relalg/dialect/RelalgTypes.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

#define GET_OP_CLASSES
#include "arcise/dialects/relalg/dialect/tablegen/RelalgOps.cpp.inc"