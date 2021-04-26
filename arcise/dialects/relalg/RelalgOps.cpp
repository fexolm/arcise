#include "arcise/dialects/relalg/RelalgOps.h"
#include "arcise/dialects/relalg/RelalgDialect.h"
#include "arcise/dialects/relalg/RelalgTypes.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

#define GET_OP_CLASSES
#include "arcise/dialects/relalg/tablegen/RelalgOps.cpp.inc"