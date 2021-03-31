#include "dialects/relalg/RelalgOps.h"
#include "dialects/relalg/RelalgDialect.h"
#include "dialects/relalg/RelalgTypes.h"
#include "dialects/relalg/transforms/
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

#define GET_OP_CLASSES
#include "dialects/relalg/tablegen/RelalgOps.cpp.inc"

namespace arcise::dialects::relalg {

void relalg::getProjectionOp(
    mlir::OwningRewritePatternList &results, mlir::MLIRContext *context) {
  results.insert<GetNullBitmapFromParent>(context);
}

} // namespace arcise::dialects::relalg
