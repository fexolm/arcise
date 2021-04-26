#include "arcise/dialects/arrow/ArrowOps.h"
#include "arcise/dialects/arrow/ArrowDialect.h"
#include "arcise/dialects/arrow/ArrowTypes.h"
#include "arcise/dialects/arrow/transforms/SimplifyArrayMaterialization.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "arcise/dialects/arrow/tablegen/ArrowOps.cpp.inc"

namespace arcise::dialects::arrow {
void GetLengthOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &results, mlir::MLIRContext *context) {
  results.insert<GetLengthFromParent>(context);
}

void GetDataBufferOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &results, mlir::MLIRContext *context) {
  results.insert<GetDataBufferFromParent>(context);
}

void GetNullBitmapOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &results, mlir::MLIRContext *context) {
  results.insert<GetNullBitmapFromParent>(context);
}
} // namespace arcise::dialects::arrow
