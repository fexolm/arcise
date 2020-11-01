#include "dialects/arrow/ArrowDialect.h"
#include "dialects/arrow/ArrowOps.h"
#include "dialects/arrow/ArrowTypes.h"
#include "dialects/arrow/passes/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <iostream>

namespace arcise::dialects::arrow {
template <typename UnwrapOp, typename WrapOp>
struct SimplifyRedundantWrapping : public mlir::OpRewritePattern<UnwrapOp> {
  SimplifyRedundantWrapping(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<UnwrapOp>(context, 1) {}

  mlir::LogicalResult
  matchAndRewrite(UnwrapOp op, mlir::PatternRewriter &rewriter) const override {
    auto operation = op.getOperation();
    auto parent = operation->getOperand(0).getDefiningOp();
    if (mlir::isa<WrapOp>(parent)) {
      parent->dump();
      rewriter.replaceOp(op, parent->getOperands());
    }
    return success();
  }
};
} // namespace arcise::dialects::arrow