#include "arcise/dialects/arrow/ArrowDialect.h"
#include "arcise/dialects/arrow/ArrowOps.h"
#include "arcise/dialects/arrow/ArrowTypes.h"
#include "arcise/dialects/arrow/passes/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace arcise::dialects::arrow {
struct MoveAllocationsOnTop : public mlir::OpRewritePattern<mlir::AllocOp> {
  MoveAllocationsOnTop(mlir::MLIRContext *ctx)
      : mlir::OpRewritePattern<mlir::AllocOp>(ctx, 1) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::AllocOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getOperands().size() == 1) {
      auto sizeValue = op.getOperand(0);
      rewriter.setInsertionPointAfterValue(sizeValue);
      auto newAlloc = rewriter.create<mlir::AllocOp>(
          op.getLoc(), op.getResult().getType(), sizeValue);
      rewriter.replaceOp(op, {newAlloc});
    }
    return mlir::success();
  }
};

struct MoveAllocationsOnTopPass
    : public mlir::PassWrapper<MoveAllocationsOnTopPass, mlir::FunctionPass> {
  void runOnFunction() final {
    auto func = getFunction();
    mlir::OwningRewritePatternList patterns;
    patterns.insert<MoveAllocationsOnTop>(&getContext());

    mlir::FrozenRewritePatternList frozenPatterns(std::move(patterns));

    patterns.insert<MoveAllocationsOnTop>(&getContext());
    func.walk([&](mlir::Operation *op) {
      mlir::applyOpPatternsAndFold(op, frozenPatterns);
    });
  }
};

std::unique_ptr<mlir::Pass> createMoveAllocationsOnTopPass() {
  return std::make_unique<MoveAllocationsOnTopPass>();
}
} // namespace arcise::dialects::arrow