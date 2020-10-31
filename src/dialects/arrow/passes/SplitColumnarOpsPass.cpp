#pragma once
#include "dialects/arrow/ArrowDialect.h"
#include "dialects/arrow/ArrowOps.h"
#include "dialects/arrow/ArrowTypes.h"
#include "dialects/arrow/passes/Passes.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <iostream>

namespace arcise::dialects::arrow {
template <typename Op, bool ReturnsBool>
class SplitColumnarOps : public mlir::OpRewritePattern<Op> {
public:
  SplitColumnarOps(mlir::MLIRContext *ctx) : OpRewritePattern<Op>(ctx, 1) {}

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    mlir::Operation *operation = op.getOperation();

    auto loc = operation->getLoc();

    auto columnType = (*operation->operand_type_begin()).dyn_cast<ColumnType>();

    if (!columnType) {
      return mlir::failure();
    }
    mlir::Type resultColumnType = op.res().getType();

    auto resultType = resultColumnType.cast<ColumnType>();

    typename Op::Adaptor binaryAdaptor(op.getOperands());

    auto unwrapLhs = rewriter.create<UnwrapColumnOp>(
        loc, columnType.getChunks(), binaryAdaptor.lhs());

    std::vector<mlir::Value> results;

    mlir::Value rhs = binaryAdaptor.rhs();
    if (rhs.getType().isa<ColumnType>()) {
      auto unwrapRhs =
          rewriter.create<UnwrapColumnOp>(loc, columnType.getChunks(), rhs);
      for (int i = 0; i < columnType.getChunksCount(); i++) {
        results.push_back(rewriter.create<Op>(loc, resultType.getChunk(i),
                                              unwrapLhs.res()[i],
                                              unwrapRhs.res()[i]));
      }
    } else {
      for (int i = 0; i < columnType.getChunksCount(); i++) {
        results.push_back(rewriter.create<Op>(loc, resultType.getChunk(i),
                                              unwrapLhs.res()[i], rhs));
      }
    }
    rewriter.replaceOp(
        op, {rewriter.create<MakeColumnOp>(loc, resultType, results)});
    return mlir::success();
  }
};

struct SplitColumnarOpsPass
    : public mlir::PassWrapper<SplitColumnarOpsPass, mlir::FunctionPass> {

  void runOnFunction() override {
    auto func = getFunction();
    mlir::OwningRewritePatternList patterns;
    patterns.insert<SplitColumnarOps<SumOp, false>,
                    SplitColumnarOps<ConstMulOp, false>,
                    SplitColumnarOps<GeOp, true>>(&getContext());

    func.walk([&](mlir::Operation *op) {
      mlir::applyOpPatternsAndFold(op, patterns);
    });
  }
};

std::unique_ptr<Pass> createSplitColumnarOpsPass() {
  return std::make_unique<SplitColumnarOpsPass>();
}
} // namespace arcise::dialects::arrow