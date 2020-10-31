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

    auto resultElementType =
        resultColumnType.cast<ColumnType>().getElementType();

    typename Op::Adaptor binaryAdaptor(op.getOperands());

    auto res = rewriter.create<CreateEmptyColumnOp>(loc, columnType);
    for (int i = 0; i < columnType.getChunksCount(); i++) {
      auto chunkType = rewriter.getType<ArrayType>(
          columnType.getElementType(), columnType.getChunkLengths()[i]);

      auto resultType = rewriter.getType<ArrayType>(
          resultElementType, columnType.getChunkLengths()[i]);

      auto lhs =
          rewriter.create<GetChunkOp>(loc, chunkType, binaryAdaptor.lhs(), i);
      mlir::Value rhs = binaryAdaptor.rhs();
      if (rhs.getType().isa<ColumnType>()) {
        rhs = rewriter.create<GetChunkOp>(loc, resultType, rhs, i);
      }
      rewriter.create<SetChunkOp>(
          loc, res, i, rewriter.create<Op>(loc, resultType, lhs, rhs));
    }
    rewriter.replaceOp(op, {res});
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

    std::vector<std::pair<mlir::Value, std::vector<mlir::Value>>>
        columnToChunks;

    std::vector<Operation *> opsToErase;

    mlir::OpBuilder builder(&getContext());
    func.walk([&](mlir::Operation *op) {
      if (auto allocOp = mlir::dyn_cast<CreateEmptyColumnOp>(op)) {
        builder.setInsertionPointAfter(op);
        auto col = allocOp.getResult();
        auto columnType = col.getType().cast<ColumnType>();
        std::vector<mlir::Value> chunks;
        columnToChunks.push_back(
            {col,
             std::vector<mlir::Value>(columnType.getChunksCount(), nullptr)});
        opsToErase.push_back(op);
      }

      if (auto getColumnOp = mlir::dyn_cast<GetColumnOp>(op)) {
        builder.setInsertionPointAfter(op);
        auto name = getColumnOp.name();
        auto col = getColumnOp.res();
        auto columnType = col.getType().cast<ColumnType>();
        std::vector<mlir::Value> chunks;

        for (int i = 0; i < columnType.getChunksCount(); i++) {
          auto arrayType = builder.getType<ArrayType>(
              columnType.getElementType(), columnType.getChunkLengths()[i]);

          chunks.push_back(
              builder.create<GetArrayOp>(op->getLoc(), arrayType, name, i));
        }
        columnToChunks.push_back({col, chunks});
        opsToErase.push_back(op);
      }

      if (auto setChunkOp = mlir::dyn_cast<SetChunkOp>(op)) {
        builder.setInsertionPointAfter(op);
        auto idx = setChunkOp.idx();
        auto &colAndChunks =
            *std::find_if(columnToChunks.begin(), columnToChunks.end(),
                          [&](auto p) { return p.first == setChunkOp.col(); });
        assert(!colAndChunks.second[idx]);
        colAndChunks.second[idx] = setChunkOp.arr();
        opsToErase.push_back(op);
      }

      if (auto getChunkOp = mlir::dyn_cast<GetChunkOp>(op)) {
        auto idx = getChunkOp.idx();
        auto &colAndChunks =
            *std::find_if(columnToChunks.begin(), columnToChunks.end(),
                          [&](auto p) { return p.first == getChunkOp.col(); });

        getChunkOp.replaceAllUsesWith(colAndChunks.second[idx]);
        opsToErase.push_back(op);
      }

      if (auto returnColumnOp = mlir::dyn_cast<ReturnColumnOp>(op)) {
        builder.setInsertionPoint(op);
        auto col = returnColumnOp.col();
        auto name = returnColumnOp.name();

        auto &colAndChunks =
            *std::find_if(columnToChunks.begin(), columnToChunks.end(),
                          [&](auto p) { return p.first == col; });

        auto &chunks = colAndChunks.second;
        for (int i = 0; i < chunks.size(); i++) {
          builder.create<ReturnArrayOp>(op->getLoc(), chunks[i], name, i);
        }

        opsToErase.push_back(op);
      }
    });

    for (auto *op : opsToErase) {
      op->erase();
    }
  }
};

std::unique_ptr<Pass> createSplitColumnarOpsPass() {
  return std::make_unique<SplitColumnarOpsPass>();
}
} // namespace arcise::dialects::arrow