#pragma once

#include "dialects/arrow/ArrowOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

namespace arcise::dialects::arrow {
struct GetLengthFromParent : public mlir::OpRewritePattern<GetLengthOp> {
  GetLengthFromParent(mlir::MLIRContext *context);

  mlir::LogicalResult
  matchAndRewrite(GetLengthOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

struct GetDataBufferFromParent
    : public mlir::OpRewritePattern<GetDataBufferOp> {
  GetDataBufferFromParent(mlir::MLIRContext *context);

  mlir::LogicalResult
  matchAndRewrite(GetDataBufferOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

struct GetNullBitmapFromParent
    : public mlir::OpRewritePattern<GetNullBitmapOp> {
  GetNullBitmapFromParent(mlir::MLIRContext *context);

  mlir::LogicalResult
  matchAndRewrite(GetNullBitmapOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace arcise::dialects::arrow
