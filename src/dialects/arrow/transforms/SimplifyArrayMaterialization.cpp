#include "SimplifyArrayMaterialization.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
namespace arcise::dialects::arrow {
GetLengthFromParent::GetLengthFromParent(mlir::MLIRContext *context)
    : mlir::OpRewritePattern<GetLengthOp>(context, 1) {}

mlir::LogicalResult
GetLengthFromParent::matchAndRewrite(GetLengthOp op,
                                     mlir::PatternRewriter &rewriter) const {
  mlir::Value array = op.array();
  if (auto makeArrayOp = mlir::dyn_cast<MakeArrayOp>(array.getDefiningOp())) {
    rewriter.replaceOp(op, {rewriter.create<mlir::DimOp>(
                               op.getLoc(), makeArrayOp.data_buffer(), 0)});
  } else if (auto getColumnOp =
                 mlir::dyn_cast<GetColumnOp>(array.getDefiningOp())) {
    rewriter.replaceOp(op, {rewriter.create<GetRowsCountOp>(
                               op.getLoc(), rewriter.getIndexType(),
                               getColumnOp.recordBatch())});
  }
  return mlir::success();
}

GetDataBufferFromParent::GetDataBufferFromParent(mlir::MLIRContext *context)
    : mlir::OpRewritePattern<GetDataBufferOp>(context, 1) {}

mlir::LogicalResult GetDataBufferFromParent::matchAndRewrite(
    GetDataBufferOp op, mlir::PatternRewriter &rewriter) const {
  mlir::Value array = op.array();
  if (auto makeArrayOp = mlir::dyn_cast<MakeArrayOp>(array.getDefiningOp())) {
    rewriter.replaceOp(op, makeArrayOp.data_buffer());
  }
  return mlir::success();
}

GetNullBitmapFromParent::GetNullBitmapFromParent(mlir::MLIRContext *context)
    : mlir::OpRewritePattern<GetNullBitmapOp>(context, 1) {}

mlir::LogicalResult GetNullBitmapFromParent::matchAndRewrite(
    GetNullBitmapOp op, mlir::PatternRewriter &rewriter) const {
  mlir::Value array = op.array();
  if (auto makeArrayOp = mlir::dyn_cast<MakeArrayOp>(array.getDefiningOp())) {
    rewriter.replaceOp(op, makeArrayOp.null_bitmap());
  }
  return mlir::success();
}
} // namespace arcise::dialects::arrow
