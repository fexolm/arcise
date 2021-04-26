#include "arcise/dialects/arrow/ArrowDialect.h"
#include "arcise/dialects/arrow/ArrowOps.h"
#include "arcise/dialects/arrow/ArrowTypes.h"
#include "arcise/dialects/arrow/passes/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace arcise::dialects::arrow {

static mlir::Value insertAllocAndDealloc(mlir::PatternRewriter &rewriter,
                                         mlir::Location loc,
                                         mlir::MemRefType type,
                                         mlir::Value size) {
  auto alloc = rewriter.create<mlir::AllocOp>(loc, type, size);

  auto *parentBlock = alloc.getOperation()->getBlock();
  auto dealloc = rewriter.create<mlir::DeallocOp>(loc, alloc);
  dealloc.getOperation()->moveBefore(&parentBlock->back());
  return alloc;
}

template <typename BinaryOp, typename LoweredBinaryOpBuilder>
struct BinaryOpLowering : public mlir::ConversionPattern {
  BinaryOpLowering(mlir::MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    auto arrayType = (*op->operand_type_begin()).cast<ArrayType>();
    auto resultType = (*op->result_type_begin()).cast<ArrayType>();

    mlir::Value zeroConstant =
        rewriter.create<mlir::ConstantOp>(loc, rewriter.getIndexAttr(0));

    typename BinaryOp::Adaptor binaryAdaptor(operands);
    mlir::Value lhs = binaryAdaptor.lhs();
    mlir::Value rhs = binaryAdaptor.rhs();

    mlir::Value arrayLength =
        rewriter.create<GetLengthOp>(loc, rewriter.getIndexType(), lhs);

    auto nullBitmapType = mlir::MemRefType::get(-1, rewriter.getI1Type());
    auto dataBufferType = mlir::MemRefType::get(-1, arrayType.getElementType());
    auto resultBufferType =
        mlir::MemRefType::get(-1, resultType.getElementType());

    auto lhsBuffer = rewriter.create<GetDataBufferOp>(loc, dataBufferType, lhs);
    auto lhsBitmap = rewriter.create<GetNullBitmapOp>(loc, nullBitmapType, lhs);

    mlir::Value resultBuffer =
        insertAllocAndDealloc(rewriter, loc, resultBufferType, arrayLength);

    mlir::Value resultBitmap;
    if (rhs.getType().isa<ArrayType>()) {
      auto rhsBuffer =
          rewriter.create<GetDataBufferOp>(loc, dataBufferType, rhs);
      auto rhsBitmap =
          rewriter.create<GetNullBitmapOp>(loc, nullBitmapType, rhs);

      // we don't need to allocate bitmap buffer if second operand is constant
      resultBitmap =
          insertAllocAndDealloc(rewriter, loc, nullBitmapType, arrayLength);

      mlir::buildAffineLoopNest(
          rewriter, loc, zeroConstant, arrayLength, 1,
          [&](mlir::OpBuilder &builder, mlir::Location loc,
              mlir::ValueRange ivs) {
            auto loadedLhsBuffer =
                builder.create<mlir::AffineLoadOp>(loc, lhsBuffer, ivs);
            auto loadedRhsBuffer =
                builder.create<mlir::AffineLoadOp>(loc, rhsBuffer, ivs);

            mlir::Value bufferResult = LoweredBinaryOpBuilder::create(
                arrayType.getElementType().isIntOrIndex(), builder, loc,
                loadedLhsBuffer, loadedRhsBuffer);
            builder.create<mlir::AffineStoreOp>(loc, bufferResult, resultBuffer,
                                                ivs);

            auto loadedLhsBitmap =
                builder.create<mlir::AffineLoadOp>(loc, lhsBitmap, ivs);
            auto loadedRhsBitmap =
                builder.create<mlir::AffineLoadOp>(loc, rhsBitmap, ivs);

            mlir::Value bitmapResult = builder.create<mlir::AndOp>(
                loc, builder.getI1Type(), loadedLhsBitmap, loadedRhsBitmap);

            builder.create<mlir::AffineStoreOp>(loc, bitmapResult, resultBitmap,
                                                ivs);
          });
    } else {
      mlir::buildAffineLoopNest(
          rewriter, loc, zeroConstant, arrayLength, 1,
          [&](mlir::OpBuilder &builder, mlir::Location loc,
              mlir::ValueRange ivs) {
            auto loadedLhsBuffer =
                builder.create<mlir::AffineLoadOp>(loc, lhsBuffer, ivs);
            mlir::Value bufferResult = LoweredBinaryOpBuilder::create(
                arrayType.getElementType().isIntOrIndex(), builder, loc,
                loadedLhsBuffer, rhs);
            builder.create<mlir::AffineStoreOp>(loc, bufferResult, resultBuffer,
                                                ivs);
          });
      resultBitmap = lhsBitmap;
    }

    rewriter.replaceOp(op, {rewriter.create<MakeArrayOp>(
                               loc, resultType, resultBitmap, resultBuffer)});
    return mlir::success();
  }
};

template <typename IntegralOp, typename FloatingPointOp>
struct ArithmeticOpBuilder {
  static mlir::Value create(bool isIntegral, mlir::OpBuilder &builder,
                            const mlir::Location &loc, const mlir::Value &lhs,
                            const mlir::Value &rhs) {
    if (isIntegral) {
      return builder.create<IntegralOp>(loc, lhs, rhs);
    } else {
      return builder.create<FloatingPointOp>(loc, lhs, rhs);
    }
  }
};

template <typename Op> struct ExactOpBuilder {
  static mlir::Value create(bool isIntegral, mlir::OpBuilder &builder,
                            const mlir::Location &loc, const mlir::Value &lhs,
                            const mlir::Value &rhs) {
    return builder.create<Op>(loc, lhs, rhs);
  }
};

template <mlir::CmpIPredicate IPredicate, mlir::CmpFPredicate FPredicate>
struct ComparisonOpBuilder {
  static mlir::Value create(bool isIntegral, mlir::OpBuilder &builder,
                            const mlir::Location &loc, const mlir::Value &lhs,
                            const mlir::Value &rhs) {
    if (isIntegral) {
      return builder.create<mlir::CmpIOp>(loc, IPredicate, lhs, rhs);
    } else {
      return builder.create<mlir::CmpFOp>(loc, FPredicate, lhs, rhs);
    }
  }
};

using ArrowEqOpLowering =
    BinaryOpLowering<EqOp, ComparisonOpBuilder<mlir::CmpIPredicate::eq,
                                               mlir::CmpFPredicate::OEQ>>;
using ArrowNeqOpLowering =
    BinaryOpLowering<NeqOp, ComparisonOpBuilder<mlir::CmpIPredicate::ne,
                                                mlir::CmpFPredicate::ONE>>;
using ArrowGeOpLowering =
    BinaryOpLowering<GeOp, ComparisonOpBuilder<mlir::CmpIPredicate::sge,
                                               mlir::CmpFPredicate::OGE>>;
using ArrowLeOpLowering =
    BinaryOpLowering<LeOp, ComparisonOpBuilder<mlir::CmpIPredicate::sle,
                                               mlir::CmpFPredicate::OLE>>;
using ArrowGtOpLowering =
    BinaryOpLowering<GtOp, ComparisonOpBuilder<mlir::CmpIPredicate::sgt,
                                               mlir::CmpFPredicate::OGT>>;
using ArrowLtOpLowering =
    BinaryOpLowering<LtOp, ComparisonOpBuilder<mlir::CmpIPredicate::slt,
                                               mlir::CmpFPredicate::OLT>>;

using ArrowSumOpLowering =
    BinaryOpLowering<SumOp, ArithmeticOpBuilder<mlir::AddIOp, mlir::AddFOp>>;
using ArrowMulOpLowering =
    BinaryOpLowering<MulOp, ArithmeticOpBuilder<mlir::MulIOp, mlir::MulFOp>>;
using ArrowDivOpLowering =
    BinaryOpLowering<DivOp,
                     ArithmeticOpBuilder<mlir::SignedDivIOp, mlir::DivFOp>>;
using ArrowSubOpLowering =
    BinaryOpLowering<SubOp, ArithmeticOpBuilder<mlir::SubIOp, mlir::SubFOp>>;

using ArrowAndOpLowering = BinaryOpLowering<AndOp, ExactOpBuilder<mlir::AndOp>>;
using ArrowOrOpLowering = BinaryOpLowering<OrOp, ExactOpBuilder<mlir::OrOp>>;

struct ArrowToAffineLoweringPass
    : public mlir::PassWrapper<ArrowToAffineLoweringPass, mlir::FunctionPass> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::AffineDialect, mlir::StandardOpsDialect>();
  }

  void runOnFunction() override {
    auto function = getFunction();

    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<mlir::AffineDialect, mlir::StandardOpsDialect>();

    target.addIllegalDialect<ArrowDialect>();
    // target.addLegalOp<FilterOp>();
    target.addLegalOp<GetColumnOp>();
    target.addLegalOp<MakeArrayOp>();
    target.addLegalOp<GetDataBufferOp>();
    target.addLegalOp<GetNullBitmapOp>();
    target.addLegalOp<GetLengthOp>();
    target.addLegalOp<GetRowsCountOp>();
    target.addLegalOp<MakeRecordBatchOp>();

    mlir::OwningRewritePatternList patterns;
    patterns.insert<ArrowEqOpLowering, ArrowNeqOpLowering, ArrowGeOpLowering,
                    ArrowLeOpLowering, ArrowGtOpLowering, ArrowLtOpLowering,
                    ArrowSumOpLowering, ArrowMulOpLowering, ArrowDivOpLowering,
                    ArrowSubOpLowering, ArrowAndOpLowering, ArrowOrOpLowering>(
        &getContext());

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();

    std::vector<mlir::Value> resultArrays;

    function.walk([&](mlir::Operation *op) {
      if (auto returnOp = mlir::dyn_cast<mlir::ReturnOp>(op)) {
        mlir::Value recordBatchResult = returnOp.getOperand(0);
        for (auto arr : recordBatchResult.getDefiningOp()->getOperands()) {
          for (auto memref : arr.getDefiningOp()->getOperands()) {
            resultArrays.push_back(memref);
          }
        }
      }
    });

    std::vector<mlir::Operation *> opsToErase;
    function.walk([&](mlir::Operation *op) {
      if (auto deallocOp = mlir::dyn_cast<mlir::DeallocOp>(op)) {
        auto operand = op->getOperand(0);
        if (std::any_of(resultArrays.begin(), resultArrays.end(),
                        [&](auto &arr) { return operand == arr; })) {
          opsToErase.push_back(op);
        }
      }
    });

    for (auto &op : opsToErase) {
      op->erase();
    }
  }
}; // namespace arcise::dialects::arrow
std::unique_ptr<mlir::Pass> createLowerToAffinePass() {
  return std::make_unique<ArrowToAffineLoweringPass>();
}
} // namespace arcise::dialects::arrow
