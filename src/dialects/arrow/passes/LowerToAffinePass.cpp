#include "dialects/arrow/ArrowDialect.h"
#include "dialects/arrow/ArrowOps.h"
#include "dialects/arrow/ArrowTypes.h"
#include "dialects/arrow/passes/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace arcise::dialects::arrow {

static mlir::Value insertAllocAndDealloc(mlir::MemRefType type,
                                         mlir::Location loc,
                                         mlir::PatternRewriter &rewriter) {
  auto alloc = rewriter.create<mlir::AllocOp>(loc, type);

  auto *parentBlock = alloc.getOperation()->getBlock();
  alloc.getOperation()->moveBefore(&parentBlock->front());

  auto dealloc = rewriter.create<mlir::DeallocOp>(loc, alloc);
  dealloc.getOperation()->moveBefore(&parentBlock->back());
  return alloc;
}

template <typename BinaryOp, typename LoweredBinaryOpBuilder, bool IsConst>
struct BinaryOpLowering : public mlir::ConversionPattern {
  BinaryOpLowering(mlir::MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    auto arrayType = (*op->operand_type_begin()).cast<ArrayType>();
    auto resultType = (*op->result_type_begin()).cast<ArrayType>();

    mlir::SmallVector<int64_t, 1> lbs(1, 0);
    mlir::SmallVector<int64_t, 1> ubs(1, arrayType.getLength());
    mlir::SmallVector<int64_t, 1> steps(1, 1);

    typename BinaryOp::Adaptor binaryAdaptor(operands);
    mlir::Value lhs = binaryAdaptor.lhs();
    mlir::Value rhs = binaryAdaptor.rhs();

    auto nullBitmapType =
        mlir::MemRefType::get(arrayType.getLength(), rewriter.getI1Type());

    auto dataBufferType = mlir::MemRefType::get(arrayType.getLength(),
                                                arrayType.getElementType());

    auto resultBufferType = mlir::MemRefType::get(resultType.getLength(),
                                                  resultType.getElementType());

    auto unwrapLhs = rewriter.create<UnwrapArrayOp>(loc, nullBitmapType,
                                                    dataBufferType, lhs);

    auto lhsBuffer = unwrapLhs.data_buffer();
    auto lhsBitmap = unwrapLhs.null_bitmap();

    mlir::Value resultBuffer =
        insertAllocAndDealloc(resultBufferType, loc, rewriter);
    mlir::Value resultBitmap;
    if (rhs.getType().isa<ArrayType>()) {
      auto unwrapRhs = rewriter.create<UnwrapArrayOp>(loc, nullBitmapType,
                                                      dataBufferType, rhs);
      auto rhsBuffer = unwrapRhs.data_buffer();
      auto rhsBitmap = unwrapRhs.null_bitmap();
      resultBitmap = insertAllocAndDealloc(nullBitmapType, loc, rewriter);

      mlir::buildAffineLoopNest(
          rewriter, loc, lbs, ubs, steps,
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
          rewriter, loc, lbs, ubs, steps,
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

using ArrowEqOpLowering = BinaryOpLowering<
    EqOp,
    ComparisonOpBuilder<mlir::CmpIPredicate::eq, mlir::CmpFPredicate::OEQ>,
    false>;
using ArrowNeqOpLowering = BinaryOpLowering<
    NeqOp,
    ComparisonOpBuilder<mlir::CmpIPredicate::ne, mlir::CmpFPredicate::ONE>,
    false>;
using ArrowGeOpLowering = BinaryOpLowering<
    GeOp,
    ComparisonOpBuilder<mlir::CmpIPredicate::sge, mlir::CmpFPredicate::OGE>,
    false>;
using ArrowLeOpLowering = BinaryOpLowering<
    LeOp,
    ComparisonOpBuilder<mlir::CmpIPredicate::sle, mlir::CmpFPredicate::OLE>,
    false>;
using ArrowGtOpLowering = BinaryOpLowering<
    GtOp,
    ComparisonOpBuilder<mlir::CmpIPredicate::sgt, mlir::CmpFPredicate::OGT>,
    false>;
using ArrowLtOpLowering = BinaryOpLowering<
    LtOp,
    ComparisonOpBuilder<mlir::CmpIPredicate::slt, mlir::CmpFPredicate::OLT>,
    false>;
using ArrowConstEqOpLowering = BinaryOpLowering<
    ConstEqOp,
    ComparisonOpBuilder<mlir::CmpIPredicate::eq, mlir::CmpFPredicate::OEQ>,
    true>;
using ArrowConstNeqOpLowering = BinaryOpLowering<
    ConstNeqOp,
    ComparisonOpBuilder<mlir::CmpIPredicate::ne, mlir::CmpFPredicate::ONE>,
    true>;
using ArrowConstGeOpLowering = BinaryOpLowering<
    ConstGeOp,
    ComparisonOpBuilder<mlir::CmpIPredicate::sge, mlir::CmpFPredicate::OGE>,
    true>;
using ArrowConstLeOpLowering = BinaryOpLowering<
    ConstLeOp,
    ComparisonOpBuilder<mlir::CmpIPredicate::sle, mlir::CmpFPredicate::OLE>,
    true>;
using ArrowConstGtOpLowering = BinaryOpLowering<
    ConstGtOp,
    ComparisonOpBuilder<mlir::CmpIPredicate::sgt, mlir::CmpFPredicate::OGT>,
    true>;
using ArrowConstLtOpLowering = BinaryOpLowering<
    ConstLtOp,
    ComparisonOpBuilder<mlir::CmpIPredicate::slt, mlir::CmpFPredicate::OLT>,
    true>;
using ArrowSumOpLowering =
    BinaryOpLowering<SumOp, ArithmeticOpBuilder<mlir::AddIOp, mlir::AddFOp>,
                     false>;
using ArrowMulOpLowering =
    BinaryOpLowering<MulOp, ArithmeticOpBuilder<mlir::MulIOp, mlir::MulFOp>,
                     false>;
using ArrowDivOpLowering = BinaryOpLowering<
    DivOp, ArithmeticOpBuilder<mlir::SignedDivIOp, mlir::DivFOp>, false>;
using ArrowSubOpLowering =
    BinaryOpLowering<SubOp, ArithmeticOpBuilder<mlir::SubIOp, mlir::SubFOp>,
                     false>;
using ArrowConstSumOpLowering =
    BinaryOpLowering<ConstSumOp,
                     ArithmeticOpBuilder<mlir::AddIOp, mlir::AddFOp>, true>;
using ArrowConstMulOpLowering =
    BinaryOpLowering<ConstMulOp,
                     ArithmeticOpBuilder<mlir::MulIOp, mlir::MulFOp>, true>;
using ArrowConstDivOpLowering = BinaryOpLowering<
    ConstDivOp, ArithmeticOpBuilder<mlir::SignedDivIOp, mlir::DivFOp>, true>;
using ArrowConstSubOpLowering =
    BinaryOpLowering<ConstSubOp,
                     ArithmeticOpBuilder<mlir::SubIOp, mlir::SubFOp>, true>;
using ArrowAndOpLowering =
    BinaryOpLowering<AndOp, ExactOpBuilder<mlir::AndOp>, false>;
using ArrowOrOpLowering =
    BinaryOpLowering<OrOp, ExactOpBuilder<mlir::OrOp>, false>;

struct ArrowToAffineLoweringPass
    : public PassWrapper<ArrowToAffineLoweringPass, FunctionPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::AffineDialect, mlir::StandardOpsDialect>();
  }
  void runOnFunction() final {
    auto function = getFunction();

    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<mlir::AffineDialect, mlir::StandardOpsDialect>();

    target.addIllegalDialect<ArrowDialect>();
    // target.addLegalOp<FilterOp>();
    target.addLegalOp<GetColumnOp>();
    target.addLegalOp<UnwrapArrayOp>();
    target.addLegalOp<UnwrapColumnOp>();
    target.addLegalOp<MakeArrayOp>();
    target.addLegalOp<MakeColumnOp>();
    target.addLegalOp<MakeTableOp>();

    mlir::OwningRewritePatternList patterns;
    patterns.insert<
        ArrowEqOpLowering, ArrowNeqOpLowering, ArrowGeOpLowering,
        ArrowLeOpLowering, ArrowGtOpLowering, ArrowLtOpLowering,
        ArrowConstEqOpLowering, ArrowConstNeqOpLowering, ArrowConstGeOpLowering,
        ArrowConstLeOpLowering, ArrowConstGtOpLowering, ArrowConstLtOpLowering,
        ArrowSumOpLowering, ArrowMulOpLowering, ArrowDivOpLowering,
        ArrowSubOpLowering, ArrowConstSumOpLowering, ArrowConstMulOpLowering,
        ArrowConstDivOpLowering, ArrowConstSubOpLowering, ArrowAndOpLowering,
        ArrowOrOpLowering>(&getContext());

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(applyPartialConversion(getFunction(), target, patterns)))
      signalPassFailure();

    std::vector<mlir::Value> resultArrays;

    function.walk([&](mlir::Operation *op) {
      if (auto returnOp = mlir::dyn_cast<mlir::ReturnOp>(op)) {
        mlir::Value tableResult = returnOp.getOperand(0);
        for (auto col : tableResult.getDefiningOp()->getOperands()) {
          for (auto arr : col.getDefiningOp()->getOperands()) {
            for (auto memref : arr.getDefiningOp()->getOperands()) {
              resultArrays.push_back(memref);
            }
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
};
std::unique_ptr<Pass> createLowerToAffinePass() {
  return std::make_unique<ArrowToAffineLoweringPass>();
}
} // namespace arcise::dialects::arrow
