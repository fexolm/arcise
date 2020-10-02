#include "dialects/arrow/ArrowDialect.h"
#include "dialects/arrow/ArrowOps.h"
#include "dialects/arrow/ArrowTypes.h"
#include "dialects/arrow/passes/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace arcise::dialects::arrow {
static mlir::MemRefType arrayToMemRef(ArrayType type) {
  return mlir::MemRefType::get(type.length(), type.elementType());
}

static mlir::Value castToMemref(mlir::OpBuilder &builder,
                                const mlir::Value &val) {
  if (val.getType().isa<ArrayType>()) {
    auto arrayType = val.getType().cast<ArrayType>();
    return builder.create<CastToMemrefOp>(
        val.getLoc(),
        mlir::MemRefType::get({(int64_t)arrayType.length()},
                              arrayType.elementType()),
        val);
  }
  return val;
}

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
    mlir::Type nestedType = arrayType.elementType();

    auto resultType = (*op->result_type_begin()).cast<ArrayType>();

    auto memRefType = arrayToMemRef(resultType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    mlir::SmallVector<int64_t, 1> lbs(1, 0);
    mlir::SmallVector<int64_t, 1> ubs(1, arrayType.length());
    mlir::SmallVector<int64_t, 1> steps(1, 1);

    typename BinaryOp::Adaptor binaryAdaptor(operands);
    auto lhs = castToMemref(rewriter, binaryAdaptor.lhs());
    auto rhs = castToMemref(rewriter, binaryAdaptor.rhs());
    mlir::buildAffineLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](mlir::OpBuilder &builder, mlir::Location loc,
            mlir::ValueRange ivs) {
          auto loadedLhs = builder.create<mlir::AffineLoadOp>(loc, lhs, ivs);

          mlir::Value loadedRhs;
          if (IsConst) {
            loadedRhs = rhs;
          } else {
            loadedRhs = builder.create<mlir::AffineLoadOp>(loc, rhs, ivs);
          }

          mlir::Value valueToStore = LoweredBinaryOpBuilder::create(
              nestedType.isIntOrIndex(), builder, loc, loadedLhs, loadedRhs);
          auto store_op = builder.create<mlir::AffineStoreOp>(loc, valueToStore,
                                                              alloc, ivs);
        });
    rewriter.replaceOp(op, alloc);
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
    target.addLegalOp<FilterOp>();
    target.addLegalOp<CastToMemrefOp>();

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
  }
};
std::unique_ptr<Pass> createLowerToAffinePass() {
  return std::make_unique<ArrowToAffineLoweringPass>();
}
} // namespace arcise::dialects::arrow
