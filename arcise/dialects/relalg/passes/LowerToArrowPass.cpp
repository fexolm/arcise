#include "arcise/dialects/arrow/ArrowDialect.h"
#include "arcise/dialects/arrow/ArrowOps.h"
#include "arcise/dialects/arrow/ArrowTypes.h"
#include "arcise/dialects/relalg/RelalgDialect.h"
#include "arcise/dialects/relalg/RelalgOps.h"
#include "arcise/dialects/relalg/RelalgTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace arcise::dialects::relalg {

class GetRecordBatchOpConvertionPattern
    : public mlir::OpConversionPattern<ScanOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ScanOp op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arrow::GetRecordBatchOp>(op,
                                                         op.table_nameAttr());
    return mlir::success();
  }
};

struct RelalgToArrowLoweringPass
    : public mlir::PassWrapper<RelalgToArrowLoweringPass, mlir::FunctionPass> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<arrow::ArrowDialect>();
  }

  void runOnFunction() override {
    auto function = getFunction();

    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<arrow::ArrowDialect, mlir::StandardOpsDialect>();

    mlir::OwningRewritePatternList patterns;
    // patterns.insert<>(&getContext());

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();
  }
};
std::unique_ptr<mlir::Pass> createLowerToArrowPass() {
  return std::make_unique<RelalgToArrowLoweringPass>();
}
} // namespace arcise::dialects::relalg
