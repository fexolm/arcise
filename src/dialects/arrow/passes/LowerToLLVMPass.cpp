#include "dialects/arrow/ArrowDialect.h"
#include "dialects/arrow/ArrowOps.h"
#include "dialects/arrow/ArrowTypes.h"
#include "dialects/arrow/passes/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"

// TODO: replace i64 type to index type where neccesary

namespace arcise::dialects::arrow {

static mlir::Optional<mlir::Type>
convertArrayType(mlir::MLIRContext *context, arrow::ArrayType type,
                 mlir::TypeConverter *converter) {
  mlir::Type elementType = type.getElementType();

  auto nullBitmapType =
      mlir::MemRefType::get(-1, mlir::IntegerType::get(1, context));
  auto dataBufferType = mlir::MemRefType::get(-1, elementType);

  auto llvmDataBuffer =
      converter->convertType(dataBufferType).cast<mlir::LLVM::LLVMType>();

  auto llvmNullBitmap =
      converter->convertType(nullBitmapType).cast<mlir::LLVM::LLVMType>();

  auto indexType = mlir::IndexType::get(context);

  auto llvmLengthType =
      converter->convertType(indexType).cast<mlir::LLVM::LLVMType>();

  return mlir::LLVM::LLVMStructType::getLiteral(
      context, {llvmDataBuffer, llvmNullBitmap, llvmLengthType});
}

static mlir::Optional<mlir::Type>
convertRecordBatchType(mlir::MLIRContext *context, arrow::RecordBatchType type,
                       mlir::TypeConverter *converter) {
  mlir::SmallVector<mlir::LLVM::LLVMType, 16> fieldTypes;
  auto indexType = mlir::IndexType::get(context);

  fieldTypes.push_back(
      converter->convertType(indexType).cast<mlir::LLVM::LLVMType>());

  for (auto &t : type.getColumnTypes()) {
    fieldTypes.push_back(
        converter->convertType(t).cast<mlir::LLVM::LLVMType>());
  }

  return mlir::LLVM::LLVMStructType::getLiteral(context, fieldTypes);
}

static void
populateArrowToLLVMTypeConvertions(mlir::MLIRContext *context,
                                   mlir::LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [context, converter = &typeConverter](arrow::ArrayType type) {
        return convertArrayType(context, type, converter);
      });
  typeConverter.addConversion(
      [context, converter = &typeConverter](arrow::RecordBatchType type) {
        return convertRecordBatchType(context, type, converter);
      });
}

class GetColumnOpConvertionPattern
    : public mlir::OpConversionPattern<GetColumnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(GetColumnOp op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto rb = op.recordBatch();
    auto columnName = op.columnName();

    auto rbType = rb.getType().cast<RecordBatchType>();

    int fieldIdx =
        1 + std::distance(rbType.getColumnNames().begin(),
                          std::find(rbType.getColumnNames().begin(),
                                    rbType.getColumnNames().end(), columnName));

    auto resultType = typeConverter->convertType(op.res().getType())
                          .cast<mlir::LLVM::LLVMType>();

    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
        op, resultType, rewriter.getRemappedValue(rb),
        rewriter.getIndexArrayAttr(fieldIdx));

    return mlir::success();
  }
};

class GetDataBufferOpConvertionPattern
    : public mlir::OpConversionPattern<GetDataBufferOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(GetDataBufferOp op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resultType = typeConverter->convertType(op.res().getType())
                          .cast<mlir::LLVM::LLVMType>();

    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
        op, resultType, rewriter.getRemappedValue(op.array()),
        rewriter.getIndexArrayAttr(0));

    return mlir::success();
  }
};

class GetNullBitmapOpConvertionPattern
    : public mlir::OpConversionPattern<GetNullBitmapOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(GetNullBitmapOp op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resultType = typeConverter->convertType(op.res().getType())
                          .cast<mlir::LLVM::LLVMType>();

    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
        op, resultType, rewriter.getRemappedValue(op.array()),
        rewriter.getIndexArrayAttr(1));

    return mlir::success();
  }
};

class GetLengthOpConvertionPattern
    : public mlir::OpConversionPattern<GetLengthOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(GetLengthOp op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resultType = typeConverter->convertType(op.res().getType())
                          .cast<mlir::LLVM::LLVMType>();

    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
        op, resultType, rewriter.getRemappedValue(op.array()),
        rewriter.getIndexArrayAttr(2));

    return mlir::success();
  }
};

class GetRowsCountOpConvertionPattern
    : public mlir::OpConversionPattern<GetRowsCountOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(GetRowsCountOp op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resultType = typeConverter->convertType(op.res().getType())
                          .cast<mlir::LLVM::LLVMType>();

    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
        op, resultType, rewriter.getRemappedValue(op.recordBatch()),
        rewriter.getIndexArrayAttr(0));

    return mlir::success();
  }
};

class MakeArrayOpConvertionPattern
    : public mlir::OpConversionPattern<MakeArrayOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(MakeArrayOp op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto nullBitmap = op.null_bitmap();
    auto dataBuf = op.data_buffer();

    auto len = rewriter.create<mlir::DimOp>(op.getLoc(), dataBuf, 0);

    auto llvmArrayType = typeConverter->convertType(op.res().getType());

    mlir::Value res =
        rewriter.create<mlir::LLVM::UndefOp>(op.getLoc(), llvmArrayType);
    res = rewriter.create<mlir::LLVM::InsertValueOp>(
        op.getLoc(), rewriter.getRemappedValue(res),
        rewriter.getRemappedValue(dataBuf), rewriter.getIndexArrayAttr(0));
    res = rewriter.create<mlir::LLVM::InsertValueOp>(
        op.getLoc(), rewriter.getRemappedValue(res),
        rewriter.getRemappedValue(nullBitmap), rewriter.getIndexArrayAttr(1));
    res = rewriter.create<mlir::LLVM::InsertValueOp>(
        op.getLoc(), rewriter.getRemappedValue(res),
        rewriter.getRemappedValue(len), rewriter.getIndexArrayAttr(2));

    rewriter.replaceOp(op, {res});

    return mlir::success();
  }
};

class MakeRecordBatchOpConvertionPattern
    : public mlir::OpConversionPattern<MakeRecordBatchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(MakeRecordBatchOp op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    auto llvmRBType = typeConverter->convertType(op.res().getType());

    mlir::Value res =
        rewriter.create<mlir::LLVM::UndefOp>(op.getLoc(), llvmRBType);

    res = rewriter.create<mlir::LLVM::InsertValueOp>(
        op.getLoc(), rewriter.getRemappedValue(res),
        rewriter.getRemappedValue(op.length()), rewriter.getIndexArrayAttr(0));

    for (size_t i = 0; i < op.columns().size(); i++) {
      res = rewriter.create<mlir::LLVM::InsertValueOp>(
          op.getLoc(), res, rewriter.getRemappedValue(op.columns()[i]),
          rewriter.getIndexArrayAttr(1 + i));
    }

    rewriter.replaceOp(op, {res});

    return mlir::success();
  }
};

struct ArrowAffineToLLVMLoweringPass
    : public mlir::PassWrapper<ArrowAffineToLLVMLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::AffineDialect, mlir::StandardOpsDialect,
                    mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto ctx = &getContext();

    mlir::LLVMTypeConverter converter(ctx);

    populateArrowToLLVMTypeConvertions(ctx, converter);

    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
    target.addIllegalDialect<ArrowDialect>();

    mlir::OwningRewritePatternList patterns;
    patterns
        .insert<GetColumnOpConvertionPattern, GetDataBufferOpConvertionPattern,
                GetNullBitmapOpConvertionPattern, GetLengthOpConvertionPattern,
                GetRowsCountOpConvertionPattern, MakeArrayOpConvertionPattern,
                MakeRecordBatchOpConvertionPattern>(converter, ctx);

    mlir::populateAffineToStdConversionPatterns(patterns, ctx);
    mlir::populateLoopToStdConversionPatterns(patterns, ctx);
    mlir::populateStdToLLVMConversionPatterns(converter, patterns);

    if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                               std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
  return std::make_unique<ArrowAffineToLLVMLoweringPass>();
}
} // namespace arcise::dialects::arrow