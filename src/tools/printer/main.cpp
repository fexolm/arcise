#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"

#include "dialects/arrow/ArrowDialect.h"
#include "dialects/arrow/ArrowOps.h"
#include "dialects/arrow/ArrowTypes.h"
#include "dialects/arrow/passes/Passes.h"
#include <iostream>

int main(int argc, char **argv) {

  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "arcise\n");

  namespace AD = arcise::dialects::arrow;
  mlir::registerAllDialects();
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  registry.insert<AD::ArrowDialect>();
  registry.insert<mlir::StandardOpsDialect>();

  mlir::MLIRContext ctx;

  ctx.printStackTraceOnDiagnostic(true);
  ctx.printOpOnDiagnostic(true);
  registry.loadAll(&ctx);

  mlir::PassManager pm(&ctx, true);
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  pm.addPass(AD::createSplitColumnarOpsPass());
  pm.addPass(AD::createLowerToAffinePass());

  pm.addPass(mlir::createLoopFusionPass());
  pm.addPass(mlir::createMemRefDataFlowOptPass());

  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // pm.addPass(mlir::createAffineParallelizePass());
  pm.addPass(mlir::createSuperVectorizePass(256));

  mlir::OpBuilder builder(&ctx);

  auto loc = builder.getUnknownLoc();

  mlir::ModuleOp module = mlir::ModuleOp::create(loc);

  std::vector<size_t> sizes = {123412, 12342, 43223};

  std::vector<AD::ArrayType> resArrays;
  std::vector<AD::ArrayType> colArrays;

  for (auto size : sizes) {
    resArrays.push_back(
        builder.getType<AD::ArrayType>(builder.getI1Type(), size));
    colArrays.push_back(
        builder.getType<AD::ArrayType>(builder.getI64Type(), size));
  }

  auto resType =
      builder.getType<AD::ColumnType>(builder.getI1Type(), resArrays);

  auto columnsType =
      builder.getType<AD::ColumnType>(builder.getI64Type(), colArrays);

  auto func = builder.getFunctionType({}, {});

  auto funcOp = builder.create<mlir::FuncOp>(loc, "f", func);

  auto block = funcOp.addEntryBlock();

  builder.setInsertionPointToStart(block);

  auto c1 = builder.create<AD::GetColumnOp>(loc, columnsType, "a");

  auto c2 = builder.create<AD::GetColumnOp>(loc, columnsType, "b");

  auto c3 = builder.create<AD::GetColumnOp>(loc, columnsType, "c");

  mlir::Value res = builder.create<AD::SumOp>(loc, columnsType, c1, c2);

  res = builder.create<AD::ConstMulOp>(
      loc, columnsType, res,
      builder.create<mlir::ConstantOp>(loc, builder.getI64Type(),
                                       builder.getI64IntegerAttr(5)));

  res = builder.create<AD::GeOp>(loc, resType, c3, res);

  builder.create<AD::ReturnColumnOp>(loc, res, "res");

  builder.create<mlir::ReturnOp>(loc);

  module.push_back(funcOp);

  mlir::applyPassManagerCLOptions(pm);

  if (mlir::failed(pm.run(module)))
    std::cerr << "FAIL" << std::endl;

  return 0;
}