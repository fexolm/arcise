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
  pm.addPass(AD::createSplitColumnarOpsPass());

  pm.addPass(AD::createLowerToAffinePass());
  pm.addPass(mlir::createLoopFusionPass());

  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  mlir::OpBuilder builder(&ctx);

  auto loc = builder.getUnknownLoc();

  mlir::ModuleOp module = mlir::ModuleOp::create(loc);

  std::vector<size_t> sizes = {128, 128, 64};

  auto resType = builder.getType<AD::ColumnType>(builder.getI1Type(), 3, sizes);

  auto columnsType =
      builder.getType<AD::ColumnType>(builder.getI64Type(), 3, sizes);

  auto func = builder.getFunctionType({}, {});

  auto funcOp = builder.create<mlir::FuncOp>(loc, "f", func);

  auto block = funcOp.addEntryBlock();

  builder.setInsertionPointToStart(block);

  auto c1 = builder.create<AD::GetColumnOp>(loc, columnsType, 1);

  auto c2 = builder.create<AD::GetColumnOp>(loc, columnsType, 2);

  auto c3 = builder.create<AD::GetColumnOp>(loc, columnsType, 3);

  mlir::Value res = builder.create<AD::SumOp>(loc, columnsType, c1, c2);

  res = builder.create<AD::ConstMulOp>(
      loc, columnsType, res,
      builder.create<mlir::ConstantOp>(loc, builder.getI64Type(),
                                       builder.getI64IntegerAttr(5)));

  res = builder.create<AD::GeOp>(loc, resType, c3, res);

  builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
  // auto returnOp =
  //     builder.create<mlir::ReturnOp>(builder.getUnknownLoc(), resType, res);

  module.push_back(funcOp);

  mlir::applyPassManagerCLOptions(pm);

  if (mlir::failed(pm.run(module)))
    std::cerr << "FAIL" << std::endl;

  return 0;
}