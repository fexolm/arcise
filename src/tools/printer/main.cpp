#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
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
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  registry.insert<AD::ArrowDialect>();
  registry.insert<mlir::StandardOpsDialect>();

  mlir::MLIRContext ctx;

  ctx.printStackTraceOnDiagnostic(true);
  ctx.printOpOnDiagnostic(true);
  registry.loadAll(&ctx);

  mlir::PassManager pm(&ctx, true);
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(AD::createLowerToAffinePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(AD::createMoveAllocationsOnTopPass());
  pm.addPass(mlir::createMemRefDataFlowOptPass());

  mlir::OpBuilder builder(&ctx);

  auto loc = builder.getUnknownLoc();

  mlir::ModuleOp module = mlir::ModuleOp::create(loc);

  mlir::Type I1Array = builder.getType<AD::ArrayType>(builder.getI1Type());
  mlir::Type I64Array = builder.getType<AD::ArrayType>(builder.getI64Type());

  auto inputType = AD::RecordBatchType::get(&ctx, {"a", "b", "c"},
                                            {I64Array, I64Array, I64Array});

  auto outputType = AD::RecordBatchType::get(&ctx, {"res"}, {I1Array});

  auto func = builder.getFunctionType({inputType}, {outputType});

  auto funcOp = builder.create<mlir::FuncOp>(loc, "f", func);

  auto block = funcOp.addEntryBlock();

  builder.setInsertionPointToStart(block);

  auto input = funcOp.getArgument(0);

  auto c1 = builder.create<AD::GetColumnOp>(loc, I64Array, input, "a");

  auto c2 = builder.create<AD::GetColumnOp>(loc, I64Array, input, "b");

  auto c3 = builder.create<AD::GetColumnOp>(loc, I64Array, input, "c");

  mlir::Value res = builder.create<AD::SumOp>(loc, I64Array, c1, c2);

  res = builder.create<AD::MulOp>(loc, I64Array, res, c3);

  res = builder.create<AD::GeOp>(
      loc, I1Array, res,
      builder.create<mlir::ConstantOp>(loc, builder.getI64Type(),
                                       builder.getI64IntegerAttr(5)));

  mlir::Value output =
      builder.create<AD::MakeRecordBatchOp>(loc, outputType, res);

  builder.create<mlir::ReturnOp>(loc, output);

  module.push_back(funcOp);

  mlir::applyPassManagerCLOptions(pm);

  if (mlir::failed(pm.run(module)))
    std::cerr << "FAIL" << std::endl;

  return 0;
}