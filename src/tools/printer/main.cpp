#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "dialects/arrow/ArrowDialect.h"
#include "dialects/arrow/ArrowOps.h"
#include "dialects/arrow/ArrowTypes.h"
#include "dialects/arrow/passes/Passes.h"

#include <iostream>

int main(int argc, char **argv) {
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

  mlir::PassManager pm(&ctx);
  pm.addPass(AD::createLowerToAffinePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createLoopFusionPass());
  pm.addPass(mlir::createMemRefDataFlowOptPass());

  mlir::OpBuilder builder(&ctx);

  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

  auto in_arr = builder.getType<AD::ArrayType>(builder.getI64Type(), 15);

  auto in_val = builder.getI64Type();

  auto res = builder.getType<AD::ArrayType>(builder.getI1Type(), 15);

  auto func = builder.getFunctionType({in_arr, in_val}, {});

  auto funcOp = mlir::FuncOp::create(builder.getUnknownLoc(), "main", func);

  auto block = funcOp.addEntryBlock();

  builder.setInsertionPointToStart(block);

  auto eq1 = builder.create<AD::ConstGeOp>(builder.getUnknownLoc(), res,
                                           funcOp.getArgument(0),
                                           funcOp.getArgument(1));

  auto eq2 = builder.create<AD::ConstGeOp>(builder.getUnknownLoc(), res,
                                           funcOp.getArgument(0),
                                           funcOp.getArgument(1));

  builder.create<AD::OrOp>(builder.getUnknownLoc(), res, eq1, eq2);

  auto returnOp = builder.create<mlir::ReturnOp>(builder.getUnknownLoc());

  module.push_back(funcOp);
  module.verify();

  module.dump();

  if (mlir::failed(pm.run(module)))
    std::cerr << "FAIL" << std::endl;

  module.dump();
  return 0;
}