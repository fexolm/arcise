#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "dialects/arrow/ArrowDialect.h"
#include "dialects/arrow/ArrowTypes.h"

int main(int argc, char **argv) {
  mlir::registerAllDialects();
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  registry.insert<arcise::dialects::ArrowDialect>();
  registry.insert<mlir::StandardOpsDialect>();

  mlir::MLIRContext ctx;

  registry.loadAll(&ctx);

  mlir::OpBuilder builder(&ctx);

  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

  auto input =
      builder.getType<arcise::dialects::ArrayType>(builder.getI64Type());

  auto func = builder.getFunctionType({input}, {});

  auto funcOp = mlir::FuncOp::create(builder.getUnknownLoc(), "main", func);

  auto block = funcOp.addEntryBlock();

  builder.setInsertionPointToStart(block);

  auto returnOp = builder.create<mlir::ReturnOp>(builder.getUnknownLoc());

  module.push_back(funcOp);

  module.dump();
  return 0;
}