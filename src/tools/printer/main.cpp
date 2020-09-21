#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "dialects/arrow/ArrowDialect.h"
#include "dialects/arrow/ArrowOps.h"
#include "dialects/arrow/ArrowTypes.h"

int main(int argc, char **argv) {
  namespace AD = arcise::dialects::arrow;
  mlir::registerAllDialects();
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  registry.insert<AD::ArrowDialect>();
  registry.insert<mlir::StandardOpsDialect>();

  mlir::MLIRContext ctx;

  registry.loadAll(&ctx);

  mlir::OpBuilder builder(&ctx);

  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

  auto in_arr = builder.getType<AD::ChunkedArrayType>(builder.getI32Type());

  auto in_val = builder.getI64Type();

  auto res = builder.getType<AD::ChunkedArrayType>(builder.getI1Type());

  auto func = builder.getFunctionType({in_arr, in_val}, {res});

  auto funcOp = mlir::FuncOp::create(builder.getUnknownLoc(), "main", func);

  auto block = funcOp.addEntryBlock();

  builder.setInsertionPointToStart(block);

  auto eq = builder.create<AD::ConstGeOp>(builder.getUnknownLoc(), res,
                                          funcOp.getArgument(0),
                                          funcOp.getArgument(1));

  auto returnOp = builder.create<mlir::ReturnOp>(builder.getUnknownLoc());

  eq.verify();
  module.push_back(funcOp);
  module.verify();

  module.dump();
  return 0;
}