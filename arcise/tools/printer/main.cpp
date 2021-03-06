#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"

#include "arcise/dialects/arrow/ArrowDialect.h"
#include "arcise/dialects/arrow/ArrowOps.h"
#include "arcise/dialects/arrow/ArrowTypes.h"
#include "arcise/dialects/arrow/passes/Passes.h"
#include <iostream>

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Target/LLVMIR.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"

int dumpLLVMIR(mlir::ModuleOp module) {
  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(3, 0, nullptr);

  auto maybeEngine =
      mlir::ExecutionEngine::create(module, nullptr, optPipeline);

  assert(maybeEngine && "failed to construct an execution engine");

  auto &engine = maybeEngine.get();

  // auto invocationResult = engine->invoke("main");

  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}

int main(int argc, char **argv) {

  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "arcise\n");

  namespace AD = arcise::dialects::arrow;
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<AD::ArrowDialect>();

  mlir::MLIRContext ctx;

  ctx.printStackTraceOnDiagnostic(true);
  ctx.printOpOnDiagnostic(true);
  registry.loadAll(&ctx);

  mlir::PassManager pm(&ctx, mlir::OpPassManager::Nesting::Implicit);
  pm.enableVerifier();
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(AD::createLowerToAffinePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  // pm.addPass(AD::createMoveAllocationsOnTopPass());
  pm.addPass(mlir::createMemRefDataFlowOptPass());
  pm.addPass(mlir::createSuperVectorizePass());
  pm.addPass(AD::createLowerToLLVMPass());

  mlir::OpBuilder builder(&ctx);

  auto loc = builder.getUnknownLoc();

  mlir::ModuleOp module = mlir::ModuleOp::create(loc);

  mlir::Type I1Array = builder.getType<AD::ArrayType>(builder.getI1Type());
  mlir::Type I64Array = builder.getType<AD::ArrayType>(builder.getI64Type());

  auto rbType = AD::RecordBatchType::get(&ctx, {"a", "b", "c"},
                                         {I64Array, I64Array, I64Array});

  auto outputType = AD::RecordBatchType::get(&ctx, {"res"}, {I1Array});

  auto func = builder.getFunctionType({}, {outputType});

  auto funcOp = builder.create<mlir::FuncOp>(loc, "f", func);

  auto block = funcOp.addEntryBlock();

  builder.setInsertionPointToStart(block);

  auto rb = builder.create<AD::GetRecordBatchOp>(loc, rbType, "col1");

  auto c1 = builder.create<AD::FetchColumnOp>(loc, I64Array, rb, "a");

  auto c2 = builder.create<AD::FetchColumnOp>(loc, I64Array, rb, "b");

  auto c3 = builder.create<AD::FetchColumnOp>(loc, I64Array, rb, "c");

  mlir::Value res = builder.create<AD::SumOp>(loc, I64Array, c1, c2);

  res = builder.create<AD::MulOp>(loc, I64Array, res, c3);

  res = builder.create<AD::GeOp>(
      loc, I1Array, res,
      builder.create<mlir::ConstantOp>(loc, builder.getI64Type(),
                                       builder.getI64IntegerAttr(5)));

  mlir::Value output = builder.create<AD::MakeRecordBatchOp>(
      loc, outputType,
      builder.create<AD::GetLengthOp>(loc, builder.getIndexType(), res), res);

  builder.create<mlir::ReturnOp>(loc, output);

  module.push_back(funcOp);

  mlir::applyPassManagerCLOptions(pm);

  if (mlir::failed(pm.run(module)))
    std::cerr << "FAIL" << std::endl;

  dumpLLVMIR(module);
  return 0;
}
