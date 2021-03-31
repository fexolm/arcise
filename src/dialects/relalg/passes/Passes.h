#pragma once

#include "mlir/Pass/Pass.h"

#include <memory>
namespace arcise::dialects::relalg {
std::unique_ptr<mlir::Pass> createLowerToAffinePass();
std::unique_ptr<mlir::Pass> createSplitColumnarOpsPass();
std::unique_ptr<mlir::Pass> createSimplifyRedundantWrappingPass();
std::unique_ptr<mlir::Pass> createMoveAllocationsOnTopPass();
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
} // namespace arcise::dialects::relalg