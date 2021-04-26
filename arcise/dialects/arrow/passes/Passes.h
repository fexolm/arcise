#pragma once

#include "mlir/Pass/Pass.h"

#include <memory>

namespace arcise::dialects::arrow {
std::unique_ptr<mlir::Pass> createLowerToAffinePass();
std::unique_ptr<mlir::Pass> createSplitColumnarOpsPass();
std::unique_ptr<mlir::Pass> createSimplifyRedundantWrappingPass();
std::unique_ptr<mlir::Pass> createMoveAllocationsOnTopPass();
std::unique_ptr<mlir::Pass> createLowerToArrowPass();
} // namespace arcise::dialects::arrow