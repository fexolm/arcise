#pragma once

#include "mlir/Pass/Pass.h"

#include <memory>
namespace arcise::dialects::arrow {
std::unique_ptr<mlir::Pass> createLowerToAffinePass();
std::unique_ptr<mlir::Pass> createSplitColumnarOpsPass();
} // namespace arcise::dialects::arrow