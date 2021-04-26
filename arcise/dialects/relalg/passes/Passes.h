#pragma once

#include "mlir/Pass/Pass.h"

#include <memory>

namespace arcise::dialects::relalg {
std::unique_ptr<mlir::Pass> createLowerToArrowPass();
} // namespace arcise::dialects::relalg