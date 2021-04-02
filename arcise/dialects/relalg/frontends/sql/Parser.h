#pragma once

#include <memory>
#include <string_view>

namespace arcise::dialects::relalg::frontends::sql {

struct SqlNode;
std::shared_ptr<SqlNode> parse_sql(std::string_view sql, bool debug = false);
} // namespace arcise::dialects::relalg::frontends::sql
