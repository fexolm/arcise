#include "arcise/dialects/relalg/frontends/sql/AST.h"

namespace arcise::dialects::relalg::frontends::sql {
void ColumnLit::visit(SqlVisitor &visitor) const { visitor.visit(*this); }
void IntLit::visit(SqlVisitor &visitor) const { visitor.visit(*this); }
void RealLit::visit(SqlVisitor &visitor) const { visitor.visit(*this); }
void StrLit::visit(SqlVisitor &visitor) const { visitor.visit(*this); }
void BoolLit::visit(SqlVisitor &visitor) const { visitor.visit(*this); }
void PredicateExpr::visit(SqlVisitor &visitor) const { visitor.visit(*this); }
void CompareExpr::visit(SqlVisitor &visitor) const { visitor.visit(*this); }
void ArithmeticOp::visit(SqlVisitor &visitor) const { visitor.visit(*this); }
void ProjectionNode::visit(SqlVisitor &visitor) const { visitor.visit(*this); }
void FilterNode::visit(SqlVisitor &visitor) const { visitor.visit(*this); }
void JoinNode::visit(SqlVisitor &visitor) const { visitor.visit(*this); }
void SelectNode::visit(SqlVisitor &visitor) const { visitor.visit(*this); }
void TableScanNode::visit(SqlVisitor &visitor) const { visitor.visit(*this); }

void SqlVisitor::visit(const ColumnLit &) {}
void SqlVisitor::visit(const IntLit &) {}
void SqlVisitor::visit(const RealLit &) {}
void SqlVisitor::visit(const StrLit &) {}
void SqlVisitor::visit(const BoolLit &) {}
void SqlVisitor::visit(const PredicateExpr &) {}
void SqlVisitor::visit(const CompareExpr &) {}
void SqlVisitor::visit(const ArithmeticOp &) {}
void SqlVisitor::visit(const ProjectionNode &) {}
void SqlVisitor::visit(const FilterNode &) {}
void SqlVisitor::visit(const JoinNode &) {}
void SqlVisitor::visit(const SelectNode &) {}
void SqlVisitor::visit(const TableScanNode &) {}

ColumnLit::ColumnLit(const std::string &name) : name(name) {}
IntLit::IntLit(int64_t value) : value(value) {}
RealLit::RealLit(double value) : value(value) {}
BoolLit::BoolLit(bool value) : value(value) {}
TableScanNode::TableScanNode(const std::string &name) : table_name(name) {}

} // namespace arcise::dialects::relalg::frontends::sql