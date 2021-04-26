#pragma once

#include <memory>
#include <string>
#include <vector>

namespace arcise::dialects::relalg::frontends::sql {
struct SqlVisitor;

struct SqlNode {
  virtual void visit(SqlVisitor &visitor) const = 0;
  virtual ~SqlNode() = default;
};

struct Expr : SqlNode {};
struct LiteralExpr : virtual Expr {};
struct ArithmeticExpr : virtual Expr {};
struct BooleanExpr : virtual Expr {};

struct ColumnLit : LiteralExpr, ArithmeticExpr, BooleanExpr {
  std::string name;
  ColumnLit() = default;
  explicit ColumnLit(const std::string &name);
  void visit(SqlVisitor &visitor) const override;
};

struct IntLit : ArithmeticExpr, LiteralExpr {
  int64_t value;
  IntLit() = default;
  explicit IntLit(int64_t value);
  void visit(SqlVisitor &visitor) const override;
};

struct RealLit : ArithmeticExpr, LiteralExpr {
  double value;
  RealLit() = default;
  explicit RealLit(double value);
  void visit(SqlVisitor &visitor) const override;
};

struct StrLit : LiteralExpr {
  std::string value;
  void visit(SqlVisitor &visitor) const override;
};

struct BoolLit : BooleanExpr, LiteralExpr {
  bool value;
  BoolLit() = default;
  explicit BoolLit(bool value);
  void visit(SqlVisitor &visitor) const override;
};

struct PredicateExpr : BooleanExpr {
  enum class Op {
    AND,
    OR,
  };
  std::shared_ptr<BooleanExpr> lhs;
  std::shared_ptr<BooleanExpr> rhs;
  Op op;
  void visit(SqlVisitor &visitor) const override;
};

struct CompareExpr : BooleanExpr {
  enum class Op {
    GT,
    LT,
    GE,
    LE,
    EQ,
    NEQ,
  };
  std::shared_ptr<ArithmeticExpr> lhs;
  std::shared_ptr<ArithmeticExpr> rhs;
  Op op;
  void visit(SqlVisitor &visitor) const override;
};

struct ArithmeticOp : ArithmeticExpr {
  enum class Op {
    Sum,
    Sub,
    Div,
    Mul,
  };
  std::shared_ptr<ArithmeticExpr> lhs;
  std::shared_ptr<ArithmeticExpr> rhs;
  Op op;
  void visit(SqlVisitor &visitor) const override;
};

struct ProjectionNode : SqlNode {
  std::vector<std::shared_ptr<Expr>> exprs;
  void visit(SqlVisitor &visitor) const override;
};

struct FilterNode : SqlNode {
  std::shared_ptr<BooleanExpr> expr;
  void visit(SqlVisitor &visitor) const override;
};

struct ScanNode : SqlNode {};

struct JoinNode : SqlNode {
  std::shared_ptr<ColumnLit> lhs;
  std::shared_ptr<ColumnLit> rhs;
  std::shared_ptr<ScanNode> scan;
  void visit(SqlVisitor &visitor) const override;
};

struct TableScanNode : ScanNode {
  std::string table_name;
  TableScanNode() = default;
  explicit TableScanNode(const std::string &name);
  void visit(SqlVisitor &visitor) const override;
};

struct SelectNode : ScanNode {
  std::shared_ptr<ProjectionNode> projection;
  std::shared_ptr<ScanNode> scan;
  std::shared_ptr<FilterNode> filter;
  std::shared_ptr<JoinNode> join;
  void visit(SqlVisitor &visitor) const override;
};

struct SqlVisitor {
  virtual void visit(const ColumnLit &);
  virtual void visit(const IntLit &);
  virtual void visit(const RealLit &);
  virtual void visit(const StrLit &);
  virtual void visit(const BoolLit &);
  virtual void visit(const PredicateExpr &);
  virtual void visit(const CompareExpr &);
  virtual void visit(const ArithmeticOp &);
  virtual void visit(const ProjectionNode &);
  virtual void visit(const FilterNode &);
  virtual void visit(const JoinNode &);
  virtual void visit(const SelectNode &);
  virtual void visit(const TableScanNode &);

  virtual ~SqlVisitor() = default;
};
} // namespace arcise::dialects::relalg::frontends::sql