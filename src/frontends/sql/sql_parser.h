#pragma once
#include <memory>
#include <string>
#include <vector>

namespace frontends::sql {
struct SqlVisitor;

struct SqlNode {
  virtual void visit(SqlVisitor &visitor) const = 0;
};

struct Expr : SqlNode {};
struct LiteralExpr : virtual Expr {};
struct ArithmeticExpr : virtual Expr {};
struct BooleanExpr : virtual Expr {};

struct ColumnLit : LiteralExpr, ArithmeticExpr, BooleanExpr {
  std::string name;
  ColumnLit() = default;
  ColumnLit(const std::string &name);
  void visit(SqlVisitor &visitor) const override;
};

struct IntLit : ArithmeticExpr, LiteralExpr {
  int64_t value;
  IntLit() = default;
  IntLit(int64_t value);
  void visit(SqlVisitor &visitor) const override;
};

struct RealLit : ArithmeticExpr, LiteralExpr {
  double value;
  RealLit() = default;
  RealLit(double value);
  void visit(SqlVisitor &visitor) const override;
};

struct StrLit : LiteralExpr {
  std::string value;
  void visit(SqlVisitor &visitor) const override;
};

struct BoolLit : BooleanExpr, LiteralExpr {
  bool value;
  BoolLit() = default;
  BoolLit(bool value);
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

struct GroupByNode : SqlNode {
  std::vector<std::shared_ptr<ColumnLit>> fields;
  void visit(SqlVisitor &visitor) const override;
};

struct SelectNode : SqlNode {
  std::shared_ptr<ProjectionNode> projection;
  std::string table_name;
  std::shared_ptr<FilterNode> filter;
  std::shared_ptr<GroupByNode> groub_by;
  void visit(SqlVisitor &visitor) const override;
};

struct SqlVisitor {
  virtual void visit(const ColumnLit &) = 0;
  virtual void visit(const IntLit &) = 0;
  virtual void visit(const RealLit &) = 0;
  virtual void visit(const StrLit &) = 0;
  virtual void visit(const BoolLit &) = 0;
  virtual void visit(const PredicateExpr &) = 0;
  virtual void visit(const CompareExpr &) = 0;
  virtual void visit(const ArithmeticOp &) = 0;
  virtual void visit(const ProjectionNode &) = 0;
  virtual void visit(const FilterNode &) = 0;
  virtual void visit(const GroupByNode &) = 0;
  virtual void visit(const SelectNode &) = 0;
};

const std::shared_ptr<SqlNode> parse_sql(std::string sql);
} // namespace frontends::sql