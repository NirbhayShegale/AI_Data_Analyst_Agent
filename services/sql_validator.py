
import sqlglot
from sqlglot import exp


ALLOWED_TABLE = "my_table"


def validate_sql(sql: str) -> str:
    """Allow a single read-only query against the uploaded dataset."""

    sql = sql.strip()

    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except sqlglot.errors.ParseError as exc:
        raise ValueError("The generated SQL is invalid.") from exc

    # Exactly one non-empty SQL statement.
    statements = [stmt for stmt in statements if stmt is not None]

    if len(statements) != 1:
        raise ValueError("Only one SQL statement is allowed.")

    query = statements[0]

    # SELECT and SELECT-based CTEs are read-only query structures.
    if not isinstance(query, exp.Query):
        raise ValueError("Only read-only SELECT queries are allowed.")

    # Allow the uploaded table and aliases defined by WITH clauses.
    cte_names = {
        cte.alias_or_name.lower()
        for cte in query.find_all(exp.CTE)
    }

    for table in query.find_all(exp.Table):
        table_name = table.name.lower()

        if table_name != ALLOWED_TABLE and table_name not in cte_names:
            raise ValueError(
                f"Access to table '{table.name}' is not allowed."
            )

        # Disallow database/catalog qualification.
        if table.db or table.catalog:
            raise ValueError("Qualified table names are not allowed.")

    # Reject explicit commands, DDL, and DML even if nested.
    blocked_types = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Create,
        exp.Alter,
        exp.Command,
    )

    for node in query.walk():
        if isinstance(node, blocked_types):
            raise ValueError("Unsafe SQL operation detected.")

    return query.sql(dialect="duckdb")
