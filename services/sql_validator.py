
import sqlglot
from sqlglot import exp


ALLOWED_TABLE = "my_table"

BLOCKED_OPERATIONS = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Command,
)


def validate_sql(sql: str) -> str:
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL query is empty.")

    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except sqlglot.errors.ParseError as exc:
        raise ValueError("Generated SQL could not be parsed.") from exc

    statements = [statement for statement in statements if statement]

    if len(statements) != 1:
        raise ValueError("Only one SQL statement is allowed.")

    query = statements[0]

    if not isinstance(query, exp.Query):
        raise ValueError("Only read-only SELECT queries are allowed.")

    for node in query.walk():
        if isinstance(node, BLOCKED_OPERATIONS):
            raise ValueError("Unsafe SQL operation detected.")

    cte_names = {
        cte.alias_or_name.lower()
        for cte in query.find_all(exp.CTE)
    }

    for table in query.find_all(exp.Table):
        table_name = table.name.lower()

        if table.db or table.catalog:
            raise ValueError("Qualified table names are not allowed.")

        if table_name != ALLOWED_TABLE and table_name not in cte_names:
            raise ValueError(
                f"Access to table '{table.name}' is not allowed."
            )

    return query.sql(dialect="duckdb")
