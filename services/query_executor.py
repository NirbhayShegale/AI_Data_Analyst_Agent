
import duckdb
import pandas as pd

from services.sql_validator import validate_sql


MAX_RESULT_ROWS = 1000


def execute_query(df: pd.DataFrame, sql: str) -> tuple[pd.DataFrame, bool]:
    """
    Execute read-only SQL against the uploaded DataFrame.

    Returns:
        result_df: up to MAX_RESULT_ROWS rows
        truncated: whether additional rows were available
    """
    safe_sql = validate_sql(sql)

    con = duckdb.connect(":memory:")

    try:
        con.execute("SET enable_external_access = false")
        con.execute("SET memory_limit = '1GB'")
        con.execute("SET threads = 2")

        con.register("my_table", df)

        # Limit returned rows without limiting the source dataset
        # or changing aggregate calculations.
        limited_sql = (
            f"SELECT * FROM ({safe_sql}) AS agent_result "
            f"LIMIT {MAX_RESULT_ROWS + 1}"
        )

        result_df = con.execute(limited_sql).df()

        truncated = len(result_df) > MAX_RESULT_ROWS
        result_df = result_df.head(MAX_RESULT_ROWS)

        return result_df, truncated

    finally:
        con.close()
