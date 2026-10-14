
import duckdb
import pandas as pd

from services.sql_validator import validate_sql


MAX_RESULT_ROWS = 1000


def execute_query(
    df: pd.DataFrame,
    sql: str,
) -> tuple[pd.DataFrame, bool]:

    safe_sql = validate_sql(sql)
    connection = duckdb.connect(":memory:")

    try:
        connection.execute("SET enable_external_access = false")
        connection.execute("SET memory_limit = '1GB'")
        connection.execute("SET threads = 2")

        connection.register("my_table", df)

        # Limit output rows, not source rows.
        # Aggregations therefore use the complete dataset.
        limited_sql = (
            f"SELECT * FROM ({safe_sql}) AS agent_result "
            f"LIMIT {MAX_RESULT_ROWS + 1}"
        )

        result_df = connection.execute(limited_sql).df()

        truncated = len(result_df) > MAX_RESULT_ROWS
        result_df = result_df.head(MAX_RESULT_ROWS)

        return result_df, truncated

    finally:
        connection.close()
