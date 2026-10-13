
import json

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage


def generate_answer(
    question: str,
    sql: str,
    result_df: pd.DataFrame,
    llm,
    truncated: bool = False,
) -> str:
    """Explain query results using evidence returned by DuckDB."""

    # Limit the context sent to the LLM.
    visible_results = result_df.head(100)

    result_json = visible_results.to_json(
        orient="records",
        date_format="iso",
        force_ascii=False,
    )

    system_prompt = f"""
You are an AI Data Analyst.

Answer the user's question using the actual SQL results below.

Rules:
- Do not invent figures, records, or explanations.
- Explain the important numbers in simple language.
- If the results are empty, say no matching records were found.
- Distinguish observed patterns from possible explanations.
- If the data cannot establish a cause, say so.
- The provided result context contains at most 100 rows.
- Result rows returned to the application were truncated:
  {truncated}
- Never claim that the visible rows are the complete result
  when they are not.

Executed SQL:
{sql}

Number of rows returned to the application: {len(result_df)}

Result data:
{result_json}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ])

    return response.content.strip()
