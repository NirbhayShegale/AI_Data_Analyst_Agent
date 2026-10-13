
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage


SYSTEM_PROMPT = """
You are an expert DuckDB SQL analyst.

Convert the user's question into one SQL query for the
uploaded dataset registered as my_table.

Rules:
- Return SQL only, without Markdown fences.
- Generate one read-only SELECT query. WITH ... SELECT is allowed.
- Use only columns in the supplied schema.
- Use DuckDB SQL syntax.
- Do not invent column names or data.
- For top-N questions, use ORDER BY and LIMIT.
- For date questions, use appropriate date filters.
- Use the supplied current date for "this month" and similar phrases.
- If the schema cannot answer the question, return:
  SELECT 'INSUFFICIENT_SCHEMA' AS status
"""


def generate_sql(question: str, schema: str, llm) -> str:
    """Generate SQL from a natural-language question."""
    prompt = f"""
Current date: {date.today().isoformat()}

Available table:
my_table

Schema:
{schema}

User question:
{question}

Generate the SQL query.
"""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    return response.content.strip()
