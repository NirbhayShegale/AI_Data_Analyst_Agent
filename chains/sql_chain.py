
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def create_sql_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You generate DuckDB SQL for a read-only data analyst.

The uploaded DataFrame is registered as my_table.

Rules:
- Return exactly one SQL query.
- Generate only read-only SELECT queries.
- Use only columns present in the schema.
- Use DuckDB-compatible SQL.
- Never access another table or external file.
- Do not generate INSERT, UPDATE, DELETE, DROP, CREATE, or ALTER.
- Do not wrap the SQL in Markdown.
- Use ORDER BY and LIMIT for top-N questions.
- For a question that cannot be answered from the schema,
  return exactly:
  SELECT 'INSUFFICIENT_SCHEMA' AS status

Schema:
{schema}
"""
        ),
        ("human", "{question}"),
    ])

    return prompt | llm | StrOutputParser()
