
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def create_answer_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an AI Data Analyst.

Answer the user's question using the SQL results provided.

Rules:
- Use only the supplied result data.
- Do not invent numbers, records, or statistics.
- Explain calculations clearly and concisely.
- If no rows matched, state that clearly.
- Distinguish observations from possible explanations.
- Do not claim causation unless the evidence supports it.
- If the results are truncated, mention this when relevant.

Question:
{question}

Executed SQL:
{sql}

Number of returned rows:
{row_count}

Results truncated:
{truncated}

SQL result data in JSON:
{results_json}
"""
        ),
        ("human", "Explain the results for the user's question."),
    ])

    return prompt | llm | StrOutputParser()
