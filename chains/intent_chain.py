
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


VALID_INTENTS = {"table", "answer", "both", "general"}


def create_intent_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You classify questions for an AI Data Analyst Agent.

Choose one intent:

table:
The user wants rows, records, a list, top-N results,
or a data table.

answer:
The user wants a calculation, metric, summary, or
explanation from the uploaded dataset, without asking
to display a table.

both:
The user explicitly wants an explanation and a table.

general:
The user greets the assistant or asks about its
capabilities without requiring dataset analysis.

Return valid JSON only:
{{"intent": "table", "reason": "short explanation"}}

Examples:
"Show the top 10 customers" -> table
"What is average sales?" -> answer
"Compare regions and show a table" -> both
"Hello, what can you do?" -> general

Do not generate SQL or answer the data question.
"""
        ),
        ("human", "{question}"),
    ])

    return prompt | llm | StrOutputParser()


def parse_intent(raw: str) -> dict:
    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "intent": "answer",
            "reason": "Invalid classifier response; used safe fallback.",
        }

    intent = result.get("intent")

    if intent not in VALID_INTENTS:
        return {
            "intent": "answer",
            "reason": "Unsupported intent; used safe fallback.",
        }

    return {
        "intent": intent,
        "reason": str(result.get("reason", "")),
    }
