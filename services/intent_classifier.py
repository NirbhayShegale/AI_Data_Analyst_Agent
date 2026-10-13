
import json

from langchain_core.messages import HumanMessage, SystemMessage


VALID_INTENTS = {"table", "answer", "both", "general"}


def classify_intent(question: str, llm) -> dict:
    """
    Decide which output the user wants.

    Returns:
        {"intent": "table" | "answer" | "both" | "general",
         "reason": "..."}
    """

    system_prompt = """
You are an intent classifier for an AI Data Analyst Agent.

Classify the user's request into exactly one intent:

1. table
   The user mainly wants records, rows, a list, a data table,
   top-N results, or column information.

2. answer
   The user wants a numerical answer, explanation, summary,
   calculation, trend interpretation, or insight without
   explicitly requesting a result table.

3. both
   The user explicitly wants both a table/list of results
   AND an explanation, or asks an analytical question that
   requires an explanation supported by a breakdown table.

4. general
   The user asks a greeting, asks what the agent can do, or
   asks a general question that does not require their dataset.

Rules:
- "Show me the top 20 sales records" -> table
- "What is the total sales?" -> answer
- "Why are sales falling? Show affected regions" -> both
- "Hello, what can you do?" -> general
- Asking for SQL alone counts as table.
- Do not generate SQL or answer the data question.
- Return valid JSON only, with keys "intent" and "reason".
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ])

    raw = response.content.strip()

    # Handle models that wrap JSON in Markdown fences.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Safe fallback for unexpected model output.
        return {
            "intent": "answer",
            "reason": "Defaulted because intent classification was invalid.",
        }

    intent = result.get("intent")

    if intent not in VALID_INTENTS:
        return {
            "intent": "answer",
            "reason": "Defaulted because the intent was unsupported.",
        }

    return {
        "intent": intent,
        "reason": str(result.get("reason", "")),
    }
