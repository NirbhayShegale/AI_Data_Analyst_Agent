
from typing import TypedDict, Any

import pandas as pd
from langgraph.graph import StateGraph, START, END

from chains.intent_chain import (
    create_intent_chain,
    parse_intent,
)
from chains.sql_chain import create_sql_chain
from chains.answer_chain import create_answer_chain
from chains.general_chain import create_general_chain

from services.sql_validator import validate_sql
from services.query_executor import execute_query


class AgentState(TypedDict, total=False):
    question: str
    df: Any
    schema: str
    intent: str
    reason: str
    sql: str
    result_df: Any
    truncated: bool
    answer: str


def create_analyst_graph(llm):

    # --------------------------------------------------
    # Create reusable LCEL chains
    # --------------------------------------------------

    intent_chain = create_intent_chain(llm)
    sql_chain = create_sql_chain(llm)
    answer_chain = create_answer_chain(llm)
    general_chain = create_general_chain(llm)

    # --------------------------------------------------
    # Node 1: Classify the user's question
    # --------------------------------------------------

    def classify_node(state: AgentState):
        raw = intent_chain.invoke({
            "question": state["question"],
        })

        classification = parse_intent(raw)

        return {
            "intent": classification["intent"],
            "reason": classification["reason"],
        }

    # --------------------------------------------------
    # Node 2: Generate and validate SQL
    # --------------------------------------------------

    def sql_node(state: AgentState):
        generated_sql = sql_chain.invoke({
            "question": state["question"],
            "schema": state["schema"],
        }).strip()

        # Remove accidental Markdown code fences.
        if generated_sql.startswith("```"):
            generated_sql = (
                generated_sql
                .split("\n", 1)[-1]
                .rsplit("```", 1)[0]
                .strip()
            )

        safe_sql = validate_sql(generated_sql)

        return {"sql": safe_sql}

    # --------------------------------------------------
    # Node 3: Execute SQL
    # --------------------------------------------------

    def execute_node(state: AgentState):
        result_df, truncated = execute_query(
            state["df"],
            state["sql"],
        )

        return {
            "result_df": result_df,
            "truncated": truncated,
        }

    # --------------------------------------------------
    # Node 4: Explain results
    # --------------------------------------------------

    def answer_node(state: AgentState):
        result_df = state["result_df"]

        results_json = result_df.head(100).to_json(
            orient="records",
            date_format="iso",
            force_ascii=False,
        )

        answer = answer_chain.invoke({
            "question": state["question"],
            "sql": state["sql"],
            "row_count": len(result_df),
            "truncated": state["truncated"],
            "results_json": results_json,
        })

        return {"answer": answer.strip()}

    # --------------------------------------------------
    # Node 5: Handle general conversation
    # --------------------------------------------------

    def general_node(state: AgentState):
        answer = general_chain.invoke({
            "question": state["question"],
        })

        return {"answer": answer.strip()}

    # --------------------------------------------------
    # Node 6: Handle insufficient schema
    # --------------------------------------------------

    def insufficient_schema_node(state: AgentState):
        return {
            "answer": (
                "I couldn't answer this question using the available "
                "columns. Please check the dataset schema or ask a "
                "question about fields that exist in your dataset."
            )
        }

    # --------------------------------------------------
    # Routing functions
    # --------------------------------------------------

    def route_intent(state: AgentState) -> str:
        if state["intent"] == "general":
            return "general"

        return "sql"

    def route_sql(state: AgentState) -> str:
        if "INSUFFICIENT_SCHEMA" in state["sql"]:
            return "insufficient"

        return "execute"

    def route_after_execution(state: AgentState) -> str:
        if state["intent"] in {"answer", "both"}:
            return "answer"

        return "end"

    # --------------------------------------------------
    # Build the LangGraph
    # --------------------------------------------------

    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_node)
    graph.add_node("generate_sql", sql_node)
    graph.add_node("execute_sql", execute_node)
    graph.add_node("generate_answer", answer_node)
    graph.add_node("general", general_node)
    graph.add_node("insufficient_schema", insufficient_schema_node)

    graph.add_edge(START, "classify")

    graph.add_conditional_edges(
        "classify",
        route_intent,
        {
            "general": "general",
            "sql": "generate_sql",
        },
    )

    graph.add_conditional_edges(
        "generate_sql",
        route_sql,
        {
            "insufficient": "insufficient_schema",
            "execute": "execute_sql",
        },
    )

    graph.add_conditional_edges(
        "execute_sql",
        route_after_execution,
        {
            "answer": "generate_answer",
            "end": END,
        },
    )

    graph.add_edge("generate_answer", END)
    graph.add_edge("general", END)
    graph.add_edge("insufficient_schema", END)

    return graph.compile()

