
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from services.data_loader import load_dataset, preprocess_dataset

from services.llm_client import get_llm
from services.intent_classifier import classify_intent
from services.sql_generator import generate_sql
from services.sql_validator import validate_sql
from services.query_executor import execute_query
from services.answer_generator import generate_answer


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="📊",
    layout="wide",
)

st.title("AI Data Analyst Agent")
st.caption("Ask questions about your uploaded dataset.")


# --------------------------------------------------
# 2. LLM CONFIGURATION
# --------------------------------------------------

def get_setting(name: str, default: str = "") -> str:
    value = os.getenv(name)

    if value:
        return value

    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


api_key = get_setting("GROQ_API_KEY")
model_name = get_setting("GROQ_MODEL", "qwen/qwen3.8-27b")

if not api_key:
    st.error("Configure GROQ_API_KEY in your .env file or app secrets.")
    st.stop()

llm = get_llm(api_key, model_name)


# --------------------------------------------------
# 3. UPLOAD AND PREPROCESS DATA
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is None:
    st.info("Upload a dataset to start asking questions.")
    st.stop()

try:
    original_df = load_dataset(uploaded_file)
    df = preprocess_dataset(original_df)

    if df.empty or len(df.columns) == 0:
        st.error("The processed dataset is empty.")
        st.stop()

except Exception as exc:
    st.error(f"Could not load or preprocess the dataset: {exc}")
    st.stop()


# --------------------------------------------------
# 4. DATASET OVERVIEW AND SCHEMA
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Rows", f"{len(df):,}")
col2.metric("Columns", len(df.columns))
col3.metric("Missing cells",f"{int(df.isna().sum().sum()):,}",)

with st.expander("Preview dataset"):
    st.dataframe(df.head(10), use_container_width=True)

schema = "\n".join(
    f"- {column!r}: {dtype}"
    for column, dtype in df.dtypes.items()
)

with st.expander("Available columns"):
    st.code(schema)


# --------------------------------------------------
# 5. GENERAL QUESTION HANDLER
# --------------------------------------------------

def answer_general_question(question: str) -> str:
    """Answer greetings and general capability questions."""

    response = llm.invoke([
        SystemMessage(content="""
You are a helpful AI Data Analyst Agent.

Answer greetings and general questions about your capabilities.
Explain that you can analyze the user's uploaded dataset,
return tables, calculate metrics, and explain trends.

Do not claim to have analyzed the dataset unless you actually
performed a data query.
Keep the response concise.
"""),
        HumanMessage(content=question),
    ])

    return response.content.strip()


# --------------------------------------------------
# 6. CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# 7. INTENT-BASED ROUTING
# --------------------------------------------------

question = st.chat_input("Ask a question about your data...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    saved_response = ""

    with st.chat_message("assistant"):
        try:
            # STEP A: Understand what the user wants.
            with st.spinner("Understanding your question..."):
                classification = classify_intent(question, llm)

            intent = classification["intent"]

            # Display only when debugging the application.
            # Remove this expander for a cleaner production UI.
            with st.expander("Request details"):
                st.write("Detected intent:", intent)
                st.write("Reason:", classification["reason"])

            # ------------------------------------------
            # ROUTE 1: GENERAL CONVERSATION
            # ------------------------------------------

            if intent == "general":
                answer = answer_general_question(question)

                st.markdown(answer)
                saved_response = answer

            # ------------------------------------------
            # ROUTES 2, 3 AND 4: DATA QUESTIONS
            # ------------------------------------------

            else:
                # Generate SQL only when data is needed.
                with st.spinner("Generating SQL..."):
                    generated_sql = generate_sql(
                        question,
                        schema,
                        llm,
                    )

                # Validate before execution.
                safe_sql = validate_sql(generated_sql)

                # Execute against the full processed dataset.
                with st.spinner("Analyzing your data..."):
                    result_df, truncated = execute_query(
                        df,
                        safe_sql,
                    )

                # --------------------------------------
                # ROUTE 2: TABLE ONLY
                # --------------------------------------

                if intent == "table":
                    st.subheader("Query results")

                    st.dataframe(
                        result_df,
                        use_container_width=True,
                    )

                    if truncated:
                        st.info(
                            "The result has more rows than displayed."
                        )

                    with st.expander("View generated SQL"):
                        st.code(safe_sql, language="sql")

                    saved_response = (
                        f"**Query results**\n\n"
                        f"{len(result_df)} result rows returned."
                    )

                # --------------------------------------
                # ROUTE 3: ANSWER ONLY
                # --------------------------------------

                elif intent == "answer":
                    with st.spinner("Preparing your answer..."):
                        answer = generate_answer(
                            question=question,
                            sql=safe_sql,
                            result_df=result_df,
                            llm=llm,
                            truncated=truncated,
                        )

                    st.markdown(answer)

                    # SQL is available but not shown by default.
                    with st.expander("View generated SQL"):
                        st.code(safe_sql, language="sql")

                    saved_response = answer

                # --------------------------------------
                # ROUTE 4: BOTH TABLE AND ANSWER
                # --------------------------------------

                elif intent == "both":
                    st.subheader("Analysis")

                    with st.spinner("Interpreting query results..."):
                        answer = generate_answer(
                            question=question,
                            sql=safe_sql,
                            result_df=result_df,
                            llm=llm,
                            truncated=truncated,
                        )

                    st.markdown(answer)

                    st.subheader("Supporting data")

                    st.dataframe(
                        result_df,
                        use_container_width=True,
                    )

                    if truncated:
                        st.info(
                            "The result has more rows than displayed."
                        )

                    with st.expander("View generated SQL"):
                        st.code(safe_sql, language="sql")

                    saved_response = (
                        f"**Analysis**\n\n{answer}\n\n"
                        f"**Supporting data**\n\n"
                        f"{len(result_df)} result rows returned."
                    )

        except Exception as exc:
            st.error(f"Could not complete your request: {exc}")

            saved_response = (
                "I couldn't complete that request. "
                "Please rephrase your question or check the dataset."
            )

    # Save assistant message for future Streamlit reruns.
    st.session_state.messages.append({
        "role": "assistant",
        "content": saved_response,
    })
