
import hashlib

import streamlit as st

from data_loader import load_dataset, preprocess_dataset

from services.llm_client import get_config, get_llm
from agents.analyst_graph import create_analyst_graph


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="📊",
    layout="wide",
)

st.title("AI Data Analyst Agent")
st.caption(
    "Ask questions about your CSV or Excel dataset using natural language."
)


# --------------------------------------------------
# 2. SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None

if "dataset_df" not in st.session_state:
    st.session_state.dataset_df = None

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None

if "analyst_graph" not in st.session_state:
    st.session_state.analyst_graph = None

if "graph_model" not in st.session_state:
    st.session_state.graph_model = None


# --------------------------------------------------
# 3. INITIALIZE GROQ AND LANGGRAPH
# --------------------------------------------------

try:
    api_key, model_name = get_config()

    llm = get_llm(api_key, model_name)

    # Rebuild the graph only if its model configuration changes.
    if st.session_state.graph_model != model_name:
        st.session_state.analyst_graph = create_analyst_graph(llm)
        st.session_state.graph_model = model_name

except Exception as exc:
    st.error(f"LLM initialization failed: {exc}")
    st.stop()


# --------------------------------------------------
# 4. UPLOAD DATASET
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is None:
    st.info("Upload a dataset to start.")
    st.stop()


# Detect whether a different file was uploaded.
file_hash = hashlib.sha256(
    uploaded_file.getvalue()
).hexdigest()

dataset_id = f"{uploaded_file.name}:{file_hash}"


if st.session_state.dataset_id != dataset_id:
    try:
        with st.spinner("Loading and preprocessing dataset..."):
            original_df = load_dataset(uploaded_file)
            cleaned_df = preprocess_dataset(original_df)

            if cleaned_df is None or cleaned_df.empty:
                st.error(
                    "The dataset is empty after preprocessing."
                )
                st.stop()

            st.session_state.dataset_df = cleaned_df
            st.session_state.dataset_id = dataset_id
            st.session_state.dataset_name = uploaded_file.name

            # Results from a previous dataset must not be mixed
            # with results from a new dataset.
            st.session_state.messages = []

    except Exception as exc:
        st.error(f"Dataset loading failed: {exc}")
        st.stop()


df = st.session_state.dataset_df


# --------------------------------------------------
# 5. DATASET OVERVIEW
# --------------------------------------------------

st.success(f"Dataset loaded: {st.session_state.dataset_name}")

col1, col2, col3 = st.columns(3)

col1.metric("Rows", f"{len(df):,}")
col2.metric("Columns", len(df.columns))
col3.metric(
    "Missing values",
    f"{int(df.isna().sum().sum()):,}",
)

with st.expander("Preview dataset"):
    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Dataset schema"):
    schema_df = df.dtypes.astype(str).reset_index()
    schema_df.columns = ["Column", "Data Type"]

    st.dataframe(
        schema_df,
        use_container_width=True,
        hide_index=True,
    )


schema = "\n".join(
    f"- {column}: {dtype}"
    for column, dtype in df.dtypes.items()
)


# --------------------------------------------------
# 6. RENDER SAVED CHAT HISTORY
# --------------------------------------------------

st.divider()
st.subheader("Chat with your data")

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message.get("content"):
            st.markdown(message["content"])

        if message.get("sql"):
            with st.expander("View SQL"):
                st.code(message["sql"], language="sql")

        result_df = message.get("result_df")

        if result_df is not None:
            if result_df.empty:
                st.info("No matching records were found.")
            else:
                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True,
                )

        if message.get("truncated"):
            st.warning(
                "The displayed results were limited to 1,000 rows."
            )


# --------------------------------------------------
# 7. PROCESS A NEW QUESTION
# --------------------------------------------------

question = st.chat_input("Ask a question about your dataset...")

if question:

    # Store the user query before processing.
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    try:
        with st.spinner("Running the agent graph..."):

            result = st.session_state.analyst_graph.invoke({
                "question": question,
                "df": df,
                "schema": schema,
            })

        intent = result["intent"]
        answer = result.get("answer")
        sql = result.get("sql")
        result_df = result.get("result_df")
        truncated = result.get("truncated", False)

        # Choose the response text and whether to show a table.
        if intent == "table":
            content = "Here are the query results."
            display_df = result_df

        elif intent == "answer":
            content = answer or "No answer was generated."
            display_df = None

        elif intent == "both":
            content = answer or "Here are the query results."
            display_df = result_df

        else:
            content = answer or "How can I help you?"
            display_df = None

        # Save everything needed to recreate this response
        # when Streamlit reruns.
        st.session_state.messages.append({
            "role": "assistant",
            "content": content,
            "sql": sql,
            "result_df": (
                display_df.copy()
                if display_df is not None
                else None
            ),
            "truncated": truncated,
        })

    except Exception as exc:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "The agent could not process this question.\n\n"
                f"**Error:** `{str(exc)}`"
            ),
            "sql": None,
            "result_df": None,
            "truncated": False,
        })

    # Rerender the complete conversation, including older tables.
    st.rerun()


# --------------------------------------------------
# 8. CLEAR CHAT
# --------------------------------------------------

st.divider()

if st.button("Clear chat history"):
    st.session_state.messages = []
    st.rerun()
