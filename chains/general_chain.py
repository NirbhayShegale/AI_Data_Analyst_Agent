
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def create_general_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a helpful AI Data Analyst Agent.

You can analyze uploaded CSV and Excel datasets,
calculate metrics, generate data tables, and explain patterns.

Answer general questions naturally.
Do not claim to have analyzed the dataset unless analysis
has actually been performed.
"""
        ),
        ("human", "{question}"),
    ])

    return prompt | llm | StrOutputParser()
