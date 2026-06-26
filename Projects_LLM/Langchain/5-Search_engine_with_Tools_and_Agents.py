import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
)
from langchain_community.utilities import (
    ArxivAPIWrapper,
    WikipediaAPIWrapper,
)

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# Load env
load_dotenv()

# --- Tools ---
arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
)

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
)

search = DuckDuckGoSearchRun()

tools = [search, arxiv, wiki]

# --- Define State ---
class AgentState(TypedDict):
    messages: List

# --- Tool Executor ---
def tool_node(state: AgentState):
    last_message = state["messages"][-1]

    tool_outputs = []

    for tool in tools:
        try:
            result = tool.run(last_message.content)
            tool_outputs.append(result)
        except Exception:
            continue

    return {
        "messages": state["messages"] + [AIMessage(content="\n".join(tool_outputs))]
    }

# --- LLM Node ---
def llm_node(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

# --- Decide Next Step ---
def should_continue(state: AgentState):
    last_message = state["messages"][-1].content.lower()

    if "search" in last_message or "arxiv" in last_message or "wiki" in last_message:
        return "tools"
    return END

# --- Streamlit UI ---
st.set_page_config(page_title="LangGraph Chatbot", layout="wide")
st.title("🔎 LangGraph Chat with Tools")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

# Session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    st.chat_message(role).write(msg.content)

# Input
user_input = st.chat_input("Ask anything...")

if api_key and user_input:
    # LLM init
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        streaming=True,
    )

    # Build Graph
    graph = StateGraph(AgentState)

    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("llm")

    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )

    graph.add_edge("tools", "llm")

    app = graph.compile()

    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    # Run graph
    with st.chat_message("assistant"):
        result = app.invoke({"messages": st.session_state.messages})

        final_msg = result["messages"][-1]
        st.write(final_msg.content)

        st.session_state.messages = result["messages"]

elif not api_key:
    st.warning("Please enter your Groq API key.")