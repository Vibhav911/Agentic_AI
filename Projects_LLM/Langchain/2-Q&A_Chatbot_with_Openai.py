import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()


# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "Q&A Chatbot with OPENAI"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's queries"),
        ("user", "question:{question}")
    ]
)

def generate_response(question, api_key, engine, temperature, max_tokens):  # temperature - level of creativity from 0 to 1
    llm = ChatOpenAI(model=engine, api_key=api_key)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({"question":question})
    return answer


# Title of the app
st.title("Q&A Chatbot using OpenAI")

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Open AI API Key:", type="password")

# Drop Down to select various Open AI Models
llm = st.sidebar.selectbox("Select OPEN AI Model", ["gpt-4o-mini-2024-07-18", "o1-mini-2024-09-12", "o3-mini-2025-01-31"])

# Adjusting the response Parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens  = st.sidebar.slider("Max Tokens", min_value=50, max_value= 300, value=150)

# Main Interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(question=user_input, api_key=api_key, engine=llm, temperature=temperature, max_tokens=max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")