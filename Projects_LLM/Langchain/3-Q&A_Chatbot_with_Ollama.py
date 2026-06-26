from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
import streamlit as st

import os
from dotenv import load_dotenv
load_dotenv()

# Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "Q&A Chatbot with OLLAMA"

# Prompt Template
prompt = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant. Please respond to the user's queries"),
        ("user", "question:{question}")
    ]
)

def generate_response(question,engine, temperature, max_tokens):
    llm = OllamaLLM(model=engine, temperature= temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({"question":question})
    return answer

# Title of the app
st.title("Q&A Chatbot using OLLAMA")

# Sidebar for settings
st.sidebar.title("Settings")

# Drop Down to select various Open AI Models
engine = st.sidebar.selectbox("Select OLLAMA  Model", ["gemma3:1b", "deepseek-r1:1.5b"])

# Adjusting the response Parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens  = st.sidebar.slider("Max Tokens", min_value=50, max_value= 300, value=150)

# Main Interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, engine, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")
