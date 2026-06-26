import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

# Setting up the Streamlit app
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant")
st.title("Text to Math Problem Solver using Google Gemma 2")

# Taking the Groq API Key
groq_api_key = st.sidebar.text_input(" Enter your Groq API Key", type="password")

if not groq_api_key:
    st.info("Please enter your Groq API Key to continue")
    st.stop()

# Groq Model
llm = ChatGroq(api_key=groq_api_key, model="gemma2-9b-it")

# Initializing the Tools
wiki_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    description="Use this tool to search Wikipedia for information on the topics mentioned",
    func=wiki_wrapper.run
)

# Initialize the math tool
math_chain = LLMMathChain(llm=llm)
calculator = Tool(
    name="Calculator",
    description="Use this tool to calculate the answer to the math relatedquestion. Only Mathematical input needs to be provided",
    func=math_chain.run
)

prompt= """
You are a Math Problem Solver. You are given a question and you need to solve it using the tools provided.
Logically break down the question into smaller steps and solve it step by step.

Question: {question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# Combining all tools into a chain
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning_Tool",
    description="A tool for anwering logic-based and reasoning based questions",
    func=chain.run
)

# Initialize the agent
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parse_errors=True,
)

if "messages" not in st.session_state:
    st.session_state.messages=[
        {
            "role": "assistant",
            "content": "Hi, I'm the Math Problem Solver who can answer all your math questions. How can I help you today?"
        }
    ]

# Display the chat history
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])
    

# Let's start the interaction
question = st.text_area("Enter your question here:", "How many cookies did you sell if you sold 320 chocolate cookies and 270 vanilla cookies?")

if st.button("Find the answer"):
    if question:
        with st.spinner("Solving the problem..."):
            st.session_state.messages.append({"role": "user","content": question})
            st.chat_message("user").write(question)
            
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write("Response")
            st.success(response)
    else:
        st.warning("Please enter a question to solve")
            
        
        
        