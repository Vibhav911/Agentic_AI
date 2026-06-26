import streamlit as st
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import psycopg2
from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(page_title="Langchain: Chat with Postgresql DB")
st.title("Langchain: Chat with SQL DB") 

INJECTION_WARNING = """
                    SQL agents can be vulnerable to promopt injections. 
                    Use a DB role with limited permissions.
                    """
                    
api_key = st.sidebar.text_input("Provide your Groq API Key", type="password")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["Use Postgresql Database-Student", "Connect to your SQL Database"]
selected_opt = st.sidebar.radio(label="Choose the Database which you want to chat with",
                                options=radio_opt)

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provide my SQL Host")
    mysql_user = st.sidebar.text_input("Provide my SQL User")
    mysql_password = st.sidebar.text_input("Provide my SQL Password")
    mysql_db = st.sidebar.text_input("Provide my SQL Database")
else:
    db_uri = LOCALDB
    mysql_host = os.getenv("MYSQL_HOST")
    mysql_user = os.getenv("MYSQL_USER")
    mysql_password = os.getenv("MYSQL_PASSWORD")
    mysql_db = os.getenv("MYSQL_DB")
    mysql_port = os.getenv("MYSQL_PORT")



if not db_uri:
    st.info("Please enter the database information and uri")
    
if not api_key:
    st.info("Please enter your Groq API Key")
    
# LLM Model
llm = ChatGroq(api_key=api_key, model_name="gemma2-9b-it", streaming=True)

@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None, mysql_port=None):
    if db_uri == LOCALDB:
        return SQLDatabase(create_engine(f"postgresql+psycopg2://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db}"))
        
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all the MySQL credentials")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))
    
if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db, mysql_port)
    
    
# Toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear Message History"):
    st.session_state['messages'] = [{"role": "assistant",
                                    "content": "Hi, I'm the SQL Agent. How can I assist you today?"
                                    }]
    
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])
    
user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role":"user", "content":user_query})
    st.chat_message("user").write(user_query)
    
    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[streamlit_callback])
        st.session_state.messages.append({"role":"assistant", "content":response})
        st.write(response)
    
    
    
    
    


