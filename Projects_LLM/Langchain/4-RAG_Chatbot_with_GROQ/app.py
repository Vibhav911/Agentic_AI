import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

# Load the GROQ API KEY
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(model="gemma2-9b-it", groq_api_key = groq_api_key)

# Creating a prompt
prompt = ChatPromptTemplate.from_template(
    """
        Answer the question based on the provided context only.
        Please provide the most accurate response based on the questions
        <context>
        {context}
        Question: {input}
    """
)

st.title("RAG Chatbot with GROQ")
def create_vector_embeddings():
    if "vectors" not in st.session_state:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLm-L6-v2")
        st.session_state.embeddings = embeddings
        st.session_state.loader = PyPDFDirectoryLoader("research_papers") # Data Ingestion Step
        st.session_state.docs = st.session_state.loader.load() # Document Loading Step
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # Text Splitting Step
        st.session_state.final_document = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_document, st.session_state.embeddings)
        

user_prompt = st.text_input("Enter your query from the Research Paper")

if st.button("Document Embeddings"):
    create_vector_embeddings()
    st.write("Vector Database is ready")
    
if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({"input":user_prompt})
    st.write(response['answer'])
    
    # Streamlit Expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-------------------------------")

