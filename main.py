import openai
from langchain import SerpAPIWrapper
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI

import streamlit as st
import os

openai.api_key = st.secrets['OPENAI_API_KEY']
GPT_MODEL_VERSION = 'gpt-4'
if 'OPENAI_ORG' in st.secrets:
    openai.organization = st.secrets['OPENAI_ORG']

@st.cache_resource(show_spinner=False)
def setup():
    loader = PyPDFLoader("Resume.pdf")
    docs = loader.load_and_split()
    model = load_langchain_model(docs)
    return model

@st.cache_resource(show_spinner=False)    
def load_langchain_model(_docs):
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(_docs, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
    # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model = GPT_MODEL_VERSION, streaming = True), chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

def query_langchain_model(model, query):
    ans = model({"query": query})
    return ans["result"]

model = setup()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Talk to My Resumé")
st.markdown("Ask anything about my background, experience, skills, etc and this chatbot will try to answer it. Built using chromadb, langchain, and python. If this bot can't answer your question, feel free to ask me :)")

# Display chat messages from history on app rerun

if prompt := st.chat_input("Ask something: "):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = query_langchain_model(model, prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})