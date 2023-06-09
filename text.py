
import streamlit as st

import tiktoken
import openai
import os
import time

from langchain.document_loaders import DirectoryLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

flag = False


# 2 parts - first train the model with the file , store it and query again and again

@st.cache_resource  # 👈 Add the caching decorator
def load_model(fileName):
    # start_time = time.time()

    # Save the uploaded file
    with open("user_feedback_100.txt", "wb") as f:
        f.write(fileName.read())

    os.environ["OPENAI_API_KEY"] = anthropic_api_key
    loader = DirectoryLoader("", glob="user_feedback_100.txt")
    txt_docs = loader.load_and_split()

    # Create embeddings
    embeddings = OpenAIEmbeddings()
    # Write in DB
    txt_docsearch = Chroma.from_documents(txt_docs, embeddings)

    # Define LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    # Create Retriever
    # In case answers are cut-off or you get error messages (token limit)
    # use different chain_type
    qa_txt = RetrievalQA.from_chain_type(llm=llm, 
                                        chain_type="stuff",
                                        retriever=txt_docsearch.as_retriever())
    
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"loading model Execution time: {execution_time} seconds")

    return qa_txt


def query(question, model):
    # start_time = time.time()
    # query = "First summarize, then categorize and then priortize what cx needs/requesting/facing issue"
    response = model.run(question)
    # print("hassssss")
    st.write("### Answer")
    st.write(response)

    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"query Execution time: {execution_time} seconds")


with st.sidebar:
    anthropic_api_key = st.text_input('OpenAI API Key',key='file_qa_api_key')

st.title("Vyapar InsightX")
uploaded_file = st.file_uploader("Upload an article", type="txt")
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)


if uploaded_file and question and anthropic_api_key:
    # print("-----beginning ----- ")
    # print(uploaded_file.name)
    query(question, load_model(uploaded_file))


