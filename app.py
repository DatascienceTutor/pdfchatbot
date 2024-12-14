import os
import shutil
import openai
import tiktoken
import chromadb
from dotenv import load_dotenv

from langchain_community.document_loaders import OnlinePDFLoader,UnstructuredPDFLoader,PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter,CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader

import streamlit as st

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    collection_name="chr_collection"
    local_embd_loc="chr_vect_embedding"
    chr_directory=os.path.join(os.getcwd(),local_embd_loc)
    if os.path.exists(chr_directory):
        shutil.rmtree(chr_directory)

    embeddings=OpenAIEmbeddings()
    vectDB=Chroma.from_texts(text_chunks,
                                embeddings,
                                collection_name=collection_name,
                                persist_directory=chr_directory)
    vectDB.persist()
    return vectDB

def main():
    st.set_page_config(page_title="PDF Chatbot Application",page_icon=":books:")
    st.title("PDF Chatbot :books:")
    st.markdown("### Reach out to me at **mvsreejith0@gmail.com** to understand more.")

    prompt=st.chat_input("Ask your question....")

    if "message" not in st.session_state.keys():
        st.session_state['message']=[{"role":"assistant","content":"I can provide answer based on the uploaded documents"}]

    if "raw_text" not in st.session_state:
        st.session_state["raw_text"] = ""
    if "text_chunks" not in st.session_state:
        st.session_state["text_chunks"] = []
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] =None

    if prompt:
        st.session_state['message'].append({"role":"user","content":prompt})

    for msg in st.session_state['message']:
        with st.chat_message(msg['role']):
            st.write(msg['content'])
    with st.sidebar:
        user_api_key = st.text_input("Enter your OpenAI API key", type="password")
        if st.button("Submit"):
            if user_api_key:
                os.environ["OPENAI_API_KEY"] = user_api_key.strip()
                openai.api_key = os.getenv("OPENAI_API_KEY")
                models = openai.models.list()
                model_names=[x.id for x in models.data]
                if len(model_names)>0:
                    st.write("✅ API key is valid!")
                else:
                    st.write("❌ Invalid API key")
            else:
                st.warning("Please enter API key to communicate with Open AI server")
        st.subheader("Your Documents")
        pdf_docs=st.file_uploader("Upload or Drag Your files here",accept_multiple_files=True)
        process=st.button("Process")

    if pdf_docs and process:
        with st.spinner("Processing"):
            st.session_state['raw_text']=get_pdf_text(pdf_docs)
            st.session_state['text_chunks']=get_text_chunks(st.session_state['raw_text'])
            st.session_state['vectorstore']=get_vectorstore(st.session_state['text_chunks'])   
            
    if st.session_state['message'][-1]["role"]!="assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                memory=ConversationBufferMemory(memory_key="chat_history",
                                    return_messages=True)
                chatmodel=ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo",streaming=True)
                chatQA=ConversationalRetrievalChain.from_llm (chatmodel,
                                                            st.session_state['vectorstore'].as_retriever(),
                                                            memory=memory
                                                            )
                response=chatQA({"question": prompt, "chat_history": memory.load_memory_variables({})["chat_history"]})
                assistant_message = response["answer"]
                # st.write(assistant_message)
                streamed_response_container = st.empty()
                streamed_content = ""
                for word in response['answer']:
                    streamed_content += word 
                    streamed_response_container.write(streamed_content)
                st.session_state['message'].append({"role": "assistant", "content": assistant_message})

if __name__=="__main__":
    main()