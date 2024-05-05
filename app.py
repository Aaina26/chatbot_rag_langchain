import json
import os
import sys
import streamlit as st

#for embedding model and llm
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama 

#for Data ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

#Vector embedding and vector store
from langchain_community.vectorstores import FAISS

#llm model
from langchain.prompts import PromptTemplate
from langchain.chains import retrieval_qa

#Olamma embedding
Ollama_embeddings=OllamaEmbeddings()

def data_ingestion():
    loader=PyPDFLoader("resume.pdf")
    documents=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
    docs=text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        Ollama_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_llama2_llm():
    llm=Ollama(model="llama2")
    return llm

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end. Ifyou don't know the answer,
just say you don't know, don't try to make up an answer
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = retrieval_qa.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    
    st.header("")

    user_question = st.text_input("Ask a Question from the resume.")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")
    
    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            #faiss_index = FAISS.load_local("faiss_index", Ollama_embeddings)
            llm=get_llama2_llm()
            docs=data_ingestion()
            faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()


