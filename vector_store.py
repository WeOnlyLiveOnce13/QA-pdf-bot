from langchain_community.vectorstores import Chroma
from embed_model import watsonx_embedding


def vector_database(chunks):
    embedding_model = watsonx_embedding()  
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

