from vector_store import vector_database
from text_splitter import text_splitter
from pdf_loader import document_loader 

def retriever(file):
    # Load the document
    splits = document_loader(file)
    
    # Split the document into chunks
    chunks = text_splitter(splits)
    
    # Create a vector database from the chunks
    vectordb = vector_database(chunks)
    
    # Convert the vector database to a retriever
    retriever = vectordb.as_retriever()
    
    return retriever
