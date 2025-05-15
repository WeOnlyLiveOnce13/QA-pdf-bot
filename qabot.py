#from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from llm_init import get_llm
from retriever import retriever

### ---- Suppress warnings from codes --
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')



def retriever_qa(file, query):
    
    load_dotenv()
    
    watsonx_API = os.getenv('WATSONX_APIKEY')
    project_id = os.getenv('PROJECT_ID')
    url = "https://eu-de.ml.cloud.ibm.com"
    
    # Get the LLM
    model = get_llm()
    
    # Get the retriever object
    retriever_obj = retriever(file)
    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    # Set up the QA chain using the LLM and retriever
    qa_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever_obj, qa_chain)


    # Invoke the QA chain with the query
    response = rag_chain.invoke({"input": query})
    
    # Return the result from the response
    return response['answer']







