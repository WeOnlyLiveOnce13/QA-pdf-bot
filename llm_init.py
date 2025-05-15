import os
from dotenv import load_dotenv
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

from ibm_watsonx_ai import Credentials

def get_llm():
    #model_id = "mistralai/mixtral-8x7b-instruct-v01"
    load_dotenv()
    
    watsonx_API = os.getenv('WATSONX_APIKEY')
    project_id = os.getenv('PROJECT_ID')
    #url = "https://us-south.ml.cloud.ibm.com"
    #"https://eu-de.ml.cloud.ibm.com"
     
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }
    
    
    #mistralai_model = Model(
    #    model_id=ModelTypes.MIXTRAL_8X7B_INSTRUCT_V01,                   
   ##     params=parameters,
    #    credentials=Credentials(
    #               url = url,
     #              api_key = watsonx_API
    #              ),
    #    project_id=project_id
        
    #    model=mistralai_model)#
    
    
    watsonx_llm = WatsonxLLM(
        model_id="mistralai/mixtral-8x7b-instruct-v01",
        url="https://eu-de.ml.cloud.ibm.com",
        apikey=watsonx_API,
        project_id=project_id,
        params=parameters,
    )

    return watsonx_llm
