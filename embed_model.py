import os
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials



def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {
            "input_text": True
        },
    }
    load_dotenv()
    
    watsonx_API = os.getenv('WATSONX_APIKEY')
    project_id = os.getenv('PROJECT_ID')
    url = 'https://eu-de.ml.cloud.ibm.com'
    
    watsonx_embedding = Embeddings(
        model_id=EmbeddingTypes.IBM_SLATE_125M_ENG,
        credentials=Credentials(
            api_key = watsonx_API,
            url = url),
        project_id=project_id,
        params=embed_params,
    )
    return watsonx_embedding


