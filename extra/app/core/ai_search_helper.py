import os
import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery


OPENAI_API_TYPE = os.environ.get("OPENAI_API_TYPE")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION")
OPENAI_API_MODEL = os.environ.get("OPENAI_API_MODEL")
AZURE_AI_SEARCH_ENDPOINT = os.environ.get("AZURE_AI_SEARCH_ENDPOINT")
AZURE_AI_SEARCH_KEY = os.environ.get("AZURE_AI_SEARCH_KEY")
AZURE_AI_SEARCH_INDEX = os.environ.get("AZURE_AI_SEARCH_INDEX")
NO_OF_CONTEXT = os.environ.get("NO_OF_CONTEXT")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
openai.api_type = OPENAI_API_TYPE
openai.api_key = OPENAI_API_KEY
openai.api_version = OPENAI_API_VERSION
openai.api_base = AZURE_OPENAI_ENDPOINT


def get_search_connection():
    credential = AzureKeyCredential(AZURE_AI_SEARCH_KEY)
    client = SearchClient(
        endpoint=AZURE_AI_SEARCH_ENDPOINT,
        index_name=AZURE_AI_SEARCH_INDEX,
        credential=credential,
    )
    return client



def search_documents(query):
    client = get_search_connection()
    try:
        embedding = (
            openai.embeddings.create(input=query, model="text-embedding-ada-002")
            .data[0]
            .embedding
        )
    except Exception as e:
        raise e
    vector_query = VectorizedQuery(
        vector=embedding,
        k_nearest_neighbors=NO_OF_CONTEXT,
        fields="textVector",
        exhaustive=True,
    )
    results = client.search(
        search_text=query,
        vector_queries=[vector_query],
        select=["uid", "content"],
        top=NO_OF_CONTEXT,
    )
    client.close()
    return results


