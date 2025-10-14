import json
from langchain_core.tools import tool

from src.config import RAGConfig
from src.rag.vector_store import ChromaDBVectorStore




chroma_db_vector_store = ChromaDBVectorStore()




@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """
    Retrieve information related to a query.

    You have the tool `retrieve`. Use `retrieve` in the following circumstances:
        - User is asking about some term you are totally unfamiliar with (it might be new).
        - User explicitly asks you to browse or provide links to references.

    Given a query that requires retrieval, you call the search function to get a list of results.
    """
    if query == "__NONE__":
        return "No results found.", []

    retrieved_docs = chroma_db_vector_store.vector_store.similarity_search(query, k = RAGConfig.retrieval_k)

    if not retrieved_docs:
        return "No results found.", []

    serialized = "\n\n".join(
        (f"{doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs




# 사용할 도구 리스트
TOOLS_LIST = [retrieve, ]
TOOLS_DICT = [
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": "You have the tool `retrieve`. Use `retrieve` in the following circumstances:\n - User is asking about some term you are totally unfamiliar with (it might be new).\n - User explicitly asks you to browse or provide links to references.\n\n Given a query that requires retrieval, you call the retrieve function to get a list of results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A search query composed of the essential keywords from the user's question. For example: 'Tell me the name of the largest bird' -> 'the largest bird'"
                    }
                },
                "required": ["query"]
            }
        }
    },
]


def tool_dict_bind():
    tool_prompt = "\n\nTools:\n\nGiven the following functions, respond to the user's prompt.\n\nFirst, determine if a function call is necessary.\n\nIf a function call is NOT necessary:\n - Respond directly to the user. Do not use JSON.\n\nIf a function call is necessary:\n - Respond ONLY with a JSON for the function call with its proper arguments.\n - Respond in the format {\"name\": function name, \"parameters\": {\"argument name\": \"value\"}}. Do not use variables.\n\n"

    for dictionary in TOOLS_DICT:
        tool_prompt += json.dumps(dictionary, indent=4)
        tool_prompt += "\n\n"

    return tool_prompt