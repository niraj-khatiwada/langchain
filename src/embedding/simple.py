import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv(override=True)

AI_SERVER_URL = os.getenv("AI_SERVER_URL")

llm = OpenAIEmbeddings(
    base_url=AI_SERVER_URL,
    check_embedding_ctx_length=False,  # Important
)

res = input("Ask anything:\n")

vec = llm.embed_query(
    text=res,
)

print(vec)
