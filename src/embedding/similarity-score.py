import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv(override=True)

AI_SERVER_URL = os.getenv("AI_SERVER_URL")

llm = OpenAIEmbeddings(
    base_url=AI_SERVER_URL,
    check_embedding_ctx_length=False,  # Important
)

res1 = input("Enter first phrase:\n")
res2 = input("Enter second phrase:\n")

vec1 = llm.embed_query(
    text=res1,
)
vec2 = llm.embed_query(
    text=res2,
)

print("Similarity Score= ", np.dot(vec1, vec2))
