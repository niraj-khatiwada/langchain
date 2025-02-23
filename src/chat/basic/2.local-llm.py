import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

AI_SERVER_URL = os.getenv("AI_SERVER_URL")

chat = ChatOpenAI(base_url=AI_SERVER_URL)

question = input("Enter your question:\n")

res = chat.invoke(input=question)
print(res.content)
