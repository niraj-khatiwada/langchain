import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise Exception("OPENAI_API_KEY not found.")

chat = ChatOpenAI(api_key=OPENAI_API_KEY)

question = input("Enter your question:\n")

res = chat.invoke(input=question)
print(res.content)
