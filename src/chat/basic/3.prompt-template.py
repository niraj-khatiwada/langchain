import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.globals import set_debug
from langchain.prompts import PromptTemplate

set_debug(True)

load_dotenv(override=True)

AI_SERVER_URL = os.getenv("AI_SERVER_URL")

chat = ChatOpenAI(base_url=AI_SERVER_URL)

template = PromptTemplate(
    template='What do we call "hello" in {language}?', input_variables=["language"]
)

language = input("Enter your language:\n")

res = chat.invoke(input=template.format(language=language))

print(res.content)
