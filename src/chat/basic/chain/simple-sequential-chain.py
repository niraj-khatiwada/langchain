import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.globals import set_debug
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

set_debug(True)

load_dotenv(override=True)

AI_SERVER_URL = os.getenv("AI_SERVER_URL")

chat = ChatOpenAI(base_url=AI_SERVER_URL)

topic_template = PromptTemplate(
    template="""
    You're an experienced story teller. You need to craft an impactful title for creating a story on the topic {topic}.
    Answer exactly with one title.
    """,
    input_variables=["topic"],
)

story_template = PromptTemplate(
    template="""
    Write a powerful story on {title} in no more than 250 words.
    """,
    input_variables=["title"],
)


topic_chain = (
    topic_template | chat | StrOutputParser() | (lambda title: ({"title": title}))
)
story_chain = story_template | chat
chain = topic_chain | story_chain

topic = input("In what topic do you want to craft your story?:\n")

res = chain.invoke(input={"topic": topic})

print(res.content)
