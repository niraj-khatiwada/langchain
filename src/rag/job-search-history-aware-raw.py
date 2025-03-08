# Raw version, without use of Langchain confusing retrieval chains
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories.file import FileChatMessageHistory
from langchain.globals import set_debug

set_debug(True)

load_dotenv(override=True)

if __name__ == "__main__":
    AI_SERVER_URL = os.getenv("AI_SERVER_URL")
    VECTOR_DB_URL = os.getenv("VECTOR_DB_URL")

    llm = ChatOpenAI(
        base_url=AI_SERVER_URL,
        api_key=".",
    )

    embedding = OpenAIEmbeddings(
        base_url=AI_SERVER_URL,
        check_embedding_ctx_length=False,  # Important
        api_key=".",
    )

    prompt_template = ChatPromptTemplate.from_messages(
        messages=[
            (
                "system",
                """You're an assistant that is designed to answer questions for the given context only. 
                If certain data from the context is not relevant with the question asked, filter them out and exclude them in the response.
                Here's the context: {context}.
                -------------------------------------------------
                Always return the value in plain text with no formatting.
                """,  # The key {context} is used internally so should be present as exact key indicating the document context
            ),
            (
                "system",
                "Now, here's the history of the messages for this chat:",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "{input}",
            ),
        ]
    )

    vector_db = QdrantClient(url=VECTOR_DB_URL)

    chat_history = FileChatMessageHistory(
        file_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "job-chat.history.json"
        ),
    )

    while True:
        question = input("Ask anything?\n")
        hits = vector_db.query_points(
            collection_name="job",
            query=embedding.embed_query(question),
        ).points
        print(hits)
        matches = []
        for hit in hits:
            matches.append(hit.payload["page_content"])
        prompt = prompt_template.format(
            context=str(matches),
            input=question,
            chat_history=chat_history.messages,
        )
        res = llm.invoke(
            input=prompt,
        )
        print(res)
        chat_history.add_user_message(question)
        chat_history.add_ai_message(res.content)
