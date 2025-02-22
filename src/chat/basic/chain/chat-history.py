import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.globals import set_debug
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories.file import FileChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

set_debug(True)

load_dotenv(override=True)

AI_SERVER_URL = os.getenv("AI_SERVER_URL")

chat = ChatOpenAI(base_url=AI_SERVER_URL)

question_template = ChatPromptTemplate.from_messages(
    messages=[
        (
            "system",
            "You'll be asked certain questions that you need to answer in just one sentence no more than 100 words.",
        ),
        (
            "human",
            'The question is "{question}"',
        ),
        (
            "system",
            "Do not acknowledge and directly return the answer.",
        ),
        (
            "system",
            "Here's the history of the messages for this chat:",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
    ]
)

chain = question_template | chat


# history = ChatMessageHistory() # Memory History

history = FileChatMessageHistory(
    file_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "chat.history.json"
    )
)

chain_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=lambda _: history,
    input_messages_key=["question"],
    history_messages_key="chat_history",
)

while True:
    question = input("Ask me anything?:\n")

    res = chain_with_history.invoke(
        input={"question": question},
        config={"configurable": {"session_id": "1"}},
    )

    print(res.content)
