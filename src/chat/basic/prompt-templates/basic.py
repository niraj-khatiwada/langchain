import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug
from langchain_core.prompts import PromptTemplate

if __name__ == "__main__":
    load_dotenv(override=True)
    set_debug(True)

    OPEN_AI_BASE_URL = os.getenv("OPEN_AI_BASE_URL")
    OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

    llm = ChatOpenAI(
        base_url=OPEN_AI_BASE_URL,
        api_key=OPEN_AI_API_KEY,
    )

    # Prompt Template will be Human only
    prompt_template = PromptTemplate(
        template="""Convert this question into Spanish language.
    The question is: {input}
    """,
        input_variables=["input"],
    )

    # Langchain Common Expression Language(LCEL)
    chain = prompt_template | llm

    res = chain.invoke(
        input=prompt_template.form(
            input="Hello",
        )
    )

    print(res.text())
