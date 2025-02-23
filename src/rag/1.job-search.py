import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_community.document_loaders import JSONLoader
from langchain_qdrant import QdrantVectorStore
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts.chat import ChatPromptTemplate


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

    vector_db = QdrantClient(url=VECTOR_DB_URL)

    def load_jobs():
        def job_metadata_func(match: dict, metadata: dict):
            metadata["id"] = match.get("id")
            return metadata

        jobs_json = JSONLoader(
            file_path="docs/jobs.json",
            text_content=False,
            jq_schema=".jobs[]",
            content_key="{title, company, location, salary, type}",
            metadata_func=job_metadata_func,
            is_content_key_jq_parsable=True,
        ).load()

        store = QdrantVectorStore.from_documents(
            documents=jobs_json,
            embedding=embedding,
            collection_name="job",
            url=VECTOR_DB_URL,
        )
        return store

    collection_exists = vector_db.collection_exists("job")
    if not collection_exists:
        vector_db.create_collection(
            collection_name="job",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        store = load_jobs()

    if collection_exists:
        yn = input("Do you want to add jobs data to vector db? (y/n)")
        if yn.strip() == "y":
            store = load_jobs()
        else:
            store = QdrantVectorStore.from_existing_collection(
                embedding=embedding,
                collection_name="job",
                url=VECTOR_DB_URL,
            )

    prompt_template = ChatPromptTemplate.from_messages(
        messages=[
            (
                "system",
                """You're an assistant that is designed to answer questions for the given context only. 
                Here's the context:
                {context}""",  # The key {context} is used internally so should be present as exact key indicating the document context
            ),
            (
                "human",
                "The question is: {input}",  # The key {input} is used internally so should be present as exact key indicating the user question
            ),
        ]
    )
    retriever = store.as_retriever()
    qa_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt_template,
    )
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=qa_chain,
    )

    while True:
        question = input("Ask anything?\n")
        res = rag_chain.invoke({"input": question})
        print(res)
