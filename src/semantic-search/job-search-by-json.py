import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_community.document_loaders import JSONLoader
from langchain_qdrant import QdrantVectorStore

load_dotenv(override=True)

if __name__ == "__main__":
    AI_SERVER_URL = os.getenv("AI_SERVER_URL")
    VECTOR_DB_URL = os.getenv("VECTOR_DB_URL")

    embedding = OpenAIEmbeddings(
        base_url=AI_SERVER_URL,
        check_embedding_ctx_length=False,  # Important
    )

    vector_db = QdrantClient(url=VECTOR_DB_URL)

    collection_exists = vector_db.collection_exists("job")
    if not collection_exists:
        vector_db.create_collection(
            collection_name="job",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    if collection_exists:
        yn = input("Do you want to add jobs data to vector db? (y/n)")
        if yn.strip() == "y":

            def job_metadata_func(match: dict, metadata: dict):
                metadata["id"] = match.get("id")
                return metadata

            jobs_json = JSONLoader(
                file_path="docs/jobs.json",
                text_content=False,
                jq_schema=".jobs[]",
                content_key="{title, company, location}",
                metadata_func=job_metadata_func,
                is_content_key_jq_parsable=True,
            ).load()

            store = QdrantVectorStore.from_documents(
                documents=jobs_json,
                embedding=embedding,
                collection_name="job",
                url=VECTOR_DB_URL,
            )
        else:
            store = QdrantVectorStore.from_existing_collection(
                embedding=embedding,
                collection_name="job",
                url=VECTOR_DB_URL,
            )

    search = input("What/Where are you searching your job?\n")
    vec = embedding.embed_query(text=search)
    res = store.similarity_search_by_vector(vec)
    print(res)
