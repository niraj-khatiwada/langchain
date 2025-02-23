import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

    if not vector_db.collection_exists("job"):
        vector_db.create_collection(
            collection_name="job",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    jobs_doc = TextLoader(file_path="docs/jobs.txt").load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
    chunks = text_splitter.split_documents(documents=jobs_doc)

    store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name="job-txt",
        url=VECTOR_DB_URL,
    )

    search = input("What job are you searching for?\n")
    vec = embedding.embed_query(text=search)
    res = store.similarity_search_by_vector(vec)
    print(res)
