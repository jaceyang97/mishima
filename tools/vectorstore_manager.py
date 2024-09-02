from typing import List

import chromadb

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

class VectorStoreManager:
    def __init__(self, database_path: str, model: str = "text-embedding-3-large"):
        self.database_path = database_path
        self.model = model
        self.vectorstore_client = chromadb.PersistentClient(path=database_path)
        self.openai_embeddings = OpenAIEmbeddings(model=self.model)

    def create_vectorstore(self, collection_name: str) -> Chroma:
        return Chroma(
            client=self.vectorstore_client,
            collection_name=collection_name,
            embedding_function=self.openai_embeddings,
            persist_directory=self.database_path,
        )

    def delete_collection(self, collection_name: str) -> None:
        self.vectorstore_client.delete_collection(collection_name)

    def list_collections(self) -> List[str]:
        return self.vectorstore_client.list_collections()
