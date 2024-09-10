from typing import List

import chromadb

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import List, Optional


class VectorStoreManager:
    def __init__(self, database_path: str, model: str = "text-embedding-3-large"):
        self.database_path = database_path
        self.model = model
        self.vectorstore_client = chromadb.PersistentClient(path=database_path)
        self.openai_embeddings = OpenAIEmbeddings(model=self.model)

    def create_vectorstore(self, collection_name: str) -> Chroma:
        """创建一个新的向量存储集合。"""
        return Chroma(
            client=self.vectorstore_client,
            collection_name=collection_name,
            embedding_function=self.openai_embeddings,
            persist_directory=self.database_path,
        )

    def delete_collection(self, collection_name: str) -> None:
        """删除指定的向量存储集合。"""
        self.vectorstore_client.delete_collection(collection_name)

    def list_collections(self) -> List[str]:
        """列出所有的向量存储集合。"""
        return self.vectorstore_client.list_collections()

    def get_collection_info(self, collection_name: str) -> Optional[dict]:
        """获取指定集合的信息，如果集合不存在则返回None。"""
        collections = self.vectorstore_client.list_collections()
        if collection_name in collections:
            return self.vectorstore_client.get_collection(collection_name)
        return None
