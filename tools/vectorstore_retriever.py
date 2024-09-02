from typing import List

from tqdm import tqdm
from langchain_core.documents import Document
from langchain_chroma import Chroma

from tools.text_chunker import ChunkInfo
from tools import llm_api

class VectorStoreRetriever:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def convert_chunks_to_documents(self, chunks: List[ChunkInfo]) -> List[Document]:
        documents: List[Document] = []

        for chunk in tqdm(chunks, desc="Processing Chunks"):
            summary = llm_api.summarize(chunk.content)
            if not summary:
                print("No summary returned!")
            doc = Document(
                page_content=summary,
                metadata={
                    "original_text": chunk.content,
                    "num_sentences": chunk.num_sentences,
                    "chunk_length": chunk.chunk_length
                }
            )
            documents.append(doc)
            
        return documents
    
    def add_to_vectorstore(self, documents: List[Document]):
        try:
            self.vectorstore.add_documents(documents=documents)
        except Exception as e:
            print(f"Error adding documents to vectorstore: {e}")

    def process_chunks(self, chunks: List[ChunkInfo]) -> None:
        documents = self.convert_chunks_to_documents(chunks)
        self.add_to_vectorstore(documents)
