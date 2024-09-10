from typing import List, Dict, Union

import numpy as np
import spacy
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

from langchain_openai import OpenAIEmbeddings

class ChunkInfo(BaseModel):
    content: str
    num_sentences: int
    chunk_length: int

class TextChunker:
    def __init__(self, model: str = "text-embedding-3-large", context_window: int = 1, percentile_threshold: int = 80):
        self.model = model
        self.context_window = context_window
        self.percentile_threshold = percentile_threshold
        self.sentences: List[Dict[str, Union[int, str, List[float]]]] = []
        self.openai_embeddings = OpenAIEmbeddings(model=self.model)

    def split_text_into_sentences(self, text: str) -> None:
        nlp = spacy.load("zh_core_web_sm")
        self.sentences = [{'index': idx, 'sentence': sent.text.strip()} 
                          for idx, sent in enumerate(nlp(text).sents) if sent.text.strip()]

    def combine_sentences(self) -> None:
        for idx, sentence_dict in enumerate(self.sentences):
            start_idx = max(idx - self.context_window, 0)
            end_idx = min(idx + self.context_window + 1, len(self.sentences))
            sentence_dict['combined_sentence'] = ' '.join(sent['sentence'] for sent in self.sentences[start_idx:end_idx])

    def compute_embeddings(self) -> None:
        embeddings = self.openai_embeddings.embed_documents(
            [sent['combined_sentence'] for sent in self.sentences]
        )
        for idx, sentence_dict in enumerate(self.sentences):
            sentence_dict['combined_sentence_embedding'] = embeddings[idx]

    def calculate_cosine_distances(self) -> List[float]:
        embeddings = np.array([sent['combined_sentence_embedding'] for sent in self.sentences])
        cosine_distances = 1 - cosine_similarity(embeddings[:-1], embeddings[1:]).diagonal()  # type: ignore
        for idx, distance in enumerate(cosine_distances):
            self.sentences[idx]['distance_to_next'] = distance
        return cosine_distances.tolist()

    def to_chunks(self, distances: List[float]) -> List[ChunkInfo]:
        distance_threshold = np.percentile(distances, self.percentile_threshold)
        indices_above_threshold = np.flatnonzero(distances > distance_threshold).tolist()
        chunks_info: List[ChunkInfo] = []
        start_idx = 0

        for end_idx in indices_above_threshold + [len(self.sentences)]:
            chunk_text = ' '.join(sent['sentence'] for sent in self.sentences[start_idx:end_idx])  # type: ignore
            chunks_info.append(ChunkInfo(content=chunk_text, num_sentences=end_idx - start_idx, chunk_length=len(chunk_text)))
            start_idx = end_idx

        return chunks_info

    def process_text(self, text: str) -> List[ChunkInfo]:
        self.split_text_into_sentences(text)
        self.combine_sentences()
        self.compute_embeddings()
        distances = self.calculate_cosine_distances()
        return self.to_chunks(distances)