import spacy
import numpy as np
from sentence_transformers import SentenceTransformer


def split_text_into_sentences_nlp(text: str) -> list:
    nlp = spacy.load("zh_core_web_sm")
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences


def group_sentences_by_semantics(sentences: list, threshold=0.5, window_size=3) -> list:
    model = SentenceTransformer("shibing624/text2vec-base-chinese")
    embeddings = model.encode(sentences)

    # Initialize grouped sentences
    grouped_sentences = []
    current_group = [sentences[0]]

    for i in range(1, len(sentences)):
        similarities = []
        for j in range(max(0, i - window_size), i):
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            similarities.append(sim)

        avg_similarity = np.mean(similarities)

        # If average similarity is above the threshold, add to the current group
        if avg_similarity > threshold:
            current_group.append(sentences[i])
        else:
            # Otherwise, start a new group
            grouped_sentences.append(" ".join(current_group))
            current_group = [sentences[i]]

    # Add the last group
    grouped_sentences.append(" ".join(current_group))

    return grouped_sentences
