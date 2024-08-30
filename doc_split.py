import spacy
from sentence_transformers import SentenceTransformer
import numpy as np


def split_text_into_sentences_nlp(text):
    # Load the Chinese NLP model
    nlp = spacy.load("zh_core_web_sm")

    # Process the text with SpaCy
    doc = nlp(text)

    # Extract sentences from the processed text
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences


def group_sentences_by_semantics(sentences, model, threshold=0.5, window_size=3):
    # Generate sentence embeddings
    embeddings = model.encode(sentences)

    # Initialize grouped sentences
    grouped_sentences = []
    current_group = [sentences[0]]

    # Iterate over sentences and compute similarity
    for i in range(1, len(sentences)):
        # Calculate similarity between current sentence and previous sentences in the window
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


# Read the text from the file
file_path = "text/the_sound_of_waves_chinese.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# Use NLP to split the text into sentences
sentences = split_text_into_sentences_nlp(text)

# Load a pre-trained sentence transformer model

# 'paraphrase-multilingual-MiniLM-L12-v2'
# 'DMetaSoul/sbert-chinese-general-v2'
# 'uer/sbert-base-chinese-nli'
# 'shibing624/text2vec-base-chinese'
model = SentenceTransformer("shibing624/text2vec-base-chinese")

# Group sentences based on semantic similarity
grouped_sentences = group_sentences_by_semantics(
    sentences, model, threshold=0.5, window_size=5
)

# Output the first 5 groups
for i, group in enumerate(grouped_sentences[:5]):  # Display only the first 5 groups
    print(f"Group {i+1}:")
    print(group)
    print("-" * 20)
