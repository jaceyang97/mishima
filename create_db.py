import os
import json  # Import json for serialization
from tools.input_tag import analyze_writing_style
from tools.doc_split import split_text_into_sentences_nlp, group_sentences_by_semantics
from tools.caching import *
from sentence_transformers import SentenceTransformer
import chromadb

chroma_client = chromadb.PersistentClient(path="database")
collection_name = "mishima_texts"
collection = chroma_client.get_or_create_collection(name=collection_name)

# Load model for generating embeddings
model = SentenceTransformer("shibing624/text2vec-base-chinese")


def main(file_path: str):
    # Define cache file paths
    sentence_cache_file = "cache/sentences.pkl"
    group_cache_file = "cache/grouped_sentences.pkl"

    # Load cached sentences if available
    sentences = load_cached_data(sentence_cache_file)
    if sentences is None:
        # Read the text from the file
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        # Split the text into sentences
        sentences = split_text_into_sentences_nlp(text)
        # Cache the sentences
        cache_data(sentences, sentence_cache_file)
    else:
        print("Loaded sentences from cache.")

    # Load cached grouped sentences if available
    grouped_sentences = load_cached_data(group_cache_file)
    if grouped_sentences is None:
        # Group sentences based on semantic similarity
        grouped_sentences = group_sentences_by_semantics(
            sentences, threshold=0.5, window_size=5
        )
        # Cache the grouped sentences
        cache_data(grouped_sentences, group_cache_file)
    else:
        print("Loaded grouped sentences from cache.")

    # Generate and store embeddings and tags for each group
    for i, group in enumerate(grouped_sentences[:3]):
        print(f"Group {i+1}:")
        print(group)
        print("-" * 20)

        # Generate tags for the current group
        response = analyze_writing_style(group)
        tags = response["tags"]
        print(f"Tags for Group {i+1}:")
        print(tags)
        print("=" * 40)

        # Convert metadata to a JSON string for Chroma compatibility
        tags_json = json.dumps(tags)  # Convert the tags dictionary to a JSON string

        # Generate embeddings for the group
        embedding = model.encode(
            group
        ).tolist()  # Convert to list for Chroma compatibility

        # Upsert data into Chroma
        collection.upsert(
            ids=[str(i)],
            embeddings=[embedding],
            metadatas=[{"tags": tags_json, "original_text": group}],
        )


if __name__ == "__main__":
    # Specify the path to your text file
    file_path = "text/the_sound_of_waves_chinese.txt"

    # Ensure cache directory exists
    os.makedirs("cache", exist_ok=True)

    # Run the main function
    main(file_path)
