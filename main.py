import os
from tools.doc_split import TextChunker
from tools.caching import cache_data, load_cached_data
from pathlib import Path

os.environ["OPENAI_API_KEY"] = str(os.getenv("OPENAI_API_KEY"))

if __name__ == "__main__":
    text_name = "the_sound_of_waves_chinese"

    file_path = Path(f"text/{text_name}.txt")
    cache_file_path = Path(f"cache/{text_name}_cache.pkl")

    # Load the text
    with open(file_path, encoding="utf-8") as file:
        text = file.read()

    # Load cached chunks if they exist
    cached_chunks = load_cached_data(cache_file_path)

    if cached_chunks is not None:
        print("Using cached chunks.")
        chunks = cached_chunks
    else:
        print("Processing text...")
        text_chunker = TextChunker()
        chunks = text_chunker.process_text(text)
        
        # Cache the processed chunks
        cache_data(chunks, cache_file_path)
        print("Chunks cached successfully.")

    # Print the chunks
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:\n{chunk}\n")
