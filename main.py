import os
from pathlib import Path
from PIL import Image
import gradio as gr
from loguru import logger

from tools.text_chunker import TextChunker
from tools.vectorstore_retriever import VectorStoreRetriever
from tools.vectorstore_manager import VectorStoreManager
from tools.caching import cache_data, load_cached_data
from tools import llm_api

os.environ["OPENAI_API_KEY"] = str(os.getenv("OPENAI_API_KEY"))

text_name = "the_sound_of_waves_chinese"
file_path = Path(f"data/text/{text_name}.txt")
cache_file_path = Path(f"cache/{text_name}_cache.pkl")

with open(file_path, encoding="utf-8") as file:
    text = file.read()

cached_chunks = load_cached_data(cache_file_path)

if cached_chunks is not None:
    logger.info("Using cached chunks.")
    chunks = cached_chunks
else:
    logger.info("Processing text...")
    text_chunker = TextChunker()
    chunks = text_chunker.process_text(text)
    
    cache_data(chunks, cache_file_path)
    logger.info("Chunks cached successfully.")

# for chunk in chunks[:50]:
#     logger.debug(chunk)
#     logger.debug("\n")

database_path = "database"
collection_name = "mishima_texts"
vs_manager = VectorStoreManager(database_path=database_path)
vs = vs_manager.create_vectorstore(collection_name=collection_name)

vs_retriever = VectorStoreRetriever(vectorstore=vs)
# this line is unnecessary if documents are already in vectorstore
# vs_retriever.process_chunks(chunks)

# Wrapper function for Gradio
def generate_response(user_input: str):
    response = llm_api.stylize(user_input=user_input, vectorstore=vs)
    return response

# Temp modification to image size
original_image = Image.open("assets/mishima_1966.png")
new_size = (int(original_image.width * 0.1), int(original_image.height * 0.1))
resized_image = original_image.resize(new_size)

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>Mishima Yukio Stylizer</h1>")
    gr.Image(resized_image, show_label=False)

    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(lines=10, placeholder="Enter text here...", label="Input")
        with gr.Column():
            output_box = gr.Textbox(lines=10, placeholder="Output will appear here...", label="Output", interactive=False)

    with gr.Row():
         generate_button = gr.Button("Stylize", scale=0)
    
    generate_button.click(fn=generate_response, inputs=input_box, outputs=output_box, show_api=True)

if __name__ == "__main__":
    demo.launch()