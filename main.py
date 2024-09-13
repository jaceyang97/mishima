import os
from pathlib import Path
from PIL import Image
import gradio as gr
from tools import llm_api
from tools.caching import cache_data, load_cached_data
from tools.text_chunker import TextChunker
from tools.vectorstore_manager import VectorStoreManager
from tools.vectorstore_retriever import VectorStoreRetriever

# 设置OpenAI API密钥
os.environ["OPENAI_API_KEY"] = str(os.getenv("OPENAI_API_KEY"))

# 文件和缓存路径
text_name = "the_sound_of_waves_chinese"
file_path = Path(f"data/text/{text_name}.txt")
cache_file_path = Path(f"cache/{text_name}_cache.pkl")

# 读取文本
with open(file_path, encoding="utf-8") as file:
    text = file.read()

# 加载缓存的文本块
cached_chunks = load_cached_data(cache_file_path)

if cached_chunks is not None:
    print("使用缓存的文本块。")
    chunks = cached_chunks
else:
    print("正在处理文本...")
    text_chunker = TextChunker()
    chunks = text_chunker.process_text(text)
    cache_data(chunks, cache_file_path)
    print("文本块成功缓存。")

# 创建向量存储
database_path = "database"
collection_name = "mishima_texts"
vs_manager = VectorStoreManager(database_path=database_path)
vs = vs_manager.create_vectorstore(collection_name=collection_name)

# 处理文本块并添加到向量存储
# vs_retriever = VectorStoreRetriever(vectorstore=vs)
# vs_retriever.process_chunks(chunks)

# 生成模仿风格的响应
def generate_response(user_input: str):
    return llm_api.stylize(user_input=user_input, vectorstore=vs)

# 调整图像大小
original_image = Image.open("assets/mishima_1966.png")
resized_image = original_image.resize((int(original_image.width * 0.1), int(original_image.height * 0.1)))

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>三岛由纪夫风格化工具</h1>")
    gr.Image(resized_image, show_label=False)

    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(lines=10, placeholder="在此输入文本...", label="输入")
        with gr.Column():
            output_box = gr.Textbox(lines=10, placeholder="输出将在此显示...", label="输出", interactive=False)

    with gr.Row():
        generate_button = gr.Button("模仿")
    
    generate_button.click(fn=generate_response, inputs=input_box, outputs=output_box, show_api=True)

# 启动Gradio界面
if __name__ == "__main__":
    demo.launch()