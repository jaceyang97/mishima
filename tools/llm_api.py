from openai import OpenAI
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document

client = OpenAI()

def summarize(text: str) -> str:
    prompt = (
        "You are an expert literary analyst and summarizer."
        "Your task is to concisely summarize the following text."
        "Focus on capturing the core subject matter, text structure, tone, perspective, and emotions."
        "Use the text's original language."
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
        )

        # Safely access the message content
        message = completion.choices[0].message
        return str(message.content)

    except Exception as e:
        print(f"Error during summarization: {e}")
        return ""
    
def mimic_mishima_style(user_input: str, vectorstore: Chroma, k: int=5) -> str:
    # Step 1: Summarize the input text
    summarized_text = summarize(user_input)
    
    # Step 2: Query the vectorstore using the summarized text
    results: List[Document] = vectorstore.similarity_search(query=summarized_text, k=k)

    # Step 3: Extract the most relevant original texts
    most_relevant_texts = [result.metadata['original_text'] for result in results]

    # Step 4: Generate a new text imitating Mishima Yukio's style
    labeled_texts = "\n\n".join([f"文本 {idx + 1}:\n {text}" for idx, text in enumerate(most_relevant_texts)])
    
    imitation_prompt = (
        "你是一位擅长模仿三岛由纪夫写作风格的文学专家。"
        "根据用户输入的文本，运用三岛由纪夫独特的写作风格重新诠释，保持原文本的核心内容与情节不变。"
        "注意，他的写作风格充满对美的执着，富有象征性，文字紧张而精确。"
        "请不要添加任何新的句子，尽量保持文本长度一致。\n\n"
        "原文内容：\n\n"
        "{labeled_texts}\n\n"
        "请在此处开始你的新文本："
    ).format(labeled_texts=labeled_texts)
    
    print(imitation_prompt)

    completion = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": imitation_prompt},
            {"role": "user", "content": user_input}
        ],
    )

    message = completion.choices[0].message
    if message.refusal is not None:
        print("warning!")

    return str(message.content)