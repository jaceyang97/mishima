from typing import List
import yaml

from loguru import logger
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document

with open("config/models.yaml", "r") as file:
    config = yaml.safe_load(file)

SUMMARY_MODEL = config['models']['summary']
STYLE_MODEL = config['models']['style']
STYLE_TEMPERATURE = config['models']['temperature']

client = OpenAI()

def summarize(text: str) -> str:
    #TODO: refine language, probably change to chinese
    system_prompt = (
        "你是一位擅长文学分析和总结的专家。"
        "你的任务是简明扼要地总结以下文本。"
        "请专注于捕捉核心主题、文本结构、语气、视角和情感。"
        "使用文本的原始语言，并注意三岛由纪夫的写作风格："
        "他的作品常常充满对美的执着，富有象征性，文字紧张而精确，"
        "并且有时使用晦涩的词语。"
    )

    try:
        completion = client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
        )

        return str(completion.choices[0].message.content)

    except Exception as e:
        logger.warning(f"Error during summarization: {e}")
        return ""

#TODO: Could decrease k to 2-3 if chunking and search is more acurate. But more is never bad as long as context length is good.    
def stylize(user_input: str, vectorstore: Chroma, k: int = 5) -> str:
    summarized_text = summarize(user_input)
    results: List[Document] = vectorstore.similarity_search(query=summarized_text, k=k)
    most_relevant_texts = [result.metadata['original_text'] for result in results]

    original_texts = "\n\n".join([f"原文 {idx + 1}:\n {text}" for idx, text in enumerate(most_relevant_texts)])
    logger.debug(original_texts)

    #TODO: refine language
    #TODO: 减少回答中"仿佛"的使用次数
    system_prompt = (
        "你是一位擅长模仿三岛由纪夫写作风格的文学专家。"
        "根据用户输入的文本，运用三岛由纪夫独特的写作风格重新诠释，保持原文本的核心内容与情节不变。"
        "注意：他的写作风格充满对美的执着，富有象征性，文字紧张而精确，有时使用晦涩词语。" 
        "不添加任何新的句子。尽量保持文本长度一致。减少<仿佛>的使用次数。"
        "{original_texts}"
    ).format(original_texts=original_texts)

    completion = client.chat.completions.create(
        model=STYLE_MODEL, 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=STYLE_TEMPERATURE
    )

    message = completion.choices[0].message
    if message.refusal is not None:
        logger.warning("warning!")

    return str(message.content)