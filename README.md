# Mishima Writing Stylizer

⚠️ **Important Announcement (2025-02-01)** ⚠️
***
> **This repository is no longer being actively maintained or supported.**  
> As someone who greatly enjoys Yukio Mishima's work but lacks domain expertise in literature, I have come to realize that simply combining original passages with style prompts does not adequately capture the essence of Mishima's writing.  
>  
> **While this project was an attempt to emulate his literary style**, the complexity of his work requires a deeper understanding that goes beyond what was originally intended here.  
>  
> **I encourage others** with the necessary expertise to explore this concept further.
***

## Overview
This project aims to transform a given text input into a style reminiscent of Yukio Mishima's writing. The workflow involves setting up a database of Mishima's works, finding similar texts, and composing prompts for generating the final output using LLMs.

![Mishima Works Banner](assets/mishima_works_banner.jpg)

*Image Source: [PR Times](https://prtimes.jp/main/html/rd/p/000000051.000047877.html)*

## Steps

<img src="assets/mishima_architecture.png" alt="Architecture" width="640" height="605"/>
*Figure 1: Proposed system architecture diagram showing the complete workflow from text ingestion to styled output generation*

1. **Set Up a Vector Database of Mishima's Works**:
   - **Text Collection**:
     - Store original texts of Mishima's works. Automating this process is ideal due to the tedious nature of manual collection.
     - Need to consider publishing versions, translators, language of choice, and copyrights.
   - **Sentence Splitting**:
     - Use the NLP module `spacy` to split texts into sentences. The base text model is Chinese (`zh_core_web_sm`).
   - **Semantic Chunking**:
     - Use embedding to find the breakpoints of certain percentile threshold between sentences with a context window 
   - **Summary Storing**
     - Include attributes such as subject matter, sentence structure, emotions etc...

2. **Find Top N Similar Texts**
   - Given a text input, first summarize it using the same prompt for stroing documents, then search the vectorstore to find the top N similar texts.

3. **Compose a Stylizer Prompt**
   - Create a structured prompt for generating the output, including:
     - An arbitrary prompt to set the role and instructions.
     - Few-shot examples retrieved from vectorstore.
     - Raw user input.

4. **Generate the Response**
   - Use the composed prompt to generate a response in the style of Yukio Mishima then display it on web.


## TODO List

| Task                                                                 | Status     | Comment                                                                                     | Priority  |
|:---------------------------------------------------------------------|:-----------|:-------------------------------------------------------------------------------------------|:----------|
| Prompt - generate prompt for prompt engineering                      | Pending    | I expect a lot of the work will be done here, finding optimized prompt. Research on effective prompting is required, especially for novel text generation. This step will be revised constantly.                                                                                            | High      |
| Data - replace text with English                                     | Pending    | Need to try out both English and Chinese to figure out which performs better. Intuition and model profile suggest English.                                                                                            | Medium    |
| Prompt - add high-dimensional data about style                      | Pending    | Optimize the summarization part, or look at other query transformations [must add domain expertise] | Medium    |
| UIUX - Use stream output to mimic handwriting                       | Pending    | UX improvement                                                                             | Medium    |
| Chunking - if sentence is short, compare surrounding embeddings     | Pending    | Slight optimization on chunking gives better original text during prompting, which is a part but might not be as important as the prompt itself.                                                                         | Low       |
| Evaluation - add benchmark                                           | Pending    | Need a way to measure how good the result is. Probably human eval, maybe benchmark.                                                                                            | Low       |
| Database - record text source and subject                            | Pending    | Better categorization and display in frontend.                                                                                            | Low       |
| UIUX - display top n original texts + metadata                      | Pending    | Makes the whole application look better                                                                                            | Low       |
| Model - switch to better OpenAI English model                       | Completed  | It is intuitive to use better model. Switched to gpt-4o from gpt-4-mini, and result is emperically significantly better.                                                                                            | High      |
| Model - Find model to fine-tune on Chinese/novels                   | Cancelled  | Unless there exist an industry-standard level LLM, finding a niche model probably means I need to localize the model hosting service, losing access to OpenAI API.                                                                                            | High      |