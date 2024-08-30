# Mishima Text Stylizer (WORK IN PROGRESS)

## Overview
This project aims to transform a given text input into a style reminiscent of Yukio Mishima's writing. The workflow involves setting up a database of Mishima's works, finding similar texts, and composing prompts for generating the final output using LLMs.

![Mishima Works Banner](assets/mishima_works_banner.jpg)

*Image Source: [PR Times](https://prtimes.jp/main/html/rd/p/000000051.000047877.html)*

## Steps

1. **Set Up a Vector Database of Mishima's Works**:
   - **Text Collection**:
     - Store original texts of Mishima's works. Automating this process is ideal due to the tedious nature of manual collection.
     - Consider publishing versions, translators, language of choice, and copyrights.
   - **Sentence Splitting**:
     - Use the NLP module `spacy` to split texts into sentences. The base text model is Chinese (`zh_core_web_sm`).

2. **Group Sentences into Semantically Coherent Clusters**:
   - **NLP and Model Selection**:
     - Use the `sentence-transformers` module with the `shibing624/text2vec-base-chinese` model.
   - **Algorithm and Parameters**:
     - Apply `Cosine Similarity` with a threshold of 0.5 and a sliding window of 5.
     - Note: These parameters are currently arbitrary and need optimization. The threshold can be refined through empirical testing. The sliding window may depend on the literature style; some styles may feature longer sentences, while others have shorter ones.

3. **Include Metadata of Writing Style**:
   - **Metadata Tags**:
     - Include attributes such as personality traits, emotional intelligence, etc.
   - **Challenges in Defining Writing Styles**:
     1. No definitive list of writing styles exists. The provided [writing style guide](https://viktorbezdek.github.io/definitive-llm-writing-style-guide/) is comprehensive but needs refinement to a smaller set of styles to:
        - Enhance the relevance of similarity search results when users query the existing works' database.
        - Reduce the token length of API calls, which is cost-sensitive.
     2. When selecting writing styles, the guide's universality should be balanced with the need for specificity to Mishima's texts. Prioritize identifying Mishima's most distinctive writing styles. Domain expertise is needed here.

4. **Find Top N Similar Texts**
   - Given a text input, search the vector database to find the top N similar texts.
   - Use a similarity algorithm; traditional NLP techniques (like TF-IDF, cosine similarity, or word embeddings) may suffice depending on the complexity required.

5. **Compose a Prompt**
   - Create a structured prompt for generating the output:
     - Start with an arbitrary prompt to set the style and tone.
     - Include few-shot examples to guide the generation process.
     - Incorporate the user input, potentially with modifications to align with Mishima's style.

6. **Generate the Response**
   - Use the composed prompt to generate a response in the style of Yukio Mishima.