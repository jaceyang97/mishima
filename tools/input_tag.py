import os
from openai import OpenAI
from writings_styles.style import StyleAnalyzeResponse

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def analyze_writing_style(text: str) -> dict:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert at identifying writing styles, particularly those of Yukio Mishima. "
                    "Analyze the following text and provide metadata based on the most relevant writing style attributes: "
                    "personality traits (openness, conscientiousness, neuroticism, assertiveness, ambition), "
                    "communication style (formal, emotional, analytical, serious, expressive), "
                    "tone (ironic, assertive, confident, persuasive, serious), "
                    "and narrative style (first-person, stream of consciousness, irony, metaphor, foreshadowing). "
                    "Each attribute represents a narrative style technique, categorized as low, medium, or high."
                    "Also, provide a list of subjects discussed in the text. Use name-entity recognition labeling."
                ),
            },
            {"role": "user", "content": text},
        ],
        response_format=StyleAnalyzeResponse,
    )

    style_response = response.choices[0].message

    result = {"original_text": text, "tags": {}}
    if style_response.parsed:
        result["tags"] = style_response.parsed.model_dump()
    elif style_response.refusal:
        result["tags"]["error"] = style_response.refusal

    return result
