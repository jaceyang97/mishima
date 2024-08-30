from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class PersonalityTraits(BaseModel):
    openness: str
    conscientiousness: str
    extraversion: str
    agreeableness: str
    neuroticism: str
    assertiveness: str
    empathy: str
    optimism: str
    humility: str
    ambition: str
    curiosity: str
    honesty: str
    risk_taking: str
    adaptability: str
    tolerance: str


class StyleResponse(BaseModel):
    personality_traits: PersonalityTraits
    # emotional_intelligence: str
    # cognitive_style: str
    # values_and_beliefs: str
    # cultural_background: str
    # communication_style: str
    # language_register: str
    # jargon: str
    # tone: str
    # structure: str
    # rhetorical_techniques: str
    # narrative_devices: str
    # bias_and_assumptions: str


text = "歌岛是个人口一千四百、方圆不到四公里的小岛。 歌岛有两处景致最美。 一处是人代神社，坐落在岛的最高点，朗西北而建。 从这里极目远望，可以望及伊势海的周边，歌岛就位于其湾口。"

# Use OpenAI API to generate a response
response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are an expert at identifying writing styles."
            "Analyze the following text and provide metadata based on writing style guide.",
        },
        {"role": "user", "content": text},
    ],
    response_format=StyleResponse,
)

# Extract the content from the response
style_response = response.choices[0].message

if style_response.parsed:
    print(style_response.parsed)
elif style_response.refusal:
    print(style_response.refusal)
