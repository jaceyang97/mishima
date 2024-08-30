from pydantic import BaseModel


class PersonalityTraits(BaseModel):
    openness: str
    conscientiousness: str
    # extraversion: str
    # agreeableness: str
    neuroticism: str
    assertiveness: str
    # empathy: str
    # optimism: str
    # humility: str
    ambition: str
    # curiosity: str
    # honesty: str
    # risk_taking: str
    # adaptability: str
    # tolerance: str


class EmotionalIntelligence(BaseModel):
    self_awareness: str
    self_regulation: str
    motivation: str
    empathy: str
    social_skills: str
    emotional_perception: str
    emotional_expression: str
    assertiveness: str
    independence: str
    optimism: str
    stress_tolerance: str
    self_actualization: str
    adaptability: str
    trust: str
    conflict_resolution: str


class CognitiveStyle(BaseModel):
    analytical: str
    creative: str
    strategic: str
    detail_oriented: str
    big_picture_thinking: str
    systematic: str
    intuitive: str
    practical: str
    scientific: str
    abstract_thinking: str
    logical_thinking: str
    verbal: str
    visual: str
    concrete_thinking: str
    sequential_thinking: str


class ValuesAndBeliefs(BaseModel):
    conservative: str
    liberal: str
    libertarian: str
    progressive: str
    traditional: str
    religious: str
    secular: str
    humanist: str
    environmentalist: str
    capitalist: str
    socialist: str
    patriotic: str
    globalist: str
    individualistic: str
    collectivistic: str


class CulturalBackground(BaseModel):
    african: str
    asian: str
    european: str
    latin_american: str
    middle_eastern: str
    north_american: str
    oceanian: str
    scandinavian: str
    slavic: str
    germanic: str
    romance: str
    mountainous: str
    coastal: str
    island: str
    tropical: str


class CommunicationStyle(BaseModel):
    # direct: str
    # indirect: str
    formal: str
    # informal: str
    # factual: str
    emotional: str
    analytical: str
    # intuitive: str
    # verbose: str
    # concise: str
    # confident: str
    # hesitant: str
    # humorous: str
    serious: str
    # respectful: str
    # friendly: str
    # loud: str
    # soft_spoken: str
    # fast_paced: str
    # slow_paced: str
    # visual: str
    # auditory: str
    # kinesthetic: str
    # functional: str
    expressive: str


class LanguageRegister(BaseModel):
    formal: str
    informal: str
    colloquial: str
    slang: str
    jargon: str
    technical: str
    academic: str
    literary: str
    poetic: str
    nostalgic: str
    inspirational: str
    humorous: str
    sarcastic: str
    ironic: str
    sincere: str
    exaggerated: str
    understated: str
    persuasive: str
    informative: str
    instructional: str
    conversational: str
    oratorical: str
    sermon: str
    baby_talk: str
    caretaker_speech: str
    motherese: str
    debate: str
    ceremonial: str
    prayer: str
    ritual: str
    folkloric: str
    intimate: str
    casual: str


class Jargon(BaseModel):
    medical: str
    legal: str
    technical: str
    scientific: str
    academic: str
    business: str
    marketing: str
    financial: str
    political: str
    military: str
    sports: str
    arts: str
    culinary: str
    fashion: str
    music: str
    film: str
    gaming: str
    social_media: str
    internet: str
    crypto: str
    programming: str
    engineering: str
    architecture: str
    psychology: str
    education: str
    journalism: str
    publishing: str


class Slang(BaseModel):
    regional: str
    ethnic: str
    cultural: str
    generational: str
    occupational: str
    argot: str
    cant: str
    jargon: str
    colloquialisms: str
    idioms: str
    euphemisms: str
    dysphemisms: str
    profanity: str
    obscenity: str
    internet: str
    social_media: str
    memes: str
    emojis: str
    niche: str
    outdated: str
    trendy: str
    playful: str
    ironic: str
    sarcastic: str
    derogatory: str
    complimentary: str
    nicknames: str
    abbreviations: str
    acronyms: str
    blends: str
    clippings: str
    reduplications: str


class Politeness(BaseModel):
    formal_address: str
    informal_address: str
    titles: str
    honorifics: str
    deference_expressions: str
    hedging_language: str
    face_saving_strategies: str
    indirect_speech_acts: str
    indirect_requests: str
    apologies: str
    gratitude_expressions: str
    compliments: str
    refusals: str
    disagreements: str
    interruptions: str
    turn_taking: str
    silence: str
    eye_contact: str
    gestures: str
    proximity: str


class GenderedLanguage(BaseModel):
    masculine_coded: str
    feminine_coded: str
    gender_neutral: str


class AgeSpecificLanguage(BaseModel):
    infantile: str
    childlike: str
    adolescent: str
    teenspeak: str
    youthful: str
    middle_aged: str
    elderly: str
    dated: str
    old_fashioned: str
    anachronistic: str
    timeless: str
    ageless: str
    baby_boomer: str
    gen_x: str
    millennial: str
    gen_z: str
    boomer: str
    zoomer: str
    silver_surfer: str
    wise: str
    innocent: str
    naive: str
    mature: str
    infantilizing: str
    patronizing: str
    ageist: str


class SocioeconomicLanguage(BaseModel):
    upper_class: str
    middle_class: str
    working_class: str
    blue_collar: str
    white_collar: str
    pink_collar: str
    poverty: str
    wealthy: str
    privileged: str
    disadvantaged: str
    upwardly_mobile: str
    downwardly_mobile: str
    homeless: str
    bougie: str
    ghetto: str
    trailer_trash: str
    redneck: str
    ozark: str
    wasp: str
    yuppie: str
    hipster: str
    basic: str
    elitist: str
    snobbish: str
    nouveau_riche: str
    old_money: str
    welfare_queen: str
    food_stamps: str
    unemployed: str
    student: str
    graduate: str
    dropout: str
    entrepreneur: str
    startup: str
    corporate: str
    freelance: str
    retired: str
    pensioner: str
    disabled: str


class Tone(BaseModel):
    # humorous: str
    # sarcastic: str
    ironic: str
    # friendly: str
    optimistic: str
    assertive: str
    confident: str
    # playful: str
    persuasive: str


class Voice(BaseModel):
    # conversational: str
    sarcastic: str
    # humorous: str
    # enthusiastic: str
    # informal: str
    # friendly: str
    assertive: str
    confident: str
    # optimistic: str
    persuasive: str
    # playful: str
    ironic: str


class NarrativeStyle(BaseModel):
    first_person: str
    stream_of_consciousness: str
    # nonlinear: str
    # flashback: str
    foreshadowing: str
    # cliffhanger: str
    # plot_twist: str
    irony: str
    metaphor: str
    # simile: str
    # satire: str
    # unreliable_narrator: str
    # vignette: str


class StyleResponse(BaseModel):
    personality_traits: PersonalityTraits
    # emotional_intelligence: EmotionalIntelligence
    # cognitive_style: CognitiveStyle
    # values_and_beliefs: ValuesAndBeliefs
    # cultural_background: CulturalBackground
    communication_style: CommunicationStyle
    # language_register: LanguageRegister
    # jargon: Jargon
    # slang: Slang
    # politeness: Politeness
    # genderedlanguage: GenderedLanguage
    # age_specific_language: AgeSpecificLanguage
    # socioeconomiclanguage: SocioeconomicLanguage
    tone: Tone
    # voice: Voice
    narrative_style: NarrativeStyle


class StyleAnalyzeResponse(BaseModel):
    subject: list[str]
    style: StyleResponse
