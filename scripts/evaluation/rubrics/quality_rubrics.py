"""Definition of grading rubrics for dialogue quality evaluation."""

from enum import Enum


class QualityRubrics(Enum):
    REC_RELEVANCE = "Recommendation Relevance\n5: The agent consistently "
    "provides informative and helpful recommendations that meet exactly the "
    "user's needs and preferences.\n4: The agent mostly provides informative "
    "and helpful recommendations, with only a few instances where the "
    "recommendations do not meet the user's needs or preferences.\n3: The agent"
    " occasionally provides informative and helpful recommendations, but there"
    " are several instances where the recommendations do not meet the user's "
    "needs or preferences.\n2: The agent rarely provides informative and "
    "helpful recommendations, often offering recommendations that do not meet "
    "the user's needs or preferences.\n1: The agent consistently fails to "
    "provide informative and helpful recommendations, offering recommendations "
    "that do not meet the user's needs or preferences."
    COM_STYLE = "Communication Style\n5: The agent consistently communicates "
    "clearly and concisely, avoiding ambiguity and unnecessary complexity in "
    "its utterances.\n4: The agent mostly communicates clearly and concisely, "
    "with only a few instances of ambiguous or overly complex utterances.\n3: "
    "The agent occasionally communicates clearly and concisely, but there are "
    "several instances of ambiguity or unnecessary complexity in its "
    "utterances.\n2: The agent rarely communicates clearly and concisely, "
    "often using ambiguous or overly complex language in its utterances.\n1: "
    "The agent consistently fails to communicate clearly and concisely, "
    "making it difficult to understand its utterances."
    FLUENCY = "Fluency\n5: The agent's responses are indistinguishable from "
    "those of a real human.\n4: The agent's responses closely resemble those "
    "of a real human, with only a few instances where the language or style "
    "feels slightly artificial.\n3: The agent's responses sometimes resemble "
    "those of a real human, but there are several instances where the language"
    " or style feels noticeably artificial.\n2: The agent's responses rarely "
    "resemble those of a real human, often sounding robotic or unnatural in "
    "language or style.\n1: The agent's responses consistently fail to "
    "resemble those of a real human, sounding highly robotic or unnatural."
    CONV_FLOW = "Conversational Flow\n5: The conversation flows naturally and "
    "smoothly, with seamless transitions between utterances.\n4: The "
    "conversation mostly flows naturally and smoothly, with only a few "
    "instances of abrupt or disjointed transitions.\n3: The conversation "
    "occasionally flows naturally and smoothly, but there are several "
    "instances of abrupt or disjointed transitions.\n2: The conversation "
    "rarely flows naturally and smoothly, often feeling disjointed or lacking"
    " coherence.\n1: The conversation consistently fails to flow naturally "
    "and smoothly, with disjointed transitions and lack of coherence."
    OVERALL_SAT = "Overall Satisfaction\n5: The user is highly satisfied with "
    "the conversation, finding it engaging and informative.\n4: The user is "
    "mostly satisfied with the conversation, with only a few minor points of "
    "dissatisfaction.\n3: The user is somewhat satisfied with the conversation,"
    " but there are several aspects that could be improved.\n2: The user is "
    "mostly dissatisfied with the conversation, with only a few minor points "
    "of satisfaction.\n1: The user is highly dissatisfied with the "
    "conversation, finding it dull and uninformative."