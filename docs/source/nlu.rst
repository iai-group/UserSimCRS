Natural Language Understanding
==============================

The natural language understanding (NLU) component is responsible for obtaining a structured representation of text utterances. This typically entails dialogue act recognition, a dialogue act comprises an intent and its associated slot-value pairs. 
In addition to this, the NLU can also do satisfaction prediction. This is the user's satisfaction with the agents response.

UserSimCRS is compatible with NLU components from the `DialogueKit <https://iai-group.github.io/DialogueKit/main/>`_ library, which provides basic NLU functionalities. Additionally, we implement :py:class:`usersimcrs.nlu.lm.LMDialogueActExtractor`, a dialogue act extractor based on a large language model. 

LLM-based NLU
-------------

**Prerequisites**: the LLM used for the dialogue acts recognition should be hosted on a ollama server.

The prompt used for the dialogue acts recognition should have the following placeholders:

- *utterance*: Placeholder for the text utterance to be processed.

Please note that the expected format for the dialogue acts recognition model output is a string representation of the dialogue acts, where each dialogue act is formatted as `intent(slot1=value1, slot2=value2, ...)`. Multiple dialogue acts are separated by a pipe (`|`).
After parsing the model output, invalid dialogue acts are discarded.
