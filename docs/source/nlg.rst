Natural Language Generation
===========================

The natural language generation (NLG) component is responsible for generating text utterances based on the structured representation provided by the dialogue policy. The generated utterances can be further customized based on user preferences and satisfaction levels.

UserSimCRS is compatible with NLU components from the `DialogueKit <https://iai-group.github.io/DialogueKit/main/>`_ library, which provides basic NLG functionalities. Additionally, we implement :py:class:`LLMGenerativeNLG`, a generative NLG component using a large language model to produce text responses based on the structured dialogue acts.

Generative NLG
--------------

**Prerequisites**: the LLM used for the generation should be hosted on a Ollama server.

The prompt used for the utterance generation should have the following placeholders:

- *dialogue_acts*: Placeholder for the string representation of the dialogue acts to be used for generating the utterance. A dialogue act is stringified as follows: `intent(slot1=value1, slot2,...)`; multiple dialogue acts are separated by `|`.
- *annotations*: Placeholder for the string representation of additional annotations that can be used to customize the generated utterance, such as satisfaction level. Annotations are stringified as follows: `annotation1=value1\nannotation2\n...`.

.. todo: An example of prompt is available at: ``.
