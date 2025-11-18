LLM-based simulators
====================

UserSimCRS implements two user simulators relying on a large language model (LLM) to generate utterances: 

- **Single-Prompt Simulator**: It generates utterances based on a single prompt that includes the task description, user persona, information need, and conversational context.
- **Dual-Prompt Simulator**: It generates utterances based on two prompts: one to decide whether to continue the conversation or not (the "stopping prompt"). If the decision is to continue, it generates the next utterance using the main generation prompt (identical to the one used by the single-prompt simulator); otherwise, it sends a default utterance to stop the conversation.

We present the different prompts and available LLM interfaces below.

Prompts
-------

:py:class:`usersimcrs.simulator.llm.prompt.prompt.Prompt`

The base class for all prompts. It provides the basic structure and methods to build prompts for LLM-based simulators. 

Note that currently, the prompts are built with a zero-shot approach in mind, but they can be adapted to use few-shot or in-context learning by providing examples in the prompt (especially in the task description).

Utterance Generation Prompt
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:class:`usersimcrs.simulator.llm.prompt.utterance_generation_prompt.UtteranceGenerationPrompt`

The prompt is inspired by `[Terragni et al., 2023] <https://arxiv.org/abs/2306.00774>`_. It includes the following information: task description, information need, conversational context, and optionally the simulated user persona. The prompt is built as follows:

  {task_description}

  PERSONA: {persona}

  REQUIREMENTS: You are looking for a {item_type} with the following characteristics: {constraints}. Once you find a suitable {item_type}, make sure to get the following information: {requests}.

  HISTORY:   
  
  {conversational_context}

The persona section is included if the simulated user persona is provided. The placeholder *item_type* is replaced by the type of item the user is looking for such as a restaurant or a movie. The *constraints* and *requests* are extracted from the information need. The *conversational_context* is the history of the conversation up to the current utterance, hence, it is updated each time an utterance is received (agent utterance) or generated (simulated user utterance).

The default task description is:
  
  You are a USER discussing with an ASSISTANT. Given the conversation history, you need to generate the next USER message in the most natural way possible. The conversation is about getting a recommendation according to the REQUIREMENTS. You must fulfill all REQUIREMENTS as the conversation progresses (you don't need to fulfill them all at once). After getting all the necessary information, you can terminate the conversation by sending '\end'. You may also terminate the conversation is stuck in a loop or the ASSISTANT is not helpful by sending '\giveup'. Be precise with the REQUIREMENTS, clear and concise.

Stop Conversation Prompt
^^^^^^^^^^^^^^^^^^^^^^^^

:pyclass:`usersimcrs.simulator.llm.prompt.stop_prompt.StopPrompt`

The stop conversation prompt is used to indicate that the conversation should end. It is built as follows:

  {task_description}

  PERSONA: {persona}
  
  HISTORY:
  
  {conversational_context}
  
  Continue?

The default task description is:

  As a USER interacting with an ASSISTANT to receive a recommendation, analyze the conversation history to determine if it is progressing productively. If the conversation has been stuck in a loop with repeated misunderstandings across multiple turns, return 'FALSE' to indicate the conversation should be terminated. Otherwise, return 'TRUE' to indicate that the conversation should continue. Only return 'TRUE' or 'FALSE' without any additional information.

LLM interface
-------------

:py:mod:`usersimcrs.llm_interfaces`

The LLM interface is responsible for interacting with the large language model to generate responses. Currently, two LLM interfaces are supported: Ollama and OpenAI. 

Ollama
^^^^^^

:py:class:`usersimcrs.llm_interfaces.ollama_interface.OllamaLLMInterface`

This interface is used to interact with a LLM that is hosted on the `Ollama platform <https://ollama.com>`_. The interface sends requests to the `Ollama API <https://github.com/ollama/ollama/blob/main/docs/api.md>`_ to generate the responses. 

This interface is configured with a YAML file that includes: the model name, the host URL, whether to stream the responses, and the LLM specific options. An example configuration is shown below: 

.. code-block:: yaml

    model: "llama3"
    host: OLLAMA_HOST_URL
    stream: true
    options:
      max_tokens: 100
      temperature: 0.5
      top_p: 0.9
      top_k: 0
      ...


OpenAI
^^^^^^

:py:class:`usersimcrs.llm_interfaces.openai_interface.OpenAILLMInterface`

This interface interacts with models hosted on the OpenAI platform using their `API <https://openai.com/api/>`_. The interface sends requests to the OpenAI API to generate the responses.

This interface is configured with a YAML file that includes: the model name, the API key, and the LLM specific options. An example configuration is shown below:

.. code-block:: yaml

    model: "GPT-4o"
    api_key: YOUR_API_KEY
    options:
      max_tokens: 100
      seed: 42
      temperature: 0.5
      ...


**Reference**

Silvia Terragni, Modestas Filipavicius, Nghia Khau, Bruna Guedes, André Manso, and Roland Mathis. 2023. In-Context Learning User Simulators for Task-Oriented Dialog Systems. arXiv:2306.00774 [cs.CL].