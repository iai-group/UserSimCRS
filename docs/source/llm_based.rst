LLM-based simulator
===================

This simulator relies on a large language model (LLM) to generate utterances. Currently, a single zero-shot prompt is supported for the response generation. The interactions with the LLM are managed by the LLM interface.

Prompt
------

:py:class:`usersimcrs.simulator.llm.prompt.Prompt`

The prompt is inspired by the work of Terragni et al. [1]_ It includes the following information: task description, information need, optionally the simulated user persona, and conversational context. The prompt is built as follows:

| {task_description} PERSONA: {persona} REQUIREMENTS: You are looking for a {item_type} with the following characteristics: {constraints}. Once you find a suitable {item_type}, make sure to get the following information: {requests}.
| {conversational_context}

The persona section is included if the simulated user persona is provided. The placeholder *item_type* is replaced by the type of item the user is looking for such as a restaurant or a movie. The *constraints* and *requests* are extracted from the information need. The *conversational_context* is the history of the conversation up to the current utterance, hence, it is updated each time an utterance is received (agent utterance) or generated (simulated user utterance).


LLM interface
-------------

:py:mod:`usersimcrs.simulator.llm.interface`

The LLM interface is responsible for interacting with the large language model to generate responses. Currently, two LLM interfaces are supported: Ollama and OpenAI. 

Ollama
^^^^^^

:py:class:`usersimcrs.simulator.llm.interface.ollama_interface.OllamaLLMInterface`

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

:py:class:`usersimcrs.simulator.llm.interface.openai_interface.OpenAILLMInterface`

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


**Footnotes**

.. [1] Silvia Terragni, Modestas Filipavicius, Nghia Khau, Bruna Guedes, André Manso, and Roland Mathis. 2023. In-Context Learning User Simulators for Task-Oriented Dialog Systems. arXiv:2306.00774 [cs.CL].