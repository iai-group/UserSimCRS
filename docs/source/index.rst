UserSimCRS documentation
========================

UserSimCRS is an extensible user simulation toolkit for evaluating conversational recommender systems.

It is built on top of the `DialogueKit <iai-group.github.io/DialogueKit/main/>`_ library, which provides basic dialogue management functionalities.

UserSimCRS currently supports the following user simulation approaches:

- **Agenda-based simulation**: This approach follows the architecture of a typical task-based dialogue system, which consists of natural language understanding, response generation, and natural language generation components.
- **LLM-based simulation**: This approach relies on large language models (LLMs).

UserSimCRS also includes a dedicated user modeling component in order to make simulation more human-like. It provides datasets formatted to be serve as training data for the components that require it, such as the natural language understanding and interaction model components; they can also be used for in-context learning for LLM-based components.

.. image:: _static/UserSimCRS-Overview.png
    :width: 400
    :alt: Overview of UserSimCRS architecture

This toolkit offers repeatable and reproducible means of evaluation that can complement human evaluation.
UserSimCRS is designed to work with existing conversational recommender systems, without needing access to source code or knowledge of their inner workings.
UserSimCRS can also be extended with other simulation approaches and more advanced user modeling capabilities.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   usage
   agenda_based
   llm_based
   user_modeling
   configuration
   :ref:`modindex`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

