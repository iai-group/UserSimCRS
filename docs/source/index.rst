UserSimCRS documentation
========================

UserSimCRS is an extensible user simulation toolkit for evaluating conversational recommender systems.

It is built on top of the `DialogueKit <iai-group.github.io/DialogueKit/main/>`_ library, which provides basic dialogue management functionalities.

UserSimCRS currently supports the following user simulation approaches:

- **Agenda-based simulation**: This approach follows the architecture of a typical task-based dialogue system, which consists of natural language understanding, dialogue policy, and natural language generation components.
- **LLM-based simulation**: This approach relies on prompting large language models (LLMs).

UserSimCRS also includes:

- User modeling capabilities to simulate different user behaviors and preferences
- Support for various datasets using a unified data format and LLM-based utils for augmentation
- Item collections using a unified representation of items
- Integration with IAI MovieBot v1.0.1 and CRSs available in the `CRS Arena <https://huggingface.co/spaces/iai-group/CRSArena>`_
- Evaluation utility using LLM-as-a-judge


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

