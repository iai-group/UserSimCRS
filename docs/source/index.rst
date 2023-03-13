UserSimCRS |release| documentation
==================================

UserSimCRS is an extensible user simulation toolkit for evaluating conversational recommender systems.

It is built on top of the `DialogueKit <iai-group.github.io/DialogueKit/main/>`_ library, which provides basic dialogue management functionalities.

UserSimCRS follows the architecture of a typical task-based dialogue system, which consists of natural language understanding, response generation, and natural language generation components. Additionally, there is a dedicated user modeling component in order to make simulation more human-like.

.. image:: _static/UserSimCRS-Overview.png
    :width: 700
    :alt: Conceptual overview of the user simulator.

This toolkit offers repeatable and reproducible means of evaluation for conversational recommender systems that can complement human evaluation.
UserSimCRS is designed to work with existing conversational recommender systems, without needing access to source code or knowledge of their inner workings.
UserSimCRS can also be extended with other types of simulators and user modeling options.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   usage
   interaction_model
   components
   configuration
   :ref:`modindex`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

