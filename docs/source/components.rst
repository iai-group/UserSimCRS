Main Components
===============

.. image:: _static/UserSimCRS-Overview.png
    :width: 700
    :alt: Conceptual overview of the user simulator.


Natural language understanding (NLU)
------------------------------------

The NLU is responsible for obtaining a structured representation of text utterances. This entails intent classification and entity recognition. In addition to this, the NLU can also do satisfaction prediction. This is the users satisfaction with the agents response.

Response generation
-------------------

Response generation is currently developed with an agenda-based simulator [2]_ in mind, however, it could be replaced with other approaches in the future. Following Zhang and Balog [4]_ , response generation is based on an interaction model, which is responsible for initializing the agenda and updating it. Updates to the agenda can be summarized as follows: if the agent responds in an expected manner, the interaction model pulls the next action off the agenda; otherwise, it either repeats the same action as the previous turn or samples a new action. Additionally, motivated by recent work in [1]_ and [3]_, we also introduce a user satisfaction prediction component. In addition to deciding which action to take next, the generated response also includes a turn-level user satisfaction estimate. This may be utilized by the NLG component when generating the text of the user utterance.

User modeling
-------------

User modeling consists of three sub-components: preference model, context model, and persona. The responsibility of the preference model is to capture users’ individual tastes and thus allow for a personalized experience. Following Zhang and Balog [4]_, it is modeled as a personal knowledge graph, where nodes can indicate either an item or an attribute. Novel to our work is the modeling of persona, which is used to capture user-specific traits, e.g., user cooperativeness, and context, which can characterize the situation of the user.

Natural language generation (NLG) 
---------------------------------

Following the work in [4]_, the NLG component is template-based, that is, given the output of the response generation module, a fitting textual response is chosen and may be instantiated with preferences. Additionally, we extend template-based generation to be conditioned on user satisfaction, such that users could use for example stronger language when getting dissatisfied with the system.

Footnotes
---------

.. [1] Alexandre Salle, Shervin Malmasi, Oleg Rokhlenko, and Eugene Agichtein. 2021. Studying the Effectiveness of Conversational Search Refinement Through User Simulation. In Proceedings of the 43rd European Conference on IR Research (ECIR ’21). 587–602.


.. [2] Jost Schatzmann, Blaise Thomson, Karl Weilhammer, Hui Ye, and Steve Young. 2007. Agenda-Based User Simulation for Bootstrapping a POMDP Dialogue System. In Human Language Technologies 2007: The Conference of the North American Chapter of the Association for Computational Linguistics; Companion Volume, Short Papers (NAACL ’07).

.. [3] Weiwei Sun, Shuo Zhang, Krisztian Balog, Zhaochun Ren, Pengjie Ren, Zhumin Chen, and Maarten de Rijke. 2021. Simulating User Satisfaction for the Evaluation of Task-Oriented Dialogue Systems. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’21). 2499–2506.

.. [4] Shuo Zhang and Krisztian Balog. 2020. Evaluating Conversational Recommender Systems via User Simulation. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’20). 1512–1520.