Agenda-based simulator
======================

The agenda-based simulator `[Schatzmann et al., 2007] <https://aclanthology.org/N07-2038/>`_ follows the architecture of a typical task-based dialogue system, which consists of natural language understanding (NLU), dialogue policy, and natural language generation (NLG) components. It is designed to ensure that the simulated user adheres to a predetermined dialogue strategy by maintaining an agenda (or stack) of actions.

Please refer to :doc:`NLU <nlu>` and :doc:`NLG <nlg>` for more details on the NLU and NLG components, respectively.

Dialogue Policy
---------------

The simulated user's decision-making is modeled as a Markov Decision Process. At each turn, it determines the next action to execute based on the current state of this agenda.

Specifically, the simulator's next action is determined by the agent response. 
If the agent responds expectedly, the next user action is pulled from the top of the agenda; otherwise, the simulator samples the next user action based on transition probabilities from responses in historical dialogues.

Agenda initialization
^^^^^^^^^^^^^^^^^^^^^

The agenda (:py:class:`usersimcrs.simulator.agenda_based.agenda.Agenda`) is initialized based on the :doc:`information need <information_need>` of the simulated user. Specifically, the agenda is initialized with the following steps:

1. (optional) Add the start intent
2. Add dialogue acts with the *INFORM* intent for each constraint in the information need
3. Add dialogue acts with the *REQUEST* intent for each request in the information need
4. Add the stop intent

Example
"""""""

For example, the following information need:

.. code-block:: json
    
    {
        "constraints": {
            "genre": "comedy"
        },
        "requests": ["plot"],
        "target_items": ["Jump Street", "The Hangover"]
    }

will result in the following agenda:

.. code-block:: python

    [
        START(),
        INFORM("genre", "comedy"),
        REQUEST("plot"),
        STOP()
    ]

Agenda update
^^^^^^^^^^^^^

The agenda is updated after each agent utterance by the :doc:`interaction model <interaction_model>`. The interaction model determines if new actions should be created or sampled and added to the agenda. For example, if the agent recommends an item, the interaction model may decide to create an action to express a preference regarding the recommended item.

**Reference**

Jost Schatzmann, Blaise Thomson, Karl Weilhammer, Hui Ye, and Steve Young. 2007. Agenda-Based User Simulation for Bootstrapping a POMDP Dialogue System. In Human Language Technologies 2007: The Conference of the North American Chapter of the Association for Computational Linguistics; Companion Volume, Short Papers (NAACL '07).