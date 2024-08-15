Agenda-based simulator
======================

The agenda-based simulator is designed to ensure that the simulated user adheres to a predetermined dialogue strategy by maintaining an agenda (or stack) of actions. At each turn, it determines the next action to execute based on the current state of this agenda. The simulated user's decision-making is modeled as a Markov Decision Process.

Specifically, the simulator's next action is determined by the agent response. 
If the agent responds expectedly, the next user action is pulled from the top of the agenda; otherwise, the simulator samples the next user action based on transition probabilities from responses in historical dialogues.

Agenda initialization
---------------------

The agenda (:py:class:`usersimcrs.simulator.agenda_based.agenda.Agenda`) is initialized based on the :doc:`information need <information_need>` of the simulated user. Specifically, the agenda is initialized with the following steps:

1. (optional) Add the start intent
2. Add dialogue acts with the *INFORM* intent for each constraint in the information need
3. Add dialogue acts with the *REQUEST* intent for each request in the information need
4. Add the stop intent

Example
^^^^^^^

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

.. code-block:: json

    [
        START_INTENT(),
        INFORM("genre", "comedy"),
        REQUEST("plot"),
        STOP_INTENT()
    ]

Agenda update
-------------

The agenda is updated after each agent utterance by the interaction model. The interaction model determines if new actions should be created or sampled and added to the agenda. For example, if the agent recommends an item, the interaction model may decide to create an action to express a preference regarding the recommended item. More details on the interaction model are provided :doc:`here <interaction_model>`.
