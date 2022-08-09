Setting up an agent
===================

How to connect
-----------------

*TODO: what interface to implement for the agent; examples of how to connect Telegram and FB applications*

Data
-----

Intent scheme
^^^^^^^^^^^^^

The *intent scheme* defines the space of actions supported by the conversational agent and the user simulator.


Annotated conversation logs
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Training data is accepted in JSON format, where each conversation is an entry in a dictionary, with utterances represented as a list of dictionaries.  Each utterance needs to include the participant, utterance text, and intent annotations, and can include arbitrary additional annotations.

.. code-block:: json

    {'conversation ID':
    [
        {
            "participant": "USER or AGENT",
            "utterance": "text",
            "intent": "REVEAL, ... ALLCAPS",
            "annotation_1": "value_1",
            "annotation_2": "value_2",
        },
        {...}
    ]
    ..
    }


The intent scheme is to be defined in a separate file, see [Intents](Intents.md).


Example:

.. code-block:: json
        
    {
        "1018216123": [
            {...},
            {
                "participant": "AGENT",
                "utterance": "Guide me. Any genres you like?",
                "intent": "ELICIT",
                "request": "GENRE"
            },
            {
                "participant": "USER",
                "utterance": "Action",
                "intent": "DISCLOSE"
            },
            {
                "participant": "AGENT",
                "utterance": "There is a movie named \"Lone Wolf McQuade\". Have you watched it?",
                "intent": "LIST"
            },
            {
                "participant": "USER",
                "utterance": "No",
                "intent": "NOTE"
            },
            [...]
        ]
    }


Conversation logs are to be placed under  `data/agents/{agent}/annotated_dialogues/` as one or multiple JSON files.

What kind of data is needed to enable the simulator → JSON files containing annotations for dialogues

Measures
--------

*TODO: How to define custom measures*
