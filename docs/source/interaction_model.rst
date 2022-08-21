Interaction model
================

The interaction model defines the user-agent interactions that can take place in a conversation. We define it in terms a set of *user intents*, a set of *agent intents*, and *expected agent responses* to each user intent.
Below, we specify the YAML format that is used for defining an interaction model and then discuss a specific interaction model (CIR6) that is shipped with the toolkit.

Format
------

* **user_intents**:  List of all user intent where each should contain at least **expected_agent_intents**.

  - Additionally, if a user intent is dependent on the preference model, this should be indicated via another key, i.e., **preference_contingent**.
  - Similarly, intents that are used to remove preferences should be indicated in another key, **remove_user_preference**.

* **sub_intents**: Intents that are a variation of an other intent can be listed as **sub_intents** of the main intent. We separate them with a **"."**, where the former part indicates the main intent, and the latter the sub intent. For example, REVEAL.EXPAND is a sub intent of REVEAL (main intent).
* **REWARD**: The reward settings for automatic assessment of simulated dialogues.

Example
^^^^^^^

.. code-block:: yaml
  
  user_intents:
    NOTE.YES:
      expected_agent_intents:
        - INQUIRE.ELICIT
        - REVEAL
        - REVEAL.SIMILAR
      preference_contingent: CONSUMED
    REVEAL.REVISE:
      expected_agent_intents:
        - ...
      remove_user_preference: true

  REWARD:
    full_set_points: 20
    missing_intent_penalties:
      - INQUIRE: 4
      - ...
    repeat_penalty: 1
    cost: 1

Adding a new interaction model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to add a new interaction model, a YAML file with the same format as the example above must be provided. The path to this file can be provided either in the configuration file or the command line, see :ref:`Configuration`.

Intents
-------

TODO: Update DKs intent link page once it is available
Intents are dialogue actions that can be performed by the participants in a dialogue. UserSimCRS uses DialogueKit's `implementation <https://github.com/iai-group/dialoguekit/tree/main/docs>`_ for intents.


CIR6  
^^^^
UserSimCRS by default implements the :ref:`CIR6 intent schema <Intent Schema>` which includes intents shown below.

User intents
""""""""""""

+-----------+------------------------------------------------------------------------------------------------------------+
| Intent    | Description                                                                                                |
+===========+============================================================================================================+
| COMPLETE  | Indicate the end of the conversation                                                                       |
+-----------+------------------------------------------------------------------------------------------------------------+
| DISCLOSE  | Information need expressed either actively or in response to the agent's question                          |
+-----------+------------------------------------------------------------------------------------------------------------+
| REVEAL    | Revising, refining, or expanding constraints and requirements                                              |
+-----------+------------------------------------------------------------------------------------------------------------+
| INQUIRE   | Ask for related items or similar options                                                                   |
+-----------+------------------------------------------------------------------------------------------------------------+
| NAVIGATE  | Actions around navigating a list of recommendations as well as questions about a certain recommended item  |
+-----------+------------------------------------------------------------------------------------------------------------+
| NOTE      | Mark or save specific items                                                                                |
+-----------+------------------------------------------------------------------------------------------------------------+


Agent intents
"""""""""""""

+-----------+----------------------------------------------------------+
| Intent    | Description                                              |
+===========+==========================================================+
| END       | Indicate the end of the conversation                     |
+-----------+----------------------------------------------------------+
| INQUIRE   | Ask for user preferences                                 |
+-----------+----------------------------------------------------------+
| REVEAL    | Display the recommendations fully, partially or further  |
+-----------+----------------------------------------------------------+
| TRAVERSE  | Actions in response to user navigate actions             |
+-----------+----------------------------------------------------------+
| RECORD    | Record the rated items                                   |
+-----------+----------------------------------------------------------+

