Intent schema
=============

The interaction model defines the user-system interactions that can take place in a conversation. We define it in the form of an *intent schema*, as specified below.

Format
------

* **user_intents**:  List of all user intent where each should contain at least **expected_agent_intents**.
  * Additionally, if a user intent is dependent on the preference model, this should be indicated via another key, i.e., **preference_contingent**.
  * Similarly, intents that are used to remove preferences should be indicated in another key, **remove_user_preference**.
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
