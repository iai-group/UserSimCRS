# Intent schema
The interaction model requires the intent schema to follow a specific format, as introduced below.

## Rules:
  * **user_intents** must encapsulate both user and expected agent intents.
  * **sub-intents** are separated by a **"."**; The former part indicates main intent, e.g., REVEAL.EXPAND is a sub-intent of REVEAL (main intent)
  * Each user intent should at least contain one key, i.e., the expected agent
  * intents. Additionally, if a user intent is dependent on the preference model, this should be indicated via another key, i.e., preference_contingent. 
  * Similarly, intents that are used to remove preferences should be indicated in another key.
  * The keys **preference_contingent** and **remove_user_preference** should be used only where necessary.
  * Example:
    ```yaml
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
    ```

Additionally, this file should contain the **REWARD** settings for evaluation purposes. We expect it to be in the form:
```yaml
REWARD:
  full_set_points: int (max score to start with)
  missing_intent_penalties:
    - intent: penalty (penalty associated with intent)
    - ...
  repeat_penalty: int (penalty for consecutive repeated intents)
  cost: int (cost for each turn)
```