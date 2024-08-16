Interaction model
================= 

The interaction model is based on (Zhang & Balog, 2020) [1]_ and defines the allowed transitions between dialogue acts based on their intents. In our implementation, the interaction model is also responsible for the updating of the agenda based on a predefined dialogue strategy.

Define allowed transitions
--------------------------

The interaction model defines the user-agent interactions in terms of *intents* from their respective dialogue acts. The model specifies a set of *user intents* including required ones, a set of *agent intents*, and *expected agent responses* to each user intent. 

Format
^^^^^^

Below, we specify the YAML format that is used for defining an interaction model.

* **required_intents**: List of minimum required intents for the user.
* **user_intents**:  List of all user intents; each should minimally specify **expected_agent_intents**.

  - Additionally, if a user intent is dependent on the preference model, this should be indicated via another key, i.e., **preference_contingent**.
  - Similarly, intents that are used to remove preferences should be indicated in another key, **remove_user_preference**.

* **sub_intents**: Intents that are a variation of another intent can be listed as **sub_intents** of the same main intent. We separate them with a **"."**, where the former part indicates the main intent, and the latter the sub-intent. For example, REVEAL.EXPAND is a sub-intent of REVEAL (main intent).
* **agent_elicit_intents**: List of intents that the agent can use to elicit preferences/need from the user. 
* **agent_set_retrieval**: List of intents that the agent can use to reveal information to the user.
* **agent_inquire_intents**: List of intents that the agent can use to ask the user if they want to know more.
* **REWARD**: The reward settings for automatic assessment of simulated dialogues.

An example of interaction model is available at: `data/interaction_models/crs_v1.yaml`.

New interaction models can be added by providing a YAML file with the same format as the example above. The path to this file can be provided either in the configuration file or the command line, see :ref:`Configuration`.

Agenda update
-------------

The agenda is updated based on the last agent dialogue acts and the current state of the conversation. For each agent dialogue act, the interaction model performs a push operation on the agenda stack. We consider four cases:

1. **Agent elicits**: In case the agent elicits information from the user, the interaction model may push a new dialogue act to disclose the information elicited.
2. **Agent recommends**: In case the agent recommends an item, the interaction model may push a new dialogue act to express a preference regarding the recommended item (e.g., like, dislike, already consumed).
3. **Agent inquires**: In case the agent inquires if the user wants to know more about a specific item, the interaction model may push a new dialogue act to request a slot in the information need or a random one.
4. **None of the above**: In case none of the above cases apply, the interaction model checks if it is coherent to continue with the current agenda or if a new dialogue act should be sampled to keep the conversation going. In the latter case, the new dialogue act is sampled based on the transition probabilities from historical dialogues.

Once all the push operations are done, the agenda is cleaned to discard unnecessary dialogue acts, e.g., duplicates.

Transition probabilities matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The interaction model uses transition probabilities matrices to sample new dialogue acts. These matrices are built from historical dialogues when the model is initialized. The transition probabilities are calculated based on the frequency of transitions between intents in the historical dialogues. We consider two matrices:

* *Single intent*: The set of intents from an utterance is considered individually. That is, the transition probabilities are calculated based on the frequency of each intent following another intent.
* *Compound intent*: The set of intents from an utterance is considered as a whole, i.e., the sequence of intents is considered as a single entity. That is, the transition probabilities are calculated based on the frequency of each sequence of intents following another sequence of intents.

For example the following consecutive sequence of dialogue acts:

| > Agent: [GREETINGS(), ELICIT(genre=?)]  
| > User: [GREETINGS(), DISCLOSE(genre=action)]

will result in the following transition probabilities matrices:

*Single intent*:

+-----------+-----------+----------+
|           | GREETINGS | DISCLOSE |
+-----------+-----------+----------+
| GREETINGS | 0.5       | 0.5      |
+-----------+-----------+----------+
| ELICIT    | 0.5       | 0.5      |
+-----------+-----------+----------+

*Compound intent*:

+-------------------+--------------------+
|                   | GREETINGS_DISCLOSE |
+-------------------+--------------------+
| GREETINGS_ELICIT  | 1                  |
+-------------------+--------------------+

**Footnotes**

.. [1] Shuo Zhang and Krisztian Balog. 2020. Evaluating Conversational Recommender Systems via User Simulation. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20). 1512--1520.