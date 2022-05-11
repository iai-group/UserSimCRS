# MovieBot dialogue files

The `.yaml` files are generated from `annotated_dialogues.json` using DialogueKit package. They are necessary in order to train the Rasa DIET classifier. 

### A short description of these files:
  - `annotated_dialogues_rasa_agent.yaml`: A file containing examples of agent utterances for all possible intents/actions that the agent can take.
  - `annotated_dialogues_rasa_user.yaml`: Similar to the agent file, but for user.
  - `annotated_dialogues_reformat.yaml`: This file is essentially the same as `annotated_dialogues.json`, but in `.yaml` format.
  - `annotated_dialogues_types_w_examples.yaml`: A file with slots and example values.