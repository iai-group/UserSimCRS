Setting up a user simulator
----------------------------

Requirements
============

To run the simulation, you will need the following:

1. A YAML file with domain-specific slot names that will be used for the preference model.
2. A CSV file containing the item collection. This file must at least contain 2 columns labeled *ID* and *NAME*.
3. A CSV file containing the item ratings in the shape of user ID, item ID, and rating triples.
4. A YAML file containing the users’ and agent’s intent space, as well as the set of expected agents for each user intent, is required for the interaction model. The CIR6 interaction model shipped with library offers a baseline starting point, which may be further tailored according to the behavior and capabilities of the target CRS.
5. A small sample of annotated dialogues with the target CRS. The sample size depends on the complexity of the system, in terms of action space and language variety, but is generally in the order of 5-50 dialogues. The sample of dialogues must contain utterance-level annotations in terms of intents and entities, as this is required to train the NLU and NLG components. Note that the slots used for annotation should be the same as the ones defined in the domain file (1) and intents should follow the ones defined in the interaction model (4)

Configuration
=============

The agenda based simulator has a number of parameters that can be customized.
These can be either provided in a YAML config file and/or via the command line. Note that arguments passed through the command line will override those in the config file.

Simulator parameters
^^^^^^^^^^^^^^^^^^^^

* agent_id: id of the agent tested.
* output_name: specifies the output name for the simulation configuration that will be stored under `data/runs` at the end of the simulation
* agent_uri: URI to communicate with the agent. By default we assume that the agent has an HTTP API
* domain: A YAML file with domain-specific slot names
* intents: Path to the intent schema file
* items: Path to items file
* ratings: Path to ratings file
* dialogues: Path to domain config file
* intent_classifier: Intent classifier model to be used. Only supports DialogueKit intent classifier
* rasa_dialogues: File with Rasa annotated dialogues. Only needed when using a DIET intent classifier

Configuration example
^^^^^^^^^^^^^^^^^^^^^

Below you have the default configuration to run simulation with the IAI MovieBot as the conversational agent.

.. todo:: Add args for context and persona.

.. code-block:: yaml
  
  agent_id: "IAI MovieBot"
  output_name: "moviebot"
  # By default, the agent has an HTTP API.
  agent_uri: "http://127.0.0.1:5001"
  domain: data/domains/movies.yaml
  intents: data/interaction_models/crs_v1.yaml
  items: data/movielens-25m-sample/movies_w_keywords.csv
  ratings: data/movielens-25m-sample/ratings.csv
  dialogues: data/agents/moviebot/annotated_dialogues.json
  intent_classifier: "cosine"
  # If using the diet classifier the following file needs to be provided. 
  # rasa_dialogues: data/agents/moviebot/annotated_dialogues_rasa_agent.yml