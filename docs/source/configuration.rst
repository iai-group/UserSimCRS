Experiment Configuration
========================

This page describes how to configure the simulation-based evaluation of a conversational recommender system (CRS).  

Requirements
------------

To run the simulation, the following are needed:

1. **Domain:** A YAML file with domain-specific slot names that will be used for the preference model.
2. **Item collection:** A CSV file containing the item collection. This file must contain at least 2 columns labeled *ID* and *NAME*.
3. **Preference data:** A CSV file containing the item ratings in the shape of user ID, item ID, and rating triples.


Configuration
-------------

The configuration of an experiment is defined in a YAML file (some parameters can also be passed via the command line). The configuration file is divided into three main sections:

1. **General parameters**: These parameters define the general settings of the simulation. It includes the following fields:
  - `output_name`: Experiment name.
  - `debug`: A boolean flag to activate debug mode.
  - `fix_random_seed`: A boolean flag to fix the random seed for reproducibility.
  - `num_simulated_dialogues`: Number of simulated dialogues to be generated during the experiment. 

2. **Agent configuration**: Configuration of the agent to be tested in the simulation. It includes the following fields:
  - `agent_id`: Agent identifier.
  - `agent_uri`: URI to communicate with the agent (by default, it is assumed that the agent has an HTTP API).
  - `agent_class_path`: Import path to the CRS wrapper class. This is used to instantiate the agent in the simulation. More details on CRS wrapper is provided :doc:`here <crs_wrapper>`.
  - `agent_config` (optional): Dictionary with additional configuration parameters for the agent.

3. **Simulator configuration**: Configuration of the user simulator. It includes at least the following fields:
  - `simulator_id`: User simulator identifier.
  - `simulator_class_path`: Import path to the user simulator class. This is used to instantiate the user simulator in the simulation.
  - `domain`: A YAML file with domain-specific slot names.
  - `items`: Path to items file.
  - `id_col`: Name of the CSV field containing item id.
  - `domain_mapping`: CSV field mapping to create item based on domain slots.
  - `ratings`: Path to ratings file.
  - `historical_ratings_ratio`: Ratio ([0..1]) of ratings to be used as historical data.


Note that additional parameters specific to the user simulator should be added in the simulator configuration section. For example, it can be parameters related to the natural language understanding components or related to the large language model used for the simulation. We discuss some of parameters for agenda-based and LLM-based simulators below. In general, the parameters correspond to the ones defined in the specific simulator class.

Agenda-based simulation configuration
"""""""""""""""""""""""""""""""""""""

To build an agenda-based user simulator, you need to provide an **interaction model**. It is a YAML file containing the users' and agent's intent space, as well as the set of expected agent intents for each user intent, is required for the interaction model. The CIR6 interaction model shipped with library offers a baseline starting point, which may be further tailored according to the behavior and capabilities of the CRS. The path to the interaction model file should be specified in the configuration file under the `intents` parameter.

A default configuration to experiment with the IAI MovieBot, as the conversational agent, and agenda-based user simulator with supervised NLU and NLG is provided in `config/default/config_default.yaml`.

Supervised NLU and NLG
^^^^^^^^^^^^^^^^^^^^^^

To use supervised NLU and NLG components in the simulation, you need access to:

  * **Annotated sample dialogues:** A small sample of annotated dialogues with the CRS. The sample size depends on the complexity of the system, in terms of action space and language variety, but is generally in the order of 5-50 dialogues. The sample of dialogues must contain utterance-level annotations in terms of intents and entities, as this is required to train the NLU and NLG components. Note that the slots used for annotation should be the same as the ones defined in the domain file and intents should follow the ones defined in the interaction model.


Associated configuration parameters are:

  
  * `dialogues`: Path to domain config file.
  * `intent_classifier`: Intent classifier model to be used. Only supports DialogueKit intent classifiers.
  * `nlg`: NLG type to be used, set to "conditional" for template-based NLG.
  * `debug`: Flag (boolean) to activate debug mode.


LLM-based NLU and NLG
^^^^^^^^^^^^^^^^^^^^^

Additional parameters for LLM-based NLU and NLG components are:

  * `intent_classifier`: Intent classifier set to "llm" for LLM-based dialogue act extraction.
  * `intent_classifier_config`: Configuration file for `LLMDialogueActsExtractor`.
  * `nlg`: NLG type set to "llm" for LLM-based NLG.

    - `nlg_class_path`: Import path to the LLM-based NLG class.
    - `nlg_args`: Dictionary with additional configuration parameters for the LLM-based NLG class.


LLM-based simulation configuration
""""""""""""""""""""""""""""""""""

Additional parameters for the LLM-based user simulators are:

  * `llm_interface_class_path`: Import path to the LLM interface class. This is used to instantiate the LLM interface in the simulation.
  * `llm_interface_args`: Dictionary with additional configuration parameters for the LLM interface.
  * `item_type`: Type of items to be recommended.


Optional parameters for the LLM-based simulators include:

  * `task_definition`: Task description to be used in the utterance generation prompt.
  * `stop_definition` (only for `DualPromptUserSimulator`): Task description to be used in the stop decision prompt.


.. todo: A default configuration to experiment with the IAI MovieBot, as the conversational agent, and single prompt user simulator is provided in `config/default/config_default.yaml`.
