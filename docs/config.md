# Configuration parameters

The agenda based simulator has a number of parameters that can be customized.
These can be either provided in a YAML config file and/or via the command line. Note that arguments passed through the command line will override those in the config file.

* `agent_id`: Id of the agent tested.
* `output_name`: Specifies the output name for the simulation configuration that will be stored under `data/runs` at the end of the simulation.
* `agent_uri`: URI to communicate with the agent. By default we assume that the agent has an HTTP API.
* `domain`: A YAML file with domain-specific slot names.
* `intents`: Path to the intent schema file.
* `items`: Path to items file.
* `id_col`: Name of the CSV field containing item id.
* `domain_mapping`: CSV field mapping to create item based on domain slots.
* `ratings`: Path to ratings file.
* `dialogues`: Path to domain config file.
* `intent_classifier`: Intent classifier model to be used. Only supports DialogueKit intent classifier.
* `rasa_dialogues`: File with Rasa annotated dialogues. Only needed when using a DIET intent classifier.
* `debug`: Flag (boolean) to activate debug mode.

## Example config file

Below you have the default configuration to run simulation with the IAI MovieBot as the conversational agent.

```yaml
agent_id: "IAI MovieBot"
output_name: "moviebot"
# By default, the agent has an HTTP API.
agent_uri: "http://127.0.0.1:5001"

domain: data/domains/movies.yaml
intents: data/interaction_models/crs_v1.yaml

items: data/movielens-25m-sample/movies_w_keywords.csv
id_col: movieId
domain_mapping:
  title:
    slot: TITLE
  genres:
    slot: GENRE
    multi-valued: True
    delimiter: "|"
  keywords:
    slot: KEYWORD
    multi-valued: True
    delimiter: "|"
ratings: data/movielens-25m-sample/ratings.csv

dialogues: data/agents/moviebot/annotated_dialogues.json
intent_classifier: "cosine"
# If using the DIET classifier the following file needs to be provided. 
# rasa_dialogues: data/agents/moviebot/annotated_dialogues_rasa_agent.yml

debug: False
```
