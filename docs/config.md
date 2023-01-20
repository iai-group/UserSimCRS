# Configuration parameters

The agenda based simulator has a number of parameters that can be customized.
These can be either provided in a YAML config file and/or via the command line. Note that arguments passed through the command line will override those in the config file.

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
ratings: data/movielens-25m-sample/ratings.csv
dialogues: data/agents/moviebot/annotated_dialogues.json
intent_classifier: "cosine"
# If using the diet classifier the following file needs to be provided. 
# rasa_dialogues: data/agents/moviebot/annotated_dialogues_rasa_agent.yml
```
