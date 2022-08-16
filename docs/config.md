# Configuration parameters

The agenda based simulator has a number of parameters that can be customized by
the users of UserSimCRS. These can be either provided in a config file and/or via the command line. Note that arguments passed through the command line will override its counterpart in the config file.

## Example config file
The config file must be a .ini file. All parameters should be under a SETTINGS flag, see example below.
```ini
[SETTINGS]
ontology=data\ontology.yaml
items=data\movielens-25m-sample\movies_w_keywords.csv
ratings=data\movielens-25m-sample\ratings.csv
dialogues=data\agents\moviebot\annotated_dialogues_v2.json
im=data\interaction_models\cir6_v2.yaml
```

## Usage
* All parameters in a .ini file:
  ```shell
  python -m usersimcrs.run_simulation -config configs/agenda_based.ini
  ```
* Overriding parameters via command line: 
  ```shell
  python -m usersimcrs.run_simulation -config configs/agenda_based.ini -ontology data\ontology.yaml
  ```