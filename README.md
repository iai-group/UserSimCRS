# UserSimCRS

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![docs](https://img.shields.io/github/actions/workflow/status/iai-group/UserSimCRS/build_docs.yaml?label=docs&branch=main)
![Tests](https://img.shields.io/github/actions/workflow/status/iai-group/UserSimCRS/merge.yaml?label=Tests&branch=main)
![Coverage Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/NoB0/cd558f4b76df656b67277f8ae214b7e0/raw/coverage.UserSimCRS.main.json)
![Python version](https://img.shields.io/badge/python-3.9-blue)

UserSimCRS is an extensible user simulation toolkit for evaluating conversational recommender systems.  

It is built on top of the [DialogueKit](https://github.com/iai-group/dialoguekit) library, which provides basic dialogue management functionalities.

UserSimCRS currently supports the following user simulation approaches:

  - **Agenda-based simulation**: This approach follows the architecture of a typical task-based dialogue system, which consists of natural language understanding, dialogue policy, and natural language generation components.
  - **LLM-based simulation**: This approach relies on prompting large language models (LLMs).

UserSimCRS also includes:

  - User modeling capabilities to simulate different user behaviors and preferences
  - Support for various datasets using a unified data format and LLM-based utils for augmentation
  - Item collections using a unified representation of items
  - Integration with IAI MovieBot v1.0.1 and CRSs available in the [CRS Arena](https://huggingface.co/spaces/iai-group/CRSArena)
  - Evaluation utility using LLM-as-a-judge

<img src="docs/source/_static/UserSimCRS-Overview.png" width="600px" alt="UserSimCRS architecture" />

We refer to the [documentation](https://iai-group.github.io/UserSimCRS/main) for details. Specifically, see [this page](https://iai-group.github.io/UserSimCRS/main/configuration.html) on how to set up an existing agent to be evaluated using UserSimCRS.

## Installation

The recommended version of Python is 3.9.  
The easiest way to install UserSimCRS and all of its dependencies is by using pip:

```shell
python -m pip install -r requirements/requirements.txt
```

To work on the documentation you also need to install other dependencies:

```shell
python -m pip install -r requirements/docs_requirements.txt
```

## Usage

A YAML configuration file is necessary to start the simulation; see [default configuration](config/default/config_default.yaml) for an example.  
Run the following command to start the simulation:

```shell
python -m usersimcrs.run_simulation -c <path_to_config.yaml>
```

### Example

This example shows how to run simulation using the default configuration and the [IAI MovieBot](https://github.com/iai-group/MovieBot) as the conversational agent.

1. Start IAI MovieBot locally

  - Download IAI MovieBot v1.0.1 [here](https://github.com/iai-group/MovieBot/releases/tag/v1.0.1)
  - Follow the IAI MovieBot installation instructions
  - Create a folder `conversation_history` in the folder you run MovieBot from.
  - Start the IAI MovieBot locally: `python -m moviebot.run -c config/moviebot_config_no_integration.yaml`

Note: you need to update the parameter `agent_uri` in the configuration in case MovieBot does not run on the default URI (i.e., `http://127.0.0.1:5001`).

2. Run simulation

```shell
python -m usersimcrs.run_simulation -c config/default/config_default.yaml
```

After the simulation, the YAML configuration is saved under `data/runs` using the `output_name` parameter.
The simulated dialogue is saved under `dialogue_export`.

## Conventions

We follow the [IAI Python Style Guide](https://github.com/iai-group/styleguide/tree/main/python).

## Contributors

UserSimCRS is developed and maintained by the [IAI group](https://iai.group) at the University of Stavanger.

(Alphabetically ordered by last name)

  - Jafar Afzali (2022)
  - Krisztian Balog (2021-present)
  - Nolwenn Bernard (2022-present)
  - Aleksander Drzewiecki (2022)
  - Shuo Zhang (2021)

We welcome contributions both on the high level (feedback and ideas) as well as on the more technical level (pull requests). See our [contribution guidelines](https://github.com/iai-group/guidelines/blob/main/github/Contribution.md) for more details.

## Publication

If you are using our simulation tool, please cite the following paper:

```
@inproceedings{Afzali:2023:WSDM,
  author = {Afzali, Jafar and Drzewiecki, Aleksander Mark and Balog, Krisztian and Zhang, Shuo},
  title = {UserSimCRS: A User Simulation Toolkit for Evaluating Conversational Recommender Systems},
  year = {2023},
  booktitle = {Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining},
  pages = {1160--1163},
  series = {WSDM '23}
}
```
