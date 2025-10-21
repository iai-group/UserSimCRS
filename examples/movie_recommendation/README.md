# Example: Movie Recommendation

This folder contains all configuration files and scripts to run simulation-based evaluation of CRS trained for movie recommendation with multiple user simulators configured/trained with different datasets.

## Conversational Recommender Systems

  * IAI MovieBot 1.0.1: rules-based CRS
  * CRSs from CRS Arena

IAI MovieBot implementation is available [here](https://github.com/iai-group/MovieBot/releases/tag/v1.0.1), while the implementations of CRSs from the CRS Arena are available [here](https://github.com/iai-group/iEvaLM-CRS).

## User Simulators

  * ABUS: agenda-based user simulator
  * SP-LLMUS: single prompt LLM-based user simulator
  * DP-LLMUS: dual prompt LLM-based user simulator

Configurations of the user simulators for the different datasets are available in the `config` folder of each dataset subfolder.

## Running Evaluation

The evaluation is performed in two steps:

1. Generate interaction data between the different CRS and user simulator pairs using the following command:

*IAI MovieBot* (config files starting with `config_iai_crs_`):

```bash
python -m usersimcrs.run_simulation \
    -c <path_to_config_file>
```

*CRSs from CRS Arena* (config files starting with `config_ievalm_`):

```bash
python -m usersimcrs.run_simulation \
    -c <path_to_config_file> \
    --agent_id <agent_id> \
    --agent_uri <agent_uri> \
    --output_name <output_name>
````

2. Compute evaluation metrics using the following command:

*User satisfaction*

```bash
python -m scripts.evaluation.satisfaction_evaluation \
        --dialogues <path_to_synthetic_dialogues>  
```

*Conversation quality aspects*

```bash
python -m scripts.evaluation.quality_evaluation \
    --dialogues <path_to_synthetic_dialogues>  \
    --ollama_config config/llm_interface/config_ollama_information_need.yaml \
    --output <path_to_save_results>
```

*Note*: The commands assume that you have already set up the environment and installed all necessary dependencies as described in the main README file.

## Results

The generated synthetic dialogues for ECIR 2026 submission are saved in the `synthetic_dialogues` folder, and the conversation quality aspects' scores are saved in the `results` folder.

## Troubleshooting

  * Cannot run the scripts: Ensure that you have execution permissions. You can set the permissions using `chmod +x script_name.sh`.
  * Errors related to missing dependencies: Ensure that all required libraries and tools are installed in your environment.
  * Issues with specific CRS: check that the CRS API endpoints are correctly configured and that the services are running.
