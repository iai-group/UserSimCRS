Evaluation
==========

UserSimCRS evaluates conversational recommender systems (CRSs) on exported dialogues. The evaluation pipeline loads dialogues from a JSON file, computes one or more metrics, and stores the results as JSON together with the resolved configuration.

A default evaluation configuration is provided in `config/default/config_evaluation.yaml`.


Usage
-----

Run evaluation with:

.. code-block:: shell

    python -m usersimcrs.run_evaluation -c <path_to_config.yaml>


Some parameters can also be overridden from the command line, for example:

.. code-block:: shell

    python -m usersimcrs.run_evaluation \
      -c config/default/config_evaluation.yaml \
      --dialogues data/datasets/moviebot/annotated_dialogues.json \
      --metrics satisfaction success_rate \
      --output data/evaluation/results.json


Configuration
-------------

The evaluation configuration is defined in a YAML file. The main parameters are:

  * `dialogues`: Path to the dialogues JSON file.
  * `metrics`: List of metrics to compute.
  * `output`: Path to the JSON file where evaluation results will be saved.
  * `quality_aspects`: Quality aspects to evaluate when `quality` is included in `metrics`.
  * `quality_llm_interface`: LLM interface configuration used by the quality metric.
  * `user_nlu_config`: Configuration file used to instantiate the user-side NLU for utility metrics.
  * `agent_nlu_config`: Configuration file used to instantiate the agent-side NLU for utility metrics.
  * `recommendation_intent_labels`: Intent labels that mark recommendation turns.
  * `accept_intent_labels`: Intent labels that mark acceptance.
  * `reject_intent_labels`: Intent labels that mark rejection.


The following metrics are currently supported:

  * `quality`
  * `satisfaction`
  * `success_rate`
  * `successful_recommendation_round_ratio`
  * `reward_per_dialogue_length`


Metric Overview
---------------

Quality
"""""""

The quality metric uses an LLM to score each dialogue aspect separately. The supported aspects are defined by ``QualityRubrics``:

  * `REC_RELEVANCE`
  * `COM_STYLE`
  * `FLUENCY`
  * `CONV_FLOW`
  * `OVERALL_SAT`


When `quality` is requested, the configuration must include `quality_llm_interface`.


Satisfaction
""""""""""""

The satisfaction metric uses the pre-trained DialogueKit satisfaction classifier and returns one score per dialogue.


Utility Metrics
"""""""""""""""

The utility metrics are:

  * **Success rate**: Returns `1.0` if at least one recommendation was accepted in the dialogue, otherwise `0.0`.
  * **Successful recommendation round ratio**: Returns the ratio of accepted recommendation rounds to all recommendation rounds in the dialogue.
  * **Reward per dialogue length**: Returns the number of accepted recommendations divided by the total number of utterances in the dialogue.


If the input dialogues are not already annotated, UserSimCRS annotates them in place using the NLU components loaded from `user_nlu_config` and `agent_nlu_config`.

When any utility metric is requested, the following configuration fields are required:

  * `user_nlu_config`
  * `agent_nlu_config`
  * `recommendation_intent_labels`
  * `accept_intent_labels`
  * `reject_intent_labels`


Output
------

The evaluation script writes two files:

  * The JSON result file specified by `output`.
  * A companion metadata file with the suffix `.meta.yaml`, containing the resolved configuration.


The result JSON contains:

  * `dialogues_path`: Path to the evaluated dialogues.
  * `metrics_requested`: List of requested metrics.
  * `metrics`: Metric results.


For `satisfaction` and all utility metrics, each metric entry contains:

  * `per_dialogue`: Mapping from conversation ID to score.
  * `summary_by_agent`: Aggregate statistics per agent (`count`, `min`, `max`, `mean`, `stdev`).


For `quality`, the output is grouped by aspect. Each aspect contains its own `per_dialogue` scores and `summary_by_agent` statistics.
