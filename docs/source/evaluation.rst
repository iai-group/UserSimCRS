Evaluation
==========

UserSimCRS evaluates conversational recommender systems (CRSs) on previously generated synthetic dialogues. The evaluation pipeline loads dialogues from a JSON file, computes one or more metrics, and stores the results as JSON together with the copy of the configuration used.

A default evaluation configuration is provided in `config/default/config_evaluation.yaml`.


Usage
-----

Run evaluation with:

.. code-block:: shell

    python -m usersimcrs.run_evaluation -c <path_to_config.yaml>


Some parameters can also be overridden from the command line, for example:

.. code-block:: shell

    python -m usersimcrs.run_evaluation \
      -c <path_to_evaluation_config.yaml> \
      --dialogues data/datasets/moviebot/annotated_dialogues.json \
      --metrics satisfaction success_rate \
      --output-dir data/evaluation

Run ``python -m usersimcrs.run_evaluation -h`` for the full list of available command-line arguments. The configuration fields used by these arguments are described below.


Configuration
-------------

The evaluation configuration is defined in a YAML file. The main parameters are:

  * `dialogues`: Path to the dialogues JSON file.
  * `metrics`: List of metrics to compute.
  * `output_dir`: Directory where evaluation results and metadata will be saved.
  * `quality_aspects`: Quality aspects to evaluate when `quality` is included in `metrics`.
  * `quality_llm_interface`: LLM interface configuration used by the quality metric.
  * `annotate_dialogues`: Whether dialogues should be annotated before metric computation.
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

:py:class:`usersimcrs.evaluation.quality_metric.QualityMetric`

The quality metric uses an LLM to score each dialogue aspect separately. The supported aspects are defined by ``QualityRubrics``:

  * `REC_RELEVANCE`: Recommendation relevance measures how closely the recommended items align with the user’s preferences and needs.
  * `COM_STYLE`: Communication style corresponds to the conciseness and clarity of the responses.
  * `FLUENCY`: Fluency is the degree of naturalness of the responses compared to human-generated responses.
  * `CONV_FLOW`: Conversational flow assesses the coherence and consistency of the conversation.
  * `OVERALL_SAT`: Overall satisfaction encapsulates the user’s holistic experience. 


When `quality` is requested, the configuration must include `quality_llm_interface`.


Satisfaction
""""""""""""

:py:class:`usersimcrs.evaluation.satisfaction_metric.SatisfactionMetric`

The satisfaction metric uses the pre-trained DialogueKit satisfaction classifier and returns one score per dialogue.


User Utility Metrics
""""""""""""""""""""

The user utility metrics capture recommendation outcomes from annotated dialogues. If the input dialogues are not already annotated, they can be annotated before evaluation by enabling `annotate_dialogues` and providing `user_nlu` and `agent_nlu` configurations. For additional context on their role in the evaluation setup, see `Bernard and Balog, 2025 <https://doi.org/10.1145/3767695.3769478>`_.


Success Rate
''''''''''''

:py:class:`usersimcrs.evaluation.success_rate_metric.SuccessRateMetric`

Returns `1.0` if at least one recommendation was accepted in the dialogue, otherwise `0.0`.


Successful Recommendation Round Ratio
''''''''''''''''''''''''''''''''''''''

:py:class:`usersimcrs.evaluation.successful_recommendation_round_ratio_metric.SuccessfulRecommendationRoundRatioMetric`

Returns the ratio of accepted recommendation rounds to all recommendation rounds in the dialogue.


Reward per Dialogue Length
''''''''''''''''''''''''''

:py:class:`usersimcrs.evaluation.reward_per_dialogue_length_metric.RewardPerDialogueLengthMetric`

Returns the number of accepted recommendations divided by the total number of utterances in the dialogue.

When any user utility metric is requested, the following configuration fields are required:

  * `recommendation_intent_labels`
  * `accept_intent_labels`
  * `reject_intent_labels`

When `annotate_dialogues` is enabled, the following configuration fields are also required:

  * `user_nlu`
  * `agent_nlu`


Output
------

The evaluation script writes two files:

  * `results.json` in the directory specified by `output_dir`.
  * `config_evaluation.meta.yaml` in the same directory, containing a copy of the configuration used.


The result JSON contains:

  * `dialogues_path`: Path to the evaluated dialogues.
  * `metrics_requested`: List of requested metrics.
  * `metrics`: Metric results.


For `satisfaction` and all user utility metrics, each metric entry contains:

  * `per_dialogue`: Mapping from conversation ID to score.
  * `summary_by_agent`: Aggregate statistics per agent (`count`, `min`, `max`, `mean`, `stdev`).


For `quality`, the output is grouped by aspect. Each aspect contains its own `per_dialogue` scores and `summary_by_agent` statistics.

Example output structure:

.. code-block:: json

    {
      "dialogues_path": "data/datasets/moviebot/annotated_dialogues.json",
      "metrics_requested": ["satisfaction", "success_rate", "quality"],
      "metrics": {
        "satisfaction": {
          "per_dialogue": {
            "conv_001": 0.82
          },
          "summary_by_agent": {
            "moviebot": {
              "count": 1,
              "min": 0.82,
              "max": 0.82,
              "mean": 0.82,
              "stdev": 0.0
            }
          }
        },
        "success_rate": {
          "per_dialogue": {
            "conv_001": 1.0
          },
          "summary_by_agent": {
            "moviebot": {
              "count": 1,
              "min": 1.0,
              "max": 1.0,
              "mean": 1.0,
              "stdev": 0.0
            }
          }
        },
        "quality": {
          "REC_RELEVANCE": {
            "per_dialogue": {
              "conv_001": 4.5
            },
            "summary_by_agent": {
              "moviebot": {
                "count": 1,
                "min": 4.5,
                "max": 4.5,
                "mean": 4.5,
                "stdev": 0.0
              }
            }
          }
        }
      }
    }
