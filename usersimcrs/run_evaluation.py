"""Console application for running evaluation."""

import argparse
import json
import os
from collections import defaultdict
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

import confuse
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.intent import Intent
from dialoguekit.nlu.models.satisfaction_classifier import (
    SatisfactionClassifierSVM,
)
from dialoguekit.utils.dialogue_reader import json_to_dialogues

from usersimcrs.evaluation.base_metric import BaseMetric
from usersimcrs.evaluation.dialogue_annotation import annotate_dialogues
from usersimcrs.evaluation.quality_metric import QualityMetric
from usersimcrs.evaluation.quality_rubrics import QualityRubrics
from usersimcrs.evaluation.reward_per_dialogue_length_metric import (
    RewardPerDialogueLengthMetric,
)
from usersimcrs.evaluation.satisfaction_metric import SatisfactionMetric
from usersimcrs.evaluation.success_rate_metric import SuccessRateMetric
from usersimcrs.evaluation.successful_recommendation_round_ratio_metric import (
    SuccessfulRecommendationRoundRatioMetric,
)
from usersimcrs.utils.simulation_utils import get_NLU, get_llm_interface

DEFAULT_CONFIG_PATH = "config/default/config_evaluation.yaml"
SUPPORTED_METRICS = [
    "quality",
    "satisfaction",
    "success_rate",
    "successful_recommendation_round_ratio",
    "reward_per_dialogue_length",
]


def parse_args() -> argparse.Namespace:
    """Defines accepted arguments and returns the parsed values.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(prog="run_evaluation.py")
    parser.add_argument(
        "-c",
        "--config-file",
        help=(
            "Path to configuration file to overwrite default values. "
            "Defaults to None."
        ),
    )
    parser.add_argument("--dialogues", type=str, help="Dialogues JSON file.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=SUPPORTED_METRICS,
        help="Metrics to compute.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        help="Directory to save evaluation results and metadata.",
    )
    parser.add_argument(
        "--quality_aspects",
        nargs="+",
        help="Quality aspects to evaluate.",
    )
    parser.add_argument(
        "--annotate_dialogues",
        action="store_const",
        const=True,
        help="Annotate dialogues before computing metrics.",
    )
    parser.add_argument(
        "--reject_intent_labels",
        nargs="+",
        help="Intent labels corresponding to rejection.",
    )
    parser.add_argument(
        "--accept_intent_labels",
        nargs="+",
        help="Intent labels corresponding to acceptance.",
    )
    parser.add_argument(
        "--recommendation_intent_labels",
        nargs="+",
        help="Intent labels corresponding to recommendation.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_const",
        const=True,
        help="Debug mode.",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> confuse.Configuration:
    """Loads config from default file, custom file, and CLI overrides.

    Args:
        args: Arguments parsed with argparse.

    Returns:
        Resolved evaluation configuration.
    """
    config = confuse.Configuration("usersimcrs")
    config.set_file(DEFAULT_CONFIG_PATH)
    if args.config_file:
        config.set_file(args.config_file)
    config.set_args(args, dots=True)

    validate_config(config)

    output_dir = config["output_dir"].get()
    output_stem, output_extension = os.path.splitext(output_dir)
    if output_extension:
        output_dir = output_stem
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, "config_evaluation.meta.yaml"), "w"
    ) as f:
        f.write(config.dump())

    return config


def validate_config(config: confuse.Configuration) -> None:
    """Validates evaluation config.

    Args:
        config: Configuration generated from YAML configuration file.

    Raises:
      ValueError: If quality evaluation is requested without an LLM
        interface, if an unknown quality aspect is configured, or if
        dialogue annotation is requested without user and agent NLU
        sections.
    """
    metrics = config["metrics"].get()
    if "quality" in metrics and "quality_llm_interface" not in config:
        raise ValueError("Quality evaluation requires `quality_llm_interface`.")

    quality_aspects = config["quality_aspects"].get()
    supported_aspects = [aspect.name for aspect in QualityRubrics]
    invalid_aspects = [
        aspect for aspect in quality_aspects if aspect not in supported_aspects
    ]
    if invalid_aspects:
        raise ValueError(
            f"Unknown quality aspect(s): {invalid_aspects}. "
            f"Supported aspects: {supported_aspects}"
        )

    if config["annotate_dialogues"].get():
        if not config["user_nlu"].get(None):
            raise ValueError(
                "`user_nlu` is required when `annotate_dialogues` is True."
            )
        if not config["agent_nlu"].get(None):
            raise ValueError(
                "`agent_nlu` is required when `annotate_dialogues` is True."
            )


def annotate_for_metrics(
    dialogues: List[Dialogue], config: confuse.Configuration
) -> None:
    """Annotates dialogues for metrics that require dialogue acts.

    Args:
        dialogues: Dialogues to annotate in place.
        config: Evaluation configuration.
    """
    user_nlu = get_NLU(config, nlu_config_key="user_nlu")
    agent_nlu = get_NLU(config, nlu_config_key="agent_nlu")
    annotate_dialogues(dialogues, user_nlu, agent_nlu)


def get_summary_by_agent(
    dialogues: List[Dialogue], scores: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """Aggregates metric scores by agent.

    Args:
        dialogues: Evaluated dialogues.
        scores: Per-dialogue scores keyed by conversation ID.

    Returns:
        Descriptive score statistics keyed by agent ID.
    """
    grouped_scores: Dict[str, List[float]] = defaultdict(list)
    for dialogue in dialogues:
        grouped_scores[dialogue.agent_id].append(
            scores[dialogue.conversation_id]
        )

    return {
        agent_id: {
            "count": len(agent_scores),
            "min": min(agent_scores),
            "max": max(agent_scores),
            "mean": mean(agent_scores),
            "stdev": stdev(agent_scores) if len(agent_scores) > 1 else 0.0,
        }
        for agent_id, agent_scores in grouped_scores.items()
    }


def build_metric_registry(
    config: confuse.Configuration, metrics: List[str]
) -> Dict[str, BaseMetric]:
    """Builds metric instances.

    Args:
        config: Evaluation configuration.
        metrics: Names of metrics to evaluate.

    Returns:
        Metric instances keyed by metric name.
    """
    registry: Dict[str, BaseMetric] = {}
    if "quality" in metrics:
        registry["quality"] = QualityMetric(
            llm_interface=get_llm_interface(
                config["quality_llm_interface"].get()
            )
        )
    if "satisfaction" in metrics:
        registry["satisfaction"] = SatisfactionMetric(
            classifier=SatisfactionClassifierSVM()
        )
    if "success_rate" in metrics:
        registry["success_rate"] = SuccessRateMetric()
    if "successful_recommendation_round_ratio" in metrics:
        registry[
            "successful_recommendation_round_ratio"
        ] = SuccessfulRecommendationRoundRatioMetric()
    if "reward_per_dialogue_length" in metrics:
        registry["reward_per_dialogue_length"] = RewardPerDialogueLengthMetric()
    return registry


def evaluate_metric(
    metric: BaseMetric,
    dialogues: List[Dialogue],
    quality_aspects: Optional[List[str]] = None,
    utility_intents: Optional[Dict[str, List[Intent]]] = None,
) -> Dict[str, Any]:
    """Evaluates one metric and returns serialized results.

    Args:
        metric: Metric instance.
        dialogues: Dialogues to evaluate.
        quality_aspects: Quality aspects to evaluate for quality metrics.
        utility_intents: Utility intent arguments for utility metrics.

    Returns:
        Serialized metric result.
    """
    if metric.name == "quality":
        aspect_results = {}
        for aspect in quality_aspects or []:
            scores = metric.evaluate_dialogues(dialogues, aspect=aspect)
            aspect_results[aspect] = {
                "per_dialogue": scores,
                "summary_by_agent": get_summary_by_agent(dialogues, scores),
            }
        return {"aspects": aspect_results}

    utility_intents = utility_intents or {}
    if metric.name in {
        "success_rate",
        "successful_recommendation_round_ratio",
    }:
        scores = metric.evaluate_dialogues(dialogues, **utility_intents)
    elif metric.name == "reward_per_dialogue_length":
        scores = metric.evaluate_dialogues(
            dialogues,
            acceptance_intents=utility_intents["acceptance_intents"],
        )
    else:
        scores = metric.evaluate_dialogues(dialogues)

    return {
        "per_dialogue": scores,
        "summary_by_agent": get_summary_by_agent(dialogues, scores),
    }


def print_summary(results: Dict[str, Any]) -> None:
    """Prints a concise terminal summary.

    Args:
        results: Serialized evaluation results.
    """
    for metric_name, metric_result in results["metrics"].items():
        print(f"Metric: {metric_name}")
        if metric_name == "quality":
            for aspect_name, aspect_result in metric_result["aspects"].items():
                print(f"  Aspect: {aspect_name}")
                for agent_id, stats in aspect_result[
                    "summary_by_agent"
                ].items():
                    print(
                        f"    Agent: {agent_id} | mean={stats['mean']:.3f} "
                        f"stdev={stats['stdev']:.3f}"
                    )
            continue

        for agent_id, stats in metric_result["summary_by_agent"].items():
            print(
                f"  Agent: {agent_id} | mean={stats['mean']:.3f} "
                f"stdev={stats['stdev']:.3f}"
            )


def main() -> None:
    """Runs evaluation based on the resolved configuration."""
    args = parse_args()
    config = load_config(args)

    metrics = config["metrics"].get()
    quality_aspects = config["quality_aspects"].get()
    dialogues = json_to_dialogues(config["dialogues"].get())
    if config["annotate_dialogues"].get():
        annotate_for_metrics(dialogues, config)

    utility_intents = {
        "recommendation_intents": [
            Intent(label)
            for label in config["recommendation_intent_labels"].get()
        ],
        "acceptance_intents": [
            Intent(label) for label in config["accept_intent_labels"].get()
        ],
        "rejection_intents": [
            Intent(label) for label in config["reject_intent_labels"].get()
        ],
    }
    metric_registry = build_metric_registry(config, metrics)

    results: Dict[str, Any] = {
        "dialogues_path": config["dialogues"].get(),
        "metrics_requested": metrics,
        "metrics": {},
    }

    for metric_name in metrics:
        results["metrics"][metric_name] = evaluate_metric(
            metric_registry[metric_name],
            dialogues,
            quality_aspects=quality_aspects,
            utility_intents=utility_intents,
        )

    output_dir = config["output_dir"].get()
    output_stem, output_extension = os.path.splitext(output_dir)
    if output_extension:
        output_dir = output_stem

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print_summary(results)


if __name__ == "__main__":
    main()
