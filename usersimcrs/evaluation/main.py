"""Unified script for evaluating dialogues with selected metrics."""

import argparse
import json
import os
from collections import defaultdict
from statistics import mean, stdev
from typing import Dict, List, Mapping, Sequence

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.nlu.models.satisfaction_classifier import (
    SatisfactionClassifierSVM,
)
from dialoguekit.utils.dialogue_reader import json_to_dialogues

from usersimcrs.evaluation.base_metric import BaseMetric
from usersimcrs.evaluation.quality_metric import QualityMetric
from usersimcrs.evaluation.quality_rubrics import QualityRubrics
from usersimcrs.evaluation.satisfaction_metric import SatisfactionMetric
from usersimcrs.evaluation.utility_metric import (
    RewardPerDialogueLengthMetric,
    SuccessRateMetric,
    SuccessfulRecommendationRoundRatioMetric,
)
from usersimcrs.llm_interfaces.ollama_interface import OllamaLLMInterface

SUPPORTED_METRICS = [
    "quality",
    "satisfaction",
    "success_rate",
    "successful_recommendation_round_ratio",
    "reward_per_dialogue_length",
]


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(prog="usersimcrs.evaluation.main")
    parser.add_argument(
        "--dialogues",
        type=str,
        required=True,
        help="Path to the dialogues JSON file.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        choices=SUPPORTED_METRICS,
        help="List of metrics to compute.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save evaluation results as JSON.",
    )
    parser.add_argument(
        "--ollama_config",
        type=str,
        help="Path to Ollama config file (required when quality is selected).",
    )
    parser.add_argument(
        "--quality_aspects",
        nargs="+",
        default=[aspect.name for aspect in QualityRubrics],
        help=(
            "Quality aspects to evaluate. "
            "Defaults to all aspects in QualityRubrics."
        ),
    )
    parser.add_argument(
        "--user_nlu_config",
        type=str,
        help=(
            "Path to user NLU config (required for utility metrics: "
            "success_rate, successful_recommendation_round_ratio, "
            "reward_per_dialogue_length)."
        ),
    )
    parser.add_argument(
        "--agent_nlu_config",
        type=str,
        help=(
            "Path to agent NLU config (required for utility metrics: "
            "success_rate, successful_recommendation_round_ratio, "
            "reward_per_dialogue_length)."
        ),
    )
    parser.add_argument(
        "--reject_intent_labels",
        nargs="+",
        default=["REJ"],
        help="Intent labels corresponding to rejection.",
    )
    parser.add_argument(
        "--accept_intent_labels",
        nargs="+",
        default=["ACC"],
        help="Intent labels corresponding to acceptance.",
    )
    parser.add_argument(
        "--recommendation_intent_labels",
        nargs="+",
        default=["REC-S", "REC-E"],
        help="Intent labels corresponding to recommendation.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Validates metric-specific CLI requirements."""
    if "quality" in args.metrics and not args.ollama_config:
        raise ValueError(
            "The --ollama_config argument is required when using quality."
        )

    utility_metrics = {
        "success_rate",
        "successful_recommendation_round_ratio",
        "reward_per_dialogue_length",
    }
    if utility_metrics.intersection(set(args.metrics)):
        if not args.user_nlu_config or not args.agent_nlu_config:
            raise ValueError(
                "Both --user_nlu_config and --agent_nlu_config are required "
                "for utility metrics."
            )

    invalid_aspects = [
        aspect
        for aspect in args.quality_aspects
        if aspect not in [enum_aspect.name for enum_aspect in QualityRubrics]
    ]
    if invalid_aspects:
        raise ValueError(
            f"Unknown quality aspect(s): {invalid_aspects}. "
            f"Supported aspects: {[aspect.name for aspect in QualityRubrics]}"
        )


def _build_metric_registry(args: argparse.Namespace) -> Dict[str, BaseMetric]:
    """Builds metric instances keyed by metric name."""
    registry: Dict[str, BaseMetric] = {}
    if "quality" in args.metrics:
        llm_interface = OllamaLLMInterface(
            configuration_path=args.ollama_config,
            default_response="",
        )
        registry["quality"] = QualityMetric(llm_interface=llm_interface)
    if "satisfaction" in args.metrics:
        registry["satisfaction"] = SatisfactionMetric(
            classifier=SatisfactionClassifierSVM()
        )
    if "success_rate" in args.metrics:
        registry["success_rate"] = SuccessRateMetric(
            user_nlu_config_path=args.user_nlu_config,
            agent_nlu_config_path=args.agent_nlu_config,
        )
    if "successful_recommendation_round_ratio" in args.metrics:
        registry[
            "successful_recommendation_round_ratio"
        ] = SuccessfulRecommendationRoundRatioMetric(
            user_nlu_config_path=args.user_nlu_config,
            agent_nlu_config_path=args.agent_nlu_config,
        )
    if "reward_per_dialogue_length" in args.metrics:
        registry["reward_per_dialogue_length"] = RewardPerDialogueLengthMetric(
            user_nlu_config_path=args.user_nlu_config,
            agent_nlu_config_path=args.agent_nlu_config,
        )
    return registry


def _summarize_by_agent(
    dialogues: Sequence[Dialogue], scores: Mapping[str, float]
) -> Dict[str, Dict[str, float]]:
    """Returns aggregate statistics by agent."""
    conversation_to_agent = {
        dialogue.conversation_id: dialogue.agent_id for dialogue in dialogues
    }
    grouped_scores: Dict[str, List[float]] = defaultdict(list)
    for conversation_id, score in scores.items():
        agent_id = conversation_to_agent.get(conversation_id, "unknown")
        grouped_scores[agent_id].append(score)

    summary: Dict[str, Dict[str, float]] = {}
    for agent_id, agent_scores in grouped_scores.items():
        summary[agent_id] = {
            "count": float(len(agent_scores)),
            "min": min(agent_scores),
            "max": max(agent_scores),
            "mean": mean(agent_scores),
            "stdev": stdev(agent_scores) if len(agent_scores) > 1 else 0.0,
        }
    return summary


def _evaluate_metric(
    metric_name: str,
    metric: BaseMetric,
    dialogues: Sequence[Dialogue],
    args: argparse.Namespace,
) -> Dict[str, object]:
    """Runs one metric and returns per-dialogue scores and summary."""
    if metric_name == "quality":
        per_aspect: Dict[str, Dict[str, Dict[str, float]]] = {}
        for aspect in args.quality_aspects:
            per_dialogue = metric.evaluate_dialogues(
                list(dialogues),
                aspect=aspect,
            )
            per_aspect[aspect] = {
                "per_dialogue": per_dialogue,
                "summary_by_agent": _summarize_by_agent(
                    dialogues, per_dialogue
                ),
            }
        return {"aspects": per_aspect}

    eval_kwargs = {}
    if metric_name in {
        "success_rate",
        "successful_recommendation_round_ratio",
        "reward_per_dialogue_length",
    }:
        eval_kwargs = {
            "recommendation_intent_labels": args.recommendation_intent_labels,
            "acceptance_intent_labels": args.accept_intent_labels,
            "rejection_intent_labels": args.reject_intent_labels,
        }

    per_dialogue_scores = metric.evaluate_dialogues(
        list(dialogues), **eval_kwargs
    )
    return {
        "per_dialogue": per_dialogue_scores,
        "summary_by_agent": _summarize_by_agent(dialogues, per_dialogue_scores),
    }


def _print_brief_summary(results: Mapping[str, object]) -> None:
    """Prints a concise summary in the terminal."""
    metric_results = results.get("metrics", {})
    if not isinstance(metric_results, dict):
        return
    for metric_name, metric_result in metric_results.items():
        print(f"Metric: {metric_name}")
        if metric_name == "quality":
            aspects = metric_result.get("aspects", {})
            for aspect_name, aspect_result in aspects.items():
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
    args = parse_args()
    _validate_args(args)

    dialogues = json_to_dialogues(args.dialogues)
    metric_registry = _build_metric_registry(args)

    results: Dict[str, object] = {
        "dialogues_path": args.dialogues,
        "metrics_requested": args.metrics,
        "metrics": {},
    }

    for metric_name in args.metrics:
        metric = metric_registry[metric_name]
        results["metrics"][metric_name] = _evaluate_metric(
            metric_name,
            metric,
            dialogues,
            args,
        )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    _print_brief_summary(results)


if __name__ == "__main__":
    main()
