"""Console application for running simulation."""

import argparse
import json
import logging
import os
import random

import confuse
from tqdm import tqdm

import usersimcrs.utils.simulation_utils as utils
from usersimcrs.simulation_platform import SimulationPlatform

DEFAULT_CONFIG_PATH = "config/default/config_default.yaml"
OUTPUT_DIR = "data/runs"

logging.basicConfig(
    format="[%(asctime)s] %(levelname)-12s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger()


def main(config: confuse.Configuration) -> None:
    """Executes the specified configuration.

    Loads domain and interaction model. Initializes agent and user. Runs the
    simulation.

    Args:
        config: Configuration generated from YAML configuration file.
        agent: Conversational agent.
    """
    agent_class, agent_config = utils.get_agent_information(config)
    (
        simulator_id,
        simulator_class,
        simulator_config,
    ) = utils.get_simulator_information(config)

    platform = SimulationPlatform(agent_class, agent_config)
    platform.start()

    for _ in tqdm(
        range(config["num_simulated_dialogues"].get()),
        desc=" Simulated dialogues",
    ):
        platform.connect(simulator_id, simulator_class, simulator_config)
        platform.disconnect(simulator_id)


def parse_args() -> argparse.Namespace:
    """Defines accepted arguments and returns the parsed values.

    Returns:
        A namespace object containing the arguments.
    """

    parser = argparse.ArgumentParser(prog="run_simulation.py")
    parser.add_argument(
        "-c",
        "--config-file",
        help=(
            "Path to configuration file to overwrite default values. "
            "Defaults to None."
        ),
    )
    parser.add_argument(
        "-a",
        "--agent_id",
        type=str,
        help=("Id of the agent tested. Defaults to 'IAI MovieBot'."),
    )
    parser.add_argument(
        "--agent_uri",
        type=str,
        help="URI to communicate with the agent. By default we assume that the"
        " agent has an HTTP API.",
    )
    parser.add_argument(
        "-o",
        "--output_name",
        type=str,
        help="Specifies the output name for the simulation configuration.",
    )
    parser.add_argument(
        "--domain", type=str, help="Path to domain config file."
    )
    parser.add_argument(
        "--intents", type=str, help="Path to the intent schema file."
    )
    parser.add_argument("--items", type=str, help="Path to items file.")
    parser.add_argument(
        "--id_col", type=str, help="Name of the CSV field containing item id."
    )
    parser.add_argument(
        "--domain_mapping",
        type=json.loads,
        help="String form of field mapping.",
    )
    parser.add_argument("--ratings", type=str, help="Path to ratings file.")
    parser.add_argument(
        "--historical_ratings_ratio",
        type=float,
        help="Ratio of ratings to be used as historical data.",
    )
    parser.add_argument(
        "--dialogues", type=str, help="Path to the annotated dialogues file."
    )
    parser.add_argument(
        "--intent_classifier",
        choices=["cosine", "diet"],
        help="Intent classifier model to be used. Defaults to cosine.",
    )
    parser.add_argument(
        "--rasa_dialogues",
        type=str,
        help="Path to the Rasa annotated dialogues file.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_const",
        const=True,
        help=("Debug mode. Defaults to False."),
    )
    parser.add_argument(
        "--fix_random_seed",
        action="store_const",
        const=True,
        help=("Fix random seed. Defaults to False."),
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> confuse.Configuration:
    """Loads config from config file and command line parameters.

    Loads default values from `config/default/config_default.yaml`. Values are
    then updated with any value specified in the command line arguments.

    Args:
        args: Arguments parsed with argparse.
    """
    # Load default config
    config = confuse.Configuration("usersimcrs")
    config.set_file(DEFAULT_CONFIG_PATH)

    # Load additional config (update defaults).
    if args.config_file:
        config.set_file(args.config_file)

    # Update config from command line arguments
    config.set_args(args, dots=True)

    # Save run config to metadata file
    output_name = config["output_name"].get()
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    output_config_file = os.path.join(OUTPUT_DIR, f"{output_name}.meta.yaml")
    with open(output_config_file, "w") as f:
        f.write(config.dump())

    return config


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)

    if config["debug"].get():
        logger.setLevel(logging.DEBUG)

    if config["fix_random_seed"].get():
        random.seed(42)

    main(config)
