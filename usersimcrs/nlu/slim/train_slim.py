"""Training script for SLIM model."""

import argparse
import logging
import os
from datetime import datetime
from typing import Dict, Tuple

import confuse
import torch
import torch.utils
import torch.utils.data
import yaml
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from usersimcrs.nlu.slim.dataset import Participant, SLIMDataset
from usersimcrs.nlu.slim.slim import SLIM
from usersimcrs.nlu.slim.slim_trainer import SLIMTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

DEFAULT_CONFIG_PATH = "config/slim/config_training_default.yaml"


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(prog="train_slim.py")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to config file to overwrite default values. "
        "Defaults to None.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the data.",
    )
    parser.add_argument(
        "--participant",
        type=Participant,
        choices=list(Participant),
        help="Participant to train the model for. Defaults to ALL",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to save the model. Defaults to data/models/slim.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Whether to log training metrics. Defaults to True.",
    )
    # Model parameters
    model_group = parser.add_argument_group("Model", "Model parameters")
    model_group.add_argument(
        "--bert_model",
        type=str,
        help="BERT model to use. Defaults to bert-base-uncased.",
    )
    model_group.add_argument(
        "--dropout",
        type=float,
        help="Dropout rate for the model. Defaults to 0.1.",
    )
    model_group.add_argument(
        "--use_crf",
        action="store_true",
        help="Whether to use CRF for slot filling. Defaults to True.",
    )
    model_group.add_argument(
        "--max_slot",
        type=int,
        help="Maximum number of slots in an utterance. Defaults to 6.",
    )
    model_group.add_argument(
        "--intent_labels",
        type=str,
        help="Path to intent labels. Defaults to "
        "data/datasets/iard/all_actions.yaml",
    )
    model_group.add_argument(
        "--slot_labels",
        type=str,
        help="Path to slot labels. Defaults to "
        "data/datasets/iard/slot_labels.yaml",
    )
    # Training parameters
    training_group = parser.add_argument_group(
        "Training", "Training parameters"
    )
    training_group.add_argument(
        "--max_epochs",
        type=int,
        help="Maximum number of epochs for training. Defaults to 10.",
    )
    training_group.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate for training. Defaults to 5e-5.",
    )
    training_group.add_argument(
        "--weight_decay",
        type=float,
        help="Weight decay for training. Defaults to 0.",
    )
    training_group.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training. Defaults to 32.",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> confuse.Configuration:
    """Loads configuration from YAML file and command-line arguments.

    Args:
        args: Command-line arguments.

    Returns:
        Configuration object.
    """
    config = confuse.Configuration("SLIM", loader=yaml.Loader)
    config.set_file(DEFAULT_CONFIG_PATH)

    # Load specified config file, updates default values
    if args.config:
        config.set_file(args.config)

    # Update config with command-line arguments
    config.set_args(args, dots=True)

    return config


def prepare_data_splits(
    config: confuse.Configuration,
    participant: Participant,
    intent_labels_map: Dict[str, int],
    slot_labels_map: Dict[str, int],
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """Prepares data splits for training, validation, and testing.

    Args:
        config: Configuration object.
        participant: Participant to train the model for.
        intent_labels_map: Mapping of intent labels to indices.
        slot_labels_map: Mapping of slot labels to indices.

    Returns:
        Data loaders for training, validation, and testing.
    """
    data_path = config["data_path"].get(str)
    logger.info(f"Loading data from {data_path}")
    dataset = SLIMDataset(
        data_path,
        intent_labels_map,
        slot_labels_map,
        participant,
        max_slot=config["max_slot"].get(int),
    )

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [0.98, 0.01, 0.01],
        generator=torch.Generator().manual_seed(config["seed"].get(int)),
    )

    batch_size = config["batch_size"].get(int)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tensorboard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = None
    if config["logs"].get(bool):
        writer = SummaryWriter(f"logs/slim_training_{timestamp}")

    participant = Participant(config["participant"].get())

    # Load data
    intent_labels = yaml.safe_load(open(config["intent_labels"].get(str)))
    intent_labels_map: Dict[str, int] = {
        label: i for i, label in enumerate(intent_labels)
    }

    slot_labels = yaml.safe_load(open(config["slot_labels"].get()))
    slot_labels_map: Dict[str, int] = {
        label: i for i, label in enumerate(slot_labels)
    }

    train_loader, valid_loader, test_loader = prepare_data_splits(
        config,
        participant,
        intent_labels_map,
        slot_labels_map,
    )

    # Initialize model
    slim = SLIM(
        intent_labels_map,
        slot_labels_map,
        config["dropout"].get(float),
        config["use_crf"].get(bool),
        config["max_slot"].get(int),
    )
    slim.to(device)

    # Initialize optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in slim.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config["weight_decay"].get(float),
        },
        {
            "params": [
                p
                for n, p in slim.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=float(config["learning_rate"].get()),
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * config["max_epochs"].get(int),
    )

    # Initialize trainer
    trainer = SLIMTrainer(slim, participant, optimizer, scheduler, device)

    # Train model
    model_output_path = os.path.join(
        config["output_dir"].get(), f"slim_model_{participant}_{timestamp}"
    )
    trainer.train(
        train_loader,
        valid_loader,
        config["max_epochs"].get(int),
        model_output_path,
        writer,
    )
    logger.info("Training complete")
    logger.info(f"Best model saved at {model_output_path}")

    # Evaluate model
    logger.info("====\nEvaluating the model")
    metrics = trainer.evaluate(test_loader)
    logger.info(f"Metrics:\n{metrics}")
    logger.info("====")

    if writer is not None:
        writer.close()

    config["model_path"] = model_output_path
    with open(os.path.join(model_output_path, "config.yaml"), "w") as f:
        f.write(config.dump())
