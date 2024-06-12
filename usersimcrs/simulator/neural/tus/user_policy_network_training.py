"""Script to train the user policy network of the transformer-based US."""

import argparse
import logging
import os
from datetime import datetime

import pandas as pd
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import MulticlassConfusionMatrix

from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.simulator.neural.core.transformer import (
    TransformerEncoderModel,
)
from usersimcrs.simulator.neural.tus.tus_dataset import TUSDataset
from usersimcrs.simulator.neural.tus.tus_feature_handler import (
    TUSFeatureHandler,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class Trainer:
    def __init__(
        self,
        model: TransformerEncoderModel,
        loss_function: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        max_turn_feature_length: int,
    ) -> None:
        """Initializes the trainer.

        Args:
            model: User policy network.
            loss_function: Loss function.
            optimizer: Optimizer.
            max_turn_feature_length: Maximum length of the input features.
        """
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.max_turn_feature_length = max_turn_feature_length

    def parse_output(self, output: torch.Tensor) -> torch.Tensor:
        """Parses the model output.

        Args:
            output: Model output.

        Returns:
            Parsed output.
        """
        _output = output[:, 1 : self.max_turn_feature_length + 1, :]
        _output = torch.reshape(
            _output,
            (
                _output.shape[0] * _output.shape[1],
                _output.shape[-1],
            ),
        )
        return _output

    def get_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Computes the loss.

        Args:
            prediction: Predicted values.
            target: Target values.

        Returns:
            Loss value.
        """
        _prediction = self.parse_output(prediction)
        return self.loss_function(_prediction, target.view(-1))

    def train_one_epoch(
        self, data_loader: torch.utils.data.DataLoader
    ) -> float:
        """Trains the model for one epoch.

        Args:
            data_loader: Training data.

        Returns:
            Average loss.
        """
        self.model.train()
        total_loss = 0.0
        for batch in data_loader:
            self.optimizer.zero_grad()
            input_feature = batch["input"]
            mask = batch["mask"]
            label = batch["label"]
            output = self.model(input_feature, mask)

            loss = self.get_loss(output, label)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(data_loader)

    def train(
        self,
        training_data: torch.utils.data.DataLoader,
        validation_data: torch.utils.data.DataLoader,
        num_epochs: int,
        model_path: str,
        tb_writer: SummaryWriter = None,
    ) -> None:
        """Trains the user policy network for a given number of epochs.

        Args:
            training_data: Training data.
            validation_data: Validation data.
            num_epochs: Number of epochs.
            model_path: Path to save the model.
            tb_writer: Tensorboard writer. Defaults to None.
        """
        best_loss = float("inf")
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch(training_data)
            # Compute validation loss
            valid_loss = self.get_validation_loss(validation_data)
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} -- "
                f"Train Loss: {train_loss:.4f} / "
                f"Validation Loss: {valid_loss:.4f}"
            )

            if tb_writer is not None:
                # Log losses to tensorboard
                tb_writer.add_scalar("Loss/Train", train_loss, epoch)
                tb_writer.add_scalar("Loss/Validation", valid_loss, epoch)
                tb_writer.flush()

            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(self.model.state_dict(), model_path)

    def get_validation_loss(
        self, validation_data: torch.utils.data.DataLoader
    ) -> float:
        """Computes the validation loss.

        Args:
            validation_data: Validation data.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in validation_data:
                input_feature = batch["input"]
                mask = batch["mask"]
                label = batch["label"]
                output = self.model(input_feature, mask)

                loss = self.get_loss(output, label)
                total_loss += loss.item()

        return total_loss / len(validation_data)

    def evaluate(
        self,
        test_data: torch.utils.data.DataLoader,
        tb_writer: SummaryWriter = None,
    ) -> pd.DataFrame:
        """Evaluates the user policy network.

        Args:
            test_data: Test data.
            tb_writer: Tensorboard writer. Defaults to None.

        Returns:
            Dataframe with accuracy, precision, recall, and F1 score for each
              class (i.e., possible values for a slot).
        """
        confusion_matrix = MulticlassConfusionMatrix(num_classes=6)
        self.model.eval()
        self.model.zero_grad()
        with torch.no_grad():
            for batch in test_data:
                input_feature = batch["input"]
                mask = batch["mask"]
                label = batch["label"]
                output = self.model(input_feature, mask)

                loss = self.get_loss(output, label)

                predictions = self.parse_output(output)
                targets = label.view(-1)
                confusion_matrix.update(predictions, targets)

                if tb_writer is not None:
                    tb_writer.add_scalar("Loss/Test", loss.item())
                    tb_writer.flush()

        return self.compute_metrics(confusion_matrix)

    def compute_metrics(
        self, confusion_matrix: MulticlassConfusionMatrix
    ) -> pd.DataFrame:
        """Computes metrics based on the confusion matrix.

        Args:
            confusion_matrix: Confusion matrix.

        Returns:
            Metrics.
        """
        matrix = confusion_matrix.compute()
        num_classes = matrix.shape[0]
        metrics = dict()

        for i in range(num_classes):
            tp = matrix[i, i].item()
            fp = sum(matrix[:, i]).item() - tp
            fn = sum(matrix[i, :]).item() - tp
            tn = sum(sum(matrix)).item() - tp - fp - fn

            accuracy = (
                (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0
            )
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0
            )
            metrics[i] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
            }

        metrics["macro avg"] = {
            "precision": sum([v["precision"] for v in metrics.values()])
            / num_classes,
            "recall": sum([v["recall"] for v in metrics.values()])
            / num_classes,
            "f1": sum([v["f1"] for v in metrics.values()]) / num_classes,
            "accuracy": sum([v["accuracy"] for v in metrics.values()])
            / num_classes,
        }

        return pd.DataFrame(metrics).T


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Namespace object containing the arguments.
    """
    parser = argparse.ArgumentParser(prog="tus_training.py")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the data file.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Path to the domain configuration file.",
    )
    parser.add_argument(
        "--agent_actions_path",
        type=str,
        help="Path to the agent actions file.",
    )
    parser.add_argument(
        "--max_turn_feature_length",
        type=int,
        default=75,
        help="Maximum length of the input features.",
    )
    parser.add_argument(
        "--context_depth",
        type=int,
        default=2,
        help="Number of turns used to build input representations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/models",
        help="Output directory.",
    )
    parser.add_argument(
        "--logs", action="store_true", help="Enable Tensorboard logs."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Tensorboard writer
    writer = None
    if args.logs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(f"logs/tus_training_{timestamp}")

    # Initialize feature handler
    domain = SimulationDomain(args.domain)
    with open(args.agent_actions_path, "r") as f:
        agent_actions = yaml.safe_load(f)
    feature_handler = TUSFeatureHandler(
        domain, args.max_turn_feature_length, args.context_depth, agent_actions
    )
    feature_handler_path = os.path.join(
        args.output_dir, "feature_handler.joblib"
    )
    feature_handler.save_handler(feature_handler_path)

    # Load data
    logger.info(f"Loading data from {args.data_path}")
    dataset = TUSDataset(args.data_path, feature_handler)
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [0.8, 0.1, 0.1],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True
    )

    # Define user policy network
    # TODO: Make config an argument
    config = {
        "input_dim": 37,
        "output_dim": 6,
        "num_encoder_layers": 2,
        "nhead": 4,
        "hidden_dim": 200,
    }
    user_policy_network = TransformerEncoderModel(**config)
    user_policy_network.to(device)

    # Define loss function
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # Define optimizer
    optimizer = torch.optim.Adam(
        user_policy_network.parameters(), lr=args.learning_rate
    )

    # Initialize trainer
    trainer = Trainer(
        user_policy_network,
        loss_function,
        optimizer,
        args.max_turn_feature_length,
    )
    model_path = os.path.join(args.output_dir, "user_policy_network.pt")
    trainer.train(
        train_data_loader,
        valid_data_loader,
        args.num_epochs,
        model_path,
        tb_writer=writer,
    )

    # Evaluate the model
    logger.info("====\nEvaluating the model")
    metrics = trainer.evaluate(test_data_loader)
    logger.info(f"Metrics:\n{metrics}")
    logger.info("====")

    if writer is not None:
        writer.close()
