"""Trainer for SLIM model."""

import json
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.utils
import torch.utils.data
from seqeval.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import MultilabelAccuracy

from usersimcrs.nlu.slim.dataset import Participant
from usersimcrs.nlu.slim.slim import SLIM

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class SLIMTrainer:
    def __init__(
        self,
        model: SLIM,
        participant: Participant,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        device: torch.device,
    ) -> None:
        """Initializes the trainer.

        Args:
            model: SLIM model.
            participant: Participant the model is trained on.
            optimizer: Optimizer.
            scheduler: Scheduler.
            device: Device to run the model on.
        """
        self.participant = participant
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_one_epoch(
        self, data_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float, float, float]:
        """Trains the model for one epoch.

        Args:
            data_loader: DataLoader for training data.

        Returns:
            Average training losses for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_intent_loss = 0.0
        total_slot_loss = 0.0
        total_tag_intent_loss = 0.0
        num_batches = len(data_loader)
        for batch in data_loader:
            inputs = {
                "input_ids": batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
                "token_type_ids": batch["token_type_ids"].to(self.device),
                "intent_label_ids": batch["intent_label_ids"].to(self.device),
                "slot_label_ids": batch["slot_label_ids"].to(self.device),
                "tag_intent_mask": batch["tag_intent_mask"].to(self.device),
                "tag_intent_label_ids": batch["tag_intent_label_ids"].to(
                    self.device
                ),
            }
            self.optimizer.zero_grad()
            outputs = self.model(**inputs)

            losses = outputs[0]
            loss = losses[0]  # Total loss
            intent_loss = losses[1]
            slot_loss = losses[2]
            tag_intent_loss = losses[3]
            loss.backward()
            total_loss += loss.item()
            total_intent_loss += intent_loss.item()
            total_slot_loss += slot_loss.item()
            total_tag_intent_loss += tag_intent_loss.item()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

        return (
            total_loss / num_batches,
            total_intent_loss / num_batches,
            total_slot_loss / num_batches,
            total_tag_intent_loss / num_batches,
        )

    def get_validation_loss(
        self, data_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float, float, float]:
        """Computes validation loss.

        Args:
            data_loader: DataLoader for validation data.

        Returns:
            Average validation losses for the epoch.
        """
        self.model.eval()
        total_loss = 0.0
        total_intent_loss = 0.0
        total_slot_loss = 0.0
        total_tag_intent_loss = 0.0
        num_batches = len(data_loader)
        with torch.no_grad():
            for batch in data_loader:
                inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                    "token_type_ids": batch["token_type_ids"].to(self.device),
                    "intent_label_ids": batch["intent_label_ids"].to(
                        self.device
                    ),
                    "slot_label_ids": batch["slot_label_ids"].to(self.device),
                    "tag_intent_mask": batch["tag_intent_mask"].to(
                        self.device
                    ),
                    "tag_intent_label_ids": batch["tag_intent_label_ids"].to(
                        self.device
                    ),
                }
                outputs = self.model(**inputs)

                losses = outputs[0]
                loss = losses[0]
                intent_loss = losses[1]
                slot_loss = losses[2]
                tag_intent_loss = losses[3]

                total_loss += loss.item()
                total_intent_loss += intent_loss.item()
                total_slot_loss += slot_loss.item()
                total_tag_intent_loss += tag_intent_loss.item()

        return (
            total_loss / num_batches,
            total_intent_loss / num_batches,
            total_slot_loss / num_batches,
            total_tag_intent_loss / num_batches,
        )

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        max_epochs: int,
        model_path: str,
        tb_writer: SummaryWriter = None,
    ) -> None:
        """Trains the model.

        Args:
            train_loader: DataLoader for training data.
            valid_loader: DataLoader for validation data.
            max_epochs: Maximum number of epochs.
            model_path: Path to save the model.
            tb_writer: Tensorboard writer. Defaults to None.
        """
        best_loss = float("inf")
        for epoch in range(max_epochs):
            (
                train_loss,
                train_intent_loss,
                train_slot_loss,
                train_tag_intent_loss,
            ) = self.train_one_epoch(train_loader)
            (
                valid_loss,
                valid_intent_loss,
                valid_slot_loss,
                valid_tag_intent_loss,
            ) = self.get_validation_loss(valid_loader)

            if tb_writer is not None:
                tb_writer.add_scalar("Loss/Train", train_loss, epoch)
                tb_writer.add_scalar("Loss/Valid", valid_loss, epoch)
                tb_writer.add_scalar(
                    "Intent loss/Train", train_intent_loss, epoch
                )
                tb_writer.add_scalar(
                    "Intent loss/Valid", valid_intent_loss, epoch
                )
                tb_writer.add_scalar("Slot loss/Train", train_slot_loss, epoch)
                tb_writer.add_scalar("Slot loss/Valid", valid_slot_loss, epoch)
                tb_writer.add_scalar(
                    "Tag Intent loss/Train", train_tag_intent_loss, epoch
                )
                tb_writer.add_scalar(
                    "Tag Intent loss/Valid", valid_tag_intent_loss, epoch
                )

            logger.info(
                f"Epoch {epoch + 1}/{max_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Valid Loss: {valid_loss:.4f}"
            )

            if valid_loss < best_loss:
                best_loss = valid_loss
                self.save_pretrained(model_path)

    def evaluate(
        self, data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        """Evaluates the model.

        Args:
            data_loader: DataLoader for evaluation data.

        Returns:
            Evaluation metrics.
        """
        self.model.eval()
        self.model.zero_grad()

        intent_predictions = None
        intent_labels = None

        slot_predictions = None
        slot_labels = None

        tag_intent_predictions = None
        tag_intent_labels = None

        with torch.no_grad():
            for batch in data_loader:
                inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                    "token_type_ids": batch["token_type_ids"].to(self.device),
                    "intent_label_ids": batch["intent_label_ids"].to(
                        self.device
                    ),
                    "slot_label_ids": batch["slot_label_ids"].to(self.device),
                    "tag_intent_mask": batch["tag_intent_mask"].to(
                        self.device
                    ),
                    "tag_intent_label_ids": batch["tag_intent_label_ids"].to(
                        self.device
                    ),
                }
                outputs = self.model(**inputs)
                logits = outputs[1]

                intent_labels = (
                    np.append(
                        intent_labels,
                        batch["intent_label_ids"].detach().cpu().numpy(),
                        axis=0,
                    )
                    if intent_labels is not None
                    else batch["intent_label_ids"].detach().cpu().numpy()
                )
                intent_logits = logits[0]
                intent_preds = torch.as_tensor(
                    intent_logits > 0.5, dtype=torch.int32
                )
                intent_predictions = (
                    np.append(
                        intent_predictions,
                        intent_preds.detach().cpu().numpy(),
                        axis=0,
                    )
                    if intent_predictions is not None
                    else intent_preds.detach().cpu().numpy()
                )

                slot_labels = (
                    np.append(
                        slot_labels,
                        batch["slot_label_ids"].detach().cpu().numpy(),
                        axis=0,
                    )
                    if slot_labels is not None
                    else batch["slot_label_ids"].detach().cpu().numpy()
                )
                slot_logits = logits[1]
                slot_preds = (
                    np.array(self.model.crf.decode(slot_logits))
                    if self.model.use_crf
                    else torch.argmax(slot_logits, dim=2)
                    .detach()
                    .cpu()
                    .numpy()
                )
                slot_predictions = (
                    np.append(
                        slot_predictions,
                        slot_preds,
                        axis=0,
                    )
                    if slot_predictions is not None
                    else slot_preds
                )

                _tag_intent_labels = batch["tag_intent_label_ids"]
                tag_intent_labels = (
                    np.append(
                        tag_intent_labels,
                        _tag_intent_labels.detach().cpu().numpy(),
                        axis=0,
                    )
                    if tag_intent_labels is not None
                    else _tag_intent_labels.detach().cpu().numpy()
                )
                tag_intent_logits = logits[2]
                tag_intent_preds = tag_intent_logits.view(
                    _tag_intent_labels.size(0), _tag_intent_labels.size(1), -1
                )
                tag_intent_predictions = (
                    np.append(
                        tag_intent_predictions,
                        torch.argmax(tag_intent_preds, dim=2)
                        .detach()
                        .cpu()
                        .numpy(),
                        axis=0,
                    )
                    if tag_intent_predictions is not None
                    else torch.argmax(tag_intent_preds, dim=2)
                    .detach()
                    .cpu()
                    .numpy()
                )

        # Intent classification accuracy
        multilabel_accuracy = MultilabelAccuracy()
        intent_accuracy = (
            multilabel_accuracy.update(
                torch.tensor(intent_predictions), torch.tensor(intent_labels)
            )
            .compute()
            .item()
        )

        # Slot filling classification report
        slot_classification_report = (
            self._get_slot_filling_classification_report(
                slot_predictions, slot_labels
            )
        )

        # Tag-intent classification report
        tag_intent_classification_report = (
            self._get_tag_intent_classification_report(
                tag_intent_predictions, tag_intent_labels
            )
        )

        return {
            "Intent Accuracy": intent_accuracy,
            "Slot Class. Report": slot_classification_report,
            "Tag-Intent Class. Report": tag_intent_classification_report,
        }

    def _get_slot_filling_classification_report(
        self, slot_preds: np.ndarray, slot_labels: np.ndarray
    ) -> Dict[str, float]:
        """Computes classification report for slot filling.

        Args:
            slot_preds: Predictions for slot filling.
            slot_labels: Ground truth labels for slot filling.

        Returns:
            Classification report.
        """
        slot_label_predictions: List[List[str]] = [
            [] for _ in range(slot_labels.shape[0])
        ]
        slot_label_labels: List[List[str]] = [
            [] for _ in range(slot_labels.shape[0])
        ]
        for i in range(slot_labels.shape[0]):
            for j in range(slot_labels.shape[1]):
                if slot_labels[i, j] == self.model.slot_labels_map["PAD"]:
                    continue
                slot_label_labels[i].append(
                    self.model.reverse_slot_labels_map[slot_labels[i, j]]
                )
                slot_label_predictions[i].append(
                    self.model.reverse_slot_labels_map[slot_preds[i, j]]
                )

        return classification_report(
            slot_label_labels, slot_label_predictions, output_dict=True
        )

    def _get_tag_intent_classification_report(
        self, tag_intent_preds: np.ndarray, tag_intent_labels: np.ndarray
    ) -> Dict[str, float]:
        """Computes classification report for tag-intent classification.

        Args:
            tag_intent_preds: Predictions for tag-intent classification.
            tag_intent_labels: Ground truth labels for tag-intent
              classification.

        Returns:
            Classification report.
        """
        slot_intent_predictions: List[List[str]] = [
            [] for _ in range(tag_intent_labels.shape[0])
        ]
        slot_intent_labels: List[List[str]] = [
            [] for _ in range(tag_intent_labels.shape[0])
        ]
        for i in range(tag_intent_labels.shape[0]):
            for j in range(tag_intent_labels.shape[1]):
                if (
                    tag_intent_labels[i, j]
                    == self.model.intent_labels_map["PAD"]
                ):
                    continue
                intent_label = self.model.reverse_intent_labels_map[
                    tag_intent_labels[i, j]
                ]
                slot_intent_labels[i].append(f"B-{intent_label}")
                intent_label_predicted = self.model.reverse_intent_labels_map[
                    tag_intent_preds[i, j]
                ]
                slot_intent_predictions[i].append(
                    f"B-{intent_label_predicted}"
                )

        return classification_report(
            slot_intent_labels, slot_intent_predictions, output_dict=True
        )

    def save_pretrained(self, model_dir: str) -> None:
        """Saves the model to disk.

        Args:
            model_dir: Path to save the model.
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)

        metadata = {
            "intent_labels_map": self.model.intent_labels_map,
            "slot_labels_map": self.model.slot_labels_map,
            "num_mask": self.model.num_mask,
            "dropout": self.model.dropout_rate,
            "use_crf": self.model.use_crf,
            "participant": self.participant.value,
        }
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
