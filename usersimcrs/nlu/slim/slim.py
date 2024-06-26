"""SLIM model for dialogue act annotation.

The model performs joint multi-intent detection, slot filling, and slot-intent
mapping.

Reference:
SLIM: Explicit Slot-Intent Mapping with BERT for Joint Multi-Intent Detection
and Slot Filling, Cai et al., 2021, https://arxiv.org/abs/2108.11711

Source:
https://github.com/TRUMANCFY/SLIM
"""

from __future__ import annotations

import json
import os
import re
from itertools import groupby
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.annotation import Annotation
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.intent import Intent
from dialoguekit.core.utterance import Utterance
from seqeval.metrics.sequence_labeling import get_entities
from torchcrf import CRF
from transformers import BertModel, BertTokenizer

from usersimcrs.nlu.slim.classifiers import (
    MultiIntentClassifier,
    TagIntentClassifier,
    TokenClassifier,
)
from usersimcrs.nlu.slim.dataset import SPLIT_CHARS

_BERT_BASE_MODEL = "bert-base-uncased"


class SLIM(nn.Module):
    def __init__(
        self,
        intent_labels_map: Dict[int, str],
        slot_labels_map: Dict[int, str],
        dropout: float = 0.1,
        use_crf: bool = False,
        num_mask: int = 4,
        max_length: int = 128,
    ) -> None:
        """Initializes SLIM model.

        Args:
            intent_labels_map: Mapping of intent labels to indices.
            slot_labels_map: Mapping of slot labels to indices.
            dropout: Dropout rate. Defaults to 0.1.
            use_crf: Whether to use CRF for slot filling. Defaults to False.
            num_mask: Assumed number of slots in one utterance. Defaults to 4.
            max_length: Maximum sequence length. Defaults to 128.
        """
        super(SLIM, self).__init__()

        self.max_length = max_length
        self.num_mask = num_mask
        self.dropout_rate = dropout

        self.intent_labels_map = intent_labels_map
        self.reverse_intent_labels_map: Dict[int, str] = dict(
            map(reversed, self.intent_labels_map.items())
        )
        self.slot_labels_map = slot_labels_map
        self.reverse_slot_labels_map: Dict[int, str] = dict(
            map(reversed, self.slot_labels_map.items())
        )
        self.num_intent_labels = len(intent_labels_map)
        self.num_slot_labels = len(slot_labels_map)

        self.bert = BertModel.from_pretrained(_BERT_BASE_MODEL)
        self.tokenizer = BertTokenizer.from_pretrained(_BERT_BASE_MODEL)

        self.multi_intent_classifier = MultiIntentClassifier(
            self.bert.config.hidden_size,
            self.num_intent_labels,
            dropout_rate=dropout,
        )
        self.tag_intent_classifier = TagIntentClassifier(
            2 * self.bert.config.hidden_size,
            self.num_intent_labels,
            dropout=dropout,
        )
        self.slot_classifier = TokenClassifier(
            self.bert.config.hidden_size,
            self.num_slot_labels,
            dropout_rate=dropout,
        )

        self.use_crf = use_crf
        if self.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def _get_intent_loss(
        self, intent_logits: torch.Tensor, intent_label_ids: torch.Tensor
    ) -> torch.Tensor:
        """Calculates intent loss.

        Args:
            intent_logits: Intent logits.
            intent_label_ids: Intent labels.

        Returns:
            Intent loss.
        """
        if self.num_intent_labels == 1:
            fct = nn.MSELoss()
            return fct(
                intent_logits.view(-1),
                intent_label_ids.view(-1, self.num_intent_labels),
            )

        fct = nn.BCELoss()
        return fct(
            intent_logits.view(-1, self.num_intent_labels) + 1e-10,
            intent_label_ids.view(-1, self.num_intent_labels),
        )

    def _get_slot_loss(
        self,
        slot_logits: torch.Tensor,
        slot_label_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        ignore_index: int = 0,
    ) -> torch.Tensor:
        """Calculates slot loss.

        Args:
            slot_logits: Slot logits.
            slot_label_ids: Slot labels.
            attention_mask: Attention mask. Defaults to None.
            ignore_index: Index to ignore (i.e., does not contribute to input
              gradient). Defaults to 0.

        Returns:
            Slot loss.
        """
        if self.use_crf:
            return torch.tensor(
                -1
                * self.crf(
                    slot_logits,
                    slot_label_ids,
                    mask=attention_mask.byte(),
                    reduction="mean",
                )
            )

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = slot_logits.view(-1, self.num_slot_labels)[
                active_loss
            ]
            active_labels = slot_label_ids.view(-1)[active_loss]
            return loss_fct(active_logits, active_labels)

        return loss_fct(
            slot_logits.view(-1, self.num_slot_labels), slot_label_ids.view(-1)
        )

    def _get_tag_intent_loss(
        self,
        tag_intent_logits: torch.Tensor,
        tag_intent_label_ids: torch.Tensor,
        ignore_index: int = 0,
    ) -> torch.Tensor:
        """Calculates tag-intent loss.

        Args:
            tag_intent_logits: Tag-intent logits.
            tag_intent_label_ids: Tag-intent labels.
            ignore_index: Index to ignore (i.e., does not contribute to input
              gradient). Defaults to 0.

        Returns:
            Tag-intent loss.
        """
        fct = nn.NLLLoss(ignore_index=ignore_index)
        return fct(
            torch.log(tag_intent_logits + 1e-10), tag_intent_label_ids.view(-1)
        )

    def _create_tag_intent_mask(
        self, slot_logits: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Creates tag-intent mask from predicted slot labels.

        Args:
            slot_logits: Slot logits.
            attention_mask: Attention mask.

        Returns:
            Tag-intent mask.
        """
        if self.use_crf:
            slot_predictions = self.crf.decode(slot_logits)
        else:
            slot_predictions = torch.argmax(slot_logits, dim=2)

        slot_bios: List[List[str]] = [
            [] for _ in range(slot_predictions.size(0))
        ]
        for i in range(slot_predictions.size(0)):
            for j in range(slot_predictions.size(1)):
                if attention_mask[i][j] == 1:
                    slot_bios[i].append(
                        self.reverse_slot_labels_map[
                            slot_predictions[i][j].item()
                        ]
                    )
        tag_intent_masks = []
        tag_intent_pad = [tuple([0 for _ in range(self.num_mask)])]

        for i, slot_bio in enumerate(slot_bios):
            entities = get_entities(slot_bio)
            entities = entities[: self.num_mask]
            tag_intent_mask = [
                [0 for _ in slot_bio] for _ in range(self.num_mask)
            ]
            for j, (_, start, end) in enumerate(entities):
                tag_intent_mask[j][start : end + 1] = [  # noqa: E203
                    1 / (end - start + 1)
                ] * (end - start + 1)

            tag_intent_mask = tag_intent_mask[: self.max_length - 2]
            tag_intent_mask = (
                tag_intent_pad  # [CLS]
                + tag_intent_mask
                + tag_intent_pad  # [SEP]
            )
            # Pad to max_length
            tag_intent_mask += tag_intent_pad * (
                self.max_length - len(tag_intent_mask)
            )
            tag_intent_mask = list(zip(*tag_intent_mask))
            tag_intent_mask = [list(mask) for mask in tag_intent_mask]
            tag_intent_masks.append(tag_intent_mask)

        return torch.tensor(tag_intent_masks, dtype=torch.float32)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        intent_label_ids: torch.Tensor = None,
        slot_label_ids: torch.Tensor = None,
        tag_intent_mask: torch.Tensor = None,
        tag_intent_label_ids: torch.Tensor = None,
        ignore_index: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs forward pass.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask. Defaults to None.
            token_type_ids: Token type IDs. Defaults to None.
            intent_label_ids: Intent labels. Defaults to None.
            slot_label_ids: Slot labels. Defaults to None.
            tag_intent_mask: Tag-intent mask. Defaults to None.
            tag_intent_label_ids: Tag-intent labels. Defaults to None.
            ignore_index: Index to ignore (i.e., does not contribute to input
              gradient). Defaults to 0.

        Returns:
            Tuple with losses, logits, hidden states, and attention weights.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        total_loss = torch.tensor(0.0)
        intent_loss = None
        slot_loss = None
        tag_intent_loss = None

        # Multi-intent detection
        intent_logits = self.multi_intent_classifier(pooled_output)
        if intent_label_ids is not None:
            intent_loss = self._get_intent_loss(intent_logits, intent_label_ids)
            total_loss += intent_loss

        # Slot filling
        slot_logits = self.slot_classifier(sequence_output)
        if slot_label_ids is not None:
            slot_loss = self._get_slot_loss(
                slot_logits, slot_label_ids, attention_mask, ignore_index
            )
            total_loss += slot_loss

        # Slot-intent mapping
        if tag_intent_mask is None:
            # Build tag-intent mask from predicted slots
            tag_intent_mask = self._create_tag_intent_mask(
                slot_logits, attention_mask
            )

        tag_intent_vector = torch.einsum(
            "bml,bld->bmd", tag_intent_mask, sequence_output
        )
        cls_token = pooled_output.unsqueeze(1)
        cls_token = cls_token.repeat(1, self.num_mask, 1)
        tag_intent_vector = torch.cat((cls_token, tag_intent_vector), dim=2)
        tag_intent_vector = tag_intent_vector.view(
            tag_intent_vector.size(0) * tag_intent_vector.size(1), -1
        )

        tag_intent_logits = self.tag_intent_classifier(tag_intent_vector)
        intent_probs = intent_logits.unsqueeze(1).repeat(1, self.num_mask, 1)
        intent_probs = intent_probs.view(
            intent_probs.size(0) * intent_probs.size(1), -1
        )
        tag_intent_logits = tag_intent_logits * intent_probs
        tag_intent_logits = tag_intent_logits.div(
            tag_intent_logits.sum(dim=1, keepdim=True)
        )

        if tag_intent_label_ids is not None:
            tag_intent_loss = self._get_tag_intent_loss(
                tag_intent_logits, tag_intent_label_ids, ignore_index
            )
            total_loss += tag_intent_loss

        return (
            [total_loss, intent_loss, slot_loss, tag_intent_loss],
            (intent_logits, slot_logits, tag_intent_logits),
        ) + outputs[2:]

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        tag_intent_mask: torch.Tensor = None,
    ) -> List[DialogueAct]:
        """Predicts dialogue acts from intents, slots, and tag-intents.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs. Defaults to None.
            tag_intent_mask: Tag-intent mask. Defaults to None.

        Returns:
            List of predicted dialogue acts.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                tag_intent_mask=tag_intent_mask,
            )
            _, (_, slot_logits, tag_intent_logits) = outputs[:2]

            # Slot filling
            if self.use_crf:
                slot_predictions = self.crf.decode(slot_logits)
            else:
                slot_predictions = torch.argmax(slot_logits, dim=2)

            slot_labels: List[List[str]] = [
                [] for _ in range(slot_predictions.size(0))
            ]
            for i in range(slot_predictions.size(0)):
                for j in range(slot_predictions.size(1)):
                    if attention_mask[i][j] == 1:
                        slot_labels[i].append(
                            self.reverse_slot_labels_map[
                                slot_predictions[i][j].item()
                            ]
                        )

            # Slot-intent mapping
            slot_intent_predictions = torch.argmax(tag_intent_logits, dim=1)
            slot_intents = [
                self.reverse_intent_labels_map[intent.item()]
                for intent in slot_intent_predictions
            ]

            # Create dialogue acts
            dialogue_acts = []

            entities = get_entities(slot_labels[0])
            entities = entities[: self.num_mask]
            for intent, entities in groupby(
                zip(slot_intents, entities), key=lambda x: x[0]
            ):
                dialogue_act = DialogueAct(intent=Intent(intent))

                for _, (slot, start, end) in entities:
                    value = self.tokenizer.decode(
                        input_ids[0][start : end + 1],  # noqa: E203
                        skip_special_tokens=True,
                    ).strip()
                    dialogue_act.annotations.append(
                        Annotation(slot=slot, value=value)
                    )
                dialogue_acts.append(dialogue_act)

        return dialogue_acts

    def annotate_utterance(self, utterance: Utterance) -> AnnotatedUtterance:
        """Annotates an utterance with dialogue acts.

        Args:
            utterance: Utterance to annotate.

        Returns:
            Annotated utterance.
        """
        text = utterance.text.strip()
        words = re.split(SPLIT_CHARS, text)
        tokenized_utterance = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            return_attention_mask=True,
            padding="max_length",
            max_length=self.max_length,
        )

        dialogue_acts = self.predict(
            tokenized_utterance["input_ids"],
            tokenized_utterance["attention_mask"],
            tokenized_utterance["token_type_ids"],
            tag_intent_mask=None,
        )

        annotated_utterance = AnnotatedUtterance.from_utterance(utterance)
        annotated_utterance.add_dialogue_acts(dialogue_acts)
        return annotated_utterance

    @classmethod
    def from_pretrained(cls, model_dir: str) -> SLIM:
        """Loads a pretrained SLIM model.

        Args:
            model_dir: Directory containing the pretrained model.

        Raises:
            FileNotFoundError: If the model directory does not exist or does not
              contain the model files.

        Returns:
            Pretrained SLIM model.
        """
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        model_path = os.path.join(model_dir, "model.pt")
        metadata_path = os.path.join(model_dir, "metadata.json")

        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(
                "Model files 'model.pt' and 'metadata.json' not found in "
                f"{model_dir}"
            )

        # Load state dictionary
        state_dict = torch.load(model_path)

        # Load metadata
        args = json.load(open(metadata_path, "r"))

        print(f"Loading model training on {args['participant']} utterances.")

        # Initialize model
        model = cls(
            intent_labels_map=args["intent_labels_map"],
            slot_labels_map=args["slot_labels_map"],
            dropout=args["dropout"],
            use_crf=args["use_crf"],
            num_mask=args["num_mask"],
        )
        model.load_state_dict(state_dict)
        return model
