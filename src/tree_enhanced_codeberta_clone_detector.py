import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from tree_enhanced_codeberta import TreeEnhancedRoberta
from transformers import AutoModelForMaskedLM
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Union, Tuple


class TreeEnhancedCodeBERTaCloneDetection(nn.Module):
    def __init__(self, config, yaml_config):
        super().__init__()
        self.num_labels = config.num_labels
        self.config = config

        self.extra_embeddings = yaml_config['model']['extra_embeddings']

        if self.extra_embeddings:
            self.roberta = TreeEnhancedRoberta(config, yaml_config)
        else:
            self.roberta = AutoModelForMaskedLM.from_config(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 2)
        )

    def forward(
        self,
        input_ids_1: Optional[torch.LongTensor] = None,
        attention_mask_1: Optional[torch.FloatTensor] = None,
        depths_1: Optional[torch.Tensor] = None,
        sibling_indices_1: Optional[torch.Tensor] = None,
        tree_attention_mask_1: Optional[torch.Tensor] = None,
        input_ids_2: Optional[torch.LongTensor] = None,
        attention_mask_2: Optional[torch.FloatTensor] = None,
        depths_2: Optional[torch.Tensor] = None,
        sibling_indices_2: Optional[torch.Tensor] = None,
        tree_attention_mask_2: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        if self.extra_embeddings:
            outputs_1 = self.roberta(
                input_ids_1,
                attention_mask=attention_mask_1,
                depths=depths_1,
                sibling_indices=sibling_indices_1,
                tree_attention_mask=tree_attention_mask_1,
            ).last_hidden_state
            outputs_2 = self.roberta(
                input_ids_2,
                attention_mask=attention_mask_2,
                depths=depths_2,
                sibling_indices=sibling_indices_2,
                tree_attention_mask=tree_attention_mask_2,
            ).last_hidden_state
        else:
            outputs_1 = self.roberta(input_ids_1, attention_mask=attention_mask_1).last_hidden_state
            outputs_2 = self.roberta(input_ids_2, attention_mask=attention_mask_2).last_hidden_state

        sequence_output_1 = outputs_1[:, 0, :]
        sequence_output_2 = outputs_2[:, 0, :]

        combined_output = torch.cat((sequence_output_1, sequence_output_2), dim=-1)

        logits = self.classifier(combined_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            self.config.problem_type = "single_label_classification"

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )