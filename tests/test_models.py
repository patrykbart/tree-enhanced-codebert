"""Tests for the model wrappers — MLM and clone detector.

All forward passes use a tiny in-code ``RobertaConfig`` (no HuggingFace
download) and run on CPU. We check output shapes, loss computation, the
``weighted_sum`` attribute wiring, and the ``extra_embeddings`` branch of the
clone detector.
"""
import pytest
import torch

from tree_enhanced_codeberta_mlm import TreeEnhancedRoberta, TreeEnhancedRobertaForMaskedLM
from tree_enhanced_codeberta_clone_detector import TreeEnhancedCodeBERTaCloneDetection


def _tree_inputs(config, batch=2, seq=8, max_tree=16):
    torch.manual_seed(0)
    pad = config.pad_token_id
    input_ids = torch.randint(5, config.vocab_size, (batch, seq))
    input_ids[input_ids == pad] = pad + 1
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones(batch, seq, dtype=torch.long),
        "depths": torch.randint(0, max_tree, (batch, seq)),
        "sibling_indices": torch.randint(0, max_tree, (batch, seq)),
        "tree_attention_mask": torch.ones(batch, seq, dtype=torch.long),
    }


class TestTreeEnhancedRoberta:
    def test_forward_produces_hidden_states(self, tiny_hf_config, make_yaml_config):
        yc = make_yaml_config(sum_embeddings=True, weighted_sum=False, concat_embeddings=False)
        model = TreeEnhancedRoberta(tiny_hf_config, yc).eval()
        out = model(**_tree_inputs(tiny_hf_config))
        assert out.last_hidden_state.shape == (2, 8, tiny_hf_config.hidden_size)
        assert torch.isfinite(out.last_hidden_state).all()


class TestMaskedLM:
    @pytest.mark.parametrize("mode", ["add", "add_weighted", "concat"])
    def test_logits_shape_all_modes(self, tiny_hf_config, make_yaml_config, mode):
        kwargs = {
            "add": dict(sum_embeddings=True, weighted_sum=False, concat_embeddings=False),
            "add_weighted": dict(sum_embeddings=True, weighted_sum=True, concat_embeddings=False),
            "concat": dict(sum_embeddings=False, weighted_sum=False, concat_embeddings=True),
        }[mode]
        yc = make_yaml_config(**kwargs)
        model = TreeEnhancedRobertaForMaskedLM(tiny_hf_config, yc).eval()
        out = model(**_tree_inputs(tiny_hf_config))
        assert out.logits.shape == (2, 8, tiny_hf_config.vocab_size)

    def test_weighted_sum_attribute_tracks_config(self, tiny_hf_config, make_yaml_config):
        weighted = make_yaml_config(sum_embeddings=True, weighted_sum=True, concat_embeddings=False)
        plain = make_yaml_config(sum_embeddings=True, weighted_sum=False, concat_embeddings=False)
        assert TreeEnhancedRobertaForMaskedLM(tiny_hf_config, weighted).weighted_sum is True
        assert TreeEnhancedRobertaForMaskedLM(tiny_hf_config, plain).weighted_sum is False

    def test_loss_computed_with_labels(self, tiny_hf_config, make_yaml_config):
        yc = make_yaml_config(sum_embeddings=True, weighted_sum=False, concat_embeddings=False)
        model = TreeEnhancedRobertaForMaskedLM(tiny_hf_config, yc).eval()
        inputs = _tree_inputs(tiny_hf_config)
        labels = inputs["input_ids"].clone()
        labels[:, 0] = -100  # ignored position
        out = model(labels=labels, **inputs)
        assert out.loss is not None
        assert out.loss.ndim == 0
        assert out.loss.item() > 0

    def test_backward_pass_produces_gradients(self, tiny_hf_config, make_yaml_config):
        yc = make_yaml_config(sum_embeddings=True, weighted_sum=True, concat_embeddings=False)
        model = TreeEnhancedRobertaForMaskedLM(tiny_hf_config, yc).train()
        inputs = _tree_inputs(tiny_hf_config)
        out = model(labels=inputs["input_ids"].clone(), **inputs)
        out.loss.backward()
        # the learnable fusion weight should receive a gradient
        assert model.roberta.embeddings.depth_weight.grad is not None


class TestCloneDetector:
    def _pair_inputs(self, config, batch=2, seq=8, max_tree=16):
        a = _tree_inputs(config, batch, seq, max_tree)
        b = _tree_inputs(config, batch, seq, max_tree)
        return {
            "input_ids_1": a["input_ids"], "attention_mask_1": a["attention_mask"],
            "depths_1": a["depths"], "sibling_indices_1": a["sibling_indices"],
            "tree_attention_mask_1": a["tree_attention_mask"],
            "input_ids_2": b["input_ids"], "attention_mask_2": b["attention_mask"],
            "depths_2": b["depths"], "sibling_indices_2": b["sibling_indices"],
            "tree_attention_mask_2": b["tree_attention_mask"],
        }

    def test_tree_enhanced_branch_logits(self, tiny_hf_config, make_yaml_config):
        tiny_hf_config.num_labels = 2
        yc = make_yaml_config(sum_embeddings=True, weighted_sum=False,
                              concat_embeddings=False, extra_embeddings=True)
        model = TreeEnhancedCodeBERTaCloneDetection(tiny_hf_config, yc).eval()
        out = model(**self._pair_inputs(tiny_hf_config))
        assert out.logits.shape == (2, 2)

    def test_baseline_branch_ignores_tree_tensors(self, tiny_hf_config, make_yaml_config):
        # extra_embeddings=False -> baseline. In real usage (finetune_clone_
        # detection.py:71) the placeholder AutoModelForMaskedLM built in
        # __init__ is replaced by `prev_model.roberta`, a plain RobertaModel
        # that exposes `.last_hidden_state`. We mirror that weight transfer
        # here so the forward path matches how the model is actually run.
        from transformers import RobertaModel

        tiny_hf_config.num_labels = 2
        yc = make_yaml_config(sum_embeddings=False, weighted_sum=False,
                              concat_embeddings=False, extra_embeddings=False)
        model = TreeEnhancedCodeBERTaCloneDetection(tiny_hf_config, yc).eval()
        model.roberta = RobertaModel(tiny_hf_config).eval()  # as finetune does
        out = model(**self._pair_inputs(tiny_hf_config))
        assert out.logits.shape == (2, 2)

    def test_loss_with_labels(self, tiny_hf_config, make_yaml_config):
        tiny_hf_config.num_labels = 2
        yc = make_yaml_config(sum_embeddings=True, weighted_sum=False,
                              concat_embeddings=False, extra_embeddings=True)
        model = TreeEnhancedCodeBERTaCloneDetection(tiny_hf_config, yc).eval()
        inputs = self._pair_inputs(tiny_hf_config)
        out = model(labels=torch.tensor([0, 1]), **inputs)
        assert out.loss is not None and out.loss.item() > 0
