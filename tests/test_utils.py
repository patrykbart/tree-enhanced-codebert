"""Tests for the shared helpers in ``src/utils.py``."""
from pathlib import Path

import numpy as np
import pytest

import utils

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


class TestSetSeed:
    def test_python_random_is_deterministic(self):
        import random

        utils.set_seed(42)
        a = [random.random() for _ in range(5)]
        utils.set_seed(42)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_numpy_is_deterministic(self):
        utils.set_seed(7)
        a = np.random.rand(5)
        utils.set_seed(7)
        b = np.random.rand(5)
        assert np.array_equal(a, b)

    def test_torch_is_deterministic(self):
        import torch

        utils.set_seed(123)
        a = torch.rand(5)
        utils.set_seed(123)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        utils.set_seed(1)
        a = np.random.rand(5)
        utils.set_seed(2)
        b = np.random.rand(5)
        assert not np.array_equal(a, b)


class TestLoadConfig:
    @pytest.mark.parametrize("name", ["add", "concat", "add-weighted", "original"])
    def test_loads_real_project_configs(self, name):
        cfg = utils.load_config(CONFIGS_DIR / f"{name}.yaml")
        assert isinstance(cfg, dict)
        # Every config carries the master switch and an experiment block.
        assert "extra_embeddings" in cfg["model"]
        assert cfg["experiment"]["wandb_project"] == "CodeBERTa-small-v1"

    def test_fusion_flags_are_mutually_consistent(self):
        # add.yaml -> pure sum, concat.yaml -> concat, add-weighted -> weighted sum
        add = utils.load_config(CONFIGS_DIR / "add.yaml")["model"]
        assert add["sum_embeddings"] and not add["weighted_sum"] and not add["concat_embeddings"]

        concat = utils.load_config(CONFIGS_DIR / "concat.yaml")["model"]
        assert concat["concat_embeddings"] and not concat["sum_embeddings"]

        weighted = utils.load_config(CONFIGS_DIR / "add-weighted.yaml")["model"]
        assert weighted["sum_embeddings"] and weighted["weighted_sum"]

    def test_roundtrip_tmp_file(self, tmp_path):
        import yaml

        data = {"a": 1, "nested": {"b": [1, 2, 3]}}
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(data))
        assert utils.load_config(p) == data


class TestComputeMetrics:
    def test_perfect_predictions(self):
        labels = np.array([0, 1, 0, 1])
        # logits whose argmax matches labels exactly
        logits = np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
        m = utils.compute_metrics((logits, labels))
        assert m["accuracy"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_known_imperfect_case(self):
        # 3/4 correct: last sample predicted 0 but labelled 1.
        labels = np.array([0, 1, 0, 1])
        logits = np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 0.0]])
        m = utils.compute_metrics((logits, labels))
        assert m["accuracy"] == pytest.approx(0.75)
        # Macro F1 for this confusion is known: class0 P=2/3 R=1, class1 P=1 R=1/2
        assert m["f1"] == pytest.approx(0.7333333, abs=1e-4)

    def test_returns_expected_keys(self):
        labels = np.array([0, 1])
        logits = np.array([[1.0, 0.0], [0.0, 1.0]])
        m = utils.compute_metrics((logits, labels))
        assert set(m) == {"accuracy", "precision", "recall", "f1"}
