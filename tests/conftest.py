"""Shared pytest fixtures and path setup for the test suite.

The project's modules under ``src/`` import each other by bare module name
(e.g. ``from utils import ...``, ``from tree_enhanced_embeddings import ...``)
rather than as a package, so ``src/`` must be on ``sys.path`` for imports to
resolve. We do that here once for the whole suite.

All tests are designed to run on CPU with no network access: model configs are
built in-code (no HuggingFace download), tokenizers are stubbed, and W&B / the
HF Hub are never touched.
"""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def tiny_hf_config():
    """A minimal RoBERTa config small enough for fast CPU forward passes.

    Mirrors the structure of ``huggingface/CodeBERTa-small-v1`` but with tiny
    dimensions so the tests run in milliseconds.
    """
    from transformers import RobertaConfig

    return RobertaConfig(
        vocab_size=200,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        type_vocab_size=1,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        layer_norm_eps=1e-5,
        hidden_dropout_prob=0.0,  # deterministic forward passes
        attention_probs_dropout_prob=0.0,
    )


def _yaml_config(sum_embeddings, weighted_sum, concat_embeddings,
                 extra_embeddings=True, max_depth=16, max_sibling_index=16):
    """Build a ``yaml_config``-shaped dict like the parsed ``configs/*.yaml``."""
    return {
        "model": {
            "extra_embeddings": extra_embeddings,
            "max_depth": max_depth,
            "max_sibling_index": max_sibling_index,
            "sum_embeddings": sum_embeddings,
            "weighted_sum": weighted_sum,
            "concat_embeddings": concat_embeddings,
        }
    }


@pytest.fixture
def make_yaml_config():
    """Factory fixture returning a yaml_config dict for a chosen fusion mode."""
    return _yaml_config


@pytest.fixture(params=["add", "add_weighted", "concat"])
def yaml_config_mode(request):
    """Parametrized over the three mutually-exclusive fusion modes."""
    mode = request.param
    if mode == "add":
        return mode, _yaml_config(sum_embeddings=True, weighted_sum=False, concat_embeddings=False)
    if mode == "add_weighted":
        return mode, _yaml_config(sum_embeddings=True, weighted_sum=True, concat_embeddings=False)
    return mode, _yaml_config(sum_embeddings=False, weighted_sum=False, concat_embeddings=True)


class StubTokenizer:
    """Minimal stand-in for the CodeBERTa tokenizer used by the parser.

    ``parse_dataset`` only uses two methods:
      * ``convert_tokens_to_ids`` — maps each leaf-node text to an id. Here we
        hash the token into a small range, returning a fixed ``unk_id`` for the
        empty string so UNK handling can be exercised deterministically.
      * ``encode("")`` — used to fetch the ``<s>``/``</s>`` special-token ids;
        the parser indexes ``[0]`` and ``[1]`` of the result.
    """

    bos_id = 0
    eos_id = 2
    unk_id = 3

    def convert_tokens_to_ids(self, tokens):
        single = isinstance(tokens, str)
        seq = [tokens] if single else tokens
        ids = [self.unk_id if t == "" else (5 + (hash(t) % 100)) for t in seq]
        return ids[0] if single else ids

    def encode(self, text):
        # Real tokenizer returns [bos, eos] for the empty string.
        return [self.bos_id, self.eos_id]


@pytest.fixture
def stub_tokenizer():
    return StubTokenizer()
