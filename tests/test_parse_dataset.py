"""Tests for the data pipeline in ``src/parse_dataset.py``.

``enhance_code`` walks the tree-sitter AST and emits one entry per *leaf* node;
``process_example`` aligns those leaves to the dataset's ``func_code_tokens``,
adds special tokens, truncates and pads to ``max_length``. We use a real
tree-sitter Python parser (no network) and a stub tokenizer (see conftest).
"""
import pytest
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

import parse_dataset


@pytest.fixture
def py_parser():
    return Parser(Language(tspython.language()))


class TestEnhanceCode:
    def test_returns_four_aligned_lists(self, py_parser):
        code = "x = 1"
        tokens, mask, depths, siblings = parse_dataset.enhance_code(code, py_parser)
        assert len(tokens) == len(mask) == len(depths) == len(siblings)
        assert len(tokens) > 0

    def test_emits_only_leaf_tokens(self, py_parser):
        # `x = 1` has leaves: identifier `x`, operator `=`, integer `1`.
        tokens, mask, depths, siblings = parse_dataset.enhance_code("x = 1", py_parser)
        assert tokens == ["x", "=", "1"]
        # All emitted nodes are genuine leaves -> mask all ones.
        assert mask == [1, 1, 1]

    def test_depths_increase_with_nesting(self, py_parser):
        # A nested call should produce deeper leaves than a bare name.
        flat, _, flat_depths, _ = parse_dataset.enhance_code("x", py_parser)
        nested, _, nested_depths, _ = parse_dataset.enhance_code("f(g(x))", py_parser)
        assert max(nested_depths) > max(flat_depths)

    def test_sibling_indices_are_nonnegative(self, py_parser):
        _, _, _, siblings = parse_dataset.enhance_code("a + b + c", py_parser)
        assert all(s >= 0 for s in siblings)
        assert max(siblings) >= 1  # at least one node is a later sibling


class TestProcessExample:
    MAX_LEN = 32

    def _run(self, code, func_tokens, tok):
        return parse_dataset.process_example(code, func_tokens, "python", tok, self.MAX_LEN)

    def test_output_keys_and_lengths(self, stub_tokenizer):
        code = "x = 1"
        out = self._run(code, ["x", "=", "1"], stub_tokenizer)
        assert set(out) == {"input_ids", "attention_mask", "depths",
                            "sibling_indices", "tree_attention_mask"}
        for key in out:
            assert len(out[key]) == self.MAX_LEN, key

    def test_special_tokens_wrap_sequence(self, stub_tokenizer):
        out = self._run("x = 1", ["x", "=", "1"], stub_tokenizer)
        # bos/eos ids from the stub at the boundaries of the content.
        assert out["input_ids"][0] == stub_tokenizer.bos_id
        # 3 content tokens -> eos at index 4.
        assert out["input_ids"][4] == stub_tokenizer.eos_id
        # special-token positions carry depth/sibling -1 and tree-mask 0.
        assert out["depths"][0] == -1 and out["sibling_indices"][0] == -1
        assert out["tree_attention_mask"][0] == 0
        assert out["depths"][4] == -1 and out["tree_attention_mask"][4] == 0

    def test_padding_fills_remainder(self, stub_tokenizer):
        out = self._run("x = 1", ["x", "=", "1"], stub_tokenizer)
        # content(3) + 2 specials = 5 real positions; rest padded.
        assert out["input_ids"][5:] == [0] * (self.MAX_LEN - 5)
        assert out["attention_mask"][:5] == [1, 1, 1, 1, 1]
        assert out["attention_mask"][5:] == [0] * (self.MAX_LEN - 5)
        assert out["depths"][5:] == [-1] * (self.MAX_LEN - 5)
        assert out["tree_attention_mask"][5:] == [0] * (self.MAX_LEN - 5)

    def test_unaligned_tokens_get_masked(self, stub_tokenizer):
        # If func_code_tokens omits a leaf the parser produced, that leaf's
        # position must be flagged with tree_attention_mask = 0.
        code = "x = 1"  # parser leaves: x, =, 1
        out = self._run(code, ["x", "1"], stub_tokenizer)  # drop "="
        # leaf order is x(aligned), =(unaligned->0), 1(...). Index 1 is the body
        # position for "x" after the bos at 0; "=" is at body index 2.
        # The unaligned "=" leaf should be masked.
        body_mask = out["tree_attention_mask"][1:4]
        assert 0 in body_mask  # the dropped token forced a masked position

    def test_truncation_to_max_length(self, stub_tokenizer):
        # Build code with many leaves so it exceeds MAX_LEN before specials.
        names = [f"v{i}" for i in range(self.MAX_LEN + 10)]
        code = " + ".join(names)
        out = self._run(code, names, stub_tokenizer)
        for key in out:
            assert len(out[key]) == self.MAX_LEN, key
        # last position is padding/eos region, never overflows
        assert out["attention_mask"][-1] in (0, 1)


class TestProcessBatch:
    def test_batches_multiple_examples(self, stub_tokenizer):
        batch = {
            "whole_func_string": ["x = 1", "a + b"],
            "func_code_tokens": [["x", "=", "1"], ["a", "+", "b"]],
            "language": ["python", "python"],
        }
        out = parse_dataset.process_batch(batch, stub_tokenizer, 32)
        assert len(out["input_ids"]) == 2
        assert all(len(row) == 32 for row in out["input_ids"])
        assert set(out) == {"input_ids", "attention_mask", "depths",
                            "sibling_indices", "tree_attention_mask"}


def test_unsupported_language_raises(stub_tokenizer):
    with pytest.raises(ValueError, match="Unsupported language"):
        parse_dataset.process_example("x = 1", ["x"], "cobol", stub_tokenizer, 32)
