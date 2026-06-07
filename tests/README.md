# Tests

Unit tests for the Tree-Enhanced CodeBERTa code. They are deliberately **fast,
CPU-only, and hermetic** — no GPU, no network, no W&B, and no HuggingFace Hub
access. Model configs are built in-code with tiny dimensions, tokenizers are
stubbed, and tree-sitter grammars (bundled as Python wheels) provide real ASTs.

## Running

```bash
pdm run test            # via the PDM script
# or, in any env with the deps + pytest:
pytest
```

`pythonpath = ["src"]` (in `pyproject.toml`) and `tests/conftest.py` both put
`src/` on the import path, because the source modules import one another by bare
module name (`from utils import ...`).

## Layout

| File | Covers |
|------|--------|
| `test_utils.py` | `set_seed` determinism, `load_config` against the real `configs/*.yaml`, `compute_metrics` on known confusion matrices |
| `test_embeddings.py` | `TreeEnhancedRobertaEmbeddings`: the 3 fusion modes (sum / weighted-sum / concat), `-1` & `tree_attention_mask` zero-masking, learnable-weight wiring, invalid-mode error, position-id helper |
| `test_parse_dataset.py` | `enhance_code` AST leaf walk, `process_example`/`process_batch` alignment, special tokens, padding, truncation, UNK/unaligned masking |
| `test_models.py` | `TreeEnhancedRobertaForMaskedLM` and `TreeEnhancedCodeBERTaCloneDetection` forward/backward passes, loss, `weighted_sum` wiring, both `extra_embeddings` branches |

## Notes

- The clone-detector baseline branch (`extra_embeddings: false`) is only
  functional after `finetune_clone_detection.py` reassigns
  `model.roberta = prev_model.roberta`. The test mirrors that weight transfer;
  the `AutoModelForMaskedLM` created in `__init__` is a placeholder.
- The fixtures use `hidden_dropout_prob = 0` so forward passes are
  deterministic and equality assertions are stable.
