"""Tests for ``TreeEnhancedRobertaEmbeddings`` — the core contribution.

These exercise the three mutually-exclusive fusion modes, the ``-1`` /
``tree_attention_mask`` zero-masking of depth & sibling embeddings, the
learnable-weight wiring, and the position-id helper. Everything runs on CPU
with a tiny in-code config.
"""
import pytest
import torch

from tree_enhanced_embeddings import TreeEnhancedRobertaEmbeddings


def _make_inputs(batch=2, seq=6, vocab=200, pad_id=1, max_tree=16):
    torch.manual_seed(0)
    input_ids = torch.randint(5, vocab, (batch, seq))
    # ensure no accidental pad ids in the body
    input_ids[input_ids == pad_id] = pad_id + 1
    depths = torch.randint(0, max_tree, (batch, seq))
    sibling_indices = torch.randint(0, max_tree, (batch, seq))
    tree_attention_mask = torch.ones(batch, seq, dtype=torch.long)
    token_type_ids = torch.zeros(batch, seq, dtype=torch.long)
    return input_ids, depths, sibling_indices, tree_attention_mask, token_type_ids


class TestConstruction:
    def test_add_mode_has_no_weights_or_fusion(self, tiny_hf_config, make_yaml_config):
        yc = make_yaml_config(sum_embeddings=True, weighted_sum=False, concat_embeddings=False)
        emb = TreeEnhancedRobertaEmbeddings(tiny_hf_config, yc)
        assert not hasattr(emb, "fusion_layer")
        assert not hasattr(emb, "word_weight")

    def test_weighted_mode_creates_five_scalar_params(self, tiny_hf_config, make_yaml_config):
        yc = make_yaml_config(sum_embeddings=True, weighted_sum=True, concat_embeddings=False)
        emb = TreeEnhancedRobertaEmbeddings(tiny_hf_config, yc)
        for name in ("word_weight", "token_type_weight", "position_weight",
                     "depth_weight", "sibling_index_weight"):
            p = getattr(emb, name)
            assert isinstance(p, torch.nn.Parameter)
            assert p.shape == torch.Size([])
            assert p.item() == pytest.approx(1.0)

    def test_concat_mode_builds_fusion_layer(self, tiny_hf_config, make_yaml_config):
        yc = make_yaml_config(sum_embeddings=False, weighted_sum=False, concat_embeddings=True)
        emb = TreeEnhancedRobertaEmbeddings(tiny_hf_config, yc)
        assert hasattr(emb, "fusion_layer")
        first = emb.fusion_layer[0]
        # concatenates all 5 embedding types -> hidden_size
        assert first.in_features == tiny_hf_config.hidden_size * 5
        assert first.out_features == tiny_hf_config.hidden_size

    def test_tree_table_sizes_follow_yaml(self, tiny_hf_config, make_yaml_config):
        yc = make_yaml_config(sum_embeddings=True, weighted_sum=False,
                              concat_embeddings=False, max_depth=11, max_sibling_index=13)
        emb = TreeEnhancedRobertaEmbeddings(tiny_hf_config, yc)
        assert emb.depth_embeddings.num_embeddings == 11
        assert emb.sibling_index_embeddings.num_embeddings == 13


class TestForward:
    def test_output_shape_all_modes(self, tiny_hf_config, yaml_config_mode):
        _, yc = yaml_config_mode
        emb = TreeEnhancedRobertaEmbeddings(tiny_hf_config, yc).eval()
        input_ids, depths, sib, tmask, ttids = _make_inputs()
        out = emb(input_ids=input_ids, token_type_ids=ttids, depths=depths,
                  sibling_indices=sib, tree_attention_mask=tmask)
        assert out.shape == (2, 6, tiny_hf_config.hidden_size)
        assert torch.isfinite(out).all()

    def test_invalid_mode_raises(self, tiny_hf_config, make_yaml_config):
        yc = make_yaml_config(sum_embeddings=False, weighted_sum=False, concat_embeddings=False)
        emb = TreeEnhancedRobertaEmbeddings(tiny_hf_config, yc).eval()
        input_ids, depths, sib, tmask, ttids = _make_inputs()
        with pytest.raises(ValueError, match="Invalid embedding mode"):
            emb(input_ids=input_ids, token_type_ids=ttids, depths=depths,
                sibling_indices=sib, tree_attention_mask=tmask)


class TestTreeMasking:
    """The depth/sibling contribution must vanish for ``-1`` values and for
    positions where ``tree_attention_mask == 0``."""

    def test_minus_one_depths_equal_full_mask(self, tiny_hf_config, make_yaml_config):
        # With all depths/siblings == -1, the tree contribution is masked to
        # zero regardless of tree_attention_mask, so toggling the mask must not
        # change the output in add mode.
        yc = make_yaml_config(sum_embeddings=True, weighted_sum=False, concat_embeddings=False)
        emb = TreeEnhancedRobertaEmbeddings(tiny_hf_config, yc).eval()
        input_ids, _, _, _, ttids = _make_inputs()
        depths = torch.full_like(input_ids, -1)
        sib = torch.full_like(input_ids, -1)

        with torch.no_grad():
            out_mask_on = emb(input_ids=input_ids, token_type_ids=ttids, depths=depths,
                              sibling_indices=sib,
                              tree_attention_mask=torch.ones_like(input_ids))
            out_mask_off = emb(input_ids=input_ids, token_type_ids=ttids, depths=depths,
                               sibling_indices=sib,
                               tree_attention_mask=torch.zeros_like(input_ids))
        assert torch.allclose(out_mask_on, out_mask_off, atol=1e-6)

    def test_real_depths_change_output(self, tiny_hf_config, make_yaml_config):
        # Sanity: when depths are valid (not -1) and the mask is on, the tree
        # embeddings DO affect the output (otherwise the feature is a no-op).
        yc = make_yaml_config(sum_embeddings=True, weighted_sum=False, concat_embeddings=False)
        emb = TreeEnhancedRobertaEmbeddings(tiny_hf_config, yc).eval()
        input_ids, depths, sib, tmask, ttids = _make_inputs()

        with torch.no_grad():
            out_active = emb(input_ids=input_ids, token_type_ids=ttids, depths=depths,
                             sibling_indices=sib, tree_attention_mask=tmask)
            out_masked = emb(input_ids=input_ids, token_type_ids=ttids, depths=depths,
                             sibling_indices=sib,
                             tree_attention_mask=torch.zeros_like(tmask))
        assert not torch.allclose(out_active, out_masked, atol=1e-4)


class TestPositionIds:
    def test_padding_and_increment(self, tiny_hf_config, make_yaml_config):
        yc = make_yaml_config(sum_embeddings=True, weighted_sum=False, concat_embeddings=False)
        emb = TreeEnhancedRobertaEmbeddings(tiny_hf_config, yc)
        pad = tiny_hf_config.pad_token_id  # 1
        # tokens: [t, t, t, pad, pad]
        input_ids = torch.tensor([[5, 6, 7, pad, pad]])
        pos = emb.create_position_ids_from_input_ids(input_ids, pad)
        # non-pad positions count up from pad+1; pad positions stay at pad.
        assert pos.tolist() == [[pad + 1, pad + 2, pad + 3, pad, pad]]
