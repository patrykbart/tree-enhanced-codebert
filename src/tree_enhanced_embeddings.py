import torch
import torch.nn as nn


class TreeEnhancedRobertaEmbeddings(nn.Module):
    def __init__(self, config, yaml_config):
        super().__init__()
        self.yaml_config = yaml_config

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

        # Additional embeddings for tree structure
        self.depth_embeddings = nn.Embedding(yaml_config['model']['max_depth'], config.hidden_size)
        self.sibling_index_embeddings = nn.Embedding(yaml_config['model']['max_sibling_index'], config.hidden_size)

    def forward(
        self, 
        input_ids=None, 
        token_type_ids=None, 
        position_ids=None, 
        inputs_embeds=None, 
        past_key_values_length=0,
        depths=None,
        sibling_indices=None,
        tree_attention_mask=None
    ):
        position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # Add tree structure embeddings
        # Mask -1 values to zero vectors
        depths_mask = (depths != -1).unsqueeze(-1)
        sibling_indices_mask = (sibling_indices != -1).unsqueeze(-1)

        depth_embeddings = self.depth_embeddings(depths.clamp(min=0)) * depths_mask * tree_attention_mask.unsqueeze(-1)
        sibling_index_embeddings = self.sibling_index_embeddings(sibling_indices.clamp(min=0)) * sibling_indices_mask * tree_attention_mask.unsqueeze(-1)

        if self.yaml_config['model']['sum_embeddings']:
            pass
            embeddings += depth_embeddings + sibling_index_embeddings
        else:
            raise ValueError("Invalid embedding mode")

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_input_ids(self, input_ids, padding_idx, past_key_values_length=0):
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx