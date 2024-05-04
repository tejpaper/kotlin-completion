import torch
import torch.nn as nn


class HookedEmbedding(nn.Module):
    def __init__(self, emb: nn.Embedding, hook_idx: int) -> None:
        super().__init__()

        self.main_emb = emb
        self.hook_idx = hook_idx
        self.hooked_emb = nn.Embedding(num_embeddings=1,
                                       embedding_dim=emb.embedding_dim,
                                       device=emb.weight.device,
                                       dtype=emb.weight.dtype)
        self.hooked_emb.weight.data.copy_(emb.weight[None, hook_idx])
        self.zero = torch.tensor(0, device=emb.weight.device)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.where((input_ids == self.hook_idx).unsqueeze(-1),
                           self.hooked_emb(self.zero), self.main_emb(input_ids))


def disable_decoder_attn_mask(_phi_decoder_layer: nn.Module, args, kwargs) -> tuple:
    attention_mask = kwargs.get('attention_mask')

    if attention_mask is not None:  # expand the last row of each matrix
        attention_mask[:] = attention_mask[..., -1, :].unsqueeze(-2).expand_as(attention_mask)

    return args, kwargs


@torch.no_grad()
def adapt_phi(phi: nn.Module, mask_idx: int) -> nn.Module:
    def select_hidden_state(_phi_model: nn.Module, _args, kwargs, output):
        input_ids = kwargs.get('input_ids')
        output.last_hidden_state = output[0][input_ids.to(output[0].device) == mask_idx]
        return output

    # emphasize mask embedding
    phi.model.embed_tokens = HookedEmbedding(phi.model.embed_tokens, mask_idx)

    # encoder model, logits for mask token only
    phi.model.layers[0].register_forward_pre_hook(disable_decoder_attn_mask, with_kwargs=True)
    phi.model.register_forward_hook(select_hidden_state, with_kwargs=True)

    # shorten classification head
    phi.lm_head.weight.data = phi.lm_head.weight.data[:mask_idx]
    phi.lm_head.bias.data = phi.lm_head.bias.data[:mask_idx]
    phi.lm_head.out_features, phi.lm_head.in_features = phi.lm_head.weight.shape

    return phi


def set_model_mode(model: nn.Module, training: bool = False) -> nn.Module:
    for params in model.parameters():
        params.requires_grad_(training)
    return model.train(training)
