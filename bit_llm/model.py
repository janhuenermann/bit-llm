import re
from typing import Tuple


import torch
from lightning.pytorch import LightningModule
from torch import Tensor, nn
from torch.nn import functional as F


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

    def forward(self, x):
        return F.linear(x, self.weight)


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_key_value_heads: int = 0):
        super().__init__()
        num_key_value_heads = (
            num_key_value_heads if num_key_value_heads > 0 else num_heads
        )
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = Linear(hidden_dim, self.head_dim * num_heads)
        self.k_proj = Linear(hidden_dim, self.head_dim * num_key_value_heads)
        self.v_proj = Linear(hidden_dim, self.head_dim * num_key_value_heads)
        self.o_proj = Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor, rotary: Tensor) -> Tensor:
        q = self.q_proj(x).view(*x.shape[:-1], self.num_heads, self.head_dim)
        k = self.k_proj(x).view(*x.shape[:-1], self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(*x.shape[:-1], self.num_key_value_heads, self.head_dim)
        q, k = _apply_rotary(q, k, rotary)

        if self.num_key_value_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_key_value_heads, -2)
            v = v.repeat_interleave(self.num_heads // self.num_key_value_heads, -2)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).flatten(-2))


class Mlp(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_dim, intermediate_size)
        self.up_proj = Linear(hidden_dim, intermediate_size)
        self.down_proj = Linear(intermediate_size, hidden_dim)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x), inplace=True) * self.up_proj(x))


class Norm(nn.Module):
    def __init__(self, hidden_size: int, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        upcasted_x = x.float()
        upcasted_x = upcasted_x * torch.rsqrt(
            upcasted_x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon
        )
        return self.weight * upcasted_x.to(x.dtype)


class Block(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
    ):
        super().__init__()
        self.self_attn = Attention(hidden_dim, num_attention_heads, num_key_value_heads)
        self.mlp = Mlp(hidden_dim, intermediate_size)
        self.input_layernorm = Norm(hidden_dim)
        self.post_attention_layernorm = Norm(hidden_dim)

    def forward(self, x, rotary):
        x = x + self.self_attn(self.input_layernorm(x), rotary)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Llama(LightningModule):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        lr: float = 5e-5,
        vocab_size: int = 32000,
        max_sequence_length: int = 2048,
        should_init_weights: bool = True,
        rope_theta: float = 10000.0,
    ):
        super().__init__()

        self.lr = lr

        self.layers = nn.ModuleList(
            [
                Block(
                    hidden_dim,
                    intermediate_size,
                    num_attention_heads,
                    num_key_value_heads,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = Norm(hidden_dim)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = Linear(hidden_dim, vocab_size)

        if should_init_weights:
            self.apply(_weight_init)

        self._register_load_state_dict_pre_hook(self._load_weights_hook)
        self.register_buffer(
            "rotary",
            _compute_rotary(
                dim=hidden_dim // num_attention_heads,
                max_sequence_length=max_sequence_length,
                base=rope_theta,
            ),
            persistent=False,
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, self.rotary[:, : x.shape[1]])
        return self.norm(x)

    def training_step(self, batch):
        ids = batch["input_ids"]
        labels = batch["labels"]
        x = self.embed_tokens(ids)
        x = self.forward(x)
        logits = self.lm_head(x[:, :-1])
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels[:, 1:].flatten(), ignore_index=-1
        )
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.max_steps,
            pct_start=0.03,
            verbose=False,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "step",
            },
        }

    @torch.inference_mode()
    def generate(self, prompt, max_len=50):
        assert (
            prompt.ndim == 2
        ), f"Expected 2D input, but got tensor of shape {prompt.shape}"

        ids = prompt
        for _ in range(max_len):
            x = self.embed_tokens(ids)
            x = self.forward(x)
            logits = self.lm_head(x[:, -1])
            ids = torch.cat([ids, logits.argmax(-1).unsqueeze(-1)], dim=-1)

        return ids

    def _load_weights_hook(self, state_dict, prefix, *args):
        # Remove `model.*` prefix from the state_dict
        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            state_dict[re.sub(r"^model\.", "", k)] = v


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(q, k, rotary) -> Tuple[Tensor, Tensor]:
    """
    Apply rotary positional embedding to the queries and keys.

    Args:
        q: queries Tensor [batch, sequence, n_heads, head_dim]
        k: keys Tensor [batch, sequence, n_heads, head_dim]
        rotary: rotary positional embedding Tensor [2, sequence, head_dim]
    """
    sin, cos = rotary.unsqueeze(2).unbind(0)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _compute_rotary(
    dim: int, max_sequence_length: int = 2048, base: float = 10000.0, device=None
):
    """
    Computes rotary embeddings for a given dimension and maximum sequence length.

    Args:
        dim: the dimension of the embedding
        max_sequence_length: the maximum sequence length
        base: the base for the exponential
        device: the device to use

    Returns:
        A tensor of shape [2, max_sequence_length, dim]
    """
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
    )
    t = torch.arange(max_sequence_length, device=device, dtype=torch.int64)
    emb = torch.outer(t.type_as(inv_freq), inv_freq).repeat(1, 2)
    out = torch.stack((emb.sin(), emb.cos()), dim=0).to(torch.get_default_dtype())
    return out


def _weight_init(module: nn.Module):
    if isinstance(module, Linear):
        nn.init.normal_(module.weight, std=0.02)


# def get_quantized_state_dict(model: Llama):
#     quantized_state_dict = {}
#     for n, m in model.named_modules():
#         if isinstance(m, Linear):
#             quantized_state_dict[n + ".weight"] = m.encode()
#         elif isinstance(m, Norm):
#             quantized_state_dict[n + ".weight"] = m.weight.detach().clone().half()
#     quantized_state_dict = {k: v.cpu().numpy() for k, v in quantized_state_dict.items()}
#     return quantized_state_dict
