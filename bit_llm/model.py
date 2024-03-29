import re
from typing import Tuple
import math

import torch
from lightning.pytorch import LightningModule
from torch import Tensor, nn
from torch.nn import functional as F

# Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py


class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

    def forward(self, x):
        x = F.linear(x, self.quantize_weight())
        return x * (self.weight.abs().mean() + self.eps)

    def quantize_weight(self):
        # Quantize the weights to -1, 0, 1
        Wq = self.weight / (self.weight.abs().mean() + self.eps)
        return torch.round(Wq).clamp(-1, 1) + (Wq - Wq.detach())

    @torch.no_grad()
    def encode(self):
        """
        Pack the quantized weights in 1.6 bit, i.e. 5 values per byte.
        """
        powers = 3 ** torch.arange(5, device=self.weight.device)
        Wq = 1 + self.quantize_weight().detach().flatten()
        if len(Wq) % 5 != 0:
            Wq = torch.cat((Wq, torch.zeros(5 - len(Wq) % 5, dtype=Wq.dtype, device=Wq.device)))
        return torch.sum(Wq.reshape(-1, 5) * powers, dim=-1, dtype=torch.uint8)

    @torch.no_grad()
    def load_quantized(self, data):
        """
        Load the quantized weights from the packed data.
        """
        powers = 3 ** torch.arange(5, device=self.weight.device)
        Wq = (data[:, None] // powers).flatten()[:self.weight.data.numel()] % 3
        return Wq.view(self.weight.data.shape).to(self.weight.dtype) - 1.0


class BitEmbedding(BitLinear):
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__(hidden_dim, vocab_size)

    def forward(self, x):
        return F.embedding(x, self.quantize_weight())


class LlamaAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = BitLinear(hidden_dim, hidden_dim)
        self.k_proj = BitLinear(hidden_dim, hidden_dim)
        self.v_proj = BitLinear(hidden_dim, hidden_dim)
        self.o_proj = BitLinear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor, rotary: Tensor) -> Tensor:
        q = self.q_proj(x).view(*x.shape[:-1], self.num_heads, self.head_dim)
        k = self.k_proj(x).view(*x.shape[:-1], self.num_heads, self.head_dim)
        v = self.v_proj(x).view(*x.shape[:-1], self.num_heads, self.head_dim)
        q, k = _apply_rotary(q, k, rotary)
        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).flatten(-2))


class LlamaMLP(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = BitLinear(hidden_dim, intermediate_size)
        self.up_proj = BitLinear(hidden_dim, intermediate_size)
        self.down_proj = BitLinear(intermediate_size, hidden_dim)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x), inplace=True) * self.up_proj(x))


class LlamaNorm(nn.Module):
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


class LlamaLayer(nn.Module):
    def __init__(
        self, hidden_dim: int, num_attention_heads: int, intermediate_size: int
    ):
        super().__init__()
        self.self_attn = LlamaAttention(hidden_dim, num_attention_heads)
        self.mlp = LlamaMLP(hidden_dim, intermediate_size)
        self.input_layernorm = LlamaNorm(hidden_dim)
        self.post_attention_layernorm = LlamaNorm(hidden_dim)

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
        intermediate_size: int,
        lr: float = 5e-5,
        vocab_size: int = 32000,
        max_sequence_length: int = 2048,
        should_init_weights: bool = True,
    ):
        super().__init__()

        self.lr = lr

        self.layers = nn.ModuleList(
            [
                LlamaLayer(hidden_dim, num_attention_heads, intermediate_size)
                for _ in range(num_layers)
            ]
        )
        self.norm = LlamaNorm(hidden_dim)
        self.embed_tokens = BitEmbedding(vocab_size, hidden_dim)
        self.lm_head = BitLinear(hidden_dim, vocab_size)

        if should_init_weights:
            self.apply(_weight_init)

        self._register_load_state_dict_pre_hook(self._load_weights_hook)
        self.register_buffer(
            "rotary",
            _compute_rotary(hidden_dim // num_attention_heads, max_sequence_length),
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
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels[:, 1:].flatten(), ignore_index=-1)
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, total_steps=self.trainer.max_steps, pct_start=0.03, verbose=False
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "frequency": 1, "interval": "step"}}

    @torch.inference_mode()
    def generate(self, prompt, max_len=50):
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
    dim: int, max_sequence_length: int = 2048, base: int = 10000, device=None
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
    if isinstance(module, (BitLinear, BitEmbedding)):
        nn.init.normal_(module.weight, std=0.02)


def get_quantized_state_dict(model: Llama):
    quantized_state_dict = {}
    for n, m in model.named_modules():
        if isinstance(m, BitLinear):
            quantized_state_dict[n + ".weight"] = m.encode()
        elif isinstance(m, LlamaNorm):
            quantized_state_dict[n + ".weight"] = m.weight.detach().clone().half()
    quantized_state_dict = {k: v.cpu().numpy() for k, v in quantized_state_dict.items()}
    return quantized_state_dict
