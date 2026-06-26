"""Minimal EVA02 backbone implementation for inference.

This module implements the EVA02 base/14 448px feature extractor used by
``eva02_base_patch14_448.mim_in22k_ft_in22k_in1k`` without depending on timm.
The module and parameter names intentionally mirror timm's EVA implementation so
checkpoints trained with the previous timm backbone can be loaded unchanged.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_2tuple(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def _drop_path(
    x: torch.Tensor,
    drop_prob: float,
    training: bool,
    scale_by_keep: bool = True,
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """2D image to patch embedding with timm-compatible names."""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 448,
        patch_size: Union[int, Tuple[int, int]] = 14,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = _to_2tuple(img_size)
        self.patch_size = _to_2tuple(patch_size)
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        if (height, width) != self.img_size:
            raise ValueError(
                f"Input image size ({height}x{width}) doesn't match "
                f"model ({self.img_size[0]}x{self.img_size[1]})."
            )
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


def _freq_bands(
    num_bands: int,
    temperature: float = 10000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    exp = torch.arange(0, num_bands, dtype=torch.float32, device=device) / num_bands
    return 1.0 / (temperature**exp)


def _build_rotary_pos_embed(
    feat_shape: Tuple[int, int],
    dim: int,
    temperature: float = 10000.0,
    ref_feat_shape: Optional[Tuple[int, int]] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    bands = _freq_bands(dim // 4, temperature=temperature, device=device)
    coords = [
        torch.arange(size, device=device, dtype=torch.float32)
        for size in feat_shape
    ]
    if ref_feat_shape is not None:
        coords = [
            coord / size * ref_size
            for coord, size, ref_size in zip(coords, feat_shape, ref_feat_shape)
        ]
    grid = torch.stack(torch.meshgrid(coords, indexing="ij"), dim=-1)
    pos = grid.unsqueeze(-1) * bands
    sin_emb = pos.sin().to(dtype=dtype).reshape(feat_shape[0] * feat_shape[1], -1)
    cos_emb = pos.cos().to(dtype=dtype).reshape(feat_shape[0] * feat_shape[1], -1)
    sin_emb = sin_emb.repeat_interleave(2, dim=-1)
    cos_emb = cos_emb.repeat_interleave(2, dim=-1)
    return torch.cat((sin_emb, cos_emb), dim=-1)


class RotaryEmbeddingCat(nn.Module):
    """Concatenated sin/cos RoPE matching timm's ``RotaryEmbeddingCat``."""

    def __init__(
        self,
        dim: int,
        feat_shape: Tuple[int, int],
        ref_feat_shape: Tuple[int, int] = (16, 16),
        temperature: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape
        self.temperature = temperature
        num_pos = feat_shape[0] * feat_shape[1]
        self.register_buffer(
            "pos_embed",
            torch.empty(num_pos, dim * 2),
            persistent=False,
        )
        self._init_buffers()

    def _init_buffers(self) -> None:
        self.pos_embed.copy_(
            _build_rotary_pos_embed(
                feat_shape=self.feat_shape,
                dim=self.dim,
                temperature=self.temperature,
                ref_feat_shape=self.ref_feat_shape,
                device=self.pos_embed.device,
                dtype=self.pos_embed.dtype,
            )
        )

    def get_embed(self) -> torch.Tensor:
        return self.pos_embed


def _rot(x: torch.Tensor) -> torch.Tensor:
    return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).reshape(x.shape)


def _apply_rot_embed_cat(x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    sin_emb, cos_emb = emb.chunk(2, dim=-1)
    return x * cos_emb + _rot(x) * sin_emb


class EvaAttention(nn.Module):
    """EVA attention with separate q/k/v projections and RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        num_prefix_tokens: int = 1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.num_prefix_tokens = num_prefix_tokens

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.qkv = None
        self.q_bias = None
        self.k_bias = None
        self.v_bias = None
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            npt = self.num_prefix_tokens
            q = torch.cat(
                [q[:, :, :npt, :], _apply_rot_embed_cat(q[:, :, npt:, :], rope)],
                dim=2,
            ).type_as(v)
            k = torch.cat(
                [k[:, :, :npt, :], _apply_rot_embed_cat(k[:, :, npt:, :], rope)],
                dim=2,
            ).type_as(v)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(batch_size, seq_len, channels)
        x = self.norm(x)
        x = self.proj(x)
        return self.proj_drop(x)


class SwiGLU(nn.Module):
    """SwiGLU MLP with timm-compatible parameter names."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        self.fc1_g = nn.Linear(in_features, hidden_features)
        self.fc1_x = nn.Linear(in_features, hidden_features)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return self.drop2(x)


class EvaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        drop_path: float,
        norm_layer: type[nn.Module],
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EvaAttention(dim=dim, num_heads=num_heads)
        self.init_values = None
        self.gamma_1 = None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = SwiGLU(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            norm_layer=norm_layer,
        )
        self.gamma_2 = None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, rope: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


def _global_pool_nlc(
    x: torch.Tensor,
    pool_type: str = "avg",
    num_prefix_tokens: int = 1,
) -> torch.Tensor:
    if pool_type == "token":
        return x[:, 0]
    if pool_type == "avg":
        return x[:, num_prefix_tokens:].mean(dim=1)
    raise ValueError(f"Unsupported pool type: {pool_type}")


class Eva(nn.Module):
    """EVA02 Base/14 feature extractor compatible with timm checkpoints."""

    def __init__(
        self,
        img_size: int = 448,
        patch_size: int = 14,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4 * 2 / 3,
        num_classes: int = 0,
        global_pool: str = "avg",
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        norm_layer = lambda dim: nn.LayerNorm(dim, eps=1e-6)
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1
        self.no_embed_class = False
        self.grad_checkpointing = False

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.reg_token = None
        self.cls_embed = True
        self.pos_embed = nn.Parameter(torch.empty(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)
        self.patch_drop = None
        self.rope_mixed = False
        self.rope = RotaryEmbeddingCat(
            dim=embed_dim // num_heads,
            feat_shape=self.patch_embed.grid_size,
            ref_feat_shape=(16, 16),
        )
        self.norm_pre = nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                EvaBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.Identity()
        self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim)
        self.head_drop = nn.Dropout(0.0)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        with torch.no_grad():
            for layer_id, layer in enumerate(self.blocks):
                scale = math.sqrt(2.0 * (layer_id + 1))
                layer.attn.proj.weight.div_(scale)
                layer.mlp.fc2.weight.div_(scale)

    def _pos_embed(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rot_pos_embed = self.rope.get_embed().unsqueeze(0).unsqueeze(1)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        return self.pos_drop(x), rot_pos_embed

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        x = self.norm_pre(x)
        for block in self.blocks:
            x = block(x, rope=rot_pos_embed)
        return self.norm(x)

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = _global_pool_nlc(
            x,
            pool_type=self.global_pool,
            num_prefix_tokens=self.num_prefix_tokens,
        )
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.forward_head(x)


def eva02_base_patch14_448(num_classes: int = 0) -> Eva:
    return Eva(
        img_size=448,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4 * 2 / 3,
        num_classes=num_classes,
    )

def eva02_large_patch14_448(num_classes: int = 0) -> Eva:
    return Eva(
        img_size=448,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        num_classes=num_classes,
    )

"""
| Параметр                                | `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k` | `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` |
| --------------------------------------- | -----------------------------------------------: | -----------------------------------------------: |
| Вариант                                 |                                       EVA02-Base |                                      EVA02-Large |
| Input                                   |                                          448×448 |                                          448×448 |
| Patch size                              |                                            14×14 |                                            14×14 |
| Patch grid                              |                                            32×32 |                                            32×32 |
| Tokens до pooling                       |                 1024 patch tokens + 1 CLS = 1025 |                 1024 patch tokens + 1 CLS = 1025 |
| Embedding dim                           |                                              768 |                                             1024 |
| Transformer blocks / depth              |                                               12 |                                               24 |
| Attention heads                         |                                               12 |                                               16 |
| Head dim                                |                                               64 |                                               64 |
| MLP ratio                               |                                              8/3 |                                              8/3 |
| MLP hidden width, по оригинальному коду |                          `int(768 * 8/3) = 2048` |                         `int(1024 * 8/3) = 2730` |
| Params, timm model card                 |                                           ~87.1M |                                          ~305.1M |
| GMACs @448, timm model card             |                                           ~107.1 |                                           ~362.3 |
| Feature shape до head                   |                                 `(B, 1025, 768)` |                                `(B, 1025, 1024)` |
"""
