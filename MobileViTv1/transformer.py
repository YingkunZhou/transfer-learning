from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from activations import HardSwish


class MultiHeadAttention(nn.Module):
    """
    This layer applies a multi-head self- or cross-attention as described in
    `Attention is all you need <https://arxiv.org/abs/1706.03762>`_ paper

    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        num_heads (int): Number of heads in multi-head attention
        attn_dropout (float): Attention dropout. Default: 0.0
        bias (bool): Use bias or not. Default: ``True``

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input

    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            attn_dropout: float = 0.0,
            bias: bool = True,
            coreml_compatible: Optional[bool] = False,
            *args,
            **kwargs
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )

        self.qkv_proj = nn.Linear(in_features=embed_dim, out_features=3 * embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        # out_features dim == in_features dim
        self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.coreml_compatible = coreml_compatible

    def forward_default(
            self,
            x_q: Tensor,
            x_kv: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # [N, S, C]
        b_sz, S_len, in_channels = x_q.shape

        if x_kv is None:
            # self-attention
            # [N, S, C] --> [N, S, 3C] --> [N, S, 3, h, c] where C = hc
            qkv = self.qkv_proj(x_q).reshape(b_sz, S_len, 3, self.num_heads, -1)
            # [N, S, 3, h, c] --> [N, h, 3, S, C]
            qkv = qkv.transpose(1, 3).contiguous()

            # [N, h, 3, S, C] --> [N, h, S, C] x 3
            query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        else:
            T_len = x_kv.shape[1]

            # cross-attention
            # [N, S, C]
            query = F.linear(
                x_q,
                weight=self.qkv_proj.weight[: self.embed_dim, ...],
                bias=self.qkv_proj.bias[: self.embed_dim]
                if self.qkv_proj.bias is not None
                else None,
            )
            # [N, S, C] --> [N, S, h, c] --> [N, h, S, c]
            query = (
                query.reshape(b_sz, S_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

            # [N, T, C] --> [N, T, 2C]
            kv = F.linear(
                x_kv,
                weight=self.qkv_proj.weight[self.embed_dim:, ...],
                bias=self.qkv_proj.bias[self.embed_dim:]
                if self.qkv_proj.bias is not None
                else None,
            )
            # [N, T, 2C] --> [N, T, 2, h, c]
            kv = kv.reshape(b_sz, T_len, 2, self.num_heads, self.head_dim)
            # [N, T, 2, h, c] --> [N, h, 2, T, c]
            kv = kv.transpose(1, 3).contiguous()
            key, value = kv[:, :, 0], kv[:, :, 1]

        query = query * self.scaling

        # [N h, T, c] --> [N, h, c, T]
        key = key.transpose(-1, -2)

        # QK^T
        # [N, h, S, c] x [N, h, c, T] --> [N, h, S, T]
        attn = torch.matmul(query, key)

        batch_size, num_heads, num_src_tokens, num_tgt_tokens = attn.shape
        if attn_mask is not None:
            # attn_mask shape should be the same as attn
            assert list(attn_mask.shape) == [
                batch_size,
                num_src_tokens,
                num_tgt_tokens,
            ], "Shape of attention mask should be [{}, {}, {}]. Got: {}".format(
                batch_size, num_src_tokens, num_tgt_tokens, attn_mask.shape
            )
            # [N, S, T] --> [N, 1, S, T]
            attn_mask = attn_mask.unsqueeze(1)
            attn = attn + attn_mask

        if key_padding_mask is not None:
            # Do not attend to padding positions
            # key padding mask size is [N, T]
            assert key_padding_mask.dim() == 2 and list(key_padding_mask.shape) == [
                batch_size,
                num_tgt_tokens,
            ], "Key_padding_mask should be 2-dimension with shape [{}, {}]. Got: {}".format(
                batch_size, num_tgt_tokens, key_padding_mask.shape
            )
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1)
                .unsqueeze(2)
                .to(torch.bool),  # [N, T] --> [N, 1, 1, T]
                float("-inf"),
            )

        attn_dtype = attn.dtype
        attn_as_float = self.softmax(attn.float())
        attn = attn_as_float.to(attn_dtype)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, h, S, T] x [N, h, T, c] --> [N, h, S, c]
        out = torch.matmul(attn, value)

        # [N, h, S, c] --> [N, S, h, c] --> [N, S, C]
        out = out.transpose(1, 2).reshape(b_sz, S_len, -1)
        out = self.out_proj(out)

        return out

    def forward_pytorch(
            self,
            x_q: Tensor,
            x_kv: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        coremltools - RuntimeError: PyTorch convert function for op 'scaled_dot_product_attention' not implemented.
        """
        out, _ = F.multi_head_attention_forward(
            query=x_q,
            key=x_kv if x_kv is not None else x_q,
            value=x_kv if x_kv is not None else x_q,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=torch.empty([0]),
            in_proj_bias=self.qkv_proj.bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.attn_dropout.p,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=self.qkv_proj.weight[: self.embed_dim, ...],
            k_proj_weight=self.qkv_proj.weight[
                          self.embed_dim: 2 * self.embed_dim, ...
                          ],
            v_proj_weight=self.qkv_proj.weight[2 * self.embed_dim:, ...],
        )
        return out

    def forward_tracing(
            self,
            x_q: Tensor,
            x_kv: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if x_kv is None:
            # [N, S, C] --> # [N, S, 3C] Here, T=S
            qkv = self.qkv_proj(x_q)
            # # [N, S, 3C] --> # [N, S, C] x 3
            query, key, value = torch.chunk(qkv, chunks=3, dim=-1)
        else:
            # [N, S, C]
            query = F.linear(
                x_q,
                weight=self.qkv_proj.weight[: self.embed_dim, ...],
                bias=self.qkv_proj.bias[: self.embed_dim]
                if self.qkv_proj.bias is not None
                else None,
            )

            # [N, T, C] --> [N, T, 2C]
            kv = F.linear(
                x_kv,
                weight=self.qkv_proj.weight[self.embed_dim:, ...],
                bias=self.qkv_proj.bias[self.embed_dim:]
                if self.qkv_proj.bias is not None
                else None,
            )
            key, value = torch.chunk(kv, chunks=2, dim=-1)

        query = query * self.scaling

        # [N, S, C] --> [N, S, c] x h, where C = c * h
        query = torch.chunk(query, chunks=self.num_heads, dim=-1)

        # [N, T, C] --> [N, T, c] x h, where C = c * h
        value = torch.chunk(value, chunks=self.num_heads, dim=-1)
        # [N, T, C] --> [N, T, c] x h, where C = c * h
        key = torch.chunk(key, chunks=self.num_heads, dim=-1)

        wt_out = []
        for h in range(self.num_heads):
            attn_h = torch.matmul(query[h], key[h].transpose(-1, -2))
            attn_h = self.softmax(attn_h)
            attn_h = self.attn_dropout(attn_h)
            out_h = torch.matmul(attn_h, value[h])
            wt_out.append(out_h)

        wt_out = torch.cat(wt_out, dim=-1)
        wt_out = self.out_proj(wt_out)
        return wt_out

    def forward(
            self,
            x_q: Tensor,
            x_kv: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
            *args,
            **kwargs
    ) -> Tensor:
        if self.coreml_compatible:
            # For CoreML, we follow batch-first format. Make sure the input is of the form
            # [Batch , Sequence, Hidden_dim]
            return self.forward_tracing(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        elif kwargs.get("use_pytorch_mha", False):
            # pytorch uses sequence-first format. Make sure that input is of the form [Sequence, Batch, Hidden dim]
            return self.forward_pytorch(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            # our default implementation format follows batch-first format. Make sure the input is of the form
            # [Batch , Sequence, Hidden_dim]
            return self.forward_default(
                x_q=x_q,
                x_kv=x_kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )


class TransformerEncoder(nn.Module):
    """
    This class defines the pre-norm `Transformer encoder <https://arxiv.org/abs/1706.03762>`_
    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        ffn_latent_dim (int): Inner dimension of the FFN
        num_heads (int) : Number of heads in multi-head attention. Default: 8
        attn_dropout (float): Dropout rate for attention in multi-head attention. Default: 0.0
        dropout (float): Dropout rate. Default: 0.0
        ffn_dropout (float): Dropout between FFN layers. Default: 0.0

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
            self,
            embed_dim: int,
            ffn_latent_dim: int,
            coreml_compatible: bool,
            num_heads: Optional[int] = 8,
            attn_dropout: Optional[float] = 0.0,
            dropout: Optional[float] = 0.0,
            ffn_dropout: Optional[float] = 0.0,
            *args,
            **kwargs
    ) -> None:
        super().__init__()

        attn_unit = MultiHeadAttention(
            embed_dim,
            num_heads,
            attn_dropout=attn_dropout,
            bias=True,
            coreml_compatible = coreml_compatible
        )

        self.pre_norm_mha = nn.Sequential(
            nn.LayerNorm(embed_dim),
            attn_unit,
            nn.Dropout(p=dropout)
        )

        self.pre_norm_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            nn.SiLU(),
            # HardSwish(),
            nn.Dropout(p=ffn_dropout),
            nn.Linear(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            nn.Dropout(p=dropout)
        )
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        # multi-head attention
        res = x
        x = self.pre_norm_mha(x)
        x = x + res

        # feed forward network
        x = x + self.pre_norm_ffn(x)
        return x
