from typing import Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class ConvLayer(nn.Module):
    """
    Applies a 2D convolution over an input

    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): Kernel size for convolution.
        stride (Union[int, Tuple[int, int]]): Stride for convolution. Default: 1
        groups (Optional[int]): Number of groups in convolution. Default: 1
        bias (Optional[bool]): Use bias. Default: ``False``
        use_norm (Optional[bool]): Use normalization layer after convolution. Default: ``True``
        use_act (Optional[bool]): Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        For depth-wise convolution, `groups=C_{in}=C_{out}`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
        act_layer = nn.SiLU
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)

        padding = (
            int((kernel_size[0] - 1) / 2),
            int((kernel_size[1] - 1) / 2),
        )

        block = nn.Sequential()

        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=bias
        )

        block.add_module(name="conv", module=conv_layer)

        if use_norm:
            norm_layer = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)
            block.add_module(name="norm", module=norm_layer)

        if use_act:
            block.add_module(name="act", module=act_layer())

        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


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
        # FIXME: output_dim == embed_dim
        self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.coreml_compatible = coreml_compatible
        self.use_pytorch_mha = False

    def forward_default(self, x_q: Tensor) -> Tensor:
        # [N, S, C]
        b_sz, S_len, in_channels = x_q.shape

        # self-attention
        # [N, S, C] --> [N, S, 3C] --> [N, S, 3, h, c] where C = hc
        qkv = self.qkv_proj(x_q).reshape(b_sz, S_len, 3, self.num_heads, -1)
        # [N, S, 3, h, c] --> [N, h, 3, S, C]
        qkv = qkv.transpose(1, 3).contiguous()

        # [N, h, 3, S, C] --> [N, h, S, C] x 3
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        query = query * self.scaling

        # [N h, T, c] --> [N, h, c, T]
        key = key.transpose(-1, -2)

        # QK^T
        # [N, h, S, c] x [N, h, c, T] --> [N, h, S, T]
        attn = torch.matmul(query, key)

        batch_size, num_heads, num_src_tokens, num_tgt_tokens = attn.shape

        """ FIXME
        attn_dtype = attn.dtype
        attn_as_float = self.softmax(attn.float())
        attn = attn_as_float.to(attn_dtype)
        """
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, h, S, T] x [N, h, T, c] --> [N, h, S, c]
        out = torch.matmul(attn, value)

        # [N, h, S, c] --> [N, S, h, c] --> [N, S, C]
        out = out.transpose(1, 2).reshape(b_sz, S_len, -1)
        out = self.out_proj(out)

        return out

    def forward_pytorch(self, x_q: Tensor) -> Tensor:
        """
        coremltools - RuntimeError: PyTorch convert function for op 'scaled_dot_product_attention' not implemented.
        """
        out, _ = F.multi_head_attention_forward(
            query=x_q, key=x_q, value=x_q,
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
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
            use_separate_proj_weight=True, # FIXME!
            q_proj_weight=self.qkv_proj.weight[: self.embed_dim, ...],
            k_proj_weight=self.qkv_proj.weight[self.embed_dim: 2 * self.embed_dim, ...],
            v_proj_weight=self.qkv_proj.weight[2 * self.embed_dim:, ...],
        )
        return out

    def forward_tracing(self, x_q: Tensor) -> Tensor:
        # [N, S, C] --> # [N, S, 3C] Here, T=S
        qkv = self.qkv_proj(x_q)
        # # [N, S, 3C] --> # [N, S, C] x 3
        query, key, value = torch.chunk(qkv, chunks=3, dim=-1)

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

    def forward(self, x_q: Tensor) -> Tensor:
        if self.coreml_compatible:
            # For CoreML, we follow batch-first format. Make sure the input is of the form
            # [Batch , Sequence, Hidden_dim]
            return self.forward_tracing(x_q=x_q)
        elif self.use_pytorch_mha:
            # pytorch uses sequence-first format. Make sure that input is of the form [Sequence, Batch, Hidden dim]
            return self.forward_pytorch(x_q=x_q)
        else:
            # our default implementation format follows batch-first format. Make sure the input is of the form
            # [Batch , Sequence, Hidden_dim]
            return self.forward_default(x_q=x_q)


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
        transformer_norm_layer (Optional[str]): Normalization layer. Default: layer_norm

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
            self,
            embed_dim: int,
            ffn_latent_dim: int,
            num_heads: Optional[int] = 8,
            attn_dropout: Optional[float] = 0.0,
            dropout: Optional[float] = 0.0,
            ffn_dropout: Optional[float] = 0.0,
            act_layer = nn.SiLU,
            coreml_compatible: Optional[bool] = False,
            *args,
            **kwargs
    ) -> None:
        super().__init__()

        attn_unit = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
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
            act_layer(),
            nn.Dropout(p=ffn_dropout),
            nn.Linear(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            nn.Dropout(p=dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        # multi-head attention
        x = x + self.pre_norm_mha(x)
        # feed forward network
        x = x + self.pre_norm_ffn(x)
        return x
