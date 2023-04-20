from edgenext_bn_hs import EdgeNeXtBNHS

import torch
from torch import nn
from timm.models.layers import trunc_normal_
from layers import LayerNorm, PositionalEncodingFourier
from sdta_encoder import SDTAEncoder
from conv_encoder import ConvEncoder
from timm.models.registry import register_model


class EdgeNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[24, 48, 88, 168],
                 global_block=[0, 1, 1, 1], global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
                 drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1., expan_ratio=4,
                 kernel_sizes=[3, 5, 7, 9], heads=[8, 8, 8, 8], use_pos_embd_xca=[False, True, False, False],
                 use_pos_embd_global=False, d2_scales=[2, 2, 3, 4], act_layer=nn.GELU, **kwargs):
        super().__init__()
        for g in global_block_type:
            assert g in ['None', 'SDTA']
        if use_pos_embd_global:
            self.pos_embd = PositionalEncodingFourier(dim=dims[0])
        else:
            self.pos_embd = None
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                if j > depths[i] - global_block[i] - 1:
                    if global_block_type[i] == 'SDTA':
                        stage_blocks.append(SDTAEncoder(dim=dims[i], act_layer=act_layer, drop_path=dp_rates[cur + j],
                                                        expan_ratio=expan_ratio, scales=d2_scales[i],
                                                        use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i]))
                    else:
                        raise NotImplementedError
                else:
                    stage_blocks.append(ConvEncoder(dim=dims[i], act_layer=act_layer, drop_path=dp_rates[cur + j],
                                                    layer_scale_init_value=layer_scale_init_value,
                                                    expan_ratio=expan_ratio, kernel_size=kernel_sizes[i]))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # Final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head_dropout = nn.Dropout(0.0)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):  # TODO: MobileViT is using 'kaiming_normal' for initializing conv layers
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        if self.pos_embd:
            B, C, H, W = x.shape
            x = x + self.pos_embd(B, H, W)
        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        return self.norm(x.mean([-2, -1]))  # Global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(self.head_dropout(x))
        return x


"""
-- Main Models
    XX-Small -> 1.3M
    X-Small -> 2.3M
    Small -> 5.6M
"""


@register_model
def edgenext_xx_small(pretrained=False, **kwargs):
    # 1.33M & 260.58M @ 256 resolution
    # 71.23% Top-1 accuracy
    # No AA, Color Jitter=0.4, No Mixup & Cutmix, DropPath=0.0, BS=4096, lr=0.006, multi-scale-sampler
    # Jetson FPS=51.66 versus 47.67 for MobileViT_XXS
    # For A100: FPS @ BS=1: 212.13 & @ BS=256: 7042.06 versus
    #           FPS @ BS=1:  96.68 & @ BS=256: 4624.71 for MobileViT_XXS
    model = EdgeNeXt(depths=[2, 2, 6, 2], dims=[24, 48, 88, 168],
                     heads=[4, 4, 4, 4], **kwargs)

    return model


@register_model
def edgenext_x_small(pretrained=False, **kwargs):
    # 2.34M & 538.0M @ 256 resolution
    # 75.00% Top-1 accuracy
    # No AA, No Mixup & Cutmix, DropPath=0.0, BS=4096, lr=0.006, multi-scale-sampler
    # Jetson FPS=31.61 versus 28.49 for MobileViT_XS
    # For A100: FPS @ BS=1: 179.55 & @ BS=256: 4404.95 versus
    #           FPS @ BS=1:  94.55 & @ BS=256: 2361.53 for MobileViT_XS
    model = EdgeNeXt(depths=[3, 3, 9, 3], dims=[32, 64, 100, 192],
                     heads=[4, 4, 4, 4], **kwargs)

    return model


@register_model
def edgenext_small(pretrained=False, **kwargs):
    # 5.59M & 1260.59M @ 256 resolution
    # 79.43% Top-1 accuracy
    # AA=True, No Mixup & Cutmix, DropPath=0.1, BS=4096, lr=0.006, multi-scale-sampler
    # Jetson FPS=20.47 versus 18.86 for MobileViT_S
    # For A100: FPS @ BS=1: 172.33 & @ BS=256: 3010.25 versus
    #           FPS @ BS=1:  93.84 & @ BS=256: 1785.92 for MobileViT_S
    model = EdgeNeXt(depths=[3, 3, 9, 3], dims=[48, 96, 160, 304], **kwargs)

    return model


@register_model
def edgenext_base(pretrained=False, **kwargs):
    # 18.51M & 3840.93M @ 256 resolution
    # 82.5% (normal) 83.7% (USI) Top-1 accuracy
    # AA=True, Mixup & Cutmix, DropPath=0.1, BS=4096, lr=0.006, multi-scale-sampler
    model = EdgeNeXt(depths=[3, 3, 9, 3], dims=[80, 160, 288, 584], **kwargs)

    return model


"""
    Using BN & HSwish instead of LN & GeLU
"""


@register_model
def edgenext_xx_small_bn_hs(pretrained=False, **kwargs):
    # 1.33M & 259.53M @ 256 resolution
    # 70.33% Top-1 accuracy
    # For A100: FPS @ BS=1: 219.66 & @ BS=256: 10359.98
    model = EdgeNeXtBNHS(depths=[2, 2, 6, 2], dims=[24, 48, 88, 168], expan_ratio=4,
                         global_block=[0, 1, 1, 1],
                         global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS', 'SDTA_BN_HS'],
                         use_pos_embd_xca=[False, True, False, False],
                         kernel_sizes=[3, 5, 7, 9],
                         heads=[4, 4, 4, 4],
                         d2_scales=[2, 2, 3, 4],
                         **kwargs)

    return model


@register_model
def edgenext_x_small_bn_hs(pretrained=False, **kwargs):
    # 2.34M & 535.84M @ 256 resolution
    # 74.87% Top-1 accuracy
    # For A100: FPS @ BS=1: 179.25 & @ BS=256: 6059.59
    model = EdgeNeXtBNHS(depths=[3, 3, 9, 3], dims=[32, 64, 100, 192], expan_ratio=4,
                         global_block=[0, 1, 1, 1],
                         global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS', 'SDTA_BN_HS'],
                         use_pos_embd_xca=[False, True, False, False],
                         kernel_sizes=[3, 5, 7, 9],
                         heads=[4, 4, 4, 4],
                         d2_scales=[2, 2, 3, 4],
                         **kwargs)

    return model


@register_model
def edgenext_small_bn_hs(pretrained=False, **kwargs):
    # 5.58M & 1257.28M @ 256 resolution
    # 78.39% Top-1 accuracy
    # For A100: FPS @ BS=1: 174.68 & @ BS=256: 3808.19
    model = EdgeNeXtBNHS(depths=[3, 3, 9, 3], dims=[48, 96, 160, 304], expan_ratio=4,
                         global_block=[0, 1, 1, 1],
                         global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS', 'SDTA_BN_HS'],
                         use_pos_embd_xca=[False, True, False, False],
                         kernel_sizes=[3, 5, 7, 9],
                         d2_scales=[2, 2, 3, 4],
                         **kwargs)

    return model
