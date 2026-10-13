"""SDTv3 backbone extracted from the original MMSegmentation fork."""
from __future__ import annotations
from collections import OrderedDict
import torch
from torch import Tensor, nn
from mmengine.logging import print_log
from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader
from mmseg.registry import MODELS
from .attention import SegmentationAttention
from .quantization import MultiSpikeNorm

class SpikeSepConv(nn.Module):
    def __init__(self, dim: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = dim * expansion
        self.layers = nn.Sequential(
            MultiSpikeNorm(), nn.Conv2d(dim, hidden, 1, bias=False), nn.BatchNorm2d(hidden),
            MultiSpikeNorm(), nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False), nn.BatchNorm2d(hidden),
            MultiSpikeNorm(), nn.Conv2d(hidden, dim, 1, bias=False), nn.BatchNorm2d(dim))
    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)

class SpikeMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(MultiSpikeNorm(), nn.Conv2d(dim, hidden_dim, 1, bias=False),
                                    nn.BatchNorm2d(hidden_dim), MultiSpikeNorm(),
                                    nn.Conv2d(hidden_dim, dim, 1, bias=False), nn.BatchNorm2d(dim))
    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)

class ConvBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.conv = SpikeSepConv(dim)
        self.mlp = nn.Sequential(MultiSpikeNorm(), nn.Conv2d(dim, dim * mlp_ratio, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(dim * mlp_ratio), MultiSpikeNorm(),
                                 nn.Conv2d(dim * mlp_ratio, dim, 3, padding=1, bias=False), nn.BatchNorm2d(dim))
    def forward(self, inputs: Tensor) -> Tensor:
        features = inputs + self.conv(inputs)
        return features + self.mlp(features)

class AttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, mode: str,
                 shiftmax_levels: int, init_value: float = 1e-6) -> None:
        super().__init__()
        self.conv = SpikeSepConv(dim)
        self.attn = SegmentationAttention(dim, num_heads, 4, mode, shiftmax_levels)
        self.mlp = SpikeMLP(dim, dim * mlp_ratio)
        self.layer_scale1 = nn.Parameter(torch.full((dim,), init_value))
        self.layer_scale2 = nn.Parameter(torch.full((dim,), init_value))
        self.layer_scale3 = nn.Parameter(torch.full((dim,), init_value))
    @staticmethod
    def _scale(value: Tensor) -> Tensor:
        return value[None, :, None, None]
    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs + self.conv(inputs) * self._scale(self.layer_scale1)
        x = x + self.attn(x) * self._scale(self.layer_scale2)
        return x + self.mlp(x) * self._scale(self.layer_scale3)

class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, padding: int, quantize: bool) -> None:
        super().__init__()
        self.quantizer = MultiSpikeNorm() if quantize else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
    def forward(self, inputs: Tensor) -> Tensor:
        return self.norm(self.conv(self.quantizer(inputs)))

@MODELS.register_module()
class Spiking_vit_MetaFormerv2(BaseModule):
    """5M SDTv3 four-stage feature backbone.

    The historical class name is retained so existing configs can use the
    extracted plugin without model-type changes.
    """
    def __init__(self, img_size_h: int = 512, img_size_w: int = 512,
                 patch_size: int = 16, in_channels: int = 3, num_classes: int = 150,
                 embed_dim=(32, 64, 128, 192), num_heads: int = 8,
                 mlp_ratios: int = 4, qkv_bias: bool = False, depths: int = 8,
                 sr_ratios: int = 1, T: int = 4, decode_mode: str = "QTrick",
                 attention_mode: str = "ssa", attention_max_value: int = 16,
                 init_cfg=None, pretrained=None, **kwargs) -> None:
        del img_size_h, img_size_w, patch_size, num_classes, qkv_bias, sr_ratios, pretrained, kwargs
        super().__init__(init_cfg=init_cfg)
        if len(embed_dim) != 4:
            raise ValueError("embed_dim must contain four stage widths")
        if depths != 8:
            raise ValueError("this extracted 5M backbone expects depths=8 (6+2 attention blocks)")
        self.T, self.decode_mode = T, decode_mode
        self.attention_mode, self.attention_max_value = attention_mode, attention_max_value
        self.downsample1_1 = Downsample(in_channels, embed_dim[0] // 2, 7, 2, 3, False)
        self.ConvBlock1_1 = nn.ModuleList([ConvBlock(embed_dim[0] // 2, mlp_ratios)])
        self.downsample1_2 = Downsample(embed_dim[0] // 2, embed_dim[0], 3, 2, 1, True)
        self.ConvBlock1_2 = nn.ModuleList([ConvBlock(embed_dim[0], mlp_ratios)])
        self.downsample2 = Downsample(embed_dim[0], embed_dim[1], 3, 2, 1, True)
        self.ConvBlock2_1 = nn.ModuleList([ConvBlock(embed_dim[1], mlp_ratios)])
        self.ConvBlock2_2 = nn.ModuleList([ConvBlock(embed_dim[1], mlp_ratios)])
        self.downsample3 = Downsample(embed_dim[1], embed_dim[2], 3, 2, 1, True)
        self.block3 = nn.ModuleList([AttentionBlock(embed_dim[2], num_heads, mlp_ratios,
                                    attention_mode, attention_max_value) for _ in range(6)])
        self.downsample4 = Downsample(embed_dim[2], embed_dim[3], 3, 1, 1, True)
        self.block4 = nn.ModuleList([AttentionBlock(embed_dim[3], num_heads, mlp_ratios,
                                    attention_mode, attention_max_value) for _ in range(2)])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight); nn.init.zeros_(module.bias)

    def init_weights(self) -> None:
        if self.init_cfg is None:
            print_log(f"No pretrained weights for {self.__class__.__name__}; training from scratch")
            return
        checkpoint = CheckpointLoader.load_checkpoint(self.init_cfg["checkpoint"], map_location="cpu")
        source = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        state = OrderedDict()
        for key, value in source.items():
            key = key.removeprefix("backbone.")
            key = key.replace(".attn.q_conv.0.weight", ".attn.q_proj.weight")
            key = key.replace(".attn.k_conv.0.weight", ".attn.k_proj.weight")
            key = key.replace(".attn.v_conv.0.weight", ".attn.v_proj.weight")
            if key.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")) and value.ndim == 4:
                value = value.squeeze(-1).squeeze(-1)
            if not any(part in key for part in (".q_conv.1.", ".k_conv.1.", ".v_conv.1.")):
                state[key] = value
        incompatible = self.load_state_dict(state, strict=False)
        print_log(f"Loaded backbone with {len(incompatible.missing_keys)} missing and "
                  f"{len(incompatible.unexpected_keys)} unexpected keys")

    @staticmethod
    def _run(blocks: nn.ModuleList, inputs: Tensor) -> Tensor:
        for block in blocks:
            inputs = block(inputs)
        return inputs

    def forward(self, inputs: Tensor):
        x1 = self._run(self.ConvBlock1_1, self.downsample1_1(inputs))
        x2 = self._run(self.ConvBlock1_2, self.downsample1_2(x1))
        x3 = self._run(self.ConvBlock2_2, self._run(self.ConvBlock2_1, self.downsample2(x2)))
        x = self._run(self.block3, self.downsample3(x3))
        x4 = self._run(self.block4, self.downsample4(x))
        return [x1, x2, x3, x4] if self.decode_mode == "QTrick" else x4
