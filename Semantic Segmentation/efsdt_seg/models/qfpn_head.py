"""Quantized semantic FPN decode head."""
from __future__ import annotations
import math
from torch import nn
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import Upsample, resize
from mmseg.registry import MODELS
from .quantization import MultiSpikeNorm

@MODELS.register_module()
class QFPNHead(BaseDecodeHead):
    def __init__(self, feature_strides, T=4, **kwargs):
        super().__init__(input_transform="multiple_select", **kwargs)
        if len(feature_strides) != len(self.in_channels) or min(feature_strides) != feature_strides[0]:
            raise ValueError("feature_strides must align with inputs and start at the smallest stride")
        self.feature_strides = feature_strides
        self.output_spike = MultiSpikeNorm(T)
        self.scale_heads = nn.ModuleList()
        for stride in feature_strides:
            modules = []
            steps = max(1, int(math.log2(stride) - math.log2(feature_strides[0])))
            for _ in range(steps):
                modules.extend([MultiSpikeNorm(T), nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=False),
                                nn.BatchNorm2d(self.channels)])
                if stride != feature_strides[0]:
                    modules.append(Upsample(scale_factor=2, mode="bilinear", align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*modules))

    def forward(self, inputs):
        features = self._transform_inputs(inputs)
        output = self.scale_heads[0](features[0])
        for index in range(1, len(self.feature_strides)):
            output = output + resize(self.scale_heads[index](features[index]), size=output.shape[2:],
                                     mode="bilinear", align_corners=self.align_corners)
        return self.cls_seg(self.output_spike(output))
