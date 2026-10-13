"""Quantized feature pyramid neck."""
from __future__ import annotations
from torch import nn
from mmengine.model import BaseModule
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from .quantization import MultiSpikeNorm

@MODELS.register_module()
class QFPN(BaseModule):
    def __init__(self, in_channels, out_channels, num_outs=4, start_level=0,
                 end_level=-1, T=4, upsample_cfg=None,
                 init_cfg=dict(type="Xavier", layer="Conv2d", distribution="uniform"), **kwargs):
        super().__init__(init_cfg)
        del kwargs
        self.in_channels, self.out_channels = list(in_channels), out_channels
        self.start_level = start_level
        self.end_level = len(in_channels) if end_level == -1 else end_level
        self.num_outs = num_outs
        if num_outs != self.end_level - start_level:
            raise ValueError("QFPN requires one output for every selected input level")
        self.upsample_cfg = dict(mode="nearest") if upsample_cfg is None else dict(upsample_cfg)
        selected = self.in_channels[start_level:self.end_level]
        self.lateral_spikes = nn.ModuleList([MultiSpikeNorm(T) for _ in selected])
        self.lateral_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(c, out_channels, 1),
                                            nn.BatchNorm2d(out_channels)) for c in selected])
        self.output_spikes = nn.ModuleList([MultiSpikeNorm(T) for _ in selected])
        self.output_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                           nn.BatchNorm2d(out_channels)) for _ in selected])

    def forward(self, inputs):
        if len(inputs) != len(self.in_channels):
            raise ValueError("input feature count does not match in_channels")
        laterals = [conv(spike(inputs[index + self.start_level]))
                    for index, (spike, conv) in enumerate(zip(self.lateral_spikes, self.lateral_convs))]
        for index in range(len(laterals) - 1, 0, -1):
            laterals[index - 1] = laterals[index - 1] + resize(
                laterals[index], size=laterals[index - 1].shape[2:], **self.upsample_cfg)
        return tuple(conv(spike(value)) for value, spike, conv in
                     zip(laterals, self.output_spikes, self.output_convs))
