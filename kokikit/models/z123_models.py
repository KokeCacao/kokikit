# A diffuser version implementation of Zero1to3 (https://github.com/cvlab-columbia/zero123), ICCV 2023
# by Xin Kong

import torch
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin


class CCProjection(ModelMixin, ConfigMixin):

    def __init__(self, in_channel=772, out_channel=768):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.projection = torch.nn.Linear(in_channel, out_channel)

    def forward(self, x):
        return self.projection(x)
