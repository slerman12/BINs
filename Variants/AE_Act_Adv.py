import math
from tonic.torch import models, normalizers
import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules import Module
from torch.nn import functional as F
from Variants.model import MLB as ModelMLB, BIN as ModelBIN


def actor_critic_bin():
    return models.ActorCritic(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=MLB((64, 64)),
            head=models.DetachedScaleGaussianPolicyHead()),
        critic=models.Critic(
            encoder=models.ObservationEncoder(),
            torso=models.MLP((64, 64), torch.nn.Tanh),
            head=models.ValueHead()),
        observation_normalizer=normalizers.MeanStd())


class MLB(ModelMLB):
    def initialize(self, input_size):
        sizes = [input_size] + list(self.sizes)
        layers = []
        for i in range(len(sizes) - 1):
            layers += [BIN(sizes[i], sizes[i + 1], not i, is_last_layer=i == len(sizes) - 2)]
        self.model = torch.nn.Sequential(*layers)
        return sizes[-1]


class Gate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, threshold=0.001):
        output = input.copy()
        output[input < threshold] = 0
        output[input >= threshold] = 1
        return output


class BIN(ModelBIN):
    def __init__(self, in_features: int, out_features: int, is_embed_layer: bool, is_last_layer: bool) -> None:
        super(BIN, self).__init__(in_features, out_features, is_embed_layer, is_last_layer)
        self.gate = Gate()
        if self.is_embed_layer:
            self.linear = torch.nn.Linear(in_features, out_features)

    @property
    def dendrites(self):
        return torch.sigmoid(self.weights).detach()

    def forward(self, input: Tensor) -> Tensor:
        if self.is_embed_layer:
            return torch.sigmoid(self.linear(input))
        else:
            signal = F.linear(input, self.weights  else self.dendrites)
            activations = self.gate(signal)
            synapses = activations * self.neurotransmitters
            self.logger["activations"] = activations[0][0].data.numpy()
            self.logger["synapses"] = synapses[0][0].data.numpy()
            # todo is dendrites indexed correctly?
            self.logger["inputs"] = (input[0][:10] * torch.sigmoid(self.dendrites[0][0][:10])).data.numpy()
            return synapses
