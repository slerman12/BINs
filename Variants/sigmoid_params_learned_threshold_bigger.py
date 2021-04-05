import math
from tonic.torch import models, normalizers
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules import Module
from torch.nn import functional as F
from Variants.model import MLB as ModelMLB, BIN as ModelBIN


def actor_critic_bin():
    return models.ActorCritic(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=MLB((256, 256, 256)),
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


class BIN(ModelBIN):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weights: Tensor
    activation_potential: Tensor

    def __init__(self, in_features: int, out_features: int, is_embed_layer: bool, is_last_layer: bool) -> None:
        super(BIN, self).__init__(in_features, out_features, is_embed_layer, is_last_layer)
        self.activation_potential = Parameter(torch.Tensor(out_features))

    def forward(self, input: Tensor) -> Tensor:
        signal = F.linear(input, self.dendrites if self.is_embed_layer else torch.sigmoid(self.dendrites))
        activations = torch.sigmoid(signal - self.activation_potential)
        synapses = activations * self.neurotransmitters
        self.logger["activations"] = activations[0][0].data.numpy()
        self.logger["synapses"] = synapses[0][0].data.numpy()
        # todo is dendrites indexed correctly?
        self.logger["inputs"] = (input[0][:10] * torch.sigmoid(self.dendrites[0][0][:10])).data.numpy()
        return synapses