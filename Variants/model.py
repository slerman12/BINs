import math
from tonic.torch import models, normalizers
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules import Module
from torch.nn import functional as F


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


class MLB(torch.nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes

    def initialize(self, input_size):
        sizes = [input_size] + list(self.sizes)
        layers = []
        for i in range(len(sizes) - 1):
            layers += [BIN(sizes[i], sizes[i + 1], not i, is_last_layer=i == len(sizes) - 2)]
        self.model = torch.nn.Sequential(*layers)
        return sizes[-1]

    def forward(self, inputs):
        # print(self.model(inputs))
        return self.model(inputs)


class BIN(Module):
    r"""Applies a BIN transformation to the incoming data:

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weights: the learnable incoming weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = BIN(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weights: Tensor
    activation_potential: Tensor

    def __init__(self, in_features: int, out_features: int, is_embed_layer: bool, is_last_layer: bool) -> None:
        super(BIN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_embed_layer = is_embed_layer
        self.is_last_layer = is_last_layer
        self.weights = Parameter(torch.Tensor(out_features, in_features))
        self.activation_potential = Parameter(torch.Tensor(out_features))
        self.neurotransmitters = torch.ones(out_features)
        self.neurotransmitters[self.out_features // 2:] = -self.neurotransmitters[self.out_features // 2:]
        self.reset_parameters()
        self.logger = {"neurotransmitters": self.neurotransmitters}

        def hook(module, grad_input, grad_output):
            self.logger.setdefault("grads", []).append(grad_output[0])

        self.register_backward_hook(hook)

    def compile_logs(self):
        index = len(self.logger["activations"]) - 1
        for log in ["activations", "synapses", "inputs", "grads"]:
            self.logger[log] = self.logger[log][index][min(len(self.logger[log][index]), 10):]
        return self.logger

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def forward(self, input: Tensor) -> Tensor:
        act_potential = F.linear(input, self.weights if self.is_embed_layer else torch.sigmoid(self.weights))
        activations = torch.sigmoid(act_potential)
        synapses = activations * self.neurotransmitters
        self.logger["activations"] = activations[0][0].data.numpy()
        self.logger["synapses"] = synapses[0][0].data.numpy()
        # todo is dendrites indexed correctly?
        self.logger["inputs"] = (input[0][:10] * torch.sigmoid(self.weights[0][0][:10])).data.numpy()
        return synapses

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )