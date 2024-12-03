import torch.nn as nn
import torch
from .utils import get_weights_url
import math


class DnCNN(nn.Module):
    r"""
    DnCNN convolutional denoiser.

    The architecture was introduced by Zhang et al. in https://arxiv.org/abs/1608.03981 and is composed of a series of
    convolutional layers with ReLU activation functions. The number of layers can be specified by the user. Unlike the
    original paper, this implementation does not include batch normalization layers.

    The network can be initialized with pretrained weights, which can be downloaded from an online repository. The
    pretrained weights are trained with the default parameters of the network, i.e. 20 layers, 64 channels and biases.

    :param int in_channels: input image channels
    :param int out_channels: output image channels
    :param int depth: number of convolutional layers
    :param bool bias: use bias in the convolutional layers
    :param int nf: number of channels per convolutional layer
    :param str, None pretrained: use a pretrained network. If ``pretrained=None``, the weights will be initialized at random
        using Pytorch's default initialization. If ``pretrained='download'``, the weights will be downloaded from an
        online repository (only available for architecture with depth 20, 64 channels and biases).
        It is possible to download weights trained via the regularization method in https://epubs.siam.org/doi/abs/10.1137/20M1387961
        using ``pretrained='download_lipschitz'``.
        Finally, ``pretrained`` can also be set as a path to the user's own pretrained weights.
        See :ref:`pretrained-weights <pretrained-weights>` for more details.
    :param bool train: training or testing mode
    :param str device: gpu or cpu
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        depth=20,
        bias=False,
        nf=64,
        pretrained="download",
        train=False,
        device="cpu",
        residual=True,
        last_act=None,
        log=False,
    ):
        super(DnCNN, self).__init__()

        self.depth = depth
        self.residual = residual

        self.in_conv = nn.Conv2d(
            in_channels, nf, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.conv_list = nn.ModuleList(
            [
                nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=bias)
                for _ in range(self.depth - 2)
            ]
        )
        self.out_conv = nn.Conv2d(
            nf, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

        self.nl_list = nn.ModuleList([nn.ReLU() for _ in range(self.depth - 1)])

        # if pretrain and ckpt_path is not None:
        #    self.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage), strict=True)
        if last_act == "relu":
            self.fn_act = nn.ReLU()
        elif last_act == "softplus":
            self.fn_act = nn.Softplus()
        elif last_act == "tanh":
            self.fn_act = nn.Tanh()
        elif last_act is None:
            self.fn_act = None

        else:
            raise ValueError("last_act must be None or 'relu'")

        if pretrained is not None:
            if pretrained.startswith("download"):
                name = ""
                if bias and depth == 20:
                    if pretrained == "download_lipschitz":
                        if in_channels == 3 and out_channels == 3:
                            name = "dncnn_sigma2_lipschitz_color.pth"
                        elif in_channels == 1 and out_channels == 1:
                            name = "dncnn_sigma2_lipschitz_gray.pth"
                    else:
                        if in_channels == 3 and out_channels == 3:
                            name = "dncnn_sigma2_color.pth"
                        elif in_channels == 1 and out_channels == 1:
                            name = "dncnn_sigma2_gray.pth"

                if name == "":
                    raise Exception(
                        "No pretrained weights were found online that match the chosen architecture"
                    )
                url = get_weights_url(model_name="dncnn", file_name=name)
                ckpt = torch.hub.load_state_dict_from_url(
                    url, map_location=lambda storage, loc: storage, file_name=name
                )
            else:
                ckpt = torch.load(pretrained, map_location=lambda storage, loc: storage)
            self.load_state_dict(ckpt, strict=True)

        if not train:
            self.eval()
            for _, v in self.named_parameters():
                v.requires_grad = False
        else:
            self.apply(weights_init_kaiming)

        if device is not None:
            self.to(device)

        self.log = log

    def forward(self, x, sigma=None):
        r"""
        Run the denoiser on noisy image. The noise level is not used in this denoiser.

        :param torch.Tensor x: noisy image
        :param float sigma: noise level (not used)
        """

        if self.log:
            x = torch.log(x)

        # if isinstance(sigma, torch.Tensor):
        #     if sigma.ndim > 0:
        #         noise_level_map = sigma.view(x.size(0), 1, 1, 1)
        #         noise_level_map = noise_level_map.expand(-1, 1, x.size(2), x.size(3))
        #     else:
        #         noise_level_map = torch.ones(
        #             (x.size(0), 1, x.size(2), x.size(3)), device=x.device
        #         ) * sigma[None, None, None, None].to(x.device)
            
        # else:
        #     noise_level_map = (
        #         torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device)
        #      )
        # x = torch.cat((x, noise_level_map), 1)

        x1 = self.in_conv(x)
        x1 = self.nl_list[0](x1)

        for i in range(self.depth - 2):
            x_l = self.conv_list[i](x1)
            x1 = self.nl_list[i + 1](x_l)

        out = self.out_conv(x1)
        
        if self.log:
            out = torch.exp(out)

        if self.fn_act is not None:
            out = self.fn_act(out)
        
        if self.residual:
            out = out + x
        # if self.log:
        #     out = torch.exp(out) - 1

        return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2.0 / 9.0 / 64.0)).clamp_(
            -0.025, 0.025
        )
        nn.init.constant(m.bias.data, 0.0)
