from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from util.misc import NestedTensor

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.

    Copy-pasted a second time from detr.models.backbone
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(pl.LightningModule):

    def __init__(
        self,
        backbone: nn.Module,
        num_channels: int,
        train_backbone: bool = True,
        return_interm_layers: bool = False,
    ) -> None:
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor) -> NestedTensor:
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):

    def __init__(
        self,
        model_name: str,
        train_backbone: bool = True,
        return_interm_layers: bool = False,
        dilation: bool = False,
    ) -> None:
        backbone = getattr(torchvision.models, model_name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True, norm_layer=FrozenBatchNorm2d
        )
        num_channels = 512 if model_name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, num_channels, train_backbone=train_backbone, return_interm_layers=return_interm_layers)


class Joiner(pl.LightningModule):

    def __init__(
        self,
        backbone: nn.Module,
        position_embedding: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_channels = backbone.num_channels
        self.position_embedding = position_embedding

    def forward(self, tensor_list: NestedTensor) -> Tuple[List[NestedTensor], List[torch.Tensor]]:
        xs = self.backbone(tensor_list)
        out: List[NestedTensor] = []
        pos: List[torch.Tensor] = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self.position_embedding(x).to(x.tensors.dtype))

        return out, pos


# TODO: replace this with Hydra
def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    return model
