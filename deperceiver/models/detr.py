from typing import Dict, Any, Union, List
import sys
import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from datamodules import get_coco_api_from_dataset
from metrics.coco_eval import CocoEvaluator
from util.misc import NestedTensor, nested_tensor_from_tensor_list, interpolate

from .postprocess import PostProcess

DETRInput = Union[List, torch.Tensor, NestedTensor]


class DETR(pl.LightningModule):

    def __init__(
        self,
        backbone: nn.Module,
        transformer: nn.Module,
        criterion: nn.Module,
        num_classes: int,
        num_queries: int,
        args: Any,
        aux_loss: bool = False,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.args = args
        self.criterion  = criterion
        self.postprocess = PostProcess()

    def forward(self, samples: DETRInput) -> Dict[str, Any]:
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None

        projected_src = self.input_proj(src)
        hs = self.transformer(projected_src, mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # Make torchscript happy
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def _step(self, batch, batch_idx):
        samples, targets = batch
        outputs = self(samples)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_unscaled = {f'{k}_unscaled': v for k, v in loss_dict.items()}
        loss_dict_scaled = {k: v * weight_dict[k] for k, v in loss_dict.items()}
        losses_scaled = sum(loss_dict_scaled.values())

        loss_value = losses_scaled.item()

        self.log_dict(loss_dict_unscaled)
        self.log_dict(loss_dict_scaled)
        self.log('loss', loss_value)

        return losses, loss_value, outputs

    def training_step(self, batch, batch_idx):
        losses, loss_value, _ = self._step(batch, batch_idx)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        return losses

    def on_validation_epoch_start(self) -> None:
        base_ds = get_coco_api_from_dataset(self.trainer.datamodule.dataset_val)
        self.evaluator = CocoEvaluator(base_ds, ('bbox'))

    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        losses, loss_value, outputs = self._step(batch, batch_idx)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = self.postprocess(outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        self.evaluator.update(res)

    def on_validation_epoch_end(self) -> None:
        self.evaluator.synchronize_between_processes()
        if not self.global_rank == 0:
            return

        self.evaluator.accumulate()
        stats = self.evaluator.summarize()

        self.log('ap', stats['bbox'][0])
        self.log('ap50', stats['bbox'][1])
        self.log('ap75', stats['bbox'][2])
        self.log('ap_s', stats['bbox'][3])
        self.log('ap_m', stats['bbox'][4])
        self.log('ap_l', stats['bbox'][5])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.lr_drop, verbose=True)

        return [self.optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]


class MLP(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x