import argparse

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from deperceiver.models.backbone import build_backbone
from deperceiver.models.transformer import build_transformer
from deperceiver.models.detr import DETR
from deperceiver.losses.matcher import build_matcher
from deperceiver.losses.set_criterion import SetCriterion
from deperceiver.datamodules.coco_datamodule import CocoDataModule

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # Trainer params
    parser.add_argument('--gpus', default=10, type=int, help='number of gpus')
    parser.add_argument('--run_name', default=None, type=str, help='name of the run')
    parser.add_argument('--amp', action='store_true', help='use amp for mixed precision training')
    parser.add_argument('--use_bfloat', action='store_true', help='use bfloat16 for mixed precision training')

    return parser

def main(args):
    seed_everything(args.seed)

    # Build our model
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    
    criterion = SetCriterion(91, matcher=matcher, weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses)

    model = DETR(
        backbone,
        transformer,
        criterion,
        num_classes=91,
        num_queries=args.num_queries,
        args=args,
        aux_loss=args.aux_loss,
    )

    datamodule = CocoDataModule(args)

    lr_monitor = LearningRateMonitor()
    wandb_logger = WandbLogger(
        name=args.run_name,
        project='DePerceiver',
        log_model=True,
        entity='deperceiver',
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val/ap',
        mode='max',
        save_top_k=3,
    )

    precision = 32
    if args.amp:
        precision = 'bf16' if args.use_bfloat else 16

    trainer = Trainer(
        gpus=args.gpus,
        accelerator='ddp',
        plugins=[DDPPlugin(find_unused_parameters=False)],
        precision=precision,
        default_root_dir=args.output_dir,
        gradient_clip_val=args.clip_max_norm,
        max_epochs=args.epochs,
        logger=wandb_logger,
        replace_sampler_ddp=False,
        callbacks=[lr_monitor, checkpoint_callback],
    )
    wandb_logger.watch(model)
    
    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
