import argparse

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from deperceiver.models.naive_deperceiver import NaiveDePerceiver

from deperceiver.models.backbone import build_backbone
from deperceiver.models.perceiver_io import PerceiverDecoder, PerceiverEncoder, PerceiverIO, PerceiverMultipleDecoder
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

    # Logging params
    parser.add_argument('--no-wandb', dest='wandb', action='store_false',
                        help='disable wandb logging')
    parser.add_argument('--project', default='deperceiver', type=str,
                        help='wandb project name')

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
    parser.add_argument('--downsample_factor', default=32, type=int,
                        help="Factor to use for downsampling the image features in backbone (default 32)")
    parser.add_argument('--multiscale', action='store_true', help='If true, we use multiscale backbone')

    # * Transformer
    parser.add_argument('--num_latents', default=1024, type=int,
                        help="Number of latents to use")
    parser.add_argument('--enc_blocks', default=3, type=int,
                        help="Number of encoding blocks in the transformer")
    parser.add_argument('--enc_layers_per_block', default=6, type=int,
                        help="Number of layers per block in the transformer")
    parser.add_argument('--dec_layers', default=3, type=int,
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
    parser.add_argument('--resume_from_checkpoint', default=None, type=str,
                        help='resume from checkpoint. does not yet reload weights')

    return parser

def main(args):
    seed_everything(args.seed)

    # Build our model
    backbone = build_backbone(args)
    perceiver_encoder = PerceiverEncoder(
        num_latents=args.num_latents,
        latent_dim=256,
        input_dim=256,
        num_self_attn_per_block=args.enc_layers_per_block,
        num_blocks=args.enc_blocks,
        num_cross_attn_heads=1,
        num_self_attn_heads=8,
        cross_attn_widening_factor=1,
        self_attn_widening_factor=1,
    )

    if args.aux_loss:
        perceiver_decoder = PerceiverMultipleDecoder(
            latent_dim=256,
            query_dim=256,
            num_layers=args.dec_layers,
            return_intermediate=args.aux_loss,
        )
    else:
        perceiver_decoder = PerceiverDecoder(
            latent_dim=256,
            query_dim=256,
        )

    perceiver = PerceiverIO(
        perceiver_encoder, perceiver_decoder
    )

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    
    # Use DeTR set criterion
    criterion = SetCriterion(91, matcher=matcher, weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses)

    model = NaiveDePerceiver(
        backbone,
        perceiver,
        criterion,
        num_classes=91,
        num_queries=args.num_queries,
        args=args,
    )

    datamodule = CocoDataModule(args)

    # logging for wandb visualizations
    lr_monitor = LearningRateMonitor()
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.run_name,
            project=args.project,
            log_model=True,
            #entity='deperceiver',
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
        logger=wandb_logger if args.wandb else True,
        replace_sampler_ddp=False,
        callbacks=[lr_monitor, checkpoint_callback],
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    if args.wandb:
        wandb_logger.watch(model)
    
    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
