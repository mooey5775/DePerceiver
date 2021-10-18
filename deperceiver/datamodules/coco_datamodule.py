import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, DistributedSampler

from . import build_dataset
from deperceiver.util.misc import collate_fn


class CocoDataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.dataset_train = build_dataset(image_set='train', args=args)
        self.dataset_val = build_dataset(image_set='val', args=args)
        # self.dataset_test = build_dataset(image_set='test', args=args)

    def train_dataloader(self):
        sampler_train = DistributedSampler(self.dataset_train)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train,
            self.args.batch_size,
            drop_last=True,
        )
        return DataLoader(
            self.dataset_train,
            batch_sampler=batch_sampler_train,
            collate_fn=collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        sampler_val = DistributedSampler(self.dataset_val)
        return DataLoader(
            self.dataset_val,
            self.args.batch_size,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    # def test_dataloader(self):
    #     sampler_test = DistributedSampler(self.dataset_test)
    #     return DataLoader(
    #         self.dataset_test,
    #         self.args.batch_size,
    #         sampler=sampler_test,
    #         drop_last=False,
    #         collate_fn=collate_fn,
    #         num_workers=self.args.num_workers,
    #     )