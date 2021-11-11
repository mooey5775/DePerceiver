import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchmetrics import Accuracy

from deperceiver.models.perceiver_io import *
from deperceiver.models.perceiver_io.positional_encoding import fourier_encoding


class DebugModel(pl.LightningModule):
    num_latents = 128
    latent_dim = 256
    num_classes = 10
    num_bands = 10
    input_dim = num_bands * 2 * 2 + 2 + 3

    def __init__(self):
        super().__init__()

        self.position_encoding = fourier_encoding(
            dims=(32, 32),
            num_bands=10,
            resolutions=(32, 32),
        )

        self.encoder = PerceiverEncoder(
            num_latents=self.num_latents,
            latent_dim=self.latent_dim,
            input_dim=self.input_dim,
            num_self_attn_per_block=6,
        )

        self.decoder = ClassificationDecoder(
            num_classes=self.num_classes,
            latent_dim=self.latent_dim,
        )

        self.perceiver = PerceiverIO(self.encoder, self.decoder)

        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandAugment(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])        

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])   

        self.train_dataset = CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=train_transforms,
        )

        self.test_dataset = CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=test_transforms,
        )

        self.accuracy = Accuracy()
        self.train_accuracy = Accuracy()

    def forward(self, x):
        batch_size = x.shape[0]

        pos_enc = self.position_encoding.to(self.device)

        pos_enc = pos_enc.repeat(batch_size, 1, 1, 1)
        pos_enc = pos_enc.permute(0, 3, 1, 2)

        x = torch.cat([x, pos_enc], dim=1)
        x = x.permute(0, 2, 3, 1).view(batch_size, -1, self.input_dim)
        x = self.perceiver(x)

        return x

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        outputs = self(imgs)
        loss = F.cross_entropy(outputs, labels)
        accuracy = self.train_accuracy(outputs, labels)

        self.log(f'train/loss', loss)
        self.log(f'train/accuracy', accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        outputs = self(imgs)
        loss = F.cross_entropy(outputs, labels)
        accuracy = self.accuracy(outputs, labels)

        self.log(f'val/loss', loss)
        self.log(f'val/accuracy', accuracy)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.1)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=4,
        )
