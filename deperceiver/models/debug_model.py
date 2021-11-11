import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchmetrics import Accuracy
import math

from deperceiver.models.perceiver_io import *
from deperceiver.models.perceiver_io.positional_encoding import fourier_encoding


def _apply_op(
    img, op_name: str, magnitude: float, interpolation: transforms.InterpolationMode, fill
):
    if op_name == "ShearX":
        img = transforms.functional.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(magnitude), 0.0],
            interpolation=interpolation,
            fill=fill,
        )
    elif op_name == "ShearY":
        img = transforms.functional.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(magnitude)],
            interpolation=interpolation,
            fill=fill,
        )
    elif op_name == "TranslateX":
        img = transforms.functional.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = transforms.functional.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = transforms.functional.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = transforms.functional.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = transforms.functional.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = transforms.functional.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = transforms.functional.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = transforms.functional.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = transforms.functional.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = transforms.functional.autocontrast(img)
    elif op_name == "Equalize":
        img = transforms.functional.equalize(img)
    elif op_name == "Invert":
        img = transforms.functional.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img

class RandAugment(torch.nn.Module):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: transforms.InterpolationMode = transforms.InterpolationMode.NEAREST,
        fill = None,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins, image_size):
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def forward(self, img):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * transforms.functional.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        for _ in range(self.num_ops):
            op_meta = self._augmentation_space(self.num_magnitude_bins, transforms.functional._get_image_size(img))
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img


    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_ops={num_ops}"
        s += ", magnitude={magnitude}"
        s += ", num_magnitude_bins={num_magnitude_bins}"
        s += ", interpolation={interpolation}"
        s += ", fill={fill}"
        s += ")"
        return s.format(**self.__dict__)

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
            RandAugment(),
            transforms.ToTensor(),
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
