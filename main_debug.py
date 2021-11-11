import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from deperceiver.models.debug_model import DebugModel

if __name__ == '__main__':
    model = DebugModel()
    wandb_logger = WandbLogger(
        name='debug_cifar',
        project='DePerceiver',
        log_model=True,
        entity='deperceiver',
    )

    trainer = Trainer(
        gpus=4,
        accelerator='ddp',
        logger=wandb_logger,
        gradient_clip_val=10,
    )
    wandb_logger.watch(model)

    trainer.fit(model)