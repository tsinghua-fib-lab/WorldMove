from argparse import ArgumentParser

import pytorch_lightning as pl
import setproctitle
import yaml
import torch
from dataset.module import DataModule

from model import RegionDiff
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    pl.seed_everything(cfg["seed"], workers=True)
    # torch.set_num_threads(8)
    setproctitle.setproctitle(cfg["title"])

    model = RegionDiff(cfg)
    datamodule = DataModule(cfg)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=args.save, name="logs", version=cfg["version"]
    )
    ckpt_callback = ModelCheckpoint()
    callbacks = [lr_monitor, ckpt_callback]
    trainer = pl.Trainer(
        logger=tb_logger,
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        strategy=DDPStrategy(
            find_unused_parameters=False, gradient_as_bucket_view=True
        ),
        callbacks=callbacks,
        max_epochs=cfg["trainer"]["max_epochs"],
        check_val_every_n_epoch=cfg["trainer"]["val_interval"],
    )
    trainer.fit(model, datamodule, ckpt_path=cfg["trainer"]["ckpt_path"])


if __name__ == "__main__":
    main()
