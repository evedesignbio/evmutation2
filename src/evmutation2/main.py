"""
Code related to training the model

Useful parameters/functions:
    model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")

    # automatically restores model, epoch, step, LR schedulers, etc...
    trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")

    trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
    # saves checkpoints to 'some/path/' at every epoch end
    # trainer = Trainer(default_root_dir="some/path/")
    # Trainer(precision="16-mixed")  # todo: bfloat16
    # trainer.fit(model=model, train_dataloaders=train_loader, valid_loader)
    # trainer = Trainer(enable_checkpointing=False)
    # trainer = Trainer(accelerator="gpu", devices="auto")
    # trainer.fit(model, datamodule=imagenet)
"""
from lightning.pytorch.cli import LightningCLI
from evmutation2.model import Model
from evmutation2.data import SeqStructDataModule


def cli_main():
    cli = LightningCLI(Model, SeqStructDataModule)


if __name__ == "__main__":
    cli_main()
