import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.datamodule import HAM10000DataModule
from src.model import MobileNetLightningModel


def main() -> None:
    model = MobileNetLightningModel()
    datamodule = HAM10000DataModule()

    start_time = time.time()
    callbacks = [ModelCheckpoint(save_top_k=-1, mode="max", monitor="val_acc")]
    trainer = pl.Trainer(precision="bf16", max_epochs=50, callbacks=callbacks)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

    runtime = (time.time() - start_time) / 60
    print(f"Training took {runtime:.2f} min in total.")


if __name__ == "__main__":
    main()
