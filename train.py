import argparse
import time

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from src.consts import get_alpha
from src.datamodule import HAM10000DataModule
from src.model import MobileNetLightningModel
from src.save_setup import save_setup


def main() -> None:
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-m",
        "--minified",
        help="only applicable in simulated mode, run with minified dataset",
        action=argparse.BooleanOptionalAction,
    )
    argParser.add_argument("-a", "--alpha", help="alpha parameter of focal loss", type=str)
    argParser.add_argument(
        "-g", "--gamma", help="gamma parameter of focal loss", type=float, default=2
    )
    argParser.add_argument("-e", "--epochs", help="number of epochs", type=int, required=True)

    args = argParser.parse_args()
    alpha = get_alpha(args.alpha)

    model = MobileNetLightningModel(alpha=alpha, gamma=args.gamma)
    datamodule = HAM10000DataModule(minified=args.minified)

    start_time = time.time()
    callbacks = [ModelCheckpoint(save_top_k=-1, mode="max", monitor="val_acc")]
    logs_path = "classic-models/"
    logger = loggers.TensorBoardLogger(save_dir=logs_path, name="")
    save_setup(f"{logs_path}version_{logger.version}")

    trainer = pl.Trainer(max_epochs=args.epochs, callbacks=callbacks, logger=logger)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

    runtime = (time.time() - start_time) / 60
    print(f"Training took {runtime:.2f} min in total.")


if __name__ == "__main__":
    main()
