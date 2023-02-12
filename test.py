import argparse

import pytorch_lightning as pl

from src.datamodule import HAM10000DataModule
from src.get_model import get_model


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-p", "--path", help="path of the stored model", type=str, required=True
    )
    arg_parser.add_argument(
        "-f",
        "--federated",
        help="use federated storage format",
        action=argparse.BooleanOptionalAction,
    )
    args = arg_parser.parse_args()

    model = get_model(federated=args.federated, path=args.path)

    # {'test_acc': 0.8534688949584961, 'test_loss': 0.2627546191215515}
    # model = MobileNetLightningModel.load_from_checkpoint("lightning_logs/version_32/checkpoints/epoch=23-step=5015.ckpt")
    # {'test_acc': 0.8241626620292664, 'test_loss': 0.28230786323547363}
    # model = MobileNetLightningModel.load_from_checkpoint("lightning_logs/version_32/checkpoints/epoch=49-step=10449.ckpt")
    # {'test_acc': 0.8355262875556946, 'test_loss': 0.3977561891078949}

    # disable randomness, dropout, etc...
    model.eval()

    datamodule = HAM10000DataModule()
    trainer = pl.Trainer(precision="bf16")
    trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
