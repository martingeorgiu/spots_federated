import pytorch_lightning as pl
from src.model import MobileNetLightningModel
from src.datamodule import HAM10000DataModule
from pytorch_lightning.callbacks import ModelCheckpoint
import time

def main() -> None:

    model = MobileNetLightningModel()
    datamodule = HAM10000DataModule()

    start_time = time.time()
    callbacks = [ModelCheckpoint(save_top_k=-1, mode='max', monitor="val_acc")]
    trainer = pl.Trainer( precision='bf16', max_epochs=10,callbacks=callbacks,log_every_n_steps=10)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

    runtime = (time.time() - start_time)/60
    print(f"Training took {runtime:.2f} min in total.")


if __name__ == '__main__':    
    main()