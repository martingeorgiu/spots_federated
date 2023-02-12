import pytorch_lightning as pl
import torch
import torchmetrics
from torchvision import models

from src.consts import lesion_type_dict
from src.focal_loss import FocalLoss


# LightningModule that receives a PyTorch model as input
class LightningModel(pl.LightningModule):
    def __init__(self, model, num_classes, learning_rate: float):
        super().__init__()

        self.learning_rate = learning_rate
        # The inherited PyTorch module
        self.model = model

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=["model"])

        task = "binary" if num_classes == 1 else "multiclass"
        # Set up attributes for computing the accuracy
        self.train_acc = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task=task, num_classes=num_classes)

        self.loss_fn = FocalLoss(class_num=num_classes)

    # Defining the forward method is only necessary
    # if you want to use a Trainer's .predict() method (optional)
    def forward(self, x):
        return self.model(x)

    # A common forward step to compute the loss and labels
    # this is used for training, validation, and testing below
    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)
        loss = self.loss_fn(logits, true_labels)
        # loss = torch.nn.functional.cross_entropy(logits, true_labels, weight=torch.FloatTensor(weights))
        predicted_labels = torch.argmax(logits, dim=1)

        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss)

        self.train_acc(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=True)
        self.model.train()

        return loss  # this is passed to the optimzer for training

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("val_loss", loss)
        self.val_acc(predicted_labels, true_labels)
        self.log(
            "val_acc",
            self.val_acc,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("test_loss", loss)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class MobileNetLightningModel(LightningModel):
    input_size = 224

    def __init__(self, learning_rate: float = 0.001):
        no_of_classes = len(lesion_type_dict)
        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = torch.nn.Linear(num_ftrs, no_of_classes)
        super().__init__(model, no_of_classes, learning_rate)


class DenseNetLightningModel(LightningModel):
    input_size = 224

    def __init__(self, learning_rate: float = 0.001):
        no_of_classes = len(lesion_type_dict)
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, no_of_classes)
        super().__init__(model, no_of_classes, learning_rate)
