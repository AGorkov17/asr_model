from typing import Tuple

import hydra
import pytorch_lightning as pl
import torch
import thop
from torchmetrics import Accuracy


class KWS(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters({"model": conf.model._target_})

        self.conf = conf

        self.model = hydra.utils.instantiate(conf.model)
        self.train_acc = Accuracy(
            task="multiclass", num_classes=conf.model.n_classes, top_k=1
        )
        self.valid_acc = Accuracy(
            task="multiclass", num_classes=conf.model.n_classes, top_k=1
        )
        self.valid_class_acc = Accuracy(
            task="multiclass",
            num_classes=conf.model.n_classes,
            top_k=1,
            average=None,
        )

        self.loss = (
            hydra.utils.instantiate(conf.loss)
            if conf.get("loss") is not None
            else torch.nn.NLLLoss()
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.model(inputs)
        preds = scores.argmax(1)
        return scores, preds

    def on_train_start(self):

        features_params = self.conf.train_dataloader.dataset.transforms[0]

        sample_inputs = torch.randn(
            1,
            features_params.n_mels,
            features_params.sample_rate // features_params.hop_length + 1,
            device=self.device,
        )
        macs, params = thop.profile(
            self.model,
            inputs=(sample_inputs,),
        )
        print(f"Model parameters: {int(params):,}")
        print(f"Model MACs: {int(macs):,}")
        self.log("model/MACs", macs)
        self.log("model/Params", params)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        _, inputs, labels = batch

        scores, preds = self.forward(inputs)

        loss = self.loss(scores, labels)
        self.train_acc.update(preds, labels)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=labels.shape[0],
        )
        self.log(
            "train/accuracy",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "lr",
            self.optimizers().param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
        )

        return {"loss": loss}

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        _, inputs, labels = batch

        scores, preds = self.forward(inputs)

        loss = self.loss(scores, labels)
        self.valid_acc.update(preds, labels)
        self.valid_class_acc.update(preds, labels)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=labels.shape[0],
        )
        self.log(
            "val/accuracy",
            self.valid_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss}

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        ids, inputs, _ = batch

        _, preds = self.forward(inputs)

        return ids, preds

    def on_validation_epoch_end(self):
        class_accuracy = self.valid_class_acc.compute()
        for index, keyword in enumerate(
            self.conf.val_dataloader.dataset.idx_to_keyword
        ):
            self.log(f"val/recall_{keyword}", class_accuracy[index])
        self.valid_class_acc.reset()

    def train_dataloader(self):
        return hydra.utils.instantiate(self.conf.train_dataloader)

    def val_dataloader(self):
        return hydra.utils.instantiate(self.conf.val_dataloader)

    def predict_dataloader(self):
        return hydra.utils.instantiate(self.conf.predict_dataloader)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.conf.optim,
            params=self.model.parameters(),
        )
        if self.conf.get("scheduler") is None:
            return {"optimizer": optimizer}

        scheduler = hydra.utils.instantiate(
            self.conf.scheduler,
            optimizer=optimizer,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
