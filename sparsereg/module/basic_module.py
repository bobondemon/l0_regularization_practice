import copy

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

# [Debug]
from sparsereg.model.basic_l0_blocks import L0Gate


class CIFARModule(pl.LightningModule):
    def __init__(self, model, optimizer_name, optimizer_hparams, multi_stepLR_hparams, lambda_l0=1.0):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
            multi_stepLR_hparams - Hyperparameters for the `MultiStepLR`
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = copy.deepcopy(model)

        # [Debug]: monitoring MaskL0 parameters
        self.l0gate_moduels = None
        if not self.model.fix_and_open_gate:
            self.l0gate_moduels = [m for m in self.modules() if type(m) is L0Gate]

        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

        print(f"===== self.hparams.multi_stepLR_hparams={self.hparams.multi_stepLR_hparams}")

    def on_train_start(self) -> None:
        if not self.model.fix_and_open_gate:
            # which means we want to use l0_gating module for learning, so we need to reset the l0 parameters
            self.model.reset_l0_parameters()

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **self.hparams.multi_stepLR_hparams)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        reg = self.model.regularization()
        acc_loss = self.loss_module(preds, labels)
        loss = acc_loss + self.hparams.lambda_l0 * reg
        loss_dict = {"train_loss": loss, "acc_loss": acc_loss, "reg": reg}
        self.log_dict(loss_dict)

        # [Debug]: monitoring one MaskL0 module's parameters
        if self.l0gate_moduels is not None:
            log_l0_param_dict = {
                "l0_module0_param0": self.l0gate_moduels[0].qz_loga[0],
                "l0_module0_paramlast": self.l0gate_moduels[0].qz_loga[-1],
                "l0_module10_param0": self.l0gate_moduels[10].qz_loga[0],
                "l0_module10_paramlast": self.l0gate_moduels[10].qz_loga[-1],
            }
            self.log_dict(log_l0_param_dict)

        acc = (preds.argmax(dim=-1) == labels).float().mean()
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)

        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)

    def cal_sparsity(self):
        l0_param_num, full_param_num = self.model.cal_full_and_l0_param_num()
        return 1.0 - l0_param_num / full_param_num
