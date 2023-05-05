import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from model.modules import *

class ModelInterface(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.cnn = None
        self.query = None
        self.cross_attention = None
        self.self_attention = None # Add CLS token
        self.classify_head = nn.Linear()

        self.criterion = nn.CrossEntropyLoss()
        self.metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task='binary', num_classes=self.args.num_classes,
                                                                           average = 'micro')])
    
    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(dim=1)
        x = self.cross_attention(x, self.query)
        x = self.self_attention(x)
        x = self.classify_head(x[:, 0])
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        loss = self.criterion(logit, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        loss = self.criterion(logit, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'logit': logit, 'target': y}

    def validation_epoch_end(self, val_step_outputs):
        val_logits = torch.stack([x['logit'] for x in val_step_outputs], dim = 0)
        val_targets = torch.stack([x['target'] for x in val_step_outputs], dim = 0)
        val_preds = torch.argmax(val_logits, dim = 1)

        self.log('val_acc', self.metrics(val_preds, val_targets )['BinaryAccuracy'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.5, 0.9), weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epochs , 0.000005)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}