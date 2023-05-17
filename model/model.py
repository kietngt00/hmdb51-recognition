import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from model.modules import *
from model.attention import CrossAttention, SelfAttention
from einops import rearrange, repeat
from utils import get_label_dict

class ModelInterface(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.label_dict = get_label_dict()

        self.cnn = None
        self.cross_attention = None
        self.self_attention = SelfAttention(**args.self_attention)

        self.criterion = nn.CrossEntropyLoss()
        self.metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task='binary', num_classes=self.args.num_classes,
                                                                           average='micro')])
    
    def forward(self, x):
        x = self.cnn(x)
        x = rearrange(x, '(N C T H W -> N T H W C')
        if self.cross_attention is None:
            self.args.cross_attention.input_channels = x.shape[-1]
            self.cross_attention = CrossAttention(*self.args.cross_attention)
        x = self.cross_attention(x)
        x = self.self_attention(x)
        return x

    def training_step(self, batch, batch_idx):
        video, target = batch['video'], batch['label']
        import pdb; pdb.set_trace() # Debugging
        target = torch.tensor([self.label_dict[x] for x in target], dtype=video.device)
        logit = self(video)
        loss = self.criterion(logit, target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        video, target = batch['video'], batch['label']
        target = torch.tensor([self.label_dict[x] for x in target], dtype=video.device)
        logit = self(video)
        loss = self.criterion(logit, target)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return {'logit': logit, 'target': target}

    def on_validation_epoch_end(self, val_step_outputs):
        val_logits = torch.stack([x['logit'] for x in val_step_outputs], dim = 0)
        val_targets = torch.stack([x['target'] for x in val_step_outputs], dim = 0)
        val_preds = torch.argmax(val_logits, dim = 1)

        self.log('val_acc', self.metrics(val_preds, val_targets )['BinaryAccuracy'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.5, 0.9), weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epochs , 0.000005)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}