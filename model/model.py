import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from model.attention import CrossAttention, SelfAttention
from einops import rearrange
from utils import get_label_dict
from model.R2plus1D import r2plus1d_18

class ModelInterface(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.label_dict = get_label_dict()

        self.cnn = r2plus1d_18(pretrained=True).cuda()
        self.cnn.fc = nn.Linear(512, 51)
        # self.cross_attention = None
        # self.self_attention = SelfAttention(**args.self_attention).cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task='multiclass', num_classes=self.args.num_classes,
                                                                           average='micro')])
        self.validation_step_outputs = []
    
    def forward(self, x):
        x = self.cnn(x)
        # x = rearrange(x, 'N C T H W -> N T H W C')
        # if self.cross_attention is None:
        #     self.args.cross_attention.input_channels = x.shape[-1]
        #     self.cross_attention = CrossAttention(**self.args.cross_attention).cuda()
        # x = self.cross_attention(x)
        # x = self.self_attention(x)
        return x

    def training_step(self, batch, batch_idx):
        video, target = batch['video'], batch['label']
        target = torch.tensor([self.label_dict[x] for x in target], device=video.device)
        logit = self(video)
        loss = self.criterion(logit, target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=video.shape[0])
        return loss
    
    def validation_step(self, batch, batch_idx):
        video, target = batch['video'], batch['label']
        target = torch.tensor([self.label_dict[x] for x in target], device=video.device)
        logit = self(video)
        loss = self.criterion(logit, target)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=video.shape[0])
        self.validation_step_outputs.append({'logit': logit, 'target': target})
        return loss

    def on_validation_epoch_end(self):
        val_logits = torch.cat([x['logit'] for x in self.validation_step_outputs], dim = 0)
        val_targets = torch.cat([x['target'] for x in self.validation_step_outputs], dim = 0)

        val_preds = torch.argmax(val_logits, dim = 1)
        self.log('val_acc', self.metrics(val_preds, val_targets)['MulticlassAccuracy'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.5, 0.9), weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epochs , 0.000005)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}