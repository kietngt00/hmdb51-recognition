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
        self.save_hyperparameters()
        self.args = args
        self.label_dict = get_label_dict()

        self.cnn = r2plus1d_18(pretrained=True).cuda()

        self.cross_attention = None
        self.self_attention = SelfAttention(**args.self_attention).cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task='multiclass', num_classes=self.args.num_classes,
                                                                           average='micro')])
        self.validation_step_outputs = {}
        self.test_step_outputs = {}
    
    def forward(self, x):
        x = self.cnn(x)
        x = rearrange(x, 'N C T H W -> N T H W C')
        if self.cross_attention is None:
            self.args.cross_attention.input_channels = x.shape[-1]
            self.cross_attention = CrossAttention(**self.args.cross_attention).cuda()
        x = self.cross_attention(x)
        x = self.self_attention(x)
        return x

    def training_step(self, batch, batch_idx):
        video, target = batch['video'], batch['label']
        target = torch.tensor([self.label_dict[x] for x in target], device=video.device)
        logit = self(video)
        loss = self.criterion(logit, target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=video.shape[0])
        return loss
    
    def validation_step(self, batch, batch_idx):
        video, target, video_index = batch['video'], batch['label'], batch['video_index']
        target = torch.tensor([self.label_dict[x] for x in target], device=video.device)
        logit = self(video)
        loss = self.criterion(logit, target)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=video.shape[0])

        # val_step_outputs: {vid_idx: {'logits': [...], 'target': target}}
        for i, vid_idx in enumerate(video_index):
            vid_idx = vid_idx.item()
            if vid_idx not in self.validation_step_outputs:
                self.validation_step_outputs[vid_idx] = {'logits': [], 'target': None}
            self.validation_step_outputs[vid_idx]['logits'].append(logit[i].cpu())
            self.validation_step_outputs[vid_idx]['target'] = target[i].item()

        return loss

    def test_step(self, batch, batch_idx):
        video, target, video_index = batch['video'], batch['label'], batch['video_index']
        target = torch.tensor([self.label_dict[x] for x in target], device=video.device)
        logit = self(video)
        loss = self.criterion(logit, target)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=video.shape[0])

        # test_step_outputs: {vid_idx: {'logits': [...], 'target': target}}
        for i, vid_idx in enumerate(video_index):
            vid_idx = vid_idx.item()
            if vid_idx not in self.test_step_outputs:
                self.test_step_outputs[vid_idx] = {'logits': [], 'target': None}
            self.test_step_outputs[vid_idx]['logits'].append(logit[i].cpu())
            self.test_step_outputs[vid_idx]['target'] = target[i].item()

        return loss

    def on_validation_epoch_end(self):
        # Video prediction is the majority vote of its clip predictions - 1 video is sampled in multiple clips
        val_preds, val_targets = [], []
        for _, output in self.validation_step_outputs.items():
            clip_preds = torch.argmax(torch.stack(output['logits']), dim = 1) # Get prediction for each clip of a video
            val_preds.append(torch.mode(clip_preds, dim = 0).values.item())   # Majority vote for video prediction
            val_targets.append(output['target'])                              # Get target for video
        val_preds, val_targets = torch.tensor(val_preds), torch.tensor(val_targets)

        self.log('val_acc', self.metrics(val_preds, val_targets)['MulticlassAccuracy'], 
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()

        # Accuracy from all clips
        # val_logits = torch.cat([x['logit'] for x in self.validation_step_outputs], dim = 0)
        # val_targets = torch.cat([x['target'] for x in self.validation_step_outputs], dim = 0)

        # val_preds = torch.argmax(val_logits, dim = 1)
    
    def on_test_epoch_end(self):
        test_preds, test_targets = [], []
        for _, output in self.test_step_outputs.items():
            clip_preds = torch.argmax(torch.stack(output['logits']), dim = 1)  # Get prediction for each clip of a video
            test_preds.append(torch.mode(clip_preds, dim = 0).values.item())   # Majority vote for video prediction
            test_targets.append(output['target'])                              # Get target for video
        test_preds, test_targets = torch.tensor(test_preds), torch.tensor(test_targets)

        self.log('test_acc', self.metrics(test_preds, test_targets)['MulticlassAccuracy'], 
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}