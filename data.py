from uu import decode
import torch
import pytorch_lightning as pl
import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    CenterCrop,
    Resize,
)

def get_train_transform(num_frames):
    return Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    # RandomShortSideScale(min_size=128, max_size=160),
                    Resize((128, 171), antialias=False),
                    RandomCrop(112),
                    RandomHorizontalFlip(p=0.5),
                  ]
                ),
              ),
            ]
        )

def get_val_transform(num_frames):
    return Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    # ShortSideScale(size=128),
                    Resize((128, 171),antialias=False),
                    CenterCrop(112),
                  ]
                ),
              ),
            ]
        )

class HMDB51DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.clip_duration = args.clip_duration
        self.video_path_prefix = args.video_path_prefix
        self.num_frames = args.num_frames

    def setup(self, stage=None):
        train_transform = get_train_transform(self.num_frames)
        self.train_dataset = pytorchvideo.data.Hmdb51(
                data_path=self.data_path,
                split_id=1,
                split_type='train',
                clip_sampler=pytorchvideo.data.RandomClipSampler(self.clip_duration),
                transform=train_transform,
                video_path_prefix=self.video_path_prefix,
                decode_audio=False
            )

        val_transform = get_val_transform(self.num_frames)
        self.val_dataset = pytorchvideo.data.Hmdb51(
                data_path=self.data_path,
                split_id=1,
                split_type='test',
                clip_sampler=pytorchvideo.data.UniformClipSampler(self.clip_duration),
                transform=val_transform,
                video_path_prefix=self.video_path_prefix,
                decode_audio=False
            )
        
        self.test_dataset = pytorchvideo.data.Hmdb51(
                data_path=self.data_path,
                split_id=1,
                split_type='unused',
                clip_sampler=pytorchvideo.data.UniformClipSampler(self.clip_duration),
                transform=val_transform,
                video_path_prefix=self.video_path_prefix,
                decode_audio=False
            )


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )