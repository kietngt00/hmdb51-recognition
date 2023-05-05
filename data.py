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
)

def get_train_transform():
    return Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(8),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(224),
                    RandomHorizontalFlip(p=0.5),
                  ]
                ),
              ),
            ]
        )

def get_val_transform():
    return Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(8),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    ShortSideScale(size=320),
                    CenterCrop(224),
                  ]
                ),
              ),
            ]
        )
class HMDB51DataModule(pl.LightningDataModule):
    # Dataset configuration
    _DATA_PATH = '/root/project/dataset/annotations'
    _CLIP_DURATION = 2  # Duration of sampled clip for each video
    _BATCH_SIZE = 8
    _NUM_WORKERS = 8  # Number of parallel processes fetching data

    def setup(self, stage=None):
        train_transform = get_train_transform()
        self.train_dataset = pytorchvideo.data.Hmdb51(
                data_path=self._DATA_PATH,
                split_id=1,
                split_type='train',
                clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
                transform=train_transform,
                video_path_prefix='/root/project/dataset/hmdb51'
            )

        val_transform = get_val_transform()
        self.val_dataset = pytorchvideo.data.Hmdb51(
                data_path=self._DATA_PATH,
                split_id=1,
                split_type='test',
                clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self._CLIP_DURATION),
                transform=val_transform,
                video_path_prefix='/root/project/dataset/hmdb51'
            )


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )