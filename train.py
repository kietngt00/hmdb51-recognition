import argparse
from pathlib import Path
import numpy as np
import glob

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from data import DataModule
from model import ModelInterface
from utils import *

#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml',type=str)
    args = parser.parse_args()
    return args

#---->main
def main(cfg):

    #---->Initialize seed
    pl.seed_everything(cfg.General.seed)

    #---->load loggers
    cfg.load_loggers = load_loggers(cfg)

    #---->load callbacks
    cfg.callbacks = load_callbacks(cfg)

    #---->Define Data 
    dm = DataModule()

    #---->Define Model
    model = ModelInterface()
    
    #---->Instantiate Trainer
    trainer = Trainer(
        num_sanity_val_steps=0, 
        logger=cfg.load_loggers,
        callbacks=cfg.callbacks,
        max_epochs= 200,
        accelerator='gpu',
        deterministic=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
    )

    #---->train or test
    trainer.fit(model=model, datamodule=dm)

if __name__ == '__main__':

    args = make_parse()
    cfg = read_yaml(args.config)

    main(cfg)
 