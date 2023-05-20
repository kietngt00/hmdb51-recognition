import argparse
from pathlib import Path
import numpy as np
import glob

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from data import HMDB51DataModule
from model.model import ModelInterface
from utils import *

#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml',type=str)
    parser.add_argument('--stage', default='train',type=str)
    parser.add_argument('--ckp_path', default=None,type=str)
    args = parser.parse_args()
    return args

#---->main
def main(cfg):

    #---->Initialize seed
    pl.seed_everything(cfg.General.seed)

    #---->load loggers
    logger = load_loggers(cfg)

    #---->load callbacks
    callbacks = load_callbacks(cfg)

    #---->Define Data 
    dm = HMDB51DataModule(cfg.Data)

    #---->Define Model
    model = ModelInterface(cfg.Model)
    
    #---->Instantiate Trainer
    trainer = Trainer(
        num_sanity_val_steps=0, 
        logger=logger,
        callbacks=callbacks,
        max_epochs= cfg.General.epochs,
        accelerator='gpu',
        deterministic=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        # limit_train_batches=1,
        # limit_val_batches=2,
    )

    #---->train or test
    if cfg.stage == 'train': 
        trainer.fit(model=model, datamodule=dm)
        trainer.test(datamodule=dm)
    elif cfg.stage == 'test': 
        model = ModelInterface.load_from_checkpoint(cfg.ckp_path)
        model.eval()
        trainer.test(model=model, datamodule=dm)

if __name__ == '__main__':
    args = make_parse()
    cfg = read_yaml(args.config)
    cfg.stage = args.stage
    cfg.ckp_path = args.ckp_path

    main(cfg)
 