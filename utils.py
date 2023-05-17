from pathlib import Path

#---->read yaml
import yaml
from addict import Dict
def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

#---->load Loggers
from pytorch_lightning import loggers as pl_loggers

def load_loggers(cfg):

    log_path = cfg.General.log_path
    Path(log_path).mkdir(exist_ok=True, parents=True)
    
    #---->TensorBoard
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_path, name='', version='')
    print(f'---->TensorBoard dir: {tb_logger.log_dir}')
    
    return tb_logger


#---->load Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
def load_callbacks(cfg):

    callbacks = []

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=cfg.General.patience,
        verbose=True,
        mode='min'
    )
    callbacks.append(early_stop_callback)

    if cfg.General.stage == 'train' :
        callbacks.append(ModelCheckpoint(monitor = 'val_acc',
                                         dirpath = str(cfg.log_path),
                                         filename = '{epoch:02d}-{val_acc:.4f}',
                                         verbose = True,
                                         save_last = True,
                                         save_top_k = 1,
                                         mode = 'max',
                                         save_weights_only = True))
    return callbacks
