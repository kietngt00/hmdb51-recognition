from pathlib import Path
import os

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

    log_path = os.path.join(cfg.General.log_path, str(cfg.Model.lr) + '_' + str(cfg.Data.batch_size)) # example: logs/0.001_32
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

    # early_stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0.00,
    #     patience=cfg.General.patience,
    #     verbose=True,
    #     mode='min'
    # )
    # callbacks.append(early_stop_callback)
    log_path = os.path.join(cfg.General.log_path, str(cfg.Model.lr) + '_' + str(cfg.Data.batch_size))
    callbacks.append(ModelCheckpoint(monitor = 'val_acc',
                                        dirpath = log_path,
                                        filename = '{epoch:02d}-{val_acc:.4f}',
                                        verbose = True,
                                        save_last = True,
                                        save_top_k = 1,
                                        mode = 'max',
                                        save_weights_only = True))
    return callbacks

def get_label_dict():
    idxs, labels = [], []
    with open('dataset/hmdb_labels.txt', 'r') as f:
        lines = f.readlines()

    for line in lines:
        idx, label = line.strip().split(' ')
        idxs.append(int(idx)-1)
        labels.append(label)

    dct = dict(zip(labels, idxs))
    return dct