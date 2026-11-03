"""Required Grid Search and Random Search on TRAIN/VALIDATION only."""
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
import torch
from src.models.transfer import build_model
from src.data.cub import stratified_subset,make_loader
from src.training.engine import train_phase
from src.utils.config import seed_everything,prepare_output


def tune_resnet_head(cfg,loaders,train_labels,class_names,split_sha256):
    root=prepare_output(cfg)
    tr=stratified_subset(loaders['train'].dataset,cfg['tuning_train_images'],train_labels,cfg['seed'])
    va=stratified_subset(loaders['validation'].dataset,cfg['tuning_val_images'],train_labels,cfg['seed'])
    grid=[{'search':'grid','lr':float(lr),'weight_decay':float(wd)}
          for lr,wd in product([1e-3,3e-4],[0.0,1e-4])]
    rng=np.random.default_rng(cfg['seed'])
    randoms=[{'search':'random','lr':float(10**rng.uniform(-4,-2.7)),
              'weight_decay':float(10**rng.uniform(-6,-3))} for _ in range(4)]
    entries=[]
    for trial,params in enumerate(grid+randoms,1):
        seed_everything(cfg['seed'])
        tr_dl=make_loader(tr,cfg['batch_size'],True,cfg['num_workers'],cfg['seed'])
        va_dl=make_loader(va,cfg['batch_size'],False,cfg['num_workers'],cfg['seed'])
        model=build_model('resnet18',num_classes=len(class_names),pretrained=True)
        model=model.to('cuda' if torch.cuda.is_available() else 'cpu')
        _,metrics=train_phase(model,'resnet18','head',tr_dl,va_dl,
              cfg,class_names,split_sha256,params['lr'],params['weight_decay'],
              cfg['tuning_epochs'],f'tuning_{params["search"]}_{trial}')
        entries.append({'trial':trial,**params,**metrics})
        pd.DataFrame(entries).to_csv(root/'metrics'/'tuning_trials.csv',index=False)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    frame=pd.DataFrame(entries)
    comparison=frame.groupby('search',as_index=False).agg(
        best_val_accuracy=('val_accuracy','max'),trials=('trial','count'),
        total_seconds=('duration_seconds','sum'))
    comparison.to_csv(root/'metrics'/'grid_vs_random.csv',index=False)
    best=frame.sort_values(['val_accuracy','val_loss'],ascending=[False,True]).iloc[0]
    return float(best['lr']),float(best['weight_decay']),frame,comparison
