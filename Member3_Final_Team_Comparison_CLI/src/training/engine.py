"""Reusable fine-tuning engine: training loops, checkpoint selection, MLflow."""
from pathlib import Path
import json
import time
import shutil
import pandas as pd
import torch
from torch import nn
from src.models.transfer import build_model,set_trainable,frozen_bn_eval,trainable_count
from src.utils.config import seed_everything,prepare_output


def init_tracking(output_dir):
    import mlflow
    tracking=Path(output_dir)/'tracking'
    tracking.mkdir(parents=True,exist_ok=True)
    mlflow.set_tracking_uri(f'sqlite:///{(tracking / "mlflow.db").resolve().as_posix()}')
    mlflow.set_experiment('CUB200_Member3_Transfer_Learning')
    return mlflow


def one_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    training=optimizer is not None
    model.train(training)
    if training:
        frozen_bn_eval(model)
    loss_sum,correct,total=0.0,0,0
    ctx=torch.enable_grad() if training else torch.inference_mode()
    with ctx:
        for batch in loader:
            x=batch['image'].to(device,non_blocking=True)
            y=batch['label'].to(device,non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=(device.type=='cuda')):
                logits=model(x)
                loss=criterion(logits,y)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            loss_sum+=float(loss.detach())*len(y)
            correct+=(logits.argmax(dim=1)==y).sum().item()
            total+=len(y)
    if not total:
        raise ValueError('Empty DataLoader')
    return {'loss':loss_sum/total,'accuracy':correct/total}


def train_phase(model,architecture,stage,train_loader,val_loader,cfg,class_names,
                split_sha256,lr,weight_decay,epochs,run_name,checkpoint_path=None):
    """Train one stage, return model restored to best validation checkpoint.

    The held-out test set is not used here.
    """
    mlflow=init_tracking(cfg['output_dir'])
    device=next(model.parameters()).device
    set_trainable(model,architecture,stage)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                lr=float(lr),weight_decay=float(weight_decay))
    criterion=nn.CrossEntropyLoss(label_smoothing=cfg.get('label_smoothing',0.05))
    scaler=torch.amp.GradScaler('cuda',enabled=(device.type=='cuda'))
    best=(-1.0,float('inf'))
    best_state=None
    rows=[]
    started=time.perf_counter()
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({'architecture':architecture,'stage':stage,
                'epochs':epochs,'lr':float(lr),'weight_decay':float(weight_decay),
                'batch_size':cfg['batch_size'],'seed':cfg['seed'],
                'label_smoothing':cfg.get('label_smoothing',0.05),
                'train_images':len(train_loader.dataset),'val_images':len(val_loader.dataset),
                'trainable_parameters':trainable_count(model),'split_sha256':split_sha256})
        for epoch in range(1,epochs+1):
            tr=one_epoch(model,train_loader,device,criterion,optimizer,scaler)
            va=one_epoch(model,val_loader,device,criterion)
            row={'epoch':epoch,'train_accuracy':tr['accuracy'],'val_accuracy':va['accuracy'],
                 'train_loss':tr['loss'],'val_loss':va['loss']}
            rows.append(row)
            for key,val in row.items():
                if key!='epoch':
                    mlflow.log_metric(key,val,step=epoch)
            print(f'{run_name} {epoch}/{epochs}: train {tr["accuracy"]:.2%}, val {va["accuracy"]:.2%} '
                  f'| loss {tr["loss"]:.3f}/{va["loss"]:.3f}',flush=True)
            if (va['accuracy']>best[0] or
                (va['accuracy']==best[0] and va['loss']<best[1])):
                best=(va['accuracy'],va['loss'])
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        elapsed=time.perf_counter()-started
        mlflow.log_metrics({'best_val_accuracy':best[0], 'best_val_loss':best[1],
                            'duration_seconds':elapsed})
        if checkpoint_path is not None:
            target=Path(checkpoint_path)
            target.parent.mkdir(parents=True,exist_ok=True)
            torch.save({'architecture':architecture,'stage':stage,
                 'model_state_dict':best_state,'class_names':list(class_names),
                 'best_val_accuracy':best[0],'best_val_loss':best[1],
                 'config':{k:v for k,v in cfg.items() if k!='class_names'},
                 'split_sha256':split_sha256,
                 'completed_epochs':epochs},target)
            mlflow.log_param('checkpoint_path',str(target.resolve()))
        mlflow.log_dict({'history':rows},'history.json')
    if best_state is None:
        raise RuntimeError('No best state produced')
    model.load_state_dict(best_state)
    return pd.DataFrame(rows),{'val_accuracy':best[0],'val_loss':best[1],
                                'duration_seconds':elapsed}


def run_model_training(architecture,cfg,loaders,class_names,split_sha256,
                       head_lr=None,weight_decay=None):
    """Two-stage training with test-blind validation-based final checkpoint selection."""
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(cfg['seed'])
    root=prepare_output(cfg)
    head_lr=cfg['head_lr'] if head_lr is None else head_lr
    weight_decay=cfg['weight_decay'] if weight_decay is None else weight_decay
    model=build_model(architecture,num_classes=len(class_names),pretrained=True).to(device)
    ckpt=root/'checkpoints'
    h,hi=train_phase(model,architecture,'head',loaders['train'],loaders['validation'],
                     cfg,class_names,split_sha256,head_lr,weight_decay,cfg['head_epochs'],
                     f'{architecture}_head',ckpt/f'{architecture}_head.pt')
    h.to_csv(root/'history'/f'{architecture}_head.csv',index=False)
    f,fi=train_phase(model,architecture,'finetune',loaders['train'],loaders['validation'],
                     cfg,class_names,split_sha256,cfg['finetune_lr'],weight_decay,cfg['finetune_epochs'],
                     f'{architecture}_finetune',ckpt/f'{architecture}_finetune.pt')
    f.to_csv(root/'history'/f'{architecture}_finetune.csv',index=False)
    chosen='finetune' if (fi['val_accuracy']>hi['val_accuracy'] or
                         (fi['val_accuracy']==hi['val_accuracy'] and fi['val_loss']<hi['val_loss'])) else 'head'
    final=ckpt/f'{architecture}_best.pt'
    shutil.copy2(ckpt/f'{architecture}_{chosen}.pt',final)
    # Important: if validation preferred the earlier stage, load those weights into the returned model.
    saved=torch.load(final,map_location='cpu',weights_only=True)
    model.load_state_dict(saved['model_state_dict'])
    model.eval()
    summary={'architecture':architecture,'selected_stage':chosen,'best_val_accuracy':saved['best_val_accuracy'],
             'best_val_loss':saved['best_val_loss'],'best_checkpoint':str(final),
             'split_sha256':split_sha256}
    (root/'metrics'/f'{architecture}_selection.json').write_text(json.dumps(summary,indent=2))
    return model,{'head':h,'finetune':f},summary
