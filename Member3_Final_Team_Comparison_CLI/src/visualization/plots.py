"""Training, confusion-matrix, and model-comparison visualization functions."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.data.cub import IMAGENET_MEAN,IMAGENET_STD


def denormalize(tensor):
    mean=np.asarray(IMAGENET_MEAN).reshape(1,1,3)
    std=np.asarray(IMAGENET_STD).reshape(1,1,3)
    img=tensor.detach().float().cpu().permute(1,2,0).numpy()
    return np.clip(img*std+mean,0,1)


def plot_training(histories,title,path):
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for stage,df in histories.items():
        for ax,metric in zip(axes,['accuracy','loss']):
            ax.plot(df['epoch'],df['train_'+metric],label=stage+' train')
            ax.plot(df['epoch'],df['val_'+metric],label=stage+' validation',linestyle='--')
    for ax,metric in zip(axes,['accuracy','loss']):
        ax.set(xlabel='Epoch',ylabel=metric.title())
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig,path)
    return fig


def plot_confusion(cm,path,title):
    norm=cm/np.maximum(cm.sum(1,keepdims=True),1)
    fig,ax=plt.subplots(figsize=(12,10))
    im=ax.imshow(norm,cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
    ax.set(xlabel='Predicted class index',ylabel='True class index',title=title)
    fig.colorbar(im,ax=ax,shrink=.8)
    fig.tight_layout()
    _save(fig,path)
    return fig


def plot_comparison(table,path):
    fig,ax=plt.subplots(figsize=(8,4))
    labels=table['Model'].tolist()
    scores=(table['accuracy']*100).tolist()
    ax.bar(labels,scores)
    ax.set(ylabel='Official test accuracy (%)',title='CUB-200 scratch vs transfer comparison')
    ax.set_ylim(0,100)
    ax.tick_params(axis='x',labelrotation=12)
    for i,score in enumerate(scores):ax.text(i,score+1,f'{score:.2f}%',ha='center')
    fig.tight_layout()
    _save(fig,path)
    return fig


def _save(fig,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160)
