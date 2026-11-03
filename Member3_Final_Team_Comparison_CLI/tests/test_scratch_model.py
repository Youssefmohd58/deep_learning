"""Offline tests for Member 2 architecture/checkpoint integration."""
from pathlib import Path
import torch
from src.models.bird_cnn import BirdCNN
from src.models.scratch import load_scratch_checkpoint,checkpoint_kind
from src.data.scratch import scratch_eval_transform
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]

def test_checkpoint_load_and_outputs():
    ck=ROOT/'team_assets'/'member2'/'best_model.pth'
    net,meta=load_scratch_checkpoint(ck)
    assert checkpoint_kind(ck)=='scratch_cnn'
    assert meta['epoch']==78
    assert 60 < meta['val_accuracy'] < 61
    net.eval()
    torch.set_num_threads(2)
    with torch.no_grad():
        got=net(torch.zeros(1,3,64,64))
    assert got.shape==(1,200)

def test_eval_preprocessing_shape():
    im=Image.new('RGB',(350,230))
    got=scratch_eval_transform(128)(im)
    assert got.shape==(3,128,128)
