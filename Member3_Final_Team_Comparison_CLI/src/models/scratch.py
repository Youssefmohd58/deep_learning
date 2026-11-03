"""Load Member 2's final 5-block scratch CNN checkpoint, without retraining.

This checkpoint does not embed its class-name mapping or training manifest hash.
The official CUB-200 label IDs and supplied saved manifest are verified by data
loading in the shared evaluation CLI. Only load checkpoints from trusted sources.
"""
from pathlib import Path
import torch
from src.models.bird_cnn import BirdCNN

SCRATCH_ARCH = 'scratch_cnn'

def load_scratch_checkpoint(path, device='cpu'):
    path=Path(path)
    state=torch.load(path,map_location='cpu',weights_only=True)
    required={'model_state_dict','num_classes','epoch','val_accuracy'}
    if not isinstance(state,dict) or not required.issubset(state):
        raise ValueError('Not a Member 2 checkpoint: missing keys '+str(required-set(state)))
    if int(state['num_classes'])!=200:
        raise ValueError('Member 2 checkpoint must have 200 classes')
    model=BirdCNN(num_classes=200,dropout=0.5)
    model.load_state_dict(state['model_state_dict'],strict=True)
    return model.to(device).eval(),state

def checkpoint_kind(path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(state,dict):
        raise ValueError('Checkpoint must be a mapping')
    if 'architecture' in state:
        return str(state['architecture'])
    if 'model_state_dict' in state and any(str(k).startswith('encoder.')
            for k in state['model_state_dict']):
        return SCRATCH_ARCH
    raise ValueError('Unknown checkpoint structure')
