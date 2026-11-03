"""Quick offline unit tests. Run: python -m pytest -q tests"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pytest

from src.utils.config import load_config
from src.models.transfer import build_model, set_trainable, load_checkpoint, extract_features
from src.evaluation.classification import compare_predictions, metrics_for
from src.data.cub import get_or_create_split, transforms_for


def test_config_and_shared_split_setup(tmp_path):
    config=load_config('configs/classification.yaml')
    assert config['num_classes']==200
    assert config['head_epochs']==3 and config['finetune_epochs']==3
    assert 'split_manifest' in config
    assert Path(config['split_manifest']).is_file()

    class FakeTrain:
        def __init__(self):
            # Fake 5,994 items balanced across 200 integer classes.
            self.labels=np.arange(5994,dtype=int)%200
        def __len__(self):return len(self.labels)
        def __getitem__(self,k):
            if k=='label':return self.labels.tolist()
            raise KeyError(k)

    fake=FakeTrain()
    config['output_dir']=str(tmp_path/'out')
    config.pop('split_manifest')  # Only for this synthetic split-generation test.
    train_ids,val_ids,digest=get_or_create_split(fake,config)
    assert len(train_ids)==4795 and len(val_ids)==1199
    assert len(digest)==64
    ids_again,vals_again,digest_again=get_or_create_split(fake,config)
    assert (train_ids,val_ids,digest)==(ids_again,vals_again,digest_again)
    assert len(set(fake.labels[val_ids]))==200


def test_model_heads_and_checkpoint(tmp_path):
    torch.set_num_threads(2)
    for architecture,features in [('resnet18',512),('efficientnet_b0',1280)]:
        net=build_model(architecture,200,pretrained=False).eval()
        set_trainable(net,architecture,'head')
        head=sum(p.numel() for p in net.parameters() if p.requires_grad)
        set_trainable(net,architecture,'finetune')
        fine=sum(p.numel() for p in net.parameters() if p.requires_grad)
        assert head>0 and fine>head
        with torch.inference_mode():
            x=torch.rand(1,3,64,64)
            assert tuple(net(x).shape)==(1,200)
            assert tuple(extract_features(net,architecture,x).shape)==(1,features)
        checkpoint=tmp_path/f'{architecture}.pt'
        torch.save({'architecture':architecture,
                    'class_names':[f'Species {i}' for i in range(200)],
                    'model_state_dict':net.state_dict()},checkpoint)
        loaded,names,payload=load_checkpoint(checkpoint)
        assert names[5]=='Species 5'
        assert payload['architecture']==architecture
        del loaded,net


def test_prediction_metrics_and_alignment():
    first=pd.DataFrame({'row_index':[0,1,2],'y_true':[0,1,1],
                        'y_pred':[0,0,1],'top5_hit':[1,1,1]})
    second=pd.DataFrame({'row_index':[2,0,1],'y_true':[1,0,1],
                         'y_pred':[1,0,1],'top5_hit':[1,1,1]})
    pairs=compare_predictions(('resnet18',first),('efficientnet_b0',second))
    assert len(pairs)==2
    metrics,cm,_=metrics_for(first,[f'Class {i}' for i in range(200)])
    assert np.isclose(metrics['accuracy'],2/3)
    assert cm.shape==(200,200)
    second.loc[second.row_index==0,'y_true']=3
    with pytest.raises(ValueError):
        compare_predictions(('resnet18',first),('efficientnet_b0',second))
