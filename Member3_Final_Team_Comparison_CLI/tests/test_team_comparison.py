"""No dataset download required: test three-model prediction alignment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.cli import compare_command
from src.utils.config import load_config,prepare_output
from src.evaluation.classification import compare_predictions


def test_comparison_requires_matching_official_images(tmp_path,monkeypatch):
    cfg=load_config('configs/classification.yaml')
    cfg['output_dir']=str(tmp_path/'out')
    out=prepare_output(cfg)
    y_true=np.arange(5794)%200
    for name,shift in [('resnet18',0),('efficientnet_b0',1),('scratch_cnn',2)]:
        frame=pd.DataFrame({'row_index':np.arange(5794),
            'image_id':[f'bird_{i}.jpg' for i in range(5794)],
            'y_true':y_true,
            'y_pred':(y_true+shift)%200})
        frame.to_csv(out/'predictions'/f'{name}_test_predictions.csv',index=False)
    # Only metadata is required, not the large pretrained checkpoint.
    import src.cli as cli
    monkeypatch.setattr(cli,'checkpoint_for',lambda cfg,arch:'not_needed.pt')
    monkeypatch.setattr(cli,'load_checkpoint',
       lambda path,device: (None,[f'Bird {i}' for i in range(200)],{}))
    result=compare_command(cfg)
    assert len(result)==3
    assert float(result.iloc[0]['accuracy'])==1.0
    assert (out/'metrics'/'three_model_comparison.csv').is_file()
    assert (out/'figures'/'three_model_comparison.png').is_file()

    frame=pd.read_csv(out/'predictions'/'scratch_cnn_test_predictions.csv')
    frame.loc[0,'image_id']='wrong_bird.jpg'
    frame.to_csv(out/'predictions'/'scratch_cnn_test_predictions.csv',index=False)
    with pytest.raises(ValueError,match='filenames'):
        compare_command(cfg)
