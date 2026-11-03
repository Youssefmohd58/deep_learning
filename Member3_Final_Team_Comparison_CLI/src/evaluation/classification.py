"""Final held-out CUB classification evaluation and stable prediction exports."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,classification_report


@torch.inference_mode()
def predict_loader(model,loader,device,test_filenames=None):
    model.eval()
    rows=[]
    for batch in loader:
        x=batch['image'].to(device,non_blocking=True)
        logits=model(x)
        probs=logits.float().softmax(1)
        confidences, pred=probs.max(1)
        top5=probs.topk(5,dim=1).indices.cpu().tolist()
        for rid,y,p,conf,k5 in zip(batch['row_index'].tolist(),batch['label'].tolist(),
                                    pred.cpu().tolist(),confidences.cpu().tolist(),top5):
            rid=int(rid)
            item={'row_index':rid,'y_true':int(y),'y_pred':int(p),
                  'confidence':float(conf),'top5_hit':int(int(y) in k5)}
            if test_filenames is not None:
                item['image_id']=str(test_filenames[rid])
            rows.append(item)
    frame=pd.DataFrame(rows).sort_values('row_index').reset_index(drop=True)
    if frame['row_index'].duplicated().any():
        raise ValueError('Duplicate row_index in prediction set')
    if test_filenames is not None and len(frame)!=len(test_filenames):
        raise ValueError('Evaluation must include every official test image')
    return frame


def metrics_for(predictions,class_names):
    y=predictions['y_true'].to_numpy(dtype=int)
    p=predictions['y_pred'].to_numpy(dtype=int)
    classes=len(class_names)
    if len(y)==0 or y.min()<0 or y.max()>=classes or p.min()<0 or p.max()>=classes:
        raise ValueError('Class IDs out of range')
    pr,re,f1,_=precision_recall_fscore_support(y,p,labels=range(classes),average='macro',zero_division=0)
    weighted=precision_recall_fscore_support(y,p,labels=range(classes),average='weighted',zero_division=0)[2]
    metrics={'samples':len(y),'accuracy':float(accuracy_score(y,p)),
             'precision_macro':float(pr),'recall_macro':float(re),
             'f1_macro':float(f1),'f1_weighted':float(weighted)}
    if 'top5_hit' in predictions:
        metrics['top5_accuracy']=float(predictions['top5_hit'].mean())
    cm=confusion_matrix(y,p,labels=range(classes))
    report=pd.DataFrame(classification_report(y,p,labels=range(classes),
                 target_names=class_names,zero_division=0,output_dict=True)).T
    return metrics,cm,report


def save_evaluation(predictions,class_names,out_dir,tag):
    root=Path(out_dir)
    (root/'predictions').mkdir(parents=True,exist_ok=True)
    (root/'metrics').mkdir(parents=True,exist_ok=True)
    metrics,cm,report=metrics_for(predictions,class_names)
    predictions.to_csv(root/'predictions'/f'{tag}_test_predictions.csv',index=False)
    report.to_csv(root/'metrics'/f'{tag}_per_class.csv')
    np.save(root/'metrics'/f'{tag}_confusion_matrix.npy',cm)
    (root/'metrics'/f'{tag}_test_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    return metrics,cm


def compare_predictions(*named_frames):
    """Compare strictly aligned test predictions from multiple architectures."""
    if not named_frames:
        raise ValueError('No prediction frames')
    key='row_index'
    base=named_frames[0][1].set_index(key).sort_index()
    results=[]
    for name, frame in named_frames:
        aligned=frame.set_index(key).sort_index()
        if not aligned.index.equals(base.index):
            raise ValueError('Models have different test-image sets')
        if not np.array_equal(aligned['y_true'].to_numpy(),base['y_true'].to_numpy()):
            raise ValueError('Models have different reference labels')
        if 'image_id' in aligned and 'image_id' in base and not np.array_equal(
                aligned['image_id'].astype(str).to_numpy(),
                base['image_id'].astype(str).to_numpy()):
            raise ValueError('Models have different original image filenames')
        results.append((name,aligned))
    return results
