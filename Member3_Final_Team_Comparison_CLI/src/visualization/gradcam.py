"""Grad-CAM for correct and incorrect classifications with a model-agnostic interface."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from src.models.transfer import get_cam_layer
from src.visualization.plots import denormalize


class GradCAM:
    def __init__(self,model,target_layer):
        self.model=model
        self.activation=None
        self.gradient=None
        self.handle=target_layer.register_forward_hook(self._hook)

    def _hook(self,module,inputs,output):
        self.activation=output
        output.register_hook(self._gradient_hook)

    def _gradient_hook(self,gradient):
        self.gradient=gradient

    def compute(self,image_tensor,target_class):
        self.model.eval()
        self.gradient=None
        device=next(self.model.parameters()).device
        x=image_tensor.unsqueeze(0).to(device).detach().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits=self.model(x)
            logits[0,int(target_class)].backward()
            if self.gradient is None:
                raise RuntimeError('Grad-CAM gradients were not collected')
            weights=self.gradient.float().mean(dim=(2,3),keepdim=True)
            heat=(weights*self.activation.float()).sum(dim=1,keepdim=True).relu()
            heat=F.interpolate(heat,size=x.shape[-2:],mode='bilinear',align_corners=False)
            heat=heat[0,0].detach().cpu().numpy()
        heat-=heat.min()
        heat/=max(float(heat.max()),1e-8)
        return heat

    def close(self):
        self.handle.remove()


def plot_cases(model,architecture,predictions,dataset,class_names,path):
    """Choose cases by row_index instead of relying on accidental row ordering."""
    if not {'row_index','y_true','y_pred'}.issubset(predictions.columns):
        raise ValueError('Predictions must have row_index, y_true, y_pred')
    lookup={int(rid):pos for pos,rid in enumerate(dataset.indices)}
    chosen=[]
    for correct in (True,False):
        selected=predictions.loc[(predictions['y_true']==predictions['y_pred'])==correct]
        if len(selected):
            chosen.append((selected.iloc[0],correct))
    if not chosen:
        raise ValueError('No cases to explain')
    cam=GradCAM(model,model.encoder[-1] if architecture=='scratch_cnn' else get_cam_layer(model,architecture))
    fig,axes=plt.subplots(len(chosen),3,figsize=(13,4.2*len(chosen)),squeeze=False)
    try:
        for row,(pred_row,correct) in enumerate(chosen):
            rid=int(pred_row['row_index'])
            sample=dataset[lookup[rid]]
            x=sample['image']
            true=int(pred_row['y_true'])
            predicted=int(pred_row['y_pred'])
            orig=denormalize(x)
            pred_heat=cam.compute(x,predicted)
            axes[row,0].imshow(orig)
            axes[row,0].set_title(('CORRECT' if correct else 'INCORRECT')+' — '+class_names[true],fontsize=9)
            axes[row,1].imshow(orig)
            axes[row,1].imshow(pred_heat,cmap='jet',alpha=.4,vmin=0,vmax=1)
            axes[row,1].set_title('Predicted: '+class_names[predicted],fontsize=9)
            axes[row,2].imshow(orig)
            if correct:
                axes[row,2].set_title('Original image')
            else:
                heat_true=cam.compute(x,true)
                axes[row,2].imshow(heat_true,cmap='jet',alpha=.4,vmin=0,vmax=1)
                axes[row,2].set_title('True class: '+class_names[true],fontsize=9)
            for a in axes[row]:a.axis('off')
    finally:
        cam.close()
    fig.suptitle(f'{architecture} — Grad-CAM examples')
    fig.tight_layout()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160)
    return fig
