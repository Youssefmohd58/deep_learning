"""Team CUB-200 command-line interface: pretrained Member 3 and Member 2 scratch CNN.

Run from the VS Code terminal in the project root, e.g.:
    python -m src.cli pipeline --config configs/classification.yaml --tune
    python -m src.cli predict --checkpoint outputs/member3/checkpoints/resnet18_best.pt --image bird.jpg
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from src.utils.config import load_config, seed_everything, prepare_output
from src.models.transfer import ARCHITECTURES, build_model, load_checkpoint, get_cam_layer
from src.models.scratch import load_scratch_checkpoint, checkpoint_kind, SCRATCH_ARCH
from src.data.scratch import scratch_test_loader, scratch_eval_transform
from src.data.cub import load_cub_loaders, transforms_for
from src.training.engine import run_model_training
from src.training.tuning import tune_resnet_head
from src.evaluation.classification import predict_loader, save_evaluation, compare_predictions, metrics_for
from src.visualization.plots import plot_training, plot_confusion, plot_comparison
from src.visualization.gradcam import GradCAM, plot_cases


EVAL_ARCHITECTURES = (*ARCHITECTURES, SCRATCH_ARCH)


def device_for():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def checkpoint_for(cfg, arch):
    if arch==SCRATCH_ARCH:
        path=Path(cfg['scratch_checkpoint'])
        hint='Use the supplied Member 2 checkpoint in team_assets/member2.'
    else:
        path=Path(cfg['output_dir'])/'checkpoints'/f'{arch}_best.pt'
        hint='Train the pretrained model first or copy a compatible trusted checkpoint here.'
    if not path.exists():
        raise FileNotFoundError(f'Checkpoint missing: {path}. {hint}')
    return path


def train_command(args, cfg):
    output=prepare_output(cfg)
    loaders, names, split_hash, labels=load_cub_loaders(cfg,include_test=False)
    head_lr, wd=float(cfg['head_lr']), float(cfg['weight_decay'])
    if args.tune:
        head_lr, wd, trials, methods=tune_resnet_head(cfg,loaders,labels,names,split_hash)
        print('\nGrid vs Random Search:\n'+methods.to_string(index=False))
        print(f'Chosen ResNet head hyperparameters: LR={head_lr:.6g}, weight decay={wd:.6g}')
    selected=ARCHITECTURES if args.model=='all' else (args.model,)
    for architecture in selected:
        model,histories,selection=run_model_training(architecture,cfg,loaders,names,
                                                    split_hash,head_lr,wd)
        plot_training(histories,f'{architecture}: train vs validation',
                      output/'figures'/f'{architecture}_learning_curves.png')
        print(json.dumps(selection,indent=2))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print('\nTraining finished. Best checkpoints selected using validation data only.')
    return selected


def load_any_checkpoint(checkpoint,cfg,device,known_class_names=None):
    arch=checkpoint_kind(checkpoint)
    if arch==SCRATCH_ARCH:
        model,payload=load_scratch_checkpoint(checkpoint,device)
        if known_class_names is not None:
            names=known_class_names
        else:
            names=json.loads(Path(cfg['class_names_file']).read_text(encoding='utf-8'))
        if len(names)!=200:
            raise ValueError('Scratch model class mapping must have 200 names')
        return model,names,dict(payload,architecture=SCRATCH_ARCH)
    model,names,payload=load_checkpoint(checkpoint,device)
    return model,names,payload


def eval_loader_for(architecture,loaders,cfg):
    return (scratch_test_loader(loaders['test'],cfg)
            if architecture==SCRATCH_ARCH else loaders['test'])


def evaluate_one(checkpoint,cfg,loaders,expected_names,split_hash,
                 generate_gradcam=False):
    output=prepare_output(cfg)
    model,class_names,payload=load_any_checkpoint(
        checkpoint,cfg,device_for(),known_class_names=expected_names)
    architecture=payload['architecture']
    if class_names!=expected_names:
        raise ValueError('Checkpoint class names/order mismatch with CUB-200')
    if payload.get('split_sha256') and payload['split_sha256']!=split_hash:
        raise ValueError('Member 3 checkpoint split differs from the team manifest')
    if architecture==SCRATCH_ARCH:
        print('Member 2 checkpoint from epoch',payload.get('epoch'),
              'with VAL accuracy',payload.get('val_accuracy'),
              'percent. This is not test accuracy.')
    dl=eval_loader_for(architecture,loaders,cfg)
    preds=predict_loader(model,dl,device_for(),loaders['test_filenames'])
    if len(preds)!=5794 or preds['row_index'].tolist()!=list(range(5794)):
        raise ValueError('Incomplete official CUB-200 test predictions')
    metrics,cm=save_evaluation(preds,class_names,output,architecture)
    plot_confusion(cm,output/'figures'/f'{architecture}_confusion_matrix.png',
                   f'{architecture}: normalized CUB-200 test confusion matrix')
    print(f'\n{architecture}: held-out official test metrics')
    print(json.dumps(metrics,indent=2))
    if generate_gradcam:
        import matplotlib.pyplot as plt
        fig=plot_cases(model,architecture,preds,dl.dataset,class_names,
             output/'figures'/f'{architecture}_gradcam_examples.png')
        plt.close(fig)
    if architecture==SCRATCH_ARCH:
        evidence={
            'model':'Member 2 final 5-block CNN, trained from scratch',
            'checkpoint_epoch':int(payload['epoch']),
            'reported_validation_accuracy_percent':float(payload['val_accuracy']),
            'evaluation_input_size':int(cfg['scratch_image_size']),
            'manifest_sha256':split_hash,
            'checkpoint_sha256':__import__('hashlib').sha256(Path(checkpoint).read_bytes()).hexdigest(),
            'test_images':len(preds),
            'note':'Validation score is not test accuracy; test metrics are saved separately.'}
        (output/'metrics'/'scratch_cnn_provenance.json').write_text(
            json.dumps(evidence,indent=2),encoding='utf-8')
    del model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return metrics


def evaluate_command(args,cfg,selected_override=None,with_gradcam=False):
    loaders,names,split_hash,_=load_cub_loaders(cfg,include_test=True)
    # Cache canonical label mapping for offline single-image inference.
    output=prepare_output(cfg)
    (output/'metrics'/'class_names.json').write_text(json.dumps(names,indent=2))
    if selected_override is not None:
        selected=list(selected_override)
    elif args.checkpoint:
        selected=[checkpoint_kind(Path(args.checkpoint))]
    else:
        selected=list(EVAL_ARCHITECTURES if args.model=='all' else (args.model,))
    results={}
    for architecture in selected:
        path=Path(args.checkpoint) if args.checkpoint else checkpoint_for(cfg,architecture)
        actual=checkpoint_kind(path)
        if actual!=architecture:
            raise ValueError(f'Checkpoint is {actual}, but requested {architecture}')
        results[architecture]=evaluate_one(path,cfg,loaders,names,split_hash,
                                           generate_gradcam=with_gradcam)
    return results


def compare_command(cfg,models=None):
    output=prepare_output(cfg)
    selected=tuple(models or EVAL_ARCHITECTURES)
    if len(selected)<2:
        raise ValueError('Comparison requires at least two models')
    paths=[output/'predictions'/f'{arch}_test_predictions.csv' for arch in selected]
    missing=[str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError('First evaluate the requested checkpoints. Missing: '+str(missing))
    named=list(zip(selected,[pd.read_csv(path) for path in paths]))
    aligned=compare_predictions(*named)
    for arch,frame in aligned:
        if len(frame)!=5794 or frame.index.tolist()!=list(range(5794)):
            raise ValueError(f'{arch} does not cover all 5,794 official test rows')
    # Use a pretrained checkpoint's class metadata, or saved public dataset mapping.
    preferred=next((arch for arch in selected if arch in ARCHITECTURES),None)
    if preferred:
        _,names,_=load_checkpoint(checkpoint_for(cfg,preferred),device='cpu')
    else:
        names=json.loads(Path(cfg['class_names_file']).read_text(encoding='utf-8'))
    rows=[]
    for arch,frame in aligned:
        metrics,_,_=metrics_for(frame.reset_index(),names)
        rows.append({'Model':arch,**metrics})
    table=pd.DataFrame(rows)
    table.to_csv(output/'metrics'/'three_model_comparison.csv',index=False)
    fig=plot_comparison(table,output/'figures'/'three_model_comparison.png')
    import matplotlib.pyplot as plt
    plt.close(fig)
    print('\nAll compared models used the same 5,794 official test images:')
    print(table.to_string(index=False))
    return table


def predict_command(args,cfg):
    if args.top_k < 1 or args.top_k > 200:
        raise ValueError('--top-k must be between 1 and 200')
    model,names,payload=load_any_checkpoint(args.checkpoint,cfg,device_for())
    image=Image.open(args.image).convert('RGB')
    arch=payload['architecture']
    size=(int(cfg['scratch_image_size']) if arch==SCRATCH_ARCH else
          int(payload.get('config',{}).get('image_size',cfg['image_size'])))
    transform=scratch_eval_transform(size) if arch==SCRATCH_ARCH else transforms_for(False,size)
    image_input=transform(image).unsqueeze(0).to(device_for())
    with torch.inference_mode():
        probabilities=model(image_input).softmax(dim=1)[0]
    confidences, ids=probabilities.topk(args.top_k)
    predictions=[{'class_id':int(cls),'species':names[int(cls)],'probability':float(conf)}
                 for conf,cls in zip(confidences.cpu(),ids.cpu())]
    output={'image':str(Path(args.image).resolve()),'model':payload['architecture'],
            'predictions':predictions}
    print(json.dumps(output,indent=2))
    if args.output:
        dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(output,indent=2),encoding='utf-8')
    return output


def gradcam_command(args,cfg):
    import matplotlib.pyplot as plt
    from src.visualization.plots import denormalize
    model,names,payload=load_any_checkpoint(args.checkpoint,cfg,device_for())
    architecture=payload['architecture']
    image=Image.open(args.image).convert('RGB')
    size=(int(cfg['scratch_image_size']) if architecture==SCRATCH_ARCH else
          int(payload.get('config',{}).get('image_size',cfg['image_size'])))
    transform=scratch_eval_transform(size) if architecture==SCRATCH_ARCH else transforms_for(False,size)
    x=transform(image)
    with torch.inference_mode():
        predicted=int(model(x.unsqueeze(0).to(device_for())).argmax(1).item())
    target=predicted if args.class_id is None else args.class_id
    if target<0 or target>=len(names):
        raise ValueError(f'--class-id must be 0..{len(names)-1}')
    extractor=GradCAM(model,model.encoder[-1] if architecture==SCRATCH_ARCH else get_cam_layer(model,architecture))
    try:
        heat=extractor.compute(x,target)
    finally:
        extractor.close()
    photo=denormalize(x)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].imshow(photo);axes[0].set_title('Input bird image')
    axes[1].imshow(photo);axes[1].imshow(heat,cmap='jet',alpha=0.42,vmin=0,vmax=1)
    axes[1].set_title(f'Grad-CAM for: {names[target]}')
    for axis in axes:axis.axis('off')
    fig.tight_layout()
    dest=Path(args.output or Path(cfg['output_dir'])/'figures'/f'{architecture}_single_gradcam.png')
    dest.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest,dpi=160);plt.close(fig)
    print('Saved Grad-CAM:',dest.resolve())
    return dest


def check_command(cfg):
    """Offline check; no Hugging Face downloads or ImageNet weight downloads."""
    print('Device:',device_for())
    print('Config:',cfg['project_root'])
    torch.set_num_threads(min(torch.get_num_threads(),2))
    for architecture in EVAL_ARCHITECTURES:
        model=(load_scratch_checkpoint(cfg['scratch_checkpoint'])[0] if architecture==SCRATCH_ARCH
               else build_model(architecture,num_classes=200,pretrained=False).eval())
        with torch.inference_mode():
            shape=tuple(model(torch.zeros(1,3,64,64)).shape)
        assert shape==(1,200),f'{architecture} produced {shape}'
        print(f'{architecture}: verified output shape {shape}')
        del model
    print('CLI sanity check PASSED (this is not a training run).')


def build_parser():
    parser=argparse.ArgumentParser(description='Team classification evaluation and training CLI')
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('check','train','evaluate','predict','gradcam','compare','pipeline'):
        p=sub.add_parser(command,help={
            'check':'Check installed PyTorch and model output shapes without downloading data',
            'train':'Train/fine-tune one or both pretrained CNNs',
            'evaluate':'Evaluate pretrained and scratch models on official CUB-200 test split',
            'predict':'Classify a local bird image using a saved checkpoint',
            'gradcam':'Save a Grad-CAM explanation for a local image',
            'compare':'Compare aligned scratch CNN, ResNet-18 and EfficientNet-B0 predictions',
            'pipeline':'Train pretrained models, then evaluate and compare all three CNNs'}[command])
        p.add_argument('--task',choices=['classification'],default='classification')
        p.add_argument('--config',default='configs/classification.yaml')
        if command in ('train','evaluate','pipeline'):
            p.add_argument('--model',choices=['all',*(EVAL_ARCHITECTURES if command=='evaluate' else ARCHITECTURES)],default='all')
        if command in ('train','pipeline'):
            p.add_argument('--tune',action='store_true',help='Run required Grid and Random Search')
        if command in ('predict','gradcam'):
            p.add_argument('--checkpoint',required=True,help='Path to a trusted .pt checkpoint')
            p.add_argument('--image',required=True,help='Local path to a bird image')
            p.add_argument('--output',help='Optional JSON/PNG output file')
        if command=='evaluate':
            p.add_argument('--checkpoint',help='Evaluate a specific trusted checkpoint instead of --model')
        if command=='compare':
            p.add_argument('--models',nargs='+',choices=EVAL_ARCHITECTURES,
                           default=list(EVAL_ARCHITECTURES),help='Choose 2 or 3 previously evaluated models')
        if command=='predict':
            p.add_argument('--top-k',type=int,default=5)
        if command=='gradcam':
            p.add_argument('--class-id',type=int,default=None,help='Class to explain, default predicted class')
    return parser


def main(argv=None):
    parser=build_parser()
    args=parser.parse_args(argv)
    cfg=load_config(args.config)
    seed_everything(int(cfg['seed']))
    if args.command=='check':
        check_command(cfg)
    elif args.command=='train':
        train_command(args,cfg)
    elif args.command=='evaluate':
        evaluate_command(args,cfg)
    elif args.command=='predict':
        predict_command(args,cfg)
    elif args.command=='gradcam':
        gradcam_command(args,cfg)
    elif args.command=='compare':
        compare_command(cfg,args.models)
    elif args.command=='pipeline':
        selected=train_command(args,cfg)
        evaluate_command(args,cfg,selected_override=selected,with_gradcam=True)
        # Also evaluate the already-trained Member 2 checkpoint; never train it here.
        if set(selected)==set(ARCHITECTURES):
            evaluate_command(args,cfg,selected_override=[SCRATCH_ARCH])
            compare_command(cfg)
        else:
            print('Train/evaluation complete. Compare requires both models.')
        print('\nAll outputs:',Path(cfg['output_dir']).resolve())


if __name__=='__main__':
    main()
