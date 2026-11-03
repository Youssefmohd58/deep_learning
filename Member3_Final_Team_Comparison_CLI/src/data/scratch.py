"""Member 2's *inference* preprocessing for scratch CNN, exact 128px specification.

Training augmentation is irrelevant for final test evaluation. Model-specific
test preprocessing is required because Member 2 trained at 128px, unlike the
pretrained 224px models. Both models see the same source CUB-200 test images.
"""
import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from src.data.cub import CUBClassificationDataset,make_loader,IMAGENET_MEAN,IMAGENET_STD

def scratch_eval_transform(size=128):
    return T.Compose([
        T.Lambda(lambda im: im.convert('RGB')),
        T.Resize(round(size*(256/224)),interpolation=InterpolationMode.BILINEAR),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN,IMAGENET_STD),
    ])

def scratch_test_loader(pretrained_test_loader,cfg):
    dataset=pretrained_test_loader.dataset
    source=dataset.data
    result=CUBClassificationDataset(source,range(len(source)),
             scratch_eval_transform(int(cfg.get('scratch_image_size',128))))
    return make_loader(result,batch_size=cfg['batch_size'],shuffle=False,
                       num_workers=cfg['num_workers'],seed=cfg['seed'])
