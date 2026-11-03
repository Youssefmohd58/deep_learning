"""Member 3 pretrained CNN encoders and 200-class classification heads."""
import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights, EfficientNet_B0_Weights

ARCHITECTURES = ('resnet18', 'efficientnet_b0')


def build_model(architecture, num_classes=200, pretrained=True):
    if architecture == 'resnet18':
        m = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif architecture == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features,num_classes)
    else:
        raise ValueError(f'Unknown model: {architecture}; choose {ARCHITECTURES}')
    return m


def set_trainable(model, architecture, stage):
    """head: new classifier; finetune: head plus final convolutional stage."""
    if stage not in ('head','finetune'):
        raise ValueError(f'Invalid stage: {stage}')
    for p in model.parameters():
        p.requires_grad_(False)
    if architecture=='resnet18':
        for p in model.fc.parameters():
            p.requires_grad_(True)
        if stage=='finetune':
            for p in model.layer4.parameters():
                p.requires_grad_(True)
    elif architecture=='efficientnet_b0':
        for p in model.classifier.parameters():
            p.requires_grad_(True)
        if stage=='finetune':
            for p in model.features[-2:].parameters():
                p.requires_grad_(True)
    else:
        raise ValueError(architecture)


def frozen_bn_eval(model):
    """Keep running statistics fixed for frozen BatchNorm layers in train mode."""
    for layer in model.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            params=list(layer.parameters(recurse=False))
            if params and all(not p.requires_grad for p in params):
                layer.eval()


def get_cam_layer(model, architecture):
    return model.layer4[-1] if architecture=='resnet18' else model.features[-1]


def trainable_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(path, device='cpu'):
    """Load checkpoints produced by this package; also understands earlier M3 checkpoint schema.

    Only open checkpoints created by you or another trusted person. PyTorch checkpoint
    deserialization can execute code for legacy pickled contents.
    """
    try:
        payload=torch.load(path,map_location='cpu',weights_only=True)
    except Exception as exc:
        raise RuntimeError('Cannot load checkpoint safely. Use a trusted state_dict checkpoint.') from exc
    arch=payload['architecture']
    names=payload['class_names']
    model=build_model(arch,num_classes=len(names),pretrained=False)
    model.load_state_dict(payload['model_state_dict'],strict=True)
    return model.to(device).eval(),names,payload


def extract_features(model, architecture, images):
    """Get encoder vectors before the classifier for possible downstream reuse."""
    if architecture == 'resnet18':
        x = model.conv1(images)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        for layer in (model.layer1,model.layer2,model.layer3,model.layer4):
            x=layer(x)
        return torch.flatten(model.avgpool(x),1)
    if architecture == 'efficientnet_b0':
        return torch.flatten(model.avgpool(model.features(images)),1)
    raise ValueError(architecture)
