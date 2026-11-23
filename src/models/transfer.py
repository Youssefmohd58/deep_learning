"""Member 3: ImageNet-pretrained encoders with their final classifier replaced by a 200-class head."""
import torch
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights

TRANSFER_ARCHITECTURES = ('resnet18', 'efficientnet_b0')


def build_transfer_model(architecture, num_classes=200, pretrained=True):
    if architecture == 'resnet18':
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f'Unknown transfer model {architecture!r}; choose from {TRANSFER_ARCHITECTURES}')
    return model


def head_parameters(model, architecture):
    return model.fc.parameters() if architecture == 'resnet18' else model.classifier.parameters()


def set_trainable(model, architecture, stage):
    """'head': only the new classifier trains. 'finetune': the classifier plus the last stage.

    ResNet-18's last stage is layer4; EfficientNet-B0's are its last two feature blocks.
    Earlier layers keep their ImageNet features (generic edges/textures), which also limits
    overfitting on 4,795 training images.
    """
    if stage not in ('head', 'finetune'):
        raise ValueError(f'Invalid transfer stage {stage!r}')
    for p in model.parameters():
        p.requires_grad_(False)
    for p in head_parameters(model, architecture):
        p.requires_grad_(True)
    if stage == 'finetune':
        last = model.layer4 if architecture == 'resnet18' else model.features[-2:]
        for p in last.parameters():
            p.requires_grad_(True)


def frozen_bn_eval(model):
    """Keep running statistics fixed for BatchNorm layers whose weights are frozen."""
    for layer in model.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            params = list(layer.parameters(recurse=False))
            if params and all(not p.requires_grad for p in params):
                layer.eval()


def transfer_cam_layer(model, architecture):
    return model.layer4[-1] if architecture == 'resnet18' else model.features[-1]


def transfer_features(model, architecture, images):
    """Encoder embedding before the classifier (512-d ResNet-18, 1280-d EfficientNet-B0)."""
    if architecture == 'resnet18':
        x = model.maxpool(model.relu(model.bn1(model.conv1(images))))
        for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
            x = layer(x)
        return torch.flatten(model.avgpool(x), 1)
    if architecture == 'efficientnet_b0':
        return torch.flatten(model.avgpool(model.features(images)), 1)
    raise ValueError(architecture)
