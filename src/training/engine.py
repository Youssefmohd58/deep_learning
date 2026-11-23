"""The single training/validation loop used by every model and by hyperparameter tuning."""
from __future__ import annotations

import torch

from src.models.transfer import frozen_bn_eval


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, scheduler=None,
              grad_clip=None, freeze_bn=False) -> dict:
    """One pass over ``loader``. Trains if an optimizer is given, otherwise only evaluates.

    Returns mean loss and accuracy (0-1). ``scheduler`` is stepped once per batch
    (OneCycle); ``freeze_bn`` keeps frozen BatchNorm layers in eval mode (transfer learning).
    """
    training = optimizer is not None
    model.train(training)
    if training and freeze_bn:
        frozen_bn_eval(model)
    use_amp = device.type == 'cuda'
    loss_sum, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            x = batch['image'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if grad_clip:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            loss_sum += float(loss.detach()) * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    if total == 0:
        raise ValueError('DataLoader is empty')
    return {'loss': loss_sum / total, 'accuracy': correct / total}
