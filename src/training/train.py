import torch


def train_one_epoch(model, train_loader, criterion, optimizer, device,
                    scheduler=None, scaler=None):
    model.train()

    total_loss = 0
    total_correct = 0
    total_images = 0
    use_amp = scaler is not None

    for batch in train_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_images += images.size(0)

    if total_images == 0:
        raise ValueError("Training DataLoader is empty")

    return total_loss / total_images, 100 * total_correct / total_images


def validate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0
    total_correct = 0
    total_images = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_images += images.size(0)

    if total_images == 0:
        raise ValueError("Validation DataLoader is empty")

    return total_loss / total_images, 100 * total_correct / total_images