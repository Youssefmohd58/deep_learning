def calculate_accuracy(outputs, labels):
    predictions = outputs.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return 100 * correct / labels.size(0)