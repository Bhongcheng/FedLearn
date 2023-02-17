import torch
from .utils import *

__all__ = ["evaluate_model", "evaluate_model_classwise"]


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Evaluate model accuracy for the given dataloader"""

    model.eval()
    model.to(device)

    count = 0
    correct = 0

    for data, targets in dataloader:
        data, targets = data.to(device), targets.to(device)

        logits = model(data)
        pred = logits.max(dim=1)[1]

        correct += (targets == pred).sum().item()
        count += data.size(0)

    accuracy = round(correct / count, 4)

    return accuracy


@torch.no_grad()
def evaluate_model_classwise(
    model, dataloader, num_classes, device,
):
    """Evaluate class-wise accuracy for the given dataloader."""

    model.eval()
    model.to(device)

    classwise_count = torch.Tensor([0 for _ in range(num_classes)]).to(device)
    classwise_correct = torch.Tensor([0 for _ in range(num_classes)]).to(device)

    for data, targets in dataloader:
        data, targets = data.to(device), targets.to(device)

        logits = model(data)
        preds = logits.max(dim=1)[1]

        for class_idx in range(num_classes):
            class_elem = targets == class_idx
            classwise_count[class_idx] += class_elem.sum().item()
            classwise_correct[class_idx] += (targets == preds)[class_elem].sum().item()

    classwise_accuracy = classwise_correct / classwise_count
    accuracy = round(classwise_accuracy.mean().item(), 4)

    return classwise_accuracy.cpu(), accuracy
