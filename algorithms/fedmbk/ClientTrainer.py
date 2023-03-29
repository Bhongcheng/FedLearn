import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """
    

    def train(self):
        """Local training"""

        self.model.train()
        self.model.to(self.device)
        # Keep global model weights
        self._keep_global()

        targets = self.trainloader.dataset.targets
        sample_per_class = torch.zeros(self.num_classes)
        from collections import Counter
        counter = Counter(targets)
        for class_idx, count in counter.items():
            sample_per_class[class_idx] = count

        local_size = self.datasize

        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                self.optimizer.zero_grad()

                # forward pass
                data, targets = data.to(self.device), targets.to(self.device)
                logits = self.model(data)

                loss1 = balanced_softmax_loss(targets, logits, sample_per_class)

                loss2 = self.criterion(logits, targets)

                g_logits = self.g_model(data)
                loss3 = _ntd_loss(logits, g_logits, targets, self.num_classes, self.algo_params.tau)

                loss = loss1 + loss2 + loss3
                # backward pass
                loss.backward()
                self.optimizer.step()

        local_results = self._get_local_stats()

        return local_results, local_size

    def _keep_global(self):
        """Keep distributed global model's weight"""
        self.g_model = copy.deepcopy(self.model)

        for params in self.g_model.parameters():
            params.requires_grad = False


# https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification
def balanced_softmax_loss(labels, logits, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss

def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).to(logits.device)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits

def _ntd_loss(logits, dg_logits, targets, num_classes, tau):
    """Not-tue Distillation Loss"""

    # Get smoothed local model prediction
    logits = refine_as_not_true(logits, targets, num_classes)
    pred_probs = F.log_softmax(logits / tau, dim=1)

    # Get smoothed global model prediction
    with torch.no_grad():
        dg_logits = refine_as_not_true(dg_logits, targets, num_classes)
        dg_probs = torch.softmax(dg_logits / tau, dim=1)

    KLDiv = nn.KLDivLoss(reduction="batchmean")
    loss = (tau ** 2) * KLDiv(pred_probs, dg_probs)

    return loss