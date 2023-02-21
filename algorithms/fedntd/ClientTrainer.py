import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer

__all__ = ["ClientTrainer"]


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

        local_size = self.datasize

        global_model = copy.deepcopy(self.model)
        for par in global_model.parameters():
            par.requirese_grad = False

        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                self.optimizer.zero_grad()

                # forward pass
                data, targets = data.to(self.device), targets.to(self.device)
                logits, dg_logits = self.model(data), global_model(data)

                loss_ce = self.criterion(logits, targets)
                loss_ntd = _ntd_loss(logits, dg_logits, targets, self.num_classes, self.algo_params.tau)
                loss = loss_ce + self.algo_params.beta * loss_ntd

                # backward pass
                loss.backward()
                self.optimizer.step()

        local_results = self._get_local_stats()

        return local_results, local_size