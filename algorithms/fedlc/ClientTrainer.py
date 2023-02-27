import os
import sys
import torch
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer

__all__ = ["ClientTrainer"]


def logit_calibrated_loss(labels_count, logits, labels, tau):
    cal_logit = torch.exp(
        logits
        - (
            tau
            * torch.pow(labels_count, -1 / 4)
            .unsqueeze(0)
            .expand((logits.shape[0], -1))
        )
    )
    y_logit = torch.gather(cal_logit, dim=-1, index=labels.unsqueeze(1))
    loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))

    return loss.sum() / logits.shape[0]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """
    

    def train(self):
        """Local training"""

        labels_count = torch.zeros(self.num_classes, device=self.device)
        labels_counter = Counter(self.trainloader.dataset.targets)
        for cls, count in labels_counter.items():
            labels_count[cls] = max(1e-8, count)

        self.model.train()
        self.model.to(self.device)

        local_size = self.datasize

        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                self.optimizer.zero_grad()

                # forward pass
                data, targets = data.to(self.device), targets.to(self.device)
                output = self.model(data)
                loss = logit_calibrated_loss(labels_count, output, targets, self.algo_params.tau)

                # backward pass
                loss.backward()
                self.optimizer.step()

        local_results = self._get_local_stats()

        return local_results, local_size