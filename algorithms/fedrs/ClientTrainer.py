import os
import sys
import torch
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer

__all__ = ["ClientTrainer"]


def get_dist_vec(dataloader, num_classes):
    """Calculate distribution vector for local set"""
    targets = dataloader.dataset.targets
    dist_vec = torch.zeros(num_classes)
    counter = Counter(targets)

    for class_idx, count in counter.items():
        dist_vec[class_idx] = count

    dist_vec /= len(targets)

    return dist_vec



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

        dist = get_dist_vec(self.trainloader, self.num_classes)
        cdist = dist / dist.max()
        cdist = cdist * (1.0 - self.algo_params.alpha) + self.algo_params.alpha
        cdist = cdist.reshape((1, -1))
        cdist = cdist.to(self.device)

        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                self.optimizer.zero_grad()

                # forward pass
                data, targets = data.to(self.device), targets.to(self.device)
                features, _ = self.model(data, get_features = True)
                w = self.model.classifier.weight

                output = cdist * features.mm(w.transpose(0, 1))
                loss = self.criterion(output, targets)

                # backward pass
                loss.backward()
                self.optimizer.step()

        local_results = self._get_local_stats()

        return local_results, local_size