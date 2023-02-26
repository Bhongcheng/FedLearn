import os
import sys
import copy
from torch import nn
import torch

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

        # Keep global model and prev local model
        self._keep_global()
        self._keep_prev_local()

        self.model.train()
        self.model.to(self.device)

        local_size = self.datasize

        ce = nn.CrossEntropyLoss()
        sim = nn.CosineSimilarity(dim = -1)

        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                self.optimizer.zero_grad()

                # forward pass
                data, targets = data.to(self.device), targets.to(self.device)
                z, logits = self.model(data, get_features=True)
                loss1 = ce(logits, targets)

                # for moon contrast
                z_prev, _ = self.prev_model(data, get_features=True)
                z_g, _ = self.g_model(data, get_features=True)
                positive = sim(z, z_g).reshape(-1, 1)
                negative = sim(z, z_prev).reshape(-1, 1)
                moon_logits = torch.cat([positive, negative], dim=1)
                moon_logits /= self.algo_params.tau
                moon_labels = torch.zeros(z.size(0)).to(self.device).long()
                loss2 = ce(moon_logits, moon_labels)

                loss = loss1 + self.algo_params.mu * loss2
                
                # backward pass
                loss.backward()
                self.optimizer.step()

        local_results = self._get_local_stats()

        return local_results, local_size
    
    def _keep_global(self):
        """Keep distributed global model's weight"""
        self.g_model = copy.deepcopy(self.model)
        self.g_model.to(self.device)

        for params in self.g_model.parameters():
            params.requires_grad = False

    def download_global(self, server_weights, server_optimizer, prev_weights):
        """Load model & Optimizer"""
        self.model.load_state_dict(server_weights)
        self.optimizer.load_state_dict(server_optimizer)
        self.prev_weights = prev_weights

    def _keep_prev_local(self):
        """Keep distributed global model's weight"""
        self.prev_model = copy.deepcopy(self.model)
        self.prev_model.load_state_dict(self.prev_weights)
        self.prev_model.to(self.device)

        for params in self.prev_model.parameters():
            params.requires_grad = False