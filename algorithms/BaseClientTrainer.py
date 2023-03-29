from .measures import *
import copy
from torch import nn
import torch


__all__ = ["BaseClientTrainer"]


class BaseClientTrainer:
    def __init__(self, algo_params, model, local_epochs, device, num_classes):
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """

        # algorithm-specific parameters
        self.algo_params = algo_params

        # model & optimizer
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
        self.criterion = nn.CrossEntropyLoss()
        self.local_epochs = local_epochs
        self.device = device
        self.num_classes = num_classes
        self.datasize = None
        self.trainloader = None
        self.testloader = None
    

    def _get_local_stats(self):
        local_results = {}

        local_results["train_acc"] = evaluate_model(
            self.model, self.trainloader, self.device
        )
        local_results["classwise_accuracy"], local_results["test_acc"] = evaluate_model_classwise(
            self.model, self.testloader, self.num_classes, device=self.device,
        )

        return local_results
    

    def download_global(self, server_weights, server_optimizer):
        """Load model & Optimizer"""
        self.model.load_state_dict(server_weights)
        self.optimizer.load_state_dict(server_optimizer)
    

    def upload_local(self):
        """Uploads local model's parameters"""
        local_weights = copy.deepcopy(self.model.state_dict())

        return local_weights
    

    def reset(self):
        """Clean existing setups"""
        self.datasize = None
        self.trainloader = None
        self.testloader = None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
