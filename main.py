import argparse
import pprint
import warnings
import torch
import algorithms
from torch import optim
from torch.optim import lr_scheduler
import numpy as np
import random
from train_tools import *
import os
import wandb
from utils import *

warnings.filterwarnings("ignore")

# Set torch base print precision
torch.set_printoptions(10)

ALGO = {
    "fedavg": algorithms.fedavg.Server,
    "fedlc": algorithms.fedlc.Server,
    "fedrs": algorithms.fedrs.Server,
    "fedntd": algorithms.fedntd.Server,
    "fedprox": algorithms.fedprox.Server,
}

SCHEDULER = {
    "step": lr_scheduler.StepLR,
    "multistep": lr_scheduler.MultiStepLR,
    "cosine": lr_scheduler.CosineAnnealingLR,
}


def _random_seeder(seed):
    """Fix randomness"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_setups(args):
    """Get train configuration"""

    # Fix randomness for data distribution
    np.random.seed(19980815)
    random.seed(19980815)

    # Distribute the data to clients
    data_distributed = data_distributer(**args.data_setups)

    # Fix randomness for experiment
    _random_seeder(args.train_setups.seed)
    model = create_models(
        args.train_setups.model.name,
        args.data_setups.dataset_name,
        **args.train_setups.model.params,
    )

    # Optimization setups
    optimizer = optim.SGD(model.parameters(), **args.train_setups.optimizer.params)
    scheduler = None

    if args.train_setups.scheduler.enabled:
        scheduler = SCHEDULER[args.train_setups.scheduler.name](
            optimizer, **args.train_setups.scheduler.params
        )

    # Algorith-specific global server container
    algo_params = args.train_setups.algo.params
    server = ALGO[args.train_setups.algo.name](
        algo_params,
        model,
        data_distributed,
        optimizer,
        scheduler,
        **args.train_setups.scenario,
    )

    return server


def main(args):
    """Execute experiment"""

    # Load the configuration
    server = _get_setups(args)

    # Conduct FL
    server.run()

    # Save the final global model
    model_path = os.path.join(wandb.run.dir, "model.pth")
    torch.save(server.model.state_dict(), model_path)

    # Upload model to wandb
    wandb.save(model_path)


# Parser arguments for terminal execution
parser = argparse.ArgumentParser(description="Process Configs")
parser.add_argument("--config_path", default="./config/fedprox.json", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--n_clients", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--partition_method", type=str)
parser.add_argument("--partition_s", type=int)
parser.add_argument("--partition_alpha", type=float)
parser.add_argument("--model_name", type=str)
parser.add_argument("--n_rounds", type=int)
parser.add_argument("--sample_ratio", type=float)
parser.add_argument("--local_epochs", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--momentum", type=float)
parser.add_argument("--wd", type=float)
parser.add_argument("--algo_name", type=str)
parser.add_argument("--device", type=str)
parser.add_argument("--seed", type=int)
parser.add_argument("--group", type=str)
parser.add_argument("--exp_name", type=str)
args = parser.parse_args()

args.partition_s = 2

if __name__ == "__main__":
    # Load configuration from .json file
    opt = ConfLoader(args.config_path).opt

    # Overwrite config by parsed arguments
    opt = config_overwriter(opt, args)

    # Print configuration dictionary pretty
    print("")
    print("=" * 50 + " Configuration " + "=" * 50)
    pp = pprint.PrettyPrinter(compact=True)
    pp.pprint(opt)
    print("=" * 120)

    # Initialize W&B
    wandb.init(config=opt, **opt.wandb_setups)

    # How many batches to wait before logging training status
    wandb.config.log_interval = 10

    # Execute expreiment
    main(opt)