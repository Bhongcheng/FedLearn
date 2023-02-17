import torch
import torch.nn.functional as F

__all__ = [
    "sampled_clients_identifier",
]



def sampled_clients_identifier(data_distributed, sampled_clients):
    """Identify local datasets information (distribution, size)"""

    local_dist_list, local_size_list = [], []

    for client_idx in sampled_clients:
        local_dist = torch.Tensor(data_distributed["data_map"])[client_idx]
        local_dist = F.normalize(local_dist, dim=0, p=1)
        local_dist_list.append(local_dist.tolist())

        local_size = data_distributed["local"][client_idx]["datasize"]
        local_size_list.append(local_size)

    return local_dist_list, local_size_list