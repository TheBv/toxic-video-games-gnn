from torch_geometric.utils import degree
import torch

def degree_pna(dataset):
  max_degree = -1
  for data in dataset:
      d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
      max_degree = max(max_degree, int(d.max()))

  # Compute the in-degree histogram tensor
  deg = torch.zeros(max_degree + 1, dtype=torch.long)
  for data in dataset:
      d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
      deg += torch.bincount(d, minlength=deg.numel())
  return deg