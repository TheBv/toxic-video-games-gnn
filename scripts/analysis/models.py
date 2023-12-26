import torch
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential, Dropout
from torch.optim.lr_scheduler import ReduceLROnPlateau

class BinaryClass(torch.nn.Module):
    def __init__(self, deg):
        super().__init__()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(2):
            conv = PNAConv(in_channels=5, out_channels=5,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=1, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(5))

        self.mlp = Sequential(Linear(5, 5), ReLU(), Linear(5, 10),  ReLU(),
                              Linear(10, 1))

    def forward(self, x, edge_index, edge_attr, batch):
        x = x
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch)
        return self.mlp(x)

class MultiClass(torch.nn.Module):
    def __init__(self, deg, out):
        super().__init__()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(2):
            conv = PNAConv(in_channels=5, out_channels=5,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=1, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(5))

        self.mlp = Sequential(Linear(5, 5), ReLU(), Linear(5, 10), ReLU(),
                              Linear(10, out))

    def forward(self, x, edge_index, edge_attr, batch):
        x = x
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch)
        return self.mlp(x)