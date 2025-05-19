import torch
import torch.nn as nn

from torch_geometric.data import Data
import torch_geometric.nn as tgnn


class MUTAGModel(nn.Module):
    """
    Network as used in the paper, three GCN layers all with output sizes of 128 followed by a max pooling.
    Activation function is ReLU
    """

    def __init__(self):
        super().__init__()

        # define layers
        self.layer_input: nn.Module = tgnn.GCNConv(7, 128)
        self.layer_hidden1: nn.Module = tgnn.GCNConv(128, 128)
        self.layer_hidden2: nn.Module = tgnn.GCNConv(128, 128)

        # activation function and pooling
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, batch_data):
        # graph layers:
        x = self.activation(self.layer_input(x, edge_index))
        x = self.activation(self.layer_hidden1(x, edge_index))
        x = self.activation(self.layer_hidden2(x, edge_index))

        # pooling
        x, _ = torch.max(x, dim=-1)
        print(x.size())
        x = tgnn.pool.global_max_pool(x, batch_data)
        print(x.size())
        return x
