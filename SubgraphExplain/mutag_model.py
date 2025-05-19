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

        # define gnn layers
        self.gnn_layer_input: nn.Module = tgnn.GCNConv(7, 128)
        self.gnn_layer_hidden1: nn.Module = tgnn.GCNConv(128, 128)
        self.gnn_layer_hidden2: nn.Module = tgnn.GCNConv(128, 128)

        # define linear layer:
        self.linear_layer: nn.Module = nn.Linear(128, 1)

        # activation function and pooling
        self.gnn_activation = nn.ReLU()
        self.linear_activation = nn.ELU()

    def forward(self, x, edge_index, batch_data):
        # graph layers:
        x = self.gnn_activation(self.gnn_layer_input(x, edge_index))
        x = self.gnn_activation(self.gnn_layer_hidden1(x, edge_index))
        x = self.gnn_activation(self.gnn_layer_hidden2(x, edge_index))

        # pooling
        x = tgnn.pool.global_max_pool(x, batch_data)

        # linear layer:
        x = self.linear_activation(self.linear_layer(x))

        return x.squeeze()
