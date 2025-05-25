import torch
import torch.nn as nn

from torch_geometric.data import Data
import torch_geometric.nn as tgnn


class MUTAGModel(nn.Module):
    """
    Network as used in the paper, three GCN layers all with output sizes of hidden_size followed by a max pooling. Afterwards, a single hidden_size-2 linear layer to aggregate the outputs. Activation function for the GCN layers is ReLU.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size: int = hidden_size

        # define gnn layers
        self.gnn_layer_input: nn.Module = tgnn.GCNConv(7, hidden_size)
        self.gnn_layer_hidden1: nn.Module = tgnn.GCNConv(hidden_size, hidden_size)
        self.gnn_layer_hidden2: nn.Module = tgnn.GCNConv(hidden_size, hidden_size)

        # define linear layer:
        self.linear_layer: nn.Module = nn.Linear(hidden_size, 2)

        # activation function and pooling
        self.gnn_activation = nn.ReLU()

    def forward(self, x, edge_index, batch_data):
        # graph layers:
        x = self.gnn_activation(self.gnn_layer_input(x, edge_index))
        x = self.gnn_activation(self.gnn_layer_hidden1(x, edge_index))
        x = self.gnn_activation(self.gnn_layer_hidden2(x, edge_index))

        # pooling
        x = tgnn.pool.global_max_pool(x, batch_data)

        # linear layer:
        x = self.linear_layer(x)

        return x
