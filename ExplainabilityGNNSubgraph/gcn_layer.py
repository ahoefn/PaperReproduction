import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    """Basic GraphCNN layer, based on https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html"""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.projection: nn.Module = nn.Linear(input_size, output_size)

    def forward(self, node_inputs: torch.Tensor, adjacency_matrix: torch.Tensor):
        """
        Inputs:
            feature_count - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adjacency_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
            Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
            Shape: [batch_size, num_nodes, num_nodes]
        """
        neighbour_count: torch.Tensor = adjacency_matrix.sum(dim=-1, keepdim=True)
        node_inputs = self.projection(node_inputs)
        node_inputs = torch.bmm(adjacency_matrix, node_inputs)
        node_inputs = node_inputs / neighbour_count
