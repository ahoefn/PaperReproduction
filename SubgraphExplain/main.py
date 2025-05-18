import torch
from torch_geometric.datasets import TUDataset

if __name__ == "__main__":
    dataset = TUDataset("ExplainabilityGNNSubgraph\data", "MUTAG")
    print(dataset)
