import torch
from torch_geometric.datasets import TUDataset
import torch_geometric.utils
import torch_geometric.transforms as tgt

import mutag_data
import mutag_model

import torch_geometric.nn as tgnn

if __name__ == "__main__":
    mutag = mutag_data.MUTAGDataLoader()
    # mutag.plot_graphs()

    model = mutag_model.MUTAGModel()
    # print(model)

    data = next(iter(mutag.train_loader))
    x = model(data["x"], data["edge_index"])
    print(x)
    for d in mutag.train_loader:
        print(d)
