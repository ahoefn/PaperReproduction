import torch

import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

# for graph plotting
import networkx as nx
import matplotlib.pyplot as plt


class MUTAGDataLoader:
    def __init__(self):
        dataset = TUDataset("SubgraphExplain\\data", name="MUTAG")
        dataset.shuffle()
        train_data = dataset[:150]
        test_data = dataset[150:]
        self.train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(test_data)

    def plot_graphs(self):
        """
        plots 4 random graphs from the dataset
        """
        data_iter = iter(self.train_loader.dataset)
        datagraphs = [to_networkx(next(data_iter)) for i in range(4)]
        fig, axes = plt.subplots(2, 2)
        for index, graph in enumerate(datagraphs):
            x = index % 2
            y = int((index - x) / 2)
            plt.sca(axes[x, y])
            nx.draw(graph)

        plt.show()
