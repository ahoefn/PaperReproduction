import matplotlib.pyplot as plt
import random

import numpy as np

import torch
from torch.ao.quantization import prepare_for_propagation_comparison
import torch_geometric.data as tgd

from mutag_data import MUTAGDataLoader
from mutag_model import MUTAGModel
from trainer import Trainer
import model_persistence


def initialize_seeds():
    torch.manual_seed(2101)
    random.seed(2101)


def train_model(name: str, epochs: int, mutag_data: MUTAGDataLoader) -> Trainer:
    model = MUTAGModel()

    trainer = Trainer(model, mutag_data)
    trainer.step(epochs)
    model_persistence.save_model(trainer, name)

    return trainer


def load_model(name: str, mutag_data: MUTAGDataLoader) -> Trainer:
    model = MUTAGModel()
    trainer = Trainer(model, mutag_data)
    model_persistence.load_model(trainer, name)
    return trainer


if __name__ == "__main__":
    initialize_seeds()

    data = MUTAGDataLoader()
    model = MUTAGModel()

    trainer = train_model("0524-2000e-2", 2000, data)
    # trainer = load_model("0524-2000e", data)

    print(f"Train accuracy: {trainer.train_accuracy[-1]}")
    print(f"Test accuracy: {trainer.test_accuracy[-1]}")
    print(f"Train loss: {trainer.train_loss[-1]}")
    print(f"Test loss: {trainer.test_loss[-1]}")

    fig, axs = plt.subplots(2)
    trainer.plot_losses(axs[0])
    trainer.plot_accuracy(axs[1])

    plt.show()
