import matplotlib.pyplot as plt

import torch
import torch_geometric.data as tgd

from mutag_data import MUTAGDataLoader
from mutag_model import MUTAGModel
from trainer import Trainer
import model_persistence


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

    data = MUTAGDataLoader()
    model = MUTAGModel()

    trainer = train_model("0524-2000e", 2000, data)

    fig, axs = plt.subplots(2)

    trainer.plot_losses(axs[0])
    trainer.plot_accuracy(axs[1])

    plt.show()
