import matplotlib.pyplot as plt
import random

import numpy as np
import math

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


def train_model(
    name: str, hidden_size: int, epochs: int, mutag_data: MUTAGDataLoader
) -> Trainer:
    model = MUTAGModel(hidden_size)

    trainer = Trainer(model, mutag_data)
    trainer.step(epochs)
    model_persistence.save_model(trainer, name)

    return trainer


def load_model(name: str, hidden_size: int, mutag_data: MUTAGDataLoader) -> Trainer:
    model = MUTAGModel(hidden_size)
    trainer = Trainer(model, mutag_data)
    model_persistence.load_model(trainer, name)
    return trainer


def print_accuracies(trainers: dict[str, Trainer]):
    """Print accuracies in general and for the last 20 epochs"""
    fig, axs = plt.subplots(2)
    for name, trainer in trainers.items():
        accuracy = trainer.test_accuracy
        axs[0].plot(accuracy, label=name)
        axs[1].plot(accuracy[-20:-1], label=name)

    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()

    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    plt.show()


if __name__ == "__main__":
    initialize_seeds()

    data = MUTAGDataLoader()

    trainer128 = load_model("0524-2000e-2", 128, data)
    trainer32 = load_model("0524-2000e-hidden32", 32, data)
    trainer16 = load_model("0524-2000e-hidden16", 16, data)
    trainer_dict: dict[str, Trainer] = {
        "128": trainer128,
        "32": trainer32,
        "16": trainer16,
    }

    print(f"Final test accuracy of 128 size: {trainer128.test_accuracy[-1]:0.3f}")
    print(f"Best test accuracy of 128 size: {max(trainer128.test_accuracy):0.3f}")

    print(f"Final test accuracy of 32 size: {trainer32.test_accuracy[-1]:0.3f}")
    print(f"Best test accuracy of 32 size: {max(trainer32.test_accuracy):0.3f}")

    print(f"Final test accuracy of 16 size: {trainer16.test_accuracy[-1]:0.3f}")
    print(f"Best test accuracy of 16 size: {max(trainer16.test_accuracy):0.3f}")

    print_accuracies(trainer_dict)
