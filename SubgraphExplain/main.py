import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from mutag_data import MUTAGDataLoader
from mutag_model import MUTAGModel
from trainer import Trainer


if __name__ == "__main__":
    mutag_data = MUTAGDataLoader()
    model = MUTAGModel()

    trainer = Trainer(model, mutag_data)
    trainer.step(2000)

    print(f"Final loss is: {trainer.train_loss[-1]}")
    print(f"Final accuracy is: {trainer.train_accuracy[-1]}")

    trainer.plot_losses()
    trainer.plot_accuracy()
    plt.show()
