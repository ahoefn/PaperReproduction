import matplotlib.pyplot as plt

from mutag_data import MUTAGDataLoader
from mutag_model import MUTAGModel
from trainer import Trainer


if __name__ == "__main__":
    mutag_data = MUTAGDataLoader()
    model = MUTAGModel()
    trainer = Trainer(model, mutag_data)
    # trainer.step(2000)
    # trainer.plot_losses()
    # plt.show()

    data = next(iter(mutag_data.train_loader))
    x = model(data["x"], data["edge_index"], data.batch)
