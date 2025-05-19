import matplotlib.pyplot as plt

from mutag_data import MUTAGDataLoader
from mutag_model import MUTAGModel
from trainer import Trainer


if __name__ == "__main__":
    mutag_data = MUTAGDataLoader()
    model = MUTAGModel()
    trainer = Trainer(model, mutag_data)
    trainer.step(100)
    trainer.plot_losses()
    plt.show()

    # data = next(iter(mutag_data.train_loader))
    # x = model(data["x"], data["edge_index"], data.batch)
    # print(x.size())

    # loss1 = trainer.loss_module(x.squeeze(), data["y"].float())
    # print(loss1)

    # y = data["y"].float().unsqueeze(1)
    # # print(y)
    # loss2 = trainer.loss_module(x, y)
    # print(loss2)
