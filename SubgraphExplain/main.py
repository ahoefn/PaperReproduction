import matplotlib.pyplot as plt

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
    trainer = train_model("test", 10, data)
    trainer_loaded = load_model("test", data)

    # check if model paramaters are indeed the same
    for trained_params, loaded_params in zip(
        trainer.model.parameters(), trainer_loaded.model.parameters()
    ):
        if trained_params.ne(loaded_params).sum() > 0:
            print(
                f"Error: not all parameters are equal. Problem in {trained_params} and {loaded_params}"
            )

    fig_loss = plt.figure(
        figsize=(
            12,
            12,
        )
    )
    trainer.plot_losses(fig_loss)
    trainer_loaded.train_loss = [x + 0.1 for x in trainer_loaded.train_loss]
    trainer_loaded.plot_losses(fig_loss)

    fig_accuracy = plt.figure(
        figsize=(
            12,
            12,
        )
    )
    trainer.plot_accuracy(fig_accuracy)
    trainer_loaded.plot_accuracy(fig_accuracy)
    plt.show()
