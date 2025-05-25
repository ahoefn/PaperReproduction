"""Module for loading and saving models"""

import pathlib

import torch

from trainer import Trainer


def save_model(trainer: Trainer, name: str) -> None:
    """saving model plus hyperparameters"""

    save_data = {
        "model_type": type(trainer.model),
        "model": trainer.model.state_dict(),
        "hidden_size": trainer.model.hidden_size,
        "optimizer_type": type(trainer.optimizer),
        "optimizer": trainer.optimizer.state_dict(),
        "loss_type": type(trainer.loss_module),
        "train_loss": trainer.train_loss,
        "test_loss": trainer.test_loss,
        "train_accuracy": trainer.train_accuracy,
        "test_accuracy": trainer.test_accuracy,
        "epochs": trainer.epochs,
    }

    # make checkpoints directory if it does not exist
    pathlib.Path("SubgraphExplain\\checkpoints").mkdir(parents=True, exist_ok=True)

    # save model
    torch.save(save_data, "SubgraphExplain\\checkpoints\\" + name + ".pt")


def load_model(trainer: Trainer, name: str, weights_only: bool = False) -> Trainer:
    """loading model plus hyperparameters"""

    path = "SubgraphExplain\\checkpoints\\" + name + ".pt"

    # if weights_only, we don't load hyperparams etc
    if weights_only:
        trainer.model.load_state_dict(torch.load(path, weights_only=True))
        return trainer

    load_data = torch.load(path, weights_only=False)

    # load model params
    if not load_data["model_type"] == type(trainer.model):
        print(
            f'Warning: model type of loaded data "{load_data["model_type"]}" does not match current model type "{type(trainer.model)}"'
        )
    trainer.model.load_state_dict(load_data["model"])

    # load optimizer
    if not load_data["optimizer_type"] == type(trainer.optimizer):
        print(
            f'Warning: optimizer type of loaded data "{load_data["optimizer_type"]}" does not match current optimizer type "{type(trainer.optimizer)}"'
        )
    trainer.optimizer.load_state_dict(load_data["optimizer"])

    # give warning of loss modules don't match
    if not load_data["loss_type"] == type(trainer.loss_module):
        print(
            f'Warning: loss module of loaded data "{load_data["loss_type"]}" does not match current loss module "{type(trainer.loss_module)}"'
        )

    # load logs en epoch counts
    trainer.train_loss = load_data["train_loss"]
    trainer.test_loss = load_data["test_loss"]
    trainer.train_accuracy = load_data["train_accuracy"]
    trainer.test_accuracy = load_data["test_accuracy"]
    trainer.epochs = load_data["epochs"]

    return trainer
