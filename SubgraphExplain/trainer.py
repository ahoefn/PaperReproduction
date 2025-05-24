from tqdm import tqdm

import matplotlib.pyplot as plt

import numpy as np

import torch.nn as nn
import torch.optim

from mutag_model import MUTAGModel
from mutag_data import MUTAGDataLoader


class Trainer:
    def __init__(self, model: MUTAGModel, data: MUTAGDataLoader):
        self.model: MUTAGModel = model
        self.data: MUTAGDataLoader = data

        # set hyperparameters
        learning_rate = 0.005
        weight_decay = 0

        # keep trach of total epochs
        self.epochs = 0

        # optimizer and loss
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.loss_module = nn.CrossEntropyLoss()

        # loss logs
        self.train_loss: list[float] = []
        self.test_loss: list[float] = []

        # accuracy logs
        self.train_accuracy: list[float] = []
        self.test_accuracy: list[float] = []

    def step(self, epochs: int = 1) -> None:
        self.model.train()

        for epoch in tqdm(range(epochs)):
            # train step and log
            loss, accuracy = self.train_step()
            self.train_loss.append(loss)
            self.train_accuracy.append(accuracy)

            # test and log
            # test_loss = self.get_test_loss()
            # self.test_loss.append(test_loss)

        self.epochs += epochs

    def train_step(self) -> tuple[float, float]:
        loss_list: list[float] = []
        correct_prediction_bools = []
        for data in self.data.train_loader:
            # calculate loss
            outputs = self.model(data["x"], data["edge_index"], data["batch"])
            expected_outputs = data["y"]
            batch_loss: torch.Tensor = self.loss_module(outputs, expected_outputs)

            # update model
            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=2.0)
            self.optimizer.step()
            loss_list.append(batch_loss.item())

            # get accuracy by checking which output is higher
            _, predictions = torch.max(outputs, dim=-1)
            is_prediction_correct: torch.Tensor = predictions.eq(data["y"])
            correct_prediction_bools.append(is_prediction_correct.cpu().numpy())

        # calculate averages and return
        average_loss = sum(loss_list) / len(loss_list)
        average_accuracy = np.concatenate(correct_prediction_bools, axis=0).mean()
        return average_loss, average_accuracy

    def get_test_loss(self) -> float:
        data = self.data.test_loader.dataset
        outputs = self.model(data["x"], data["edge_index"], data["batch"])
        expected_outputs = data["y"]
        loss: torch.Tensor = self.loss_module(outputs, expected_outputs)
        return loss.item()

    # saving and loading models plus hyperparameters
    def save_model(self, name: str) -> None:
        save_data = {
            "model_type": type(self.model),
            "model": self.model.state_dict(),
            "optimizer_type": type(self.optimizer),
            "optimizer": self.optimizer.state_dict(),
            "loss_type": type(self.loss_module),
            "train_loss": self.train_loss,
            "test_loss": self.test_loss,
            "train_accuracy": self.train_accuracy,
            "test_accuracy": self.test_accuracy,
            "epochs": self.epochs,
        }

        torch.save(save_data, "SubgraphExplain\\checkpoints\\" + name + ".pt")

    def load_model(self, name: str, weights_only: bool = False) -> None:
        path = "SubgraphExplain\\checkpoints\\" + name + ".pt"

        # if weights_only, we don't load hyperparams etc
        if weights_only:
            self.model.load_state_dict(torch.load(path, weights_only=True))
            return

        load_data = torch.load(path, weights_only=False)

        # load model params
        if not load_data["model_type"] == type(self.model):
            print(
                f'Warning: model type of loaded data "{load_data["model_type"]}" does not match current model type "{type(self.model)}"'
            )
        self.model.load_state_dict(load_data["model"])

        # load optimizer
        if not load_data["optimizer_type"] == type(self.optimizer):
            print(
                f'Warning: optimizer type of loaded data "{load_data["optimizer_type"]}" does not match current optimizer type "{type(self.optimizer)}"'
            )
        self.optimizer.load_state_dict(load_data["optimizer"])

        # give warning of loss modules don't match
        if not load_data["loss_type"] == type(self.loss_module):
            print(
                f'Warning: loss module of loaded data "{load_data["loss_type"]}" does not match current loss module "{type(self.loss_module)}"'
            )

        # load logs en epoch counts
        self.train_loss = load_data["train_loss"]
        self.test_loss = load_data["test_loss"]
        self.train_accuracy = load_data["train_accuracy"]
        self.test_accuracy = load_data["test_accuracy"]
        self.epochs = load_data["epochs"]

    # plotting functions:
    def plot_losses(self, fig=plt.figure(figsize=(12, 12))) -> None:
        fig = plt.figure(fig.number)
        # plt.plot(self.test_loss, label="test loss")
        plt.plot(self.train_loss, label="train loss")

    def plot_accuracy(self, fig=plt.figure(figsize=(12, 12))) -> None:
        fig = plt.figure(fig.number)
        # plt.plot(self.test_loss, label="test loss")
        plt.plot(self.train_accuracy, label="train accuracy")
