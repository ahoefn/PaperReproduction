from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.axes as pltax

import numpy as np

import torch
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
            test_loss, test_accuracy = self.test_step()
            self.test_loss.append(test_loss)
            self.test_accuracy.append(test_accuracy)

        self.epochs += epochs

    def train_step(self) -> tuple[float, float]:
        self.model.train()

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

    @torch.no_grad()
    def test_step(self) -> tuple[float, float]:
        self.model.eval()

        loss_list: list[float] = []
        correct_prediction_bools = []
        for data in self.data.test_loader:
            # calculate loss
            outputs = self.model(data["x"], data["edge_index"], data["batch"])
            expected_outputs = data["y"]
            batch_loss: torch.Tensor = self.loss_module(outputs, expected_outputs)

            # append loss
            loss_list.append(batch_loss.item())

            # get accuracy by checking which output is higher
            _, predictions = torch.max(outputs, dim=-1)
            is_prediction_correct: torch.Tensor = predictions.eq(data["y"])
            correct_prediction_bools.append(is_prediction_correct.cpu().numpy())

        # calculate averages and return
        average_loss = sum(loss_list) / len(loss_list)
        average_accuracy = np.concatenate(correct_prediction_bools, axis=0).mean()
        return average_loss, average_accuracy

    # plotting functions:
    def plot_losses(self, ax: pltax.Axes) -> None:
        ax.plot(self.train_loss, label="train loss")
        ax.plot(self.test_loss, label="test loss")

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()

    def plot_accuracy(self, ax: pltax.Axes) -> None:
        ax.plot(self.train_accuracy, label="train accuracy")
        ax.plot(self.test_accuracy, label="test accuracy")

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.legend()
