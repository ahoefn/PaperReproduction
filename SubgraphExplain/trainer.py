from tqdm import tqdm

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim

from mutag_model import MUTAGModel
from mutag_data import MUTAGDataLoader


class Trainer:
    def __init__(self, model: MUTAGModel, data: MUTAGDataLoader):
        self.model: MUTAGModel = model
        self.data: MUTAGDataLoader = data

        # optimizer and loss
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
        self.loss_module = nn.BCEWithLogitsLoss()

        # loss logs
        self.train_loss: list[float] = []
        self.test_loss: list[float] = []

    def step(self, epochs: int = 1) -> None:
        self.model.train()

        for epoch in tqdm(range(epochs)):
            # train step and log
            loss = self.train_step()
            self.train_loss.append(loss)

            # test and log
            # test_loss = self.get_test_loss()
            # self.test_loss.append(test_loss)

    def train_step(self) -> float:
        for data in self.data.train_loader:
            self.optimizer.zero_grad()

            # calculate loss
            outputs = self.model(data["x"], data["edge_index"], data["batch"])
            expected_outputs = data["y"].float()
            loss: torch.Tensor = self.loss_module(outputs, expected_outputs)

            # update model
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def get_test_loss(self) -> float:
        data = self.data.test_loader.dataset
        outputs = self.model(data["x"], data["edge_index"], data["batch"])
        expected_outputs = data["y"]
        loss: torch.Tensor = self.loss_module(outputs, expected_outputs)
        return loss.item()

    # plotting functions:
    def plot_losses(self) -> None:
        plt.figure(figsize=(12, 12))
        plt.plot(self.test_loss, label="test loss")
        plt.plot(self.train_loss, label="train loss")
