from tqdm import tqdm

import torch.nn as nn
import torch.optim

from mutag_model import MUTAGModel
from mutag_data import DataLoader

class Trainer:
    def __init__(self, model : MUTAGModel, data : DataLoader):
        self.model : MUTAGModel = model
        self.data : DataLoader = data

        # optimizer and loss
        self.optimizer = torch.optim.SGD(model.parameters())
        self.loss_func = nn.CrossEntropyLoss()

        # loss logs
        self.train_loss : list[float] = []
        self.test_loss : list[float] = []

    def step(self, epochs : int = 1) -> None:
        self.model.train()

        # train
        for epoch in range(epochs):
            loss = self.train_step()

            # loss log:
            self.train_loss.append(loss)

            test_loss = self.get_test_loss()
            self.test_loss.append(loss)

    def train_step(self) -> float:
        for data in 

    def get_test_loss(self) -> float:
        pass



