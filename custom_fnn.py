import torch
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class NNModel(nn.Module):
    def __init__(self, input_dim, class_size):
        super(NNModel, self).__init__()
        self.l1 = nn.Linear(input_dim, 520)  # 768차원의 cls token을..
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, class_size)  # 6개로 classification하기 위해..

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


class CustomNNModel:

    def __init__(self, input_dim, class_size, lr):
        self.device = "cuda" # if torch.cuda.is_available() else "cpu"
        self.model = NNModel(input_dim, class_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.input_dim, self.class_size = input_dim, class_size

    def train_by_data(self, batch_data, answers):

        dataset = TensorDataset(batch_data, answers)
        dataloader = DataLoader(dataset, shuffle=False, batch_size=32)
        for (step, batch) in enumerate(dataloader):


            self.model.train()
            batch = tuple(t.cuda() for t in batch)
            self.optimizer.zero_grad()

            enc_output, label = batch
            enc_output = enc_output.view(-1, self.input_dim).to(self.device)
            label = label.to(self.device)

            hypothesis = self.model(enc_output)
            loss = self.criterion(hypothesis, label)
            loss.requires_grad_(True)
            loss.backward()
            self.optimizer.step()

    def train_by_data_new(self, batch_data, answers):

        self.model.train()
        self.optimizer.zero_grad()

        batch_data = batch_data.view(-1, self.input_dim).to(self.device)
        answers = answers.to(self.device)

        hypothesis = self.model(batch_data)
        loss = self.criterion(hypothesis, answers)
        loss.requires_grad_(True)
        loss.backward()
        self.optimizer.step()

        print(loss)

