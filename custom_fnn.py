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
        self.first = True
        self.loss_maximum = 0
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
        loss.backward(retain_graph=True)
        self.optimizer.step()

        if self.first:
            self.loss_maximum = loss
            self.first = False
        real_loss = (loss / self.loss_maximum).clone().detach()

        results = torch.argmax(hypothesis, dim=1)
        # tensor([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        #         3, 3, 3, 3, 3, 3, 3, 3], device='cuda:0')
        # 형식의 데이터
        return results, real_loss

    def test_data(self, batch_data):
        self.model.train()
        self.optimizer.zero_grad()

        batch_data = batch_data.view(-1, self.input_dim).to(self.device)

        hypothesis = self.model(batch_data)
        answer = torch.argmax(hypothesis, dim=1)
        loss = self.criterion(hypothesis, answer)
        real_loss = (loss / self.loss_maximum).clone().detach()
        return real_loss


