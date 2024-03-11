import torch
import torch.nn as nn


class Train:
    def __init__(self,
                 epoch: int,
                 batch_size: int,
                 optimizer: str,
                 loss_fn: str,
                 optimizer_parameters,
                 net,
                 train_loader,
                 validation_loader,
                 ) -> None:
        super().__init__()
        self.epoch = epoch
        self.batch_size = batch_size
        self.optimizer = getattr(torch.optim, optimizer)(optimizer_parameters)
        self.loss_fn = getattr(nn, loss_fn)()
        self.net = net
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.total_loss = 0

        self.correct = 0
        self.accuracy = 0
        self.num_data = 0

    def train(self):
        self.net.train()
        for epoch in range(self.epoch):
            self.total_loss = 0
            for idx, (seq, label) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.net(seq)
                loss = self.loss_fn(output, label)
                loss.backward()
                self.correct += (label == output).sum().item()
                self.total_loss += loss
                self.num_data += seq.size(0)

            self.accuracy = self.correct / self.num_data

    def validation(self):
        self.net.eval()

        for idx, (seq, label) in enumerate(self.validation_loader):
            with torch.no_grad():
                output = self.net(seq)
                self.correct + (label == output).sum().item()
                self.num_data += seq.size(0)
            self.accuracy = self.correct / self.num_data



