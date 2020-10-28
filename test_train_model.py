# testing file
# lets play around a little with testing perf benching of deep learning

# Lib Imports
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy


################### Basic Model ######################################
class MNISTModel(pl.LightningModule):

    def __init__(self):
        super(MNISTModel, self).__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

################## Running the main model #####################

if __name__ == '__main__':
    
    # Init our model
    mnist_model = MNISTModel()

    # Init DataLoader from MNIST Dataset
    train_ds = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=32, num_workers = 8)

    # Initialize a trainer
    # gpus = 1 
    # distributed_backend='ddp_cpu',
    trainer = pl.Trainer(max_epochs=3, progress_bar_refresh_rate=20)
    #trainer = pl.Trainer(gpus=2, distributed_backend='ddp', max_epochs=3, progress_bar_refresh_rate=20)

    # Train the model ⚡
    trainer.fit(mnist_model, train_loader)

