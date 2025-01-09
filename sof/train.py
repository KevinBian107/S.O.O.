import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from train_supp import train_supp
from train_soo import train_sofppo_agent

def train():
    
    # given that we have intention searched data, we do the following:
    train_supp()
    train_sofppo_agent()
    
if __name__ == '__main__':
    train()