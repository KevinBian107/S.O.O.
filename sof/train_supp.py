import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from config import args_supp
from models import UPN
from optimization_utils import *

def train_model(model, dataloader, optimizer):
    '''Ensure data is correct, is all in the data, must use consistent non stop data'''
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_forward_loss = 0
    total_inverse_loss = 0
    total_consistency_loss = 0
    for states, actions, next_states in dataloader:
        optimizer.zero_grad()
        loss, recon_loss, forward_loss, inverse_loss, consistency_loss = compute_upn_loss(model, states, actions, next_states)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_forward_loss += forward_loss.item()
        total_inverse_loss += inverse_loss.item()
        total_consistency_loss += consistency_loss.item()
    return (total_loss / len(dataloader), total_recon_loss / len(dataloader),
            total_forward_loss / len(dataloader), total_inverse_loss / len(dataloader),
            total_consistency_loss / len(dataloader))

def validate_model(model, dataloader):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_forward_loss = 0
    total_inverse_loss = 0
    total_consistency_loss = 0
    with torch.no_grad():
        for states, actions, next_states in dataloader:
            loss, recon_loss, forward_loss, inverse_loss, consistency_loss = compute_upn_loss(model, states, actions, next_states)
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_forward_loss += forward_loss.item()
            total_inverse_loss += inverse_loss.item()
            total_consistency_loss += consistency_loss.item()
    return (total_loss / len(dataloader), total_recon_loss / len(dataloader),
            total_forward_loss / len(dataloader), total_inverse_loss / len(dataloader),
            total_consistency_loss / len(dataloader))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() and args_supp.cuda else "cpu")
    save_dir = os.path.join(os.getcwd(), 'sof', 'data')
    os.makedirs(save_dir, exist_ok=True)
    data_path = os.path.join(save_dir, args_supp.imitate_data_path)
    states, actions, next_states = load_supp_data(file_path=data_path)
    
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Next states shape: {next_states.shape}")
    
    split = int(0.8 * len(states))
    train_states, train_actions, train_next_states = states[:split], actions[:split], next_states[:split]
    val_states, val_actions, val_next_states = states[split:], actions[split:], next_states[split:]

    train_dataset = TensorDataset(train_states, train_actions, train_next_states)
    val_dataset = TensorDataset(val_states, val_actions, val_next_states)

    train_dataloader = DataLoader(train_dataset, batch_size=args_supp.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args_supp.batch_size, shuffle=False)

    state_dim = states.shape[-1]
    action_dim = actions.shape[-1]

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    model = UPN(state_dim, action_dim, args_supp.latent_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args_supp.learning_rate, weight_decay=1e-5)

    train_losses = []
    val_losses = []

    for epoch in range(args_supp.num_epochs):
        train_loss = train_model(model, train_dataloader, optimizer)
        val_loss = validate_model(model, val_dataloader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{args_supp.num_epochs}")
        print(f"Train - Total: {train_loss[0]:.4f}, Recon: {train_loss[1]:.4f}, Forward: {train_loss[2]:.4f}, Inverse: {train_loss[3]:.4f}, Consistency: {train_loss[4]:.4f}")
        print(f"Val   - Total: {val_loss[0]:.4f}, Recon: {val_loss[1]:.4f}, Forward: {val_loss[2]:.4f}, Inverse: {val_loss[3]:.4f}, Consistency: {val_loss[4]:.4f}")

    plot_supp_losses(train_losses, val_losses)

    save_dir = os.path.join(os.getcwd(), 'sof', 'params', 'supp')
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, args_supp.save_supp_path)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at: {model_path}")