import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F

from tqdm import tqdm

def train(model, inputs, targets, batch_size=32, epochs=100, lr=0.001, weight_decay = 1e-4, device='cpu'):
    """
    Trains a PyTorch model using MSE loss and Adam optimizer with optional L2 regularization.

    Args:
        model (nn.Module): The model to train.
        inputs (Tensor): Input features (shape: [N, ...]). 
        targets (Tensor): Target outputs (shape: [N, ...]).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train.
        lr (float): Learning rate.
        weight_decay (float): L2 regularization factor (0.0 means no regularization).
        device (str): 'cpu' or 'cuda' for GPU training.
    """
    model.to(device)
  
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_features, train_labels = next(iter(dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
 
    model.train()
    for epoch in range(epochs):  # Number of epochs
        print(f'Starting epoch {epoch+1}/{epochs}')
        # Add tqdm progress bar to dataloader
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}', leave=False)
        epoch_loss = 0.0
        # display progress bar 
        # bug: correct usage and dimension
        for inputs_b, targets_b in progress_bar:
            # print(f'Processing batch with input shape: {inputs_b.shape}, target shape: {targets_b.shape}')
            # Zero the gradients
            optimizer.zero_grad()
            inputs_b, targets_b = inputs_b.to(device), targets_b.to(device)
            # Forward pass
            outputs_b = model(inputs_b)

            # Compute the loss
            B, T, C = outputs_b.shape
            outputs_b = outputs_b.view(B * T, C)
            targets_b = targets_b.view(B*T)
            # print(f'Output shape: {outputs_b.shape}, Target shape: {targets_b.shape}')
            # bug: shape of inputs
            loss = F.cross_entropy(outputs_b, targets_b)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            epoch_loss += loss.item() * inputs_b.size(0)

            # Optional: Update progress bar description
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')