import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim import Adam

import config
from data.prepare_data import train_loader, val_loader

def train_model(model : nn.Module):
    optimizer = Adam(
        model.parameters, 
        weight_decay=config.WEIGHT_DECAY
        )
    
    criterion = nn.MSELoss()

    scaler = GradScaler('cuda')

    all_train_losses = []
    all_val_losses = []

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)

            optimizer.zero_grad()

            with autocast('cuda'):
                preds = model(x)
                loss = criterion(preds.view(-1), y.view(-1))

            scaler.scale(loss).backward()
            scaler.scale(optimizer)
            scaler.update()

            train_loss += loss

        train_loss = train_loss / len(train_loader)
        all_train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x,y in val_loader:
                x = x.to(config.DEVICE)
                y = y.to(config.DEVICE)

                preds = model(x)

                loss = criterion(preds.view(-1), y.view(-1))

                val_loss += loss

        val_loss = val_loss / len(val_loader)
        all_val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{config.EPOCHS} | Train loss : {train_loss} | Val loss : {val_loss}")

    # Visualize
    x_axis = [epoch+1 for epoch in range(config.EPOCHS)]
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=x_axis, y=all_train_losses, label='Train loss')
    sns.lineplot(x=x_axis, y=all_val_losses, label='Val loss')
    plt.title("Train and Val loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


            