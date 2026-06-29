import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam

import config
from data.prepare_data import train_loader, val_loader

def train_model(model : nn.Module):
    optimizer = Adam(
        model.parameters(), 
        lr=config.LEARNING_RATE
        )
    
    criterion = nn.BCEWithLogitsLoss()

    scaler = GradScaler()

    best_val_loss = float('inf')
    all_train_losses = []
    all_val_losses = []

    print('Training started...')
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)

            optimizer.zero_grad()

            with autocast():
                preds = model(x)
                loss = criterion(preds.squeeze(1), y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        all_train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x,y in val_loader:
                x = x.to(config.DEVICE)
                y = y.to(config.DEVICE)

                preds = model(x)

                loss = criterion(preds.squeeze(1), y)

                val_loss += loss.item()

        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        all_val_losses.append(val_loss)

        if best_val_loss > val_loss:
            best_val_loss = val_loss
            os.makedirs('weights', exist_ok=True)
            torch.save(model.state_dict(), 'weights/best_model.pt')

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