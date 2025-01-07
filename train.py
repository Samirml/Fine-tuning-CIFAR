import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from utils import calc_accuracy


def train_loop(config, model, train_loader, val_loader, device):
    """
    Training loop for the model.

    Args:
        config (Config): Configuration parameters for the training process.
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device (CPU or GPU) for training.

    Returns:
        tuple: (list of training losses, list of validation accuracies, list of epochs)
    """
    optimizer = Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    train_losses = []
    val_accuracies = []
    epochs = []

    for epoch in range(config.n_epochs):
        print(f"Epoch {epoch + 1}/{config.n_epochs}")
        running_loss = 0.0
        for i, (img_batch, labels) in enumerate(tqdm(train_loader)):
            img_batch, labels = img_batch.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(img_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % config.eval_every == 0:
                val_acc = calc_accuracy(model, val_loader, device)
                val_accuracies.append(val_acc)
                epochs.append(epoch + (i + 1) / len(train_loader))
                running_loss = 0.0

        train_losses.append(running_loss / len(train_loader))
        val_acc = calc_accuracy(model, val_loader, device)
        val_accuracies.append(val_acc)
        epochs.append(epoch + 1)

    return train_losses, val_accuracies, epochs


def fine_tune_loop(config, model, train_loader, val_loader, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_accuracy = []
    val_accuracy = []

    for epoch in range(config.n_epochs):
        model.train()
        correct, total = 0, 0

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        train_accuracy.append(train_acc)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        val_accuracy.append(val_acc)

        print(f'Epoch {epoch + 1}/{config.n_epochs}, Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}')

    return train_accuracy, val_accuracy