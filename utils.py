import torch
import matplotlib.pyplot as plt

def plot_fine_tuning_results(results, epoch_counts):
    plt.figure(figsize=(12, 8))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    labels = [f'Validation Accuracy ({epochs} epochs)' for epochs in epoch_counts]

    for i, val_acc in enumerate(results):
        plt.plot(range(1, len(val_acc) + 1), val_acc, label=labels[i], color=colors[i], linewidth=2)

    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Validation Accuracy', fontsize=14)
    plt.title('Validation Accuracy for Different Epoch Counts', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
def plot_training_results(epochs, train_losses, val_accuracies):
    plt.figure(figsize=(14, 7))
    plt.style.use('default')

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='royalblue', linestyle='-', marker='o')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss over Epochs', fontsize=15)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True)

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='seagreen', linestyle='-', marker='s')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Validation Accuracy over Epochs', fontsize=15)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def calc_accuracy(model, loader, device):
    """
    Calculate the accuracy of the model on the given data loader.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): DataLoader for the dataset.
        device (torch.device): The device (CPU or GPU).

    Returns:
        float: The accuracy of the model on the dataset.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for img_batch, labels in loader:
            img_batch, labels = img_batch.to(device), labels.to(device)
            outputs = model(img_batch)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct / total
