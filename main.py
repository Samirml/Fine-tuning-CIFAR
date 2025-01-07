import torch
from torch.utils.data import DataLoader, random_split
from dataset import CifarDataset
from models import SimpleCNNModel, ModifiedResNet18
from train import train_loop
from config import Config
from utils import plot_training_results, plot_fine_tuning_results
from train import fine_tune_loop


def main():
    """
    Main function to run the training and fine-tuning processes.
    This function initializes the dataset, model, and config, then starts the training and evaluation.
    """
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CIFAR-10 dataset and split into training and validation sets
    train_dataset = CifarDataset(train=True)
    test_dataset = CifarDataset(train=False)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Train SimpleCNNModel
    print("\nTraining SimpleCNNModel from scratch:")
    model = SimpleCNNModel()
    train_losses, val_accuracies, epochs = train_loop(config, model, train_loader, val_loader, device)
    plot_training_results(epochs, train_losses, val_accuracies)

    # Fine-tuning ModifiedResNet18
    print("\nFine-tuning ModifiedResNet18:")
    configs = [Config(lr=1e-3, n_epochs=20, eval_every=1),
               Config(lr=1e-3, n_epochs=50, eval_every=1),
               Config(lr=1e-3, n_epochs=100, eval_every=1)]

    results = []
    for config in configs:
        model = ModifiedResNet18(num_classes=10)  # Create a new model for each experiment
        _, val_acc = fine_tune_loop(config, model, train_loader, val_loader, device)
        results.append(val_acc)

    plot_fine_tuning_results(results, [1, 2, 3])


if __name__ == "__main__":
    main()
