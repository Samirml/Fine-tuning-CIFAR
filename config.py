from dataclasses import dataclass

@dataclass
class Config:
    """
    Configuration class for storing model training parameters.

    Attributes:
        seed (int): Seed value for reproducibility (default is 0).
        batch_size (int): Batch size for training (default is 64).
        img_size (int): Image size for the model (default is 32).
        n_epochs (int): Number of training epochs (default is 20).
        eval_every (int): Frequency of evaluation on validation data (default is 2000).
        lr (float): Learning rate (default is 1e-3).
    """
    seed: int = 0
    batch_size: int = 64
    img_size: int = 32
    n_epochs: int = 20
    eval_every: int = 2000
    lr: float = 1e-3

