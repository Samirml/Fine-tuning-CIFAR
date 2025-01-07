# Fine-tuning CIFAR


## Description
This project demonstrates how to fine-tune deep learning models on the CIFAR dataset using PyTorch.
It includes code for training, evaluating, and visualizing the performance of the models.
The goal is to provide a comprehensive example of fine-tuning neural networks for image classification tasks.



## Technologies used
- **Python 3.8+**
- **PyTorch**
- **torchvision**
- **albumentations**
- **tqdm** 
- **seaborn**
- **matplotlib**

  
## Scheme of the program operation
 It includes two models:

1. A simple convolutional neural network (SimpleCNNModel) trained from scratch.
2. A fine-tuned pre-trained ResNet-18 model (ModifiedResNet18), leveraging transfer learning.
   
The code is structured into separate files to make the project modular, easy to maintain, and reusable. 

## How It Works
**Dataset Loading:**
The CIFAR-10 dataset is loaded and transformed into tensors with augmentation (resizing, flipping, normalization).
**Model Training:**
For the SimpleCNNModel, the network is trained from scratch using the train_loop() function.
For the ModifiedResNet18, the model is fine-tuned using the fine_tune_loop() function. The convolutional layers are frozen, and only the fully connected layers are trained.
**Evaluation:** 
After every few batches, the validation accuracy is computed to track the model’s performance on unseen data.
**Visualization:**
The plot_training_results() function visualizes the training loss and validation accuracy over epochs.
The plot_fine_tuning_results() function compares fine-tuning results across different training durations.

The program was tested on GPU T4 x2 on kaggle platform. 
To speed up performance, we recommend running on a graphics card with a GPU.

## Customizing the Project
You can easily modify the following parameters to adjust the training process:

Hyperparameters: Edit the Config class in config.py to change settings such as batch size, learning rate, and the number of epochs.
Model Selection: In main.py, you can switch between training SimpleCNNModel or fine-tuning ModifiedResNet18.
Data Augmentation: You can customize the transformations applied to the dataset in dataset.py.



In order to launch a project, you need to:

1. Clone repository
```bash
git clone https://github.com/Samirml/Fine-tuning-CIFAR
cd Fine-tuning-CIFAR
```
2. Install the required packages:
Linux:
```bash
pip install -r requirements.txt
```
3. Run the training script:
Linux:
```bash
python main.py
```



## License
The idea was taken from Karpov.Courses 
(https://karpov.courses/deep-learning?_gl=1*gvc6ll*_ga*NDI1MzY4NTU3LjE3MjM5NzU4OTE.*_ga_DZP7KEXCQQ*MTcyNTg3MzAyNi4xMTYuMC4xNzI1ODczMDI2LjYwLjAuMA..).

## Authors and contacts
To contact the author, write to the following email: samiralzgulfx@gmail.com



