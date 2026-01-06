import torch
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "corruptmnist"

def corrupt_mnist_data():
    train_images, train_labels = [], []
    for i in range(6):
        #data = torch.load(DATA_Path / f"train_images_{i}.pt")
        train_images.append(torch.load(DATA_PATH / f"train_images_{i}.pt"))
        train_labels.append(torch.load(DATA_PATH / f"train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_labels = torch.cat(train_labels)

    test_images = torch.load(DATA_PATH / "test_images.pt")
    test_labels = torch.load(DATA_PATH / "test_target.pt")

    train_images = train_images.unsqueeze(1).float() 
    test_images = test_images.unsqueeze(1).float()
    train_labels = train_labels.long()
    test_labels = test_labels.long()

    train_set = torch.utils.data.TensorDataset(train_images, train_labels)
    test_set = torch.utils.data.TensorDataset(test_images, test_labels)
    return train_set, test_set

def show_image_and_label(image, label):
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f'Label: {label.item()}')
    plt.axis('off')
    plt.show()
    