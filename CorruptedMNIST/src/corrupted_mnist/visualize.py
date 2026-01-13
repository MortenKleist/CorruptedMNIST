import torch
from corrupted_mnist.model import Network
from corrupted_mnist.data import corrupt_mnist as corrupt_mnist_data
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()
                        else "cpu")
wandb.init(project="CorruptedMNIST")
def visualize():
    """Visualize model predictions."""
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    model = Network().to(device)
    model.load_state_dict(torch.load(PROJECT_ROOT/"models"/"model.pth"))
    model.eval()
    model.fc2 = torch.nn.Identity()  # Remove final layer to get embeddings
    _, test_set = corrupt_mnist_data()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
    embeddings = []
    labels_list = []
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            features = model(images)
            embeddings.append(features)
            labels_list.append(labels)
        embeddings = torch.cat(embeddings).cpu().numpy()
        labels_list = torch.cat(labels_list).cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    for i in range(10):
        idxs = labels_list == i
        plt.scatter(embeddings_2d[idxs, 0], embeddings_2d[idxs, 1], label=str(i))
    plt.legend()
    plt.title("t-SNE Visualization of Model Predictions")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(PROJECT_ROOT/"reports"/"figures"/"tsne_visualization.png")
    wandb.log({"t-SNE": wandb.Image(plt)})
    plt.close()


if __name__ == "__main__":
    visualize()