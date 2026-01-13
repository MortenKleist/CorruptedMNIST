from corrupted_mnist.model import Network
from corrupted_mnist.data import corrupt_mnist
from corrupted_mnist.visualize import visualize
import torch
import matplotlib.pyplot as plt
import wandb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()
                        else "cpu")

project = "CorruptedMNIST"
config = {
    "learning_rate": 1e-3,
    "epochs": 10,
    "batch_size": 32,
} 

def train():
    with wandb.init(project=project, config=config) as run:
        dataset = corrupt_mnist()
        model = Network().to(device)
        train_set, _ = dataset
        # add rest of your training code here
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"]) 
        stats = {"training_loss": [],
                "training_accuracy": []}
        for epoch in range(config["epochs"]):  # Dummy epoch loop
            for i, (images, labels) in enumerate(torch.utils.data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                stats["training_loss"].append(loss.item())
                # Dummy accuracy calculation    
                accuracy = (outputs.argmax(dim=1) == labels).float().mean().item()
                stats["training_accuracy"].append(accuracy)
                run.log({"training_loss": loss.item(), "training_accuracy": accuracy})
                if i % 100 == 0:
                    print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
        print("Training complete")
        torch.save(model.state_dict(), PROJECT_ROOT /"models"/"model.pth")
        artifact = wandb.Artifact(
            name="corrupt_mnist_model",
            type="model",
            description="A model trained to classify corrupt MNIST images"
        )
        artifact.add_file("model.pth")
        run.log_artifact(artifact)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(stats["training_loss"], label="Training Loss")
        ax[0].set_title("Training Loss")
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("Loss")
        ax[0].legend()
        ax[1].plot(stats["training_accuracy"], label="Training Accuracy", color='orange')
        ax[1].set_title("Training Accuracy")
        ax[1].set_xlabel("Iteration")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()
        fig.savefig(PROJECT_ROOT/"reports"/"figures"/"training_stats.png")

if __name__ == "__main__":
    train()
    
