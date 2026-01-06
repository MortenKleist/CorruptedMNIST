import torch
import typer
from data import corrupt_mnist_data 
from model import Network
import matplotlib.pyplot as plt

app = typer.Typer()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()
                        else "cpu")

@app.command()
def train(lr: float = 1e-3) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = Network().to(device)
    train_set, _ = corrupt_mnist_data()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    stats = {"training_loss": [],
             "training_accuracy": []}
    for epoch in range(10):  # Dummy epoch loop
        for i, (images, labels) in enumerate(torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)):
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
            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
    print("Training complete")
    torch.save(model.state_dict(), "model.pth")
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
    fig.savefig("training_stats.png")



@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = Network().to(device)
    model.load_state_dict(torch.load(model_checkpoint))
    _, test_set = corrupt_mnist_data()
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).float().sum().item()
        print(f"Test Accuracy: {100 * correct / total}%")


if __name__ == "__main__":
    app()