import torch
from corrupted_mnist.model import Network
from corrupted_mnist.data import corrupt_mnist as corrupt_mnist_data

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()
                        else "cpu")

def evaluate():
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")

    # TODO: Implement evaluation logic here
    model = Network().to(device)
    model.load_state_dict(torch.load("/models/model.pth"))
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
    evaluate()