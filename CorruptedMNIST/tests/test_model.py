from corrupted_mnist.model import Network
from tests import _PATH_DATA
import torch

def test_model():
    model = Network()
    x = torch.rand(1, 1, 28, 28)
    output = model(x)
    assert output.shape == (1, 10), "Output shape should be (1, 10) for 10 classes"