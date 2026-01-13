from torch.utils.data import Dataset
from corrupted_mnist.data import corrupt_mnist
from tests import _PATH_DATA
import torch
import os.path


@pytest.mark.skipif(not os.path.exists(_PATH_DATA ))
def test_data():
    N_train = 30000  # expected number of training samples
    N_test = 5000   # expected number of test samples
    train, test = corrupt_mnist()
    for dataset in (train, test):
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert 0 <= y <= 9
    assert len(train) == N_train, "Length of training set should be {N_train}"
    assert len(test) == N_test, "Length of test set should be {N_test}"
    train_target = torch.unique(train.tensors[1])
    assert (train_target == torch.arange(10)).all(), "Training set should contain all classes from 0 to 9"
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(10)).all(), "Test set should contain all classes from 0 to 9"
   
