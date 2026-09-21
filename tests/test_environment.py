import torch


def test_pytorch_imports():
    assert torch.__version__


def test_cuda_available():
    assert torch.cuda.is_available()


def test_gpu_is_volta_or_newer():
    major, minor = torch.cuda.get_device_capability(0)
    assert (major, minor) >= (7, 0)
