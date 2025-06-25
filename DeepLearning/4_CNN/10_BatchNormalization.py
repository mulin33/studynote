import torch
from torch import nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps=1e-5):
    """Batch normalization for 2D inputs."""
    