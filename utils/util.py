import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)



def to_numpy(tensor):
    """
    Converts a tensor to a numpy array.

    Args:
        tensor (torch.Tensor): The tensor to be converted.

    Returns:
        numpy_array (numpy.ndarray): The converted numpy array.
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()