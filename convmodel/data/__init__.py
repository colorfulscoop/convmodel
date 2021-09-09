"""
This module treats the difference of definition of BufferedShuffleDataset in PyTorch between 1.8 and 1.9.

PyTorch torch.utils.data.BufferedShuffleDataset was moved to totorch.utils.data.datapipes.iter.Shuffle in PyTorch 1.9.
Therefore, this module provides clas BufferedShuffleDataset which works both PyTorch 1.8 and 1.9.

This import should be done prior to other import
"""
try:
    # This import will succeed with PyTroch >= 1.9
    from torch.utils.data.datapipes.iter import Shuffle as BufferedShuffleDataset
except ImportError:
    # This import will succeed with PyTroch == 1.8
    from torch.utils.data import BufferedShuffleDataset

from .conv_dataset import ConversationDataset
from .training import ConversationExample
from .training import ConversationExampleError
