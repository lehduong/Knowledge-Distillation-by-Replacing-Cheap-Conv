from torch import nn
import torch
import numpy as np 


class RandomMask2d(nn.Module):
    """
        A mask layer that drop some filters of an input, use this instead of dropout for encourage the
         network learning to map from the intrinsic filters to ghost filters
    """
    def __init__(self, in_channels, droprate=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.droprate = droprate 
        self.mask = self.create_mask()

    def create_mask(self):
        """
            Randomly create a mask to drop some layer 
        """
        num_dropped_filters = int(self.in_channels*self.droprate)
        mask = np.ones((1, self.in_channels, 1, 1))
        dropped_idx = np.random.choice(self.in_channels, num_dropped_filters, False)
        mask[0][dropped_idx] = 0
        mask = torch.FloatTensor(mask)

        return mask 

    def forward(self, x): 
        return self.mask*x