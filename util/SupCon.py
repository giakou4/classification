import torch
import torch.nn as nn
import torch.nn.functional as F


class SupCon(nn.Module):
    """backbone + projection head"""
    def __init__(self, model, head='mlp', feat_dim=128):
        super(SupCon, self).__init__()
        
        self.dim_in = model._to_linear
        self.encoder = model
        
        if head == 'linear':
            self.head = nn.Linear(self.dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.dim_in, self.dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(self.dim_in, feat_dim)
            )
        else:
            raise NotImplementedError('Head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat