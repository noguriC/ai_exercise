import torch.nn as nn
import torch.nn.init


class FC_layer(nn.Module):
    def __init__(self, in_feature_dim, out_feature_dim):
        super(FC_layer, self).__init__()
        self.linear1 = nn.Linear(in_feature_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, out_feature_dim)

    def forward(self, x):
        x = self.bn1(self.linear1(x))
        x = self.linear2(x)

        return x