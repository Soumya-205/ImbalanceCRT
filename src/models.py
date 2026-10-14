import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(64, 32)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits

@torch.no_grad()
def predict_scores(model, X):
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(X))).numpy()