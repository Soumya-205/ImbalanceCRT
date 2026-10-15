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
        return self.net(x).squeeze(-1)  # logits (raw scores, before sigmoid)


@torch.no_grad()
def predict_logits(model, X):
    """Scores used for thresholding. Logits, not probabilities: sigmoid in float32
    rounds to exactly 1.0 for logits >= ~17, creating ties that make CRC's
    threshold coarse. Sigmoid preserves order, so CRC on logits is equivalent
    but tie-free."""
    model.eval()
    return model(torch.from_numpy(X)).numpy()