import torch.nn as nn


class FunctionFitter(nn.Module):
    def __init__(self,
                 n_dims: int = 1,
                 mlp_ratio: float = 4.0,
                 hidden_layers: int = 3,
                 ):
        super(FunctionFitter, self).__init__()
        width = int(n_dims * mlp_ratio)
        activation = nn.ReLU
        assert hidden_layers >= 1, "Must have at least one hidden layer"
        layers = [nn.Linear(n_dims, width), activation()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(width, width),
                       nn.BatchNorm1d(width),
                       activation()]
        layers += [nn.Linear(width, 1)]
        mlp = nn.Sequential(*layers)

        layers = [nn.Linear(n_dims, width), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers += [nn.Linear(width, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)