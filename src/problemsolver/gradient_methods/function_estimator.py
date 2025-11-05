import attrs
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader # provides an iterable of the dataset
import torch.optim as optim
from problemsolver.function_generators.fun_nonlinear import generate_affine_transformation


class TestPolynomial:
    def __init__(self, a: np.ndarray):
        self.a = a  # Coefficients of the polynomial, highest degree first for horner's method

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.polyval(self.a, x)

class FunctionFitter(nn.Module):
    def __init__(self,
                 n_dims: int = 1,
                 mlp_ratio: float = 4.0,
                 hidden_layers: int = 3,
                 ):
        super(FunctionFitter, self).__init__()
        width = int(n_dims * mlp_ratio)
        assert hidden_layers >= 1, "Must have at least one hidden layer"
        layers = [nn.Linear(n_dims, width), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers += [nn.Linear(width, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def fit_function_problem(polynomial_degree = 4,
                        learning_rate = 1e-2,
                        weight_decay = 1e-2,
                        max_epochs = 100, # number of epochs
                        batch_size = 64,
                        samples_per_epoch = int(10e3),
                        data_seed = 42,
                        evaluation_scale = 10.0,
                        mlp_ratio: float = 50.0,
                        hidden_layers: int = 1,
                           ):

    n_dims = 1
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Define reference problem
    np.random.seed(data_seed)
    test_fun = TestPolynomial(a=np.random.randn(polynomial_degree))

    torch.manual_seed(data_seed)
    model = FunctionFitter(n_dims=1,
                           mlp_ratio=mlp_ratio,
                           hidden_layers=hidden_layers)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, threshold=1e-4,
                                                     threshold_mode='rel', min_lr=1e-6)
    criterion = nn.MSELoss()

    epoch_stats = {}
    old_lr = learning_rate

    batches_per_epoch = int(samples_per_epoch // batch_size)

    # Note: We retain the "Epoch" terminology even though we are generating new samples each epoch.
    # In this context, an "epoch" is a convenience term for
    for epoch in range(max_epochs):
        epoch_start_clock = time.time()
        model.train()  # Set the model to training mode
        running_loss = 0.0
        x_full = evaluation_scale * torch.rand(int(samples_per_epoch), n_dims, dtype=torch.float32) - evaluation_scale * 0.5
        y_full = torch.tensor(test_fun.evaluate(x_full.numpy()), dtype=torch.float32)

        for i in range(batch_size, len(x_full), batch_size):
            # Generate batch data
            x = x_full[i - batch_size:i]
            y = y_full[i - batch_size:i]
            optimizer.zero_grad()  # Clear gradients
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)  # Forward pass
            loss = criterion(y_hat, y)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item()
        epoch_time = time.time() - epoch_start_clock
        mean_sample_loss = running_loss / batches_per_epoch
        scheduler.step(mean_sample_loss)
        if scheduler.get_last_lr()[0] != old_lr:
            print(f"Epoch {epoch+1}: reducing learning rate to {scheduler.get_last_lr()[0]:.2e}")
            old_lr = scheduler.get_last_lr()[0]
            if old_lr <= 1e-6:
                print(f'Early stopping at epoch {epoch+1} due to minimal learning rate.')
                break
        epoch_stats[epoch] = {'loss': mean_sample_loss,
                              't_elapsed': epoch_time}
        print(f'Epoch {epoch+1}, Loss: {mean_sample_loss:.4f}, time: {epoch_time:.3f}')
    plt.scatter(x.cpu().squeeze(), y.cpu().squeeze(), label='Reference Function')
    plt.scatter(x.cpu().squeeze(), y_hat.detach().cpu().squeeze(), label='Fitted Function')
    plt.show()
    epoch_stats = pd.DataFrame.from_dict(epoch_stats, orient='index')
    return epoch_stats


def sweep_hyperparameters():
    base_params = dict(
                       polynomial_degree = 4,
                       max_epochs=20,
                       samples_per_epoch=10e3,
                       batch_size = 512,
                       learning_rate=1e-2,
                       weight_decay=1e-2,
                       mlp_ratio=10.0,
                       hidden_layers=1
                       )
    param_name, param_values = 'mlp_ratio', [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    all_losses = {}
    summary_stats = pd.DataFrame(columns=["total_time", "final_loss", "n_epochs"])
    for p in param_values:
        sweep_params = base_params.copy()
        sweep_params[param_name] = p
        print(f"Running with {param_name}={p}")
        epoch_stats = fit_function_problem(**sweep_params)
        all_losses[p] = epoch_stats['loss']
        summary_stats.loc[p, "total_time"] = epoch_stats['t_elapsed'].sum()
        summary_stats.loc[p, "final_loss"] = epoch_stats['loss'].iloc[-1]
        summary_stats.loc[p, "n_epochs"] = len(epoch_stats)

    all_losses = pd.DataFrame.from_dict(all_losses, orient='columns')
    all_losses.to_pickle(f'./plots/output/function_hyperparam_sweep_{param_name}.pkl')
    ax = all_losses.plot(title=f"Hyperparameter sweep for {param_name}", logy=True)
    ax.grid(axis='y', which='both')
    plt.tight_layout()
    plt.savefig(f'./plots/function_hyperparam_sweep_{param_name}.png')
    plt.show()

if __name__ == "__main__":
    fit_function_problem()
