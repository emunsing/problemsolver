import attrs
import cProfile
import pstats
from pstats import SortKey
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader # provides an iterable of the dataset
import torch.optim as optim
from itertools import product

class TestPolynomial:
    def __init__(self, a: np.ndarray):
        self.a = a  # Coefficients of the polynomial, highest degree first for horner's method

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.polyval(self.a, x)

class MultiPoly:
    """
    Random multivariate polynomial evaluator.
    Supports fixed coefficients, efficient re-evaluation,
    and arbitrary cross-terms up to a given total degree.
    """
    def __init__(self, n_vars: int, degree: int = 7, seed: int | None = None):
        self.n_vars = n_vars
        self.degree = degree
        rng = np.random.default_rng(seed)

        # Enumerate all monomial powers (tuples like (2,0,1))
        self.powers = [p for p in product(range(degree + 1), repeat=n_vars)
                       if sum(p) <= degree]

        # Random coefficients for each term
        self.coefs = rng.standard_normal(len(self.powers))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Evaluate polynomial at N×n_vars array X."""
        assert X.shape[1] == self.n_vars
        terms = [self.coefs[i] * np.prod(X ** p, axis=1)
                 for i, p in enumerate(self.powers)]
        return np.sum(terms, axis=0)[:, None] # Ensure we return a [n, 1] tensor


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


def fit_function_problem(
        test_fun,
        student_model,
        n_dims,
        learning_rate = 1e-2,
        weight_decay = 1e-2,
        max_epochs = 100, # number of epochs
        batch_size = 64,
        samples_per_epoch = int(10e3),
        rel_stopping_threshold: float | None = 1e-4,  # Early stopping threshold for MultiPoly problems
        data_seed: int | None = None,
        evaluation_scale = 10.0,
        plot=True,
        device=None,
):

    # Note: On a macbook laptop, "cpu" is faster than "mps" for these small models.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)

    # Define reference problem
    if data_seed is not None:
        np.random.seed(data_seed)
        torch.manual_seed(data_seed)

    print(f"Number of trainable parameters: {sum(p.numel() for p in student_model.parameters() if p.requires_grad)}")
    student_model = student_model.to(device)

    optimizer = optim.AdamW(student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, threshold=1e-4,
                                                     threshold_mode='rel', min_lr=1e-6)
    criterion = nn.MSELoss()

    epoch_stats = {}
    old_lr = learning_rate

    batches_per_epoch = int(samples_per_epoch // batch_size)

    # Continually generating new samples by moving this into the epoch loop, but this adds significant overhead (20%)
    x_full = evaluation_scale * torch.rand(int(samples_per_epoch), n_dims, dtype=torch.float32) - evaluation_scale * 0.5
    with torch.no_grad():
        y_full = test_fun.forward(x_full)
    x_mean, x_std = x_full.mean(), x_full.std()
    y_mean, y_std = y_full.mean(), y_full.std()

    # Normalize
    x_full = (x_full - x_mean) / x_std
    y_full = (y_full - y_mean) / y_std
    print("Y std: ", y_full.std())

    rolling_loss = [np.inf] * 5
    # Note: We retain the "Epoch" terminology even though we are continually generating new samples.
    # In this context, an "epoch" is a convenience term for a cycle of logging and learning rate schedule evaluation.
    for epoch in range(max_epochs):
        epoch_start_clock = time.time()
        student_model.train()  # Set the model to training mode
        running_loss = 0.0

        # For shuffling data:
        idx = np.random.permutation(len(y_full))
        x_full = x_full[idx]
        y_full = y_full[idx]

        for i in range(batch_size, len(x_full), batch_size):
            # Generate batch data
            x = x_full[i - batch_size:i]
            y = y_full[i - batch_size:i]
            optimizer.zero_grad()  # Clear gradients
            x = x.to(device)
            y = y.to(device)
            y_hat = student_model(x)  # Forward pass
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
        rolling_loss.append(mean_sample_loss)
        rolling_loss.pop(0)
        if rel_stopping_threshold is not None and np.mean(rolling_loss) < rel_stopping_threshold * n_dims:  # Empirically seems appropriate
            print(f'Early stopping at epoch {epoch+1} due to meeting loss threshold.')
            break

    if plot:
        if n_dims == 1:
            plt.scatter(x.cpu().squeeze(), y.cpu().squeeze(), label='Reference Function')
            plt.scatter(x.cpu().squeeze(), y_hat.detach().cpu().squeeze(), label='Fitted Function')
            plt.show()
        elif n_dims == 2:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            with torch.no_grad():
                y_hat = student_model(x_full.to(device)).cpu()

            vmin = min(y_full.min(), y_hat.min())
            vmax = max(y_full.max(), y_hat.max())

            ax[0].scatter(x_full[:, 0], x_full[:, 1], c=y_full, vmin=vmin, vmax=vmax)
            ax[1].scatter(x_full[:, 0], x_full[:, 1], c=y_hat, vmin=vmin, vmax=vmax)
            plt.show()
        else:
            print("Plotting is only supported for 1D or 2D input data.")

    epoch_stats = pd.DataFrame.from_dict(epoch_stats, orient='index')
    return epoch_stats

def fit_function_problem_1d():
    n_dims, polynomial_degree = 1, 4
    mlp_ratio, hidden_layers = 10.0, 1
    test_fun = MultiPoly(n_vars=n_dims, degree=polynomial_degree, seed=42)
    student_model = FunctionFitter(n_dims=n_dims, mlp_ratio=mlp_ratio, hidden_layers=hidden_layers)

    return fit_function_problem(test_fun=test_fun,
                                student_model=student_model,
                                n_dims=n_dims,
                                data_seed=None,
                                max_epochs=20,
                                samples_per_epoch=int(10e3),
                                batch_size=512,
                                learning_rate=1e-2,
                                weight_decay=1e-2,
                                plot=True)

def fit_student_teacher_problem_1d():
    n_dims, mlp_ratio, hidden_layers = 1, 10.0, 3
    torch.manual_seed(42)
    teacher = FunctionFitter(n_dims=n_dims, mlp_ratio=mlp_ratio, hidden_layers=hidden_layers)
    student = FunctionFitter(n_dims=n_dims, mlp_ratio=mlp_ratio, hidden_layers=hidden_layers)

    return fit_function_problem(test_fun=teacher,
                                student_model=student,
                                n_dims=n_dims,
                                data_seed=None,
                                max_epochs=20,
                                samples_per_epoch=int(10e3),
                                batch_size=512,
                                learning_rate=1e-2,
                                weight_decay=1e-2,
                                # rel_stopping_threshold=None,
                                plot=True)

def fit_student_teacher_problem_nd(n_dims=2, plot=True):
    mlp_ratio, hidden_layers = 10.0, 3
    teacher = FunctionFitter(n_dims=n_dims, mlp_ratio=mlp_ratio, hidden_layers=hidden_layers)
    student = FunctionFitter(n_dims=n_dims, mlp_ratio=mlp_ratio, hidden_layers=hidden_layers)

    return fit_function_problem(test_fun=teacher,
                                student_model=student,
                                n_dims=n_dims,
                                data_seed=None,
                                max_epochs=20,
                                samples_per_epoch=int(10e3),
                                batch_size=512,
                                learning_rate=1e-2,
                                weight_decay=1e-2,
                                # rel_stopping_threshold=None,
                                plot=plot)


def fit_function_problem_2d():
    n_dims, polynomial_degree = 2, 4
    mlp_ratio, hidden_layers = 10.0, n_dims
    test_fun = MultiPoly(n_vars=n_dims, degree=polynomial_degree, seed=42)
    student_model = FunctionFitter(n_dims=n_dims, mlp_ratio=mlp_ratio, hidden_layers=hidden_layers)

    return fit_function_problem(test_fun=test_fun,
                                student_model=student_model,
                                n_dims=n_dims,
                                data_seed=None,
                                max_epochs=20,
                                samples_per_epoch=int(10e3),
                                batch_size=512,
                                learning_rate=1e-2,
                                weight_decay=1e-2,
                                plot=True)

def fit_function_problem_n_d():
    n_dims, polynomial_degree = 5, 4
    mlp_ratio, hidden_layers = 10.0, n_dims
    test_fun = MultiPoly(n_vars=n_dims, degree=polynomial_degree, seed=42)
    student_model = FunctionFitter(n_dims=n_dims, mlp_ratio=mlp_ratio, hidden_layers=hidden_layers)

    start_time = time.time()
    res = fit_function_problem(test_fun=test_fun,
                                student_model=student_model,
                                n_dims=n_dims,
                                data_seed=None,
                                max_epochs=200,
                                samples_per_epoch=int(10e3),
                                batch_size=512,
                                learning_rate=1e-2,
                                weight_decay=1e-2,
                                plot=True,
                                device="cpu")
    print(f"Done in {time.time() - start_time:.2f} seconds")
    return res

def fit_many_student_problems():
    n_dims = 2
    n_tests = 100
    summary_stats = pd.DataFrame(columns=["total_time", "final_loss", "n_epochs"])
    for p in range(100):
        epoch_stats = fit_student_teacher_problem_nd(n_dims=n_dims, plot=False)
        summary_stats.loc[p, "total_time"] = epoch_stats['t_elapsed'].sum()
        summary_stats.loc[p, "final_loss"] = epoch_stats['loss'].iloc[-1]
        summary_stats.loc[p, "n_epochs"] = len(epoch_stats)
    all_losses.hist()

def sweep_hyperparameters():
    n_dims, polynomial_degree = 5, 4
    mlp_ratio, hidden_layers = 10.0, n_dims
    test_fun = MultiPoly(n_vars=n_dims, degree=polynomial_degree, seed=42)
    student_model = FunctionFitter(n_dims=n_dims, mlp_ratio=mlp_ratio, hidden_layers=hidden_layers)

    base_params = dict(test_fun=test_fun,
                       student_model=student_model,
                       n_dims=n_dims,
                       max_epochs=1000,
                       samples_per_epoch=int(10e3),
                       batch_size = 512,
                       learning_rate=1e-2,
                       weight_decay=1e-2,
                       plot=False,
                       device="cpu",
                       )
    param_name, param_values = 'hidden_layers', [max(1, int(np.floor(n_dims*0.6))), n_dims, int(np.ceil(n_dims * 1.5))]
    param_name, param_values = 'batch_size', [64, 128, 256, 512, 1024]
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
    print("Summary Stats:")
    print(summary_stats)

def plot_n_layers():
    # Verification of architecture by plotting 1-D test case for a variety of numbers of hidden layers
    layer_range = [0, 1, 2, 5, 10, 20]
    mlp_width = 10
    x = np.linspace(-10, 10, 1000)
    fig, ax, = plt.subplots(nrows=len(layer_range), ncols=1, figsize=(8, 10))
    activation = nn.ReLU()
    for r, n_hidden_layers in enumerate(layer_range):
        layers = [nn.Linear(1, mlp_width), activation]
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(mlp_width, mlp_width),
                       nn.BatchNorm1d(mlp_width),
                       activation]
        layers += [nn.Linear(mlp_width, 1)]
        mlp = nn.Sequential(*layers)

        with torch.no_grad():
            y = mlp.forward(torch.tensor(x.tolist()).reshape(-1,1))
        ax[r].plot(x, y)
        ax[r].set_ylabel(n_hidden_layers)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    sweep_hyperparameters()
    pr.disable()
    stats = pstats.Stats(pr)
    stats.strip_dirs()
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)
