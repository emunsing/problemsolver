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

class Rastrigin(nn.Module):
    def __init__(self,
                 A:torch.Tensor|None = None,
                 b:torch.Tensor|None = None,
                 n_dims:int|None = None,
                 ):
        super(Rastrigin, self).__init__()
        assert (A is None) == (b is None), "Either A and b must both be defined, or both be None"
        if A is None:
            assert b is None
            assert n_dims is not None, "if A and B are None, must define n_dim"
            self.A = nn.Parameter(0.1 * torch.randn(n_dims, n_dims))
            self.b = nn.Parameter(torch.zeros(n_dims))
        else:
            self.A = A
            self.b = b

    def forward(self, x):
        """
        x: torch.Tensor of shape [batch_size, dim]
        Returns: torch.Tensor of shape [batch_size]
        """
        r = 10.0
        z = (x - self.b) @ self.A.T  # Shape: [batch_size, dim]
        result = r * x.shape[1] + torch.sum(z ** 2 - r * torch.cos(2 * np.pi * z), dim=1) + 1
        return result


@attrs.define
class ProblemSummary:
    reference_model: Rastrigin
    final_model: Rastrigin
    summary_stats: pd.DataFrame

def fit_coordinate_problem(n_dims = 2,
                            learning_rate = 1e-4,
                            weight_decay = 1e-2,
                            num_epochs = 75, # number of epochs
                            batch_size = 128,
                            dataset_size = int(200e3),
                            data_seed = 42,
                            model_seed = 42,
                           loss_eps=1.0) -> ProblemSummary:

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Define reference problem
    np.random.seed(data_seed)
    A_mat, b = generate_affine_transformation(n_dims)
    reference_problem = Rastrigin(A=torch.tensor(A_mat, dtype=torch.float32),
                                  b=torch.tensor(b, dtype=torch.float32))

    # Create dataset
    torch.manual_seed(model_seed)
    x_data = torch.stack([2 * torch.rand(n_dims, dtype=torch.float32) - 1 for i in range(dataset_size)])  # This should be in the range [-1, 1]
    with torch.no_grad():
        y_data = reference_problem.forward(x_data)
    train_dataset = TensorDataset(x_data, y_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    torch.manual_seed(model_seed)
    model = Rastrigin(n_dims=n_dims)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    epoch_stats = {}
    previous_loss = np.inf

    for epoch in range(num_epochs):
        epoch_start_clock = time.time()
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()  # Clear gradients
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)  # Forward pass
            loss = criterion(y_hat, y)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item()
        epoch_time = time.time() - epoch_start_clock
        mean_sample_loss = running_loss / len(train_loader)  # loss.item() is the average loss per item
        if previous_loss - mean_sample_loss < loss_eps:
            print(f'Early stopping at epoch {epoch+1} due to minimal loss improvement.')
            break
        previous_loss = mean_sample_loss
        epoch_stats[epoch] = {'loss': mean_sample_loss,
                              't_elapsed': epoch_time}
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {mean_sample_loss:.4f}, time: {epoch_time:.3f}')
    epoch_stats = pd.DataFrame.from_dict(epoch_stats, orient='index')
    results = ProblemSummary(reference_model=reference_problem,
                             final_model=model,
                             summary_stats=epoch_stats)
    return results


def describe_difference(delta):
    delta = torch.abs(delta)
    return f"median {delta.median():.2f}, mean {delta.mean():.2f}, max {delta.max():.2f}"

def run_one_problem():
    n_dims = 10
    model_seed = 42
    dataset_size = int(100e3)
    batch_size = 128
    problem_summary = fit_coordinate_problem(
        n_dims=n_dims,
        model_seed=model_seed,
        dataset_size=dataset_size,
        batch_size=batch_size,
    )
    a_gap = problem_summary.reference_model.A.data - problem_summary.final_model.A.data.to('cpu')
    b_gap = problem_summary.reference_model.b.data - problem_summary.final_model.b.data.to('cpu')

    problem_summary.summary_stats['loss'].plot(
        title=f"N_dims {n_dims} Seed {model_seed}, data size {dataset_size}, batch size {batch_size}\n"
               f"A gap: {describe_difference(a_gap)}\n"
               f"b gap: {describe_difference(b_gap)}\n"
              f"Loss at epoch {len(problem_summary.summary_stats)}:  {problem_summary.summary_stats['loss'].iloc[-1]:.2f}"
        )
    plt.tight_layout()
    plt.show()
    print("Done")


if __name__ == "__main__":
    run_one_problem()