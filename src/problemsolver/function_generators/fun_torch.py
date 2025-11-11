import torch
from typing import Callable, Any
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset # provides an iterable of the dataset
from problemsolver.function_generators import ProblemFunction
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging

logger = logging.getLogger(__name__)

class FunctionFitter(nn.Module):
    def __init__(self,
                 n_dims: int = 1,
                 mlp_ratio: float = 4.0,
                 hidden_layers: int = 3,
                 ):
        super(FunctionFitter, self).__init__()
        self.n_dims = n_dims
        width = int(n_dims * mlp_ratio)
        activation = nn.ReLU
        assert hidden_layers >= 1, "Must have at least one hidden layer"
        layers = [nn.Linear(n_dims, width), activation()]
        for _ in range(hidden_layers-1):
            layers += [nn.Linear(width, width),
                       nn.BatchNorm1d(width),
                       activation()]
        layers += [nn.Linear(width, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLPTestProblem(ProblemFunction):
    def __init__(self,
                 ref_problem: nn.Module,
                 model: nn.Module,
                 n_train_samples: int = 10e3,
                 n_test_samples: int = 1e3,
                 batch_size: int=32,
                 max_epochs=100,
                 min_loss=1e-4,
                 max_epoch_time=5.0,
                 device=None,
                 ):
        super(MLPTestProblem, self).__init__()
        self.max_epochs = max_epochs
        self.min_loss = min_loss
        self.max_epoch_time = max_epoch_time

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)

        n_dims = int(ref_problem.n_dims)
        x_full = torch.randn(int(n_train_samples + n_test_samples), n_dims, dtype=torch.float32)
        with torch.no_grad():
            y_full = ref_problem.forward(x_full)

        n_train_samples = int(n_train_samples)
        self.x_train = x_full[:n_train_samples]
        self.y_train = y_full[:n_train_samples]
        self.x_test = x_full[n_train_samples:]
        self.y_test = y_full[n_train_samples:]

        self.ref_problem = ref_problem
        self.unfitted_model = model
        self.batch_size = batch_size
        self.optimizer: torch.optim.Optimizer | None = None  # This is the class definition. It will be set post-init by the problem wrapper. It will be instantiated in fit_and_report_loss
        self.fitted = None

    def fit_and_report_loss(self, **kwargs) -> float:
        # Kwargs are optimizer kwargs
        # Standard SGD loop, testing the optimizer performance to reach the target
        training_model = deepcopy(self.unfitted_model)
        logger.info(
            f"Number of trainable parameters: {sum(p.numel() for p in training_model.parameters() if p.requires_grad)}")

        training_model.to(self.device)
        train_loader = DataLoader(TensorDataset(self.x_train, self.y_train),
                                  batch_size=self.batch_size,
                                  shuffle=True)

        optimizer = self.optimizer(training_model.parameters(), **kwargs)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3,
                                      threshold=1e-4,
                                      threshold_mode='rel', min_lr=1e-6)
        criterion = nn.MSELoss()

        mean_sample_loss = np.inf

        for epoch in range(self.max_epochs):
            epoch_start = time.time()
            training_model.train()
            running_loss = 0.0

            for x, y in train_loader:
                optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = training_model(x)  # Forward pass
                loss = criterion(y_hat, y)  # Calculate loss
                loss.backward()  # Backward pass
                optimizer.step()
                running_loss += loss.item()

            epoch_elapsed = time.time() - epoch_start
            mean_sample_loss = running_loss / len(train_loader)
            scheduler.step(mean_sample_loss)

            if scheduler.get_last_lr()[0] <= 1e-6:
                logger.info(f'Early stopping at epoch {epoch + 1} due to minimal learning rate.')
                break

            if mean_sample_loss <= self.min_loss:
                logger.info(f'Early stopping at epoch {epoch + 1} due to loss target achieved.')
                break

            logger.info(f'Epoch {epoch + 1}/{self.max_epochs}, Loss: {mean_sample_loss:.6f}, Time Elapsed: {epoch_elapsed:.3f}s')
            if epoch_elapsed > self.max_epoch_time:
                raise TimeoutError(f"Epoch took too long; {epoch_elapsed:.3f} > {self.max_epoch_time}")
                
        self.fitted = training_model
        return mean_sample_loss


def generate_mlp_test_models(n_samples, n_dims, mlp_ratio=10.0, hidden_layers=2,
                             n_train_samples=10e3,
                             n_test_samples=1e3,
                             batch_size=32,
                             max_epochs=50,
                             device=None,
                             ) -> list[ProblemFunction]:
    test_functions = []
    for i in range(n_samples):
        ref_problem = FunctionFitter(n_dims=n_dims, mlp_ratio=mlp_ratio, hidden_layers=hidden_layers)
        student_model = FunctionFitter(n_dims=n_dims, mlp_ratio=mlp_ratio, hidden_layers=hidden_layers)
        test_func = MLPTestProblem(ref_problem=ref_problem,
                                   model=student_model,
                                   n_train_samples=n_train_samples,
                                   n_test_samples=n_test_samples,
                                   batch_size=batch_size,
                                   max_epochs=max_epochs,
                                   device=device,
                                   )
        test_functions.append((test_func, None))
    return test_functions


