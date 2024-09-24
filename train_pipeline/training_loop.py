import random
from functools import partial
from itertools import starmap

from more_itertools import consume

import mlflow
import torch
from torch import nn
from torchinfo import summary

from data_loader import load_MNIST_data
from model import DigitClassifier
from trainer import Trainer


device = "cuda" if torch.cuda.is_available() else "cpu"


def accuracy_fn(test_labels, predicted_labels):
    """
    Calculate accuracy of the model.

    Args:
    test_labels (Tensor): Test labels
    predicted_labels (Tensor): Predicted labels
    """
    tp = int(torch.eq(test_labels, predicted_labels).sum())
    return tp / len(test_labels) * 100


def training_run(
    run_name,
    batch_sizes,
    learning_rates,
    epoch_choices,
    train_data_loader,
    test_data_loader,
):
    """
    Train the model with random selection of hyperparameters.
    Log the model, parameters and metrics using MLflow.

    Args:
    run_name (str): Run name
    batch_sizes (list): List of batch sizes
    learning_rates (list): List of learning rates
    epoch_choices (list): List of epochs
    train_data_loader (DataLoader): Training data loader
    test_data_loader (DataLoader): Testing data loader
    """

    batch_size = random.choice(batch_sizes)
    epochs = random.choice(epoch_choices)
    learning_rate = random.choice(learning_rates)

    model = DigitClassifier().to(device)
    model_name = "digit-classifier"
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = Trainer(model, optimizer, loss_fn, accuracy_fn)

    log_params = {
        "model": model.__class__.__name__,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": optimizer.__class__.__name__,
        "loss_fn": loss_fn.__class__.__name__,
        "torch_version": torch.__version__,
    }

    mlflow.pytorch.log_model(model, "model")

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(log_params)

        print(f"\nRunning for {epochs} epochs")
        print(f"Batch size is {batch_size}")
        print(f"Learning rate is {learning_rate}\n")

        # Train
        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            train_loss, train_accuracy = trainer.training(
                data_loader=train_data_loader,
            )
            test_loss, test_accuracy = trainer.evaluate(
                data_loader=test_data_loader,
            )

            # Log metrics.
            mlflow.log_metrics(
                {
                    "train_accuracy": train_accuracy,
                    "train_loss": train_loss,
                    "test_accuracy": test_accuracy,
                    "test_loss": test_loss,
                },
                step=epoch,
            )

        # Log model.
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name=model_name,
        )


def generate_test_names(test_runs):
    """
    Generate test names for the training runs.

    Args:
    test_runs (int): Number of tests for this run.
    """
    return (f"test_{i}" for i in range(test_runs))


def execute_tuning(
    run_num,
    batch_sizes=[64, 32, 16],
    learning_rates=[0.01, 0.001, 0.0001],
    epoch_choices=[1, 2, 3],
    job_id="default",
    num_runs=1,
):
    """
    This function allows for hyper parameter tuning of the model.

    1. Load the MNIST data.
    2. Generate run names.
    3. Execute training runs.
    4. Log the runs.

    Args:
    run_num (int): Test number
    batch_sizes (list): List of batch sizes
    learning_rates (list): List of learning rates
    epoch_choices (list): List of epochs
    num_runs (int): Number of runs
    """

    train_data_loader, test_data_loader = load_MNIST_data(root="./data", batch_size=64)

    # Use a parent run to encapsulate the child runs.
    with mlflow.start_run(run_name=f"hyprm_tuning_job_{job_id}_run_{run_num}"):

        # Partial application of the log_run function.
        log_current_run = partial(
            training_run,
            batch_sizes=batch_sizes,
            learning_rates=learning_rates,
            epoch_choices=epoch_choices,
            train_data_loader=train_data_loader,
            test_data_loader=test_data_loader,
        )

        # Generate test names and apply training_run function to each test name.
        tests = starmap(
            log_current_run,
            ((test_name,) for test_name in generate_test_names(num_runs)),
        )
        # Consume the iterator to execute the tests.
        consume(tests)
