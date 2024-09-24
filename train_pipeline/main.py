import yaml
from more_itertools import consume

import mlflow
from training_loop import execute_tuning
import uuid


def load_config():
    """
    Load the configuration file
    """
    with open("hyperparameter_tuning/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    """
    Main function to execute hyperparameter tuning.
    """

    config = load_config()
    print("\nHyper parameter tuning configuration:")
    print(yaml.dump(config, default_flow_style=False, sort_keys=False, indent=4))

    mlflow.set_tracking_uri("http://host.docker.internal:5050")
    mlflow.set_experiment(config["experiment_name"])

    job_id = str(uuid.uuid4())[:5]
    print(f"Starting hyperparameter tuning Job ID: {job_id}\n")

    execute_tuning(
        run_num=1,
        batch_sizes=config["batch_sizes"],
        learning_rates=config["learning_rates"],
        epoch_choices=config["epoch_choices"],
        job_id=job_id,
        num_runs=config["num_runs"],
    )


if __name__ == "__main__":
    main()
