import argparse
import datetime
import os
from pathlib import Path

import flwr as fl

from client import client_fn
from src.consts import get_alpha
from src.fed_strategies import FedAvgSaved, FedOptSaved, FedProxSaved, aggregate_eval, aggregate_fit
from src.save_setup import save_setup
from src.tensorboard import tensorboard


def get_strategy(
    strategy: str, no_clients: int, log_dir: str, proximal_mu: float
) -> fl.server.strategy.Strategy:
    print(f"Running using strategy: {strategy}")
    log_dir_models = log_dir + "/rounds"
    os.makedirs(log_dir_models, exist_ok=True)

    if strategy == "fedavg":
        return tensorboard(logdir=log_dir)(FedAvgSaved)(
            log_dir_models,
            no_clients,
            fit_metrics_aggregation_fn=aggregate_fit,
            evaluate_metrics_aggregation_fn=aggregate_eval,
        )
    elif strategy == "fedprox":
        return tensorboard(logdir=log_dir)(FedProxSaved)(
            log_dir_models,
            no_clients,
            proximal_mu=proximal_mu,
            fit_metrics_aggregation_fn=aggregate_fit,
            evaluate_metrics_aggregation_fn=aggregate_eval,
        )
    elif strategy == "fedopt":
        return tensorboard(logdir=log_dir)(FedOptSaved)(
            log_dir_models,
            no_clients,
            fit_metrics_aggregation_fn=aggregate_fit,
            evaluate_metrics_aggregation_fn=aggregate_eval,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def main() -> None:
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-s",
        "--simulated",
        help="run in simulated mode",
        action=argparse.BooleanOptionalAction,
    )
    argParser.add_argument(
        "-m",
        "--minified",
        help="only applicable in simulated mode, run with minified dataset",
        action=argparse.BooleanOptionalAction,
    )
    argParser.add_argument("--strategy", help="type of strategy to use", type=str, required=True)
    argParser.add_argument("-r", "--rounds", help="number of rounds", type=int, required=True)
    argParser.add_argument("-c", "--no_clients", help="number of clients", type=int, default=3)
    argParser.add_argument(
        "-te", "--train_epochs", help="number of train epochs per round", type=int, default=1
    )
    argParser.add_argument(
        "-pmu",
        "--proximal_mu",
        help="proximal_mu - only applicable for FedProx strategy",
        type=float,
        default=0.3,
    )
    argParser.add_argument(
        "-a", "--alpha", help="alpha parameter of focal loss (only used when simulated)", type=str
    )
    argParser.add_argument(
        "-g",
        "--gamma",
        help="gamma parameter of focal loss (only used when simulated)",
        type=float,
        default=2,
    )

    args = argParser.parse_args()
    simulated = args.simulated
    strategy = args.strategy
    rounds = args.rounds
    no_clients = args.no_clients
    train_epochs = args.train_epochs
    proximal_mu = args.proximal_mu
    alpha = get_alpha(args.alpha)

    # Define strategy
    start_time = datetime.datetime.now().replace(microsecond=0).isoformat()
    suffix = "-simulated" if simulated else ""
    log_dir = f"federated-models{suffix}/{strategy}/{start_time}"
    strategy = get_strategy(strategy, no_clients, log_dir, proximal_mu)
    save_setup(log_dir)
    if simulated:
        fl.simulation.start_simulation(
            client_fn=lambda cid: client_fn(
                unit=int(cid),
                no_units=no_clients,
                minified=args.minified,
                train_epochs=train_epochs,
                alpha=alpha,
                gamma=args.gamma,
            ),
            num_clients=no_clients,
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
        )
    else:
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
        )


if __name__ == "__main__":
    main()
