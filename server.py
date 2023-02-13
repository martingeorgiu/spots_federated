import argparse
import datetime

import flwr as fl

from client import client_fn
from src.fed_strategies import FedAvgSaved, FedOptSaved, FedProxSaved, aggregate_eval, aggregate_fit
from src.tensorboard import tensorboard


def get_strategy(strategy: str, no_clients: int, log_dir: str) -> fl.server.strategy.Strategy:
    if strategy == "fedavg":
        return fl.server.strategy.FedAvg(
            min_available_clients=no_clients, min_fit_clients=no_clients
        )
    elif strategy == "fedprox":
        return tensorboard(logdir=log_dir)(FedProxSaved)(
            log_dir,
            no_clients,
            proximal_mu=0.2,
            fit_metrics_aggregation_fn=aggregate_fit,
            evaluate_metrics_aggregation_fn=aggregate_eval,
        )
    elif strategy == "fedopt":
        return FedOptSaved(log_dir, no_clients)
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
    # argParser.add_argument("--strategy", help="type of strategy to use", type=str, required=True)
    argParser.add_argument("-r", "--rounds", help="number of rounds", type=int, required=True)
    argParser.add_argument("-c", "--no_clients", help="number of clients", type=int, default=3)

    args = argParser.parse_args()
    simulated = args.simulated
    rounds = args.rounds
    no_clients = args.no_clients

    # Define strategy
    start_time = datetime.datetime.now().replace(microsecond=0).isoformat()
    suffix = "-simulated" if simulated else ""
    log_dir = f"federated-models{suffix}/{start_time}"
    # strategy = tensorboard(logdir=log_dir)(FedOptSaved)(
    #     log_dir,
    #     no_clients,
    #     fit_metrics_aggregation_fn=aggregate_fit,
    #     evaluate_metrics_aggregation_fn=aggregate_eval,
    # )
    strategy = tensorboard(logdir=log_dir)(FedProxSaved)(
        log_dir,
        no_clients,
        proximal_mu=0.3,
        fit_metrics_aggregation_fn=aggregate_fit,
        evaluate_metrics_aggregation_fn=aggregate_eval,
    )

    if simulated:
        fl.simulation.start_simulation(
            client_fn=lambda cid: client_fn(
                unit=int(cid), no_units=no_clients, minified=args.minified
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
