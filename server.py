import argparse
import datetime

import flwr as fl

from client import client_fn
from src.fed_strategies import FedProxSaved, aggregate_eval, aggregate_fit
from src.tensorboard import tensorboard


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
    strategy = tensorboard(logdir=log_dir)(FedProxSaved)(
        log_dir,
        no_clients,
        proximal_mu=0.2,
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
