import argparse
import flwr as fl
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import os
import datetime
from flwr.server.strategy import FedAvg
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    Metrics,
    parameters_to_ndarrays
)
from flwr.common.typing import MetricsAggregationFn
from flwr.server.utils import tensorboard
from flwr.server.client_proxy import ClientProxy

from client import client_fn

class SaveModelStrategy(FedAvg):
    def __init__(self, log_dir:str, no_clients:int,evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None):
        super().__init__(min_available_clients=no_clients,min_fit_clients=no_clients,evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)
        self.log_dir = log_dir        

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        print(f"Saving round {server_round}")
        # print(f"aggregated_parameters: {aggregated_parameters}")
        print(f"aggregated_metrics: {aggregated_metrics}")
        
        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.save(f"{self.log_dir}/round-{server_round}-weights.npy", aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["test_acc"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def main() -> None:
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-s", "--simulated", help="run in simulated mode", action=argparse.BooleanOptionalAction)
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
    strategy = tensorboard(logdir=log_dir)(SaveModelStrategy)(
        log_dir,
        no_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )


    if(simulated):
        fl.simulation.start_simulation(
            client_fn=lambda cid: client_fn(unit=int(cid),no_units=no_clients),
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