from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import FitRes, Metrics, Parameters, Scalar, parameters_to_ndarrays
from flwr.common.typing import MetricsAggregationFn
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedOpt, FedProx


class FedAvgSaved(FedAvg):
    def __init__(self, log_dir: str, no_clients: int, **kwargs):
        super().__init__(
            min_available_clients=no_clients,
            min_fit_clients=no_clients,
            **kwargs,
        )
        self.log_dir = log_dir

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(f"aggregate_fit started of round {server_round}")
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        store_aggregated_parameters(self.log_dir, server_round, aggregated_parameters)
        return aggregated_parameters, aggregated_metrics


class FedProxSaved(FedProx):
    def __init__(self, log_dir: str, no_clients: int, **kwargs):
        super().__init__(
            min_available_clients=no_clients,
            min_fit_clients=no_clients,
            **kwargs,
        )
        self.log_dir = log_dir

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(f"aggregate_fit started of round {server_round}")
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        store_aggregated_parameters(self.log_dir, server_round, aggregated_parameters)
        return aggregated_parameters, aggregated_metrics


class FedOptSaved(FedOpt):
    def __init__(self, log_dir: str, no_clients: int, **kwargs):
        super().__init__(
            min_available_clients=no_clients,
            min_fit_clients=no_clients,
            **kwargs,
        )
        self.log_dir = log_dir

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(f"aggregate_fit started of round {server_round}")
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        store_aggregated_parameters(self.log_dir, server_round, aggregated_parameters)
        return aggregated_parameters, aggregated_metrics


def store_aggregated_parameters(
    log_dir: str,
    server_round: int,
    aggregated_parameters: Optional[Parameters],
) -> None:
    if aggregated_parameters is not None:
        # Convert `Parameters` to `List[np.ndarray]`
        aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)

        # Save aggregated_ndarrays
        print(f"Saving a checkpoint of round {server_round} in aggregated_ndarrays...")
        np.save(f"{log_dir}/round-{server_round}-weights.npy", aggregated_ndarrays)


def aggregate_metric(metrics: List[Tuple[int, Metrics]], key: str) -> float:
    # Multiply metric of each client by number of examples used
    weighted_metrics = 0
    no_examples = 0
    for num_examples, m in metrics:
        value = m.get(key)
        if value is None:
            continue
        weighted_metrics += num_examples * value
        no_examples += num_examples

    if no_examples == 0:
        return 0.0

    # Aggregate and return custom metric (weighted average)
    return weighted_metrics / no_examples


def aggregate_fit(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    train_acc = aggregate_metric(metrics, "train_acc")
    val_loss = aggregate_metric(metrics, "val_loss")
    val_acc = aggregate_metric(metrics, "val_acc")

    return {
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }


def aggregate_eval(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    eval_acc = aggregate_metric(metrics, "eval_acc")

    return {
        "eval_acc": eval_acc,
    }
