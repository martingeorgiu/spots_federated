from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import FitRes, Metrics, Parameters, Scalar, parameters_to_ndarrays
from flwr.common.typing import MetricsAggregationFn
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class SaveModelStrategy(FedAvg):
    def __init__(
        self,
        log_dir: str,
        no_clients: int,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ):
        super().__init__(
            min_available_clients=no_clients,
            min_fit_clients=no_clients,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.log_dir = log_dir

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        print(f"aggregate_fit started of round {server_round}")

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving a checkpoint of round {server_round} in aggregated_ndarrays...")
            np.save(f"{self.log_dir}/round-{server_round}-weights.npy", aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics


def aggregate_fit(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    train_acc = [num_examples * m["train_acc"] for num_examples, m in metrics]
    val_loss = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_acc = [num_examples * m["val_acc"] for num_examples, m in metrics]
    no_examples = sum([num_examples for num_examples, _ in metrics])

    aggregated = {
        "train_acc": sum(train_acc) / no_examples,
        "val_loss": sum(val_loss) / no_examples,
        "val_acc": sum(val_acc) / no_examples,
    }
    return aggregated


def aggregate_eval(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["eval_acc"] for num_examples, m in metrics]
    no_examples = sum([num_examples for num_examples, _ in metrics])

    # Aggregate and return custom metric (weighted average)
    return {"eval_acc": sum(accuracies) / no_examples}
