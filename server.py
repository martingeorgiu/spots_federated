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
    parameters_to_ndarrays
)
from flwr.server.utils import tensorboard
from flwr.server.client_proxy import ClientProxy

class SaveModelStrategy(FedAvg):
    def __init__(self, log_dir:str):
        super().__init__(min_available_clients=3,min_fit_clients=3)
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


def main() -> None:
    # Define strategy
    start_time = datetime.datetime.now().replace(microsecond=0).isoformat()
    log_dir = f"federated-models/{start_time}"
    strategy = tensorboard(logdir=log_dir)(SaveModelStrategy)(log_dir)

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()