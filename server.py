import flwr as fl
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        print("1fdasdfjsafjaslfjasdfljasljflasjf")
        print(aggregated_parameters)
        print(aggregated_metrics)
        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
            print("2fdasdfjsafjaslfjasdfljasljflasjf")

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.save(f"federated-models/round-{server_round}-weights.npy", aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics


def main() -> None:
    # Define strategy
    strategy = SaveModelStrategy(

    )

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()