# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower TensorBoard utilities."""
import os
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast

try:
    import tensorflow as tf
except ImportError:
    tf = None

from flwr.common import EvaluateRes, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

TBW = TypeVar("TBW")


def tensorboard(logdir: str) -> Callable[[Strategy], TBW]:
    """TensorBoard logger for Flower strategies.

    It will log loss, num_examples and all metrics which are of type float or int.

    This can either be used as a decorator as shown in the example variant 1
    or directly as shown in the example variant 2.

    Example:
        # Variant 1
        @tensorboard(logdir=LOGDIR)
        class CustomStrategy(FedAvg):
            pass

        strategy = CustomStrategy()

        # Variant 2
        strategy = tensorboard(logdir=LOGDIR)(FedAvg)()
    """
    print(
        "\n\t\033[32mStart TensorBoard with the following parameters"
        + f"\n\t$ tensorboard --logdir {logdir}\033[39m\n"
    )
    # Create logdir if it does not yet exist

    # To allow multiple runs and group those we will create a subdir
    # in the logdir which is named as number of directories in logdir + 1
    # run_id = str(
    #     len([name for name in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, name))])
    # )
    # run_id = run_id + "-" + datetime.now().strftime("%Y%m%dT%H%M%S")
    logdir_run = os.path.join(logdir, "tensorboard")
    os.makedirs(logdir, exist_ok=True)

    def decorator(strategy_class: Strategy) -> TBW:
        """Return overloaded Strategy Wrapper."""

        class TBWrapper(strategy_class):  # type: ignore
            """Strategy wrapper which hooks into some methods for TensorBoard
            logging."""

            def log_result(
                self,
                loss_aggregated: Optional[float],
                metrics_aggregated,
                results,
                server_round,
                prefix,
            ) -> None:
                # Server logs
                writer = tf.summary.create_file_writer(os.path.join(logdir_run, "server"))

                # Write aggregated loss
                with writer.as_default(step=server_round):  # pylint: disable=not-context-manager
                    if loss_aggregated is not None:
                        tf.summary.scalar(
                            f"server/{prefix}_loss_aggregated", loss_aggregated, step=server_round
                        )
                    for key, value in metrics_aggregated.items():
                        if type(value) in [int, float]:
                            tf.summary.scalar(f"server/{key}_aggregated", value)
                    writer.flush()

                if len(results) == 0:
                    return

                # Client logs
                for client, evaluate_res in results:
                    loss, num_examples, metrics = (
                        evaluate_res.loss,
                        evaluate_res.num_examples,
                        evaluate_res.metrics,
                    )

                    writer = tf.summary.create_file_writer(
                        os.path.join(logdir_run, "clients", client.cid)
                    )
                    with writer.as_default(  # pylint: disable=not-context-manager
                        step=server_round
                    ):
                        tf.summary.scalar(f"clients/{prefix}_loss", loss)
                        tf.summary.scalar(f"clients/{prefix}_num_examples", num_examples)
                        if metrics is not None:
                            for key, value in metrics.items():
                                if type(value) in [int, float]:
                                    tf.summary.scalar(f"clients/{key}", value)
                        writer.flush()

            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                """Hooks into aggregate_fit for TensorBoard logging
                purpose."""
                # Execute decorated function and extract results for logging
                # They will be returned at the end of this function but also
                # used for logging
                aggregated_parameters, metrics_aggregated = super().aggregate_fit(
                    server_round, results, failures
                )

                self.log_result(None, metrics_aggregated, [], server_round, "fit")

                return aggregated_parameters, metrics_aggregated

            def aggregate_evaluate(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, EvaluateRes]],
                failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
            ) -> Tuple[Optional[float], Dict[str, Scalar]]:
                """Hooks into aggregate_evaluate for TensorBoard logging
                purpose."""
                # Execute decorated function and extract results for logging
                # They will be returned at the end of this function but also
                # used for logging
                loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
                    server_round,
                    results,
                    failures,
                )

                self.log_result(loss_aggregated, metrics_aggregated, results, server_round, "eval")
                return loss_aggregated, metrics_aggregated

        return cast(TBW, TBWrapper)

    return decorator
