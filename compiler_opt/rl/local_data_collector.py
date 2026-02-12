# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module for collecting data directly from corpus without workers."""

import time
from collections.abc import Callable, Iterator

from absl import logging
from tf_agents.trajectories import trajectory

from compiler_opt.rl import best_trajectory
from compiler_opt.rl import compilation_runner
from compiler_opt.rl import corpus
from compiler_opt.rl import data_collector
from compiler_opt.rl import policy_saver
from compiler_opt.distributed import worker
from pdb import set_trace


class LocalDataCollector(data_collector.DataCollector):
    """Class for local data collection - no workers, direct corpus access."""

    def __init__(
        self,
        cps: corpus.Corpus,
        num_modules: int,
        worker_pool: worker.WorkerPool,
        parser: Callable[[list[str]], Iterator[trajectory.Trajectory]],
        reward_stat_map: dict[str, dict[str, compilation_runner.RewardStat]] = {},
        best_trajectory_repo: best_trajectory.BestTrajectoryRepo | None = None,
    ):
        super().__init__()
        
        self._corpus = cps
        self._num_modules = num_modules
        self._parser = parser
        self._reward_stat_map = reward_stat_map
        self._best_trajectory_repo = best_trajectory_repo

    def collect_data(
        self,
        policy: policy_saver.Policy,
        model_id: int,
    ) -> tuple[Iterator[trajectory.Trajectory], dict[str, dict[str, float]]]:
        """Collect data directly from corpus without workers.

        Args:
            policy: a policy_saver.Policy object to collect data with.
            model_id: the model identifier for logging purposes.

        Returns:
            A tuple of (trajectory iterator, monitor dictionary).
        """
        # Sample modules from corpus
        sampled_module_names = self._corpus.sample(k=self._num_modules, sort=False)
        logging.info('Sampling %d modules for collection', len(sampled_module_names))

        # Process modules directly - no workers involved
        sequence_examples: list[str] = []
        successful_work: list[tuple[str, list[str], list[float]]] = []
        
        start_time = time.time()
        
        for module_name in sampled_module_names:
            try:
                # Load module spec directly from corpus
                result = loaded_module_spec = self._corpus.load_module_spec(module_name.name)
                
                # Simulate compilation with policy to get trajectory data
                # This is a placeholder - replace with your actual data collection logic
                #result = self._collect_data_for_module(loaded_module_spec, policy)
                
                #if result and result.serialized_sequence_examples:
                set_trace()
                if result:
                    # Successfully collected data for this module
                    #successful_work.append((module_name, result.serialized_sequence_examples, result.rewards))
                    successful_work.append((module_name.name, result.loaded_ir))
                    #sequence_examples.extend(result.serialized_sequence_examples)
                else:
                    logging.warning(f"Failed to collect data for module: {module_name}")
                    
            except Exception as e:
                logging.error(f"Exception during data collection for module {module_name}: {e}")
                continue
        
        total_time = time.time() - start_time
        logging.info(
            '%d of %d modules finished in %.2f seconds',
            len(successful_work),
            len(sampled_module_names),
            total_time,
        )

        # Early return if no data was collected
        if not sequence_examples:
            logging.warning('No sequence examples collected. Returning empty iterators.')
            return iter([]), {}

        # Parse trajectories from sequence examples
        trajectories_iter = self._parser(sequence_examples)

        # Build monitor dictionary
        total_trajectory_length = sum(len(seq_ex) for _, seq_ex, _ in successful_work)
        monitor_dict: dict[str, dict[str, float]] = {
            'default': {
                'success_modules': len(successful_work),
                'total_trajectory_length': total_trajectory_length,
            },
        }

        # Collect rewards
        all_rewards = []
        for _, _, rewards in successful_work:
            all_rewards.extend(rewards)
        
        monitor_dict['reward_distribution'] = data_collector.build_distribution_monitor(all_rewards)

        # Handle best trajectory repository if provided
        if self._best_trajectory_repo is not None:
            for module_name, seq_exs, rewards in successful_work:
                for i, (sequence_example, reward) in enumerate(zip(seq_exs, rewards)):
                    identifier = f"{model_id}_{i}"
                    self._best_trajectory_repo.update_if_better_trajectory(
                        module_name, identifier, reward, sequence_example
                    )

        return trajectories_iter, monitor_dict

    # def _collect_data_for_module(
    #     self,
    #     loaded_module_spec: corpus.LoadedModuleSpec,
    #     policy: policy_saver.Policy,
    # ): #-> compilation_runner.WorkResult:
    #     """Collect trajectory data for a single module directly.

    #     Replace this with your actual data collection logic.
    #     """
    #     # This is a placeholder implementation
    #     # You would replace this with your actual compilation logic
    #     return compilation_runner.WorkResult(
    #         serialized_sequence_examples=["dummy_data"],
    #         rewards=[0.0],
    #         length=1,
    #         policy_rewards=[0.0],
    #         keys=["dummy_key"],
    #     )

    def close_pool(self) -> None:
        """Clean up resources."""
        logging.info('Closing data collector - no workers to close')

    def on_dataset_consumed(self, dataset_iterator: Iterator[trajectory.Trajectory]):
        """Callback when dataset has been consumed."""
        pass