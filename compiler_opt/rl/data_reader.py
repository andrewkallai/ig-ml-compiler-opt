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
"""util function to create training datasets."""

from collections.abc import Callable, Iterator

import tensorflow as tf
from tf_agents.trajectories import trajectory

from compiler_opt.rl import agent_config


def create_parser_fn(
    agent_cfg: agent_config.AgentConfig
) -> Callable[[str], trajectory.Trajectory]:
  """Create a parser function for reading from a serialized tf.SequenceExample.

  Args:
    agent_name: AgentName, enum type of the agent.
    time_step_spec: time step spec of the optimization problem.
    action_spec: action spec of the optimization problem.

  Returns:
    A callable that takes scalar serialized proto Tensors and emits
    `Trajectory` objects containing parsed tensors.
  """

  def _parser_fn(serialized_proto):
    """Helper function that is returned by create_`parser_fn`."""
    # We copy through all context features at each frame, so even though we know
    # they don't change from frame to frame, they are still sequence features
    # and stored in the feature list.
    context_features = {}
    # pylint: disable=g-complex-comprehension
    sequence_features = {
        tensor_spec.name:
            tf.io.FixedLenSequenceFeature(
                shape=tensor_spec.shape, dtype=tensor_spec.dtype)
        for tensor_spec in agent_cfg.time_step_spec.observation.values()
    }
    sequence_features[
        agent_cfg.action_spec.name] = tf.io.FixedLenSequenceFeature(
            shape=agent_cfg.action_spec.shape,
            dtype=agent_cfg.action_spec.dtype)
    sequence_features[
        agent_cfg.time_step_spec.reward.name] = tf.io.FixedLenSequenceFeature(
            shape=agent_cfg.time_step_spec.reward.shape,
            dtype=agent_cfg.time_step_spec.reward.dtype)
    sequence_features.update(agent_cfg.get_policy_info_parsing_dict())

    # pylint: enable=g-complex-comprehension
    with tf.name_scope('parse'):
      _, parsed_sequence = tf.io.parse_single_sequence_example(
          serialized_proto,
          context_features=context_features,
          sequence_features=sequence_features)
      # TODO(yundi): make the transformed reward configurable.
      action = parsed_sequence[agent_cfg.action_spec.name]
      reward = tf.cast(parsed_sequence[agent_cfg.time_step_spec.reward.name],
                       tf.float32)

      policy_info = agent_cfg.process_parsed_sequence_and_get_policy_info(
          parsed_sequence)

      del parsed_sequence[agent_cfg.time_step_spec.reward.name]
      del parsed_sequence[agent_cfg.action_spec.name]
      full_trajectory = trajectory.from_episode(
          observation=parsed_sequence,
          action=action,
          policy_info=policy_info,
          reward=reward)
      return full_trajectory

  return _parser_fn


def create_sequence_example_dataset_fn(
    agent_cfg: agent_config.AgentConfig, batch_size: int,
    train_sequence_length: int) -> Callable[[list[str]], Iterator[trajectory.Trajectory]]:
  """Get a generator that yields batches of parsed trajectories.

  Avoids tf.data.Dataset overhead (Iterator::Root::Prefetch, etc.) by parsing
  eagerly and yielding batches from a Python generator.

  Args:
    agent_cfg: agent config with specs.
    batch_size: int, batch size B.
    train_sequence_length: int, trajectory sequence length T.

  Returns:
    A callable that takes a list of serialized sequence examples and returns
      an infinite iterator yielding batched `Trajectory` with shape [B, T, ...].
  """
  parser_fn = create_parser_fn(agent_cfg)

  def _batched_generator(sequence_examples: list[str]):
    trajectories = []
    for serialized in sequence_examples:
      if not serialized:
        continue
      traj = parser_fn(tf.constant(serialized))
      if tf.size(traj.reward) > 2:
        trajectories.append(traj)

    if not trajectories:
      return

    num_frames_per_batch = train_sequence_length * batch_size

    def slice_traj(traj, start, length):
      return tf.nest.map_structure(
          lambda t: t[start:start + length], traj)

    def concat_frames(trajs):
      return tf.nest.map_structure(
          lambda *ts: tf.concat(list(ts), axis=0), *trajs)

    all_frames = concat_frames(trajectories)

    num_frames = tf.shape(
        tf.nest.flatten(all_frames)[0])[0].numpy()

    all_batches = []
    for b in range(num_frames // num_frames_per_batch):
      start = b * num_frames_per_batch
      batch_chunks = []
      for t in range(batch_size):
        chunk_start = start + t * train_sequence_length
        chunk = slice_traj(all_frames, chunk_start, train_sequence_length)
        batch_chunks.append(chunk)

      batched = tf.nest.map_structure(
          lambda *ts: tf.stack(ts), *batch_chunks)
      all_batches.append(batched)

    idx = 0
    while True:
      if idx < len(all_batches):
        yield all_batches[idx]
        idx += 1
      else:
        idx = 0

  return _batched_generator


# TODO(yundi): PyType check of input_dataset as Type[tf.data.Dataset] is not
# working.
def create_file_dataset_fn(
    agent_cfg: agent_config.AgentConfig,
    batch_size: int,
    train_sequence_length: int,
    input_dataset,
    shuffle_repeat_count: int | None) -> Callable[[list[str]], tf.data.Dataset]:
  """Get a function that creates an dataset from files.

  Args:
    agent_name: AgentName, enum type of the agent.
    time_step_spec: time step spec of the optimization problem.
    action_spec: action spec of the optimization problem.
    batch_size: int, batch_size B.
    train_sequence_length: int, trajectory sequence length T.
    input_dataset: A tf.data.Dataset subclass object.
    shuffle_repeat_count: The number of times to repeat a shuffled version of
      the dataset, or None to repeat indefinitely.

  Returns:
    A callable that takes file path(s) and returns a `tf.data.Dataset`.
      Iterating over this dataset yields `trajectory.Trajectory` instances with
      shape `[B, T, ...]`.
  """
  #files_buffer_size = 100
  files_buffer_size = 2
  num_readers = 10
  num_map_threads = 8
  #shuffle_buffer_size = 1024
  shuffle_buffer_size = 2
  #trajectory_shuffle_buffer_size = 1024
  trajectory_shuffle_buffer_size = 2

  parser_fn = create_parser_fn(agent_cfg)

  def _file_dataset_fn(data_path):
    dataset = (
        tf.data.Dataset.list_files(data_path).shuffle(
            files_buffer_size).interleave(
                input_dataset, cycle_length=num_readers, block_length=1)
        # Due to a bug in collection, we sometimes get empty rows.
        .filter(lambda string: tf.strings.length(string) > 0)
        .apply(tf.data.experimental.shuffle_and_repeat(
                 shuffle_buffer_size,
                 count=shuffle_repeat_count))
        .map(parser_fn, num_parallel_calls=num_map_threads)
        # Only keep sequences of length 2 or more.
        .filter(lambda traj: tf.size(traj.reward) > 2))

    # TODO(yundi): window and subsample data.
    # TODO(yundi): verify the shuffling is correct.
    dataset = (
        dataset.unbatch().batch(
            train_sequence_length,
            drop_remainder=True))
            # drop_remainder=True).shuffle(trajectory_shuffle_buffer_size).batch(
            #     batch_size,
            #     drop_remainder=True))
                #drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
    return dataset

  return _file_dataset_fn


def create_tfrecord_dataset_fn(
    agent_cfg: agent_config.AgentConfig, batch_size: int,
    train_sequence_length: int,
    shuffle_repeat_count: int | None = None
) -> Callable[[list[str]], tf.data.Dataset]:
  """Get a function that creates an dataset from tfrecord.

  Args:
    agent_name: AgentName, enum type of the agent.
    time_step_spec: time step spec of the optimization problem.
    action_spec: action spec of the optimization problem.
    batch_size: int, batch_size B.
    train_sequence_length: int, trajectory sequence length T.
    shuffle_repeat_count: The number of times to repeat a shuffled version of
      the dataset, or None to repeat indefinitely.

  Returns:
    A callable that takes tfrecord path(s) and returns a `tf.data.Dataset`.
      Iterating over this dataset yields `trajectory.Trajectory` instances with
      shape `[B, T, ...]`.
  """
  return create_file_dataset_fn(
      agent_cfg,
      batch_size,
      train_sequence_length,
      input_dataset=tf.data.TFRecordDataset,
      shuffle_repeat_count=shuffle_repeat_count)
