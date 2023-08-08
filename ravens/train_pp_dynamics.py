"""Ravens main training script for MCTS primitives."""

import datetime
import os
from absl import app
from absl import flags
import numpy as np
# from ravens.dataset_pp_dynamics import DatasetPPDynamics
# from ravens.dynamics.pp_dynamics_trainer import PPDynamicsTrainer

from dataset_pp_dynamics import DatasetPPDynamics
from dynamics.pp_dynamics_trainer import PPDynamicsTrainer

# import tensorflow as tf
import torch
from torch.utils.tensorboard import SummaryWriter

import sys



flags.DEFINE_string('data_dir', './data_train', '')
flags.DEFINE_string('root_dir', './', '')
flags.DEFINE_string('model_name', 'resnet_lite', '')
flags.DEFINE_integer('n_demos', 10, '')
flags.DEFINE_integer('n_steps', 60000, '')
flags.DEFINE_integer('interval', 10000, '')
flags.DEFINE_integer('gpu', 0, '')
flags.DEFINE_integer('n_runs', 1, '')
flags.DEFINE_bool('debug', False, '')

FLAGS = flags.FLAGS

task_list = [
  'stack-tower',
  'stack-pyramid',  
  'stack-square',
  'put-block-base',
  'stack-palace',
  'stack-t']

def main(unused_argv):
  # Configure which GPU to use.
  # cfg = tf.config.experimental
  # gpus = cfg.list_physical_devices('GPU')
  # if not gpus:
  #   print('No GPUs detected. Running with CPU.')
  # else:
  #   cfg.set_visible_devices(gpus[FLAGS.gpu], 'GPU')

  ## Pytorch's way to specify GPU
  if not torch.cuda.is_available():
    print('No GPUs detected. Running with CPU.')
  else:
      # Specify the GPU
      device = torch.device(f'cuda:{FLAGS.gpu}')
      print(f'Running with GPU: {device}')


  # Dataset
  dataset_list = [os.path.join(FLAGS.data_dir, f'{task}-mcts-pp-train') for task in task_list]
  dataset = DatasetPPDynamics(dataset_list)

  # Training.
  train_dir = f'{FLAGS.root_dir}/dynamics_models/vf_{FLAGS.n_demos}'
  for train_run in range(FLAGS.n_runs):
    name = f'pp-dynamics-task{len(task_list)}-demo{FLAGS.n_demos}-{FLAGS.model_name}-seed{train_run}-step{FLAGS.n_steps}'

    # Set up tensorboard logger for pytorch
    curr_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(
        train_dir, 
        'logs', 
        name,
        curr_time, 
        'train')
    writer = SummaryWriter(log_dir)

    # Initialize trainer.
    np.random.seed(train_run)
    # tf.random.set_seed(train_run)
    torch.manual_seed(train_run)
    trainer = PPDynamicsTrainer(
      model_name=FLAGS.model_name,
      repeat_H_lambda=5)

    # Limit random sampling during training to a fixed dataset.
    for i in range(len(task_list)):
      max_demos = dataset.n_episodes_list[i]
      assert max_demos >= FLAGS.n_demos
      episodes = np.random.choice(range(max_demos), FLAGS.n_demos, False) # choose n_demos number of unique demos (each demo has a numbering)
      dataset.set(i, episodes)
    
    # Train agent and save snapshots.
    # wayne: the training loop is here?
    while trainer.total_steps < FLAGS.n_steps: # n_steps by default is 60K
      print("the CWD is", os.getcwd())
      print("SYSTEM path:", sys.path)
      trainer.train_pp(dataset, writer=writer, debug=FLAGS.debug)

      if trainer.total_steps % FLAGS.interval == 0:
        trainer.save_to_dir(train_dir, name)

if __name__ == '__main__':
  app.run(main)