# coding=utf-8
# Copyright 2021 The Ravens Authors.
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

"""MLP ground-truth state module."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MlpModel(nn.Module):
  """MLP ground-truth state module."""
  def __init__(self, batch_size, d_obs, d_action, activation=F.relu, mdn=False, dropout=0.2, use_sinusoid=True):
      super(MlpModel, self).__init__()
      self.normalize_input = True
      self.use_sinusoid = use_sinusoid

      if self.use_sinusoid:
          k = 3
      else:
          k = 1

      self.fc1 = nn.Linear(d_obs * k, 128)
      self.drop1 = nn.Dropout(dropout)
      self.fc2 = nn.Linear(128 * k, 128)
      self.drop2 = nn.Dropout(dropout)
      self.fc3 = nn.Linear(128 * k, d_action)

      self.mdn = mdn
      if self.mdn:
          k = 26
          self.mu = nn.Linear(128 * k, d_action * k)
          self.logvar = nn.Linear(128 * k, k)
          self.pi = nn.Linear(128 * k, k)
          self.softmax = nn.Softmax(dim=1)
          self.temperature = 2.5

  def reset_states(self):
      pass

  def set_normalization_parameters(self, obs_train_parameters):
      self.obs_train_mean = obs_train_parameters["mean"]
      self.obs_train_std = obs_train_parameters["std"]

  def forward(self, x):
      def cs(input_tensor):
          if self.use_sinusoid:
              sin = torch.sin(input_tensor)
              cos = torch.cos(input_tensor)
              return torch.cat((input_tensor, cos, sin), dim=1)
          else:
              return input_tensor

      obs = x.clone()

      x = self.drop1(self.fc1(cs(obs)))
      x = self.drop2(self.fc2(torch.cat((x, cs(obs)), dim=1)))
      x = torch.cat((x, cs(obs)), dim=1)

      if not self.mdn:
          x = self.fc3(x)
          return x
      else:
          pi = self.pi(x)
          pi = pi / self.temperature
          pi = self.softmax(pi)

          mu = self.mu(x)
          var = torch.exp(self.logvar(x))
          return (pi, mu, var)

