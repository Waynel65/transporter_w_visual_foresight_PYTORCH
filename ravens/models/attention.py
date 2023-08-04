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

"""Attention module."""

import numpy as np
from ravens.models.resnet import ResNet36_4s
from ravens.models.resnet import ResNet43_8s
from ravens.utils import utils
import tensorflow as tf
from tensorflow_addons import image as tfa_image

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF


class Attention:
  """Attention module."""

  def __init__(self, in_shape, n_rotations, preprocess, lite=False):
    self.n_rotations = n_rotations
    self.preprocess = preprocess

    max_dim = np.max(in_shape[:2])

    self.padding = np.zeros((3, 2), dtype=int)
    pad = (max_dim - np.array(in_shape[:2])) / 2
    self.padding[:2] = pad.reshape(2, 1)

    in_shape = np.array(in_shape)
    in_shape += np.sum(self.padding, axis=1)
    in_shape = tuple(in_shape)

    # Initialize fully convolutional Residual Network with 43 layers and
    # 8-stride (3 2x2 max pools and 3 2x bilinear upsampling)
    if lite:
      # d_in, d_out = ResNet36_4s(in_shape, 1)
      self.model = ResNet36_4s(in_shape[2], out_channel).to(self.device) # Wayne: instantiate the model here
    else:
      # d_in, d_out = ResNet43_8s(in_shape, 1)
      self.model = ResNet43_8s(in_shape[2], out_channel).to(self.device) # Wayne: instantiate the model here


    # self.model = tf.keras.models.Model(inputs=[d_in], outputs=[d_out])
    # self.optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
    # self.metric = tf.keras.metrics.Mean(name='loss_attention')

    self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)


  def forward(self, in_img, softmax=True):
    in_data = np.pad(in_img, self.padding, mode='constant')
    in_data = self.preprocess(in_data)
    in_shape = (1,) + in_data.shape
    in_data = in_data.reshape(in_shape)
    in_tens = torch.from_numpy(in_data).float()

    # Rotate input.
    pivot = torch.tensor(in_data.shape[1:3]) / 2
    rvecs = self.get_se2(self.n_rotations, pivot)
    in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
    for i in range(self.n_rotations):
        in_tens[i] = TF.rotate(in_tens[i], rvecs[i])

    # Forward pass.
    logits = []
    for x in torch.split(in_tens, 1):
        logits.append(self.model(x))
    logits = torch.cat(logits, dim=0)

    # Rotate back output.
    rvecs = self.get_se2(self.n_rotations, pivot, reverse=True)
    for i in range(self.n_rotations):
        logits[i] = TF.rotate(logits[i], -rvecs[i]) # assuming rvecs are in degrees
    c0 = self.padding[:2, 0]
    c1 = c0 + torch.tensor(in_img.shape[:2])
    logits = logits[:, c0[0]:c1[0], c0[1]:c1[1], :]

    logits = logits.permute(3, 1, 2, 0)
    output = logits.view(1, -1)
    if softmax:
        output = F.softmax(output, dim=-1)
        output = output.view(logits.shape[1:])
    return output

  def train(self, in_img, p, theta, backprop=True):
    self.metric.reset_states()
    output = self.forward(in_img, softmax=False)

    # Get label.
    theta_i = theta / (2 * np.pi / self.n_rotations)
    theta_i = int(np.round(theta_i)) % self.n_rotations
    label_size = in_img.shape[:2] + (self.n_rotations,)
    label = np.zeros(label_size)
    label[p[0], p[1], theta_i] = 1
    label = label.reshape(1, -1)
    label = torch.from_numpy(label).float()

    # Get loss.
    loss = F.cross_entropy(output, label)
    loss = torch.mean(loss)

    # Backpropagate
    if backprop:
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        # self.metric(loss)

    return loss.item()

  def load(self, path):
    self.model.load_weights(path)

  def save(self, filename):
    self.model.save(filename)

  def get_se2(self, n_rotations, pivot, reverse=False):
    """Get SE2 rotations discretized into n_rotations angles counter-clockwise."""
    rvecs = []
    for i in range(n_rotations):
      theta = i * 2 * np.pi / n_rotations
      theta = -theta if reverse else theta
      rmat = utils.get_image_transform(theta, (0, 0), pivot)
      rvec = rmat.reshape(-1)[:-1]
      rvecs.append(rvec)
    return np.array(rvecs, dtype=np.float32)
