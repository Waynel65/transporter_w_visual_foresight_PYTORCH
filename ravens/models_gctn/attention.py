#!/usr/bin/env python

import os
import sys
import cv2
import numpy as np
# import tensorflow as tf
# import tensorflow_addons as tfa

from ravens.models_gctn.resnet import ResNet43_8s
from ravens.utils_gctn import utils

import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F


class Attention:
    """Daniel: attention model implemented as an hourglass FCN.

    Used for the picking network, and for placing if doing the 'no-transport'
    ablation. By default our TransporterAgent class (using this for picking)
    has num_rotations=1, leaving rotations to the Transport component. Input
    shape is (320,160,6) with 3 height maps, and then we change it so the H
    and W are both 320 (to support rotations).

    In the normal Transporter model, this component only uses one rotation,
    so the label is just sized at (320,160,1).
    """

    def __init__(self, input_shape, num_rotations, preprocess):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_rotations = num_rotations
        self.preprocess = preprocess

        max_dim = np.max(input_shape[:2])

        self.padding = np.zeros((3, 2), dtype=int)
        pad = (max_dim - np.array(input_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)

        input_shape = np.array(input_shape)
        input_shape += np.sum(self.padding, axis=1)
        input_shape = tuple(input_shape)

        # Initialize fully convolutional Residual Network with 43 layers and
        # 8-stride (3 2x2 max pools and 3 2x bilinear upsampling)
        # d_in, d_out = ResNet43_8s(input_shape, 1)
        # self.model = tf.keras.models.Model(inputs=[d_in], outputs=[d_out])
        # self.optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
        # self.metric = tf.keras.metrics.Mean(name='attention_loss')

        # d_in, d_out = ResNet43_8s(in_shape, 1)
        self.model = ResNet43_8s(input_shape[2], 1).to(self.device) # Wayne: instantiate the model here
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        

    def forward(self, in_img, softmax=True):
        # print(f"[DEBUG] Original in_img shape: {in_img.shape}")

        in_data = np.pad(in_img, self.padding, mode='constant')
        # print(f"[DEBUG] in_data shape after padding: {in_data.shape}")

        in_data = self.preprocess(in_data)
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape)
        in_tens = torch.from_numpy(in_data).float().to(self.device)

        # Rotate input.
        pivot = torch.tensor(in_data.shape[1:3]) / 2
        # print(f"[DEBUG] pivot: {pivot}")

        rvecs = self.get_se2(self.num_rotations, pivot)
        in_tens = in_tens.repeat(self.num_rotations, 1, 1, 1)
        # print(f"[DEBUG] in_tens shape after repeat: {in_tens.shape}")

        rotated_tens = torch.empty_like(in_tens)
        for i in range(self.num_rotations):
            rvec = rvecs[i]
            angle = np.arctan2(rvec[1], rvec[0]) * 180 / np.pi
            rotated_tens[i] = TF.rotate(in_tens[i], angle)
        in_tens = rotated_tens


        # print(f"[DEBUG] in_tens shape after rotation: {in_tens.shape}")    

        # Forward pass.
        logits = []
        for x in torch.split(in_tens, 1):
            x = x.permute(0, 3, 1, 2)
            out = self.model(x)
            # print(f"[DEBUG] out shape before concatenation: {out.shape}")
            logits.append(out)
        logits = torch.cat(logits, dim=0)
        # print(f"[DEBUG] logits shape after concatenation: {logits.shape}")

        # Rotate back output.
        rvecs = self.get_se2(self.num_rotations, pivot, reverse=True)
        rotated_logits = torch.empty_like(logits)
        for i in range(self.num_rotations):
            rvec = rvecs[i]
            angle = np.arctan2(rvec[1], rvec[0]) * 180 / np.pi
            rotated_logits[i] = TF.rotate(logits[i], angle)
        logits = rotated_logits


        c0 = torch.tensor(self.padding[:2, 0])
        # print(f"[DEBUG] c0 shape after concatenation: {c0.shape}")
        c1 = c0 + torch.tensor(in_img.shape[:2])
        # print(f"[DEBUG] c1 shape after concatenation: {c1.shape}")
        logits = logits[:, c0[0]:c1[0], c0[1]:c1[1], :]

        # print(f"[DEBUG] logits shape after slicing: {logits.shape}")

        logits = logits.permute(3, 1, 2, 0)
        output = logits.reshape(1, -1)
        print(f"[DEBUG] Final output shape: {output.shape}")

        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.view(logits.shape[1:])
        return output


    def train(self, in_img, p, theta, backprop=True):
        # self.metric.reset_states()
        print(f"[ATTENTION in train] in_img has shape of {in_img.shape}")
        output = self.forward(in_img, softmax=False)

        # Get label.
        theta_i = theta / (2 * np.pi / self.num_rotations)
        theta_i = int(np.round(theta_i)) % self.num_rotations
        label_size = in_img.shape[:2] + (self.num_rotations,)
        label = np.zeros(label_size)
        label[p[0], p[1], theta_i] = 1
        label = label.reshape(1, -1)
        label = torch.from_numpy(label).long()
        label = label.to(self.device)
        print(f"[ATTENTION in train] Output has shape of {output.shape}")
        print(f"[ATTENTION in train] Label has shape of {label.shape}")

        # Get loss.
        # loss_fn = torch.nn.BCEWithLogitsLoss()  # Cross-entropy loss with logits
        # loss = loss_fn(output, label)
        # loss_mean = loss.mean()

        label_indices = torch.argmax(label, dim=-1)  # Convert from one-hot to class indices
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(output, label_indices)
        loss = torch.mean(loss)

        # Backpropagate
        if backprop:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            # self.metric(loss)
        
        return loss.item()

    def load(self, path):
        # self.model.load_weights(path)
        # raise NotImplementedError("need to write load for attention")
        self.model.load_state_dict(torch.load(path))

    def save(self, filename):
        # self.model.save(filename)
        torch.save(self.model.state_dict(), filename)

    def get_se2(self, num_rotations, pivot, reverse=False):
        '''
        Get SE2 rotations discretized into num_rotations angles counter-clockwise.
        Returns list (np.array) where each item is a flattened SE2 rotation matrix.
        '''
        rvecs = []
        for i in range(num_rotations):
            theta = i * 2 * np.pi / num_rotations
            theta = -theta if reverse else theta
            rmat = utils.get_image_transform(theta, (0, 0), pivot)
            rvec = rmat.reshape(-1)[:-1]
            rvecs.append(rvec)
        return np.array(rvecs, dtype=np.float32)

    def get_attention_heatmap(self, attention):
        """Given attention, get a human-readable heatmap.

        https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html  
        In my normal usage, the attention is already softmax-ed but just be
        aware in case it's not. Also be aware of RGB vs BGR mode. We should
        ensure we're in BGR mode before saving. Also with RAINBOW mode, red =
        hottest (highest attention values), green=medium, blue=lowest.

        Note: to see the grayscale only (which may be easier to interpret,
        actually...) save `vis_attention` just before applying the colormap.
        """
        # Options: cv2.COLORMAP_PLASMA, cv2.COLORMAP_JET, etc.
        #attention = tf.reshape(attention, (1, np.prod(attention.shape)))
        #attention = tf.nn.softmax(attention)
        vis_attention = np.float32(attention).reshape((320, 160))
        vis_attention = vis_attention - np.min(vis_attention)
        vis_attention = 255 * vis_attention / np.max(vis_attention)
        vis_attention = cv2.applyColorMap(np.uint8(vis_attention), cv2.COLORMAP_RAINBOW)
        vis_attention = cv2.cvtColor(vis_attention, cv2.COLOR_RGB2BGR)
        return vis_attention