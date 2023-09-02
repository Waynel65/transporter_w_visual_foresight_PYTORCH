#!/usr/bin/env python

import os
import sys

import cv2
import numpy as np
# import tensorflow as tf
# import tensorflow_addons as tfa
import matplotlib.pyplot as plt

from ravens.models import ResNet43_8s
from ravens import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import pdb

class TripleResnet(nn.Module):
    """
        This class will encapsulate the three resnets used
        in the original TensorFlow version of TransportGoal
        We have to manually define this because PyTorch doesn't
        support one line encapulation of multiple models. 
        We still need to define the init and forward function
    """

    def __init__(self, in_channel, output_dim):
        super(TripleResnet, self).__init__()

        self.resnet1 = ResNet43_8s(in_channel, output_dim)
        self.resnet2 = ResNet43_8s(in_channel, output_dim)
        self.resnet3 = ResNet43_8s(in_channel, output_dim)

    def forward(self, in_tensor, goal_tensor):


        in_logits = self.resnet1(in_tensor)
        kernel_nocrop_logits = self.resnet2(in_tensor)
        goal_logits = self.resnet3(goal_tensor)

        return in_logits, kernel_nocrop_logits, goal_logits


class TransportGoal:
    """Daniel: Transporter for the placing module, with goal images.

    Built on top of the normal Transporters class, with three FCNs. We assume
    by nature that we have a goal image. We also crop after the query, and
    will not use per-pixel losses, so ignore those vs normal transporters.
    """

    def __init__(self, input_shape, num_rotations, crop_size, preprocess):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_rotations = num_rotations
        self.crop_size = crop_size  # crop size must be N*16 (e.g. 96)
        self.preprocess = preprocess

        self.pad_size = int(self.crop_size / 2)
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = self.pad_size

        input_shape = np.array(input_shape)
        input_shape[0:2] += self.pad_size * 2
        input_shape = tuple(input_shape)
        self.odim = output_dim = 3

        print(f"input shape is {input_shape}")

        self.model = TripleResnet(input_shape[2], output_dim)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # 3 fully convolutional ResNets. Third one is for the goal.
        # in0, out0 = ResNet43_8s(input_shape, output_dim, prefix='s0_')
        # in1, out1 = ResNet43_8s(input_shape, output_dim, prefix='s1_')
        # in2, out2 = ResNet43_8s(input_shape, output_dim, prefix='s2_')

        # self.resnet1 = ResNet43_8s(input_shape[2], output_dim).to(self.device)
        # self.resnet2 = ResNet43_8s(input_shape[2], output_dim).to(self.device)
        # self.resnet3 = ResNet43_8s(input_shape[2], output_dim).to(self.device)

        # self.model = tf.keras.Model(inputs=[in0, in1, in2], outputs=[out0, out1, out2])
        # self.optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
        # self.metric = tf.keras.metrics.Mean(name='transport_loss')


    def forward(self, in_img, goal_img, p, apply_softmax=True):
        """Forward pass of our goal-conditioned Transporter.

        Relevant shapes and info:

            in_img and goal_img: (320,160,6)
            p: integer pixels on in_img, e.g., [158, 30]
            self.padding: [[32,32],[32,32],0,0]], with shape (3,2)

        Run input through all three networks, to get output of the same
        shape, except that the last channel is 3 (output_dim). Then, the
        output for one stream has the convolutional kernels for another. Call
        tf.nn.convolution. That's it, and the operation is be differentiable,
        so that gradients apply to all the FCNs.

        I actually think cropping after the query network is easier, because
        otherwise we have to do a forward pass, then call tf.multiply, then
        do another forward pass, which splits up the computation.
        """

        # pdb.set_trace()
        print(f"[TRANS_goal] in_img.shape: {in_img.shape}")
        assert in_img.shape == goal_img.shape, f'{in_img.shape}, {goal_img.shape}'

        # input image --> Torch tensor
        input_unproc = np.pad(in_img, self.padding, mode='constant')    # (384,224,6)
        input_data = self.preprocess(input_unproc.copy())               # (384,224,6)
        input_shape = (1,) + input_data.shape
        input_data = input_data.reshape(input_shape)                    # (1,384,224,6)
        in_tensor = torch.from_numpy(input_data).float().permute(0, 3, 1, 2)  # (1,6,384,224)
        # in_tensor = in_tensor.to(self.device)
        print(f"[TRANS_goal] in_tensor.shape: {in_tensor.shape}")

        # goal image --> Torch tensor
        goal_unproc = np.pad(goal_img, self.padding, mode='constant')   # (384,224,6)
        goal_data = self.preprocess(goal_unproc.copy())                 # (384,224,6)
        goal_shape = (1,) + goal_data.shape
        goal_data = goal_data.reshape(goal_shape)                       # (1,384,224,6)
        goal_tensor = torch.from_numpy(goal_data).float().permute(0, 3, 1, 2) # (1,6,384,224)
        # goal_tensor = goal_tensor.to(self.device)
        print(f"[TRANS_goal] goal_tensor.shape: {goal_tensor.shape}")

        # Get SE2 rotation vectors for cropping.
        pivot = np.array([p[1], p[0]]) + self.pad_size # here p is based on the output of attention
        rvecs = self.get_se2(self.num_rotations, pivot)
        print(f"[TRANS_goal] RVECS have a shape of {rvecs.shape}")

        # pdb.set_trace()
        # pytorch convention start #
        in_logits, kernel_nocrop_logits, goal_logits = self.model(in_tensor, goal_tensor)
        # conduct re-permute here to avoid problems
        # in_logits = in_logits.permute(0, 2, 3, 1)
        # kernel_nocrop_logits = kernel_nocrop_logits.permute(0, 2, 3, 1)
        # goal_logits = goal_logits.permute(0, 2, 3, 1)
        
        # Use features from goal logits and combine with input and kernel.
        goal_x_in_logits     = goal_logits * in_logits
        goal_x_kernel_logits = goal_logits * kernel_nocrop_logits

        # Crop the kernel_logits about the picking point and get rotations.
        crop = goal_x_kernel_logits.clone()                                 # (1,3,384,224)
        crop = crop.repeat(self.num_rotations, 1, 1, 1)                     # (24,3,384,224)

        rotated_crop = torch.empty_like(crop)
        for i in range(self.num_rotations):
            rvec = rvecs[i]
            angle = np.arctan2(rvec[1], rvec[0]) * 180 / np.pi
            rotated_crop[i] = T.functional.rotate(crop[i], angle, interpolation=T.InterpolationMode.NEAREST)
        crop = rotated_crop
        
        # pdb.set_trace()

        # slicing based on torch's convention
        kernel = crop[:, :,
        p[0]:(p[0] + self.crop_size),
        p[1]:(p[1] + self.crop_size)]

        # print(f"[TRANS_GOAL] kernel shape: {kernel.shape} | the rest: {(self.num_rotations, self.crop_size, self.crop_size, self.odim)}")
        # assert kernel.shape == (self.num_rotations, self.crop_size, self.crop_size, self.odim)
        assert kernel.shape == (self.num_rotations, self.odim, self.crop_size, self.crop_size)
        # at this point we should have kernel shape == (36,3,64,64)

        kernel = F.pad(kernel, (0, 1, 0, 1)) # this gives (36,3,65,65)
        # ! do we really need depth convolution here?
        # kernel_shape = kernel.shape
        # kernel = kernel.view(kernel_shape[0]*kernel_shape[1], 1, kernel_shape[2], kernel_shape[3])                            
        # output = F.conv2d(goal_x_in_logits, kernel, groups=3) # cross-convolution

        output = F.conv2d(goal_x_in_logits, kernel) # regular convolution
        output = (1 / (self.crop_size**2)) * output # normalization

        # output = output.permute(0, 2, 3, 1)

        # pdb.set_trace()

        if apply_softmax:
            output_shape = output.shape
            output = output.view(1,-1)
            output = F.softmax(output, dim=1)
            output = output.view(output_shape[1:])
            # output = output.detach().numpy()
        print(f"[DEBUG in trans_goal] final output shape: {output.shape}")

        return output


    def train(self, in_img, goal_img, p, q, theta):
        """Transport Goal training.

        Both `in_img` and `goal_img` have the color and depth. Much is
        similar to the attention model: (a) forward pass, (b) get angle
        discretizations, (c) make the label consider rotations in the last
        axis, but only provide the label to one single (pixel,rotation).
        """
        # self.metric.reset_states()
        output = self.forward(in_img, goal_img, p, apply_softmax=False) # still in pytorch format and on device
        output = output.to(self.device)

        # Compute label
        itheta = theta / (2 * np.pi / self.num_rotations)
        itheta = int(np.round(itheta)) % self.num_rotations
        label_size = in_img.shape[:2] + (self.num_rotations,)
        label = np.zeros(label_size)
        label[q[0], q[1], itheta] = 1
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).float().to(self.device)

        # Compute loss after re-shaping the output.
        output = output.view(1, -1)
        print(f"[trans_goal] output shape: {output.shape} | label shape: {label.shape}")
        # loss = F.cross_entropy(output, label)
        # loss = torch.mean(loss)

        # loss_fn = torch.nn.BCEWithLogitsLoss()  # Cross-entropy loss with logits
        # loss = loss_fn(output, label)
        # loss_mean = loss.mean()

        label_indices = torch.argmax(label, dim=-1)  # Convert from one-hot to class indices
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(output, label_indices)
        loss = torch.mean(loss)
        # Backward pass and optimization
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # self.metric.update(loss.item())

        return loss.item()

    def get_se2(self, num_rotations, pivot):
        '''
        Get SE2 rotations discretized into num_rotations angles counter-clockwise.
        '''
        rvecs = []
        for i in range(num_rotations):
            theta = i * 2 * np.pi / num_rotations
            rmat = utils.get_image_transform(theta, (0, 0), pivot)
            rvec = rmat.reshape(-1)[:-1]
            rvecs.append(rvec)
        return np.array(rvecs, dtype=np.float32)

    def save(self, fname):
        # self.model.save(fname)
        torch.save(self.model.state_dict(), fname)

    def load(self, fname):
        # self.model.load_weights(fname)
        raise NotImplementedError("need to write load for transport goal")

    #-------------------------------------------------------------------------
    # Visualization.
    #-------------------------------------------------------------------------

    def visualize_images(self, p, in_img, input_data, crop):
        def get_itheta(theta):
            itheta = theta / (2 * np.pi / self.num_rotations)
            return np.int32(np.round(itheta)) % self.num_rotations

        plt.subplot(1, 3, 1)
        plt.title(f'Perturbed', fontsize=15)
        plt.imshow(np.array(in_img[:, :, :3]).astype(np.uint8))
        plt.subplot(1, 3, 2)
        plt.title(f'Process/Pad', fontsize=15)
        plt.imshow(input_data[0, :, :, :3])
        plt.subplot(1, 3, 3)
        # Let's stack two crops together.
        theta1 = 0.0
        theta2 = 90.0
        itheta1 = get_itheta(theta1)
        itheta2 = get_itheta(theta2)
        crop1 = crop[itheta1, :, :, :3]
        crop2 = crop[itheta2, :, :, :3]
        barrier = np.ones_like(crop1)
        barrier = barrier[:4, :, :] # white barrier of 4 pixels
        stacked = np.concatenate((crop1, barrier, crop2), axis=0)
        plt.imshow(stacked)
        plt.title(f'{theta1}, {theta2}', fontsize=15)
        plt.suptitle(f'pick: {p}', fontsize=15)
        plt.tight_layout()
        plt.show()
        #plt.savefig('viz.png')

    def visualize_transport(self, p, in_img, input_data, crop, kernel):
        """Like the attention map, let's visualize the transport data from a
        trained model.

        https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html
        In my normal usage, the attention is already softmax-ed but just be
        aware in case it's not. Also be aware of RGB vs BGR mode. We should
        ensure we're in BGR mode before saving. Also with RAINBOW mode,
        red=hottest (highest attention values), green=medium, blue=lowest.

        See also:
        https://matplotlib.org/3.3.0/api/_as_gen/matplotlib.pyplot.subplot.html

        crop.shape: (24,64,64,6)
        kernel.shape = (65,65,3,24)
        """
        def colorize(img):
            # I don't think we have to convert to BGR here...
            img = img - np.min(img)
            img = 255 * img / np.max(img)
            img = cv2.applyColorMap(np.uint8(img), cv2.COLORMAP_RAINBOW)
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        kernel = (tf.transpose(kernel, [3, 0, 1, 2])).numpy()

        # Top two rows: crops from processed RGBD. Bottom two: output from FCN.
        nrows = 4
        ncols = 12
        assert self.num_rotations == nrows * (ncols / 2)
        idx = 0
        fig, ax = plt.subplots(nrows, ncols, figsize=(12,6))
        for _ in range(nrows):
            for _ in range(ncols):
                plt.subplot(nrows, ncols, idx+1)
                plt.axis('off')  # Ah, you need to put this here ...
                if idx < self.num_rotations:
                    plt.imshow(crop[idx, :, :, :3])
                else:
                    # Offset because idx goes from 0 to (rotations * 2) - 1.
                    _idx = idx - self.num_rotations
                    processed = colorize(img=kernel[_idx, :, :, :])
                    plt.imshow(processed)
                idx += 1
        plt.tight_layout()
        plt.show()

    def visualize_logits(self, logits, name):
        """Given logits (BEFORE tf.nn.convolution), get a heatmap.

        Here we apply a softmax to make it more human-readable. However, the
        tf.nn.convolution with the learned kernels happens without a softmax
        on the logits. [Update: wait, then why should we have a softmax,
        then? I forgot why we did this ...]
        """
        original_shape = logits.shape
        logits = tf.reshape(logits, (1, np.prod(original_shape)))
        # logits = tf.nn.softmax(logits)  # Is this necessary?
        vis_transport = np.float32(logits).reshape(original_shape)
        vis_transport = vis_transport[0]
        vis_transport = vis_transport - np.min(vis_transport)
        vis_transport = 255 * vis_transport / np.max(vis_transport)
        vis_transport = cv2.applyColorMap(np.uint8(vis_transport), cv2.COLORMAP_RAINBOW)

        # Only if we're saving with cv2.imwrite()
        vis_transport = cv2.cvtColor(vis_transport, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'tmp/logits_{name}.png', vis_transport)

        plt.subplot(1, 1, 1)
        plt.title(f'Logits: {name}', fontsize=15)
        plt.imshow(vis_transport)
        plt.tight_layout()
        plt.show()

    def get_transport_heatmap(self, transport):
        """Given transport output, get a human-readable heatmap.

        https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html  
        In my normal usage, the attention is already softmax-ed but just be
        aware in case it's not. Also be aware of RGB vs BGR mode. We should
        ensure we're in BGR mode before saving. Also with RAINBOW mode, red =
        hottest (highest attention values), green=medium, blue=lowest.
        """
        # Options: cv2.COLORMAP_PLASMA, cv2.COLORMAP_JET, etc.
        #transport = tf.reshape(transport, (1, np.prod(transport.shape)))
        #transport = tf.nn.softmax(transport)
        assert transport.shape == (320, 160, self.num_rotations), transport.shape
        vis_images = []
        for idx in range(self.num_rotations):
            t_img = transport[:, :, idx]
            vis_transport = np.float32(t_img)
            vis_transport = vis_transport - np.min(vis_transport)
            vis_transport = 255 * vis_transport / np.max(vis_transport)
            vis_transport = cv2.applyColorMap(np.uint8(vis_transport), cv2.COLORMAP_RAINBOW)
            vis_transport = cv2.cvtColor(vis_transport, cv2.COLOR_RGB2BGR)
            vis_images.append(vis_transport)
        return vis_images