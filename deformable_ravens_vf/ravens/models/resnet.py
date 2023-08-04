#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
# from packaging import version
# import tensorflow as tf
import torch
import torch.nn as nn


# Check TensorFlow version
# print("Detected TensorFlow version: ", tf.__version__)
print("Detected PyTorch version: ", torch.__version__)
# assert version.parse(tf.__version__).release[0] >= 2, 'This code requires TensorFlow 2.0 or above.'

# # Set eager execution
# if not tf.executing_eagerly():
#     tf.compat.v1.enable_eager_execution()


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, include_batchnorm=False):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = filters

        self.conv1 = nn.Conv2d(in_channels, filters1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(filters1) if include_batchnorm else nn.Identity()

        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(filters2) if include_batchnorm else nn.Identity()

        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(filters3) if include_batchnorm else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = F.relu(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, stride=2, include_batchnorm=False):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters

        self.conv1 = nn.Conv2d(in_channels, filters1, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(filters1) if include_batchnorm else nn.Identity()

        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(filters2) if include_batchnorm else nn.Identity()

        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(filters3) if include_batchnorm else nn.Identity()

        self.shortcut = nn.Conv2d(in_channels, filters3, kernel_size=1, stride=stride)
        self.bn_shortcut = nn.BatchNorm2d(filters3) if include_batchnorm else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        identity = self.bn_shortcut(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = F.relu(out)

        return out



class ResNet43_8s(nn.Module):
    def __init__(self, input_channels, output_dim, include_batchnorm=False, cutoff_early=False):
        super(ResNet43_8s, self).__init__()

        self.cutoff_early = cutoff_early

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64) if include_batchnorm else nn.Identity()

        self.stage2a = ConvBlock(64, [64, 64, 64], kernel_size=3, stride=1, include_batchnorm=include_batchnorm)
        self.stage2b = IdentityBlock(64, [64, 64, 64], kernel_size=3, include_batchnorm=include_batchnorm)

        self.stage3a = ConvBlock(64, [128, 128, 128], kernel_size=3, stride=2, include_batchnorm=include_batchnorm)
        self.stage3b = IdentityBlock(128, [128, 128, 128], kernel_size=3, include_batchnorm=include_batchnorm)

        self.stage4a = ConvBlock(128, [256, 256, 256], kernel_size=3, stride=2, include_batchnorm=include_batchnorm)
        self.stage4b = IdentityBlock(256, [256, 256, 256], kernel_size=3, include_batchnorm=include_batchnorm)

        self.stage5a = ConvBlock(256, [512, 512, 512], kernel_size=3, stride=2, include_batchnorm=include_batchnorm)
        self.stage5b = IdentityBlock(512, [512, 512, 512], kernel_size=3, include_batchnorm=include_batchnorm)

        self.stage6a = ConvBlock(512, [256, 256, 256], kernel_size=3, stride=1, include_batchnorm=include_batchnorm)
        self.stage6b = IdentityBlock(256, [256, 256, 256], kernel_size=3, include_batchnorm=include_batchnorm)

        self.stage7a = ConvBlock(256, [128, 128, 128], kernel_size=3, stride=1, include_batchnorm=include_batchnorm)
        self.stage7b = IdentityBlock(128, [128, 128, 128], kernel_size=3, include_batchnorm=include_batchnorm)

        self.stage8a = ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, include_batchnorm=include_batchnorm)
        self.stage8b = IdentityBlock(64, [64, 64, 64], kernel_size=3, include_batchnorm=include_batchnorm)

        self.stage9a = ConvBlock(64, [16, 16, output_dim], kernel_size=3, stride=1, include_batchnorm=include_batchnorm, activation=False)
        self.stage9b = IdentityBlock(output_dim, [16, 16, output_dim], kernel_size=3, include_batchnorm=include_batchnorm, activation=False)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.stage2a(x)
        x = self.stage2b(x)

        if self.cutoff_early:
            return x

        x = self.stage3a(x)
        x = self.stage3b(x)

        x = self.stage4a(x)
        x = self.stage4b(x)

        x = self.stage5a(x)
        x = self.stage5b(x)

        x = self.upsample(x)

        x = self.stage6a(x)
        x = self.stage6b(x)

        x = self.upsample(x)

        x = self.stage7a(x)
        x = self.stage7b(x)

        x = self.upsample(x)

        x = self.stage8a(x)
        x = self.stage8b(x)

        x = self.upsample(x)

        x = self.stage9a(x)
        x = self.stage9b(x)

        return x



class ResNet36_4s(nn.Module):
    def __init__(self, input_channels, output_dim, include_batchnorm=False, cutoff_early=False):
        super(ResNet36_4s, self).__init__()

        self.cutoff_early = cutoff_early

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64) if include_batchnorm else nn.Identity()

        self.conv_block1 = ConvBlock(64, [64, 64, output_dim if cutoff_early else 64], kernel_size=5 if cutoff_early else 3, stride=1, include_batchnorm=include_batchnorm)
        self.identity_block1 = IdentityBlock(64, [64, 64, output_dim if cutoff_early else 64], kernel_size=5 if cutoff_early else 3, include_batchnorm=include_batchnorm)

        if not cutoff_early:
            self.conv_block2 = ConvBlock(64, [64, 64, 64], kernel_size=3, stride=2, include_batchnorm=include_batchnorm)
            self.identity_block2 = IdentityBlock(64, [64, 64, 64], kernel_size=3, include_batchnorm=include_batchnorm)

            self.conv_block3 = ConvBlock(64, [64, 64, 64], kernel_size=3, stride=2, include_batchnorm=include_batchnorm)
            self.identity_block3 = IdentityBlock(64, [64, 64, 64], kernel_size=3, include_batchnorm=include_batchnorm)

            self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            self.conv_block4 = ConvBlock(64, [64, 64, 64], kernel_size=3, stride=1, include_batchnorm=include_batchnorm)
            self.identity_block4 = IdentityBlock(64, [64, 64, 64], kernel_size=3, include_batchnorm=include_batchnorm)

            self.upsample_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            self.conv_block5 = ConvBlock(64, [16, 16, output_dim], kernel_size=3, stride=1, include_batchnorm=include_batchnorm)
            # self.identity_block5 = IdentityBlock(64, [16, 16, output_dim], kernel_size=3, include_batchnorm=include_batchnorm)
            self.identity_block5 = IdentityBlock(output_dim, [16, 16, output_dim], kernel_size=3, include_batchnorm=include_batchnorm)
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv_block1(x)
        x = self.identity_block1(x)

        if self.cutoff_early:
            return x

        x = self.conv_block2(x)
        x = self.identity_block2(x)

        x = self.conv_block3(x)
        x = self.identity_block3(x)

        x = self.upsample_2(x)

        x = self.conv_block4(x)
        x = self.identity_block4(x)

        x = self.upsample_3(x)

        x = self.conv_block5(x)
        x = self.identity_block5(x)

        return x
