import torch
import tensorflow as tf
import tensorflow_addons as tfa

import pdb
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T

from ravens import utils

pdb.set_trace()
# PyTorch

# assuming this is the output from the resnet model
# randomly generate tensors of shape (1,3,224,224)

in_logits = torch.rand(1,3,10,10)
kernel_nocrop_logits = torch.rand(1,3,10,10)
goal_logits = torch.rand(1,3,10,10)

in_logits_tf = tf.convert_to_tensor(in_logits.permute(0, 2, 3, 1).numpy())
kernel_nocrop_logits_tf = tf.convert_to_tensor(kernel_nocrop_logits.permute(0, 2, 3, 1).numpy())
goal_logits_tf = tf.convert_to_tensor(goal_logits.permute(0, 2, 3, 1).numpy())

def get_se2(num_rotations, pivot):
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

def testing_torch(in_logits, kernel_nocrop_logits, goal_logits):
    # to test if permute back to tf convention before rotation will cause any problems
    in_logits = in_logits.permute(0, 2, 3, 1)
    kernel_nocrop_logits = kernel_nocrop_logits.permute(0, 2, 3, 1)
    goal_logits = goal_logits.permute(0, 2, 3, 1)

    # Use features from goal logits and combine with input and kernel.
    goal_x_in_logits     = goal_logits * in_logits
    goal_x_kernel_logits = goal_logits * kernel_nocrop_logits

    num_rotations = 24
    p = [130,33]

    pad_size = int(64 / 2)
    pivot = np.array([p[1], p[0]]) + pad_size # here p is based on the output of attention
    rvecs = get_se2(num_rotations, pivot)

    crop = goal_x_kernel_logits.clone()                                 
    crop = crop.repeat(num_rotations, 1, 1, 1)

    rotated_crop = torch.empty_like(crop)
    for i in range(num_rotations):
        rvec = rvecs[i]
        angle = np.arctan2(rvec[1], rvec[0]) * 180 / np.pi
        rotated_crop[i] = T.functional.rotate(crop[i], angle, interpolation=T.InterpolationMode.NEAREST)
    crop = rotated_crop

    kernel = crop[:,
            p[0]:(p[0] + self.crop_size),
            p[1]:(p[1] + self.crop_size),
            :]

    # need to permute back to pytorch convention
    goal_x_in_logits = goal_x_in_logits.permute(0, 3, 1, 2)
    kernel = kernel.permute(0, 3, 1, 2)
    kernel = F.pad(kernel, (0, 1, 0, 1)) # this gives (36,3,65,65)
    output = F.conv2d(goal_x_in_logits, kernel) # regular convolution with output shape (1,36,160,160)
    output = (1 / (self.crop_size**2)) * output # normalization

    # permute back to tensorflow convention before output
    output = output.permute(0, 2, 3, 1)

    return output


def testing_tf(in_logits, kernel_nocrop_logits, goal_logits):
    # Use features from goal logits and combine with input and kernel.
    goal_x_in_logits     = tf.multiply(goal_logits, in_logits)
    goal_x_kernel_logits = tf.multiply(goal_logits, kernel_nocrop_logits)


    num_rotations = 24
    
    p = [130,33]
    pad_size = int(64 / 2)
    pivot = np.array([p[1], p[0]]) + pad_size # here p is based on the output of attention
    rvecs = get_se2(num_rotations, pivot)

    # Crop the kernel_logits about the picking point and get rotations.
    crop = tf.identity(goal_x_kernel_logits)                            # (1,384,224,3)
    crop = tf.repeat(crop, repeats=num_rotations, axis=0)          # (24,384,224,3)
    crop = tfa.image.transform(crop, rvecs, interpolation='NEAREST')    # (24,384,224,3)
    kernel = crop[:,
                    p[0]:(p[0] + self.crop_size),
                    p[1]:(p[1] + self.crop_size),
                    :]
    print(f"[TRANS_GOAL] kernel shape: {kernel.shape} | the rest: {(self.num_rotations, self.crop_size, self.crop_size, self.odim)}")

    # Cross-convolve `in_x_goal_logits`. Padding kernel: (24,64,64,3) --> (65,65,3,24).
    kernel_paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
    kernel = tf.pad(kernel, kernel_paddings, mode='CONSTANT')
    kernel = tf.transpose(kernel, [1, 2, 3, 0])
    output = tf.nn.convolution(goal_x_in_logits, kernel, data_format="NHWC")
    output = (1 / (self.crop_size**2)) * output



torch_out = testing_torch(in_logits, kernel_nocrop_logits, goal_logits)
tf_out = testing_tf(in_logits_tf, kernel_nocrop_logits_tf, goal_logits_tf)

# compare if they are the same in terms of value positions and values
print(torch_out)
print(tf_out)



# # TensorFlow
# goal_x_in_logits_tf = tf.convert_to_tensor(goal_x_in_logits_torch.permute(0, 2, 3, 1).numpy()) # changing to NHWC format
# kernel_tf = tf.convert_to_tensor(kernel_torch.permute(0,2,3,1).numpy()) # to match TF's order

# output_torch = torch.nn.functional.conv2d(goal_x_in_logits_torch, kernel_torch)

# kernel_tf_transposed = tf.transpose(kernel_tf, [1, 2, 3, 0]) # 
# output_tf = tf.nn.convolution(goal_x_in_logits_tf, kernel_tf_transposed, data_format="NHWC")

# output_tf_torch = torch.tensor(output_tf.numpy().transpose(0, 3, 1, 2)) # Convert TF tensor to PyTorch format (NCHW)
# diff = torch.nn.functional.mse_loss(output_torch, output_tf_torch)
# print(diff.item())
