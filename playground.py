import torch
import tensorflow as tf
import tensorflow_addons as tfa

import pdb
import numpy as np
# import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T

import matplotlib.pyplot as plt

# from ravens import utils

# pdb.set_trace()
# PyTorch

# assuming this is the output from the resnet model
# randomly generate tensors of shape (1,3,224,224)

pdb.set_trace()
def create_custom_tensor():
    row = torch.arange(1, 15).unsqueeze(0)
    tensor = row.repeat(14, 1).unsqueeze(0).unsqueeze(0)
    return tensor

in_logits = create_custom_tensor()
kernel_nocrop_logits = create_custom_tensor()
goal_logits = create_custom_tensor()

in_logits_tf = tf.convert_to_tensor(in_logits.permute(0, 2, 3, 1).numpy())
kernel_nocrop_logits_tf = tf.convert_to_tensor(kernel_nocrop_logits.permute(0, 2, 3, 1).numpy())
goal_logits_tf = tf.convert_to_tensor(goal_logits.permute(0, 2, 3, 1).numpy())

# if we permute back to tf convention before rotation, would they be the same? => yes

def get_image_transform(theta, trans, pivot=[0, 0]):
    # Get 2D rigid transformation matrix that rotates an image by theta (in
    # radians) around pivot (in pixels) and translates by trans vector (in
    # pixels)
    pivot_T_image = np.array([[1., 0., -pivot[0]],
                              [0., 1., -pivot[1]],
                              [0., 0.,        1.]])
    image_T_pivot = np.array([[1., 0., pivot[0]],
                              [0., 1., pivot[1]],
                              [0., 0.,       1.]])
    transform = np.array([[np.cos(theta), -np.sin(theta), trans[0]],
                          [np.sin(theta), np.cos(theta), trans[1]],
                          [0.,            0.,            1.]])
    return np.dot(image_T_pivot, np.dot(transform, pivot_T_image))

def get_se2(num_rotations, pivot):
    '''
    Get SE2 rotations discretized into num_rotations angles counter-clockwise.
    '''
    rvecs = []
    for i in range(num_rotations):
        theta = i * 2 * np.pi / num_rotations
        rmat = get_image_transform(theta, (0, 0), pivot)
        rvec = rmat.reshape(-1)[:-1]
        rvecs.append(rvec)
    return np.array(rvecs, dtype=np.float32)

def rotate_tensor(input_tensor, rvecs, pivot):
    # Convert rvecs to angles (in radians)
    rvecs_tensor = torch.tensor(rvecs, dtype=torch.float32)
    angles = torch.atan2(rvecs_tensor[:, 1], rvecs_tensor[:, 0])
    
    # Compute affine transformation matrices
    theta = torch.zeros((angles.shape[0], 2, 3), device=angles.device)
    theta[:, 0, 0] = torch.cos(angles)
    theta[:, 0, 1] = -torch.sin(angles)
    theta[:, 1, 0] = torch.sin(angles)
    theta[:, 1, 1] = torch.cos(angles)
    
    # Adjust for the pivot
    pivot = torch.tensor(pivot, device=angles.device).float()
    theta[:, :, 2] = (1 - theta[:, :, 0] - theta[:, :, 1]) * pivot.unsqueeze(0)

    # Compute rotation using affine transformations
    grid = F.affine_grid(theta, input_tensor.size())
    output = F.grid_sample(input_tensor, grid, mode='nearest')
    return output



def testing_torch_no_prepermute(in_logits, kernel_nocrop_logits, goal_logits):


    # Use features from goal logits and combine with input and kernel.
    goal_x_in_logits     = goal_logits * in_logits
    goal_x_kernel_logits = goal_logits * kernel_nocrop_logits


    num_rotations = 24
    p = [5,5]
    crop_size = 4

    pad_size = int(crop_size / 2)
    pivot = np.array([p[1], p[0]]) + pad_size # here p is based on the output of attention
    rvecs = get_se2(num_rotations, pivot)

    crop = goal_x_kernel_logits.clone()                                 
    crop = crop.repeat(num_rotations, 1, 1, 1)

    rotated_crop = rotate_tensor(crop, rvecs, pivot)
    

    rotated_crop = rotated_crop.permute(0, 2, 3, 1)
    return rotated_crop

    # pdb.set_trace()
    # kernel = crop[:, :,
    #             p[0]:(p[0] + crop_size),
    #             p[1]:(p[1] + crop_size)]

    # kernel = kernel.permute(0, 2, 3, 1)


    # # need to permute back to pytorch convention
    # # goal_x_in_logits = goal_x_in_logits.permute(0, 3, 1, 2)
    # # kernel = kernel.permute(0, 3, 1, 2)
    # kernel = F.pad(kernel, (0, 1, 0, 1)) # this gives (36,3,65,65)
    # output = F.conv2d(goal_x_in_logits, kernel) # regular convolution with output shape (1,36,160,160)
    # output = (1 / (crop_size**2)) * output # normalization

    # # permute back to tensorflow convention before output
    # output = output.permute(0, 2, 3, 1)

    # return output

def testing_torch(in_logits, kernel_nocrop_logits, goal_logits):
    # to test if permute back to tf convention before rotation will cause any problems
    pdb.set_trace()
    in_logits = in_logits.permute(0, 2, 3, 1)
    kernel_nocrop_logits = kernel_nocrop_logits.permute(0, 2, 3, 1)
    goal_logits = goal_logits.permute(0, 2, 3, 1)

    # Use features from goal logits and combine with input and kernel.
    goal_x_in_logits     = goal_logits * in_logits
    goal_x_kernel_logits = goal_logits * kernel_nocrop_logits

    num_rotations = 24
    p = [5,5]
    crop_size = 4

    pad_size = int(crop_size / 2)
    pivot = np.array([p[1], p[0]]) + pad_size # here p is based on the output of attention
    rvecs = get_se2(num_rotations, pivot)

    crop = goal_x_kernel_logits.clone()                                 
    crop = crop.repeat(num_rotations, 1, 1, 1)


    rotated_crop = torch.empty_like(crop)
    for i in range(num_rotations):
        rvec = rvecs[i]
        angle = np.arctan2(rvec[1], rvec[0]) * 180 / np.pi
        rotated_crop[i] = TF.rotate(crop[i], angle)
    crop = rotated_crop
    # pdb.set_trace()

    return crop

    # kernel = crop[:,
    #         p[0]:(p[0] + crop_size),
    #         p[1]:(p[1] + crop_size),
    #         :]

    # # need to permute back to pytorch convention
    # goal_x_in_logits = goal_x_in_logits.permute(0, 3, 1, 2)
    # kernel = kernel.permute(0, 3, 1, 2)
    # kernel = F.pad(kernel, (0, 1, 0, 1)) # this gives (36,3,65,65)
    # output = F.conv2d(goal_x_in_logits, kernel) # regular convolution with output shape (1,36,160,160)
    # output = (1 / (crop_size**2)) * output # normalization

    # # permute back to tensorflow convention before output
    # output = output.permute(0, 2, 3, 1)

    # return output


def testing_tf(in_logits, kernel_nocrop_logits, goal_logits):
    # Use features from goal logits and combine with input and kernel.
    pdb.set_trace()
    goal_x_in_logits     = tf.multiply(goal_logits, in_logits)
    goal_x_kernel_logits = tf.multiply(goal_logits, kernel_nocrop_logits)


    num_rotations = 24
    p = [5,5]
    crop_size = 4

    pad_size = int(crop_size / 2)
    pivot = np.array([p[1], p[0]]) + pad_size # here p is based on the output of attention
    rvecs = get_se2(num_rotations, pivot)

    # Crop the kernel_logits about the picking point and get rotations.
    crop = tf.identity(goal_x_kernel_logits)                            # (1,384,224,3)
    crop = tf.repeat(crop, repeats=num_rotations, axis=0)          # (24,384,224,3)

    crop = tfa.image.transform(crop, rvecs, interpolation='NEAREST')    # (24,384,224,3)
    return crop

    # pdb.set_trace()

    # kernel = crop[:,
    #                 p[0]:(p[0] + crop_size),
    #                 p[1]:(p[1] + crop_size),
    #                 :]
    # # this becomes (24,64,64,3)

    # # return kernel 

    # # print(f"[TRANS_GOAL] kernel shape: {kernel.shape} | the rest: {(self.num_rotations, self.crop_size, self.crop_size, self.odim)}")

    # # Cross-convolve `in_x_goal_logits`. Padding kernel: (24,64,64,3) --> (65,65,3,24).
    # kernel_paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
    # kernel = tf.pad(kernel, kernel_paddings, mode='CONSTANT')
    # kernel = tf.transpose(kernel, [1, 2, 3, 0])
    # output = tf.nn.convolution(goal_x_in_logits, kernel, data_format="NHWC")
    # output = (1 / (crop_size**2)) * output

    # return output



# torch_out = testing_torch(in_logits, kernel_nocrop_logits, goal_logits)
torch_no_permute = testing_torch_no_prepermute(in_logits, kernel_nocrop_logits, goal_logits)
tf_out = testing_tf(in_logits_tf, kernel_nocrop_logits_tf, goal_logits_tf)
pdb.set_trace()
# compare if they are the same in terms of value positions and values
# print("pytorch output")
# print(torch_out)

print("tensorflow output")
print(tf_out)

print("pytorch output")
print(torch_no_permute)

# print(np.allclose(torch_1.numpy(), tf_1, atol=1e-6))
# print(np.allclose(torch_2.numpy(), tf_2, atol=1e-6))
print(np.allclose(torch_no_permute.numpy(), tf_out, atol=1e-6))

# plt.imshow(torch_no_permute[0,0,:,:].numpy())
# plt.show()
# plt.imshow(tf_out[0,:,:,0].numpy())
# plt.show()




# # TensorFlow
# goal_x_in_logits_tf = tf.convert_to_tensor(goal_x_in_logits_torch.permute(0, 2, 3, 1).numpy()) # changing to NHWC format
# kernel_tf = tf.convert_to_tensor(kernel_torch.permute(0,2,3,1).numpy()) # to match TF's order

# output_torch = torch.nn.functional.conv2d(goal_x_in_logits_torch, kernel_torch)

# kernel_tf_transposed = tf.transpose(kernel_tf, [1, 2, 3, 0]) # 
# output_tf = tf.nn.convolution(goal_x_in_logits_tf, kernel_tf_transposed, data_format="NHWC")

# output_tf_torch = torch.tensor(output_tf.numpy().transpose(0, 3, 1, 2)) # Convert TF tensor to PyTorch format (NCHW)
# diff = torch.nn.functional.mse_loss(output_torch, output_tf_torch)
# print(diff.item())
