import torch
import tensorflow as tf
import tensorflow_addons as tfa

import pdb
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T

import matplotlib.pyplot as plt



# preliminary function for creating tensors
def create_custom_tensor():
    """
        this creates a tensor of shape (1,1,14,14)
    """
    row = torch.arange(1, 15).unsqueeze(0)
    tensor = row.repeat(14, 1).unsqueeze(0).unsqueeze(0)
    return tensor

# helper functions for the functions below
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

    input_tensor = input_tensor.float()
    # Compute rotation using affine transformations
    grid = F.affine_grid(theta, input_tensor.size())
    output = F.grid_sample(input_tensor, grid, mode='nearest')
    return output


def testing_tf(in_logits, kernel_nocrop_logits, goal_logits):
    """
        this is the original forward function in the tensorflow codebase
        Note that here I am leaving out the part where we get output from model
        (basically meaning that we are testing the part after model output)
        !The goal is to recreate the output from this function!
    """

    pdb.set_trace()
    # here we pretend goal_logits, in_logits, kernel_nocrop_logits are output from the model
    goal_x_in_logits     = tf.multiply(goal_logits, in_logits)
    goal_x_kernel_logits = tf.multiply(goal_logits, kernel_nocrop_logits)


    # these are some hardcoded values for testing purposes
    num_rotations = 24
    p = [5,5]
    crop_size = 4
    pad_size = int(crop_size / 2)

    # getting the rotation pivot and the rotation vector
    pivot = np.array([p[1], p[0]]) + pad_size # here p is based on the output of attention
    rvecs = get_se2(num_rotations, pivot)

    # Crop the kernel_logits about the picking point and get rotations.
    crop = tf.identity(goal_x_kernel_logits)                            # (1,384,224,3)
    crop = tf.repeat(crop, repeats=num_rotations, axis=0)          # (24,384,224,3)

    # rotating the crop 24 times according to rvecs
    crop = tfa.image.transform(crop, rvecs, interpolation='NEAREST')    # (24,384,224,3)

    # cropping the height and width dimension for kernel
    kernel = crop[:,
                    p[0]:(p[0] + crop_size),
                    p[1]:(p[1] + crop_size),
                    :]


    # print(f"[TRANS_GOAL] kernel shape: {kernel.shape} | the rest: {(self.num_rotations, self.crop_size, self.crop_size, self.odim)}")

    # Cross-convolve `in_x_goal_logits`. Padding kernel: (24,64,64,3) --> (65,65,3,24).
    kernel_paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
    kernel = tf.pad(kernel, kernel_paddings, mode='CONSTANT')
    kernel = tf.transpose(kernel, [1, 2, 3, 0])
    output = tf.nn.convolution(goal_x_in_logits, kernel, data_format="NHWC")
    output = (1 / (crop_size**2)) * output

    return output

def testing_torch(in_logits, kernel_nocrop_logits, goal_logits):

    # Use features from goal logits and combine with input and kernel.
    goal_x_in_logits     = goal_logits * in_logits
    goal_x_kernel_logits = goal_logits * kernel_nocrop_logits

    # these are some hardcoded values for testing purposes
    num_rotations = 24
    p = [5,5]
    crop_size = 4
    pad_size = int(crop_size / 2)

    # getting the rotation pivot and the rotation vector
    pivot = np.array([p[1], p[0]]) + pad_size # here p is based on the output of attention
    rvecs = get_se2(num_rotations, pivot)

    crop = goal_x_kernel_logits.clone()                                 
    crop = crop.repeat(num_rotations, 1, 1, 1)

    # !THIS IS THE PART WHERE IT STARTS TO GO WRONG
    # apparently this does not recreate the rotation effect in tensorflow
    rotated_crop = torch.empty_like(crop)
    for i in range(num_rotations):
        rvec = rvecs[i]
        angle = np.arctan2(rvec[1], rvec[0]) * 180 / np.pi
        rotated_crop[i] = TF.rotate(crop[i], angle)
    crop = rotated_crop

    # note how we are cropping the kernel differently because of the different convention in dimension
    kernel = crop[:, :,
                p[0]:(p[0] + crop_size),
                p[1]:(p[1] + crop_size)]

    # finally a 2D convolution before output
    kernel = F.pad(kernel, (0, 1, 0, 1)) # this gives (36,3,65,65)
    output = F.conv2d(goal_x_in_logits, kernel) # regular convolution with output shape (1,36,160,160)
    output = (1 / (crop_size**2)) * output # normalization

    # permute back to tensorflow convention before output
    # so that we can compare with the output from tensorflow
    output = output.permute(0, 2, 3, 1)

    return output


# define a main function
if __name__ == "__main__":
    in_logits = create_custom_tensor()
    kernel_nocrop_logits = create_custom_tensor()
    goal_logits = create_custom_tensor()

    # convert to tensorflow tensors with permuted dimensions
    in_logits_tf = tf.convert_to_tensor(in_logits.permute(0,2,3,1).numpy())
    kernel_nocrop_logits_tf = tf.convert_to_tensor(kernel_nocrop_logits.permute(0,2,3,1).numpy())
    goal_logits_tf = tf.convert_to_tensor(goal_logits.permute(0,2,3,1).numpy())

    # run each version's function
    tf_out = testing_tf(in_logits_tf, kernel_nocrop_logits_tf, goal_logits_tf)
    torch_out = testing_torch(in_logits, kernel_nocrop_logits, goal_logits)
    pdb.set_trace()

    print("tensorflow output")
    print(tf_out)

    print("pytorch output")
    print(torch_out)

    print("are they the same?")
    print(np.allclose(torch_out.numpy(), tf_out, atol=1e-6))