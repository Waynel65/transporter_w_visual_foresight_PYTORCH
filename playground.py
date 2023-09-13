import torch
import tensorflow as tf
import pdb

pdb.set_trace()
# PyTorch
goal_x_in_logits_torch = torch.randn(1, 3, 224, 224)
kernel_torch = torch.randn(36, 3, 65, 65)

# TensorFlow
goal_x_in_logits_tf = tf.convert_to_tensor(goal_x_in_logits_torch.permute(0, 2, 3, 1).numpy()) # changing to NHWC format
kernel_tf = tf.convert_to_tensor(kernel_torch.permute(0,2,3,1).numpy()) # to match TF's order

# output_torch = torch.nn.functional.conv2d(goal_x_in_logits_torch, kernel_torch)
output = torch.nn.functional.conv2d(goal_x_in_logits, kernel, groups=goal_x_in_logits.shape[1])

kernel_tf_transposed = tf.transpose(kernel_tf, [1, 2, 3, 0])
output_tf = tf.nn.convolution(goal_x_in_logits_tf, kernel_tf_transposed, data_format="NHWC")

output_tf_torch = torch.tensor(output_tf.numpy().transpose(0, 3, 1, 2)) # Convert TF tensor to PyTorch format (NCHW)
diff = torch.nn.functional.mse_loss(output_torch, output_tf_torch)
print(diff.item())
