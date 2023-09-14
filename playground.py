import torch
import tensorflow as tf
import pdb

pdb.set_trace()
# # PyTorch
# goal_x_in_logits_torch = torch.ones(1, 3, 224, 224)
# kernel_torch = torch.randn(36,3,65, 65)

# # TensorFlow
# goal_x_in_logits_tf = tf.convert_to_tensor(goal_x_in_logits_torch.permute(0, 2, 3, 1).numpy()) # changing to NHWC format
# kernel_tf = tf.convert_to_tensor(kernel_torch.permute(0,2,3,1).numpy()) # to match TF's order

# output_torch = torch.nn.functional.conv2d(goal_x_in_logits_torch, kernel_torch)

# kernel_tf_transposed = tf.transpose(kernel_tf, [1, 2, 3, 0]) # 
# output_tf = tf.nn.convolution(goal_x_in_logits_tf, kernel_tf_transposed, data_format="NHWC")

# output_tf_torch = torch.tensor(output_tf.numpy().transpose(0, 3, 1, 2)) # Convert TF tensor to PyTorch format (NCHW)
# diff = torch.nn.functional.mse_loss(output_torch, output_tf_torch)
# print(diff.item())

# pytorch output
# save
# torch.save(in_logits, 'in_logits.pth')
# torch.save(kernel_nocrop_logits, 'kernel_nocrop_logits.pth')
# torch.save(goal_logits, 'goal_logits.pth')

# now load 
in_logits = torch.load('in_logits.pth')
kernel_nocrop_logits = torch.load('kernel_nocrop_logits.pth')
goal_logits = torch.load('goal_logits.pth')
# convert to numpy
in_logits_np = in_logits.detach().cpu().numpy()
kernel_nocrop_logits_np = kernel_nocrop_logits.detach().cpu().numpy()
goal_logits_np = goal_logits.detach().cpu().numpy()

# tensorflow output
# save
# np.save('in_logits_tf.npy', in_logits_np) # Save to disk
# np.save('kernel_nocrop_logits_tf.npy', kernel_nocrop_logits_np)
# np.save('goal_logits_tf.npy', goal_logits_np)

# now load
in_logits_tf = np.load('in_logits_tf.npy')
kernel_nocrop_logits_tf = np.load('kernel_nocrop_logits_tf.npy')
goal_logits_tf = np.load('goal_logits_tf.npy')

# compare
diff = np.mean(np.square(in_logits_np - in_logits_tf))
print(f"[DEBUG] diff between in_logits: {diff}")
diff = np.mean(np.square(kernel_nocrop_logits_np - kernel_nocrop_logits_tf))
print(f"[DEBUG] diff between kernel_nocrop_logits: {diff}")
diff = np.mean(np.square(goal_logits_np - goal_logits_tf))
print(f"[DEBUG] diff between goal_logits: {diff}")
