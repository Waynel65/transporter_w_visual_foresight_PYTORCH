"""PP dynamics module."""

import numpy as np
from ravens.models.resnet import ResNet36_4s
from ravens.models.resnet import ResNet43_8s
from ravens.utils import utils
# import tensorflow as tf
import torch
# from tensorflow_addons import image as tfa_image


class PPDynamics(object):
  """PP dynamics module."""
  
  def __init__(self, in_shape, out_channel, mask_size, model_name):
    # torch device definition
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Padding.
    self.mask_size = mask_size
    self.pad_size = int(self.mask_size / 2)
    self.padding = np.zeros((3, 2), dtype=int)
    self.padding[:2, :] = self.pad_size
    in_shape = np.array(in_shape)
    in_shape[0:2] += self.pad_size * 2
    in_shape = tuple(in_shape)
    print(f"[PPDynamics] in_shape after padding: {in_shape}")
    
    # Initialize the model.
    self.model_name = model_name
    print(f"[PPDynamics] Model is {self.model_name}")
    if self.model_name == 'resnet_lite':
      # d_in, d_out = ResNet36_4s(in_shape, out_channel)
      self.model = ResNet36_4s(in_shape, out_channel).to(device) # Wayne: instantiate the model here
    elif self.model_name == 'resnet':
      self.model = ResNet43_8s(in_shape, out_channel).to(device) # Wayne: instantiate the model here
    # self.model = tf.keras.models.Model(inputs=[d_in], outputs=[d_out])
    # self.optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
    self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
    # self.metric = tf.keras.metrics.Mean(name='loss_dynamics')

    # Means and variances for color and depth images.
    self.color_mean = 0.18877631
    self.depth_mean = 0.00509261
    self.color_std = 0.07276466
    self.depth_std = 0.00903967


  def train_pp(self, init_img, target_img, p0, p1, p1_theta, backprop=True, repeat_H_lambda=1, h_only=False):
    """Train the PP dynamics."""
    """The actual training part of the VF model """

    self.optim.zero_grad()

    # Debug
    if False:
      import matplotlib
      matplotlib.use('TkAgg')
      import matplotlib.pyplot as plt
      if not h_only:
        f, ax = plt.subplots(1, 4)
        max_height = 0.14
        normalize = matplotlib.colors.Normalize(vmin=0.0, vmax=max_height)
        init_img[p0[0], p0[1], :] = 0.0
        target_img[p1[0], p1[1], :] = 0.0
        ax[0].imshow(init_img[:, :, :3] / 255.0)
        ax[1].imshow(init_img[:, :, 3], norm=normalize)
        ax[2].imshow(target_img[:, :, :3] / 255.0)
        ax[3].imshow(target_img[:, :, 3], norm=normalize)
        plt.show()
      else:
        f, ax = plt.subplots(2)
        ax[0].imshow(init_img[:, :, 0])
        ax[1].imshow(target_img[:, :, 0])
        plt.show()

    # Subtrack means and divide by std.
    init_img = self.preprocess_input(init_img, h_only)
    target_img = self.preprocess_target(target_img, h_only)        
    
    # Add padding.
    init_img  = np.pad(init_img, self.padding, mode='constant')
    target_img = np.pad(target_img, self.padding, mode='constant')

    # Convert numpy arrays to PyTorch tensors
    init_img = torch.from_numpy(init_img).to(self.device).float()
    target_img = torch.from_numpy(target_img).to(self.device).float()

    # Forward pass.
    out_tens = self.forward_pp(init_img, p0, p1, p1_theta)

    # Get loss.
    diff = torch.abs(target_img - out_tens)
    b, h, w, c = diff.shape

    if h_only:
      loss = torch.mean(diff)
    else:
      loss_R = torch.sum(diff[:, :, :, 0])
      loss_G = torch.sum(diff[:, :, :, 1])
      loss_B = torch.sum(diff[:, :, :, 2])
      loss_H = torch.sum(diff[:, :, :, 3])
      loss = (loss_R + loss_G + loss_B + repeat_H_lambda * loss_H) / (b * h * w * c)

    # Backpropagate
    if backprop:
      loss.backward()
      self.optim.step()
    
    return loss.item()



    def forward_pp(self, init_img, p0, p1, p1_theta):
      """Forward pass."""
      
      # PyTorch uses (C,H,W) convention and TensorFlow uses (H,W,C) so you might need to transpose your data before using it
      init_img = torch.from_numpy(init_img.transpose((2, 0, 1))).to(self.device)

      # Pick mask.
      init_shape = init_img.shape
      pick_mask = torch.zeros((init_shape[1], init_shape[2]), device=self.device)
      pick_mask[p0[0]:(p0[0]+self.mask_size), p0[1]:(p0[1]+self.mask_size)] = 1.0

      # Place mask 
      # a square mask that has the same size as the image and is centered at p_pick
      # but only contain positive values and 0s
      # positive values for the square region centered at p_pick (the area around picking position)
      # 0s for everywhere else
      pivot = torch.tensor([p0[1], p0[0]]).to(self.device) + self.pad_size
      rmat = utils.get_image_transform(p1_theta, (0, 0), pivot)
      rvec = rmat.view(-1)[:-1]
      init_tens_rot = torchvision.transforms.functional.affine(init_img, angle=0, translate=(0, 0), scale=1, shear=0, resample=0, fillcolor=None)
      crop = init_tens_rot[p0[0]:(p0[0] + self.mask_size),
                          p0[1]:(p0[1] + self.mask_size), :]
      place_mask = torch.zeros(init_shape, device=self.device) 
      place_mask[:, p1[0]:(p1[0]+self.mask_size), p1[1]:(p1[1]+self.mask_size)] = crop

      # Concateante init_img, pick_mask, and place_mask.
      # this in_img will be the input into the network
      in_img = torch.cat([init_img, pick_mask.unsqueeze(0), place_mask], dim=0)

      # Debug
      if False:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        print(f"in_img: {in_img.shape}")
        print(f"pick_mask: {pick_mask.shape}")
        print(f"place_mask: {place_mask.shape}")
        
        # # h_only
        # f, ax = plt.subplots(1, 5)
        # ax[0].imshow(in_img[:, :, 0])
        # ax[1].imshow(in_img[:, :, 1])
        # ax[2].imshow(in_img[:, :, 2])
        # ax[3].imshow(in_img[:, :, 0] + in_img[:, :, 1])
        # ax[4].imshow(in_img[:, :, 0] + in_img[:, :, 2])
        
        # full
        f, ax = plt.subplots(4)
        ax[0].imshow(in_img[:, :, 3] + pick_mask)
        ax[1].imshow(in_img[:, :, 3] + place_mask[:, :, 3])
        ax[2].imshow(in_img[:, :, 0])
        ax[3].imshow(in_img[:, :, 0] + place_mask[:, :, 0])
        plt.show()    

      # Forward pass.
      out_tens = self.model(in_img.unsqueeze(0)) # to account for the batch_size

      return out_tens


  def imagine(self, init_img, p0, p1, p1_theta, h_only):
    """Imagine the image after taking the action."""
    
    # Subtract means and divide by std.
    init_img = self.preprocess_input(init_img, h_only)

    # Add padding.
    init_img = np.pad(init_img, self.padding, mode='constant')

    # Forward pass.
    out_tens = self.forward_pp(init_img, p0, p1, p1_theta)

    # Postprocess output tensor.
    out_data = out_tens.numpy()[0][self.pad_size:(-self.pad_size), self.pad_size:(-self.pad_size)]
    out_img = self.postprocess_output(out_data, h_only)

    # Debug paper
    if False:
      import matplotlib
      matplotlib.use('TkAgg')
      import matplotlib.pyplot as plt
      plt.imshow(out_img[:, :, :3]/ 255.0)
      plt.show()

    return out_img

  def test_pp(self, init_img, target_img, p0, p1, p1_theta, repeat_H_lambda, h_only):
    """Test PP Dynamics."""

    # Subtract means and divide by std.
    init_img = self.preprocess_input(init_img, h_only)
    target_img = self.preprocess_input(target_img, h_only)

    # Add padding.
    init_img = np.pad(init_img, self.padding, mode='constant')
    target_img = np.pad(target_img, self.padding, mode='constant')

    # Forward pass.
    out_tens = self.forward_pp(init_img, p0, p1, p1_theta)

    # Get loss.
    target_tens = tf.convert_to_tensor(target_img, dtype=tf.float32)
    diff = tf.abs(target_tens - out_tens)
    b, h, w, c = diff.shape
    if h_only:
      rgb_loss = 0.0
      height_loss = tf.reduce_mean(diff)
    else:
      assert c == 4
      loss_R = tf.reduce_sum(diff[:, :, :, 0])
      loss_G = tf.reduce_sum(diff[:, :, :, 1])
      loss_B = tf.reduce_sum(diff[:, :, :, 2])
      loss_H = tf.reduce_sum(diff[:, :, :, 3])
      rgb_loss = (loss_R + loss_G + loss_B) / (b * h * w * 3)
      height_loss = loss_H / (b * h * w)

    # Postprocess output tensor.
    out_data = out_tens.numpy()[0][self.pad_size:(-self.pad_size), self.pad_size:(-self.pad_size)]
    out_img = self.postprocess_output(out_data, h_only)

    return out_img, np.float32(rgb_loss), np.float32(height_loss)

  def load(self, path):
    """Load model weights."""
    self.model.load_weights(path)
  
  def save(self, filename):
    """Save the model."""
    self.model.save(filename)

  def preprocess_input(self, img, h_only=False):
    """Pre-process input (subtract mean, divide by std)."""

    if h_only:
      assert img.shape[-1] == 1
      img[:, :, 0] = (img[:, :, 0] - self.depth_mean) / self.depth_std
    else:
      # assert img.shape[-1] == 6
      img[:, :, :3] = (img[:, :, :3] / 255 - self.color_mean) / self.color_std
      img[:, :, 3:] = (img[:, :, 3:] - self.depth_mean) / self.depth_std
    return img
  
  def preprocess_target(self, img, h_only=False):
    """Pre-process input (subtract mean, divide by std)."""

    if h_only:
      assert img.shape[-1] == 1
      img[:, :, 0] = (img[:, :, 0] - self.depth_mean) / self.depth_std 
    else:
      assert img.shape[-1] == 4
      img[:, :, :3] = (img[:, :, :3] / 255 - self.color_mean) / self.color_std
      img[:, :, -1] = (img[:, :, -1] - self.depth_mean) / self.depth_std
    return img

  def postprocess_output(self, img, h_only=False):
    """Post-process output (add mean, multiply by std)."""

    if h_only:
      img[:, :, 0] = img[:, :, 0] * self.depth_std + self.depth_mean
    else:
      img[:, :, :3] = 255 * (img[:, :, :3] * self.color_std + self.color_mean)
      img[:, :, -1] = img[:, :, -1] * self.depth_std + self.depth_mean
    return img