import warnings
warnings.filterwarnings("ignore")
from torch.utils import data
from torch import nn, optim
from vis_tools import *
from datasets import *
from models import *
import argparse, os
import itertools
import torch
import pdb
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--loss_curves', action='store_true',
                    help='save loss curves')
parser.add_argument('--random', action="store_true",
                    help='random latent as input')
parser.add_argument('--fid', action="store_true",
                    help='encoded latent as input')
parser.add_argument('--lpips', action='store_true',
                    help='generate images LPIPS score')
opt = parser.parse_args()

checkpoint = torch.load('./checkpoints/bicyclegan_19.pth')

# Plot training losses
if opt.__dict__['loss_curves']:
  loss_l1_image_list = checkpoint['loss_l1_image_list']
  loss_l1_z_list = checkpoint['loss_l1_z_list']
  loss_gan_list = checkpoint['loss_gan_list']
  loss_gan_vae_list = checkpoint['loss_gan_vae_list']
  loss_kl_list = checkpoint['loss_kl_list']
  total_loss_list = checkpoint['total_loss_list']

  plt.plot(loss_l1_image_list)
  plt.title('l1_image_loss')
  plt.savefig('l1_image_loss.png')
  plt.close()

  plt.plot(loss_l1_z_list)
  plt.title('l1_z_loss')
  plt.savefig('l1_z_loss.png')
  plt.close()

  plt.plot(loss_gan_list)
  plt.title('gan_loss')
  plt.savefig('gan_loss.png')
  plt.close()

  plt.plot(loss_gan_vae_list)
  plt.title('gan_vae_loss')
  plt.savefig('gan_vae_loss.png')
  plt.close()

  plt.plot(loss_kl_list)
  plt.title('kl_loss')
  plt.savefig('kl_loss.png')
  plt.close()

  plt.plot(total_loss_list)
  plt.title('total_loss')
  plt.savefig('total_loss.png')
  plt.close()

# Training Configurations 
# (You may put your needed configuration here. Please feel free to add more or use argparse. )
img_dir = './data/edges2shoes/val/'
img_shape = (3, 128, 128) # Please use this image dimension faster training purpose
num_epochs = 20
batch_size = 16
lr_rate = 2e-4  	      # Adam optimizer learning rate
betas = (0.5, 0.999)			  # Adam optimizer beta 1, beta 2
lambda_pixel = 10       # Loss weights for pixel loss
lambda_latent = 0.5      # Loss weights for latent regression 
lambda_kl = 0.01          # Loss weights for kl divergence
latent_dim = 8         # latent dimension for the encoded images from domain B
gpu_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Normalize image tensor
def norm(image):
  return (image/255.0-0.5)*2.0

# Denormalize image tensor
def denorm(tensor):
  return ((tensor+1.0)/2.0)*255.0

# Reparameterization helper function 
# (You may need this helper function here or inside models.py, depending on your encoder implementation)
def reparameterization(mu, logvar):    
  std = torch.exp(logvar/2)
  eps = torch.randn_like(std)
  z = mu + std*eps
  return z

# torch.manual_seed(1); np.random.seed(1)

# Define DataLoader
dataset = Edge2Shoe(img_dir)
print('len(dataset): {}'.format(len(dataset)))
loader = data.DataLoader(dataset, batch_size=batch_size)
print('len(loader): {}'.format(len(loader)))
# Loss functions
mae_loss = torch.nn.L1Loss().to(gpu_id)
mse_loss = torch.nn.MSELoss().to(gpu_id)


# Define generator, encoder and discriminators
generator = Generator(latent_dim, img_shape).to(gpu_id)
encoder = Encoder(latent_dim).to(gpu_id)
encoder.load_state_dict(checkpoint['encoder'])
generator.load_state_dict(checkpoint['generator'])
encoder.eval()
generator.eval()
counter = 0
for idx, data in tqdm(enumerate(loader)):
  ########## Process Inputs ##########
  edge_tensor, rgb_tensor = data
  edge_tensor, rgb_tensor = norm(edge_tensor).to(gpu_id), norm(rgb_tensor).to(gpu_id)
  real_A = edge_tensor
  real_B = rgb_tensor

  if opt.__dict__['random']:
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    vis_real_A = denorm(real_A[0].detach()).cpu().data.numpy().astype(np.uint8)
    vis_real_B = denorm(real_B[0].detach()).cpu().data.numpy().astype(np.uint8)
    axs[0].imshow(vis_real_A.transpose(1, 2, 0))
    axs[0].set_title('edge')
    axs[1].imshow(vis_real_B.transpose(1, 2, 0))
    axs[1].set_title('rgb')

    rand_z = torch.randn(3, real_A.shape[0], latent_dim).to(gpu_id)
    for i in range(3):
      fake_B_cLR = generator(real_A, rand_z[i])
      # -------------------------------
      #  Visualization
      # ------------------------------
      vis_fake_B_cLR = denorm(fake_B_cLR[0].detach()).cpu().data.numpy().astype(np.uint8)
      axs[i + 2].imshow(vis_fake_B_cLR.transpose(1, 2, 0))
      axs[i + 2].set_title('fake_'+str(i+1))

    plt.savefig('./imgs_infer_'+str(idx)+'.png')
    plt.close()

  elif opt.__dict__['fid']:
    mu, logvar = encoder(real_B)
    z = reparameterization(mu, logvar)
    fake_B_cVAE = generator(real_A, z)
    for i in range(batch_size):
      counter += 1
      vis_fake_B_cVAE = denorm(fake_B_cVAE[i].detach()).cpu().data.numpy().astype(np.uint8).transpose(1, 2, 0)
      plt.imsave('./imgs_fid/gen/fake_B_' + str(idx) + '_' + str(i) + '.png', vis_fake_B_cVAE)
      plt.imsave('./imgs_fid/real/real_B_' + str(idx) + '_' + str(i) + '.png', denorm(real_B[i].detach()).cpu().data.numpy().astype(np.uint8).transpose(1, 2, 0))
      if counter == 200:
        break
  
  elif opt.__dict__['lpips']:
    rand_z = torch.randn(batch_size, real_A.shape[0], latent_dim).to(gpu_id)
    for i in range(batch_size):
        fake_B_cLR = generator.forward(real_A, rand_z[i])
        # -------------------------------
        #  Visualization and Save
        # ------------------------------
        vis_fake_B_cLR = denorm(fake_B_cLR[0].detach()).cpu().data.numpy().astype(np.uint8)
        plt.imsave('./imgs_lpips/' + str(counter) + '.png', vis_fake_B_cLR.transpose(1, 2, 0))
        counter += 1

