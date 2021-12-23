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
import time
import pdb

# Training Configurations 
# (You may put your needed configuration here. Please feel free to add more or use argparse. )
img_dir = './data/edges2shoes/train/'
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

#save models
os.makedirs("./ckpts", exist_ok = True)

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

def zero_grad_all():
    optimizer_E.zero_grad()
    optimizer_G.zero_grad()
    optimizer_D_VAE.zero_grad()
    optimizer_D_LR.zero_grad()

# Random seeds (optional)
torch.manual_seed(17); np.random.seed(17)

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
D_VAE = Discriminator().to(gpu_id)
D_LR = Discriminator().to(gpu_id)

# Define optimizers for networks
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=lr_rate, betas=betas)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=betas)
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=lr_rate, betas=betas)
optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=lr_rate, betas=betas)

# For adversarial loss (optional to use)
valid = 1; fake = 0
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
torch.autograd.set_detect_anomaly(True)

# loss recording
loss_l1_image_list = []
loss_l1_z_list = []
loss_kl_list = []
loss_gan_list = []
loss_gan_vae_list = []
total_loss_list = []

epoch_start = 0
resume = False
if resume:
  checkpoint = torch.load('./ckpts/bicyclegan_13.pth')
  epoch_start = checkpoint['epoch'] + 1
  encoder.load_state_dict(checkpoint['encoder'])
  generator.load_state_dict(checkpoint['generator'])
  D_VAE.load_state_dict(checkpoint['D_VAE'])
  D_LR.load_state_dict(checkpoint['D_LR'])
  optimizer_G.load_state_dict(checkpoint['optimizer_G'])
  optimizer_E.load_state_dict(checkpoint['optimizer_E'])
  optimizer_D_VAE.load_state_dict(checkpoint['optimizer_D_VAE'])
  optimizer_D_LR.load_state_dict(checkpoint['optimizer_D_LR'])
  loss_l1_image_list = checkpoint['loss_l1_image_list']
  loss_l1_z_list = checkpoint['loss_l1_z_list']
  loss_gan_list = checkpoint['loss_gan_list']
  loss_gan_vae_list = checkpoint['loss_gan_vae_list']
  loss_kl_list = checkpoint['loss_kl_list']
  total_loss_list = checkpoint['total_loss_list']
  
# Training
for e in range(epoch_start, num_epochs):
  encoder.train()
  generator.train()
  D_LR.train()
  D_VAE.train()
  running_loss_l1_image = 0
  running_loss_l1_z = 0
  running_loss_gan = 0
  running_loss_gan_vae = 0
  running_loss_kl = 0
  running_total_loss = 0
  for idx, data in enumerate(loader):

    ########## Process Inputs ##########
    edge_tensor, rgb_tensor = data
    edge_tensor, rgb_tensor = norm(edge_tensor).to(gpu_id), norm(rgb_tensor).to(gpu_id)
    real_A = edge_tensor
    real_B = rgb_tensor

    mu, logvar = encoder(real_B)
    z = reparameterization(mu, logvar)
    fake_B_cVAE = generator(real_A, z)
    rand_z = torch.randn(real_A.shape[0], latent_dim).to(gpu_id)
    fake_B_cLR = generator(real_A, rand_z)
    mu_, logvar_ = encoder(fake_B_cLR)
    #-------------------------------
    #  Train Generator and Encoder
    #------------------------------
    for param in D_VAE.parameters():
      param.requires_grad = False
    for param in D_LR.parameters():
      param.requires_grad = False
    optimizer_E.zero_grad()
    optimizer_G.zero_grad()

    loss_G = 0.0
    # cVAE-GAN
    # mu, logvar = encoder(real_B)
    # z = reparameterization(mu, logvar)
    # fake_B_cVAE = generator(real_A, z)
    loss_G_cVAE = loss_discriminator_one(fake_B_cVAE, D_VAE, mse_loss, Tensor)
    # cLR-GAN
    # rand_z = torch.randn(real_A.shape[0], latent_dim).to(gpu_id)
    # fake_B_cLR = generator(real_A, rand_z)
    loss_G_cLR = loss_discriminator_one(fake_B_cLR, D_LR, mse_loss, Tensor)
    # KLD
    loss_kl = lambda_kl * loss_kld(mu, logvar)
    # recon loss, cVAE-GAN
    loss_l1_image = lambda_pixel * loss_generator(generator, real_A, z, real_B, mae_loss)
    # sum
    loss_G = loss_G_cVAE + loss_G_cLR + loss_kl + loss_l1_image
    # zero_grad_all()
    loss_G.backward(retain_graph=True)
    
    

    # mu_, logvar_ = encoder(fake_B_cLR.detach())
    for param in encoder.parameters():
      param.requires_grad = False
    # loss_latent = lambda_latent * loss_z(fake_B_cLR, encoder, rand_z, mae_loss)
    loss_latent = lambda_latent * mae_loss(mu_, rand_z)
    # zero_grad_all()
    loss_latent.backward()
    for param in encoder.parameters():
      param.requires_grad = True

    optimizer_E.step()
    optimizer_G.step()

    #----------------------------------
    #  Train Discriminator (cVAE-GAN)
    #----------------------------------
    for param in D_VAE.parameters():
      param.requires_grad = True
    for param in D_LR.parameters():
      param.requires_grad = True

    optimizer_D_VAE.zero_grad()
    optimizer_D_LR.zero_grad()
    # mu, logvar = encoder(real_B)
    # z = reparameterization(mu, logvar)
    # fake_B_cVAE = generator(real_A, z)
    # rand_z = torch.randn(real_A.shape[0], latent_dim).to(gpu_id)
    # fake_B_cLR = generator(real_A, rand_z)

    loss_D = 0.0
    loss_D_gan_vae = loss_discriminator(fake_B_cVAE.detach(), D_VAE, real_B, mse_loss, Tensor)
    #---------------------------------
    #  Train Discriminator (cLR-GAN)
    #---------------------------------
    loss_D_gan = loss_discriminator(fake_B_cLR.detach(), D_LR, real_B, mse_loss, Tensor)
    # wrap up
    loss_D = loss_D_gan_vae + loss_D_gan
    # zero_grad_all()
    loss_D.backward()
    optimizer_D_VAE.step()
    optimizer_D_LR.step()

    """ Optional TODO: 
      1. You may want to visualize results during training for debugging purpose
      2. Save your model every few iterations
    """
    running_loss_l1_image += loss_l1_image.item() / lambda_pixel
    running_loss_l1_z += loss_latent.item() / lambda_latent
    running_loss_gan += (loss_D_gan + loss_G_cLR).item()
    running_loss_gan_vae += (loss_D_gan_vae + loss_G_cVAE).item()
    running_loss_kl += loss_kl.item() / lambda_kl
    running_total_loss += (loss_latent + loss_G + loss_D).item()

    if (idx+1) % 500 == 0:
      print('Train Epoch: {} {:.0f}% \tTotal Loss: {:.6f} \tLoss_l1_image: {:.6f} \tLoss_l1_z: {:.6f} \tLoss_GAN: {:.6f} \tLoss_GAN_VAE: {:.6f} \tLoss_KL: {:.6f}'.format
              (e, 100. * idx / len(loader), running_total_loss / (idx+1), running_loss_l1_image / (idx+1),
                running_loss_l1_z / (idx+1), running_loss_gan / (idx+1),
                running_loss_gan_vae / (idx+1), running_loss_kl / (idx+1)
                ))
      
      vis_real_A = denorm(real_A[0].detach()).cpu().data.numpy().astype(np.uint8)
      vis_fake_B_cVAE = denorm(fake_B_cVAE[0].detach()).cpu().data.numpy().astype(np.uint8)
      vis_real_B = denorm(real_B[0].detach()).cpu().data.numpy().astype(np.uint8)
      vis_fake_B_cLR = denorm(fake_B_cLR[0].detach()).cpu().data.numpy().astype(np.uint8)
      fig, axs = plt.subplots(2, 2, figsize=(5, 5))

      axs[0, 0].imshow(vis_real_A.transpose(1, 2, 0))
      axs[0, 0].set_title('real images')
      axs[0, 1].imshow(vis_fake_B_cVAE.transpose(1, 2, 0))
      axs[0, 1].set_title('generated images')
      axs[1, 0].imshow(vis_real_B.transpose(1, 2, 0))
      axs[1, 1].imshow(vis_fake_B_cLR.transpose(1, 2, 0))
      plt.savefig('./imgs/training_epoch_'+str(e)+'_'+str(idx)+'.png')

      
  loss_l1_image_list.append(running_loss_l1_image/len(loader))
  loss_l1_z_list.append(running_loss_l1_z/len(loader))
  loss_gan_list.append(running_loss_gan/len(loader))
  loss_gan_vae_list.append(running_loss_gan_vae/len(loader))
  loss_kl_list.append(running_loss_kl/len(loader))
  total_loss_list.append(running_total_loss/len(loader))

  print("saving...")
  ckpt = {
    'epoch':e,
    'generator':generator.state_dict(),
    'encoder':encoder.state_dict(),
    'D_VAE':D_VAE.state_dict(),
    'D_LR':D_LR.state_dict(),
    'optimizer_G': optimizer_G.state_dict(),
    'optimizer_E': optimizer_E.state_dict(),
    'optimizer_D_VAE':optimizer_D_VAE.state_dict(),
    'optimizer_D_LR':optimizer_D_LR.state_dict(),
    'loss_l1_image_list': loss_l1_image_list,
    'loss_l1_z_list': loss_l1_z_list,
    'loss_gan_list': loss_gan_list,
    'loss_gan_vae_list': loss_gan_vae_list,
    'loss_kl_list': loss_kl_list,
    'total_loss_list': total_loss_list
  }
  torch.save(ckpt, './checkpoints/bicyclegan_{}.pth'.format(e))

