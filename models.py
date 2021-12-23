from torchvision.models import resnet18
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import pdb
from torch.autograd import Variable

##############################
#        Encoder 
##############################
class Encoder(nn.Module):
    def __init__(self, latent_dim):
      super(Encoder, self).__init__()
      """ The encoder used in both cVAE-GAN and cLR-GAN, which encode image B or B_hat to latent vector
          This encoder uses resnet-18 to extract features, and further encode them into a distribution
          similar to VAE encoder. 

          Note: You may either add "reparametrization trick" and "KL divergence" or in the train.py file
          
          Args in constructor: 
              latent_dim: latent dimension for z 

          Args in forward function: 
              img: image input (from domain B)

          Returns: 
              mu: mean of the latent code 
              logvar: sigma of the latent code 
      """

      # Extracts features at the last fully-connected
      resnet18_model = resnet18(pretrained=True)      
      self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])
      self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)

      # Output is mu and log(var) for reparameterization trick used in VAEs
      self.fc_mu = nn.Linear(256, latent_dim)
      self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, img):
      out = self.feature_extractor(img)
      out = self.pooling(out)
      out = out.view(out.size(0), -1)
      mu = self.fc_mu(out)
      logvar = self.fc_logvar(out)
      return mu, logvar


##############################
#        Generator 
##############################
class Generator(nn.Module):
    """ The generator used in both cVAE-GAN and cLR-GAN, which transform A to B
        
        Args in constructor: 
            latent_dim: latent dimension for z 
            image_shape: (channel, h, w), you may need this to specify the output dimension (optional)
        
        Args in forward function: 
            x: image input (from domain A)
            z: latent vector (encoded B)

        Returns: 
            fake_B: generated image in domain B
    """
    # use U-Net as suggested
    def __init__(self, latent_dim, img_shape):
      super(Generator, self).__init__()
      channels, self.h, self.w = img_shape
      # (TODO: add layers...)
      self.latent_dim = latent_dim

      self.down1 = nn.Sequential(
        nn.Conv2d(in_channels = channels + latent_dim, out_channels = 64, kernel_size = 4, stride = 2, padding = 1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
      )
      self.down2 = nn.Sequential(
        nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
        nn.InstanceNorm2d(128, affine=True),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
      )
      self.down3 = nn.Sequential(
        nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1),
        nn.InstanceNorm2d(256, affine=True),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
      )
      self.down4 = nn.Sequential(
        nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4, stride = 2, padding = 1),
        nn.InstanceNorm2d(512, affine=True),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
      )
      self.down5 = nn.Sequential(
        nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 4, stride = 2, padding = 1),
        nn.InstanceNorm2d(512, affine=True),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
      )

      self.up1 = nn.Sequential(
        nn.ConvTranspose2d(in_channels = 512, out_channels = 512, kernel_size = 4, stride = 2, padding = 1),
        nn.InstanceNorm2d(512, affine=True),
        nn.ReLU(inplace=True)
      )
      self.up2 = nn.Sequential(
        nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 4, stride = 2, padding = 1),
        nn.InstanceNorm2d(512, affine=True),
        nn.ReLU(inplace=True)
      )
      self.up3 = nn.Sequential(
        nn.ConvTranspose2d(in_channels = 1024, out_channels = 256, kernel_size = 4, stride = 2, padding = 1),
        nn.InstanceNorm2d(256, affine=True),
        nn.ReLU(inplace=True)
      )
      self.up4 = nn.Sequential(
        nn.ConvTranspose2d(in_channels = 512, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
        nn.InstanceNorm2d(128, affine=True),
        nn.ReLU(inplace=True)
      )
      self.up5 = nn.Sequential(
        nn.ConvTranspose2d(in_channels = 256, out_channels = 64, kernel_size = 4, stride = 2, padding = 1),
        nn.InstanceNorm2d(64, affine=True),
        nn.ReLU(inplace=True)
      )

      self.out = nn.Sequential(
        nn.ConvTranspose2d(in_channels = 128, out_channels = 3, kernel_size = 4, stride = 2, padding = 1),
        nn.Tanh()
      )

    def forward(self, x, z):
      # (TODO: add layers...)
      # reshaping z and concat with x, from original implementation
      z = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
      try:
        x_and_z = torch.cat([x, z], 1)
      except:
        print('Error, x: {}, z: {}'.format(x.shape, z.shape))
      # downsample
      d1 = self.down1(x_and_z) # -64
      d2 = self.down2(d1) # 64-128
      d3 = self.down3(d2) # 128-256
      d4 = self.down4(d3) # 256-512
      d5 = self.down5(d4) # 512-512
      # upsample
      u1 = d5 # 512-512
      u2 = self.up2(torch.cat([u1, d5], dim=1)) # 1024-512
      u3 = self.up3(torch.cat([u2, d4], dim=1)) # 1024-256
      u4 = self.up4(torch.cat([u3, d3], dim=1)) # 512-128
      u5 = self.up5(torch.cat([u4, d2], dim=1)) # 256-64
      # output
      output = self.out(torch.cat([u5, d1], dim=1)) # 128-3
      return output

##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    # use PatchGAN discriminator as suggested; optimize two slightly D as suggested
    def __init__(self, in_channels=3):
      super(Discriminator, self).__init__()
      """ The discriminator used in both cVAE-GAN and cLR-GAN
          
          Args in constructor: 
              in_channels: number of channel in image (default: 3 for RGB)

          Args in forward function: 
              x: image input (real_B, fake_B)

          Returns: 
              discriminator output: could be a single value or a matrix depending on the type of GAN
      """
      self.D = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.ZeroPad2d((1,0,1,0)),
        nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, padding=1),
        nn.Sigmoid()
      )
      
    def forward(self, x):
      output = self.D(x)
      return output

##############################
#        Loss
##############################
def loss_generator(G, real_img, z, target_img, criterion):
  fake_img = G(real_img, z)
  assert fake_img.shape == target_img.shape
  return criterion(fake_img, target_img)

def loss_discriminator_one(img, D, criterion, Tensor):
  img_d = D(img)
  valid = Variable(Tensor(np.ones(img_d.shape)), requires_grad=False)
  loss = criterion(img_d, valid)
  return loss

def loss_discriminator_zero(img, D, criterion, Tensor):
  img_d = D(img)
  valid = Variable(Tensor(np.zeros(img_d.shape)), requires_grad=False)
  loss = criterion(img_d, valid)
  return loss

def loss_discriminator(fake_img, D, real_img, criterion, Tensor):
  real_loss = loss_discriminator_one(real_img, D, criterion, Tensor)
  fake_loss = loss_discriminator_zero(fake_img, D, criterion, Tensor)
  return real_loss + fake_loss

def loss_kld(mu, logvar):
  return torch.sum(0.5*(mu**2 + torch.exp(logvar) - logvar - 1))

def loss_z(fake_img, encoder, z0, criterion):
  mu, logvar = encoder(fake_img.detach())
  z_loss = criterion(mu, z0)
  return z_loss
