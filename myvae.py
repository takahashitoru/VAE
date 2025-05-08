import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

##########################################################################
# Ouput 64x64 pixels
##########################################################################

class Encoder64(nn.Module):
  def __init__(self, z_dim):
    super().__init__()
    self.lr  = nn.Linear(64*64, 2048)
    self.lr2 = nn.Linear(2048, 1024)
    self.lr3 = nn.Linear(1024, 512)
    self.lr4 = nn.Linear(512, 256)
    self.lr5 = nn.Linear(256, 128)
    self.lr_ave = nn.Linear(128, z_dim) # average
    self.lr_dev = nn.Linear(128, z_dim) # log(sigma^2)
    self.relu = nn.ReLU()
  
  def forward(self, x):
    x = self.lr(x)
    x = self.relu(x)
    x = self.lr2(x)
    x = self.relu(x)
    x = self.lr3(x)
    x = self.relu(x)
    x = self.lr4(x)
    x = self.relu(x)
    x = self.lr5(x)
    x = self.relu(x)
    ave = self.lr_ave(x)
    log_dev = self.lr_dev(x)

    ep = torch.randn_like(ave)
    z = ave + torch.exp(log_dev / 2) * ep
    return z, ave, log_dev

class Decoder64(nn.Module):
  def __init__(self, z_dim):
    super().__init__()
    self.lr = nn.Linear(z_dim, 128)
    self.lr2 = nn.Linear(128, 256)
    self.lr3 = nn.Linear(256, 512)
    self.lr4 = nn.Linear(512, 1024)
    self.lr5 = nn.Linear(1024, 2048)
    self.lr6 = nn.Linear(2048, 64*64)
    self.relu = nn.ReLU()
  
  def forward(self, z):
    x = self.lr(z)
    x = self.relu(x)
    x = self.lr2(x)
    x = self.relu(x)
    x = self.lr3(x)
    x = self.relu(x)
    x = self.lr4(x)
    x = self.relu(x)
    x = self.lr5(x)
    x = self.relu(x)
    x = self.lr6(x)
    x = torch.sigmoid(x)
    return x

class VAE64(nn.Module):
  def __init__(self, z_dim):
    super().__init__()
    self.encoder = Encoder64(z_dim)
    self.decoder = Decoder64(z_dim)
  
  def forward(self, x):
    z, ave, log_dev = self.encoder(x)
    x = self.decoder(z)
    return x, z, ave, log_dev

  def criterion(self, predict, target, ave, log_dev):
    bce_loss = F.binary_cross_entropy(predict, target, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_dev - ave**2 - log_dev.exp())
    loss = bce_loss + kl_loss
    return loss

  def z2img(self, z, threshold = 0.5):
    x = self.decoder(z)
    x_np = x.to('cpu').detach().numpy().copy()
    img = np.reshape(x_np, (64, 64))
    img = 1.0 - img
    img = (img > threshold) * 1.0
    return img

  def z2png(self, z, png_file, threshold = 0.5):
    img = self.z2img(z, threshold)
    img_inverted = 1 - img # Invert black and white
    img_pil = Image.fromarray((img_inverted * 255).astype(np.uint8))
    img_pil.save(png_file, "PNG")

  
  
