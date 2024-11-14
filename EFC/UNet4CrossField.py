#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:02:47 2024

@author: Richard Frazin
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


if not torch.cuda.is_available():
   print("Read the Sign.  No GPU, No Service!")
   assert False
device = torch.device('cuda')

in_channels = 2; out_channels = 2  # the images are complex valued.
# Define the UNet Model
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder
        self.dec4 = self.conv_block(512, 256)
        self.dec3 = self.conv_block(256, 128)
        self.dec2 = self.conv_block(128, 64)
        self.dec1 = self.conv_block(64, 32)

        # Output layer
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder
        dec4 = self.dec4(F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=False))
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec3 = self.dec3(F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=False))
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec2 = self.dec2(F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=False))
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec1 = self.dec1(F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=False))
        dec1 = torch.cat([dec1, enc1], dim=1)

        # Output layer
        out = self.out_conv(dec1)
        return out

# 3. Define the Complex Image Dataset Class
class ComplexImageDataSet(torch.utils.data.Dataset):
    def __init__(self, input_images, target_images, transform=None):
        """
        Args:
            input_images (array-like or tensor): Complex-valued input images.
            target_images (array-like or tensor): Complex-valued target images.
            transform (callable, optional): Optional transform to be applied to the input and target.
        """
        self.input_images = input_images  # List or array of complex-valued input images
        self.target_images = target_images  # List or array of complex-valued target images
        self.transform = transform  # Transform function to be applied to each image (input & target)

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.input_images)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            input_img (tensor): A tensor of shape (2, H, W) where the first channel is the real part,
                                 and the second channel is the imaginary part.
            target_img (tensor): Same as input_img but for the target image.
        """
        input_img = self.input_images[idx]  # Get the complex-valued input image
        target_img = self.target_images[idx]  # Get the complex-valued target image

        # Convert the complex-valued images into two-channel tensors (real and imaginary)
        input_real = input_img.real
        input_imag = input_img.imag
        target_real = target_img.real
        target_imag = target_img.imag

        # Stack the real and imaginary parts into a single tensor with shape (2, H, W)
        input_img = torch.stack([input_real, input_imag], dim=0)  # (2, H, W)
        target_img = torch.stack([target_real, target_imag], dim=0)  # (2, H, W)

        # Apply any transformations (if provided)
        if self.transform:
            input_img, target_img = self.transform(input_img, target_img)

        return input_img, target_img

# 4. Helper Functions for Saving and Loading Checkpoints and Models
def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save the model and optimizer state dict to a file.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at epoch {epoch}, loss {loss:.4f} to {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """
    Load the model and optimizer state dict from a checkpoint file.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filepath}, epoch {epoch}, loss {loss:.4f}")
    return epoch, loss

def save_model_for_inference(model, filepath):
    """
    Save the trained model for inference.
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved for inference to {filepath}")

def load_model_for_inference(model, filepath):
    """
    Load a trained model for inference.
    """
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded for inference from {filepath}")

# 5. Training and Checkpoint Saving Logic
def train_model(model, train_loader, optimizer, criterion, epochs, checkpoint_dir, checkpoint_freq):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Save checkpoint at specified frequency
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_filepath = f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, checkpoint_filepath)

# 6. Visualization Function
def visualize_results(model, input_image, target_image):
    model.eval()
    with torch.no_grad():
        output = model(input_image)

    input_img = input_image[0].cpu().numpy()
    target_img = target_image[0].cpu().numpy()
    output_img = output[0].cpu().numpy()

    # Plot real and imaginary parts for input, target, and output images
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot input (real and imaginary parts)
    axs[0].imshow(input_img[0], cmap='gray')
    axs[0].set_title('Input Image (Real part)')

    axs[1].imshow(target_img[0], cmap='gray')
    axs[1].set_title('Target Image (Real part)')

    axs[2].imshow(output_img[0], cmap='gray')
    axs[2].set_title('Predicted Image (Real part)')

    plt.show()
