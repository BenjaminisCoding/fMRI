from deepinv.utils.metric import cal_psnr
import matplotlib.pyplot as plt
import numpy as np 
import torch 
import fastmri

def plot_rec(x, x_hat):

    # Compute reconstruction metric
    psnr = cal_psnr(x.abs(), x_hat.abs()).item()
    
    plt.figure(figsize=(15,5))
    plt.subplot(131)
    plt.imshow(np.real(x_hat).squeeze(), cmap='viridis')
    plt.title('Reconstruction, PSNR = ' + str(psnr))
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(np.real(x_hat-x).squeeze(), cmap='viridis')
    plt.title('Error')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(np.real(x).squeeze(), cmap='viridis')
    plt.title('Target')
    plt.axis('off')
    plt.show()

def plot_img(x, back):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot original image
    axes[0].imshow(x[0,0])
    axes[0].axis('off')
    axes[0].set_title('Original Image')

    # Plot reconstructed image
    axes[1].imshow(torch.real(back[0,0]))
    axes[1].axis('off')
    axes[1].set_title('Reconstruction (real value)')

    # Plot reconstructed image, abs
    axes[2].imshow(torch.abs(back[0,0]))
    axes[2].axis('off')
    axes[2].set_title('Reconstruction (abs value)')
    
    plt.show()
    return 


def plot_images_coils(images):
    """
    Plot a grid of images.
    
    Args:
    - images (torch.Tensor): Tensor of shape (1, num_images, height, width) containing the images.
    """
    if images.shape[-1] == 2: ### complex values
        images = fastmri.complex_abs(images)
    num_images = images.size(1)
    num_rows = (num_images + 4) // 5  # Calculate number of rows needed for the grid
    fig, axes = plt.subplots(num_rows, 5, figsize=(15, num_rows*3))  # Create subplots grid
    
    for i in range(num_images):
        row = i // 5
        col = i % 5
        img = images[0, i]  # Get the ith image from the batch
        axes[row, col].imshow(img, cmap='gray')  # Plot the image
        axes[row, col].axis('off')  # Hide axis
        axes[row, col].set_title(f"Image {i+1}")  # Set title
    
    # Hide empty subplots if any
    for i in range(num_images, num_rows * 5):
        row = i // 5
        col = i % 5
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
