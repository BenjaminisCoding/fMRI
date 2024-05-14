from deepinv.utils.metric import cal_psnr
import matplotlib.pyplot as plt
import numpy as np 
import torch 

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
