from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import h5py
from data_load import *
import sklearn


# Load HR image
img = load_data(type='target', slice=20, echo=1, coil=1)

# Save original dimensions
orig_ky, orig_kx = img.shape
target = max(orig_kx, orig_ky)

# Pad the image with zeros to make it square
pad_y = target - orig_ky
pad_x = target - orig_kx
pad_y_before = pad_y // 2
pad_y_after  = pad_y - pad_y_before
pad_x_before = pad_x // 2
pad_x_after  = pad_x - pad_x_before

img_padded = np.pad(
    img,
    pad_width=((pad_y_before, pad_y_after), (pad_x_before, pad_x_after)),
    mode='constant',
    constant_values=0
)

# Normalize padded image and save as original reference
original_image = np.abs(img_padded)
org_min = original_image.min()
org_max = original_image.max()
original = (original_image - org_min) / (org_max - org_min)
org_img = (original * 255).astype(np.uint8)
Image.fromarray(org_img).save('...')

# Image to k-space
fourrier = np.fft.fft2(img_padded)
k_space = np.fft.fftshift(fourrier)

# Save k-space (log scale)
k_space_abs = np.abs(k_space)
k_space_log = np.log1p(k_space_abs)
log_min = k_space_log.min()
log_max = k_space_log.max()
k_space_save = (k_space_log - log_min) / (log_max - log_min)
k_space_save = (k_space_save * 255).astype(np.uint8)
Image.fromarray(k_space_save).save('C:/Users/floor/OneDrive - UvA/Bureaublad/kspacepad.tif')

# Save a true copy of the original k-space
k_space_original = k_space.copy()

# Set parameters
# Define k_space shape
ky, kx = k_space.shape

# Define middle point of k_space
cx = kx//2
cy = ky//2

# Define the radius of the k_space circle; the shortest centre coordinate defines this
r= min(cx, cy)

# Define thickness of the blade (chose yourself)
choose_width = 10

# Make a mesh grid of the kspace, matrix indexing
y, x = np.meshgrid(np.arange(ky), np.arange(kx), indexing='ij') 
# Center the coordinates around the middle of kspace
Xc = x-cx
Yc = y-cy

# Choose the amount of blades & define the angles
amount_blades = 25
angles = np.linspace(0.0, math.pi, amount_blades, endpoint= False) #exclude pi itself (same as 0)

#Compute signal power
signal_power = np.mean(np.abs(k_space_original)**2)
noise_level = 12.4 #20 dB means signal is 100x stronger than noise (10 dB = 10x, 30 dB = 1000x)
sigma = np.sqrt(signal_power/ (2*10** (noise_level/10)))
print(f'signal power is {signal_power:.4e}')
print(f'noise sigma is {sigma:.4e}')

# Initiate list library for rotations and k_space
blade_dict = {}

# Loop over all blades; rotate blade and take sample and save
for i, theta in enumerate(angles):
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Inverse rotation -->  from x,y (coordinate grid) to blade coordinates (u,v)
    u = Xc*cos_t + Yc*sin_t
    v = -Xc*sin_t + Yc*cos_t

    # Only take the values that lay within the blade length (r) and within half the width of the blade
    in_blade = ((u>=-r)&
                (u<=r)&
                (v>=(-choose_width/2))&
                (v<=(choose_width/2)))
    
    # Add the noise to the kspace
    k_space = k_space_original
 
    # Create a mask with for the values that fall within the blade
    mask_blade = in_blade
    # Fill the blade with the values from k_space
    k_space_blade = k_space * mask_blade 
   
    # Transform kspace into image
    img_from_blade = np.fft.ifft2(np.fft.ifftshift(k_space_blade))

    # Magnitude + min-max normalization
    img_mag = np.abs(img_from_blade)

    img_min = img_mag.min()
    img_max = img_mag.max()

    img_mag = (img_mag - img_min) / (img_max - img_min) 
    img_mag = np.clip(img_mag, 0, 1) # makes sure the data stays between 0 and 1 (exclude possible artefacts)
    img_uint8 = (img_mag * 255).astype(np.uint8)

    #save rotation, kspace data & image to dictionary
    blade_dict[i] = {
        "theta": float(theta),
        "kspace": k_space_blade,
        "mask": mask_blade,
        "image":img_uint8
    }

    Image.fromarray(img_uint8, mode='L').save(r'C:\Users\floor\OneDrive - UvA\JAAR 3\Stage\SR script\image\AHEAD\img' + str(i + 1) + '.tif')

    #save blade
    kspace_log = np.log10(1 + np.abs(k_space_blade))
    plt.figure(figsize=(5, 5))
    plt.imshow(kspace_log, cmap='gray')
    plt.title(f'k-space blade magnitude (log scale), theta={theta:.2f} rad')
    plt.axis('off')

    plt.savefig(r'C:\Users\floor\OneDrive - UvA\JAAR 3\Stage\SR script\image\AHEAD\blade' + str(i + 1) + '.tif',
                dpi=300, bbox_inches='tight')
    plt.close()
