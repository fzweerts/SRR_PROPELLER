from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import h5py
from data_load import *
import sklearn
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim

# Collect all blades and masks
blades = []
masks = []

#Loop over the entire dictionary
for i in sorted(blade_dict.keys()):
    # Select all the k_spaces in the dictionary and append the blades list
    k_blade=blade_dict[i]["kspace"].astype(np.complex64)
    blades.append(k_blade)
    
    # Make a binary mask, with 1 where there is data (blade), and 0 where there is no data
    mask = blade_dict[i]["mask"].astype(np.float32)
    masks.append(mask)

# Stack info into a 3D array
blades = np.stack(blades, axis=0)
masks = np.stack(masks, axis=0)

# N= number of blades, H= height, W=width
N, H, W = blades.shape

#normalize by dividing with the sum of all kspace pixels --> root mean square
scale      = np.sqrt(np.sum(np.abs(blades) ** 2) / np.sum(masks > 0)) #RMS of all kspace pizels
blades     = blades / scale        # normalized blades
sigma_norm = sigma / scale      # normalized sigma

#add noise after normalization  
for n in range(N):
    noise = (np.random.normal(0, sigma_norm, (H, W)) +
             1j * np.random.normal(0, sigma_norm, (H, W))).astype(np.complex64)
    # only add noise to the pixels within the blade (mask), since outside is 0
    blades[n] += noise * masks[n]


# Initialize k-space guess, this is just the sum of all blades
k_init = np.sum(blades, axis=0)
# Turn into image domain, making it the initial HR image estimate
X =  np.fft.ifft2(np.fft.ifftshift(k_init)).astype(np.complex64)

#Initialize IBP parameters
max_iter = 100
iteration = 0
tol = 0.05  
alpha = 0.2

# Main IBP loop
while iteration < max_iter:
    print(f"on IBP interation {iteration}")

    # Save old estimate to measure change later
    X_old = X.copy()

    # Initialize gradient accumulator in image domain
    G_data = np.zeros_like(X, dtype=np.complex64)

    # Forward model: turn current image estimate into k_space
    K_est = np.fft.fftshift(np.fft.fft2(X))

    # Loop over each blade; compute k-space error and back-project
    for n in range(N):
        mask_n = masks[n]
        blade_n = blades[n]

        # "Pick" the k-space values of this blade of the estimates k-space by multiplying with the mask
        K_est_n = K_est * mask_n

        # Compute the error of the blade: difference predicted and measured blade
        E_n = K_est_n - blade_n

        # Back projection of the k-space error into image space
        g_n = np.fft.ifft2(np.fft.ifftshift(E_n))

        # Acumulate the gradient for all blades
        G_data += g_n
    
    #Tikhonov regularization: ||X||^2 --> (derivative for gradient) 2*X
    G_reg = 2*X
    # Add the regularization to the gradient
    G = G_data + alpha*G_reg

    #Determine the optimal lambda
    #forward projection of G to k space
    EG = np.fft.fftshift(np.fft.fft2(G)) 

    # Back project G per blade
    EHEg = np.zeros_like(G, dtype=np.complex64)
    for n in range(N):
        EHEg += np.fft.ifft2(np.fft.ifftshift(EG * masks[n]))

    # Include regularization term in denominator: E^H E G + 2*alpha*G
    EHEg_reg  = EHEg + 2.0 * alpha * G

    # lambda = <G, G> / <G, E^H E G>
    numerator   = np.sum(np.abs(G) ** 2)
    denominator = np.real(np.sum(np.conj(G) * EHEg_reg)) #use np.real to make sure output for lambda is real (not complex), np.conj is needed
    lamda = numerator / (denominator)

    print(f"lamda is {lamda:.4e}")

    # Update the gradient by moving the image estimate opposite to the gradient
    X = X - lamda * G

    # Save the generated image every 4 iterations to look at the convergence
    if iteration in range(0, max_iter, 10):
        X_mag = np.abs(X)
        X_min = X_mag.min()
        X_max = X_mag.max()
        X_norm = (X_mag - X_min) / (X_max - X_min) 
        X_clipped = np.clip(X_norm, 0, 1) # makes sure the data point stay between 0 and 1 (to exclude possible artefacts)
        X_iter = (X_clipped * 255).astype(np.uint8)
        Image.fromarray(X_iter, mode='L').save(r'C:\Users\floor\OneDrive - UvA\JAAR 3\Stage\SR script\image\AHEAD\IBP_iter' + str(iteration + 1) + '.tif')

    # Compute the change
    diff = X - X_old

    rel_change = np.linalg.norm(diff.ravel()) / (np.linalg.norm(X_old.ravel()) + 1e-12)

    # Stop if the relative change falls below the chosen tolerance
    if rel_change < tol:
        print("Converged: relative change below tolerance.")
        break

    iteration += 1

    
# magnitude + min max normalization
img_mag = np.abs(X)
img_min = img_mag.min()
img_max = img_mag.max()
img_mag = (img_mag - img_min) / (img_max - img_min) 
HR_image = np.clip(img_mag, 0, 1) # makes sure the data point stay between 0 and 1 (to exclude possible artefacts)


# Compute the MSE to compute the PSNR
MSE_img = np.mean((original - HR_image)**2)
PSNR_img = 20*log10(1/(np.sqrt(MSE_img)))
print(f'MSE is {MSE_img} \nPSNR is {PSNR_img}')

HR_save = (HR_image * 255).astype(np.uint8)
Image.fromarray(HR_save).save('C:/Users/floor/OneDrive - UvA/Bureaublad/result.tif')

# Calculate mean and std
mu_o, sigma_o = original.mean(), original.std()
mu_r, sigma_r = HR_image.mean(), HR_image.std()

# Linear correction: match mean & std
HR_match = (HR_image - mu_r) * (sigma_o / (sigma_r + 1e-12)) + mu_o
HR_match = np.clip(HR_match, 0, 1)

# Define ROI
mask = original > 0.05  # choose threshold for ROI

orig_roi = original[mask]
hr_roi   = HR_image[mask]

orig_roi_img = original.copy()
hr_roi_img   = HR_image.copy()
orig_roi_img[~mask] = 0
hr_roi_img[~mask]   = 0

ssim_full = ssim(original, HR_image, data_range=1.0)
ssim_roi  = ssim(orig_roi_img, hr_roi_img, data_range=1.0)
print("SSIM full image:", ssim_full)
print("SSIM in ROI:    ", ssim_roi)
