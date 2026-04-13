#Part 2

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel

sigma = 2
rho = 2.5
k = 0.05
theta_corn = 0.005
s = 1.5
N = 4

#2.1.1


def compute_structure_tensor(I, sigma, rho):

    # smoothing
    I_sigma = gaussian_filter(I, sigma)

    # gradients
    Ix = sobel(I_sigma, axis=1)
    Iy = sobel(I_sigma, axis=0)

    # products
    Ix2 = Ix**2
    Iy2 = Iy**2
    Ixy = Ix * Iy

    # integration scale
    J1 = gaussian_filter(Ix2, rho)
    J2 = gaussian_filter(Ixy, rho)
    J3 = gaussian_filter(Iy2, rho)

    return J1, J2, J3



#I = cv2.imread("sun.jpg")  #για solar
I = cv2.imread("cells_of_blood.jpg")  #για blood_cells

I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
I = I.astype(np.float64)

J1, J2, J3= compute_structure_tensor(I, sigma, rho)


#2.1.2 


lambda_plus  = 0.5*(J1 + J3 + np.sqrt((J1 - J3)**2 + 4*(J2**2)))
lambda_minus = 0.5*(J1 + J3 - np.sqrt((J1 - J3)**2 + 4*(J2**2)))

#κάνω normalization γιατί τα δεδομένα της εικόνας έχουν τιμές εκτός του σωστού εύρους
lambda_plus_norm = (lambda_plus - lambda_plus.min()) / (lambda_plus.max() - lambda_plus.min())
lambda_minus_norm = (lambda_minus - lambda_minus.min()) / (lambda_minus.max() - lambda_minus.min())


plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("lambda_plus")
plt.imshow(lambda_plus_norm, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("lambda_minus")
plt.imshow(lambda_minus_norm, cmap='gray')
plt.axis('off')

plt.show()

#παρατηρήσεις στην αναφορά

#2.1.3 

from part2_utilities import disk_strel

R = lambda_minus * lambda_plus - k*(lambda_minus + lambda_plus)**2

Rmax = np.max(R)

passes_threshold = R > theta_corn * Rmax

ns = int(np.ceil(3*sigma)*2 + 1)
disk_kernel = disk_strel(ns)

is_local_max = (R == cv2.dilate(R, disk_kernel))

corners = is_local_max & passes_threshold
#corners_scales.append(corners)

y, x = np.where(corners)

sigma_arr = sigma * np.ones_like(x)

kp_data = np.stack([x, y, sigma_arr], axis=1)

from part2_utilities import render_interest_points

render_interest_points(I, kp_data)

plt.show()


#2.2.1

sigmas = []
rhos = []

for i in range(N):

    sigma_i = (s**i) * sigma
    rho_i   = (s**i) * rho

    sigmas.append(sigma_i)
    rhos.append(rho_i)
    
print(sigmas)
print(rhos)

#2.2.2 


LoG_scales = []

for i in range(N):

    sigma_i = sigma*(s**i)

    I_sigma = gaussian_filter(I, sigma_i)

    LoG = sigma_i**2 * np.abs(cv2.Laplacian(I_sigma, cv2.CV_64F))

    LoG_scales.append(LoG)
    
#print(LoG_scales)
    
       
kp_data = []

for i in range(1, N-1):

    sigma_i = sigma*(s**i)

    #corners = corners_scales[i]

    y, x = np.where(corners)

    for j in range(len(x)):

        val = LoG_scales[i][y[j], x[j]]

        if (val > LoG_scales[i-1][y[j], x[j]] and
            val > LoG_scales[i+1][y[j], x[j]]):

            kp_data.append([x[j], y[j], sigma_i])

render_interest_points(I, np.array(kp_data))
plt.show()      

#2.3.1

#Gaussian smoothing
from scipy.ndimage import gaussian_filter

I_sigma = gaussian_filter(I, sigma)

#δεύτερες παράγωγοι
import cv2

Lxx = cv2.Sobel(I_sigma, cv2.CV_64F, 2, 0)
Lyy = cv2.Sobel(I_sigma, cv2.CV_64F, 0, 2)
Lxy = cv2.Sobel(I_sigma, cv2.CV_64F, 1, 1)

#ορίζουσα
R = Lxx * Lyy - Lxy**2
plt.imshow(R, cmap='gray')
plt.title("Blob response")
plt.axis('off')

#2.3.2
#blobs
Rmax = np.max(R)
passes_threshold = R > theta_corn * Rmax

ns = int(np.ceil(3*sigma)*2+1)
disk_kernel = disk_strel(ns)

is_local_max = (R == cv2.dilate(R, disk_kernel))

blobs = is_local_max & passes_threshold
#kp_data
y, x = np.where(blobs)

sigma_arr = sigma*np.ones_like(x)

kp_data = np.stack([x,y,sigma_arr],axis=1)

render_interest_points(I, kp_data)
plt.show()

#2.4.1
sigmas = [sigma*(s**i) for i in range(N)]

blobs_scales = []
LoG_scales = []
#Blob detector
for i in range(N):

    sigma_i = sigma*(s**i)

    I_sigma = gaussian_filter(I, sigma_i)

    # Hessian derivatives
    Lxx = cv2.Sobel(I_sigma, cv2.CV_64F, 2, 0)
    Lyy = cv2.Sobel(I_sigma, cv2.CV_64F, 0, 2)
    Lxy = cv2.Sobel(I_sigma, cv2.CV_64F, 1, 1)

    R = Lxx*Lyy - Lxy**2
#Threshold + local maxima    
Rmax = np.max(R)

passes_threshold = R > theta_corn * Rmax

ns = int(np.ceil(3*sigma_i)*2 + 1)

disk_kernel = disk_strel(ns)

is_local_max = (R == cv2.dilate(R, disk_kernel))

blobs = is_local_max & passes_threshold

blobs_scales.append(blobs)
#Laplacian
LoG = sigma_i**2 * np.abs(cv2.Laplacian(I_sigma, cv2.CV_64F))
LoG_scales.append(LoG)
#scale
kp_data = []

for i in range(1, len(blobs_scales)-1):

    blobs = blobs_scales[i]

    y, x = np.where(blobs)

    sigma_i = sigma*(s**i)

    for j in range(len(x)):

        val = LoG_scales[i][y[j], x[j]]

        if (val > LoG_scales[i-1][y[j], x[j]] and
            val > LoG_scales[i+1][y[j], x[j]]):

            kp_data.append([x[j], y[j], sigma_i])
            
#visualization


kp_data = np.array(kp_data)



if kp_data.ndim == 1:
    kp_data = kp_data.reshape(-1,3)

render_interest_points(I, kp_data)
plt.show()