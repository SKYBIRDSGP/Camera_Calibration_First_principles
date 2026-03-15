# Since we had calculated the intrinsic matrix, we will first calculate the extrinsic matrix 
import cv2
import numpy as np
import glob
import os

# Loading the K and H matrices 
# as well as the coordinate points for the image and it's absolute coordinates
K = np.load("K.npy")
H = np.load("H.npy")

# Using Zhang's method 
K_inv = np.linalg.inv(K)
extrinsic_mtx = []

for h in H:
    
    B = K_inv @ h

    r1 = B[:,0]
    r2 = B[:,1]
    t  = B[:,2]

    l = 1 / np.linalg.norm(r1)

    r1 = l * r1
    r2 = l * r2
    t  = l * t
    r3 = np.cross(r1,r2)

    R =np.column_stack((r1, r2, r3))

    # Since the R we get now is not perfect, because of the noise in the data, we use SVD to get the optimal value of the rotation matrix R
    U, S, Vt = np.linalg.svd(R)
    R = U @ Vt

    M_ext = np.hstack((R, t.reshape(3,1)))
    extrinsic_mtx.append(M_ext)

extrinsic_mtx = np.array(extrinsic_mtx)
np.save("M_ext", extrinsic_mtx)