# Our aim here is to compute the distortion parameters : K1 and K2 from the image data, and the other parameters we have obtained through the second and third tasks, compute the error,
# and then using the least square method, we have to find the optimal values for K1 and K2, such that we are minimizing the least square error

import numpy as np
import cv2
from scipy.optimize import least_squares
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pattern_size = (10,7)
images = glob.glob("assets/*.jpg")
images.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

# importing the intrinsics and the extrinsics initially along with the world points
M_int = np.load("../Homography/K.npy")
M_ext = np.load("../Distortion_detection/M_ext.npy")
W_pts = np.load("../Distortion_detection/World_mtx.npy")

num_imgs = W_pts.shape[0]

projn_points = []
corner_points = []
history = []

for i in range(num_imgs):
    img = cv2.imread(images[i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    P = M_int @ M_ext[i]
    proj_coordinates = (P @ W_pts[i].T).T
    proj_points = proj_coordinates[:,:2] / proj_coordinates[:, 2][:, None]

    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:  
        points_cal = corners.reshape(-1,2)
        projn_points.append(proj_points)
        corner_points.append(points_cal)

    else:
        print("Error in detecting the image!!!")

# Getting the reprojection error
def reprojection_error(params, projn_points, corner_points, K):
    K1, K2 = params

    f_x = K[0,0]
    f_y = K[1,1]
    c_x = K[0,2]
    c_y = K[1,2]

    errors = []

    for points, corners in zip(projn_points, corner_points):
        for (u,v), corner in zip(points, corners):

            x = (u - c_x)/f_x
            y = (v - c_y)/f_y

            r2 = x*x + y*y

            distortion = K1*r2 + K2*(r2**2)

            u_dist = u + (u - c_x) * distortion
            v_dist = v + (v - c_y) * distortion

            u_obv, v_obv = corner 
            # our error is: e = predicted coordinate - observed coordinated
            errors.append(u_obv - u_dist)
            errors.append(v_obv - v_dist)
    
    errors = np.array(errors)
    history.append(np.sum(errors**2))
    
    return errors
    
initial = [1,1]

# using non-linear least square method to optimize our error
result = least_squares(
    reprojection_error,
    initial,
    args=(projn_points, corner_points, M_int)
)

K1, K2 = result.x

D = [K1 , K2]
np.save("D.npy", D) # Saves our Distorton Parameters

print("\nEstimated Distortion Parameters :")
print("K1 = ", K1)
print("K2 = ", K2)

plt.plot(history)
plt.xlabel("Iterations")
plt.ylabel("Reprojection Error")
plt.title("Reprojection Error while Optimization")

plt.savefig("reprojection_error.png", dpi=300)
print("Plot saved as reprojection_error.png")