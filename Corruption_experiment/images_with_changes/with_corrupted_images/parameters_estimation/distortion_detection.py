# The aim is to initially assume zero distortion, then calculate and plot the reprojection errors,
# and based on these errors, we will then determine the type of distortion
import cv2 
import numpy as np
import glob
import os

pattern_size = (10,7)
images = glob.glob("assets/*.jpg")
images.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

# Loading the intrinsic and extrinsic matrices along with the coordinate points for images in world and image frame
K = np.load("K.npy")
img_points = np.load("image_points.npy")
world_points = np.load("world_points.npy")
M_ext = np.load("M_ext.npy")

num_images = len(img_points)
num_points = pattern_size[0] * pattern_size[1]

world_points = world_points.reshape(num_images, num_points, 2)
num_images = world_points.shape[0]
num_points = world_points.shape[1]

Z =np.zeros((num_images, num_points, 1))
ones = np.ones((num_images, num_points, 1))

world_mtx = np.concatenate((world_points, Z, ones), axis=2)
np.save("World_mtx", world_mtx)
print(world_mtx.shape[0])

image_errors = []
all_errors = []
all_points = []

for i in range(num_images):
    img = cv2.imread(images[i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    P = K @ M_ext[i]
    proj_coordinates = (P @ world_mtx[i].T).T
    proj_points = proj_coordinates[:,:2] / proj_coordinates[:, 2][:, None]

    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:             
        points_cal = corners.reshape(-1,2)

        error = proj_points - points_cal
        error_mag = np.linalg.norm(error, axis=1)

        mean_error = np.mean(error_mag)
        image_errors.append(mean_error)
        all_errors.append(error)
        all_points.append(points_cal)

        print(f" Error for the image {i+1} is : ", mean_error)

        for j in range(len(points_cal)):
            p1 = tuple(points_cal[j].astype(int))
            p2 = tuple(proj_points[j].astype(int))

            cv2.circle(img, p1, 2, (0,255,0), -1)
            cv2.circle(img, p2, 2, (0,0,255), -1)

            cv2.arrowedLine(img, p1, p2, (255,0,0), 1)
        store_path = 'output'
        os.makedirs(store_path, exist_ok=True)

        img_name = f"output/reproj_{i+1}.jpg"
        cv2.imwrite(img_name, img)
    
    else:
        print("Error detecting the image!!")

cv2.destroyAllWindows()