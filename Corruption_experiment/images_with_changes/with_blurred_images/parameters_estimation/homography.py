# Gives us the intrinsics of the camera
import cv2
import numpy as np
import glob
import os

pattern_size = (10,7)
side = 25   # size of the square
world_points = []

image_point_coordinates = []
global_point_coordinates = []
for i in range(7):
    for j in range(10):
        points = (j * side , i * side)
        world_points.append(points)

images = glob.glob("assets/*.jpg")
images.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

def Matrix_A(world_coordinates, image_coordinates):
    A = []

    for i in range(len(world_coordinates)):
        X , Y = world_coordinates[i]
        u , v = image_coordinates[i]

        row_1 = [-X, -Y, -1, 0, 0, 0, u*X, u*Y, u]
        row_2 = [0, 0, 0, -X, -Y, -1, v*X, v*Y, v]

        A.append(row_1)
        A.append(row_2)
    
    return np.array(A)

for image in images:
    img = cv2.imread(image)
    blurred_image = cv2.GaussianBlur(img, (9, 9), 0)
    gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gs_img, pattern_size, None)

    if ret:
        print(f"Corners in the Checkerboard are detected successfully for {image} !\n")
        corners = corners.reshape(-1,2)

        image_point_coordinates.append(corners)
        global_point_coordinates.append(np.array(world_points))
        cv2.drawChessboardCorners(img,pattern_size, corners, ret)
        cv2.imshow("Corner detection", img)
        cv2.waitKey(200)

    else:
        print("Checkerboard not detected!!!")

np.save("image_points.npy", image_point_coordinates) 
np.save("world_points.npy", global_point_coordinates)

homography_mtx = []

for i in range(len(image_point_coordinates)):

    world_coordinate_points = global_point_coordinates[i]
    image_coordinate_points = image_point_coordinates[i]

    A = Matrix_A(world_coordinate_points, image_coordinate_points)

    # Using SVD to get the matrix A simplified so as to find the eigen vectors corresponding to the smallest eigen values
    U, S, Vt = np.linalg.svd(A)

    # Getting those eigen values from Vt, which will be its last row.
    h = Vt[-1]
    H = h.reshape(3,3)

    # Normalizing the Homography matrix
    H = H / H[2,2]

    homography_mtx.append(H)
    # print(f"Homography Matrix for image {i+1} is :\n", H)


np.save("H.npy", homography_mtx)

# Now, obtaining the Intinsic Matrix 
def V_matrix(H, i, j):
    return np.array([
        H[0,i]*H[0,j],
        H[0,i]*H[1,j] + H[1,i]*H[0,j],
        H[1,i]*H[1,j],
        H[2,i]*H[0,j] + H[0,i]*H[2,j],
        H[2,i]*H[1,j] + H[1,i]*H[2,j],
        H[2,i]*H[2,j]
    ])     

V = []

for H in homography_mtx:
    V_12 = V_matrix(H,0,1)
    V_11 = V_matrix(H,0,0)
    V_22 = V_matrix(H,1,1)

    V.append(V_12)
    V.append(V_11 - V_22)

V = np.array(V)
# print(V.shape)

# Now, extracting the B matrix from V using SVD
U, S, Vt = np.linalg.svd(V)

# Getting those eigen values from Vt, which will be its last row.
b = Vt[-1] 

# Formulating the B matrix
B11, B12, B22, B13, B23, B33 = b 

B = np.array([
    [B11, B12, B13],
    [B12, B22, B23],
    [B13, B23, B33]
])

if B[0,0]<0:
    B = -B

# Using Chloesky decomposition to get L and thus, ultimately get K
L = np.linalg.cholesky(B)
K = np.linalg.inv(L.T)

K = K / K[2,2]
np.save("K.npy", K)

print("Intrinsic Matrix K :\n")
print(K)

f_x = K[0,0]
f_y = K[1,1]
o_x = K[0,2]
o_y = K[1,2]

print(f"The effective focal length in x : {f_x}")
print(f"The effective focal length in y : {f_y}")
print(f"The coordinates of the Principle point are : ( {o_x} , {o_y} )")

cv2.destroyAllWindows()