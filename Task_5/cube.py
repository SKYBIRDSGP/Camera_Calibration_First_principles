import cv2
import os
import glob
import numpy as np

M_int = np.load("../Task_2/K.npy")
D = np.load("../Task_4/D.npy")
W_pts = np.load("../Task_3/World_mtx.npy")
aD = np.array([D[0], D[1], 0, 0, 0], dtype=np.float32)

rows = 7
cols = 10

cube_points = np.float32([
    [0,0,0],
    [90,0,0],
    [90,60,0],
    [0,60,0],
    [0,0,-60],
    [90,0,-60],
    [90,60,-60],
    [0,60,-60]
])

def drawCube(img, img_points):
    img_points = np.int32(img_points).reshape(-1,2)

    cv2.drawContours(img, [img_points[:4]], -1, (255,0,0), 3)

    for i,j in zip(range(4),range(4,8)):
        cv2.line(img, tuple(img_points[i]), tuple(img_points[j]), (0,255,255), 3)
    
    cv2.drawContours(img, [img_points[4:]], -1, (0,0,255),3)

    return img

# Function to turn on the video, and then the algo will perform for each frame of the same
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret :
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ret_corner, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
    
    if ret_corner:
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11,11),
            (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        img_points = corners
        obj_points = W_pts[0][:,:3].astype(np.float32)

        # Using the PnP method to determine the pose 
        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            M_int,
            D
        )

        # Cube projection
        imgpts, _ = cv2.projectPoints(
            cube_points,
            rvec,
            tvec,
            M_int,
            D
        )
        frame = drawCube(frame, imgpts)
    
    cv2.imshow("Cube Projection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()