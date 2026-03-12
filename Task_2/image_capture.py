import cv2 
import os

# Initializing the camera to capture the images
cam = cv2.VideoCapture(0)

store_path = 'assets'
os.makedirs(store_path, exist_ok=True)

img_id = 0

print("Starting the Image capturing sequence.\n")
print("------------------------------------------------\n")
print("Press 'c' to capture the image:\n")
print("Prexx 'x' to exit:\n")

while True:
    # Accessing the data through the camera
    ret, frame = cam.read()
    if not ret:
        print("Error accessing the image frames.")
        break
    
    cv2.imshow("Camera feed", frame)

    key = cv2.waitKey(1) & 0xFF
    # Storing the image as per the user input
    if key == ord('c'):
        img_name = f"{store_path}/image_{img_id+1}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} saved.")
        img_id += 1

    elif key == ord('x'):
        print("Images saved, exiting.")
        break

cam.release()
cv2.destroyAllWindows()