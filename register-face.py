import cv2
import os

name = input("Enter your name: ")
save_path = f"photos/{name}"

os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
count = 0
max_images = 10  # Number of images to capture

while count < max_images:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Face (press 'space' to save)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Spacebar key
        filename = f"{save_path}/{name}_{count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
