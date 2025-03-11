import cv2
import face_recognition

# Load the image file
image_path = "photos\Ryu.jpg"
image = face_recognition.load_image_file(image_path)

# Resize the image to a smaller size
scale_percent = 80  # Percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Find all facial landmarks in the resized image
face_landmarks_list = face_recognition.face_landmarks(resized_image)

# Convert the resized image to BGR color (which OpenCV uses)
image_bgr = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

# Draw facial landmarks on the resized image
for face_landmarks in face_landmarks_list:
    for facial_feature in face_landmarks.keys():
        for point in face_landmarks[facial_feature]:
            cv2.circle(image_bgr, point, 2, (0, 0, 255), -1)

# Display the resized image with landmarks
cv2.imshow('Facial Landmarks', image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()