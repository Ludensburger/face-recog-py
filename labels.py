import cv2
import face_recognition
import time
import os

# Load all known faces
known_face_encodings = []
known_face_names = []

# Path to photos directory of known faces
photos_path = "photos"

# Load and encode all images from /photos a.k.a. known faces
for person in os.listdir(photos_path):
    person_dir = os.path.join(photos_path, person)
    if os.path.isdir(person_dir):
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(person)
    else:
        image_path = person_dir
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(person)[0])

# Find a working camera index
cap = None
for i in range(3):  # Try 0, 1, 2
    temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if temp_cap.read()[0]:
        cap = temp_cap
        print(f"Camera index {i} is working.")
        break
    temp_cap.release()

if not cap:
    print("No working camera found.")
    exit()

# Lower resolution for performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# FPS tracking
start_time = time.time()
frame_count = 0

# Process every fourth frame to save time
process_this_frame = 0

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)

    # Resize for performance
    small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)  # Reduce size further
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame % 4 == 0:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = []
        face_landmarks_list = []

        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)

    process_this_frame += 1

    # Draw facial landmarks with index numbers
    for face_landmarks in face_landmarks_list:
        for facial_feature, points in face_landmarks.items():
            for index, point in enumerate(points):
                scaled_point = (point[0] * 5, point[1] * 5)  # Scale back up to full size
                
                # Draw the landmark point
                cv2.circle(frame, scaled_point, 1, (0, 0, 255), -1)

                # Label with index number
                text = str(index)  # Convert index to string
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)[0]
                label_position = (scaled_point[0] + 5, scaled_point[1] - 5)  # Offset for better visibility
                
                cv2.putText(frame, text, label_position, cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 0), 1)

    # Show the frame
    cv2.imshow('Face Recognition', frame)

    # OPTIONAL: Display
    # Print FPS every second
    frame_count += 1
    if time.time() - start_time >= 1:
        print(f"FPS: {frame_count}")
        frame_count = 0
        start_time = time.time()
    # OPTIONAL: Display

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()