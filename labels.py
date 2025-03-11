import cv2
import face_recognition
import time
import os

# Facial Landmarks Detected:
    # chin: 17 points, left ear to right ear
    # left_eyebrow: 5 points, left to right
    # right_eyebrow: 5 points, keft to right
    # nose_bridge: 4 points, top to bottom
    # nose_tip: 5 points, left to right
    # left_eye: 6 points, starting from left corner and proceeding clockwise
    # right_eye: 6 points, starting from left corner and proceeding clockwise
    # top_lip: 12 points, starting from the left corner and proceeding clockwise above and around the mouth
    # bottom_lip: 12 points, starting from the right corner and proceeding counterclockwise below and around the mouth


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

# Mid resolution for performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# FPS tracking
start_time = time.time()
frame_count = 0

# Process every fourth frame to save time
process_this_frame = 0

# Flag to ensure landmarks are printed only once
landmarks_printed = False

# Facial features to skip
# Guide: chin, left_eyebrow, right_eyebrow, nose_tip, top_lip, bottom_lip
features_to_skip = []

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

            if face_landmarks_list and not landmarks_printed:
                print("\nFacial Landmarks Detected:")
                for face_landmarks in face_landmarks_list:
                    for facial_feature, points in face_landmarks.items():
                        print(f"{facial_feature}: {len(points)} points")
                landmarks_printed = True

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
            if facial_feature in features_to_skip:
                continue

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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()