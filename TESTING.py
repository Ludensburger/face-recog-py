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

# Process every other frame to save time
process_this_frame = True

face_locations = []
face_names = []
face_landmarks_list = []

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize for performance
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
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

    process_this_frame = not process_this_frame

    # Draw rectangles and labels
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up to full size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    # Draw facial landmarks
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            for point in face_landmarks[facial_feature]:
                point = (point[0] * 4, point[1] * 4)  # Scale back up to full size
                cv2.circle(frame, point, 1, (0, 0, 255), -1)

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