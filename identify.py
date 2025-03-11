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

    # Draw facial landmarks
    for face_landmarks in face_landmarks_list:
        for facial_feature, points in face_landmarks.items():
            for point in points:
                point = (point[0] * 5, point[1] * 5)  # Scale back up to full size
                cv2.circle(frame, point, 1, (0, 0, 255), -1)

            # Add labels for specific facial features

            # where point [0][0] is x and point [0][1] is y
            if facial_feature == 'left_eye':
                text = 'Left Eye'
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]  # Get text width and height
                label_position = (points[0][0] * 5 - 125, points[0][1] * 5 - 40)
                
                cv2.putText(frame, text, label_position, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
                
                # Adjust line connection point to the end of the text
                line_end_position = (label_position[0] + text_size[0], label_position[1])  
                cv2.line(frame, (points[0][0] * 5, points[0][1] * 5), line_end_position, (255, 255, 255), 1)

            elif facial_feature == 'right_eye':
                # Choose a specific point for the right eye (e.g., the first point)
                specific_point = points[3]  # Change the index to choose a different point
                label_position = (specific_point[0] * 5 + 75, specific_point[1] * 5 - 40)
                cv2.putText(frame, 'Right Eye', label_position, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
                cv2.line(frame, (specific_point[0] * 5, specific_point[1] * 5), label_position, (255, 255, 255), 1)

                
            elif facial_feature == 'nose_bridge': 
                # or you can use nose_tip

                text = 'Nose'
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]  # Get text width and height
                
                # Choose a specific point for the nose (e.g., the 2nd point in the list)
                specific_point = points[3]  # Change index to select a different landmark if needed

                # Adjust label position relative to the specific point
                label_position = (specific_point[0] * 5 - 150, specific_point[1] * 5)

                cv2.putText(frame, text, label_position, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

                # Adjust line connection point to the end of the text
                line_end_position = (label_position[0] + text_size[0], label_position[1])  
                cv2.line(frame, (specific_point[0] * 5, specific_point[1] * 5), line_end_position, (255, 255, 255), 1)

                # label_position = (points[0][0] * 5 - 100, points[0][1] * 5 + 40)
                # cv2.putText(frame, 'Nose', label_position, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
                # cv2.line(frame, (points[0][0] * 5, points[0][1] * 5), label_position, (255, 255, 255), 1)
            
            elif facial_feature == 'top_lip':
                text = 'Mouth'
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]  # Get text width and height
                
                # Choose a specific point for the nose (e.g., the 2nd point in the list)
                specific_point = points[3]  # Change index to select a different landmark if needed

                # Adjust label position relative to the specific point
                label_position = (specific_point[0] * 5 - 150, specific_point[1] * 5 + 25)

                cv2.putText(frame, text, label_position, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1)

                # Adjust line connection point to the end of the text
                line_end_position = (label_position[0] + text_size[0], label_position[1])  
                cv2.line(frame, (specific_point[0] * 5, specific_point[1] * 5), line_end_position, (255, 255, 255), 1)

                # label_position = (points[0][0] * 5 - 75, points[0][1] * 5 + 60)
                # cv2.putText(frame, 'Mouth', label_position, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1)
                # cv2.line(frame, (points[0][0] * 5, points[0][1] * 5), label_position, (255, 255, 255), 1)

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