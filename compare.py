import cv2
import face_recognition

# Load the first image file
image_path1 = r"photos\compare\younger.jpg"
image1 = face_recognition.load_image_file(image_path1)

# Load the second image file
image_path2 = r"photos\compare\current.jpg"
image2 = face_recognition.load_image_file(image_path2)

# Find face encodings in both images
face_encodings1 = face_recognition.face_encodings(image1)
face_encodings2 = face_recognition.face_encodings(image2)

if len(face_encodings1) == 1 and len(face_encodings2) == 1:
    # Compare the first face in each image
    face_encoding1 = face_encodings1[0]
    face_encoding2 = face_encodings2[0]

    # Compare faces
    results = face_recognition.compare_faces([face_encoding1], face_encoding2)
    face_distance = face_recognition.face_distance([face_encoding1], face_encoding2)

    # Define a threshold for face comparison
    threshold = 0.4

    # Calculate accuracy
    accuracy = (1 - face_distance[0]) * 100

    # Print the results
    print(f"Are the two faces the same? {results[0]}")
    print(f"Face distance: {face_distance[0]}")
    print(f"Threshold: {threshold}")
    print(f"Accuracy: {accuracy:.2f}%")

    print("Is this the same person?")

    if face_distance[0] <= threshold:
        print("Yes")
    else:
        print("No")

    # Display the images with landmarks
    for image, face_landmarks_list in zip([image1, image2], [face_recognition.face_landmarks(image1), face_recognition.face_landmarks(image2)]):
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for face_landmarks in face_landmarks_list:
            for facial_feature in face_landmarks.keys():
                for point in face_landmarks[facial_feature]:
                    cv2.circle(image_bgr, point, 2, (0, 0, 255), -1)
        cv2.imshow('Facial Landmarks', image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print("Could not find faces in one or both images.")