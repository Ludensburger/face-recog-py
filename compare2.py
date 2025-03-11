import cv2
import face_recognition

def preprocess_image(image_path):
    image = face_recognition.load_image_file(image_path)
    # Normalize lighting (without converting to grayscale)
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab_image)
    l = cv2.equalizeHist(l)
    normalized_image = cv2.merge((l, a, b))
    normalized_image = cv2.cvtColor(normalized_image, cv2.COLOR_LAB2RGB)
    return normalized_image

# Load and preprocess the first image file
image_path1 = r"C:\Users\Bernard\Desktop\ryu-files\code\python\cs-elec2\face-recog-py\photos\compare\younger.jpg"
image1 = preprocess_image(image_path1)

# Load and preprocess the second image file
image_path2 = r"C:\Users\Bernard\Desktop\ryu-files\code\python\cs-elec2\face-recog-py\photos\compare\current.jpg"
image2 = preprocess_image(image_path2)

# Find face encodings in both images
face_encodings1 = face_recognition.face_encodings(image1)
face_encodings2 = face_recognition.face_encodings(image2)

if len(face_encodings1) > 0 and len(face_encodings2) > 0:
    # Compare the first face in each image
    face_encoding1 = face_encodings1[0]
    face_encoding2 = face_encodings2[0]

    # Compare faces
    results = face_recognition.compare_faces([face_encoding1], face_encoding2, tolerance=0.5)  # Adjusted threshold
    face_distance = face_recognition.face_distance([face_encoding1], face_encoding2)

    # Calculate accuracy
    accuracy = (1 - face_distance[0]) * 100

    # Print the results
    print(f"Are the two faces the same? {results[0]}")
    print(f"Face distance: {face_distance[0]}")
    print(f"Threshold: 0.5")  # Adjusted threshold
    print(f"Accuracy: {accuracy:.2f}%")

    print("Is this the same person?")

    if accuracy >= 50:  # Adjusted threshold
        print("Yes")
    else:
        print("No")
else:
    print("Could not find faces in one or both images.")