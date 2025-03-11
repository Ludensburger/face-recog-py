import cv2
import face_recognition
import os

# Load the image file
# image_path = "photos\\Ryu.jpg"
image_path = "photos\\Art.jpg"
image_name = image_path.split("\\")[-1]
image = face_recognition.load_image_file(image_path)

# Find all facial landmarks in the original image
face_landmarks_list = face_recognition.face_landmarks(image)

# Convert the original image to BGR (OpenCV format)
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Calculate font size based on image size
image_height, image_width = image_bgr.shape[:2]
font_scale = max(image_width, image_height) / 1000  # Adjust the divisor to control the font size

# Calculate dynamic offset based on image size
offset_x = int(image_width * 0.1)
offset_y = int(image_height * 0.05)

# Print the size of the image
print(f"Image size: {image_width}x{image_height}")


if image_width > image_height:
    # Landscape orientation
    extra_offset_x = image_width / 10
    extra_offset_y = image_height / 10
else:
    # Portrait orientation
    extra_offset_x = image_width / 6
    extra_offset_y = image_height / 7

# Function to draw labels and connect lines to a specific point
def draw_label(image, text, feature_point, label_offset, text_color, line_color, specific_point_index, connect_to_last_letter=True):
    x, y = feature_point[specific_point_index]
    label_x, label_y = int(x + label_offset[0]), int(y + label_offset[1])  # Ensure integer values

    # Calculate text width for line connection
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)[0]
    if connect_to_last_letter:
        label_end_x = int(label_x + text_size[0])  # Convert to integer
    else:
        label_end_x = label_x  

    # Draw label text
    # cv2.putText(image, text, (label_x, label_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, 1)
    cv2.putText(image, text, (label_x, label_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, 2, lineType=cv2.LINE_4)

    # Draw connecting line
    cv2.line(image, (x, y), (label_end_x, label_y), line_color, 1)

# Define label offsets, colors, and connection preferences
feature_settings = {
    'left_eye': {'label': 'Left Eye', 'offset': (offset_x - extra_offset_x * 2.6, offset_y - extra_offset_y), 'text_color': (0, 255, 0), 'line_color': (255, 255, 255), 'specific_point_index': 0, 'connect_to_last_letter': True},
    'right_eye': {'label': 'Right Eye', 'offset': ((offset_x + extra_offset_x)/4.1, -offset_y), 'text_color': (0, 255, 0), 'line_color': (255, 255, 255), 'specific_point_index': 3, 'connect_to_last_letter': False},
    'nose_bridge': {'label': 'Nose', 'offset': (offset_x - extra_offset_x * 2.8, offset_y - 40), 'text_color': (0, 0, 255), 'line_color': (255, 255, 255), 'specific_point_index': 3, 'connect_to_last_letter': True},
    'top_lip': {'label': 'Mouth', 'offset': (offset_x - extra_offset_x * 2.8, offset_y), 'text_color': (255, 0, 0), 'line_color': (255, 255, 255), 'specific_point_index': 11, 'connect_to_last_letter': True},
}

# Draw facial landmarks and labels
for face_landmarks in face_landmarks_list:
    for facial_feature, points in face_landmarks.items():
        for point in points:
            cv2.circle(image_bgr, point, 2, (0, 0, 255), -1)

        # Draw labels and lines for specific features
        if facial_feature in feature_settings:
            settings = feature_settings[facial_feature]
            draw_label(image_bgr, settings['label'], points, 
                       settings['offset'], settings['text_color'], settings['line_color'], 
                       settings['specific_point_index'], settings['connect_to_last_letter'])

# Resize the image based on a percentage
resize_percent = 75  # Percentage of the original size
width = int(image_bgr.shape[1] * resize_percent / 100)
height = int(image_bgr.shape[0] * resize_percent / 100)
resized_image = cv2.resize(image_bgr, (width, height))

# Save the image with landmarks and labels in the same format as the input image
output_dir = os.path.join("photos", "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
image_name_without_ext = os.path.splitext(image_name)[0]
output_path = os.path.join(output_dir, image_name_without_ext + "_with_landmarks.jpg")
cv2.imwrite(output_path, resized_image)

# Show the image with landmarks
cv2.imshow('Facial Landmarks', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
