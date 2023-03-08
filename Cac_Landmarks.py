import cv2
import math

# Define your line drawing function
def draw_line(image, start_point, end_point, color, thickness):
    cv2.line(image, start_point, end_point, color, thickness)

# Define your landmarks calculator function
def landmarks_calculator(features):
    for feature in features:
        landmarks_coordinates = feature.landmarks
        nose_length = nose_length_calculator(landmarks_coordinates[0][28:32])
        print("check", nose_length)
        # Draw the line on the image
        image = feature.image
        draw_line(image, landmarks_coordinates[0][28], landmarks_coordinates[0][31], (0, 255, 0), 2)
        cv2.imshow("Image with Line", image)
        cv2.waitKey(0)

# Define your line length calculator function
def line_length_calculator(arr_of_dots):
    total_distance = 0.0
    for i in range(len(arr_of_dots) - 1):
        x1, y1 = arr_of_dots[i]
        x2, y2 = arr_of_dots[i + 1]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_distance += distance
    return total_distance

# Define your nose length calculator function
def nose_length_calculator(arr_of_dots):
    return line_length_calculator(arr_of_dots)

