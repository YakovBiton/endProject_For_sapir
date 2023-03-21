import cv2
import math
import numpy as np
# Define your line drawing function
def draw_line_direct(image, start_point, end_point, color, thickness):
    cv2.line(image, start_point, end_point, color, thickness)

def draw_line(image, arr_of_dots, color, thickness):
    for i in range(len(arr_of_dots) - 1):
        start_point = arr_of_dots[i]
        end_point = arr_of_dots[i + 1]
        cv2.line(image, start_point, end_point, color, thickness)


# Define your landmarks calculator function
def landmarks_calculator(features):
    X = []
    y = []
    for feature in features:
        ratio_features = [[]]
        landmarks_coordinates = feature.landmarks
        nose_length = nose_length_calculator(landmarks_coordinates[0][27:31])
        nose_width = nose_wide_calculator(landmarks_coordinates[0][31:36])
        nose_ratio = nose_length / nose_width
        #mouth 
        left_to_mouth = line_length_calculator([get_midpoint(landmarks_coordinates[0][3], landmarks_coordinates[0][4]) ,  landmarks_coordinates[0][48]])
        right_to_mouth = line_length_calculator([get_midpoint(landmarks_coordinates[0][12], landmarks_coordinates[0][13]) ,  landmarks_coordinates[0][54]])
        mouth_middle_ratio = left_to_mouth / right_to_mouth
        print("Nose length:", nose_length)
        print("Nose wide:", nose_width)
        ratio_features[0].append(nose_ratio) # Append nose_ratio to the sub-list
        ratio_features[0].append(mouth_middle_ratio) # Append mouth_middle_ratio to the sub-list
        feature.ratio_features = ratio_features
        fm , number , sex = feature.label.split('-')[0 : 3]
        print(fm , number , sex)
        if fm == 'FMD' and sex == 'M.jpg' :
            X.append([nose_ratio, mouth_middle_ratio])
        if fm == 'FMD' and sex == 'D.jpg' :
            y.append([nose_ratio, mouth_middle_ratio])
        
       # X.append([nose_ratio, mouth_middle_ratio])
        # y.append(int(feature.label.split('-')[0]))
        # Draw the line on the image
        image = feature.image
        draw_line(image, landmarks_coordinates[0][27:31], (0, 255, 0), 2)
        draw_line(image, landmarks_coordinates[0][31:36], (0, 0, 255), 2)
        draw_line_direct(image, get_midpoint(landmarks_coordinates[0][12], landmarks_coordinates[0][13]),landmarks_coordinates[0][54], (255,0,0), 2)
        draw_line_direct(image, get_midpoint(landmarks_coordinates[0][3], landmarks_coordinates[0][4]),landmarks_coordinates[0][48], (255,0,0), 2)
        # show lines on the face
        # cv2.imshow("Image with Line", image)
        # cv2.waitKey(0)
    X = np.array(X)
    y = np.array(y)

    return features , X , y   

# Define your line length calculator function
def line_length_calculator(arr_of_dots):
    total_distance = 0.0
    for i in range(len(arr_of_dots) - 1):
        x1, y1 = arr_of_dots[i]
        x2, y2 = arr_of_dots[i + 1]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_distance += distance
    return total_distance

# nose length calculator function
def nose_length_calculator(arr_of_dots):
    return line_length_calculator(arr_of_dots)
# nose wide calculator function
def nose_wide_calculator(arr_of_dots):
    return line_length_calculator(arr_of_dots)

def get_midpoint(point1, point2):
    x = int((point1[0] + point2[0]) / 2)
    y = int((point1[1] + point2[1]) / 2)
    return (x, y)