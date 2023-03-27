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
        feature.ratio_features = [nose_ratio , mouth_middle_ratio]


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
    for feature in features:
        fm , number , sex = feature.label.split('-')[0 : 3]
        print(fm , number , sex)
        if feature.belongs_to_set == '$':
            if fm == 'FMD' and sex == 'F.jpg':
                for f in features:
                    if f.family_type == 'FMD' and f.family_number == number and f.member_type == 'M.jpg':
                        print('im here')
                        for f1 in features:
                            if f1.family_type == 'FMD' and f1.family_number == number and f1.member_type == 'D.jpg':
                                feature.belongs_to_set = 'x'
                                f.belongs_to_set = 'x'
                                f1.belongs_to_set = 'y'
                                X.append([feature.ratio_features[0], feature.ratio_features[1], f.ratio_features[0], f.ratio_features[1]])
                                y.append([f1.ratio_features[0], f1.ratio_features[1]])
                                print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set, " and ", f1.label, " belongs to ", f1.belongs_to_set)  
            elif fm == 'FMD' and sex == 'M.jpg':
                for f in features:
                    if f.family_type == 'FMD' and f.family_number == number and f.member_type == 'F.jpg':
                        print('im here')
                        for f1 in features:
                            if f1.family_type == 'FMD' and f1.family_number == number and f1.member_type == 'D.jpg':
                                feature.belongs_to_set = 'x'
                                f.belongs_to_set = 'x'
                                f1.belongs_to_set = 'y'
                                X.append([f.ratio_features[0], f.ratio_features[1], feature.ratio_features[0], feature.ratio_features[1]])
                                y.append([f1.ratio_features[0], f1.ratio_features[1]])
                                print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set, " and ", f1.label, " belongs to ", f1.belongs_to_set)                  
            elif fm == 'FMD' and sex == 'D.jpg':
                for f in features:
                    if f.family_type == 'FMD' and f.family_number == number and f.member_type == 'F.jpg':
                        print('im here')
                        for f1 in features:
                            if f1.family_type == 'FMD' and f1.family_number == number and f1.member_type == 'M.jpg':
                                feature.belongs_to_set = 'y'
                                f.belongs_to_set = 'x'
                                f1.belongs_to_set = 'x'
                                X.append([f.ratio_features[0], f.ratio_features[1], f1.ratio_features[0], f1.ratio_features[1]])
                                y.append([feature.ratio_features[0], feature.ratio_features[1]])
                                print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set, " and ", f1.label, " belongs to ", f1.belongs_to_set)
            elif fm == 'FMS' and sex == 'F.jpg':
                for f in features:
                    if f.family_type == 'FMS' and f.family_number == number and f.member_type == 'M.jpg':
                        print('im here')
                        for f1 in features:
                            if f1.family_type == 'FMS' and f1.family_number == number and f1.member_type == 'S.jpg':
                                feature.belongs_to_set = 'x'
                                f.belongs_to_set = 'x'
                                f1.belongs_to_set = 'y'
                                X.append([feature.ratio_features[0], feature.ratio_features[1], f.ratio_features[0], f.ratio_features[1]])
                                y.append([f1.ratio_features[0], f1.ratio_features[1]])
                                print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set, " and ", f1.label, " belongs to ", f1.belongs_to_set)
            elif fm == 'FMS' and sex == 'M.jpg':
                for f in features:
                    if f.family_type == 'FMS' and f.family_number == number and f.member_type == 'F.jpg':
                        print('im here')
                        for f1 in features:
                            if f1.family_type == 'FMS' and f1.family_number == number and f1.member_type == 'S.jpg':
                                feature.belongs_to_set = 'x'
                                f.belongs_to_set = 'x'
                                f1.belongs_to_set = 'y'
                                X.append([f.ratio_features[0], f.ratio_features[1], feature.ratio_features[0], feature.ratio_features[1]])
                                y.append([f1.ratio_features[0], f1.ratio_features[1]])
                                print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set, " and ", f1.label, " belongs to ", f1.belongs_to_set)
            elif fm == 'FMS' and sex == 'S.jpg':
                for f in features:
                    if f.family_type == 'FMS' and f.family_number == number and f.member_type == 'F.jpg':
                        print('im here')
                        for f1 in features:
                            if f1.family_type == 'FMS' and f1.family_number == number and f1.member_type == 'M.jpg':
                                feature.belongs_to_set = 'y'
                                f.belongs_to_set = 'x'
                                f1.belongs_to_set = 'x'
                                X.append([f.ratio_features[0], f.ratio_features[1], f1.ratio_features[0], f1.ratio_features[1]])
                                y.append([feature.ratio_features[0], feature.ratio_features[1]])
                                print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set, " and ", f1.label, " belongs to ", f1.belongs_to_set)
        elif fm == 'FMSD' and sex == 'F.jpg':
                for f in features:
                    if f.family_type == 'FMSD' and f.family_number == number and f.member_type == 'M.jpg':
                        print('im here')
                        for f1 in features:
                            if f1.family_type == 'FMSD' and f1.family_number == number and f1.member_type == 'D.jpg':
                                feature.belongs_to_set = 'x'
                                f.belongs_to_set = 'x'
                                f1.belongs_to_set = 'y'
                                X.append([feature.ratio_features[0], feature.ratio_features[1], f.ratio_features[0], f.ratio_features[1]])
                                y.append([f1.ratio_features[0], f1.ratio_features[1]])
                                print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set, " and ", f1.label, " belongs to ", f1.belongs_to_set)  
        elif fm == 'FMSD' and sex == 'M.jpg':
                for f in features:
                    if f.family_type == 'FMSD' and f.family_number == number and f.member_type == 'F.jpg':
                        print('im here')
                        for f1 in features:
                            if f1.family_type == 'FMSD' and f1.family_number == number and f1.member_type == 'D.jpg':
                                feature.belongs_to_set = 'x'
                                f.belongs_to_set = 'x'
                                f1.belongs_to_set = 'y'
                                X.append([f.ratio_features[0], f.ratio_features[1], feature.ratio_features[0], feature.ratio_features[1]])
                                y.append([f1.ratio_features[0], f1.ratio_features[1]])
                                print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set, " and ", f1.label, " belongs to ", f1.belongs_to_set) 
        elif fm == 'FMSD' and sex == 'D.jpg':
                for f in features:
                    if f.family_type == 'FMSD' and f.family_number == number and f.member_type == 'F.jpg':
                        print('im here')
                        for f1 in features:
                            if f1.family_type == 'FMSD' and f1.family_number == number and f1.member_type == 'M.jpg':
                                feature.belongs_to_set = 'y'
                                f.belongs_to_set = 'x'
                                f1.belongs_to_set = 'x'
                                X.append([f.ratio_features[0], f.ratio_features[1], f1.ratio_features[0], f1.ratio_features[1]])
                                y.append([feature.ratio_features[0], feature.ratio_features[1]])
                                print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set, " and ", f1.label, " belongs to ", f1.belongs_to_set)  
        elif fm == 'FMSD' and sex == 'S.jpg':
                for f in features:
                    if f.family_type == 'FMSD' and f.family_number == number and f.member_type == 'F.jpg':
                        print('im here')
                        for f1 in features:
                            if f1.family_type == 'FMSD' and f1.family_number == number and f1.member_type == 'M.jpg':
                                feature.belongs_to_set = 'y'
                                f.belongs_to_set = 'x'
                                f1.belongs_to_set = 'x'
                                X.append([f.ratio_features[0], f.ratio_features[1], f1.ratio_features[0], f1.ratio_features[1]])
                                y.append([feature.ratio_features[0], feature.ratio_features[1]])
                                print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set, " and ", f1.label, " belongs to ", f1.belongs_to_set)                                                                
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