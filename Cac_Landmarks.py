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
        angle_features = [[]]
        landmarks_coordinates = feature.landmarks
    # Ratio-based features
        nose_length = nose_length_calculator(landmarks_coordinates[0][27:31])
        nose_width_long = nose_wide_calculator(landmarks_coordinates[0][31:36])
        nose_ratio = nose_length / nose_width_long
    # Face width and height ratio
        face_width = euclidean_distance(landmarks_coordinates[0][0], landmarks_coordinates[0][16])
        face_height = euclidean_distance(landmarks_coordinates[0][27], landmarks_coordinates[0][8])
        face_ratio = face_width / face_height
    #mouth 
        left_to_mouth = line_length_calculator([get_midpoint(landmarks_coordinates[0][3], landmarks_coordinates[0][4]), landmarks_coordinates[0][48]])
        right_to_mouth = line_length_calculator([get_midpoint(landmarks_coordinates[0][12], landmarks_coordinates[0][13]), landmarks_coordinates[0][54]])
        mouth_middle_ratio = left_to_mouth / right_to_mouth
        mouth_width = euclidean_distance(landmarks_coordinates[0][48], landmarks_coordinates[0][54])
        nose_width_short = euclidean_distance(landmarks_coordinates[0][31], landmarks_coordinates[0][35])
        mouth_nose_ratio = mouth_width / nose_width_short
        eye_distance = euclidean_distance(landmarks_coordinates[0][39], landmarks_coordinates[0][42])
       # eye_mouth_ratio = eye_distance / mouth_width
    # Calculate the angle between nose and mouth lines
        nose_line_start = landmarks_coordinates[0][27]
        nose_line_end = landmarks_coordinates[0][30]
        mouth_line_start = get_midpoint(landmarks_coordinates[0][3], landmarks_coordinates[0][4])
        mouth_line_end = landmarks_coordinates[0][48]
        angle_between_nose_mouth = angle_between_lines(nose_line_start, nose_line_end, mouth_line_start, mouth_line_end)
        angle_features[0].append(angle_between_nose_mouth)
    
        # Angle-based features
        angle_nose_inner_eye_corners = angle_between_points(landmarks_coordinates[0][39], landmarks_coordinates[0][30], landmarks_coordinates[0][42])
        angle_features[0].append(angle_nose_inner_eye_corners)

        angle_right_eye_right_corner = angle_between_points(landmarks_coordinates[0][37], landmarks_coordinates[0][36], landmarks_coordinates[0][41])
        angle_features[0].append(angle_right_eye_right_corner)

        angle_left_eye_right_corner = angle_between_points(landmarks_coordinates[0][43], landmarks_coordinates[0][42], landmarks_coordinates[0][47])
        angle_features[0].append(angle_left_eye_right_corner)


        ratio_features[0].append(face_ratio)
        ratio_features[0].append(nose_ratio) # Append nose_ratio to the sub-list
        ratio_features[0].append(mouth_middle_ratio) # Append mouth_middle_ratio to the sub-list
        ratio_features[0].append(mouth_nose_ratio)
        
        
        skin_color_Red = feature.skin_color[0]
        skin_color_Green = feature.skin_color[1]
        skin_color_Blue = feature.skin_color[2]
        face_embeddings_array = np.array(feature.face_embeddings)
        feature.ratio_features = [face_ratio, nose_ratio, mouth_middle_ratio, mouth_nose_ratio]
        feature.angle_features = [angle_between_nose_mouth, angle_nose_inner_eye_corners, angle_right_eye_right_corner, angle_left_eye_right_corner]
        feature.color_features = [skin_color_Red , skin_color_Green , skin_color_Blue]
        
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
    
    return features

def set_X_y(features):
    X = []
    y = []
    for feature in features:
        fm , number , sex = feature.label.split('-')[0 : 3]
        if feature.belongs_to_set == '$':
            if fm == 'FMD' and sex == 'F.jpg':
                for f in features:
                    if f.family_type == 'FMD' and f.family_number == number and f.member_type == 'M.jpg':
                        for f1 in features:
                            if f1.family_type == 'FMD' and f1.family_number == number and f1.member_type == 'D.jpg':
                                feature.belongs_to_set = 'x'
                                f.belongs_to_set = 'x'
                                f1.belongs_to_set = 'y'
                                X.append([*feature.ratio_features, *feature.angle_features, *feature.color_features, *f.ratio_features, *f.angle_features, *f.color_features])
                                y.append([*f1.ratio_features, *f1.angle_features, *f1.color_features])
                                print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set, " and ", f1.label, " belongs to ", f1.belongs_to_set)  
            elif fm == 'FMD' and sex == 'M.jpg':
                for f in features:
                    if f.family_type == 'FMD' and f.family_number == number and f.member_type == 'F.jpg':
                        for f1 in features:
                            if f1.family_type == 'FMD' and f1.family_number == number and f1.member_type == 'D.jpg':
                                feature.belongs_to_set = 'x'
                                f.belongs_to_set = 'x'
                                f1.belongs_to_set = 'y'
                                X.append([*f.ratio_features, *f.angle_features, *f.color_features, *feature.ratio_features, *feature.angle_features, *feature.color_features])
                                y.append([*f1.ratio_features, *f1.angle_features, *f1.color_features])
                                print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set, " and ", f1.label, " belongs to ", f1.belongs_to_set)                  
            elif fm == 'FMD' and sex == 'D.jpg':
                for f in features:
                    if f.family_type == 'FMD' and f.family_number == number and f.member_type == 'F.jpg':
                        for f1 in features:
                            if f1.family_type == 'FMD' and f1.family_number == number and f1.member_type == 'M.jpg':
                                feature.belongs_to_set = 'y'
                                f.belongs_to_set = 'x'
                                f1.belongs_to_set = 'x'
                                X.append([*f.ratio_features, *f.angle_features, *f.color_features, *f1.ratio_features, *f1.angle_features, *f1.color_features])
                                y.append([*feature.ratio_features, *feature.angle_features, *feature.color_features])
                                print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set, " and ", f1.label, " belongs to ", f1.belongs_to_set)
            elif fm == 'FMS' and sex == 'F.jpg':
                for f in features:
                    if f.family_type == 'FMS' and f.family_number == number and f.member_type == 'M.jpg':
                        for f1 in features:
                            if f1.family_type == 'FMS' and f1.family_number == number and f1.member_type == 'S.jpg':
                                feature.belongs_to_set = 'x'
                                f.belongs_to_set = 'x'
                                f1.belongs_to_set = 'y'
                                X.append([*feature.ratio_features, *feature.angle_features, *feature.color_features, *f.ratio_features, *f.angle_features, *f.color_features])
                                y.append([*f1.ratio_features, *f1.angle_features, *f1.color_features])
                                print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set, " and ", f1.label, " belongs to ", f1.belongs_to_set)
            elif fm == 'FMS' and sex == 'M.jpg':
                for f in features:
                    if f.family_type == 'FMS' and f.family_number == number and f.member_type == 'F.jpg':
                        for f1 in features:
                            if f1.family_type == 'FMS' and f1.family_number == number and f1.member_type == 'S.jpg':
                                feature.belongs_to_set = 'x'
                                f.belongs_to_set = 'x'
                                f1.belongs_to_set = 'y'
                                X.append([*f.ratio_features, *f.angle_features, *f.color_features, *feature.ratio_features, *feature.angle_features, *feature.color_features])
                                y.append([*f1.ratio_features, *f1.angle_features, *f1.color_features])
                                print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set, " and ", f1.label, " belongs to ", f1.belongs_to_set)
            elif fm == 'FMS' and sex == 'S.jpg':
                for f in features:
                    if f.family_type == 'FMS' and f.family_number == number and f.member_type == 'F.jpg':
                        for f1 in features:
                            if f1.family_type == 'FMS' and f1.family_number == number and f1.member_type == 'M.jpg':
                                feature.belongs_to_set = 'y'
                                f.belongs_to_set = 'x'
                                f1.belongs_to_set = 'x'
                                X.append([*f.ratio_features, *f.angle_features, *f.color_features, *f1.ratio_features, *f1.angle_features, *f1.color_features])
                                y.append([*feature.ratio_features, *feature.angle_features, *feature.color_features])
                                print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set, " and ", f1.label, " belongs to ", f1.belongs_to_set)
            elif fm == 'FMSD' and sex == 'F.jpg':
                for f in features:
                    if f.family_type == 'FMSD' and f.family_number == number and f.member_type == 'M.jpg':
                        for f1 in features:
                            if f1.family_type == 'FMSD' and f1.family_number == number and f1.member_type == 'D.jpg':
                                feature.belongs_to_set = 'x'
                                f.belongs_to_set = 'x'
                                f1.belongs_to_set = 'y'
                                X.append([*feature.ratio_features, *feature.angle_features, *feature.color_features, *f.ratio_features, *f.angle_features, *f.color_features])
                                y.append([*f1.ratio_features, *f1.angle_features, *f1.color_features])
                                print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set, " and ", f1.label, " belongs to ", f1.belongs_to_set)  
            elif fm == 'FMSD' and sex == 'M.jpg':
                for f in features:
                    if f.family_type == 'FMSD' and f.family_number == number and f.member_type == 'F.jpg':
                        for f1 in features:
                            if f1.family_type == 'FMSD' and f1.family_number == number and f1.member_type == 'D.jpg':
                                feature.belongs_to_set = 'x'
                                f.belongs_to_set = 'x'
                                f1.belongs_to_set = 'y'
                                X.append([*f.ratio_features, *f.angle_features, *f.color_features, *feature.ratio_features, *feature.angle_features, *feature.color_features])
                                y.append([*f1.ratio_features, *f1.angle_features, *f1.color_features])
                                print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set, " and ", f1.label, " belongs to ", f1.belongs_to_set) 
            elif fm == 'FMSD' and sex == 'D.jpg':
                for f in features:
                    if f.family_type == 'FMSD' and f.family_number == number and f.member_type == 'F.jpg':
                        for f1 in features:
                            if f1.family_type == 'FMSD' and f1.family_number == number and f1.member_type == 'M.jpg':
                                feature.belongs_to_set = 'y'
                                f.belongs_to_set = 'x'
                                f1.belongs_to_set = 'x'
                                X.append([*f.ratio_features, *f.angle_features, *f.color_features, *f1.ratio_features, *f1.angle_features, *f1.color_features])
                                y.append([*feature.ratio_features, *feature.angle_features, *feature.color_features])
                                print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set, " and ", f1.label, " belongs to ", f1.belongs_to_set)  
            elif fm == 'FMSD' and sex == 'S.jpg':
                for f in features:
                    if f.family_type == 'FMSD' and f.family_number == number and f.member_type == 'F.jpg':
                        for f1 in features:
                            if f1.family_type == 'FMSD' and f1.family_number == number and f1.member_type == 'M.jpg':
                                feature.belongs_to_set = 'y'
                                f.belongs_to_set = 'x'
                                f1.belongs_to_set = 'x'
                                X.append([*f.ratio_features, *f.angle_features, *f.color_features, *f1.ratio_features, *f1.angle_features, *f1.color_features])
                                y.append([*feature.ratio_features, *feature.angle_features, *feature.color_features])
                                print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set, " and ", f1.label, " belongs to ", f1.belongs_to_set)                                                                
    X = np.array(X)
    y = np.array(y)

    return features , X , y   


def set_X_y_father_classifier(features):
    X = []
    y = []
    for feature in features:
        fm , number , sex = feature.label.split('-')[0 : 3]
        if feature.belongs_to_set == '$':
            if fm == 'FMD' and sex == 'F.jpg':
                for f in features:
                    if f.family_type == 'FMD' and f.family_number == number and f.member_type == 'D.jpg':
                        feature.belongs_to_set = 'x'
                        f.belongs_to_set = 'y'
                        X.append([*feature.ratio_features, *feature.angle_features, *feature.color_features])
                        y.append([ *f.ratio_features, *f.angle_features, *f.color_features])
                        print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set)  
            elif fm == 'FMD' and sex == 'D.jpg':
                for f in features:
                    if f.family_type == 'FMD' and f.family_number == number and f.member_type == 'F.jpg':
                        feature.belongs_to_set = 'y'
                        f.belongs_to_set = 'x'
                        X.append([*f.ratio_features, *f.angle_features, *f.color_features])
                        y.append([ *feature.ratio_features, *feature.angle_features, *feature.color_features])
                        print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set)                  
            elif fm == 'FMS' and sex == 'S.jpg':
                for f in features:
                    if f.family_type == 'FMS' and f.family_number == number and f.member_type == 'F.jpg':
                        feature.belongs_to_set = 'y'
                        f.belongs_to_set = 'x'
                        X.append([*f.ratio_features, *f.angle_features, *f.color_features])
                        y.append([ *feature.ratio_features, *feature.angle_features, *feature.color_features])
                        print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set) 
            elif fm == 'FMS' and sex == 'F.jpg':
                for f in features:
                    if f.family_type == 'FMS' and f.family_number == number and f.member_type == 'S.jpg':
                        feature.belongs_to_set = 'x'
                        f.belongs_to_set = 'y'
                        X.append([*feature.ratio_features, *feature.angle_features, *feature.color_features])
                        y.append([ *f.ratio_features, *f.angle_features, *f.color_features])
                        print(feature.label + " belongs to ", feature.belongs_to_set, " and ", f.label + " belongs to ", f.belongs_to_set) 
                                                                            
    X = np.array(X)
    y = np.array(y)

    return  X , y 

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

def angle_between_lines(line1_start, line1_end, line2_start, line2_end):
    # Calculate the vectors for each line
    vec1 = np.array(line1_end) - np.array(line1_start)
    vec2 = np.array(line2_end) - np.array(line2_start)

    # Calculate the cosine of the angle between the lines
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # Convert cosine to angle in degrees
    angle = np.arccos(cos_angle) * 180 / np.pi

    return angle



def angle_between_points(a, b, c):
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    cosine_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (math.sqrt(ba[0]**2 + ba[1]**2) * math.sqrt(bc[0]**2 + bc[1]**2))
    angle = math.acos(cosine_angle)

    return math.degrees(angle)

def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

