import os
import cv2
from Find_Child import *
from Cac_Landmarks import *
def extract_son_daughter_attributes(directory_path):
    son_daughter_attributes = []

    # Iterate through the files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file is a son or daughter image
        if filename.endswith("S.jpg") or filename.endswith("D.jpg"):
            # Read the image
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            print("im here")
            # Extract the attributes using the provided function
            attributes = extract_attributes(image, filename)

            # Append the attributes to the son_daughter_attributes list
            if attributes is not None:
                son_daughter_attributes.append(attributes)

    return son_daughter_attributes

def cal_son_daughter_landmarks_and_save(features):
    with open("C:\\kobbi\\endProject\\TSKinFace_Data\\All_Pairs_SD\\pairs_childrens.txt", 'w') as file:
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
            eye_mouth_ratio = eye_distance / mouth_width
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
            ratio_features[0].append(eye_mouth_ratio)
            
            feature.ratio_features = [face_ratio, nose_ratio, mouth_middle_ratio, mouth_nose_ratio, eye_mouth_ratio]
            feature.angle_features = [angle_between_nose_mouth, angle_nose_inner_eye_corners, angle_right_eye_right_corner, angle_left_eye_right_corner]
             # Write the features to the text file
            file.write(f"Child: {feature.image_name}\n")
            file.write("Ratio Features:\n")
            file.write(f"face_ratio: {face_ratio}\n")
            file.write(f"nose_ratio: {nose_ratio}\n")
            file.write(f"mouth_middle_ratio: {mouth_middle_ratio}\n")
            file.write(f"mouth_nose_ratio: {mouth_nose_ratio}\n")
            file.write(f"eye_mouth_ratio: {eye_mouth_ratio}\n")

            file.write("\nAngle Features:\n")
            file.write(f"angle_between_nose_mouth: {angle_between_nose_mouth}\n")
            file.write(f"angle_nose_inner_eye_corners: {angle_nose_inner_eye_corners}\n")
            file.write(f"angle_right_eye_right_corner: {angle_right_eye_right_corner}\n")
            file.write(f"angle_left_eye_right_corner: {angle_left_eye_right_corner}\n")
            file.write("\n-----------------------------------------\n\n")


