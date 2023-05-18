from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from DataSetRead import *
from Cac_Landmarks import *
from Classifier_PyTorch import *
import sqlite3
import json
from res18_check import *

def find_child(father_Img_path,mother_Img_path):
    
    print("Father image path:", father_Img_path)
    print("Mother image path:", mother_Img_path)

    father_Img = cv2.imread(father_Img_path)
    mother_Img = cv2.imread(mother_Img_path)

    # Check if the images were loaded successfully
    if father_Img is None:
        print("Failed to load father image:", father_Img_path)
        return
    if mother_Img is None:
        print("Failed to load mother image:", mother_Img_path)
        return
    face_embeddings_father = extract_embedding(father_Img_path)
    face_embeddings_mother = extract_embedding(mother_Img_path)
    
    feature_resnet_father = extract_features_resnet(father_Img_path, model_resnet)
    feature_resnet_mother = extract_features_resnet(mother_Img_path, model_resnet)
    father_Img_name = os.path.basename(father_Img_path)
    mother_Img_name = os.path.basename(mother_Img_path)


    father_features = extract_attributes(father_Img , father_Img_name)
    mother_features = extract_attributes(mother_Img , mother_Img_name)
    parents_Features = [father_features , mother_features]
    X = cac_parents_landmarks(parents_Features)
    
    model = ChildFaceFeaturesNet()
    model.load_state_dict(torch.load('C://kobbi//endProject//py_torch_model//model.pth'))
    model.eval()
    print("X  : " ,len(X))
    predicted_child_attributes = model(torch.tensor(X, dtype=torch.float32).unsqueeze(0)).detach().numpy()
    print("predicted child : " , len(predicted_child_attributes))
    closest_children_CNN, closest_children_embedding_father, closest_children_embedding_mother, closest_children_resnet_father, closest_children_resnet_mother = find_N_closest_child(predicted_child_attributes[0], face_embeddings_father, face_embeddings_mother, feature_resnet_father, feature_resnet_mother)
    children_lists = [
    closest_children_embedding_father,
    closest_children_embedding_mother,
    closest_children_resnet_father,
    closest_children_resnet_mother,
    ]

    top_10_best_matches = find_best_match(children_lists)
    print(top_10_best_matches)

    #print("The 5 closest children from CNN are:")
    #for child in closest_children_CNN:
   #     print(child)
    






def cac_parents_landmarks(features):
   
   features_after_cac = landmarks_calculator(features)
   X = [*features_after_cac[0].ratio_features, *features_after_cac[0].angle_features, *features_after_cac[0].color_features, *features_after_cac[1].ratio_features, *features_after_cac[1].angle_features, *features_after_cac[1].color_features]
   return X
   """ 
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
    X.append([*features[0].ratio_features, *features[0].angle_features,*features[1].ratio_features,*features[1].angle_features]) """
   


def extract_attributes(image, file_name):
    label = makeLabel(file_name)
    family_type, family_number, member_type = label.split('-')[0:3]
    image_resize = resizeImage(image)
    image_name = os.path.basename(file_name)
    landmarks = np.array(extract_landmarks(image))

    if landmarks.shape != (0,):
        hair_color, skin_color = extract_hair_and_skin_color(image, landmarks,image_name)
        face_embeddings = face_recognition.face_encodings(image)
        dominant_eye_color, eye_palette = extract_eye_color(image, landmarks , "output_eye_image.jpg")
        belongs_to_set = '$'
        face_features = FaceFeatures(landmarks, face_embeddings, hair_color, skin_color, label, image, image_name, family_type, family_number, member_type, belongs_to_set)
        return face_features
    else:
        print("Failed to extract landmarks")
        return None
    

def find_closest_children(predicted_child_attributes, file_path="C:\\kobbi\\endProject\\TSKinFace_Data\\All_Pairs_SD\\pairs_childrens.txt", top_k=5):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    children = []
    i = 0
    while i < len(lines):
        if "Child:" in lines[i]:
            child_name = lines[i].strip().split(': ')[1]
            ratio_features = [float(x.split(': ')[1]) for x in lines[i + 2:i + 7]]
            angle_features = [float(x.split(': ')[1]) for x in lines[i + 9:i + 13]]
            child_attributes = np.concatenate((ratio_features, angle_features))
            children.append((child_name, child_attributes))
            i += 15
        else:
            i += 1

    # Calculate similarity between predicted_child_attributes and each child
    similarities = []
    for child_name, child_attributes in children:
        similarity = np.dot(predicted_child_attributes, child_attributes) / (np.linalg.norm(predicted_child_attributes) * np.linalg.norm(child_attributes))
        similarities.append((child_name, similarity))

    # Sort the children based on similarity and return the top_k closest children
    closest_children = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    return [child[0] for child in closest_children]  # Return only the names of the top_k closest children

def find_N_closest_child(predicted_child_attributes, face_embeddings_father, face_embeddings_mother, feature_resnet_father, feature_resnet_mother, N=5):
    # Connect to the SQLite database
    conn = sqlite3.connect("C:\\kobbi\\endProject\\TSKinFace_Data\\image_data.db")
    cursor = conn.cursor()
    # Define the query to fetch landmarks for fathers and mothers from FMS families
    query = """
    SELECT image_full_name, info
    FROM image_data
    WHERE image_full_name LIKE 'FMSD_%_S.%' OR image_full_name LIKE 'FMSD_%_D.%'
    """
    # Execute the query and fetch the results
    cursor.execute(query)
    results = cursor.fetchall()
    # Set the print options to display all array elements
    np.set_printoptions(threshold=5000)

    # Initialize lists to store the closest children
    closest_children_CNN = []
    closest_children_embedding_father = []
    closest_children_embedding_mother = []
    closest_children_resnet_father = []
    closest_children_resnet_mother = []

    for result in results:
        image_full_name, info_str = result
        info = json.loads(info_str)  # Parse the info string into a dictionary
        embedding = info['face_embeddings']
        resnet = info['feature_resnet']
        ratio_features = info['ratio_features']
        angle_features = info['angle_features']
        color_features = info['color_features']
        child_attributes = [*ratio_features, *angle_features, *color_features]
        print(predicted_child_attributes)
        print(child_attributes)
        #similarity_attributes = np.dot(predicted_child_attributes, child_attributes) / (
       #     np.linalg.norm(predicted_child_attributes) * np.linalg.norm(child_attributes)
       # )

       # closest_children_CNN.append((image_full_name, similarity_attributes))

        similarity_embedding_father = cosine_similarity_matrix(embedding, face_embeddings_father)
        similarity_embedding_mother = cosine_similarity_matrix(embedding, face_embeddings_mother)

        closest_children_embedding_father.append((image_full_name, similarity_embedding_father))
        closest_children_embedding_mother.append((image_full_name, similarity_embedding_mother))

        similarity_resnet_father = cosine_similarity(resnet, feature_resnet_father)
        similarity_resnet_mother = cosine_similarity(resnet, feature_resnet_mother)

        closest_children_resnet_father.append((image_full_name, similarity_resnet_father))
        closest_children_resnet_mother.append((image_full_name, similarity_resnet_mother))

    # Sort the lists based on the similarity values
    closest_children_CNN.sort(key=lambda x: x[1], reverse=True)
    closest_children_embedding_father.sort(key=lambda x: x[1], reverse=True)
    closest_children_embedding_mother.sort(key=lambda x: x[1], reverse=True)
    closest_children_resnet_father.sort(key=lambda x: x[1], reverse=True)
    closest_children_resnet_mother.sort(key=lambda x: x[1], reverse=True)

    # Get the top N closest children
    closest_children_CNN = closest_children_CNN[:N]
    closest_children_embedding_father = closest_children_embedding_father[:N]
    closest_children_embedding_mother = closest_children_embedding_mother[:N]
    closest_children_resnet_father = closest_children_resnet_father[:N]
    closest_children_resnet_mother = closest_children_resnet_mother[:N]

    # Close the database connection
    conn.close()

    return closest_children_CNN, closest_children_embedding_father, closest_children_embedding_mother, closest_children_resnet_father, closest_children_resnet_mother

def find_best_match(children_lists):
    child_counter = {}

    for child_list in children_lists:
        for child, _ in child_list:
            if child in child_counter:
                child_counter[child] += 1
            else:
                child_counter[child] = 1

    # Sort children by their count (number of lists they appear in)
    sorted_children = sorted(child_counter.items(), key=lambda x: x[1], reverse=True)

    # Return the top 10 children
    return sorted_children[:10]


class FaceFeatures:
    def __init__(self, landmarks, face_embeddings, hair_color, skin_color, label, image, image_name, family_type, family_number, member_type, belongs_to_set):
        self.landmarks = landmarks
        self.face_embeddings = face_embeddings
        self.hair_color = hair_color
        self.skin_color = skin_color
        self.label = label
        self.image = image
        self.image_name = image_name
        self.family_type = family_type
        self.family_number = family_number
        self.member_type = member_type
        self.belongs_to_set = belongs_to_set