from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from DataSetRead import *
from Cac_Landmarks import landmarks_calculator
from Classifier_PyTorch import *
from Classifier_Father import *
import sqlite3
import json
from res18_check import *
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

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


    father_features = extract_features_from_image(father_Img_path)
    print(father_features.landmarks)
    mother_features = extract_features_from_image(mother_Img_path)
    parents_Features = [father_features , mother_features]
    features_after_cac = landmarks_calculator(parents_Features)
    parents_Features_fin = [*features_after_cac[0].ratio_features, *features_after_cac[0].angle_features, *features_after_cac[0].color_features, *features_after_cac[1].ratio_features, *features_after_cac[1].angle_features, *features_after_cac[1].color_features]
    predicted_child_attributes = predict_child_with_keras_model(parents_Features_fin)
    #X = cac_parents_landmarks(parents_Features)
    """model = ChildFaceFeaturesNetModified()
    model.load_state_dict(torch.load('C://kobbi//endProject//py_torch_model//model.pth'))
    model.eval()
    print("X  : " ,len(X))
    predicted_child_attributes = model(torch.tensor(X, dtype=torch.float32).unsqueeze(0)).detach().numpy()"""
    print("predicted child : " , len(predicted_child_attributes))
    closest_children_CNN, closest_children_embedding_father, closest_children_embedding_mother, closest_children_resnet_father, closest_children_resnet_mother = find_N_closest_child(predicted_child_attributes, face_embeddings_father, face_embeddings_mother, feature_resnet_father, feature_resnet_mother)
    children_lists = [
    closest_children_CNN,
    closest_children_embedding_father,
    closest_children_embedding_mother,
    closest_children_resnet_father,
    closest_children_resnet_mother
    
    ]
    
    top_N_best_matches = find_best_match(children_lists , 10)
    print(top_N_best_matches)
    print(closest_children_CNN)
    return top_N_best_matches
    #print("The 5 closest children from CNN are:")
    #for child in closest_children_CNN:
   #     print(child)
    






def cac_parents_landmarks(features):
   
   features_after_cac = landmarks_calculator(features)
   X = [*features_after_cac[0].ratio_features, *features_after_cac[0].angle_features, *features_after_cac[0].color_features, *features_after_cac[1].ratio_features, *features_after_cac[1].angle_features, *features_after_cac[1].color_features]
   return X

def predict_child_with_keras_model(X):
    # Load the Keras model from disk
    model = load_model('C://kobbi//endProject//tensorflow_model//model.h5')

    # Load the scaler fitted on the training data
    scaler_father = joblib.load('C://kobbi//endProject//tensorflow_model//scaler_father.pkl')
    scaler_mother = joblib.load('C://kobbi//endProject//tensorflow_model//scaler_mother.pkl')

    X = np.array(X)  # Convert list to numpy array
    X = X.reshape(-1, 34)  # Reshape the 1D array into a 2D array if necessary

    # Preprocess the data in the same way as for training
    X_father = scaler_father.transform(X[:,:17])  # First half of the features are father's
    X_mother = scaler_mother.transform(X[:,17:])  # Second half of the features are mother's

    # Make predictions on the new data
    predicted_child_attributes = model.predict([X_father, X_mother])
    
    print("X  : " ,len(X))
    print("predicted child : " , len(predicted_child_attributes))
    return predicted_child_attributes
   
def cac_father_landmarks(features):
    features_after_cac = landmarks_calculator(features)

    X = [*features_after_cac[0].ratio_features, *features_after_cac[0].angle_features, *features_after_cac[0].color_features]
    return X

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
       # face_features = FaceFeatures(landmarks, face_embeddings, hair_color, skin_color, label, image, image_name, family_type, family_number, member_type, belongs_to_set)
       # return face_features
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
       # Flatten predicted_child_attributes to 1D
        predicted_child_attributes = predicted_child_attributes.flatten()

        # Calculate cosine similarity
        #similarity_attributes = np.dot(predicted_child_attributes, child_attributes) / (np.linalg.norm(predicted_child_attributes) * np.linalg.norm(child_attributes))
        similarity_attributes = cosine_similarity(child_attributes,predicted_child_attributes)
        closest_children_CNN.append((image_full_name, similarity_attributes))


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

def find_best_match(children_lists , N):
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
    return sorted_children[:N]


"""class FaceFeatures:
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
        self.belongs_to_set = belongs_to_set"""