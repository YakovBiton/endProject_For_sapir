import os
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
from deepface.commons import functions
import dlib
from PIL import Image
from collections.abc import Mapping
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import colorthief
from DomiColor import *
from keras_vggface.utils import preprocess_input
import face_recognition
from res18_check import extract_features_resnet
from res18_check import extract_embedding
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import sqlite3
import json

########################     ########################################
# handle all the extractions of the data we need from the images for the calculation and the training
########################     ########################################
#set the directory
directory = 'C:\\kobbi\\endProject\\TSKinFace_Data\\Azura_Test'
model_path = 'C:\kobbi\endProject\shape_predictor_68_face_landmarks.dat'
bad_photos_path = 'C:\\kobbi\\endProject\\TSKinFace_Data\\Azura_Test_Copy'
detector2 = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)
# Load a pre-trained ResNet18 model
model_resnet = models.resnet50(pretrained=True)

# Remove the last layer to use the model for feature extraction
model_resnet = torch.nn.Sequential(*(list(model_resnet.children())[:-1]))

count = 1
def count_plus1():
   global count
   count += 1
path_resize = "C:/kobbi/endProject/TSKinFace_Data/Azura_Test/test/resize_pics"
# This key will serve all examples in this document.
"""KEY = "3bf95fe5f0554d7f8f7bf5d877076c0c"
headers = {"Ocp-Apim-Subscription-Key": KEY}


# This endpoint will be used in all examples in this quickstart.
ENDPOINT = "https://eastus.api.cognitive.microsoft.com/face/v1.0/detect[?returnFaceId][&returnFaceLandmarks][&returnFaceAttributes][&recognitionModel][&returnRecognitionModel][&detectionModel]&subscription-key=<3bf95fe5f0554d7f8f7bf5d877076c0c>"
endpoint = "https://sapirendprojectfaceapi.cognitiveservices.azure.com"""

# Create a client for interacting with the Face API
"""credentials = CognitiveServicesCredentials(KEY)
face_client = FaceClient(ENDPOINT, credentials)"""
def extract_features(directory):
    features = []
    labels = []
    # Now call this function at the start of your feature extraction
    color_info_dict = read_eye_hair_color_info("C:\\kobbi\\endProject\\TSKinFace_Data\\image_eye_hair_color_info.txt")
    #loop through subdirectories  
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        #loop through files
        for file in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file)
            if os.path.isfile(file_path):
               
                #get the name of the file (photo)
                file_name = os.path.basename(file_path)
                label = makeLabel(file_name)
                # Extract information from filename
                family_type , family_number , member_type  = label.split('-')[0 : 3] 
                # FMD or FMS
                # number of the family
                # M, F, D, or S
                # Load the image and extract the landmarks
                image = cv2.imread(file_path)
                image_resize = resizeImage(image) # Resize the image still doesn't used
                image_name = os.path.basename(file_path)
                image_name_withoutend, _ = os.path.splitext(os.path.basename(file_path))  # This will give you the filename without the extension
                # Check if the data is in the database
                info = retrieve_from_database(image_name)
                if info is not None:
                    landmarks = info["landmarks"]
                else:
                    landmarks = np.array(extract_landmarks(image))
                if landmarks.shape != (0,):
                    hair_color_null, skin_color = extract_hair_and_skin_color(image,landmarks,image_name)
                   #dominant_eye_color,eye_palette = extract_eye_color(image, landmarks, file_name)
                    if info is not None:
                        eye_color = color_info_dict[image_name_withoutend]["eye_color"]
                    else:
                        eye_color = 0
                    
                    if info is not None:
                        hair_color = color_info_dict[image_name_withoutend]["hair_color"]
                    else:
                        hair_color = 0
                    belongs_to_set = '$'
                    if info is not None:
                        face_embeddings = info["face_embeddings"]
                    else:
                        face_embeddings = extract_embedding(file_path)
                    if info is not None:    
                        feature_resnet = info["feature_resnet"]
                    else:
                        feature_resnet = extract_features_resnet(file_path, model_resnet)
                    # With these lines:
                    ######   features_VGGFace still does not work ######
                    #features_VGGFace = extract_VGGFace_features(file_path)
                    #if features_VGGFace is None:
                    #    print("Skipping image", file_path)
                    #    continue
                    #features_VGGFace_array = np.array(features_VGGFace)
                    face_features = FaceFeatures(landmarks, face_embeddings, feature_resnet, hair_color,eye_color, skin_color, label, image, image_name, family_type, family_number, member_type, belongs_to_set)
                  #  landmarks = np.append(landmarks, [hair_color, skin_color])
                   # landmarks = np.expand_dims(landmarks, axis=-1)
                   # eye_color(image)
                   # landmarks[3] = 
                    features.append(face_features)
                    labels.append(subdir)
                else:
                    file_path_bad = os.path.join(bad_photos_path, file_name)
                    copyBefore = image.copy()
                    cv2.imwrite(file_path_bad, copyBefore)
    
    
        #for feature in features:
        #print("feature", feature.name , "68 is ", feature.landmarks[0][35][0]) 
        #print(feature.skin_color)
        #print(feature.hair_color)
       
        #np.set_printoptions(threshold=sys.maxsize)   
    return features

############################# for one image ##################################

def extract_features_from_image(file_path):
    color_info_dict = read_eye_hair_color_info("C:\\kobbi\\endProject\\TSKinFace_Data\\image_eye_hair_color_info.txt")
    file_name = os.path.basename(file_path)
    label = makeLabel(file_name)
    # Extract information from filename
    family_type , family_number , member_type  = label.split('-')[0 : 3] 
    # FMD or FMS
    # number of the family
    # M, F, D, or S
    # Load the image and extract the landmarks
    image = cv2.imread(file_path)
    image_resize = resizeImage(image) # Resize the image still doesn't used
    image_name = os.path.basename(file_path)
    # Check if the data is in the database
    info = retrieve_from_database(image_name)
    if info is not None:
        landmarks = info["landmarks"]
    else:
        landmarks = np.array(extract_landmarks(image))
    # Preprocess the landmarks
    image_name_withoutend, _ = os.path.splitext(os.path.basename(file_path))  # This will give you the filename without the extension
    # landmarks = np.array(landmarks).flatten()
    # Append the features and labels
    if landmarks.shape != (0,):
        hair_color, skin_color = extract_hair_and_skin_color(image,landmarks,image_name)
        eye_color = color_info_dict[image_name_withoutend]["eye_color"]
        hair_color = color_info_dict[image_name_withoutend]["hair_color"]
        dominant_eye_color,eye_palette = extract_eye_color(image, landmarks, file_name)
        belongs_to_set = '$'
        if info is not None:
            face_embeddings = info["face_embeddings"]
        else:
            face_embeddings = extract_embedding(file_path)
        if info is not None:    
            feature_resnet = info["feature_resnet"]
        else:
            feature_resnet = extract_features_resnet(file_path, model_resnet)
        # With these lines:
        ######   features_VGGFace still does not work ######
        #features_VGGFace = extract_VGGFace_features(file_path)
        #if features_VGGFace is None:
        #    print("Skipping image", file_path)
        #    continue
        #features_VGGFace_array = np.array(features_VGGFace)
        face_features = FaceFeatures(landmarks, face_embeddings, feature_resnet, hair_color,eye_color, skin_color, label, image, image_name, family_type, family_number, member_type, belongs_to_set)
        #  landmarks = np.append(landmarks, [hair_color, skin_color])
        # landmarks = np.expand_dims(landmarks, axis=-1)
        # eye_color(image)
        # landmarks[3] = 
        return face_features
    else:
        file_path_bad = os.path.join(bad_photos_path, file_name)
        copyBefore = image.copy()
        cv2.imwrite(file_path_bad, copyBefore)


def retrieve_from_database(image_full_name):
    # Connect to the SQLite database
    conn = sqlite3.connect("C:\\kobbi\\endProject\\TSKinFace_Data\\image_data.db")
    cursor = conn.cursor()

   # Query the database for the image data
    
    cursor.execute("""
    SELECT info FROM image_data WHERE image_full_name = ?
    """, (image_full_name,))

    result = cursor.fetchone()
    if result is None:
        print("Image did not exit in database")
        return None

    # The data is stored as a JSON string, so we need to convert it back to a Python dictionary
    info = json.loads(result[0])

    # Convert the lists back to NumPy arrays
    info["landmarks"] = np.array([[tuple(point) for point in info["landmarks"]]]) # extra dimension added
    info["face_embeddings"] = np.array(info["face_embeddings"])
    info["feature_resnet"] = np.array(info["feature_resnet"])
    info["ratio_features"] = np.array(info["ratio_features"])
    info["angle_features"] = np.array(info["angle_features"])
    info["color_features"] = np.array(info["color_features"])

    return info


def extract_features_from_dataBase():
    # Connect to the SQLite database
    conn = sqlite3.connect("C:\\kobbi\\endProject\\TSKinFace_Data\\image_data.db")
    cursor = conn.cursor()
    # Define the query to fetch landmarks for fathers and mothers from FMS families
    query = """
    SELECT image_full_name, info
    FROM image_data
    WHERE image_full_name LIKE 'FMS_%_%.%' OR image_full_name LIKE 'FMD_%_%.%'
    """
    # Execute the query and fetch the results
    cursor.execute(query)
    results = cursor.fetchall()




def extract_landmarks(image):
    # Detect the faces
    faces = detector2(image, 1)
    # If cant find face return an empty list
    if len(faces) == 0:
        return []
    # Extract the landmarks for each face
    landmarks = []
    for face in faces:
        shape = predictor(image, face)
        landmarks.append(np.array([[point.x, point.y] for point in shape.parts()]))
    copyImage = image.copy()
    for shape in landmarks:
        for point in shape:
            x, y = point
            cv2.circle(copyImage, (x, y), 2, (255, 0, 0), -1)
    # show the landmarks on the face
    #cv2.imshow("Facial Landmarks", copyImage)
    #cv2.waitKey(0)
    return landmarks

# Define a function to read the text file into a dictionary
def read_eye_hair_color_info(file_path):
    color_info_dict = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    current_key = ""
    for line in lines:
        line = line.strip()
        if line.endswith(':'):
            current_key = line[:-1]  # Remove the ":" at the end
        elif "Eyes color:" in line:
            eye_color = line.split(": ")[1]
            color_info_dict.setdefault(current_key, {})["eye_color"] = eye_color
        elif "Hair color:" in line:
            hair_color = line.split(": ")[1]
            color_info_dict.setdefault(current_key, {})["hair_color"] = hair_color
    return color_info_dict

def extract_hair_and_skin_color(image,landmarks,image_name):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    height, width, _ = image.shape # extract the image width and height
    rectangular_area_right_hair = extract_bounding_box(landmarks[0][16][0]+2 , landmarks[0][16][1]-7, width, height)
    rectangular_area_left_hair = extract_bounding_box(landmarks[0][0][0] , landmarks[0][1][1]-20, width, height)
    rectangular_area_left_skin = extract_bounding_box(landmarks[0][31][0]-6 , landmarks[0][32][1]-6, width, height)
    rectangular_area_right_skin = extract_bounding_box(landmarks[0][35][0]+6 , landmarks[0][35][1]-6, width, height)
    # Get the skin region points
    rectangular_area_left_skin_points = get_skin_region_points(rectangular_area_left_skin)
    rectangular_area_right_skin_points = get_skin_region_points(rectangular_area_right_skin)
    # Crop the skin regions
    left_skin_region = crop_skin_region(image, rectangular_area_left_skin_points)
    right_skin_region = crop_skin_region(image, rectangular_area_right_skin_points)
    # Combine the skin regions and save the image
    combined_skin_region = combine_skin_regions(left_skin_region, right_skin_region)
    output_directory = "C:/kobbi/endProject/TSKinFace_Data/Azura_Test/test/only_bounding_boxes/"
    os.makedirs(output_directory, exist_ok=True)
    output_image_path = os.path.join(output_directory, f"{os.path.splitext(image_name)[0]}_skin.png")
    cv2.imwrite(output_image_path, combined_skin_region)
    right_hair = extract_color_from_region(image,rectangular_area_right_hair)
    left_hair = extract_color_from_region(image,rectangular_area_left_hair)

    # Convert the average color of the rectangular areas to the same color space
    # print(right_skin + " and with left " + left_skin + )
    hair_mask = ((right_hair[0]+left_hair[0]) / 2,(right_hair[1]+left_hair[1]) / 2 , (right_hair[2]+left_hair[2]) /2)
    skin_color = dominant_color_cluster(combined_skin_region)
    #mean_skin = cv2.mean(image, mask=skin_mask)
    return hair_mask , skin_color

def extract_bounding_box(point_x, point_y, image_width, image_height):
    x, y = point_x, point_y
    margin = 2
    x1, y1 = max(x - margin, 0), max(y - margin, 0)
    x2, y2 = min(x + margin, image_width), min(y + margin, image_height)
    rectangular_area = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
    return rectangular_area

def get_skin_region_points(rectangular_area):
    points = []
    for point in rectangular_area:
        x, y = point[0], point[1]
        points.append((x, y))
    
    return points

def crop_skin_region(image, points):
    x_min = min([p[0] for p in points])
    x_max = max([p[0] for p in points])
    y_min = min([p[1] for p in points])
    y_max = max([p[1] for p in points])

    width = x_max - x_min
    height = y_max - y_min

    if width > height:
        y_max = y_min + width
    else:
        x_max = x_min + height

    return image[y_min:y_max, x_min:x_max]

# def extract_bounding_box(point_x,point_y) :
#     x,y = point_x,point_y
#     rectangular_area = [[x-2, y-2], [x-2, y+2], [x+2, y+2], [x+2, y-2]]
#     return rectangular_area  

def extract_color_from_region(image , rectangular_area):
    # Create a copy of the original image
    image_with_bounding_box = image.copy()
    # Draw the bounding box on the image_with_bounding_box
    cv2.fillPoly(image_with_bounding_box, [np.array(rectangular_area)], (0, 255, 0))

    # Save the image with the bounding box
    
    path = "C:/kobbi/endProject/TSKinFace_Data/Azura_Test/test/bounding_boxes/"
    path_for_only = "C:/kobbi/endProject/TSKinFace_Data/Azura_Test/test/only_bounding_boxes/"
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = "image_with_bounding_box" + str(count) + ".jpg"
    file_path = os.path.join(path, file_name)
    cv2.imwrite(file_path, image_with_bounding_box)
     # Crop the region from the original image
    x1, y1 = rectangular_area[0]
    x2, y2 = rectangular_area[2]
    cropped_image = image[y1:y2, x1:x2]
    if cropped_image is not None and cropped_image.size != 0:
        if not os.path.exists(path_for_only):
            os.makedirs(path_for_only)
        file_name2 = "only_bounding_box" + str(count) + ".jpg"
        file_path2 = os.path.join(path_for_only, file_name2)
        new_size = (128, 128)
        copyBefore = cropped_image.copy()
        image_resize = cv2.resize(copyBefore, new_size)
        cv2.imwrite(file_path2, image_resize)
        count_plus1()
        return dominant_color(copyBefore)
    # Create a black image with the same shape as the original image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # Create a polygon with the rectangular_area points
    cv2.fillPoly(mask, [np.array(rectangular_area)], 255)
    # Use the mask to extract the color
    color = cv2.mean(image, mask=mask)
    count_plus1()
    return color

def extract_eye_color(image, landmarks, image_name) :
    left_eye_points = get_eye_region(landmarks, 36, 41)
    right_eye_points = get_eye_region(landmarks, 42, 47)

    # Draw bounding boxes around the eye regions
    image_with_boxes = draw_eye_bounding_boxes(image.copy(), left_eye_points, right_eye_points)
    #cv2.imshow("Eye Bounding Boxes", image_with_boxes)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    left_eye_region = crop_eye_region(image, left_eye_points)
    right_eye_region = crop_eye_region(image, right_eye_points)
    combined_eye_region = combine_eye_regions(left_eye_region, right_eye_region)

    # Save the combined_eye_region image
    output_directory = "C:/kobbi/endProject/TSKinFace_Data/Azura_Test/test/bounding_eyes/"
    os.makedirs(output_directory, exist_ok=True)
    output_image_path = os.path.join(output_directory, f"combined_eye_{image_name}")
    cv2.imwrite(output_image_path, combined_eye_region)


    dominant_color, palette = three_most_dominant_colors(combined_eye_region)
    print(dominant_color)
    if is_blueish(palette):
        print(f"{image_name} has a blueish tone")

    if is_greenish(palette):
        print(f"{image_name} has a greenish tone")
    return dominant_color,palette



def get_eye_region(landmarks, start_point, end_point):
    points = []
    for i in range(start_point, end_point + 1):
        x, y = landmarks[0][i][0], landmarks[0][i][1]
        points.append((x, y))
    
    return points

def crop_eye_region(image, points):
    x_min = min([p[0] for p in points])
    x_max = max([p[0] for p in points])
    y_min = min([p[1] for p in points])
    y_max = max([p[1] for p in points])

    return image[y_min:y_max, x_min:x_max]

def combine_eye_regions(left_eye_region, right_eye_region):
    combined_height = max(left_eye_region.shape[0], right_eye_region.shape[0])
    combined_width = left_eye_region.shape[1] + right_eye_region.shape[1]
    
    combined_eye_region = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    
    combined_eye_region[0:left_eye_region.shape[0], 0:left_eye_region.shape[1]] = left_eye_region
    combined_eye_region[0:right_eye_region.shape[0], left_eye_region.shape[1]:] = right_eye_region
    
    return combined_eye_region

def combine_skin_regions(left_skin_region, right_skin_region):
    combined_height = max(left_skin_region.shape[0], right_skin_region.shape[0])
    combined_width = left_skin_region.shape[1] + right_skin_region.shape[1]

    combined_skin_region = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    combined_skin_region[0:left_skin_region.shape[0], 0:left_skin_region.shape[1]] = left_skin_region
    combined_skin_region[0:right_skin_region.shape[0], left_skin_region.shape[1]:] = right_skin_region

    return combined_skin_region



def is_blueish(palette):
    for color in palette:
        r, g, b = color
        if b > r and b > g:
            return True
    return False

def is_greenish(palette):
    for color in palette:
        r, g, b = color
        if g > r and g > b:
            return True
    return False

def draw_eye_bounding_boxes(image, left_eye_points, right_eye_points):
    left_eye_hull = cv2.convexHull(np.array(left_eye_points))
    right_eye_hull = cv2.convexHull(np.array(right_eye_points))

    cv2.drawContours(image, [left_eye_hull], -1, (0, 255, 0), 1)
    cv2.drawContours(image, [right_eye_hull], -1, (0, 255, 0), 1)

    return image


def resizeImage(image):
    file_name22 = "image_resize" + str(count) + ".jpg"
    file_path22 = os.path.join(path_resize, file_name22)
    new_size = (256, 256)
    copyBefore = image.copy()
    image_resize = cv2.resize(copyBefore, new_size)
    cv2.imwrite(file_path22, image_resize)
    return image_resize

def makeLabel(photo_name):
    label = photo_name.split("-")[0:3]
    label = "-".join(label)
    return label


def extract_VGGFace_features(img_path):
    # Extract face using the desired detector backend
    detector_backend = 'opencv'
    face_imgs = functions.extract_faces(img=img_path, target_size=(224, 224), detector_backend=detector_backend)

    if len(face_imgs) > 0:
        # If a face is detected, use the first one (assuming there's only one face in the image)
        face_img = face_imgs[0][0]  # Get the face image from the first tuple

        # Check if the face image is empty before processing
        if face_img.size == 0:
            print(f"Empty face image detected in {img_path}. Skipping this image.")
            return None

        # Convert the face image to a numpy array and make sure it's in the correct format
        x = np.array(face_img, dtype=np.float64)
        x = np.expand_dims(x, axis=0)

        # Extract features with VGGFace
        features_VGGFace = DeepFace.represent(x, model_name='VGG-Face', enforce_detection=False)

        # Convert features_VGGFace to a NumPy array
        features_VGGFace_array = np.array(features_VGGFace)

        return features_VGGFace_array
    else:
        # If no face is detected, return None
        return None








def check_VGGFace_features_extraction(features):
    all_extracted = True
    for face_features in features:
        if face_features.features_VGGFace is None or len(face_features.features_VGGFace) == 0:
            print(f"VGGFace features not extracted for image {face_features.image_name}")
            all_extracted = False
    return all_extracted

def string_to_array(s):
    s = s.strip('[]')
    s = s.split(',')
    array = np.array([float(value) for value in s])
    return array

"""def extract_hair_color(image_path):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
    headers = {
        "Ocp-Apim-Subscription-Key": KEY,
        "Content-Type": "application/octet-stream"
    }
    params = {'returnFaceId': 'false', 'detectionModel': 'detection_03'}
    response = requests.post(f'{endpoint}/detect', headers=headers, params=params, data=image_data)
    faces = loads(response.text)
    
    response_det = face_client.face.detect_with_stream(
        image=image_data,
        detection_model='detection_03',
        recognition_model='recognition_04',
        return_face_landmarks=True
    )"""
"""
    print("Detected Faces:")
    for face in faces:
        print(face)
    image = io.BytesIO(image_data)
    response = face_client.face.detect_with_stream(
        image=image
    )
   # print(vars(response[0]))
   # response = requests.post(f"{ENDPOINT}/face/v1.0/detect", headers=headers, params=params, data=image_data)
   # hair_color_data = response.json()["hairColor"]
    hair_color_data = response[0].face_attributes.hair.hair_color
    return hair_color_data
"""


class FaceFeatures:
    def __init__(self, landmarks, face_embeddings, feature_resnet, hair_color, eye_color, skin_color, label, image , image_name, family_type, family_number, member_type, belongs_to_set):
        self.landmarks = landmarks
        self.face_embeddings = face_embeddings
        self.feature_resnet = feature_resnet
        self.hair_color = hair_color
        self.eye_color = eye_color
        self.skin_color = skin_color
        self.label = label
        self.image = image
        self.image_name = image_name
        self.family_type = family_type
        self.family_number = family_number
        self.member_type = member_type
        self.belongs_to_set = belongs_to_set

