import os
import cv2
import numpy as np
import io
import dlib
from collections.abc import Mapping
import requests
from json import loads
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
#set the directory
directory = 'C:\\kobbi\\endProject\\TSKinFace_Data\\Azura_Test'
model_path = 'C:\kobbi\endProject\shape_predictor_68_face_landmarks.dat'
detector2 = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)
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
    #loop through subdirectories  
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        #loop through files
        for file in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file)
            if os.path.isfile(file_path):
                """"
                # Read the image file and convert it to a byte array
                with open(file_path, "rb") as image_file:
                    image_data = image_file.read()
                 # Define the payload for the API call
                params = {"returnFaceLandmarks": True}
                payload = image_data
                # Make the API call
                response = requests.post(endpoint, headers=headers, params=params, data=payload)
                """
                """hair_colorAzura = extract_hair_color(file_path)
                print("micro Azura hair color : " + hair_colorAzura)"""
                # Load the image and extract the landmarks
                image = cv2.imread(file_path)
                resizeImage(image)
                image_name = os.path.basename(file_path)
                landmarks = np.array(extract_landmarks(image))
                # Preprocess the landmarks
                # landmarks = np.array(landmarks).flatten()
                # Append the features and labels
                if landmarks.shape != (0,):
                    hair_color, skin_color = extract_hair_and_skin_color(image,landmarks)
                    face_features = FaceFeatures(landmarks, hair_color, skin_color, subdir,image_name)
                  #  landmarks = np.append(landmarks, [hair_color, skin_color])
                   # landmarks = np.expand_dims(landmarks, axis=-1)
                   # eye_color(image)
                   # landmarks[3] = 
                    features.append(face_features)
                    labels.append(subdir)

   
    
    for feature in features:
       print("feature", feature.name , "68 is ", feature.landmarks[0][35][0]) 
       print(feature.skin_color)
       #np.set_printoptions(threshold=sys.maxsize) 
    return features


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
    """ for shape in landmarks:
        for point in shape:
            x, y = point
            cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
    cv2.imshow("Facial Landmarks", image)
    cv2.waitKey(0)"""
    return landmarks

def extract_hair_and_skin_color(image,landmarks):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rectangular_area_right_hair = extract_bounding_box(landmarks[0][16][0]+2 , landmarks[0][16][1]-7)
    rectangular_area_left_hair = extract_bounding_box(landmarks[0][0][0] , landmarks[0][1][1]-20)
    rectangular_area_left_skin = extract_bounding_box(landmarks[0][31][0]-6 , landmarks[0][32][1]-6)
    rectangular_area_right_skin = extract_bounding_box(landmarks[0][35][0]+6 , landmarks[0][35][1]-6)
    right_skin = extract_color_from_region(hsv,rectangular_area_right_skin)
    left_skin = extract_color_from_region(image,rectangular_area_left_skin)
    right_hair = extract_color_from_region(image,rectangular_area_right_hair)
    left_hair = extract_color_from_region(image,rectangular_area_left_hair)
    # Convert the average color of the rectangular areas to the same color space
    # print(right_skin + " and with left " + left_skin + )
    hair_mask = ((right_hair[0]+left_hair[0]) / 2,(right_hair[1]+left_hair[1]) / 2 , (right_hair[2]+left_hair[2]) /2)
    skin_mask = ((right_skin[0]+left_skin[0]) / 2,(right_skin[1]+left_skin[1]) / 2 , (right_skin[2]+left_skin[2]) /2)
    #mean_skin = cv2.mean(image, mask=skin_mask)
    return hair_mask , skin_mask

def extract_bounding_box(point_x,point_y) :
    x,y = point_x,point_y
    rectangular_area = [[x-2, y-2], [x-2, y+2], [x+2, y+2], [x+2, y-2]]
    return rectangular_area  

def extract_color_from_region(image, rectangular_area):
    # Create a copy of the original image
    image_with_bounding_box = image.copy()
    # Draw the bounding box on the image_with_bounding_box
    cv2.fillPoly(image_with_bounding_box, [np.array(rectangular_area)], (0, 255, 0))

    # Save the image with the bounding box
    
    path = "C:/kobbi/endProject/TSKinFace_Data/Azura_Test/test/bounding_boxes/"
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = "image_with_bounding_box" + str(count) + ".jpg"
    file_path = os.path.join(path, file_name)
    cv2.imwrite(file_path, image_with_bounding_box)
    count_plus1()
    # Create a black image with the same shape as the original image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # Create a polygon with the rectangular_area points
    cv2.fillPoly(mask, [np.array(rectangular_area)], 255)
    # Use the mask to extract the color
    color = cv2.mean(image, mask=mask)
    return color

def resizeImage(image):
    file_name22 = "image_resize" + str(count) + ".jpg"
    file_path22 = os.path.join(path_resize, file_name22)
    new_size = (256, 256)
    copyBefore = image.copy()
    image_resize = cv2.resize(copyBefore, new_size)
    cv2.imwrite(file_path22, image_resize)

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
    def __init__(self, landmarks, hair_color, skin_color, label,name):
        self.landmarks = landmarks
        self.hair_color = hair_color
        self.skin_color = skin_color
        self.label = label
        self.name = name

