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
    for shape in landmarks:
        for point in shape:
            x, y = point
            cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
    cv2.imshow("Facial Landmarks", image)
    cv2.waitKey(0)
    return landmarks

def extract_hair_and_skin_color(image,landmarks):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    upper_hair = np.array([180, 255, 50])
    rectangular_area_left_skin = extract_bounding_box(landmarks[0][32][0]-4 , landmarks[0][32][1]+4)
    rectangular_area_right_skin = extract_bounding_box(landmarks[0][35][0]+4 , landmarks[0][35][1]+4)
    right_skin = extract_color_from_region(image,rectangular_area_right_skin)
    left_skin = extract_color_from_region(image,rectangular_area_left_skin)
    # Convert the average color of the rectangular areas to the same color space
    # print(right_skin + " and with left " + left_skin + )
    hair_mask = cv2.inRange(hsv, upper_hair, upper_hair)
    skin_mask = ((right_skin[0]+left_skin[0]) / 2,(right_skin[1]+left_skin[1]) / 2 , (right_skin[2]+left_skin[2]) /2)
    hair_color = cv2.mean(image, mask=hair_mask)
    #mean_skin = cv2.mean(image, mask=skin_mask)
    return hair_color , skin_mask

def extract_bounding_box(point_x,point_y) :
    x,y = point_x,point_y
    rectangular_area = [[x-1, y-1], [x-1, y+1], [x+1, y+1], [x+1, y-1]]
    return rectangular_area  

def extract_color_from_region(image, rectangular_area):
    # Create a black image with the same shape as the original image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # Create a polygon with the rectangular_area points
    cv2.fillPoly(mask, [np.array(rectangular_area)], 255)
    # Use the mask to extract the color
    color = cv2.mean(image, mask=mask)
    return color

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

