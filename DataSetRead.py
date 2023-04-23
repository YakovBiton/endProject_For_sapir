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

#set the directory
directory = 'C:\\kobbi\\endProject\\TSKinFace_Data\\Azura_Test'
model_path = 'C:\kobbi\endProject\shape_predictor_68_face_landmarks.dat'
bad_photos_path = 'C:\\kobbi\\endProject\\TSKinFace_Data\\Azura_Test_Copy'
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
                landmarks = np.array(extract_landmarks(image))
                # Preprocess the landmarks
                # landmarks = np.array(landmarks).flatten()
                # Append the features and labels
                if landmarks.shape != (0,):
                    hair_color, skin_color = extract_hair_and_skin_color(image,landmarks)
                    belongs_to_set = '$'
                    # With these lines:
                    ######   features_VGGFace still does not work ######
                    #features_VGGFace = extract_VGGFace_features(file_path)
                    #if features_VGGFace is None:
                    #    print("Skipping image", file_path)
                    #    continue
                    #features_VGGFace_array = np.array(features_VGGFace)
                    face_features = FaceFeatures(landmarks, hair_color, skin_color, label, image, image_name, family_type, family_number, member_type, belongs_to_set)
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
   
    
    for feature in features:
       print("feature", feature.name , "68 is ", feature.landmarks[0][35][0]) 
       #print(feature.skin_color)
       #print(feature.hair_color)
       
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
    copyImage = image.copy()
    for shape in landmarks:
        for point in shape:
            x, y = point
            cv2.circle(copyImage, (x, y), 2, (255, 0, 0), -1)
    # show the landmarks on the face
    # cv2.imshow("Facial Landmarks", copyImage)
    # cv2.waitKey(0)
    return landmarks

def extract_hair_and_skin_color(image,landmarks):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    height, width, _ = image.shape # extract the image width and height
    rectangular_area_right_hair = extract_bounding_box(landmarks[0][16][0]+2 , landmarks[0][16][1]-7, width, height)
    rectangular_area_left_hair = extract_bounding_box(landmarks[0][0][0] , landmarks[0][1][1]-20, width, height)
    rectangular_area_left_skin = extract_bounding_box(landmarks[0][31][0]-6 , landmarks[0][32][1]-6, width, height)
    rectangular_area_right_skin = extract_bounding_box(landmarks[0][35][0]+6 , landmarks[0][35][1]-6, width, height)
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

def extract_bounding_box(point_x, point_y, image_width, image_height):
    x, y = point_x, point_y
    margin = 2
    x1, y1 = max(x - margin, 0), max(y - margin, 0)
    x2, y2 = min(x + margin, image_width), min(y + margin, image_height)
    rectangular_area = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
    return rectangular_area

# def extract_bounding_box(point_x,point_y) :
#     x,y = point_x,point_y
#     rectangular_area = [[x-2, y-2], [x-2, y+2], [x+2, y+2], [x+2, y-2]]
#     return rectangular_area  

def extract_color_from_region(image, rectangular_area):
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
    def __init__(self, landmarks ,hair_color, skin_color, label, image , name, family_type, family_number, member_type, belongs_to_set):
        self.landmarks = landmarks
        self.hair_color = hair_color
        self.skin_color = skin_color
        self.label = label
        self.image = image
        self.name = name
        self.family_type = family_type
        self.family_number = family_number
        self.member_type = member_type
        self.belongs_to_set = belongs_to_set

