import os
import cv2
import numpy as np
import tensorflow as tf
#set the directory
directory = 'C:\\kobbi\\endProject\\TSKinFace_Data\\TSKinFace_cropped'

def extract_features(directory):
    #we load mobileNetV2 model 
    model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    features = []
    labels = []
    #loop through subdirectories and extract image files 
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        #if os.path.isdir(subdir_path):
        #loop through files
        for file in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file)
            if os.path.isfile(file_path):
            #we load image with CV2     
                image = cv2.imread(file_path)
                # Preprocess of the image
                image = cv2.resize(image, (224, 224))
                image = image / 255.0
                image = np.expand_dims(image, axis=0)
                #extract the features from the image
                feature = model.predict(image)
                #decrase the features to 1D array
               
                features.append(feature)
                labels.append(subdir)

    features = np.concatenate(features, axis=0)          
    
    return features, labels              