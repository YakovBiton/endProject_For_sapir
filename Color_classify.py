from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from DataSetRead import *

# Assume you have a dataset of colors with their corresponding labels
colors_skin = [[255, 0, 0], [0, 255, 0], [185,185,185], [210,180,140], [255, 255, 0], [130,130,160], [111,55,55], [255, 255, 255], [100, 42, 42] , [25, 25, 25]]
labels_skin = ["red", "green", "light white", "tan white", "yellow", "dark white", "light brown", "pale white", "brown", "dark black"] 
colors_hair = [[255, 0, 0], [0, 255, 0], [185,185,185],  [255, 255, 0], [111,55,55], [255, 255, 255], [111, 65, 65] , [45, 45, 45]]
labels_hair = ["red", "green", "blonde",  "yellow", "light brown", "pale white", "brown", "dark black"] 
#whitee = purple  
# Create the k-NN classifier
knnSkin = KNeighborsClassifier(n_neighbors=3)
knnHair = KNeighborsClassifier(n_neighbors=2)
knnSkin.fit(colors_skin, labels_skin)
knnHair.fit(colors_hair, labels_hair)
def predict_Skin_Colors(features):
    knn_skin_colors = []
    for  feature in features:
        print(feature.skin_color)
        predictedColor = knnSkin.predict([feature.skin_color])
        name = feature.name
        knn_skin_colors.append([predictedColor, name , feature.skin_color])
        print("name:  " + name + "   skin color is:  " + str(predictedColor))
    return knn_skin_colors

def predict_hair_Colors(features):
    knn_hair_colors = []
    for  feature in features:
        print(feature.hair_color)
        predictedColor = knnHair.predict([feature.hair_color])
        name = feature.name
        knn_hair_colors.append([predictedColor, name , feature.hair_color])
        print("name:  " + name + "   hair color is:  " + str(predictedColor))
    return knn_hair_colors
    
    