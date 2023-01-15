from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from DataSetRead import *

# Assume you have a dataset of colors with their corresponding labels
colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 255, 255], [165, 42, 42] , [0, 0, 0]]
labels = ["red", "green", "blue", "yellow", "whitee", "whitee", "white", "brown", "black"] 
#whitee = purple  
# Create the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(colors, labels)

def predict_Colors(features):
    print("Predict im here...")
    knn_skin_colors = []
    for  feature in features:
        print(feature.skin_color)
        predictedColor = knn.predict([feature.skin_color])
        name = feature.name
        knn_skin_colors.append([predictedColor, name])
        print("name:  " + name + "   skin color is:  " + str(predictedColor))
    return knn_skin_colors


    
    