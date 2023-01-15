from DataSetRead import *
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_data(features):
   flat_features =[]
   # if isinstance(features, list):
    # features = np.array(features)
   for i, feature in enumerate(features):
        flat_features.append(feature.reshape(-1))
#  for i, feature in enumerate(flat_features):
     #  print("feature", i, "has shape", feature.shape) 
   # flat_features = features.reshape(-1, features.shape[-1])
   #reducing dim with pca
   pca =PCA(n_components=2)
   features_2d = pca.fit_transform(flat_features)
  # pca.fit(flat_features)
   #reduced_features = pca.transform(flat_features)
  # X_embedded = TSNE(n_components=2,perplexity=10).fit_transform(flat_features)
  # plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels)
   plt.scatter(features_2d[:, 0], features_2d[:, 1], cmap='viridis')
   plt.colorbar()
   plt.show()
   plt.show()

def visualize_skin_color(predictions):
     #Create an empty figure
     fig = plt.figure()
     # Loop through the predictions
     for prediction in predictions:
          color = prediction[0][0]
          x = prediction[1].skin_color[0]
          y = prediction[1].skin_color[1]
          z = prediction[1].skin_color[2]
     #     plt.scatter(x, y, z, c=color, marker='o')
     # Add the labels for the axis
     plt.xlabel('Red')
     plt.ylabel('Green')
     # plt.zlabel('Blue')      

def visualize_skin_color2(knn_skin_colors):
     predicted_labels = [x[0][0] for x in knn_skin_colors]
     image_names = [x[1] for x in knn_skin_colors]

     # Plot the results
     plt.scatter(image_names, predicted_labels)
     plt.xlabel('Image Name')
     plt.ylabel('Predicted Color')
     plt.show()

