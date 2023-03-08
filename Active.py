# Import the extract_features and visualize_data functions from the DataSetRead and Visualize_Data modules
from DataSetRead import *
from Visualize_Data import visualize_color , visualize_shape
from Color_classify import predict_Skin_Colors , predict_hair_Colors
import random
from Cac_Landmarks import *
# Set the directory where the images are stored
directory = 'C:\\kobbi\\endProject\\TSKinFace_Data\\Azura_Test'




# Extract the features
features = extract_features(directory)
landmarks_calculator(features)


# fraction = 0.1

# # Select a random subset of the data
# num_samples_to_keep = int(len(features) * fraction)
# subset_indices = random.sample(range(len(features)), num_samples_to_keep)

# # Extract the subset features
# subset_features = [features[i] for i in subset_indices]
# #visualize_shape(features)
# predict_skin_colors = predict_Skin_Colors(features)
# predict_hair_colors = predict_hair_Colors(features)
# visualize_color(predict_skin_colors)
# #visualize_skin_color(predict_skin_colors)
# visualize_color(predict_hair_colors)
# # subset_labels = [labels[i] for i in subset_indices]

# # Visualize the data
# #visualize_data(subset_features)

# # Visualize the data
# # visualize_data(features, labels)
