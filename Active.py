# Import the extract_features and visualize_data functions from the DataSetRead and Visualize_Data modules
from DataSetRead import extract_features
from Visualize_Data import visualize_data
import random
# Set the directory where the images are stored
directory = 'C:\\kobbi\\endProject\\TSKinFace_Data\\TSKinFace_cropped'




# Extract the features and labels from the images
features, labels = extract_features(directory)

# Set the fraction of the data to keep (e.g. 0.1 to keep 10%)
keep_fraction = 0.01

# Select a random subset of the data
num_samples_to_keep = int(len(features) * keep_fraction)
subset_indices = random.sample(range(len(features)), num_samples_to_keep)

# Extract the subset of the features and labels
subset_features = [features[i] for i in subset_indices]
subset_labels = [labels[i] for i in subset_indices]

# Visualize the data
visualize_data(subset_features, subset_labels)

# Visualize the data
# visualize_data(features, labels)
