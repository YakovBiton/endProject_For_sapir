import numpy as np
from deepface import DeepFace
from deepface.commons import functions
from PIL import Image

# Load an image of a face
img_path = 'C:/kobbi/endProject/TSKinFace_Data/All_Single_SD/all/FMD-1-F.jpg'
img = Image.open(img_path)

# Ensure the image is in RGB format
if img.mode != 'RGB':
    img = img.convert('RGB')

# Resize the image
img = img.resize((224, 224))

# Convert the image to a numpy array and normalize it
x = np.array(img)
x = functions.preprocess_face(x, target_size=(224, 224), grayscale=False)

# Extract features with VGGFace
features = DeepFace.represent(x, model_name='VGG-Face', enforce_detection=False)
# Print the feature vector
print("Feature vector length:", len(features))
print("Feature vector values:\n", features)
