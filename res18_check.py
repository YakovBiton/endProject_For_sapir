import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.distance import cosine
import face_recognition
import cv2
import numpy as np

# Load a pre-trained ResNet18 model
model = models.resnet50(pretrained=True)

# Remove the last layer to use the model for feature extraction
model = torch.nn.Sequential(*(list(model.children())[:-1]))

# Load and preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)

# Extract features from an image
def extract_features_resnet(image_path, model):
    model.eval()
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)
    return output.squeeze()

def extract_embedding(image_path):
    image = cv2.imread(image_path)
    face_embeddings = face_recognition.face_encodings(image)
    face_embeddings_array = np.array(face_embeddings)
    face_embeddings2 = [*face_embeddings_array[0]]
    return face_embeddings2

# Example usage
image_path_parent = 'C:\\kobbi\\endProject\\TSKinFace_Data\\All_Pairs_SD\\FMSD\\FMSD-18-M.jpg'
image_path_child = 'C:\\kobbi\\endProject\\TSKinFace_Data\\All_Pairs_SD\\FMSD\\FMSD-14-D.jpg'

features1_resnet = extract_features_resnet(image_path_parent, model)
features2_resnet = extract_features_resnet(image_path_child, model)
features1_emb = extract_embedding(image_path_parent)
features2_emb = extract_embedding(image_path_child)


def cosine_similarity(features1, features2):
    return 1 - cosine(features1, features2)

def cosine_similarity_matrix(matrix1, matrix2):
    # Convert the lists to NumPy arrays (if they are not already)
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)

    # Flatten the matrices
    flat_matrix1 = matrix1.flatten()
    flat_matrix2 = matrix2.flatten()
    
    # Normalize the flattened matrices
    flat_matrix1_normalized = flat_matrix1 / np.linalg.norm(flat_matrix1)
    flat_matrix2_normalized = flat_matrix2 / np.linalg.norm(flat_matrix2)
    
    # Compute the cosine similarity
    cosine_similarity_value = np.dot(flat_matrix1_normalized, flat_matrix2_normalized)
    
    return cosine_similarity_value

# Example usage
#similarity = cosine_similarity(features1_resnet, features2_resnet)
#print(f'Cosine similarity between the two images: {similarity}')
#similarity2 = cosine_similarity_matrix(features1_emb, features2_emb)
#print(f'Cosine similarity between the two images: {similarity2}')
