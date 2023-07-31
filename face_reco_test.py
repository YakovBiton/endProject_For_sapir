import face_recognition
import numpy as np
########################     ########################################
# check if the face embedding working
########################     ########################################
# Load the image file
image = face_recognition.load_image_file('C:\\kobbi\\endProject\\TSKinFace_Data\\All_Pairs_SD\\FMSD\\FMSD-17-F.jpg')
test = [1,2,4]
# Calculate the facial embeddings
face_embeddings = face_recognition.face_encodings(image)
face_embeddings_array = np.array(face_embeddings)
# The face_embeddings variable now contains a 128-dimensional vector for each face found in the image
print(face_embeddings_array[0])
print(test)
