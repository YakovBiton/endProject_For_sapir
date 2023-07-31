import urllib.request
import tarfile
########################     ########################################
# download the dlib library 
########################     ########################################
url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

# Download the file from the URL
urllib.request.urlretrieve(url, 'shape_predictor_68_face_landmarks.dat.bz2')

# Extract the tar file
tar = tarfile.open('shape_predictor_68_face_landmarks.dat.bz2', "r:bz2")
tar.extractall()
tar.close()