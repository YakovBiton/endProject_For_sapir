from keras.models import load_model as keras_load_model
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
########################     ########################################
# handle the models in the GUI
########################     ########################################
def load_model_from_path(filepath):  # Rename your function
    return keras_load_model(filepath)  # Call the correctly named function

def evaluate_model(model, data, labels):

    # Reshape data into the form (number of samples, 3, 17)
    n_samples = data.shape[0]
    trips = data.reshape(n_samples, 3, 17)

    # Slice the data into respective 'mother', 'father', and 'child' parts
    trip_mother, trip_father, trip_child = np.split(trips, indices_or_sections=3, axis=1)

    # Remove the singleton dimension
    trip_mother = np.squeeze(trip_mother, axis=1)
    trip_father = np.squeeze(trip_father, axis=1)
    trip_child = np.squeeze(trip_child, axis=1)

    # Make the predictions
    y_pred = model.predict([trip_mother, trip_father, trip_child])

    # Since we used a sigmoid activation in the final layer, the output will be a probability in the range [0, 1].
    # We can convert this to a binary prediction by rounding to the nearest integer: 0 or 1.
    y_pred_binary = np.round(y_pred)

    # Evaluate predictions
    accuracy = accuracy_score(labels, y_pred_binary)
    confusion_matrix_score =  confusion_matrix(labels, y_pred_binary)

    return accuracy, confusion_matrix_score
