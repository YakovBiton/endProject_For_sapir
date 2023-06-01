from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
def evaluate_keras_model(X_new, y_new):
    # Load the model from disk
    model = load_model('C://kobbi//endProject//tensorflow_model//model.h5')

    # Preprocess the data in the same way as for training
    scaler = StandardScaler()
    X_new = X_new.reshape(-1, 22)  # Reshape the 1D array into a 2D array if necessary
    X_father_new = scaler.fit_transform(X_new[:,:11])  # First half of the features are father's
    X_mother_new = scaler.fit_transform(X_new[:,11:])  # Second half of the features are mother's

    # Make predictions on the new data
    y_pred = model.predict([X_father_new, X_mother_new])
    # Calculate Mean Squared Error for each feature
    mse_features = np.mean(np.power(y_new - y_pred, 2), axis=0)
    print('Mean Squared Error for each feature: ', mse_features)

    # Calculate Mean Absolute Error for each feature
    mae_features = np.mean(np.abs(y_new - y_pred), axis=0)
    print('Mean Absolute Error for each feature: ', mae_features)
    # Compute the Mean Squared Error of the predictions
    mse = mean_squared_error(y_new, y_pred)
    print('Mean Squared Error on new data:', mse)

def predict_and_evaluate_new_data(data, labels):
    from sklearn.metrics import accuracy_score, confusion_matrix
    import numpy as np

    # Load the trained model
    model = load_model('C://kobbi//endProject//tensorflow_model//model_triple.h5')

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
    print("Accuracy:", accuracy_score(labels, y_pred_binary))
    print("Confusion matrix:\n", confusion_matrix(labels, y_pred_binary))

    return y_pred_binary


    

