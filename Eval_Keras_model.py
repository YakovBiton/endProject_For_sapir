from tensorflow.keras.models import load_model
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
    

