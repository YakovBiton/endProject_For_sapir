import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
########################     ########################################
# attempt to train 2 classifier one for each parent and then use thrid classifier with the outputs
########################     ########################################
def build_model(input_shape):
    input_father = Input(shape=input_shape)
    input_mother = Input(shape=input_shape)

    father = Dense(256, activation='relu')(input_father)
    father = Dense(128, activation='relu')(father)

    mother = Dense(256, activation='relu')(input_mother)
    mother = Dense(128, activation='relu')(mother)

    combined = concatenate([father, mother])

    z = Dense(128, activation='relu')(combined)
    z = Dense(64, activation='relu')(z)
    z = Dense(17)(z)

    return Model(inputs=[input_father, input_mother], outputs=z)

def train_keras_classifier(X , y ):
    scaler_father = StandardScaler()
    scaler_mother = StandardScaler()

    X = X.reshape(-1, 34)  # Reshape the 1D array into a 2D array

    X_father = scaler_father.fit_transform(X[:,:17])  # First half of the features are father's
    X_mother = scaler_mother.fit_transform(X[:,17:])  # Second half of the features are mother's

    # Save the scalers
    joblib.dump(scaler_father, 'C://kobbi//endProject//tensorflow_model//scaler_father.pkl')
    joblib.dump(scaler_mother, 'C://kobbi//endProject//tensorflow_model//scaler_mother.pkl')

    X_father_train, X_father_val, X_mother_train, X_mother_val, y_train, y_val = train_test_split(X_father, X_mother, y, test_size=0.2, random_state=42)

    model = build_model((17,))
    model.compile(loss='mse', optimizer=Adam())

    # Define the early stopping criteria and model checkpoint to save the best model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('C://kobbi//endProject//tensorflow_model//model.h5', save_best_only=True)

    history = model.fit([X_father_train, X_mother_train], y_train, epochs=150, batch_size=64, validation_data=([X_father_val, X_mother_val], y_val), callbacks=[early_stopping, model_checkpoint])