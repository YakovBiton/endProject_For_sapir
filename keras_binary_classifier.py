import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
########################     ########################################
# binary (father and mother are referred as one) classification classifier
########################     ########################################
def build_model_binary(input_shape):
    input_father = Input(shape=input_shape)
    input_mother = Input(shape=input_shape)

    father = Dense(128, activation='relu')(input_father)
    father = Dense(64, activation='relu')(father)

    mother = Dense(128, activation='relu')(input_mother)
    mother = Dense(64, activation='relu')(mother)

    combined = concatenate([father, mother])

    z = Dense(64, activation='relu')(combined)
    z = Dense(32, activation='relu')(z)
    z = Dense(1, activation='sigmoid')(z)  # change here

    return Model(inputs=[input_father, input_mother], outputs=z)

def train_keras_classifier_binary(X , y ):
    scaler_father = StandardScaler()
    scaler_mother = StandardScaler()

    X = X.reshape(-1, 22)  

    X_father = scaler_father.fit_transform(X[:,:11])  
    X_mother = scaler_mother.fit_transform(X[:,11:])  

    joblib.dump(scaler_father, 'C://kobbi//endProject//tensorflow_model//scaler_father.pkl')
    joblib.dump(scaler_mother, 'C://kobbi//endProject//tensorflow_model//scaler_mother.pkl')

    X_father_train, X_father_val, X_mother_train, X_mother_val, y_train, y_val = train_test_split(X_father, X_mother, y, test_size=0.2, random_state=42)

    model = build_model_binary((11,))
    model.compile(loss='binary_crossentropy', optimizer=Adam())  # change here

    history = model.fit([X_father_train, X_mother_train], y_train, epochs=300, batch_size=64, validation_data=([X_father_val, X_mother_val], y_val))

    model.save('C://kobbi//endProject//tensorflow_model//model_binary.h5')
