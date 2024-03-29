from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
########################     ########################################
# classification classifier that handle each person separately 
########################     ########################################
def build_model_keras_triple(input_shape=(17,)):
    # Define the three inputs
    father_input = Input(input_shape)
    mother_input = Input(input_shape)
    child_input = Input(input_shape)

    # Define a simple MLP architecture
    def create_base_network(input):
        x = Dense(128, activation='relu')(input)
        x = Dense(64, activation='relu')(x)
        return x

    # Use the architecture for all three inputs
    processed_father = create_base_network(father_input)
    processed_mother = create_base_network(mother_input)
    processed_child = create_base_network(child_input)

    # Combine the outputs of the three branches
    combined = concatenate([processed_father, processed_mother, processed_child])

    # Add final dense layers and compile model
    dense = Dense(32, activation="relu")(combined)
    dense = Dense(17, activation="relu")(dense)
    output = Dense(1, activation="sigmoid")(dense)

    model = Model([father_input, mother_input, child_input], output)
    model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

import matplotlib.pyplot as plt

def trip_keras(trips, labels):
    pairs = np.array(trips)
    labels = np.array(labels)

    # Assume that the shape of each individual data sample is (17,)
    model = build_model_keras_triple((17,))

    # Split the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(pairs, labels, test_size=0.2, random_state=42)

    # Slice the data into respective 'father', 'mother', and 'child' parts
    X_train_father, X_train_mother, X_train_child = np.split(X_train, indices_or_sections=3, axis=1)
    X_test_father, X_test_mother, X_test_child = np.split(X_test, indices_or_sections=3, axis=1)

    # Define the early stopping callback
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)

    # Train the model
    history = model.fit([X_train_mother, X_train_father, X_train_child], y_train, 
                        epochs=50, 
                        validation_data=([X_test_mother, X_test_father, X_test_child], y_test), 
                        callbacks=[early_stopping])

    # Plot the history for accuracy
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Evaluate the model
    y_pred = model.predict([X_test_mother, X_test_father, X_test_child])

    model.save('C://kobbi//endProject//tensorflow_model//model_triple.h5')

    print("Accuracy:", accuracy_score(y_test, np.round(y_pred)))
    print("Confusion matrix:\n", confusion_matrix(y_test, np.round(y_pred)))
