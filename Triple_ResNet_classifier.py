from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras import regularizers

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

########################     ########################################
# triple resnet classifier (does not work need super computer)
########################     ########################################
def build_model_keras_triple_resnet(input_shape=(2048,)):
    # Define the three inputs
    father_input = Input(input_shape)
    mother_input = Input(input_shape)
    child_input = Input(input_shape)
# Define a simple MLP architecture
    def create_base_network(input):
        x = Dense(1024, activation='relu')(input)
        x = Dense(512, activation='relu')(x)
        return x

    # Use the architecture for all three inputs
    processed_father = create_base_network(father_input)
    processed_mother = create_base_network(mother_input)
    processed_child = create_base_network(child_input)

    # Combine the outputs of the three branches
    combined = concatenate([processed_father, processed_mother, processed_child])

    # Add final dense layers and compile model
    dense = Dense(256, activation="relu")(combined)
    dense = Dense(128, activation="relu")(dense)
    output = Dense(1, activation="sigmoid")(dense)

    model = Model([father_input, mother_input, child_input], output)
    model.compile(optimizer=Adam(0.1), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def trip_resnet_keras(trips, labels):
    pairs = np.array(trips)
    labels = np.array(labels)

    # Assume that the shape of each individual data sample is (2048,)
    model = build_model_keras_triple_resnet((2048,))

    # Split the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(pairs, labels, test_size=0.2, random_state=42)

    # Slice the data into respective 'father', 'mother', and 'child' parts
    X_train_father, X_train_mother, X_train_child = np.split(X_train, indices_or_sections=3, axis=1)
    X_test_father, X_test_mother, X_test_child = np.split(X_test, indices_or_sections=3, axis=1)

    # Squeeze unnecessary dimensions
    X_train_father = np.squeeze(X_train_father, axis=1)
    X_train_mother = np.squeeze(X_train_mother, axis=1)
    X_train_child = np.squeeze(X_train_child, axis=1)

    X_test_father = np.squeeze(X_test_father, axis=1)
    X_test_mother = np.squeeze(X_test_mother, axis=1)
    X_test_child = np.squeeze(X_test_child, axis=1)
    # Define the early stopping callback
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True)

    # Train the model
    history = model.fit([X_train_mother, X_train_father, X_train_child], y_train, 
                        epochs=150, 
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

    model.save('C://kobbi//endProject//tensorflow_model//model_triple_resnet.h5')

    print("Accuracy:", accuracy_score(y_test, np.round(y_pred)))
    print("Confusion matrix:\n", confusion_matrix(y_test, np.round(y_pred)))
