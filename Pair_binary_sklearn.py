from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
########################     ########################################
# binary resnet classifier (does not work need super computer)
########################     ########################################
def build_model_keras_pair(input_shape):
    # Define the inputs
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # Define a simple MLP architecture
    def create_base_network():
        input = Input(shape=input_shape)
        x = Dense(1024, activation='relu')(input)
        x = Dense(512, activation='relu')(x)
        return Model(input, x)

    # Reuse the same architecture for both inputs
    base_network = create_base_network()
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Combine the outputs of the two branches
    combined = concatenate([processed_a, processed_b])

    # Add final dense layers and compile model
    dense = Dense(256, activation="relu")(combined)
    dense = Dense(128, activation="relu")(dense)
    output = Dense(1, activation="sigmoid")(dense)

    model = Model([input_a, input_b], output)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def pair_keras(pairs, labels):
    pairs = np.array(pairs)
    labels = np.array(labels)

    # Assume that the shape of each individual data sample is (2048,)
    model = build_model_keras_pair((2048,))

    # Split the pairs into separate arrays for training
    parent_features = np.squeeze(pairs[:, 0])
    child_features = np.squeeze(pairs[:, 1])
    
    # Split the data into training and testing datasets
    X_train_a, X_test_a, y_train, y_test = train_test_split(parent_features, labels, test_size=0.2, random_state=42)
    X_train_b, X_test_b, _, _ = train_test_split(child_features, labels, test_size=0.2, random_state=42)

    # Train the model
    history = model.fit([X_train_a, X_train_b], y_train, epochs=10, validation_data=([X_test_a, X_test_b], y_test))

    # Evaluate the model
    y_pred = model.predict([X_test_a, X_test_b])

    print("Accuracy:", accuracy_score(y_test, np.round(y_pred)))
    print("Confusion matrix:\n", confusion_matrix(y_test, np.round(y_pred)))
