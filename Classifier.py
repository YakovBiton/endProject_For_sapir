import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split



def landmarks_classifier(features , x , y):
        
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print('X shape:', x.shape)
    print('Y shape:', y.shape)
    print('X example:', x[0])
    print('Y example:', y[9])

    # Define the Keras model
    model = Sequential()
    model.add(Dense(10, input_dim=2, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Test the model on the testing set
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Accuracy:', accuracy)