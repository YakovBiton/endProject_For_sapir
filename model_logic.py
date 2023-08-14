from keras.models import load_model as keras_load_model
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
################################################################
import os
import sys
import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import Constants_GPT
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

def interact_with_gpt4(query, chat_history , PERSIST = False):
    if not query:
        return "No query provided."

    if PERSIST and os.path.exists("persist"):
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = DirectoryLoader("data_for_GPT/")
        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    result = chain({"question": query, "chat_history": chat_history})
    return result['answer']
