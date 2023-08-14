import tkinter as tk
import model_logic
import Gui
from DataSetRead import extract_features
from Cac_Landmarks import landmarks_calculator , set_trips_labels_features
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

directory_For_Pairs = 'C:\\kobbi\\endProject\\TSKinFace_Data\\All_Pairs_SD'
########################     ########################################
# main for GUI
########################     ########################################
def compare_models():
    # Get the model paths from the entry fields
    model_1_path = model_1_entry.get()
    model_2_path = model_2_entry.get()

    # Load the models
    model_1 = model_logic.load_model_from_path(model_1_path)  # Call the renamed function
    model_2 = model_logic.load_model_from_path(model_2_path)  # Call the renamed function

    # Run your evaluation function and get the results
    result_1 = model_logic.evaluate_model(model_1, data ,labels)
    result_2 = model_logic.evaluate_model(model_2, data ,labels)

    # Display the results in the text field
    comparison_result_field.insert(tk.INSERT, f"Model 1 result: {result_1}\n")
    comparison_result_field.insert(tk.INSERT, f"Model 2 result: {result_2}\n")

chat_history = []  # Keep track of chat history

def ask_gpt4():
    feature_type = feature_type_combo.get()  # Get selected feature type (ratio or angle)
    facial_part_1 = facial_part_1_combo.get()  # Get selected facial part 1
    facial_part_2 = facial_part_2_combo.get()  # Get selected facial part 2

    # Construct the query
    query = (f"Excellent, I see that you're ready! Now, let's create a new {feature_type} "
             f"feature involving the {facial_part_1} and the {facial_part_2}. Utilizing the dlib 68 points landmarks method and "
             "existing functions, please calculate any necessary distances, midpoints or angles, and formulate a new "
             f"{feature_type} feature that describes the relationship between these facial parts. Please provide only the calculations "
             "and final ratio feature without additional explanation or steps.")
    response = model_logic.interact_with_gpt4(query, chat_history)
    gpt4_result_field.insert(tk.INSERT, f"GPT-4 response: {response}\n")
    chat_history.append((query, response))


features = extract_features(directory_For_Pairs)
features_after_cal = landmarks_calculator(features)
data ,labels = set_trips_labels_features(features_after_cal)
root, model_1_entry, model_2_entry, comparison_result_field, feature_type_combo, facial_part_1_combo, facial_part_2_combo, gpt4_result_field = Gui.create_gui(compare_models, ask_gpt4)
################################################################
# coonection to Ai Model and features files
################################################################

os.environ["OPENAI_API_KEY"] = Constants_GPT.APIKEY
# ...

INTRO_MESSAGE = (
    "Welcome, esteemed artificial intelligence model! You have been specifically designed and trained as a feature-making machine, "
    "capable of generating and calculating intricate geometric features based on facial landmarks. Your abilities are essential in a critical "
    "mission: determining biological relationships between individuals, specifically identifying a child's biological parents from facial images.\n\n"
    "Your task will involve utilizing the dlib 68 points landmarks method to create and define new ratio and angle features. For ratio features, "
    "you'll be working with existing functions such as \"euclidean_distance\" , \"get_midpoint\" , \"angle_between_lines\" and \"angle_between_points\" , and you'll be encouraged to calculate new points "
    "or distances as needed. For angle features, you can use functions like \"angle_between_lines(line1_start, line1_end, line2_start, line2_end)\" "
    "and \"angle_between_points(a, b, c)\" to compute the required angles.\n\n"
    "You will be asked to generate specific calculations, points, and final features without additional explanations or detailed steps. The purpose "
    "of these features is to measure and compare facial characteristics that might indicate family resemblances. This could be pivotal in understanding "
    "the geometry of faces and helping to ascertain if a child is indeed the biological offspring of two given parents.\n\n"
    "Your understanding and expertise in this area are vital. Your responses should be concise and directly related to the task at hand. Here's an example "
    "of the input you might receive for ratio features:\n"
    "\"Excellent, I see that you're ready! Now, let's create a new ratio feature involving the nose and the face. Utilizing the dlib 68 points landmarks method "
    "and existing functions, please calculate any necessary distances, midpoints or angles, and formulate a new ratio feature that describes the relationship between "
    "these facial parts. Please provide only the calculations and final ratio feature without additional explanation or steps.\"\n\n"
    "And an example response might be:\n"
    "\"nose_width = euclidean_distance(landmarks_coordinates[0][31], landmarks_coordinates[0][35])\n"
    "face_width = euclidean_distance(landmarks_coordinates[0][0], landmarks_coordinates[0][16])\n"
    "nose_face_width_ratio = nose_width / face_width\"\n\n"
    "For angle features, you might receive this input:\n"
    "\"Excellent, I see that you're ready! Now, let's create a new angle feature involving the nose and the mouth. Utilizing the dlib 68 points landmarks method "
    "and existing functions, please calculate any necessary distances, midpoints or angles, and formulate a new angle feature that describes the relationship between "
    "these facial parts. Please provide only the calculations and final angle feature without additional explanation or steps.\"\n\n"
    "An expected response might be:\n"
    "nose_line_start = landmarks_coordinates[0][27]\n"
    "nose_line_end = landmarks_coordinates[0][30]\n"
    "mouth_line_start = get_midpoint(landmarks_coordinates[0][3], landmarks_coordinates[0][4])\n"
    "mouth_line_end = landmarks_coordinates[0][48]\n"
    "angle_between_nose_mouth = angle_between_lines(nose_line_start, nose_line_end, mouth_line_start, mouth_line_end)\n\n"
    "If you are clear on your role and the expectations, please respond with: \"I understand and am ready for the new feature description.\""
)




# Send the introductory message to GPT-4 and display the response
intro_response = model_logic.interact_with_gpt4(INTRO_MESSAGE, chat_history)
gpt4_result_field.insert(tk.INSERT, f"GPT-4 response: {intro_response}\n")
chat_history.append((INTRO_MESSAGE, intro_response))


root.mainloop()
#