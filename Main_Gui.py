import tkinter as tk
import model_logic
import Gui
from DataSetRead import extract_features
from Cac_Landmarks import landmarks_calculator , set_trips_labels_features
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
    result_field.insert(tk.INSERT, f"Model 1 result: {result_1}\n")
    result_field.insert(tk.INSERT, f"Model 2 result: {result_2}\n")
features = extract_features(directory_For_Pairs)
features_after_cal = landmarks_calculator(features)
data ,labels = set_trips_labels_features(features_after_cal)
root, model_1_entry, model_2_entry, result_field = Gui.create_gui(compare_models)
root.mainloop()
