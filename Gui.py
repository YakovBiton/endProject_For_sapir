import tkinter as tk
from tkinter import scrolledtext, filedialog
import tkinter.ttk as ttk  # module for combo boxes
########################     ########################################
# GUI system for displaying models efficiently
########################     ########################################
def browse_file(entry_field):
    filename = filedialog.askopenfilename()
    entry_field.delete(0, tk.END)  # Clear the entry field
    entry_field.insert(0, filename)  # Insert the selected file path



def create_gui(compare_function, ask_gpt4_function):
    root = tk.Tk()
    root.geometry('800x600')  # size of the GUI
    root.title('Model Comparison GUI')

    # Create labels
    model_1_label = tk.Label(root, text="Model 1 Path:")
    model_1_label.grid(row=0, column=0)

    model_2_label = tk.Label(root, text="Model 2 Path:")
    model_2_label.grid(row=1, column=0)

    # Create entry fields
    model_1_entry = tk.Entry(root, width=50)
    model_1_entry.grid(row=0, column=1)

    model_2_entry = tk.Entry(root, width=50)
    model_2_entry.grid(row=1, column=1)

    # Create Browse buttons
    browse_1_button = tk.Button(root, text="Browse", command=lambda: browse_file(model_1_entry))
    browse_1_button.grid(row=0, column=2)

    browse_2_button = tk.Button(root, text="Browse", command=lambda: browse_file(model_2_entry))
    browse_2_button.grid(row=1, column=2)

    # Create compare button
    compare_button = tk.Button(root, text="Compare", command=compare_function)
    compare_button.grid(row=2, column=0, columnspan=3)

    # Create result field for model comparison
    comparison_result_field = scrolledtext.ScrolledText(root, width=75, height=10)
    comparison_result_field.grid(row=3, column=0, columnspan=3)

    # Define facial parts for the dropdown menus
    facial_parts = ["chin", "eyes", "nose", "mouth", "left_ear", "right_ear"]

    # Labels and combo boxes for GPT-4 interaction
    feature_type_label = tk.Label(root, text="Please make a new")
    feature_type_label.grid(row=4, column=0)
    feature_type_combo = ttk.Combobox(root, values=["ratio", "angle"], state="readonly")
    feature_type_combo.grid(row=4, column=1)
    feature_type_combo.current(0)

    facial_part_1_label = tk.Label(root, text="feature with the use of")
    facial_part_1_label.grid(row=5, column=0)
    facial_part_1_combo = ttk.Combobox(root, values=facial_parts, state="readonly")
    facial_part_1_combo.grid(row=5, column=1)
    facial_part_1_combo.current(0)

    facial_part_2_label = tk.Label(root, text="and")
    facial_part_2_label.grid(row=6, column=0)
    facial_part_2_combo = ttk.Combobox(root, values=facial_parts, state="readonly")
    facial_part_2_combo.grid(row=6, column=1)
    facial_part_2_combo.current(1)

    # Create a button to send a query to GPT-4 (renamed to "Make a Call")
    make_call_button = tk.Button(root, text="Make a Call", command=ask_gpt4_function)
    make_call_button.grid(row=7, column=0, columnspan=2)

    # Create result field for GPT-4 interaction
    gpt4_result_field = scrolledtext.ScrolledText(root, width=75, height=10)
    gpt4_result_field.grid(row=8, column=0, columnspan=3)

    return root, model_1_entry, model_2_entry, comparison_result_field, feature_type_combo, facial_part_1_combo, facial_part_2_combo, gpt4_result_field
