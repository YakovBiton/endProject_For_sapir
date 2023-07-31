import tkinter as tk
from tkinter import scrolledtext, filedialog
########################     ########################################
# GUI system for displaying models efficiently
########################     ########################################
def browse_file(entry_field):
    filename = filedialog.askopenfilename()
    entry_field.delete(0, tk.END)  # Clear the entry field
    entry_field.insert(0, filename)  # Insert the selected file path

def create_gui(compare_function):
    root = tk.Tk()
    root.geometry('600x400')
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

    # Create result field
    result_field = scrolledtext.ScrolledText(root, width=50, height=10)
    result_field.grid(row=3, column=0, columnspan=3)

    return root, model_1_entry, model_2_entry, result_field
