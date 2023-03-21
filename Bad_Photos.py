import os

# Set the directory path
directory_path = "C:\\kobbi\\endProject\\TSKinFace_Data\\Azura_Test_Copy"

def fun(directory_path):
    files = os.listdir(directory_path)

    # Create a new text file to store the file names
    try:
        with open(os.path.join(directory_path, 'file_names.txt'), 'w') as f:
            # Write each file name to the text file, one per line
            for file_name in files:
                # Use os.path.basename to extract the file name from the full path
                f.write(os.path.basename(file_name) + '\n')
        # No need to explicitly close the file as the "with" statement handles that for us
    except Exception as e:
        print(e)

fun(directory_path)
