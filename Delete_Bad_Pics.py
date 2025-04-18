import os
########################     ########################################
# after deleting images that dont work with the dlib library we need to delete the rest of the family images . this functions handle it
########################     ########################################
dir1 = 'C:\\kobbi\\endProject\\TSKinFace_Data\\All_Pairs_SD\\FMSD'
dir2 = 'C:\\kobbi\\endProject\\TSKinFace_Data\\Azura_Test_Copy'

# Get the list of file names in both directories
dir1_files = os.listdir(dir1)
dir2_files = os.listdir(dir2)

# Create a dictionary of file names and numbers in dir2
dir2_dict = {}
for file_name in dir2_files:
    name, number, name2 = file_name.split('-')
    dir2_dict[(name, number)] = True

# Loop through the files in dir1 and delete those with the same name and number as in dir2
for file_name in dir1_files:
    name, number, _ = file_name.split('-')
    if (name, number) in dir2_dict:
        file_path = os.path.join(dir1, file_name)
        os.remove(file_path)
        print(f"{file_name} has been deleted from {dir1}.")
