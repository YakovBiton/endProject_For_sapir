import os
import matplotlib.pyplot as plt
from Find_Child import find_child

# Define the directory where the images are stored
directory = 'C:\\kobbi\\endProject\\TSKinFace_Data\\All_Pairs_SD\\FMSD\\'

def check_score():

    # Initialize the counters
    counter = 0
    strong_match_counter = 0
    right_families_counter = 0

    # Loop over all the families
    for i in range(1, 251):  # assuming family numbers are from 1 to 250
        try:
            # Define the paths to the father's and mother's images
            img_father_path = os.path.join(directory, f'FMSD-{i}-F.jpg')
            img_mother_path = os.path.join(directory, f'FMSD-{i}-M.jpg')
            
            # Check if both images exist
            if not os.path.exists(img_father_path) or not os.path.exists(img_mother_path):
                print(f"Father's or mother's image for family {i} is missing.")
                continue
            counter +=1        

            # Call the find_child function
            top_N_best_matches = find_child(img_father_path , img_mother_path)
            
            # Check if the actual child is present in the top N results
            actual_child_names = [f'FMSD-{i}-S.jpg']
            child_presence_count = 0
            for match in top_N_best_matches:
                if match[0] in actual_child_names:
                    right_families_counter += 1
                    child_presence_count += match[1]
                    if child_presence_count >= 2:
                        strong_match_counter += 1
                    break  # no need to check the rest of the matches for this family
        except Exception as e:
            print(f'An error occurred for family {i}: {e}')

    # Print the number of times the actual child was present in the top N results
    print(f'The actual child was present in the top N results for {right_families_counter} out of {counter} families.')
    print(f'The actual child was a strong match in {strong_match_counter} out of {counter} families.')

    # Plotting the data
    categories = ['Total Families', 'possible families', 'Strong matches']
    values = [counter, right_families_counter, strong_match_counter]
    
    plt.bar(categories, values, color=['skyblue', 'blue', 'navy'])
    plt.xlabel('Categories')
    plt.ylabel('Number of Families')
    plt.title('Child Matching Results')
    plt.show()

