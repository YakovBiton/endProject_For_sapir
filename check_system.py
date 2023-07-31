import os
import matplotlib.pyplot as plt
from Find_Child import find_child

########################     ########################################
# check how good is the system 
########################     ########################################
# Define the directory where the images are stored
directory = 'C:\\kobbi\\endProject\\TSKinFace_Data\\All_Pairs_SD\\FMSD\\'

def check_score():

    # Initialize the counters
    counter = 0
    strong_match_counter = 0
    right_families_counter = 0

    # Loop over all the families
    for i in range(141, 160):  # assuming family numbers are from 1 to 250
       # try:
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
            actual_child_names_son = [f'FMSD-{i}-S.jpg']
            child_presence_count = 0
            for match in top_N_best_matches:
                if match[0] in actual_child_names_son:
                    right_families_counter += 1
                    child_presence_count += match[1]
                    if child_presence_count >= 5:
                        strong_match_counter += 1
                    break  # no need to check the rest of the matches for this family
        #except Exception as e:
        #    print(f'An error occurred for family {i}: {e}')

    # Print the number of times the actual child was present in the top N results
    # Print the percentage of times the actual child was present in the top N results
    print(f'The actual child was present in the top N results for {(right_families_counter/counter)*100}% of the families.')
    print(f'The actual child was a strong match in {(strong_match_counter/counter)*100}% of the families.')

    # Plotting the data
    categories = ['Possible Match', 'Strong Matches']
    values = [round((right_families_counter/counter)*100, 2), round((strong_match_counter/counter)*100, 2)]  # percentages

    plt.bar(categories, values, color=['blue', 'navy'])
    plt.xlabel('Categories')
    plt.ylabel('Percentage of Families (%)')
    plt.title('Child Matching Results')
    plt.ylim([0, 100])  # Set the limits of y-axis from 0 to 100
    plt.show()



def find_all_children_score_regrestion():
    #Initialize counters
    counters_sons = [0]*5  # Assuming 5 lists in the output of find_child function
    total_families = 0
    missing_families = 0
    counters_dauther = [0]*5 
    for i in range(1, 300):  # assuming family numbers are from 1 to 250
        # Define the paths to the father's and mother's images
        img_father_path = os.path.join(directory, f'FMSD-{i}-F.jpg')
        img_mother_path = os.path.join(directory, f'FMSD-{i}-M.jpg')

        # Check if both images exist
        if not os.path.exists(img_father_path) or not os.path.exists(img_mother_path):
            print(f"Father's or mother's image for family {i} is missing.")
            missing_families += 1
            continue

        # Compute the children lists
        children_lists = find_child(img_father_path, img_mother_path)

        # Iterate over each list
        for j, child_list in enumerate(children_lists):
            # Check if the correct child is in the list
            if any(child_name.startswith(f'FMSD-{i}-S.jpg') for child_name, _ in child_list):
                counters_sons[j] += 1

        for j, child_list in enumerate(children_lists):
            # Check if the correct child is in the list
            if any(child_name.startswith(f'FMSD-{i}-D.jpg') for child_name, _ in child_list):
                counters_dauther[j] += 1
        total_families += 1

    print("Total families:", total_families)
    print("Missing families:", missing_families)
    print("Correct sons found counts:", counters_sons)
    print("Correct dauther found counts:", counters_dauther)
