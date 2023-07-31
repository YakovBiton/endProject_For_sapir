from DataSetRead import *
from Visualize_Data import visualize_color , visualize_shape
from Color_classify import predict_Skin_Colors , predict_hair_Colors
import random
from Cac_Landmarks import *
from Classifier import *
from Classifier_PyTorch import *
from Eval_Model import *
from Find_Child import *
from Child_Data_Maker import *
from Features_Classifier import *
from Images_DataBase import *
from Classifier_Father import *
from Eval_Keras_model import *
from check_system import check_score , find_all_children_score_regrestion
from keras_binary_classifier import *
from Pair_binary_sklearn import *
from Triple_classifier import *
from Triple_ResNet_classifier import *
# the directory where the images are stored
directory = 'C:\\kobbi\\endProject\\TSKinFace_Data\\Azura_Test'
directory2 = 'C:\\kobbi\\endProject\\TSKinFace_Data\\Bigger_test'
directory_For_Singles = 'C:\\kobbi\\endProject\\TSKinFace_Data\\All_Single_SD'
directory_For_Pairs = 'C:\\kobbi\\endProject\\TSKinFace_Data\\All_Pairs_SD'
directory_For_All = 'C:\\kobbi\\endProject\\TSKinFace_Data\\All_Data'
model_path = 'C:\\kobbi\\endProject\\py_torch_model\\model.pth'
directory_For_argi = 'C:\\kobbi\\endProject\\TSKinFace_Data\\argi'
img_father_path = 'C:\\kobbi\\endProject\\TSKinFace_Data\\All_Data\\FMD_FMS_FMSD\\FMSD-98-F.jpg'
img_mother_path = 'C:\\kobbi\\endProject\\TSKinFace_Data\\All_Data\\FMD_FMS_FMSD\\FMSD-98-M.jpg'

########################      ######################################## 
# Active functions for all the processes
########################     ######################################## 


# Extract the features from the images and add them to the Database
features = extract_features(directory_For_Pairs)
features_after_cal = landmarks_calculator(features)
#add_To_DataBase(features_after_cal)
########################     ######################################## 

#x ,y = set_X_y(features_after_cal)
#neural_Classifier(x , y)
#train_keras_classifier(x , y )

########################  binary classifier  ######################################## 
#x,y = set_pairs_labels(features_after_cal)
#pair_keras(x, y)
########################  binary classifier  ######################################## 

########################  triple classifier  ######################################## 
x,y = set_trips_labels_features(features_after_cal)
#trip_keras(x,y)
########################  triple classifier  ########################################
#
#x,y = set_trips_labels_resnet(features_after_cal)  
#trip_resnet_keras(x,y)
predict_and_evaluate_new_data(x,y)
########################  father classifier  ######################################## 
#x ,y = set_X_y_father_classifier(features_after_cal)
#neural_Classifier_father(x ,y)
########################  features classifier be importens  ######################################## 
#feature_regression(x,y)
########################  features classifier be importens  ######################################## 

#eval model activation:
#features , input_data ,true_label = landmarks_calculator(features)
#evaluate_keras_model(x, y)
#evaluate(model_path , input_data , true_label)

########################  find child from data base  ######################################## 
#topmatches = find_child(img_father_path , img_mother_path)
#check_score()
#find_all_children_score_regrestion()
########################  find child from data base  ######################################## 

########################  delete child from data base  ######################################## 
#delete_from_database("FMSD-160-M.jpg")
########################  delete child from data base  ######################################## 

###################### create data base       ##########################################
#son_daughter_featurs = extract_son_daughter_attributes("C:\\kobbi\\endProject\\TSKinFace_Data\\All_Pairs_SD\\FMSD")
#cal_son_daughter_landmarks_and_save(son_daughter_featurs)
###################### create data base       ##########################################


# fraction = 0.1

# # Select a random subset of the data
# num_samples_to_keep = int(len(features) * fraction)
# subset_indices = random.sample(range(len(features)), num_samples_to_keep)

# # Extract the subset features
# subset_features = [features[i] for i in subset_indices]
# #visualize_shape(features)
# predict_skin_colors = predict_Skin_Colors(features)
# predict_hair_colors = predict_hair_Colors(features)
# visualize_color(predict_skin_colors)
# #visualize_skin_color(predict_skin_colors)
# visualize_color(predict_hair_colors)
# # subset_labels = [labels[i] for i in subset_indices]

# # Visualize the data
# #visualize_data(subset_features)

# # Visualize the data
# # visualize_data(features, labels)
