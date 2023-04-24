import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from Classifier_PyTorch import ChildFaceFeaturesNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate(model_path, input_data, true_labels):
    model_class = ChildFaceFeaturesNet
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        input_data_tensor = torch.tensor(input_data, dtype=torch.float32)
        true_labels_tensor = torch.tensor(true_labels, dtype=torch.float32)
        predicted_labels_tensor = model(input_data_tensor)
        mse = mean_squared_error(true_labels_tensor, predicted_labels_tensor)
        mae = mean_absolute_error(true_labels_tensor, predicted_labels_tensor)
        r2 = r2_score(true_labels_tensor, predicted_labels_tensor)
        print("Mean Squared Error:", mse)
        print("Mean Absolute Error:", mae)
        print("R^2 Score:", r2)

