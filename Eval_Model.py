import torch
from sklearn.metrics import accuracy_score
from Classifier_PyTorch import FacialFeaturesNet
def evaluate(model_path, input_data , true_labels):
    
    model_class = FacialFeaturesNet
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval() # Set the model to evaluation mode
    
    with torch.no_grad():
        input_data_tensor = torch.tensor(input_data, dtype=torch.float32)
        true_labels_tensor = torch.tensor(true_labels, dtype=torch.float32)
        predicted_labels_tensor = model(input_data_tensor)
        predicted_labels = predicted_labels_tensor.argmax(dim=1)
        true_labels = true_labels_tensor.argmax(dim=1)
        accuracy = accuracy_score(true_labels, predicted_labels)
        print('Accuracy:', accuracy)

    #return output.numpy()
