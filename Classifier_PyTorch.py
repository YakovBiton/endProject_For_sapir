import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler # Added for normalization


class ChildFaceFeaturesNet(nn.Module):
    
    def __init__(self):
        super(ChildFaceFeaturesNet, self).__init__()
        self.fc1 = nn.Linear(22, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32,11)
        self.dropout = nn.Dropout(0.5) # Added dropout for regularization

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x) # Added dropout after the first layer
        x = torch.relu(self.fc2(x))
        x = self.dropout(x) # Added dropout after the second layer
        x = torch.relu(self.fc3(x))
        x = self.dropout(x) # Added dropout after the third layer
        x = self.fc4(x)
        return x

class ChildFaceFeaturesNetModified(nn.Module):
    
    def __init__(self):
        super(ChildFaceFeaturesNetModified, self).__init__()

        self.single_layers = nn.ModuleList([nn.Linear(2, 8) for _ in range(11)])

        # Final layers
        self.fc1 = nn.Linear(88, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 11)

        self.dropout = nn.Dropout(0.5) # Added dropout for regularization

    def forward(self, x):
        outputs = []
        for i in range(11):
            x_single = torch.relu(self.single_layers[i](torch.cat((x[:, i:i+1], x[:, i+11:i+12]), dim=1)))
            x_single = self.dropout(x_single) # Added dropout after the first layer
            outputs.append(x_single)

        x = torch.cat(outputs, dim=1)

        # Final layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x) # Added dropout after the first layer
        x = torch.relu(self.fc2(x))
        x = self.dropout(x) # Added dropout after the second layer
        x = self.fc3(x)

        return x
    
model = ChildFaceFeaturesNetModified()

def neural_Classifier(X , y ):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Normalize the input data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        batch_size = 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        epochs = 300
        train_losses = []
        val_losses = []
        patience = 100
        best_val_loss = float('inf')
        counter = 0
        for epoch in range(epochs):
            running_train_loss = 0.0
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * X_batch.size(0)
            train_loss = running_train_loss / len(train_loader)
            train_losses.append(train_loss)
            running_val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    running_val_loss += loss.item() * X_batch.size(0)
            val_loss = running_val_loss / len(val_loader)
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break
            if epoch % 10 == 0:
                print('Epoch [{}/{}], Train Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch+1, epochs, train_loss, val_loss))

        torch.save(model.state_dict(), 'C://kobbi//endProject//py_torch_model//model.pth')
        
