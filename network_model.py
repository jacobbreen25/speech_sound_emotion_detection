import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import seaborn as sns

def load_csv_data(file_path):
    """
    Loads a CSV file with the first column as labels and the rest as features.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        tuple: (features_tensor, labels_tensor)
    """
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Separate labels (first column) and features (remaining columns)
    labels = df.iloc[:, 0]
    features = df.iloc[:, 1:]
    
    # Convert to PyTorch tensors
    features_tensor = torch.tensor(features.values, dtype=torch.float32)
    labels_tensor = torch.tensor(labels.values, dtype=torch.long)  # Use long for classification
    
    return features_tensor, labels_tensor

def plot_confusion_matrix(conf_matrix, class_names):
    """
    Plots a confusion matrix as a heatmap.
    
    Args:
        conf_matrix (ndarray): Confusion matrix.
        class_names (list): List of class names corresponding to matrix rows/columns.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

def get_predictions(model, dataloader, device='cpu'):
    """
    Gets all predictions from the test DataLoader.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test set.
        device (str): Device to run the model on ('cpu' or 'cuda').
    
    Returns:
        torch.Tensor: All predictions.
        torch.Tensor: Corresponding true labels.
    """
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation
        for batch in dataloader:
            inputs, labels = batch  # Unpack the batch
            # inputs = inputs.to(device)
            # labels = labels.to(device)
            
            outputs = model(inputs)  # Forward pass
            preds = torch.argmax(outputs, dim=1)  # Get predicted class
            
            all_preds.append(preds)
            all_labels.append(labels)
    
    # Concatenate all batches into a single tensor
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    # print(all_labels)
    # print(all_preds)
    
    return all_preds, all_labels

# Define the neural network model
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=None, output_size=9):
        super(FullyConnectedNN, self).__init__()
        hidden_sizes = [3168, 1584, 792, 396, 198, 99]
        # hidden_sizes = [100]
        if hidden_sizes is None:
            hidden_sizes = self.get_hidden_sizes(input_size,output_size)

        # Define the layers
        layer_sizes = [input_size]+ hidden_sizes + [output_size]
        print(f"Layer Sizes: {layer_sizes}")

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            size_in = layer_sizes[i]
            size_out = layer_sizes[i + 1]
            print(f"> Added Linear Layer [{size_in} -> {size_out}]")
            self.layers.append(nn.Linear(size_in, size_out))
        
        self.relu = nn.ReLU()  # Activation function
        # self.softmax = nn.Softmax(dim=1)  # Output activation for classification
    
    def forward(self, x):
        # Apply activation after all layers except the last
        for layer in self.layers[:-1]:  
        # for layer in self.layers: 
            x = self.relu(layer(x))
        # Softmax the final layer
        # x = self.softmax(self.layers[-1](x))
        x = self.layers[-1](x)
        return x

    def get_hidden_sizes(self,input_size,output_size):
        hidden_sizes = []
        num_layers = 5
        diff = abs(input_size-output_size)
        for i in range(num_layers):
            s = input_size - (i+1)*(diff//num_layers)
            hidden_sizes.append(s)

        return hidden_sizes
    

if __name__ == "__main__":
    # Example data: Create some synthetic data (100 samples, 20 features)
    # X = torch.randn(100, input_size)
    # y = torch.randint(0, 8, (100,))  # Random integer labels in [0, 7]

    data_file = "./data_v3.csv"
    X, y = load_csv_data(data_file)
    print(f"x shape {X.shape}")
    print(f"y shape {y.shape}")

    input_size = X.shape[1]
    model = FullyConnectedNN(input_size)
    # Print the model architecture
    print(model)
    
    # Prepare DataLoader
    # TODO: Split data in to a train, test, and validation set
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Loss for classification
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    # Training loop
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        val_loss = 0.0

        model.train()
        for batch_X, batch_y in train_dataloader:
            # Zero the gradient buffers
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)

            # print(outputs, batch_y)
            
            # Compute the loss
            loss = criterion(outputs, batch_y)
            # print(f"LOSS: {loss}")
            # Backward pass
            loss.backward()
            
            # Update the weights
            optimizer.step()

            running_loss += loss.item()
            # validation after each epoch
        avg_loss = running_loss / len(train_dataloader)
        train_losses.append(avg_loss)  # Store training loss
        model.eval()
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch_X, batch_y in val_dataloader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_val_preds.append(preds)
                all_val_labels.append(batch_y)

        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)  # Store validation loss
        all_val_preds = torch.cat(all_val_preds).cpu()
        all_val_labels = torch.cat(all_val_labels).cpu()
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    #Test model after training and generate a confusion matrix
    print("Getting predictions")
    labels =["silence", "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    test_predictions, test_labels = get_predictions(model,test_dataloader)
    # Compute confusion matrix
    conf_matrix = confusion_matrix(test_labels, test_predictions)
    plot_confusion_matrix(conf_matrix,labels)

    # Plot training and validation losses
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    #Save model to test with new data later
    model_path = './models'
    model_name = 'model_v2.pth'
    torch.save(model.state_dict(), join(model_path,model_name))
    print("Model saved successfully!")