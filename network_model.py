import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

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


# Define the neural network model
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=None, output_size=8):
        super(FullyConnectedNN, self).__init__()
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
        self.softmax = nn.Softmax(dim=1)  # Output activation for classification
    
    def forward(self, x):
        # Apply activation after all layers except the last
        for layer in self.layers[:-1]:  
            x = self.relu(layer(x))
        # Softmax the final layer
        x = self.softmax(self.layers[-1](x))
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

    data_file = "./data_small.csv"
    X, y = load_csv_data(data_file)
    print(f"x shape {X.shape}")
    print(f"y shape {y.shape}")

    input_size = X.shape[1]
    model = FullyConnectedNN(input_size)
    # Print the model architecture
    print(model)
    
    # Prepare DataLoader
    # TODO: Split data in to a train, test, and validation set
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Loss for classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            # Zero the gradient buffers
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            
            # Compute the loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Update the weights
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    #TODO: Test model after training and generate a confusion matrix

    #TODO: Save model to test with new data later
