from network_model import FullyConnectedNN
import torch
import csv
from os.path import join

# Make sure to define the same model architecture
# First we need the number of training features for input size (or just hardcode it if you know it)
training_dataset = "data_small.csv"
with open(training_dataset, 'r') as file:
    reader = csv.reader(file)
    first_line = next(reader)  # Get the first line
    num_features = len(first_line[1:])
print(f"Features found in dataset: {num_features}")
model = FullyConnectedNN(num_features)
model_dir = "./models"
model_name = "model_v0.pth"

# Load only the state dictionary safely
state_dict = torch.load(join(model_dir,model_name), weights_only=True)
# Load it into your model
model.load_state_dict(state_dict)
# Set the model to evaluation mode
model.eval()  

print("Model loaded successfully!")