import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import joblib

# Load dataset
df = pd.read_csv('LastPlease/Depression Professional Dataset.csv')
df = df.dropna()

# Feature selection
selected_features = [
    'Gender', 'Age', 'Work Pressure', 'Job Satisfaction',
    'Sleep Duration', 'Dietary Habits',
    'Have you ever had suicidal thoughts ?', 'Work Hours',
    'Financial Stress', 'Family History of Mental Illness'
]
X = df[selected_features]
y = df['Depression']

# Convert target variable to numeric
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Define preprocessing pipeline
preprocessor = Pipeline(steps=[
    ('one_hot', OneHotEncoder(handle_unknown='ignore')),
    ('scaler', StandardScaler(with_mean=False))
])

# Apply preprocessing
X = preprocessor.fit_transform(X)

# Save preprocessor for reuse
joblib.dump(preprocessor, './model/preprocessor.pkl')
print("Preprocessor saved as './model/preprocessor.pkl'")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Shape of processed input: {X.shape}")

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.todense(), dtype=torch.float32)  # .todense() for sparse matrices
X_test_tensor = torch.tensor(X_test.todense(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define Neural Network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Initialize model, loss function, and optimizer
input_size = X_train_tensor.shape[1]
model = NeuralNetwork(input_size)
criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluating the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_labels = (y_pred > 0.5).float()
    accuracy = (y_pred_labels.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item()
    print(f"Accuracy on test data: {accuracy * 100:.2f}%")

# Save the model
torch.save(model.state_dict(), './model/neural_network_model.pth')
print("Neural Network model has been saved as './model/neural_network_model.pth'")
