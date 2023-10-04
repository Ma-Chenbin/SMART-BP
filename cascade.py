import torch
import h2o
import numpy as np
from feats_extract import extract_morphological_features

# Load the pre-trained 1D-ResNet model for stage 1
model_stage1 = torch.load('1D_ResNet_with_SE.pt')
model_stage1.eval()

# Load the trained H2O AutoML models for stage 2 (one model per blood pressure category)
model_normal = h2o.load_model('h2o_automl_model_normal')
model_hypotensive = h2o.load_model('h2o_automl_model_hypotensive')
model_hypertensive = h2o.load_model('h2o_automl_model_hypertensive')

# Test data: PPG signal
ppg_signal = np.load("./datasets/test/ppg_ex.npy")

# Preprocess the PPG signal if required (e.g., normalization)

# Stage 1: Predict blood pressure categories
ppg_tensor = torch.Tensor(ppg_signal).unsqueeze(0).unsqueeze(1)  # Convert to tensor
with torch.no_grad():
    output_stage1 = model_stage1(ppg_tensor)

# Get the predicted blood pressure category (assuming softmax output)
predicted_category = torch.argmax(output_stage1, dim=1).item()

# Stage 2: Predict blood pressure value based on the blood pressure category
ppg_morphological_features = extract_morphological_features(ppg_signal)  # Extract morphological features
input_data = h2o.H2OFrame(ppg_morphological_features.reshape(1, -1))  # Convert to H2OFrame

if predicted_category == 0:
    # Blood pressure is normal
    blood_pressure_prediction = model_normal.predict(input_data)[0][0]
elif predicted_category == 1:
    # Blood pressure is hypotensive
    blood_pressure_prediction = model_hypotensive.predict(input_data)[0][0]
else:
    # Blood pressure is hypertensive
    blood_pressure_prediction = model_hypertensive.predict(input_data)[0][0]

# Output the predicted blood pressure
print("Predicted blood pressure: ", blood_pressure_prediction)
