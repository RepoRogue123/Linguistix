import streamlit as st
import numpy as np
import librosa
import torch
import torch.nn as nn
import pickle
import os

# Define the MLP Model architecture (same as in mlp_lda.ipynb)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=50):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_mlp_model():
    if 'mlp_model' not in st.session_state:
        try:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(current_dir, "mlp_lda_model.pkl")
            
            with open(model_path, 'rb') as f:
                model_dict = pickle.load(f)
            
            # Create model with the same architecture
            model = MLP(
                input_size=model_dict['input_dim'],
                hidden_size=model_dict['hidden_size'],
                output_size=model_dict['output_size']
            )
            
            # Load the trained weights
            model.load_state_dict(model_dict['model_state_dict'])
            model.eval()  # Set to evaluation mode
            
            st.session_state.mlp_model = model
            st.session_state.mlp_accuracy = model_dict['accuracy']
            return True
        except Exception as e:
            st.error(f"Error loading MLP model: {e}")
            return False
    return True

def process_audio(file):
    import time
    start_time = time.time()

    # Load and preprocess audio same as Model1
    y, sr = librosa.load(file, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    max_pad_len = 100
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    mfcc_flat = mfcc.flatten().reshape(1, -1)
    lda_features = np.dot(mfcc_flat, st.session_state.lda_eigenvectors)

    # Convert to PyTorch tensor for MLP model
    input_tensor = torch.tensor(lda_features, dtype=torch.float32)
    
    # Make prediction with MLP model
    with torch.no_grad():
        outputs = st.session_state.mlp_model(input_tensor)
        predicted_label = torch.argmax(outputs, dim=1).item()
    
    # Get speaker name
    speaker_name = st.session_state.inv_label_map[predicted_label]
    
    processing_time = time.time() - start_time
    return speaker_name, processing_time

def main():
    if 'resources_loaded' not in st.session_state:
        st.error("Please start from the Home page to load the basic resources first!")
        return

    st.title("Speaker Recognition using MLP with LDA")
    
    # Load MLP model
    if not load_mlp_model():
        st.warning("Failed to load the MLP model. Please check if the model file exists.")
        return
    
    
    st.write("Upload a `.wav` file to classify it using the MLP-LDA model.")

    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        
        with st.spinner('Analyzing audio...'):
            speaker_name, proc_time = process_audio(uploaded_file)
        
        # Show results
        st.success(f"Prediction: {speaker_name}")
        st.info(f"Processing time: {proc_time:.4f} seconds")
        

if __name__ == "__main__":
    main()