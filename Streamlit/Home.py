import streamlit as st
import numpy as np
import dill as pickle
import os
import sys
import base64

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from naive_bayes import NaiveBayesClassifier

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_img = base64.b64encode(img_data).decode()
    custom_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{b64_img}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    .stMarkdown, .stText, .stTitle {{
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }}
    
    .stSidebar {{
        background-color: rgba(0,0,0,0.7);
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def load_resources():
    if 'resources_loaded' not in st.session_state:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load all resources
        st.session_state.lda_eigenvectors = np.load(os.path.join(current_dir, "lda_eigenvectors.npy"))
        st.session_state.y_labels = np.load(os.path.join(current_dir, "y_labels.npy"))
        
        # Load the classifier
        with open(os.path.join(current_dir, "naive_bayes_model.pkl"), "rb") as f:
            st.session_state.nb_classifier = pickle.load(f)
        
        # Load label mapping
        with open(os.path.join(current_dir, "label_map.pkl"), "rb") as f:
            label_map = pickle.load(f)
            st.session_state.inv_label_map = {v: k for k, v in label_map.items()}
        
        st.session_state.resources_loaded = True

def main():
    # Assuming you'll place your background image in the same directory
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "https://recfaces.com/wp-content/uploads/2021/06/voice-recognition-830x571.jpg")
    if os.path.exists(image_path):
        add_bg_from_local(image_path)
    
    st.title("Linguistix - Audio Classification")
    load_resources()
    st.write("Welcome to Linguistix! Please select a model from the sidebar.")

if __name__ == "__main__":
    main()