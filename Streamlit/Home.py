import streamlit as st
import numpy as np
import dill as pickle
import os
import sys
import base64

st.set_page_config(page_title="Linguistix", page_icon="🎙️", layout="centered")

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from naive_bayes import NaiveBayesClassifier
from gmm_semi_supervised import GaussianMixtureModel

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

def add_custom_sidebar_button():
    custom_css = """
    <style>
    /* Hide the default arrow button */
    [data-testid="collapsedControl"] {
        display: none;
    }

    /* Add a custom text label for the sidebar */
    .stSidebar {
        position: relative;
    }
    .stSidebar::before {
        content: "Sidebar"; /* Text label for the sidebar */
        font-size: 18px;
        position: absolute;
        top: 10px;
        left: 10px;
        cursor: pointer;
        color: white;
        font-weight: bold;
    }
    .stSidebar:hover::before {
        color: #f0f0f0; /* Change color on hover */
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def load_resources():
    if 'resources_loaded' not in st.session_state:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load all resources
        st.session_state.x_features = np.load(os.path.join(current_dir, "X_lda.npy"))
        st.session_state.lda_eigenvectors = np.load(os.path.join(current_dir, "lda_eigenvectors.npy"))
        st.session_state.y_labels = np.load(os.path.join(current_dir, "y_labels.npy"))
        
        # Load the classifier
        with open(os.path.join(current_dir, "naive_bayes_model.pkl"), "rb") as f:
            st.session_state.nb_classifier = pickle.load(f)
        
        # Load label mapping
        with open(os.path.join(current_dir, "label_map.pkl"), "rb") as f:
            label_map = pickle.load(f)
            st.session_state.inv_label_map = {v: k for k, v in label_map.items()}
        
        # Load GMM model for speaker recognition
        try:
            with open(os.path.join(current_dir, "best_semi_supervised_gmm.pkl"), "rb") as f:
                st.session_state.gmm_model = pickle.load(f)
                #st.success("GMM model loaded successfully!")
        except Exception as e:
            st.warning("GMM model not loaded. Error: " + str(e))
        
        st.session_state.resources_loaded = True

def main():
    # Add custom sidebar button
    #add_custom_sidebar_button()

    # Assuming you'll place your background image in the same directory
    # image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image.jpg")
    # if os.path.exists(image_path):
    #     add_bg_from_local(image_path)
    
# Title and Introduction
    st.title("🎙️ Linguistix - Speaker Recognition")
    st.markdown(
        """
        Welcome to **Linguistix**, a cutting-edge platform for **Speaker Recognition**. 
        Using advanced models, Linguistix provides 
        accurate and efficient speaker identification for your audio data.
        
        Linguistix gives you a great variety of models to choose from, thus giving a good experience to the user.
        Our models are trained on a diverse dataset, ensuring high accuracy and reliability.
        
        Upload your audio files and let Linguistix identify the speaker with precision!
        """
    )
    
    # Add the image in its original size
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image.jpg")
    if os.path.exists(image_path):
        st.image(image_path, caption="Empowering Speaker Recognition with Linguistix", use_column_width=False)
    
    # Load resources
    load_resources()

    # Sidebar or navigation info
    st.write("Navigate through the sidebar to explore different models and functionalities.")
    st.write("This project is made for the course CSL2050 - Pattern Recognition and Machine Learning at IIT Jodhpur")
    st.write("GitHub Repository Link: https://github.com/RepoRogue123/Linguistix")
    st.write("The project is developed by:")
    st.write("1. **Shashank Parchure** - B23CM1059")
    st.write("2. **Vyankatesh Deshpande** - B23CS1079")    
    st.write("3. **Atharva Honparkhe** - B23EE1006")
    st.write("4. **Abhinash Roy** - B23CS1003")    
    st.write("5. **Namya Dhingra** - B23CS1040")
    st.write("6. **Damarasingu Akshaya Sree** - B23EE1085")   


if __name__ == "__main__":
    main()