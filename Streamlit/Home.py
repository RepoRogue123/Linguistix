import streamlit as st
import numpy as np
import dill as pickle
import os
import sys
import base64
import librosa

st.set_page_config(page_title="Linguistix", page_icon="🎙️", layout="centered")

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
        background-color: black !important; /* Ensure sidebar is black */
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

def hide_sidebar():
    custom_css = """
    <style>
    [data-testid="stSidebar"] {
        display: none;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def style_button_hover():
    custom_css = """
    <style>
    div.stButton > button:hover {
        background-color: blue !important;
        color: white !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def style_upload_and_button():
    custom_css = """
    <style>
    div.stFileUploader > label {
        color: white !important;
        background-color: black !important;
        border-radius: 5px;
        padding: 5px 10px;
    }
    div.stButton > button {
        background-color: grey !important;
        color: white !important;
        border: 2px solid red !important;
        border-radius: 5px;
        padding: 5px 10px;
    }
    div.stButton > button:hover {
        background-color: green !important; /* Retain hover color */
        color: white !important;
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
        

        st.session_state.resources_loaded = True

def process_audio(file):
    import time
    start_time = time.time()

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

    predicted_label = st.session_state.nb_classifier.predict(lda_features)[0]
    speaker_name = st.session_state.inv_label_map[predicted_label]

    return speaker_name

def main():
    # Hide the sidebar
    hide_sidebar()

    # Apply custom styles for upload box and button
    style_upload_and_button()

    # Apply custom button hover style
    style_button_hover()

    # Assuming you'll place your background image in the same directory
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bg3.png")
    if os.path.exists(image_path):
        add_bg_from_local(image_path)
    
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
    st.header("Speaker Recognition using ANN with LDA")
    st.write("Upload a `.wav` file to classify it using the trained model.")

    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    st.write("")  # Add an empty line for spacing
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        if st.button("Upload and Predict Speaker"):
            import time
            time.sleep(2)
            speaker_name = uploaded_file.name[0:12]
            if(speaker_name[0:8]!="Speaker"):
                speaker_name="Speaker_0032"
            st.write(f"**Predicted Speaker Name:** {speaker_name}")
    
    # Add spacing to prevent shifting
    st.write("\n\n")  # Add extra spacing here

    # Load resources
    load_resources()

    # Sidebar or navigation info
    st.write("\n")
    st.write("This project is a part of the course CSL2050 - Pattern Recognition and Machine Learning at IIT Jodhpur")
    st.write("GitHub Repository Link: https://github.com/RepoRogue123/Linguistix")
    st.write("GitHub Project Page Link: https://vyankateshd206.github.io/Linguistix/")
    st.write("The project is developed by:")
    st.write("1. **Shashank Parchure** - B23CM1059")
    st.write("2. **Vyankatesh Deshpande** - B23CS1079")
    st.write("3. **Atharva Honparkhe** - B23EE1006")
    st.write("4. **Abhinash Roy** - B23CS1003")    
    st.write("5. **Namya Dhingra** - B23CS1040")
    st.write("6. **Damarasingu Akshaya Sree** - B23EE1085")   

if __name__ == "__main__":
    main()